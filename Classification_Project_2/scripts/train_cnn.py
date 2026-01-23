#!/usr/bin/env python3
"""
Train CNN model on habitat patches (CPU-friendly, memmap-safe).
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SPLIT_DIR
from scripts.eval_utils import load_split_indices, weighted_f1, macro_f1, report_per_class_f1


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    label_smoothing: float = 0.05
    patience: int = 7
    scheduler: str = "cosine"
    use_log1p: int = 1
    augment: int = 0
    aug_flip: float = 0.5
    aug_rot: int = 1
    aug_noise_std: float = 0.01
    aug_gain: float = 0.1


class PatchDataset(Dataset):
    def __init__(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        use_log1p: bool,
        augment: bool,
        aug_flip: float,
        aug_rot: bool,
        aug_noise_std: float,
        aug_gain: float,
    ):
        self.patches = patches
        self.labels = labels
        self.indices = indices
        self.mean = mean
        self.std = std
        self.use_log1p = use_log1p
        self.augment = augment
        self.aug_flip = aug_flip
        self.aug_rot = aug_rot
        self.aug_noise_std = aug_noise_std
        self.aug_gain = aug_gain

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if self.aug_flip > 0.0:
            if np.random.rand() < self.aug_flip:
                x = np.flip(x, axis=1)
            if np.random.rand() < self.aug_flip:
                x = np.flip(x, axis=2)
        if self.aug_rot:
            k = np.random.randint(0, 4)
            if k:
                x = np.rot90(x, k, axes=(1, 2))
        if self.aug_noise_std > 0.0:
            x = x + np.random.normal(0, self.aug_noise_std, size=x.shape).astype(np.float32)
        if self.aug_gain > 0.0:
            gains = np.random.uniform(1.0 - self.aug_gain, 1.0 + self.aug_gain, size=(x.shape[0], 1, 1))
            x = x * gains.astype(np.float32)
        return x

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        x = self.patches[real_idx].astype(np.float32, copy=False)
        if self.augment:
            x = self._augment(x)
        if self.use_log1p:
            x = np.log1p(np.maximum(x, 0.0))
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        x = torch.from_numpy(np.ascontiguousarray(x))
        y = torch.tensor(self.labels[real_idx], dtype=torch.long)
        return x, y


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.net(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def make_model(arch: str, in_ch: int, num_classes: int, dropout: float) -> nn.Module:
    if arch == "small_cnn":
        return SmallCNN(in_ch, num_classes, dropout)
    try:
        import timm
    except Exception as exc:
        raise RuntimeError("timm is required for non-small_cnn architectures") from exc

    if arch in {
        "resnet18",
        "resnet34",
        "tf_efficientnet_b0",
        "mobilenetv3_large_100",
        "convnext_tiny",
    }:
        return timm.create_model(arch, pretrained=True, in_chans=in_ch, num_classes=num_classes)
    raise ValueError(f"Unknown arch: {arch}")


def compute_class_weights(labels: np.ndarray, power: float, num_classes: int) -> torch.Tensor:
    values, counts = np.unique(labels, return_counts=True)
    freq = counts.astype(np.float32)
    weights = 1.0 / (freq ** power)
    weights = weights / weights.mean()
    full = np.ones(num_classes, dtype=np.float32)
    for v, w in zip(values, weights):
        full[int(v)] = w
    return torch.tensor(full, dtype=torch.float32)


def compute_mean_std(patches: np.ndarray, train_idx: np.ndarray, use_log1p: bool) -> Tuple[np.ndarray, np.ndarray]:
    num_channels = patches.shape[1]
    total = 0
    sum_c = np.zeros(num_channels, dtype=np.float64)
    sumsq_c = np.zeros(num_channels, dtype=np.float64)

    batch_size = 256
    for start in range(0, len(train_idx), batch_size):
        idx = train_idx[start:start + batch_size]
        batch = patches[idx].astype(np.float32, copy=False)
        if use_log1p:
            batch = np.log1p(np.maximum(batch, 0.0))
        sum_c += batch.sum(axis=(0, 2, 3))
        sumsq_c += np.square(batch).sum(axis=(0, 2, 3))
        total += batch.shape[0] * batch.shape[2] * batch.shape[3]

    mean = (sum_c / total).astype(np.float32)
    var = (sumsq_c / total) - np.square(mean, dtype=np.float64)
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray]:
    model.eval()
    preds = []
    targets = []
    probs = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            preds.append(np.argmax(p, axis=1))
            targets.append(y.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    val_probs = np.vstack(probs)
    return weighted_f1(y_true, y_pred), macro_f1(y_true, y_pred), val_probs


def load_params(params_path: Path) -> TrainConfig:
    if not params_path.exists():
        return TrainConfig()
    data = json.loads(params_path.read_text(encoding="utf-8"))
    cfg = TrainConfig()
    for field in cfg.__dataclass_fields__:
        if field in data:
            setattr(cfg, field, data[field])
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN model (memmap-safe)")
    parser.add_argument(
        "--arch",
        choices=[
            "small_cnn",
            "resnet18",
            "resnet34",
            "tf_efficientnet_b0",
            "mobilenetv3_large_100",
            "convnext_tiny",
        ],
        required=True,
    )
    parser.add_argument("--params_json", type=Path, required=True)
    parser.add_argument("--patches_path", type=Path, default=ROOT / "data" / "train" / "patches.npy")
    parser.add_argument("--labels_path", type=Path, default=ROOT / "artifacts" / "features" / "y.npy")
    parser.add_argument("--split_dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_class_weights", type=int, default=1)
    parser.add_argument("--weight_power", type=float, default=0.25)
    args = parser.parse_args()

    torch.set_num_threads(4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.patches_path.exists():
        raise FileNotFoundError(f"Missing patches: {args.patches_path}")
    if not args.labels_path.exists():
        raise FileNotFoundError(f"Missing labels: {args.labels_path}")

    cfg = load_params(args.params_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx = load_split_indices(args.split_dir)
    patches = np.load(args.patches_path, mmap_mode="r")
    labels = np.load(args.labels_path)
    num_classes = int(labels.max()) + 1
    in_ch = int(patches.shape[1])

    mean, std = compute_mean_std(patches, train_idx, use_log1p=bool(cfg.use_log1p))
    norm_path = args.out_dir / "norm.npz"
    np.savez(norm_path, mean=mean, std=std, use_log1p=bool(cfg.use_log1p))

    train_ds = PatchDataset(
        patches,
        labels,
        train_idx,
        mean,
        std,
        bool(cfg.use_log1p),
        bool(cfg.augment),
        float(cfg.aug_flip),
        bool(cfg.aug_rot),
        float(cfg.aug_noise_std),
        float(cfg.aug_gain),
    )
    val_ds = PatchDataset(
        patches,
        labels,
        val_idx,
        mean,
        std,
        bool(cfg.use_log1p),
        False,
        0.0,
        False,
        0.0,
        0.0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(32, min(256, cfg.batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    device = torch.device("cpu")
    model = make_model(args.arch, in_ch=in_ch, num_classes=num_classes, dropout=cfg.dropout).to(device)

    if args.use_class_weights:
        class_weights = compute_class_weights(labels[train_idx], args.weight_power, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_wf1 = -1.0
    best_epoch = -1
    bad_epochs = 0

    log_path = args.out_dir / "train_log.csv"
    start_time = time.time()
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_weighted_f1", "val_macro_f1"])
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            total_loss = 0.0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)

            train_loss = total_loss / len(train_loader.dataset)
            wf1, mf1, _ = evaluate(model, val_loader, device)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{wf1:.4f}", f"{mf1:.4f}"])
            f.flush()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(wf1)
                else:
                    scheduler.step()

            if wf1 > best_wf1 + 1e-6:
                best_wf1 = wf1
                best_epoch = epoch
                bad_epochs = 0
                torch.save(model.state_dict(), args.out_dir / "model.pt")
            else:
                bad_epochs += 1

            print(f"Epoch {epoch:02d} | loss {train_loss:.4f} | wf1 {wf1:.4f} | mf1 {mf1:.4f}")
            if bad_epochs >= cfg.patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(args.out_dir / "model.pt", map_location=device))
    wf1, mf1, val_probs = evaluate(model, val_loader, device)
    np.save(args.out_dir / "val_probs.npy", val_probs.astype(np.float32))

    worst = report_per_class_f1(labels[val_idx], np.argmax(val_probs, axis=1), top_k=10)
    meta: Dict[str, object] = {
        "arch": args.arch,
        "params": asdict(cfg),
        "norm_path": str(norm_path),
        "num_classes": num_classes,
        "in_ch": in_ch,
        "use_class_weights": bool(args.use_class_weights),
        "weight_power": args.weight_power if args.use_class_weights else None,
        "best_epoch": best_epoch,
        "weighted_f1": wf1,
        "macro_f1": mf1,
        "worst_classes": worst,
        "duration_sec": round(time.time() - start_time, 2),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    try:
        import timm

        meta["timm"] = timm.__version__
    except Exception:
        meta["timm"] = None
    try:
        import subprocess

        meta["git_hash"] = (
            subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        )
    except Exception:
        meta["git_hash"] = "unknown"

    with open(args.out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {args.out_dir / 'model.pt'}")
    print(f"Weighted F1: {wf1:.4f} | Macro F1: {mf1:.4f}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    main()
