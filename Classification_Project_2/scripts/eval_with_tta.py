#!/usr/bin/env python3
"""
Evaluate CNN model with Test-Time Augmentation (TTA).
Applies flips and rotations, averages predictions.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SPLIT_DIR
from scripts.eval_utils import load_split_indices, weighted_f1, macro_f1


class TTADataset(Dataset):
    """Dataset that yields TTA variants of each sample."""
    
    def __init__(
        self,
        patches: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        use_log1p: bool,
    ):
        self.patches = patches
        self.indices = indices
        self.mean = mean
        self.std = std
        self.use_log1p = use_log1p
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        if self.use_log1p:
            x = np.log1p(np.maximum(x, 0.0))
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return x.astype(np.float32)
    
    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        x = self.patches[real_idx].astype(np.float32, copy=False)
        
        # Generate 8 TTA variants: original + 3 rotations + 4 flips
        variants = []
        
        # Original and rotations
        for k in range(4):
            rotated = np.rot90(x, k, axes=(1, 2))
            variants.append(self._preprocess(rotated.copy()))
        
        # Horizontal flip and rotations
        x_flip = np.flip(x, axis=2).copy()
        for k in range(4):
            rotated = np.rot90(x_flip, k, axes=(1, 2))
            variants.append(self._preprocess(rotated.copy()))
        
        # Stack all variants: (8, C, H, W)
        stacked = np.stack([np.ascontiguousarray(v) for v in variants], axis=0)
        return torch.from_numpy(stacked), real_idx


class SmallCNN(torch.nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 192, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.fc = torch.nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x)


def evaluate_with_tta(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 71,
) -> np.ndarray:
    """Returns averaged TTA probabilities for each sample."""
    model.eval()
    
    all_probs = []
    all_indices = []
    
    with torch.no_grad():
        for variants, indices in loader:
            # variants: (B, 8, C, H, W)
            B, num_tta, C, H, W = variants.shape
            
            # Reshape to process all TTA variants at once
            variants_flat = variants.view(B * num_tta, C, H, W).to(device)
            
            # Get predictions
            logits = model(variants_flat)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            # Reshape and average
            probs = probs.reshape(B, num_tta, num_classes)
            avg_probs = probs.mean(axis=1)  # (B, num_classes)
            
            all_probs.append(avg_probs)
            all_indices.extend(indices.numpy().tolist())
    
    return np.vstack(all_probs), np.array(all_indices)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN with TTA")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--norm_path", type=Path, required=True)
    parser.add_argument("--patches_path", type=Path, default=ROOT / "data" / "train" / "patches.npy")
    parser.add_argument("--labels_path", type=Path, default=ROOT / "artifacts" / "features" / "y.npy")
    parser.add_argument("--split_dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    # Load normalization stats
    npz = np.load(args.norm_path)
    mean = npz["mean"].astype(np.float32)
    std = npz["std"].astype(np.float32)
    use_log1p = bool(npz.get("use_log1p", np.array(True)).item())
    
    # Load data
    patches = np.load(args.patches_path, mmap_mode="r")
    labels = np.load(args.labels_path)
    train_idx, val_idx = load_split_indices(args.split_dir)
    y_val = labels[val_idx]
    
    # Create dataset and loader
    dataset = TTADataset(patches, val_idx, mean, std, use_log1p)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Load model
    device = torch.device("cpu")
    model = SmallCNN(in_ch=15, num_classes=71)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # Evaluate with TTA
    print("Running TTA evaluation (8 variants per sample)...")
    probs, indices = evaluate_with_tta(model, loader, device)
    
    # Compute metrics
    preds = np.argmax(probs, axis=1)
    wf1 = weighted_f1(y_val, preds)
    mf1 = macro_f1(y_val, preds)
    
    # Save results
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "val_probs_tta.npy", probs.astype(np.float32))
    
    meta = {
        "model_path": str(args.model_path),
        "norm_path": str(args.norm_path),
        "use_log1p": use_log1p,
        "weighted_f1": float(wf1),
        "macro_f1": float(mf1),
        "tta_variants": 8,
    }
    with open(args.out_dir / "tta_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Weighted F1 (TTA): {wf1:.4f} | Macro F1: {mf1:.4f}")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
