#!/usr/bin/env python3
"""
Extract image embeddings using pretrained timm models on CPU.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import timm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SPLIT_DIR
from scripts.eval_utils import load_split_indices


class ImagePathDataset(Dataset):
    def __init__(self, paths, transform, root: Path):
        self.paths = paths
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        if isinstance(p, bytes):
            p = p.decode("utf-8")
        path = Path(p)
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img) if self.transform is not None else img
        return idx, tensor

def _to_uint8_rgb(patch: np.ndarray) -> np.ndarray:
    # --- 1) Ensure we end up with HWC and exactly 3 channels ---
    if patch.ndim == 2:
        p = np.stack([patch, patch, patch], axis=-1)  # HWC
    elif patch.ndim == 3:
        # If it looks like CHW (your case: (15,35,35)), transpose to HWC
        if patch.shape[0] >= 3 and patch.shape[-1] not in {1, 3}:
            p = np.transpose(patch, (1, 2, 0))  # HWC
        else:
            p = patch  # assume already HWC

        # Now force exactly 3 channels
        if p.shape[-1] == 1:
            p = np.repeat(p, 3, axis=-1)
        elif p.shape[-1] >= 3:
            p = p[..., :3]
        else:
            # Rare case: 2 channels -> pad a 3rd
            pad = np.zeros((p.shape[0], p.shape[1], 3 - p.shape[-1]), dtype=p.dtype)
            p = np.concatenate([p, pad], axis=-1)
    else:
        raise ValueError(f"Unexpected patch ndim={patch.ndim}")

    # --- 2) Robust scaling (per-channel percentiles) to 0..255 ---
    p = p.astype(np.float32, copy=False)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

    lo = np.percentile(p, 1.0, axis=(0, 1), keepdims=True)
    hi = np.percentile(p, 99.0, axis=(0, 1), keepdims=True)
    hi = np.maximum(hi, lo + 1e-6)

    p = (p - lo) / (hi - lo)
    p = np.clip(p, 0.0, 1.0) * 255.0

    return p.astype(np.uint8)



class PatchArrayDataset(Dataset):
    def __init__(self, patches: np.ndarray, transform):
        self.patches = patches
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        patch = _to_uint8_rgb(patch)
        img = Image.fromarray(patch, mode="RGB")
        tensor = self.transform(img) if self.transform is not None else img
        return idx, tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_embedding_dim(model, sample_batch: torch.Tensor) -> int:
    with torch.inference_mode():
        output = model(sample_batch)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if isinstance(output, dict):
            output = next(iter(output.values()))
        if output.ndim > 2:
            output = output.flatten(1)
        return int(output.shape[1])


def main():
    parser = argparse.ArgumentParser(description="Extract image embeddings on CPU")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_paths_npy", type=Path)
    input_group.add_argument("--patches_npy", type=Path)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--labels_path", type=Path, default=ROOT / "artifacts" / "features" / "y.npy")
    args = parser.parse_args()

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_seed(args.seed)

    image_paths = None
    patches = None
    if args.patches_npy is not None:
        patches = np.load(args.patches_npy, mmap_mode="r")
        num_samples = len(patches)
    else:
        try:
            image_paths = np.load(args.image_paths_npy)
        except ValueError:
            image_paths = np.load(args.image_paths_npy, allow_pickle=True)
        num_samples = len(image_paths)
    y_all = np.load(args.labels_path)
    if num_samples != len(y_all):
        raise ValueError(
            f"inputs length {num_samples} does not match labels length {len(y_all)}"
        )

    train_idx, val_idx = load_split_indices(SPLIT_DIR)

    model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
    model.eval()
    model.to("cpu")

    data_config = timm.data.resolve_data_config({}, model=model)
    data_config["input_size"] = (3, args.img_size, args.img_size)
    transform = timm.data.create_transform(**data_config, is_training=False)

    if patches is not None:
        dataset = PatchArrayDataset(patches, transform)
    else:
        dataset = ImagePathDataset(image_paths, transform, ROOT)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    emb_dim = getattr(model, "num_features", None)
    X_all = None
    if emb_dim is not None:
        X_all = np.zeros((num_samples, int(emb_dim)), dtype=np.float32)

    total_batches = len(loader)
    for batch_idx, (idxs, batch) in enumerate(loader):
        batch = batch.to("cpu")
        if emb_dim is None:
            emb_dim = resolve_embedding_dim(model, batch[:1])
            X_all = np.zeros((num_samples, emb_dim), dtype=np.float32)

        if batch_idx % 50 == 0:
            print(f"[emb] {batch_idx}/{total_batches}")

        with torch.inference_mode():
            output = model(batch)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if isinstance(output, dict):
                output = next(iter(output.values()))
            if output.ndim > 2:
                output = output.flatten(1)
            output = output.detach().cpu().numpy().astype(np.float32)

        X_all[idxs.numpy()] = output

    if X_all is None:
        raise RuntimeError("No embeddings were produced; check image paths and dataset.")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_path, X_all)

    meta = {
        "model_name": args.model_name,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "embedding_dim": int(X_all.shape[1]),
        "num_samples": int(X_all.shape[0]),
        "image_paths_npy": str(args.image_paths_npy) if args.image_paths_npy else None,
        "patches_npy": str(args.patches_npy) if args.patches_npy else None,
        "labels_path": str(args.labels_path),
        "train_idx_len": int(len(train_idx)),
        "val_idx_len": int(len(val_idx)),
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "numpy_version": np.__version__,
        "timestamp_utc": datetime.utcnow().isoformat(),
    }
    meta_path = args.out_path.parent / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved embeddings to {args.out_path}")


if __name__ == "__main__":
    main()
