#!/usr/bin/env python3
"""
Create a stratified train/val split with rare-class handling.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def main():
    seed = 42
    val_frac = 0.15
    rare_threshold = 3

    data_dir = Path(__file__).parent.parent / "data"
    csv_path = data_dir / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    labels = df["vistgerd_idx"].to_numpy()

    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)

        if n < rare_threshold:
            train_idx.extend(cls_idx.tolist())
            continue

        n_val = max(1, int(round(n * val_frac)))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)

    out_dir = Path(__file__).parent.parent / "artifacts" / "split_seed42"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_idx.npy", train_idx)
    np.save(out_dir / "val_idx.npy", val_idx)

    print(f"Saved train_idx.npy: {len(train_idx)}")
    print(f"Saved val_idx.npy: {len(val_idx)}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
