#!/usr/bin/env python3
"""
Compute per-channel mean/std on training split.
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import NORM_STATS_PATH, TRAIN_DIR, SPLIT_DIR


def main():
    patches_path = TRAIN_DIR / "patches.npy"
    if not patches_path.exists():
        raise FileNotFoundError(f"Missing {patches_path}")

    train_idx = np.load(SPLIT_DIR / "train_idx.npy")
    patches = np.load(patches_path, mmap_mode="r")
    subset = patches[train_idx]

    mean = subset.mean(axis=(0, 2, 3))
    std = subset.std(axis=(0, 2, 3)) + 1e-6

    NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NORM_STATS_PATH, mean=mean, std=std)
    print(f"Saved {NORM_STATS_PATH}")


if __name__ == "__main__":
    main()
