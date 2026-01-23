#!/usr/bin/env python3
"""
Evaluation utilities for habitat classification.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_split_indices(split_dir: Path):
    train_idx = np.load(split_dir / "train_idx.npy")
    val_idx = np.load(split_dir / "val_idx.npy")
    return train_idx, val_idx


def weighted_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="weighted")


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro")


def report_per_class_f1(y_true, y_pred, top_k: int = 10):
    per_class = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=np.arange(71),
        zero_division=0,
    )
    worst_idx = np.argsort(per_class)[:top_k]
    return list(zip(worst_idx.tolist(), per_class[worst_idx].tolist()))
