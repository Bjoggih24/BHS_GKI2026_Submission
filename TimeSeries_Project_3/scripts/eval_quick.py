#!/usr/bin/env python3
"""
Quick, biased evaluation on training samples (rough sanity check).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import predict
from utils import compute_score, normalize_timestamps


def _pick_indices(n: int, max_samples: int) -> np.ndarray:
    if max_samples <= 0 or max_samples >= n:
        return np.arange(n)
    if max_samples == 1:
        return np.array([n // 2])
    return np.linspace(0, n - 1, max_samples, dtype=int)


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
        category=UserWarning,
    )
    parser = argparse.ArgumentParser(description="Quick eval on training data (biased).")
    parser.add_argument("--train-path", default="data/train_full.npz")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    data_path = Path(args.train_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    data = np.load(data_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    timestamps = normalize_timestamps(data["timestamps"])

    idx = _pick_indices(len(X), args.max_samples)
    X = X[idx]
    y = y[idx]
    timestamps = timestamps[idx]

    start = perf_counter()
    preds = []
    for x, ts in zip(X, timestamps):
        preds.append(predict(x, str(ts), None, None))
    preds = np.array(preds, dtype=np.float32)
    elapsed = perf_counter() - start

    score = compute_score(y, preds, X)
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))

    print("Quick eval (biased; uses training samples):")
    print(f"  samples: {len(X)}")
    print(f"  score:   {score:.4f}")
    print(f"  rmse:    {rmse:.4f}")
    print(f"  time:    {elapsed:.2f}s")

    # Running mean skill every 10 samples (rough stability check)
    if len(X) >= 10:
        print("\nRunning mean skill (every 10 samples):")
        for end in range(10, len(X) + 1, 10):
            score_k = compute_score(y[:end], preds[:end], X[:end])
            print(f"  n={end:4d} | score={score_k:.4f}")


if __name__ == "__main__":
    main()
