#!/usr/bin/env python3
"""
Filter an existing train.npz by timestamp range and strict target rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter NPZ dataset by time.")
    parser.add_argument("--in", dest="in_path", type=str, required=True, help="Input npz path")
    parser.add_argument("--out", type=str, required=True, help="Output npz path")
    parser.add_argument("--start", type=str, default=None, help="Start time filter (origin timestamp)")
    parser.add_argument("--end", type=str, default=None, help="End time filter (origin timestamp)")
    parser.add_argument("--strict-target", type=int, default=1, help="Drop samples with NaNs in y window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(in_path, allow_pickle=True)
    X = data["X_train"]
    y = data["y_train"]
    timestamps = data["timestamps"]
    sensor_names = data["sensor_names"]

    ts_pd = pd.to_datetime(timestamps, utc=True, errors="coerce")
    bad = np.where(ts_pd.isna())[0]
    if len(bad) > 0:
        raise ValueError(f"{len(bad)} timestamps failed to parse (NaT). Example idx={bad[:5]}")

    start_ts = pd.to_datetime(args.start, utc=True, errors="coerce") if args.start else None
    end_ts = pd.to_datetime(args.end, utc=True, errors="coerce") if args.end else None

    mask = np.ones(len(ts_pd), dtype=bool)
    if start_ts is not None and not pd.isna(start_ts):
        mask &= ts_pd >= start_ts
    if end_ts is not None and not pd.isna(end_ts):
        mask &= ts_pd <= end_ts

    if args.strict_target:
        mask &= ~np.isnan(y).any(axis=(1, 2))

    if not np.any(mask):
        raise ValueError("No samples left after filtering.")

    X_f = X[mask]
    y_f = y[mask]
    ts_f = np.asarray(timestamps)[mask]

    np.savez_compressed(
        out_path,
        X_train=X_f,
        y_train=y_f,
        timestamps=ts_f,
        sensor_names=sensor_names,
    )

    print(f"Saved {out_path}")
    print(f"Samples: {X_f.shape[0]}")
    print(f"X_train: {X_f.shape} dtype={X_f.dtype}")
    print(f"y_train: {y_f.shape} dtype={y_f.dtype}")


if __name__ == "__main__":
    main()
