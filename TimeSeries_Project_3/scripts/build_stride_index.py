#!/usr/bin/env python3
"""
Build stride index lists for NN window slicing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HISTORY_LENGTH = 672
HORIZON = 72


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stride index lists.")
    parser.add_argument("--series_dir", type=str, default="artifacts/datasets")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start_ts", type=str, default=None)
    parser.add_argument("--end_ts", type=str, default=None)
    parser.add_argument("--require_strict_y", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    series_dir = Path(args.series_dir)
    values = np.load(series_dir / "series_values.npy")
    times = np.load(series_dir / "series_times.npy")

    ts = pd.to_datetime(times, errors="coerce", utc=True)
    if ts.isna().any():
        bad = np.where(ts.isna())[0]
        raise ValueError(f"NaT found in series_times: idx={bad[:5]}")

    start_ts = pd.to_datetime(args.start_ts, utc=True, errors="coerce") if args.start_ts else None
    end_ts = pd.to_datetime(args.end_ts, utc=True, errors="coerce") if args.end_ts else None

    t0_min = HISTORY_LENGTH
    t0_max = values.shape[0] - HORIZON
    if t0_max <= t0_min:
        raise ValueError("Series too short for history+horizon windows.")

    idx_list = []
    for t0 in range(t0_min, t0_max, args.stride):
        ts0 = ts[t0]
        if start_ts is not None and ts0 < start_ts:
            continue
        if end_ts is not None and ts0 > end_ts:
            continue
        idx_list.append(t0)

    if not idx_list:
        raise ValueError("No indices created with current filters.")

    idx_arr = np.array(idx_list, dtype=np.int64)
    out_train = series_dir / f"idx_stride{args.stride}_train.npy"
    np.save(out_train, idx_arr)

    if args.require_strict_y:
        strict_mask = []
        for t0 in idx_arr:
            y_window = values[t0 : t0 + HORIZON]
            strict_mask.append(np.isfinite(y_window).all())
        strict_idx = idx_arr[np.array(strict_mask, dtype=bool)]
        out_strict = series_dir / f"idx_stride{args.stride}_strict.npy"
        np.save(out_strict, strict_idx)
        print(f"Saved {out_strict} ({len(strict_idx)})")

    print(f"Saved {out_train} ({len(idx_arr)})")


if __name__ == "__main__":
    main()
