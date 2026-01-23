#!/usr/bin/env python3
"""
Quick sanity check for weather alignment vs training timestamps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_utils import WEATHER_NUMERIC_COLS, build_horizon_times, build_weather_features, load_weather_forecasts_agg
from utils import load_training_data


def _summarize(weather_agg, timestamps, idxs, label: str) -> None:
    total_counts = 0
    total_finite = 0
    col_finite = np.zeros(len(WEATHER_NUMERIC_COLS), dtype=np.int64)
    col_counts = np.zeros(len(WEATHER_NUMERIC_COLS), dtype=np.int64)

    for j, i in enumerate(idxs):
        ts = str(timestamps[i])
        horizon_times = build_horizon_times(ts)

        if j == 0:
            print(f"[debug] {label} first_timestamp:", ts)
            print(f"[debug] {label} horizon_times_tz:", getattr(horizon_times, "tz", None))
            print(f"[debug] {label} horizon_times_sample:", horizon_times[:5])

        wf = build_weather_features(horizon_times, None, weather_agg)
        finite_mask = np.isfinite(wf)

        total_counts += finite_mask.size
        total_finite += int(finite_mask.sum())
        col_finite += finite_mask.sum(axis=0).astype(np.int64)
        col_counts += finite_mask.shape[0]

    overall_frac = total_finite / max(total_counts, 1)
    print(f"[weather:{label}] n={len(idxs)} overall_finite_frac={overall_frac:.4f}")

    for col, f_cnt, c_cnt in zip(WEATHER_NUMERIC_COLS, col_finite, col_counts):
        frac = float(f_cnt) / max(int(c_cnt), 1)
        print(f"[weather:{label}] {col}: finite_frac={frac:.4f}")


def main() -> None:
    X, y, timestamps, _ = load_training_data()
    data_dir = ROOT / "data"
    weather_agg = load_weather_forecasts_agg(data_dir)

    print("[debug] weather_agg_is_none:", weather_agg is None)
    if weather_agg is not None:
        print("[debug] weather_agg_shape:", weather_agg.shape)
        print("[debug] weather_agg_index_minmax:", weather_agg.index.min(), weather_agg.index.max())
        print("[debug] weather_agg_index_tz:", getattr(weather_agg.index, "tz", None))
        print("[debug] weather_agg_index_sample:", weather_agg.index[:5])

    n_check = min(200, len(timestamps))

    ts_all = pd.to_datetime(timestamps, errors="coerce", utc=True)

    idxs_span = np.linspace(0, len(timestamps) - 1, n_check).astype(int)
    _summarize(weather_agg, timestamps, idxs_span, label="span")

    cutoff = pd.Timestamp("2022-06-08", tz="UTC")
    valid = np.where(ts_all >= cutoff)[0]
    if valid.size == 0:
        print("[weather:post_2022] no timestamps >= 2022-06-08 found")
        return

    idxs_post = valid[np.linspace(0, valid.size - 1, min(n_check, valid.size)).astype(int)]
    _summarize(weather_agg, timestamps, idxs_post, label="post_2022")


if __name__ == "__main__":
    main()
