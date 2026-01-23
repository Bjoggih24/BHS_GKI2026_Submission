#!/usr/bin/env python3
"""
Sanity check for weather observations alignment vs training timestamps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_utils import WEATHER_OBS_COLS, load_weather_observations_agg
from utils import load_training_data


def main() -> None:
    X, y, timestamps, _ = load_training_data()
    data_dir = ROOT / "data"
    obs_agg = load_weather_observations_agg(data_dir)

    print("[debug] obs_agg_is_none:", obs_agg is None)
    if obs_agg is None:
        return

    print("[debug] obs_agg_shape:", obs_agg.shape)
    print("[debug] obs_agg_index_minmax:", obs_agg.index.min(), obs_agg.index.max())
    print("[debug] obs_agg_index_tz:", getattr(obs_agg.index, "tz", None))
    print("[debug] obs_agg_index_sample:", obs_agg.index[:5])

    ts_all = pd.to_datetime(timestamps, errors="coerce", utc=True)
    cutoff = pd.Timestamp("2022-06-08", tz="UTC")
    valid = np.where(ts_all >= cutoff)[0]
    if valid.size == 0:
        print("[weather_obs] no timestamps >= 2022-06-08 found")
        return

    n_check = min(200, valid.size)
    idxs_post = valid[np.linspace(0, valid.size - 1, n_check).astype(int)]

    total_counts = 0
    total_finite = 0
    col_finite = np.zeros(len(obs_agg.columns), dtype=np.int64)
    col_counts = np.zeros(len(obs_agg.columns), dtype=np.int64)

    for j, i in enumerate(idxs_post):
        t0 = pd.to_datetime(str(timestamps[i]), errors="coerce", utc=True).tz_localize(None)
        hist_index = pd.date_range(end=t0 - pd.Timedelta(hours=1), periods=672, freq="h")
        aligned = obs_agg.reindex(hist_index)

        if j == 0:
            print("[debug] first_timestamp:", str(timestamps[i]))
            print("[debug] hist_index_sample:", hist_index[:5])

        finite_mask = np.isfinite(aligned.to_numpy())
        total_counts += finite_mask.size
        total_finite += int(finite_mask.sum())
        col_finite += finite_mask.sum(axis=0).astype(np.int64)
        col_counts += finite_mask.shape[0]

    overall_frac = total_finite / max(total_counts, 1)
    print(f"[weather_obs] n={len(idxs_post)} overall_finite_frac={overall_frac:.4f}")
    for col, f_cnt, c_cnt in zip(obs_agg.columns, col_finite, col_counts):
        frac = float(f_cnt) / max(int(c_cnt), 1)
        print(f"[weather_obs] {col}: finite_frac={frac:.4f}")


if __name__ == "__main__":
    main()
