#!/usr/bin/env python3
"""
Report weather forecast coverage using build_weather_features_asof.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from feature_utils import build_horizon_times, build_weather_features_asof, load_weather_forecasts_runs
from utils import load_training_data, normalize_timestamps


def main() -> None:
    X, y, timestamps, sensor_names = load_training_data()
    timestamps = normalize_timestamps(timestamps)
    ts_pd = pd.to_datetime(timestamps, errors="coerce", utc=True)
    bad = np.where(ts_pd.isna())[0]
    if len(bad) > 0:
        raise ValueError(f"{len(bad)} timestamps failed to parse (NaT). Example idx={bad[:5]}")

    data_dir = ROOT / "data"
    runs_pack = load_weather_forecasts_runs(data_dir)
    runs, issue_times = (runs_pack if runs_pack is not None else (None, None))
    if runs is None or issue_times is None:
        print("[weather_coverage] no runs available")
        return

    fracs = []
    for ts in timestamps:
        ht = build_horizon_times(str(ts))
        wf = build_weather_features_asof(ht, str(ts), runs, issue_times)
        fracs.append(float(np.isfinite(wf).mean()))

    fracs = np.array(fracs, dtype=np.float32)
    zero_frac = float(np.mean(fracs == 0.0))
    print("finite_frac==0:", zero_frac)
    print(
        "finite_frac stats:",
        "mean", float(np.mean(fracs)),
        "median", float(np.median(fracs)),
        "p10", float(np.percentile(fracs, 10)),
        "p90", float(np.percentile(fracs, 90)),
    )

    threshold = 0.5
    good_idx = np.where(fracs >= threshold)[0]
    if len(good_idx) == 0:
        print("No timestamps with decent coverage.")
        return

    first_good = timestamps[good_idx[0]]
    print("earliest decent coverage:", first_good)
    before = np.mean(pd.to_datetime(timestamps, utc=True) < pd.to_datetime(first_good, utc=True))
    after = 1.0 - before
    print("pct before:", float(before), "pct after:", float(after))


if __name__ == "__main__":
    main()
