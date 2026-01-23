#!/usr/bin/env python3
"""
End-to-end data pipeline checks (train.npz, weather forecasts, observations).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_utils import (
    WEATHER_NUMERIC_COLS,
    build_horizon_times,
    build_weather_features_asof,
    load_weather_observations_agg,
    load_weather_forecasts_runs,
)
from utils import load_training_data


def _print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _train_npz_checks() -> dict:
    _print_header("A) train.npz integrity")
    X, y, timestamps, sensor_names = load_training_data()
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("timestamps shape:", timestamps.shape)
    print("sensor_names shape:", sensor_names.shape)
    print("y NaN frac:", float(np.isnan(y).mean()))
    print("X NaN frac:", float(np.isnan(X).mean()))

    ts_utc = pd.to_datetime(timestamps, errors="coerce", utc=True)
    ts_naive = pd.to_datetime(timestamps, errors="coerce")
    print("timestamps min/max (utc):", ts_utc.min(), ts_utc.max(), "nan_frac:", float(ts_utc.isna().mean()))
    print("timestamps tz-aware (utc):", getattr(ts_utc.tz, "zone", None))
    print("timestamps tz-aware (naive):", getattr(ts_naive.tz, "zone", None))

    # monotonicity + stride
    ts_sorted = np.sort(ts_utc.dropna().values)
    if len(ts_sorted) > 1:
        diffs = np.diff(ts_sorted).astype("timedelta64[h]").astype(int)
        print("median diff hours:", float(np.median(diffs)))
        print("min/max diff hours:", int(np.min(diffs)), int(np.max(diffs)))
    else:
        print("not enough timestamps for diff stats")

    ok = float(np.isnan(y).mean()) == 0.0
    return {"TRAIN_NPZ_OK": ok}


def _obs_checks() -> dict:
    _print_header("B) weather observations integrity")
    data_dir = ROOT / "data"
    obs_agg = load_weather_observations_agg(data_dir)
    print("[debug] obs_agg_is_none:", obs_agg is None)
    if obs_agg is None:
        return {"OBS_OK": False}

    print("[debug] obs_agg_shape:", obs_agg.shape)
    print("[debug] obs_agg_index_minmax:", obs_agg.index.min(), obs_agg.index.max())
    print("[debug] obs_agg_index_tz:", getattr(obs_agg.index, "tz", None))

    _, _, timestamps, _ = load_training_data()
    ts_all = pd.to_datetime(timestamps, errors="coerce", utc=True)
    cutoff = pd.Timestamp("2022-06-08", tz="UTC")
    valid = np.where(ts_all >= cutoff)[0]
    if valid.size == 0:
        print("[weather_obs] no timestamps >= 2022-06-08 found")
        return {"OBS_OK": False}

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

    return {"OBS_OK": overall_frac > 0.99}


def _infer_issue_valid_cols(df: pd.DataFrame) -> tuple[str, str]:
    issue = pd.to_datetime(df["date_time"], errors="coerce", utc=True)
    valid = pd.to_datetime(df["value_date"], errors="coerce", utc=True)
    lead_a = (valid - issue) / pd.Timedelta(hours=1)
    lead_b = (issue - valid) / pd.Timedelta(hours=1)
    print("median lead_a hours:", lead_a.median())
    print("median lead_b hours:", lead_b.median())
    if lead_a.median() < 0:
        return "value_date", "date_time"
    return "date_time", "value_date"


def _forecast_checks() -> dict:
    _print_header("C) weather forecasts integrity + lead time")
    df = pd.read_csv(ROOT / "data" / "weather_forecasts.zip", compression="zip")
    issue_col, valid_col = _infer_issue_valid_cols(df)
    print("issue_col:", issue_col, "valid_col:", valid_col)
    issue = pd.to_datetime(df[issue_col], errors="coerce", utc=True)
    valid = pd.to_datetime(df[valid_col], errors="coerce", utc=True)
    lead_hours = (valid - issue) / pd.Timedelta(hours=1)

    print("lead_hours min/median/max:", lead_hours.min(), lead_hours.median(), lead_hours.max())
    print("lead_hours < 0 frac:", float((lead_hours < 0).mean()))
    print("lead_hours value counts (top 20):")
    print(lead_hours.value_counts(dropna=True).head(20))

    rows_per_valid = valid.dt.floor("h")
    print("rows per valid_time describe:")
    print(rows_per_valid.groupby(rows_per_valid).size().describe())

    return {"FORECAST_SCHEMA_OK": True}


def _leak_check() -> dict:
    _print_header("D) dataset contains future runs (leak risk if aggregated)")
    df = pd.read_csv(ROOT / "data" / "weather_forecasts.zip", compression="zip")
    issue_col, valid_col = _infer_issue_valid_cols(df)
    issue = pd.to_datetime(df[issue_col], errors="coerce", utc=True)
    valid = pd.to_datetime(df[valid_col], errors="coerce", utc=True)
    issue_norm = issue.dt.floor("h")
    valid_norm = valid.dt.floor("h")
    df = df.assign(issue_norm=issue_norm, valid_norm=valid_norm).dropna(subset=["issue_norm", "valid_norm"])

    issue_max_by_valid = df.groupby("valid_norm")["issue_norm"].max()

    _, _, timestamps, _ = load_training_data()
    ts_all = pd.to_datetime(timestamps, errors="coerce", utc=True)
    cutoff = pd.Timestamp("2022-06-08", tz="UTC")
    valid_ts = ts_all[ts_all >= cutoff]
    if valid_ts.empty:
        print("[leak] no timestamps >= 2022-06-08 found")
        return {"FORECAST_LEAK_RISK_HIGH": False, "RECOMMEND_ASOF_FIX": False}

    n_check = min(200, len(valid_ts))
    idxs = np.linspace(0, len(valid_ts) - 1, n_check).astype(int)
    t0s = valid_ts[idxs]

    horizons = [0, 12, 24, 48, 72]
    future_count = 0
    total_pairs = 0
    deltas = []

    for t0 in t0s:
        t0 = pd.Timestamp(t0).tz_convert("UTC")
        for h in horizons:
            v = (t0 + pd.Timedelta(hours=h)).floor("h")
            issue_max = issue_max_by_valid.get(v)
            total_pairs += 1
            if issue_max is not None and issue_max > t0:
                future_count += 1
                deltas.append((issue_max - t0) / pd.Timedelta(hours=1))

    frac = future_count / max(total_pairs, 1)
    avg_adv = float(np.mean(deltas)) if deltas else 0.0
    print("future-issued fraction:", frac)
    print("avg future advantage hours:", avg_adv)

    high_risk = frac > 0.05
    return {"DATASET_HAS_FUTURE_RUNS": high_risk, "RECOMMEND_ASOF_FIX": high_risk}


def _asof_check() -> dict:
    _print_header("E) ASOF forecast usage sanity")
    data_dir = ROOT / "data"
    runs_pack = load_weather_forecasts_runs(data_dir)
    if runs_pack is None:
        print("[asof] runs not available")
        return {"ASOF_NO_LEAK_OK": False}

    runs, issue_times = runs_pack
    if issue_times is None or len(issue_times) == 0:
        print("[asof] no issue_times found")
        return {"ASOF_NO_LEAK_OK": False}

    _, _, timestamps, _ = load_training_data()
    ts_all = pd.to_datetime(timestamps, errors="coerce", utc=True)
    cutoff = pd.Timestamp("2022-06-08", tz="UTC")
    valid_ts = ts_all[ts_all >= cutoff]
    if valid_ts.empty:
        print("[asof] no timestamps >= 2022-06-08 found")
        return {"ASOF_NO_LEAK_OK": False}

    n_check = min(200, len(valid_ts))
    idxs = np.linspace(0, len(valid_ts) - 1, n_check).astype(int)
    t0s = valid_ts[idxs]

    future_issue_count = 0
    total_pairs = 0
    finite_fracs = []

    issue_times = pd.DatetimeIndex(issue_times)
    for t0 in t0s:
        t0 = pd.Timestamp(t0).tz_convert("UTC").tz_localize(None).floor("h")
        k = issue_times.searchsorted(t0, side="right") - 1
        if k < 0:
            continue
        issue_star = issue_times[k]
        if issue_star > t0:
            future_issue_count += 1

        run_df = runs.xs(issue_star, level=0)
        ht = pd.date_range(start=t0, periods=72, freq="h")
        aligned = run_df.reindex(ht)
        finite_frac = float(np.isfinite(aligned.to_numpy()).mean())
        finite_fracs.append(finite_frac)
        total_pairs += 1

    frac_future = future_issue_count / max(total_pairs, 1)
    mean_frac = float(np.mean(finite_fracs)) if finite_fracs else 0.0
    min_frac = float(np.min(finite_fracs)) if finite_fracs else 0.0
    print("asof_used_future_issue_frac:", frac_future)
    print("asof_coverage_mean:", mean_frac, "min:", min_frac)

    return {"ASOF_NO_LEAK_OK": frac_future == 0.0}


def _final_weather_feature_coverage_check() -> dict:
    _print_header("F) FINAL weather feature coverage (after ASOF backtracking + cutoffs)")

    data_dir = ROOT / "data"
    runs_pack = load_weather_forecasts_runs(data_dir)
    if runs_pack is None:
        print("[final_weather] runs not available")
        return {"FINAL_WEATHER_OK": False}

    runs, issue_times = runs_pack
    if issue_times is None or len(issue_times) == 0:
        print("[final_weather] no issue_times found")
        return {"FINAL_WEATHER_OK": False}

    _, _, timestamps, _ = load_training_data()
    ts_all = pd.to_datetime(timestamps, errors="coerce", utc=True)

    cutoff = pd.Timestamp("2022-06-08", tz="UTC")
    valid_ts = ts_all[ts_all >= cutoff]
    if valid_ts.empty:
        print("[final_weather] no timestamps >= 2022-06-08 found")
        return {"FINAL_WEATHER_OK": False}

    n_check = min(300, len(valid_ts))
    idxs = np.linspace(0, len(valid_ts) - 1, n_check).astype(int)
    t0s = valid_ts[idxs]

    fracs = []
    all_nan_count = 0

    for t0 in t0s:
        t0_str = str(pd.Timestamp(t0).tz_convert("UTC"))
        ht = build_horizon_times(t0_str)

        wf = build_weather_features_asof(
            horizon_times=ht,
            forecast_start=t0_str,
            runs=runs,
            issue_times=issue_times,
        )
        finite_frac = float(np.isfinite(wf).mean())
        fracs.append(finite_frac)

        if finite_frac < 1e-6:
            all_nan_count += 1

    mean_frac = float(np.mean(fracs)) if fracs else 0.0
    min_frac = float(np.min(fracs)) if fracs else 0.0
    pct_all_nan = 100.0 * all_nan_count / max(len(fracs), 1)

    print("final_weather_finite_mean:", mean_frac)
    print("final_weather_finite_min :", min_frac)
    print("final_weather_all_nan_pct:", pct_all_nan)

    ok = (mean_frac > 0.85) and (pct_all_nan < 10.0)
    return {"FINAL_WEATHER_OK": ok}


def main() -> None:
    flags = {}
    flags.update(_train_npz_checks())
    flags.update(_obs_checks())
    flags.update(_forecast_checks())
    flags.update(_leak_check())
    flags.update(_asof_check())
    flags.update(_final_weather_feature_coverage_check())

    _print_header("PASS/FAIL SUMMARY")
    for k, v in flags.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
