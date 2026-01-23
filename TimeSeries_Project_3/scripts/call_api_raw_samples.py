#!/usr/bin/env python3
"""
Call the running API with raw data-derived samples.
Builds API-format payloads from sensor_timeseries.csv + weather CSVs.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

HISTORY_LENGTH = 672
HORIZON = 72


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call API with raw data samples")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", type=str, default="2022-07-01")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--api-url", type=str, default="http://localhost:8080/predict")
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def load_sensor_data(data_dir: Path) -> pd.DataFrame:
    csv_path = data_dir / "sensor_timeseries.csv"
    df = pd.read_csv(csv_path)
    df["CTime"] = pd.to_datetime(df["CTime"], errors="coerce")
    df = df.sort_values("CTime").set_index("CTime")
    df = df.asfreq("h")
    return df


def load_weather_forecasts(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "weather_forecasts.csv"
    if not path.exists():
        path = data_dir / "weather_forecasts.zip"
    df = pd.read_csv(path, compression="zip" if path.suffix == ".zip" else None)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["value_date"] = pd.to_datetime(df["value_date"], errors="coerce")
    return df


def load_weather_obs(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "weather_observations.csv"
    if not path.exists():
        path = data_dir / "weather_observations.zip"
    df = pd.read_csv(path, compression="zip" if path.suffix == ".zip" else None, low_memory=False)
    # Normalize to tz-naive UTC for consistent comparisons
    df["timi"] = pd.to_datetime(df["timi"], errors="coerce", utc=True).dt.tz_localize(None)
    return df


def get_weather_forecast_for_sample(fcst_df: pd.DataFrame, forecast_start: pd.Timestamp) -> Optional[np.ndarray]:
    # Select forecasts with issue time <= forecast_start and valid time in horizon
    mask = (
        (fcst_df["date_time"] >= forecast_start)
        & (fcst_df["date_time"] < forecast_start + pd.Timedelta(hours=HORIZON))
        & (fcst_df["value_date"] <= forecast_start)
    )
    df_window = fcst_df[mask].copy()
    if len(df_window) == 0:
        return None
    df_window["value_date_norm"] = df_window["value_date"].dt.floor("h")
    latest_issue = df_window["value_date_norm"].max()
    df_latest = df_window[df_window["value_date_norm"] == latest_issue]

    cols = [
        "date_time", "station_id", "temperature", "windspeed", "cloud_coverage",
        "gust", "humidity", "winddirection", "dewpoint", "rain_accumulated", "value_date",
    ]
    available_cols = [c for c in cols if c in df_latest.columns]
    df_out = df_latest[available_cols].copy()

    if "date_time" in df_out.columns:
        df_out["date_time"] = df_out["date_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if "value_date" in df_out.columns:
        df_out["value_date"] = df_out["value_date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df_out.values


def get_weather_obs_for_sample(obs_df: pd.DataFrame, forecast_start: pd.Timestamp) -> Optional[np.ndarray]:
    t_start = forecast_start - pd.Timedelta(hours=HISTORY_LENGTH)
    t_end = forecast_start
    mask = (obs_df["timi"] >= t_start) & (obs_df["timi"] < t_end)
    df_window = obs_df[mask].copy()
    if len(df_window) == 0:
        return None
    df_window["timi"] = df_window["timi"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df_window.values


def to_jsonable(arr: Optional[np.ndarray]) -> Optional[list]:
    if arr is None:
        return None
    result = []
    for row in np.array(arr):
        row_list = []
        for v in row:
            if isinstance(v, (np.floating, float)) and (np.isnan(v) or np.isinf(v)):
                row_list.append(None)
            elif isinstance(v, np.generic):
                row_list.append(v.item())
            else:
                row_list.append(v)
        result.append(row_list)
    return result


def compute_skill(y_pred: np.ndarray, y_true: np.ndarray, y_baseline: np.ndarray) -> float:
    rmse_model = np.sqrt(np.mean((y_pred - y_true) ** 2))
    rmse_baseline = np.sqrt(np.mean((y_baseline - y_true) ** 2))
    if rmse_baseline < 1e-9:
        return np.nan
    return 1.0 - rmse_model / rmse_baseline


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("[1] Loading raw sensor data...")
    sensor_df = load_sensor_data(data_dir)

    print("[2] Loading raw weather data...")
    fcst_df = load_weather_forecasts(data_dir)
    obs_df = load_weather_obs(data_dir)

    min_date = pd.Timestamp(args.start_date)
    max_date = pd.Timestamp(args.end_date) if args.end_date else sensor_df.index.max() - pd.Timedelta(hours=HISTORY_LENGTH + HORIZON)

    all_times = pd.date_range(start=min_date, end=max_date, freq="h")
    if len(all_times) == 0:
        raise RuntimeError("No valid timestamps in range")

    selected_times = sorted(random.sample(list(all_times), min(args.n_samples, len(all_times))))

    print(f"[3] Sampling {len(selected_times)} timestamps from {min_date} to {max_date}")

    successes = 0
    failures = 0
    times = []
    skills = []

    for i, t in enumerate(selected_times):
        forecast_start = t + pd.Timedelta(hours=HISTORY_LENGTH)

        hist = sensor_df.loc[t: forecast_start - pd.Timedelta(hours=1)]
        if len(hist) != HISTORY_LENGTH:
            print(f"  [{i+1}] Skipped - history length {len(hist)}")
            continue

        targets = sensor_df.loc[forecast_start: forecast_start + pd.Timedelta(hours=HORIZON - 1)]
        if len(targets) != HORIZON:
            print(f"  [{i+1}] Skipped - target length {len(targets)}")
            continue

        sensor_history = hist.to_numpy(dtype=np.float32)
        y_true = targets.to_numpy(dtype=np.float32)
        wf = get_weather_forecast_for_sample(fcst_df, forecast_start)
        wo = get_weather_obs_for_sample(obs_df, forecast_start)

        payload = {
            "sensor_history": to_jsonable(sensor_history),
            "timestamp": forecast_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "weather_forecast": to_jsonable(wf),
            "weather_history": to_jsonable(wo),
        }

        try:
            start = pd.Timestamp.utcnow()
            resp = requests.post(args.api_url, json=payload, timeout=args.timeout)
            elapsed = (pd.Timestamp.utcnow() - start).total_seconds()
            if resp.status_code == 200:
                successes += 1
                times.append(elapsed)
                preds = np.array(resp.json().get("predictions"))
                if preds.shape == (HORIZON, sensor_history.shape[1]):
                    baseline = sensor_history[-HORIZON:, :]
                    skill = compute_skill(preds, y_true, baseline)
                    skills.append(skill)
                    print(
                        f"  [{i+1}] OK {elapsed:.2f}s | skill={skill:.3f} | "
                        f"wf={None if wf is None else wf.shape} wo={None if wo is None else wo.shape}"
                    )
                else:
                    print(
                        f"  [{i+1}] OK {elapsed:.2f}s | bad pred shape {preds.shape} | "
                        f"wf={None if wf is None else wf.shape} wo={None if wo is None else wo.shape}"
                    )
            else:
                failures += 1
                print(f"  [{i+1}] FAIL {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            failures += 1
            print(f"  [{i+1}] ERROR: {e}")

    print(f"\nDone. Success: {successes}, Failures: {failures}")
    if times:
        print(f"Latency: mean={np.mean(times):.2f}s, p95={np.percentile(times, 95):.2f}s")
    if skills:
        skills_arr = np.array(skills, dtype=np.float64)
        print(
            f"Skill: mean={np.nanmean(skills_arr):.4f}, std={np.nanstd(skills_arr):.4f}, "
            f"neg%={(skills_arr < 0).mean() * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
