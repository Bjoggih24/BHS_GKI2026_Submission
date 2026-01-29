#!/usr/bin/env python3
"""
Evaluate hosted API by sending realistic request payloads.

Simulates EXACTLY what the competition website would send:
- sensor_history: (672, 45) sliced directly from raw sensor_timeseries.csv
- timestamp: ISO 8601 string
- weather_forecast: rows from weather_forecasts.csv for the 72h forecast window
- weather_history: rows from weather_observations.csv for the 672h history window

This is the most accurate local reproduction of the competition eval flow.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*DtypeWarning.*")
warnings.filterwarnings("ignore", message=".*mixed types.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import compute_score

HISTORY_LENGTH = 672
HORIZON = 72


def _pick_indices(n: int, max_samples: int) -> np.ndarray:
    if max_samples <= 0 or max_samples >= n:
        return np.arange(n)
    if max_samples == 1:
        return np.array([n // 2])
    return np.linspace(0, n - 1, max_samples, dtype=int)


def _safe_value(val):
    """Convert value for JSON: NaN/inf -> None, numpy types -> python types."""
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(val, (np.floating, np.integer)):
        v = val.item()
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if pd.isna(val):
        return None
    return val


def _df_rows_to_list(df: pd.DataFrame) -> list[list]:
    """Convert DataFrame rows to nested list, handling NaN for JSON."""
    return [[_safe_value(v) for v in row] for row in df.values]


def _load_sensor_timeseries(data_dir: Path) -> pd.DataFrame:
    """Load raw sensor timeseries and reindex to hourly."""
    csv_path = data_dir / "sensor_timeseries.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    df["CTime"] = pd.to_datetime(df["CTime"], errors="coerce")
    df = df.sort_values("CTime").set_index("CTime")
    df = df.asfreq("h")  # Reindex to hourly, exposing gaps as NaN
    return df


def _slice_sensor_history(sensor_df: pd.DataFrame, forecast_start: pd.Timestamp) -> np.ndarray:
    """
    Slice 672 hours of sensor history ending at forecast_start - 1h.
    Returns (672, N_SENSORS) array with NaN replaced by 0 for JSON.
    """
    hist_end = forecast_start - pd.Timedelta(hours=1)
    hist_start = forecast_start - pd.Timedelta(hours=HISTORY_LENGTH)
    try:
        sensor_hist = sensor_df.loc[hist_start:hist_end]
    except Exception:
        return None
    if len(sensor_hist) != HISTORY_LENGTH:
        return None
    arr = sensor_hist.values.astype(np.float32)
    # Replace NaN with 0 for JSON serialization (matches real API behavior)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _slice_targets(sensor_df: pd.DataFrame, forecast_start: pd.Timestamp) -> np.ndarray:
    """
    Slice 72 hours of targets starting at forecast_start.
    Returns (72, N_SENSORS) array or None if incomplete.
    """
    target_end = forecast_start + pd.Timedelta(hours=HORIZON - 1)
    try:
        targets = sensor_df.loc[forecast_start:target_end]
    except Exception:
        return None
    if len(targets) != HORIZON:
        return None
    return targets.values.astype(np.float32)


def _filter_weather_forecast(
    forecasts_df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    horizon_hours: int = 72,
) -> list[list]:
    """
    Filter weather_forecasts.csv to rows for the 72h forecast window.
    
    - date_time (col 0): the target time the forecast is FOR
    - value_date (col 10): when the forecast was issued (must be <= forecast_start)
    
    Returns rows where date_time is in [forecast_start, forecast_start + 72h)
    and value_date <= forecast_start (no future leakage).
    """
    df = forecasts_df.copy()
    
    df["_target_time"] = pd.to_datetime(df["date_time"], errors="coerce", utc=True)
    df["_target_time"] = df["_target_time"].dt.tz_localize(None)
    df["_issue_time"] = pd.to_datetime(df["value_date"], errors="coerce", utc=True)
    df["_issue_time"] = df["_issue_time"].dt.tz_localize(None)
    
    horizon_end = forecast_start + pd.Timedelta(hours=horizon_hours)
    mask = (
        (df["_target_time"] >= forecast_start) &
        (df["_target_time"] < horizon_end) &
        (df["_issue_time"] <= forecast_start)
    )
    
    filtered = df.loc[mask].drop(columns=["_target_time", "_issue_time"])
    return _df_rows_to_list(filtered)


def _filter_weather_history(
    observations_df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    history_hours: int = 672,
) -> list[list]:
    """
    Filter weather_observations.csv to rows for the 672h history window.
    
    - timi (col 1): observation timestamp
    
    Returns rows where timi is in [forecast_start - 672h, forecast_start).
    """
    df = observations_df.copy()
    
    df["_obs_time"] = pd.to_datetime(df["timi"], errors="coerce", utc=True)
    df["_obs_time"] = df["_obs_time"].dt.tz_localize(None)
    
    history_start = forecast_start - pd.Timedelta(hours=history_hours)
    mask = (
        (df["_obs_time"] >= history_start) &
        (df["_obs_time"] < forecast_start)
    )
    
    filtered = df.loc[mask].drop(columns=["_obs_time"])
    return _df_rows_to_list(filtered)


def _generate_eval_timestamps(sensor_df: pd.DataFrame, stride: int, min_date: str, max_date: str) -> list[pd.Timestamp]:
    """Generate evaluation timestamps matching train_full.npz generation logic."""
    min_dt = pd.Timestamp(min_date).floor("h")
    max_dt = pd.Timestamp(max_date).floor("h")
    
    earliest_start = max(min_dt, sensor_df.index.min() + pd.Timedelta(hours=HISTORY_LENGTH))
    latest_start = min(max_dt, sensor_df.index.max() - pd.Timedelta(hours=HORIZON))
    
    timestamps = pd.date_range(start=earliest_start, end=latest_start, freq=f"{stride}h")
    return list(timestamps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval via hosted API using RAW data (simulates website exactly)."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--api-url", default="http://0.0.0.0:8080/predict")
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--stride", type=int, default=24, help="Stride between samples (hours)")
    parser.add_argument("--min-date", default="2022-07-01", help="Start date for samples")
    parser.add_argument("--max-date", default="2024-12-31", help="End date for samples")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-weather", action="store_true", help="Skip weather data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load raw sensor data
    print("Loading raw sensor_timeseries.csv...")
    sensor_df = _load_sensor_timeseries(data_dir)
    print(f"  sensors: {len(sensor_df.columns)} columns")
    print(f"  range: {sensor_df.index.min()} -> {sensor_df.index.max()}")

    # Generate eval timestamps
    all_timestamps = _generate_eval_timestamps(sensor_df, args.stride, args.min_date, args.max_date)
    print(f"  potential samples: {len(all_timestamps)}")

    # Pick subset
    idx = _pick_indices(len(all_timestamps), args.max_samples)
    eval_timestamps = [all_timestamps[i] for i in idx]
    print(f"  evaluating: {len(eval_timestamps)} samples")

    # Load weather CSVs (once)
    forecasts_df = None
    observations_df = None
    if not args.no_weather:
        print("\nLoading weather CSVs...")
        forecasts_df = pd.read_csv(data_dir / "weather_forecasts.csv", low_memory=False)
        observations_df = pd.read_csv(data_dir / "weather_observations.csv", low_memory=False)
        print(f"  forecasts: {len(forecasts_df)} rows")
        print(f"  observations: {len(observations_df)} rows")

    print(f"\nEvaluating {len(eval_timestamps)} samples via API (RAW data)...")
    preds_list = []
    y_list = []
    X_list = []
    start = perf_counter()

    with httpx.Client(timeout=args.timeout) as client:
        for i, forecast_start in enumerate(eval_timestamps, start=1):
            forecast_start = forecast_start.floor("h")
            
            # Slice sensor history directly from raw CSV
            sensor_history = _slice_sensor_history(sensor_df, forecast_start)
            if sensor_history is None:
                print(f"  [{i:3d}/{len(eval_timestamps)}] ts={forecast_start} | SKIPPED (sensor gap)")
                continue
            
            # Slice targets for scoring
            targets = _slice_targets(sensor_df, forecast_start)
            if targets is None:
                print(f"  [{i:3d}/{len(eval_timestamps)}] ts={forecast_start} | SKIPPED (target gap)")
                continue
            
            # Skip if targets have NaN (can't score)
            if np.isnan(targets).any():
                print(f"  [{i:3d}/{len(eval_timestamps)}] ts={forecast_start} | SKIPPED (target NaN)")
                continue

            # Build weather arrays for this specific time window
            weather_forecast = None
            weather_history = None
            if forecasts_df is not None:
                weather_forecast = _filter_weather_forecast(forecasts_df, forecast_start)
            if observations_df is not None:
                weather_history = _filter_weather_history(observations_df, forecast_start)

            payload = {
                "sensor_history": sensor_history.tolist(),
                "timestamp": forecast_start.isoformat(),
                "weather_forecast": weather_forecast,
                "weather_history": weather_history,
            }

            resp = client.post(args.api_url, json=payload)
            resp.raise_for_status()
            pred = np.asarray(resp.json()["predictions"], dtype=np.float32)
            
            preds_list.append(pred)
            y_list.append(targets)
            X_list.append(sensor_history)

            # Compute running score
            y_so_far = np.array(y_list, dtype=np.float32)
            preds_so_far = np.array(preds_list, dtype=np.float32)
            X_so_far = np.array(X_list, dtype=np.float32)
            score_so_far = compute_score(y_so_far, preds_so_far, X_so_far)

            wf_rows = len(weather_forecast) if weather_forecast else 0
            wh_rows = len(weather_history) if weather_history else 0
            ts_str = forecast_start.strftime("%Y-%m-%dT%H:%M")
            print(f"  [{i:3d}/{len(eval_timestamps)}] ts={ts_str} | wf={wf_rows:5d} wh={wh_rows:5d} | running skill={score_so_far:.4f}")

    elapsed = perf_counter() - start

    if len(preds_list) == 0:
        print("\nNo valid samples evaluated!")
        return

    preds = np.asarray(preds_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    X = np.asarray(X_list, dtype=np.float32)
    
    final_score = compute_score(y, preds, X)
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))

    print("\n" + "=" * 60)
    print("API eval results (RAW data, biased since uses training period):")
    print(f"  samples: {len(preds)}")
    print(f"  skill:   {final_score:.4f}")
    print(f"  rmse:    {rmse:.4f}")
    print(f"  time:    {elapsed:.2f}s ({elapsed/len(preds):.2f}s per sample)")
    print("=" * 60)


if __name__ == "__main__":
    main()
