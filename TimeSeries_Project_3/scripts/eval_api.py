#!/usr/bin/env python3
"""
Evaluate hosted API by sending realistic request payloads.

Simulates what the competition website would send:
- sensor_history: (672, 45) from train_full.npz
- timestamp: ISO 8601 string
- weather_forecast: rows from weather_forecasts.csv for the 72h forecast window
- weather_history: rows from weather_observations.csv for the 672h history window
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

from utils import compute_score, normalize_timestamps


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
    
    # Parse timestamps
    df["_target_time"] = pd.to_datetime(df["date_time"], errors="coerce", utc=True)
    df["_target_time"] = df["_target_time"].dt.tz_localize(None)
    df["_issue_time"] = pd.to_datetime(df["value_date"], errors="coerce", utc=True)
    df["_issue_time"] = df["_issue_time"].dt.tz_localize(None)
    
    # Filter to forecast window
    horizon_end = forecast_start + pd.Timedelta(hours=horizon_hours)
    mask = (
        (df["_target_time"] >= forecast_start) &
        (df["_target_time"] < horizon_end) &
        (df["_issue_time"] <= forecast_start)  # No future leakage
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
    
    # Parse timestamps
    df["_obs_time"] = pd.to_datetime(df["timi"], errors="coerce", utc=True)
    df["_obs_time"] = df["_obs_time"].dt.tz_localize(None)
    
    # Filter to history window
    history_start = forecast_start - pd.Timedelta(hours=history_hours)
    mask = (
        (df["_obs_time"] >= history_start) &
        (df["_obs_time"] < forecast_start)
    )
    
    filtered = df.loc[mask].drop(columns=["_obs_time"])
    return _df_rows_to_list(filtered)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval via hosted API (simulates website calls)."
    )
    parser.add_argument("--train-path", default="data/train_full.npz")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--api-url", default="http://0.0.0.0:8080/predict")
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-weather", action="store_true", help="Skip weather data")
    args = parser.parse_args()

    data_path = Path(args.train_path)
    data_dir = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    # Load training samples
    data = np.load(data_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    timestamps = normalize_timestamps(data["timestamps"])

    idx = _pick_indices(len(X), args.max_samples)
    X = X[idx]
    y = y[idx]
    timestamps = timestamps[idx]

    # Load weather CSVs (once)
    forecasts_df = None
    observations_df = None
    if not args.no_weather:
        print("Loading weather CSVs...")
        forecasts_df = pd.read_csv(data_dir / "weather_forecasts.csv", low_memory=False)
        observations_df = pd.read_csv(data_dir / "weather_observations.csv", low_memory=False)
        print(f"  forecasts: {len(forecasts_df)} rows")
        print(f"  observations: {len(observations_df)} rows")

    print(f"\nEvaluating {len(X)} samples via API...")
    preds = []
    scores = []
    start = perf_counter()

    with httpx.Client(timeout=args.timeout) as client:
        for i, (x, ts) in enumerate(zip(X, timestamps), start=1):
            # Parse timestamp
            forecast_start = pd.Timestamp(str(ts)).floor("h")
            
            # Build weather arrays for this specific time window
            weather_forecast = None
            weather_history = None
            if forecasts_df is not None:
                weather_forecast = _filter_weather_forecast(forecasts_df, forecast_start)
            if observations_df is not None:
                weather_history = _filter_weather_history(observations_df, forecast_start)

            # Clean sensor history (replace NaN with 0 for JSON)
            x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            payload = {
                "sensor_history": x_clean.tolist(),
                "timestamp": str(ts),
                "weather_forecast": weather_forecast,
                "weather_history": weather_history,
            }

            resp = client.post(args.api_url, json=payload)
            resp.raise_for_status()
            pred = np.asarray(resp.json()["predictions"], dtype=np.float32)
            preds.append(pred)

            # Compute running score
            y_so_far = y[:i]
            preds_so_far = np.array(preds, dtype=np.float32)
            X_so_far = X[:i]
            score_so_far = compute_score(y_so_far, preds_so_far, X_so_far)
            scores.append(score_so_far)

            wf_rows = len(weather_forecast) if weather_forecast else 0
            wh_rows = len(weather_history) if weather_history else 0
            print(f"  [{i:3d}/{len(X)}] ts={ts[:16]} | wf={wf_rows:4d} wh={wh_rows:5d} | running skill={score_so_far:.4f}")

    elapsed = perf_counter() - start

    preds = np.asarray(preds, dtype=np.float32)
    final_score = compute_score(y, preds, X)
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))

    print("\n" + "=" * 60)
    print("API eval results (biased; uses training samples):")
    print(f"  samples: {len(X)}")
    print(f"  skill:   {final_score:.4f}")
    print(f"  rmse:    {rmse:.4f}")
    print(f"  time:    {elapsed:.2f}s ({elapsed/len(X):.2f}s per sample)")
    print("=" * 60)


if __name__ == "__main__":
    main()
