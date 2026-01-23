#!/usr/bin/env python3
"""
Local validation harness that simulates the EXACT API call format.

This script tests the inference pipeline with per-station weather data
(21 columns for observations, 11 columns for forecasts) to ensure
everything works correctly before deploying.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import predict, baseline_model, HISTORY_LENGTH, HORIZON
from utils import load_training_data, compute_score, normalize_timestamps


def load_raw_weather_observations(data_dir: Path) -> pd.DataFrame:
    """Load raw weather observations (21 columns per-station format)."""
    import zipfile
    
    with zipfile.ZipFile(data_dir / 'weather_observations.zip', 'r') as z:
        with z.open('weather_observations.csv') as f:
            df = pd.read_csv(f, low_memory=False)
    
    # Parse timestamps
    df['timi'] = pd.to_datetime(df['timi'], utc=True, errors='coerce')
    return df


def load_raw_weather_forecasts(data_dir: Path) -> pd.DataFrame:
    """Load raw weather forecasts (11 columns per-station format)."""
    import zipfile
    
    with zipfile.ZipFile(data_dir / 'weather_forecasts.zip', 'r') as z:
        with z.open('weather_forecasts.csv') as f:
            df = pd.read_csv(f, low_memory=False)
    
    # Parse timestamps
    df['date_time'] = pd.to_datetime(df['date_time'], utc=True, errors='coerce')
    df['value_date'] = pd.to_datetime(df['value_date'], utc=True, errors='coerce')
    return df


def build_api_weather_history(obs_df: pd.DataFrame, forecast_start: str) -> np.ndarray:
    """
    Build weather history in the exact format the API sends (21 columns per-station).
    
    Returns numpy array matching what the API would send.
    """
    t0 = pd.to_datetime(forecast_start, utc=True)
    hist_start = t0 - pd.Timedelta(hours=HISTORY_LENGTH)
    
    # Filter to relevant time window
    mask = (obs_df['timi'] >= hist_start) & (obs_df['timi'] < t0)
    df_window = obs_df[mask].copy()
    
    if len(df_window) == 0:
        print(f"WARNING: No weather observations in time window [{hist_start}, {t0})")
        return None
    
    # Convert to array in the same format API sends
    # Columns must match: stod, timi, f, fg, fsdev, d, dsdev, t, tx, tn, rh, td, p, r, tg, tng, 
    #                     _rescued_data, value_date, lh_created_date, lh_modified_date, lh_is_deleted
    cols = ['stod', 'timi', 'f', 'fg', 'fsdev', 'd', 'dsdev', 't', 'tx', 'tn', 
            'rh', 'td', 'p', 'r', 'tg', 'tng', '_rescued_data', 'value_date',
            'lh_created_date', 'lh_modified_date', 'lh_is_deleted']
    
    # Convert timestamps to strings (as API sends them)
    df_out = df_window[cols].copy()
    df_out['timi'] = df_out['timi'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
    
    return df_out.values


def build_api_weather_forecast(fcst_df: pd.DataFrame, forecast_start: str) -> np.ndarray:
    """
    Build weather forecast in the exact format the API sends (11 columns per-station).
    
    Returns numpy array matching what the API would send.
    """
    t0 = pd.to_datetime(forecast_start, utc=True)
    t_end = t0 + pd.Timedelta(hours=HORIZON)
    
    # Filter to relevant time window (forecasts FOR the horizon period)
    # date_time is the time being predicted, value_date is when forecast was issued
    mask = (fcst_df['date_time'] >= t0) & (fcst_df['date_time'] < t_end)
    
    # Only use forecasts issued BEFORE the forecast start (no data leakage)
    mask = mask & (fcst_df['value_date'] < t0)
    
    df_window = fcst_df[mask].copy()
    
    if len(df_window) == 0:
        print(f"WARNING: No weather forecasts in time window [{t0}, {t_end})")
        return None
    
    # Get the most recent forecast run
    df_window['value_date_norm'] = df_window['value_date'].dt.floor('h')
    latest_issue = df_window['value_date_norm'].max()
    df_latest = df_window[df_window['value_date_norm'] == latest_issue]
    
    # Columns: date_time, station_id, temperature, windspeed, cloud_coverage, gust, 
    #          humidity, winddirection, dewpoint, rain_accumulated, value_date
    cols = ['date_time', 'station_id', 'temperature', 'windspeed', 'cloud_coverage',
            'gust', 'humidity', 'winddirection', 'dewpoint', 'rain_accumulated', 'value_date']
    
    df_out = df_latest[cols].copy()
    df_out['date_time'] = df_out['date_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_out['value_date'] = df_out['value_date'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    return df_out.values


def _call_predict_api(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: np.ndarray | None,
    weather_history: np.ndarray | None,
) -> np.ndarray:
    from fastapi.testclient import TestClient
    from new_api import app

    client = TestClient(app)
    def _sanitize(obj):
        if isinstance(obj, list):
            return [_sanitize(x) for x in obj]
        if pd.isna(obj):
            return None
        if isinstance(obj, (float, np.floating)):
            return obj if np.isfinite(obj) else None
        return obj

    def _to_jsonable(arr: np.ndarray) -> list:
        return _sanitize(np.asarray(arr, dtype=object).tolist())

    payload = {
        "sensor_history": _to_jsonable(sensor_history),
        "timestamp": timestamp,
    }
    if weather_forecast is not None:
        payload["weather_forecast"] = _to_jsonable(weather_forecast)
    if weather_history is not None:
        payload["weather_history"] = _to_jsonable(weather_history)
    resp = client.post("/predict", json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    return np.asarray(data["predictions"], dtype=np.float32)


def test_pipeline(n_test: int, use_api: bool, cutoff: str, start: str | None, end: str | None):
    """Test the inference pipeline with real data in API format."""
    print("=" * 60)
    print("LOCAL VALIDATION HARNESS - Testing API Pipeline")
    print("=" * 60)
    
    # Load training data
    data_dir = ROOT / "data"
    print("\n[1] Loading training data...")
    X, y, timestamps, sensor_names = load_training_data()
    timestamps = normalize_timestamps(timestamps)
    print(f"    Loaded {len(X)} samples")
    
    # Load raw weather data
    print("\n[2] Loading raw weather data...")
    obs_df = load_raw_weather_observations(data_dir)
    fcst_df = load_raw_weather_forecasts(data_dir)
    print(f"    Observations: {len(obs_df)} rows, {len(obs_df.columns)} columns")
    print(f"    Forecasts: {len(fcst_df)} rows, {len(fcst_df.columns)} columns")
    
    # Filter to post-2022 samples (where we have forecast data)
    ts_pd = pd.to_datetime(timestamps, utc=True, errors='coerce')
    cutoff_dt = pd.to_datetime(cutoff, utc=True) if cutoff else None
    start_dt = pd.to_datetime(start, utc=True) if start else None
    end_dt = pd.to_datetime(end, utc=True) if end else None

    mask = np.ones(len(ts_pd), dtype=bool)
    if cutoff_dt is not None:
        mask &= ts_pd >= cutoff_dt
    if start_dt is not None:
        mask &= ts_pd >= start_dt
    if end_dt is not None:
        mask &= ts_pd < end_dt

    post_ids = np.where(mask)[0]
    range_desc = []
    if start_dt is not None:
        range_desc.append(f"start={start_dt}")
    if end_dt is not None:
        range_desc.append(f"end={end_dt}")
    if cutoff_dt is not None:
        range_desc.append(f"cutoff={cutoff_dt}")
    range_desc = ", ".join(range_desc) if range_desc else "no filters"
    print(f"\n[3] Found {len(post_ids)} samples ({range_desc})")
    
    # Test on a few samples
    n_test = min(n_test, len(post_ids))
    test_ids = post_ids[-n_test:]  # Use last N samples (most recent)
    
    print(f"\n[4] Testing {n_test} samples...")
    print("-" * 60)
    
    all_scores = []
    baseline_scores = []
    
    for i, idx in enumerate(test_ids):
        ts = timestamps[idx]
        sensor_history = X[idx]  # (672, 45)
        y_true = y[idx]  # (72, 45)
        
        # Build API-format weather data
        weather_history = build_api_weather_history(obs_df, ts)
        weather_forecast = build_api_weather_forecast(fcst_df, ts)
        
        # Debug: check what we got
        obs_shape = weather_history.shape if weather_history is not None else None
        fcst_shape = weather_forecast.shape if weather_forecast is not None else None
        
        # Call predict (the actual inference function)
        try:
            if use_api:
                predictions = _call_predict_api(
                    sensor_history=sensor_history,
                    timestamp=ts,
                    weather_forecast=weather_forecast,
                    weather_history=weather_history,
                )
            else:
                predictions = predict(
                    sensor_history=sensor_history,
                    timestamp=ts,
                    weather_forecast=weather_forecast,
                    weather_history=weather_history,
                )
            
            # Validate output
            assert predictions.shape == (HORIZON, 45), f"Wrong shape: {predictions.shape}"
            assert not np.any(np.isnan(predictions)), "Predictions contain NaN!"
            assert not np.any(np.isinf(predictions)), "Predictions contain Inf!"
            assert np.all(predictions >= 0), "Negative predictions!"
            
            # Compute score for this sample
            rmse_model = np.sqrt(np.mean((y_true - predictions) ** 2))
            
            # Baseline comparison
            baseline = baseline_model(sensor_history)
            rmse_baseline = np.sqrt(np.mean((y_true - baseline) ** 2))
            
            skill = 1 - rmse_model / rmse_baseline if rmse_baseline > 0 else 0.0
            all_scores.append(skill)
            baseline_scores.append(rmse_baseline)
            
            status = "OK" if skill > 0 else "WARN (worse than baseline)"
            print(f"    Sample {i+1}/{n_test}: ts={ts[:19]} | "
                  f"obs_rows={obs_shape[0] if obs_shape else 'None':>5} | "
                  f"fcst_rows={fcst_shape[0] if fcst_shape else 'None':>4} | "
                  f"skill={skill:.4f} | {status}")
            
        except Exception as e:
            print(f"    Sample {i+1}/{n_test}: ts={ts[:19]} | ERROR: {e}")
            all_scores.append(None)
    
    print("-" * 60)
    
    # Summary
    valid_scores = [s for s in all_scores if s is not None]
    if valid_scores:
        print(f"\n[5] SUMMARY:")
        print(f"    Samples tested: {n_test}")
        print(f"    Successful: {len(valid_scores)}")
        print(f"    Mean skill: {np.mean(valid_scores):.4f}")
        print(f"    Min skill: {np.min(valid_scores):.4f}")
        print(f"    Max skill: {np.max(valid_scores):.4f}")
        
        if np.mean(valid_scores) < 0:
            print("\n    WARNING: Model performing worse than baseline!")
        elif np.mean(valid_scores) < 0.2:
            print("\n    WARNING: Model performing below expected (~0.3+)")
        else:
            print("\n    SUCCESS: Model performing reasonably well")
    else:
        print("\n    ERROR: All samples failed!")
    
    return valid_scores


def test_weather_feature_extraction():
    """Test weather feature extraction specifically."""
    print("\n" + "=" * 60)
    print("Testing Weather Feature Extraction")
    print("=" * 60)
    
    from feature_utils import (
        build_weather_obs_summary_features,
        build_weather_features,
        WEATHER_OBS_COLS,
        WEATHER_NUMERIC_COLS,
    )
    
    data_dir = ROOT / "data"
    obs_df = load_raw_weather_observations(data_dir)
    
    # Pick a timestamp
    ts = "2023-06-15T12:00:00Z"
    
    # Build weather history in API format
    weather_history = build_api_weather_history(obs_df, ts)
    
    if weather_history is not None:
        print(f"\n1. Weather history array:")
        print(f"   Shape: {weather_history.shape}")
        print(f"   Expected columns (21): stod, timi, f, fg, fsdev, d, dsdev, t, tx, tn,")
        print(f"                          rh, td, p, r, tg, tng, _rescued_data, value_date,")
        print(f"                          lh_created_date, lh_modified_date, lh_is_deleted")
        print(f"   Sample row: {weather_history[0][:5]}...")
        
        # Test the feature extraction
        print(f"\n2. Calling build_weather_obs_summary_features()...")
        obs_feats = build_weather_obs_summary_features(ts, weather_history, obs_agg=None)
        print(f"   Output shape: {obs_feats.shape}")
        print(f"   Expected: (72, {len(WEATHER_OBS_COLS) * 3})")
        print(f"   NaN count: {np.isnan(obs_feats).sum()} / {obs_feats.size}")
        print(f"   Sample values: {obs_feats[0, :6]}")
        
        if np.all(np.isnan(obs_feats)):
            print("\n   ERROR: All features are NaN! Something is wrong with parsing.")
        elif np.isnan(obs_feats).sum() > obs_feats.size * 0.5:
            print("\n   WARNING: More than 50% of features are NaN")
        else:
            print("\n   OK: Features extracted successfully")
    else:
        print("   ERROR: Could not build weather history")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate API-format inference.")
    parser.add_argument("--n-test", type=int, default=10, help="Number of samples to test.")
    parser.add_argument("--all", action="store_true", help="Test all eligible samples.")
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Call the FastAPI /predict endpoint instead of predict() directly.",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2022-07-01T00:00:00Z",
        help="Only test samples on/after this timestamp.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start timestamp (inclusive) for testing window.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End timestamp (exclusive) for testing window.",
    )
    parser.add_argument(
        "--debug-model",
        action="store_true",
        help="Enable model debug logging (prints active model and paths once).",
    )
    args = parser.parse_args()

    if args.debug_model:
        os.environ["MODEL_DEBUG"] = "1"

    test_weather_feature_extraction()

    n_test = 10 if args.n_test is None else args.n_test
    if args.all:
        n_test = 10**9
    test_pipeline(
        n_test=n_test,
        use_api=args.use_api,
        cutoff=args.cutoff,
        start=args.start,
        end=args.end,
    )
