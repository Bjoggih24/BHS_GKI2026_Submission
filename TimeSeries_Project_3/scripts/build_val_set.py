"""
Build a clean validation set from scratch, mimicking real inference.

Uses data after 2022-06 when weather forecasts become available.
Each sample has:
  - 672 hours of sensor history (X)
  - 72 hours of targets (y)  
  - 672 hours of weather observations history
  - 72 hours of weather forecasts

Usage:
    python scripts/build_val_set.py --n-samples 100 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd


HISTORY_LENGTH = 672  # 4 weeks
HORIZON = 72          # 3 days
MIN_DATE = "2022-07-01"  # After weather forecasts start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build validation set from scratch")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of validation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/val_clean.npz", help="Output file")
    return parser.parse_args()


def load_sensor_data(data_dir: Path) -> pd.DataFrame:
    """Load raw sensor timeseries."""
    csv_path = data_dir / "sensor_timeseries.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    
    df = pd.read_csv(csv_path)
    df["CTime"] = pd.to_datetime(df["CTime"], errors="coerce")
    df = df.sort_values("CTime").set_index("CTime")
    
    # Reindex to hourly to expose gaps
    df = df.asfreq("h")
    return df


def load_weather_forecasts(data_dir: Path) -> pd.DataFrame:
    """Load weather forecasts."""
    candidates = [
        data_dir / "weather_forecasts.csv",
        data_dir / "weather_forecasts.zip",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Missing weather_forecasts.csv or .zip")
    
    df = pd.read_csv(path, compression="zip" if path.suffix == ".zip" else None)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce", utc=True).dt.tz_localize(None)
    df["value_date"] = pd.to_datetime(df["value_date"], errors="coerce", utc=True).dt.tz_localize(None)
    return df


def load_weather_obs(data_dir: Path) -> pd.DataFrame:
    """Load weather observations."""
    candidates = [
        data_dir / "weather_observations.csv",
        data_dir / "weather_observations.zip",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Missing weather_observations.csv or .zip")
    
    df = pd.read_csv(path, compression="zip" if path.suffix == ".zip" else None)
    df["timi"] = pd.to_datetime(df["timi"], errors="coerce", utc=True).dt.tz_localize(None)
    return df


def get_weather_forecast_for_sample(
    fcst_df: pd.DataFrame, 
    forecast_start: pd.Timestamp,
    horizon: int = HORIZON
) -> np.ndarray | None:
    """
    Get weather forecast array for a sample, mimicking API format.
    Returns array of shape (~936, 11) or None if not available.
    """
    t_end = forecast_start + pd.Timedelta(hours=horizon)
    
    # Filter to forecasts FOR the horizon period, issued BEFORE forecast_start
    mask = (
        (fcst_df["date_time"] >= forecast_start) & 
        (fcst_df["date_time"] < t_end) &
        (fcst_df["value_date"] < forecast_start)
    )
    df_window = fcst_df[mask].copy()
    
    if len(df_window) == 0:
        return None
    
    # Get most recent forecast run
    df_window["value_date_norm"] = df_window["value_date"].dt.floor("h")
    latest_issue = df_window["value_date_norm"].max()
    df_latest = df_window[df_window["value_date_norm"] == latest_issue]
    
    # API format columns
    cols = ['date_time', 'station_id', 'temperature', 'windspeed', 'cloud_coverage',
            'gust', 'humidity', 'winddirection', 'dewpoint', 'rain_accumulated', 'value_date']
    
    # Filter to available columns
    available_cols = [c for c in cols if c in df_latest.columns]
    df_out = df_latest[available_cols].copy()
    
    # Convert timestamps to strings (like API sends)
    if 'date_time' in df_out.columns:
        df_out['date_time'] = df_out['date_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    if 'value_date' in df_out.columns:
        df_out['value_date'] = df_out['value_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df_out.values


def get_weather_obs_for_sample(
    obs_df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    history_length: int = HISTORY_LENGTH
) -> np.ndarray | None:
    """
    Get weather observations array for a sample, mimicking API format.
    Returns array of shape (N, 21) or None if not available.
    """
    t_start = forecast_start - pd.Timedelta(hours=history_length)
    t_end = forecast_start
    
    mask = (obs_df["timi"] >= t_start) & (obs_df["timi"] < t_end)
    df_window = obs_df[mask].copy()
    
    if len(df_window) == 0:
        return None
    
    # Convert timestamp to string (like API sends)
    df_window["timi"] = df_window["timi"].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df_window.values


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("[1] Loading sensor data...")
    sensor_df = load_sensor_data(data_dir)
    sensor_names = list(sensor_df.columns)
    print(f"    Sensors: {len(sensor_names)}, Range: {sensor_df.index.min()} -> {sensor_df.index.max()}")
    
    print("[2] Loading weather forecasts...")
    fcst_df = load_weather_forecasts(data_dir)
    print(f"    Forecasts: {len(fcst_df)} rows")
    
    print("[3] Loading weather observations...")
    obs_df = load_weather_obs(data_dir)
    print(f"    Observations: {len(obs_df)} rows")
    
    # Find valid sample start times
    min_date = pd.Timestamp(MIN_DATE)
    max_date = sensor_df.index.max() - pd.Timedelta(hours=HISTORY_LENGTH + HORIZON)
    
    # Get all possible hourly start times
    all_times = pd.date_range(start=min_date, end=max_date, freq="h")
    print(f"[4] Possible sample times: {len(all_times)} (from {min_date} to {max_date})")
    
    # Randomly select n_samples
    if len(all_times) < args.n_samples:
        print(f"    Warning: Only {len(all_times)} possible times, using all")
        selected_times = all_times.tolist()
    else:
        selected_times = random.sample(list(all_times), args.n_samples)
    selected_times = sorted(selected_times)
    
    print(f"[5] Building {len(selected_times)} validation samples...")
    
    X_list = []
    y_list = []
    ts_list = []
    weather_fcst_list = []
    weather_obs_list = []
    skipped = 0
    
    for i, t in enumerate(selected_times):
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(selected_times)}...")
        
        # Forecast start is end of history
        forecast_start = t + pd.Timedelta(hours=HISTORY_LENGTH)
        
        # Get sensor history (672 hours before forecast_start)
        hist_start = t
        hist_end = forecast_start
        sensor_hist = sensor_df.loc[hist_start:hist_end - pd.Timedelta(hours=1)]
        
        if len(sensor_hist) != HISTORY_LENGTH:
            skipped += 1
            continue
        
        # Get targets (72 hours after forecast_start)
        target_start = forecast_start
        target_end = forecast_start + pd.Timedelta(hours=HORIZON)
        targets = sensor_df.loc[target_start:target_end - pd.Timedelta(hours=1)]
        
        if len(targets) != HORIZON:
            skipped += 1
            continue
        
        # Skip if targets have NaN (can't evaluate)
        if targets.isna().any().any():
            skipped += 1
            continue
        
        # Get weather forecast (for horizon period)
        weather_fcst = get_weather_forecast_for_sample(fcst_df, forecast_start)
        if weather_fcst is None or len(weather_fcst) < 100:
            skipped += 1
            continue
        
        # Get weather observations (for history period)
        weather_obs = get_weather_obs_for_sample(obs_df, forecast_start)
        
        X_list.append(sensor_hist.values)
        y_list.append(targets.values)
        ts_list.append(forecast_start.isoformat())
        weather_fcst_list.append(weather_fcst)
        weather_obs_list.append(weather_obs)
    
    print(f"[6] Built {len(X_list)} samples (skipped {skipped})")
    
    if len(X_list) == 0:
        raise ValueError("No valid samples created!")
    
    # Convert to arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    timestamps = np.array(ts_list)
    
    # Save
    output_path = Path(args.output)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        timestamps=timestamps,
        sensor_names=np.array(sensor_names),
        # Store weather as object arrays since shapes vary
        weather_forecasts=np.array(weather_fcst_list, dtype=object),
        weather_observations=np.array(weather_obs_list, dtype=object),
    )
    
    print(f"[7] Saved to {output_path}")
    print(f"    X: {X.shape}")
    print(f"    y: {y.shape}")
    print(f"    timestamps: {len(timestamps)}")
    print(f"    Date range: {timestamps[0]} -> {timestamps[-1]}")


if __name__ == "__main__":
    main()
