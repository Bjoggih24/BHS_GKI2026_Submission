"""
Build clean training set from scratch, matching real inference scenario.

Uses data from 2022-07 onwards (when weather forecasts available).
Each sample has:
  - 672 hours of sensor history (X)
  - 72 hours of targets (y)
  - Weather forecast array (API format)
  - Weather observations array (API format)

Usage:
    python scripts/build_train_full.py
    python scripts/build_train_full.py --stride 24 --min-date 2022-07-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HISTORY_LENGTH = 672  # 4 weeks
HORIZON = 72          # 3 days
MIN_DATE_DEFAULT = "2022-07-01"  # After weather forecasts start
MAX_DATE_DEFAULT = "2024-12-31"  # End of available data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean training set")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--stride", type=int, default=24, help="Stride in hours (24=daily)")
    parser.add_argument("--min-date", type=str, default=MIN_DATE_DEFAULT, help="Start date")
    parser.add_argument("--max-date", type=str, default=MAX_DATE_DEFAULT, help="End date")
    parser.add_argument("--output", type=str, default="data/train_full.npz", help="Output file")
    # Batch processing to avoid OOM
    parser.add_argument("--batch-start", type=int, default=0, help="Start sample index (for batch processing)")
    parser.add_argument("--batch-size", type=int, default=0, help="Number of samples to process (0=all)")
    return parser.parse_args()


def load_sensor_data(data_dir: Path) -> pd.DataFrame:
    """Load raw sensor timeseries."""
    csv_path = data_dir / "sensor_timeseries.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    
    print(f"    Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df["CTime"] = pd.to_datetime(df["CTime"], errors="coerce")
    df = df.sort_values("CTime").set_index("CTime")
    
    # Reindex to hourly to expose gaps (matching API timestamp flooring)
    df = df.asfreq("h")
    return df


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    
    print("[1] Loading sensor data...")
    sensor_df = load_sensor_data(data_dir)
    sensor_names = list(sensor_df.columns)
    print(f"    Sensors: {len(sensor_names)}")
    print(f"    Range: {sensor_df.index.min()} -> {sensor_df.index.max()}")
    
    # NOTE: Weather data will be loaded on-demand during training (saves memory)
    
    # Determine valid sample range
    min_date = pd.Timestamp(args.min_date).floor("h")
    max_date = pd.Timestamp(args.max_date).floor("h")
    
    # Adjust for history and horizon requirements
    earliest_start = max(min_date, sensor_df.index.min() + pd.Timedelta(hours=HISTORY_LENGTH))
    latest_start = min(max_date, sensor_df.index.max() - pd.Timedelta(hours=HORIZON))
    
    print(f"[2] Sample range: {earliest_start} -> {latest_start}")
    print(f"    Stride: {args.stride} hours")
    
    # Generate sample timestamps
    sample_times = pd.date_range(start=earliest_start, end=latest_start, freq=f"{args.stride}h")
    total_samples = len(sample_times)
    print(f"    Potential samples: {total_samples}")
    
    # Handle batch processing
    batch_start = args.batch_start
    batch_end = total_samples if args.batch_size == 0 else min(batch_start + args.batch_size, total_samples)
    sample_times = sample_times[batch_start:batch_end]
    
    if args.batch_size > 0:
        print(f"    Processing batch: samples {batch_start} to {batch_end} ({len(sample_times)} samples)")
    
    print(f"[3] Building training samples (sensor data only, weather loaded on-demand during training)...")
    
    X_list = []
    y_list = []
    ts_list = []
    skipped_sensor = 0
    skipped_target_nan = 0
    
    for i, forecast_start in enumerate(sample_times):
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(sample_times)}...")
        
        # Floor timestamp for consistency with API
        forecast_start = forecast_start.floor("h")
        
        # Get sensor history (672 hours before forecast_start)
        hist_start = forecast_start - pd.Timedelta(hours=HISTORY_LENGTH)
        hist_end = forecast_start - pd.Timedelta(hours=1)
        
        try:
            sensor_hist = sensor_df.loc[hist_start:hist_end]
        except Exception:
            skipped_sensor += 1
            continue
        
        if len(sensor_hist) != HISTORY_LENGTH:
            skipped_sensor += 1
            continue
        
        # Get targets (72 hours starting at forecast_start)
        target_start = forecast_start
        target_end = forecast_start + pd.Timedelta(hours=HORIZON - 1)
        
        try:
            targets = sensor_df.loc[target_start:target_end]
        except Exception:
            skipped_sensor += 1
            continue
        
        if len(targets) != HORIZON:
            skipped_sensor += 1
            continue
        
        # Skip if targets have NaN (can't evaluate)
        if targets.isna().any().any():
            skipped_target_nan += 1
            continue
        
        X_list.append(sensor_hist.values.astype(np.float32))
        y_list.append(targets.values.astype(np.float32))
        ts_list.append(forecast_start.floor("h").isoformat())
    
    print(f"[4] Results:")
    print(f"    Valid samples: {len(X_list)}")
    print(f"    Skipped (sensor gaps): {skipped_sensor}")
    print(f"    Skipped (target NaN): {skipped_target_nan}")
    
    if len(X_list) == 0:
        raise ValueError("No valid samples created!")
    
    # Convert to arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    timestamps = np.array(ts_list)
    
    # Save (adjust filename for batch processing)
    output_path = Path(args.output)
    if args.batch_size > 0:
        # Add batch info to filename
        stem = output_path.stem
        output_path = output_path.parent / f"{stem}_batch{args.batch_start:04d}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        timestamps=timestamps,
        sensor_names=np.array(sensor_names),
        # NOTE: Weather data is NOT stored here to save memory
        # Weather will be loaded on-demand during training from raw CSVs
    )
    
    print(f"[5] Saved to {output_path}")
    print(f"    X: {X.shape}")
    print(f"    y: {y.shape}")
    print(f"    timestamps: {len(timestamps)}")
    print(f"    Date range: {timestamps[0]} -> {timestamps[-1]}")


if __name__ == "__main__":
    main()
