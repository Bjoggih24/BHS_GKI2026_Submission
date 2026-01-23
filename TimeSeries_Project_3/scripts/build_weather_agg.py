"""
Build weather aggregate files for training.
Creates smaller, pre-aggregated weather data that fits in memory.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def build_forecast_agg():
    """Build hourly aggregated weather forecasts."""
    print("[1] Building weather forecast aggregates...")
    
    path = DATA_DIR / "weather_forecasts.csv"
    print(f"    Loading {path}...")
    
    # Load in chunks and aggregate
    agg_list = []
    for chunk in pd.read_csv(path, low_memory=False, chunksize=500000):
        chunk["date_time"] = pd.to_datetime(chunk["date_time"], errors="coerce", utc=True).dt.tz_localize(None)
        
        # Aggregate by hour across all stations
        numeric_cols = ["temperature", "windspeed", "cloud_coverage", "dewpoint", "rain_accumulated"]
        available_cols = [c for c in numeric_cols if c in chunk.columns]
        
        hourly = chunk.groupby(chunk["date_time"].dt.floor("h"))[available_cols].mean()
        agg_list.append(hourly)
    
    # Combine all chunks
    full_agg = pd.concat(agg_list)
    full_agg = full_agg.groupby(full_agg.index).mean()  # Re-aggregate overlaps
    full_agg = full_agg.sort_index()
    
    out_path = DATA_DIR / "weather_agg.csv"
    full_agg.to_csv(out_path)
    print(f"    Saved to {out_path} ({len(full_agg)} rows)")
    return full_agg


def build_obs_agg():
    """Build hourly aggregated weather observations."""
    print("[2] Building weather observation aggregates...")
    
    path = DATA_DIR / "weather_observations.csv"
    print(f"    Loading {path}...")
    
    # Load in chunks and aggregate
    agg_list = []
    for chunk in pd.read_csv(path, low_memory=False, chunksize=500000):
        chunk["timi"] = pd.to_datetime(chunk["timi"], errors="coerce", utc=True).dt.tz_localize(None)
        
        # Map column indices to names (based on README)
        col_mapping = {"t": 7, "f": 2, "r": 13, "td": 11, "rh": 10, "p": 12, "fg": 3}
        
        for name, idx in col_mapping.items():
            if idx < len(chunk.columns):
                chunk[name] = pd.to_numeric(chunk.iloc[:, idx], errors="coerce")
        
        # Aggregate by hour across all stations
        obs_cols = ["t", "f", "r", "td", "rh", "p", "fg"]
        available_cols = [c for c in obs_cols if c in chunk.columns]
        
        hourly = chunk.groupby(chunk["timi"].dt.floor("h"))[available_cols].mean()
        agg_list.append(hourly)
    
    # Combine all chunks
    full_agg = pd.concat(agg_list)
    full_agg = full_agg.groupby(full_agg.index).mean()  # Re-aggregate overlaps
    full_agg = full_agg.sort_index()
    
    out_path = DATA_DIR / "weather_obs_agg.csv"
    full_agg.to_csv(out_path)
    print(f"    Saved to {out_path} ({len(full_agg)} rows)")
    return full_agg


def main():
    build_forecast_agg()
    build_obs_agg()
    print("\nDone!")


if __name__ == "__main__":
    main()
