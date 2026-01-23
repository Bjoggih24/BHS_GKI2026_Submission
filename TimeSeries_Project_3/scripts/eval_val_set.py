"""
Evaluate model on the clean validation set.

Mimics the real inference scenario with weather data passed through the API path.

Usage:
    python scripts/eval_val_set.py
    python scripts/eval_val_set.py --val-path data/val_clean.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import predict, baseline_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model on validation set")
    parser.add_argument("--val-path", type=str, default="data/val_clean.npz", help="Validation set path")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample scores")
    return parser.parse_args()


def compute_skill(y_true: np.ndarray, y_pred: np.ndarray, y_baseline: np.ndarray) -> float:
    """Compute skill score: 1 - RMSE(pred) / RMSE(baseline)"""
    rmse_pred = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rmse_base = np.sqrt(np.mean((y_true - y_baseline) ** 2))
    if rmse_base < 1e-9:
        return 0.0
    return 1.0 - rmse_pred / rmse_base


def main() -> None:
    args = parse_args()
    val_path = Path(args.val_path)
    
    if not val_path.exists():
        print(f"Error: {val_path} not found. Run build_val_set.py first.")
        return
    
    print(f"[1] Loading validation set from {val_path}...")
    data = np.load(val_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    timestamps = data["timestamps"]
    weather_forecasts = data["weather_forecasts"]
    weather_observations = data["weather_observations"]
    
    print(f"    Samples: {len(X)}")
    print(f"    X shape: {X.shape}")
    print(f"    y shape: {y.shape}")
    
    print(f"[2] Evaluating model...")
    skills = []
    errors = 0
    
    for i in range(len(X)):
        sensor_history = X[i]
        y_true = y[i]
        ts = timestamps[i]
        weather_fcst = weather_forecasts[i]
        weather_obs = weather_observations[i]
        
        try:
            # Call predict with weather data (mimics API)
            y_pred = predict(
                sensor_history=sensor_history,
                timestamp=ts,
                weather_forecast=weather_fcst,
                weather_history=weather_obs,
            )
            
            # Compute baseline
            y_baseline = baseline_model(sensor_history)
            
            # Compute skill
            skill = compute_skill(y_true, y_pred, y_baseline)
            skills.append(skill)
            
            if args.verbose:
                print(f"    [{i+1:3d}/{len(X)}] {ts}: skill={skill:.4f}")
        
        except Exception as e:
            errors += 1
            if args.verbose:
                print(f"    [{i+1:3d}/{len(X)}] {ts}: ERROR - {e}")
    
    if not skills:
        print("No successful predictions!")
        return
    
    skills = np.array(skills)
    
    print(f"[3] Results:")
    print(f"    Successful: {len(skills)}/{len(X)}")
    print(f"    Errors: {errors}")
    print()
    print(f"    Mean skill:   {skills.mean():.4f}")
    print(f"    Std skill:    {skills.std():.4f}")
    print(f"    Min skill:    {skills.min():.4f}")
    print(f"    Max skill:    {skills.max():.4f}")
    print(f"    Median skill: {np.median(skills):.4f}")
    print()
    print(f"    Negative samples: {(skills < 0).sum()}/{len(skills)} ({100*(skills < 0).mean():.1f}%)")
    print(f"    >0.3 samples:     {(skills > 0.3).sum()}/{len(skills)} ({100*(skills > 0.3).mean():.1f}%)")
    print(f"    >0.5 samples:     {(skills > 0.5).sum()}/{len(skills)} ({100*(skills > 0.5).mean():.1f}%)")


if __name__ == "__main__":
    main()
