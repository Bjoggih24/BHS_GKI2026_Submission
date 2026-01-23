"""
Train multiple models for ensemble.

Models:
  1. LGBM per-sensor (with weather) - current best
  2. ExtraTrees per-sensor (with weather) - diversity
  3. LGBM 2024-only per-sensor (with weather) - recent patterns
  4. LGBM per-sensor (no weather) - fallback/diversity

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --train-path data/train_full.npz --models lgbm,extratrees
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
import joblib

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from feature_utils import (
    build_time_features,
    build_history_features,
    build_weather_features,
    build_weather_obs_summary_features,
    build_horizon_times,
    HORIZON,
    HISTORY_LENGTH,
)

N_SENSORS = 45

# Global weather data cache (loaded once, shared across all samples)
_WEATHER_FCST_AGG = None
_WEATHER_OBS_AGG = None


def load_weather_aggregates(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-aggregated weather data for feature building."""
    global _WEATHER_FCST_AGG, _WEATHER_OBS_AGG
    
    if _WEATHER_FCST_AGG is not None and _WEATHER_OBS_AGG is not None:
        return _WEATHER_FCST_AGG, _WEATHER_OBS_AGG
    
    print("    Loading weather aggregates for feature building...")
    
    # Load forecast aggregate
    fcst_agg_path = data_dir / "weather_agg.csv"
    if fcst_agg_path.exists():
        _WEATHER_FCST_AGG = pd.read_csv(fcst_agg_path, index_col=0, parse_dates=True)
    else:
        _WEATHER_FCST_AGG = pd.DataFrame()
    
    # Load observation aggregate  
    obs_agg_path = data_dir / "weather_obs_agg.csv"
    if obs_agg_path.exists():
        _WEATHER_OBS_AGG = pd.read_csv(obs_agg_path, index_col=0, parse_dates=True)
    else:
        _WEATHER_OBS_AGG = pd.DataFrame()
    
    print(f"    Forecast agg: {len(_WEATHER_FCST_AGG)} rows, Obs agg: {len(_WEATHER_OBS_AGG)} rows")
    
    return _WEATHER_FCST_AGG, _WEATHER_OBS_AGG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multiple models")
    parser.add_argument("--train-path", type=str, default="data/train_full.npz", help="Training data path")
    parser.add_argument("--output-dir", type=str, default="artifacts/runs", help="Output directory")
    parser.add_argument("--models", type=str, default="lgbm,extratrees,lgbm_2024,lgbm_noweather",
                        help="Comma-separated list of models to train")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs")
    return parser.parse_args()


def build_features_for_sample(
    sensor_history: np.ndarray,
    timestamp: str,
    sensor_idx: int,
    use_weather: bool = True,
    weather_fcst_agg: pd.DataFrame = None,
    weather_obs_agg: pd.DataFrame = None,
) -> np.ndarray:
    """
    Build feature matrix (72, n_features) for one sensor in one sample.
    Uses the SAME feature engineering as inference.
    Weather is loaded from aggregated DataFrames (not per-sample arrays).
    """
    horizon_times = build_horizon_times(timestamp)
    time_feats = build_time_features(horizon_times)
    hist_feats = build_history_features(sensor_history[:, sensor_idx])
    
    if use_weather:
        # Use aggregated weather data (same as inference fallback)
        weather_feats = build_weather_features(horizon_times, None, weather_agg=weather_fcst_agg)
        obs_feats = build_weather_obs_summary_features(timestamp, None, obs_agg=weather_obs_agg)
    else:
        # No weather - use zeros
        weather_feats = np.zeros((HORIZON, 5), dtype=np.float32)
        obs_feats = np.zeros((HORIZON, 21), dtype=np.float32)
    
    # Add seasonal features (day of year)
    ts = pd.Timestamp(timestamp)
    day_of_year = ts.dayofyear
    day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    seasonal_feats = np.full((HORIZON, 2), [day_of_year_sin, day_of_year_cos], dtype=np.float32)
    
    # Combine all features
    feats = np.hstack([time_feats, hist_feats, weather_feats, obs_feats, seasonal_feats])
    
    return feats


def build_dataset(
    train_data: dict,
    sensor_idx: int,
    use_weather: bool = True,
    min_date: str | None = None,
    weather_fcst_agg: pd.DataFrame = None,
    weather_obs_agg: pd.DataFrame = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) dataset for a single sensor.
    
    Returns:
        X: (n_samples * 72, n_features)
        y: (n_samples * 72,)
    """
    X_all = []
    y_all = []
    
    timestamps = train_data["timestamps"]
    X_train = train_data["X"]
    y_train = train_data["y"]
    
    for i in range(len(timestamps)):
        ts = str(timestamps[i])  # Ensure string for compatibility
        
        # Filter by date if specified
        if min_date is not None:
            sample_date = pd.Timestamp(ts)
            if sample_date < pd.Timestamp(min_date):
                continue
        
        sensor_hist = X_train[i]
        targets = y_train[i][:, sensor_idx]  # (72,)
        
        try:
            feats = build_features_for_sample(
                sensor_hist, ts, sensor_idx, use_weather,
                weather_fcst_agg=weather_fcst_agg,
                weather_obs_agg=weather_obs_agg,
            )
            X_all.append(feats)
            y_all.append(targets)
        except Exception as e:
            print(f"    Warning: Failed to build features for sample {i}: {e}")
            continue
    
    if len(X_all) == 0:
        return np.array([]), np.array([])
    
    X = np.vstack(X_all)  # (n_samples * 72, n_features)
    y = np.concatenate(y_all)  # (n_samples * 72,)
    
    return X, y


def train_lgbm_per_sensor(
    train_data: dict,
    output_dir: Path,
    use_weather: bool = True,
    min_date: str | None = None,
    model_name: str = "lgbm_ps_v2",
    weather_fcst_agg: pd.DataFrame = None,
    weather_obs_agg: pd.DataFrame = None,
) -> Path:
    """Train LightGBM model per sensor."""
    print(f"\n[LGBM] Training {model_name}...")
    print(f"    use_weather={use_weather}, min_date={min_date}")
    
    run_dir = output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    models = []
    feature_means = []
    feature_stds = []
    
    lgbm_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
    }
    
    for s in range(N_SENSORS):
        print(f"    Sensor {s+1}/{N_SENSORS}...", end=" ", flush=True)
        
        X, y = build_dataset(
            train_data, s, use_weather=use_weather, min_date=min_date,
            weather_fcst_agg=weather_fcst_agg, weather_obs_agg=weather_obs_agg
        )
        
        if len(X) == 0:
            print("SKIP (no data)")
            models.append(None)
            feature_means.append(np.zeros(1))
            feature_stds.append(np.ones(1))
            continue
        
        # Standardize features
        feat_mean = np.nanmean(X, axis=0)
        feat_std = np.nanstd(X, axis=0)
        feat_std[feat_std < 1e-6] = 1.0
        
        X_norm = (X - feat_mean) / feat_std
        X_norm = np.nan_to_num(X_norm, nan=0.0)
        
        # Train model
        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(X_norm, y)
        
        models.append(model)
        feature_means.append(feat_mean)
        feature_stds.append(feat_std)
        
        print(f"OK (n={len(X)})")
    
    # Save models
    models_path = run_dir / "models.joblib"
    joblib.dump(models, models_path)
    
    # Save normalization stats
    np.savez(
        run_dir / "norm_stats.npz",
        feature_means=np.array(feature_means, dtype=object),
        feature_stds=np.array(feature_stds, dtype=object),
    )
    
    # Save config
    config = {
        "model_type": model_name,
        "use_weather": use_weather,
        "min_date": min_date,
        "lgbm_params": lgbm_params,
        "n_sensors": N_SENSORS,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"    Saved to {run_dir}")
    return run_dir


def train_extratrees_per_sensor(
    train_data: dict,
    output_dir: Path,
    use_weather: bool = True,
    min_date: str | None = None,
    model_name: str = "et_ps_v1",
    weather_fcst_agg: pd.DataFrame = None,
    weather_obs_agg: pd.DataFrame = None,
) -> Path:
    """Train ExtraTrees model per sensor."""
    print(f"\n[ExtraTrees] Training {model_name}...")
    print(f"    use_weather={use_weather}, min_date={min_date}")
    
    run_dir = output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    models = []
    feature_means = []
    feature_stds = []
    
    et_params = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 10,
        "min_samples_split": 20,
        "n_jobs": -1,
        "random_state": 42,
    }
    
    for s in range(N_SENSORS):
        print(f"    Sensor {s+1}/{N_SENSORS}...", end=" ", flush=True)
        
        X, y = build_dataset(
            train_data, s, use_weather=use_weather, min_date=min_date,
            weather_fcst_agg=weather_fcst_agg, weather_obs_agg=weather_obs_agg
        )
        
        if len(X) == 0:
            print("SKIP (no data)")
            models.append(None)
            feature_means.append(np.zeros(1))
            feature_stds.append(np.ones(1))
            continue
        
        # Standardize features
        feat_mean = np.nanmean(X, axis=0)
        feat_std = np.nanstd(X, axis=0)
        feat_std[feat_std < 1e-6] = 1.0
        
        X_norm = (X - feat_mean) / feat_std
        X_norm = np.nan_to_num(X_norm, nan=0.0)
        
        # Train model
        model = ExtraTreesRegressor(**et_params)
        model.fit(X_norm, y)
        
        models.append(model)
        feature_means.append(feat_mean)
        feature_stds.append(feat_std)
        
        print(f"OK (n={len(X)})")
    
    # Save models
    models_path = run_dir / "models.joblib"
    joblib.dump(models, models_path)
    
    # Save normalization stats
    np.savez(
        run_dir / "norm_stats.npz",
        feature_means=np.array(feature_means, dtype=object),
        feature_stds=np.array(feature_stds, dtype=object),
    )
    
    # Save config
    config = {
        "model_type": model_name,
        "use_weather": use_weather,
        "min_date": min_date,
        "et_params": et_params,
        "n_sensors": N_SENSORS,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"    Saved to {run_dir}")
    return run_dir


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_path)
    output_dir = Path(args.output_dir)
    data_dir = ROOT / "data"
    
    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        print("Run build_train_full.py first!")
        return
    
    print(f"[1] Loading training data from {train_path}...")
    train_data = dict(np.load(train_path, allow_pickle=True))
    print(f"    Samples: {len(train_data['timestamps'])}")
    print(f"    X shape: {train_data['X'].shape}")
    print(f"    y shape: {train_data['y'].shape}")
    
    # Load weather aggregates for feature building
    weather_fcst_agg, weather_obs_agg = load_weather_aggregates(data_dir)
    
    models_to_train = [m.strip().lower() for m in args.models.split(",")]
    print(f"\n[2] Training models: {models_to_train}")
    
    trained_paths = {}
    
    # Model 1: LGBM with weather (full data)
    if "lgbm" in models_to_train:
        path = train_lgbm_per_sensor(
            train_data, output_dir,
            use_weather=True, min_date=None,
            model_name="lgbm_ps_full_v2",
            weather_fcst_agg=weather_fcst_agg, weather_obs_agg=weather_obs_agg
        )
        trained_paths["lgbm"] = str(path)
    
    # Model 2: ExtraTrees with weather
    if "extratrees" in models_to_train:
        path = train_extratrees_per_sensor(
            train_data, output_dir,
            use_weather=True, min_date=None,
            model_name="et_ps_full_v1",
            weather_fcst_agg=weather_fcst_agg, weather_obs_agg=weather_obs_agg
        )
        trained_paths["extratrees"] = str(path)
    
    # Model 3: LGBM 2024-only
    if "lgbm_2024" in models_to_train:
        path = train_lgbm_per_sensor(
            train_data, output_dir,
            use_weather=True, min_date="2024-01-01",
            model_name="lgbm_ps_2024only_v1",
            weather_fcst_agg=weather_fcst_agg, weather_obs_agg=weather_obs_agg
        )
        trained_paths["lgbm_2024"] = str(path)
    
    # Model 4: LGBM no weather (diversity)
    if "lgbm_noweather" in models_to_train:
        path = train_lgbm_per_sensor(
            train_data, output_dir,
            use_weather=False, min_date=None,
            model_name="lgbm_ps_noweather_v1",
            weather_fcst_agg=weather_fcst_agg, weather_obs_agg=weather_obs_agg
        )
        trained_paths["lgbm_noweather"] = str(path)
    
    print("\n[3] Training complete!")
    print("    Trained models:")
    for name, path in trained_paths.items():
        print(f"      {name}: {path}")
    
    # Save summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "train_path": str(train_path),
            "timestamp": datetime.now().isoformat(),
            "models": trained_paths,
        }, f, indent=2)
    print(f"\n    Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
