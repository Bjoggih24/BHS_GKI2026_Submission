"""
Prediction functions for new trained models (LGBM v2, ExtraTrees).

These models use the same feature engineering as training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from feature_utils import (
    build_time_features,
    build_history_features,
    build_weather_features,
    build_weather_obs_summary_features,
    build_horizon_times,
    HORIZON,
)


_BASE = Path(__file__).resolve().parent

# Cache for loaded models
_MODEL_CACHE: dict = {}


def _resolve(p: str | Path | None) -> Path | None:
    """Resolve path relative to this file's directory."""
    if p is None:
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else (_BASE / pp)


def _load_model_bundle(run_dir: Path) -> dict | None:
    """Load models and normalization stats from a run directory."""
    cache_key = str(run_dir)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    
    run_dir = _resolve(run_dir)
    if run_dir is None or not run_dir.exists():
        return None
    
    models_path = run_dir / "models.joblib"
    norm_path = run_dir / "norm_stats.npz"
    config_path = run_dir / "config.json"
    
    if not models_path.exists():
        return None
    
    try:
        models = joblib.load(models_path)
        
        norm_data = {}
        if norm_path.exists():
            data = np.load(norm_path, allow_pickle=True)
            norm_data["feature_means"] = data["feature_means"]
            norm_data["feature_stds"] = data["feature_stds"]
        
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        
        bundle = {
            "models": models,
            "feature_means": norm_data.get("feature_means"),
            "feature_stds": norm_data.get("feature_stds"),
            "config": config,
        }
        
        _MODEL_CACHE[cache_key] = bundle
        return bundle
    
    except Exception as e:
        print(f"Error loading model bundle from {run_dir}: {e}")
        return None


def build_features_for_prediction(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray],
    weather_history: Optional[np.ndarray],
    sensor_idx: int,
    use_weather: bool = True,
    weather_agg: Optional[pd.DataFrame] = None,
    obs_agg: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Build feature matrix (72, n_features) for one sensor.
    EXACTLY matches training feature engineering.
    """
    horizon_times = build_horizon_times(timestamp)
    time_feats = build_time_features(horizon_times)
    hist_feats = build_history_features(sensor_history[:, sensor_idx])
    
    if use_weather:
        weather_feats = build_weather_features(horizon_times, weather_forecast, weather_agg=weather_agg)
        obs_feats = build_weather_obs_summary_features(timestamp, weather_history, obs_agg=obs_agg)
    else:
        # No weather - use zeros (same as training)
        weather_feats = np.zeros((HORIZON, 5), dtype=np.float32)
        obs_feats = np.zeros((HORIZON, 21), dtype=np.float32)
    
    # Add seasonal features (day of year) - SAME as training
    ts = pd.Timestamp(timestamp)
    day_of_year = ts.dayofyear
    day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    seasonal_feats = np.full((HORIZON, 2), [day_of_year_sin, day_of_year_cos], dtype=np.float32)
    
    # Combine all features
    feats = np.hstack([time_feats, hist_feats, weather_feats, obs_feats, seasonal_feats])
    
    return feats


def predict_lgbm_v2(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray],
    weather_history: Optional[np.ndarray],
    run_dir: str | Path,
    weather_agg: Optional[pd.DataFrame] = None,
    obs_agg: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Predict using LGBM v2 per-sensor models.
    
    Returns: (72, 45) predictions
    """
    bundle = _load_model_bundle(run_dir)
    if bundle is None:
        # Fallback to baseline
        return _baseline_model(sensor_history)
    
    models = bundle["models"]
    feature_means = bundle["feature_means"]
    feature_stds = bundle["feature_stds"]
    config = bundle["config"]
    
    use_weather = config.get("use_weather", True)
    n_sensors = sensor_history.shape[1]
    
    preds = np.zeros((HORIZON, n_sensors), dtype=np.float32)
    
    for s in range(n_sensors):
        model = models[s] if s < len(models) else None
        
        if model is None:
            # Use lag-72 baseline for this sensor
            preds[:, s] = sensor_history[-HORIZON:, s]
            continue
        
        # Build features
        feats = build_features_for_prediction(
            sensor_history, timestamp, weather_forecast, weather_history, s, use_weather,
            weather_agg=weather_agg, obs_agg=obs_agg
        )
        
        # Normalize
        if feature_means is not None and s < len(feature_means):
            feat_mean = feature_means[s]
            feat_std = feature_stds[s]
            feats = (feats - feat_mean) / feat_std
        
        feats = np.nan_to_num(feats, nan=0.0)
        
        # Predict
        preds[:, s] = model.predict(feats)
    
    # Ensure non-negative
    preds = np.maximum(preds, 0.0)
    
    return preds


def predict_extratrees_v1(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray],
    weather_history: Optional[np.ndarray],
    run_dir: str | Path,
    weather_agg: Optional[pd.DataFrame] = None,
    obs_agg: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Predict using ExtraTrees per-sensor models.
    
    Returns: (72, 45) predictions
    """
    # Same logic as LGBM v2
    return predict_lgbm_v2(
        sensor_history, timestamp, weather_forecast, weather_history, run_dir,
        weather_agg=weather_agg, obs_agg=obs_agg
    )


def _baseline_model(sensor_history: np.ndarray) -> np.ndarray:
    """Simple lag-72 baseline."""
    return sensor_history[-HORIZON:].copy()


def predict_ensemble_v2(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray],
    weather_history: Optional[np.ndarray],
    model_configs: list[dict],
    weights: list[float],
) -> np.ndarray:
    """
    Ensemble prediction from multiple models.
    
    model_configs: list of {"type": str, "run_dir": str}
    weights: corresponding weights (should sum to 1)
    
    Returns: (72, 45) predictions
    """
    if len(model_configs) != len(weights):
        raise ValueError("model_configs and weights must have same length")
    
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()  # Normalize
    
    ensemble_pred = None
    
    for config, weight in zip(model_configs, weights):
        model_type = config.get("type", "lgbm_v2")
        run_dir = config.get("run_dir")
        
        if model_type in ("lgbm_v2", "lgbm_ps_full_v2", "lgbm_ps_2024only_v1", "lgbm_ps_noweather_v1"):
            pred = predict_lgbm_v2(
                sensor_history, timestamp, weather_forecast, weather_history, run_dir
            )
        elif model_type in ("extratrees_v1", "et_ps_full_v1"):
            pred = predict_extratrees_v1(
                sensor_history, timestamp, weather_forecast, weather_history, run_dir
            )
        else:
            # Unknown type, use baseline
            pred = _baseline_model(sensor_history)
        
        if ensemble_pred is None:
            ensemble_pred = weight * pred
        else:
            ensemble_pred += weight * pred
    
    # Ensure non-negative
    ensemble_pred = np.maximum(ensemble_pred, 0.0)
    
    return ensemble_pred
