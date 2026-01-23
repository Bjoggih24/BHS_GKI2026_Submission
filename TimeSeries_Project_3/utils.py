"""
Utility functions for hot water demand forecasting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

HISTORY_LENGTH = 672  # 4 weeks of hourly data
HORIZON = 72  # 3 days ahead
N_SENSORS = 45


def _ffill_1d(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    mask = ~np.isnan(out)
    if not np.any(mask):
        return out
    idx = np.where(mask, np.arange(out.size), 0)
    np.maximum.accumulate(idx, out=idx)
    return out[idx]


def _bfill_1d(arr: np.ndarray) -> np.ndarray:
    return _ffill_1d(arr[::-1])[::-1]


def load_training_data(
    data_dir: str = None,
    dataset_npz_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training data for model development.

    Returns:
        X_train: (N, 672, 45) - sensor history for each sample
        y_train: (N, 72, 45) - target values to predict
        timestamps: (N,) - datetime of first forecast hour for each sample
        sensor_names: (45,) - names of sensors
    """
    if dataset_npz_path:
        data = np.load(Path(dataset_npz_path), allow_pickle=True)
    else:
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        else:
            data_dir = Path(data_dir)
        data = np.load(data_dir / "train.npz", allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    if X_train.shape[1:] != (HISTORY_LENGTH, N_SENSORS):
        raise ValueError(f"X_train has wrong shape: {X_train.shape}")
    if y_train.shape[1:] != (HORIZON, N_SENSORS):
        raise ValueError(f"y_train has wrong shape: {y_train.shape}")

    return (
        X_train,
        y_train,
        data["timestamps"],
        data["sensor_names"],
    )


def normalize_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """
    Normalize timestamps to string array (bytes -> str, objects -> str).
    """
    if timestamps.dtype.kind in {"S", "O", "U"}:
        ts_list = []
        for ts in timestamps:
            if isinstance(ts, bytes):
                ts_list.append(ts.decode("utf-8"))
            else:
                ts_list.append(str(ts))
        return np.array(ts_list)
    return timestamps.astype(str)


def load_weather_data(data_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load weather data.

    Returns:
        weather_forecasts: DataFrame with weather forecasts
        weather_observations: DataFrame with weather observations
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    else:
        data_dir = Path(data_dir)

    forecasts = pd.read_csv(data_dir / "weather_forecasts.csv")
    observations = pd.read_csv(data_dir / "weather_observations.csv")

    return forecasts, observations


def make_time_split(
    timestamps: np.ndarray,
    val_fraction: float = 0.15,
    gap_hours: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make a deterministic time-based train/val split by sorting timestamps.
    Returns indices into the original arrays.
    """
    ts = pd.to_datetime(timestamps, errors="coerce", utc=True)
    bad = np.where(ts.isna())[0]
    if len(bad) > 0:
        raise ValueError(f"{len(bad)} timestamps failed to parse (NaT). Example idx={bad[:5]}")
    tsv = ts.values.astype("datetime64[ns]")
    order = np.lexsort((np.arange(len(tsv)), tsv))
    n = len(order)
    n_val = max(1, int(val_fraction * n))
    split = n - n_val
    t_split = tsv[order[split]]
    if gap_hours > 0:
        t_train_max = t_split - np.timedelta64(gap_hours, "h")
        train_idx = np.where(tsv < t_train_max)[0]
    else:
        train_idx = np.where(tsv < t_split)[0]
    val_idx = np.where(tsv >= t_split)[0]
    if len(train_idx) == 0:
        raise ValueError("Gap too large: train split became empty.")
    train_idx = train_idx[np.argsort(tsv[train_idx])]
    val_idx = val_idx[np.argsort(tsv[val_idx])]
    return train_idx, val_idx


def compute_baseline_predictions(X: np.ndarray) -> np.ndarray:
    """
    Generate baseline predictions using lag-72.

    Args:
        X: Input history, shape (n_samples, 672, 45) or (672, 45)

    Returns:
        Predictions, shape (n_samples, 72, 45) or (72, 45)
    """
    single_sample = X.ndim == 2
    if single_sample:
        X = X[np.newaxis, :]

    baseline = X[:, HISTORY_LENGTH - HORIZON:HISTORY_LENGTH, :].copy()

    sensor_means = np.nanmean(X, axis=(0, 1))
    sensor_means = np.where(np.isnan(sensor_means), 0.0, sensor_means)

    n_samples = baseline.shape[0]
    for i in range(n_samples):
        for s in range(N_SENSORS):
            v = baseline[i, :, s]
            if np.isnan(v).any():
                v = _ffill_1d(v)
                v = _bfill_1d(v)
                if np.isnan(v).any():
                    v = np.where(np.isnan(v), sensor_means[s], v)
                baseline[i, :, s] = v

    if single_sample:
        return baseline[0]
    return baseline


def compute_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray
) -> float:
    """
    Compute the competition score.

    The score is a weighted average of per-sensor skill scores.
    Skill = 1 - (RMSE_model / RMSE_baseline)

    Sensors are weighted by sqrt(mean_flow).

    Args:
        y_true: Ground truth, shape (n_samples, 72, 45)
        y_pred: Predictions, shape (n_samples, 72, 45)
        X: Input history, shape (n_samples, 672, 45)

    Returns:
        Score (0 = baseline performance, higher = better)
    """
    # Generate baseline predictions
    y_baseline = compute_baseline_predictions(X)

    # Flatten to (n_samples * horizon, n_sensors)
    y_true_flat = y_true.reshape(-1, N_SENSORS)
    y_pred_flat = y_pred.reshape(-1, N_SENSORS)
    y_baseline_flat = y_baseline.reshape(-1, N_SENSORS)

    # Compute weights based on sqrt of mean flow
    mean_flows = np.abs(y_true_flat).mean(axis=0)
    sqrt_flows = np.sqrt(mean_flows + 1e-6)
    weights = sqrt_flows / sqrt_flows.sum()

    # Compute skill per sensor
    skills = []
    for s in range(N_SENSORS):
        rmse_model = np.sqrt(np.mean((y_true_flat[:, s] - y_pred_flat[:, s]) ** 2))
        rmse_baseline = np.sqrt(np.mean((y_true_flat[:, s] - y_baseline_flat[:, s]) ** 2))

        if (not np.isfinite(rmse_model)) or (not np.isfinite(rmse_baseline)) or rmse_baseline <= 1e-6:
            skill = 0.0
        else:
            skill = 1 - rmse_model / rmse_baseline

        skills.append(skill)

    skills = np.nan_to_num(np.array(skills), nan=0.0, posinf=0.0, neginf=0.0)
    score_raw = np.sum(skills * weights)
    if not np.isfinite(score_raw):
        score_raw = 0.0
    score = float(score_raw)

    return score


def evaluate_model(predict_fn, X: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Evaluate a prediction function.

    Args:
        predict_fn: Function that takes (672, 45) and returns (72, 45)
        X: Input history, shape (n_samples, 672, 45)
        y_true: Ground truth, shape (n_samples, 72, 45)

    Returns:
        Dictionary with evaluation metrics
    """
    # Generate predictions
    y_pred = np.array([predict_fn(x, "", None, None) for x in X])

    # Compute score
    score = compute_score(y_true, y_pred, X)

    # Overall RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return {
        'score': score,
        'rmse': rmse,
        'n_samples': len(X)
    }
