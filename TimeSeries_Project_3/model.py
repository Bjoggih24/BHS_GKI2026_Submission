"""
Model for hot water demand forecasting.

This file contains the predict() function that will be called by the API.
Replace baseline_model() with your own implementation.

Input:
  - sensor_history: (672, 45) array - 4 weeks of hourly sensor data
  - timestamp: datetime string (ISO format) - when the forecast starts
  - weather_forecast: (72, n_features) array - weather forecasts for next 72h (optional)
  - weather_history: (672, n_features) array - weather observations for past 672h (optional)

Output:
  - predictions: (72, 45) array - 3 days of predictions for 45 sensors
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from feature_utils import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_FEATURE_CONFIG_V2,
    DEFAULT_FEATURE_CONFIG_V2_NOTOTAL,
    WEATHER_NUMERIC_COLS,
    WEATHER_OBS_COLS,
    build_features_for_sensor,
    build_history_features,
    build_horizon_times,
    build_time_features,
    build_weather_features,
    build_weather_features_asof,
    build_weather_features_asof_v2,
    build_weather_features_v2,
    build_weather_obs_summary_features,
    build_weather_obs_summary_features_v2,
    load_weather_forecasts_agg,
    load_weather_forecasts_runs,
    load_weather_observations_agg,
)
from utils import compute_baseline_predictions

HISTORY_LENGTH = 672  # 4 weeks
HORIZON = 72  # 3 days
N_SENSORS = 45
TOTAL_IDX = 44


_ARTIFACTS = None
_LGBM_MODELS = None
_LGBM_POSTPROC = None
_LGBM_BEST_ITERS = None
_GRU_MODEL = None
_WEATHER_AGG = None
_WEATHER_OBS_AGG = None
_H72_ARTIFACTS = None
_TOTAL_SHARES_ARTIFACTS = None
_NN_TABULAR_CACHE = {}
_LGBM_GLOBAL_RESIDUAL = None
_LGBM_GLOBAL_META = None
_WEATHER_RUNS = None
_WEATHER_ISSUE_TIMES = None
_LGBM_PS_POST2022_MODELS = None
_LGBM_PS_POST2022_POSTPROC = None
_LGBM_PS_POST2022_RUN_DIR = None
_LGBM_PS_POST2022_BEST_ITERS = None
_SENSOR_NAMES = None
_TOTAL_IDX = None
_TCN_BUNDLE = None
_ACTIVE_LOGGED = False

_BASE = Path(__file__).resolve().parent


def _resolve(path: Optional[Path | str]) -> Optional[Path]:
    if path is None:
        return None
    pp = Path(path)
    if pp.is_absolute():
        return pp
    return _BASE / pp


def _debug_enabled() -> bool:
    return os.environ.get("MODEL_DEBUG", "").strip().lower() in {"1", "true", "yes"}


def _path_status(path: Optional[Path]) -> str:
    if path is None:
        return "None"
    return f"{path} (exists={path.exists()})"


def _maybe_log_active(active: Optional[dict]) -> None:
    global _ACTIVE_LOGGED
    if _ACTIVE_LOGGED or not _debug_enabled():
        return
    _ACTIVE_LOGGED = True
    if not active:
        print("[model] active_model.json missing or empty; using fallback.", file=sys.stderr)
        return

    mtype = active.get("type")
    print(f"[model] active type: {mtype}", file=sys.stderr)
    if mtype == "ensemble":
        for idx, spec in enumerate(active.get("models", [])):
            run_dir = _resolve(spec.get("run_dir"))
            artifact = _resolve(spec.get("artifact_path"))
            print(
                f"[model] ensemble[{idx}]: {spec.get('type')} "
                f"run_dir={_path_status(run_dir)} "
                f"artifact_path={_path_status(artifact)}",
                file=sys.stderr,
            )
    else:
        run_dir = _resolve(active.get("run_dir"))
        artifact = _resolve(active.get("artifact_path"))
        if run_dir or artifact:
            print(
                f"[model] run_dir={_path_status(run_dir)} "
                f"artifact_path={_path_status(artifact)}",
                file=sys.stderr,
            )


def _load_artifacts(override_path: Optional[Path] = None) -> Optional[dict]:
    global _ARTIFACTS
    if _ARTIFACTS is not None:
        return _ARTIFACTS

    path = override_path or (Path(__file__).parent / "artifacts" / "linear_ridge_v1.npz")
    if not path.exists():
        _ARTIFACTS = None
        return None

    data = np.load(path, allow_pickle=True)
    _ARTIFACTS = {
        "coefs": data["coefs"],
        "intercepts": data["intercepts"],
        "feature_means": data["feature_means"],
        "feature_stds": data["feature_stds"],
        "feature_names": data["feature_names"],
        "alpha_total": float(data["alpha_total"]),
        "reconcile_beta": float(data["reconcile_beta"]),
        "blend_lambdas": data["blend_lambdas"] if "blend_lambdas" in data.files else None,
        "valid_cols": data["valid_cols"] if "valid_cols" in data.files else None,
    }
    return _ARTIFACTS


def _load_sensor_names() -> Optional[np.ndarray]:
    global _SENSOR_NAMES
    if _SENSOR_NAMES is not None:
        return _SENSOR_NAMES

    candidates = [
        Path(__file__).parent / "artifacts" / "datasets" / "sensor_names.npy",
        Path(__file__).parent / "data" / "train.npz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix == ".npz":
                data = np.load(path, allow_pickle=True)
                if "sensor_names" in data.files:
                    names = data["sensor_names"]
                else:
                    continue
            else:
                names = np.load(path, allow_pickle=True)
            _SENSOR_NAMES = np.array([str(n) for n in names])
            return _SENSOR_NAMES
        except Exception:
            continue

    _SENSOR_NAMES = None
    return None


def _get_total_idx(n_sensors: int) -> int:
    global _TOTAL_IDX
    if _TOTAL_IDX is None:
        _TOTAL_IDX = TOTAL_IDX
    if _TOTAL_IDX != TOTAL_IDX:
        _TOTAL_IDX = TOTAL_IDX
    if n_sensors <= TOTAL_IDX:
        raise ValueError(f"Expected total_idx={TOTAL_IDX} within n_sensors={n_sensors}")
    return _TOTAL_IDX


def _load_active_model() -> Optional[dict]:
    active_path = Path(__file__).parent / "artifacts" / "active_model.json"
    if not active_path.exists():
        return None
    with active_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_weather_agg() -> Optional[pd.DataFrame]:
    global _WEATHER_AGG
    if _WEATHER_AGG is not None:
        return _WEATHER_AGG
    data_dir = Path(__file__).parent / "data"
    _WEATHER_AGG = load_weather_forecasts_agg(data_dir)
    return _WEATHER_AGG


def _get_weather_runs() -> tuple[Optional[pd.DataFrame], Optional[pd.DatetimeIndex]]:
    global _WEATHER_RUNS, _WEATHER_ISSUE_TIMES
    if _WEATHER_RUNS is not None or _WEATHER_ISSUE_TIMES is not None:
        return _WEATHER_RUNS, _WEATHER_ISSUE_TIMES
    data_dir = Path(__file__).parent / "data"
    runs_pack = load_weather_forecasts_runs(data_dir)
    if runs_pack is None:
        _WEATHER_RUNS, _WEATHER_ISSUE_TIMES = None, None
    else:
        _WEATHER_RUNS, _WEATHER_ISSUE_TIMES = runs_pack
    return _WEATHER_RUNS, _WEATHER_ISSUE_TIMES


def _get_weather_obs_agg() -> Optional[pd.DataFrame]:
    global _WEATHER_OBS_AGG
    if _WEATHER_OBS_AGG is not None:
        return _WEATHER_OBS_AGG
    data_dir = Path(__file__).parent / "data"
    _WEATHER_OBS_AGG = load_weather_observations_agg(data_dir)
    return _WEATHER_OBS_AGG


def _load_ridge_h72_artifacts(override_path: Optional[Path] = None) -> Optional[dict]:
    global _H72_ARTIFACTS
    if _H72_ARTIFACTS is not None:
        return _H72_ARTIFACTS

    path = override_path or (Path(__file__).parent / "artifacts" / "linear_ridge_h72_v2.npz")
    if not path.exists():
        _H72_ARTIFACTS = None
        return None

    data = np.load(path, allow_pickle=True)
    _H72_ARTIFACTS = {
        "coefs": data["coefs"],
        "intercepts": data["intercepts"],
        "feature_means": data["feature_means"],
        "feature_stds": data["feature_stds"],
        "feature_names": data["feature_names"],
        "alpha_total": float(data["alpha_total"]),
        "reconcile_beta": float(data["reconcile_beta"]),
        "blend_lambdas": data["blend_lambdas"] if "blend_lambdas" in data.files else None,
        "valid_cols": data["valid_cols"] if "valid_cols" in data.files else None,
    }
    return _H72_ARTIFACTS


def _load_total_shares_artifacts(override_path: Optional[Path] = None) -> Optional[dict]:
    global _TOTAL_SHARES_ARTIFACTS
    if _TOTAL_SHARES_ARTIFACTS is not None:
        return _TOTAL_SHARES_ARTIFACTS

    path = override_path or (Path(__file__).parent / "artifacts" / "ridge_total_shares_v1.npz")
    if not path.exists():
        _TOTAL_SHARES_ARTIFACTS = None
        return None

    data = np.load(path, allow_pickle=True)
    _TOTAL_SHARES_ARTIFACTS = {
        "total_coefs": data["total_coefs"],
        "total_intercepts": data["total_intercepts"],
        "total_feature_means": data["total_feature_means"],
        "total_feature_stds": data["total_feature_stds"],
        "total_valid_cols": data["total_valid_cols"],
        "share_coefs": data["share_coefs"],
        "share_intercepts": data["share_intercepts"],
        "share_feature_means": data["share_feature_means"],
        "share_feature_stds": data["share_feature_stds"],
        "share_valid_cols": data["share_valid_cols"],
        "feature_names": data["feature_names"],
        "blend_lambdas": data["blend_lambdas"] if "blend_lambdas" in data.files else None,
        "reconcile_beta": float(data["reconcile_beta"]) if "reconcile_beta" in data.files else 1.0,
    }
    return _TOTAL_SHARES_ARTIFACTS


def _load_lgbm_models(run_dir: Path, n_sensors: int) -> list:
    global _LGBM_MODELS
    if _LGBM_MODELS is not None:
        return _LGBM_MODELS

    try:
        import lightgbm as lgb
    except Exception:
        return []

    models = []
    models_dir = run_dir / "models"
    for s in range(n_sensors):
        path = models_dir / f"sensor_{s:02d}.txt"
        if not path.exists():
            models.append(None)
            continue
        booster = lgb.Booster(model_file=str(path))
        models.append(booster)

    _LGBM_MODELS = models
    return models


def _load_lgbm_global_residual(model_path: Optional[Path] = None) -> Optional[dict]:
    global _LGBM_GLOBAL_RESIDUAL, _LGBM_GLOBAL_META
    if _LGBM_GLOBAL_RESIDUAL is not None and _LGBM_GLOBAL_META is not None:
        return {"booster": _LGBM_GLOBAL_RESIDUAL, "meta": _LGBM_GLOBAL_META}

    try:
        import lightgbm as lgb
    except Exception:
        return None

    model_path = model_path or (Path(__file__).parent / "artifacts" / "lgbm_global_residual_v1.txt")
    meta_path = model_path.with_suffix(".meta.json")
    if not model_path.exists() or not meta_path.exists():
        return None

    booster = lgb.Booster(model_file=str(model_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _LGBM_GLOBAL_RESIDUAL = booster
    _LGBM_GLOBAL_META = meta
    return {"booster": booster, "meta": meta}


def _load_lgbm_best_iters(run_dir: Path, n_sensors: int) -> Optional[np.ndarray]:
    global _LGBM_BEST_ITERS
    if _LGBM_BEST_ITERS is not None and _LGBM_BEST_ITERS.shape[0] == n_sensors:
        return _LGBM_BEST_ITERS
    p = run_dir / "best_iters.npy"
    if not p.exists():
        _LGBM_BEST_ITERS = None
        return None
    arr = np.load(p, allow_pickle=True)
    if arr.shape[0] != n_sensors:
        _LGBM_BEST_ITERS = None
        return None
    _LGBM_BEST_ITERS = arr
    return _LGBM_BEST_ITERS


def _load_lgbm_postproc(run_dir: Path) -> Optional[dict]:
    global _LGBM_POSTPROC
    if _LGBM_POSTPROC is not None and _LGBM_POSTPROC.get("run_dir") == str(run_dir):
        return _LGBM_POSTPROC
    postproc_path = run_dir / "postproc.npz"
    if not postproc_path.exists():
        _LGBM_POSTPROC = {"run_dir": str(run_dir), "blend_lambdas": None}
        return _LGBM_POSTPROC
    data = np.load(postproc_path, allow_pickle=True)
    blend_lambdas = data["blend_lambdas"] if "blend_lambdas" in data.files else None
    _LGBM_POSTPROC = {"run_dir": str(run_dir), "blend_lambdas": blend_lambdas}
    return _LGBM_POSTPROC


def _load_lgbm_ps_post2022_models(run_dir: Path, n_sensors: int) -> list:
    global _LGBM_PS_POST2022_MODELS, _LGBM_PS_POST2022_RUN_DIR
    if _LGBM_PS_POST2022_MODELS is not None and _LGBM_PS_POST2022_RUN_DIR == str(run_dir):
        return _LGBM_PS_POST2022_MODELS

    try:
        import lightgbm as lgb
    except Exception:
        return []

    models_dir = run_dir / "models"
    models = []
    for s in range(n_sensors):
        path = models_dir / f"sensor_{s:02d}.txt"
        if not path.exists():
            models.append(None)
            continue
        models.append(lgb.Booster(model_file=str(path)))

    _LGBM_PS_POST2022_MODELS = models
    _LGBM_PS_POST2022_RUN_DIR = str(run_dir)
    return models


def _load_lgbm_ps_post2022_postproc(postproc_path: Path, run_dir: Path) -> Optional[dict]:
    global _LGBM_PS_POST2022_POSTPROC, _LGBM_PS_POST2022_RUN_DIR
    if _LGBM_PS_POST2022_POSTPROC is not None and _LGBM_PS_POST2022_RUN_DIR == str(run_dir):
        return _LGBM_PS_POST2022_POSTPROC
    if not postproc_path.exists():
        return None
    data = np.load(postproc_path, allow_pickle=True)
    postproc = {
        "blend_lambdas": data["blend_lambdas"],
        "reconcile_beta": float(data["reconcile_beta"]),
        "residual_mode": int(data["residual_mode"]) if "residual_mode" in data.files else 0,
        "feature_set_version": int(data["feature_set_version"]) if "feature_set_version" in data.files else 1,
        "include_total_context": int(data["include_total_context"]) if "include_total_context" in data.files else 0,
    }
    _LGBM_PS_POST2022_POSTPROC = postproc
    _LGBM_PS_POST2022_RUN_DIR = str(run_dir)
    return postproc


def _load_lgbm_ps_post2022_best_iters(run_dir: Path, n_sensors: int) -> Optional[np.ndarray]:
    global _LGBM_PS_POST2022_BEST_ITERS, _LGBM_PS_POST2022_RUN_DIR
    if _LGBM_PS_POST2022_BEST_ITERS is not None and _LGBM_PS_POST2022_RUN_DIR == str(run_dir):
        return _LGBM_PS_POST2022_BEST_ITERS
    p = run_dir / "best_iters.npy"
    if not p.exists():
        _LGBM_PS_POST2022_BEST_ITERS = None
        return None
    arr = np.load(p, allow_pickle=True)
    
    # Handle dict format (0-dim object array containing dict)
    if arr.shape == () and isinstance(arr.item(), dict):
        d = arr.item()
        arr = np.zeros(n_sensors, dtype=np.int32)
        for k, v in d.items():
            if 0 <= k < n_sensors:
                arr[k] = v
    
    # Validate shape
    if arr.ndim != 1 or arr.shape[0] < n_sensors:
        _LGBM_PS_POST2022_BEST_ITERS = None
        return None
    _LGBM_PS_POST2022_BEST_ITERS = arr
    _LGBM_PS_POST2022_RUN_DIR = str(run_dir)
    return _LGBM_PS_POST2022_BEST_ITERS


def _load_gru_model(run_dir: Path):
    global _GRU_MODEL
    if _GRU_MODEL is not None:
        return _GRU_MODEL

    try:
        import torch
    except Exception:
        return None

    from nn_models import GRUForecast

    stats_path = run_dir / "stats.npz"
    weights_path = run_dir / "gru_best.pt"
    if not stats_path.exists() or not weights_path.exists():
        return None

    stats = np.load(stats_path, allow_pickle=True)
    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]

    # Infer model input dims
    input_dim = 45 + 45 + 6  # X + mask + time_history
    horizon_cov_dim = 6 + 7  # time_horizon + weather_horizon
    model = GRUForecast(input_dim=input_dim, horizon_cov_dim=horizon_cov_dim)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    _GRU_MODEL = {
        "model": model,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return _GRU_MODEL


def _build_obs_history_matrix(
    forecast_start: str,
    weather_history: Optional[np.ndarray],
    obs_agg: Optional[pd.DataFrame],
) -> np.ndarray:
    t0 = pd.to_datetime(forecast_start, errors="coerce", utc=True).tz_localize(None)
    hist_index = pd.date_range(end=t0 - pd.Timedelta(hours=1), periods=HISTORY_LENGTH, freq="h")

    mat = None
    if weather_history is not None:
        if getattr(weather_history, "ndim", 0) == 2 and weather_history.shape[1] == len(WEATHER_OBS_COLS):
            mat = np.asarray(weather_history, dtype=np.float32)
            if mat.shape[0] != HISTORY_LENGTH:
                mat = mat[-HISTORY_LENGTH:]
        else:
            try:
                df = pd.DataFrame(weather_history)
                ts = pd.to_datetime(df.iloc[:, 1], errors="coerce", utc=True).dt.floor("h").dt.tz_localize(None)
                df["timi_norm"] = ts
                df = df.dropna(subset=["timi_norm"])

                mapping = {
                    "f": 2,
                    "fg": 3,
                    "t": 7,
                    "rh": 10,
                    "td": 11,
                    "p": 12,
                    "r": 13,
                }
                for k, idx in mapping.items():
                    df[k] = pd.to_numeric(df.iloc[:, idx], errors="coerce")

                agg = df.groupby("timi_norm")[list(mapping.keys())].mean().sort_index()
                agg = agg.reindex(columns=WEATHER_OBS_COLS)
                agg = agg.reindex(hist_index)
                mat = agg.to_numpy(dtype=np.float32)
            except Exception:
                mat = None

    if mat is None and obs_agg is not None:
        aligned = obs_agg.reindex(hist_index)
        aligned = aligned.reindex(columns=WEATHER_OBS_COLS)
        mat = aligned.to_numpy(dtype=np.float32)

    if mat is None:
        mat = np.full((HISTORY_LENGTH, len(WEATHER_OBS_COLS)), np.nan, dtype=np.float32)

    return mat


def _build_tcn_inputs(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray],
    weather_history: Optional[np.ndarray],
    obs_agg: Optional[pd.DataFrame],
    runs: Optional[pd.DataFrame],
    issue_times: Optional[pd.DatetimeIndex],
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    horizon_times = build_horizon_times(timestamp)
    hist_times = pd.date_range(
        start=pd.to_datetime(timestamp) - pd.Timedelta(hours=HISTORY_LENGTH),
        periods=HISTORY_LENGTH,
        freq="h",
    )

    x_hist = np.asarray(sensor_history, dtype=np.float32)
    x_mask = np.isfinite(x_hist).astype(np.float32)
    x_norm = (x_hist - x_mean) / x_std
    x_norm = np.where(np.isfinite(x_norm), x_norm, 0.0)

    time_hist = build_time_features(hist_times)
    time_hor = build_time_features(horizon_times)

    obs_hist = _build_obs_history_matrix(timestamp, weather_history, obs_agg)
    obs_hist = np.nan_to_num(obs_hist, nan=0.0)
    obs_sum = build_weather_obs_summary_features_v2(timestamp, weather_history, obs_agg=obs_agg)
    obs_sum = np.nan_to_num(obs_sum, nan=0.0)

    if weather_forecast is not None:
        fcst = build_weather_features_v2(horizon_times, weather_forecast)
    else:
        fcst = build_weather_features_asof_v2(horizon_times, timestamp, runs, issue_times)
    fcst = np.nan_to_num(fcst, nan=0.0)

    return x_norm, x_mask, time_hist, time_hor, obs_hist, obs_sum, fcst


def _load_tcn_bundle(run_dir: Path) -> Optional[dict]:
    global _TCN_BUNDLE
    if _TCN_BUNDLE is not None and _TCN_BUNDLE.get("run_dir") == str(run_dir):
        return _TCN_BUNDLE

    try:
        import torch
    except Exception:
        return None

    stats_path = run_dir / "stats.npz"
    weights_path = run_dir / "tcn_best.pt"
    if not stats_path.exists() or not weights_path.exists():
        return None

    try:
        from scripts.nn.models import TCNForecast
    except Exception:
        return None

    cfg = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    stats = np.load(stats_path, allow_pickle=True)
    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]

    model = TCNForecast(
        in_channels=45 + 45 + 6 + 7,
        hidden=int(cfg.get("hidden", 64)),
        blocks=int(cfg.get("blocks", 6)),
        kernel_size=int(cfg.get("kernel_size", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    _TCN_BUNDLE = {
        "run_dir": str(run_dir),
        "model": model,
        "x_mean": x_mean.astype(np.float32),
        "x_std": np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": np.where(y_std < 1e-6, 1.0, y_std).astype(np.float32),
        "residual": bool(cfg.get("residual", False)),
        "target": cfg.get("target", "direct"),
    }
    return _TCN_BUNDLE


def _standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    features = np.where(np.isnan(features), mean, features)
    return (features - mean) / std


def _reconcile_keep_total(
    preds: np.ndarray, beta: float, total_idx: Optional[int] = None, eps: float = 1e-6
) -> np.ndarray:
    out = preds.copy()
    n_sensors = out.shape[1]
    total_idx = TOTAL_IDX if total_idx is None else int(total_idx)
    if total_idx < 0 or total_idx >= n_sensors:
        raise ValueError(f"Invalid total_idx={total_idx} for n_sensors={n_sensors}")
    part_idx = [i for i in range(n_sensors) if i != total_idx]
    total = out[:, total_idx]
    parts = out[:, part_idx]
    sum_parts = parts.sum(axis=1)
    scale = total / (sum_parts + eps)
    scale = np.clip(scale, 0.0, 5.0)
    parts_scaled = parts * scale[:, None]
    out[:, part_idx] = (1 - beta) * parts + beta * parts_scaled
    out[:, total_idx] = total
    return out


def _predict_ridge_h72(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray],
    weather_history: Optional[np.ndarray],
    artifact_path: Optional[Path] = None,
) -> np.ndarray:
    artifacts = _load_ridge_h72_artifacts(Path(artifact_path) if artifact_path else None)
    if artifacts is None:
        return weekly_mean_model_v1(sensor_history)

    horizon_times = build_horizon_times(timestamp)
    time_feats = build_time_features(horizon_times)
    if weather_forecast is None:
        runs, issue_times = _get_weather_runs()
        if runs is None or issue_times is None:
            weather_feats = np.full((HORIZON, len(WEATHER_NUMERIC_COLS)), np.nan, dtype=np.float32)
        else:
            weather_feats = build_weather_features_asof(horizon_times, timestamp, runs, issue_times)
    else:
        weather_feats = build_weather_features(horizon_times, weather_forecast, _get_weather_agg())
    obs_feats = build_weather_obs_summary_features(timestamp, weather_history, obs_agg=_get_weather_obs_agg())

    # Debug: log weather feature stats
    if _debug_enabled():
        wf_all_zero = int(np.all(weather_feats == 0))
        wf_nans = int(np.isnan(weather_feats).sum())
        of_all_zero = int(np.all(obs_feats == 0))
        of_nans = int(np.isnan(obs_feats).sum())
        print(f"[ridge_h72] ts={timestamp} weather_feats: all_zero={wf_all_zero}, nans={wf_nans} | obs_feats: all_zero={of_all_zero}, nans={of_nans}")

    n_sensors = sensor_history.shape[1]
    n_features = len(DEFAULT_FEATURE_CONFIG.names)
    preds = np.zeros((HORIZON, n_sensors), dtype=np.float32)

    for s in range(n_sensors):
        hist_feats = build_history_features(sensor_history[:, s])
        feats = np.hstack([time_feats, hist_feats, weather_feats, obs_feats])
        if feats.shape[1] != n_features:
            raise ValueError("Feature dimension mismatch at inference.")

        mean = artifacts["feature_means"][s]
        std = artifacts["feature_stds"][s]
        feats = _standardize(feats, mean, std)
        valid_cols = artifacts.get("valid_cols")
        if valid_cols is not None:
            feats[:, ~valid_cols[s]] = 0.0

        preds[:, s] = (feats * artifacts["coefs"][s]).sum(axis=1) + artifacts["intercepts"][s]

    preds = np.maximum(preds, 0.0)

    blend_lambdas = artifacts.get("blend_lambdas")
    if blend_lambdas is not None:
        baseline = baseline_model(sensor_history)
        preds = preds * (1 - blend_lambdas[None, :]) + baseline * blend_lambdas[None, :]
        preds = np.maximum(preds, 0.0)

    reconcile_beta = artifacts.get("reconcile_beta", 1.0)
    total_idx = _get_total_idx(n_sensors)
    preds = _reconcile_keep_total(preds, beta=reconcile_beta, total_idx=total_idx)
    return preds


def _predict_total_shares(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray],
    weather_history: Optional[np.ndarray],
    artifact_path: Optional[Path] = None,
) -> np.ndarray:
    artifacts = _load_total_shares_artifacts(Path(artifact_path) if artifact_path else None)
    if artifacts is None:
        return weekly_mean_model_v1(sensor_history)

    horizon_times = build_horizon_times(timestamp)
    time_feats = build_time_features(horizon_times)
    if weather_forecast is None:
        runs, issue_times = _get_weather_runs()
        if runs is None or issue_times is None:
            weather_feats = np.full((HORIZON, len(WEATHER_NUMERIC_COLS)), np.nan, dtype=np.float32)
        else:
            weather_feats = build_weather_features_asof(horizon_times, timestamp, runs, issue_times)
    else:
        weather_feats = build_weather_features(horizon_times, weather_forecast, _get_weather_agg())
    obs_feats = build_weather_obs_summary_features(timestamp, weather_history, obs_agg=_get_weather_obs_agg())

    n_sensors = sensor_history.shape[1]
    total_idx = _get_total_idx(n_sensors)
    n_features = len(DEFAULT_FEATURE_CONFIG.names)

    # Total prediction
    hist_total = build_history_features(sensor_history[:, total_idx])
    feats_total = np.hstack([time_feats, hist_total, weather_feats, obs_feats])
    if feats_total.shape[1] != n_features:
        raise ValueError("Feature dimension mismatch at inference.")
    feats_total = _standardize(feats_total, artifacts["total_feature_means"], artifacts["total_feature_stds"])
    valid_cols_total = artifacts.get("total_valid_cols")
    if valid_cols_total is not None:
        feats_total[:, ~valid_cols_total] = 0.0
    total_pred = (feats_total * artifacts["total_coefs"]).sum(axis=1) + artifacts["total_intercepts"]
    total_pred = np.maximum(total_pred, 0.0)

    # Share predictions
    part_idx = [i for i in range(n_sensors) if i != total_idx]
    share_pred = np.zeros((HORIZON, n_sensors - 1), dtype=np.float32)
    for j, s in enumerate(part_idx):
        hist_feats = build_history_features(sensor_history[:, s])
        feats = np.hstack([time_feats, hist_feats, weather_feats, obs_feats])
        feats = _standardize(feats, artifacts["share_feature_means"][s], artifacts["share_feature_stds"][s])
        valid_cols = artifacts.get("share_valid_cols")
        if valid_cols is not None:
            feats[:, ~valid_cols[s]] = 0.0
        share_pred[:, j] = (feats * artifacts["share_coefs"][s]).sum(axis=1) + artifacts["share_intercepts"][s]

    share_pred = np.maximum(share_pred, 0.0)
    sum_sh = share_pred.sum(axis=1, keepdims=True) + 1e-6
    share_norm = share_pred / sum_sh

    preds = np.zeros((HORIZON, n_sensors), dtype=np.float32)
    preds[:, total_idx] = total_pred
    preds[:, part_idx] = share_norm * total_pred[:, None]

    blend_lambdas = artifacts.get("blend_lambdas")
    if blend_lambdas is not None:
        baseline = baseline_model(sensor_history)
        preds = preds * (1 - blend_lambdas[None, :]) + baseline * blend_lambdas[None, :]
        preds = np.maximum(preds, 0.0)

    reconcile_beta = artifacts.get("reconcile_beta", 1.0)
    preds = _reconcile_keep_total(preds, beta=reconcile_beta, total_idx=total_idx)
    return preds


def _load_nn_tabular(run_dir: Path):
    """Load NN tabular model and normalization stats."""
    import torch
    from torch import nn
    
    global _NN_TABULAR_CACHE
    key = str(run_dir)
    if key in _NN_TABULAR_CACHE:
        return _NN_TABULAR_CACHE[key]
    
    if not run_dir.exists():
        return None
    
    # Load norm stats
    norm_path = run_dir / "norm_stats.npz"
    model_path = run_dir / "best_model.pt"
    config_path = run_dir / "config.json"
    
    if not norm_path.exists() or not model_path.exists():
        return None
    
    norm_stats = np.load(norm_path, allow_pickle=True)
    feat_mean = norm_stats["feat_mean"]
    feat_std = norm_stats["feat_std"]
    target_mean = float(norm_stats["target_mean"])
    target_std = float(norm_stats["target_std"])
    
    input_dim = len(feat_mean)
    hidden, num_layers, dropout = 512, 4, 0.15
    
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
            hidden = cfg.get("hidden", 512)
            num_layers = cfg.get("num_layers", 4)
            dropout = cfg.get("dropout", 0.15)
    
    # Build MLP
    class TabularMLP(nn.Module):
        def __init__(self, input_dim, hidden, num_layers, dropout):
            super().__init__()
            layers = [nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            for _ in range(num_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
            layers.append(nn.Linear(hidden, 1))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)
    
    model = TabularMLP(input_dim, hidden, num_layers, dropout)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    
    bundle = {
        "model": model,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }
    _NN_TABULAR_CACHE[key] = bundle
    return bundle


def _predict_nn_tabular(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray],
    weather_history: Optional[np.ndarray],
    run_dir: Path,
) -> np.ndarray:
    """Make predictions using NN tabular model."""
    import torch
    
    bundle = _load_nn_tabular(run_dir)
    if bundle is None:
        return weekly_mean_model_v1(sensor_history)
    
    model = bundle["model"]
    feat_mean = bundle["feat_mean"]
    feat_std = bundle["feat_std"]
    target_mean = bundle["target_mean"]
    target_std = bundle["target_std"]
    
    runs, issue_times = _get_weather_runs()
    obs_agg = _get_weather_obs_agg()
    
    n_sensors = sensor_history.shape[1]
    preds = np.zeros((HORIZON, n_sensors), dtype=np.float32)
    
    # Baseline and lag features
    baseline = sensor_history[HISTORY_LENGTH - HORIZON:HISTORY_LENGTH, :]
    lag24 = sensor_history[HISTORY_LENGTH - 24 - HORIZON:HISTORY_LENGTH - 24, :]
    lag168 = sensor_history[HISTORY_LENGTH - 168 - HORIZON:HISTORY_LENGTH - 168, :]
    total_baseline = baseline[:, TOTAL_IDX]
    total_recent_mean = np.nanmean(sensor_history[-24:, TOTAL_IDX])
    
    horizon_times = build_horizon_times(timestamp)
    time_feats = build_time_features(horizon_times)
    if weather_forecast is None:
        weather_feats = build_weather_features_asof(horizon_times, timestamp, runs, issue_times)
    else:
        weather_feats = build_weather_features(horizon_times, weather_forecast, _get_weather_agg())
    obs_feats = build_weather_obs_summary_features(timestamp, weather_history, obs_agg)
    
    with torch.no_grad():
        for s in range(n_sensors):
            hist_feats = build_history_features(sensor_history[:, s])
            
            batch_feats = []
            for h in range(HORIZON):
                baseline_val = baseline[h, s]
                lag24_val = lag24[h, s]
                lag168_val = lag168[h, s]
                diff_24 = baseline_val - lag24_val if not np.isnan(lag24_val) else 0.0
                diff_168 = baseline_val - lag168_val if not np.isnan(lag168_val) else 0.0
                total_val = total_baseline[h]
                ratio_to_total = baseline_val / (total_val + 1e-6) if total_val > 0 else 0.0
                
                feats = np.concatenate([
                    time_feats[h],
                    hist_feats[h],
                    weather_feats[h],
                    obs_feats[h],
                    [h / HORIZON],
                    [s / N_SENSORS],
                    [baseline_val],
                    [lag24_val if not np.isnan(lag24_val) else baseline_val],
                    [lag168_val if not np.isnan(lag168_val) else baseline_val],
                    [diff_24],
                    [diff_168],
                    [total_val],
                    [ratio_to_total],
                    [total_recent_mean],
                ])
                batch_feats.append(feats)
            
            batch_feats = np.array(batch_feats, dtype=np.float32)
            feat_mean_row = feat_mean[None, :]
            feat_std_row = feat_std[None, :]
            mask = np.isfinite(batch_feats)
            batch_feats = np.where(mask, batch_feats, feat_mean_row)
            batch_feats = (batch_feats - feat_mean_row) / feat_std_row
            batch_feats = np.clip(batch_feats, -8.0, 8.0)
            
            batch_t = torch.from_numpy(batch_feats)
            pred_norm = model(batch_t).squeeze(-1).numpy()
            
            # Denormalize (residual + baseline)
            pred_residual = pred_norm * target_std + target_mean
            preds[:, s] = baseline[:, s] + pred_residual
    
    preds = np.maximum(preds, 0.0)
    # Final NaN cleanup - replace any remaining NaN with baseline
    if np.isnan(preds).any():
        base_fallback = sensor_history[HISTORY_LENGTH - HORIZON:HISTORY_LENGTH, :].copy()
        base_fallback = np.nan_to_num(base_fallback, nan=np.nanmean(sensor_history))
        preds = np.where(np.isnan(preds), base_fallback, preds)
    return preds


def _predict_ensemble(
    models: list[dict],
    weights: list[float],
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray],
    weather_history: Optional[np.ndarray],
) -> np.ndarray:
    if not models:
        return weekly_mean_model_v1(sensor_history)
    if len(weights) != len(models):
        weights = [1.0 / len(models)] * len(models)

    preds_list = []
    for spec in models:
        mtype = spec.get("type")
        artifact_path = spec.get("artifact_path")
        if mtype == "ridge_h72":
            pred = _predict_ridge_h72(sensor_history, timestamp, weather_forecast, weather_history, artifact_path)
        elif mtype == "total_shares_v1":
            pred = _predict_total_shares(sensor_history, timestamp, weather_forecast, weather_history, artifact_path)
        elif mtype == "lgbm_global_residual_v1":
            pred = _predict_lgbm_global_residual(
                sensor_history,
                timestamp,
                weather_forecast,
                weather_history,
                _resolve(artifact_path),
            )
        elif mtype == "lgbm_per_sensor_post2022_v1":
            run_dir = _resolve(spec.get("run_dir"))
            postproc_path = _resolve(spec.get("artifact_path"))
            if run_dir is None or postproc_path is None:
                continue
            pred = _predict_lgbm_per_sensor_post2022_v1(
                sensor_history, timestamp, weather_forecast, weather_history, run_dir, postproc_path
            )
        elif mtype == "lgbm_per_sensor_post2022_v2":
            run_dir = _resolve(spec.get("run_dir"))
            postproc_path = _resolve(spec.get("artifact_path"))
            if run_dir is None or postproc_path is None:
                continue
            pred = _predict_lgbm_per_sensor_post2022_v2(
                sensor_history, timestamp, weather_forecast, weather_history, run_dir, postproc_path
            )
        elif mtype in {"nn_tcn_stride", "tcn"}:
            run_dir = _resolve(spec.get("run_dir"))
            if run_dir is None:
                continue
            tcn_bundle = _load_tcn_bundle(run_dir)
            if tcn_bundle is None:
                continue
            import torch

            runs, issue_times = _get_weather_runs()
            obs_agg = _get_weather_obs_agg()
            x_norm, x_mask, time_hist, time_hor, obs_hist, obs_sum, fcst = _build_tcn_inputs(
                sensor_history,
                timestamp,
                weather_forecast,
                weather_history,
                obs_agg=obs_agg,
                runs=runs,
                issue_times=issue_times,
                x_mean=tcn_bundle["x_mean"],
                x_std=tcn_bundle["x_std"],
            )
            x_in = np.concatenate([x_norm, x_mask, time_hist, obs_hist], axis=-1).astype(np.float32)
            x_in_t = torch.from_numpy(x_in).unsqueeze(0).permute(0, 2, 1)
            time_hor_t = torch.from_numpy(time_hor).unsqueeze(0).float()
            fcst_t = torch.from_numpy(fcst).unsqueeze(0).float()
            obs_sum_t = torch.from_numpy(obs_sum).unsqueeze(0).float()
            with torch.no_grad():
                pred = tcn_bundle["model"](x_in_t, time_hor_t, fcst_t, obs_sum_t).cpu().numpy()[0]
            if tcn_bundle["residual"]:
                if tcn_bundle.get("target") == "residual_baseline":
                    baseline_raw = compute_baseline_predictions(sensor_history[np.newaxis, ...])[0]
                else:
                    baseline_raw = x_norm[-HORIZON:, :] * tcn_bundle["x_std"] + tcn_bundle["x_mean"]
                baseline_norm = (baseline_raw - tcn_bundle["y_mean"]) / tcn_bundle["y_std"]
                pred = pred + baseline_norm
            pred = pred * tcn_bundle["y_std"] + tcn_bundle["y_mean"]
            pred = pred.astype(np.float32)
        elif mtype == "nn_tabular":
            run_dir = _resolve(spec.get("run_dir"))
            if run_dir is None:
                continue
            pred = _predict_nn_tabular(
                sensor_history, timestamp, weather_forecast, weather_history, run_dir
            )
        elif mtype in {"lgbm_v2", "extratrees_v1"}:
            # New model format using models_v2.py
            run_dir = _resolve(spec.get("run_dir"))
            if run_dir is None:
                continue
            from models_v2 import predict_lgbm_v2, predict_extratrees_v1
            # Load weather aggregates for on-demand feature building
            weather_agg = _get_weather_agg()
            obs_agg = _get_weather_obs_agg()
            if mtype == "lgbm_v2":
                pred = predict_lgbm_v2(
                    sensor_history, timestamp, weather_forecast, weather_history,
                    run_dir, weather_agg=weather_agg, obs_agg=obs_agg
                )
            else:
                pred = predict_extratrees_v1(
                    sensor_history, timestamp, weather_forecast, weather_history,
                    run_dir, weather_agg=weather_agg, obs_agg=obs_agg
                )
        else:
            continue
        preds_list.append(pred)

    if not preds_list:
        return weekly_mean_model_v1(sensor_history)
    weights_arr = np.array(weights[: len(preds_list)], dtype=np.float32)
    weights_arr = weights_arr / weights_arr.sum()
    out = np.zeros_like(preds_list[0])
    for w, p in zip(weights_arr, preds_list):
        out += w * p
    return out


def _predict_lgbm_global_residual(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray],
    weather_history: Optional[np.ndarray],
    model_path: Optional[Path] = None,
) -> np.ndarray:
    bundle = _load_lgbm_global_residual(model_path)
    if bundle is None:
        return weekly_mean_model_v1(sensor_history)

    booster = bundle["booster"]
    meta = bundle["meta"]
    horizon_times = build_horizon_times(timestamp)
    time_feats = build_time_features(horizon_times)
    if weather_forecast is None:
        runs, issue_times = _get_weather_runs()
        if runs is None or issue_times is None:
            weather_feats = np.full((HORIZON, len(WEATHER_NUMERIC_COLS)), np.nan, dtype=np.float32)
        else:
            weather_feats = build_weather_features_asof(horizon_times, timestamp, runs, issue_times)
    else:
        weather_feats = build_weather_features(horizon_times, weather_forecast, _get_weather_agg())
    obs_feats = build_weather_obs_summary_features(timestamp, weather_history, obs_agg=_get_weather_obs_agg())

    n_sensors = sensor_history.shape[1]
    n_features = len(DEFAULT_FEATURE_CONFIG.names)
    rows = HORIZON * n_sensors
    X_tab = np.zeros((rows, n_features + 2), dtype=np.float32)
    row = 0
    for s in range(n_sensors):
        hist_feats = build_history_features(sensor_history[:, s])
        for h in range(HORIZON):
            feats = np.hstack([time_feats[h], hist_feats[h], weather_feats[h], obs_feats[h]]).astype(np.float32)
            X_tab[row, :-2] = feats
            X_tab[row, -2:] = np.array([s, h], dtype=np.int32)
            row += 1

    best_it = int(meta.get("best_iteration", 0))
    if best_it > 0:
        resid = booster.predict(X_tab, num_iteration=best_it)
    else:
        resid = booster.predict(X_tab)
    resid = resid.reshape(HORIZON, n_sensors)
    preds = baseline_model(sensor_history) + resid
    preds = np.maximum(preds, 0.0)

    reconcile_beta = float(meta.get("reconcile_beta", 0.0))
    if reconcile_beta > 0.0:
        total_idx = _get_total_idx(n_sensors)
        preds = _reconcile_keep_total(preds, beta=reconcile_beta, total_idx=total_idx)
    return preds


def _predict_lgbm_per_sensor_post2022_v1(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray],
    weather_history: Optional[np.ndarray],
    run_dir: Path,
    postproc_path: Path,
) -> np.ndarray:
    n_sensors = sensor_history.shape[1]
    models = _load_lgbm_ps_post2022_models(run_dir, n_sensors)
    if not models:
        return weekly_mean_model_v1(sensor_history)

    postproc = _load_lgbm_ps_post2022_postproc(postproc_path, run_dir)
    if postproc is None:
        return weekly_mean_model_v1(sensor_history)

    best_iters = _load_lgbm_ps_post2022_best_iters(run_dir, n_sensors)
    horizon_times = build_horizon_times(timestamp)
    time_feats = build_time_features(horizon_times)
    if weather_forecast is None:
        runs, issue_times = _get_weather_runs()
        if runs is None or issue_times is None:
            weather_feats = np.full((HORIZON, len(WEATHER_NUMERIC_COLS)), np.nan, dtype=np.float32)
        else:
            weather_feats = build_weather_features_asof(horizon_times, timestamp, runs, issue_times)
    else:
        weather_feats = build_weather_features(horizon_times, weather_forecast, _get_weather_agg())
    obs_feats = build_weather_obs_summary_features(timestamp, weather_history, obs_agg=_get_weather_obs_agg())

    horizon_id = np.arange(HORIZON, dtype=np.int32).reshape(-1, 1)
    preds = np.zeros((HORIZON, n_sensors), dtype=np.float32)
    base = compute_baseline_predictions(sensor_history[np.newaxis, ...])[0]

    for s in range(n_sensors):
        model = models[s] if s < len(models) else None
        if model is None:
            preds[:, s] = base[:, s]
            continue
        hist_feats = build_history_features(sensor_history[:, s])
        feats = np.hstack([time_feats, hist_feats, weather_feats, obs_feats])
        feats = np.hstack([feats, horizon_id])
        num_it = None
        if best_iters is not None and s < best_iters.shape[0]:
            num_it = int(best_iters[s])
        preds[:, s] = model.predict(feats, num_iteration=num_it) if num_it else model.predict(feats)

    if postproc.get("residual_mode", 0) == 1:
        preds = preds + base
    preds = np.maximum(preds, 0.0)
    blend_lambdas = postproc.get("blend_lambdas")
    if blend_lambdas is not None:
        preds = preds * (1 - blend_lambdas[None, :]) + base * blend_lambdas[None, :]
        preds = np.maximum(preds, 0.0)

    reconcile_beta = float(postproc.get("reconcile_beta", 0.0))
    total_idx = _get_total_idx(n_sensors)
    preds = _reconcile_keep_total(preds, beta=reconcile_beta, total_idx=total_idx)
    return preds


def _predict_lgbm_per_sensor_post2022_v2(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray],
    weather_history: Optional[np.ndarray],
    run_dir: Path,
    postproc_path: Path,
) -> np.ndarray:
    n_sensors = sensor_history.shape[1]
    models = _load_lgbm_ps_post2022_models(run_dir, n_sensors)
    if not models:
        return weekly_mean_model_v1(sensor_history)

    postproc = _load_lgbm_ps_post2022_postproc(postproc_path, run_dir)
    if postproc is None:
        return weekly_mean_model_v1(sensor_history)

    best_iters = _load_lgbm_ps_post2022_best_iters(run_dir, n_sensors)
    runs, issue_times = _get_weather_runs()
    obs_agg = _get_weather_obs_agg()

    include_total = bool(postproc.get("include_total_context", 0))
    config = DEFAULT_FEATURE_CONFIG_V2 if include_total else DEFAULT_FEATURE_CONFIG_V2_NOTOTAL
    total_idx = _get_total_idx(n_sensors)

    preds = np.zeros((HORIZON, n_sensors), dtype=np.float32)
    base = compute_baseline_predictions(sensor_history[np.newaxis, ...])[0]

    for s in range(n_sensors):
        model = models[s] if s < len(models) else None
        if model is None:
            preds[:, s] = base[:, s]
            continue
        total_hist = sensor_history[:, total_idx] if include_total else None
        feats = build_features_for_sensor(
            sensor_history_1d=sensor_history[:, s],
            forecast_start=timestamp,
            weather_forecast=weather_forecast,
            weather_agg=None,
            weather_history=weather_history,
            obs_agg=obs_agg,
            total_history_1d=total_hist,
            runs=runs,
            issue_times=issue_times,
            config=config,
        )
        num_it = None
        if best_iters is not None and s < best_iters.shape[0]:
            num_it = int(best_iters[s])
        preds[:, s] = model.predict(feats, num_iteration=num_it) if num_it else model.predict(feats)

    preds = np.maximum(preds, 0.0)
    blend_lambdas = postproc.get("blend_lambdas")
    if blend_lambdas is not None:
        preds = preds * (1 - blend_lambdas[None, :]) + base * blend_lambdas[None, :]
        preds = np.maximum(preds, 0.0)

    reconcile_beta = float(postproc.get("reconcile_beta", 0.0))
    preds = _reconcile_keep_total(preds, beta=reconcile_beta, total_idx=total_idx)
    return preds


def predict(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray] = None,
    weather_history: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Predict hot water demand for all 45 sensors, 72 hours ahead.

    Args:
        sensor_history: (672, 45) array of past sensor readings
        timestamp: ISO format datetime string for the first forecast hour
        weather_forecast: (72, n) array of weather forecasts (optional)
        weather_history: (672, n) array of weather observations (optional)

    Returns:
        (72, 45) array of predictions
    """
    assert TOTAL_IDX == 44
    if sensor_history.shape[1] <= TOTAL_IDX:
        raise ValueError(f"Expected total_idx={TOTAL_IDX} within n_sensors={sensor_history.shape[1]}")
    active = _load_active_model()
    _maybe_log_active(active)
    if active and active.get("type") == "ensemble":
        models = active.get("models", [])
        weights = active.get("weights", [])
        preds = _predict_ensemble(models, weights, sensor_history, timestamp, weather_forecast, weather_history)
        preds = np.maximum(preds, 0.0)
        reconcile_beta = float(active.get("reconcile_beta", 0.0))
        total_idx = active.get("total_idx")
        if reconcile_beta > 0.0:
            preds = _reconcile_keep_total(preds, beta=reconcile_beta, total_idx=total_idx)
        return preds
    if active and active.get("type") == "lgbm_global_residual_v1":
        artifact_path = _resolve(active.get("artifact_path"))
        return _predict_lgbm_global_residual(
            sensor_history, timestamp, weather_forecast, weather_history, artifact_path
        )
    if active and active.get("type") == "lgbm_per_sensor_post2022_v1":
        run_dir = _resolve(active.get("run_dir"))
        postproc_path = _resolve(active.get("artifact_path"))
        if run_dir is None or postproc_path is None:
            return weekly_mean_model_v1(sensor_history)
        return _predict_lgbm_per_sensor_post2022_v1(
            sensor_history, timestamp, weather_forecast, weather_history, run_dir, postproc_path
        )
    if active and active.get("type") == "lgbm_per_sensor_post2022_v2":
        run_dir = _resolve(active.get("run_dir"))
        postproc_path = _resolve(active.get("artifact_path"))
        if run_dir is None or postproc_path is None:
            return weekly_mean_model_v1(sensor_history)
        return _predict_lgbm_per_sensor_post2022_v2(
            sensor_history, timestamp, weather_forecast, weather_history, run_dir, postproc_path
        )
    if active and active.get("type") == "lightgbm":
        run_dir = _resolve(active.get("run_dir"))
        if run_dir is None:
            return weekly_mean_model_v1(sensor_history)
        n_sensors = sensor_history.shape[1]
        models = _load_lgbm_models(run_dir, n_sensors)
        best_iters = _load_lgbm_best_iters(run_dir, n_sensors)
        if models:
            horizon_times = build_horizon_times(timestamp)
            time_feats = build_time_features(horizon_times)
            if weather_forecast is None:
                runs, issue_times = _get_weather_runs()
                if runs is None or issue_times is None:
                    weather_feats = np.full(
                        (HORIZON, len(WEATHER_NUMERIC_COLS)), np.nan, dtype=np.float32
                    )
                else:
                    weather_feats = build_weather_features_asof(horizon_times, timestamp, runs, issue_times)
            else:
                weather_feats = build_weather_features(horizon_times, weather_forecast, _get_weather_agg())
            obs_feats = build_weather_obs_summary_features(timestamp, weather_history, obs_agg=_get_weather_obs_agg())

            preds = np.zeros((HORIZON, n_sensors), dtype=np.float32)
            for s in range(n_sensors):
                model = models[s]
                if model is None:
                    preds[:, s] = weekly_mean_model_v1(sensor_history)[:, s]
                    continue
                hist_feats = build_history_features(sensor_history[:, s])
                feats = np.hstack([time_feats, hist_feats, weather_feats, obs_feats])
                num_it = int(best_iters[s]) if best_iters is not None else model.current_iteration()
                preds[:, s] = model.predict(feats, num_iteration=num_it)

            postproc = _load_lgbm_postproc(run_dir)
            blend_lambdas = None if postproc is None else postproc.get("blend_lambdas")
            if blend_lambdas is not None:
                baseline = baseline_model(sensor_history)
                preds = preds * (1 - blend_lambdas[None, :]) + baseline * blend_lambdas[None, :]
            preds = np.maximum(preds, 0.0)
            return preds

    if active and active.get("type") == "gru":
        run_dir = _resolve(active.get("run_dir"))
        if run_dir is None:
            return weekly_mean_model_v1(sensor_history)
        gru_bundle = _load_gru_model(run_dir)
        if gru_bundle is not None:
            import torch

            horizon_times = build_horizon_times(timestamp)
            time_horizon = build_time_features(horizon_times)
            if weather_forecast is None:
                runs, issue_times = _get_weather_runs()
                if runs is None or issue_times is None:
                    weather_horizon = np.full(
                        (HORIZON, len(WEATHER_NUMERIC_COLS)), np.nan, dtype=np.float32
                    )
                else:
                    weather_horizon = build_weather_features_asof(horizon_times, timestamp, runs, issue_times)
            else:
                weather_horizon = build_weather_features(horizon_times, weather_forecast, _get_weather_agg())
            time_horizon = np.nan_to_num(time_horizon, nan=0.0)
            weather_horizon = np.nan_to_num(weather_horizon, nan=0.0)

            history_mask = ~np.isnan(sensor_history)
            x_mean = gru_bundle["x_mean"]
            x_std = np.where(gru_bundle["x_std"] < 1e-6, 1.0, gru_bundle["x_std"])
            x_mean = np.nan_to_num(x_mean, nan=0.0)
            x_std = np.nan_to_num(x_std, nan=1.0)
            x_norm = (sensor_history - x_mean) / x_std
            x_norm = np.where(np.isnan(x_norm), 0.0, x_norm)

            time_history = build_time_features(
                pd.date_range(
                    start=pd.to_datetime(timestamp) - pd.Timedelta(hours=HISTORY_LENGTH),
                    periods=HISTORY_LENGTH,
                    freq="h",
                )
            )

            history = np.concatenate([x_norm, history_mask.astype(np.float32), time_history], axis=-1)
            horizon_cov = np.concatenate([time_horizon, weather_horizon], axis=-1)
            history = np.nan_to_num(history, nan=0.0)
            horizon_cov = np.nan_to_num(horizon_cov, nan=0.0)

            history_t = torch.from_numpy(history).unsqueeze(0).float()
            cov_t = torch.from_numpy(horizon_cov).unsqueeze(0).float()
            with torch.no_grad():
                pred = gru_bundle["model"](history_t, cov_t).cpu().numpy()[0]
            pred = pred * gru_bundle["y_std"] + gru_bundle["y_mean"]
            return pred.astype(np.float32)

    if active and active.get("type") in {"nn_tcn_stride", "tcn"}:
        run_dir = _resolve(active.get("run_dir"))
        if run_dir is None:
            return weekly_mean_model_v1(sensor_history)
        tcn_bundle = _load_tcn_bundle(run_dir)
        if tcn_bundle is not None:
            import torch

            runs, issue_times = _get_weather_runs()
            obs_agg = _get_weather_obs_agg()
            x_norm, x_mask, time_hist, time_hor, obs_hist, obs_sum, fcst = _build_tcn_inputs(
                sensor_history,
                timestamp,
                weather_forecast,
                weather_history,
                obs_agg=obs_agg,
                runs=runs,
                issue_times=issue_times,
                x_mean=tcn_bundle["x_mean"],
                x_std=tcn_bundle["x_std"],
            )

            if time_hor.shape[-1] != 6:
                raise ValueError(f"time_hor dim mismatch: expected 6, got {time_hor.shape[-1]}")
            if fcst.shape[-1] != 12:
                raise ValueError(f"fcst dim mismatch: expected 12, got {fcst.shape[-1]}")
            if obs_sum.shape[-1] != 27:
                raise ValueError(f"obs_sum dim mismatch: expected 27, got {obs_sum.shape[-1]}")

            x_in = np.concatenate([x_norm, x_mask, time_hist, obs_hist], axis=-1).astype(np.float32)
            x_in_t = torch.from_numpy(x_in).unsqueeze(0).permute(0, 2, 1)
            time_hor_t = torch.from_numpy(time_hor).unsqueeze(0).float()
            fcst_t = torch.from_numpy(fcst).unsqueeze(0).float()
            obs_sum_t = torch.from_numpy(obs_sum).unsqueeze(0).float()
            with torch.no_grad():
                pred = tcn_bundle["model"](x_in_t, time_hor_t, fcst_t, obs_sum_t).cpu().numpy()[0]
            if tcn_bundle["residual"]:
                if tcn_bundle.get("target") == "residual_baseline":
                    baseline_raw = compute_baseline_predictions(sensor_history[np.newaxis, ...])[0]
                else:
                    baseline_raw = x_norm[-HORIZON:, :] * tcn_bundle["x_std"] + tcn_bundle["x_mean"]
                baseline_norm = (baseline_raw - tcn_bundle["y_mean"]) / tcn_bundle["y_std"]
                pred = pred + baseline_norm
            pred = pred * tcn_bundle["y_std"] + tcn_bundle["y_mean"]
            return pred.astype(np.float32)

    if active and active.get("type") in {"lgbm_v2", "extratrees_v1"}:
        # New model format using models_v2.py
        run_dir = _resolve(active.get("run_dir"))
        if run_dir is None:
            return weekly_mean_model_v1(sensor_history)
        from models_v2 import predict_lgbm_v2, predict_extratrees_v1
        weather_agg = _get_weather_agg()
        obs_agg = _get_weather_obs_agg()
        if active.get("type") == "lgbm_v2":
            return predict_lgbm_v2(
                sensor_history, timestamp, weather_forecast, weather_history,
                run_dir, weather_agg=weather_agg, obs_agg=obs_agg
            )
        else:
            return predict_extratrees_v1(
                sensor_history, timestamp, weather_forecast, weather_history,
                run_dir, weather_agg=weather_agg, obs_agg=obs_agg
            )

    if active and active.get("type") == "ridge_h72":
        artifact_path = _resolve(active.get("artifact_path"))
        return _predict_ridge_h72(sensor_history, timestamp, weather_forecast, weather_history, artifact_path)

    if active and active.get("type") == "total_shares_v1":
        artifact_path = _resolve(active.get("artifact_path"))
        return _predict_total_shares(sensor_history, timestamp, weather_forecast, weather_history, artifact_path)

    if active and active.get("type") == "ridge":
        artifact_path = _resolve(active.get("artifact_path"))
        artifacts = _load_artifacts(artifact_path)
    else:
        artifacts = _load_artifacts()
    if artifacts is None:
        return weekly_mean_model_v1(sensor_history)

    horizon_times = build_horizon_times(timestamp)
    time_feats = build_time_features(horizon_times)
    if weather_forecast is None:
        runs, issue_times = _get_weather_runs()
        if runs is None or issue_times is None:
            weather_feats = np.full((HORIZON, len(WEATHER_NUMERIC_COLS)), np.nan, dtype=np.float32)
        else:
            weather_feats = build_weather_features_asof(horizon_times, timestamp, runs, issue_times)
    else:
        weather_feats = build_weather_features(horizon_times, weather_forecast, _get_weather_agg())
    obs_feats = build_weather_obs_summary_features(timestamp, weather_history, obs_agg=_get_weather_obs_agg())

    n_sensors = sensor_history.shape[1]
    n_features = len(DEFAULT_FEATURE_CONFIG.names)
    preds = np.zeros((HORIZON, n_sensors), dtype=np.float32)

    for s in range(n_sensors):
        hist_feats = build_history_features(sensor_history[:, s])
        feats = np.hstack([time_feats, hist_feats, weather_feats, obs_feats])
        if feats.shape[1] != n_features:
            raise ValueError("Feature dimension mismatch at inference.")

        mean = artifacts["feature_means"][s]
        std = artifacts["feature_stds"][s]
        feats = _standardize(feats, mean, std)
        valid_cols = artifacts.get("valid_cols")
        if valid_cols is not None:
            feats[:, ~valid_cols[s]] = 0.0
        preds[:, s] = feats @ artifacts["coefs"][s] + artifacts["intercepts"][s]

    preds = np.maximum(preds, 0.0)

    # Blend toward baseline per sensor (if available)
    blend_lambdas = artifacts.get("blend_lambdas")
    if blend_lambdas is not None:
        baseline = baseline_model(sensor_history)
        preds = preds * (1 - blend_lambdas[None, :]) + baseline * blend_lambdas[None, :]
        preds = np.maximum(preds, 0.0)

    # Reconcile parts toward total (assume last column is total)
    reconcile_beta = artifacts.get("reconcile_beta", 1.0)
    total_idx = _get_total_idx(n_sensors)
    preds = _reconcile_keep_total(preds, beta=reconcile_beta, total_idx=total_idx)

    return preds


def baseline_model(sensor_history: np.ndarray) -> np.ndarray:
    """
    Baseline: Use values from 72 hours ago.

    For predicting hour t+h, use the value from hour t+h-72.
    This assumes the pattern from 3 days ago will repeat.
    """
    base = sensor_history[HISTORY_LENGTH - HORIZON:HISTORY_LENGTH, :].copy()

    sensor_means = np.nanmean(sensor_history, axis=0)
    sensor_means = np.where(np.isnan(sensor_means), 0.0, sensor_means)

    for s in range(base.shape[1]):
        v = base[:, s]
        if np.isnan(v).any():
            mask = ~np.isnan(v)
            if np.any(mask):
                idx = np.where(mask, np.arange(v.size), 0)
                np.maximum.accumulate(idx, out=idx)
                v = v[idx]
                v_rev = v[::-1]
                mask2 = ~np.isnan(v_rev)
                if np.any(mask2):
                    idx2 = np.where(mask2, np.arange(v_rev.size), 0)
                    np.maximum.accumulate(idx2, out=idx2)
                    v_rev = v_rev[idx2]
                v = v_rev[::-1]
            if np.isnan(v).any():
                v = np.where(np.isnan(v), sensor_means[s], v)
            base[:, s] = v

    return base


def weekly_mean_model_v1(sensor_history: np.ndarray) -> np.ndarray:
    """
    Weekly seasonal mean: average the same hour across the last 4 weeks.

    For horizon h, use indices:
        t+h-168, t+h-336, t+h-504, t+h-672
    """
    predictions = np.zeros((HORIZON, N_SENSORS))

    for h in range(HORIZON):
        idx0 = HISTORY_LENGTH - 168 + h
        idxs = [idx0 - 168 * k for k in range(4)]
        values = sensor_history[idxs]
        pred = np.nanmean(values, axis=0)

        # Fallback to lag-72 if all values are NaN for a sensor
        fallback = sensor_history[HISTORY_LENGTH - HORIZON + h]
        pred = np.where(np.isnan(pred), fallback, pred)

        predictions[h] = pred

    return predictions
