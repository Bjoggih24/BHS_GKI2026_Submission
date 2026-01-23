#!/usr/bin/env python3
"""
Train global LightGBM on residuals using time-based split by sample timestamp.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils import compute_score, load_training_data


def main() -> None:
    artifact_root = Path(__file__).resolve().parents[2] / "artifacts"
    data = np.load(artifact_root / "global_residual_v1.npz", allow_pickle=True)
    X_tab = data["X_tab"]
    y_resid = data["y_resid"]
    sample_id = data["sample_id"]
    sensor_id = data["sensor_id"]
    horizon_id = data["horizon_id"]
    timestamps = data["timestamp"]

    # Build time-based split by sample timestamp (sample-level)
    change = np.flatnonzero(sample_id[1:] != sample_id[:-1]) + 1
    first_rows = np.r_[0, change]
    unique_ids = sample_id[first_rows]
    sample_ts = timestamps[first_rows]
    sample_ts_pd = pd.to_datetime(sample_ts, errors="coerce", utc=True)
    bad = np.where(sample_ts_pd.isna())[0]
    if len(bad) > 0:
        raise ValueError(f"{len(bad)} sample timestamps failed to parse (NaT). Example idx={bad[:5]}")
    order = np.argsort(sample_ts_pd.values)
    n_samples = len(order)
    n_val = max(1, int(0.15 * n_samples))
    val_ids = unique_ids[order[-n_val:]]

    is_val = np.isin(sample_id, val_ids)
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]

    # Build feature matrix with categorical IDs
    X_base = X_tab[:, :-2]
    X2 = np.concatenate(
        [X_base, sensor_id[:, None].astype(np.int32), horizon_id[:, None].astype(np.int32)],
        axis=1,
    )
    cat_cols = [X2.shape[1] - 2, X2.shape[1] - 1]

    train_data = lgb.Dataset(X2[train_idx], label=y_resid[train_idx], categorical_feature=cat_cols, free_raw_data=False)
    val_data = lgb.Dataset(X2[val_idx], label=y_resid[val_idx], categorical_feature=cat_cols, free_raw_data=False)

    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        max_bin=255,
        verbose=-1,
        num_threads=0,
    )

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(200)],
    )

    model_path = artifact_root / "lgbm_global_residual_v1.txt"
    booster.save_model(str(model_path))

    meta = {
        "feature_count": int(X2.shape[1]),
        "categorical_cols": cat_cols,
        "n_train_rows": int(len(train_idx)),
        "n_val_rows": int(len(val_idx)),
        "best_iteration": int(booster.best_iteration or 0),
    }
    meta_path = model_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Evaluate score on val samples (by reconstructing predictions)
    # Rebuild sample-level predictions
    resid_val = booster.predict(X2[val_idx], num_iteration=booster.best_iteration)
    preds = np.zeros((len(val_idx),), dtype=np.float32)
    preds[:] = resid_val

    # Build y_pred for val samples
    # Need baseline + residual -> compute on actual samples from train.npz
    X, y, ts_train, sensor_names = load_training_data()
    n_sensors = y.shape[-1]
    n_val_samples = len(val_ids)
    # Map sample ids for val
    val_sample_ids = sorted(list(val_ids))
    sample_to_pos = {sid: i for i, sid in enumerate(val_sample_ids)}
    y_pred = np.zeros((n_val_samples, 72, n_sensors), dtype=np.float32)
    filled = np.zeros_like(y_pred, dtype=bool)

    # Baseline
    from utils import compute_baseline_predictions
    baseline = compute_baseline_predictions(X[val_sample_ids])

    # Fill residuals
    for idx, row in enumerate(val_idx):
        sample = sample_id[row]
        pos = sample_to_pos[sample]
        s = int(sensor_id[row])
        h = int(horizon_id[row])
        y_pred[pos, h, s] = baseline[pos, h, s] + preds[idx]
        filled[pos, h, s] = True

    assert filled.all()
    y_pred = np.maximum(y_pred, 0.0)
    score_val = float(compute_score(y[val_sample_ids], y_pred, X[val_sample_ids]))
    print(f"score_val: {score_val:.6f}")

    # Update active model only if better than best ridge_h72
    best_ridge = None
    leaderboard = artifact_root / "leaderboard.csv"
    if leaderboard.exists():
        try:
            df = pd.read_csv(leaderboard)
            ridge_rows = df[df["model_type"] == "ridge_h72"]
            if not ridge_rows.empty:
                best_ridge = ridge_rows["score"].max()
        except Exception:
            best_ridge = None

    if best_ridge is None or score_val > float(best_ridge):
        active_path = artifact_root / "active_model.json"
        active_path.write_text(
            json.dumps(
                {
                    "type": "lgbm_global_residual_v1",
                    "artifact_path": str(model_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("active_model.json updated to lgbm_global_residual_v1")


if __name__ == "__main__":
    main()
