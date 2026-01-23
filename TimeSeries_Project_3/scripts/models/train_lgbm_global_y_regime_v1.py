#!/usr/bin/env python3
"""
Train regime-split global LightGBM models for direct y prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.exp_utils import append_csv, make_run_dir, write_json
from utils import compute_score, load_training_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train regime-split global LGBM on y.")
    parser.add_argument("--dataset", type=str, default="artifacts/global_y_v1.npz")
    parser.add_argument("--cutoff-ts", type=str, default="2022-06-09T00:00:00Z")
    parser.add_argument("--name", type=str, default="lgbm_global_y_regime_v1")
    parser.add_argument("--num-boost-round", type=int, default=5000)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = ROOT / "artifacts"
    data = np.load(ROOT / args.dataset, allow_pickle=True)
    X_tab = data["X_tab"]
    y_target = data["y_target"]
    sample_id = data["sample_id"]
    sensor_id = data["sensor_id"]
    horizon_id = data["horizon_id"]
    timestamps = data["timestamp"]

    # sample-level split
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

    cutoff = pd.to_datetime(args.cutoff_ts, errors="coerce", utc=True)
    if pd.isna(cutoff):
        raise ValueError(f"cutoff-ts failed to parse: {args.cutoff_ts}")

    # Build feature matrix with categorical IDs
    X_base = X_tab[:, :-2]
    X2 = np.concatenate(
        [X_base, sensor_id[:, None].astype(np.int32), horizon_id[:, None].astype(np.int32)],
        axis=1,
    )
    cat_cols = [X2.shape[1] - 2, X2.shape[1] - 1]

    # Regime masks
    sample_ts_full = pd.to_datetime(timestamps, errors="coerce", utc=True)
    if np.any(sample_ts_full.isna()):
        raise ValueError("NaT found in row timestamps.")
    is_post = sample_ts_full >= cutoff

    train_pre = train_idx[~is_post[train_idx]]
    train_post = train_idx[is_post[train_idx]]
    val_pre = val_idx[~is_post[val_idx]]
    val_post = val_idx[is_post[val_idx]]

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

    run_root = artifact_root / "runs"
    run_paths = make_run_dir(run_root, args.name)
    write_json(
        run_paths.config_path,
        {
            "model_type": "lgbm_global_y_regime",
            "dataset": args.dataset,
            "cutoff_ts": args.cutoff_ts,
            "num_boost_round": args.num_boost_round,
            "early_stopping_rounds": args.early_stopping_rounds,
            "categorical_cols": cat_cols,
        },
    )

    models_dir = run_paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    def _train_one(split_name: str, tr_idx: np.ndarray, va_idx: np.ndarray):
        if len(tr_idx) == 0 or len(va_idx) == 0:
            return None, None
        train_data = lgb.Dataset(X2[tr_idx], label=y_target[tr_idx], categorical_feature=cat_cols, free_raw_data=False)
        val_data = lgb.Dataset(X2[va_idx], label=y_target[va_idx], categorical_feature=cat_cols, free_raw_data=False)
        booster = lgb.train(
            params,
            train_data,
            num_boost_round=args.num_boost_round,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(args.early_stopping_rounds)],
        )
        model_path = models_dir / f"{split_name}.txt"
        booster.save_model(str(model_path))
        return booster, model_path

    booster_pre, model_pre_path = _train_one("pre", train_pre, val_pre)
    booster_post, model_post_path = _train_one("post", train_post, val_post)

    # Predict on val set
    preds = np.zeros((len(val_idx),), dtype=np.float32)
    if booster_pre is not None and len(val_pre) > 0:
        preds[np.isin(val_idx, val_pre)] = booster_pre.predict(
            X2[val_pre], num_iteration=booster_pre.best_iteration
        )
    if booster_post is not None and len(val_post) > 0:
        preds[np.isin(val_idx, val_post)] = booster_post.predict(
            X2[val_post], num_iteration=booster_post.best_iteration
        )

    # Reconstruct y_pred for score
    X, y, ts_train, sensor_names = load_training_data()
    n_sensors = y.shape[-1]
    val_sample_ids = sorted(list(val_ids))
    sample_to_pos = {sid: i for i, sid in enumerate(val_sample_ids)}
    y_pred = np.zeros((len(val_sample_ids), 72, n_sensors), dtype=np.float32)
    filled = np.zeros_like(y_pred, dtype=bool)

    for idx, row in enumerate(val_idx):
        sample = sample_id[row]
        pos = sample_to_pos[sample]
        s = int(sensor_id[row])
        h = int(horizon_id[row])
        y_pred[pos, h, s] = preds[idx]
        filled[pos, h, s] = True

    assert filled.all()
    y_pred = np.maximum(y_pred, 0.0)
    score_val = float(compute_score(y[val_sample_ids], y_pred, X[val_sample_ids]))

    meta = {
        "model_type": "lgbm_global_y_regime",
        "run_dir": str(run_paths.run_dir),
        "cutoff_ts": args.cutoff_ts,
        "pre_model_path": str(model_pre_path) if model_pre_path else None,
        "post_model_path": str(model_post_path) if model_post_path else None,
        "best_iteration_pre": int(booster_pre.best_iteration) if booster_pre is not None else None,
        "best_iteration_post": int(booster_post.best_iteration) if booster_post is not None else None,
        "score_val": score_val,
        "n_train_rows_pre": int(len(train_pre)),
        "n_train_rows_post": int(len(train_post)),
        "n_val_rows_pre": int(len(val_pre)),
        "n_val_rows_post": int(len(val_post)),
    }
    write_json(run_paths.meta_path, meta)
    write_json(run_paths.metrics_path, {"score_val": score_val})

    append_csv(
        artifact_root / "leaderboard.csv",
        {
            "run_dir": str(run_paths.run_dir),
            "model_type": "lgbm_global_y_regime",
            "score": score_val,
        },
    )

    print(f"score_val: {score_val:.6f}")


if __name__ == "__main__":
    main()
