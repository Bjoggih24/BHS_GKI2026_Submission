#!/usr/bin/env python3
"""
Tune ensemble weights between ridge_h72 and lgbm_global_residual_v1 on tune_idx.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import _predict_ridge_h72, _predict_lgbm_global_residual
from scripts.exp_utils import append_csv, make_run_dir, write_json
from utils import compute_score, load_training_data


def main() -> None:
    artifact_root = Path(__file__).resolve().parents[2] / "artifacts"
    ridge_path = artifact_root / "linear_ridge_h72_v2.npz"
    lgbm_path = artifact_root / "lgbm_global_residual_v1.txt"

    data = np.load(ridge_path, allow_pickle=True)
    if "tune_idx" in data.files:
        tune_idx = data["tune_idx"].astype(int)
    else:
        tune_idx = data["val_idx"].astype(int)
    val_idx = data["val_idx"].astype(int)

    X, y, timestamps, sensor_names = load_training_data()

    def _preds_for_idx(indices: np.ndarray):
        preds_r = np.zeros((len(indices), 72, y.shape[-1]), dtype=np.float32)
        preds_l = np.zeros((len(indices), 72, y.shape[-1]), dtype=np.float32)
        for j, i in enumerate(indices):
            preds_r[j] = _predict_ridge_h72(X[i], str(timestamps[i]), None, None, ridge_path)
            preds_l[j] = _predict_lgbm_global_residual(X[i], str(timestamps[i]), None, None, lgbm_path)
        return preds_r, preds_l

    preds_ridge, preds_lgbm = _preds_for_idx(tune_idx)
    preds_ridge_val, preds_lgbm_val = _preds_for_idx(val_idx)

    best_w = 0.0
    best_score = -1.0
    for w in np.linspace(0.0, 1.0, 21):
        preds = w * preds_ridge + (1.0 - w) * preds_lgbm
        score = compute_score(y[tune_idx], preds, X[tune_idx])
        if score > best_score:
            best_score = score
            best_w = float(w)

    # Compare to ridge alone
    ridge_score = compute_score(y[tune_idx], preds_ridge, X[tune_idx])
    ridge_val_score = compute_score(y[val_idx], preds_ridge_val, X[val_idx])
    # Always log run artifacts
    run_root = artifact_root / "runs"
    run_paths = make_run_dir(run_root, "ensemble_tune")
    config = {
        "model_type": "ensemble",
        "ridge_artifact_path": str(ridge_path),
        "lgbm_artifact_path": str(lgbm_path),
        "weights_grid": list(np.linspace(0.0, 1.0, 21)),
        "tune_idx_source": "tune_idx" if "tune_idx" in data.files else "val_idx",
    }
    write_json(run_paths.config_path, config)

    metrics = {
        "ridge_score": float(ridge_score),
        "ensemble_score": float(best_score),
        "ridge_val_score": float(ridge_val_score),
        "best_w": float(best_w),
    }
    write_json(run_paths.metrics_path, metrics)

    active = {
        "type": "ensemble",
        "models": [
            {"type": "ridge_h72", "artifact_path": str(ridge_path)},
            {"type": "lgbm_global_residual_v1", "artifact_path": str(lgbm_path)},
        ],
        "weights": [best_w, 1.0 - best_w],
    }
    write_json(run_paths.meta_path, {"run_dir": str(run_paths.run_dir), "active_model": active})
    (run_paths.run_dir / "ensemble_weights.json").write_text(json.dumps(active, indent=2), encoding="utf-8")

    leaderboard = artifact_root / "leaderboard.csv"
    append_csv(
        leaderboard,
        {
            "run_dir": str(run_paths.run_dir),
            "model_type": "ensemble",
            "score": float(best_score),
            "ridge_score": float(ridge_score),
        },
    )

    ens_val = best_w * preds_ridge_val + (1.0 - best_w) * preds_lgbm_val
    ens_val_score = compute_score(y[val_idx], ens_val, X[val_idx])
    metrics["ensemble_val_score"] = float(ens_val_score)

    if ens_val_score > ridge_val_score:
        (artifact_root / "active_model.json").write_text(json.dumps(active, indent=2), encoding="utf-8")
        print(f"best_w={best_w:.2f} score={best_score:.6f} val={ens_val_score:.6f}")
    else:
        print(
            f"ensemble not better than ridge (ridge_val={ridge_val_score:.6f}, ens_val={ens_val_score:.6f})"
        )


if __name__ == "__main__":
    main()
