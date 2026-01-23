#!/usr/bin/env python3
"""
Run tabular v2 sweeps and top-k ensemble tuning.
"""

import sys
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ARTIFACTS_DIR


def run(cmd, cwd=None):
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def ensure_features_v2() -> Path:
    feats_dir = ARTIFACTS_DIR / "features"
    X_path = feats_dir / "X_tabular_v2.npy"
    if not X_path.exists():
        run([sys.executable, "scripts/precompute_tabular_features.py", "--version", "v2"], cwd=ROOT)
    return X_path


def find_best_cnn_probs():
    best = None
    for meta_path in ARTIFACTS_DIR.rglob("meta.json"):
        if "cnn" not in meta_path.parts:
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        wf1 = meta.get("weighted_f1")
        if wf1 is None:
            continue
        probs_path = meta_path.parent / "val_probs.npy"
        if not probs_path.exists():
            continue
        if best is None or wf1 > best["weighted_f1"]:
            best = {
                "name": meta_path.parent.name,
                "weighted_f1": wf1,
                "macro_f1": meta.get("macro_f1"),
                "probs_path": probs_path,
            }
    return best


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "run_v2_001"
    run_dir = ARTIFACTS_DIR / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    feats_dir = ARTIFACTS_DIR / "features"
    X_path = ensure_features_v2()
    y_path = feats_dir / "y.npy"

    tab_configs = [
        ("ET1", "extratrees", {"n_estimators": 2000, "max_features": 0.35, "min_samples_leaf": 3, "n_jobs": -1}),
        ("ET2", "extratrees", {"n_estimators": 2000, "max_features": "sqrt", "min_samples_leaf": 1, "n_jobs": -1}),
        ("ET3", "extratrees", {"n_estimators": 3000, "max_features": 0.2, "min_samples_leaf": 5, "n_jobs": -1}),
        ("HGB1", "hgb", {"max_depth": 8, "learning_rate": 0.07, "max_iter": 1500, "min_samples_leaf": 20, "l2_regularization": 1e-3}),
        ("HGB2", "hgb", {"max_depth": 10, "learning_rate": 0.05, "max_iter": 2000, "min_samples_leaf": 30, "l2_regularization": 1e-2}),
    ]

    xgb_configs = [
        ("XGB1", {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0}),
        ("XGB2", {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 1.0, "min_child_weight": 1, "reg_lambda": 1.0}),
        ("XGB3", {"max_depth": 10, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 1.0}),
        ("XGB4", {"max_depth": 6, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 1.0, "min_child_weight": 1, "reg_lambda": 5.0}),
        ("XGB5", {"max_depth": 8, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 5.0}),
        ("XGB6", {"max_depth": 10, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 1.0, "min_child_weight": 1, "reg_lambda": 1.0}),
        ("XGB7", {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 5.0}),
        ("XGB8", {"max_depth": 6, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 1.0}),
    ]

    lgbm_configs = [
        ("LGBM1", {"num_leaves": 63, "learning_rate": 0.05, "min_data_in_leaf": 10, "feature_fraction": 0.8, "bagging_fraction": 0.8}),
        ("LGBM2", {"num_leaves": 127, "learning_rate": 0.05, "min_data_in_leaf": 30, "feature_fraction": 0.8, "bagging_fraction": 0.8}),
        ("LGBM3", {"num_leaves": 255, "learning_rate": 0.05, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8}),
        ("LGBM4", {"num_leaves": 63, "learning_rate": 0.03, "min_data_in_leaf": 30, "feature_fraction": 0.8, "bagging_fraction": 0.8}),
        ("LGBM5", {"num_leaves": 127, "learning_rate": 0.03, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8}),
        ("LGBM6", {"num_leaves": 255, "learning_rate": 0.03, "min_data_in_leaf": 10, "feature_fraction": 0.8, "bagging_fraction": 0.8}),
        ("LGBM7", {"num_leaves": 127, "learning_rate": 0.05, "min_data_in_leaf": 10, "feature_fraction": 0.8, "bagging_fraction": 0.8}),
        ("LGBM8", {"num_leaves": 63, "learning_rate": 0.03, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8}),
    ]

    leaderboard = []
    prob_paths = []
    prob_names = []

    for name, model, params in tab_configs:
        cfg_path = run_dir / f"{name}_params.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        out_dir = run_dir / "tabular" / name
        run(
            [
                sys.executable,
                "scripts/train_tabular.py",
                "--model",
                model,
                "--params_json",
                str(cfg_path),
                "--features_path",
                str(X_path),
                "--labels_path",
                str(y_path),
                "--out_dir",
                str(out_dir),
            ],
            cwd=ROOT,
        )
        meta = json.loads((out_dir / "meta.json").read_text())
        leaderboard.append({"name": name, **meta})
        prob_paths.append(str(out_dir / "val_probs.npy"))
        prob_names.append(name)

    for name, grid in xgb_configs:
        params = {
            "objective": "multi:softprob",
            "num_class": 71,
            "tree_method": "hist",
            "n_estimators": 5000,
            "n_jobs": -1,
            **grid,
        }
        cfg_path = run_dir / f"{name}_params.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        out_dir = run_dir / "tabular" / name
        run(
            [
                sys.executable,
                "scripts/train_tabular.py",
                "--model",
                "xgb",
                "--params_json",
                str(cfg_path),
                "--features_path",
                str(X_path),
                "--labels_path",
                str(y_path),
                "--out_dir",
                str(out_dir),
            ],
            cwd=ROOT,
        )
        meta = json.loads((out_dir / "meta.json").read_text())
        leaderboard.append({"name": name, **meta})
        prob_paths.append(str(out_dir / "val_probs.npy"))
        prob_names.append(name)

    for name, grid in lgbm_configs:
        params = {
            "objective": "multiclass",
            "num_class": 71,
            "n_estimators": 8000,
            "n_jobs": -1,
            "bagging_freq": 1,
            **grid,
        }
        cfg_path = run_dir / f"{name}_params.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        out_dir = run_dir / "tabular" / name
        run(
            [
                sys.executable,
                "scripts/train_tabular.py",
                "--model",
                "lgbm",
                "--params_json",
                str(cfg_path),
                "--features_path",
                str(X_path),
                "--labels_path",
                str(y_path),
                "--out_dir",
                str(out_dir),
            ],
            cwd=ROOT,
        )
        meta = json.loads((out_dir / "meta.json").read_text())
        leaderboard.append({"name": name, **meta})
        prob_paths.append(str(out_dir / "val_probs.npy"))
        prob_names.append(name)

    best_cnn = find_best_cnn_probs()
    if best_cnn:
        prob_paths.append(str(best_cnn["probs_path"]))
        prob_names.append(f"cnn_{best_cnn['name']}")
        leaderboard.append(
            {
                "name": f"cnn_{best_cnn['name']}",
                "model": "cnn",
                "weighted_f1": best_cnn["weighted_f1"],
                "macro_f1": best_cnn.get("macro_f1", 0.0),
            }
        )

    ens_dir = run_dir / "ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "scripts/tune_topk_ensemble.py",
            "--prob_paths",
            *prob_paths,
            "--names",
            *prob_names,
            "--out_path",
            str(ens_dir / "ensemble_topK.json"),
        ],
        cwd=ROOT,
    )

    leaderboard_path = run_dir / "leaderboard.csv"
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        f.write("name,weighted_f1,macro_f1,model\n")
        for entry in sorted(leaderboard, key=lambda x: x["weighted_f1"], reverse=True):
            f.write(
                f"{entry['name']},{entry['weighted_f1']:.4f},{entry.get('macro_f1',0.0):.4f},{entry.get('model','tabular')}\n"
            )

    print(f"Saved leaderboard to {leaderboard_path}")


if __name__ == "__main__":
    main()
