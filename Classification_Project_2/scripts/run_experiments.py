#!/usr/bin/env python3
"""
Run tabular + CNN sweeps and ensemble tuning.
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


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "run_gym_001"
    run_dir = ARTIFACTS_DIR / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    feats_dir = ARTIFACTS_DIR / "features"
    X_path = feats_dir / "X_tabular.npy"
    y_path = feats_dir / "y.npy"
    if not X_path.exists():
        run([sys.executable, "scripts/precompute_tabular_features.py"], cwd=ROOT)

    tab_configs = [
        ("ET_A", "extratrees", {"n_estimators": 1200, "max_features": "sqrt", "min_samples_leaf": 1, "n_jobs": -1}),
        ("ET_B", "extratrees", {"n_estimators": 1600, "max_features": 0.35, "min_samples_leaf": 3, "n_jobs": -1}),
        ("ET_C", "extratrees", {"n_estimators": 1600, "max_features": 0.2, "min_samples_leaf": 5, "n_jobs": -1}),
        ("HGB_A", "hgb", {"max_depth": 8, "learning_rate": 0.07, "max_iter": 600, "min_samples_leaf": 20, "l2_regularization": 1e-3}),
        ("RF_A", "randomforest", {"n_estimators": 2000, "max_features": "sqrt", "min_samples_leaf": 2, "n_jobs": -1}),
    ]

    leaderboard = []
    tab_prob_paths = []

    for name, model, params in tab_configs:
        cfg_path = run_dir / f"{name}_params.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

        out_dir = run_dir / "tabular" / name
        run([
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
        ], cwd=ROOT)

        meta = json.loads((out_dir / "meta.json").read_text())
        leaderboard.append({"name": name, **meta})
        tab_prob_paths.append(str(out_dir / "val_probs.npy"))

    cnn_configs = [
        ("cnn_small_ce_ls", "resnet_small", "ce"),
        ("cnn_small_mildw_ls", "resnet_small", "ce_mildw"),
        ("cnn_small_se_mixer_ce_ls", "resnet_small_se_mixer", "ce"),
        ("cnn_medium_ce_ls", "resnet_medium", "ce"),
    ]

    cnn_prob_paths = []
    for name, arch, loss in cnn_configs:
        out_dir = run_dir / "cnn" / name
        run([
            sys.executable,
            "scripts/train_cnn.py",
            "--arch",
            arch,
            "--loss",
            loss,
            "--label_smoothing",
            "0.05",
            "--epochs",
            "40",
            "--patience",
            "7",
            "--out_dir",
            str(out_dir),
        ], cwd=ROOT)
        meta = json.loads((out_dir / "meta.json").read_text())
        leaderboard.append({"name": name, **meta})
        cnn_prob_paths.append(str(out_dir / "val_probs.npy"))

    ens_dir = run_dir / "ensemble"
    run([
        sys.executable,
        "scripts/ensemble_from_probs.py",
        "--tab_probs_list",
        *tab_prob_paths,
        "--cnn_probs_list",
        *cnn_prob_paths,
        "--out_dir",
        str(ens_dir),
    ], cwd=ROOT)

    # Write leaderboard summary
    leaderboard_path = run_dir / "leaderboard.csv"
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        f.write("name,weighted_f1,macro_f1,model\n")
        for entry in sorted(leaderboard, key=lambda x: x["weighted_f1"], reverse=True):
            f.write(f"{entry['name']},{entry['weighted_f1']:.4f},{entry['macro_f1']:.4f},{entry.get('model','cnn')}\n")

    print(f"Saved leaderboard to {leaderboard_path}")


if __name__ == "__main__":
    main()
