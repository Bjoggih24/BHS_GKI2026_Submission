#!/usr/bin/env python3
"""
Run a sequence of tabular trainings and log outputs.
"""

import argparse
import json
import subprocess
from pathlib import Path


DEFAULT_CONFIGS = {
    "XGB1": {
        "objective": "multi:softprob",
        "num_class": 71,
        "tree_method": "hist",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 20000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
        "gamma": 0.0,
        "n_jobs": -1,
    },
    "XGB2": {
        "objective": "multi:softprob",
        "num_class": 71,
        "tree_method": "hist",
        "max_depth": 7,
        "learning_rate": 0.03,
        "n_estimators": 30000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 10.0,
        "reg_alpha": 0.5,
        "gamma": 0.0,
        "n_jobs": -1,
    },
    "LGBM1": {
        "objective": "multiclass",
        "num_class": 71,
        "n_estimators": 20000,
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
        "n_jobs": -1,
    },
    "LGBM2": {
        "objective": "multiclass",
        "num_class": 71,
        "n_estimators": 30000,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_child_samples": 40,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_lambda": 10.0,
        "reg_alpha": 0.5,
        "n_jobs": -1,
    },
    "ET1": {
        "n_estimators": 2000,
        "max_features": 0.25,
        "min_samples_leaf": 1,
        "n_jobs": -1,
    },
    "ET2": {
        "n_estimators": 2000,
        "max_features": 0.35,
        "min_samples_leaf": 3,
        "n_jobs": -1,
    },
}


def ensure_default_config(name: str, path: Path):
    if path.exists():
        return
    if name not in DEFAULT_CONFIGS:
        raise ValueError(f"No default config for '{name}'. Create {path} manually.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_CONFIGS[name], indent=2), encoding="utf-8")


def run_one(cmd, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        return proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Run tabular training batch")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--features_path", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_class_weights", type=int, default=1)
    parser.add_argument("--weight_power", type=float, default=0.25)
    parser.add_argument("--write_default_configs", type=int, default=0)
    args = parser.parse_args()

    if not (len(args.configs) == len(args.models) == len(args.names)):
        raise ValueError("configs, models, and names must have the same length")

    root = Path(__file__).resolve().parents[1]
    for cfg, model, name in zip(args.configs, args.models, args.names):
        cfg_path = Path(cfg)
        if args.write_default_configs:
            ensure_default_config(name, cfg_path)
        out_dir = args.out_root / name
        log_path = args.out_root / f"{name}.log"
        cmd = [
            str(root / "scripts" / "train_tabular.py"),
            "--model",
            model,
            "--params_json",
            cfg,
            "--features_path",
            str(args.features_path),
            "--labels_path",
            str(args.labels_path),
            "--out_dir",
            str(out_dir),
            "--seed",
            str(args.seed),
            "--use_class_weights",
            str(args.use_class_weights),
            "--weight_power",
            str(args.weight_power),
        ]
        cmd = ["python3", *cmd]
        print(f"\n=== Running {name} ({model}) ===")
        exit_code = run_one(cmd, log_path)
        if exit_code != 0:
            raise SystemExit(f"{name} failed with exit code {exit_code}. See {log_path}")


if __name__ == "__main__":
    main()
