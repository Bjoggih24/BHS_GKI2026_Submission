#!/usr/bin/env python3
"""
Overnight run: v3 tabular features + 1-2 CNN seeds (sequential, logged).
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"

TABULAR_MODELS = [
    ("ET_overfit_v3", "extratrees", "configs/overnight_v3/ET_overfit_v3.json"),
    ("ET_overfit_v3_b", "extratrees", "configs/overnight_v3/ET_overfit_v3_b.json"),
    ("ET_overfit_v3_c", "extratrees", "configs/overnight_v3/ET_overfit_v3_c.json"),
    ("ET_overfit_v3_d", "extratrees", "configs/overnight_v3/ET_overfit_v3_d.json"),
    ("ET_A_v3", "extratrees", "configs/overnight_v3/ET_A_v3.json"),
    ("ET_C_v3", "extratrees", "configs/overnight_v3/ET_C_v3.json"),
    ("LGBM_A_v3", "lgbm", "configs/overnight_v3/LGBM_A_v3.json"),
    ("XGB_A_v3", "xgb", "configs/overnight_v3/XGB_A_v3.json"),
    ("MLP_TORCH_E_v3", "torch_mlp", "configs/tabular_targeted/MLP_TORCH_E.json"),
    ("TRANSFORMER_TORCH_SMALL_v3", "torch_transformer", "configs/tabular_targeted/TRANSFORMER_TORCH_SMALL.json"),
]

CNN_MODELS = [
    ("cnn_small_2_seed4", "configs/overnight_v3/cnn_small_2_seed4.json", 4),
    ("cnn_small_2_nolog1p_seed5", "configs/overnight_v3/cnn_small_2_nolog1p_seed5.json", 5),
    ("cnn_small_2_seed6", "configs/cnn_targeted/cnn_small_2.json", 6),
    ("cnn_small_2_aug_ls02_seed7", "configs/cnn_targeted/cnn_small_2_aug_ls02.json", 7),
    ("cnn_small_2_aug_ls08_seed8", "configs/cnn_targeted/cnn_small_2_aug_ls08.json", 8),
    # Additional seeds for ensemble diversity
    ("cnn_small_2_aug_ls02_seed9", "configs/cnn_targeted/cnn_small_2_aug_ls02.json", 9),
    ("cnn_small_2_aug_ls02_seed10", "configs/cnn_targeted/cnn_small_2_aug_ls02.json", 10),
]


def next_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        candidate = base / f"run_overnight_{idx:03d}"
        if not candidate.exists():
            return candidate
        idx += 1


def log_line(fp: Path, msg: str):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


def run_cmd(cmd, cwd, stdout_path: Path, stderr_path: Path):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        cmd_line = "$ " + " ".join(cmd) + "\n"
        out.write(cmd_line)
        err.write(cmd_line)
        out.flush()
        err.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=err, env=env, text=True)
        return proc.wait()


def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def ensure_features_v3() -> Path:
    feats_dir = ARTIFACTS_DIR / "features"
    X_path = feats_dir / "X_tabular_v3.npy"
    if not X_path.exists():
        code = run_cmd(
            [sys.executable, "scripts/precompute_tabular_features.py", "--version", "v3"],
            ROOT,
            stdout_path=ARTIFACTS_DIR / "features" / "v3_stdout.txt",
            stderr_path=ARTIFACTS_DIR / "features" / "v3_stderr.txt",
        )
        if code != 0:
            raise RuntimeError(f"precompute v3 failed with exit_code={code}")
        if not X_path.exists():
            raise RuntimeError("precompute v3 reported success but X_tabular_v3.npy is missing")
    return X_path


def preflight_checks() -> None:
    required = [
        ARTIFACTS_DIR / "features" / "y.npy",
        ARTIFACTS_DIR / "split_seed42" / "train_idx.npy",
        ARTIFACTS_DIR / "split_seed42" / "val_idx.npy",
        ROOT / "scripts" / "train_tabular.py",
        ROOT / "scripts" / "train_cnn.py",
        ROOT / "scripts" / "precompute_tabular_features.py",
    ]
    for _, _, cfg_rel in TABULAR_MODELS:
        required.append(ROOT / cfg_rel)
    for _, cfg_rel, _ in CNN_MODELS:
        required.append(ROOT / cfg_rel)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def main():
    parser = argparse.ArgumentParser(description="Overnight v3 tabular + CNN sweep")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--experiments_subdir", type=str, default="experiments/overnight_v3_runs")
    args = parser.parse_args()

    preflight_checks()
    run_dir = next_run_dir(ARTIFACTS_DIR / args.experiments_subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / "run_log.txt"
    summary_csv = run_dir / "summary.csv"
    summary_json = run_dir / "summary.json"

    log_line(run_log, f"[run] {run_dir}")
    log_line(run_log, f"git: {get_git_hash()}")

    X_path = ensure_features_v3()
    y_path = ARTIFACTS_DIR / "features" / "y.npy"

    cols = [
        "name",
        "model",
        "exit_code",
        "weighted_f1",
        "macro_f1",
        "best_iteration",
        "best_score",
        "feature_version",
        "val_probs_shape",
        "duration_sec",
    ]
    summary_rows = []
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)

    for name, model, cfg_rel in TABULAR_MODELS:
        out_dir = run_dir / "tabular" / name
        stdout_path = out_dir / "stdout.txt"
        stderr_path = out_dir / "stderr.txt"
        cmd_path = out_dir / "cmd.txt"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_tabular.py"),
            "--model",
            model,
            "--params_json",
            str(ROOT / cfg_rel),
            "--features_path",
            str(X_path),
            "--labels_path",
            str(y_path),
            "--out_dir",
            str(out_dir),
            "--seed",
            "42",
            "--use_class_weights",
            "1",
            "--weight_power",
            "0.25",
        ]
        cmd_path.parent.mkdir(parents=True, exist_ok=True)
        cmd_path.write_text(" ".join(cmd), encoding="utf-8")
        log_line(run_log, f"[start] {name} ({model})")
        start = time.time()
        if args.dry_run:
            log_line(run_log, f"[dry_run] {' '.join(cmd)}")
            exit_code = 0
        else:
            exit_code = run_cmd(cmd, ROOT, stdout_path, stderr_path)
        duration = time.time() - start

        row = {
            "name": name,
            "model": model,
            "exit_code": exit_code,
            "weighted_f1": None,
            "macro_f1": None,
            "best_iteration": None,
            "best_score": None,
            "feature_version": None,
            "val_probs_shape": None,
            "duration_sec": round(duration, 2),
        }
        meta_path = out_dir / "meta.json"
        probs_path = out_dir / "val_probs.npy"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            row["weighted_f1"] = meta.get("weighted_f1")
            row["macro_f1"] = meta.get("macro_f1")
            row["best_iteration"] = meta.get("best_iteration")
            row["best_score"] = meta.get("best_score")
            row["feature_version"] = meta.get("feature_version")
        if probs_path.exists():
            try:
                import numpy as np

                p = np.load(probs_path, mmap_mode="r")
                row["val_probs_shape"] = tuple(p.shape)
            except Exception:
                row["val_probs_shape"] = None

        summary_rows.append(row)
        with summary_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([row[c] for c in cols])
        summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
        log_line(run_log, f"[done] {name} exit_code={exit_code} duration={duration:.2f}s")

    for name, cfg_rel, seed in CNN_MODELS:
        out_dir = run_dir / "cnn" / name
        stdout_path = out_dir / "stdout.txt"
        stderr_path = out_dir / "stderr.txt"
        cmd_path = out_dir / "cmd.txt"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_cnn.py"),
            "--arch",
            "small_cnn",
            "--params_json",
            str(ROOT / cfg_rel),
            "--out_dir",
            str(out_dir),
            "--seed",
            str(seed),
            "--use_class_weights",
            "1",
            "--weight_power",
            "0.25",
        ]
        cmd_path.parent.mkdir(parents=True, exist_ok=True)
        cmd_path.write_text(" ".join(cmd), encoding="utf-8")
        log_line(run_log, f"[start] {name} (cnn)")
        start = time.time()
        if args.dry_run:
            log_line(run_log, f"[dry_run] {' '.join(cmd)}")
            exit_code = 0
        else:
            exit_code = run_cmd(cmd, ROOT, stdout_path, stderr_path)
        duration = time.time() - start

        row = {
            "name": name,
            "model": "cnn",
            "exit_code": exit_code,
            "weighted_f1": None,
            "macro_f1": None,
            "best_iteration": None,
            "best_score": None,
            "feature_version": None,
            "val_probs_shape": None,
            "duration_sec": round(duration, 2),
        }
        meta_path = out_dir / "meta.json"
        probs_path = out_dir / "val_probs.npy"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            row["weighted_f1"] = meta.get("weighted_f1")
            row["macro_f1"] = meta.get("macro_f1")
            row["best_iteration"] = meta.get("best_epoch")
        if probs_path.exists():
            try:
                import numpy as np

                p = np.load(probs_path, mmap_mode="r")
                row["val_probs_shape"] = tuple(p.shape)
            except Exception:
                row["val_probs_shape"] = None

        summary_rows.append(row)
        with summary_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([row[c] for c in cols])
        summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
        log_line(run_log, f"[done] {name} exit_code={exit_code} duration={duration:.2f}s")

    log_line(run_log, "[run] complete")
    print(f"[run] {run_dir}")


if __name__ == "__main__":
    main()
