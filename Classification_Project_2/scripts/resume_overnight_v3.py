#!/usr/bin/env python3
"""
Resume overnight v3 run from where it stopped (ET_overfit_v3_c onwards).
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

# Models to run (starting from ET_overfit_v3_c)
TABULAR_MODELS = [
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
    ("cnn_small_2_aug_ls02_seed9", "configs/cnn_targeted/cnn_small_2_aug_ls02.json", 9),
    ("cnn_small_2_aug_ls02_seed10", "configs/cnn_targeted/cnn_small_2_aug_ls02.json", 10),
]


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


def main():
    parser = argparse.ArgumentParser(description="Resume overnight v3 from ET_overfit_v3_c")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the run_overnight_003 directory")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_log = run_dir / "run_log.txt"
    summary_csv = run_dir / "summary.csv"
    summary_json = run_dir / "summary.json"

    X_path = ARTIFACTS_DIR / "features" / "X_tabular_v3.npy"
    y_path = ARTIFACTS_DIR / "features" / "y.npy"
    if not X_path.exists():
        raise FileNotFoundError(f"Features not found: {X_path}")

    # Load existing summary if available
    summary_rows = []
    if summary_json.exists():
        try:
            summary_rows = json.loads(summary_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    cols = [
        "name", "model", "exit_code", "weighted_f1", "macro_f1",
        "best_iteration", "best_score", "feature_version", "val_probs_shape", "duration_sec",
    ]

    log_line(run_log, f"[resume] Resuming from ET_overfit_v3_c")

    for name, model, cfg_rel in TABULAR_MODELS:
        out_dir = run_dir / "tabular" / name
        # Skip if already completed
        meta_path = out_dir / "meta.json"
        if meta_path.exists():
            log_line(run_log, f"[skip] {name} already completed")
            print(f"[skip] {name} already completed")
            continue

        stdout_path = out_dir / "stdout.txt"
        stderr_path = out_dir / "stderr.txt"
        cmd_path = out_dir / "cmd.txt"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_tabular.py"),
            "--model", model,
            "--params_json", str(ROOT / cfg_rel),
            "--features_path", str(X_path),
            "--labels_path", str(y_path),
            "--out_dir", str(out_dir),
            "--seed", "42",
            "--use_class_weights", "1",
            "--weight_power", "0.25",
        ]
        cmd_path.parent.mkdir(parents=True, exist_ok=True)
        cmd_path.write_text(" ".join(cmd), encoding="utf-8")
        log_line(run_log, f"[start] {name} ({model})")
        print(f"[start] {name} ({model})")
        start = time.time()
        if args.dry_run:
            log_line(run_log, f"[dry_run] {' '.join(cmd)}")
            exit_code = 0
        else:
            exit_code = run_cmd(cmd, ROOT, stdout_path, stderr_path)
        duration = time.time() - start

        row = {
            "name": name, "model": model, "exit_code": exit_code,
            "weighted_f1": None, "macro_f1": None, "best_iteration": None,
            "best_score": None, "feature_version": None, "val_probs_shape": None,
            "duration_sec": round(duration, 2),
        }
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            row["weighted_f1"] = meta.get("weighted_f1")
            row["macro_f1"] = meta.get("macro_f1")
            row["best_iteration"] = meta.get("best_iteration")
            row["best_score"] = meta.get("best_score")
            row["feature_version"] = meta.get("feature_version")
        probs_path = out_dir / "val_probs.npy"
        if probs_path.exists():
            try:
                import numpy as np
                p = np.load(probs_path, mmap_mode="r")
                row["val_probs_shape"] = tuple(p.shape)
            except Exception:
                pass

        summary_rows.append(row)
        with summary_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([row[c] for c in cols])
        summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
        log_line(run_log, f"[done] {name} exit_code={exit_code} duration={duration:.2f}s")
        print(f"[done] {name} exit_code={exit_code} duration={duration:.2f}s")

    for name, cfg_rel, seed in CNN_MODELS:
        out_dir = run_dir / "cnn" / name
        meta_path = out_dir / "meta.json"
        if meta_path.exists():
            log_line(run_log, f"[skip] {name} already completed")
            print(f"[skip] {name} already completed")
            continue

        stdout_path = out_dir / "stdout.txt"
        stderr_path = out_dir / "stderr.txt"
        cmd_path = out_dir / "cmd.txt"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_cnn.py"),
            "--arch", "small_cnn",
            "--params_json", str(ROOT / cfg_rel),
            "--out_dir", str(out_dir),
            "--seed", str(seed),
            "--use_class_weights", "1",
            "--weight_power", "0.25",
        ]
        cmd_path.parent.mkdir(parents=True, exist_ok=True)
        cmd_path.write_text(" ".join(cmd), encoding="utf-8")
        log_line(run_log, f"[start] {name} (cnn)")
        print(f"[start] {name} (cnn)")
        start = time.time()
        if args.dry_run:
            log_line(run_log, f"[dry_run] {' '.join(cmd)}")
            exit_code = 0
        else:
            exit_code = run_cmd(cmd, ROOT, stdout_path, stderr_path)
        duration = time.time() - start

        row = {
            "name": name, "model": "cnn", "exit_code": exit_code,
            "weighted_f1": None, "macro_f1": None, "best_iteration": None,
            "best_score": None, "feature_version": None, "val_probs_shape": None,
            "duration_sec": round(duration, 2),
        }
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            row["weighted_f1"] = meta.get("weighted_f1")
            row["macro_f1"] = meta.get("macro_f1")
            row["best_iteration"] = meta.get("best_epoch")
        probs_path = out_dir / "val_probs.npy"
        if probs_path.exists():
            try:
                import numpy as np
                p = np.load(probs_path, mmap_mode="r")
                row["val_probs_shape"] = tuple(p.shape)
            except Exception:
                pass

        summary_rows.append(row)
        with summary_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([row[c] for c in cols])
        summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
        log_line(run_log, f"[done] {name} exit_code={exit_code} duration={duration:.2f}s")
        print(f"[done] {name} exit_code={exit_code} duration={duration:.2f}s")

    log_line(run_log, "[run] complete (resumed)")
    print("[run] complete")


if __name__ == "__main__":
    main()
