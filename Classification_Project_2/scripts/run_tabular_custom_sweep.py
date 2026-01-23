#!/usr/bin/env python3
"""
Run a small custom tabular sweep with robust logging (sequential).
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

MODELS = [
    ("ET_A", "extratrees", "configs/tabular_targeted/ET_A.json"),
    ("ET_B", "extratrees", "configs/tabular_targeted/ET_B.json"),
    ("ET_C", "extratrees", "configs/tabular_targeted/ET_C.json"),
    ("ET_overfit", "extratrees", "configs/tabular_targeted/ET_overfit.json"),
    ("LGBM_A", "lgbm", "configs/tabular_targeted/LGBM_A.json"),
]


def next_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        candidate = base / f"run_custom_{idx:03d}"
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


def preflight_checks() -> None:
    required = [
        ARTIFACTS_DIR / "features" / "X_tabular_v2.npy",
        ARTIFACTS_DIR / "features" / "y.npy",
        ARTIFACTS_DIR / "split_seed42" / "train_idx.npy",
        ARTIFACTS_DIR / "split_seed42" / "val_idx.npy",
        ROOT / "scripts" / "train_tabular.py",
    ]
    for _, _, cfg_rel in MODELS:
        required.append(ROOT / cfg_rel)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def log_versions(run_log: Path):
    log_line(run_log, f"python: {sys.version.split()[0]}")
    try:
        import numpy as np

        log_line(run_log, f"numpy: {np.__version__}")
    except Exception as exc:
        log_line(run_log, f"numpy: error {exc}")
    try:
        import sklearn

        log_line(run_log, f"sklearn: {sklearn.__version__}")
    except Exception as exc:
        log_line(run_log, f"sklearn: error {exc}")
    try:
        import lightgbm

        log_line(run_log, f"lightgbm: {lightgbm.__version__}")
    except Exception as exc:
        log_line(run_log, f"lightgbm: error {exc}")


def main():
    parser = argparse.ArgumentParser(description="Run custom tabular sweep")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--experiments_subdir", type=str, default="experiments/targeted_ET_runs")
    args = parser.parse_args()

    preflight_checks()

    base_dir = ARTIFACTS_DIR / args.experiments_subdir
    run_dir = args.run_dir if args.run_dir else next_run_dir(base_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / "run_log.txt"
    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"

    log_line(run_log, f"[run] {run_dir}")
    log_line(run_log, f"git: {get_git_hash()}")
    log_versions(run_log)

    features_path = ARTIFACTS_DIR / "features" / "X_tabular_v2.npy"
    labels_path = ARTIFACTS_DIR / "features" / "y.npy"

    summary_rows = []
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
        )

    for name, model, cfg_rel in MODELS:
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
            str(features_path),
            "--labels_path",
            str(labels_path),
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
        row = {
            "name": name,
            "model": model,
            "exit_code": 0,
            "weighted_f1": None,
            "macro_f1": None,
            "best_iteration": None,
            "best_score": None,
            "feature_version": None,
            "val_probs_shape": None,
            "duration_sec": None,
        }

        if args.dry_run:
            log_line(run_log, f"[dry_run] {' '.join(cmd)}")
            exit_code = 0
        else:
            exit_code = run_cmd(cmd, ROOT, stdout_path, stderr_path)

        duration = time.time() - start
        row["exit_code"] = exit_code
        row["duration_sec"] = round(duration, 2)

        meta_path = out_dir / "meta.json"
        probs_path = out_dir / "val_probs.npy"
        if exit_code == 0 and meta_path.exists():
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

        if exit_code != 0:
            log_line(run_log, f"[fail] {name} exit_code={exit_code} duration={duration:.2f}s")
        else:
            log_line(run_log, f"[done] {name} exit_code={exit_code} duration={duration:.2f}s")

        summary_rows.append(row)
        with summary_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    row["name"],
                    row["model"],
                    row["exit_code"],
                    row["weighted_f1"],
                    row["macro_f1"],
                    row["best_iteration"],
                    row["best_score"],
                    row["feature_version"],
                    row["val_probs_shape"],
                    row["duration_sec"],
                ]
            )
        summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    log_line(run_log, "[run] complete")
    print(f"[run] {run_dir}")


if __name__ == "__main__":
    main()
