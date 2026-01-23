#!/usr/bin/env python3
"""
Overnight tabular sweep runner (sequential, with full logging).
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
    ("XGB1", "xgb", "configs/tabular/XGB1.json"),
    ("XGB2", "xgb", "configs/tabular/XGB2.json"),
    ("LGBM1", "lgbm", "configs/tabular/LGBM1.json"),
    ("LGBM2", "lgbm", "configs/tabular/LGBM2.json"),
    ("LGBM3", "lgbm", "configs/tabular/LGBM3.json"),
    ("ET1", "extratrees", "configs/tabular/ET1.json"),
    ("ET2", "extratrees", "configs/tabular/ET2.json"),
    ("XGBT1", "xgb", "configs/tabular_targeted/XGBT1.json"),
    ("LGBMT1", "lgbm", "configs/tabular_targeted/LGBMT1.json"),
    ("ET1_BEEF", "extratrees", "configs/tabular_targeted/ET1_beef.json"),
    ("ET2_BEEF", "extratrees", "configs/tabular_targeted/ET2_beef.json"),
    ("ET3", "extratrees", "configs/tabular_targeted/ET3.json"),
    ("ET4", "extratrees", "configs/tabular_targeted/ET4.json"),
    ("ET5", "extratrees", "configs/tabular_targeted/ET5.json"),
    ("ET6", "extratrees", "configs/tabular_targeted/ET6.json"),
    ("ET7", "extratrees", "configs/tabular_targeted/ET7.json"),
    ("ET8", "extratrees", "configs/tabular_targeted/ET8.json"),
    ("ET9", "extratrees", "configs/tabular_targeted/ET9.json"),
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
        ROOT / "scripts" / "train_tabular.py",
        ARTIFACTS_DIR / "features" / "X_tabular_v2.npy",
        ARTIFACTS_DIR / "features" / "y.npy",
        ROOT / "configs" / "tabular" / "XGB1.json",
        ROOT / "configs" / "tabular" / "XGB2.json",
        ROOT / "configs" / "tabular" / "LGBM1.json",
        ROOT / "configs" / "tabular" / "LGBM2.json",
        ROOT / "configs" / "tabular" / "LGBM3.json",
        ROOT / "configs" / "tabular" / "ET1.json",
        ROOT / "configs" / "tabular" / "ET2.json",
        ROOT / "configs" / "tabular_targeted" / "XGBT1.json",
        ROOT / "configs" / "tabular_targeted" / "LGBMT1.json",
        ROOT / "configs" / "tabular_targeted" / "ET1_beef.json",
        ROOT / "configs" / "tabular_targeted" / "ET2_beef.json",
        ROOT / "configs" / "tabular_targeted" / "ET3.json",
        ROOT / "configs" / "tabular_targeted" / "ET4.json",
        ROOT / "configs" / "tabular_targeted" / "ET5.json",
        ROOT / "configs" / "tabular_targeted" / "ET6.json",
        ROOT / "configs" / "tabular_targeted" / "ET7.json",
        ROOT / "configs" / "tabular_targeted" / "ET8.json",
        ROOT / "configs" / "tabular_targeted" / "ET9.json",
        ARTIFACTS_DIR / "split_seed42" / "train_idx.npy",
        ARTIFACTS_DIR / "split_seed42" / "val_idx.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    train_tabular = (ROOT / "scripts" / "train_tabular.py").read_text(encoding="utf-8")
    required_snippets = [
        "use_xgb_train",
        "xgb.train",
        "early_stopping_rounds",
        "params_raw",
        "params_used",
        "val_probs",
    ]
    for snippet in required_snippets:
        if snippet not in train_tabular:
            raise RuntimeError(f"train_tabular.py missing expected content: {snippet}")


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
        import xgboost
        log_line(run_log, f"xgboost: {xgboost.__version__}")
    except Exception as exc:
        log_line(run_log, f"xgboost: error {exc}")
    try:
        import lightgbm
        log_line(run_log, f"lightgbm: {lightgbm.__version__}")
    except Exception as exc:
        log_line(run_log, f"lightgbm: error {exc}")


def log_feature_shapes(run_log: Path):
    import numpy as np
    X = np.load(ARTIFACTS_DIR / "features" / "X_tabular_v2.npy", mmap_mode="r")
    y = np.load(ARTIFACTS_DIR / "features" / "y.npy")
    val_idx = np.load(ARTIFACTS_DIR / "split_seed42" / "val_idx.npy")
    log_line(run_log, f"X_tabular_v2 shape: {X.shape}")
    log_line(run_log, f"y shape: {y.shape}")
    log_line(run_log, f"val_idx len: {len(val_idx)}")


def main():
    parser = argparse.ArgumentParser(description="Run overnight tabular sweep")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--experiments_subdir", type=str, default="experiments")
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
    log_feature_shapes(run_log)

    features_path = ARTIFACTS_DIR / "features" / "X_tabular_v2.npy"
    labels_path = ARTIFACTS_DIR / "features" / "y.npy"
    val_idx = None
    try:
        import numpy as np
        val_idx = np.load(ARTIFACTS_DIR / "split_seed42" / "val_idx.npy")
    except Exception:
        pass

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

    selected = MODELS
    if args.only:
        want = {name.strip() for name in args.only}
        selected = [entry for entry in MODELS if entry[0] in want]
        missing = want.difference({name for name, _, _ in MODELS})
        if missing:
            raise ValueError(f"Unknown model names in --only: {', '.join(sorted(missing))}")
        if not selected:
            raise ValueError("No models selected. Check --only arguments.")

    for name, model, cfg_rel in selected:
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

        meta_path = out_dir / "meta.json"
        probs_path = out_dir / "val_probs.npy"
        if meta_path.exists():
            try:
                meta_existing = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta_existing = None
            if meta_existing and meta_existing.get("weighted_f1") is not None:
                row["weighted_f1"] = meta_existing.get("weighted_f1")
                row["macro_f1"] = meta_existing.get("macro_f1")
                row["best_iteration"] = meta_existing.get("best_iteration")
                row["best_score"] = meta_existing.get("best_score")
                row["feature_version"] = meta_existing.get("feature_version")
                if probs_path.exists():
                    try:
                        import numpy as np

                        p = np.load(probs_path, mmap_mode="r")
                        row["val_probs_shape"] = tuple(p.shape)
                    except Exception as exc:
                        log_line(run_log, f"[warn] {name} failed loading val_probs.npy: {exc}")
                row["duration_sec"] = round(time.time() - start, 2)
                log_line(run_log, f"[skip] {name} already completed")
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
                continue

        if args.dry_run:
            log_line(run_log, f"[dry_run] {' '.join(cmd)}")
            exit_code = 0
        else:
            exit_code = run_cmd(cmd, ROOT, stdout_path, stderr_path)

        duration = time.time() - start
        row["exit_code"] = exit_code
        row["duration_sec"] = round(duration, 2)

        if exit_code != 0:
            log_line(run_log, f"[fail] {name} exit_code={exit_code} duration={duration:.2f}s")
        else:
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
                    if val_idx is not None and p.shape[0] != len(val_idx):
                        log_line(run_log, f"[warn] {name} val_probs shape {p.shape} != {len(val_idx)}")
                except Exception as exc:
                    log_line(run_log, f"[warn] {name} failed loading val_probs.npy: {exc}")
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
