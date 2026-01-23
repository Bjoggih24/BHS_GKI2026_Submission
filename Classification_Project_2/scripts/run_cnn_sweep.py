#!/usr/bin/env python3
"""
Run sequential CNN sweeps with robust logging (CPU-safe).
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

CNN_MODELS = [
    ("cnn_small_1", "small_cnn", "configs/cnn_targeted/cnn_small_1.json"),
    ("cnn_small_2", "small_cnn", "configs/cnn_targeted/cnn_small_2.json"),
    ("resnet18_15ch", "resnet18", "configs/cnn_targeted/resnet18_15ch.json"),
    ("resnet34_15ch", "resnet34", "configs/cnn_targeted/resnet34_15ch.json"),
    ("tf_efficientnet_b0_15ch", "tf_efficientnet_b0", "configs/cnn_targeted/tf_efficientnet_b0_15ch.json"),
    ("mobilenetv3_large_15ch", "mobilenetv3_large_100", "configs/cnn_targeted/mobilenetv3_large_15ch.json"),
]


def next_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        candidate = base / f"run_{idx:03d}"
        if not candidate.exists():
            return candidate
        idx += 1


def log_line(fp: Path, msg: str) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


def run_cmd(cmd, cwd, stdout_path: Path, stderr_path: Path) -> int:
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


def available_memory_gib():
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return None


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


def log_versions(run_log: Path):
    log_line(run_log, f"python: {sys.version.split()[0]}")
    try:
        import numpy as np

        log_line(run_log, f"numpy: {np.__version__}")
    except Exception as exc:
        log_line(run_log, f"numpy: error {exc}")
    try:
        import torch

        log_line(run_log, f"torch: {torch.__version__}")
    except Exception as exc:
        log_line(run_log, f"torch: error {exc}")
    try:
        import timm

        log_line(run_log, f"timm: {timm.__version__}")
    except Exception as exc:
        log_line(run_log, f"timm: error {exc}")


def preflight_checks() -> None:
    required = [
        ROOT / "data" / "train" / "patches.npy",
        ROOT / "artifacts" / "features" / "y.npy",
    ]
    for _, _, cfg_rel in CNN_MODELS:
        required.append(ROOT / cfg_rel)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def _load_meta_if_exists(meta_path: Path):
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CNN sweep with robust logging")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--experiments_subdir", type=str, default="experiments/cnn_runs")
    parser.add_argument("--abort_on_low_mem", action="store_true")
    args = parser.parse_args()

    base_dir = ARTIFACTS_DIR / args.experiments_subdir
    run_dir = next_run_dir(base_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / "run_log.txt"
    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"

    log_line(run_log, f"[run] {run_dir}")
    log_line(run_log, f"git: {get_git_hash()}")
    log_line(run_log, "note: do not run other training jobs concurrently on this VM.")
    log_versions(run_log)
    preflight_checks()

    models = list(CNN_MODELS)
    if args.only:
        want = {name.strip() for name in args.only}
        models = [entry for entry in models if entry[0] in want]
        missing = want - {name for name, _, _ in CNN_MODELS}
        if missing:
            raise ValueError(f"Unknown model names in --only: {', '.join(sorted(missing))}")
        if not models:
            raise ValueError("No models selected. Check --only arguments.")

    cols = [
        "name",
        "arch",
        "exit_code",
        "weighted_f1",
        "macro_f1",
        "best_epoch",
        "duration_sec",
        "val_probs_shape",
        "params_json",
    ]
    summary_rows = []
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)

    for name, arch, cfg_rel in models:
        mem_gib = available_memory_gib()
        if mem_gib is not None and mem_gib < 3.5:
            msg = f"[warn] low memory available: {mem_gib:.2f} GiB"
            log_line(run_log, msg)
            if args.abort_on_low_mem:
                raise RuntimeError(msg)

        out_dir = run_dir / "cnn" / name
        stdout_path = out_dir / "stdout.txt"
        stderr_path = out_dir / "stderr.txt"
        cmd_path = out_dir / "cmd.txt"
        meta_path = out_dir / "meta.json"

        row = {
            "name": name,
            "arch": arch,
            "exit_code": 0,
            "weighted_f1": None,
            "macro_f1": None,
            "best_epoch": None,
            "duration_sec": None,
            "val_probs_shape": None,
            "params_json": str(ROOT / cfg_rel),
        }

        if meta_path.exists():
            meta = _load_meta_if_exists(meta_path)
            if meta and meta.get("weighted_f1") is not None:
                log_line(run_log, f"[skip] {name} already completed")
                row.update(
                    {
                        "weighted_f1": meta.get("weighted_f1"),
                        "macro_f1": meta.get("macro_f1"),
                        "best_epoch": meta.get("best_epoch"),
                    }
                )
                if (out_dir / "val_probs.npy").exists():
                    try:
                        import numpy as np

                        p = np.load(out_dir / "val_probs.npy", mmap_mode="r")
                        row["val_probs_shape"] = list(p.shape)
                    except Exception:
                        row["val_probs_shape"] = None
                summary_rows.append(row)
                with summary_csv.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([row[c] for c in cols])
                continue

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_cnn.py"),
            "--arch",
            arch,
            "--params_json",
            str(ROOT / cfg_rel),
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

        log_line(run_log, f"[start] {name} ({arch})")
        start = time.time()
        if args.dry_run:
            log_line(run_log, f"[dry_run] {' '.join(cmd)}")
            row["duration_sec"] = 0.0
        else:
            exit_code = run_cmd(cmd, ROOT, stdout_path, stderr_path)
            row["exit_code"] = exit_code
            row["duration_sec"] = round(time.time() - start, 2)
            meta = _load_meta_if_exists(meta_path)
            if meta:
                row.update(
                    {
                        "weighted_f1": meta.get("weighted_f1"),
                        "macro_f1": meta.get("macro_f1"),
                        "best_epoch": meta.get("best_epoch"),
                    }
                )
            if (out_dir / "val_probs.npy").exists():
                try:
                    import numpy as np

                    p = np.load(out_dir / "val_probs.npy", mmap_mode="r")
                    row["val_probs_shape"] = list(p.shape)
                except Exception:
                    row["val_probs_shape"] = None

            log_line(run_log, f"[done] {name} exit_code={exit_code}")

        summary_rows.append(row)
        with summary_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([row[c] for c in cols])

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    log_line(run_log, f"[done] summary_rows={len(summary_rows)}")
    print(f"[run] {run_dir}")


if __name__ == "__main__":
    main()
