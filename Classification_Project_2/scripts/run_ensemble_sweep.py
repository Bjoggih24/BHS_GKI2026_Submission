#!/usr/bin/env python3
"""
Run sequential ensemble tuning jobs with robust logging.
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


def default_candidates(cnn_probs: Path) -> list:
    return [
        {
            "name": "ET1_ET3_CNN",
            "probs": [
                ROOT / "artifacts/experiments/run_overnight_002/tabular/ET1/val_probs.npy",
                ROOT / "artifacts/experiments/targeted_ET_runs/run_overnight_007/tabular/ET3/val_probs.npy",
                cnn_probs,
            ],
        },
        {
            "name": "ET1_LGBM2_CNN",
            "probs": [
                ROOT / "artifacts/experiments/run_overnight_002/tabular/ET1/val_probs.npy",
                ROOT / "artifacts/experiments/run_overnight_002/tabular/LGBM2/val_probs.npy",
                cnn_probs,
            ],
        },
        {
            "name": "ET1_XGB1_CNN",
            "probs": [
                ROOT / "artifacts/experiments/run_overnight_002/tabular/ET1/val_probs.npy",
                ROOT / "artifacts/experiments/run_overnight_002/tabular/XGB1/val_probs.npy",
                cnn_probs,
            ],
        },
        {
            "name": "ET2_ET3_CNN",
            "probs": [
                ROOT / "artifacts/experiments/run_overnight_002/tabular/ET2/val_probs.npy",
                ROOT / "artifacts/experiments/targeted_ET_runs/run_overnight_007/tabular/ET3/val_probs.npy",
                cnn_probs,
            ],
        },
    ]


def load_candidates(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "candidates" in data:
        data = data["candidates"]
    if not isinstance(data, list):
        raise ValueError("Candidates JSON must be a list or contain a 'candidates' list.")
    return data


def preflight_checks(candidates, labels_path: Path, split_dir: Path) -> None:
    required = [labels_path, split_dir / "train_idx.npy", split_dir / "val_idx.npy"]
    for cand in candidates:
        for p in cand["probs"]:
            required.append(p)
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequential ensemble tuning sweeps")
    parser.add_argument("--experiments_subdir", type=str, default="experiments/ensemble_runs")
    parser.add_argument(
        "--cnn_probs",
        type=Path,
        default=ROOT / "artifacts/experiments/cnn_runs/run_002/cnn/cnn_small_2/val_probs.npy",
    )
    parser.add_argument("--labels_path", type=Path, default=ROOT / "artifacts/features/y.npy")
    parser.add_argument("--split_dir", type=Path, default=ROOT / "artifacts/split_seed42")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--candidates_json", type=Path, default=None)
    args = parser.parse_args()

    run_dir = next_run_dir(ARTIFACTS_DIR / args.experiments_subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / "run_log.txt"
    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"

    log_line(run_log, f"[run] {run_dir}")
    log_line(run_log, f"git: {get_git_hash()}")
    log_line(run_log, f"cnn_probs: {args.cnn_probs}")
    log_line(run_log, f"step: {args.step}")

    if args.candidates_json is not None:
        candidates = load_candidates(args.candidates_json)
    else:
        candidates = default_candidates(args.cnn_probs)
    preflight_checks(candidates, args.labels_path, args.split_dir)

    cols = [
        "name",
        "exit_code",
        "weighted_f1",
        "macro_f1",
        "weights",
        "duration_sec",
        "out_json",
    ]
    summary_rows = []
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)

    for cand in candidates:
        name = cand["name"]
        out_dir = run_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = out_dir / "stdout.txt"
        stderr_path = out_dir / "stderr.txt"
        cmd_path = out_dir / "cmd.txt"
        out_json = out_dir / "ensemble_weights.json"

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "ensemble_val.py"),
            "--probs",
            *[str(p) for p in cand["probs"]],
            "--labels_path",
            str(args.labels_path),
            "--split_dir",
            str(args.split_dir),
            "--out_json",
            str(out_json),
            "--step",
            str(args.step),
        ]
        cmd_path.write_text(" ".join(cmd), encoding="utf-8")

        log_line(run_log, f"[start] {name}")
        start = time.time()
        exit_code = run_cmd(cmd, ROOT, stdout_path, stderr_path)
        duration = round(time.time() - start, 2)
        log_line(run_log, f"[done] {name} exit_code={exit_code} duration={duration:.2f}s")

        row = {
            "name": name,
            "exit_code": exit_code,
            "weighted_f1": None,
            "macro_f1": None,
            "weights": None,
            "duration_sec": duration,
            "out_json": str(out_json),
        }
        if out_json.exists():
            try:
                meta = json.loads(out_json.read_text(encoding="utf-8"))
                row["weighted_f1"] = meta.get("weighted_f1")
                row["macro_f1"] = meta.get("macro_f1")
                row["weights"] = meta.get("weights")
            except Exception:
                pass

        summary_rows.append(row)
        with summary_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([row[c] for c in cols])
        summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print(f"[run] {run_dir}")


if __name__ == "__main__":
    main()
