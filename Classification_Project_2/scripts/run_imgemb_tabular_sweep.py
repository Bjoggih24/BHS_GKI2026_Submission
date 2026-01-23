#!/usr/bin/env python3
"""
CPU-safe image-embedding extraction + tabular sweep runner.
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


EMBED_MODELS_BASE = [
    ("resnet18", "resnet18", 224, 16),
    ("tf_efficientnet_b0", "tf_efficientnet_b0", 224, 8),
    ("mobilenetv3_large_100", "mobilenetv3_large_100", 224, 16),
]

EMBED_MODELS_CONVNEXT = [
    ("convnext_tiny", "convnext_tiny", 224, 8),
]

TABULAR_MODELS_BASE = [
    ("ET_emb_1", "extratrees", "configs/tabular_targeted/ET_emb_1.json"),
    ("ET_emb_2", "extratrees", "configs/tabular_targeted/ET_emb_2.json"),
    ("ET_emb_3", "extratrees", "configs/tabular_targeted/ET_emb_3.json"),
    ("LGBM_emb_1", "lgbm", "configs/tabular_targeted/LGBM_emb_1.json"),
]

TABULAR_MODELS_XGB = [
    ("XGB_emb_1", "xgb", "configs/tabular_targeted/XGB_emb_1.json"),
]


def next_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        candidate = base / f"run_{idx:03d}"
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


def available_memory_gib():
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return None


def preflight_checks(include_xgb: bool, patches_npy: Path):
    required = [
        ARTIFACTS_DIR / "features" / "y.npy",
        patches_npy,
        ARTIFACTS_DIR / "split_seed42" / "train_idx.npy",
        ARTIFACTS_DIR / "split_seed42" / "val_idx.npy",
        ROOT / "scripts" / "train_tabular.py",
        ROOT / "scripts" / "extract_img_embeddings.py",
        ROOT / "configs" / "tabular_targeted" / "ET_emb_1.json",
        ROOT / "configs" / "tabular_targeted" / "ET_emb_2.json",
        ROOT / "configs" / "tabular_targeted" / "ET_emb_3.json",
        ROOT / "configs" / "tabular_targeted" / "LGBM_emb_1.json",
    ]
    if include_xgb:
        required.append(ROOT / "configs" / "tabular_targeted" / "XGB_emb_1.json")
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
        import torch

        log_line(run_log, f"torch: {torch.__version__}")
    except Exception as exc:
        log_line(run_log, f"torch: error {exc}")
    try:
        import timm

        log_line(run_log, f"timm: {timm.__version__}")
    except Exception as exc:
        log_line(run_log, f"timm: error {exc}")
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


def _load_meta_if_exists(meta_path: Path):
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Run image-embedding + tabular sweeps")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--include_convnext", action="store_true")
    parser.add_argument("--include_xgb", action="store_true")
    parser.add_argument("--experiments_subdir", type=str, default="experiments/imgemb_runs")
    parser.add_argument("--abort_on_low_mem", action="store_true")
    parser.add_argument(
        "--patches_npy",
        type=Path,
        default=ROOT / "data" / "train" / "patches.npy",
    )
    args = parser.parse_args()

    preflight_checks(include_xgb=args.include_xgb, patches_npy=args.patches_npy)

    base_dir = ARTIFACTS_DIR / args.experiments_subdir
    run_dir = next_run_dir(base_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / "run_log.txt"
    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"

    log_line(run_log, f"[run] {run_dir}")
    log_line(run_log, f"git: {get_git_hash()}")
    log_line(run_log, "note: Do not run TinyGRU/NLP training concurrently on this VM.")
    log_versions(run_log)

    embed_models = list(EMBED_MODELS_BASE)
    if args.include_convnext:
        embed_models.extend(EMBED_MODELS_CONVNEXT)

    all_names = {name for name, _, _, _ in embed_models}
    if args.only:
        want = {name.strip() for name in args.only}
        missing = want - all_names
        if missing:
            raise ValueError(f"Unknown embedding names in --only: {', '.join(sorted(missing))}")
        embed_models = [entry for entry in embed_models if entry[0] in want]
        if not embed_models:
            raise ValueError("No embedding models selected. Check --only arguments.")

    tabular_models = list(TABULAR_MODELS_BASE)
    if args.include_xgb:
        tabular_models.extend(TABULAR_MODELS_XGB)

    labels_path = ARTIFACTS_DIR / "features" / "y.npy"

    summary_rows = []
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "embedding_tag",
        "embedding_model",
        "tabular_name",
        "model",
        "exit_code",
        "weighted_f1",
        "macro_f1",
        "best_iteration",
        "best_score",
        "feature_version",
        "val_probs_shape",
        "duration_sec",
        "features_path",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)

    for tag, model_name, img_size, batch_size in embed_models:
        mem_gib = available_memory_gib()
        if mem_gib is not None and mem_gib < 3.5:
            msg = f"[warn] low memory available: {mem_gib:.2f} GiB"
            log_line(run_log, msg)
            if args.abort_on_low_mem:
                raise RuntimeError(msg)

        emb_dir = run_dir / "embeddings" / tag
        emb_path = emb_dir / "X_imgemb.npy"
        emb_stdout = emb_dir / "stdout.txt"
        emb_stderr = emb_dir / "stderr.txt"
        emb_cmd_path = emb_dir / "cmd.txt"
        emb_meta = emb_dir / "meta.json"

        emb_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "extract_img_embeddings.py"),
            "--model_name",
            model_name,
            "--img_size",
            str(img_size),
            "--batch_size",
            str(batch_size),
            "--patches_npy",
            str(args.patches_npy),
            "--out_path",
            str(emb_path),
            "--seed",
            "42",
            "--num_workers",
            "0",
            "--labels_path",
            str(labels_path),
        ]

        emb_cmd_path.parent.mkdir(parents=True, exist_ok=True)
        emb_cmd_path.write_text(" ".join(emb_cmd), encoding="utf-8")

        log_line(run_log, f"[embed] {tag} ({model_name})")
        if emb_path.exists() and emb_meta.exists():
            log_line(run_log, f"[skip] embeddings already exist for {tag}")
        elif args.dry_run:
            log_line(run_log, f"[dry_run] {' '.join(emb_cmd)}")
        else:
            exit_code = run_cmd(emb_cmd, ROOT, emb_stdout, emb_stderr)
            log_line(run_log, f"[embed_done] {tag} exit_code={exit_code}")
            if exit_code != 0:
                log_line(run_log, f"[embed_fail] {tag} skipping tabular runs")
                continue

        try:
            import numpy as np

            X_emb = np.load(emb_path, mmap_mode="r")
            log_line(run_log, f"[embed_shape] {tag}: {X_emb.shape}")
        except Exception as exc:
            log_line(run_log, f"[embed_shape_error] {tag}: {exc}")
            if not args.dry_run:
                continue

        for tab_name, model, cfg_rel in tabular_models:
            out_dir = run_dir / "tabular" / tag / tab_name
            stdout_path = out_dir / "stdout.txt"
            stderr_path = out_dir / "stderr.txt"
            cmd_path = out_dir / "cmd.txt"
            meta_path = out_dir / "meta.json"

            row = {
                "embedding_tag": tag,
                "embedding_model": model_name,
                "tabular_name": tab_name,
                "model": model,
                "exit_code": 0,
                "weighted_f1": None,
                "macro_f1": None,
                "best_iteration": None,
                "best_score": None,
                "feature_version": None,
                "val_probs_shape": None,
                "duration_sec": None,
                "features_path": str(emb_path),
            }

            if meta_path.exists():
                meta_existing = _load_meta_if_exists(meta_path)
                if meta_existing and meta_existing.get("weighted_f1") is not None:
                    log_line(run_log, f"[skip] {tag}/{tab_name} already completed")
                    row.update(
                        {
                            "exit_code": 0,
                            "weighted_f1": meta_existing.get("weighted_f1"),
                            "macro_f1": meta_existing.get("macro_f1"),
                            "best_iteration": meta_existing.get("best_iteration"),
                            "best_score": meta_existing.get("best_score"),
                            "feature_version": meta_existing.get("feature_version"),
                        }
                    )
                    summary_rows.append(row)
                    with summary_csv.open("a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([row[c] for c in cols])
                    continue

            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "train_tabular.py"),
                "--model",
                model,
                "--params_json",
                str(ROOT / cfg_rel),
                "--features_path",
                str(emb_path),
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

            log_line(run_log, f"[start] {tag}/{tab_name} ({model})")
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
                            "best_iteration": meta.get("best_iteration"),
                            "best_score": meta.get("best_score"),
                            "feature_version": meta.get("feature_version"),
                        }
                    )
                try:
                    import numpy as np

                    val_probs = np.load(out_dir / "val_probs.npy")
                    row["val_probs_shape"] = list(val_probs.shape)
                except Exception:
                    row["val_probs_shape"] = None

                log_line(run_log, f"[done] {tag}/{tab_name} exit_code={exit_code}")

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
