#!/usr/bin/env python3
# score_local.py
import argparse
import json
import math
import time
import importlib.util
import random
from pathlib import Path

import numpy as np
from datasets import load_from_disk

LN2 = math.log(2.0)


def find_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(8):
        if (p / "data" / "igc_full").exists() and (p / "submission").exists():
            return p
        p = p.parent
    raise FileNotFoundError("Could not find project root containing data/igc_full and submission/")


def load_submission_model(submission_dir: Path):
    model_path = submission_dir / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}")

    spec = importlib.util.spec_from_file_location("submission_model", str(model_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.Model(submission_dir)


def default_ctx_from_gru_config(submission_dir: Path, fallback: int) -> int:
    cfg_path = submission_dir / "gru_config.json"
    if not cfg_path.exists():
        return fallback
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return int(cfg.get("max_ctx_len", fallback))
    except Exception:
        return fallback


def bytes_from_doc(rec) -> bytes:
    # Keep this consistent with training/inference: UTF-8 bytes
    return rec["text"].encode("utf-8", errors="ignore")


def bpb_sum_from_logits_numpy(logits_batch, y_batch) -> float:
    """
    logits_batch: list[list[float]] shape (B, 256)
    y_batch: list[int] length B
    returns: sum of bpb over batch
    """
    L = np.asarray(logits_batch, dtype=np.float32)  # (B,256)
    y = np.asarray(y_batch, dtype=np.int64)         # (B,)

    # stable logsumexp over dim=1
    m = L.max(axis=1)  # (B,)
    lse = m + np.log(np.exp(L - m[:, None]).sum(axis=1))  # (B,)
    logp = L[np.arange(L.shape[0]), y] - lse              # (B,)
    bpb = (-logp / LN2)                                   # (B,)
    return float(bpb.sum())


def iter_examples_random(ds, doc_indices, max_preds: int, ctx_len: int, seed: int):
    """
    Yields (ctx_list[int], y_int) for max_preds predictions, sampling random docs and positions.
    """
    rng = random.Random(seed)
    used = 0
    n_docs = len(doc_indices)

    while used < max_preds:
        i = doc_indices[rng.randrange(n_docs)]
        b = bytes_from_doc(ds[i])
        if len(b) < 2:
            continue

        pos = rng.randrange(1, len(b))
        start = max(0, pos - ctx_len)
        ctx = list(b[start:pos])
        y = b[pos]
        yield ctx, y
        used += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=200_000, help="How many next-byte predictions to score.")
    ap.add_argument("--batch", type=int, default=1024, help="Batch size for model.predict().")
    ap.add_argument("--ctx", type=int, default=None, help="Context length (bytes) used for sampling.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for repeatable eval.")
    ap.add_argument("--max-val-docs", type=int, default=0,
                    help="Optionally limit the number of val docs used for sampling (0 = use all).")
    args = ap.parse_args()

    root = find_root(Path.cwd())
    data_dir = root / "data" / "igc_full"
    splits_dir = root / "data" / "splits"
    submission_dir = root / "submission"

    print(f"[info] root={root}")
    print(f"[info] data_dir exists={data_dir.exists()} | splits_dir exists={splits_dir.exists()}")

    ds = load_from_disk(str(data_dir))
    val_idx = json.loads((splits_dir / "val_idx.json").read_text(encoding="utf-8"))
    print(f"[info] loaded dataset docs={len(ds)} | val_docs={len(val_idx)}")

    if args.max_val_docs and args.max_val_docs > 0 and args.max_val_docs < len(val_idx):
        rng = random.Random(args.seed)
        val_idx = rng.sample(val_idx, args.max_val_docs)
        print(f"[info] using subset val_docs={len(val_idx)} (max_val_docs={args.max_val_docs})")

    if args.ctx is None:
        args.ctx = default_ctx_from_gru_config(submission_dir, fallback=512)
    print(f"[info] eval ctx_len={args.ctx} | batch={args.batch} | preds={args.bytes} | seed={args.seed}")

    model = load_submission_model(submission_dir)
    print("[info] model loaded")

    total_bpb = 0.0
    n = 0
    batch_ctx, batch_y = [], []

    t0 = time.time()
    for ctx, y in iter_examples_random(ds, val_idx, max_preds=args.bytes, ctx_len=args.ctx, seed=args.seed):
        batch_ctx.append(ctx)
        batch_y.append(y)

        if len(batch_ctx) >= args.batch:
            logits_batch = model.predict(batch_ctx)
            total_bpb += bpb_sum_from_logits_numpy(logits_batch, batch_y)
            n += len(batch_y)
            batch_ctx.clear()
            batch_y.clear()

    if batch_ctx:
        logits_batch = model.predict(batch_ctx)
        total_bpb += bpb_sum_from_logits_numpy(logits_batch, batch_y)
        n += len(batch_y)

    dt = time.time() - t0
    if n == 0:
        raise RuntimeError("No validation bytes were scored (n=0). Something went wrong with data iteration.")

    print(f"VAL bpb: {total_bpb / n:.4f} over {n} preds | {n / dt:.0f} preds/s | {dt:.1f}s")


if __name__ == "__main__":
    main()
