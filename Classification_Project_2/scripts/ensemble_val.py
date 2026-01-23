#!/usr/bin/env python3
"""
Tune ensemble weights on validation probabilities.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_utils import load_split_indices, weighted_f1, macro_f1


def generate_simplex_weights(n: int, step: float) -> List[List[float]]:
    if n == 1:
        return [[1.0]]
    weights = []
    grid = np.arange(0.0, 1.0 + 1e-9, step)

    def rec(prefix, remaining, k):
        if k == n - 1:
            if remaining < -1e-9:
                return
            w_last = float(np.clip(remaining, 0.0, 1.0))
            weights.append(prefix + [w_last])
            return
        for w in grid:
            if w > remaining + 1e-9:
                continue
            rec(prefix + [float(w)], remaining - w, k + 1)

    rec([], 1.0, 0)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble val_probs with simplex grid search")
    parser.add_argument("--probs", nargs="+", required=True)
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    if len(args.probs) > 5:
        raise ValueError("Please provide <= 5 models for simplex grid search.")

    y_all = np.load(args.labels_path)
    _, val_idx = load_split_indices(args.split_dir)
    y_val = y_all[val_idx]

    probs_list = [np.load(Path(p), mmap_mode="r") for p in args.probs]
    num_classes = None
    for i, p in enumerate(probs_list):
        if p.shape[0] != len(val_idx):
            raise ValueError(f"val_probs shape mismatch for {args.probs[i]}: {p.shape} vs {len(val_idx)}")
        if num_classes is None:
            num_classes = p.shape[1]
        elif p.shape[1] != num_classes:
            raise ValueError(
                f"class count mismatch for {args.probs[i]}: {p.shape[1]} vs {num_classes}"
            )

    weights_grid = generate_simplex_weights(len(probs_list), args.step)
    best = {"weighted_f1": -1.0, "macro_f1": -1.0, "weights": None}
    best_probs = None

    for weights in weights_grid:
        blended = np.zeros_like(probs_list[0], dtype=np.float32)
        for p, w in zip(probs_list, weights):
            blended += p * w
        preds = np.argmax(blended, axis=1)
        wf1 = weighted_f1(y_val, preds)
        if wf1 > best["weighted_f1"] + 1e-9:
            mf1 = macro_f1(y_val, preds)
            best = {"weighted_f1": float(wf1), "macro_f1": float(mf1), "weights": weights}
            best_probs = blended

    out = {
        "probs": [str(p) for p in args.probs],
        "weights": best["weights"],
        "weighted_f1": best["weighted_f1"],
        "macro_f1": best["macro_f1"],
        "step": args.step,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    if best_probs is not None:
        np.save(args.out_json.parent / "ensemble_val_probs.npy", best_probs.astype(np.float32))

    print(f"Best weighted F1: {best['weighted_f1']:.4f} | macro F1: {best['macro_f1']:.4f}")
    print(f"Saved {args.out_json}")


if __name__ == "__main__":
    main()
