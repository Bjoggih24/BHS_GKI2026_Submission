#!/usr/bin/env python3
"""
Tune top-K ensemble weights with coordinate ascent.
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ARTIFACTS_DIR, SPLIT_DIR
from scripts.eval_utils import load_split_indices, weighted_f1


def eval_weights(probs_list, weights, y_true):
    blended = np.zeros_like(probs_list[0], dtype=np.float32)
    for p, w in zip(probs_list, weights):
        blended += p * w
    preds = np.argmax(blended, axis=1)
    return weighted_f1(y_true, preds)


def coordinate_ascent(probs_list, y_true, step=0.05, max_iter=50):
    n = len(probs_list)
    weights = np.ones(n, dtype=np.float32) / n
    best_score = eval_weights(probs_list, weights, y_true)
    grid = np.arange(0.0, 1.0 + 1e-9, step)

    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(n):
            current = weights.copy()
            rest_idx = [j for j in range(n) if j != i]
            rest_sum = current[rest_idx].sum()
            best_local_w = current[i]
            best_local_score = best_score
            for w in grid:
                if n == 1:
                    new_weights = np.array([1.0], dtype=np.float32)
                else:
                    if w < 0.0 or w > 1.0:
                        continue
                    remaining = 1.0 - w
                    if remaining < 0.0:
                        continue
                    if rest_sum == 0.0:
                        rest_weights = np.ones(n - 1, dtype=np.float32) / (n - 1) * remaining
                    else:
                        rest_weights = current[rest_idx] / rest_sum * remaining
                    new_weights = current.copy()
                    new_weights[i] = w
                    new_weights[rest_idx] = rest_weights
                score = eval_weights(probs_list, new_weights, y_true)
                if score > best_local_score + 1e-6:
                    best_local_score = score
                    best_local_w = w
            if best_local_score > best_score + 1e-6:
                if n == 1:
                    weights = np.array([1.0], dtype=np.float32)
                else:
                    remaining = 1.0 - best_local_w
                    if rest_sum == 0.0:
                        rest_weights = np.ones(n - 1, dtype=np.float32) / (n - 1) * remaining
                    else:
                        rest_weights = current[rest_idx] / rest_sum * remaining
                    weights = current.copy()
                    weights[i] = best_local_w
                    weights[rest_idx] = rest_weights
                best_score = best_local_score
                improved = True
    return weights, best_score


def main():
    parser = argparse.ArgumentParser(description="Tune top-K ensemble weights")
    parser.add_argument("--prob_paths", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--model_paths", nargs="+", required=True)
    parser.add_argument("--types", nargs="+", required=True)
    parser.add_argument("--archs", nargs="*", default=None)
    parser.add_argument("--feature_version", type=str, default="v2")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--max_models", type=int, default=6)
    parser.add_argument("--split_dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--labels_path", type=Path, default=ARTIFACTS_DIR / "features" / "y.npy")
    parser.add_argument("--out_path", type=Path, default=ARTIFACTS_DIR / "ensemble_topK.json")
    args = parser.parse_args()

    if len(args.prob_paths) != len(args.names):
        raise ValueError("prob_paths and names must have the same length")
    if len(args.model_paths) != len(args.names):
        raise ValueError("model_paths and names must have the same length")
    if len(args.types) != len(args.names):
        raise ValueError("types and names must have the same length")
    if args.archs and len(args.archs) != len(args.names):
        raise ValueError("archs and names must have the same length")

    y_all = np.load(args.labels_path)
    _, val_idx = load_split_indices(args.split_dir)
    y_val = y_all[val_idx]

    probs_list = []
    for path in args.prob_paths:
        probs_list.append(np.load(path).astype(np.float32))

    scores = []
    for name, probs in zip(args.names, probs_list):
        preds = np.argmax(probs, axis=1)
        wf1 = weighted_f1(y_val, preds)
        scores.append((name, wf1))

    ranked = sorted(range(len(scores)), key=lambda i: scores[i][1], reverse=True)
    top_k = ranked[: min(args.max_models, len(ranked))]

    top_probs = [probs_list[i] for i in top_k]
    top_names = [args.names[i] for i in top_k]
    top_scores = [scores[i][1] for i in top_k]
    top_model_paths = [args.model_paths[i] for i in top_k]
    top_types = [args.types[i] for i in top_k]
    top_archs = [args.archs[i] for i in top_k] if args.archs else [None for _ in top_k]

    weights, best_score = coordinate_ascent(top_probs, y_val, step=args.step)

    models = []
    for name, path, model_path, model_type, arch, w, s in zip(
        top_names,
        [args.prob_paths[i] for i in top_k],
        top_model_paths,
        top_types,
        top_archs,
        weights,
        top_scores,
    ):
        entry = {
            "name": name,
            "type": model_type,
            "model_path": model_path,
            "weight": float(w),
            "individual_wf1": float(s),
            "prob_path": path,
        }
        if model_type == "cnn":
            if not arch:
                raise ValueError("CNN model entries must provide an arch.")
            entry["arch"] = arch
        models.append(entry)

    out = {
        "feature_version": args.feature_version,
        "models": models,
        "weighted_f1": float(best_score),
        "step": args.step,
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Best weighted F1: {best_score:.4f}")
    print(f"Saved {args.out_path}")


if __name__ == "__main__":
    main()
