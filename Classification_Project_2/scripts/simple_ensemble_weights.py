#!/usr/bin/env python3
"""
Test simpler ensemble weighting strategies that may generalize better.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ARTIFACTS_DIR, SPLIT_DIR
from scripts.eval_utils import load_split_indices, weighted_f1, macro_f1


def main():
    parser = argparse.ArgumentParser(description="Test simple ensemble weights")
    parser.add_argument("--prob_paths", nargs="+", required=True, help="Paths to val_probs.npy files")
    parser.add_argument("--names", nargs="+", required=True, help="Model names")
    parser.add_argument("--labels_path", type=Path, default=ARTIFACTS_DIR / "features" / "y.npy")
    parser.add_argument("--split_dir", type=Path, default=SPLIT_DIR)
    args = parser.parse_args()

    y_all = np.load(args.labels_path)
    _, val_idx = load_split_indices(args.split_dir)
    y_val = y_all[val_idx]

    # Load all probability files
    probs_list = []
    individual_scores = []
    for path, name in zip(args.prob_paths, args.names):
        p = np.load(path).astype(np.float32)
        probs_list.append(p)
        preds = np.argmax(p, axis=1)
        wf1 = weighted_f1(y_val, preds)
        individual_scores.append(wf1)
        print(f"{name}: {wf1:.4f}")

    n = len(probs_list)
    print(f"\n{'='*50}")
    print(f"Testing {n} models")
    print(f"{'='*50}\n")

    # Strategy 1: Uniform weights
    uniform_weights = np.ones(n) / n
    blended = sum(p * w for p, w in zip(probs_list, uniform_weights))
    preds = np.argmax(blended, axis=1)
    wf1_uniform = weighted_f1(y_val, preds)
    print(f"Uniform weights: {wf1_uniform:.4f}")

    # Strategy 2: Inverse-variance (weight by individual F1)
    scores = np.array(individual_scores)
    inv_var_weights = scores / scores.sum()
    blended = sum(p * w for p, w in zip(probs_list, inv_var_weights))
    preds = np.argmax(blended, axis=1)
    wf1_inv_var = weighted_f1(y_val, preds)
    print(f"Inverse-variance weights: {wf1_inv_var:.4f}")
    print(f"  Weights: {dict(zip(args.names, [f'{w:.3f}' for w in inv_var_weights]))}")

    # Strategy 3: Softmax temperature scaling
    for temp in [0.5, 1.0, 2.0, 5.0]:
        softmax_weights = np.exp(scores / temp) / np.exp(scores / temp).sum()
        blended = sum(p * w for p, w in zip(probs_list, softmax_weights))
        preds = np.argmax(blended, axis=1)
        wf1_softmax = weighted_f1(y_val, preds)
        print(f"Softmax (T={temp}): {wf1_softmax:.4f}")

    # Strategy 4: Top-K only (drop worst models)
    for k in range(max(1, n-2), n+1):
        top_k_idx = np.argsort(scores)[-k:]
        top_k_weights = np.zeros(n)
        top_k_weights[top_k_idx] = 1.0 / k
        blended = sum(p * w for p, w in zip(probs_list, top_k_weights))
        preds = np.argmax(blended, axis=1)
        wf1_topk = weighted_f1(y_val, preds)
        print(f"Top-{k} uniform: {wf1_topk:.4f}")

    print(f"\n{'='*50}")
    print("Recommendation: Use uniform or softmax(T=2) weights")
    print("These tend to generalize better to test set")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
