#!/usr/bin/env python3
"""
Greedy forward selection + coordinate ascent tuning for ensemble weights.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ARTIFACTS_DIR, SPLIT_DIR
from scripts.eval_utils import load_split_indices, weighted_f1


def _blend_probs(probs_list: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    blended = np.zeros_like(probs_list[0], dtype=np.float32)
    for p, w in zip(probs_list, weights):
        blended += p.astype(np.float32) * float(w)
    return blended


def _score_probs(y_val: np.ndarray, probs: np.ndarray) -> float:
    preds = np.argmax(probs, axis=1)
    return float(weighted_f1(y_val, preds))


def _normalize(weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return weights / total


def greedy_forward_selection(
    y_val: np.ndarray,
    probs_list: List[np.ndarray],
    names: List[str],
    max_models: int,
    min_improve: float,
) -> Tuple[List[int], float]:
    selected = []
    best_score = 0.0

    remaining = list(range(len(probs_list)))
    print("== Greedy forward selection ==")

    while remaining and len(selected) < max_models:
        best_candidate = None
        best_candidate_score = best_score
        for idx in remaining:
            trial = selected + [idx]
            weights = np.ones(len(trial), dtype=np.float32) / len(trial)
            probs = _blend_probs([probs_list[i] for i in trial], weights)
            score = _score_probs(y_val, probs)
            print(f"  -> try + {names[idx]:<30} | wf1={score:.4f}")
            if score > best_candidate_score + min_improve:
                best_candidate_score = score
                best_candidate = idx

        if best_candidate is None:
            print("  no candidate improves the score, stopping.")
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_score = best_candidate_score
        print(f"  + added {names[best_candidate]} | best wf1={best_score:.4f}")

    return selected, best_score


def coordinate_ascent(
    y_val: np.ndarray,
    probs_list: List[np.ndarray],
    init_weights: np.ndarray,
    step: float,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, float]:
    weights = _normalize(init_weights.copy())
    best_score = _score_probs(y_val, _blend_probs(probs_list, weights))
    print("\n== Coordinate ascent ==")
    print(f"  start wf1={best_score:.4f} | weights={weights.round(4).tolist()}")

    for iteration in range(1, max_iter + 1):
        improved = False
        for i in range(len(weights)):
            current = weights[i]
            candidates = np.clip(
                np.array([current - step, current, current + step]), 0.0, 1.0
            )
            best_local = current
            best_local_score = best_score
            for cand in candidates:
                if len(weights) == 1:
                    trial = np.array([1.0], dtype=np.float32)
                else:
                    if current >= 1.0:
                        trial = weights.copy()
                    else:
                        scale = (1.0 - cand) / max(1e-12, 1.0 - current)
                        trial = weights.copy()
                        trial[i] = cand
                        for j in range(len(weights)):
                            if j != i:
                                trial[j] = trial[j] * scale
                    trial = _normalize(trial)
                score = _score_probs(y_val, _blend_probs(probs_list, trial))
                if score > best_local_score + tol:
                    best_local_score = score
                    best_local = cand
                    best_trial = trial

            if best_local_score > best_score + tol:
                weights = best_trial
                best_score = best_local_score
                improved = True
        print(
            f"  iter {iteration:02d} | wf1={best_score:.4f} | weights={weights.round(4).tolist()}"
        )
        if not improved:
            print("  no improvement, stopping.")
            break

    return weights, best_score


def _validate_lengths(arg_name: str, values: List[str], names: List[str]) -> None:
    if values and len(values) != len(names):
        raise ValueError(f"{arg_name} must match names length ({len(names)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune ensemble weights")
    parser.add_argument("--prob_paths", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--labels_path", type=Path, default=ARTIFACTS_DIR / "features" / "y.npy")
    parser.add_argument("--split_dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--max_models", type=int, default=8)
    parser.add_argument("--min_improve", type=float, default=1e-4)
    parser.add_argument(
        "--force_all",
        action="store_true",
        help="Skip greedy selection and tune weights on all provided models.",
    )
    parser.add_argument("--coord_step", type=float, default=0.02)
    parser.add_argument("--coord_iter", type=int, default=25)
    parser.add_argument("--coord_tol", type=float, default=1e-5)
    parser.add_argument("--feature_version", default="v3")
    parser.add_argument("--out_json", type=Path, default=ARTIFACTS_DIR / "ensemble_submit.json")
    parser.add_argument("--model_paths", nargs="*")
    parser.add_argument("--model_types", nargs="*")
    parser.add_argument("--cnn_archs", nargs="*")
    parser.add_argument("--cnn_norm_paths", nargs="*")
    args = parser.parse_args()

    if len(args.prob_paths) != len(args.names):
        raise ValueError("--prob_paths and --names must have the same length.")

    _validate_lengths("--model_paths", args.model_paths or [], args.names)
    _validate_lengths("--model_types", args.model_types or [], args.names)
    _validate_lengths("--cnn_archs", args.cnn_archs or [], args.names)
    _validate_lengths("--cnn_norm_paths", args.cnn_norm_paths or [], args.names)

    y_all = np.load(args.labels_path)
    _, val_idx = load_split_indices(args.split_dir)
    y_val = y_all[val_idx]

    probs_list = [np.load(p).astype(np.float32) for p in args.prob_paths]
    for p, name in zip(probs_list, args.names):
        score = _score_probs(y_val, p)
        print(f"base {name:<30} | wf1={score:.4f}")

    if args.force_all:
        selected = list(range(len(args.names)))
        print("\n== Greedy forward selection ==")
        print("  force_all enabled, using all models.")
    else:
        max_models = min(args.max_models, len(args.names))
        selected, _ = greedy_forward_selection(
            y_val, probs_list, args.names, max_models=max_models, min_improve=args.min_improve
        )
        if not selected:
            raise RuntimeError("No models selected. Try lowering --min_improve or use --force_all.")

    selected_names = [args.names[i] for i in selected]
    selected_probs = [probs_list[i] for i in selected]
    init_weights = np.ones(len(selected), dtype=np.float32) / len(selected)

    tuned_weights, tuned_score = coordinate_ascent(
        y_val,
        selected_probs,
        init_weights,
        step=args.coord_step,
        max_iter=args.coord_iter,
        tol=args.coord_tol,
    )

    print("\n== Final selection ==")
    for name, w in zip(selected_names, tuned_weights):
        print(f"  {name:<30} weight={float(w):.4f}")
    print(f"  final wf1={tuned_score:.4f}")

    models = []
    for idx, w in zip(selected, tuned_weights):
        entry = {"name": args.names[idx], "weight": float(w)}
        if args.model_types:
            entry["type"] = args.model_types[idx]
        if args.model_paths:
            entry["model_path"] = args.model_paths[idx]
        if args.cnn_archs:
            if args.cnn_archs[idx]:
                entry["arch"] = args.cnn_archs[idx]
        if args.cnn_norm_paths:
            if args.cnn_norm_paths[idx]:
                entry["norm_path"] = args.cnn_norm_paths[idx]
        models.append(entry)

    out_cfg = {
        "feature_version": args.feature_version,
        "models": models,
        "weighted_f1": tuned_score,
        "note": "Greedy forward selection + coord ascent (scripted)",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out_cfg, indent=2), encoding="utf-8")
    print(f"\nSaved ensemble config: {args.out_json}")


if __name__ == "__main__":
    main()
