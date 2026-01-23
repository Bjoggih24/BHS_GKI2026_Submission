#!/usr/bin/env python3
"""
Tune ensemble weights using saved val_probs.
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR, SPLIT_DIR
from scripts.eval_utils import load_split_indices, weighted_f1


def main():
    parser = argparse.ArgumentParser(description="Ensemble tuning from saved probs")
    parser.add_argument("--tab_probs_list", nargs="+", required=True)
    parser.add_argument("--cnn_probs_list", nargs="+", required=True)
    parser.add_argument("--split_dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    labels = pd.read_csv(DATA_DIR / "train.csv")["vistgerd_idx"].to_numpy()
    _, val_idx = load_split_indices(args.split_dir)
    y_val = labels[val_idx]

    tab_probs = [np.load(p) for p in args.tab_probs_list]
    cnn_probs = [np.load(p) for p in args.cnn_probs_list]
    avg_tab = np.mean(tab_probs, axis=0)
    avg_cnn = np.mean(cnn_probs, axis=0)

    best_w = 0.0
    best_f1 = -1.0
    for w in np.linspace(0.0, 1.0, 21):
        p = w * avg_cnn + (1.0 - w) * avg_tab
        preds = np.argmax(p, axis=1)
        f1 = weighted_f1(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_w = w

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "ensemble_meta.json", "w", encoding="utf-8") as f:
        json.dump({"best_weight": best_w, "weighted_f1": best_f1}, f, indent=2)

    print(f"Best weight: {best_w:.2f} | weighted F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
