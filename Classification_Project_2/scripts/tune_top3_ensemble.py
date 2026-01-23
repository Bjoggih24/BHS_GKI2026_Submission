#!/usr/bin/env python3
"""
Tune weights for top-2 tabular + top-1 CNN ensemble.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ARTIFACTS_DIR, DATA_DIR, SPLIT_DIR
from scripts.eval_utils import load_split_indices, weighted_f1


def main():
    # fixed selection based on sweep
    tab_paths = [
        ARTIFACTS_DIR / "experiments" / "run_gym_001" / "tabular" / "ET_B" / "val_probs.npy",
        ARTIFACTS_DIR / "experiments" / "run_gym_001" / "tabular" / "HGB_A" / "val_probs.npy",
    ]
    cnn_path = ARTIFACTS_DIR / "experiments" / "run_gym_001" / "cnn" / "cnn_small_mildw_ls" / "val_probs.npy"

    y = pd.read_csv(DATA_DIR / "train.csv")["vistgerd_idx"].to_numpy()
    _, val_idx = load_split_indices(SPLIT_DIR)
    y_val = y[val_idx]

    tab_probs = [np.load(p) for p in tab_paths]
    cnn_probs = np.load(cnn_path)

    best = {"w_tab1": 0, "w_tab2": 0, "w_cnn": 0, "f1": -1}
    weights = np.arange(0.0, 1.01, 0.05)
    for w1 in weights:
        for w2 in weights:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-6 or w3 > 1.0 + 1e-6:
                continue
            if not np.any(np.isclose(weights, w3, atol=1e-6)):
                continue
            w3 = float(np.clip(w3, 0.0, 1.0))
            probs = w1 * tab_probs[0] + w2 * tab_probs[1] + w3 * cnn_probs
            preds = np.argmax(probs, axis=1)
            f1 = weighted_f1(y_val, preds)
            if f1 > best["f1"]:
                best = {"w_tab1": w1, "w_tab2": w2, "w_cnn": w3, "f1": f1}

    out = {
        "tabular_models": [
            {"path": "experiments/run_gym_001/tabular/ET_B/model.joblib", "weight": best["w_tab1"]},
            {"path": "experiments/run_gym_001/tabular/HGB_A/model.joblib", "weight": best["w_tab2"]},
        ],
        "cnn_models": [
            {"path": "experiments/run_gym_001/cnn/cnn_small_mildw_ls/cnn_best.pth", "arch": "resnet_small", "weight": best["w_cnn"]}
        ],
        "weighted_f1": best["f1"],
    }

    cfg_path = ARTIFACTS_DIR / "ensemble_top3.json"
    cfg_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {cfg_path}")
    print(out)
    print(f"Best weighted F1 (offline val): {best['f1']:.4f}")


if __name__ == "__main__":
    main()
