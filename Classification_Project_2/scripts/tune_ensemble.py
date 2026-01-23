#!/usr/bin/env python3
"""
Tune ensemble weight between CNN and tabular model.
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR, TRAIN_DIR, SPLIT_DIR, TABULAR_DIR, CNN_DIR, ENSEMBLE_PATH, NORM_STATS_PATH
from cnn.dataset import HabitatDataset
from cnn.model import HabitatCNN
from feature_utils import extract_tabular_features
from scripts.eval_utils import load_split_indices, weighted_f1


def build_features(patches: np.ndarray) -> np.ndarray:
    return np.stack([extract_tabular_features(p) for p in patches], axis=0)


def main():
    patches_path = TRAIN_DIR / "patches.npy"
    labels_path = DATA_DIR / "train.csv"

    df = pd.read_csv(labels_path)
    labels = df["vistgerd_idx"].to_numpy()
    patches = np.load(patches_path, mmap_mode="r")
    _, val_idx = load_split_indices(SPLIT_DIR)

    stats = np.load(NORM_STATS_PATH)
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)

    # CNN probs
    device = torch.device("cpu")
    model = HabitatCNN(in_ch=16, num_classes=71)
    model.load_state_dict(torch.load(CNN_DIR / "cnn_best.pth", map_location=device))
    model.eval()

    val_ds = HabitatDataset(patches[val_idx], labels[val_idx], mean, std, augment=False)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
    cnn_probs = []
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            cnn_probs.append(probs)
    cnn_probs = np.vstack(cnn_probs)

    # Tabular probs
    tabular = joblib.load(TABULAR_DIR / "extratrees.joblib")
    X_val = build_features(patches[val_idx])
    tree_probs = tabular.predict_proba(X_val)

    y_val = labels[val_idx]
    best_w = 0.0
    best_f1 = -1.0
    for w in np.linspace(0.0, 1.0, 21):
        probs = w * cnn_probs + (1.0 - w) * tree_probs
        preds = np.argmax(probs, axis=1)
        f1 = weighted_f1(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_w = w

    ENSEMBLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ENSEMBLE_PATH, "w", encoding="utf-8") as f:
        json.dump({"weight_cnn": best_w, "weighted_f1": best_f1}, f, indent=2)

    print(f"Best weight: {best_w:.2f} | weighted F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
