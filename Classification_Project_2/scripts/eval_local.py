#!/usr/bin/env python3
"""
Evaluate tabular, CNN, and ensemble models on the fixed split.
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
import joblib

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR, TRAIN_DIR, SPLIT_DIR, TABULAR_DIR, CNN_DIR, ENSEMBLE_PATH, NORM_STATS_PATH
from cnn.dataset import HabitatDataset
from cnn.model import HabitatCNN
from feature_utils import extract_tabular_features_v2
from scripts.eval_utils import load_split_indices, weighted_f1, macro_f1, report_per_class_f1


def build_features(patches: np.ndarray) -> np.ndarray:
    return np.stack([extract_tabular_features_v2(p) for p in patches], axis=0)


def eval_tabular(patches, labels, val_idx):
    model = joblib.load(TABULAR_DIR / "extratrees.joblib")
    X_val = build_features(patches[val_idx])
    preds = model.predict(X_val)
    return preds, model.predict_proba(X_val)


def eval_cnn(patches, labels, val_idx, mean, std):
    device = torch.device("cpu")
    model = HabitatCNN(in_ch=16, num_classes=71)
    model.load_state_dict(torch.load(CNN_DIR / "cnn_best.pth", map_location=device))
    model.eval()

    val_ds = HabitatDataset(patches[val_idx], labels[val_idx], mean, std, augment=False)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)

    preds = []
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            preds.append(np.argmax(p, axis=1))

    probs = np.vstack(probs)
    preds = np.concatenate(preds)
    return preds, probs


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

    y_val = labels[val_idx]

    tab_preds, tab_probs = eval_tabular(patches, labels, val_idx)
    cnn_preds, cnn_probs = eval_cnn(patches, labels, val_idx, mean, std)

    print("Tabular weighted F1:", weighted_f1(y_val, tab_preds))
    print("Tabular macro F1:", macro_f1(y_val, tab_preds))
    print("Tabular worst classes:", report_per_class_f1(y_val, tab_preds, top_k=10))

    print("CNN weighted F1:", weighted_f1(y_val, cnn_preds))
    print("CNN macro F1:", macro_f1(y_val, cnn_preds))
    print("CNN worst classes:", report_per_class_f1(y_val, cnn_preds, top_k=10))

    if ENSEMBLE_PATH.exists():
        with open(ENSEMBLE_PATH, "r", encoding="utf-8") as f:
            weight = json.load(f).get("weight_cnn", 0.5)
    else:
        weight = 0.5

    ens_probs = weight * cnn_probs + (1.0 - weight) * tab_probs
    ens_preds = np.argmax(ens_probs, axis=1)
    print(f"Ensemble weighted F1 (w={weight:.2f}):", weighted_f1(y_val, ens_preds))
    print("Ensemble macro F1:", macro_f1(y_val, ens_preds))
    print("Ensemble worst classes:", report_per_class_f1(y_val, ens_preds, top_k=10))


if __name__ == "__main__":
    main()
