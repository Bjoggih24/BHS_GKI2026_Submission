#!/usr/bin/env python3
"""
Precompute tabular features for all samples.
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

from config import DATA_DIR, TRAIN_DIR, ARTIFACTS_DIR
from feature_utils import extract_tabular_features, extract_tabular_features_v2, extract_tabular_features_v3


def main():
    parser = argparse.ArgumentParser(description="Precompute tabular features")
    parser.add_argument("--version", choices=["v1", "v2", "v3"], default="v1")
    args = parser.parse_args()

    patches_path = TRAIN_DIR / "patches.npy"
    labels_path = DATA_DIR / "train.csv"
    if not patches_path.exists():
        raise FileNotFoundError(f"Missing {patches_path}")

    patches = np.load(patches_path, mmap_mode="r")
    labels = pd.read_csv(labels_path)["vistgerd_idx"].to_numpy()

    if args.version == "v3":
        extractor = extract_tabular_features_v3
        version_name = "v3"
        desc = "v3_stats_quantiles_indices_ratios_grad_terrain_grid"
    elif args.version == "v2":
        extractor = extract_tabular_features_v2
        version_name = "v2"
        desc = "v2_stats_quantiles_indices_grad_grid"
    else:
        extractor = extract_tabular_features
        version_name = "v1"
        desc = "v1_mean_std_grid7x7_terrain"

    n = len(patches)
    first = extractor(patches[0])
    feats = np.zeros((n, first.shape[0]), dtype=np.float32)
    feats[0] = first
    for i in range(1, n):
        feats[i] = extractor(patches[i])

    out_dir = ARTIFACTS_DIR / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"X_tabular_{version_name}.npy", feats)
    np.save(out_dir / "y.npy", labels.astype(np.int64))

    meta = {"feature_version": version_name, "description": desc, "dim": int(feats.shape[1])}
    with open(out_dir / f"feature_version_{version_name}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {out_dir / f'X_tabular_{version_name}.npy'}")
    print(f"Saved {out_dir / 'y.npy'}")


if __name__ == "__main__":
    main()
