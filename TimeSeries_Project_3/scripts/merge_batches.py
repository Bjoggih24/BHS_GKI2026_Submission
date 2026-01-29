"""
Merge training batch files into a single train_full.npz

Usage:
    python scripts/merge_batches.py "data/train_batches/train_full_batch*.npz" --output data/train_full.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import glob

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Merge training batches")
    parser.add_argument("pattern", help="Glob pattern for batch files (e.g., data/train_batch_*.npz)")
    parser.add_argument("--output", type=str, default="data/train_full.npz", help="Output file")
    args = parser.parse_args()
    
    # Find batch files
    batch_files = sorted(glob.glob(args.pattern))
    if not batch_files:
        print(f"No files found matching: {args.pattern}")
        return
    
    print(f"Found {len(batch_files)} batch files:")
    for f in batch_files:
        print(f"  - {f}")
    
    # Load and merge
    X_all = []
    y_all = []
    ts_all = []
    sensor_names = None
    
    for batch_file in batch_files:
        print(f"\nLoading {batch_file}...")
        data = np.load(batch_file, allow_pickle=True)
        
        X_all.append(data["X"])
        y_all.append(data["y"])
        ts_all.append(data["timestamps"])
        
        if sensor_names is None:
            sensor_names = data["sensor_names"]
        
        print(f"  Samples: {len(data['timestamps'])}")
    
    # Concatenate
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    timestamps = np.concatenate(ts_all)
    
    print(f"\nMerged dataset:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  timestamps: {len(timestamps)}")
    
    # Save
    output_path = Path(args.output)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        timestamps=timestamps,
        sensor_names=sensor_names,
    )
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()