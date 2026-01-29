#!/usr/bin/env python3
"""
Prepare training data files required by the pipeline.

This script:
1) Extracts train_data_part1.zip and train_data_part2.zip into data/train/
2) Concatenates patches_part1.npy and patches_part2.npy into patches.npy
"""

from pathlib import Path
import zipfile
import numpy as np


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    part1_zip = data_dir / "train_data_part1.zip"
    part2_zip = data_dir / "train_data_part2.zip"

    extract_zip(part1_zip, data_dir)
    extract_zip(part2_zip, data_dir)

    part1_path = train_dir / "patches_part1.npy"
    part2_path = train_dir / "patches_part2.npy"
    if not part1_path.exists() or not part2_path.exists():
        raise FileNotFoundError(
            "Expected data/train/patches_part1.npy and data/train/patches_part2.npy"
        )

    part1 = np.load(part1_path, mmap_mode="r")
    part2 = np.load(part2_path, mmap_mode="r")
    patches = np.concatenate([part1, part2], axis=0)

    out_path = train_dir / "patches.npy"
    np.save(out_path, patches)

    print(f"Saved {out_path} with shape {patches.shape} and dtype {patches.dtype}")


if __name__ == "__main__":
    main()
