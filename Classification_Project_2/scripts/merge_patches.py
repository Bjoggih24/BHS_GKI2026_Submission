#!/usr/bin/env python3
"""
Merge training patches split into two parts into a single patches.npy.
"""

from pathlib import Path
import numpy as np


def main():
    data_dir = Path(__file__).parent.parent / "data" / "train"
    part1 = data_dir / "patches_part1.npy"
    part2 = data_dir / "patches_part2.npy"
    out_path = data_dir / "patches.npy"

    if not part1.exists() or not part2.exists():
        raise FileNotFoundError("Missing patches_part1.npy or patches_part2.npy")

    print(f"Loading {part1}...")
    p1 = np.load(part1)
    print(f"Loading {part2}...")
    p2 = np.load(part2)

    print(f"Part1 shape: {p1.shape}")
    print(f"Part2 shape: {p2.shape}")

    patches = np.concatenate([p1, p2], axis=0)
    print(f"Merged shape: {patches.shape}")

    np.save(out_path, patches)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
