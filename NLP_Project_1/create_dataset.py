#!/usr/bin/env python3
"""
Create the training dataset for the NLP Challenge.

Downloads the IGC-2024 (Icelandic Gigaword Corpus) from HuggingFace
and prepares it as a HuggingFace Dataset for training.

Usage:
    python create_dataset.py

This will create:
    data/igc_full/  - The full training dataset (~2.1M documents)

The validation and test sets used for evaluation are sampled from
the same IGC corpus but are held out and not included here.
"""

import os
import zipfile

from datasets import Dataset, Features, List, Value, load_dataset
from huggingface_hub import snapshot_download

SEED = 999


def collect_igc_to_flat_ds() -> Dataset:
    """
    Download and prepare the IGC-2024 dataset.

    Downloads from: https://huggingface.co/datasets/arnastofnun/IGC-2024
    """
    # Download the dataset if not already present
    if not os.path.exists("IGC-2024-snapshot"):
        print("Downloading IGC-2024 from HuggingFace...")
        snapshot_download(
            repo_id="arnastofnun/IGC-2024",
            repo_type="dataset",
            cache_dir=None,
            local_dir="IGC-2024-snapshot",
        )

        # Extract all zip files
        extracted_dir = "IGC-2024-extracted"
        os.makedirs(extracted_dir, exist_ok=True)

        for root, dirs, files in os.walk("IGC-2024-snapshot"):
            for file in files:
                if file.endswith(".zip"):
                    zip_path = os.path.join(root, file)
                    print(f"Extracting {zip_path}...")
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extracted_dir)

    # Define the schema (IGC features need explicit definition)
    features = Features(
        {
            "document": Value("string"),
            "uuid": Value("string"),
            "metadata": {
                "author": Value("string"),
                "fetch_timestamp": Value("string"),
                "xml_id": Value("string"),
                "publish_timestamp": Value("string"),
                "title": {"offset": Value("int64"), "length": Value("int64")},
                "paragraphs": List({"offset": Value("int64"), "length": Value("int64")}),
                "sentences": List({"offset": Value("int64"), "length": Value("int64")}),
                "source": Value("string"),
            },
        }
    )

    # Load all JSONL files
    print("Loading JSONL files...")
    ds = load_dataset(
        "json",
        data_files="IGC-2024-extracted/*.jsonl",
        split="train",
        features=features
    )

    # Rename 'document' to 'text' for consistency
    ds = ds.rename_column("document", "text")

    return ds


if __name__ == "__main__":
    print("=" * 60)
    print("Creating IGC Training Dataset")
    print("=" * 60)
    print()

    print("Step 1: Collecting IGC data...")
    igc_ds = collect_igc_to_flat_ds()

    print(f"Loaded {len(igc_ds):,} documents")

    print("\nStep 2: Shuffling...")
    igc_ds = igc_ds.shuffle(seed=SEED)

    print("\nStep 3: Saving to data/igc_full/...")
    os.makedirs("data", exist_ok=True)
    igc_ds.save_to_disk("data/igc_full")

    print()
    print("=" * 60)
    print("Done! Training data saved to: data/igc_full/")
    print(f"Total documents: {len(igc_ds):,}")
    print("=" * 60)
    print()
    print("You can now train your model with:")
    print("  python train_ngram.py --data data/igc_full")
