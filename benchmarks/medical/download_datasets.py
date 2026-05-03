"""
Pre-download OpenMedicalLLM Leaderboard datasets.

This script downloads all datasets without running evaluation,
useful for preparing data on systems with limited internet access.

Usage:
    python download_datasets.py --cache_dir ./datasets_cache
"""

import argparse
import os
from datasets import load_dataset

DATASET_CONFIGS = {
    "medqa": {
        "name": "MedQA (USMLE)",
        "path": "bigbio/med_qa",
        "config": "med_qa_en",
        "split": "test",
    },
    "pubmedqa": {
        "name": "PubMedQA",
        "path": "pubmedqa",
        "config": "pqa_artificial",
        "split": "train",
    },
    "medmcqa": {
        "name": "MedMCQA",
        "path": "medmcqa",
        "config": None,
        "split": "test",
    },
    "mmlu_anatomy": {
        "name": "MMLU Anatomy",
        "path": "cais/mmlu",
        "config": "anatomy",
        "split": "test",
    },
    "mmlu_professional_medicine": {
        "name": "MMLU Professional Medicine",
        "path": "cais/mmlu",
        "config": "professional_medicine",
        "split": "test",
    },
    "mmlu_college_medicine": {
        "name": "MMLU College Medicine",
        "path": "cais/mmlu",
        "config": "college_medicine",
        "split": "test",
    },
    "mmlu_college_biology": {
        "name": "MMLU College Biology",
        "path": "cais/mmlu",
        "config": "college_biology",
        "split": "test",
    },
}


def download_dataset(name: str, config: dict, cache_dir: str):
    """Download a single dataset."""
    print(f"\n📥 Downloading {config['name']}...")
    try:
        kwargs = {
            "cache_dir": cache_dir,
            "trust_remote_code": True,
        }
        if config["config"]:
            kwargs["name"] = config["config"]

        dataset = load_dataset(config["path"], **kwargs)

        # Access the specified split to ensure it's downloaded
        split_data = dataset[config["split"]]
        print(f"   ✓ Downloaded {len(split_data)} examples")
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download OpenMedicalLLM Leaderboard datasets"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./datasets_cache",
        help="Directory to cache datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to download (default: all)"
    )

    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    datasets_to_download = args.datasets if args.datasets else list(DATASET_CONFIGS.keys())

    success_count = 0
    for name in datasets_to_download:
        if name not in DATASET_CONFIGS:
            print(f"⚠ Unknown dataset: {name}, skipping...")
            continue

        if download_dataset(name, DATASET_CONFIGS[name], args.cache_dir):
            success_count += 1

    print(f"\n{'='*50}")
    print(f"Downloaded {success_count}/{len(datasets_to_download)} datasets")
    print(f"Cache location: {args.cache_dir}")


if __name__ == "__main__":
    main()
