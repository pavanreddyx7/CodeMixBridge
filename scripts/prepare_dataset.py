"""
Dataset preparation script for CodeMixBridge.
Loads raw Kannada data, applies Unicode normalization, cleans missing values,
partitions groups with seed 3407 (70% train / 15% val / 15% test),
and generates artifacts/split_manifest.json.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data import (
    clean_dataset,
    group_and_split,
    load_kannada_indiccmix_hf,
    save_split_manifest,
)
from src.prompts import expand_record_to_tasks


def main():
    print("=" * 70)
    print("CodeMixBridge — Dataset Preparation & Split Partitioning")
    print("=" * 70)

    # 1. Load data
    try:
        raw_records = load_kannada_indiccmix_hf()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Raw Records Loaded: {len(raw_records)}")

    # 2. Clean and deduplicate
    cleaned_records, clean_stats = clean_dataset(raw_records)
    print("\nCleaning Summary:")
    print(f"  - Valid Cleaned Records: {clean_stats['valid_count']}")
    print(f"  - Missing/Incomplete Records Removed: {clean_stats['missing_fields']}")
    print(f"  - Duplicate Records Removed: {clean_stats['duplicates']}")

    # 3. Group and split (70% train / 15% val / 15% test)
    splits = group_and_split(
        cleaned_records,
        seed=3407,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    print("\nData Split Allocation:")
    for s_name, recs in splits.items():
        unique_groups = len(set(r["source_id"] for r in recs))
        print(f"  - {s_name.capitalize()}: {len(recs)} records across {unique_groups} source groups")

    # 4. Save audit manifest
    manifest_path = "artifacts/split_manifest.json"
    manifest = save_split_manifest(splits, output_path=manifest_path, seed=3407)
    print(f"\n✅ Audit manifest saved to: {manifest_path}")

    # 5. Expand supervised task examples and save locally for training & evaluation
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for s_name, recs in splits.items():
        expanded = []
        for r in recs:
            expanded.extend(expand_record_to_tasks(r, include_reverse=True))
        out_json = artifacts_dir / f"{s_name}_tasks.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(expanded, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(expanded)} supervised task examples to: {out_json}")

    print("\n✅ Dataset preparation completed successfully.")


if __name__ == "__main__":
    main()
