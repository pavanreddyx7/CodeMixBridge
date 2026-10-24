"""
Dataset inspection script for CodeMixBridge.
Inspects the schema of ai4bharat/IndicCMix (ka.parquet), validates columns,
analyzes missing values, and prints 20 sample records for linguistic inspection.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data import load_kannada_indiccmix_hf, normalize_text, REQUIRED_COLUMNS


def main():
    print("=" * 70)
    print("CodeMixBridge — Kannada Dataset Inspection")
    print("Dataset: ai4bharat/IndicCMix (ka.parquet)")
    print("=" * 70)

    try:
        records = load_kannada_indiccmix_hf()
    except Exception as e:
        print(f"\n[Warning] Could not load directly from Hugging Face hub: {e}")
        print("Please ensure HF_TOKEN is set in your environment.")
        return

    print(f"\nTotal Raw Records Downloaded: {len(records)}")

    # Schema inspection
    if records:
        print(f"\nDiscovered Columns: {list(records[0].keys())}")
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in records[0]]
        if missing_cols:
            print(f"❌ Missing expected columns: {missing_cols}")
        else:
            print("✅ All required text columns present.")

    # Missing value count
    null_counts = {col: 0 for col in REQUIRED_COLUMNS}
    for r in records:
        for col in REQUIRED_COLUMNS:
            val = normalize_text(r.get(col, ""))
            if not val:
                null_counts[col] += 1

    print("\nEmpty/Missing Value Counts per Column:")
    for col, count in null_counts.items():
        print(f"  - {col}: {count} ({count/len(records)*100:.2f}%)")

    # Sample inspection: 20 records
    print("\n" + "=" * 70)
    print("Linguistic Review: First 20 Raw Kannada Records")
    print("=" * 70)

    for i, r in enumerate(records[:20], start=1):
        print(f"\n--- RECORD {i:02d} ---")
        print(f"English:                 {r.get('english', '').strip()}")
        print(f"Romanized Casual:        {r.get('romanized_casual', '').strip()}")
        print(f"Native Code-Mixed:       {r.get('native_script_codemixed', '').strip()}")
        print(f"Full Native Script:      {r.get('full_native_script', '').strip()}")


if __name__ == "__main__":
    main()
