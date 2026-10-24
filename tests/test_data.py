"""
Unit tests for data validation, cleaning, source ID stability,
and group-aware partition integrity.
"""

import os
import sys
from pathlib import Path
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    clean_dataset,
    generate_source_id,
    group_and_split,
    normalize_text,
    save_split_manifest,
    validate_record,
)


def test_normalize_text():
    raw = "  Hello   world \u00a0 test  "
    norm = normalize_text(raw)
    assert norm == "Hello world test"

    # Kannada Unicode NFC preservation
    kannada_raw = "ನಮಸ್ಕಾರ   ಸ್ನೇಹಿತರೇ"
    kannada_norm = normalize_text(kannada_raw)
    assert kannada_norm == "ನಮಸ್ಕಾರ ಸ್ನೇಹಿತರೇ"


def test_source_id_stability():
    text_a = "Are you coming to college tomorrow?"
    text_b = "  are you coming to college tomorrow?  "
    id_a = generate_source_id(text_a)
    id_b = generate_source_id(text_b)
    assert id_a.startswith("src_")
    assert id_a == id_b, "Source ID must be deterministic and case/whitespace-insensitive"


def test_validate_record():
    valid_rec = {
        "english": "Good morning",
        "native_script_codemixed": "Good morning ಗೆಳೆಯರೇ",
        "full_native_script": "ಶುಭೋದಯ ಗೆಳೆಯರೇ",
        "romanized_casual": "Good morning geleyare",
    }
    is_valid, err = validate_record(valid_rec)
    assert is_valid is True
    assert err is None

    # Incomplete record
    invalid_rec = {
        "english": "Good morning",
        "romanized_casual": "Good morning",
    }
    is_valid, err = validate_record(invalid_rec)
    assert is_valid is False
    assert "Missing required column" in err


def test_clean_dataset_deduplication():
    records = [
        {
            "english": "See you later",
            "native_script_codemixed": "Later ಸಿಗೋಣ",
            "full_native_script": "ನಂತರ ಭೇಟಿಯಾಗೋಣ",
            "romanized_casual": "Later sigona",
        },
        # Exact duplicate
        {
            "english": "See you later",
            "native_script_codemixed": "Later ಸಿಗೋಣ",
            "full_native_script": "ನಂತರ ಭೇಟಿಯಾಗೋಣ",
            "romanized_casual": "Later sigona",
        },
        # Missing field record
        {
            "english": "Broken row",
            "romanized_casual": "",
        },
    ]
    cleaned, stats = clean_dataset(records)
    assert len(cleaned) == 1
    assert stats["duplicates"] == 1
    assert stats["missing_fields"] == 1
    assert cleaned[0]["source_id"].startswith("src_")


def test_group_and_split_zero_leakage():
    # Construct 10 distinct source groups
    records = []
    for i in range(20):
        src_eng = f"Statement group {i % 10}"
        records.append({
            "source_id": generate_source_id(src_eng),
            "english": src_eng,
            "native_script_codemixed": f"CMix {i}",
            "full_native_script": f"Native {i}",
            "romanized_casual": f"Rom {i}",
        })

    splits = group_and_split(records, seed=3407, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    train_sids = set(r["source_id"] for r in splits["train"])
    val_sids = set(r["source_id"] for r in splits["val"])
    test_sids = set(r["source_id"] for r in splits["test"])

    # Strict isolation verification
    assert len(train_sids.intersection(val_sids)) == 0, "Train and Val share source groups!"
    assert len(train_sids.intersection(test_sids)) == 0, "Train and Test share source groups!"
    assert len(val_sids.intersection(test_sids)) == 0, "Val and Test share source groups!"


def test_save_split_manifest(tmp_path):
    splits = {
        "train": [{"source_id": "src_1", "english": "A"}],
        "val": [{"source_id": "src_2", "english": "B"}],
        "test": [{"source_id": "src_3", "english": "C"}],
    }
    manifest_file = tmp_path / "manifest.json"
    manifest = save_split_manifest(splits, str(manifest_file), seed=3407)
    assert manifest["splits"]["train"]["record_count"] == 1
    assert manifest["splits"]["test"]["record_count"] == 1
    assert manifest_file.exists()
