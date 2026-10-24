"""
Dataset processing, validation, group-aware splitting, and manifest generation
for CodeMixBridge on the AI4Bharat IndicCMix Kannada dataset.
"""

import hashlib
import json
import os
import random
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REQUIRED_COLUMNS = [
    "english",
    "native_script_codemixed",
    "full_native_script",
    "romanized_casual",
]


def normalize_text(text: Any) -> str:
    """
    Applies standard Unicode normalization (NFC) and cleans whitespace
    while preserving Kannada characters, diacritics, and punctuation.
    """
    if text is None:
        return ""
    # Convert to string and apply Unicode NFC normalization
    norm = unicodedata.normalize("NFC", str(text))
    # Replace non-breaking spaces with standard space
    norm = norm.replace("\u00a0", " ")
    # Collapse multiple spaces while preserving single newlines
    lines = [ " ".join(line.split()) for line in norm.splitlines() ]
    return "\n".join(lines).strip()


def generate_source_id(text: str) -> str:
    """
    Generates a deterministic 12-character hex ID based on normalized English source text.
    Ensures identical source sentences receive the same group identifier across splits.
    """
    norm = normalize_text(text).lower()
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]
    return f"src_{digest}"


def validate_record(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validates that a single raw record contains all required columns with valid string data.
    """
    for col in REQUIRED_COLUMNS:
        if col not in record:
            return False, f"Missing required column: {col}"
        val = normalize_text(record.get(col, ""))
        if not val:
            return False, f"Empty content in column: {col}"
    return True, None


def clean_dataset(raw_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Cleans raw dataset records:
    1. Removes 'audio' if present
    2. Applies Unicode normalization
    3. Validates completeness
    4. Deduplicates identical (english, romanized_casual) tuples
    5. Assigns stable source_id
    """
    cleaned: List[Dict[str, Any]] = []
    seen_keys = set()
    stats = {
        "raw_count": len(raw_records),
        "missing_fields": 0,
        "duplicates": 0,
        "valid_count": 0,
    }

    for item in raw_records:
        valid, _ = validate_record(item)
        if not valid:
            stats["missing_fields"] += 1
            continue

        norm_eng = normalize_text(item["english"])
        norm_cmix = normalize_text(item["native_script_codemixed"])
        norm_native = normalize_text(item["full_native_script"])
        norm_rom = normalize_text(item["romanized_casual"])

        dedup_key = (norm_eng.lower(), norm_rom.lower())
        if dedup_key in seen_keys:
            stats["duplicates"] += 1
            continue
        seen_keys.add(dedup_key)

        src_id = generate_source_id(norm_eng)
        cleaned.append({
            "source_id": src_id,
            "english": norm_eng,
            "native_script_codemixed": norm_cmix,
            "full_native_script": norm_native,
            "romanized_casual": norm_rom,
        })

    stats["valid_count"] = len(cleaned)
    return cleaned, stats


def group_and_split(
    records: List[Dict[str, Any]],
    seed: int = 3407,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Groups records by normalized English source text (source_id) and partitions groups.
    Ensures zero leakage across train, val, and test splits.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # Group records by source_id
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        sid = r["source_id"]
        if sid not in groups:
            groups[sid] = []
        groups[sid].append(r)

    unique_sids = sorted(list(groups.keys()))
    rng = random.Random(seed)
    rng.shuffle(unique_sids)

    n_groups = len(unique_sids)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)

    train_sids = set(unique_sids[:n_train])
    val_sids = set(unique_sids[n_train : n_train + n_val])
    test_sids = set(unique_sids[n_train + n_val :])

    # Verification: zero overlap between split group IDs
    assert len(train_sids.intersection(val_sids)) == 0, "Train and Val overlap detected!"
    assert len(train_sids.intersection(test_sids)) == 0, "Train and Test overlap detected!"
    assert len(val_sids.intersection(test_sids)) == 0, "Val and Test overlap detected!"

    train_recs = [r for sid in train_sids for r in groups[sid]]
    val_recs = [r for sid in val_sids for r in groups[sid]]
    test_recs = [r for sid in test_sids for r in groups[sid]]

    # Shuffle instances inside each split deterministically
    rng.shuffle(train_recs)
    rng.shuffle(val_recs)
    rng.shuffle(test_recs)

    return {
        "train": train_recs,
        "val": val_recs,
        "test": test_recs,
    }


def save_split_manifest(
    splits: Dict[str, List[Dict[str, Any]]],
    output_path: str = "artifacts/split_manifest.json",
    seed: int = 3407,
    dataset_revision: str = "ai4bharat/IndicCMix-ka",
) -> Dict[str, Any]:
    """
    Saves a complete manifest file tracking data partition sizes, source IDs,
    and group assignments for scientific auditability.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_revision": dataset_revision,
        "language": "Kannada",
        "seed": seed,
        "splits": {
            "train": {
                "record_count": len(splits["train"]),
                "group_count": len(set(r["source_id"] for r in splits["train"])),
                "source_ids": sorted(list(set(r["source_id"] for r in splits["train"]))),
            },
            "val": {
                "record_count": len(splits["val"]),
                "group_count": len(set(r["source_id"] for r in splits["val"])),
                "source_ids": sorted(list(set(r["source_id"] for r in splits["val"]))),
            },
            "test": {
                "record_count": len(splits["test"]),
                "group_count": len(set(r["source_id"] for r in splits["test"])),
                "source_ids": sorted(list(set(r["source_id"] for r in splits["test"]))),
            },
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def load_kannada_indiccmix_hf(hf_token: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads Kannada subset from Hugging Face repository using authenticated access.
    Reads token safely without logging credentials.
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                    token = line.split("=", 1)[1].strip()
                    os.environ["HF_TOKEN"] = token
                    break

    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(
            "ai4bharat/IndicCMix",
            data_files="ka.parquet",
            split="train",
            token=token,
        )
        if "audio" in ds.column_names:
            ds = ds.remove_columns(["audio"])
        return [dict(row) for row in ds]
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ai4bharat/IndicCMix (ka.parquet) from Hugging Face: {e}. "
            "Ensure you have accepted the dataset terms at https://huggingface.co/datasets/ai4bharat/IndicCMix "
            "and set your HF_TOKEN."
        )
