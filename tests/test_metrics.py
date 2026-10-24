"""
Unit tests for evaluation metrics, chrF calculation,
empty-output detection, and truncation heuristics.
"""

import sys
from pathlib import Path
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    check_is_empty,
    check_is_truncated,
    compute_corpus_chrf,
    compute_sentence_chrf,
    compute_task_metrics,
)


def test_sentence_chrf_identical():
    ref = "naale college ge barthiya?"
    hyp = "naale college ge barthiya?"
    score = compute_sentence_chrf(hyp, ref)
    assert score > 99.0, "Identical strings should achieve ~100 chrF score"


def test_sentence_chrf_empty():
    ref = "kannada text"
    assert compute_sentence_chrf("", ref) == 0.0
    assert compute_sentence_chrf("   ", ref) == 0.0


def test_corpus_chrf():
    refs = ["naale sigona", "shubhodaya"]
    hyps = ["naale sigona", "shubhodaya"]
    score = compute_corpus_chrf(hyps, refs)
    assert score > 99.0


def test_empty_detection():
    assert check_is_empty("") is True
    assert check_is_empty("   \n\t") is True
    assert check_is_empty("valid text") is False


def test_truncation_detection():
    # Complete sentence
    complete = "This is a finished thought."
    assert check_is_truncated(complete, max_tokens=10) is False

    # Abruptly cut off sentence without terminal punctuation
    long_cut_off = "This message continues for a very long duration and ends unexpectedly without punctuation"
    assert check_is_truncated(long_cut_off, max_tokens=15, approx_chars_per_token=3.5) is True


def test_compute_task_metrics_aggregation():
    predictions = [
        {
            "prediction": "naale sigona",
            "reference": "naale sigona",
            "duration_sec": 0.45,
        },
        {
            "prediction": "shubhodaya geleyare",
            "reference": "shubhodaya",
            "duration_sec": 0.55,
        },
        {
            "prediction": "",
            "reference": "missing text",
            "duration_sec": 0.20,
        },
    ]
    summary = compute_task_metrics(predictions)
    assert summary["count"] == 3
    assert summary["corpus_chrf"] > 0.0
    assert summary["empty_rate_pct"] == round(1/3 * 100, 2)
    assert summary["median_latency_sec"] == 0.45
