"""
Self-contained test runner executing all unit tests in tests/ without external test runners.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_data import (
    test_normalize_text,
    test_source_id_stability,
    test_validate_record,
    test_clean_dataset_deduplication,
    test_group_and_split_zero_leakage,
)
from tests.test_prompts import (
    test_build_user_prompt,
    test_unknown_task_error,
    test_format_supervised_example,
    test_expand_record_to_tasks,
    test_build_chat_messages,
)
from tests.test_metrics import (
    test_sentence_chrf_identical,
    test_sentence_chrf_empty,
    test_corpus_chrf,
    test_empty_detection,
    test_truncation_detection,
    test_compute_task_metrics_aggregation,
)
from tests.test_inference import (
    test_generate_single_with_adapter,
    test_generate_single_disable_adapter,
    test_generate_single_error_handling,
    test_generate_dual_comparison,
)


def run_all():
    tests = [
        ("test_data: normalize_text", test_normalize_text),
        ("test_data: source_id_stability", test_source_id_stability),
        ("test_data: validate_record", test_validate_record),
        ("test_data: clean_dataset_deduplication", test_clean_dataset_deduplication),
        ("test_data: group_and_split_zero_leakage", test_group_and_split_zero_leakage),
        ("test_prompts: build_user_prompt", test_build_user_prompt),
        ("test_prompts: unknown_task_error", test_unknown_task_error),
        ("test_prompts: format_supervised_example", test_format_supervised_example),
        ("test_prompts: expand_record_to_tasks", test_expand_record_to_tasks),
        ("test_prompts: build_chat_messages", test_build_chat_messages),
        ("test_metrics: sentence_chrf_identical", test_sentence_chrf_identical),
        ("test_metrics: sentence_chrf_empty", test_sentence_chrf_empty),
        ("test_metrics: corpus_chrf", test_corpus_chrf),
        ("test_metrics: empty_detection", test_empty_detection),
        ("test_metrics: truncation_detection", test_truncation_detection),
        ("test_metrics: compute_task_metrics_aggregation", test_compute_task_metrics_aggregation),
        ("test_inference: generate_single_with_adapter", test_generate_single_with_adapter),
        ("test_inference: generate_single_disable_adapter", test_generate_single_disable_adapter),
        ("test_inference: generate_single_error_handling", test_generate_single_error_handling),
        ("test_inference: generate_dual_comparison", test_generate_dual_comparison),
    ]

    print("=" * 70)
    print("CodeMixBridge — Automated Unit Test Suite Execution")
    print("=" * 70)

    passed = 0
    failed = 0

    for name, func in tests:
        try:
            func()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 70)
    print(f"Summary: {passed} passed, {failed} failed (Total: {len(tests)})")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
