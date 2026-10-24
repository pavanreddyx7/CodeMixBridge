"""
Unit tests for prompt generation, task mappings, and chat message structures.
"""

import sys
from pathlib import Path
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import (
    ALL_TASKS,
    PRIMARY_TASKS,
    REVERSE_TASKS,
    TASK_EN_TO_ROMANIZED,
    TASK_EN_TO_CODEMIXED,
    TASK_EN_TO_NATIVE,
    TASK_ROMANIZED_TO_EN,
    TASK_CODEMIXED_TO_EN,
    build_chat_messages,
    build_user_prompt,
    expand_record_to_tasks,
    format_supervised_example,
)


def test_build_user_prompt():
    prompt = build_user_prompt(TASK_EN_TO_ROMANIZED, "Where is the library?")
    assert "Convert the following English message" in prompt
    assert "Text:\nWhere is the library?" in prompt


def test_unknown_task_error():
    with pytest.raises(ValueError, match="Unknown conversion task"):
        build_user_prompt("non_existent_task", "Sample text")


def test_format_supervised_example():
    ex = format_supervised_example(
        source_id="src_123",
        task=TASK_EN_TO_CODEMIXED,
        source_text="Hello friend",
        target_text="Hello ಗೆಳೆಯಾ",
    )
    assert ex["source_id"] == "src_123"
    assert ex["task"] == TASK_EN_TO_CODEMIXED
    assert "Text:\nHello friend" in ex["input"]
    assert ex["output"] == "Hello ಗೆಳೆಯಾ"


def test_expand_record_to_tasks():
    record = {
        "source_id": "src_456",
        "english": "Call me later",
        "native_script_codemixed": "Later call ಮಾಡಿ",
        "full_native_script": "ನಂತರ ಕರೆ ಮಾಡಿ",
        "romanized_casual": "Later call madi",
    }
    # Test primary 3 tasks + 2 reverse tasks = 5 tasks
    tasks = expand_record_to_tasks(record, include_reverse=True)
    assert len(tasks) == 5
    task_names = [t["task"] for t in tasks]
    for p_task in PRIMARY_TASKS:
        assert p_task in task_names
    for r_task in REVERSE_TASKS:
        assert r_task in task_names


def test_build_chat_messages():
    user_prompt = "Hello prompt"
    # User only (inference)
    msgs = build_chat_messages(user_prompt)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == [{"type": "text", "text": user_prompt}]

    # User + Assistant (training)
    msgs_train = build_chat_messages(user_prompt, assistant_response="Kannada response")
    assert len(msgs_train) == 2
    assert msgs_train[1]["role"] == "assistant"
    assert msgs_train[1]["content"] == [{"type": "text", "text": "Kannada response"}]
