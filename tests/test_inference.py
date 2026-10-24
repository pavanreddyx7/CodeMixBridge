"""
Unit tests for deterministic inference, adapter toggling,
and error handling using mock model components.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import generate_dual_comparison, generate_single


class MockContextManager:
    def __init__(self, state_tracker, name):
        self.state_tracker = state_tracker
        self.name = name

    def __enter__(self):
        self.state_tracker.append(f"enter_{self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.state_tracker.append(f"exit_{self.name}")


class MockModel:
    def __init__(self):
        self.history = []

    def disable_adapter(self):
        return MockContextManager(self.history, "disable_adapter")

    def generate(self, **kwargs):
        # Return mock token tensor
        # Input has length 5, output has length 8 -> new tokens = 3
        return torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])


class MockTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        # Return dummy dict with input_ids of shape (1, 5)
        return {
            "input_ids": torch.tensor([[10, 20, 30, 40, 50]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

    def decode(self, tokens, **kwargs):
        return "mocked kannada response"


def test_generate_single_with_adapter():
    model = MockModel()
    tokenizer = MockTokenizer()

    pred, duration = generate_single(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        use_adapter=True,
    )
    assert pred == "mocked kannada response"
    assert duration >= 0.0
    # Adapter was NOT disabled
    assert "enter_disable_adapter" not in model.history


def test_generate_single_disable_adapter():
    model = MockModel()
    tokenizer = MockTokenizer()

    pred, duration = generate_single(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        use_adapter=False,
    )
    assert pred == "mocked kannada response"
    # disable_adapter context manager was properly called
    assert "enter_disable_adapter" in model.history
    assert "exit_disable_adapter" in model.history


def test_generate_single_error_handling():
    model = MockModel()
    tokenizer = MockTokenizer()

    # Force tokenizer exception
    tokenizer.apply_chat_template = MagicMock(side_effect=RuntimeError("Tokenization failure"))

    pred, duration = generate_single(
        model=model,
        tokenizer=tokenizer,
        prompt="Bad prompt",
    )
    assert "Generation Error: Tokenization failure" in pred
    assert duration >= 0.0


def test_generate_dual_comparison():
    model = MockModel()
    tokenizer = MockTokenizer()

    res = generate_dual_comparison(
        model=model,
        tokenizer=tokenizer,
        prompt="Hello dual",
    )
    assert "generic" in res
    assert "finetuned" in res
    assert res["generic"]["prediction"] == "mocked kannada response"
    assert res["finetuned"]["prediction"] == "mocked kannada response"
    assert "enter_disable_adapter" in model.history
