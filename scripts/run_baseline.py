"""
Baseline evaluation script for CodeMixBridge.
Runs Generic Gemma (un-fine-tuned base checkpoint) on the test split
and saves predictions with runtime metadata to results/baseline_predictions.jsonl.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_gemma_model, verify_gpu_environment
from src.evaluation import evaluate_model_on_tasks


def main():
    print("=" * 70)
    print("CodeMixBridge — Generic Gemma Baseline Evaluation")
    print("=" * 70)

    has_gpu, gpu_msg = verify_gpu_environment()
    print(gpu_msg)
    if not has_gpu:
        print("Note: Running on CPU will be extremely slow. GPU is strongly recommended.")

    # Load test split task examples
    test_file = Path("artifacts/test_tasks.json")
    if not test_file.exists():
        print(f"Error: {test_file} not found. Run scripts/prepare_dataset.py first.")
        return

    with open(test_file, "r", encoding="utf-8") as f:
        test_examples = json.load(f)

    print(f"Loaded {len(test_examples)} test task examples.")

    # Load base model without adapters
    model, tokenizer = load_gemma_model(
        base_model_name="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
        adapter_path=None,
        load_in_4bit=True,
    )

    out_jsonl = "results/baseline_predictions.jsonl"
    print(f"Evaluating Generic Gemma... (Predictions will be saved to {out_jsonl})")

    _, summary = evaluate_model_on_tasks(
        model=model,
        tokenizer=tokenizer,
        test_examples=test_examples,
        model_name="Generic Gemma",
        use_adapter=False,
        output_jsonl_path=out_jsonl,
        max_new_tokens=128,
    )

    print("\nBaseline Summary:")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
