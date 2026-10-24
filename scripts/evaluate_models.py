"""
Full model evaluation and benchmarking script for CodeMixBridge.
Evaluates Fine-Tuned Gemma on the test split, saves predictions to
results/finetuned_predictions.jsonl, loads baseline predictions,
and generates the final results/metrics.csv comparison table.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_gemma_model, verify_gpu_environment
from src.evaluation import evaluate_model_on_tasks, generate_comparison_csv
from src.metrics import compute_task_metrics


def main():
    print("=" * 70)
    print("CodeMixBridge — Model Evaluation & Benchmark Report")
    print("=" * 70)

    test_file = Path("artifacts/test_tasks.json")
    if not test_file.exists():
        print(f"Error: {test_file} not found. Run scripts/prepare_dataset.py first.")
        return

    with open(test_file, "r", encoding="utf-8") as f:
        test_examples = json.load(f)

    # Check baseline predictions
    baseline_jsonl = Path("results/baseline_predictions.jsonl")
    if not baseline_jsonl.exists():
        print(f"Warning: {baseline_jsonl} not found. Running baseline evaluation first is recommended.")
        generic_summary = {"model_name": "Generic Gemma", "overall": {}, "by_task": {}}
    else:
        with open(baseline_jsonl, "r", encoding="utf-8") as f:
            base_preds = [json.loads(line) for line in f if line.strip()]
        base_tasks = {}
        for t in set(p["task"] for p in base_preds):
            base_tasks[t] = compute_task_metrics([p for p in base_preds if p["task"] == t])
        generic_summary = {
            "model_name": "Generic Gemma",
            "overall": compute_task_metrics(base_preds),
            "by_task": base_tasks,
        }

    # Evaluate Fine-Tuned Gemma
    adapter_dir = "artifacts/adapters/gemma_4_kannada_lora"
    print(f"Loading Fine-Tuned Gemma with adapter from: {adapter_dir}")

    model, tokenizer = load_gemma_model(
        base_model_name="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
        adapter_path=adapter_dir,
        load_in_4bit=True,
    )

    out_jsonl = "results/finetuned_predictions.jsonl"
    print(f"Evaluating Fine-Tuned Gemma... (Predictions will be saved to {out_jsonl})")

    _, finetuned_summary = evaluate_model_on_tasks(
        model=model,
        tokenizer=tokenizer,
        test_examples=test_examples,
        model_name="Fine-Tuned Gemma",
        use_adapter=True,
        output_jsonl_path=out_jsonl,
        max_new_tokens=128,
    )

    # Generate comparison CSV
    print("\nGenerating final comparative benchmark metrics...")
    df = generate_comparison_csv(
        generic_summary=generic_summary,
        finetuned_summary=finetuned_summary,
        output_csv_path="results/metrics.csv",
    )
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
