"""
Evaluation and benchmarking engine for CodeMixBridge.
Runs deterministic test batch evaluations, serializes predictions to JSONL,
and produces comparative CSV metric reports.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.metrics import compute_task_metrics
from src.inference import generate_single
from src.prompts import (
    ALL_TASKS,
    PRIMARY_TASKS,
    TASK_EN_TO_ROMANIZED,
    TASK_EN_TO_CODEMIXED,
    TASK_EN_TO_NATIVE,
)


def evaluate_model_on_tasks(
    model: Any,
    tokenizer: Any,
    test_examples: List[Dict[str, str]],
    model_name: str = "Generic Gemma",
    use_adapter: bool = False,
    output_jsonl_path: Optional[str] = None,
    max_new_tokens: int = 128,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluates a specific model configuration across all given test examples.
    Records per-instance predictions, latencies, and writes to JSONL.
    Returns (all_predictions, aggregated_metrics).
    """
    predictions: List[Dict[str, Any]] = []

    for idx, ex in enumerate(test_examples, start=1):
        task = ex.get("task", "unknown")
        src_id = ex.get("source_id", f"item_{idx}")
        input_text = ex["input"]
        reference = ex["output"]

        pred_text, duration = generate_single(
            model=model,
            tokenizer=tokenizer,
            prompt=input_text,
            use_adapter=use_adapter,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

        record = {
            "source_id": src_id,
            "task": task,
            "input_text": input_text,
            "reference_text": reference,
            "prediction": pred_text,
            "model_configuration": model_name,
            "prompt_version": "v1.0",
            "generation_settings": {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": 0.0,
            },
            "duration_sec": round(duration, 3),
        }
        predictions.append(record)
        print(f"\r[{model_name}] Evaluated {idx}/{len(test_examples)} examples...", end="")

    print("\nEvaluation complete.")

    # Save to JSONL if requested
    if output_jsonl_path:
        out_file = Path(output_jsonl_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Predictions saved to: {output_jsonl_path}")

    # Compute task-level and overall metrics
    metrics_by_task = {}
    for task_name in set(p["task"] for p in predictions):
        sub_preds = [p for p in predictions if p["task"] == task_name]
        metrics_by_task[task_name] = compute_task_metrics(sub_preds)

    overall_metrics = compute_task_metrics(predictions)
    summary = {
        "model_name": model_name,
        "overall": overall_metrics,
        "by_task": metrics_by_task,
    }
    return predictions, summary


def generate_comparison_csv(
    generic_summary: Dict[str, Any],
    finetuned_summary: Dict[str, Any],
    output_csv_path: str = "results/metrics.csv",
) -> pd.DataFrame:
    """
    Compiles actual comparative evaluation results into results/metrics.csv.
    Compares Generic Gemma against Fine-Tuned Gemma.
    """
    rows = []
    for model_summary in [generic_summary, finetuned_summary]:
        m_name = model_summary["model_name"]
        by_task = model_summary.get("by_task", {})

        rom_chrf = by_task.get(TASK_EN_TO_ROMANIZED, {}).get("corpus_chrf", 0.0)
        cmix_chrf = by_task.get(TASK_EN_TO_CODEMIXED, {}).get("corpus_chrf", 0.0)
        nat_chrf = by_task.get(TASK_EN_TO_NATIVE, {}).get("corpus_chrf", 0.0)
        overall_chrf = model_summary.get("overall", {}).get("corpus_chrf", 0.0)
        empty_rate = model_summary.get("overall", {}).get("empty_rate_pct", 0.0)
        trunc_rate = model_summary.get("overall", {}).get("truncation_rate_pct", 0.0)
        med_lat = model_summary.get("overall", {}).get("median_latency_sec", 0.0)

        rows.append({
            "Model": m_name,
            "Romanized chrF": rom_chrf,
            "Code-mixed chrF": cmix_chrf,
            "Native chrF": nat_chrf,
            "Overall chrF": overall_chrf,
            "Empty Rate (%)": empty_rate,
            "Truncation Rate (%)": trunc_rate,
            "Median Latency (s)": med_lat,
        })

    df = pd.DataFrame(rows)
    out_file = Path(output_csv_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Benchmark summary saved to: {output_csv_path}")
    return df
