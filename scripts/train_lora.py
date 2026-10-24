"""
Training script for CodeMixBridge.
Applies LoRA adapters to Gemma 4 E2B using Unsloth and trains on
the supervised Kannada task examples.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_gemma_model, verify_gpu_environment
from src.training import (
    prepare_training_dataset,
    setup_lora_model,
    train_lora_adapter,
)


def main():
    print("=" * 70)
    print("CodeMixBridge — Gemma 4 LoRA Fine-Tuning")
    print("=" * 70)

    has_gpu, gpu_msg = verify_gpu_environment()
    print(gpu_msg)
    if not has_gpu:
        raise RuntimeError("GPU required for training! Please run inside Google Colab with T4 GPU.")

    # Load training task examples
    train_file = Path("artifacts/train_tasks.json")
    if not train_file.exists():
        print(f"Error: {train_file} not found. Run scripts/prepare_dataset.py first.")
        return

    with open(train_file, "r", encoding="utf-8") as f:
        train_examples = json.load(f)

    print(f"Loaded {len(train_examples)} training task examples.")

    # Load configuration
    config_file = Path("configs/training.json")
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 1. Load base model
    model, tokenizer = load_gemma_model(
        base_model_name=config["model"]["base_model"],
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=config["model"]["load_in_4bit"],
    )

    # 2. Attach LoRA adapters
    lora_cfg = config["lora"]
    model = setup_lora_model(
        model=model,
        lora_rank=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        seed=config["seed"],
    )

    # 3. Format dataset for SFT
    train_dataset, tokenizer = prepare_training_dataset(
        training_examples=train_examples,
        tokenizer=tokenizer,
    )

    # 4. Train adapter
    train_cfg = config["training"]
    metrics = train_lora_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=train_cfg["output_dir"],
        max_steps=train_cfg["max_steps"],
        learning_rate=train_cfg["learning_rate"],
        batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        seed=config["seed"],
    )

    print("\n✅ Training finished successfully!")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
