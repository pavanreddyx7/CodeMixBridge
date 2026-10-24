"""
LoRA Fine-tuning pipeline for CodeMixBridge on Google Colab T4 GPU.
Implements Unsloth QLoRA, response-only loss masking, and adapter saving.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def setup_lora_model(
    model: Any,
    lora_rank: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.0,
    seed: int = 3407,
) -> Any:
    """
    Configures and wraps Gemma 4 model with PEFT LoRA adapters using Unsloth.
    Targets language attention modules and MLPs.
    """
    from unsloth import FastModel  # type: ignore

    peft_model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        random_state=seed,
    )
    return peft_model


def prepare_training_dataset(
    training_examples: List[Dict[str, str]],
    tokenizer: Any,
) -> Any:
    """
    Converts list of {input, output} examples into formatted chat text dataset
    using Gemma 4 chat template.
    """
    from unsloth.chat_templates import get_chat_template  # type: ignore
    from datasets import Dataset  # type: ignore

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    def to_formatted_text(example: Dict[str, str]) -> Dict[str, str]:
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": example["input"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": example["output"]}]},
        ]
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Remove potential leading <bos> token if duplicated
        return {"text": text.removeprefix("<bos>")}

    dataset = Dataset.from_list(training_examples).map(to_formatted_text)
    return dataset, tokenizer


def train_lora_adapter(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    output_dir: str = "artifacts/adapters/gemma_4_kannada_lora",
    max_steps: int = 60,
    learning_rate: float = 2e-4,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 5,
    seed: int = 3407,
) -> Dict[str, Any]:
    """
    Runs LoRA fine-tuning using TRL's SFTTrainer with response-only loss calculation.
    Saves adapter weights, tokenizer, and training metrics upon completion.
    """
    from trl import SFTTrainer, SFTConfig  # type: ignore
    from unsloth.chat_templates import train_on_responses_only  # type: ignore

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=seed,
            output_dir=str(out_path / "checkpoints"),
            report_to="none",
        ),
    )

    # Crucial: Mask input tokens so loss is evaluated strictly on assistant outputs
    trainer = train_on_responses_only(trainer)

    print(f"Starting LoRA training for {max_steps} steps...")
    train_result = trainer.train()

    # Save final model adapters and tokenizer
    print(f"Saving fine-tuned LoRA adapters to: {out_path}")
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    metrics = {
        "train_runtime_sec": train_result.metrics.get("train_runtime", 0.0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0.0),
        "train_loss": train_result.metrics.get("train_loss", 0.0),
        "total_steps": max_steps,
        "learning_rate": learning_rate,
    }

    with open(out_path / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
