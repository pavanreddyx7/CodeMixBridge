"""
Model loading module for CodeMixBridge.
Loads Google Gemma 4 E2B via Unsloth in 4-bit precision, attaches LoRA adapters,
and verifies GPU/VRAM configuration.
"""

import os
import torch
from pathlib import Path
from typing import Any, Optional, Tuple


def verify_gpu_environment() -> Tuple[bool, str]:
    """
    Checks if CUDA is available and provides device name and memory summary.
    """
    if not torch.cuda.is_available():
        return False, "No CUDA GPU detected. Running in CPU mode (inference will be slow or disabled)."
    dev_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return True, f"GPU Active: {dev_name} ({total_mem:.2f} GB total VRAM)"


def load_gemma_model(
    base_model_name: str = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
    adapter_path: Optional[str] = None,
    max_seq_length: int = 1024,
    load_in_4bit: bool = True,
) -> Tuple[Any, Any]:
    """
    Loads base Gemma 4 model using Unsloth's FastModel.
    If adapter_path is provided and exists, attaches the trained LoRA weights.
    Returns (model, tokenizer).
    """
    try:
        from unsloth import FastModel  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"Unsloth is not installed in the current environment: {e}. "
            "Please run in Google Colab with T4 GPU runtime or install dependencies from requirements-colab.txt."
        )

    # 1. Load base model
    model, tokenizer = FastModel.from_pretrained(
        model_name=base_model_name,
        dtype=None,  # Auto-select bfloat16 / float16
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        full_finetuning=False,
        text_only=True,         # Drops multimodal vision tower to fit in T4 GPU VRAM
        offload_layers="auto",  # Prevents bitsandbytes CPU spill error
    )

    # 2. Attach LoRA adapter if specified
    if adapter_path and Path(adapter_path).exists():
        print(f"Loading LoRA adapter from: {adapter_path}")
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, adapter_path)
        print("LoRA adapter successfully attached.")

    return model, tokenizer
