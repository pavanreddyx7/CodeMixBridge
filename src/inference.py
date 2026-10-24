"""
Deterministic inference engine for CodeMixBridge.
Manages runtime adapter state switching (Generic Gemma vs Fine-Tuned Gemma)
with strict error recovery and execution locking.
"""

import threading
import time
from typing import Any, Dict, Optional, Tuple
import torch

from src.prompts import build_chat_messages


# Thread lock to guarantee sequential generation and prevent race conditions during adapter toggling
_INFERENCE_LOCK = threading.Lock()


def generate_single(
    model: Any,
    tokenizer: Any,
    prompt: str,
    use_adapter: bool = True,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> Tuple[str, float]:
    """
    Generates text from a single model configuration with safe adapter state management.
    - If use_adapter is False: runs inside `model.disable_adapter()` context (Generic Gemma).
    - If use_adapter is True: runs with active LoRA weights (Fine-Tuned Gemma).
    Returns (generated_text, duration_seconds).
    """
    with _INFERENCE_LOCK:
        start_time = time.perf_counter()
        messages = build_chat_messages(user_prompt=prompt)

        try:
            # Prepare model inputs with Gemma chat template
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            sampling_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
                "do_sample": do_sample,
            }
            if do_sample and temperature > 0.0:
                sampling_kwargs["temperature"] = temperature
                sampling_kwargs["top_p"] = 0.95
                sampling_kwargs["top_k"] = 64

            with torch.no_grad():
                if not use_adapter and hasattr(model, "disable_adapter"):
                    # Generic Gemma: LoRA adapter temporarily disabled
                    with model.disable_adapter():
                        outputs = model.generate(**inputs, **sampling_kwargs)
                else:
                    # Fine-Tuned Gemma: Active LoRA adapter
                    outputs = model.generate(**inputs, **sampling_kwargs)

            # Slice new tokens only
            input_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_len:]
            prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        except Exception as e:
            # Clean error recovery
            prediction = f"(Generation Error: {str(e)})"

        duration = time.perf_counter() - start_time
        return prediction, duration


def generate_dual_comparison(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """
    Executes both model configurations sequentially on the identical prompt:
    1. Generic Gemma (adapter disabled)
    2. Fine-Tuned Gemma (adapter enabled)
    Returns dictionary with predictions and inference latencies.
    """
    # 1. Generic Gemma
    generic_text, generic_lat = generate_single(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        use_adapter=False,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # 2. Fine-Tuned Gemma
    finetuned_text, finetuned_lat = generate_single(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        use_adapter=True,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    return {
        "generic": {
            "prediction": generic_text,
            "latency_sec": round(generic_lat, 3),
        },
        "finetuned": {
            "prediction": finetuned_text,
            "latency_sec": round(finetuned_lat, 3),
        },
    }
