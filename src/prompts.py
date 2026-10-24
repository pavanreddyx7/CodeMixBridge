"""
Single source of truth for prompt templates, instruction formatting,
and task mappings across training, evaluation, and user interfaces.
"""

from typing import Any, Dict, List, Optional


# Explicit conversion task identifiers
TASK_EN_TO_ROMANIZED = "english_to_romanized"
TASK_EN_TO_CODEMIXED = "english_to_codemixed"
TASK_EN_TO_NATIVE = "english_to_native"
TASK_ROMANIZED_TO_EN = "romanized_to_english"
TASK_CODEMIXED_TO_EN = "codemixed_to_english"

PRIMARY_TASKS = [
    TASK_EN_TO_ROMANIZED,
    TASK_EN_TO_CODEMIXED,
    TASK_EN_TO_NATIVE,
]

REVERSE_TASKS = [
    TASK_ROMANIZED_TO_EN,
    TASK_CODEMIXED_TO_EN,
]

ALL_TASKS = PRIMARY_TASKS + REVERSE_TASKS

# Human-readable labels for the Gradio UI and reports
TASK_LABELS: Dict[str, str] = {
    TASK_EN_TO_ROMANIZED: "English → Romanized Kannada (Casual Chat)",
    TASK_EN_TO_CODEMIXED: "English → Kannada Code-Mixed (Native Script)",
    TASK_EN_TO_NATIVE: "English → Full Kannada (Formal Native Script)",
    TASK_ROMANIZED_TO_EN: "Romanized Kannada → English (De-Mixing)",
    TASK_CODEMIXED_TO_EN: "Kannada Code-Mixed → English (Standardization)",
}

# Reverse mapping from UI label to task key
LABEL_TO_TASK: Dict[str, str] = {v: k for k, v in TASK_LABELS.items()}

# Core instruction strings: explicit, concise, and strict
TASK_INSTRUCTIONS: Dict[str, str] = {
    TASK_EN_TO_ROMANIZED: (
        "Convert the following English message into natural casual Romanized Kannada (Kanglish).\n"
        "Preserve meaning, names, numbers, dates, and negation.\n"
        "Return only the converted message."
    ),
    TASK_EN_TO_CODEMIXED: (
        "Convert the following English message into natural Kannada script code-mixed text.\n"
        "Preserve meaning, names, numbers, dates, and negation.\n"
        "Return only the converted message."
    ),
    TASK_EN_TO_NATIVE: (
        "Convert the following English message into clear, formal native Kannada script.\n"
        "Preserve meaning, names, numbers, dates, and negation.\n"
        "Return only the converted message."
    ),
    TASK_ROMANIZED_TO_EN: (
        "Translate this casual Romanized Kannada (Kanglish) message into clear standard English.\n"
        "Preserve meaning, names, numbers, dates, and negation.\n"
        "Return only the converted message."
    ),
    TASK_CODEMIXED_TO_EN: (
        "Translate this code-mixed Kannada script message into clear standard English.\n"
        "Preserve meaning, names, numbers, dates, and negation.\n"
        "Return only the converted message."
    ),
}


def build_user_prompt(task: str, source_text: str) -> str:
    """
    Constructs the exact user prompt string for a given task and input text.
    Used identically across Generic baseline, Fine-Tuned inference, and Gradio.
    """
    if task not in TASK_INSTRUCTIONS:
        raise ValueError(f"Unknown conversion task: '{task}'. Supported tasks: {ALL_TASKS}")
    instruction = TASK_INSTRUCTIONS[task]
    clean_text = str(source_text).strip()
    return f"{instruction}\n\nText:\n{clean_text}"


def format_supervised_example(
    source_id: str,
    task: str,
    source_text: str,
    target_text: str,
) -> Dict[str, str]:
    """
    Produces a standardized dictionary representation of a training or evaluation instance.
    """
    user_prompt = build_user_prompt(task, source_text)
    return {
        "source_id": source_id,
        "task": task,
        "input": user_prompt,
        "output": str(target_text).strip(),
    }


def expand_record_to_tasks(record: Dict[str, Any], include_reverse: bool = True) -> List[Dict[str, str]]:
    """
    Expands a single cleaned IndicCMix Kannada record into all active supervised task examples.
    """
    source_id = record["source_id"]
    eng = record["english"]
    cmix = record["native_script_codemixed"]
    native = record["full_native_script"]
    rom = record["romanized_casual"]

    examples = [
        format_supervised_example(source_id, TASK_EN_TO_ROMANIZED, eng, rom),
        format_supervised_example(source_id, TASK_EN_TO_CODEMIXED, eng, cmix),
        format_supervised_example(source_id, TASK_EN_TO_NATIVE, eng, native),
    ]

    if include_reverse:
        examples.append(format_supervised_example(source_id, TASK_ROMANIZED_TO_EN, rom, eng))
        examples.append(format_supervised_example(source_id, TASK_CODEMIXED_TO_EN, cmix, eng))

    return examples


def build_chat_messages(user_prompt: str, assistant_response: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Builds the standard chat messages payload compatible with tokenizer.apply_chat_template
    for Gemma 4 multimodal and text processors.
    """
    messages = [
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_response}]})
    return messages
