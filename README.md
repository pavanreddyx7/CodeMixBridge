# CodeMixBridge: Kannada-English Language Conversion

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Unsloth](https://img.shields.io/badge/Fine--Tuning-Unsloth%20QLoRA-orange.svg)](https://github.com/unslothai/unsloth)
[![Gemma 4](https://img.shields.io/badge/Model-Gemma%204%20E2B-red.svg)](https://ai.google.dev/gemma)
[![Colab T4](https://img.shields.io/badge/Hardware-Colab%20T4-purple.svg)](https://colab.research.google.com/)

> Bridging Kannada and English through parameter-efficient Gemma fine-tuning.

## Project overview

CodeMixBridge is a Python-based machine learning project that improves English-to-Kannada language conversion by fine-tuning Google’s Gemma 4 E2B instruction model. It supports three tasks: English to Romanized Kannada, English to Kannada-script code-mixed text, and English to formal native Kannada. The project uses the AI4Bharat IndicCMix Kannada dataset, containing English sentences with corresponding Romanized, code-mixed, and native-script outputs. Data is normalized, deduplicated, grouped by source, and divided into deterministic training, validation, and testing sets without leakage. Each source record is expanded into three task-specific examples. The model is loaded in pre-quantized 4-bit format using Unsloth and bitsandbytes, allowing training on a Google Colab Tesla T4 GPU. LoRA parameter-efficient fine-tuning updates small adapter matrices while keeping the original Gemma weights frozen. PyTorch performs GPU computation, Hugging Face Transformers handles tokenization and generation, TRL provides the supervised fine-tuning trainer, and PEFT manages LoRA adapters. Training uses response-only loss, gradient accumulation, an 8-bit AdamW optimizer, and a cosine learning-rate schedule. Generic Gemma and fine-tuned Gemma are evaluated on identical held-out examples using deterministic generation and SacreBLEU chrF scores. Pandas organizes results, while Gradio provides an interactive interface for comparing both models’ Kannada outputs side by side.

## Why fine-tuning adds value

Generic instruction models understand Kannada, but often respond with explanations, literal translations, inconsistent scripts, or unnatural phrasing when asked for casual Kannada. CodeMixBridge specializes Gemma for the exact output styles used by Kannada speakers. The LoRA adapter learns Kannada word order, colloquial transliteration, code-mixing patterns, script fidelity, and the requirement to return only the converted message. It also improves preservation of names, numbers, dates, and negation.

LoRA keeps the original model frozen and trains only a small set of low-rank adapter weights. This makes the experiment affordable on a T4 GPU, produces a small reusable adapter, and allows a controlled comparison between the base and fine-tuned configurations without loading two full models.

## Measured benchmark result

The controlled Colab benchmark evaluated plain and fine-tuned Gemma on the same held-out examples with deterministic decoding.

| Model | Mean chrF |
|---|---:|
| Plain Gemma | 38.86 |
| Fine-tuned Gemma | **54.85** |
| Absolute gain | **+15.99** |

This is an approximately **41.1% relative improvement** over the plain model:

```text
(54.85 - 38.86) / 38.86 × 100 ≈ 41.1%
```

The qualitative examples also show that the base model frequently begins with generic wording such as “Here are a few options,” while the fine-tuned model follows the requested format and directly produces Kannada or Kanglish text. chrF is useful for character-level similarity in morphologically rich and inconsistently Romanized text, but human review remains important because multiple translations can be valid.

## Supported tasks

1. **English → Romanized Kannada:** casual Kannada written with Latin characters.
2. **English → Kannada code-mixed text:** Kannada script with natural English mixing.
3. **English → native Kannada:** clear Kannada written fully in Kannada script.

## Dataset

The project uses the Kannada file from [AI4Bharat IndicCMix](https://huggingface.co/datasets/ai4bharat/IndicCMix).

| Stage | Records/examples |
|---|---:|
| Raw records | 104,798 |
| Clean records | 104,586 |
| Training records | 73,211 |
| Validation records | 15,688 |
| Test records | 15,687 |
| Expanded training examples | 219,633 |
| Expanded validation examples | 47,064 |
| Expanded test examples | 47,061 |

Records are Unicode-normalized, deduplicated, assigned stable SHA-256 source IDs, and split 70/15/15 by source group. Grouped splitting prevents translations derived from the same source sentence from appearing in both training and testing.

## Model and training

- **Checkpoint:** `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit`
- **Framework:** Unsloth `FastModel`
- **Method:** QLoRA / PEFT
- **LoRA rank and alpha:** 16 / 16
- **Training steps:** 1,000
- **Sequence length:** 1,024 tokens
- **Micro-batch size:** 1
- **Gradient accumulation:** 8
- **Optimizer:** 8-bit AdamW
- **Learning rate:** `2e-4`
- **Schedule:** cosine with 5% warmup
- **Loss:** assistant-response tokens only
- **Target hardware:** Google Colab Tesla T4

The training flow is:

```text
IndicCMix Kannada data
  → normalization and deduplication
  → leakage-safe grouped split
  → three task examples per record
  → Gemma chat-template formatting
  → response-only supervised loss
  → LoRA gradient updates
  → saved adapter
  → base-versus-fine-tuned evaluation
```

## Technology stack

| Purpose | Technology |
|---|---|
| Language | Python |
| Notebook runtime | Google Colab / Jupyter |
| GPU computation | PyTorch and CUDA |
| Base model and tokenizer | Hugging Face Transformers |
| Efficient model loading/training | Unsloth |
| Adapter training | PEFT LoRA |
| Supervised trainer | TRL `SFTTrainer` |
| Quantization and optimizer | bitsandbytes |
| Dataset loading | Hugging Face Datasets |
| Evaluation | SacreBLEU chrF and pandas |
| Demo interface | Gradio |

## Repository structure

```text
CodeMixBridge/
├── configs/                Training, inference, and evaluation settings
├── docs/                   Architecture, dataset, model, and evaluation docs
├── notebooks/              End-to-end Colab training notebook
├── scripts/                Dataset, training, evaluation, and app commands
├── src/                    Reusable Python implementation
├── tests/                  Automated unit tests
├── artifacts/              Split manifest and adapter output directory
└── results/                Metrics and human-review templates
```

## Quick start

```bash
git clone https://github.com/pavanreddyx7/CodeMixBridge.git
cd CodeMixBridge
pip install -r requirements.txt
python -m pytest -q
```

For full training, open [`notebooks/CodeMixBridge_Gemma_Training.ipynb`](notebooks/CodeMixBridge_Gemma_Training.ipynb) in Google Colab, select a T4 GPU, add `HF_TOKEN` to Colab Secrets, and run the notebook from top to bottom.

Launch the local Gradio application with:

```bash
python scripts/launch_app.py
```

## Evaluation protocol

- Model A uses the base Gemma checkpoint with the LoRA adapter disabled.
- Model B uses the same checkpoint with the trained LoRA adapter active.
- Both models receive identical prompts and held-out examples.
- Generation is deterministic with `do_sample=False`.
- The benchmark reports overall and per-task chrF, output-change counts, and qualitative examples.
- Human review scores meaning preservation, naturalness, and script compliance.

## License and attribution

- Code: [Apache License 2.0](LICENSE)
- Dataset: [AI4Bharat IndicCMix](https://huggingface.co/datasets/ai4bharat/IndicCMix)
- Base model: [Google Gemma](https://ai.google.dev/gemma)
- Fine-tuning: [Unsloth](https://github.com/unslothai/unsloth)
