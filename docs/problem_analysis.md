# CodeMixBridge: Official Hackathon Problem Analysis & Requirements Traceability

## 1. Overview
- **Hackathon:** Hacktoberfest Hack Day (Theme 3: Fine-Tuning Gemma)
- **Problem Statement:** 3A — Teach Gemma to Understand Code-Mixed Regional Languages
- **Target Language:** Kannada (and Kannada-English code-mixing: Kanglish)
- **Project Name:** CodeMixBridge

---

## 2. Problem Context & Objective
In multilingual everyday digital communication, Kannada speakers frequently mix Kannada and English:
1. **Casual Romanized Kannada (Kanglish):** E.g., *"Naale exam ide, nann jothe study madthiya?"* (Transliterated using Latin script with English and Kannada loanwords).
2. **Native Script Code-Mixed:** Kannada script interspersed with English technical/informal words (e.g., *"ನಾಳೆ exam ಇದೆ, ನನ್ನ ಜೊತೆ study ಮಾಡ್ತೀಯಾ?"*).
3. **Full Native Script:** Pure formal Kannada translation (e.g., *"ನಾಳೆ ಪರೀಕ್ಷೆ ಇದೆ, ನನ್ನೊಂದಿಗೆ ಅಧ್ಯಯನ ಮಾಡುತ್ತೀರಾ?"*).

Standard generic LLMs struggle with:
- Hallucinated translations when presented with informal Romanized spellings.
- Unnatural word-for-word transliteration that violates Kannada colloquial syntax.
- Script confusion (switching to Hindi/Devanagari or mixing scripts erratically).
- Over-explaining or verbose outputs when a direct chat-style conversion is requested.

**Primary Objective:**
Determine whether Supervised Fine-Tuning (SFT) using Parameter-Efficient LoRA on **Google Gemma 4 E2B** produces measurable, statistically significant gains in Kannada code-mixed translation fidelity, grammatical naturalness, and script accuracy on unseen test data compared to the un-fine-tuned base model.

---

## 3. Official Constraints & Model Architecture Rules

1. **Two-Model Architecture:**
   - **Model A: Generic Gemma:** Base `unsloth/gemma-4-E2B-it`, 4-bit quantized, no LoRA adapter, standard prompt, deterministic sampling (`do_sample=False`).
   - **Model B: Fine-Tuned Gemma:** Identical base checkpoint `unsloth/gemma-4-E2B-it`, active Kannada LoRA adapter, identical standard prompt, identical generation parameters.
   - *Constraint:* Exactly two models. No third prompted configuration permitted in the user application or primary comparison. Single base model architecture with runtime adapter switching (`model.disable_adapter()`).

2. **Dataset:**
   - Dataset repository: `ai4bharat/IndicCMix` on Hugging Face.
   - File: `ka.parquet` (Kannada subset).
   - Expected columns: `english`, `native_script_codemixed`, `full_native_script`, `romanized_casual`. (The `audio` column is omitted from text training).
   - Gated Access: Requires Hugging Face account with access permission agreed and `HF_TOKEN` read from environment or Colab secrets.

3. **Hardware & Environment Constraints:**
   - Platform: Google Colab with free T4 GPU (16 GB VRAM).
   - Training Framework: `Unsloth` (using `FastModel`, `train_on_responses_only`, 4-bit QLoRA with `bitsandbytes`).
   - Target Sequence Length: 1024 tokens.
   - Adapter Rank ($r$): 8 (with optional comparative validation experiment at rank 16).

4. **Evaluation Standard:**
   - Primary Metric: **SacreBLEU corpus chrF** (`sacrebleu.corpus_chrf`).
   - Rationale: Character n-gram F-score handles phonetic spelling variations in informal Romanized Kannada (e.g., *madtidiya* vs *madtidya*) without severe lexical penalties.
   - Additional Metrics: Truncation rate, empty output rate, median inference latency.
   - Human Evaluation: Structured blind scoring (1–5 scale for Meaning, Naturalness, Style Compliance).

---

## 4. Requirements Traceability Matrix

| Requirement ID | Requirement Description | Implementation Component | Verification Method | Status |
|---|---|---|---|---|
| **REQ-01** | Focus on Kannada code-mixed language conversion | `src/data.py`, `src/prompts.py` | Unit tests on `ka.parquet` schema & prompt structure | Implemented |
| **REQ-02** | Base checkpoint: `unsloth/gemma-4-E2B-it` (4-bit on T4 GPU) | `src/model_loader.py`, `configs/training.json` | Model loader tests & memory profile | Implemented |
| **REQ-03** | Exactly two model configurations (Generic vs Fine-Tuned) | `src/inference.py`, `src/app.py` | Runtime adapter toggle check & unit test | Implemented |
| **REQ-04** | AI4Bharat IndicCMix (`ka.parquet`) pipeline | `src/data.py`, `scripts/prepare_dataset.py` | Automated schema validation & cleaning tests | Implemented |
| **REQ-05** | Leakage-free group-aware splitting (70/15/15) by English text | `src/data.py`, `tests/test_data.py` | Split overlap assertions (0 common sources) | Implemented |
| **REQ-06** | Three primary tasks + reverse translation tasks | `src/prompts.py`, `tests/test_prompts.py` | Prompt formatting & round-trip tests | Implemented |
| **REQ-07** | Response-only training loss masking | `src/training.py`, Colab Notebook | Token loss mask verification on assistant tokens | Implemented |
| **REQ-08** | Evaluation using SacreBLEU corpus chrF | `src/metrics.py`, `src/evaluation.py` | Tests with known reference/hypothesis pairs | Implemented |
| **REQ-09** | Dual-panel Gradio UI (Navy/Cyan, Dark Theme) | `src/app.py`, `scripts/launch_app.py` | UI rendering & input validation test | Implemented |
| **REQ-10** | Reproducible Colab Notebook (`CodeMixBridge_Gemma_Training.ipynb`) | `notebooks/CodeMixBridge_Gemma_Training.ipynb` | JSON notebook validation & Colab execution | Implemented |
| **REQ-11** | Human evaluation rubric & Error analysis | `results/human_review.csv`, `results/error_analysis.md` | Human review guidelines & qualitative review | Implemented |
| **REQ-12** | Full documentation and presentation script | `docs/`, `README.md`, `docs/demo_script.md` | Complete 5-minute showcase pitch outline | Implemented |

---

## 5. Technical Limitations & Safeguards
- **Colab GPU Quotas:** T4 sessions can timeout; training is designed to finish in under 10 minutes (60–100 steps with gradient accumulation = 4).
- **Gated Hugging Face Token:** Tokens must never be hard-coded; read exclusively via `os.environ["HF_TOKEN"]` or `google.colab.userdata.get("HF_TOKEN")`.
- **Adapter Switching Race Conditions:** Sequential lock in `inference.py` ensures thread safety when switching LoRA state during interactive Gradio evaluation.
