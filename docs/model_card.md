# Model Card: CodeMixBridge Gemma 4 Kannada LoRA

## 1. Model Details
- **Model Name:** CodeMixBridge (Gemma 4 Kannada LoRA)
- **Base Foundation Model:** Google `gemma-4-E2B-it` (instruction-tuned, 4-bit quantized)
- **Developer:** CodeMixBridge Team
- **Fine-Tuning Framework:** Unsloth & Hugging Face TRL (`SFTTrainer`)
- **Adaptation Technique:** QLoRA (Rank 8, Alpha 8, Response-Only loss masking)
- **Target Language:** Kannada (`kan`) and Kannada-English code-mixing (*Kanglish*)
- **License:** Apache 2.0 (compatible with Gemma open-weights terms)

---

## 2. Intended Use & Application Scope
- **Primary Use:** Converting English messages into natural, colloquial Romanized Kannada (Kanglish) for messaging apps, social communication, and chatbots.
- **Secondary Use:** Transliterating code-mixed text into Kannada script and formal native Kannada.
- **Reverse Use:** De-mixing informal Romanized Kannada into standard English for English-speaking customer support or analytics.
- **Out-of-Scope:** Not intended for legal contracts, clinical medical translations, or formal governmental gazette generation without human verification.

---

## 3. Training Parameters & Environment
- **Platform:** Google Colab Free T4 GPU (16 GB VRAM)
- **Optimizer:** 8-bit AdamW (`adamw_8bit`)
- **Learning Rate:** $2 \times 10^{-4}$ with Cosine schedule
- **Batch Size:** 1 per device with 4 gradient accumulation steps (effective batch size = 4)
- **Sequence Length:** 1024 tokens
- **Training Steps:** 1,000 steps

---

## 4. Evaluation Protocol
- **Benchmark Metric:** SacreBLEU corpus chrF (`sacrebleu.corpus_chrf`)
- **Test Set:** 15% held-out IndicCMix Kannada partition (leakage-free)
- **Comparison Baseline:** Un-fine-tuned `unsloth/gemma-4-E2B-it` evaluated with identical prompts and deterministic generation settings (`do_sample=False`).
