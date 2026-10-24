# 🌉 CodeMixBridge: Kannada-English Language Conversion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Unsloth](https://img.shields.io/badge/Fine--Tuning-Unsloth%20QLoRA-orange.svg)](https://github.com/unslothai/unsloth)
[![Gemma 4](https://img.shields.io/badge/Base%20Model-Google%20Gemma%204%20E2B-red.svg)](https://ai.google.dev/gemma)
[![Hardware: Colab T4](https://img.shields.io/badge/Hardware-Google%20Colab%20T4-purple.svg)](https://colab.research.google.com)

> **Hacktoberfest Hack Day — Theme 3: Fine-Tuning Gemma (Problem Statement 3A)**  
> **Tagline:** Bridging Kannada and English through AI — Powered by Google Gemma & LoRA Fine-Tuning.

---

## 📌 1. Problem Statement & Objective
In multilingual everyday digital communication, millions of Kannada speakers naturally mix English with Kannada:
1. **Casual Romanized Kannada (*Kanglish*):** E.g., *"Naale exam mugida mele cafeteria ge hogona barthiya?"*
2. **Native Script Code-Mixed:** E.g., *"ನಾಳೆ exam ಮುಗಿದ ಮೇಲೆ cafeteria ಗೆ ಹೋಗೋಣ ಬರ್ತೀಯಾ?"*
3. **Full Native Script:** E.g., *"ನಾಳೆ ಪರೀಕ್ಷೆ ಮುಗಿದ ನಂತರ ಉಪಾಹಾರ ಗೃಹಕ್ಕೆ ಹೋಗೋಣ ಬರುತ್ತೀರಾ?"*

Generic base LLMs struggle with informal regional transliteration: they produce literal translations that violate Dravidian SOV syntax, confuse Kannada with Hindi, or emit wordy conversational preambles.

**Project Objective:**  
**CodeMixBridge** determines whether supervised LoRA fine-tuning of Google's **Gemma 4 (E2B)** on the **AI4Bharat IndicCMix** dataset measurably improves conversion accuracy, colloquial fluency, and script fidelity on unseen test data compared to the un-fine-tuned base model.

---

## ⚖️ 2. Two-Model Architecture Protocol

In strict compliance with hackathon guidelines, CodeMixBridge benchmarks **exactly two model configurations** sharing one underlying checkpoint:

- **Model A: Generic Gemma (Base):**
  - Original `unsloth/gemma-4-E2B-it` instruction-tuned checkpoint
  - LoRA adapter disabled (`model.disable_adapter()`)
  - Standard conversion prompt, deterministic decoding (`do_sample=False`)
- **Model B: Fine-Tuned Gemma (LoRA):**
  - Identical base checkpoint
  - Active Kannada LoRA adapter ($r=16, \alpha=16$)
  - Identical standard conversion prompt, identical generation parameters

---

## 📂 3. Repository Structure

```
CodeMixBridge/
├── README.md                           # Main project documentation
├── LICENSE                             # Apache 2.0 License
├── .gitignore                          # Git tracking exclusions
├── .env.example                        # Required credentials template
├── requirements.txt                    # Local dependencies (testing, UI, data)
├── requirements-colab.txt              # Colab GPU dependencies (Unsloth, TRL, PEFT)
├── configs/
│   ├── training.json                   # LoRA hyperparameters & training settings
│   ├── inference.json                  # Generation parameters
│   └── evaluation.json                 # Benchmark metrics configuration
├── notebooks/
│   └── CodeMixBridge_Gemma_Training.ipynb  # End-to-end reproducible Colab notebook
├── src/
│   ├── __init__.py
│   ├── data.py                         # Data normalization, grouping, zero-leakage splits
│   ├── prompts.py                      # Single source of truth for prompt templates
│   ├── training.py                     # Unsloth LoRA & SFTTrainer pipeline
│   ├── model_loader.py                 # Gemma 4 4-bit loader & GPU verification
│   ├── inference.py                    # Thread-safe dual-model inference engine
│   ├── evaluation.py                   # Automated batch evaluation & CSV reports
│   ├── metrics.py                      # SacreBLEU corpus & sentence chrF calculation
│   └── app.py                          # Dual-panel dark navy/cyan Gradio web application
├── scripts/
│   ├── inspect_dataset.py              # Dataset schema and sample inspector
│   ├── prepare_dataset.py              # Data cleaning and split manifest generator
│   ├── run_baseline.py                 # Generic Gemma baseline evaluation
│   ├── train_lora.py                   # LoRA fine-tuning execution script
│   ├── evaluate_models.py              # Comparative benchmark evaluator
│   └── launch_app.py                   # Gradio web app launcher
├── tests/
│   ├── run_tests.py                    # Self-contained test runner
│   ├── test_data.py                    # Data validation & leakage tests
│   ├── test_prompts.py                 # Prompt formatting tests
│   ├── test_metrics.py                 # chrF score & heuristic tests
│   └── test_inference.py              # Mock adapter switching tests
├── artifacts/
│   ├── split_manifest.json             # Audit record of dataset partitions
│   └── adapters/
│       └── .gitkeep                    # Storage for fine-tuned LoRA weights
├── results/
│   ├── metrics.csv                     # Measured comparative benchmark results
│   ├── human_review.csv                # Blind evaluation sheet with 1-5 rubrics
│   └── error_analysis.md               # Detailed linguistic error taxonomy & cases
└── docs/
    ├── problem_analysis.md             # Official requirements traceability
    ├── architecture.md                 # System components & memory budget
    ├── dataset_card.md                 # Dataset documentation
    ├── model_card.md                   # Model specifications & training details
    ├── training_guide.md               # Step-by-step Colab reproduction guide
    ├── evaluation_report.md            # Benchmark methodology & chrF rationale
    └── demo_script.md                  # 5-minute showcase pitch script
```

---

## 🛠️ 4. Quick Start & Setup

### Local Setup (CPU Testing & Web UI Preview)
```bash
git clone https://github.com/mithun-srinivas/HacktoberFest-HackDay.git
cd HacktoberFest-HackDay/CodeMixBridge

# Install dependencies
pip install -r requirements.txt

# Run automated unit tests
python tests/run_tests.py

# Launch Gradio interface (Preview Mode)
python scripts/launch_app.py
```

### Full GPU Training (Google Colab T4)
1. Open `notebooks/CodeMixBridge_Gemma_Training.ipynb` in [Google Colab](https://colab.research.google.com).
2. Set Runtime to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
3. Add your `HF_TOKEN` in Colab Secrets (Left sidebar key icon).
4. Run all cells (**Shift + Enter**).

---

## 🔬 5. Evaluation & Benchmark Methodology

Text generation in informal code-mixed languages is evaluated using **SacreBLEU corpus chrF** (`sacrebleu.corpus_chrf`). Character $n$-gram F-scores reward valid colloquial transliteration variations (e.g. *barthiya* vs *bartha idiya*) without the severe penalties of word-level BLEU.

| Metric | Generic Gemma (Base) | Fine-Tuned Gemma (LoRA) | Net Change |
|---|---|---|---|
| **Romanized chrF (Kanglish)** | *Per-task result pending* | *Per-task result pending* | — |
| **Code-Mixed chrF (Native)** | *Per-task result pending* | *Per-task result pending* | — |
| **Full Native chrF** | *Per-task result pending* | *Per-task result pending* | — |
| **Overall chrF** | **38.86** | **54.85** | **+15.99** |
| **Empty Output Rate (%)** | 0.0% | 0.0% | 0.0% |
| **Median Latency** | ~0.38s | ~0.39s | +0.01s |

The measured overall chrF improved by **15.99 points**, from **38.86** to **54.85** (approximately **41.1% relative improvement**). All scores in `results/metrics.csv` are generated directly from notebook execution; per-task aggregate scores will be added after they are exported from the benchmark notebook.

---

## 📱 6. Interactive Gradio Interface

CodeMixBridge features a modern dark navy (`#070B19`) and cyan (`#00F0FF`) Gradio interface:
- **Dual Model Panels:** Displays Generic Gemma on the left and Fine-Tuned Gemma on the right.
- **Task Selector:** English $\\rightarrow$ Romanized, English $\\rightarrow$ Code-Mixed, English $\\rightarrow$ Full Kannada.
- **Copy Buttons & Pre-Loaded Scenarios:** Fast demonstration of student conversations and official announcements.

---

## 📄 7. License & Citations
- **Code:** Licensed under Apache 2.0 (`LICENSE`).
- **Dataset:** [AI4Bharat IndicCMix](https://huggingface.co/datasets/ai4bharat/IndicCMix).
- **Base Model:** [Google Gemma 4 E2B](https://ai.google.dev/gemma).
- **Fine-Tuning:** [Unsloth](https://github.com/unslothai/unsloth).
