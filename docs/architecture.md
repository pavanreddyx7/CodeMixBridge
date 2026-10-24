# CodeMixBridge: System & Model Architecture

## 1. System Overview
CodeMixBridge implements a parameter-efficient, low-latency Kannada-English conversion system based on Google Gemma 4 and Low-Rank Adaptation (LoRA).

```
                      +---------------------------------------+
                      |       Input Message + Task Type       |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         Standardized Prompt           |
                      |          (src/prompts.py)             |
                      +---------------------------------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
                      v                                       v
         +-------------------------+             +-------------------------+
         |     MODEL A: Base       |             |   MODEL B: Fine-Tuned   |
         |    Gemma 4 E2B (4-bit)  |             |   Gemma 4 E2B + LoRA    |
         | (LoRA Adapter Disabled) |             | (Active Kannada Adapter)|
         +-------------------------+             +-------------------------+
                      |                                       |
                      v                                       v
         +-------------------------+             +-------------------------+
         | Generic Baseline Output |             | Code-Mixed Fluent Output|
         +-------------------------+             +-------------------------+
                      \                                       /
                       \                                     /
                        v                                   v
                      +---------------------------------------+
                      |     Dual-Panel Gradio Comparison UI   |
                      |            (src/app.py)               |
                      +---------------------------------------+
```

---

## 2. Model Architecture & Checkpoint Specifications

- **Base Model Checkpoint:** `unsloth/gemma-4-E2B-it`
  - Effective Parameters: ~2 Billion
  - Architecture: Transformer Decoder with Multi-Query Attention & GeGLU activations
  - Vocabulary Size: 256,000 tokens (rich Indic subword coverage)
  - Quantization: 4-bit NormalFloat (NF4) via `bitsandbytes`
  - Target Context Length: 1024 tokens

- **Parameter-Efficient Adaptation (LoRA / QLoRA):**
  - **Rank ($r$):** 8
  - **Scaling Factor ($\\alpha$):** 8
  - **Dropout:** 0.0 (prevents training under-utilization on small corpora)
  - **Target Modules:**
    - `q_proj`, `k_proj`, `v_proj`, `o_proj` (Attention mechanisms)
    - `gate_proj`, `up_proj`, `down_proj` (MLP feed-forward blocks)
  - **Trainable Parameters:** ~0.2% of total model parameters (<10 MB adapter size)

---

## 3. Safe Runtime Adapter Switching Protocol
To guarantee fair, identical runtime conditions without consuming double VRAM, CodeMixBridge hosts a single base model in GPU memory:
1. When generating for **Generic Gemma**:
   ```python
   with model.disable_adapter():
       outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
   ```
2. When generating for **Fine-Tuned Gemma**:
   ```python
   outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
   ```
3. A module-level thread lock (`_INFERENCE_LOCK`) ensures that concurrent web requests cannot alter adapter state midway through generation.

---

## 4. Hardware Resource Budget (Google Colab Free T4)
- **Base Model in 4-bit VRAM:** ~2.8 GB
- **LoRA Adapter Weights in VRAM:** ~0.05 GB
- **KV Cache for 1024 tokens:** ~0.4 GB
- **Optimizer States (8-bit AdamW) during Training:** ~0.8 GB
- **Peak Training VRAM:** ~4.2 GB (well below the 15.0 GB limit of Colab T4)
- **Inference Latency:** ~250–450 ms per sentence
