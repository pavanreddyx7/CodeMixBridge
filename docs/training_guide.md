# Training Guide: Running CodeMixBridge on Google Colab (T4 GPU)

This guide provides step-by-step instructions for reproducing the LoRA fine-tuning and benchmark evaluation using Google Colab's free T4 GPU.

---

## 1. Prerequisites & Credentials

1. **Google Account:** Sign in to [Google Colab](https://colab.research.google.com).
2. **Hugging Face Account & Gated Access:**
   - Log in at [Hugging Face](https://huggingface.co).
   - Visit [ai4bharat/IndicCMix](https://huggingface.co/datasets/ai4bharat/IndicCMix) and click **"Agree and access repository"**.
   - Create a User Access Token with `read` permission at [hf.co/settings/tokens](https://huggingface.co/settings/tokens).
3. **Colab Secrets:**
   - In your Colab notebook, click the **Key icon** (Secrets) on the left sidebar.
   - Add a new secret named `HF_TOKEN` with your access token.
   - Toggle **"Notebook access"** to ON.

---

## 2. Opening and Configuring the Notebook

1. Open `notebooks/CodeMixBridge_Gemma_Training.ipynb` in Google Colab:
   - Either upload the `.ipynb` file from this repository, or clone from GitHub.
2. Select the GPU Runtime:
   - Click **Runtime $\\rightarrow$ Change runtime type**.
   - Select **T4 GPU** under Hardware accelerator.
   - Click **Save**.

---

## 3. Step-by-Step Notebook Execution

Run the cells from top to bottom (**Shift + Enter**):

- **Section 1–2 (Environment & Dependencies):**
  - Verifies that `torch.cuda.is_available()` returns `True` and displays device name.
  - Installs Unsloth, Transformers, PEFT, TRL, SacreBLEU, and Gradio (~2–3 minutes).
- **Section 3–4 (Authentication & Data Loading):**
  - Reads `HF_TOKEN` from secrets and downloads `ka.parquet`.
- **Section 5–7 (Inspection, Normalization & Splitting):**
  - Applies NFC Unicode cleaning, generates deterministic source IDs, and splits data 70/15/15.
- **Section 8–10 (Base Model & Baseline Benchmark):**
  - Loads 4-bit `unsloth/gemma-4-E2B-it`.
  - Runs baseline evaluation on the held-out test subset and computes baseline chrF score.
- **Section 11–13 (LoRA Fine-Tuning):**
  - Attaches LoRA adapters ($r=16$).
  - Trains for 1,000 steps using `SFTTrainer` with `train_on_responses_only`.
- **Section 14–17 (Adapter Export & Final Comparison):**
  - Saves adapter weights to `gemma_4_kannada_lora/` and creates `gemma_4_kannada_lora.zip`.
  - Evaluates fine-tuned model and outputs the side-by-side comparison table.
  - Generates `metrics.csv`.
- **Section 18 (Gradio Application Launch):**
  - Launches the interactive dual-pane app with a public `gradio.live` link for live judging!

---

## 4. Troubleshooting & FAQ

| Symptom | Cause | Solution |
|---|---|---|
| `RuntimeError: No GPU detected` | CPU runtime active | Runtime $\\rightarrow$ Change runtime type $\\rightarrow$ T4 GPU. |
| `GatedRepoError` or 401 Unauthorized | HF_TOKEN missing or terms unaccepted | Accept dataset terms at hf.co/datasets/ai4bharat/IndicCMix and verify HF_TOKEN secret. |
| `CUDA out of memory` | KV cache exhaustion | Keep `max_new_tokens <= 128` and `per_device_train_batch_size = 1`. |
| Session Disconnected | Inactivity timeout | Click reconnect; downloaded adapters remain in local disk if saved. |
