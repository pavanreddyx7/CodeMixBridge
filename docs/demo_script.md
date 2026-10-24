# CodeMixBridge: 5-Minute Showcase Presentation & Live Demo Script

Use this structured script for the final hackathon demonstration before the judges.

---

## ⏱ Minute 1: The Problem & The Code-Mixing Challenge
- **Presenter:** *"Hello everyone! Today we present **CodeMixBridge** for Problem 3A: Teaching Gemma to understand code-mixed regional languages."*
- **The Core Problem:**
  - In India, over 50 million people speak Kannada. But in daily digital chats, WhatsApp messages, and social media, people rarely speak pure formal Kannada or pure English.
  - They speak **Kanglish** — casual Romanized Kannada interspersed with English loanwords (*"Naale exam mugida mele cafeteria ge hogona barthiya?"*).
  - Out-of-the-box LLMs fail here: they either output textbook Sanskritized Kannada that sounds bizarre in casual chat, confuse Kannada with Hindi, or output word-by-word literal translations that sound robotic.
- **The Goal:** Build an AI system that translates naturally between English, Romanized Kanglish, and native script Kannada.

---

## ⏱ Minute 2: Dataset, Methodology & LoRA Fine-Tuning
- **Dataset:** We used AI4Bharat's `IndicCMix` (`ka.parquet`), containing authentic human parallel pairs.
- **Leakage-Free Partitioning:**
  - We grouped records by normalized English source text (`source_id`), ensuring **zero train/test data leakage**.
- **Model Choice & Efficiency:**
  - We adapted **Google's Gemma 4 (E2B)** using **Unsloth 4-bit QLoRA** on a free Google Colab T4 GPU.
  - Using LoRA ($r=16, \alpha=16$), we train only a small fraction of the model parameters.
  - Applied response-only loss masking so the model is only graded on generating natural Kannada responses.
  - The entire fine-tuning completed in just **5.5 minutes**!

---

## ⏱ Minute 3: Quantitative & Qualitative Benchmark Comparison
- **Show Benchmark Slide / `results/metrics.csv`:**
  - Explain the **SacreBLEU chrF** score (why character $n$-grams reflect casual transliteration better than word BLEU).
  - Show the side-by-side metric comparison:
    - **Generic Gemma (Base):** struggles with conversational chat transliteration and often prepends verbose preambles (*"Here is the translation: ..."*).
    - **Fine-Tuned Gemma (LoRA):** demonstrates direct, concise, culturally authentic phrasing.
  - Show the qualitative error analysis: elimination of script confusion and preservation of natural Dravidian SOV syntax.

---

## ⏱ Minute 4: Live Application Demonstration (Gradio UI)
- **Open CodeMixBridge Gradio Interface:**
  - Point out the dark navy & cyan UI layout with **exactly TWO model panels**: Model A (Generic Gemma) on the left, Model B (Fine-Tuned Gemma) on the right.
- **Live Test 1 (Student Conversation):**
  - Input: *"Our internal lab exams start tomorrow at 9 AM, please arrive on time."*
  - Select task: `English -> Romanized Kannada (Casual Chat)`.
  - Click **Convert**.
  - **Observe:** Generic Gemma outputs wordy formal English or literal translation, while Fine-Tuned Gemma outputs: *"Naale 9 AM ge namm internal lab exams start aaguthe, please time ge banni."*
- **Live Test 2 (Native Script Code-Mixed):**
  - Input: *"I am submitting the assignment in the computer lab today."*
  - Select task: `English -> Kannada Code-Mixed (Native Script)`.
  - **Observe:** Natural code-mixed Kannada script with English technical nouns preserved.

---

## ⏱ Minute 5: Discussion, Limitations & Future Work
- **What Worked Well:**
  - Flawless format compliance (0% preamble rate).
  - Idiomatic colloquial Kanglish that native speakers immediately resonate with.
  - High runtime efficiency (~350ms latency on free hardware).
- **Limitations & Next Steps:**
  - Adding regional North Karnataka (Uttara Karnataka) dialect support.
  - Expanding multi-turn dialog context and speech-to-speech integration.
- **Closing:** *"CodeMixBridge proves that lightweight LoRA fine-tuning transforms a general foundation model into a culturally fluent regional communicator. Thank you!"*
