# CodeMixBridge: Linguistic & Error Analysis

## 1. Overview
This document analyzes the qualitative performance differences between **Generic Gemma 4 E2B (Model A)** and **Fine-Tuned Gemma 4 E2B LoRA (Model B)** across Kannada-English conversion tasks.

---

## 2. Qualitative Error Taxonomy

We categorize observed model errors into 8 linguistic and stylistic failure modes:

| Error Type | Description | Generic Gemma Vulnerability | Fine-Tuned Gemma Resolution |
|---|---|---|---|
| **E1: Script Confusion** | Emitting Hindi/Devanagari script instead of Kannada script when code-mixing is requested. | **High:** Generic models often bias towards Hindi for Indic code-mix prompts. | **Resolved:** Strongly conditioned on Kannada Unicode range `[U+0C80 - U+0CFF]`. |
| **E2: Verbose Chat / Over-Explanation** | Outputting conversational preambles (e.g., *"Here is the translation into casual Kannada: ..."*). | **Very High:** Generic instruction models tend to respond conversationally. | **Resolved:** Strict response-only training teaches direct output generation. |
| **E3: Literal Unnatural Transliteration** | Word-by-word phonetic mapping that violates Dravidian SOV (Subject-Object-Verb) grammar. | **High:** Produces awkward phrases that native Kannada speakers find unnatural. | **Resolved:** Learns colloquial phrasing from authentic IndicCMix conversational pairs. |
| **E4: Loss of Negation** | Dropping negative particles like *illa* (ಇಲ್ಲ), *beda* (ಬೇಡ), reversing sentence meaning. | **Moderate:** Can occasionally drop trailing negation in long clauses. | **Monitored:** Strict instruction prompt explicitly penalizes lost negation. |
| **E5: Named Entity Corruption** | Translating proper names literally (e.g., translating "May" as month vs name). | **Low to Moderate.** | **Preserved:** Retains English entities in Kanglish/Code-mixed output. |
| **E6: Numerical Value Drift** | Changing digits or quantifiers (e.g., changing "9 AM" to "morning"). | **Low.** | **Preserved:** Retains exact numerical specifications. |
| **E7: Incomplete / Truncated Generation** | Cutting off sentence before final verb auxiliary. | **Moderate** when output token budget is narrow. | **Resolved:** SFT enforces concise, self-contained single-sentence outputs. |
| **E8: Awkward Romanization Orthography** | Inconsistent phonetic spelling that obscures meaning (e.g., *aagolla* vs *aguvudilla*). | **High:** Produces rigid textbook transliterations rather than chat spelling. | **Resolved:** Learns authentic casual Romanized patterns (*sigona*, *barthiya*, *madtidira*). |

---

## 3. Concrete Comparative Case Studies

### Case 1: College Internal Announcement
- **Source (English):** *"Our internal lab exams start tomorrow at 9 AM, please arrive on time."*
- **Target Task:** `english_to_romanized` (Kanglish)
- **Generic Gemma (Base):**
  > *"Tomorrow our lab internal examination will start at 9:00 AM, please you must come on time."*
  > *(Error: Kept entirely in English or gave literal word substitution without Kannada colloquial vocabulary).*
- **Fine-Tuned Gemma (LoRA):**
  > *"Naale 9 AM ge namm internal lab exams start aaguthe, please time ge banni."*
  > *(Analysis: Natural Kanglish phrasing, correct SOV order, loanword placement, and polite verb ending 'banni').*

---

### Case 2: Code-Mixed Assignment Submission
- **Source (English):** *"I am submitting the assignment in the computer lab today."*
- **Target Task:** `english_to_codemixed`
- **Generic Gemma (Base):**
  > *"ನಾನು ಇಂದು ಕಂಪ್ಯೂಟರ್ ಲ್ಯಾಬ್‌ನಲ್ಲಿ ಅಸೈನ್‌ಮೆಂಟ್ ಅನ್ನು ಸಲ್ಲಿಸುತ್ತಿದ್ದೇನೆ."*
  > *(Error: Emitted overly formal, pure textbook Kannada rather than casual code-mixed conversational Kannada).*
- **Fine-Tuned Gemma (LoRA):**
  > *"Naanu ivathu computer lab nalli assignment submit madtidini."* / *"ನಾನು ಇವತ್ತು computer lab ನಲ್ಲಿ assignment submit ಮಾಡ್ತಿದ್ದೀನಿ."*
  > *(Analysis: Retained technical English words naturally interspersed in Kannada script).*

---

### Case 3: Formal Public Notice
- **Source (English):** *"The college library will remain closed on Sunday for maintenance."*
- **Target Task:** `english_to_native` (Formal Script)
- **Generic Gemma (Base):**
  > *"ರವಿವಾರದಂದು ಕಾಲೇಜು ಲೈಬ್ರರಿ ಬಂದ್ ಆಗಿರುತ್ತದೆ..."* (Mixes informal slang 'ಬಂದ್').
- **Fine-Tuned Gemma (LoRA):**
  > *"ನಿರ್ವಹಣಾ ಕಾರ್ಯದ ಕಾರಣ ಭಾನುವಾರ ಕಾಲೇಜು ಗ್ರಂಥಾಲಯ ಮುಚ್ಚಿರುತ್ತದೆ."*
  > *(Analysis: Proper formal native vocabulary 'ಗ್ರಂಥಾಲಯ' and 'ನಿರ್ವಹಣಾ ಕಾರ್ಯ').*

---

## 4. Where Fine-Tuning Succeeded vs Limitations
- **Major Successes:**
  - Complete elimination of conversational preambles.
  - Authentic colloquial Romanized spellings matching real messaging habits.
  - Consistent script adherence (never mixes Hindi or Arabic numerals inappropriately).
- **Remaining Challenges & Future Work:**
  - Complex nested subordinate clauses occasionally lose secondary adjectives.
  - Regional dialectal variations (e.g., North Karnataka Dharwad style vs South Karnataka Bengaluru/Mysuru style) — model predominantly reflects standard southern casual code-mixing.
