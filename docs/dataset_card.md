# Dataset Card: AI4Bharat IndicCMix (Kannada Subset)

## 1. Dataset Summary
- **Source:** AI4Bharat IndicCMix (`ai4bharat/IndicCMix` on Hugging Face)
- **Language:** Kannada (ISO 639-3: `kan`) and Kannada-English code-mixed (*Kanglish*)
- **Target File:** `ka.parquet`
- **Gated License:** Requires user agreement on Hugging Face Model Hub.
- **Modality:** Text (audio column removed for SFT text generation).

---

## 2. Dataset Schema & Fields

Each record contains parallel representations of a single conversational utterance:

| Field Name | Type | Description | Example |
|---|---|---|---|
| `english` | String | Reference English sentence | *"Are you coming to college tomorrow?"* |
| `native_script_codemixed` | String | Code-mixed Kannada written in Kannada script | *"ನಾಳೆ college ಗೆ ಬರ್ತೀಯಾ?"* |
| `full_native_script` | String | Formal, complete Kannada translation | *"ನೀವು ನಾಳೆ ಕಾಲೇಜಿಗೆ ಬರುತ್ತೀರಾ?"* |
| `romanized_casual` | String | Informal Romanized Kannada (Kanglish chat) | *"Naale college ge barthiya?"* |

---

## 3. Data Preprocessing & Leakage-Free Partitioning

1. **Unicode Normalization:** All text fields normalized to Unicode NFC, eliminating non-breaking spaces (`\u00a0`) and irregular whitespace.
2. **Deduplication:** Duplicate `(english, romanized_casual)` tuples purged (212 duplicate entries removed from 104,798 raw records, yielding 104,586 clean records).
3. **Deterministic Source Grouping:** A deterministic 12-char SHA256 identifier (`source_id`) is calculated from `normalize_text(english).lower()`.
4. **Group-Aware Splitting (Seed 3407):**
   - **Train Split (70%):** 73,211 records across 73,188 source groups (366,055 expanded task examples).
   - **Validation Split (15%):** 15,688 records across 15,683 source groups (78,440 expanded task examples).
   - **Held-Out Test Split (15%):** 15,687 records across 15,684 source groups (78,435 expanded task examples).
   - **Leakage Verification:** 0 overlapping `source_id` groups between train, val, and test sets. Tracked in `artifacts/split_manifest.json`.

---

## 4. Ethical Considerations
The dataset captures authentic, informal everyday communication. Annotations do not contain private personally identifiable information (PII). Dialectal representation primarily covers standard urban conversational Kannada.
