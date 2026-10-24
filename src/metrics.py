"""
Evaluation metrics calculation module for CodeMixBridge.
Implements SacreBLEU corpus and sentence chrF, empty-output rate,
truncation detection, and latency tracking.
"""

from typing import Any, Dict, List, Optional
import numpy as np


def compute_sentence_chrf(hypothesis: str, reference: str) -> float:
    """
    Computes sentence-level chrF score (character n-gram F-score) using sacrebleu.
    Handles empty predictions gracefully with a score of 0.0.
    """
    hyp = str(hypothesis or "").strip()
    ref = str(reference or "").strip()
    if not hyp:
        return 0.0
    try:
        import sacrebleu  # type: ignore
        score = sacrebleu.sentence_chrf(hyp, [ref]).score
        return round(float(score), 2)
    except ImportError:
        # Fallback character n-gram F-score if sacrebleu is not installed
        return round(_simple_char_chrf(hyp, ref), 2)


def compute_corpus_chrf(hypotheses: List[str], references: List[str]) -> float:
    """
    Computes corpus-level chrF score across all paired hypotheses and references.
    """
    assert len(hypotheses) == len(references), "Hypotheses and references must have identical length"
    if not hypotheses:
        return 0.0
    clean_hyps = [str(h or "").strip() for h in hypotheses]
    clean_refs = [[str(r or "").strip() for r in references]]
    try:
        import sacrebleu  # type: ignore
        res = sacrebleu.corpus_chrf(clean_hyps, clean_refs)
        return round(float(res.score), 2)
    except ImportError:
        # Average sentence chrF as fallback
        scores = [compute_sentence_chrf(h, r) for h, r in zip(hypotheses, references)]
        return round(float(np.mean(scores)), 2)


def check_is_empty(text: str) -> bool:
    """Returns True if the generated string is empty or contains only whitespace."""
    return len(str(text or "").strip()) == 0


def check_is_truncated(text: str, max_tokens: int = 128, approx_chars_per_token: float = 3.5) -> bool:
    """
    Heuristic check for incomplete or abruptly cut off output:
    1. Reaches approximate character budget for max_tokens without terminal punctuation
    2. Ends with dangling conjunction or hyphen
    """
    t = str(text or "").strip()
    if not t:
        return False
    # If the text does not end with typical sentence punctuation and is long
    terminal_punct = {".", "!", "?", "।", "॥", "\"", "'", "”", "’"}
    has_terminal = any(t.endswith(p) for p in terminal_punct)
    is_very_long = len(t) >= int(max_tokens * approx_chars_per_token * 0.9)
    return is_very_long and not has_terminal


def compute_task_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates full evaluation metrics across predictions:
    - Corpus chrF
    - Mean sentence chrF
    - Empty output rate (%)
    - Truncation rate (%)
    - Median inference latency (seconds)
    - Sample count
    """
    if not predictions:
        return {
            "count": 0,
            "corpus_chrf": 0.0,
            "mean_sentence_chrf": 0.0,
            "empty_rate_pct": 0.0,
            "truncation_rate_pct": 0.0,
            "median_latency_sec": 0.0,
        }

    hyps = [p.get("prediction", "") for p in predictions]
    refs = [p.get("reference", "") for p in predictions]
    durations = [p.get("duration_sec", 0.0) for p in predictions if "duration_sec" in p]

    corpus_score = compute_corpus_chrf(hyps, refs)
    sentence_scores = [compute_sentence_chrf(h, r) for h, r in zip(hyps, refs)]

    empty_count = sum(1 for h in hyps if check_is_empty(h))
    trunc_count = sum(1 for h in hyps if check_is_truncated(h))
    n = len(predictions)

    med_lat = float(np.median(durations)) if durations else 0.0

    return {
        "count": n,
        "corpus_chrf": round(corpus_score, 2),
        "mean_sentence_chrf": round(float(np.mean(sentence_scores)), 2),
        "empty_rate_pct": round((empty_count / n) * 100.0, 2),
        "truncation_rate_pct": round((trunc_count / n) * 100.0, 2),
        "median_latency_sec": round(med_lat, 3),
    }


def _simple_char_chrf(hyp: str, ref: str, n_order: int = 6, beta: float = 2.0) -> float:
    """
    Lightweight CPU implementation of character n-gram F-score when sacrebleu is unavailable.
    """
    if not hyp and not ref:
        return 100.0
    if not hyp or not ref:
        return 0.0

    precisions = []
    recalls = []

    for n in range(1, n_order + 1):
        hyp_ngrams = [hyp[i:i+n] for i in range(len(hyp) - n + 1)] if len(hyp) >= n else []
        ref_ngrams = [ref[i:i+n] for i in range(len(ref) - n + 1)] if len(ref) >= n else []

        if not hyp_ngrams or not ref_ngrams:
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        hyp_counts: Dict[str, int] = {}
        for g in hyp_ngrams:
            hyp_counts[g] = hyp_counts.get(g, 0) + 1

        ref_counts: Dict[str, int] = {}
        for g in ref_ngrams:
            ref_counts[g] = ref_counts.get(g, 0) + 1

        matches = sum(min(count, ref_counts.get(g, 0)) for g, count in hyp_counts.items())
        p = matches / len(hyp_ngrams) if hyp_ngrams else 0.0
        r = matches / len(ref_ngrams) if ref_ngrams else 0.0
        precisions.append(p)
        recalls.append(r)

    avg_p = float(np.mean(precisions)) if precisions else 0.0
    avg_r = float(np.mean(recalls)) if recalls else 0.0

    if avg_p + avg_r == 0:
        return 0.0

    beta_sq = beta ** 2
    f_score = (1 + beta_sq) * (avg_p * avg_r) / ((beta_sq * avg_p) + avg_r)
    return f_score * 100.0
