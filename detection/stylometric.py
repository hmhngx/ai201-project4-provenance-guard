import re
import math
from dataclasses import dataclass


@dataclass
class StylemetricResult:
    score: float            # 0–1, where 1 = likely AI
    variance_score: float   # sentence length uniformity (high = AI)
    ttr_score: float        # low vocabulary diversity (high = AI)
    punctuation_score: float  # low expressive punctuation (high = AI)


def compute_stylometric_score(text: str) -> StylemetricResult:
    """Compute AI-likelihood from structural text properties. Returns 0–1 (1 = AI)."""
    words = text.split()
    if len(words) < 10:
        return StylemetricResult(score=0.5, variance_score=0.5, ttr_score=0.5, punctuation_score=0.5)

    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return StylemetricResult(score=0.5, variance_score=0.5, ttr_score=0.5, punctuation_score=0.5)

    variance_score = _sentence_variance_score(sentences)
    ttr_score = _type_token_ratio_score(words)
    punctuation_score = _punctuation_score(text)

    score = (0.40 * variance_score) + (0.35 * ttr_score) + (0.25 * punctuation_score)
    return StylemetricResult(
        score=round(score, 4),
        variance_score=round(variance_score, 4),
        ttr_score=round(ttr_score, 4),
        punctuation_score=round(punctuation_score, 4),
    )


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def _sentence_variance_score(sentences: list[str]) -> float:
    """Low variance → high AI score. AI produces more uniform sentence lengths."""
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    # AI typically std_dev < 4; human creative writing often > 10
    # Normalize: std_dev=0 → 1.0 (AI), std_dev≥12 → 0.0 (human)
    score = max(0.0, 1.0 - (std_dev / 12.0))
    return min(1.0, score)


def _type_token_ratio_score(words: list[str]) -> float:
    """Low TTR → high AI score. AI reuses vocabulary patterns more."""
    lower_words = [w.lower().strip(".,!?;:\"'—–…()") for w in words]
    ttr = len(set(lower_words)) / len(lower_words)
    # High TTR = diverse vocabulary = human; invert to get AI score
    score = max(0.0, 1.0 - ttr)
    return min(1.0, score)


def _punctuation_score(text: str) -> float:
    """Low expressive punctuation density → high AI score."""
    expressive = sum(1 for c in text if c in '!?;:—–…()')
    total = len(text)
    if total == 0:
        return 0.5
    density = expressive / total
    # density=0 → 1.0 (AI: no expressive punct); density≥0.05 → 0.0 (human)
    score = max(0.0, 1.0 - (density * 20.0))
    return min(1.0, score)
