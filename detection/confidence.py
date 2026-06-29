from dataclasses import dataclass
from config import HUMAN_THRESHOLD, AI_THRESHOLD, LLM_WEIGHT, STYLOMETRIC_WEIGHT


@dataclass
class AttributionResult:
    attribution: str        # "human" | "ai" | "uncertain"
    confidence_score: float  # 0–1 (higher = stronger AI signal)
    llm_score: float
    stylometric_score: float


def combine_scores(llm_score: float, stylometric_score: float) -> AttributionResult:
    """Combine signal scores into a single confidence score and attribution."""
    combined = (LLM_WEIGHT * llm_score) + (STYLOMETRIC_WEIGHT * stylometric_score)
    combined = round(combined, 4)

    if combined < HUMAN_THRESHOLD:
        attribution = "human"
    elif combined > AI_THRESHOLD:
        attribution = "ai"
    else:
        attribution = "uncertain"

    return AttributionResult(
        attribution=attribution,
        confidence_score=combined,
        llm_score=round(llm_score, 4),
        stylometric_score=round(stylometric_score, 4),
    )


def generate_label(confidence_score: float, attribution: str) -> dict:
    """Generate the transparency label text shown to platform users."""
    if attribution == "human":
        return {
            "verdict": "Likely Human-Written",
            "confidence_display": "High",
            "detail": (
                "Our system analyzed this content and found strong indicators of human authorship. "
                "This is an automated assessment and is not guaranteed to be correct. "
                "If you are the creator and believe this label is incorrect, you may submit an appeal."
            ),
        }
    if attribution == "ai":
        return {
            "verdict": "Likely AI-Generated",
            "confidence_display": "High",
            "detail": (
                "Our system found strong indicators that this content may have been AI-generated. "
                "This is an automated assessment and may be incorrect. "
                "If you are the creator and believe this label is incorrect, you may submit an appeal."
            ),
        }
    return {
        "verdict": "Authorship Uncertain",
        "confidence_display": "Low — this content has not been labeled as AI-generated",
        "detail": (
            "Our system detected mixed signals and cannot determine authorship with confidence. "
            "This content has not been flagged as AI-generated. "
            "If you are the creator and disagree with this assessment, you may submit an appeal."
        ),
    }
