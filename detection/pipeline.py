from dataclasses import dataclass
from detection.llm_signal import classify_with_llm
from detection.stylometric import compute_stylometric_score
from detection.confidence import combine_scores, generate_label


@dataclass
class PipelineResult:
    attribution: str            # "human" | "ai" | "uncertain"
    confidence_score: float     # 0–1
    llm_score: float
    stylometric_score: float
    transparency_label: dict    # {verdict, confidence_display, detail}


def run_detection_pipeline(text: str) -> PipelineResult:
    """Run both detection signals and return a combined result."""
    llm_result = classify_with_llm(text)
    stylo_result = compute_stylometric_score(text)

    attribution_result = combine_scores(
        llm_score=llm_result.score,
        stylometric_score=stylo_result.score,
    )
    label = generate_label(attribution_result.confidence_score, attribution_result.attribution)

    return PipelineResult(
        attribution=attribution_result.attribution,
        confidence_score=attribution_result.confidence_score,
        llm_score=llm_result.score,
        stylometric_score=stylo_result.score,
        transparency_label=label,
    )
