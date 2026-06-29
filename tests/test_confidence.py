import pytest
from detection.confidence import combine_scores, generate_label, AttributionResult


def test_low_score_returns_human():
    result = combine_scores(llm_score=0.20, stylometric_score=0.25)
    assert result.attribution == "human"
    assert result.confidence_score < 0.35


def test_high_score_returns_ai():
    result = combine_scores(llm_score=0.85, stylometric_score=0.80)
    assert result.attribution == "ai"
    assert result.confidence_score > 0.70


def test_mid_score_returns_uncertain():
    result = combine_scores(llm_score=0.55, stylometric_score=0.50)
    assert result.attribution == "uncertain"


def test_boundary_just_below_human_threshold():
    # 0.60*0.30 + 0.40*0.40 = 0.18 + 0.16 = 0.34 → human
    result = combine_scores(llm_score=0.30, stylometric_score=0.40)
    assert result.attribution == "human"
    assert result.confidence_score < 0.35


def test_boundary_just_above_ai_threshold():
    # 0.60*0.85 + 0.40*0.50 = 0.51 + 0.20 = 0.71 → ai
    result = combine_scores(llm_score=0.85, stylometric_score=0.50)
    assert result.attribution == "ai"
    assert result.confidence_score > 0.70


def test_weights_applied_correctly():
    # LLM weight=0.60, stylometric weight=0.40
    result = combine_scores(llm_score=1.0, stylometric_score=0.0)
    assert abs(result.confidence_score - 0.60) < 0.01


def test_attribution_result_stores_individual_scores():
    result = combine_scores(llm_score=0.30, stylometric_score=0.40)
    assert result.llm_score == 0.30
    assert result.stylometric_score == 0.40


def test_label_human_has_required_fields():
    label = generate_label(0.20, "human")
    assert "Human-Written" in label["verdict"]
    assert "High" in label["confidence_display"]
    assert "appeal" in label["detail"].lower()


def test_label_uncertain_says_not_flagged():
    label = generate_label(0.52, "uncertain")
    assert "Uncertain" in label["verdict"]
    assert "not been labeled as AI-generated" in label["confidence_display"]
    assert "appeal" in label["detail"].lower()


def test_label_ai_has_required_fields():
    label = generate_label(0.85, "ai")
    assert "AI-Generated" in label["verdict"]
    assert "appeal" in label["detail"].lower()


def test_0_51_and_0_95_produce_different_verdicts():
    label_low = generate_label(0.51, "uncertain")
    label_high = generate_label(0.95, "ai")
    assert label_low["verdict"] != label_high["verdict"]
