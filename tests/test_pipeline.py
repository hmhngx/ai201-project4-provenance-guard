import pytest
from unittest.mock import patch
from detection.pipeline import run_detection_pipeline, PipelineResult
from detection.llm_signal import LLMResult
from detection.stylometric import StylemetricResult


def _mock_llm(score):
    return LLMResult(score=score)


def _mock_stylo(score):
    return StylemetricResult(
        score=score, variance_score=score, ttr_score=score, punctuation_score=score
    )


def test_returns_pipeline_result():
    with patch("detection.pipeline.classify_with_llm", return_value=_mock_llm(0.2)), \
         patch("detection.pipeline.compute_stylometric_score", return_value=_mock_stylo(0.3)):
        result = run_detection_pipeline("Some human text here with enough words.")
        assert isinstance(result, PipelineResult)


def test_human_signals_produce_human_attribution():
    with patch("detection.pipeline.classify_with_llm", return_value=_mock_llm(0.1)), \
         patch("detection.pipeline.compute_stylometric_score", return_value=_mock_stylo(0.1)):
        result = run_detection_pipeline("Human text here.")
        assert result.attribution == "human"
        assert result.confidence_score < 0.35


def test_ai_signals_produce_ai_attribution():
    with patch("detection.pipeline.classify_with_llm", return_value=_mock_llm(0.9)), \
         patch("detection.pipeline.compute_stylometric_score", return_value=_mock_stylo(0.9)):
        result = run_detection_pipeline("AI text here.")
        assert result.attribution == "ai"
        assert result.confidence_score > 0.70


def test_mixed_signals_produce_uncertain():
    with patch("detection.pipeline.classify_with_llm", return_value=_mock_llm(0.55)), \
         patch("detection.pipeline.compute_stylometric_score", return_value=_mock_stylo(0.50)):
        result = run_detection_pipeline("Borderline text here.")
        assert result.attribution == "uncertain"


def test_result_includes_all_fields():
    with patch("detection.pipeline.classify_with_llm", return_value=_mock_llm(0.1)), \
         patch("detection.pipeline.compute_stylometric_score", return_value=_mock_stylo(0.2)):
        result = run_detection_pipeline("Human text here.")
        assert hasattr(result, "llm_score")
        assert hasattr(result, "stylometric_score")
        assert hasattr(result, "confidence_score")
        assert hasattr(result, "attribution")
        assert isinstance(result.transparency_label, dict)
        assert "verdict" in result.transparency_label
        assert "detail" in result.transparency_label


def test_individual_scores_stored_separately():
    with patch("detection.pipeline.classify_with_llm", return_value=_mock_llm(0.3)), \
         patch("detection.pipeline.compute_stylometric_score", return_value=_mock_stylo(0.6)):
        result = run_detection_pipeline("Some text here.")
        assert result.llm_score == 0.3
        assert result.stylometric_score == 0.6
