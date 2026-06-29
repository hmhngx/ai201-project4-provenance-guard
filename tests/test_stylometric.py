import pytest
from detection.stylometric import compute_stylometric_score, StylemetricResult


def test_returns_stylometric_result():
    result = compute_stylometric_score(
        "Hello world. This is a test sentence with enough words to pass the minimum."
    )
    assert isinstance(result, StylemetricResult)
    assert 0.0 <= result.score <= 1.0


def test_result_contains_all_component_scores():
    result = compute_stylometric_score(
        "The cat sat on the mat. The dog ran to the park. "
        "The bird flew high in the sky and circled twice above the old oak tree."
    )
    assert 0.0 <= result.variance_score <= 1.0
    assert 0.0 <= result.ttr_score <= 1.0
    assert 0.0 <= result.punctuation_score <= 1.0


def test_uniform_sentences_score_higher_ai_than_varied():
    # Very uniform sentence lengths → higher AI signal
    uniform = (
        "The cat sat on the mat. The dog ran to the park. "
        "The bird flew past quickly."
    )
    # Very varied sentence lengths → lower AI signal
    varied = (
        "Short. This sentence is considerably longer and more complex in its structure. "
        "Brief. But this one is very very very very long indeed and goes on and on forever."
    )
    score_uniform = compute_stylometric_score(uniform)
    score_varied = compute_stylometric_score(varied)
    assert score_uniform.score > score_varied.score


def test_expressive_punctuation_lowers_ai_score():
    no_punct = "The system processed the data. It returned a result. The result was correct."
    with_punct = "The system processed the data! It returned a result — unexpectedly. Was it correct?"
    score_no = compute_stylometric_score(no_punct)
    score_with = compute_stylometric_score(with_punct)
    assert score_no.punctuation_score > score_with.punctuation_score


def test_insufficient_text_returns_uncertain():
    result = compute_stylometric_score("Short text.")
    assert result.score == 0.5
    assert result.variance_score == 0.5
    assert result.ttr_score == 0.5
    assert result.punctuation_score == 0.5


def test_empty_string_returns_uncertain():
    result = compute_stylometric_score("")
    assert result.score == 0.5


def test_score_is_between_0_and_1_for_any_input():
    inputs = [
        "a " * 50,                                       # repetitive
        "The " + " ".join(f"word{i}" for i in range(50)),  # high variety
        "Short sentence. " * 10,                         # uniform short
        "! ? — … " * 15 + " words words words words",   # high punctuation
    ]
    for text in inputs:
        result = compute_stylometric_score(text)
        assert 0.0 <= result.score <= 1.0, f"Score out of range for: {text[:40]}"
