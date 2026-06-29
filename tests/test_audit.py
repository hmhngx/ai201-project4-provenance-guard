import pytest
from audit.logger import AuditLogger


@pytest.fixture
def logger(tmp_path):
    log = AuditLogger(db_path=str(tmp_path / "test_audit.db"))
    log.init_db()
    return log


def test_log_decision_returns_content_id(logger):
    content_id = logger.log_decision(
        creator_id="user_1",
        content_snippet="The cat sat on the mat.",
        llm_score=0.15,
        stylometric_score=0.25,
        confidence_score=0.19,
        attribution="human",
        transparency_label="Likely Human-Written",
    )
    assert content_id.startswith("cnt_")


def test_get_entry_returns_logged_decision(logger):
    content_id = logger.log_decision(
        creator_id="user_1",
        content_snippet="Test content.",
        llm_score=0.80,
        stylometric_score=0.75,
        confidence_score=0.78,
        attribution="ai",
        transparency_label="Likely AI-Generated",
    )
    entry = logger.get_entry(content_id)
    assert entry is not None
    assert entry["attribution"] == "ai"
    assert entry["confidence_score"] == 0.78
    assert entry["status"] == "classified"


def test_log_appeal_updates_status_to_under_review(logger):
    content_id = logger.log_decision(
        creator_id="user_2",
        content_snippet="Human text.",
        llm_score=0.20,
        stylometric_score=0.30,
        confidence_score=0.24,
        attribution="human",
        transparency_label="Likely Human-Written",
    )
    appeal_id = logger.log_appeal(
        content_id=content_id,
        creator_id="user_2",
        reason="I wrote this myself.",
    )
    assert appeal_id.startswith("app_")
    entry = logger.get_entry(content_id)
    assert entry["status"] == "under_review"
    assert len(entry["appeals"]) == 1
    assert entry["appeals"][0]["reason"] == "I wrote this myself."


def test_get_all_entries_returns_list(logger):
    logger.log_decision("u1", "text1", 0.1, 0.2, 0.14, "human", "Likely Human-Written")
    logger.log_decision("u2", "text2", 0.8, 0.9, 0.84, "ai", "Likely AI-Generated")
    entries = logger.get_all_entries()
    assert len(entries) >= 2


def test_get_entry_returns_none_for_unknown_id(logger):
    assert logger.get_entry("cnt_doesnotexist") is None


def test_log_appeal_raises_on_unknown_content_id(logger):
    with pytest.raises(ValueError, match="not found"):
        logger.log_appeal("cnt_unknown", "user_1", "My work.")
