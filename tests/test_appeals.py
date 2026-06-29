import pytest
from unittest.mock import MagicMock
from appeals.handler import process_appeal, AppealError


def test_returns_appeal_id_and_under_review_status():
    mock_logger = MagicMock()
    mock_logger.get_entry.return_value = {"content_id": "cnt_abc", "creator_id": "user_1"}
    mock_logger.log_appeal.return_value = "app_xyz"

    result = process_appeal(
        content_id="cnt_abc",
        creator_id="user_1",
        reason="I wrote this myself.",
        logger=mock_logger,
    )
    assert result["appeal_id"] == "app_xyz"
    assert result["status"] == "under_review"
    assert "message" in result


def test_raises_on_unknown_content_id():
    mock_logger = MagicMock()
    mock_logger.get_entry.return_value = None

    with pytest.raises(AppealError, match="not found"):
        process_appeal(
            content_id="cnt_unknown",
            creator_id="user_1",
            reason="My work.",
            logger=mock_logger,
        )


def test_raises_on_creator_mismatch():
    mock_logger = MagicMock()
    mock_logger.get_entry.return_value = {
        "content_id": "cnt_abc",
        "creator_id": "user_correct",
    }

    with pytest.raises(AppealError, match="creator_id"):
        process_appeal(
            content_id="cnt_abc",
            creator_id="user_wrong",
            reason="My work.",
            logger=mock_logger,
        )


def test_raises_on_empty_reason():
    mock_logger = MagicMock()
    mock_logger.get_entry.return_value = {"content_id": "cnt_abc", "creator_id": "user_1"}

    with pytest.raises(AppealError, match="reason"):
        process_appeal(
            content_id="cnt_abc",
            creator_id="user_1",
            reason="   ",
            logger=mock_logger,
        )


def test_delegates_to_logger_with_stripped_reason():
    mock_logger = MagicMock()
    mock_logger.get_entry.return_value = {"content_id": "cnt_abc", "creator_id": "user_1"}
    mock_logger.log_appeal.return_value = "app_xyz"

    process_appeal(
        content_id="cnt_abc",
        creator_id="user_1",
        reason="  I wrote this.  ",
        logger=mock_logger,
    )
    mock_logger.log_appeal.assert_called_once_with(
        content_id="cnt_abc",
        creator_id="user_1",
        reason="I wrote this.",
    )
