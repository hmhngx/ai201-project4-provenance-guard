import pytest
import json
import os
from unittest.mock import patch, MagicMock
from detection.pipeline import PipelineResult


def _mock_pipeline(attribution="human", confidence=0.20, llm=0.15, stylo=0.28):
    return PipelineResult(
        attribution=attribution,
        confidence_score=confidence,
        llm_score=llm,
        stylometric_score=stylo,
        transparency_label={
            "verdict": "Likely Human-Written",
            "confidence_display": "High",
            "detail": (
                "Our system found strong indicators of human authorship. "
                "If you are the creator and believe this label is incorrect, "
                "you may submit an appeal."
            ),
        },
    )


@pytest.fixture
def client(tmp_path):
    os.environ["AUDIT_DB_PATH"] = str(tmp_path / "test.db")
    import app as app_module
    app_module._logger_instance = None
    flask_app = app_module.create_app(testing=True)
    with flask_app.test_client() as c:
        yield c


_VALID_CONTENT = (
    "The starling flew past the window at dusk. "
    "It circled twice, then vanished into the elm trees."
)


def test_submit_returns_200_with_all_fields(client):
    with patch("app.run_detection_pipeline", return_value=_mock_pipeline()):
        resp = client.post(
            "/submit",
            json={"content": _VALID_CONTENT, "creator_id": "user_1"},
        )
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "content_id" in data
    assert data["content_id"].startswith("cnt_")
    assert "attribution_result" in data
    assert "confidence_score" in data
    assert "transparency_label" in data


def test_submit_accepts_text_field_alias(client):
    with patch("app.run_detection_pipeline", return_value=_mock_pipeline()):
        resp = client.post(
            "/submit",
            json={"text": _VALID_CONTENT, "creator_id": "user_1"},
        )
    assert resp.status_code == 200


def test_submit_returns_400_on_missing_content(client):
    resp = client.post("/submit", json={"creator_id": "user_1"})
    assert resp.status_code == 400


def test_submit_returns_400_on_missing_creator_id(client):
    resp = client.post("/submit", json={"content": _VALID_CONTENT})
    assert resp.status_code == 400


def test_submit_returns_400_on_too_short_content(client):
    resp = client.post("/submit", json={"content": "Short.", "creator_id": "user_1"})
    assert resp.status_code == 400


def test_submit_writes_both_signal_scores_to_audit_log(client):
    with patch("app.run_detection_pipeline",
               return_value=_mock_pipeline(llm=0.15, stylo=0.28)):
        client.post("/submit", json={"content": _VALID_CONTENT, "creator_id": "user_1"})
    resp = client.get("/log")
    data = json.loads(resp.data)
    assert len(data["entries"]) == 1
    entry = data["entries"][0]
    assert entry["creator_id"] == "user_1"
    assert entry["llm_score"] == 0.15
    assert entry["stylometric_score"] == 0.28
    assert entry["status"] == "classified"


def test_log_returns_entries_list(client):
    with patch("app.run_detection_pipeline", return_value=_mock_pipeline()):
        client.post("/submit", json={"content": _VALID_CONTENT, "creator_id": "u1"})
        client.post("/submit", json={"content": _VALID_CONTENT, "creator_id": "u2"})
    resp = client.get("/log")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "entries" in data
    assert len(data["entries"]) >= 2


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert json.loads(resp.data)["status"] == "ok"


# ── Appeal endpoint tests ────────────────────────────────────────────────────

def _submit_and_get_id(client):
    """Helper: submit content and return the content_id."""
    with patch("app.run_detection_pipeline", return_value=_mock_pipeline()):
        resp = client.post("/submit", json={"content": _VALID_CONTENT, "creator_id": "user_1"})
    return json.loads(resp.data)["content_id"]


def test_appeal_returns_200_with_under_review_status(client):
    content_id = _submit_and_get_id(client)
    resp = client.post("/appeal", json={
        "content_id": content_id,
        "creator_id": "user_1",
        "reason": "I wrote this myself from personal experience.",
    })
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "under_review"
    assert "appeal_id" in data
    assert data["content_id"] == content_id


def test_appeal_accepts_creator_reasoning_alias(client):
    """Milestone spec uses 'creator_reasoning' as field name — accept both."""
    content_id = _submit_and_get_id(client)
    resp = client.post("/appeal", json={
        "content_id": content_id,
        "creator_id": "user_1",
        "creator_reasoning": "I wrote this myself.",
    })
    assert resp.status_code == 200
    assert json.loads(resp.data)["status"] == "under_review"


def test_appeal_updates_status_in_audit_log(client):
    content_id = _submit_and_get_id(client)
    client.post("/appeal", json={
        "content_id": content_id,
        "creator_id": "user_1",
        "reason": "I wrote this myself.",
    })
    log_resp = client.get("/log")
    entries = json.loads(log_resp.data)["entries"]
    entry = next(e for e in entries if e["content_id"] == content_id)
    assert entry["status"] == "under_review"
    assert len(entry["appeals"]) == 1
    assert "I wrote this myself" in entry["appeals"][0]["reason"]


def test_appeal_returns_404_on_unknown_content_id(client):
    resp = client.post("/appeal", json={
        "content_id": "cnt_doesnotexist",
        "creator_id": "user_1",
        "reason": "I wrote this.",
    })
    assert resp.status_code == 404


def test_appeal_returns_400_on_creator_mismatch(client):
    content_id = _submit_and_get_id(client)
    resp = client.post("/appeal", json={
        "content_id": content_id,
        "creator_id": "user_wrong",
        "reason": "I wrote this.",
    })
    assert resp.status_code == 400


def test_appeal_returns_400_on_missing_fields(client):
    resp = client.post("/appeal", json={"content_id": "cnt_abc"})
    assert resp.status_code == 400
