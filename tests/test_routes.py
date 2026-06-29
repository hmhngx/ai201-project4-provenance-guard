import pytest
import json
import os
from unittest.mock import patch
from detection.llm_signal import LLMResult


def _mock_llm(score=0.20):
    return LLMResult(score=score, raw_response=str(score))


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
    with patch("app.classify_with_llm", return_value=_mock_llm(0.20)):
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


def test_submit_returns_400_on_missing_content(client):
    resp = client.post("/submit", json={"creator_id": "user_1"})
    assert resp.status_code == 400


def test_submit_returns_400_on_missing_creator_id(client):
    resp = client.post("/submit", json={"content": _VALID_CONTENT})
    assert resp.status_code == 400


def test_submit_returns_400_on_too_short_content(client):
    resp = client.post("/submit", json={"content": "Short.", "creator_id": "user_1"})
    assert resp.status_code == 400


def test_submit_writes_to_audit_log(client):
    with patch("app.classify_with_llm", return_value=_mock_llm(0.20)):
        client.post(
            "/submit",
            json={"content": _VALID_CONTENT, "creator_id": "user_1"},
        )
    resp = client.get("/log")
    data = json.loads(resp.data)
    assert len(data["entries"]) == 1
    entry = data["entries"][0]
    assert entry["creator_id"] == "user_1"
    assert entry["status"] == "classified"


def test_log_returns_entries_list(client):
    with patch("app.classify_with_llm", return_value=_mock_llm()):
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
