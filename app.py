import os
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from detection.llm_signal import classify_with_llm
from audit.logger import AuditLogger
from config import RATE_LIMIT, MAX_CONTENT_CHARS, MIN_CONTENT_WORDS, HUMAN_THRESHOLD, AI_THRESHOLD

_logger_instance = None


def get_logger() -> AuditLogger:
    global _logger_instance
    if _logger_instance is None:
        db_path = os.getenv("AUDIT_DB_PATH", "audit.db")
        _logger_instance = AuditLogger(db_path=db_path)
        _logger_instance.init_db()
    return _logger_instance


def _score_to_attribution(score: float) -> str:
    if score < HUMAN_THRESHOLD:
        return "human"
    if score > AI_THRESHOLD:
        return "ai"
    return "uncertain"


def _generate_label(attribution: str) -> dict:
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


def create_app(testing: bool = False) -> Flask:
    global _logger_instance
    if testing:
        _logger_instance = None

    app = Flask(__name__)
    app.config["TESTING"] = testing

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[RATE_LIMIT],
        storage_uri="memory://",
    )

    @app.route("/health", methods=["GET"])
    @limiter.exempt
    def health():
        return jsonify({"status": "ok", "version": "1.0.0"})

    @app.route("/submit", methods=["POST"])
    @limiter.limit(RATE_LIMIT)
    def submit():
        body = request.get_json(silent=True) or {}
        # Accept both "content" and "text" field names
        content = (body.get("content") or body.get("text") or "").strip()
        creator_id = body.get("creator_id", "").strip()

        if not content:
            return jsonify({"error": "content is required"}), 400
        if not creator_id:
            return jsonify({"error": "creator_id is required"}), 400
        if len(content) > MAX_CONTENT_CHARS:
            return jsonify({"error": f"content exceeds {MAX_CONTENT_CHARS} characters"}), 400
        if len(content.split()) < MIN_CONTENT_WORDS:
            return jsonify({"error": f"content must have at least {MIN_CONTENT_WORDS} words"}), 400

        # Signal 1: LLM classifier
        llm_result = classify_with_llm(content)
        llm_score = llm_result.score

        # Signal 2: stylometric (placeholder until Milestone 4)
        stylometric_score = 0.5

        # Confidence: weighted average (M4 will use real stylometric)
        confidence_score = round((0.60 * llm_score) + (0.40 * stylometric_score), 4)
        attribution = _score_to_attribution(confidence_score)
        label = _generate_label(attribution)

        content_id = get_logger().log_decision(
            creator_id=creator_id,
            content_snippet=content[:500],
            llm_score=llm_score,
            stylometric_score=stylometric_score,
            confidence_score=confidence_score,
            attribution=attribution,
            transparency_label=label["verdict"],
        )

        return jsonify({
            "content_id": content_id,
            "attribution_result": attribution,
            "confidence_score": confidence_score,
            "transparency_label": label,
        })

    @app.route("/log", methods=["GET"])
    @limiter.exempt
    def log():
        entries = get_logger().get_all_entries()
        return jsonify({"entries": entries})

    return app


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    flask_app = create_app()
    flask_app.run(debug=True, port=5000)
