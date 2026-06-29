import os
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from detection.pipeline import run_detection_pipeline
from audit.logger import AuditLogger
from appeals.handler import process_appeal, AppealError
from config import RATE_LIMIT, MAX_CONTENT_CHARS, MIN_CONTENT_WORDS

_logger_instance = None


def get_logger() -> AuditLogger:
    global _logger_instance
    if _logger_instance is None:
        db_path = os.getenv("AUDIT_DB_PATH", "audit.db")
        _logger_instance = AuditLogger(db_path=db_path)
        _logger_instance.init_db()
    return _logger_instance


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

        result = run_detection_pipeline(content)

        content_id = get_logger().log_decision(
            creator_id=creator_id,
            content_snippet=content[:500],
            llm_score=result.llm_score,
            stylometric_score=result.stylometric_score,
            confidence_score=result.confidence_score,
            attribution=result.attribution,
            transparency_label=result.transparency_label["verdict"],
        )

        return jsonify({
            "content_id": content_id,
            "attribution_result": result.attribution,
            "confidence_score": result.confidence_score,
            "transparency_label": result.transparency_label,
        })

    @app.route("/appeal", methods=["POST"])
    @limiter.limit(RATE_LIMIT)
    def appeal():
        body = request.get_json(silent=True) or {}
        content_id = body.get("content_id", "").strip()
        creator_id = body.get("creator_id", "").strip()
        # Accept both "reason" and "creator_reasoning" field names
        reason = (body.get("reason") or body.get("creator_reasoning") or "").strip()

        if not content_id or not creator_id or not reason:
            return jsonify({
                "error": "content_id, creator_id, and reason (or creator_reasoning) are required"
            }), 400

        try:
            result = process_appeal(
                content_id=content_id,
                creator_id=creator_id,
                reason=reason,
                logger=get_logger(),
            )
            return jsonify(result)
        except AppealError as e:
            msg = str(e)
            status_code = 404 if "not found" in msg else 400
            return jsonify({"error": msg}), status_code

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
