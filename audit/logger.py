import sqlite3
import uuid
import json
from datetime import datetime, timezone


class AuditLogger:
    def __init__(self, db_path: str = "audit.db"):
        self.db_path = db_path

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    content_id TEXT PRIMARY KEY,
                    creator_id TEXT NOT NULL,
                    content_snippet TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    llm_score REAL NOT NULL,
                    stylometric_score REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    attribution TEXT NOT NULL,
                    transparency_label TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'classified',
                    appeals TEXT NOT NULL DEFAULT '[]'
                )
            """)
            conn.commit()

    def log_decision(
        self,
        creator_id: str,
        content_snippet: str,
        llm_score: float,
        stylometric_score: float,
        confidence_score: float,
        attribution: str,
        transparency_label: str,
    ) -> str:
        content_id = f"cnt_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO decisions
                   (content_id, creator_id, content_snippet, timestamp,
                    llm_score, stylometric_score, confidence_score,
                    attribution, transparency_label, status, appeals)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'classified', '[]')""",
                (
                    content_id,
                    creator_id,
                    content_snippet[:500],
                    timestamp,
                    llm_score,
                    stylometric_score,
                    confidence_score,
                    attribution,
                    transparency_label,
                ),
            )
            conn.commit()
        return content_id

    def log_appeal(self, content_id: str, creator_id: str, reason: str) -> str:
        appeal_id = f"app_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        appeal = {
            "appeal_id": appeal_id,
            "creator_id": creator_id,
            "reason": reason,
            "timestamp": timestamp,
        }
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT appeals FROM decisions WHERE content_id = ?", (content_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"content_id {content_id} not found")
            appeals = json.loads(row[0])
            appeals.append(appeal)
            conn.execute(
                "UPDATE decisions SET appeals = ?, status = 'under_review' WHERE content_id = ?",
                (json.dumps(appeals), content_id),
            )
            conn.commit()
        return appeal_id

    def get_entry(self, content_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM decisions WHERE content_id = ?", (content_id,)
            ).fetchone()
        if row is None:
            return None
        entry = dict(row)
        entry["appeals"] = json.loads(entry["appeals"])
        return entry

    def get_all_entries(self, limit: int = 100) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        entries = []
        for row in rows:
            entry = dict(row)
            entry["appeals"] = json.loads(entry["appeals"])
            entries.append(entry)
        return entries
