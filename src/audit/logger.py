"""Persistent audit trail (JSON lines + optional SQLite)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import AUDIT_DIR


class AuditLogger:
    def __init__(
        self,
        jsonl_path: Path | None = None,
        sqlite_path: Path | None = None,
    ) -> None:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = jsonl_path or (AUDIT_DIR / "decisions.jsonl")
        self.sqlite_path = sqlite_path or (AUDIT_DIR / "decisions.db")
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    return_id TEXT,
                    timestamp TEXT,
                    action TEXT,
                    risk_score REAL,
                    predicted_class TEXT,
                    payload TEXT
                )
                """
            )
            conn.commit()

    def log(self, record: dict[str, Any]) -> dict[str, Any]:
        entry = dict(record)
        entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        line = json.dumps(entry, ensure_ascii=True)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                INSERT INTO decisions (return_id, timestamp, action, risk_score, predicted_class, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(entry.get("return_id")),
                    entry.get("timestamp"),
                    entry.get("action"),
                    entry.get("risk_score"),
                    entry.get("predicted_class"),
                    line,
                ),
            )
            conn.commit()
        return entry

    def tail(self, n: int = 10) -> list[dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []
        lines = self.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(x) for x in lines[-n:]]
