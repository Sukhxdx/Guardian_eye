"""SQLite log of every event, including ones that did not alert."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class EventLog:
    """Append-only audit log."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    alerted INTEGER NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, event: dict[str, Any]) -> None:
        """Store one event. Non-alerted events are kept."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO events (event_id, timestamp, alerted, payload) VALUES (?, ?, ?, ?)",
                (
                    str(event["event_id"]),
                    str(event["timestamp"]),
                    1 if event.get("alerted") else 0,
                    json.dumps(event),
                ),
            )

    def list_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return newest events first."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM events ORDER BY timestamp DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]
