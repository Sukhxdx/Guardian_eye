"""The audit log keeps events that did not alert."""

from __future__ import annotations

from pathlib import Path

from guardianeye.serve.event_log import EventLog


def test_non_alerted_events_are_stored(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.sqlite")
    log.add(
        {
            "event_id": "a",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "alerted": False,
            "risk_level": "LOW",
        }
    )
    log.add(
        {
            "event_id": "b",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "alerted": True,
            "risk_level": "HIGH",
        }
    )
    rows = log.list_events()
    assert {row["event_id"] for row in rows} == {"a", "b"}
    assert any(row["alerted"] is False for row in rows)
