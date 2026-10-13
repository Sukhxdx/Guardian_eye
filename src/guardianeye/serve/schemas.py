"""Public event schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EventRecord(BaseModel):
    """One logged event. A detection is an indicator, not a poaching finding."""

    event_id: str
    timestamp: str
    modalities_present: list[str]
    image_evidence: dict[str, Any]
    audio_evidence: dict[str, Any]
    risk_level: str
    risk_score: float
    thresholds_used: dict[str, Any]
    snapshot_path: str | None = None
    audio_clip_path: str | None = None
    latency_ms: float
    alerted: bool
    reason: str = ""
    protocol_note: str = ""
    comparison: dict[str, str] = Field(default_factory=dict)
    frame_summaries: list[dict[str, Any]] = Field(default_factory=list)
