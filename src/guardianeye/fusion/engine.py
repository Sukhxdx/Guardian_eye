"""Load fusion config and build the event record returned by the API."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from guardianeye.fusion.rules import FusionConfig, FusionDecision


def fusion_config_from_yaml(
    payload: dict[str, Any], profile: str | None = None, decision_mode: str | None = None
) -> FusionConfig:
    """Build a ``FusionConfig`` from ``configs/fusion.yaml`` contents."""
    chosen = profile or str(payload.get("default_profile", "balanced"))
    weights = payload.get("weights", {}).get(chosen)
    if not isinstance(weights, dict):
        raise KeyError(f"Unknown weight profile {chosen}")
    levels = tuple(str(level) for level in payload.get("alert_levels", ["HIGH"]))
    classes = tuple(
        str(name) for name in payload.get("visual_indicator_classes", ["human", "vehicle"])
    )
    return FusionConfig(
        image_positive_threshold=float(payload["image_positive_threshold"]),
        audio_positive_threshold=float(payload["audio_positive_threshold"]),
        unimodal_image_threshold=float(payload["unimodal_image_threshold"]),
        unimodal_audio_threshold=float(payload["unimodal_audio_threshold"]),
        time_window_s=float(payload["time_window_s"]),
        image_weight=float(weights["image"]),
        audio_weight=float(weights["audio"]),
        decision_mode=str(decision_mode or payload.get("decision_mode", "rules")),
        weighted_high_threshold=float(payload["weighted_high_threshold"]),
        weighted_medium_threshold=float(payload["weighted_medium_threshold"]),
        alert_levels=levels,
        visual_indicator_classes=classes,
        protocol_note=str(payload.get("protocol_note", "")),
        profile=chosen,
    )


def decision_to_event(
    decision: FusionDecision,
    *,
    snapshot_path: str | None,
    audio_clip_path: str | None,
    latency_ms: float,
    protocol_note: str,
    event_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Shape a decision into the public event schema."""
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "timestamp": timestamp or datetime.now(UTC).isoformat(),
        "modalities_present": list(decision.modalities_present),
        "image_evidence": decision.image_summary,
        "audio_evidence": decision.audio_summary,
        "risk_level": decision.risk_level,
        "risk_score": decision.risk_score,
        "thresholds_used": decision.thresholds_used,
        "snapshot_path": snapshot_path,
        "audio_clip_path": audio_clip_path,
        "latency_ms": latency_ms,
        "alerted": decision.alerted,
        "reason": decision.reason,
        "protocol_note": protocol_note,
    }
