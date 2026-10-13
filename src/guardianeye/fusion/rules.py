"""Readable late-fusion rules.

HIGH requires a human or vehicle detection and a gunshot score, both at or
above threshold, inside the configured time window. MEDIUM is exactly one
strong indicator, or two strong indicators that are not coincident. LOW is
neither. A missing modality cannot produce HIGH; it must clear a stricter
threshold to count as MEDIUM.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
    """One visual detection."""

    class_name: str
    confidence: float
    timestamp_s: float = 0.0
    bbox_xyxy: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class ImageEvidence:
    """Image branch output. ``present`` means a file was supplied."""

    present: bool
    reliable: bool
    detections: tuple[Detection, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class AudioWindowScore:
    """Gunshot probability for one analysis window."""

    start_s: float
    end_s: float
    gunshot_probability: float


@dataclass(frozen=True)
class AudioEvidence:
    """Audio branch output. ``present`` means a file was supplied."""

    present: bool
    reliable: bool
    windows: tuple[AudioWindowScore, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class FusionConfig:
    """Thresholds and weights. Scores equal to a threshold count as strong."""

    image_positive_threshold: float = 0.5
    audio_positive_threshold: float = 0.5
    unimodal_image_threshold: float = 0.85
    unimodal_audio_threshold: float = 0.85
    time_window_s: float = 5.0
    image_weight: float = 0.5
    audio_weight: float = 0.5
    decision_mode: str = "rules"
    weighted_high_threshold: float = 0.75
    weighted_medium_threshold: float = 0.45
    alert_levels: tuple[str, ...] = ("HIGH",)
    visual_indicator_classes: tuple[str, ...] = ("human", "vehicle")
    protocol_note: str = "synthetic proxy protocol — not field validation"
    profile: str = "balanced"


@dataclass(frozen=True)
class FusionDecision:
    """Risk level for one event."""

    risk_level: str
    risk_score: float
    alerted: bool
    reason: str
    thresholds_used: dict[str, float | str | list[str]]
    modalities_present: tuple[str, ...]
    image_summary: dict[str, object] = field(default_factory=dict)
    audio_summary: dict[str, object] = field(default_factory=dict)


def per_class_max(evidence: ImageEvidence) -> dict[str, float]:
    """Maximum confidence for human, vehicle, and animal."""
    scores = {"human": 0.0, "vehicle": 0.0, "animal": 0.0}
    for detection in evidence.detections:
        if detection.class_name in scores:
            scores[detection.class_name] = max(scores[detection.class_name], detection.confidence)
    return scores


def indicator_score(evidence: ImageEvidence, classes: tuple[str, ...]) -> float:
    """Max confidence among visual indicator classes. Animals do not count."""
    best = 0.0
    for detection in evidence.detections:
        if detection.class_name in classes:
            best = max(best, detection.confidence)
    return best


def gunshot_score(evidence: AudioEvidence) -> float:
    """Max gunshot probability across windows."""
    if not evidence.windows:
        return 0.0
    return max(window.gunshot_probability for window in evidence.windows)


def _coincident(
    image: ImageEvidence,
    audio: AudioEvidence,
    image_threshold: float,
    audio_threshold: float,
    time_window_s: float,
    classes: tuple[str, ...],
) -> bool:
    visuals = [
        detection
        for detection in image.detections
        if detection.class_name in classes and detection.confidence >= image_threshold
    ]
    shots = [window for window in audio.windows if window.gunshot_probability >= audio_threshold]
    for detection in visuals:
        for window in shots:
            if (
                (window.start_s - time_window_s)
                <= detection.timestamp_s
                <= (window.end_s + time_window_s)
            ):
                return True
    return False


def _weighted_score(
    image_score: float,
    audio_score: float,
    use_image: bool,
    use_audio: bool,
    image_weight: float,
    audio_weight: float,
) -> float:
    total = 0.0
    weight = 0.0
    if use_image:
        total += image_weight * image_score
        weight += image_weight
    if use_audio:
        total += audio_weight * audio_score
        weight += audio_weight
    if weight <= 0:
        return 0.0
    return total / weight


def _thresholds(cfg: FusionConfig) -> dict[str, float | str | list[str]]:
    return {
        "image_positive_threshold": cfg.image_positive_threshold,
        "audio_positive_threshold": cfg.audio_positive_threshold,
        "unimodal_image_threshold": cfg.unimodal_image_threshold,
        "unimodal_audio_threshold": cfg.unimodal_audio_threshold,
        "time_window_s": cfg.time_window_s,
        "image_weight": cfg.image_weight,
        "audio_weight": cfg.audio_weight,
        "weighted_high_threshold": cfg.weighted_high_threshold,
        "weighted_medium_threshold": cfg.weighted_medium_threshold,
        "decision_mode": cfg.decision_mode,
        "profile": cfg.profile,
        "alert_levels": list(cfg.alert_levels),
    }


def _summaries(
    image: ImageEvidence,
    audio: AudioEvidence,
    classes: tuple[str, ...],
) -> tuple[dict[str, object], dict[str, object]]:
    image_summary: dict[str, object] = {
        "present": image.present,
        "reliable": image.reliable,
        "per_class_max": per_class_max(image),
        "indicator_score": indicator_score(image, classes),
        "detections": [
            {
                "class_name": detection.class_name,
                "confidence": detection.confidence,
                "timestamp_s": detection.timestamp_s,
                "bbox_xyxy": detection.bbox_xyxy,
            }
            for detection in image.detections
        ],
        "note": image.note,
    }
    audio_summary: dict[str, object] = {
        "present": audio.present,
        "reliable": audio.reliable,
        "gunshot_probability": gunshot_score(audio) if audio.windows else None,
        "windows": [
            {
                "start_s": window.start_s,
                "end_s": window.end_s,
                "gunshot_probability": window.gunshot_probability,
            }
            for window in audio.windows
        ],
        "note": audio.note,
    }
    return image_summary, audio_summary


def _finish(
    level: str,
    score: float,
    reason: str,
    cfg: FusionConfig,
    modalities: tuple[str, ...],
    image_summary: dict[str, object],
    audio_summary: dict[str, object],
) -> FusionDecision:
    if level not in {"HIGH", "MEDIUM", "LOW"}:
        raise ValueError(f"Unexpected risk level {level}")
    return FusionDecision(
        risk_level=level,
        risk_score=float(score),
        alerted=level in cfg.alert_levels,
        reason=reason,
        thresholds_used=_thresholds(cfg),
        modalities_present=modalities,
        image_summary=image_summary,
        audio_summary=audio_summary,
    )


def decide(
    image: ImageEvidence | None,
    audio: AudioEvidence | None,
    cfg: FusionConfig,
) -> FusionDecision:
    """Apply the late-fusion policy. Missing evidence is allowed."""
    image = image or ImageEvidence(present=False, reliable=False)
    audio = audio or AudioEvidence(present=False, reliable=False)
    modalities = tuple(
        name for name, flag in (("image", image.present), ("audio", audio.present)) if flag
    )
    image_summary, audio_summary = _summaries(image, audio, cfg.visual_indicator_classes)
    reliable_image = image.present and image.reliable
    reliable_audio = audio.present and audio.reliable
    image_score = indicator_score(image, cfg.visual_indicator_classes) if reliable_image else 0.0
    audio_value = gunshot_score(audio) if reliable_audio else 0.0
    score = _weighted_score(
        image_score,
        audio_value,
        reliable_image,
        reliable_audio,
        cfg.image_weight,
        cfg.audio_weight,
    )

    if not modalities:
        return _finish("LOW", score, "no_modality", cfg, modalities, image_summary, audio_summary)

    if not reliable_image and not reliable_audio:
        return _finish(
            "LOW",
            score,
            "unreliable_weights",
            cfg,
            modalities,
            image_summary,
            audio_summary,
        )

    if reliable_image and reliable_audio:
        return _both(image, audio, cfg, score, modalities, image_summary, audio_summary)

    return _unimodal(
        image if reliable_image else None,
        audio if reliable_audio else None,
        cfg,
        score,
        modalities,
        image_summary,
        audio_summary,
        degraded=image.present and audio.present,
    )


def _both(
    image: ImageEvidence,
    audio: AudioEvidence,
    cfg: FusionConfig,
    score: float,
    modalities: tuple[str, ...],
    image_summary: dict[str, object],
    audio_summary: dict[str, object],
) -> FusionDecision:
    if cfg.decision_mode == "weighted":
        if score >= cfg.weighted_high_threshold:
            level, reason = "HIGH", "weighted_high"
        elif score >= cfg.weighted_medium_threshold:
            level, reason = "MEDIUM", "weighted_medium"
        else:
            level, reason = "LOW", "weighted_low"
        return _finish(level, score, reason, cfg, modalities, image_summary, audio_summary)

    visual_strong = (
        indicator_score(image, cfg.visual_indicator_classes) >= cfg.image_positive_threshold
    )
    audio_strong = gunshot_score(audio) >= cfg.audio_positive_threshold
    coincident = _coincident(
        image,
        audio,
        cfg.image_positive_threshold,
        cfg.audio_positive_threshold,
        cfg.time_window_s,
        cfg.visual_indicator_classes,
    )
    if visual_strong and audio_strong and coincident:
        return _finish(
            "HIGH", score, "coincident_indicators", cfg, modalities, image_summary, audio_summary
        )
    if visual_strong and audio_strong:
        return _finish(
            "MEDIUM",
            score,
            "indicators_outside_time_window",
            cfg,
            modalities,
            image_summary,
            audio_summary,
        )
    if visual_strong ^ audio_strong:
        which = "image" if visual_strong else "audio"
        return _finish(
            "MEDIUM",
            score,
            f"single_indicator_{which}",
            cfg,
            modalities,
            image_summary,
            audio_summary,
        )
    return _finish(
        "LOW", score, "no_strong_indicator", cfg, modalities, image_summary, audio_summary
    )


def _unimodal(
    image: ImageEvidence | None,
    audio: AudioEvidence | None,
    cfg: FusionConfig,
    score: float,
    modalities: tuple[str, ...],
    image_summary: dict[str, object],
    audio_summary: dict[str, object],
    degraded: bool,
) -> FusionDecision:
    """Stricter single-modality path. HIGH is not reachable."""
    if image is not None:
        strong = (
            indicator_score(image, cfg.visual_indicator_classes) >= cfg.unimodal_image_threshold
        )
        kind = "image"
    elif audio is not None:
        strong = gunshot_score(audio) >= cfg.unimodal_audio_threshold
        kind = "audio"
    else:
        return _finish(
            "LOW", score, "unreliable_weights", cfg, modalities, image_summary, audio_summary
        )

    if cfg.decision_mode == "weighted" and score < (
        cfg.unimodal_image_threshold if kind == "image" else cfg.unimodal_audio_threshold
    ):
        strong = False

    prefix = "degraded_" if degraded else "unimodal_"
    if strong:
        return _finish(
            "MEDIUM", score, f"{prefix}{kind}", cfg, modalities, image_summary, audio_summary
        )
    return _finish(
        "LOW",
        score,
        f"{prefix}{kind}_below_strict_threshold",
        cfg,
        modalities,
        image_summary,
        audio_summary,
    )


def compare_modalities(
    image: ImageEvidence | None,
    audio: AudioEvidence | None,
    cfg: FusionConfig,
) -> dict[str, FusionDecision]:
    """Run image-only, audio-only, and fused decisions on the same evidence."""
    image = image or ImageEvidence(present=False, reliable=False)
    audio = audio or AudioEvidence(present=False, reliable=False)
    absent_image = ImageEvidence(present=False, reliable=False)
    absent_audio = AudioEvidence(present=False, reliable=False)
    return {
        "image_only": decide(image, absent_audio, cfg),
        "audio_only": decide(absent_image, audio, cfg),
        "fused": decide(image, audio, cfg),
    }
