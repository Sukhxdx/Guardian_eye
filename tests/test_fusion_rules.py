"""Every late-fusion path, including missing modalities and threshold boundaries."""

from __future__ import annotations

from guardianeye.fusion.rules import (
    AudioEvidence,
    AudioWindowScore,
    Detection,
    FusionConfig,
    ImageEvidence,
    compare_modalities,
    decide,
)


def _cfg(**overrides: object) -> FusionConfig:
    base = dict(
        image_positive_threshold=0.5,
        audio_positive_threshold=0.5,
        unimodal_image_threshold=0.85,
        unimodal_audio_threshold=0.85,
        time_window_s=5.0,
        image_weight=0.5,
        audio_weight=0.5,
        decision_mode="rules",
        weighted_high_threshold=0.75,
        weighted_medium_threshold=0.45,
        alert_levels=("HIGH",),
        visual_indicator_classes=("human", "vehicle"),
    )
    base.update(overrides)
    return FusionConfig(**base)  # type: ignore[arg-type]


def _image(cls: str, conf: float, timestamp: float = 0.0, reliable: bool = True) -> ImageEvidence:
    detections = () if cls == "empty" else (Detection(cls, conf, timestamp),)
    return ImageEvidence(present=True, reliable=reliable, detections=detections)


def _audio(
    prob: float, start: float = 0.0, end: float = 3.0, reliable: bool = True
) -> AudioEvidence:
    return AudioEvidence(
        present=True,
        reliable=reliable,
        windows=(AudioWindowScore(start, end, prob),),
    )


def test_high_when_human_and_gunshot_coincide() -> None:
    decision = decide(_image("human", 0.9), _audio(0.9), _cfg())
    assert decision.risk_level == "HIGH"
    assert decision.alerted is True
    assert decision.reason == "coincident_indicators"


def test_vehicle_counts_as_visual_indicator() -> None:
    decision = decide(_image("vehicle", 0.8), _audio(0.8), _cfg())
    assert decision.risk_level == "HIGH"


def test_threshold_equality_is_strong() -> None:
    decision = decide(_image("human", 0.5), _audio(0.5), _cfg())
    assert decision.risk_level == "HIGH"


def test_just_below_image_threshold_is_medium() -> None:
    decision = decide(_image("human", 0.49), _audio(0.9), _cfg())
    assert decision.risk_level == "MEDIUM"
    assert decision.alerted is False
    assert decision.reason == "single_indicator_audio"


def test_just_below_audio_threshold_is_medium() -> None:
    decision = decide(_image("vehicle", 0.9), _audio(0.49), _cfg())
    assert decision.risk_level == "MEDIUM"
    assert decision.reason == "single_indicator_image"


def test_animal_is_not_a_visual_indicator() -> None:
    decision = decide(_image("animal", 0.99), _audio(0.99), _cfg())
    assert decision.risk_level == "MEDIUM"
    assert decision.reason == "single_indicator_audio"


def test_animal_plus_background_is_low() -> None:
    decision = decide(_image("animal", 0.99), _audio(0.1), _cfg())
    assert decision.risk_level == "LOW"


def test_empty_plus_quiet_audio_is_low() -> None:
    decision = decide(_image("empty", 0.0), _audio(0.2), _cfg())
    assert decision.risk_level == "LOW"
    assert decision.reason == "no_strong_indicator"


def test_time_window_boundary_is_inclusive() -> None:
    audio = _audio(0.9, start=0.0, end=3.0)
    inside = decide(_image("human", 0.9, timestamp=8.0), audio, _cfg())
    outside = decide(_image("human", 0.9, timestamp=8.01), audio, _cfg())
    assert inside.risk_level == "HIGH"
    assert outside.risk_level == "MEDIUM"
    assert outside.reason == "indicators_outside_time_window"


def test_early_edge_of_time_window() -> None:
    audio = _audio(0.9, start=0.0, end=3.0)
    inside = decide(_image("human", 0.9, timestamp=-5.0), audio, _cfg())
    outside = decide(_image("human", 0.9, timestamp=-5.01), audio, _cfg())
    assert inside.risk_level == "HIGH"
    assert outside.risk_level == "MEDIUM"


def test_image_only_uses_stricter_threshold_and_cannot_be_high() -> None:
    high_enough = decide(_image("human", 0.85), None, _cfg())
    too_low = decide(_image("human", 0.849), None, _cfg())
    assert high_enough.risk_level == "MEDIUM"
    assert high_enough.alerted is False
    assert high_enough.reason == "unimodal_image"
    assert too_low.risk_level == "LOW"
    assert too_low.reason == "unimodal_image_below_strict_threshold"


def test_audio_only_cannot_reach_high() -> None:
    decision = decide(None, _audio(0.99), _cfg())
    assert decision.risk_level == "MEDIUM"
    assert decision.alerted is False
    weak = decide(None, _audio(0.5), _cfg())
    assert weak.risk_level == "LOW"


def test_unreliable_weights_do_not_alert() -> None:
    decision = decide(_image("human", 0.99, reliable=False), _audio(0.99, reliable=False), _cfg())
    assert decision.risk_level == "LOW"
    assert decision.alerted is False
    assert decision.reason == "unreliable_weights"


def test_one_unreliable_modality_degrades_to_unimodal() -> None:
    decision = decide(_image("human", 0.99, reliable=False), _audio(0.9), _cfg())
    assert decision.risk_level == "MEDIUM"
    assert decision.reason == "degraded_audio"


def test_no_modality() -> None:
    decision = decide(None, None, _cfg())
    assert decision.risk_level == "LOW"
    assert decision.modalities_present == ()
    assert decision.reason == "no_modality"


def test_weighted_mode_can_be_high_only_with_both() -> None:
    both = decide(_image("human", 0.9), _audio(0.9), _cfg(decision_mode="weighted"))
    only = decide(_image("human", 0.95), None, _cfg(decision_mode="weighted"))
    assert both.risk_level == "HIGH"
    assert only.risk_level == "MEDIUM"


def test_night_profile_changes_the_score() -> None:
    balanced = decide(_image("human", 1.0), _audio(0.0), _cfg(image_weight=0.5, audio_weight=0.5))
    night = decide(_image("human", 1.0), _audio(0.0), _cfg(image_weight=0.3, audio_weight=0.7))
    assert night.risk_score < balanced.risk_score


def test_alert_levels_can_include_medium() -> None:
    decision = decide(_image("human", 0.9), None, _cfg(alert_levels=("HIGH", "MEDIUM")))
    assert decision.risk_level == "MEDIUM"
    assert decision.alerted is True


def test_compare_modalities_blocks_unimodal_high() -> None:
    compared = compare_modalities(_image("human", 0.95), _audio(0.95), _cfg())
    assert compared["fused"].risk_level == "HIGH"
    assert compared["image_only"].risk_level != "HIGH"
    assert compared["audio_only"].risk_level != "HIGH"
