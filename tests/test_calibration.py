"""Calibration maps are fit from scores, or explicitly marked as not fit."""

from __future__ import annotations

import numpy as np

from guardianeye.fusion.calibrate import (
    apply_platt,
    apply_temperature,
    fit_platt,
    fit_temperature,
    softmax,
)


def test_temperature_one_matches_softmax() -> None:
    logits = np.array([[0.2, 1.0], [1.5, -0.3]])
    calibrated = apply_temperature(logits, 1.0)
    assert np.allclose(calibrated, softmax(logits))


def test_higher_temperature_flattens_the_peak() -> None:
    logits = np.array([[0.0, 3.0]])
    cool = apply_temperature(logits, 1.0)[0, 1]
    warm = apply_temperature(logits, 4.0)[0, 1]
    assert warm < cool


def test_temperature_fit_marks_thin_data() -> None:
    payload = fit_temperature(np.array([[0.0, 1.0]]), np.array([1]))
    assert payload["fitted"] is False
    assert payload["temperature"] == 1.0


def test_platt_separates_obvious_scores() -> None:
    scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.85])
    labels = np.array([0, 0, 0, 1, 1, 1])
    payload = fit_platt(scores, labels)
    assert payload["fitted"] is True
    probs = apply_platt(scores, payload["a"], payload["b"])
    assert float(probs[3:].mean()) > float(probs[:3].mean())
