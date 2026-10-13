"""Temperature scaling and Platt scaling for modality scores.

Artifacts record whether a fit actually ran. An identity map is stored when
the validation split is too small, and callers must not describe that as a fit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def softmax(logits: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Numerically stable softmax over the last axis."""
    shifted = np.asarray(logits, dtype=np.float64)
    shifted = shifted - np.max(shifted, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def nll(logits: NDArray[np.floating[Any]], labels: NDArray[np.integer[Any]]) -> float:
    """Mean negative log likelihood of integer class labels."""
    probs = softmax(logits)
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, 1.0))))


def fit_temperature(
    logits: NDArray[np.floating[Any]],
    labels: NDArray[np.integer[Any]],
) -> dict[str, Any]:
    """Grid-search a positive temperature. Returns identity when data is too thin."""
    array = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if array.ndim != 2 or len(array) < 4 or len(np.unique(targets)) < 2:
        return {
            "method": "temperature",
            "temperature": 1.0,
            "fitted": False,
            "reason": "insufficient_validation_data",
        }
    best_t = 1.0
    best_nll = nll(array, targets)
    for temperature in np.linspace(0.5, 5.0, 46):
        loss = nll(array / float(temperature), targets)
        if loss < best_nll:
            best_nll = loss
            best_t = float(temperature)
    return {
        "method": "temperature",
        "temperature": best_t,
        "fitted": True,
        "nll": best_nll,
    }


def apply_temperature(logits: NDArray[np.floating[Any]], temperature: float) -> NDArray[np.float64]:
    """Return calibrated class probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return softmax(np.asarray(logits, dtype=np.float64) / temperature)


def fit_platt(
    scores: NDArray[np.floating[Any]], labels: NDArray[np.integer[Any]]
) -> dict[str, Any]:
    """Fit a 1D logistic map ``sigmoid(a * score + b)`` by gradient steps."""
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if len(x) < 4 or len(np.unique(y)) < 2:
        return {
            "method": "platt",
            "a": 1.0,
            "b": 0.0,
            "fitted": False,
            "reason": "insufficient_validation_data",
        }
    a = 1.0
    b = 0.0
    lr = 0.1
    for _ in range(400):
        z = a * x + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        error = pred - y
        a -= lr * float(np.mean(error * x))
        b -= lr * float(np.mean(error))
    return {"method": "platt", "a": a, "b": b, "fitted": True}


def apply_platt(scores: NDArray[np.floating[Any]], a: float, b: float) -> NDArray[np.float64]:
    """Apply a fitted Platt map."""
    z = a * np.asarray(scores, dtype=np.float64) + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def save_calibration(path: Path, payload: dict[str, Any]) -> None:
    """Write a calibration artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibration(path: Path) -> dict[str, Any] | None:
    """Load a calibration artifact, or return None when it is missing."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return data
