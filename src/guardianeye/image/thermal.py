"""Thermal frame conversions for the aerial-thermal domain.

Two representations are supported and selected from ``configs/data.yaml``:

* ``replicate`` — the single thermal channel is copied to three identical
  channels so a standard RGB backbone can ingest it.
* ``pseudocolor`` — percentile-normalized iron-style false color.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def as_gray(image: NDArray[np.uint8] | NDArray[np.floating[Any]]) -> NDArray[np.float32]:
    """Collapse an array to a float32 grayscale plane in ``[0, 1]``."""
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    array = array.astype(np.float32)
    peak = float(array.max()) if array.size else 0.0
    if peak > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def replicate_thermal(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Copy one thermal channel into three identical RGB channels."""
    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = gray[..., 0]
    if gray.dtype != np.uint8:
        scaled = np.clip(np.asarray(gray, dtype=np.float32), 0.0, 1.0)
        if float(scaled.max()) <= 1.0:
            gray = (scaled * 255.0).astype(np.uint8)
        else:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def pseudocolor_thermal(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Map thermal intensity to an iron-style RGB image.

    Intensities are stretched between the 1st and 99th percentiles so a few
    saturated pixels do not flatten the rest of the frame.
    """
    gray = as_gray(image)
    low = float(np.percentile(gray, 1)) if gray.size else 0.0
    high = float(np.percentile(gray, 99)) if gray.size else 1.0
    if high <= low:
        high = low + 1e-6
    stretched = np.clip((gray - low) / (high - low), 0.0, 1.0)
    red = np.clip(1.6 * stretched, 0.0, 1.0)
    green = np.clip(1.8 * stretched - 0.45, 0.0, 1.0)
    cool = np.clip(0.85 - 1.6 * stretched, 0.0, 1.0)
    hot = np.clip(2.2 * stretched - 1.2, 0.0, 1.0)
    blue = np.where(stretched < 0.45, cool, hot)
    rgb = np.stack([red, green, blue], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def apply_thermal_representation(
    image: NDArray[np.uint8],
    mode: str,
) -> NDArray[np.uint8]:
    """Apply the configured thermal representation."""
    if mode == "replicate":
        return replicate_thermal(image)
    if mode == "pseudocolor":
        return pseudocolor_thermal(image)
    raise ValueError(f"Unknown thermal representation: {mode}")
