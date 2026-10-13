"""SpecAugment-style masks, noise, and waveform time shift."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def time_shift(
    audio: NDArray[np.float32], max_fraction: float, rng: np.random.Generator
) -> NDArray[np.float32]:
    """Roll a waveform by up to ``max_fraction`` of its length."""
    signal = np.asarray(audio, dtype=np.float32)
    if max_fraction <= 0 or len(signal) == 0:
        return signal
    shift = int(rng.uniform(-max_fraction, max_fraction) * len(signal))
    return np.roll(signal, shift)


def augment_mel(
    mel: NDArray[np.float32],
    freq_mask: int,
    time_mask: int,
    noise_std: float,
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    """Apply additive noise plus one frequency mask and one time mask."""
    out = np.asarray(mel, dtype=np.float32).copy()
    if noise_std > 0:
        noise = np.asarray(rng.normal(0.0, noise_std, size=out.shape), dtype=np.float32)
        out += noise
    if freq_mask > 0 and out.shape[0] > 1:
        width = int(rng.integers(0, min(freq_mask, out.shape[0] - 1) + 1))
        if width:
            start = int(rng.integers(0, out.shape[0] - width + 1))
            out[start : start + width, :] = 0.0
    if time_mask > 0 and out.shape[1] > 1:
        width = int(rng.integers(0, min(time_mask, out.shape[1] - 1) + 1))
        if width:
            start = int(rng.integers(0, out.shape[1] - width + 1))
            out[:, start : start + width] = 0.0
    return out
