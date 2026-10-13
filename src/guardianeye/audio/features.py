"""Waveform loading, fixed windows, and log-mel spectrograms."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def read_wav(path: Path) -> tuple[NDArray[np.float32], int]:
    """Load a mono 16-bit or 32-bit PCM WAV file into ``[-1, 1]`` float32."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width {sample_width} in {path}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.float32)
    return audio, sample_rate


def write_wav(path: Path, audio: NDArray[np.floating[Any]], sample_rate: int) -> None:
    """Write a mono 16-bit PCM WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def resample(audio: NDArray[np.floating[Any]], orig_sr: int, target_sr: int) -> NDArray[np.float32]:
    """Linear resample. Returns the input when the rates already match."""
    signal = np.asarray(audio, dtype=np.float32)
    if orig_sr == target_sr:
        return signal
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive")
    duration = len(signal) / float(orig_sr)
    target_len = max(int(round(duration * target_sr)), 1)
    if len(signal) == 0:
        return np.zeros(target_len, dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, num=len(signal), endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(target_x, source_x, signal).astype(np.float32)


def fixed_windows(
    audio: NDArray[np.floating[Any]],
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
) -> list[tuple[float, NDArray[np.float32]]]:
    """Cut ``audio`` into fixed-length windows.

    Short clips are zero-padded. Each item is ``(start_seconds, samples)``.
    """
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")
    signal = np.asarray(audio, dtype=np.float32)
    window = int(round(window_seconds * sample_rate))
    hop = int(round(hop_seconds * sample_rate))
    if window <= 0 or hop <= 0:
        raise ValueError("Window and hop must cover at least one sample")
    if len(signal) < window:
        signal = np.pad(signal, (0, window - len(signal)))
    starts = list(range(0, len(signal) - window + 1, hop))
    if not starts:
        starts = [0]
    return [(start / sample_rate, signal[start : start + window].copy()) for start in starts]


def log_mel(
    audio: NDArray[np.floating[Any]],
    sample_rate: int,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
    fmin: float = 50.0,
    fmax: float = 8000.0,
) -> NDArray[np.float32]:
    """Compute a log-mel spectrogram with shape ``(n_mels, time)``."""
    import librosa

    signal = np.asarray(audio, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=min(fmax, sample_rate / 2.0),
        power=2.0,
    )
    db = librosa.power_to_db(mel, ref=np.max)
    return db.astype(np.float32)


def spectrogram_image_to_mel(
    path: Path,
    n_mels: int,
    n_frames: int,
) -> NDArray[np.float32]:
    """Resize a supplied spectrogram image to ``(n_mels, n_frames)``.

    Vietnam releases include spectrogram pictures. They are an audio
    representation, not a third modality and not a waveform.
    """
    from PIL import Image

    image = Image.open(path).convert("L").resize((n_frames, n_mels))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def load_audio_file(path: Path, target_sr: int) -> tuple[NDArray[np.float32], int]:
    """Load WAV via the stdlib, or other formats via librosa, then resample."""
    if path.suffix.lower() == ".wav":
        try:
            audio, sample_rate = read_wav(path)
            return resample(audio, sample_rate, target_sr), target_sr
        except (wave.Error, ValueError):
            pass
    import librosa

    loaded, loaded_sr = librosa.load(path, sr=target_sr, mono=True)
    return np.asarray(loaded, dtype=np.float32), int(loaded_sr)
