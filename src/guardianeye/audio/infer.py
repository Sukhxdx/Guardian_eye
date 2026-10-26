"""Gunshot probabilities for a clip or a long recording."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from guardianeye.audio.features import fixed_windows, load_audio_file, log_mel
from guardianeye.audio.model import SmallCNN, build_model
from guardianeye.fusion.calibrate import apply_temperature, load_calibration
from guardianeye.fusion.rules import AudioEvidence, AudioWindowScore


class AudioInferencer:
    """Load a checkpoint if it exists. Missing weights are reported, not invented."""

    def __init__(
        self,
        weights: Path | None,
        sample_rate: int,
        window_seconds: float,
        hop_seconds: float,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        fmin: float,
        fmax: float,
        calibration_path: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.device = torch.device(device)
        self.model: torch.nn.Module | None = None
        self.kind = "none"
        self.experiment = ""
        self.note = "No audio checkpoint found. Audio evidence is not reliable."
        self.temperature = 1.0
        self.calibration_fitted = False
        if weights is not None and weights.is_file():
            checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
            backend = str(checkpoint.get("backend", "small_cnn"))
            self.model = build_model(backend, int(checkpoint.get("n_mels", n_mels)))
            if isinstance(self.model, SmallCNN):
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint["state_dict"])
            self.model.to(self.device).eval()
            self.kind = str(checkpoint.get("kind", "trained"))
            self.experiment = str(checkpoint.get("experiment", ""))
            self.note = f"Loaded {backend} weights ({self.kind}) from {weights.name}."
            stored = checkpoint.get("calibration") or {}
            if isinstance(stored, dict) and stored.get("method") == "temperature":
                self.temperature = float(stored.get("temperature", 1.0))
                self.calibration_fitted = bool(stored.get("fitted", False))
        if calibration_path is not None:
            external = load_calibration(calibration_path)
            if external and external.get("method") == "temperature":
                self.temperature = float(external.get("temperature", self.temperature))
                self.calibration_fitted = bool(external.get("fitted", False))

    @property
    def reliable(self) -> bool:
        return self.model is not None

    def predict_array(self, audio: np.ndarray) -> AudioEvidence:
        """Score fixed windows of an already-resampled waveform."""
        windows = fixed_windows(audio, self.sample_rate, self.window_seconds, self.hop_seconds)
        if self.model is None:
            return AudioEvidence(present=True, reliable=False, windows=(), note=self.note)
        scored: list[AudioWindowScore] = []
        self.model.eval()
        with torch.no_grad():
            for start, clip in windows:
                mel = log_mel(
                    clip,
                    self.sample_rate,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    fmin=self.fmin,
                    fmax=self.fmax,
                )
                tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(self.device)
                logits = self.model(tensor).cpu().numpy().reshape(1, -1)
                probs = apply_temperature(logits, self.temperature)[0]
                scored.append(
                    AudioWindowScore(
                        start_s=float(start),
                        end_s=float(start + self.window_seconds),
                        gunshot_probability=float(probs[1]),
                    )
                )
        return AudioEvidence(present=True, reliable=True, windows=tuple(scored), note=self.note)

    def predict_file(self, path: Path) -> AudioEvidence:
        """Score a WAV (or other librosa-readable) file.

        When no checkpoint is loaded, skip decoding. Librosa on CPU is the
        dominant cost on free Render instances and is wasted without weights.
        """
        if self.model is None:
            return AudioEvidence(present=True, reliable=False, windows=(), note=self.note)
        audio, _sr = load_audio_file(path, self.sample_rate)
        return self.predict_array(audio)
