"""Audio classifiers. The default is a small log-mel CNN.

PANNs / YAMNet-style embeddings are optional and only load from a local
checkpoint. This project does not invent embedding weights.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Three-block CNN on a single-channel log-mel spectrogram."""

    def __init__(self, n_mels: int = 64) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Linear(64 * 4 * 4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits ``(background, gunshot)``."""
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)
        encoded = self.features(inputs)
        return self.head(encoded.flatten(1))


class EmbeddingHead(nn.Module):
    """Linear classifier on a frozen embedding module."""

    def __init__(self, encoder: nn.Module, embed_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)
        with torch.no_grad():
            embedding = self.encoder(inputs)
        if isinstance(embedding, tuple | list):
            embedding = embedding[0]
        return self.head(embedding)


def load_panns_classifier(checkpoint: str | Path) -> EmbeddingHead:
    """Load a local embedding checkpoint.

    The file must be a ``torch.save`` dict with ``encoder`` (an ``nn.Module``)
    and ``embed_dim`` (int). Download PANNs yourself; no URL is assumed here.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"PANNs checkpoint not found at {path}. Set audio.panns_checkpoint "
            "to a local file, or set backend to small_cnn. GuardianEye does not "
            "download PANNs or YAMNet weights."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "encoder" not in payload or "embed_dim" not in payload:
        raise ValueError("PANNs checkpoint must be a dict with keys 'encoder' and 'embed_dim'.")
    encoder = payload["encoder"]
    if not isinstance(encoder, nn.Module):
        raise ValueError("PANNs checkpoint 'encoder' must be an nn.Module")
    return EmbeddingHead(encoder, int(payload["embed_dim"]))


def build_model(backend: str, n_mels: int, panns_checkpoint: str = "") -> nn.Module:
    """Construct the configured audio model."""
    if backend == "small_cnn":
        return SmallCNN(n_mels=n_mels)
    if backend == "panns":
        return load_panns_classifier(panns_checkpoint)
    raise ValueError(f"Unknown audio backend: {backend}")
