"""Log-mel dataset. Length equals the number of windows; nothing is dropped."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from guardianeye.audio.augment import augment_mel


class MelWindowDataset(Dataset[tuple[torch.Tensor, int]]):
    """Every manifest row is one item."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        augment: bool,
        freq_mask: int,
        time_mask: int,
        noise_std: float,
        seed: int,
    ) -> None:
        self.records = list(records)
        self.augment = augment
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.noise_std = noise_std
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        mel = np.load(record["feature"])
        if self.augment:
            rng = np.random.default_rng(self.seed + index)
            mel = augment_mel(mel, self.freq_mask, self.time_mask, self.noise_std, rng)
        tensor = torch.from_numpy(np.asarray(mel, dtype=np.float32)).unsqueeze(0)
        label = 1 if record["label"] == "gunshot" else 0
        return tensor, label


def class_weights(records: list[dict[str, Any]]) -> torch.Tensor:
    """Inverse-frequency weights. Both classes stay in the tensor even if one is rare."""
    background = sum(1 for row in records if row["label"] != "gunshot")
    gunshot = sum(1 for row in records if row["label"] == "gunshot")
    counts = np.array([max(background, 1), max(gunshot, 1)], dtype=np.float32)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)
