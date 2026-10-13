"""PANNs is selectable and refuses to invent a checkpoint."""

from __future__ import annotations

import pytest

from guardianeye.audio.model import build_model


def test_missing_panns_checkpoint_is_an_error() -> None:
    with pytest.raises(FileNotFoundError, match="small_cnn"):
        build_model("panns", n_mels=64, panns_checkpoint="")
