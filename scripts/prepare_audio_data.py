"""Resample audio, write log-mel windows, and build site-held-out splits."""

from __future__ import annotations

import argparse
import json

from guardianeye.config import load_config
from guardianeye.data.audio_prep import prepare_audio_dataset
from guardianeye.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare audio data")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    setup_logging()
    summary = prepare_audio_dataset(load_config(args.config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
