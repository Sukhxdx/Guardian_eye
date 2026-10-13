"""Convert image sources into per-domain YOLO trees."""

from __future__ import annotations

import argparse
import json

from guardianeye.config import load_config
from guardianeye.data.image_prep import prepare_image_dataset
from guardianeye.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare image data")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    setup_logging()
    summary = prepare_image_dataset(load_config(args.config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
