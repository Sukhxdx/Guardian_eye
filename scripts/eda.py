"""Regenerate class-distribution plots and imbalance ratios."""

from __future__ import annotations

import argparse
import json

from guardianeye.config import load_config
from guardianeye.data.eda import run_eda
from guardianeye.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Write EDA plots and imbalance ratios")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    setup_logging()
    payload = run_eda(load_config(args.config))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
