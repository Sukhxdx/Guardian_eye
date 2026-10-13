"""Build synthetic image+audio pairs for the fusion proxy protocol."""

from __future__ import annotations

import argparse

from guardianeye.config import load_config
from guardianeye.data.pairing import build_paired_events
from guardianeye.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic paired events")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    setup_logging()
    events = build_paired_events(load_config(args.config))
    print(f"Wrote {len(events)} synthetic proxy events")


if __name__ == "__main__":
    main()
