"""Create the tiny synthetic dataset used by tests and the demo."""

from __future__ import annotations

import argparse

from guardianeye.config import load_config, repo_root, resolve_path
from guardianeye.data.sample import generate_sample_dataset
from guardianeye.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic sample images and clips")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    audio_cfg = config["audio"]
    root = resolve_path(config["sample_root"], repo_root())
    manifest = generate_sample_dataset(
        root,
        seed=int(config.get("seed", 27)),
        sample_rate=int(audio_cfg["sample_rate"]),
        seconds=float(audio_cfg["window_seconds"]),
    )
    print(
        f"Wrote {len(manifest['images'])} images and {len(manifest['audio'])} audio rows under {root}"
    )


if __name__ == "__main__":
    main()
