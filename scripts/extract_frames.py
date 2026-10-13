"""Sample frames from a video into the image branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from guardianeye.data.frames import extract_frames
from guardianeye.image.dataset import write_jsonl
from guardianeye.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--domain", choices=["aerial_thermal", "camera_trap_rgb"], required=True)
    args = parser.parse_args()
    setup_logging()
    rows = extract_frames(args.video, args.out, args.fps, args.domain)
    write_jsonl(args.out / "frames.jsonl", rows)
    print(json.dumps({"frames": len(rows), "domain": args.domain}))


if __name__ == "__main__":
    main()
