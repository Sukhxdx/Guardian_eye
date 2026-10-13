"""Extract still frames from video. Frames join the image branch."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def extract_frames(
    video_path: Path,
    out_dir: Path,
    fps: float,
    domain: str,
) -> list[dict[str, Any]]:
    """Sample frames at ``fps`` and return manifest rows.

    ``domain`` is stored on every row so later evaluation does not pool
    aerial-thermal frames with camera-trap frames.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    native = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if native <= 0:
        native = fps
    step = max(int(round(native / fps)), 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    index = 0
    saved = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            timestamp = index / native
            name = f"{video_path.stem}_{saved:05d}.jpg"
            dest = out_dir / name
            cv2.imwrite(str(dest), frame)
            height, width = frame.shape[:2]
            rows.append(
                {
                    "id": f"{video_path.stem}_{saved:05d}",
                    "file": str(dest),
                    "domain": domain,
                    "source": "video_frames",
                    "width": int(width),
                    "height": int(height),
                    "timestamp_s": timestamp,
                    "objects": [],
                }
            )
            saved += 1
        index += 1
    capture.release()
    return rows
