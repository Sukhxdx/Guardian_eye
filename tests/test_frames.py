"""Video frames enter the image branch with a domain tag."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from guardianeye.data.frames import extract_frames


def test_extract_frames_tags_the_domain(tmp_path: Path) -> None:
    cv2 = __import__("cv2")
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    assert writer.isOpened()
    for index in range(10):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[..., 1] = index * 10
        writer.write(frame)
    writer.release()
    rows = extract_frames(video, tmp_path / "frames", fps=2, domain="aerial_thermal")
    assert rows
    assert all(row["domain"] == "aerial_thermal" for row in rows)
    assert all(row["objects"] == [] for row in rows)
    assert len(rows) < 10
