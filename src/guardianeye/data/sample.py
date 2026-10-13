"""Deterministic tiny images and clips so the pipeline runs without real datasets.

Generated media is synthetic. It is not camera-trap or gunshot field data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from guardianeye.audio.features import write_wav
from guardianeye.image.dataset import write_jsonl

SAMPLE_WIDTH = 320
SAMPLE_HEIGHT = 240


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rgb_background(rng: np.random.Generator) -> np.ndarray:
    image = np.zeros((SAMPLE_HEIGHT, SAMPLE_WIDTH, 3), dtype=np.uint8)
    image[..., 0] = rng.integers(18, 46, size=(SAMPLE_HEIGHT, SAMPLE_WIDTH), dtype=np.uint8)
    image[..., 1] = rng.integers(48, 92, size=(SAMPLE_HEIGHT, SAMPLE_WIDTH), dtype=np.uint8)
    image[..., 2] = rng.integers(16, 36, size=(SAMPLE_HEIGHT, SAMPLE_WIDTH), dtype=np.uint8)
    return image


def _thermal_background(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(12, 36, size=(SAMPLE_HEIGHT, SAMPLE_WIDTH), dtype=np.uint8)


def _paint_box(
    image: np.ndarray, box: tuple[int, int, int, int], value: int | tuple[int, int, int]
) -> None:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, x1), min(image.shape[1], x2)))
    y1, y2 = sorted((max(0, y1), min(image.shape[0], y2)))
    image[y1:y2, x1:x2] = value


def _box_for(kind: str, index: int, small: bool) -> tuple[int, int, int, int]:
    if small:
        return (12 + index, 18 + index, 26 + index, 40 + index)
    if kind == "human":
        return (40 + 15 * index, 30, 70 + 15 * index, 180)
    if kind == "vehicle":
        return (30 + 10 * index, 120, 180 + 10 * index, 190)
    return (90 + 8 * index, 70, 150 + 8 * index, 150)


def _image_plan() -> list[dict[str, Any]]:
    """Fixed roster. Empty frames stay the majority in both domains."""
    plan: list[dict[str, Any]] = []
    specs = [
        ("human", False),
        ("human", True),
        ("vehicle", False),
        ("animal", False),
        ("animal", False),
        ("empty", False),
        ("empty", False),
        ("empty", False),
        ("empty", False),
        ("empty", False),
        ("empty", False),
    ]
    for domain, source in (
        ("aerial_thermal", "sample_birdsai"),
        ("camera_trap_rgb", "sample_camera_trap"),
    ):
        prefix = "at" if domain == "aerial_thermal" else "ct"
        for index, (kind, small) in enumerate(specs):
            plan.append(
                {
                    "id": f"{prefix}_{index:03d}",
                    "domain": domain,
                    "source": source,
                    "kind": kind,
                    "small": small and kind == "human",
                }
            )
    return plan


def _audio_plan() -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for site in ("belize", "vietnam"):
        for index in range(4):
            plan.append(
                {
                    "id": f"{site}_gun_{index:03d}",
                    "site": site,
                    "label": "gunshot",
                    "variant": "impulse_shot",
                }
            )
        for index in range(8):
            plan.append(
                {
                    "id": f"{site}_bg_{index:03d}",
                    "site": site,
                    "label": "background",
                    "variant": "ambient",
                }
            )
        plan.append(
            {
                "id": f"{site}_bg_thunder",
                "site": site,
                "label": "background",
                "variant": "thunder",
            }
        )
        plan.append(
            {
                "id": f"{site}_bg_snap",
                "site": site,
                "label": "background",
                "variant": "snap",
            }
        )
    return plan


def _synth_gunshot(sample_rate: int, seconds: float, rng: np.random.Generator) -> np.ndarray:
    n = int(sample_rate * seconds)
    audio = rng.normal(0.0, 0.008, size=n).astype(np.float32)
    start = int(rng.integers(int(0.25 * sample_rate), int(0.55 * sample_rate)))
    length = int(0.18 * sample_rate)
    t = np.arange(length, dtype=np.float32)
    decay = np.exp(-t / (0.015 * sample_rate))
    burst = decay * rng.normal(0.0, 1.0, size=length).astype(np.float32)
    thump = decay * np.sin(2 * np.pi * 140.0 * t / sample_rate).astype(np.float32)
    audio[start : start + length] += 0.85 * burst + 0.35 * thump
    return np.clip(audio, -1.0, 1.0)


def _synth_background(
    sample_rate: int,
    seconds: float,
    rng: np.random.Generator,
    variant: str,
) -> np.ndarray:
    n = int(sample_rate * seconds)
    audio = rng.normal(0.0, 0.02, size=n).astype(np.float32)
    t = np.arange(n, dtype=np.float32) / sample_rate
    audio += (0.01 * np.sin(2 * np.pi * 3.0 * t)).astype(np.float32)
    if variant == "thunder":
        start = int(0.4 * sample_rate)
        length = int(1.2 * sample_rate)
        burst = rng.normal(0.0, 0.08, size=length).astype(np.float32)
        envelope = np.linspace(0.2, 1.0, length, dtype=np.float32) * np.linspace(
            1.0, 0.1, length, dtype=np.float32
        )
        audio[start : start + length] += burst * envelope
    elif variant == "snap":
        at = int(rng.integers(sample_rate, 2 * sample_rate))
        audio[at : at + 8] += 0.25
    return np.clip(audio, -1.0, 1.0)


def generate_sample_dataset(
    root: Path, seed: int = 27, sample_rate: int = 16000, seconds: float = 3.0
) -> dict[str, Any]:
    """Write the sample raw tree and a manifest. Existing files are overwritten."""
    root = Path(root)
    image_dir = root / "raw" / "images"
    audio_dir = root / "raw" / "audio"
    rng = _rng(seed)
    image_rows: list[dict[str, Any]] = []

    for spec in _image_plan():
        domain = str(spec["domain"])
        folder = image_dir / domain
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{spec['id']}.png"
        kind = str(spec["kind"])
        if domain == "aerial_thermal":
            canvas = _thermal_background(rng)
            objects: list[dict[str, Any]] = []
            if kind != "empty":
                box = _box_for(kind, int(str(spec["id"])[-3:]), bool(spec["small"]))
                _paint_box(canvas, box, 235)
                objects.append({"cls": kind, "bbox_xyxy": list(box)})
            Image.fromarray(canvas, mode="L").save(path)
        else:
            canvas = _rgb_background(rng)
            objects = []
            if kind != "empty":
                box = _box_for(kind, int(str(spec["id"])[-3:]), bool(spec["small"]))
                color = {
                    "human": (210, 120, 40),
                    "vehicle": (40, 70, 180),
                    "animal": (30, 140, 50),
                }[kind]
                _paint_box(canvas, box, color)
                objects.append({"cls": kind, "bbox_xyxy": list(box)})
            Image.fromarray(canvas, mode="RGB").save(path)
        image_rows.append(
            {
                "id": spec["id"],
                "file": str(path.relative_to(root)),
                "domain": domain,
                "source": spec["source"],
                "width": SAMPLE_WIDTH,
                "height": SAMPLE_HEIGHT,
                "objects": objects,
                "small_object": bool(spec["small"]),
            }
        )

    audio_rows: list[dict[str, Any]] = []
    for spec in _audio_plan():
        site = str(spec["site"])
        label = str(spec["label"])
        folder = audio_dir / site / label
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{spec['id']}.wav"
        clip_rng = _rng(seed + len(audio_rows) + 1)
        if label == "gunshot":
            audio = _synth_gunshot(sample_rate, seconds, clip_rng)
        else:
            audio = _synth_background(sample_rate, seconds, clip_rng, str(spec["variant"]))
        write_wav(path, audio, sample_rate)
        audio_rows.append(
            {
                "id": spec["id"],
                "file": str(path.relative_to(root)),
                "site": site,
                "label": label,
                "variant": spec["variant"],
                "sample_rate": sample_rate,
                "seconds": seconds,
            }
        )

    # A tiny spectrogram image stands in for the Vietnam spectrogram release.
    spec_path = audio_dir / "vietnam" / "spectrograms" / "vietnam_spec_000.png"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_img = rng.integers(0, 255, size=(64, 96), dtype=np.uint8)
    Image.fromarray(spec_img, mode="L").save(spec_path)
    audio_rows.append(
        {
            "id": "vietnam_spec_000",
            "file": str(spec_path.relative_to(root)),
            "site": "vietnam",
            "label": "gunshot",
            "variant": "spectrogram_image",
            "representation": "spectrogram_image",
        }
    )

    manifest = {"images": image_rows, "audio": audio_rows, "seed": seed, "kind": "synthetic_sample"}
    write_jsonl(root / "manifest_images.jsonl", image_rows)
    write_jsonl(root / "manifest_audio.jsonl", audio_rows)
    return manifest
