"""Deterministic tiny images and clips so the pipeline runs without real datasets.

Generated media is synthetic. It is not camera-trap or gunshot field data.
The roster is intentionally varied for demos while keeping empty frames and
background audio as the majority classes.
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


def _rgb_background(rng: np.random.Generator, style: str = "day") -> np.ndarray:
    image = np.zeros((SAMPLE_HEIGHT, SAMPLE_WIDTH, 3), dtype=np.uint8)
    if style == "night":
        ranges = ((4, 22), (8, 28), (10, 30))
    elif style == "dusk":
        ranges = ((30, 70), (20, 45), (15, 35))
    else:
        ranges = ((18, 46), (48, 92), (16, 36))
    for channel, (lo, hi) in enumerate(ranges):
        image[..., channel] = rng.integers(
            lo, hi, size=(SAMPLE_HEIGHT, SAMPLE_WIDTH), dtype=np.uint8
        )
    return image


def _thermal_background(rng: np.random.Generator, style: str = "cool") -> np.ndarray:
    if style == "warm_canopy":
        return rng.integers(40, 90, size=(SAMPLE_HEIGHT, SAMPLE_WIDTH), dtype=np.uint8)
    if style == "hot_spot":
        canvas = rng.integers(20, 50, size=(SAMPLE_HEIGHT, SAMPLE_WIDTH), dtype=np.uint8)
        yy, xx = np.ogrid[:SAMPLE_HEIGHT, :SAMPLE_WIDTH]
        mask = (xx - 220) ** 2 + (yy - 60) ** 2 < 35**2
        canvas[mask] = rng.integers(160, 210, size=int(mask.sum()), dtype=np.uint8)
        return canvas
    return rng.integers(12, 36, size=(SAMPLE_HEIGHT, SAMPLE_WIDTH), dtype=np.uint8)


def _paint_box(
    image: np.ndarray, box: tuple[int, int, int, int], value: int | tuple[int, int, int]
) -> None:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, x1), min(image.shape[1], x2)))
    y1, y2 = sorted((max(0, y1), min(image.shape[0], y2)))
    image[y1:y2, x1:x2] = value


def _box_for(kind: str, index: int, small: bool, shape: str = "default") -> tuple[int, int, int, int]:
    if small:
        return (12 + index * 3, 18 + index * 2, 28 + index * 3, 42 + index * 2)
    if kind == "human":
        if shape == "crouched":
            return (50 + 12 * index, 90, 95 + 12 * index, 170)
        if shape == "edge":
            return (280, 40, 310, 160)
        return (40 + 15 * index, 30, 70 + 15 * index, 180)
    if kind == "vehicle":
        if shape == "truck":
            return (20 + 8 * index, 100, 210 + 8 * index, 195)
        if shape == "partial":
            return (250, 130, 319, 185)
        return (30 + 10 * index, 120, 180 + 10 * index, 190)
    if shape == "herd":
        return (60 + 40 * (index % 3), 90, 100 + 40 * (index % 3), 140)
    return (90 + 8 * index, 70, 150 + 8 * index, 150)


def _image_plan() -> list[dict[str, Any]]:
    """Varied roster. Empty frames stay the majority in both domains."""
    plan: list[dict[str, Any]] = []
    # 18 frames / domain: empty 10 > human 3, vehicle 2, animal 3
    specs: list[dict[str, Any]] = [
        {"kind": "human", "small": False, "shape": "default", "style": "day", "scene": "patrol_human"},
        {"kind": "human", "small": True, "shape": "default", "style": "day", "scene": "canopy_small_human"},
        {"kind": "human", "small": False, "shape": "crouched", "style": "night", "scene": "night_human"},
        {"kind": "vehicle", "small": False, "shape": "default", "style": "dusk", "scene": "road_vehicle"},
        {"kind": "vehicle", "small": False, "shape": "truck", "style": "day", "scene": "truck"},
        {"kind": "animal", "small": False, "shape": "default", "style": "day", "scene": "single_animal"},
        {"kind": "animal", "small": False, "shape": "herd", "style": "dusk", "scene": "herd_animal"},
        {"kind": "animal", "small": True, "shape": "default", "style": "day", "scene": "small_animal"},
        {"kind": "empty", "small": False, "shape": "default", "style": "day", "scene": "empty_clearing"},
        {"kind": "empty", "small": False, "shape": "default", "style": "night", "scene": "empty_night"},
        {"kind": "empty", "small": False, "shape": "default", "style": "dusk", "scene": "empty_dusk"},
        {"kind": "empty", "small": False, "shape": "default", "style": "day", "scene": "empty_a"},
        {"kind": "empty", "small": False, "shape": "default", "style": "day", "scene": "empty_b"},
        {"kind": "empty", "small": False, "shape": "default", "style": "day", "scene": "empty_c"},
        {"kind": "empty", "small": False, "shape": "default", "style": "dusk", "scene": "empty_d"},
        {"kind": "empty", "small": False, "shape": "default", "style": "night", "scene": "empty_e"},
        {"kind": "empty", "small": False, "shape": "default", "style": "day", "scene": "empty_f"},
        {"kind": "empty", "small": False, "shape": "default", "style": "day", "scene": "empty_g"},
    ]
    for domain, source in (
        ("aerial_thermal", "sample_birdsai"),
        ("camera_trap_rgb", "sample_camera_trap"),
    ):
        prefix = "at" if domain == "aerial_thermal" else "ct"
        thermal_styles = ["cool", "warm_canopy", "hot_spot", "cool", "warm_canopy"]
        for index, spec in enumerate(specs):
            row = {
                "id": f"{prefix}_{index:03d}",
                "domain": domain,
                "source": source,
                "kind": spec["kind"],
                "small": bool(spec["small"]) and spec["kind"] in {"human", "animal"},
                "shape": spec["shape"],
                "style": spec["style"],
                "scene": spec["scene"],
            }
            if domain == "aerial_thermal":
                row["style"] = thermal_styles[index % len(thermal_styles)]
            plan.append(row)
    return plan


def _audio_plan() -> list[dict[str, Any]]:
    """Gunshot variants plus richer background distractors; background stays majority."""
    plan: list[dict[str, Any]] = []
    gun_variants = [
        "impulse_shot",
        "impulse_shot",
        "impulse_shot",
        "impulse_shot",
        "distant_shot",
        "distant_shot",
        "double_tap",
        "double_tap",
    ]
    bg_variants = (
        ["ambient"] * 8
        + ["thunder", "thunder", "snap", "snap", "rain", "rain", "insects", "insects", "wind", "wind"]
    )
    for site in ("belize", "vietnam"):
        for index, variant in enumerate(gun_variants):
            plan.append(
                {
                    "id": f"{site}_gun_{index:03d}",
                    "site": site,
                    "label": "gunshot",
                    "variant": variant,
                }
            )
        for index, variant in enumerate(bg_variants):
            plan.append(
                {
                    "id": f"{site}_bg_{index:03d}_{variant}",
                    "site": site,
                    "label": "background",
                    "variant": variant,
                }
            )
    return plan


def _synth_gunshot(
    sample_rate: int, seconds: float, rng: np.random.Generator, variant: str = "impulse_shot"
) -> np.ndarray:
    n = int(sample_rate * seconds)
    audio = rng.normal(0.0, 0.008, size=n).astype(np.float32)
    starts = [int(rng.integers(int(0.25 * sample_rate), int(0.55 * sample_rate)))]
    length = int(0.18 * sample_rate)
    amp = 0.85
    freq = 140.0
    if variant == "distant_shot":
        length = int(0.28 * sample_rate)
        amp = 0.35
        freq = 90.0
        audio = rng.normal(0.0, 0.015, size=n).astype(np.float32)
    elif variant == "double_tap":
        starts.append(starts[0] + int(0.22 * sample_rate))
        length = int(0.12 * sample_rate)
        amp = 0.75
    for start in starts:
        if start + length > n:
            continue
        t = np.arange(length, dtype=np.float32)
        decay = np.exp(-t / (0.015 * sample_rate))
        burst = decay * rng.normal(0.0, 1.0, size=length).astype(np.float32)
        thump = decay * np.sin(2 * np.pi * freq * t / sample_rate).astype(np.float32)
        audio[start : start + length] += amp * burst + 0.35 * thump
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
        at = int(rng.integers(sample_rate // 2, 2 * sample_rate))
        audio[at : at + 8] += 0.25
    elif variant == "rain":
        audio += rng.normal(0.0, 0.04, size=n).astype(np.float32)
        drops = rng.integers(0, n, size=40)
        audio[drops] += 0.08
    elif variant == "insects":
        audio += (0.03 * np.sin(2 * np.pi * 4200.0 * t)).astype(np.float32)
        audio += (0.02 * np.sin(2 * np.pi * 5100.0 * t + 0.4)).astype(np.float32)
    elif variant == "wind":
        gust = np.sin(2 * np.pi * 0.7 * t).astype(np.float32)
        audio += 0.05 * gust * rng.normal(0.0, 1.0, size=n).astype(np.float32)
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
        style = str(spec.get("style", "day"))
        shape = str(spec.get("shape", "default"))
        objects: list[dict[str, Any]] = []
        if domain == "aerial_thermal":
            canvas = _thermal_background(rng, style=style)
            if kind != "empty":
                box = _box_for(kind, int(str(spec["id"])[-3:]), bool(spec["small"]), shape)
                _paint_box(canvas, box, 235)
                objects.append({"cls": kind, "bbox_xyxy": list(box)})
                if shape == "herd":
                    box2 = _box_for(kind, int(str(spec["id"])[-3:]) + 1, False, "herd")
                    _paint_box(canvas, box2, 220)
                    objects.append({"cls": kind, "bbox_xyxy": list(box2)})
            Image.fromarray(canvas, mode="L").save(path)
        else:
            canvas = _rgb_background(rng, style=style if style in {"day", "night", "dusk"} else "day")
            if kind != "empty":
                box = _box_for(kind, int(str(spec["id"])[-3:]), bool(spec["small"]), shape)
                color = {
                    "human": (210, 120, 40),
                    "vehicle": (40, 70, 180),
                    "animal": (30, 140, 50),
                }[kind]
                if style == "night":
                    color = tuple(max(20, c // 2) for c in color)  # type: ignore[assignment]
                _paint_box(canvas, box, color)
                objects.append({"cls": kind, "bbox_xyxy": list(box)})
                if shape == "herd":
                    box2 = _box_for(kind, int(str(spec["id"])[-3:]) + 2, False, "herd")
                    _paint_box(canvas, box2, (20, 120, 40))
                    objects.append({"cls": kind, "bbox_xyxy": list(box2)})
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
                "scene": spec.get("scene", kind),
                "style": style,
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
        variant = str(spec["variant"])
        if label == "gunshot":
            audio = _synth_gunshot(sample_rate, seconds, clip_rng, variant)
        else:
            audio = _synth_background(sample_rate, seconds, clip_rng, variant)
        write_wav(path, audio, sample_rate)
        audio_rows.append(
            {
                "id": spec["id"],
                "file": str(path.relative_to(root)),
                "site": site,
                "label": label,
                "variant": variant,
                "sample_rate": sample_rate,
                "seconds": seconds,
            }
        )

    # Spectrogram-image stand-ins for the Vietnam spectrogram release (a few variants).
    for i, label in enumerate(("gunshot", "gunshot", "background")):
        spec_path = audio_dir / "vietnam" / "spectrograms" / f"vietnam_spec_{i:03d}.png"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        if label == "gunshot":
            spec_img = rng.integers(40, 255, size=(64, 96), dtype=np.uint8)
            spec_img[:, 40:48] = 255
        else:
            spec_img = rng.integers(0, 80, size=(64, 96), dtype=np.uint8)
        Image.fromarray(spec_img, mode="L").save(spec_path)
        audio_rows.append(
            {
                "id": f"vietnam_spec_{i:03d}",
                "file": str(spec_path.relative_to(root)),
                "site": "vietnam",
                "label": label,
                "variant": "spectrogram_image",
                "representation": "spectrogram_image",
            }
        )

    manifest = {"images": image_rows, "audio": audio_rows, "seed": seed, "kind": "synthetic_sample"}
    write_jsonl(root / "manifest_images.jsonl", image_rows)
    write_jsonl(root / "manifest_audio.jsonl", audio_rows)
    _write_showcase_index(root, image_rows, audio_rows)
    return manifest


def _write_showcase_index(
    root: Path, images: list[dict[str, Any]], audio: list[dict[str, Any]]
) -> None:
    """Named demo pairs for the Gradio sample loader."""
    by_img = {(row["domain"], row.get("scene") or ""): row for row in images}
    # also index by class for fallbacks
    by_class: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in images:
        by_class.setdefault((str(row["domain"]), str(row["objects"][0]["cls"]) if row["objects"] else "empty"), []).append(row)
    by_audio: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in audio:
        if row.get("representation") == "spectrogram_image":
            continue
        by_audio.setdefault((str(row["site"]), str(row["variant"])), []).append(row)
        by_audio.setdefault((str(row["site"]), str(row["label"])), []).append(row)

    def pick_img(domain: str, scene: str, cls: str) -> dict[str, Any] | None:
        row = by_img.get((domain, scene))
        if row:
            return row
        opts = by_class.get((domain, cls)) or []
        return opts[0] if opts else None

    def pick_aud(site: str, variant_or_label: str) -> dict[str, Any] | None:
        opts = by_audio.get((site, variant_or_label)) or []
        return opts[0] if opts else None

    catalog = [
        {
            "id": "ct_human_gunshot",
            "title": "Camera-trap human + gunshot",
            "domain": "camera_trap_rgb",
            "image_id": (pick_img("camera_trap_rgb", "patrol_human", "human") or {}).get("id"),
            "audio_id": (pick_aud("belize", "impulse_shot") or {}).get("id"),
        },
        {
            "id": "ct_night_human_distant",
            "title": "Night human + distant shot",
            "domain": "camera_trap_rgb",
            "image_id": (pick_img("camera_trap_rgb", "night_human", "human") or {}).get("id"),
            "audio_id": (pick_aud("belize", "distant_shot") or {}).get("id"),
        },
        {
            "id": "ct_vehicle_double",
            "title": "Vehicle + double-tap gunshot",
            "domain": "camera_trap_rgb",
            "image_id": (pick_img("camera_trap_rgb", "truck", "vehicle") or {}).get("id"),
            "audio_id": (pick_aud("vietnam", "double_tap") or {}).get("id"),
        },
        {
            "id": "ct_animal_insects",
            "title": "Animal + insect background",
            "domain": "camera_trap_rgb",
            "image_id": (pick_img("camera_trap_rgb", "herd_animal", "animal") or {}).get("id"),
            "audio_id": (pick_aud("belize", "insects") or {}).get("id"),
        },
        {
            "id": "ct_empty_thunder",
            "title": "Empty frame + thunder distractor",
            "domain": "camera_trap_rgb",
            "image_id": (pick_img("camera_trap_rgb", "empty_clearing", "empty") or {}).get("id"),
            "audio_id": (pick_aud("vietnam", "thunder") or {}).get("id"),
        },
        {
            "id": "ct_empty_rain",
            "title": "Empty night + rain",
            "domain": "camera_trap_rgb",
            "image_id": (pick_img("camera_trap_rgb", "empty_night", "empty") or {}).get("id"),
            "audio_id": (pick_aud("belize", "rain") or {}).get("id"),
        },
        {
            "id": "at_human_gunshot",
            "title": "Aerial thermal human + gunshot",
            "domain": "aerial_thermal",
            "image_id": (pick_img("aerial_thermal", "patrol_human", "human") or {}).get("id"),
            "audio_id": (pick_aud("vietnam", "impulse_shot") or {}).get("id"),
        },
        {
            "id": "at_vehicle_wind",
            "title": "Aerial thermal vehicle + wind",
            "domain": "aerial_thermal",
            "image_id": (pick_img("aerial_thermal", "road_vehicle", "vehicle") or {}).get("id"),
            "audio_id": (pick_aud("belize", "wind") or {}).get("id"),
        },
        {
            "id": "at_animal_snap",
            "title": "Aerial thermal animal + branch snap",
            "domain": "aerial_thermal",
            "image_id": (pick_img("aerial_thermal", "single_animal", "animal") or {}).get("id"),
            "audio_id": (pick_aud("vietnam", "snap") or {}).get("id"),
        },
        {
            "id": "at_empty_hotspot",
            "title": "Thermal empty (hot canopy) + ambient",
            "domain": "aerial_thermal",
            "image_id": (pick_img("aerial_thermal", "empty_clearing", "empty") or {}).get("id"),
            "audio_id": (pick_aud("belize", "ambient") or {}).get("id"),
        },
    ]
    catalog = [row for row in catalog if row.get("image_id") and row.get("audio_id")]
    write_jsonl(root / "showcase_pairs.jsonl", catalog)
