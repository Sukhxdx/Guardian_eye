"""Build synthetic image+audio events for the fusion proxy protocol.

Pairs are constructed from labels. They are not time-aligned field incidents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from guardianeye.config import repo_root, resolve_path
from guardianeye.image.dataset import load_jsonl, write_jsonl

# Expected levels assume a perfect indicator detector. They are scenario labels
# for the proxy protocol, not model scores.
SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "name": "human_gunshot",
        "image_class": "human",
        "audio_label": "gunshot",
        "expected_level": "HIGH",
        "failure_tags": [],
        "small_only": False,
    },
    {
        "name": "vehicle_gunshot",
        "image_class": "vehicle",
        "audio_label": "gunshot",
        "expected_level": "HIGH",
        "failure_tags": [],
        "small_only": False,
    },
    {
        "name": "animal_background",
        "image_class": "animal",
        "audio_label": "background",
        "expected_level": "LOW",
        "failure_tags": [],
        "small_only": False,
    },
    {
        "name": "empty_impulse",
        "image_class": "empty",
        "audio_label": "background",
        "audio_variant": "thunder",
        "expected_level": "LOW",
        "failure_tags": ["impulse_noise"],
        "small_only": False,
    },
    {
        "name": "vehicle_background",
        "image_class": "vehicle",
        "audio_label": "background",
        "expected_level": "MEDIUM",
        "failure_tags": [],
        "small_only": False,
    },
    {
        "name": "human_background",
        "image_class": "human",
        "audio_label": "background",
        "expected_level": "MEDIUM",
        "failure_tags": [],
        "small_only": False,
    },
    {
        "name": "empty_gunshot",
        "image_class": "empty",
        "audio_label": "gunshot",
        "expected_level": "MEDIUM",
        "failure_tags": [],
        "small_only": False,
    },
    {
        "name": "small_human_gunshot",
        "image_class": "human",
        "audio_label": "gunshot",
        "expected_level": "HIGH",
        "failure_tags": ["canopy_or_small_object"],
        "small_only": True,
    },
)


def _pick(rows: list[dict[str, Any]], used: set[str]) -> dict[str, Any] | None:
    for row in rows:
        if str(row["id"]) not in used:
            return row
    return rows[0] if rows else None


def build_paired_events(config: dict[str, Any], root: Path | None = None) -> list[dict[str, Any]]:
    """Pair processed images and clips according to ``SCENARIOS``.

    Every row is marked ``protocol = synthetic_proxy``.
    """
    root = root or repo_root()
    processed = resolve_path(config["processed_root"], root)
    images = load_jsonl(processed / "images" / "manifest.jsonl")
    clips = [
        row
        for row in load_jsonl(processed / "audio" / "manifest.jsonl")
        if row.get("representation") != "spectrogram_image" and row.get("waveform")
    ]
    events: list[dict[str, Any]] = []
    used_images: set[str] = set()
    used_audio: set[str] = set()
    for domain in ("camera_trap_rgb", "aerial_thermal"):
        domain_images = [row for row in images if row["domain"] == domain]
        for scenario in SCENARIOS:
            image_pool = [
                row
                for row in domain_images
                if row["primary_class"] == scenario["image_class"]
                and (bool(row.get("small_object")) if scenario["small_only"] else True)
            ]
            if scenario["small_only"]:
                image_pool = [row for row in image_pool if row.get("small_object")]
            audio_pool = [row for row in clips if row["label"] == scenario["audio_label"]]
            variant = scenario.get("audio_variant")
            if variant:
                variant_pool = [row for row in audio_pool if row.get("variant") == variant]
                if variant_pool:
                    audio_pool = variant_pool
            image_row = _pick(image_pool, used_images)
            audio_row = _pick(audio_pool, used_audio)
            if image_row is None or audio_row is None:
                continue
            used_images.add(str(image_row["id"]))
            used_audio.add(str(audio_row["id"]))
            events.append(
                {
                    "event_id": f"proxy_{domain}_{scenario['name']}",
                    "protocol": "synthetic_proxy",
                    "scenario": scenario["name"],
                    "expected_level": scenario["expected_level"],
                    "failure_tags": list(scenario["failure_tags"]),
                    "domain": domain,
                    "image_id": image_row["id"],
                    "image_path": image_row["image"],
                    "audio_id": audio_row["id"],
                    "audio_path": audio_row["waveform"],
                    "audio_site": audio_row["site"],
                    "image_timestamp_s": 0.0,
                    "audio_window_start_s": float(audio_row.get("start_s", 0.0)),
                    "audio_window_end_s": float(audio_row.get("start_s", 0.0))
                    + float(config.get("audio", {}).get("window_seconds", 3.0)),
                    "note": "Constructed pair for the synthetic proxy protocol. Not a field incident.",
                }
            )
    out = resolve_path(
        config.get("paired", {}).get("output", "data/processed/paired/events.jsonl"), root
    )
    write_jsonl(out, events)
    return events
