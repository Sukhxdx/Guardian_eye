"""Resample clips, cut fixed windows, and build site-held-out splits.

Training and test sites are never mixed inside a cross-site experiment.
Background clips are kept.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from guardianeye.audio.features import (
    fixed_windows,
    load_audio_file,
    log_mel,
    spectrogram_image_to_mel,
    write_wav,
)
from guardianeye.config import repo_root, resolve_path
from guardianeye.constants import AUDIO_LABELS
from guardianeye.image.dataset import load_jsonl, stratified_split, write_jsonl

logger = logging.getLogger(__name__)

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}
SPEC_SUFFIXES = {".png", ".jpg", ".jpeg"}


def records_from_sample(sample_root: Path) -> list[dict[str, Any]]:
    """Load sample audio rows with absolute file paths."""
    rows = []
    for row in load_jsonl(sample_root / "manifest_audio.jsonl"):
        rows.append({**row, "file": str(sample_root / str(row["file"])), "id": str(row["id"])})
    return rows


def records_from_raw_tree(source_dir: Path, site: str) -> list[dict[str, Any]]:
    """Read ``site/gunshot`` and ``site/background`` trees, plus spectrogram images.

    A file's parent folder name must be ``gunshot`` or ``background``.
    Spectrogram images are stored as an audio representation (``representation``
    = ``spectrogram_image``), not as waveforms.
    """
    if not source_dir.is_dir():
        logger.info("Audio source for %s is not present at %s", site, source_dir)
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        label = next((part for part in path.parts if part in AUDIO_LABELS), None)
        if label is None:
            continue
        if path.suffix.lower() in AUDIO_SUFFIXES:
            records.append(
                {
                    "id": f"{site}_{path.stem}",
                    "file": str(path),
                    "site": site,
                    "label": label,
                    "variant": "raw",
                    "representation": "waveform",
                }
            )
        elif path.suffix.lower() in SPEC_SUFFIXES:
            records.append(
                {
                    "id": f"{site}_{path.stem}",
                    "file": str(path),
                    "site": site,
                    "label": label,
                    "variant": "spectrogram_image",
                    "representation": "spectrogram_image",
                }
            )
    return records


def collect_audio_records(config: dict[str, Any], root: Path | None = None) -> list[dict[str, Any]]:
    """Collect sample and raw audio records."""
    root = root or repo_root()
    records: list[dict[str, Any]] = []
    sample_root = resolve_path(config["sample_root"], root)
    if (
        config.get("use_sample_if_raw_missing", True)
        and (sample_root / "manifest_audio.jsonl").is_file()
    ):
        records.extend(records_from_sample(sample_root))
    for name, source in config.get("audio", {}).get("sources", {}).items():
        if not source.get("enabled", True):
            continue
        raw = records_from_raw_tree(resolve_path(source["path"], root), str(source["site"]))
        if raw:
            records.extend(raw)
        else:
            logger.info("No raw audio for %s", name)
    return records


def _window_records(
    record: dict[str, Any],
    audio_cfg: dict[str, Any],
    clip_dir: Path,
    feature_dir: Path,
) -> list[dict[str, Any]]:
    sample_rate = int(audio_cfg["sample_rate"])
    window_seconds = float(audio_cfg["window_seconds"])
    hop_seconds = float(audio_cfg["hop_seconds"])
    n_mels = int(audio_cfg["n_mels"])
    representation = str(record.get("representation", "waveform"))
    windows: list[tuple[float, np.ndarray]] = []
    waveform_ok = representation != "spectrogram_image"
    if representation == "spectrogram_image":
        n_frames = max(int(window_seconds * sample_rate / int(audio_cfg["hop_length"])), 8)
        mel = spectrogram_image_to_mel(Path(str(record["file"])), n_mels, n_frames)
        feature_path = feature_dir / f"{record['id']}.npy"
        np.save(feature_path, mel)
        return [
            {
                "id": str(record["id"]),
                "parent_id": str(record["id"]),
                "site": record["site"],
                "label": record["label"],
                "variant": record.get("variant", ""),
                "representation": "spectrogram_image",
                "start_s": 0.0,
                "feature": str(feature_path),
                "waveform": "",
                "sample_rate": sample_rate,
            }
        ]

    audio, _sr = load_audio_file(Path(str(record["file"])), sample_rate)
    windows = fixed_windows(audio, sample_rate, window_seconds, hop_seconds)
    rows: list[dict[str, Any]] = []
    for index, (start, clip) in enumerate(windows):
        clip_id = str(record["id"]) if len(windows) == 1 else f"{record['id']}_w{index:02d}"
        wav_path = clip_dir / str(record["site"]) / str(record["label"]) / f"{clip_id}.wav"
        feature_path = feature_dir / f"{clip_id}.npy"
        if waveform_ok:
            write_wav(wav_path, clip, sample_rate)
        mel = log_mel(
            clip,
            sample_rate,
            n_mels=n_mels,
            n_fft=int(audio_cfg["n_fft"]),
            hop_length=int(audio_cfg["hop_length"]),
            fmin=float(audio_cfg["fmin"]),
            fmax=float(audio_cfg["fmax"]),
        )
        np.save(feature_path, mel)
        rows.append(
            {
                "id": clip_id,
                "parent_id": str(record["id"]),
                "site": record["site"],
                "label": record["label"],
                "variant": record.get("variant", ""),
                "representation": "log_mel",
                "start_s": start,
                "feature": str(feature_path),
                "waveform": str(wav_path),
                "sample_rate": sample_rate,
            }
        )
    return rows


def _cross_site_split(
    records: list[dict[str, Any]],
    train_site: str,
    test_site: str,
    val_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    """Train and validate on ``train_site`` only. Test is the entire other site."""
    train_pool = [row for row in records if row["site"] == train_site]
    test_rows = [row for row in records if row["site"] == test_site]
    held = stratified_split(
        train_pool,
        key_fn=lambda row: str(row["label"]),
        val_ratio=val_ratio,
        test_ratio=0.0,
    )
    return {"train": held["train"], "val": held["val"], "test": test_rows}


def _in_domain_split(
    records: list[dict[str, Any]],
    site: str,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    pool = [row for row in records if row["site"] == site]
    return stratified_split(
        pool, key_fn=lambda row: str(row["label"]), val_ratio=val_ratio, test_ratio=test_ratio
    )


def build_audio_splits(
    records: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Build the four required experiments. Sites are not pooled in a headline split."""
    return {
        "belize_to_vietnam": _cross_site_split(records, "belize", "vietnam", val_ratio),
        "vietnam_to_belize": _cross_site_split(records, "vietnam", "belize", val_ratio),
        "in_domain_belize": _in_domain_split(records, "belize", val_ratio, test_ratio),
        "in_domain_vietnam": _in_domain_split(records, "vietnam", val_ratio, test_ratio),
    }


def prepare_audio_dataset(config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Write windows, log-mel features, and site-held-out split files."""
    root = root or repo_root()
    audio_cfg = config["audio"]
    records = collect_audio_records(config, root)
    processed = resolve_path(config["processed_root"], root) / "audio"
    if processed.exists():
        shutil.rmtree(processed)
    clip_dir = processed / "clips"
    feature_dir = processed / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for record in records:
        manifest.extend(_window_records(record, audio_cfg, clip_dir, feature_dir))
    write_jsonl(processed / "manifest.jsonl", manifest)
    supervised = [row for row in manifest if row.get("representation") == "log_mel"]
    splits = build_audio_splits(
        supervised,
        val_ratio=float(audio_cfg.get("val_ratio", 0.15)),
        test_ratio=float(audio_cfg.get("test_ratio", 0.15)),
    )
    split_dir = processed / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    summary_splits: dict[str, dict[str, int]] = {}
    for name, parts in splits.items():
        payload = {part: [row["id"] for row in rows] for part, rows in parts.items()}
        write_jsonl(
            split_dir / f"{name}.jsonl",
            [
                {
                    "experiment": name,
                    "split": part,
                    "id": row["id"],
                    "site": row["site"],
                    "label": row["label"],
                }
                for part, rows in parts.items()
                for row in rows
            ],
        )
        (split_dir / f"{name}.ids.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        summary_splits[name] = {part: len(rows) for part, rows in parts.items()}
    by_site: dict[str, dict[str, int]] = {}
    for row in manifest:
        site_counts = by_site.setdefault(str(row["site"]), {"gunshot": 0, "background": 0})
        label = str(row["label"])
        if label in site_counts:
            site_counts[label] += 1
    logger.info("Prepared %s audio windows into %s", len(manifest), processed)
    return {"windows": len(manifest), "by_site": by_site, "splits": summary_splits}
