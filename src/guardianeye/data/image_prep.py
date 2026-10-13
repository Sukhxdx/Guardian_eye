"""Unify BIRDSAI, Caltech Camera Traps, Snapshot Serengeti, and sample images.

Outputs are YOLO directories split **per domain**. Empty images are kept.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from guardianeye.config import repo_root, resolve_path
from guardianeye.constants import BOX_CLASSES, DOMAINS
from guardianeye.image.dataset import (
    load_jsonl,
    primary_class,
    stratified_split,
    write_jsonl,
    write_ultralytics_yaml,
    write_yolo_label,
)
from guardianeye.image.thermal import apply_thermal_representation

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _map_label(raw: str, label_map: dict[str, str]) -> str | None:
    key = raw.strip().lower()
    if key not in label_map:
        return None
    return label_map[key]


def _read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def records_from_sample(sample_root: Path) -> list[dict[str, Any]]:
    """Load the sample manifest and point ``file`` at absolute paths."""
    rows = load_jsonl(sample_root / "manifest_images.jsonl")
    records: list[dict[str, Any]] = []
    for row in rows:
        file_path = sample_root / str(row["file"])
        records.append({**row, "file": str(file_path), "id": str(row["id"])})
    return records


def records_from_raw_tree(
    source_dir: Path, domain: str, source_name: str, label_map: dict[str, str]
) -> list[dict[str, Any]]:
    """Read a raw source directory.

    Accepted layouts, in order:

    * ``annotations.jsonl`` with ``file``, ``objects`` (``cls``, ``bbox_xyxy``)
    * ``annotations.csv`` with ``filename,label,x1,y1,x2,y2`` (blank coords = empty)
    * images plus sibling YOLO label files
    * images inside an ``empty/`` folder, treated as negatives
    """
    if not source_dir.is_dir():
        logger.info("Source %s is not present at %s", source_name, source_dir)
        return []
    jsonl = source_dir / "annotations.jsonl"
    if jsonl.is_file():
        return _from_jsonl(source_dir, jsonl, domain, source_name, label_map)
    csv_path = source_dir / "annotations.csv"
    if csv_path.is_file():
        return _from_csv(source_dir, csv_path, domain, source_name, label_map)
    return _from_images(source_dir, domain, source_name)


def _from_jsonl(
    source_dir: Path,
    jsonl: Path,
    domain: str,
    source_name: str,
    label_map: dict[str, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in load_jsonl(jsonl):
        file_path = Path(str(row["file"]))
        if not file_path.is_absolute():
            file_path = source_dir / file_path
        if not file_path.is_file():
            logger.warning("Missing image %s", file_path)
            continue
        objects = []
        for obj in row.get("objects", []):
            mapped = _map_label(str(obj.get("cls", "")), label_map)
            if mapped is None:
                logger.warning("Unmapped label %s in %s", obj.get("cls"), file_path.name)
                continue
            if mapped == "empty":
                continue
            objects.append({"cls": mapped, "bbox_xyxy": obj["bbox_xyxy"]})
        with Image.open(file_path) as image:
            width, height = image.size
        records.append(
            {
                "id": str(row.get("id", file_path.stem)),
                "file": str(file_path),
                "domain": domain,
                "source": source_name,
                "width": width,
                "height": height,
                "objects": objects,
            }
        )
    return records


def _from_csv(
    source_dir: Path,
    csv_path: Path,
    domain: str,
    source_name: str,
    label_map: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    header = [part.strip().lower() for part in lines[0].split(",")]
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = [part.strip() for part in line.split(",")]
        row = dict(zip(header, cells, strict=False))
        filename = row.get("filename", "")
        grouped.setdefault(filename, [])
        label = row.get("label", "")
        if not label:
            continue
        mapped = _map_label(label, label_map)
        if mapped is None:
            logger.warning("Unmapped label %s in %s", label, csv_path)
            continue
        if mapped == "empty":
            continue
        grouped[filename].append(
            {
                "cls": mapped,
                "bbox_xyxy": [
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                ],
            }
        )
    records: list[dict[str, Any]] = []
    for filename, objects in grouped.items():
        file_path = source_dir / filename
        if not file_path.is_file():
            logger.warning("Missing image %s", file_path)
            continue
        with Image.open(file_path) as image:
            width, height = image.size
        records.append(
            {
                "id": Path(filename).stem,
                "file": str(file_path),
                "domain": domain,
                "source": source_name,
                "width": width,
                "height": height,
                "objects": objects,
            }
        )
    return records


def _from_images(source_dir: Path, domain: str, source_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = path.with_suffix(".txt")
        objects: list[dict[str, Any]] = []
        if "empty" not in path.parts and label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_index = int(parts[0])
                if class_index < 0 or class_index >= len(BOX_CLASSES):
                    continue
                cx, cy, bw, bh = (float(part) for part in parts[1:])
                with Image.open(path) as image:
                    width, height = image.size
                x1 = (cx - bw / 2) * width
                y1 = (cy - bh / 2) * height
                x2 = (cx + bw / 2) * width
                y2 = (cy + bh / 2) * height
                objects.append({"cls": BOX_CLASSES[class_index], "bbox_xyxy": [x1, y1, x2, y2]})
        with Image.open(path) as image:
            width, height = image.size
        records.append(
            {
                "id": path.stem,
                "file": str(path),
                "domain": domain,
                "source": source_name,
                "width": width,
                "height": height,
                "objects": objects,
            }
        )
    if not records:
        logger.info("No images found under %s", source_dir)
    return records


def collect_image_records(config: dict[str, Any], root: Path | None = None) -> list[dict[str, Any]]:
    """Collect sample and raw records. Domains are tagged, not merged."""
    root = root or repo_root()
    image_cfg = config["image"]
    label_map = {str(k).lower(): str(v) for k, v in image_cfg.get("label_map", {}).items()}
    records: list[dict[str, Any]] = []
    sample_root = resolve_path(config["sample_root"], root)
    if (
        config.get("use_sample_if_raw_missing", True)
        and (sample_root / "manifest_images.jsonl").is_file()
    ):
        records.extend(records_from_sample(sample_root))
    for name, source in image_cfg.get("sources", {}).items():
        if not source.get("enabled", True):
            continue
        source_dir = resolve_path(source["path"], root)
        domain = str(source["domain"])
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain {domain} for source {name}")
        raw = records_from_raw_tree(source_dir, domain, name, label_map)
        if raw:
            records.extend(raw)
        elif not config.get("use_sample_if_raw_missing", True):
            logger.warning("No raw images for %s and sample fallback is disabled", name)
    return records


def prepare_image_dataset(config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Write per-domain YOLO trees and a manifest that keeps the domain field."""
    root = root or repo_root()
    records = collect_image_records(config, root)
    processed = resolve_path(config["processed_root"], root) / "images"
    if processed.exists():
        shutil.rmtree(processed)
    processed.mkdir(parents=True, exist_ok=True)
    image_cfg = config["image"]
    thermal_mode = str(image_cfg.get("thermal_representation", "replicate"))
    val_ratio = float(image_cfg.get("val_ratio", 0.15))
    test_ratio = float(image_cfg.get("test_ratio", 0.15))
    manifest: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"domains": {}, "thermal_representation": thermal_mode}

    by_domain: dict[str, list[dict[str, Any]]] = {
        domain: [] for domain in image_cfg.get("domains", DOMAINS)
    }
    for record in records:
        domain = str(record["domain"])
        by_domain.setdefault(domain, []).append(record)

    for domain, domain_records in by_domain.items():
        splits = stratified_split(
            domain_records,
            key_fn=lambda row: primary_class(list(row.get("objects", []))),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        domain_dir = processed / domain
        for split_name, split_rows in splits.items():
            image_dir = domain_dir / "images" / split_name
            label_dir = domain_dir / "labels" / split_name
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for record in split_rows:
                src = Path(str(record["file"]))
                array = _read_image(src)
                if domain == "aerial_thermal":
                    array = apply_thermal_representation(array, thermal_mode)
                if array.ndim == 2:
                    array = np.stack([array, array, array], axis=-1)
                dest = image_dir / f"{record['id']}.png"
                Image.fromarray(array).save(dest)
                height, width = array.shape[:2]
                objects = list(record.get("objects", []))
                write_yolo_label(label_dir / f"{record['id']}.txt", objects, width, height)
                manifest.append(
                    {
                        "id": record["id"],
                        "domain": domain,
                        "source": record.get("source", "unknown"),
                        "split": split_name,
                        "image": str(dest),
                        "label": str(label_dir / f"{record['id']}.txt"),
                        "primary_class": primary_class(objects),
                        "objects": objects,
                        "width": width,
                        "height": height,
                        "small_object": bool(record.get("small_object", False)),
                    }
                )
        write_ultralytics_yaml(domain_dir / "data.yaml", domain_dir)
        summary["domains"][domain] = {split_name: len(rows) for split_name, rows in splits.items()}
        summary["domains"][domain]["total"] = len(domain_records)

    write_jsonl(processed / "manifest.jsonl", manifest)
    summary["images"] = len(manifest)
    logger.info("Prepared %s images into %s", len(manifest), processed)
    return summary
