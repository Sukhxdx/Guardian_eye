"""YOLO-format helpers for the four-class image task.

Box classes are human, vehicle, and animal. Empty images are stored with an
empty label file and are never dropped.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from guardianeye.constants import BOX_CLASSES


def xyxy_to_yolo(
    box: tuple[float, float, float, float] | list[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Convert a pixel ``xyxy`` box to normalized YOLO ``cx cy w h``."""
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh


def yolo_to_xyxy(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Convert normalized YOLO ``cx cy w h`` to pixel ``xyxy``."""
    cx, cy, bw, bh = box
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return x1, y1, x2, y2


def primary_class(objects: list[dict[str, Any]]) -> str:
    """Pick a single image-level class for stratification and EDA.

    Priority is human, then vehicle, then animal. No boxes means empty.
    """
    present = {str(obj["cls"]) for obj in objects}
    for name in BOX_CLASSES:
        if name in present:
            return name
    return "empty"


def write_yolo_label(
    path: Path,
    objects: list[dict[str, Any]],
    width: int,
    height: int,
    class_to_index: dict[str, int] | None = None,
) -> None:
    """Write a YOLO label file. Empty object lists produce an empty file."""
    mapping = class_to_index or {name: index for index, name in enumerate(BOX_CLASSES)}
    lines: list[str] = []
    for obj in objects:
        cls_name = str(obj["cls"])
        if cls_name not in mapping:
            continue
        cx, cy, bw, bh = xyxy_to_yolo(obj["bbox_xyxy"], width, height)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{mapping[cls_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read YOLO rows as ``(class_index, cx, cy, w, h)``."""
    if not path.is_file():
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        rows.append(
            (int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        )
    return rows


def stratified_split(
    records: list[dict[str, Any]],
    key_fn: Any,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic per-class split. Records are sorted by ``id`` inside each class.

    The function never mixes caller-supplied groups; pass one domain or one site
    at a time.
    """
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to < 1")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(key_fn(record))].append(record)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda row: str(row["id"]))
        n = len(rows)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        # Tiny classes otherwise round down to an all-train split, which leaves
        # validation as empty frames only.
        if test_ratio > 0 and n >= 3 and n_test == 0:
            n_test = 1
        if val_ratio > 0 and n >= 2 and n_val == 0 and (n - n_test) >= 2:
            n_val = 1
        if n_test + n_val > n:
            overflow = n_test + n_val - n
            n_test = max(0, n_test - overflow)
        n_train = n - n_val - n_test
        while n_train == 0 and (n_val > 0 or n_test > 0) and n > 0:
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            n_train = n - n_val - n_test
        splits["train"].extend(rows[:n_train])
        splits["val"].extend(rows[n_train : n_train + n_val])
        splits["test"].extend(rows[n_train + n_val :])
    return splits


def write_ultralytics_yaml(
    path: Path, dataset_root: Path, names: tuple[str, ...] = BOX_CLASSES
) -> None:
    """Write an Ultralytics data YAML for one domain."""
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(names)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def count_box_instances(label_paths: list[Path]) -> dict[str, int]:
    """Count box instances per class. Empty files contribute zero boxes."""
    counts = {name: 0 for name in BOX_CLASSES}
    index_to_name = {index: name for index, name in enumerate(BOX_CLASSES)}
    for label_path in label_paths:
        for class_index, *_rest in read_yolo_label(label_path):
            name = index_to_name.get(class_index)
            if name is not None:
                counts[name] += 1
    return counts


def oversample_rare_classes(
    train_image_dir: Path,
    train_label_dir: Path,
    max_copies_per_image: int = 20,
) -> dict[str, int]:
    """Duplicate training images of rare box classes until counts match the majority.

    Empty images (empty label files) are left in place and are not deleted.
    Returns the number of copies written per class.
    """
    images = sorted(
        path
        for path in train_image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    labels = [train_label_dir / f"{path.stem}.txt" for path in images]
    for _image_path, label_path in zip(images, labels, strict=True):
        if not label_path.exists():
            label_path.write_text("", encoding="utf-8")
    counts = count_box_instances(labels)
    positive_counts = {name: count for name, count in counts.items() if count > 0}
    if not positive_counts:
        return {name: 0 for name in BOX_CLASSES}
    target = max(positive_counts.values())
    copies_written = {name: 0 for name in BOX_CLASSES}
    images_by_class: dict[str, list[Path]] = defaultdict(list)
    index_to_name = {index: name for index, name in enumerate(BOX_CLASSES)}
    for image_path in images:
        label_path = train_label_dir / f"{image_path.stem}.txt"
        classes_here = {
            index_to_name[row[0]] for row in read_yolo_label(label_path) if row[0] in index_to_name
        }
        for name in classes_here:
            images_by_class[name].append(image_path)

    for name, current in positive_counts.items():
        pool = images_by_class.get(name, [])
        if not pool or current >= target:
            continue
        copy_index = 0
        guard = 0
        while current < target and guard < max_copies_per_image * len(pool):
            source = pool[copy_index % len(pool)]
            source_copies = sum(
                1
                for path in train_image_dir.glob(f"{source.stem}_os*")
                if path.suffix == source.suffix
            )
            if source_copies >= max_copies_per_image:
                copy_index += 1
                guard += 1
                if copy_index > guard:
                    break
                continue
            dest = train_image_dir / f"{source.stem}_os{source_copies + 1}{source.suffix}"
            dest_label = train_label_dir / f"{dest.stem}.txt"
            shutil.copy2(source, dest)
            shutil.copy2(train_label_dir / f"{source.stem}.txt", dest_label)
            added = sum(
                1 for row in read_yolo_label(dest_label) if index_to_name.get(row[0]) == name
            )
            current += added
            copies_written[name] += 1
            copy_index += 1
            guard += 1
    return copies_written


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file. Missing files yield an empty list."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
