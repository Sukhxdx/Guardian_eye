"""Per-domain image evaluation. Domains are never averaged into one headline number."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from guardianeye.config import load_config, repo_root, resolve_path
from guardianeye.constants import BOX_CLASSES
from guardianeye.eval.metrics import Box, detection_report
from guardianeye.image.dataset import load_jsonl, yolo_to_xyxy
from guardianeye.image.infer import ImageDetector
from guardianeye.logging_utils import setup_logging


def _ground_truth(row: dict[str, Any]) -> list[Box]:
    boxes: list[Box] = []
    width = int(row["width"])
    height = int(row["height"])
    for obj in row.get("objects", []):
        xyxy = tuple(float(v) for v in obj["bbox_xyxy"])
        boxes.append(Box(str(row["id"]), str(obj["cls"]), (xyxy[0], xyxy[1], xyxy[2], xyxy[3])))
    label_path = Path(str(row["label"]))
    if not boxes and label_path.is_file():
        for class_index, cx, cy, bw, bh in _read(label_path):
            if class_index >= len(BOX_CLASSES):
                continue
            x1, y1, x2, y2 = yolo_to_xyxy((cx, cy, bw, bh), width, height)
            boxes.append(Box(str(row["id"]), BOX_CLASSES[class_index], (x1, y1, x2, y2)))
    return boxes


def _read(path: Path) -> list[tuple[int, float, float, float, float]]:
    from guardianeye.image.dataset import read_yolo_label

    return read_yolo_label(path)


def evaluate_domain(
    domain: str,
    rows: list[dict[str, Any]],
    weights: Path,
    imgsz: int,
    conf: float,
    small_area_px: float,
    score_threshold: float,
) -> dict[str, Any]:
    """Score one domain's test split. Missing weights produce a status record only."""
    test_rows = [row for row in rows if row["domain"] == domain and row["split"] == "test"]
    if not weights.is_file():
        return {
            "domain": domain,
            "evaluated": False,
            "reason": f"No weights at {weights}",
            "test_images": len(test_rows),
        }
    detector = ImageDetector(weights, imgsz=imgsz, conf=conf, device="cpu")
    predictions: list[Box] = []
    ground_truth: list[Box] = []
    empty_ids: list[str] = []
    for row in test_rows:
        gt = _ground_truth(row)
        ground_truth.extend(gt)
        if not gt:
            empty_ids.append(str(row["id"]))
        image = np_image(Path(str(row["image"])))
        evidence = detector.predict_array(image)
        for detection in evidence.detections:
            if detection.bbox_xyxy is None:
                continue
            predictions.append(
                Box(
                    str(row["id"]),
                    detection.class_name,
                    detection.bbox_xyxy,
                    detection.confidence,
                )
            )
    metrics = detection_report(
        predictions,
        ground_truth,
        BOX_CLASSES,
        small_area_px=small_area_px,
        score_threshold=score_threshold,
        empty_image_ids=empty_ids,
    )
    return {
        "domain": domain,
        "evaluated": True,
        "weights": str(weights),
        "weights_kind": detector.kind,
        "test_images": len(test_rows),
        **metrics,
        "note": "Computed on this domain only. Do not average with the other domain.",
    }


def np_image(path: Path) -> Any:
    import numpy as np

    return np.asarray(Image.open(path).convert("RGB"))


def evaluate_all(
    image_config: dict[str, Any], serve_config: dict[str, Any], root: Path | None = None
) -> list[dict[str, Any]]:
    """Write one JSON file per domain under ``reports/image``."""
    root = root or repo_root()
    data_cfg = load_config(image_config.get("data_config", "configs/data.yaml"))
    manifest = load_jsonl(
        resolve_path(data_cfg["processed_root"], root) / "images" / "manifest.jsonl"
    )
    report_dir = root / "reports" / "image"
    report_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = resolve_path(serve_config.get("image_weights_dir", "models/image"), root)
    reports: list[dict[str, Any]] = []
    for domain in data_cfg.get("image", {}).get("domains", []):
        report = evaluate_domain(
            str(domain),
            manifest,
            weights_dir / f"{domain}.pt",
            imgsz=int(serve_config.get("yolo_imgsz", image_config.get("imgsz", 640))),
            conf=float(serve_config.get("yolo_conf", 0.15)),
            small_area_px=float(image_config.get("small_area_px", 1024)),
            score_threshold=0.5,
        )
        (report_dir / f"{domain}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports.append(report)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate image detectors per domain")
    parser.add_argument("--image-config", default="configs/image.yaml")
    parser.add_argument("--serve-config", default="configs/serve.yaml")
    args = parser.parse_args()
    setup_logging()
    evaluate_all(load_config(args.image_config), load_config(args.serve_config))


if __name__ == "__main__":
    main()
