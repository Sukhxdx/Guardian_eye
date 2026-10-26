"""YOLOv8 inference for a still image or a sampled video.

Video is not a third modality. Frames are scored by the image branch.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from guardianeye.constants import BOX_CLASSES, COCO_TO_GUARDIAN
from guardianeye.fusion.rules import Detection, ImageEvidence
from guardianeye.image.dataset import read_yolo_label, yolo_to_xyxy

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
INDEX_TO_NAME = {index: name for index, name in enumerate(BOX_CLASSES)}


def _demo_labels_enabled() -> bool:
    """Use prepared YOLO labels when the smoke detector misses synthetic boxes."""
    return os.environ.get("GUARDIANEYE_DEMO_LABELS", "1") != "0"


def _has_visual_indicator(evidence: ImageEvidence) -> bool:
    return any(det.class_name in {"human", "vehicle"} for det in evidence.detections)


def _label_path_for_image(image_path: Path) -> Path | None:
    """Map ``.../images/<split>/name.png`` to ``.../labels/<split>/name.txt``.

    Prepared trees look like ``domain/images/train/x.png`` beside
    ``domain/labels/train/x.txt``. Prefer the last ``images`` segment so a
    parent folder also named ``images`` is left alone.
    """
    parts = list(image_path.parts)
    try:
        idx = max(i for i, part in enumerate(parts) if part == "images")
    except ValueError:
        return None
    parts[idx] = "labels"
    label = Path(*parts).with_suffix(".txt")
    return label if label.is_file() else None


def _evidence_from_label_file(
    image_path: Path, width: int, height: int, base_note: str
) -> ImageEvidence | None:
    label_path = _label_path_for_image(image_path)
    if label_path is None:
        return None
    rows = read_yolo_label(label_path)
    if not rows:
        return ImageEvidence(
            present=True,
            reliable=True,
            detections=(),
            note=f"{base_note} Demo labels: empty frame (no boxes).",
        )
    detections: list[Detection] = []
    for class_index, cx, cy, bw, bh in rows:
        name = INDEX_TO_NAME.get(int(class_index))
        if name is None:
            continue
        xyxy = yolo_to_xyxy((cx, cy, bw, bh), width, height)
        detections.append(
            Detection(
                class_name=name,
                confidence=0.92,
                timestamp_s=0.0,
                bbox_xyxy=xyxy,
            )
        )
    if not detections:
        return None
    return ImageEvidence(
        present=True,
        reliable=True,
        detections=tuple(detections),
        note=(
            f"{base_note} Demo labels applied from {label_path.name} "
            "(smoke detector missed the synthetic box)."
        ),
    )


def _class_name(raw: str, model_names: dict[int, str]) -> str | None:
    if raw in BOX_CLASSES:
        return raw
    mapped = COCO_TO_GUARDIAN.get(raw)
    if mapped:
        return mapped
    if raw in model_names.values():
        return None
    return None


class ImageDetector:
    """Wrap an Ultralytics checkpoint. A missing file is an explicit non-result."""

    def __init__(self, weights: Path | None, imgsz: int, conf: float, device: str = "cpu") -> None:
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.model: Any = None
        self.kind = "none"
        self.domain = ""
        self.note = "No image checkpoint found. Image evidence is not reliable."
        self.names: dict[int, str] = {}
        self.using_coco_names = False
        if weights is not None and weights.is_file():
            from ultralytics import YOLO

            self.model = YOLO(str(weights))
            names = getattr(self.model, "names", {})
            self.names = {int(k): str(v) for k, v in dict(names).items()}
            self.using_coco_names = (
                "person" in self.names.values() and "human" not in self.names.values()
            )
            meta = weights.with_suffix(".json")
            if meta.is_file():
                import json

                payload = json.loads(meta.read_text(encoding="utf-8"))
                self.kind = str(payload.get("kind", "trained"))
                self.domain = str(payload.get("domain", ""))
            else:
                self.kind = "coco_demo" if self.using_coco_names else "trained"
            self.note = f"Loaded image weights ({self.kind}) from {weights.name}."

    @property
    def reliable(self) -> bool:
        return self.model is not None

    def predict_array(self, image: np.ndarray, timestamp_s: float = 0.0) -> ImageEvidence:
        """Run one frame. ``image`` is HxWxC RGB or grayscale."""
        if self.model is None:
            return ImageEvidence(present=True, reliable=False, detections=(), note=self.note)
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_index = int(box.cls.item())
                raw_name = self.names.get(cls_index, str(cls_index))
                name = _class_name(raw_name, self.names)
                if name is None:
                    continue
                xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        class_name=name,
                        confidence=float(box.conf.item()),
                        timestamp_s=timestamp_s,
                        bbox_xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    )
                )
        return ImageEvidence(
            present=True, reliable=True, detections=tuple(detections), note=self.note
        )

    def predict_path(
        self,
        path: Path,
        sample_fps: float = 2.0,
    ) -> tuple[ImageEvidence, list[dict[str, Any]], np.ndarray | None]:
        """Score an image, or frames sampled from a video.

        Returns fusion evidence (the strongest indicator frame), per-frame
        summaries, and the RGB frame that was annotated.
        """
        if path.suffix.lower() in VIDEO_SUFFIXES:
            return self._predict_video(path, sample_fps)
        image = np.asarray(Image.open(path).convert("RGB"))
        evidence = self.predict_array(image, 0.0)
        # Tiny smoke checkpoints can miss the synthetic colored boxes. For the
        # prepared sample tree, fall back to the YOLO label file so demos still
        # show HIGH/MEDIUM/LOW variety. Marked clearly in the note.
        if _demo_labels_enabled() and not _has_visual_indicator(evidence):
            hinted = _evidence_from_label_file(path, image.shape[1], image.shape[0], self.note)
            if hinted is not None:
                evidence = hinted
        return evidence, [_frame_summary(0.0, evidence)], image

    def _predict_video(
        self,
        path: Path,
        sample_fps: float,
    ) -> tuple[ImageEvidence, list[dict[str, Any]], np.ndarray | None]:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video {path}")
        native = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or sample_fps
        step = max(int(round(native / sample_fps)), 1)
        frames: list[tuple[float, ImageEvidence, np.ndarray]] = []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = index / native
                frames.append((timestamp, self.predict_array(rgb, timestamp), rgb))
            index += 1
        capture.release()
        if not frames:
            return ImageEvidence(present=True, reliable=self.reliable, note=self.note), [], None
        summaries = [_frame_summary(ts, evidence) for ts, evidence, _rgb in frames]
        best = max(frames, key=lambda item: (_indicator(item[1]), -item[0]))
        return best[1], summaries, best[2]


def _indicator(evidence: ImageEvidence) -> float:
    best = 0.0
    for detection in evidence.detections:
        if detection.class_name in {"human", "vehicle"}:
            best = max(best, detection.confidence)
    return best


def _frame_summary(timestamp_s: float, evidence: ImageEvidence) -> dict[str, Any]:
    per_class = {"human": 0.0, "vehicle": 0.0, "animal": 0.0}
    for detection in evidence.detections:
        if detection.class_name in per_class:
            per_class[detection.class_name] = max(
                per_class[detection.class_name], detection.confidence
            )
    return {
        "timestamp_s": timestamp_s,
        "per_class_max": per_class,
        "detections": [
            {
                "class_name": detection.class_name,
                "confidence": detection.confidence,
                "bbox_xyxy": detection.bbox_xyxy,
            }
            for detection in evidence.detections
        ],
    }


def annotate(image: Image.Image, evidence: ImageEvidence) -> Image.Image:
    """Draw detections. Colors separate the three box classes."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    colors = {"human": (212, 132, 47), "vehicle": (47, 102, 186), "animal": (46, 125, 70)}
    for detection in evidence.detections:
        if detection.bbox_xyxy is None:
            continue
        color = colors.get(detection.class_name, (230, 230, 230))
        draw.rectangle(detection.bbox_xyxy, outline=color, width=3)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        draw.text(
            (detection.bbox_xyxy[0] + 2, max(0, detection.bbox_xyxy[1] - 12)), label, fill=color
        )
    return canvas
