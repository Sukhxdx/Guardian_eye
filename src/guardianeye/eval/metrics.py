"""Detection and binary metrics used by the evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Box:
    """One ground-truth or predicted box in pixel xyxy."""

    image_id: str
    class_name: str
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def average_precision(
    predictions: list[Box], ground_truth: list[Box], iou_threshold: float
) -> float | None:
    """All-point AP for one class. ``None`` when there is no ground truth."""
    if not ground_truth:
        return None
    if not predictions:
        return 0.0
    order = sorted(predictions, key=lambda box: box.confidence, reverse=True)
    matched: set[int] = set()
    tp: list[float] = []
    fp: list[float] = []
    for pred in order:
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(ground_truth):
            if index in matched or gt.image_id != pred.image_id:
                continue
            overlap = box_iou(pred.bbox, gt.bbox)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched.add(best_index)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / len(ground_truth)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall_points = np.concatenate(([0.0], recall, [1.0]))
    precision_points = np.concatenate(([1.0], precision, [0.0]))
    for index in range(len(precision_points) - 2, -1, -1):
        precision_points[index] = max(precision_points[index], precision_points[index + 1])
    trap = np.where(recall_points[1:] != recall_points[:-1])[0]
    return float(
        np.sum((recall_points[trap + 1] - recall_points[trap]) * precision_points[trap + 1])
    )


def detection_report(
    predictions: list[Box],
    ground_truth: list[Box],
    class_names: tuple[str, ...],
    small_area_px: float,
    score_threshold: float,
    empty_image_ids: list[str] | None = None,
) -> dict[str, Any]:
    """mAP50, mAP50-95, per-class precision/recall, small-object recall, empty-frame FPR."""
    per_class: dict[str, dict[str, float | None]] = {}
    ap50: list[float] = []
    for name in class_names:
        preds = [
            box
            for box in predictions
            if box.class_name == name and box.confidence >= score_threshold
        ]
        gts = [box for box in ground_truth if box.class_name == name]
        # Precision/recall at the score threshold use the same matching as AP's operating point.
        ap_value = average_precision(preds, gts, 0.5)
        if ap_value is not None:
            ap50.append(ap_value)
        matched_gt: set[int] = set()
        tp = 0
        ordered = sorted(preds, key=lambda box: box.confidence, reverse=True)
        for pred in ordered:
            best_iou = 0.0
            best_index = -1
            for index, gt in enumerate(gts):
                if index in matched_gt or gt.image_id != pred.image_id:
                    continue
                overlap = box_iou(pred.bbox, gt.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index >= 0 and best_iou >= 0.5:
                matched_gt.add(best_index)
                tp += 1
        fp = len(ordered) - tp
        fn = len(gts) - len(matched_gt)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        per_class[name] = {"precision": precision, "recall": recall, "ap50": ap_value}

    ap_range: list[float] = []
    thresholds = [round(0.5 + 0.05 * step, 2) for step in range(10)]
    for iou in thresholds:
        class_aps: list[float] = []
        for name in class_names:
            preds = [
                box
                for box in predictions
                if box.class_name == name and box.confidence >= score_threshold
            ]
            gts = [box for box in ground_truth if box.class_name == name]
            value = average_precision(preds, gts, iou)
            if value is not None:
                class_aps.append(value)
        if class_aps:
            ap_range.append(float(sum(class_aps) / len(class_aps)))

    small_gt = [box for box in ground_truth if box.area < small_area_px]
    small_hit = 0
    for gt in small_gt:
        hit = any(
            pred.image_id == gt.image_id
            and pred.class_name == gt.class_name
            and pred.confidence >= score_threshold
            and box_iou(pred.bbox, gt.bbox) >= 0.5
            for pred in predictions
        )
        if hit:
            small_hit += 1

    if empty_image_ids is None:
        image_ids = {box.image_id for box in ground_truth} | {box.image_id for box in predictions}
        empty_ids = [
            image_id
            for image_id in image_ids
            if not any(box.image_id == image_id for box in ground_truth)
        ]
    else:
        empty_ids = list(empty_image_ids)
    empty_fp = 0
    for image_id in empty_ids:
        if any(
            box.image_id == image_id and box.confidence >= score_threshold for box in predictions
        ):
            empty_fp += 1

    return {
        "map50": _safe_div(sum(ap50), len(ap50)) if ap50 else None,
        "map50_95": _safe_div(sum(ap_range), len(ap_range)) if ap_range else None,
        "per_class": per_class,
        "small_object_recall": _safe_div(small_hit, len(small_gt)),
        "small_object_count": len(small_gt),
        "empty_frame_fpr": _safe_div(empty_fp, len(empty_ids)),
        "empty_frame_count": len(empty_ids),
        "score_threshold": score_threshold,
        "iou_for_precision_recall": 0.5,
    }


def binary_report(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    """Precision, recall, F1, and false-positive rate for gunshot as the positive class."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores, strict=True):
        predicted = score >= threshold
        if predicted and label == 1:
            tp += 1
        elif predicted and label == 0:
            fp += 1
        elif not predicted and label == 0:
            tn += 1
        else:
            fn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    elif precision == 0 and recall == 0:
        f1 = 0.0
    pr_points: list[dict[str, float | None]] = []
    for cut in np.linspace(0.0, 1.0, 21):
        c_tp = c_fp = c_fn = 0
        for label, score in zip(labels, scores, strict=True):
            predicted = score >= float(cut)
            if predicted and label == 1:
                c_tp += 1
            elif predicted and label == 0:
                c_fp += 1
            elif not predicted and label == 1:
                c_fn += 1
        c_precision = _safe_div(c_tp, c_tp + c_fp)
        c_recall = _safe_div(c_tp, c_tp + c_fn)
        pr_points.append({"threshold": float(cut), "precision": c_precision, "recall": c_recall})
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": _safe_div(fp, fp + tn),
        "support": {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": len(labels)},
        "pr_curve": pr_points,
    }


def level_report(expected_positive: list[bool], predicted_positive: list[bool]) -> dict[str, Any]:
    """Precision/recall/F1/FPR where True is the positive decision."""
    labels = [1 if flag else 0 for flag in expected_positive]
    scores = [1.0 if flag else 0.0 for flag in predicted_positive]
    report = binary_report(labels, scores, 0.5)
    report.pop("pr_curve", None)
    return report
