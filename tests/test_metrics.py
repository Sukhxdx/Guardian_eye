"""Metric helpers. These fixtures check the math, not a trained model."""

from __future__ import annotations

from guardianeye.eval.metrics import Box, average_precision, binary_report, detection_report


def test_perfect_box_outranks_a_miss() -> None:
    gt = [Box("img", "human", (0.0, 0.0, 10.0, 10.0))]
    perfect = [Box("img", "human", (0.0, 0.0, 10.0, 10.0), 0.9)]
    missed = [Box("img", "human", (50.0, 50.0, 60.0, 60.0), 0.9)]
    perfect_score = average_precision(perfect, gt, 0.5)
    missed_score = average_precision(missed, gt, 0.5)
    assert perfect_score is not None and missed_score is not None
    assert perfect_score > missed_score


def test_empty_frame_false_positive_uses_the_supplied_ids() -> None:
    report = detection_report(
        predictions=[Box("empty-1", "human", (0.0, 0.0, 4.0, 4.0), 0.8)],
        ground_truth=[],
        class_names=("human", "vehicle", "animal"),
        small_area_px=1024,
        score_threshold=0.5,
        empty_image_ids=["empty-1", "empty-2"],
    )
    assert report["empty_frame_fpr"] == 0.5
    assert report["map50"] is None


def test_binary_report_counts_false_positives() -> None:
    report = binary_report([1, 0, 0, 1], [0.9, 0.8, 0.1, 0.2], 0.5)
    assert report["support"]["tp"] == 1
    assert report["support"]["fp"] == 1
    assert report["support"]["tn"] == 1
    assert report["support"]["fn"] == 1
    assert report["fpr"] == 0.5
