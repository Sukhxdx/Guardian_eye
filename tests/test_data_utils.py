"""Thermal conversion, boxes, windows, splits, and empty-frame retention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from guardianeye.audio.features import fixed_windows, spectrogram_image_to_mel, write_wav
from guardianeye.constants import BOX_CLASSES
from guardianeye.image.dataset import (
    oversample_rare_classes,
    primary_class,
    stratified_split,
    write_yolo_label,
    xyxy_to_yolo,
    yolo_to_xyxy,
)
from guardianeye.image.thermal import (
    apply_thermal_representation,
    pseudocolor_thermal,
    replicate_thermal,
)


def test_replicate_copies_the_thermal_channel() -> None:
    gray = np.array([[0, 255], [40, 10]], dtype=np.uint8)
    rgb = replicate_thermal(gray)
    assert rgb.shape == (2, 2, 3)
    assert np.array_equal(rgb[..., 0], rgb[..., 1])
    assert np.array_equal(rgb[..., 0], rgb[..., 2])


def test_pseudocolor_uses_more_than_one_channel() -> None:
    gray = np.linspace(0, 255, 64, dtype=np.uint8).reshape(8, 8)
    rgb = pseudocolor_thermal(gray)
    assert rgb.shape == (8, 8, 3)
    assert not np.array_equal(rgb[..., 0], rgb[..., 1])


def test_thermal_mode_switch() -> None:
    gray = np.zeros((4, 4), dtype=np.uint8)
    assert apply_thermal_representation(gray, "replicate").shape[-1] == 3
    assert apply_thermal_representation(gray, "pseudocolor").shape[-1] == 3


def test_yolo_box_roundtrip() -> None:
    converted = xyxy_to_yolo((10, 20, 30, 60), 100, 80)
    restored = yolo_to_xyxy(converted, 100, 80)
    assert np.allclose(restored, (10, 20, 30, 60))


def test_empty_label_file_is_written(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    write_yolo_label(path, [], 64, 64)
    assert path.read_text(encoding="utf-8") == ""
    assert primary_class([]) == "empty"


def test_windows_pad_short_audio_and_slide() -> None:
    short = fixed_windows(np.zeros(8, dtype=np.float32), 4, 3.0, 3.0)
    assert len(short) == 1
    assert len(short[0][1]) == 12
    long = fixed_windows(np.zeros(20, dtype=np.float32), 4, 2.0, 1.0)
    assert [start for start, _clip in long] == [0.0, 1.0, 2.0, 3.0]


def test_spectrogram_image_is_a_mel_grid(tmp_path: Path) -> None:
    image = tmp_path / "spec.png"
    Image.fromarray(np.zeros((20, 30), dtype=np.uint8), mode="L").save(image)
    mel = spectrogram_image_to_mel(image, n_mels=16, n_frames=8)
    assert mel.shape == (16, 8)


def test_small_class_still_reaches_validation() -> None:
    records = [{"id": "a", "label": "human"}, {"id": "b", "label": "human"}]
    splits = stratified_split(records, key_fn=lambda row: row["label"], val_ratio=0.15, test_ratio=0.15)
    assert len(splits["train"]) == 1
    assert len(splits["val"]) == 1
    assert splits["test"] == []


def test_splits_do_not_overlap_and_cover_every_row() -> None:
    records = [{"id": f"r{i}", "label": "gunshot" if i < 4 else "background"} for i in range(14)]
    splits = stratified_split(
        records, key_fn=lambda row: row["label"], val_ratio=0.15, test_ratio=0.15
    )
    ids = [row["id"] for part in splits.values() for row in part]
    assert len(ids) == len(set(ids)) == 14
    assert splits["train"]


def test_oversample_keeps_every_empty_image(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    for name, text in (
        ("empty_a", ""),
        ("empty_b", ""),
        ("human", f"{BOX_CLASSES.index('human')} 0.5 0.5 0.2 0.4\n"),
        ("human2", f"{BOX_CLASSES.index('human')} 0.4 0.4 0.2 0.3\n"),
        ("vehicle", f"{BOX_CLASSES.index('vehicle')} 0.5 0.5 0.4 0.2\n"),
    ):
        Image.new("RGB", (8, 8)).save(image_dir / f"{name}.png")
        (label_dir / f"{name}.txt").write_text(text, encoding="utf-8")
    before_empty = sorted(path.name for path in label_dir.glob("empty*.txt"))
    oversample_rare_classes(image_dir, label_dir, max_copies_per_image=5)
    after_empty = sorted(path.name for path in label_dir.glob("empty*.txt"))
    assert after_empty == before_empty
    assert any(path.name.startswith("vehicle_os") for path in image_dir.iterdir())


def test_write_wav_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    tone = (0.2 * np.sin(np.linspace(0, 20, 1600))).astype(np.float32)
    write_wav(path, tone, 16000)
    assert path.stat().st_size > 44
