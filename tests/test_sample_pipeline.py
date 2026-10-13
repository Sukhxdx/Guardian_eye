"""Sample data is enough to prepare splits, pairs, and EDA without real datasets."""

from __future__ import annotations

import json
from pathlib import Path

from guardianeye.config import load_config
from guardianeye.data.audio_prep import prepare_audio_dataset
from guardianeye.data.eda import run_eda
from guardianeye.data.image_prep import prepare_image_dataset
from guardianeye.data.pairing import build_paired_events
from guardianeye.data.sample import generate_sample_dataset
from guardianeye.image.dataset import load_jsonl


def _config(tmp_path: Path) -> dict[str, object]:
    config = load_config("configs/data.yaml")
    config["sample_root"] = str(tmp_path / "sample")
    config["processed_root"] = str(tmp_path / "processed")
    config["paired"]["output"] = str(tmp_path / "processed" / "paired" / "events.jsonl")
    config["use_sample_if_raw_missing"] = True
    return config


def test_sample_pipeline_keeps_domains_sites_and_majority_classes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    generate_sample_dataset(
        Path(str(config["sample_root"])),
        seed=27,
        sample_rate=16000,
        seconds=3.0,
    )
    image_summary = prepare_image_dataset(config, root=tmp_path)
    audio_summary = prepare_audio_dataset(config, root=tmp_path)
    events = build_paired_events(config, root=tmp_path)
    eda = run_eda(config, root=tmp_path)

    assert image_summary["images"] > 0
    images = load_jsonl(tmp_path / "processed" / "images" / "manifest.jsonl")
    domains = {row["domain"] for row in images}
    assert domains == {"aerial_thermal", "camera_trap_rgb"}
    for domain in domains:
        rows = [row for row in images if row["domain"] == domain]
        splits = {row["split"] for row in rows}
        assert splits <= {"train", "val", "test"}
        ids = [row["id"] for row in rows]
        assert len(ids) == len(set(ids))
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["primary_class"]] = counts.get(row["primary_class"], 0) + 1
        assert counts["empty"] > counts.get("human", 0)
        assert counts["empty"] > counts.get("vehicle", 0)
        label = Path(rows[0]["label"])
        assert label.is_file()

    audio_rows = load_jsonl(tmp_path / "processed" / "audio" / "manifest.jsonl")
    assert any(row["representation"] == "spectrogram_image" for row in audio_rows)
    split_ids = json.loads(
        (tmp_path / "processed" / "audio" / "splits" / "belize_to_vietnam.ids.json").read_text(
            encoding="utf-8"
        )
    )
    by_id = {row["id"]: row for row in audio_rows}
    train_sites = {by_id[item]["site"] for item in split_ids["train"]}
    test_sites = {by_id[item]["site"] for item in split_ids["test"]}
    assert train_sites == {"belize"}
    assert test_sites == {"vietnam"}
    assert "vietnam_spec_000" not in set(split_ids["train"]) | set(split_ids["val"]) | set(
        split_ids["test"]
    )
    for site, counts in audio_summary["by_site"].items():
        assert counts["background"] > counts["gunshot"], site

    assert events
    assert all(event["protocol"] == "synthetic_proxy" for event in events)
    levels = {event["expected_level"] for event in events}
    assert {"HIGH", "MEDIUM", "LOW"} <= levels
    assert any("impulse_noise" in event["failure_tags"] for event in events)

    aerial_counts = eda["image_class_counts"]["aerial_thermal"]
    positive = [count for count in aerial_counts.values() if count]
    assert eda["image_imbalance_ratio"]["aerial_thermal"] == max(positive) / min(positive)
    assert (tmp_path / "reports" / "eda" / "class_counts_aerial_thermal.png").is_file()
