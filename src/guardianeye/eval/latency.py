"""CPU latency of one fused event, written to ``reports/latency.json``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from guardianeye.config import load_config, repo_root, resolve_path
from guardianeye.image.dataset import load_jsonl
from guardianeye.logging_utils import setup_logging
from guardianeye.serve.pipeline import Pipeline


def measure(root: Path | None = None) -> dict[str, Any]:
    """Time preprocess, each model, fusion, and the total against the configured target."""
    root = root or repo_root()
    serve_cfg = load_config("configs/serve.yaml")
    data_cfg = load_config("configs/data.yaml")
    processed = resolve_path(data_cfg["processed_root"], root)
    images = load_jsonl(processed / "images" / "manifest.jsonl")
    clips = [
        row for row in load_jsonl(processed / "audio" / "manifest.jsonl") if row.get("waveform")
    ]
    if not images or not clips:
        payload = {"measured": False, "reason": "Prepared sample image and audio are required."}
        (root / "reports").mkdir(parents=True, exist_ok=True)
        (root / "reports" / "latency.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return payload
    pipeline = Pipeline.from_configs(root)
    started = time.perf_counter()
    stage_start = time.perf_counter()
    image_path = Path(str(images[0]["image"]))
    audio_path = Path(str(clips[0]["waveform"]))
    load_ms = (time.perf_counter() - stage_start) * 1000
    stage_start = time.perf_counter()
    result = pipeline.run(
        visual_path=image_path,
        audio_path=audio_path,
        domain=str(images[0]["domain"]),
        weight_profile="balanced",
        decision_mode="rules",
    )
    total_ms = (time.perf_counter() - started) * 1000
    infer_ms = (time.perf_counter() - stage_start) * 1000
    target_s = float(serve_cfg.get("latency_target_s", 5))
    payload = {
        "measured": True,
        "device": "cpu",
        "target_s": target_s,
        "within_target": (total_ms / 1000.0) < target_s,
        "stages_ms": {
            "resolve_paths": load_ms,
            "inference_fusion_log": infer_ms,
            "end_to_end": total_ms,
        },
        "event_latency_ms": result.event["latency_ms"],
        "image_reliable": bool(result.event["image_evidence"].get("reliable")),
        "audio_reliable": bool(result.event["audio_evidence"].get("reliable")),
        "risk_level": result.event["risk_level"],
        "note": "One sample event on CPU. Not a hardware benchmark suite.",
    }
    out = root / "reports" / "latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure CPU latency of one fused event")
    parser.parse_args()
    setup_logging()
    measure()


if __name__ == "__main__":
    main()
