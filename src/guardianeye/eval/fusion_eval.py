"""Fusion evaluation on synthetic paired events.

Results are labeled ``synthetic proxy protocol``. They are not field validation.
Image-only, audio-only, and fused decisions are scored on the same events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from guardianeye.audio.infer import AudioInferencer
from guardianeye.config import load_config, repo_root, resolve_path
from guardianeye.eval.metrics import level_report
from guardianeye.fusion.engine import fusion_config_from_yaml
from guardianeye.fusion.rules import compare_modalities
from guardianeye.image.dataset import load_jsonl
from guardianeye.image.infer import ImageDetector
from guardianeye.logging_utils import setup_logging


def _positive(level: str, operating_point: str) -> bool:
    if operating_point == "high":
        return level == "HIGH"
    return level in {"HIGH", "MEDIUM"}


def evaluate(root: Path | None = None) -> dict[str, Any]:
    """Run the proxy protocol and write ``reports/fusion/proxy.json``."""
    root = root or repo_root()
    data_cfg = load_config("configs/data.yaml")
    fusion_cfg_raw = load_config("configs/fusion.yaml")
    audio_cfg = load_config("configs/audio.yaml")
    serve_cfg = load_config("configs/serve.yaml")
    events = load_jsonl(
        resolve_path(
            data_cfg.get("paired", {}).get("output", "data/processed/paired/events.jsonl"), root
        )
    )
    report_dir = root / "reports" / "fusion"
    report_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = resolve_path(serve_cfg.get("image_weights_dir", "models/image"), root)
    audio_weights = resolve_path(serve_cfg.get("audio_weights", "models/audio/best.pt"), root)
    detectors = {
        domain: ImageDetector(
            weights_dir / f"{domain}.pt",
            imgsz=int(serve_cfg.get("yolo_imgsz", 640)),
            conf=float(serve_cfg.get("yolo_conf", 0.15)),
        )
        for domain in ("aerial_thermal", "camera_trap_rgb")
    }
    audio = AudioInferencer(
        audio_weights if audio_weights.is_file() else None,
        sample_rate=int(audio_cfg["sample_rate"]),
        window_seconds=float(audio_cfg["window_seconds"]),
        hop_seconds=float(audio_cfg["hop_seconds"]),
        n_mels=int(audio_cfg["n_mels"]),
        n_fft=int(audio_cfg["n_fft"]),
        hop_length=int(audio_cfg["hop_length"]),
        fmin=float(audio_cfg["fmin"]),
        fmax=float(audio_cfg["fmax"]),
        calibration_path=audio_weights.with_name("belize_to_vietnam_calibration.json"),
    )
    image_ready = any(detector.reliable for detector in detectors.values())
    if not events or not image_ready or not audio.reliable:
        payload = {
            "protocol": "synthetic proxy protocol",
            "evaluated": False,
            "reason": "Need paired events plus image and audio checkpoints before metrics are computed.",
            "paired_events": len(events),
            "image_weights_loaded": image_ready,
            "audio_weights_loaded": audio.reliable,
            "note": "This protocol is not field validation. No metric values are invented when models are missing.",
        }
        (report_dir / "proxy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    cfg = fusion_config_from_yaml(fusion_cfg_raw)
    rows: list[dict[str, Any]] = []
    import numpy as np
    from PIL import Image

    for event in events:
        domain = str(event["domain"])
        detector = detectors[domain]
        image = np.asarray(Image.open(event["image_path"]).convert("RGB"))
        image_ev = detector.predict_array(image, float(event.get("image_timestamp_s", 0.0)))
        audio_ev = audio.predict_file(Path(event["audio_path"]))
        compared = compare_modalities(image_ev, audio_ev, cfg)
        rows.append(
            {
                "event_id": event["event_id"],
                "scenario": event["scenario"],
                "domain": domain,
                "expected_level": event["expected_level"],
                "failure_tags": event.get("failure_tags", []),
                "image_only": compared["image_only"].risk_level,
                "audio_only": compared["audio_only"].risk_level,
                "fused": compared["fused"].risk_level,
                "fused_reason": compared["fused"].reason,
                "fused_alerted": compared["fused"].alerted,
            }
        )
    systems = {}
    for operating_point in ("high", "indicator"):
        expected_high = [_positive(str(row["expected_level"]), operating_point) for row in rows]
        systems[operating_point] = {
            name: level_report(
                expected_high, [_positive(str(row[name]), operating_point) for row in rows]
            )
            for name in ("image_only", "audio_only", "fused")
        }
    failures = [row for row in rows if row["failure_tags"] or row["fused"] != row["expected_level"]]
    payload = {
        "protocol": "synthetic proxy protocol",
        "evaluated": True,
        "note": "Pairs are constructed from labelled scenarios. This is not field validation.",
        "n_events": len(rows),
        "operating_points": {
            "high": "Positive means risk level HIGH. This is the default alert.",
            "indicator": "Positive means risk level HIGH or MEDIUM.",
        },
        "systems": systems,
        "failure_analysis": failures,
        "events": rows,
    }
    (report_dir / "proxy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate late fusion on synthetic pairs")
    parser.parse_args()
    setup_logging()
    evaluate()


if __name__ == "__main__":
    main()
