"""Run image and audio inference, fuse them, and log the event."""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from guardianeye.audio.features import load_audio_file, log_mel
from guardianeye.audio.infer import AudioInferencer
from guardianeye.config import load_config, repo_root, resolve_path
from guardianeye.fusion.engine import decision_to_event, fusion_config_from_yaml
from guardianeye.fusion.rules import AudioEvidence, ImageEvidence, compare_modalities, decide
from guardianeye.image.infer import ImageDetector, annotate
from guardianeye.serve.event_log import EventLog

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """JSON event plus display artifacts for the dashboard."""

    event: dict[str, Any]
    annotated: Image.Image | None
    spectrogram: Image.Image | None
    waveform_path: str | None


def fetch_hub_weights(repo_id: str, dest: Path) -> str | None:
    """Download a Hub repo into ``dest``. Returns an error string on failure."""
    if not repo_id:
        return None
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo_id, local_dir=str(dest))
    except Exception as exc:  # noqa: BLE001 — startup must survive a missing Hub repo
        logger.warning("Hugging Face download failed: %s", exc)
        return str(exc)
    return None


def _mel_image(
    path: Path, sample_rate: int, n_mels: int, n_fft: int, hop_length: int, fmin: float, fmax: float
) -> Image.Image:
    audio, _sr = load_audio_file(path, sample_rate)
    mel = log_mel(
        audio, sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax
    )
    span = float(mel.max() - mel.min()) or 1.0
    norm = (mel - float(mel.min())) / span
    flipped = np.flipud(norm)
    red = np.clip(flipped * 1.4, 0, 1)
    green = np.clip(flipped * 1.1 - 0.15, 0, 1)
    blue = np.clip(0.35 + flipped * 0.4, 0, 1)
    rgb = (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((max(image.width * 2, 320), 180))


class Pipeline:
    """CPU inference pipeline used by the API and the dashboard."""

    def __init__(
        self,
        root: Path,
        serve_cfg: dict[str, Any],
        fusion_cfg: dict[str, Any],
        audio_cfg: dict[str, Any],
        detectors: dict[str, ImageDetector],
        audio: AudioInferencer,
        event_log: EventLog,
        hub_error: str | None = None,
    ) -> None:
        self.root = root
        self.serve_cfg = serve_cfg
        self.fusion_cfg = fusion_cfg
        self.audio_cfg = audio_cfg
        self.detectors = detectors
        self.audio = audio
        self.event_log = event_log
        self.hub_error = hub_error
        self.snapshot_dir = resolve_path(serve_cfg["snapshot_dir"], root)
        self.audio_dir = resolve_path(serve_cfg["audio_dir"], root)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_configs(cls, root: Path | None = None) -> Pipeline:
        """Build a pipeline from the YAML files and optional Hub weights."""
        root = root or repo_root()
        serve_cfg = load_config("configs/serve.yaml")
        fusion_cfg = load_config(serve_cfg.get("fusion_config", "configs/fusion.yaml"))
        audio_cfg = load_config(serve_cfg.get("audio_config", "configs/audio.yaml"))
        repo = os.environ.get("HF_MODEL_REPO") or str(serve_cfg.get("hf_model_repo") or "")
        hub_dir = resolve_path(serve_cfg.get("hf_local_dir", "models/hf"), root)
        hub_error = fetch_hub_weights(repo, hub_dir) if repo else None
        weights_dir = resolve_path(serve_cfg.get("image_weights_dir", "models/image"), root)
        detectors: dict[str, ImageDetector] = {}
        for domain in ("aerial_thermal", "camera_trap_rgb"):
            local = weights_dir / f"{domain}.pt"
            hub = hub_dir / "image" / f"{domain}.pt"
            chosen = local if local.is_file() else hub if hub.is_file() else local
            detectors[domain] = ImageDetector(
                chosen if chosen.is_file() else None,
                imgsz=int(serve_cfg.get("yolo_imgsz", 640)),
                conf=float(serve_cfg.get("yolo_conf", 0.15)),
                device=str(serve_cfg.get("device", "cpu")),
            )
        audio_path = resolve_path(serve_cfg.get("audio_weights", "models/audio/best.pt"), root)
        hub_audio = hub_dir / "audio" / "best.pt"
        if not audio_path.is_file() and hub_audio.is_file():
            audio_path = hub_audio
        calibration = audio_path.with_name("belize_to_vietnam_calibration.json")
        audio = AudioInferencer(
            audio_path if audio_path.is_file() else None,
            sample_rate=int(audio_cfg["sample_rate"]),
            window_seconds=float(audio_cfg["window_seconds"]),
            hop_seconds=float(audio_cfg["hop_seconds"]),
            n_mels=int(audio_cfg["n_mels"]),
            n_fft=int(audio_cfg["n_fft"]),
            hop_length=int(audio_cfg["hop_length"]),
            fmin=float(audio_cfg["fmin"]),
            fmax=float(audio_cfg["fmax"]),
            calibration_path=calibration if calibration.is_file() else None,
            device=str(serve_cfg.get("device", "cpu")),
        )
        db_path = Path(os.environ.get("GUARDIANEYE_DB") or resolve_path(serve_cfg["db_path"], root))
        return cls(
            root, serve_cfg, fusion_cfg, audio_cfg, detectors, audio, EventLog(db_path), hub_error
        )

    def model_info(self) -> dict[str, Any]:
        """Honest description of which weights are loaded."""
        return {
            "image": {
                domain: {
                    "loaded": detector.reliable,
                    "kind": detector.kind,
                    "note": detector.note,
                    "coco_label_map": detector.using_coco_names,
                }
                for domain, detector in self.detectors.items()
            },
            "audio": {
                "loaded": self.audio.reliable,
                "kind": self.audio.kind,
                "experiment": self.audio.experiment,
                "calibration_fitted": self.audio.calibration_fitted,
                "note": self.audio.note,
            },
            "hub_download_error": self.hub_error,
            "metrics_policy": "Displayed metrics are read from reports/ after make eval. They are not hardcoded.",
            "protocol_note": self.fusion_cfg.get("protocol_note", ""),
        }

    def run(
        self,
        visual_path: Path | None,
        audio_path: Path | None,
        domain: str,
        weight_profile: str,
        decision_mode: str,
    ) -> RunResult:
        """Fuse one upload. Both files may be omitted only by the caller's validation."""
        started = time.perf_counter()
        if domain not in self.detectors:
            raise KeyError(f"Unknown domain {domain}")
        image_ev: ImageEvidence | None = None
        audio_ev: AudioEvidence | None = None
        annotated: Image.Image | None = None
        spectrogram: Image.Image | None = None
        frames: list[dict[str, Any]] = []
        if visual_path is not None:
            detector = self.detectors[domain]
            image_ev, frames, rgb = detector.predict_path(
                visual_path,
                sample_fps=float(self.serve_cfg.get("video_sample_fps", 2)),
            )
            if rgb is not None and image_ev is not None:
                annotated = annotate(Image.fromarray(rgb), image_ev)
        if audio_path is not None:
            audio_ev = self.audio.predict_file(audio_path)
            spectrogram = _mel_image(
                audio_path,
                self.audio.sample_rate,
                self.audio.n_mels,
                self.audio.n_fft,
                self.audio.hop_length,
                self.audio.fmin,
                self.audio.fmax,
            )
        cfg = fusion_config_from_yaml(
            self.fusion_cfg, profile=weight_profile, decision_mode=decision_mode
        )
        decision = decide(image_ev, audio_ev, cfg)
        compared = compare_modalities(image_ev, audio_ev, cfg)
        latency_ms = (time.perf_counter() - started) * 1000.0
        event = decision_to_event(
            decision,
            snapshot_path=None,
            audio_clip_path=None,
            latency_ms=latency_ms,
            protocol_note=str(self.fusion_cfg.get("protocol_note", "")),
        )
        if annotated is not None:
            dest = self.snapshot_dir / f"{event['event_id']}.jpg"
            annotated.save(dest, quality=90)
            event["snapshot_path"] = str(dest)
        stored_audio: str | None = None
        if audio_path is not None:
            dest_audio = (
                self.audio_dir / f"{event['event_id']}{audio_path.suffix.lower() or '.wav'}"
            )
            shutil.copy2(audio_path, dest_audio)
            stored_audio = str(dest_audio)
            event["audio_clip_path"] = stored_audio
        event["comparison"] = {name: item.risk_level for name, item in compared.items()}
        event["comparison_reasons"] = {name: item.reason for name, item in compared.items()}
        event["frame_summaries"] = frames
        event["domain"] = domain
        self.event_log.add(event)
        return RunResult(
            event=event,
            annotated=annotated,
            spectrogram=spectrogram,
            waveform_path=stored_audio or (str(audio_path) if audio_path else None),
        )
