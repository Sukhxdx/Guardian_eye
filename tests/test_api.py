"""API contract without trained weights. The event is still logged."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from guardianeye.audio.features import write_wav
from guardianeye.serve.api import create_app
from guardianeye.serve.pipeline import Pipeline


def test_health_model_info_and_untrained_event(tmp_path: Path, monkeypatch: object) -> None:
    db = tmp_path / "events.sqlite"
    monkeypatch.setenv("GUARDIANEYE_DB", str(db))  # type: ignore[attr-defined]
    monkeypatch.setenv("GUARDIANEYE_NO_UI", "1")  # type: ignore[attr-defined]
    pipeline = Pipeline.from_configs()
    client = TestClient(create_app(pipeline))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    info = client.get("/model-info").json()
    image_loaded = Path("models/image/camera_trap_rgb.pt").is_file()
    audio_loaded = Path("models/audio/best.pt").is_file()
    assert info["image"]["camera_trap_rgb"]["loaded"] is image_loaded
    assert info["audio"]["loaded"] is audio_loaded
    if image_loaded:
        assert info["image"]["camera_trap_rgb"]["kind"] in {"smoke", "trained", "coco_demo"}
    if audio_loaded:
        assert info["audio"]["kind"] in {"smoke", "trained"}
    assert "reports/" in info["metrics_policy"]

    image = tmp_path / "frame.png"
    audio = tmp_path / "clip.wav"
    Image.new("RGB", (32, 32), (20, 40, 20)).save(image)
    write_wav(audio, np.zeros(16000, dtype=np.float32), 16000)
    with image.open("rb") as visual, audio.open("rb") as clip:
        response = client.post(
            "/events",
            files={
                "visual": ("frame.png", visual, "image/png"),
                "audio": ("clip.wav", clip, "audio/wav"),
            },
            data={
                "domain": "camera_trap_rgb",
                "weight_profile": "balanced",
                "decision_mode": "rules",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] in {"HIGH", "MEDIUM", "LOW"}
    assert set(body["modalities_present"]) == {"image", "audio"}
    assert body["image_evidence"]["reliable"] is image_loaded
    assert body["audio_evidence"]["reliable"] is audio_loaded
    if not image_loaded and not audio_loaded:
        assert body["alerted"] is False
        assert body["reason"] == "unreliable_weights"
    listed = client.get("/events").json()
    assert listed[0]["event_id"] == body["event_id"]
    assert listed[0]["alerted"] is False

    missing = client.post("/events")
    assert missing.status_code == 400
