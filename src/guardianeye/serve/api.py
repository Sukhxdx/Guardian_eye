"""FastAPI service. The dashboard is mounted at ``/dashboard`` when enabled."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

from guardianeye import __version__
from guardianeye.logging_utils import setup_logging
from guardianeye.serve.pipeline import Pipeline
from guardianeye.serve.schemas import EventRecord

VISUAL_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}
PROFILES = {"balanced", "night", "wind"}
MODES = {"rules", "weighted"}


def _save_upload(upload: UploadFile, directory: Path, allowed: set[str]) -> Path:
    name = Path(upload.filename or "upload").name
    suffix = Path(name).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type {suffix or '(none)'}")
    dest = directory / name
    with dest.open("wb") as handle:
        handle.write(upload.file.read())
    return dest


def create_app(pipeline: Pipeline) -> FastAPI:
    """API routes. Gradio is mounted by ``main`` so tests can skip it."""
    app = FastAPI(title="GuardianEye", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "guardianeye"}

    @app.get("/model-info")
    def model_info() -> dict[str, object]:
        return pipeline.model_info()

    @app.get("/events")
    def list_events(limit: int = 50) -> list[dict[str, object]]:
        return pipeline.event_log.list_events(limit=min(max(limit, 1), 200))

    @app.post("/events", response_model=EventRecord)
    def create_event(
        visual: UploadFile | None = File(default=None),
        audio: UploadFile | None = File(default=None),
        domain: str = Form(default="camera_trap_rgb"),
        weight_profile: str = Form(default="balanced"),
        decision_mode: str = Form(default="rules"),
    ) -> EventRecord:
        if visual is None and audio is None:
            raise HTTPException(
                status_code=400, detail="Provide an image/video file, an audio file, or both."
            )
        if weight_profile not in PROFILES:
            raise HTTPException(
                status_code=400, detail=f"weight_profile must be one of {sorted(PROFILES)}"
            )
        if decision_mode not in MODES:
            raise HTTPException(
                status_code=400, detail=f"decision_mode must be one of {sorted(MODES)}"
            )
        if domain not in pipeline.detectors:
            raise HTTPException(status_code=400, detail=f"Unknown domain {domain}")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            visual_path = (
                _save_upload(visual, folder, VISUAL_SUFFIXES) if visual is not None else None
            )
            audio_path = _save_upload(audio, folder, AUDIO_SUFFIXES) if audio is not None else None
            result = pipeline.run(visual_path, audio_path, domain, weight_profile, decision_mode)
        return EventRecord.model_validate(result.event)

    @app.get("/")
    def root() -> RedirectResponse:
        if os.environ.get("GUARDIANEYE_NO_UI", "0") == "1":
            return RedirectResponse(url="/health")
        return RedirectResponse(url="/dashboard")

    return app


def main() -> None:
    """Start the API and, unless disabled, the ranger dashboard."""
    setup_logging()
    pipeline = Pipeline.from_configs()
    app = create_app(pipeline)
    host = os.environ.get("GUARDIANEYE_HOST", str(pipeline.serve_cfg.get("host", "0.0.0.0")))
    port = int(os.environ.get("GUARDIANEYE_PORT", pipeline.serve_cfg.get("port", 8742)))
    if os.environ.get("GUARDIANEYE_NO_UI", "0") != "1":
        import gradio as gr

        from guardianeye.ui.app import build_demo

        demo = build_demo(pipeline)
        app = gr.mount_gradio_app(app, demo, path="/dashboard")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
