"""Ranger dashboard.

The banner is intentional: indicators are not a finding of poaching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from guardianeye.image.dataset import load_jsonl
from guardianeye.serve.pipeline import Pipeline
from guardianeye.ui.reports_view import reports_markdown

CSS = """
.gradio-container { max-width: 1180px !important; }
.ge-banner {
  background: #1c3428;
  color: #f6f1e7;
  border-left: 8px solid #d3922a;
  padding: 14px 16px;
  font-size: 1.05rem;
  font-weight: 650;
  letter-spacing: 0.01em;
}
.ge-wrap { font-family: "Source Serif 4", Georgia, serif; }
.ge-risk {
  padding: 16px;
  border-radius: 8px;
  color: #f7f3ea;
}
.ge-risk h2 { margin: 0 0 6px 0; font-size: 1.6rem; }
.ge-card {
  border: 1px solid #d9d3c7;
  border-radius: 8px;
  padding: 10px 12px;
  background: #fbf8f2;
}
.ge-muted { color: #5c564c; font-size: 0.92rem; }
"""

LEVEL_COLORS = {"HIGH": "#8c2f1b", "MEDIUM": "#8a5a12", "LOW": "#1e4d38"}


def _risk_html(event: dict[str, Any]) -> str:
    level = str(event.get("risk_level", "LOW"))
    color = LEVEL_COLORS.get(level, "#333")
    alert = "Alert queued for review" if event.get("alerted") else "Not alerted"
    score = event.get("risk_score")
    score_text = f"{float(score):.3f}" if isinstance(score, int | float) else "—"
    return (
        f"<div class='ge-risk' style='background:{color}'>"
        f"<h2>{level}</h2>"
        f"<div>{alert}</div>"
        f"<div>Weighted score for this event: {score_text}</div>"
        f"<div>{event.get('reason', '')}</div>"
        f"</div>"
    )


def _evidence_md(title: str, payload: dict[str, Any]) -> str:
    present = payload.get("present")
    reliable = payload.get("reliable")
    note = payload.get("note") or ""
    lines = [
        f"**{title}**",
        "",
        f"File supplied: {bool(present)}",
        f"Weights reliable: {bool(reliable)}",
    ]
    if "per_class_max" in payload:
        lines.append("")
        lines.append("Max confidence by class:")
        for name, value in payload.get("per_class_max", {}).items():
            lines.append(f"- {name}: {float(value):.3f}")
    if payload.get("gunshot_probability") is not None:
        lines.append("")
        lines.append(f"Gunshot probability: {float(payload['gunshot_probability']):.3f}")
    elif "windows" in payload:
        lines.append("")
        lines.append("Gunshot probability: not produced (weights not loaded).")
    if note:
        lines.extend(["", note])
    return "\n".join(lines)


def _comparison_md(event: dict[str, Any]) -> str:
    comparison = event.get("comparison") or {}
    reasons = event.get("comparison_reasons") or {}
    lines = [
        "### Same event, three ways",
        "",
        "HIGH is only available when a human or vehicle detection and a gunshot score clear their thresholds inside the time window. A single modality stops at MEDIUM, and only if it clears the stricter unimodal threshold.",
        "",
    ]
    for key, label in (
        ("image_only", "Image only"),
        ("audio_only", "Audio only"),
        ("fused", "Fused"),
    ):
        lines.append(f"- **{label}:** {comparison.get(key, '—')} ({reasons.get(key, '')})")
    lines.append("")
    lines.append(str(event.get("protocol_note") or ""))
    return "\n".join(lines)


def _filepath(value: object) -> str | None:
    """Gradio file components sometimes return a path and sometimes a dict."""
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        raw = value.get("path") or value.get("name")
        return str(raw) if raw else None
    if isinstance(value, tuple | list):
        for item in value:
            if isinstance(item, str):
                return item
        return None
    return str(value)


def _history_rows(pipeline: Pipeline) -> list[list[str]]:
    rows: list[list[str]] = []
    for event in pipeline.event_log.list_events(limit=40):
        rows.append(
            [
                str(event.get("timestamp", "")),
                str(event.get("risk_level", "")),
                "yes" if event.get("alerted") else "no",
                ",".join(event.get("modalities_present") or []),
                str(event.get("reason", "")),
                str(event.get("event_id", "")),
            ]
        )
    return rows


def _sample_paths(pipeline: Pipeline) -> tuple[str | None, str | None]:
    processed = pipeline.root / "data" / "processed"
    images = load_jsonl(processed / "images" / "manifest.jsonl")
    audio = load_jsonl(processed / "audio" / "manifest.jsonl")
    image = next(
        (
            row
            for row in images
            if row.get("domain") == "camera_trap_rgb" and row.get("primary_class") == "human"
        ),
        None,
    )
    clip = next(
        (row for row in audio if row.get("label") == "gunshot" and row.get("waveform")), None
    )
    image_path = str(image["image"]) if image else None
    audio_path = str(clip["waveform"]) if clip else None
    return image_path, audio_path


def build_demo(pipeline: Pipeline) -> gr.Blocks:
    """Dashboard used by rangers reviewing one event at a time."""

    def on_sample() -> tuple[Any, Any, str]:
        image_path, audio_path = _sample_paths(pipeline)
        if not image_path or not audio_path:
            return None, None, "Sample files are not prepared. Run `make setup` first."
        return (
            image_path,
            audio_path,
            "Loaded a sample camera-trap frame and a sample gunshot clip. Run the event to score them.",
        )

    def on_run(
        visual: str | None,
        audio: str | None,
        domain: str,
        profile: str,
        mode: str,
    ) -> tuple[Any, ...]:
        visual_path = _filepath(visual)
        audio_path = _filepath(audio)
        if not visual_path and not audio_path:
            empty = "<div class='ge-card'>Add an image, a video, or a clip. Empty submissions are not logged.</div>"
            return (
                None,
                None,
                None,
                empty,
                "No image evidence.",
                "No audio evidence.",
                "",
                _history_rows(pipeline),
                reports_markdown(pipeline.root),
            )
        try:
            result = pipeline.run(
                Path(visual_path) if visual_path else None,
                Path(audio_path) if audio_path else None,
                domain,
                profile,
                mode,
            )
        except Exception as exc:  # noqa: BLE001 — show the failure in the panel
            message = f"<div class='ge-card'>Could not score this upload: {exc}</div>"
            return (
                None,
                None,
                None,
                message,
                "Image branch did not finish.",
                "Audio branch did not finish.",
                "",
                _history_rows(pipeline),
                reports_markdown(pipeline.root),
            )
        event = result.event
        return (
            result.annotated,
            result.waveform_path,
            result.spectrogram,
            _risk_html(event),
            _evidence_md("Image", event["image_evidence"]),
            _evidence_md("Audio", event["audio_evidence"]),
            _comparison_md(event),
            _history_rows(pipeline),
            reports_markdown(pipeline.root),
        )

    theme = gr.themes.Soft(primary_hue="green", secondary_hue="stone", neutral_hue="stone")
    with gr.Blocks(title="GuardianEye", css=CSS, theme=theme) as demo:
        gr.HTML(
            "<div class='ge-banner'>Proof-of-concept. Indicators, not confirmed poaching.</div>"
        )
        gr.Markdown(
            """
# GuardianEye

Camera-trap and aerial-thermal frames go through the image branch. Gunshot-versus-background audio goes through a separate branch. The two meet only at the decision, after each score is checked against its own threshold.

A human, a vehicle, or a gunshot-like sound is an **indicator**. It is not, on its own, evidence that someone was poaching. Video is sampled into frames and stays in the image branch.
            """
        )
        status = gr.Markdown("Upload a frame or a clip, or load the prepared sample pair.")
        with gr.Row():
            visual = gr.File(label="Image or video", file_types=["image", "video"], type="filepath")
            audio = gr.Audio(label="Audio clip", type="filepath", sources=["upload"])
        with gr.Row():
            domain = gr.Dropdown(
                label="Image domain",
                choices=["camera_trap_rgb", "aerial_thermal"],
                value="camera_trap_rgb",
            )
            profile = gr.Dropdown(
                label="Weight profile",
                choices=["balanced", "night", "wind"],
                value="balanced",
                info="Night favours audio. Wind favours vision. Rules mode still requires both indicators for HIGH.",
            )
            mode = gr.Dropdown(label="Decision mode", choices=["rules", "weighted"], value="rules")
        with gr.Row():
            sample_btn = gr.Button("Load sample pair")
            run_btn = gr.Button("Score event", variant="primary")
        with gr.Row():
            annotated = gr.Image(label="Annotated frame", type="pil")
            waveform = gr.Audio(label="Waveform", type="filepath")
            spectrogram = gr.Image(label="Log-mel spectrogram", type="pil")
        with gr.Row():
            risk = gr.HTML("<div class='ge-card'>No event yet.</div>")
            image_md = gr.Markdown("")
            audio_md = gr.Markdown("")
        comparison = gr.Markdown("")
        gr.Markdown(
            "### Event log\n\nEvery scored event is stored, including ones that did not alert."
        )
        history = gr.Dataframe(
            headers=["timestamp", "risk", "alerted", "modalities", "reason", "event id"],
            value=_history_rows(pipeline),
            interactive=False,
            wrap=True,
        )
        refresh = gr.Button("Refresh log")
        reports = gr.Markdown(reports_markdown(pipeline.root))
        gr.Markdown(
            "<span class='ge-muted'>Group 27 · Multimodal AI. Empty frames and background audio stay in training. "
            "Cross-site audio scores and per-domain image scores are reported separately. "
            "Paired-event numbers use a synthetic proxy protocol.</span>"
        )
        sample_btn.click(on_sample, outputs=[visual, audio, status])
        run_btn.click(
            on_run,
            inputs=[visual, audio, domain, profile, mode],
            outputs=[
                annotated,
                waveform,
                spectrogram,
                risk,
                image_md,
                audio_md,
                comparison,
                history,
                reports,
            ],
        )
        refresh.click(lambda: _history_rows(pipeline), outputs=[history])
    return demo
