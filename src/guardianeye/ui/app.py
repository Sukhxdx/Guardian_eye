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
.gradio-container { max-width: 1100px !important; }
.ge-banner {
  background: #1c3428;
  color: #f6f1e7;
  border-left: 8px solid #d3922a;
  padding: 14px 16px;
  font-size: 1.05rem;
  font-weight: 650;
  letter-spacing: 0.01em;
  margin-bottom: 0.75rem;
}
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
    latency = event.get("latency_ms")
    latency_text = f"{float(latency):.0f} ms" if isinstance(latency, int | float) else "—"
    return (
        f"<div class='ge-risk' style='background:{color}'>"
        f"<h2>{level}</h2>"
        f"<div>{alert}</div>"
        f"<div>Weighted score for this event: {score_text}</div>"
        f"<div>Pipeline latency: {latency_text}</div>"
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
        "HIGH needs a human/vehicle detection and a gunshot score inside the time window.",
        "",
    ]
    for key, label in (
        ("image_only", "Image only"),
        ("audio_only", "Audio only"),
        ("fused", "Fused"),
    ):
        lines.append(f"- **{label}:** {comparison.get(key, '—')} — {reasons.get(key, '')}")
    return "\n".join(lines)


def _filepath(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        for key in ("path", "name"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item
    return None


def _history_rows(pipeline: Pipeline) -> list[list[str]]:
    rows: list[list[str]] = []
    for event in pipeline.event_log.list_events(limit=25):
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


def _showcase_catalog(pipeline: Pipeline) -> list[dict[str, Any]]:
    path = pipeline.root / "data" / "sample" / "showcase_pairs.jsonl"
    if path.is_file():
        return load_jsonl(path)
    return []


def _sample_paths(
    pipeline: Pipeline, showcase_id: str | None = None
) -> tuple[str | None, str | None, str, str]:
    processed = pipeline.root / "data" / "processed"
    images = load_jsonl(processed / "images" / "manifest.jsonl")
    audio = load_jsonl(processed / "audio" / "manifest.jsonl")
    by_image = {str(row["id"]): row for row in images}
    by_audio = {str(row["id"]): row for row in audio}
    catalog = _showcase_catalog(pipeline)
    chosen = None
    if showcase_id and showcase_id != "(default human + gunshot)":
        chosen = next(
            (
                row
                for row in catalog
                if row.get("title") == showcase_id or row.get("id") == showcase_id
            ),
            None,
        )
    if chosen is None and catalog:
        chosen = catalog[0]
    if chosen is not None:
        image = by_image.get(str(chosen.get("image_id")))
        clip = by_audio.get(str(chosen.get("audio_id")))
        domain = str(chosen.get("domain") or (image or {}).get("domain") or "camera_trap_rgb")
        title = str(chosen.get("title") or chosen.get("id") or "sample pair")
        if image and clip and clip.get("waveform"):
            return (
                str(image["image"]),
                str(clip["waveform"]),
                domain,
                f"Loaded showcase: {title}. Run the event to score them.",
            )
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
    domain = str((image or {}).get("domain") or "camera_trap_rgb")
    return (
        image_path,
        audio_path,
        domain,
        "Loaded a sample camera-trap frame and a sample gunshot clip. Run the event to score them.",
    )


def build_demo(pipeline: Pipeline) -> gr.Blocks:
    """Dashboard used by rangers reviewing one event at a time."""

    showcase_titles = [str(row.get("title") or row.get("id")) for row in _showcase_catalog(pipeline)]
    if not showcase_titles:
        showcase_titles = ["(default human + gunshot)"]
    reports_panel = reports_markdown(pipeline.root)

    def on_sample(showcase_choice: str) -> tuple[Any, Any, str, str]:
        image_path, audio_path, domain, message = _sample_paths(pipeline, showcase_choice)
        if not image_path or not audio_path:
            return (
                None,
                None,
                "camera_trap_rgb",
                "Sample files are not prepared. Run `make setup` first.",
            )
        return image_path, audio_path, domain, message

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
            empty = (
                "<div class='ge-card'>Add an image, a video, or a clip. "
                "Empty submissions are not logged.</div>"
            )
            return (
                None,
                None,
                empty,
                "No image evidence.",
                "No audio evidence.",
                "",
                _history_rows(pipeline),
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
                message,
                "Image branch did not finish.",
                "Audio branch did not finish.",
                "",
                _history_rows(pipeline),
            )
        event = result.event
        return (
            result.annotated,
            result.spectrogram,
            _risk_html(event),
            _evidence_md("Image", event["image_evidence"]),
            _evidence_md("Audio", event["audio_evidence"]),
            _comparison_md(event),
            _history_rows(pipeline),
        )

    # Default Gradio theme + our CSS. Soft/stone made the page look foreign and heavier.
    with gr.Blocks(title="GuardianEye", css=CSS, analytics_enabled=False) as demo:
        gr.HTML(
            "<div class='ge-banner'>Proof-of-concept. Indicators, not confirmed poaching.</div>"
        )
        gr.Markdown(
            """
# GuardianEye

Camera-trap and aerial-thermal frames go through the image branch. Short audio goes through a separate branch. The two meet only at the decision.

A human, a vehicle, or a gunshot-like sound is an **indicator**, not proof of poaching.
            """
        )
        status = gr.Markdown("Upload a frame or a clip, or load a showcase sample.")
        with gr.Row():
            visual = gr.Image(label="Image", type="filepath", height=220)
            audio = gr.Audio(label="Audio clip", type="filepath")
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
            )
            mode = gr.Dropdown(label="Decision mode", choices=["rules", "weighted"], value="rules")
        showcase = gr.Dropdown(
            label="Showcase sample",
            choices=showcase_titles,
            value=showcase_titles[0],
        )
        with gr.Row():
            sample_btn = gr.Button("Load sample pair")
            run_btn = gr.Button("Score event", variant="primary")
        with gr.Row():
            annotated = gr.Image(label="Annotated frame", type="pil", height=200)
            spectrogram = gr.Image(label="Log-mel spectrogram", type="pil", height=200)
        risk = gr.HTML("<div class='ge-card'>No event yet.</div>")
        with gr.Row():
            image_md = gr.Markdown("")
            audio_md = gr.Markdown("")
        comparison = gr.Markdown("")
        with gr.Accordion("Event log", open=False):
            history = gr.Dataframe(
                headers=["timestamp", "risk", "alerted", "modalities", "reason", "event id"],
                value=_history_rows(pipeline),
                interactive=False,
                wrap=True,
            )
            refresh = gr.Button("Refresh log")
        with gr.Accordion("Evaluation files", open=False):
            gr.Markdown(reports_panel)
        gr.Markdown(
            "<span class='ge-muted'>Group 27 · Multimodal AI. Empty frames and background audio stay "
            "in training. Paired-event numbers use a synthetic proxy protocol. "
            "On Render, the first open after idle can take about a minute (cold start).</span>"
        )
        sample_btn.click(on_sample, inputs=[showcase], outputs=[visual, audio, domain, status])
        run_btn.click(
            on_run,
            inputs=[visual, audio, domain, profile, mode],
            outputs=[annotated, spectrogram, risk, image_md, audio_md, comparison, history],
        )
        refresh.click(lambda: _history_rows(pipeline), outputs=[history])
    return demo
