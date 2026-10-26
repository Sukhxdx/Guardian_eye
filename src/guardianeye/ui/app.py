"""Ranger dashboard.

The banner is intentional: indicators are not a finding of poaching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image as PILImage

from guardianeye.image.dataset import load_jsonl
from guardianeye.serve.pipeline import Pipeline
from guardianeye.ui.reports_view import reports_markdown

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

:root {
  --ge-ink: #102018;
  --ge-forest: #1a3d2e;
  --ge-moss: #2f6b4f;
  --ge-leaf: #3d8f68;
  --ge-amber: #c9842a;
  --ge-sand: #f3efe6;
  --ge-card: #fffcf7;
  --ge-line: #d7d0c3;
  --ge-muted: #5e675f;
  --ge-high: #9b2f1f;
  --ge-med: #9a6a14;
  --ge-low: #1f5a40;
}

.gradio-container {
  max-width: 1180px !important;
  font-family: "DM Sans", "Segoe UI", sans-serif !important;
  color: var(--ge-ink) !important;
  background:
    radial-gradient(1200px 480px at 10% -10%, rgba(61, 143, 104, 0.18), transparent 55%),
    radial-gradient(900px 420px at 100% 0%, rgba(201, 132, 42, 0.14), transparent 50%),
    linear-gradient(180deg, #f7f4ee 0%, #efebe3 100%) !important;
}

.ge-hero {
  display: grid;
  gap: 10px;
  padding: 22px 24px 20px;
  border-radius: 18px;
  background:
    linear-gradient(135deg, rgba(26, 61, 46, 0.96), rgba(47, 107, 79, 0.92)),
    var(--ge-forest);
  color: #f7f3ea;
  box-shadow: 0 18px 40px rgba(16, 32, 24, 0.18);
  border: 1px solid rgba(255,255,255,0.08);
  margin-bottom: 14px;
}
.ge-brand {
  font-family: "Fraunces", Georgia, serif;
  font-size: 2.15rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.1;
  margin: 0;
}
.ge-tagline {
  margin: 0;
  max-width: 46rem;
  color: rgba(247, 243, 234, 0.88);
  font-size: 1.02rem;
  line-height: 1.45;
}
.ge-chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
.ge-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 11px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.16);
}
.ge-chip strong { color: #f0c56d; font-weight: 700; }
.ge-banner {
  background: linear-gradient(90deg, #7a2a18, #9b2f1f);
  color: #fff7f2;
  border-left: 6px solid #f0c56d;
  padding: 11px 14px;
  font-size: 0.95rem;
  font-weight: 650;
  border-radius: 10px;
  margin-bottom: 14px;
}

.ge-section-title {
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--ge-forest);
  margin: 4px 0 2px;
}

.ge-panel {
  background: var(--ge-card);
  border: 1px solid var(--ge-line);
  border-radius: 14px;
  padding: 12px 14px;
  box-shadow: 0 8px 24px rgba(16, 32, 24, 0.04);
}

.ge-risk {
  padding: 18px 18px 16px;
  border-radius: 14px;
  color: #fff8f1;
  box-shadow: 0 12px 28px rgba(16, 32, 24, 0.12);
}
.ge-risk h2 {
  margin: 0 0 8px 0;
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.85rem;
  letter-spacing: -0.02em;
}
.ge-risk .ge-meta {
  display: grid;
  gap: 4px;
  opacity: 0.95;
  font-size: 0.95rem;
}

.ge-card {
  border: 1px solid var(--ge-line);
  border-radius: 12px;
  padding: 12px 14px;
  background: var(--ge-card);
}
.ge-muted { color: var(--ge-muted); font-size: 0.9rem; }

.ge-status {
  border-left: 4px solid var(--ge-leaf);
  background: rgba(61, 143, 104, 0.08);
  border-radius: 10px;
  padding: 10px 12px;
  color: var(--ge-forest);
  font-weight: 500;
}

button.primary, .primary {
  background: linear-gradient(180deg, var(--ge-leaf), var(--ge-moss)) !important;
  border: none !important;
}
"""

LEVEL_COLORS = {"HIGH": "#9b2f1f", "MEDIUM": "#9a6a14", "LOW": "#1f5a40"}


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
        f"<div class='ge-meta'>"
        f"<div>{alert}</div>"
        f"<div>Weighted score: {score_text}</div>"
        f"<div>Pipeline latency: {latency_text}</div>"
        f"<div>{event.get('reason', '')}</div>"
        f"</div></div>"
    )


def _evidence_md(title: str, payload: dict[str, Any]) -> str:
    present = payload.get("present")
    reliable = payload.get("reliable")
    note = payload.get("note") or ""
    lines = [
        f"### {title}",
        "",
        f"- File supplied: **{bool(present)}**",
        f"- Weights reliable: **{bool(reliable)}**",
    ]
    if "per_class_max" in payload:
        lines.append("- Max confidence by class:")
        for name, value in payload.get("per_class_max", {}).items():
            lines.append(f"  - `{name}`: {float(value):.3f}")
    if payload.get("gunshot_probability") is not None:
        lines.append(f"- Gunshot probability: **{float(payload['gunshot_probability']):.3f}**")
    elif "windows" in payload:
        lines.append("- Gunshot probability: not produced (weights not loaded)")
    if note:
        lines.extend(["", f"*{note}*"])
    return "\n".join(lines)


def _comparison_md(event: dict[str, Any]) -> str:
    comparison = event.get("comparison") or {}
    reasons = event.get("comparison_reasons") or {}
    lines = [
        "### Same event, three ways",
        "",
        "HIGH needs a human/vehicle detection **and** a gunshot score inside the time window.",
        "",
    ]
    for key, label in (
        ("image_only", "Image only"),
        ("audio_only", "Audio only"),
        ("fused", "Fused"),
    ):
        lines.append(f"- **{label}:** `{comparison.get(key, '—')}` — {reasons.get(key, '')}")
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


def _showcase_labels(catalog: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for row in catalog:
        label = row.get("label") or f"[{row.get('category', 'Sample')}] {row.get('title', row.get('id'))}"
        labels.append(str(label))
    return labels or ["(default human + gunshot)"]


def _open_preview(path: str | None) -> PILImage.Image | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    return PILImage.open(file_path).convert("RGB")


def _sample_paths(
    pipeline: Pipeline, showcase_id: str | None = None
) -> tuple[Any, str | None, str, str, str, str | None]:
    """Return preview image, audio path, domain, status HTML, expected level, image path."""
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
                if row.get("label") == showcase_id
                or row.get("title") == showcase_id
                or row.get("id") == showcase_id
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
        blurb = str(chosen.get("blurb") or "")
        expected = str(chosen.get("expected_level") or "")
        category = str(chosen.get("category") or "Sample")
        if image and clip and clip.get("waveform"):
            image_path = str(image["image"])
            status = (
                f"<div class='ge-status'><strong>{category}</strong> · {title}<br/>"
                f"{blurb}<br/><span class='ge-muted'>Proxy expected level: {expected}. "
                f"Domain: {domain}.</span></div>"
            )
            return (
                _open_preview(image_path),
                str(clip["waveform"]),
                domain,
                status,
                expected,
                image_path,
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
    status = (
        "<div class='ge-status'>Loaded a sample camera-trap frame and a sample gunshot clip. "
        "Run the event to score them.</div>"
    )
    return _open_preview(image_path), audio_path, domain, status, "—", image_path


def build_demo(pipeline: Pipeline) -> gr.Blocks:
    """Dashboard used by rangers reviewing one event at a time."""

    catalog = _showcase_catalog(pipeline)
    showcase_titles = _showcase_labels(catalog)
    reports_panel = reports_markdown(pipeline.root)
    info = pipeline.model_info()
    img_loaded = any(
        bool((info.get("image") or {}).get(d, {}).get("loaded"))
        for d in ("aerial_thermal", "camera_trap_rgb")
    )
    aud_loaded = bool((info.get("audio") or {}).get("loaded"))
    weight_chip = "Weights loaded" if (img_loaded or aud_loaded) else "Demo mode · no checkpoints"

    def on_sample(showcase_choice: str) -> tuple[Any, Any, str, str, str | None]:
        preview, audio_path, domain, message, _expected, image_path = _sample_paths(
            pipeline, showcase_choice
        )
        if preview is None or not audio_path:
            return (
                None,
                None,
                "camera_trap_rgb",
                "<div class='ge-card'>Sample files are not prepared. Run <code>make setup</code> first.</div>",
                None,
            )
        return preview, audio_path, domain, message, image_path

    def on_run(
        visual: Any,
        audio: str | None,
        domain: str,
        profile: str,
        mode: str,
        stored_image_path: str | None,
    ) -> tuple[Any, ...]:
        visual_path = stored_image_path or _filepath(visual)
        if visual_path is None and isinstance(visual, PILImage.Image):
            tmp = Path("/tmp") / "guardianeye_upload.png"
            visual.save(tmp)
            visual_path = str(tmp)
        audio_path = _filepath(audio)
        if not visual_path and not audio_path:
            empty = (
                "<div class='ge-card'>Add an image or a clip. Empty submissions are not logged.</div>"
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
            _evidence_md("Image evidence", event["image_evidence"]),
            _evidence_md("Audio evidence", event["audio_evidence"]),
            _comparison_md(event),
            _history_rows(pipeline),
        )

    theme = gr.themes.Default(
        primary_hue=gr.themes.Color(
            c50="#eef7f1",
            c100="#d7ecdf",
            c200="#b5d9c4",
            c300="#88bf9f",
            c400="#5ea47c",
            c500="#3d8f68",
            c600="#2f6b4f",
            c700="#1a3d2e",
            c800="#143226",
            c900="#102018",
            c950="#0a140f",
        ),
        secondary_hue="stone",
        neutral_hue="stone",
        font=gr.themes.GoogleFont("DM Sans"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
    ).set(
        body_background_fill="#f3efe6",
        block_background_fill="#fffcf7",
        block_border_color="#d7d0c3",
        block_label_text_color="#1a3d2e",
        button_primary_background_fill="#2f6b4f",
        button_primary_background_fill_hover="#3d8f68",
        button_primary_text_color="#ffffff",
    )

    with gr.Blocks(title="GuardianEye", css=CSS, theme=theme, analytics_enabled=False) as demo:
        gr.HTML(
            f"""
<div class="ge-hero">
  <div class="ge-brand">GuardianEye</div>
  <p class="ge-tagline">
    Multimodal review for wildlife evidence. Vision and audio stay separate until late fusion.
    A human, vehicle, or gunshot-like score is an <strong>indicator</strong> — not confirmed poaching.
  </p>
  <div class="ge-chip-row">
    <span class="ge-chip">Late fusion</span>
    <span class="ge-chip">Image + audio</span>
    <span class="ge-chip">Belize ↔ Vietnam audio</span>
    <span class="ge-chip">{weight_chip}</span>
  </div>
</div>
<div class="ge-banner">Proof-of-concept. Indicators, not confirmed poaching. Synthetic proxy pairs are labelled as such.</div>
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                gr.HTML("<div class='ge-section-title'>1 · Choose a showcase scenario</div>")
                showcase = gr.Dropdown(
                    label="Sample library",
                    choices=showcase_titles,
                    value=showcase_titles[0],
                    info="RGB day/night, aerial thermal, gunshot variants, rain/thunder/insects/wind distractors.",
                )
                with gr.Row():
                    sample_btn = gr.Button("Load sample", variant="secondary")
                    run_btn = gr.Button("Score event", variant="primary")
                status = gr.HTML(
                    "<div class='ge-status'>Pick a showcase sample, then score the event.</div>"
                )
                image_path_state = gr.State(None)
                with gr.Row():
                    visual = gr.Image(label="Image input", type="pil", height=240)
                    audio = gr.Audio(label="Audio input", type="filepath")
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
                        info="Night favours audio. Wind favours vision.",
                    )
                    mode = gr.Dropdown(
                        label="Decision mode",
                        choices=["rules", "weighted"],
                        value="rules",
                    )
            with gr.Column(scale=4):
                gr.HTML("<div class='ge-section-title'>2 · Fusion decision</div>")
                risk = gr.HTML("<div class='ge-card'>No event scored yet.</div>")
                comparison = gr.Markdown("")
                with gr.Row():
                    image_md = gr.Markdown("")
                    audio_md = gr.Markdown("")

        gr.HTML("<div class='ge-section-title'>3 · Visual outputs</div>")
        with gr.Row():
            annotated = gr.Image(label="Annotated frame", type="pil", height=220)
            spectrogram = gr.Image(label="Log-mel spectrogram", type="pil", height=220)

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
            "<p class='ge-muted'>Empty frames and background audio stay in training. "
            "Image metrics stay split by domain; audio stays site-stratified. "
            "Paired demos use the synthetic proxy protocol. Smoke models may use prepared "
            "sample labels when the detector misses a synthetic box.</p>"
        )

        sample_btn.click(
            on_sample,
            inputs=[showcase],
            outputs=[visual, audio, domain, status, image_path_state],
        )
        showcase.change(
            on_sample,
            inputs=[showcase],
            outputs=[visual, audio, domain, status, image_path_state],
        )
        run_btn.click(
            on_run,
            inputs=[visual, audio, domain, profile, mode, image_path_state],
            outputs=[annotated, spectrogram, risk, image_md, audio_md, comparison, history],
        )
        refresh.click(lambda: _history_rows(pipeline), outputs=[history])
        demo.load(
            on_sample,
            inputs=[showcase],
            outputs=[visual, audio, domain, status, image_path_state],
        )
    return demo
