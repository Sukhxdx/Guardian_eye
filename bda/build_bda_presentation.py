"""Build the GuardianEye BDA project PowerPoint for college submission.

Reads facts from reports/bda/summary.json. Does not invent detection metrics.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reports" / "bda" / "summary.json"
OUT_DOCS = ROOT / "docs" / "BDA_Project_Presentation_2026_2027.pptx"
OUT_ROOT = ROOT / "BDA_Project_Presentation_2026_2027.pptx"

# SIT AIML palette: dark navy / white / muted green
NAVY = RGBColor(0x0B, 0x1D, 0x36)
NAVY_MID = RGBColor(0x14, 0x2F, 0x52)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
OFF_WHITE = RGBColor(0xF5, 0xF7, 0xFA)
MUTED_GREEN = RGBColor(0x3D, 0x7A, 0x5F)
GREEN_SOFT = RGBColor(0x5A, 0x9E, 0x7A)
SLATE = RGBColor(0x4A, 0x55, 0x68)
LIGHT_LINE = RGBColor(0xD0, 0xD7, 0xE2)
TABLE_HEADER_BG = NAVY
TABLE_ALT = RGBColor(0xE8, 0xEE, 0xF4)


def _set_run(run, text: str, size: int, bold: bool = False, color: RGBColor = NAVY, font: str = "Calibri"):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font


def _add_textbox(slide, left, top, width, height, text, size=18, bold=False, color=NAVY, align=PP_ALIGN.LEFT, font="Calibri"):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    _set_run(p.add_run(), text, size=size, bold=bold, color=color, font=font)
    return box


def _solid_fill(shape, color: RGBColor):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _add_rect(slide, left, top, width, height, color: RGBColor):
    shape = slide.shapes.add_shape(1, left, top, width, height)  # MSO_SHAPE.RECTANGLE = 1
    _solid_fill(shape, color)
    return shape


def _blank_slide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])  # blank


def _header_bar(slide, title: str, subtitle: str | None = None):
    _add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.95), NAVY)
    _add_rect(slide, Inches(0), Inches(0.95), Inches(13.333), Inches(0.06), MUTED_GREEN)
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.22), Inches(12.3), Inches(0.55))
    tf = box.text_frame
    p = tf.paragraphs[0]
    _set_run(p.add_run(), title, size=26, bold=True, color=WHITE)
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.5), Inches(1.15), Inches(12.3), Inches(0.4))
        sp = sub.text_frame.paragraphs[0]
        _set_run(sp.add_run(), subtitle, size=13, bold=False, color=SLATE)


def _footer(slide, page: int, total: int):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(7.15), Inches(10), Inches(0.3))
    p = box.text_frame.paragraphs[0]
    _set_run(p.add_run(), "GuardianEye | BDA 0701260704 | SIT AIML | AY 2026–2027", size=10, color=SLATE)
    num = slide.shapes.add_textbox(Inches(11.5), Inches(7.15), Inches(1.3), Inches(0.3))
    np_ = num.text_frame.paragraphs[0]
    np_.alignment = PP_ALIGN.RIGHT
    _set_run(np_.add_run(), f"{page} / {total}", size=10, color=SLATE)


def _bullets(slide, left, top, width, height, items: list[str], size=16, color=NAVY, spacing=8):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = 0
        p.space_after = Pt(spacing)
        run = p.add_run()
        _set_run(run, f"•  {item}", size=size, color=color)
    return box


def _fill_cell(cell, text: str, size=12, bold=False, color=NAVY, fill: RGBColor | None = None, align=PP_ALIGN.LEFT):
    if fill is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
    else:
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    # clear existing runs
    p.clear()
    _set_run(p.add_run(), text, size=size, bold=bold, color=color)


def _style_table(table, header=True):
    for r_idx, row in enumerate(table.rows):
        for cell in row.cells:
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.name = "Calibri"


def slide_title(prs, total):
    slide = _blank_slide(prs)
    _add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), NAVY)
    _add_rect(slide, Inches(0), Inches(0), Inches(0.18), Inches(7.5), MUTED_GREEN)

    _add_textbox(
        slide, Inches(0.7), Inches(0.55), Inches(12), Inches(0.4),
        "Symbiosis Institute of Technology  ·  Department of AIML",
        size=14, color=GREEN_SOFT,
    )
    _add_textbox(
        slide, Inches(0.7), Inches(1.15), Inches(12), Inches(1.2),
        "GuardianEye: Late Fusion of Image and\nAudio Poaching Indicators",
        size=32, bold=True, color=WHITE,
    )
    _add_textbox(
        slide, Inches(0.7), Inches(2.55), Inches(12), Inches(0.4),
        "Big Data Analytics (BDA)  ·  Course code 0701260704  ·  Academic Year 2026–2027",
        size=15, color=OFF_WHITE,
    )
    _add_textbox(
        slide, Inches(0.7), Inches(3.15), Inches(12), Inches(0.35),
        "Guide: Dr. Archana Choudary, Assistant Professor, AIML",
        size=14, color=GREEN_SOFT,
    )
    _add_textbox(
        slide, Inches(0.7), Inches(3.5), Inches(12), Inches(0.35),
        "Head of the Department: Dr. Sumit Kumar",
        size=14, color=GREEN_SOFT,
    )

    members = [
        "Akhil Chivakula — 23070126009",
        "Dipti Kothari — 23070126040",
        "Khurinji Malar — 23070126059",
        "Sukhada Dhingra — 24070126505",
    ]
    box = slide.shapes.add_textbox(Inches(0.7), Inches(4.05), Inches(8), Inches(2.0))
    tf = box.text_frame
    p0 = tf.paragraphs[0]
    _set_run(p0.add_run(), "Group members", size=13, bold=True, color=MUTED_GREEN)
    for m in members:
        p = tf.add_paragraph()
        p.space_before = Pt(6)
        _set_run(p.add_run(), m, size=15, color=WHITE)

    _add_textbox(
        slide, Inches(0.7), Inches(6.7), Inches(12), Inches(0.35),
        "Proof-of-concept. Indicators, not confirmed poaching.",
        size=12, color=GREEN_SOFT,
    )
    return slide


def slide_problem(prs, page, total):
    slide = _blank_slide(prs)
    _add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), OFF_WHITE)
    _header_bar(slide, "Problem and Motivation")
    _bullets(
        slide,
        Inches(0.6),
        Inches(1.35),
        Inches(12.2),
        Inches(5.2),
        [
            "Wildlife patrol evidence arrives as camera-trap RGB, aerial thermal frames, and short audio clips.",
            "No labelled poacher class in the collections named by configuration.",
            "No public time-aligned field incidents pairing image with audio of the same moment.",
            "GuardianEye detects indicators (human / vehicle / gunshot-like sound), not confirmed poaching.",
            "One detection is not proof of poaching; late fusion informs a human review workflow.",
            "Video is not a third modality: frames are sampled into the image branch.",
        ],
        size=17,
        spacing=12,
    )
    _footer(slide, page, total)
    return slide


def slide_objectives(prs, page, total):
    slide = _blank_slide(prs)
    _add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), OFF_WHITE)
    _header_bar(slide, "Objectives (Completed Work)")
    _bullets(
        slide,
        Inches(0.6),
        Inches(1.35),
        Inches(12.2),
        Inches(5.4),
        [
            "Implement a two-branch indicator pipeline: YOLOv8 image domains + log-mel audio CNN.",
            "Keep image metrics separate for aerial_thermal and camera_trap_rgb.",
            "Stratify audio by site (Belize ↔ Vietnam) with in-domain controls.",
            "Retain empty images and background audio; address class imbalance (oversampling / weighted loss).",
            "Define late-fusion rules and evaluate fusion pairs under a labelled synthetic proxy protocol.",
            "Serve the pipeline (FastAPI + Gradio dashboard) with SQLite audit logging.",
            "Aggregate prepared manifests and the audit log with a local PySpark Big Data Analytics job.",
        ],
        size=16,
        spacing=10,
    )
    _footer(slide, page, total)
    return slide


def slide_architecture(prs, page, total):
    slide = _blank_slide(prs)
    _add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), OFF_WHITE)
    _header_bar(slide, "System Architecture")

    # Three column cards
    cards = [
        ("Image branch", ["Still / sampled video frames", "Domains: aerial_thermal, camera_trap_rgb", "YOLOv8 per domain", "Classes: human, vehicle, animal, empty"]),
        ("Audio branch", ["WAV windows or spectrogram image", "Fixed windows → log-mel", "Small CNN classifier", "Classes: gunshot, background"]),
        ("Decision & serving", ["Late-fusion rules (HIGH / MEDIUM / LOW)", "FastAPI API + Gradio dashboard", "SQLite event / audit log", "Alerts and non-alerts both recorded"]),
    ]
    lefts = [0.5, 4.6, 8.7]
    for left, (title, bullets) in zip(lefts, cards):
        card = _add_rect(slide, Inches(left), Inches(1.35), Inches(3.9), Inches(4.6), WHITE)
        card.line.color.rgb = LIGHT_LINE
        card.line.width = Pt(1)
        _add_rect(slide, Inches(left), Inches(1.35), Inches(3.9), Inches(0.55), NAVY)
        _add_textbox(slide, Inches(left + 0.15), Inches(1.42), Inches(3.6), Inches(0.4), title, size=16, bold=True, color=WHITE)
        _bullets(slide, Inches(left + 0.15), Inches(2.1), Inches(3.6), Inches(3.5), bullets, size=13, spacing=8)

    _add_textbox(
        slide, Inches(0.5), Inches(6.15), Inches(12.3), Inches(0.6),
        "HIGH requires human/vehicle and gunshot-like score within the time window. Single-modality uploads cannot reach HIGH.",
        size=13, color=SLATE,
    )
    _footer(slide, page, total)
    return slide


def slide_data_pipeline(prs, page, total, summary: dict):
    slide = _blank_slide(prs)
    _add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), OFF_WHITE)
    _header_bar(slide, "Data Pipeline and BDA Framing")
    img = summary["image"]
    aud = summary["audio"]
    _bullets(
        slide,
        Inches(0.6),
        Inches(1.3),
        Inches(12.2),
        Inches(2.6),
        [
            "Prepared manifests under data/processed/ (JSONL); configuration in configs/*.yaml; seed 27.",
            "Dataset download URLs in configs/data.yaml remain TODO — not invented.",
            f"Empty images retained: {img['empty_images_retained']} of {img['records']}; background audio retained: {aud['background_clips_retained']} of {aud['records']}.",
            f"Class imbalance (max/min): aerial_thermal {img['imbalance_ratio_max_over_min']['aerial_thermal']}, camera_trap_rgb {img['imbalance_ratio_max_over_min']['camera_trap_rgb']}; audio Belize {aud['imbalance_ratio_max_over_min']['belize']}, Vietnam {aud['imbalance_ratio_max_over_min']['vietnam']}.",
            "Handling: rare visual classes oversampled; audio uses class-weighted loss.",
            "BDA artefact aggregates indicator metadata counts — not detector mAP / F1 scores.",
        ],
        size=15,
        spacing=8,
    )

    # Small summary table
    rows, cols = 3, 4
    table = slide.shapes.add_table(rows, cols, Inches(0.6), Inches(4.55), Inches(12.1), Inches(1.7)).table
    headers = ["Corpus", "Records", "Key retention", "Imbalance note"]
    data = [
        ["Images", str(img["records"]), f"{img['empty_images_retained']} empty kept", "ratio 6.0 per domain"],
        ["Audio windows", str(aud["records"]), f"{aud['background_clips_retained']} background kept", "Belize 2.5 / Vietnam 2.0"],
    ]
    for c, h in enumerate(headers):
        _fill_cell(table.cell(0, c), h, size=12, bold=True, color=WHITE, fill=TABLE_HEADER_BG, align=PP_ALIGN.CENTER)
    for r, row in enumerate(data, start=1):
        for c, val in enumerate(row):
            fill = TABLE_ALT if r % 2 == 0 else WHITE
            _fill_cell(table.cell(r, c), val, size=12, color=NAVY, fill=fill, align=PP_ALIGN.CENTER)
    _style_table(table)
    _footer(slide, page, total)
    return slide


def slide_audio_splits(prs, page, total, summary: dict):
    slide = _blank_slide(prs)
    _add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), OFF_WHITE)
    _header_bar(slide, "Audio Site Stratification", "Belize ↔ Vietnam cross-site transfers and in-domain controls")

    splits = summary["audio_splits"]["by_experiment_and_split"]
    experiments = [
        ("belize_to_vietnam", "Belize → Vietnam"),
        ("vietnam_to_belize", "Vietnam → Belize"),
        ("in_domain_belize", "In-domain Belize"),
        ("in_domain_vietnam", "In-domain Vietnam"),
    ]

    rows, cols = 5, 4
    table = slide.shapes.add_table(rows, cols, Inches(0.6), Inches(1.45), Inches(12.1), Inches(3.4)).table
    for c, h in enumerate(["Experiment", "Train", "Val", "Test"]):
        _fill_cell(table.cell(0, c), h, size=13, bold=True, color=WHITE, fill=TABLE_HEADER_BG, align=PP_ALIGN.CENTER)
    for r, (key, label) in enumerate(experiments, start=1):
        block = splits[key]
        vals = [label, str(block["train"]), str(block["val"]), str(block["test"])]
        fill = TABLE_ALT if r % 2 == 0 else WHITE
        for c, val in enumerate(vals):
            _fill_cell(table.cell(r, c), val, size=13, color=NAVY, fill=fill, align=PP_ALIGN.CENTER)

    aud = summary["audio"]
    _bullets(
        slide,
        Inches(0.6),
        Inches(5.15),
        Inches(12.2),
        Inches(1.5),
        [
            f"Site counts: Belize {aud['by_site']['belize']} windows (gunshot {aud['by_site_and_label']['belize']['gunshot']}, background {aud['by_site_and_label']['belize']['background']}); Vietnam {aud['by_site']['vietnam']} (gunshot {aud['by_site_and_label']['vietnam']['gunshot']}, background {aud['by_site_and_label']['vietnam']['background']}).",
            f"Split-assignment rows across four experiments: {summary['audio_splits']['records']} (files: {summary['audio_splits']['files']}).",
            "Cross-site experiments expose domain shift; they are not averaged into one headline score.",
        ],
        size=14,
        spacing=6,
    )
    _footer(slide, page, total)
    return slide


def slide_synthetic_proxy(prs, page, total, summary: dict):
    slide = _blank_slide(prs)
    _add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), OFF_WHITE)
    _header_bar(slide, "Synthetic Proxy Fusion Protocol", "Constructed pairs — not time-aligned field incidents")

    proxy = summary["synthetic_proxy_pairs"]
    _bullets(
        slide,
        Inches(0.6),
        Inches(1.3),
        Inches(6.5),
        Inches(2.4),
        [
            f"Protocol label: {proxy['protocol']} ({proxy['records']} pairs).",
            f"By domain: aerial_thermal {proxy['by_domain']['aerial_thermal']}, camera_trap_rgb {proxy['by_domain']['camera_trap_rgb']}.",
            f"Expected levels: HIGH {proxy['by_expected_level']['HIGH']}, MEDIUM {proxy['by_expected_level']['MEDIUM']}, LOW {proxy['by_expected_level']['LOW']}.",
            "Pairs are built from labelled scenarios for review of fusion rules.",
        ],
        size=15,
        spacing=8,
    )

    # Scenario table
    scenarios = sorted(proxy["by_scenario"].items())
    rows = len(scenarios) + 1
    table = slide.shapes.add_table(rows, 2, Inches(7.3), Inches(1.3), Inches(5.4), Inches(4.8)).table
    _fill_cell(table.cell(0, 0), "Scenario", size=12, bold=True, color=WHITE, fill=TABLE_HEADER_BG, align=PP_ALIGN.CENTER)
    _fill_cell(table.cell(0, 1), "Pairs", size=12, bold=True, color=WHITE, fill=TABLE_HEADER_BG, align=PP_ALIGN.CENTER)
    for i, (name, count) in enumerate(scenarios, start=1):
        fill = TABLE_ALT if i % 2 == 0 else WHITE
        _fill_cell(table.cell(i, 0), name, size=11, color=NAVY, fill=fill)
        _fill_cell(table.cell(i, 1), str(count), size=11, color=NAVY, fill=fill, align=PP_ALIGN.CENTER)

    note = slide.shapes.add_textbox(Inches(0.6), Inches(5.0), Inches(6.5), Inches(1.4))
    tf = note.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    _set_run(p.add_run(), "Honesty note", size=13, bold=True, color=MUTED_GREEN)
    p2 = tf.add_paragraph()
    p2.space_before = Pt(6)
    _set_run(
        p2.add_run(),
        proxy["note"] + " Fusion evaluation under this protocol is a synthetic proxy, not field validation.",
        size=13,
        color=NAVY,
    )
    _footer(slide, page, total)
    return slide


def slide_bda_program(prs, page, total, summary: dict):
    slide = _blank_slide(prs)
    _add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), OFF_WHITE)
    _header_bar(slide, "Big Data Analytics Program", "bda/guardianeye_bda_analysis.py")

    eng = summary["engine"]
    _bullets(
        slide,
        Inches(0.6),
        Inches(1.3),
        Inches(12.2),
        Inches(3.2),
        [
            f"Engine: PySpark {eng['version']} on master {eng['master']} (Java: OpenJDK 21).",
            "Inputs: image manifest, audio windows, site-stratified split CSVs, synthetic proxy pairs, SQLite audit log.",
            "Pattern: map each record to grouping keys; reduceByKey / Spark SQL aggregations for counts and ratios.",
            "One-hour tumbling window over the event log (timestamp → window → alert / latency aggregates).",
            "Outputs: reports/bda/summary.json and reports/bda/aggregation.csv.",
            "Scope: metadata aggregation of prepared indicator corpora — not model scoring.",
        ],
        size=15,
        spacing=8,
    )

    # Event window callout
    win = summary["event_log"]["windows_1h"][0]
    box = _add_rect(slide, Inches(0.6), Inches(5.0), Inches(12.1), Inches(1.35), WHITE)
    box.line.color.rgb = MUTED_GREEN
    box.line.width = Pt(1.5)
    _add_textbox(
        slide, Inches(0.85), Inches(5.15), Inches(11.6), Inches(0.35),
        "1-hour audit window (from event log)",
        size=14, bold=True, color=MUTED_GREEN,
    )
    _add_textbox(
        slide, Inches(0.85), Inches(5.55), Inches(11.6), Inches(0.6),
        f"{win['start']} → {win['end']}   ·   alerted {win['alerted']}   ·   not alerted {win['not_alerted']}   ·   mean latency {win['mean_latency_ms']:.1f} ms",
        size=14, color=NAVY,
    )
    _footer(slide, page, total)
    return slide


def slide_results_images_audio(prs, page, total, summary: dict):
    slide = _blank_slide(prs)
    _add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), OFF_WHITE)
    _header_bar(
        slide,
        "Empirical Results — Image and Audio Counts",
        "From reports/bda/summary.json only. Not detection metrics.",
    )

    img = summary["image"]
    # Image class table
    _add_textbox(slide, Inches(0.5), Inches(1.25), Inches(6), Inches(0.35), "Image records by domain and class", size=14, bold=True, color=NAVY)
    classes = ["animal", "empty", "human", "vehicle"]
    rows, cols = 3, 5
    table = slide.shapes.add_table(rows, cols, Inches(0.5), Inches(1.65), Inches(6.2), Inches(1.7)).table
    headers = ["Domain"] + classes
    for c, h in enumerate(headers):
        _fill_cell(table.cell(0, c), h, size=11, bold=True, color=WHITE, fill=TABLE_HEADER_BG, align=PP_ALIGN.CENTER)
    for r, domain in enumerate(["aerial_thermal", "camera_trap_rgb"], start=1):
        fill = TABLE_ALT if r % 2 == 0 else WHITE
        _fill_cell(table.cell(r, 0), domain, size=10, color=NAVY, fill=fill)
        for c, cls in enumerate(classes, start=1):
            _fill_cell(table.cell(r, c), str(img["by_domain_and_class"][domain][cls]), size=11, color=NAVY, fill=fill, align=PP_ALIGN.CENTER)

    # Chart: image domain totals
    chart_data_img = _chart_categories(
        ["aerial_thermal", "camera_trap_rgb"],
        [("Images", [img["by_domain"]["aerial_thermal"], img["by_domain"]["camera_trap_rgb"]])],
    )
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(7.1),
        Inches(1.25),
        Inches(5.6),
        Inches(2.5),
        chart_data_img,
    ).chart
    chart.has_legend = False
    plot = chart.plots[0]
    plot.gap_width = 80
    series = plot.series[0]
    series.format.fill.solid()
    series.format.fill.fore_color.rgb = NAVY

    # Audio table
    aud = summary["audio"]
    _add_textbox(slide, Inches(0.5), Inches(4.0), Inches(6), Inches(0.35), "Audio windows by site and label", size=14, bold=True, color=NAVY)
    rows, cols = 3, 3
    table2 = slide.shapes.add_table(rows, cols, Inches(0.5), Inches(4.4), Inches(6.2), Inches(1.55)).table
    for c, h in enumerate(["Site", "gunshot", "background"]):
        _fill_cell(table2.cell(0, c), h, size=11, bold=True, color=WHITE, fill=TABLE_HEADER_BG, align=PP_ALIGN.CENTER)
    for r, site in enumerate(["belize", "vietnam"], start=1):
        fill = TABLE_ALT if r % 2 == 0 else WHITE
        _fill_cell(table2.cell(r, 0), site, size=11, color=NAVY, fill=fill)
        _fill_cell(table2.cell(r, 1), str(aud["by_site_and_label"][site]["gunshot"]), size=11, color=NAVY, fill=fill, align=PP_ALIGN.CENTER)
        _fill_cell(table2.cell(r, 2), str(aud["by_site_and_label"][site]["background"]), size=11, color=NAVY, fill=fill, align=PP_ALIGN.CENTER)

    chart_data_aud = _chart_categories(
        ["belize", "vietnam"],
        [
            ("gunshot", [aud["by_site_and_label"]["belize"]["gunshot"], aud["by_site_and_label"]["vietnam"]["gunshot"]]),
            ("background", [aud["by_site_and_label"]["belize"]["background"], aud["by_site_and_label"]["vietnam"]["background"]]),
        ],
    )
    chart2 = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(7.1),
        Inches(3.95),
        Inches(5.6),
        Inches(2.55),
        chart_data_aud,
    ).chart
    chart2.has_legend = True
    chart2.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart2.legend.include_in_layout = False
    colors = [MUTED_GREEN, NAVY_MID]
    for i, series in enumerate(chart2.plots[0].series):
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = colors[i]

    _footer(slide, page, total)
    return slide


def _chart_categories(categories, series_list):
    from pptx.chart.data import CategoryChartData

    chart_data = CategoryChartData()
    chart_data.categories = categories
    for name, values in series_list:
        chart_data.add_series(name, values)
    return chart_data


def slide_results_proxy_events(prs, page, total, summary: dict):
    slide = _blank_slide(prs)
    _add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), OFF_WHITE)
    _header_bar(
        slide,
        "Empirical Results — Proxy Pairs and Audit Log",
        "Metadata counts only. No fabricated mAP / F1 / precision / recall.",
    )

    proxy = summary["synthetic_proxy_pairs"]
    ev = summary["event_log"]

    # Left: proxy level chart
    _add_textbox(slide, Inches(0.5), Inches(1.25), Inches(6), Inches(0.35), "Synthetic proxy pairs by expected level", size=14, bold=True, color=NAVY)
    chart_data = _chart_categories(
        ["HIGH", "MEDIUM", "LOW"],
        [("Pairs", [proxy["by_expected_level"]["HIGH"], proxy["by_expected_level"]["MEDIUM"], proxy["by_expected_level"]["LOW"]])],
    )
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(0.5),
        Inches(1.65),
        Inches(6.0),
        Inches(3.4),
        chart_data,
    ).chart
    chart.has_legend = False
    series = chart.plots[0].series[0]
    series.format.fill.solid()
    series.format.fill.fore_color.rgb = MUTED_GREEN

    # Right: audit facts
    card = _add_rect(slide, Inches(7.0), Inches(1.35), Inches(5.7), Inches(4.5), WHITE)
    card.line.color.rgb = LIGHT_LINE
    _add_rect(slide, Inches(7.0), Inches(1.35), Inches(5.7), Inches(0.5), NAVY)
    _add_textbox(slide, Inches(7.2), Inches(1.42), Inches(5.3), Inches(0.4), "SQLite audit log (aggregated)", size=14, bold=True, color=WHITE)

    facts = [
        f"Event records: {ev['records']}",
        f"Alerted: {ev['alerted']}  ·  Not alerted: {ev['not_alerted']}",
        f"Risk level: {', '.join(f'{k}={v}' for k, v in ev['by_risk_level'].items())}",
        f"Domain: {', '.join(f'{k}={v}' for k, v in ev['by_domain'].items())}",
        f"Reason: {', '.join(f'{k}={v}' for k, v in ev['by_reason'].items())}",
        f"Proxy pairs total: {proxy['records']} (protocol={proxy['protocol']})",
        f"Seed: {summary['seed']}",
    ]
    _bullets(slide, Inches(7.2), Inches(2.1), Inches(5.3), Inches(3.4), facts, size=14, spacing=8)

    note = slide.shapes.add_textbox(Inches(0.5), Inches(5.4), Inches(12.3), Inches(1.1))
    tf = note.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    _set_run(p.add_run(), "Checkpoint status: ", size=13, bold=True, color=NAVY)
    _set_run(
        p.add_run(),
        "models/ contains no trained weights. Detection metrics are not reported. "
        "summary.json note: analytical summary of prepared indicator metadata — not model metrics.",
        size=13,
        color=NAVY,
    )
    _footer(slide, page, total)
    return slide


def slide_serving(prs, page, total, summary: dict):
    slide = _blank_slide(prs)
    _add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), OFF_WHITE)
    _header_bar(slide, "Serving and Honesty")
    _bullets(
        slide,
        Inches(0.6),
        Inches(1.35),
        Inches(12.2),
        Inches(5.2),
        [
            "Service: FastAPI on port 8742; Gradio dashboard at /dashboard.",
            "GET /health for liveness; GET /model-info reports whether each checkpoint loaded (smoke / trained / missing).",
            "POST /events accepts visual and/or audio uploads; GET /events returns the audit trail.",
            "Every decision is logged, including non-alerts.",
            "Current working tree: no checkpoint under models/ — weights are unreliable / missing.",
            f"Observed audit event reason in BDA aggregation: unreliable_weights (risk LOW; not alerted).",
            "Smoke checkpoints, if present, prove scripts run; they are not field performance.",
        ],
        size=16,
        spacing=10,
    )
    _footer(slide, page, total)
    return slide


def slide_conclusion(prs, page, total, summary: dict):
    slide = _blank_slide(prs)
    _add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), OFF_WHITE)
    _header_bar(slide, "Conclusion")
    img = summary["image"]
    aud = summary["audio"]
    proxy = summary["synthetic_proxy_pairs"]
    _bullets(
        slide,
        Inches(0.6),
        Inches(1.35),
        Inches(12.2),
        Inches(5.4),
        [
            "GuardianEye delivers a completed late-fusion indicator pipeline for wildlife evidence review.",
            "Image and audio branches remain domain- and site-aware; empty and background samples are retained.",
            "Fusion pairs are labelled as a synthetic proxy protocol — not field-validated poaching detection.",
            f"PySpark BDA job aggregated the prepared corpus: {img['records']} images, {aud['records']} audio windows, "
            f"{summary['audio_splits']['records']} split rows, {proxy['records']} proxy pairs, {summary['event_log']['records']} audit event(s).",
            "No detection accuracy figures are claimed without checkpoints; models/ has none in this tree.",
            "One detection remains an indicator, not proof of poaching.",
        ],
        size=16,
        spacing=10,
    )
    _footer(slide, page, total)
    return slide


def slide_thanks(prs, page, total):
    slide = _blank_slide(prs)
    _add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), NAVY)
    _add_rect(slide, Inches(0), Inches(0), Inches(0.18), Inches(7.5), MUTED_GREEN)

    _add_textbox(
        slide, Inches(0.7), Inches(1.8), Inches(12), Inches(0.8),
        "Thank You",
        size=40, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
    )
    _add_textbox(
        slide, Inches(0.7), Inches(2.7), Inches(12), Inches(0.4),
        "Questions and Discussion",
        size=20, color=GREEN_SOFT, align=PP_ALIGN.CENTER,
    )

    members = [
        "Akhil Chivakula — 23070126009",
        "Dipti Kothari — 23070126040",
        "Khurinji Malar — 23070126059",
        "Sukhada Dhingra — 24070126505",
    ]
    box = slide.shapes.add_textbox(Inches(3.5), Inches(3.5), Inches(6.3), Inches(2.2))
    tf = box.text_frame
    tf.word_wrap = True
    for i, m in enumerate(members):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        p.space_before = Pt(4)
        _set_run(p.add_run(), m, size=15, color=WHITE)

    _add_textbox(
        slide, Inches(0.7), Inches(5.85), Inches(12), Inches(0.35),
        "Guide: Dr. Archana Choudary  ·  Head of the Department: Dr. Sumit Kumar",
        size=12, color=GREEN_SOFT, align=PP_ALIGN.CENTER,
    )
    _add_textbox(
        slide, Inches(0.7), Inches(6.25), Inches(12), Inches(0.35),
        "BDA 0701260704  ·  SIT AIML  ·  AY 2026–2027",
        size=12, color=GREEN_SOFT, align=PP_ALIGN.CENTER,
    )
    return slide


def build() -> Path:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Build slide list dynamically so footers know total (title + content + thanks)
    # We use 13 slides total.
    total = 13

    slide_title(prs, total)
    slide_problem(prs, 2, total)
    slide_objectives(prs, 3, total)
    slide_architecture(prs, 4, total)
    slide_data_pipeline(prs, 5, total, summary)
    slide_audio_splits(prs, 6, total, summary)
    slide_synthetic_proxy(prs, 7, total, summary)
    slide_bda_program(prs, 8, total, summary)
    slide_results_images_audio(prs, 9, total, summary)
    slide_results_proxy_events(prs, 10, total, summary)
    slide_serving(prs, 11, total, summary)
    slide_conclusion(prs, 12, total, summary)
    slide_thanks(prs, 13, total)

    OUT_DOCS.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT_DOCS)
    shutil.copy2(OUT_DOCS, OUT_ROOT)
    return OUT_DOCS


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
    print(f"Copied {OUT_ROOT}")
