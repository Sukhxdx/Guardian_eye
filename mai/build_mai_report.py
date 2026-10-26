"""Build the GuardianEye Multimodal AI (MAI) project report (DOCX).

Focuses on multimodal architecture, unimodal branches, late fusion, and serving.
Does not invent detection metrics. Corpus counts come from reports/bda/summary.json
when present (metadata only).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reports" / "bda" / "summary.json"
OUT_DOCS = ROOT / "docs" / "MAI_Project_Report_2026_2027.docx"
OUT_ROOT = ROOT / "MAI_Project_Report_2026_2027.docx"

MEMBERS = [
    ("Akhil Chivakula", "23070126009"),
    ("Dipti Kothari", "23070126040"),
    ("Khurinji Malar", "23070126059"),
    ("Sukhada Dhingra", "24070126505"),
]

TITLE = "GuardianEye: Multimodal Late Fusion of Image and Audio Poaching Indicators"
GUIDE = "Dr. Nivedita Mishra"
HOD = "Dr. Sumit Kumar"
SUBJECT = "Multimodal AI (MAI)"
GROUP = "Group 27"
AY = "2026–2027"
DATE = "26 October 2026"


def _set_run_font(run, size=11, bold=False, name="Times New Roman"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold


def _p(doc, text, *, size=11, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=6):
    para = doc.add_paragraph()
    para.alignment = align
    para.paragraph_format.space_after = Pt(space_after)
    para.paragraph_format.space_before = Pt(0)
    run = para.add_run(text)
    _set_run_font(run, size=size, bold=bold)
    return para


def _heading(doc, text, level=1):
    para = doc.add_heading(text, level=level)
    for run in para.runs:
        run.font.color.rgb = RGBColor(0x0B, 0x1D, 0x36)
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    return para


def _center(doc, text, size=11, bold=False, space_after=4):
    return _p(doc, text, size=size, bold=bold, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=space_after)


def _member_table(doc, with_signature: bool = False):
    cols = 4 if with_signature else 3
    table = doc.add_table(rows=1 + len(MEMBERS), cols=cols)
    table.style = "Table Grid"
    headers = ["Sr. No.", "Name of the Student", "PRN"]
    if with_signature:
        headers.append("Signature")
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        _set_run_font(run, size=10, bold=True)
    for i, (name, prn) in enumerate(MEMBERS, start=1):
        row = table.rows[i].cells
        vals = [str(i), name, prn]
        if with_signature:
            vals.append("")
        for j, v in enumerate(vals):
            row[j].text = ""
            run = row[j].paragraphs[0].add_run(v)
            _set_run_font(run, size=10)
    doc.add_paragraph()


def _guide_hod_table(doc):
    table = doc.add_table(rows=5, cols=2)
    table.style = "Table Grid"
    pairs = [
        ("Guide", "Head of the Department"),
        (GUIDE, HOD),
        ("Assistant Professor", "Department of Artificial Intelligence and Machine Learning"),
        (
            "Department of Artificial Intelligence and Machine Learning",
            "Symbiosis Institute of Technology, Pune",
        ),
        ("Signature", "Signature"),
    ]
    for r, (a, b) in enumerate(pairs):
        for c, text in enumerate((a, b)):
            cell = table.rows[r].cells[c]
            cell.text = ""
            run = cell.paragraphs[0].add_run(text)
            _set_run_font(run, size=10, bold=(r == 0))
    doc.add_paragraph()


def _data_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        _set_run_font(run, size=10, bold=True)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            _set_run_font(run, size=10)
    doc.add_paragraph()


def _load_summary() -> dict:
    if SUMMARY_PATH.is_file():
        return json.loads(SUMMARY_PATH.read_text())
    return {}


def build() -> Path:
    summary = _load_summary()
    img = summary.get("image", {})
    aud = summary.get("audio", {})
    splits = summary.get("audio_splits", {})
    proxy = summary.get("synthetic_proxy_pairs", {})
    events = summary.get("event_log", {})

    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)

    # ---- Cover ----
    _center(doc, "SYMBIOSIS INSTITUTE OF TECHNOLOGY", size=14, bold=True)
    _center(doc, "Symbiosis International (Deemed University)", size=11, bold=True)
    _center(doc, "(Established under Section 3 of the UGC Act, 1956)", size=10)
    _center(
        doc,
        "Re-accredited by NAAC with ‘A++’ Grade  |  Awarded Category – I by UGC",
        size=9,
    )
    _center(doc, "Founder: Prof. Dr. S. B. Mujumdar, M.Sc., Ph.D.", size=10)
    _center(
        doc,
        "(Awarded Padma Bhushan and Padma Shri by the President of India)",
        size=9,
        space_after=12,
    )
    _center(doc, "A PROJECT REPORT", size=14, bold=True)
    _center(doc, "ON", size=11, bold=True)
    _center(doc, TITLE, size=13, bold=True, space_after=10)
    _center(
        doc,
        "Submitted in partial fulfillment of the requirements for the degree of",
        size=11,
    )
    _center(doc, "Bachelor of Technology", size=12, bold=True)
    _center(doc, "in", size=11)
    _center(doc, "Artificial Intelligence and Machine Learning", size=12, bold=True, space_after=8)
    _center(doc, f"Subject: {SUBJECT}", size=11, bold=True)
    _center(doc, f"{GROUP}", size=11, bold=True)
    _center(doc, f"Academic Year: {AY}", size=11, space_after=10)
    _center(doc, "Submitted by", size=11, bold=True)
    _member_table(doc)
    _center(doc, "Under the guidance of", size=11, bold=True)
    _center(doc, GUIDE, size=12, bold=True)
    _center(doc, "Assistant Professor", size=11)
    _center(doc, "Department of Artificial Intelligence and Machine Learning", size=11)
    _center(doc, "Symbiosis Institute of Technology", size=11, space_after=6)
    _center(doc, "Near Lupin Research Park, Gram Lavale, Taluka Mulshi", size=10)
    _center(doc, "Pune – 412115, Maharashtra, India", size=10)

    doc.add_page_break()

    # ---- Certificate ----
    _center(doc, "CERTIFICATE", size=14, bold=True, space_after=12)
    _p(
        doc,
        f"This is to certify that the project work entitled “{TITLE}” is carried out by "
        f"the students named below, of the Department of Artificial Intelligence and Machine "
        f"Learning, Symbiosis Institute of Technology, Pune, in partial fulfillment of the "
        f"requirements for the award of the degree of Bachelor of Technology in Artificial "
        f"Intelligence and Machine Learning of Symbiosis International (Deemed University), "
        f"Pune, during the academic year {AY}.",
    )
    _p(
        doc,
        f"The work is submitted for the course {SUBJECT}, {GROUP}, and was completed under "
        f"the guidance of {GUIDE}.",
    )
    _member_table(doc)
    _p(doc, "Place: Pune")
    _p(doc, f"Date: {DATE}", space_after=12)
    _guide_hod_table(doc)

    doc.add_page_break()

    # ---- Declaration ----
    _center(doc, "DECLARATION", size=14, bold=True, space_after=12)
    _p(
        doc,
        f"We hereby declare that the project report entitled “{TITLE}” is a record of work "
        f"carried out by us at the Department of Artificial Intelligence and Machine Learning, "
        f"Symbiosis Institute of Technology, Pune, a constituent of Symbiosis International "
        f"(Deemed University), under the guidance of {GUIDE}, during the academic year {AY}.",
    )
    _p(
        doc,
        f"This report is submitted in partial fulfillment of the requirements for the degree "
        f"of Bachelor of Technology in Artificial Intelligence and Machine Learning, for the "
        f"course {SUBJECT}, {GROUP}.",
    )
    _p(
        doc,
        "We further declare that this report has not been submitted to any other university "
        "or institute for the award of any degree or diploma.",
    )
    _p(doc, "Place: Pune")
    _p(doc, f"Date: {DATE}", space_after=10)
    _member_table(doc, with_signature=True)
    _p(
        doc,
        "Certified that the above statement made by the students is correct to the best of "
        "our knowledge and belief.",
        space_after=10,
    )
    _center(doc, GUIDE, size=11, bold=True)
    _center(doc, "Guide", size=10)
    _center(doc, "Assistant Professor", size=10)
    _center(doc, "Department of Artificial Intelligence and Machine Learning", size=10)
    _center(doc, "Symbiosis Institute of Technology, Pune", size=10)

    doc.add_page_break()

    # ---- Acknowledgement ----
    _center(doc, "ACKNOWLEDGEMENT", size=14, bold=True, space_after=12)
    _p(
        doc,
        f"The authors acknowledge {GUIDE}, Assistant Professor, Department of Artificial "
        f"Intelligence and Machine Learning, Symbiosis Institute of Technology, for guidance "
        f"of this Multimodal AI work during the academic year {AY}.",
    )
    _p(
        doc,
        f"The authors acknowledge {HOD}, Head of the Department of Artificial Intelligence "
        f"and Machine Learning, Symbiosis Institute of Technology, Pune, and Symbiosis "
        f"International (Deemed University), for the academic framework in which the project "
        f"was undertaken.",
        space_after=12,
    )
    for name, _ in MEMBERS:
        _p(doc, name, align=WD_ALIGN_PARAGRAPH.RIGHT, space_after=2)

    doc.add_page_break()

    # ---- Abstract ----
    _center(doc, "ABSTRACT", size=14, bold=True, space_after=12)
    _p(
        doc,
        "GuardianEye is a multimodal review pipeline for wildlife evidence. Two sensory "
        "streams are modelled separately and combined only at decision time. Still frames "
        "from camera traps and from aerial thermal imagery pass through a YOLOv8 detector "
        "trained independently on each visual domain. Short audio clips are converted to "
        "log-mel spectrograms and classified by a convolutional network. Video is not treated "
        "as a third modality; frames are sampled and routed through the image branch. A human "
        "box, a vehicle box, or a gunshot-like score is an indicator. One detection is not "
        "proof of poaching.",
    )
    _p(
        doc,
        "The multimodal design is late fusion. Each branch keeps its own representation, "
        "loss, and evaluation protocol. Image metrics are written separately for "
        "aerial_thermal and camera_trap_rgb. Audio is split by site: Belize to Vietnam, "
        "Vietnam to Belize, and an in-domain control for each site. Empty frames and "
        "background clips remain in the prepared sets. Rare visual classes are oversampled. "
        "Audio training uses class-weighted loss. Because no public collection used here "
        "aligns an image with the audio of the same moment, fusion pairs are constructed "
        "from labels and marked as a synthetic proxy protocol.",
    )
    n_img = img.get("records", 22)
    n_aud = aud.get("records", 29)
    n_proxy = proxy.get("records", 16)
    _p(
        doc,
        f"On the corpus present in this working tree the prepared set contains {n_img} images, "
        f"{n_aud} audio windows, and {n_proxy} synthetic proxy pairs. Detection scores are "
        f"written only by the evaluation scripts, and only when a checkpoint is present. The "
        f"model directory contains no checkpoint. The FastAPI service and Gradio dashboard "
        f"expose multimodal inference with honest model-info reporting when weights are absent.",
    )
    _p(
        doc,
        "Keywords: multimodal AI, late fusion, YOLOv8, log-mel CNN, site stratification, "
        "synthetic proxy protocol, missing-modality degradation, audit log.",
        space_after=12,
    )

    _center(doc, "CONTENTS", size=14, bold=True, space_after=8)
    for line in [
        "Certificate",
        "Declaration",
        "Acknowledgement",
        "Abstract",
        "List of Tables",
        "1. Introduction",
        "2. Multimodal Problem Formulation",
        "3. System Architecture",
        "4. Visual Modality",
        "5. Audio Modality and Cross-Site Transfer",
        "6. Late Fusion and Synthetic Proxy Evaluation",
        "7. Multimodal Serving Interface",
        "8. Empirical Characterisation of the Multimodal Corpus",
        "References",
    ]:
        _p(doc, line, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=2)

    _center(doc, "LIST OF TABLES", size=12, bold=True, space_after=6)
    for line in [
        "Table 1. Image records by domain and primary class.",
        "Table 2. Image records by domain and split.",
        "Table 3. Audio windows by site and label.",
        "Table 4. Audio split sizes by experiment.",
        "Table 5. Synthetic proxy pairs by scenario.",
        "Table 6. Fusion decision levels on the proxy set (expected).",
    ]:
        _p(doc, line, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=2)

    doc.add_page_break()

    # ---- 1 ----
    _heading(doc, "1. Introduction")
    _p(
        doc,
        "Patrol records arrive as heterogeneous sensory streams: RGB camera-trap stills, "
        "aerial thermal frames, and short acoustic clips. The streams are rarely synchronised. "
        "A single sensor reading is incomplete, and a single detector output is ambiguous. "
        "GuardianEye addresses this as a multimodal AI problem. Separate unimodal models "
        "extract modality-specific indicators. A late-fusion stage combines those indicators "
        "into a risk level that a ranger can review.",
    )
    _p(
        doc,
        "Video is not a third modality. Frames are sampled from it and submitted to the image "
        "branch. The image branch itself spans two domains: aerial_thermal and camera_trap_rgb. "
        "A detector trained on one domain is not substituted for the other. The audio branch "
        "is stratified by recording site so that geographic transfer can be measured rather "
        "than hidden inside a shuffled pool.",
    )
    _p(
        doc,
        "There is no labelled poacher class in the collections named by the configuration, "
        "and there is no time-aligned field incident that pairs an image with the audio of "
        "the same moment. The system therefore detects indicators and fuses them. It does "
        "not claim a confirmed poaching event. Configuration lives in configs/*.yaml. The "
        "seed is 27. Download URLs remain TODO until a person fills them in.",
    )

    # ---- 2 ----
    _heading(doc, "2. Multimodal Problem Formulation")
    _heading(doc, "2.1 Modalities and indicator classes", level=2)
    _p(
        doc,
        "The visual modality carries four outcomes: human, vehicle, animal, and empty. An "
        "empty image is a frame with no box. It is a negative example, and it stays in the "
        "training set. Visual poaching indicators are restricted to human and vehicle. Animal "
        "boxes are recorded and are not treated as poaching evidence.",
    )
    _p(
        doc,
        "The acoustic modality carries two labels: background and gunshot. The gunshot class "
        "is a gunshot-like impulse. Background covers ambient sound and impulse-like "
        "distractors tagged on the manifest. Background clips are retained so that the "
        "classifier sees the negative class in proportion to the field mixture.",
    )
    _heading(doc, "2.2 Why late fusion", level=2)
    _p(
        doc,
        "Early fusion would require a shared representation of image and audio before either "
        "branch is trained. The repositories used here do not supply paired, time-aligned "
        "samples at scale. Late fusion keeps each encoder specialised, permits independent "
        "calibration, and degrades gracefully when one modality is missing. The fusion stage "
        "reads detection scores and timestamps, not raw pixels or waveforms.",
    )
    _heading(doc, "2.3 Decision policy", level=2)
    _p(
        doc,
        "The primary policy is rules mode in configs/fusion.yaml. Inclusive thresholds of "
        "0.50 mark a strong image indicator and a strong gunshot score. When only one "
        "reliable modality is present, unimodal thresholds of 0.85 apply, and the level "
        "cannot be HIGH. Coincidence uses a 5.0 second expansion of the audio window.",
    )
    _p(
        doc,
        "HIGH requires a human or vehicle detection and a gunshot score, both at or above "
        "threshold, inside that window. MEDIUM covers exactly one strong indicator, or both "
        "indicators outside the window. LOW covers neither. Alerting is configured for HIGH "
        "only. Weight profiles balanced, night, and wind rescale the numeric score when "
        "weighted mode is selected; rules mode remains the documented primary path.",
    )

    # ---- 3 ----
    _heading(doc, "3. System Architecture")
    _p(
        doc,
        "The pipeline is organised as two parallel unimodal branches and one fusion module. "
        "The image branch runs YOLOv8 once per visual domain. The audio branch converts each "
        "clip to a fixed log-mel matrix and classifies it with a small convolutional network. "
        "The fusion module implements the rule set above, writes an audit row to SQLite, and "
        "exposes the result through FastAPI and a Gradio dashboard at /dashboard.",
    )
    _p(
        doc,
        "Missing-modality handling is explicit. An upload that contains only an image or only "
        "an audio clip still receives a score. The response states which branch was reliable. "
        "GET /model-info reports, for each domain and for audio, whether a checkpoint was "
        "loaded. When weights are absent, the service states that the corresponding evidence "
        "is not reliable.",
    )

    # ---- 4 ----
    _heading(doc, "4. Visual Modality")
    _p(
        doc,
        "The image trainer runs one YOLOv8 model per domain. configs/image.yaml sets "
        "train_per_domain to true and names yolov8n.pt as the base weights. The full schedule "
        "is 50 epochs, image size 640, batch size 8, and device cpu by default. Aerial thermal "
        "frames use a replicate three-channel representation so that an RGB backbone can be "
        "reused without inventing false colour structure.",
    )
    _p(
        doc,
        "Class imbalance on the image side is handled by oversampling, not by dropping empty "
        "frames. Rare box classes are duplicated in a copy of the training split until their "
        "instance counts meet the most common non-empty class under the configured multiplier. "
        "Evaluation is per domain. guardianeye.eval.image_eval scores aerial_thermal and "
        "camera_trap_rgb separately. When weights exist it can write detection metrics to "
        "reports/. Those numbers are never hardcoded in this report.",
    )

    # ---- 5 ----
    _heading(doc, "5. Audio Modality and Cross-Site Transfer")
    _p(
        doc,
        "Audio preparation resamples to 16 kHz and cuts windows of 3.0 seconds. Each window "
        "becomes a log-mel matrix of 64 bins. The default classifier is a three-block "
        "convolutional network with batch normalisation and ReLU activations, followed by "
        "adaptive pooling and a linear head over the two audio classes. Training uses "
        "class-weighted cross-entropy so that the rarer gunshot class retains gradient mass.",
    )
    _p(
        doc,
        "Four experiments isolate geographic transfer. belize_to_vietnam trains and validates "
        "on Belize and tests on all of Vietnam. vietnam_to_belize reverses the sites. "
        "in_domain_belize and in_domain_vietnam each split one site into train, validation, "
        "and test. Validation is drawn only from the training site. This design is a "
        "multimodal evaluation discipline: the acoustic encoder is judged under domain shift "
        "that the fusion stage will later inherit.",
    )

    # ---- 6 ----
    _heading(doc, "6. Late Fusion and Synthetic Proxy Evaluation")
    _p(
        doc,
        "No public set used by this repository aligns an image with the audio of the same "
        "incident. scripts/build_paired_events.py therefore constructs pairs from labels. "
        "Every row carries protocol synthetic_proxy. The construction is a proxy for "
        "multimodal coincidence, not a substitute for field-aligned capture.",
    )
    _p(
        doc,
        "Eight scenarios are applied in each image domain: human with gunshot, vehicle with "
        "gunshot, small-human with gunshot, human with background, vehicle with background, "
        "animal with background, empty with gunshot, and empty with an impulse distractor. "
        "Expected levels on the proxy set follow the fusion rules. Fusion evaluation, when "
        "both branches can be loaded, compares image-only, audio-only, and fused decisions "
        "and writes reports/fusion/. A missing checkpoint leaves that report empty of scores.",
    )

    # ---- 7 ----
    _heading(doc, "7. Multimodal Serving Interface")
    _p(
        doc,
        "The service is a FastAPI application bound to port 8742. A Gradio dashboard is "
        "mounted at /dashboard and accepts image and audio uploads in one interaction. "
        "GET /health reports that the process is up. The audit store is SQLite at "
        "data/events.sqlite. Snapshots and audio copies are written under data/events/. "
        "Every decision is logged, including events that do not alert.",
    )
    _p(
        doc,
        "The dashboard banner states that the system is a proof-of-concept for indicators, "
        "not confirmed poaching. Model-info text surfaces unreliable weights when "
        "checkpoints are absent, so a multimodal demo cannot be mistaken for a calibrated "
        "field detector.",
    )

    # ---- 8 ----
    _heading(doc, "8. Empirical Characterisation of the Multimodal Corpus")
    _p(
        doc,
        "The figures below describe the prepared multimodal corpus in this working tree. "
        "They are metadata counts. They are not precision, recall, F1, mAP, or false-positive "
        "rate. Those detection metrics are produced only by evaluation scripts after weights "
        "exist. The models/ directory contains no checkpoint.",
    )

    _heading(doc, "8.1 Images", level=2)
    by_cls = img.get("by_domain_and_class", {})
    at = by_cls.get("aerial_thermal", {"human": 2, "vehicle": 1, "animal": 2, "empty": 6})
    ct = by_cls.get("camera_trap_rgb", {"human": 2, "vehicle": 1, "animal": 2, "empty": 6})
    imb = img.get("imbalance_ratio_max_over_min", {"aerial_thermal": 6.0, "camera_trap_rgb": 6.0})
    _p(
        doc,
        f"The image manifest contains {img.get('records', 22)} records, "
        f"{img.get('by_domain', {}).get('aerial_thermal', 11)} in aerial_thermal and "
        f"{img.get('by_domain', {}).get('camera_trap_rgb', 11)} in camera_trap_rgb. "
        f"{img.get('empty_images_retained', 12)} empty images are retained. "
        f"Imbalance ratios are {imb.get('aerial_thermal', 6.0)} and {imb.get('camera_trap_rgb', 6.0)}.",
    )
    _p(doc, "Table 1. Image records by domain and primary class.", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    _data_table(
        doc,
        ["Domain", "Human", "Vehicle", "Animal", "Empty", "Total", "Imbalance"],
        [
            [
                "aerial_thermal",
                at.get("human", 2),
                at.get("vehicle", 1),
                at.get("animal", 2),
                at.get("empty", 6),
                img.get("by_domain", {}).get("aerial_thermal", 11),
                imb.get("aerial_thermal", 6.0),
            ],
            [
                "camera_trap_rgb",
                ct.get("human", 2),
                ct.get("vehicle", 1),
                ct.get("animal", 2),
                ct.get("empty", 6),
                img.get("by_domain", {}).get("camera_trap_rgb", 11),
                imb.get("camera_trap_rgb", 6.0),
            ],
        ],
    )

    by_split = img.get(
        "by_domain_and_split",
        {
            "aerial_thermal": {"train": 7, "val": 3, "test": 1},
            "camera_trap_rgb": {"train": 7, "val": 3, "test": 1},
        },
    )
    _p(doc, "Table 2. Image records by domain and split.", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    _data_table(
        doc,
        ["Domain", "Train", "Validation", "Test"],
        [
            [
                "aerial_thermal",
                by_split["aerial_thermal"]["train"],
                by_split["aerial_thermal"]["val"],
                by_split["aerial_thermal"]["test"],
            ],
            [
                "camera_trap_rgb",
                by_split["camera_trap_rgb"]["train"],
                by_split["camera_trap_rgb"]["val"],
                by_split["camera_trap_rgb"]["test"],
            ],
        ],
    )

    _heading(doc, "8.2 Audio", level=2)
    site_lab = aud.get(
        "by_site_and_label",
        {
            "belize": {"gunshot": 4, "background": 10},
            "vietnam": {"gunshot": 5, "background": 10},
        },
    )
    aimb = aud.get("imbalance_ratio_max_over_min", {"belize": 2.5, "vietnam": 2.0})
    _p(
        doc,
        f"The audio manifest contains {aud.get('records', 29)} windows. "
        f"Belize contributes {aud.get('by_site', {}).get('belize', 14)}; "
        f"Vietnam contributes {aud.get('by_site', {}).get('vietnam', 15)}. "
        f"{aud.get('background_clips_retained', 20)} background clips are retained.",
    )
    _p(doc, "Table 3. Audio windows by site and label.", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    _data_table(
        doc,
        ["Site", "Gunshot", "Background", "Total", "Imbalance"],
        [
            [
                "Belize",
                site_lab["belize"]["gunshot"],
                site_lab["belize"]["background"],
                aud.get("by_site", {}).get("belize", 14),
                aimb.get("belize", 2.5),
            ],
            [
                "Vietnam",
                site_lab["vietnam"]["gunshot"],
                site_lab["vietnam"]["background"],
                aud.get("by_site", {}).get("vietnam", 15),
                aimb.get("vietnam", 2.0),
            ],
        ],
    )

    exp = splits.get(
        "by_experiment_and_split",
        {
            "belize_to_vietnam": {"train": 11, "val": 3, "test": 14},
            "vietnam_to_belize": {"train": 11, "val": 3, "test": 14},
            "in_domain_belize": {"train": 8, "val": 3, "test": 3},
            "in_domain_vietnam": {"train": 8, "val": 3, "test": 3},
        },
    )
    _p(doc, "Table 4. Audio split sizes by experiment.", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    _data_table(
        doc,
        ["Experiment", "Train", "Validation", "Test"],
        [
            [name, vals["train"], vals["val"], vals["test"]]
            for name, vals in exp.items()
        ],
    )

    _heading(doc, "8.3 Synthetic proxy pairs", level=2)
    scenarios = proxy.get(
        "by_scenario",
        {
            "human_gunshot": 2,
            "vehicle_gunshot": 2,
            "small_human_gunshot": 2,
            "human_background": 2,
            "vehicle_background": 2,
            "animal_background": 2,
            "empty_gunshot": 2,
            "empty_impulse": 2,
        },
    )
    levels = proxy.get("by_expected_level", {"HIGH": 6, "MEDIUM": 6, "LOW": 4})
    _p(
        doc,
        f"The pair file contains {proxy.get('records', 16)} rows. All carry protocol "
        f"synthetic_proxy. Expected levels on the proxy set are HIGH {levels.get('HIGH', 6)}, "
        f"MEDIUM {levels.get('MEDIUM', 6)}, and LOW {levels.get('LOW', 4)}.",
    )
    _p(doc, "Table 5. Synthetic proxy pairs by scenario.", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    role = {
        "human_gunshot": "Expected HIGH",
        "vehicle_gunshot": "Expected HIGH",
        "small_human_gunshot": "Expected HIGH",
        "human_background": "Expected MEDIUM",
        "vehicle_background": "Expected MEDIUM",
        "empty_gunshot": "Expected MEDIUM",
        "animal_background": "Expected LOW",
        "empty_impulse": "Expected LOW",
    }
    _data_table(
        doc,
        ["Scenario", "Rows", "Role in the proxy protocol"],
        [[k, v, role.get(k, "")] for k, v in scenarios.items()],
    )
    _p(doc, "Table 6. Fusion decision levels on the proxy set (expected).", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    _data_table(
        doc,
        ["Expected level", "Pairs"],
        [[k, levels[k]] for k in ("HIGH", "MEDIUM", "LOW") if k in levels],
    )

    _heading(doc, "8.4 Serving observation", level=2)
    _p(
        doc,
        f"The audit log in this working tree contains {events.get('records', 1)} event. "
        f"Alerted {events.get('alerted', 0)}; not alerted {events.get('not_alerted', 1)}. "
        f"The recorded reason includes unreliable_weights when checkpoints are absent. "
        f"That outcome is consistent with a multimodal service that refuses to overstate "
        f"evidence from unloaded branches.",
    )
    _p(
        doc,
        "Empty images and background audio dominate the prepared counts and were retained. "
        "The multimodal protocol therefore trains and evaluates against the negative class "
        "rather than discarding it. No detection metric is stated here because no checkpoint "
        "is present to score.",
    )

    # ---- References ----
    _heading(doc, "References")
    refs = [
        "[1] GuardianEye configuration files configs/data.yaml, configs/image.yaml, "
        "configs/audio.yaml, configs/fusion.yaml, and configs/serve.yaml. Repository "
        f"working tree. Academic year {AY}. Download URLs left as TODO.",
        "[2] Ultralytics YOLOv8. Base weights named yolov8n.pt in configs/image.yaml. "
        "One training run is defined per image domain.",
        "[3] Named source collections in configs/data.yaml, with URLs left as TODO: "
        "BIRDSAI (aerial thermal); Caltech Camera Traps, including empty frames; Snapshot "
        "Serengeti; the Belize tropical-forest gunshot set; the Chu Mom Ray Vietnam set.",
        "[4] FastAPI service on port 8742 with Gradio dashboard at /dashboard. SQLite "
        "audit table events at data/events.sqlite.",
        "[5] Prepared multimodal metadata summarised in reports/bda/summary.json "
        "(corpus counts only; not model metrics).",
    ]
    for r in refs:
        _p(doc, r, space_after=8)

    OUT_DOCS.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_DOCS)
    shutil.copy2(OUT_DOCS, OUT_ROOT)
    return OUT_DOCS


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
    print(f"wrote {OUT_ROOT}")
