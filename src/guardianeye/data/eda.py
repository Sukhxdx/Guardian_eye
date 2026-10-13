"""Class-distribution plots and imbalance ratios from prepared manifests."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from guardianeye.config import repo_root, resolve_path
from guardianeye.constants import IMAGE_CLASSES
from guardianeye.image.dataset import load_jsonl


def _ratio(counts: dict[str, int]) -> float | None:
    positive = [count for count in counts.values() if count > 0]
    if len(positive) < 2:
        return None
    return max(positive) / min(positive)


def _bar(path: Path, title: str, labels: list[str], values: list[int], xlabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(labels, values, color="#2f5d50")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_eda(config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Regenerate ``reports/eda`` from the processed manifests."""
    root = root or repo_root()
    processed = resolve_path(config["processed_root"], root)
    report_dir = root / "reports" / "eda"
    report_dir.mkdir(parents=True, exist_ok=True)
    images = load_jsonl(processed / "images" / "manifest.jsonl")
    audio = load_jsonl(processed / "audio" / "manifest.jsonl")

    per_domain: dict[str, dict[str, int]] = {}
    for row in images:
        domain = str(row["domain"])
        counts = per_domain.setdefault(domain, {name: 0 for name in IMAGE_CLASSES})
        primary = str(row.get("primary_class", "empty"))
        if primary not in counts:
            counts[primary] = 0
        counts[primary] += 1

    domain_totals = Counter(str(row["domain"]) for row in images)
    _bar(
        report_dir / "domain_counts.png",
        "Images by domain",
        list(domain_totals.keys()) or ["none"],
        [domain_totals[key] for key in domain_totals] or [0],
        "domain",
    )
    for domain, counts in per_domain.items():
        _bar(
            report_dir / f"class_counts_{domain}.png",
            f"Image-level classes — {domain}",
            list(counts.keys()),
            list(counts.values()),
            "class",
        )

    audio_counts: dict[str, dict[str, int]] = {}
    for row in audio:
        site = str(row["site"])
        site_counts = audio_counts.setdefault(site, {"gunshot": 0, "background": 0})
        label = str(row["label"])
        if label in site_counts:
            site_counts[label] += 1
    if audio_counts:
        labels = []
        values = []
        for site in sorted(audio_counts):
            for label, count in audio_counts[site].items():
                labels.append(f"{site}\n{label}")
                values.append(count)
        _bar(
            report_dir / "audio_site_counts.png",
            "Audio windows by site and label",
            labels,
            values,
            "site / label",
        )

    payload: dict[str, Any] = {
        "image_class_counts": per_domain,
        "image_imbalance_ratio": {domain: _ratio(counts) for domain, counts in per_domain.items()},
        "domain_counts": dict(domain_totals),
        "audio_counts": audio_counts,
        "audio_imbalance_ratio": {site: _ratio(counts) for site, counts in audio_counts.items()},
        "note": "Counts describe the prepared dataset. They are not model metrics.",
    }
    import json

    (report_dir / "imbalance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
