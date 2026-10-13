"""Build ``reports/INDEX.md`` from files that evaluation scripts actually wrote."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from guardianeye.config import repo_root


def _read(path: Path) -> Any | None:
    if not path.is_file():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def build_index(root: Path | None = None) -> str:
    """List report files and embed their JSON. Missing runs are named as missing."""
    root = root or repo_root()
    reports = root / "reports"
    lines = [
        "# GuardianEye reports",
        "",
        "This page is generated from files on disk. If a section says a run is missing, that run has not been produced yet.",
        "",
        "Fusion numbers, when present, come from the **synthetic proxy protocol**, not from field incidents.",
        "",
    ]
    sections = [
        ("EDA", reports / "eda" / "imbalance.json"),
        ("Image / aerial thermal", reports / "image" / "aerial_thermal.json"),
        ("Image / camera trap", reports / "image" / "camera_trap_rgb.json"),
        ("Audio experiments", reports / "audio" / "experiments.json"),
        ("Fusion proxy protocol", reports / "fusion" / "proxy.json"),
        ("Latency", reports / "latency.json"),
    ]
    for title, path in sections:
        lines.append(f"## {title}")
        lines.append("")
        payload = _read(path)
        if payload is None:
            lines.append(f"No file at `{path.relative_to(root)}`.")
        else:
            lines.append(f"Source: `{path.relative_to(root)}`")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(payload, indent=2))
            lines.append("```")
        lines.append("")
    text = "\n".join(lines)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "INDEX.md").write_text(text, encoding="utf-8")
    return text
