"""Render evaluation files for the dashboard. Missing files stay missing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from guardianeye.config import repo_root


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def reports_markdown(root: Path | None = None) -> str:
    """Markdown summary of whatever ``reports/`` currently contains."""
    root = root or repo_root()
    lines = [
        "### Evaluation files",
        "",
        "Figures below are copied from `reports/`. If a run has not been produced, this panel says so.",
        "",
    ]
    fusion = _load(root / "reports" / "fusion" / "proxy.json")
    if fusion is None:
        lines.append("Fusion: no report yet. Run `make eval`.")
    elif not fusion.get("evaluated"):
        lines.append(f"Fusion: not evaluated. {fusion.get('reason', '')}")
        lines.append("")
        lines.append("Protocol: synthetic proxy protocol. This is not field validation.")
    else:
        lines.append("Fusion: synthetic proxy protocol (not field validation).")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(fusion.get("systems", {}), indent=2))
        lines.append("```")
    lines.append("")
    for domain, filename in (
        ("aerial thermal", "aerial_thermal.json"),
        ("camera trap", "camera_trap_rgb.json"),
    ):
        payload = _load(root / "reports" / "image" / filename)
        if payload is None:
            lines.append(f"Image / {domain}: no report yet.")
        elif not payload.get("evaluated"):
            lines.append(f"Image / {domain}: not evaluated. {payload.get('reason', '')}")
        else:
            view = {
                key: payload.get(key)
                for key in (
                    "weights_kind",
                    "map50",
                    "map50_95",
                    "per_class",
                    "small_object_recall",
                    "empty_frame_fpr",
                )
            }
            lines.append(f"Image / {domain}:")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(view, indent=2))
            lines.append("```")
        lines.append("")
    audio = _load(root / "reports" / "audio" / "experiments.json")
    if audio is None:
        lines.append("Audio: no report yet.")
    else:
        brief = {}
        for name, report in audio.get("experiments", {}).items():
            if report.get("evaluated"):
                metrics = report.get("metrics") or {}
                brief[name] = {
                    "weights_kind": report.get("weights_kind"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "fpr": metrics.get("fpr"),
                }
            else:
                brief[name] = {"evaluated": False, "reason": report.get("reason")}
        lines.append("Audio experiments:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(brief, indent=2))
        lines.append("```")
    return "\n".join(lines)
