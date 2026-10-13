"""Reported performance numbers have to come from ``reports/``, not the source tree."""

from __future__ import annotations

import re
from pathlib import Path

from guardianeye.config import repo_root

PATTERN = re.compile(
    r"(?i)\b(accuracy|mAP50(?:-95)?|mAP@50|f1(?:[- ]score)?|precision|recall|false[- ]positive rate)\b"
    r"[^.\n]{0,48}?\b(?:\d{2,3}\s*%|0?\.\d{2,})"
)

SCAN_ROOTS = ("src", "scripts", "configs", "deploy", "notebooks")
SCAN_FILES = ("README.md", "data/README.md", "reports/RESULTS_TEMPLATE.md")


def test_source_and_docs_do_not_hardcode_performance() -> None:
    root = repo_root()
    offenders: list[str] = []
    paths: list[Path] = []
    for name in SCAN_ROOTS:
        base = root / name
        if base.exists():
            paths.extend(
                path
                for path in base.rglob("*")
                if path.suffix in {".py", ".md", ".yaml", ".yml", ".ipynb"}
            )
    for name in SCAN_FILES:
        candidate = root / name
        if candidate.is_file():
            paths.append(candidate)
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in PATTERN.finditer(text):
            offenders.append(f"{path.relative_to(root)}: {match.group(0)}")
    assert offenders == []
