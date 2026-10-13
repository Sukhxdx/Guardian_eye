"""The dashboard copy comes from report files, or says they are missing."""

from __future__ import annotations

import json
from pathlib import Path

from guardianeye.ui.reports_view import reports_markdown


def test_missing_reports_are_named(tmp_path: Path) -> None:
    text = reports_markdown(tmp_path)
    assert "Fusion: no report yet" in text
    assert "Image / aerial thermal: no report yet" in text
    assert "Audio: no report yet" in text


def test_numbers_are_read_from_the_report_file(tmp_path: Path) -> None:
    fusion = tmp_path / "reports" / "fusion"
    fusion.mkdir(parents=True)
    marker = {"evaluated": True, "systems": {"high": {"fused": {"precision": 0.123456}}}}
    (fusion / "proxy.json").write_text(json.dumps(marker), encoding="utf-8")
    text = reports_markdown(tmp_path)
    assert "0.123456" in text
    assert "synthetic proxy protocol" in text
