"""Build the GuardianEye Multimodal AI report as an IEEE conference paper PDF.

Compiles mai/MAI_IEEE_Conference_Paper.tex with pdflatex (IEEEtran).
Does not invent detection metrics.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAI = ROOT / "mai"
TEX = MAI / "MAI_IEEE_Conference_Paper.tex"
PDF_BUILT = MAI / "MAI_IEEE_Conference_Paper.pdf"

OUTS = [
    ROOT / "docs" / "MAI_Project_Report_2026_2027.pdf",
    ROOT / "MAI_Project_Report_2026_2027.pdf",
    ROOT / "docs" / "MAI_IEEE_Conference_Paper.pdf",
    ROOT / "MAI_IEEE_Conference_Paper.pdf",
    ROOT / "docs" / "MAI_IEEE_Conference_Paper.tex",
]


def build() -> Path:
    if not TEX.is_file():
        raise SystemExit(f"missing {TEX}")
    for _ in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", TEX.name],
            cwd=MAI,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 and not PDF_BUILT.is_file():
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            raise SystemExit(f"pdflatex failed with code {proc.returncode}")
    if not PDF_BUILT.is_file():
        raise SystemExit("pdflatex produced no PDF")

    (ROOT / "docs").mkdir(parents=True, exist_ok=True)
    for dest in OUTS:
        if dest.suffix == ".tex":
            shutil.copy2(TEX, dest)
        else:
            shutil.copy2(PDF_BUILT, dest)
            print(f"wrote {dest}")
    # Remove obsolete college-format DOCX if present (IEEE PDF is the report).
    for obsolete in (
        ROOT / "docs" / "MAI_Project_Report_2026_2027.docx",
        ROOT / "MAI_Project_Report_2026_2027.docx",
    ):
        if obsolete.is_file():
            obsolete.unlink()
            print(f"removed obsolete {obsolete.name}")
    return PDF_BUILT


if __name__ == "__main__":
    path = build()
    print(f"primary {path}")
