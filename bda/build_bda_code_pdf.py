"""Render bda/guardianeye_bda_analysis.py as a submission PDF with cover page."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pygments import highlight
from pygments.formatters import LatexFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "bda" / "guardianeye_bda_analysis.py"
BUILD = ROOT / "bda" / "_code_pdf_build"
OUT_DOCS = ROOT / "docs" / "BDA_Project_Code_2026_2027.pdf"
OUT_ROOT = ROOT / "BDA_Project_Code_2026_2027.pdf"

MEMBERS = [
    ("Akhil Chivakula", "23070126009"),
    ("Dipti Kothari", "23070126040"),
    ("Khurinji Malar", "23070126059"),
    ("Sukhada Dhingra", "24070126505"),
]


def build() -> Path:
    code = SRC.read_text(encoding="utf-8")
    formatter = LatexFormatter(
        linenos=True,
        nobackground=True,
        verboptions="fontsize=\\footnotesize",
    )
    highlighted = highlight(code, PythonLexer(), formatter)
    style_defs = formatter.get_style_defs()

    member_lines = " \\\\\n".join(f"{n} --- PRN {p}" for n, p in MEMBERS)

    tex = rf"""
\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=0.9in]{{geometry}}
\usepackage{{fancyhdr}}
\usepackage{{titlesec}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{fancyvrb}}
\usepackage{{fvextra}}
\hypersetup{{colorlinks=true,urlcolor=blue}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\footnotesize GuardianEye --- BDA Project Code}}
\fancyhead[R]{{\footnotesize AY 2026--2027}}
\fancyfoot[C]{{\thepage}}
\renewcommand{{\headrulewidth}}{{0.4pt}}
{style_defs}
\title{{\textbf{{GuardianEye}}\\[6pt]
\large Big Data Analytics (BDA) --- Project Source Code\\[4pt]
\normalsize Course code 0701260704}}
\author{{
Akhil Chivakula \and Dipti Kothari \and Khurinji Malar \and Sukhada Dhingra\\[4pt]
\textit{{Department of Artificial Intelligence and Machine Learning}}\\
\textit{{Symbiosis Institute of Technology}}\\[6pt]
Guide: Dr.~Archana Choudary\\
Head of the Department: Dr.~Sumit Kumar
}}
\date{{Academic Year 2026--2027}}
\begin{{document}}
\maketitle
\begin{{abstract}}
This document contains the complete source of the Big Data Analytics coursework
artefact \texttt{{bda/guardianeye\_bda\_analysis.py}}. The program opens a local
PySpark session and aggregates prepared multimodal indicator metadata
(image manifests, site-tagged audio, site-stratified splits, synthetic proxy
pairs, and the SQLite audit log). It does not report detector precision, recall,
F1, or mAP. One detection is an indicator, not proof of poaching.
\end{{abstract}}

\section*{{Submission details}}
\begin{{itemize}}
\item Subject: Big Data Analytics (BDA)
\item File path: \texttt{{bda/guardianeye\_bda\_analysis.py}}
\item Run command: \texttt{{.venv/bin/python bda/guardianeye\_bda\_analysis.py}}
\item Engine: PySpark 4.0.1, local[2], seed 27
\item Group members:\\[2pt]
{member_lines}
\end{{itemize}}

\section*{{Source listing}}
\noindent\texttt{{bda/guardianeye\_bda\_analysis.py}}

{highlighted}

\end{{document}}
"""

    BUILD.mkdir(parents=True, exist_ok=True)
    tex_path = BUILD / "BDA_Project_Code.tex"
    tex_path.write_text(tex, encoding="utf-8")

    for _ in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=BUILD,
            capture_output=True,
            text=True,
        )
    pdf_built = BUILD / "BDA_Project_Code.pdf"
    if not pdf_built.is_file():
        raise SystemExit(proc.stdout[-2000:] + "\n" + proc.stderr[-1000:])

    OUT_DOCS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_built, OUT_DOCS)
    shutil.copy2(pdf_built, OUT_ROOT)
    print(f"wrote {OUT_DOCS}")
    print(f"wrote {OUT_ROOT}")
    return OUT_DOCS


if __name__ == "__main__":
    build()
