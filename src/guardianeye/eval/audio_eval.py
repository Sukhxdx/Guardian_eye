"""Cross-site and in-domain audio evaluation, reported side by side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from guardianeye.audio.infer import AudioInferencer
from guardianeye.config import load_config, repo_root, resolve_path
from guardianeye.eval.metrics import binary_report
from guardianeye.image.dataset import load_jsonl
from guardianeye.logging_utils import setup_logging


def _score_rows(
    inferencer: AudioInferencer, rows: list[dict[str, Any]]
) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    scores: list[float] = []
    for row in rows:
        if not row.get("waveform"):
            continue
        evidence = inferencer.predict_file(Path(str(row["waveform"])))
        if not evidence.windows:
            continue
        labels.append(1 if row["label"] == "gunshot" else 0)
        scores.append(max(window.gunshot_probability for window in evidence.windows))
    return labels, scores


def evaluate_all(audio_config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Score each experiment's test split and write ``reports/audio/experiments.json``."""
    root = root or repo_root()
    data_cfg = load_config(audio_config.get("data_config", "configs/data.yaml"))
    processed = resolve_path(data_cfg["processed_root"], root)
    by_id = {str(row["id"]): row for row in load_jsonl(processed / "audio" / "manifest.jsonl")}
    output_dir = resolve_path(audio_config.get("output_dir", "models/audio"), root)
    report_dir = root / "reports" / "audio"
    report_dir.mkdir(parents=True, exist_ok=True)
    experiments: dict[str, Any] = {}
    threshold = float(audio_config.get("decision_threshold", 0.5))
    for experiment in audio_config.get("experiments", []):
        name = str(experiment["name"])
        weights = output_dir / f"{name}.pt"
        split_path = processed / "audio" / "splits" / f"{name}.ids.json"
        if not weights.is_file() or not split_path.is_file():
            experiments[name] = {
                "evaluated": False,
                "reason": f"Missing checkpoint or split for {name}",
                "train_site": experiment.get("train_site"),
                "test_site": experiment.get("test_site"),
            }
            continue
        ids = json.loads(split_path.read_text(encoding="utf-8"))
        test_rows = [
            by_id[item]
            for item in ids.get("test", [])
            if item in by_id and by_id[item].get("waveform")
        ]
        inferencer = AudioInferencer(
            weights,
            sample_rate=int(audio_config["sample_rate"]),
            window_seconds=float(audio_config["window_seconds"]),
            hop_seconds=float(audio_config["hop_seconds"]),
            n_mels=int(audio_config["n_mels"]),
            n_fft=int(audio_config["n_fft"]),
            hop_length=int(audio_config["hop_length"]),
            fmin=float(audio_config["fmin"]),
            fmax=float(audio_config["fmax"]),
            calibration_path=output_dir / f"{name}_calibration.json",
        )
        labels, scores = _score_rows(inferencer, test_rows)
        metrics = binary_report(labels, scores, threshold) if labels else None
        experiments[name] = {
            "evaluated": metrics is not None,
            "train_site": experiment.get("train_site"),
            "test_site": experiment.get("test_site"),
            "in_domain": bool(experiment.get("in_domain", False)),
            "weights_kind": inferencer.kind,
            "calibration_fitted": inferencer.calibration_fitted,
            "test_clips": len(labels),
            "metrics": metrics,
            "reason": None if metrics is not None else "No scorable test clips",
        }
    payload = {
        "protocol": "site-held-out and in-domain controls, reported separately",
        "experiments": experiments,
        "note": "Do not collapse Belize and Vietnam into one accuracy number.",
    }
    (report_dir / "experiments.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _plot_pr(experiments, report_dir / "pr_curves.png")
    return payload


def _plot_pr(experiments: dict[str, Any], path: Path) -> None:
    series = []
    for name, report in experiments.items():
        metrics = report.get("metrics") or {}
        curve = metrics.get("pr_curve")
        if curve:
            series.append((name, curve))
    if not series:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, curve in series:
        recall = [point["recall"] if point["recall"] is not None else 0.0 for point in curve]
        precision = [
            point["precision"] if point["precision"] is not None else 0.0 for point in curve
        ]
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Audio PR curves by experiment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate audio experiments")
    parser.add_argument("--config", default="configs/audio.yaml")
    args = parser.parse_args()
    setup_logging()
    evaluate_all(load_config(args.config))


if __name__ == "__main__":
    main()
