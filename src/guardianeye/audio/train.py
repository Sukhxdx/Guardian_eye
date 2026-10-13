"""Train the audio branch, including both cross-site directions.

Validation rows always come from the training site. The held-out site is used
only as test, by ``guardianeye.eval.audio_eval``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from guardianeye.audio.dataset import MelWindowDataset, class_weights
from guardianeye.audio.model import build_model
from guardianeye.config import load_config, repo_root, resolve_path, set_seed
from guardianeye.fusion.calibrate import fit_temperature, save_calibration
from guardianeye.image.dataset import load_jsonl
from guardianeye.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _records_by_id(processed: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in load_jsonl(processed / "audio" / "manifest.jsonl")}


def _split_ids(processed: Path, experiment: str) -> dict[str, list[str]]:
    path = processed / "audio" / "splits" / f"{experiment}.ids.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split {path}. Run scripts/prepare_audio_data.py first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): [str(item) for item in values] for key, values in payload.items()}


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        if training and optimizer is not None:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        return 0.0
    return float(sum(losses) / len(losses))


def _collect_logits(
    model: nn.Module,
    records: list[dict[str, Any]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = MelWindowDataset(records, False, 0, 0, 0.0, 0)
    if len(dataset) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model.eval()
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(device)).cpu().numpy()
            logits_all.append(logits)
            labels_all.append(labels.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def train_experiment(
    config: dict[str, Any],
    experiment: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
    split_ids: dict[str, list[str]],
    output_dir: Path,
    epochs: int,
    batch_size: int,
) -> Path:
    """Train one experiment and write a checkpoint plus a calibration artifact."""
    device_name = str(config.get("device", "cpu"))
    device = torch.device(
        device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu"
    )
    train_rows = [
        records_by_id[item] for item in split_ids.get("train", []) if item in records_by_id
    ]
    val_rows = [records_by_id[item] for item in split_ids.get("val", []) if item in records_by_id]
    if not train_rows:
        raise RuntimeError(f"Experiment {experiment['name']} has no training rows")
    augment_cfg = config.get("augment", {})
    train_ds = MelWindowDataset(
        train_rows,
        augment=True,
        freq_mask=int(augment_cfg.get("freq_mask", 8)),
        time_mask=int(augment_cfg.get("time_mask", 16)),
        noise_std=float(augment_cfg.get("noise_std", 0.01)),
        seed=int(config.get("seed", 27)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = build_model(
        str(config.get("backend", "small_cnn")),
        int(config.get("n_mels", 64)),
        str(config.get("panns_checkpoint", "")),
    ).to(device)
    criterion: nn.Module
    if config.get("balance", "loss_weight") == "loss_weight":
        criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = None
        if val_rows:
            val_loader = DataLoader(
                MelWindowDataset(val_rows, False, 0, 0, 0.0, 0),
                batch_size=batch_size,
                shuffle=False,
            )
            val_loss = _run_epoch(model, val_loader, criterion, None, device)
        history.append(
            {"epoch": float(epoch + 1), "train_loss": train_loss, "val_loss": val_loss or 0.0}
        )
        logger.info(
            "%s epoch %s train_loss=%.4f val_loss=%s",
            experiment["name"],
            epoch + 1,
            train_loss,
            f"{val_loss:.4f}" if val_loss is not None else "n/a",
        )
    logits, labels = _collect_logits(model, val_rows, device)
    calibration = fit_temperature(logits, labels)
    name = str(experiment["name"])
    checkpoint = {
        "state_dict": model.state_dict(),
        "backend": config.get("backend", "small_cnn"),
        "n_mels": int(config.get("n_mels", 64)),
        "experiment": name,
        "train_site": experiment.get("train_site"),
        "test_site": experiment.get("test_site"),
        "in_domain": bool(experiment.get("in_domain", False)),
        "epochs": epochs,
        "history": history,
        "calibration": calibration,
        "kind": "smoke" if epochs <= int(config.get("smoke", {}).get("epochs", 1)) else "trained",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{name}.pt"
    torch.save(checkpoint, dest)
    save_calibration(output_dir / f"{name}_calibration.json", calibration)
    return dest


def train_all(config: dict[str, Any], smoke: bool, root: Path | None = None) -> list[Path]:
    """Train every configured experiment."""
    root = root or repo_root()
    set_seed(int(config.get("seed", 27)))
    data_cfg = load_config(config.get("data_config", "configs/data.yaml"))
    processed = resolve_path(data_cfg["processed_root"], root)
    records_by_id = _records_by_id(processed)
    output_dir = resolve_path(config.get("output_dir", "models/audio"), root)
    epochs = int(config["smoke"]["epochs"] if smoke else config["epochs"])
    batch_size = int(config["smoke"]["batch_size"] if smoke else config["batch_size"])
    allowed = set(config["smoke"].get("experiments", [])) if smoke else None
    written: list[Path] = []
    for experiment in config.get("experiments", []):
        if allowed is not None and experiment["name"] not in allowed:
            continue
        split_ids = _split_ids(processed, str(experiment["name"]))
        path = train_experiment(
            config,
            experiment,
            records_by_id,
            split_ids,
            output_dir,
            epochs,
            batch_size,
        )
        written.append(path)
        logger.info("Wrote %s", path)
    serve_name = str(config.get("serve_experiment", "belize_to_vietnam"))
    source = output_dir / f"{serve_name}.pt"
    if source.is_file():
        target = output_dir / "best.pt"
        target.write_bytes(source.read_bytes())
        logger.info("Copied %s to %s for serving", source.name, target.name)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GuardianEye audio branch")
    parser.add_argument("--config", default="configs/audio.yaml")
    parser.add_argument("--smoke", action="store_true", help="One short pass on the prepared set")
    args = parser.parse_args()
    setup_logging()
    train_all(load_config(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
