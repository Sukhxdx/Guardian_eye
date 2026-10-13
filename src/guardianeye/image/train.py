"""Train one YOLOv8 model per domain.

Rare box classes are oversampled in a copy of the train split. Empty images
in that copy are kept and are not duplicated to chase the majority count.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from guardianeye.config import load_config, repo_root, resolve_path, set_seed
from guardianeye.image.dataset import oversample_rare_classes, write_ultralytics_yaml
from guardianeye.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _stage_domain(domain_dir: Path, work_dir: Path, oversample: bool, max_copies: int) -> Path:
    """Copy a domain tree and optionally oversample only the train split."""
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(domain_dir, work_dir)
    if oversample:
        copies = oversample_rare_classes(
            work_dir / "images" / "train",
            work_dir / "labels" / "train",
            max_copies_per_image=max_copies,
        )
        logger.info("Oversample copies for %s: %s", domain_dir.name, copies)
    write_ultralytics_yaml(work_dir / "data.yaml", work_dir)
    return work_dir / "data.yaml"


def train_domain(
    domain: str,
    data_yaml: Path,
    config: dict[str, Any],
    smoke: bool,
    output_dir: Path,
) -> Path:
    """Fine-tune YOLOv8 on one domain and copy ``best.pt`` to ``output_dir``."""
    from ultralytics import YOLO

    smoke_cfg = config.get("smoke", {})
    epochs = int(smoke_cfg["epochs"] if smoke else config["epochs"])
    imgsz = int(smoke_cfg["imgsz"] if smoke else config["imgsz"])
    batch = int(smoke_cfg["batch"] if smoke else config["batch"])
    workers = int(smoke_cfg["workers"] if smoke else config.get("workers", 2))
    aug = config.get("augmentation", {})
    weights = str(config.get("base_weights", "yolov8n.pt"))
    model = YOLO(weights)
    project = resolve_path(config.get("project_dir", "models/image/runs"))
    run_name = f"{domain}_smoke" if smoke else domain
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=str(config.get("device", "cpu")),
        workers=workers,
        project=str(project),
        name=run_name,
        exist_ok=True,
        patience=int(config.get("patience", 15)) if not smoke else epochs,
        hsv_h=float(aug.get("hsv_h", 0.015)),
        hsv_s=float(aug.get("hsv_s", 0.4)),
        hsv_v=float(aug.get("hsv_v", 0.4)),
        degrees=float(aug.get("degrees", 0.0)),
        translate=float(aug.get("translate", 0.1)),
        scale=float(aug.get("scale", 0.5)),
        fliplr=float(aug.get("fliplr", 0.5)),
        mosaic=float(aug.get("mosaic", 1.0)),
        erasing=float(aug.get("erasing", 0.0)),
        close_mosaic=int(aug.get("close_mosaic", 10)) if not smoke else 0,
        pretrained=True,
        verbose=False,
        seed=int(config.get("seed", 27)),
    )
    best = project / run_name / "weights" / "best.pt"
    if not best.is_file():
        best = project / run_name / "weights" / "last.pt"
    if not best.is_file():
        raise FileNotFoundError(f"YOLO did not write weights for {domain}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{domain}.pt"
    shutil.copy2(best, dest)
    meta = {
        "kind": "smoke" if smoke else "trained",
        "domain": domain,
        "epochs": epochs,
        "imgsz": imgsz,
        "base_weights": weights,
        "dataset": str(data_yaml),
    }
    dest.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved %s", dest)
    return dest


def train_all(config: dict[str, Any], smoke: bool, root: Path | None = None) -> list[Path]:
    """Train every domain that has a prepared YOLO tree."""
    root = root or repo_root()
    set_seed(int(config.get("seed", 27)))
    data_cfg = load_config(config.get("data_config", "configs/data.yaml"))
    processed = resolve_path(data_cfg["processed_root"], root) / "images"
    output_dir = root / "models" / "image"
    work_root = root / "models" / "image" / "staged"
    written: list[Path] = []
    domains = list(data_cfg.get("image", {}).get("domains", []))
    if not config.get("train_per_domain", True):
        raise ValueError("train_per_domain must stay true so domains are not pooled")
    for domain in domains:
        domain_dir = processed / domain
        if not (domain_dir / "data.yaml").is_file():
            logger.warning("Skipping %s; no prepared data at %s", domain, domain_dir)
            continue
        yaml_path = _stage_domain(
            domain_dir,
            work_root / domain,
            oversample=bool(config.get("oversample_rare", True)),
            max_copies=int(config.get("max_copies_per_image", 20)),
        )
        written.append(train_domain(domain, yaml_path, config, smoke, output_dir))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GuardianEye image branch")
    parser.add_argument("--config", default="configs/image.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    setup_logging()
    train_all(load_config(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
