"""YAML loading, path resolution, and seeded RNGs."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def repo_root() -> Path:
    """Return the repository root.

    Prefers ``GUARDIANEYE_ROOT``, then the current directory when it looks like
    the repo, then walks parents of this file.
    """
    env = os.environ.get("GUARDIANEYE_ROOT")
    if env:
        return Path(env).resolve()
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").is_file() and (cwd / "configs").is_dir():
        return cwd
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
            return parent
    return cwd


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping. Relative paths are left unresolved."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = repo_root() / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {config_path} must be a mapping")
    return data


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    """Resolve a config path against the repository root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (root or repo_root()) / path


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when it is installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
