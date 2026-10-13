"""Process-wide logging setup."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure a single stderr handler if logging is not already set up."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
