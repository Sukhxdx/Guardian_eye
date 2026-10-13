"""Download datasets whose URLs you have filled in.

Every URL in ``configs/data.yaml`` ships as TODO. This script will not guess
a URL. When you replace TODO with a real link, it downloads the file and
checks sha256 when that field is set.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Any

from guardianeye.config import load_config, repo_root, resolve_path
from guardianeye.logging_utils import setup_logging


def sha256_file(path: Path) -> str:
    """Hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_placeholder(url: str) -> bool:
    text = url.strip()
    return (not text) or text.upper().startswith("TODO")


def download_one(name: str, spec: dict[str, Any], root: Path) -> str:
    """Download one dataset. Returns ``skipped``, ``downloaded``, or raises."""
    url = str(spec.get("url", "TODO"))
    dest = resolve_path(spec["dest"], root)
    if _is_placeholder(url):
        print(
            f"[skip] {name}: URL is still TODO.\n"
            f"       Edit configs/data.yaml -> downloads.{name}.url\n"
            f"       Destination will be {dest}\n"
            f"       Notes: {spec.get('notes', '')}"
        )
        dest.mkdir(parents=True, exist_ok=True)
        return "skipped"
    dest.mkdir(parents=True, exist_ok=True)
    filename = url.rstrip("/").split("/")[-1] or f"{name}.bin"
    target = dest / filename
    print(f"[download] {name} -> {target}")
    urllib.request.urlretrieve(url, target)  # noqa: S310 — URL comes from the user's config
    expected = str(spec.get("sha256", "")).strip()
    if expected:
        actual = sha256_file(target)
        if actual != expected.lower():
            raise SystemExit(f"Checksum mismatch for {name}: expected {expected}, got {actual}")
        print(f"[ok] {name} checksum matched")
    else:
        print(f"[ok] {name} downloaded (no sha256 configured)")
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download configured datasets")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    root = repo_root()
    failed = False
    for name, spec in config.get("downloads", {}).items():
        try:
            download_one(str(name), spec, root)
        except Exception as exc:  # noqa: BLE001 — report every dataset, then exit
            failed = True
            print(f"[error] {name}: {exc}", file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
