"""Checksum helper, and TODO urls are not guessed."""

from __future__ import annotations

from pathlib import Path

from scripts.download_data import _is_placeholder, sha256_file


def test_todo_urls_are_placeholders() -> None:
    assert _is_placeholder("TODO")
    assert _is_placeholder("TODO: fill me in")
    assert _is_placeholder("")
    assert not _is_placeholder("https://example.invalid/data.zip")


def test_sha256_changes_with_bytes(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"guardianeye")
    first = sha256_file(path)
    path.write_bytes(b"guardianeye!")
    assert sha256_file(path) != first
    assert len(first) == 64
