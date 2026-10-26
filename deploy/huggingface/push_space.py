#!/usr/bin/env python3
"""Create/update a Hugging Face Docker Space from this repo.

Requires HF_TOKEN with write access. Example:

  HF_TOKEN=hf_... HF_SPACE=YourUser/guardianeye \\
    .venv/bin/python deploy/huggingface/push_space.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    space_id = os.environ.get("HF_SPACE")
    if not token or not space_id:
        raise SystemExit("Set HF_TOKEN and HF_SPACE=user/space-name")
    root = Path(__file__).resolve().parents[2]
    api = HfApi(token=token)
    create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True, token=token)
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        shutil.copy2(root / "deploy/huggingface/Dockerfile", dest / "Dockerfile")
        shutil.copy2(root / "deploy/huggingface/Space_README.md", dest / "README.md")
        for name in (
            "requirements-render.txt",
            "pyproject.toml",
            "src",
            "configs",
            "scripts",
            "deploy",
        ):
            src = root / name
            target = dest / name
            if src.is_dir():
                shutil.copytree(src, target, dirs_exist_ok=True)
            else:
                shutil.copy2(src, target)
        # Spaces Dockerfile path is ./Dockerfile; it expects requirements-render at root.
        api.upload_folder(
            folder_path=str(dest),
            repo_id=space_id,
            repo_type="space",
            commit_message="Deploy GuardianEye Docker Space",
        )
    print(f"https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
