"""Artifact helpers for Agent_4 runs."""

from __future__ import annotations

import json
import os
import shutil
from typing import Any


RUNS_ROOT = os.path.join(os.path.dirname(__file__), "runs")


def create_session_dir(session_name: str) -> str:
    path = os.path.join(RUNS_ROOT, session_name)
    os.makedirs(path, exist_ok=True)
    return path


def create_run_dir(session_name: str, run_index: int) -> str:
    session_dir = create_session_dir(session_name)
    run_dir = os.path.join(session_dir, f"run_{run_index:03d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def copy_if_exists(src: str, dest: str) -> bool:
    if not src or not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(src, dest)
    return True
