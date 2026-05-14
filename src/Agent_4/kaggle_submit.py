"""Optional Kaggle submission helpers for Agent_4."""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import time
from typing import Any


DEFAULT_COMPETITION = "nlp-getting-started"
DEFAULT_POLL_SECONDS = 15
DEFAULT_TIMEOUT_SECONDS = 15 * 60


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def auto_submit_enabled() -> bool:
    return _env_flag("AGENT4_AUTO_SUBMIT_KAGGLE", default=False)


def competition_name() -> str:
    return os.environ.get("AGENT4_KAGGLE_COMPETITION", DEFAULT_COMPETITION).strip() or DEFAULT_COMPETITION


def poll_seconds() -> int:
    raw = os.environ.get("AGENT4_KAGGLE_POLL_SECONDS", str(DEFAULT_POLL_SECONDS))
    try:
        return max(5, int(raw))
    except ValueError:
        return DEFAULT_POLL_SECONDS


def timeout_seconds() -> int:
    raw = os.environ.get("AGENT4_KAGGLE_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
    try:
        return max(30, int(raw))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def submission_message(default_message: str) -> str:
    message = os.environ.get("AGENT4_KAGGLE_MESSAGE", "").strip()
    return message or default_message


def _discover_kaggle_cli() -> str | None:
    configured = os.environ.get("KAGGLE_CLI_PATH", "").strip()
    candidates = [
        configured,
        os.path.join(os.getcwd(), ".venv", "bin", "kaggle"),
        shutil.which("kaggle") or "",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _run_kaggle(cli_path: str, args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [cli_path, *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _parse_submissions_csv(stdout_text: str) -> list[dict[str, str]]:
    text = stdout_text.strip()
    if not text:
        return []
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, str]] = []
    for row in reader:
        cleaned = {(key or "").strip(): (value or "").strip() for key, value in row.items()}
        if any(cleaned.values()):
            rows.append(cleaned)
    return rows


def _find_submission(rows: list[dict[str, str]], message: str, file_name: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("description") == message and row.get("fileName") == file_name:
            return row
    for row in rows:
        if row.get("description") == message:
            return row
    for row in rows:
        if row.get("fileName") == file_name:
            return row
    return rows[0] if rows else None


def _is_scored(row: dict[str, str]) -> bool:
    return bool(row.get("publicScore") or row.get("privateScore"))


def _is_terminal(row: dict[str, str]) -> bool:
    status = (row.get("status") or "").strip().lower()
    if status in {"complete", "completed", "error", "failed", "invalid", "cancelled"}:
        return True
    return _is_scored(row)


def submit_and_wait(submission_path: str, default_message: str) -> dict[str, Any]:
    cli_path = _discover_kaggle_cli()
    if not cli_path:
        return {
            "enabled": True,
            "submitted": False,
            "error": "Could not find the Kaggle CLI. Set KAGGLE_CLI_PATH or install it in .venv/bin/kaggle.",
        }
    if not os.path.exists(submission_path):
        return {
            "enabled": True,
            "submitted": False,
            "error": f"Submission file not found: {submission_path}",
        }

    competition = competition_name()
    message = submission_message(default_message)
    file_name = os.path.basename(submission_path)

    try:
        submit_proc = _run_kaggle(
            cli_path,
            ["competitions", "submit", competition, "-f", submission_path, "-m", message],
            timeout=5 * 60,
        )
    except subprocess.TimeoutExpired:
        return {
            "enabled": True,
            "submitted": False,
            "competition": competition,
            "message": message,
            "error": "Timed out while uploading the Kaggle submission.",
        }

    result: dict[str, Any] = {
        "enabled": True,
        "competition": competition,
        "message": message,
        "submission_path": submission_path,
        "submit_stdout": submit_proc.stdout.strip(),
        "submit_stderr": submit_proc.stderr.strip(),
        "submit_returncode": submit_proc.returncode,
    }
    if submit_proc.returncode != 0:
        result["submitted"] = False
        result["error"] = "Kaggle submission command failed."
        return result

    result["submitted"] = True
    deadline = time.time() + timeout_seconds()
    latest_row: dict[str, str] | None = None

    while time.time() < deadline:
        try:
            poll_proc = _run_kaggle(
                cli_path,
                ["competitions", "submissions", competition, "-v", "-q"],
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            time.sleep(poll_seconds())
            continue

        result["poll_stdout"] = poll_proc.stdout.strip()
        result["poll_stderr"] = poll_proc.stderr.strip()
        result["poll_returncode"] = poll_proc.returncode
        if poll_proc.returncode == 0:
            rows = _parse_submissions_csv(poll_proc.stdout)
            latest_row = _find_submission(rows, message, file_name)
            if latest_row:
                result["latest_submission"] = latest_row
                if _is_terminal(latest_row):
                    result["status"] = latest_row.get("status", "")
                    result["public_score"] = latest_row.get("publicScore", "")
                    result["private_score"] = latest_row.get("privateScore", "")
                    result["file_name"] = latest_row.get("fileName", "")
                    result["submitted_at"] = latest_row.get("date", "")
                    result["scored"] = _is_scored(latest_row)
                    return result
        time.sleep(poll_seconds())

    result["status"] = (latest_row or {}).get("status", "pending")
    result["public_score"] = (latest_row or {}).get("publicScore", "")
    result["private_score"] = (latest_row or {}).get("privateScore", "")
    result["file_name"] = (latest_row or {}).get("fileName", file_name)
    result["submitted_at"] = (latest_row or {}).get("date", "")
    result["scored"] = _is_scored(latest_row or {})
    result["timed_out_waiting_for_score"] = True
    return result
