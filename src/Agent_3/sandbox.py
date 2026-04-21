"""Sandboxed execution for Agent_3 generated experiments."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd


DRY_RUN_TIMEOUT = 60
FULL_RUN_TIMEOUT = 1000
DATA_DIR_ENV = "DISASTER_AGENT_DATA_DIR"
DEFAULT_DATA_DIR = "data"


def _python_executable() -> str:
    repo_python = os.path.join(os.getcwd(), ".venv", "bin", "python")
    if os.path.exists(repo_python) and os.access(repo_python, os.X_OK):
        return repo_python
    return sys.executable


def _write_temp_script(
    code: str,
    dry_run: bool,
    train_fraction: float = 1.0,
    write_submission: bool = False,
    final_submission: bool = False,
) -> str:
    if dry_run:
        code = (
            "import os as _os\n"
            "_os.environ['AGENT_DRY_RUN'] = '1'\n"
            "_os.environ['AGENT_WRITE_SUBMISSION'] = '0'\n"
            "_os.environ['AGENT_FINAL_SUBMISSION'] = '0'\n"
            + code
        )
    else:
        code = (
            "import os as _os\n"
            f"_os.environ['AGENT_TRAIN_FRACTION'] = {str(train_fraction)!r}\n"
            "_os.environ.setdefault('AGENT_SAMPLE_SEED', '42')\n"
            f"_os.environ['AGENT_WRITE_SUBMISSION'] = {('1' if write_submission else '0')!r}\n"
            f"_os.environ['AGENT_FINAL_SUBMISSION'] = {('1' if final_submission else '0')!r}\n"
            + code
        )
    fd, path = tempfile.mkstemp(suffix=".py", prefix="agent3_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    return path


def _parse_metrics(stdout: str) -> dict:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("METRICS:"):
            try:
                return json.loads(line[len("METRICS:"):].strip().replace("'", '"'))
            except json.JSONDecodeError:
                return {}
    return {}


def _data_paths() -> tuple[str, str]:
    data_dir = os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR)
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    if os.path.exists(train_path) and os.path.exists(test_path):
        return train_path, test_path
    if os.path.exists("train.csv") and os.path.exists("test.csv"):
        return "train.csv", "test.csv"
    raise FileNotFoundError("Could not find train.csv and test.csv.")


def _sampled_data_dir(
    train_fraction: float,
    seed: int = 42,
    test_rows: int | None = None,
    train_rows: int | None = None,
) -> str | None:
    train_path, test_path = _data_paths()
    train = pd.read_csv(train_path)
    requested_rows = train_rows if train_rows and train_rows > 0 else None
    needs_train_sample = train_fraction < 1.0 or (requested_rows is not None and requested_rows < len(train))
    if not needs_train_sample and test_rows is None:
        return None

    sampled_dir = tempfile.mkdtemp(prefix="agent3_sampled_data_")
    if requested_rows is not None:
        train = train.sample(n=min(requested_rows, len(train)), random_state=seed).reset_index(drop=True)
    elif train_fraction < 1.0:
        train = train.sample(frac=train_fraction, random_state=seed).reset_index(drop=True)
    train.to_csv(os.path.join(sampled_dir, "train.csv"), index=False)
    if test_rows is None:
        shutil.copyfile(test_path, os.path.join(sampled_dir, "test.csv"))
    else:
        test = pd.read_csv(test_path).head(test_rows)
        test.to_csv(os.path.join(sampled_dir, "test.csv"), index=False)
    return sampled_dir


def _resolved_train_rows(train_rows: int | None) -> int | None:
    if not train_rows or train_rows <= 0:
        return None
    train_path, _ = _data_paths()
    return min(train_rows, len(pd.read_csv(train_path)))


def _run(script_path: str, timeout: int, monitor: bool = False, env: dict[str, str] | None = None) -> dict:
    start = time.time()
    python_exec = _python_executable()
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    proc = subprocess.Popen(
        [python_exec, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=process_env,
    )
    try:
        if not monitor:
            stdout, stderr = proc.communicate(timeout=timeout)
            return {
                "success": proc.returncode == 0,
                "timed_out": False,
                "stdout": stdout,
                "stderr": stderr,
            }

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                proc.kill()
                stdout, stderr = proc.communicate()
                return {
                    "success": False,
                    "timed_out": True,
                    "stdout": stdout or "",
                    "stderr": f"TIMEOUT after {timeout}s\n" + (stderr or ""),
                }
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                return {
                    "success": proc.returncode == 0,
                    "timed_out": False,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            if int(elapsed) and int(elapsed) % 30 == 0:
                print(f"  [Monitor] Still running... {int(elapsed)}s elapsed")
            time.sleep(2)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return {
            "success": False,
            "timed_out": True,
            "stdout": stdout or "",
            "stderr": f"TIMEOUT after {timeout}s\n" + (stderr or ""),
        }


def run_experiment(
    code: str,
    name: str,
    train_fraction: float = 1.0,
    train_rows: int | None = None,
    write_submission: bool = False,
    final_submission: bool = False,
    data_dir: str | None = None,
) -> dict:
    print(f"  [Sandbox] Dry run for '{name}'...")
    dry_path = _write_temp_script(code, dry_run=True, write_submission=False)
    dry_env = {DATA_DIR_ENV: data_dir} if data_dir else None
    dry_result = _run(dry_path, DRY_RUN_TIMEOUT, monitor=False, env=dry_env)
    os.unlink(dry_path)
    if not dry_result["success"]:
        print("  [Sandbox] Dry run FAILED.")
        return {
            "success": False,
            "timed_out": dry_result["timed_out"],
            "dry_run_failed": True,
            "metrics": {},
            "stdout": dry_result["stdout"],
            "stderr": dry_result["stderr"],
        }

    mode = "submission" if write_submission else "metrics-only"
    if data_dir:
        row_note = f", fixed_data_dir={data_dir}"
    else:
        resolved_rows = _resolved_train_rows(train_rows)
        row_note = (
            f", train_rows={resolved_rows} (requested {train_rows})"
            if train_rows and resolved_rows != train_rows
            else (f", train_rows={train_rows}" if train_rows else f", train_fraction={train_fraction:.2f}")
        )
    print(f"  [Sandbox] Dry run passed. Starting {mode} run{row_note}...")
    sampled_dir = None if data_dir else _sampled_data_dir(train_fraction, test_rows=None if write_submission else 1, train_rows=train_rows)
    full_path = _write_temp_script(
        code,
        dry_run=False,
        train_fraction=1.0 if sampled_dir or data_dir else train_fraction,
        write_submission=write_submission,
        final_submission=final_submission,
    )
    full_env = {DATA_DIR_ENV: data_dir or sampled_dir} if data_dir or sampled_dir else None
    try:
        full_result = _run(full_path, FULL_RUN_TIMEOUT, monitor=True, env=full_env)
    finally:
        os.unlink(full_path)
        if sampled_dir:
            shutil.rmtree(sampled_dir, ignore_errors=True)
    metrics = _parse_metrics(full_result["stdout"])
    has_metrics = "f1" in metrics
    stderr = full_result["stderr"]
    if full_result["success"] and not has_metrics:
        stderr = (stderr + "\n" if stderr else "") + "Missing METRICS line or parsable F1 output."
    return {
        "success": full_result["success"] and has_metrics,
        "process_success": full_result["success"],
        "timed_out": full_result["timed_out"],
        "dry_run_failed": False,
        "metrics": metrics,
        "stdout": full_result["stdout"],
        "stderr": stderr,
    }


def tail(text: str, n: int = 30) -> str:
    return "\n".join((text or "").strip().splitlines()[-n:])
