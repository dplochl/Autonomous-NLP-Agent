"""Sandboxed execution for Agent_3 generated experiments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time


DRY_RUN_TIMEOUT = 60
FULL_RUN_TIMEOUT = 1000


def _python_executable() -> str:
    repo_python = os.path.join(os.getcwd(), ".venv", "bin", "python")
    if os.path.exists(repo_python) and os.access(repo_python, os.X_OK):
        return repo_python
    return sys.executable


def _write_temp_script(code: str, dry_run: bool) -> str:
    if dry_run:
        code = "import os as _os\n_os.environ['AGENT_DRY_RUN'] = '1'\n" + code
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


def _run(script_path: str, timeout: int, monitor: bool = False) -> dict:
    start = time.time()
    python_exec = _python_executable()
    proc = subprocess.Popen(
        [python_exec, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
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


def run_experiment(code: str, name: str) -> dict:
    print(f"  [Sandbox] Dry run for '{name}'...")
    dry_path = _write_temp_script(code, dry_run=True)
    dry_result = _run(dry_path, DRY_RUN_TIMEOUT, monitor=False)
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

    print("  [Sandbox] Dry run passed. Starting full run...")
    full_path = _write_temp_script(code, dry_run=False)
    full_result = _run(full_path, FULL_RUN_TIMEOUT, monitor=True)
    os.unlink(full_path)
    return {
        "success": full_result["success"],
        "timed_out": full_result["timed_out"],
        "dry_run_failed": False,
        "metrics": _parse_metrics(full_result["stdout"]),
        "stdout": full_result["stdout"],
        "stderr": full_result["stderr"],
    }


def tail(text: str, n: int = 30) -> str:
    return "\n".join((text or "").strip().splitlines()[-n:])
