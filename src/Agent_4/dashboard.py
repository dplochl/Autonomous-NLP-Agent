"""Live Flask dashboard for Agent_4.

Three pages:
  /          F1 over time chart for past + current runs, error-rate stats
  /current   The currently running spec, the active prompt, the latest sweep decision
  /errors    Failed trials with outcome classification breakdown

No DB — everything is read from disk on each request (runs/, agent3_log.json).

Run:
  python3 src/Agent_4/dashboard.py
  # then open http://localhost:5050
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from glob import glob
from typing import Any

from flask import Flask, jsonify, render_template_string, abort, send_from_directory

AGENT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(AGENT_ROOT, "docs")
PROJECT_ROOT = os.path.abspath(os.path.join(AGENT_ROOT, "..", ".."))
# Top-level logs/ is the Catolica/Nicc_2 layout; fall back to the legacy
# in-tree location if the dashboard is run from a different checkout.
_TOPLEVEL_LOGS = os.path.join(PROJECT_ROOT, "logs")
_LEGACY_LOGS = os.path.join(AGENT_ROOT, "data", "logs")
LOGS_DIR = _TOPLEVEL_LOGS if os.path.isdir(_TOPLEVEL_LOGS) else _LEGACY_LOGS

# Nicc_2 layout: runs live under <repo>/runs/agent_4/<session>/. The "current"
# directory is the most recently invoked agent.py run. Fall back to the legacy
# src/Agent_4/runs/ path if a fresh in-place run wrote there.
_NICC2_RUNS = os.path.join(PROJECT_ROOT, "runs", "agent_4")
_LEGACY_RUNS = os.path.join(AGENT_ROOT, "runs")
RUNS_DIR = _NICC2_RUNS if os.path.isdir(_NICC2_RUNS) else _LEGACY_RUNS
# overall_best.json + sweep_decisions.jsonl live at the runs/ root for the
# currently-active launch. After a run finishes we archive that root into
# runs/agent_4/<version>/. The dashboard checks the live in-tree location
# first, then the "current" snapshot, then picks the most recently modified
# copy across all version archives so the /current page always has data.
_LIVE_RUNS = os.path.join(AGENT_ROOT, "runs")  # src/Agent_4/runs/ (legacy in-tree)
_CURRENT_RUN = os.path.join(_NICC2_RUNS, "current")


def _newest(*candidates: str) -> str:
    existing = [p for p in candidates if p and os.path.exists(p)]
    if not existing:
        # Last-resort: scan all version archives for the most recent file.
        scan = (
            glob(os.path.join(_NICC2_RUNS, "*", candidates[0].split("/")[-1]))
            if candidates else []
        )
        existing = [p for p in scan if os.path.exists(p)]
    return max(existing, key=os.path.getmtime) if existing else candidates[-1]


SWEEP_DECISIONS = _newest(
    os.path.join(_LIVE_RUNS, "sweep_decisions.jsonl"),
    os.path.join(_CURRENT_RUN, "sweep_decisions.jsonl"),
)
OVERALL_BEST = _newest(
    os.path.join(_LIVE_RUNS, "overall_best.json"),
    os.path.join(_CURRENT_RUN, "overall_best.json"),
)

# The dashboard prefers the repo-local logs (committed snapshots under
# src/Agent_4/data/logs/) and falls back to absolute paths on Niccolò's / Simon's
# laptops if the repo-local copies aren't present.
def _first_existing(*candidates: str) -> str | None:
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


V1_LOG = os.environ.get("V1_LOG_PATH") or _first_existing(
    os.path.join(LOGS_DIR, "v1_experiment_log.json"),
    r"C:\Users\shoxx\Downloads\apa-disaster-tweets-agent\experiment_log.json",
)
V2_LOG_PATHS = [
    p for p in (
        os.environ.get("V2_LOG_PATHS").split(";")
        if os.environ.get("V2_LOG_PATHS")
        else [
            os.path.join(LOGS_DIR, "v2_agent_v2_log.json"),
            os.path.join(LOGS_DIR, "v2_git_clone_log.json"),
            r"C:\Users\shoxx\Downloads\Agent_V2\experiment_log.json",
            r"C:\Users\shoxx\Downloads\Git_Clone\apa-disaster-tweets-agent\experiment_log.json",
            r"C:\Users\shoxx\Downloads\Git_Clone\apa-disaster-tweets-agent-1\experiment_log.json",
        ]
    )
    if p and p.strip()
]
AGENT3_LOG = os.environ.get("V3_LOG_PATH") or _first_existing(
    os.path.join(LOGS_DIR, "agent3_log.json"),          # Catolica/Nicc_2 layout
    os.path.join(LOGS_DIR, "v3_agent3_log.json"),       # legacy in-tree name
    os.path.join(PROJECT_ROOT, "agent3_log.json"),      # repo-root legacy
)
# Current Agent_4 in-launch log (only present after a recent run). Reused
# for the /current page so it can show what the live agent just did.
AGENT4_LOG = os.environ.get("V4_LOG_PATH") or _first_existing(
    os.path.join(LOGS_DIR, "agent4_log.json"),
    os.path.join(PROJECT_ROOT, "agent4_log.json"),
)

app = Flask(__name__)


# ----------------------------- data loaders ----------------------------- #

def load_json(path: str | None) -> Any:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_text(path: str, max_bytes: int = 200_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = fh.read(max_bytes)
        return data
    except FileNotFoundError:
        return ""


import re as _re_dashboard

# Architecture-version cutoffs (inclusive lower bound, YYYYMMDD).
# Anything on/after a cutoff is that version until the next cutoff:
#   < 2026-04-15           -> V1  (04-12 sklearn baseline only)
#   2026-04-15 - 2026-04-17 -> V2 (autonomous single-file, 80-trial burst)
#   2026-04-18 - 2026-05-13 -> V3 (multi-family modular agent;
#                                  04-20 sessions with `_opt` suffix are
#                                  already V3-architecture pre-production
#                                  experiments — V2 ran its last trial 04-17)
#   >= 2026-05-14          -> V4 (sweep planner)
_VERSION_CUTOFFS = [
    ("20260514", "V4"),
    ("20260418", "V3"),
    ("20260415", "V2"),
    ("00000000", "V1"),
]


def _classify_version_by_date(session_name: str, fallback: str) -> str:
    """Return V1/V2/V3/V4 based on the YYYYMMDD embedded in the session name.

    Folder name (e.g. `runs/agent_3/...`) is ignored — only the session
    timestamp matters. Anything that does not match the expected
    `<family>_<YYYYMMDD>_<HHMMSS>` pattern falls back to `fallback`.
    """
    m = _re_dashboard.search(r"(\d{8})_\d{6}", session_name or "")
    if not m:
        return fallback
    date = m.group(1)
    for cutoff, label in _VERSION_CUTOFFS:
        if date >= cutoff:
            return label
    return fallback


def _trials_from_summary_glob(summary_paths: list[str], fallback_version: str, source: str) -> list[dict[str, Any]]:
    """Walk a list of summary.json files and emit one trial dict per entry.

    Version is determined by the date in the session folder name (not by
    which root folder the summary lives in), so a `runs/agent_3/<session>`
    trial dated 2026-04-17 is correctly classified as V2 even though it
    sits under the "agent_3" code-folder.
    """
    out: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        summary = load_json(summary_path) or {}
        session = os.path.basename(os.path.dirname(summary_path))
        version = _classify_version_by_date(session, fallback_version)
        for trial in summary.get("trials", []):
            run_idx = trial.get("run_index")
            run_dir_rel = f"{session}/run_{run_idx:03d}" if isinstance(run_idx, int) and run_idx > 0 else None
            out.append({
                "version": version,
                "source": source,
                "session": session,
                "phase": summary.get("phase", "sweep"),
                "family": summary.get("family", "unknown"),
                "family_key": summary.get("family_key", ""),
                "run_index": run_idx,
                "success": bool(trial.get("success", False)),
                "f1": (trial.get("metrics") or {}).get("f1"),
                "accuracy": (trial.get("metrics") or {}).get("accuracy"),
                "outcome": trial.get("outcome", "unknown"),
                "error_summary": trial.get("error_summary", ""),
                "wall_seconds": trial.get("wall_seconds"),
                "repair_attempts": trial.get("repair_attempts", 0),
                "run_dir": run_dir_rel,
                "timestamp": None,
            })
    return out


def gather_trials() -> list[dict[str, Any]]:
    """Walk all V3 + V4 session summaries plus the V3 / V1 / V2 flat logs."""
    trials: list[dict[str, Any]] = []

    # Agent_4 runs → V4. Nicc_2 layout has two levels of sessions
    # (runs/agent_4/<bucket>/<session>/summary.json), so glob both depths.
    v4_summaries = sorted(set(
        glob(os.path.join(RUNS_DIR, "*", "summary.json")) +
        glob(os.path.join(RUNS_DIR, "*", "*", "summary.json"))
    ))
    trials.extend(_trials_from_summary_glob(v4_summaries, fallback_version="V4", source="agent4"))

    # Agent_3 runs → V3. Same summary.json shape as V4, just stored under
    # runs/agent_3/<session>/. Glob both 1- and 2-level layouts to match
    # whatever folder structure happens to be in place.
    agent3_root = os.path.join(PROJECT_ROOT, "runs", "agent_3")
    v3_summaries = sorted(set(
        glob(os.path.join(agent3_root, "*", "summary.json")) +
        glob(os.path.join(agent3_root, "*", "*", "summary.json"))
    ))
    trials.extend(_trials_from_summary_glob(v3_summaries, fallback_version="V3", source="agent3-runs"))

    # Historical Agent_3 in-launch log → V3 only used if no run folders found.
    # agent3_log.json is just the last launch's records and is fully redundant
    # with summary.json files when those exist (it would double-count).
    if not v3_summaries:
        historical = load_json(AGENT3_LOG)
        if isinstance(historical, list):
            for entry in historical:
                metrics = entry.get("metrics") or {}
                trials.append({
                    "version": "V3",
                    "source": "agent3-history",
                    "session": entry.get("run_name", "history"),
                    "phase": "history",
                    "family": entry.get("family", "unknown"),
                    "family_key": "",
                    "run_index": entry.get("run_index"),
                    "success": bool(entry.get("success", False)),
                    "f1": metrics.get("f1"),
                    "accuracy": metrics.get("accuracy"),
                    "outcome": "success" if entry.get("success") else "training_crash",
                    "error_summary": "",
                    "wall_seconds": None,
                    "repair_attempts": 0,
                    "run_dir": None,
                    "timestamp": entry.get("timestamp"),
                })

    # V1 and V2 flat experiment_log.json files (same schema, different folders).
    # V2 may have multiple snapshots — dedupe by (name, timestamp).
    seen_v2: set[tuple[str, str]] = set()
    log_sources: list[tuple[str, str]] = [("V1", V1_LOG)]
    for p in V2_LOG_PATHS:
        log_sources.append(("V2", p))

    for version_label, log_path in log_sources:
        data = load_json(log_path)
        if not isinstance(data, list):
            continue
        for entry in data:
            metrics = entry.get("metrics") or {}
            stderr = entry.get("stderr") or ""
            code_gen = entry.get("code_generated") or ""
            success = bool(entry.get("success", False))
            if success:
                outcome = "success"
            elif "Preflight validation failed" in stderr:
                outcome = "preflight_failed"
            elif "TIMEOUT" in stderr:
                outcome = "timeout"
            elif "FileNotFoundError" in stderr:
                outcome = "file_not_found"
            elif "[LLM ERROR]" in code_gen or "LLM failed" in (entry.get("llm_analysis") or ""):
                outcome = "code_gen_failed"
            elif "MISSING_METRICS_LINE" in stderr:
                outcome = "no_metrics"
            elif "Traceback" in stderr:
                outcome = "training_crash"
            else:
                outcome = "unknown_failure"

            key = (entry.get("name", "?"), entry.get("timestamp", ""))
            if version_label == "V2":
                if key in seen_v2:
                    continue
                seen_v2.add(key)

            trials.append({
                "version": version_label,
                "source": version_label.lower(),
                "session": entry.get("name", "?"),
                "phase": "baseline",
                "family": entry.get("architecture", "unknown"),
                "family_key": "",
                "run_index": entry.get("id"),
                "success": success,
                "f1": metrics.get("f1"),
                "accuracy": metrics.get("accuracy"),
                "outcome": outcome,
                "error_summary": (stderr.splitlines() or [""])[-1][:200],
                "wall_seconds": None,
                "repair_attempts": 0,
                "run_dir": None,
                "timestamp": entry.get("timestamp"),
            })
    return trials


def latest_session_dir() -> str | None:
    sessions = [
        d for d in glob(os.path.join(RUNS_DIR, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "summary.json")) is False
        and not d.endswith(".jsonl")
    ]
    candidates = [
        d for d in glob(os.path.join(RUNS_DIR, "*"))
        if os.path.isdir(d)
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def latest_run_dir(session_dir: str) -> str | None:
    if not session_dir:
        return None
    run_dirs = [d for d in glob(os.path.join(session_dir, "run_*")) if os.path.isdir(d)]
    if not run_dirs:
        return None
    return max(run_dirs, key=os.path.getmtime)


def tail_jsonl(path: str, n: int = 20) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-n:]


# ----------------------------- routes ----------------------------- #

DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><title>Agent_4 Dashboard</title>
<meta http-equiv="refresh" content="10">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0b0f1a;
    --bg-2: #111729;
    --surface: #161d33;
    --surface-2: #1d2645;
    --surface-3: #232e54;
    --border: rgba(255,255,255,0.06);
    --border-strong: rgba(255,255,255,0.12);
    --text: #e6e9f5;
    --text-muted: #8a93b3;
    --text-dim: #5f6889;
    --accent: #7c5cff;
    --accent-2: #00d4b1;
    --good: #22c55e;
    --bad: #ef4444;
    --warn: #f59e0b;
    --v1: #2563eb;
    --v2: #16a34a;
    --v3: #f59e0b;
    --v4: #9333ea;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow: 0 10px 30px -10px rgba(0,0,0,0.6), 0 4px 12px -2px rgba(0,0,0,0.3);
    --shadow-glow: 0 0 0 1px rgba(124,92,255,0.15), 0 10px 40px -10px rgba(124,92,255,0.25);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-feature-settings: 'cv02','cv03','cv04','cv11';
    background:
      radial-gradient(1200px 600px at 10% -10%, rgba(124,92,255,0.18), transparent 50%),
      radial-gradient(1000px 500px at 110% 0%, rgba(0,212,177,0.10), transparent 50%),
      var(--bg);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  header {
    background:
      linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0) 60%),
      rgba(11,15,26,0.85);
    backdrop-filter: saturate(140%) blur(14px);
    -webkit-backdrop-filter: saturate(140%) blur(14px);
    border-bottom: 1px solid var(--border);
    padding: 16px 28px;
    position: sticky; top: 0; z-index: 50;
    display: flex; align-items: center; gap: 22px;
  }
  header .brand {
    display: flex; align-items: center; gap: 12px;
    font-size: 16px; font-weight: 700; letter-spacing: 0.01em;
  }
  header .brand .logo {
    width: 28px; height: 28px; border-radius: 8px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: 0 4px 16px rgba(124,92,255,0.45);
    display: grid; place-items: center; font-size: 14px; font-weight: 800; color: #0b0f1a;
  }
  header .brand .pill-live {
    margin-left: 8px;
    background: rgba(34,197,94,0.12);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.35);
    padding: 2px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.02em;
  }
  header .brand .pill-live .dot {
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: #22c55e; margin-right: 6px;
    box-shadow: 0 0 8px rgba(34,197,94,0.8);
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: 0.5 } }
  nav { display: flex; gap: 4px; flex-wrap: wrap; }
  nav a {
    color: var(--text-muted);
    padding: 6px 12px;
    border-radius: 8px;
    text-decoration: none;
    font-weight: 500;
    font-size: 14px;
    transition: all .15s ease;
  }
  nav a:hover { color: var(--text); background: rgba(255,255,255,0.06); }
  nav a.active { color: var(--text); background: rgba(124,92,255,0.16); box-shadow: inset 0 0 0 1px rgba(124,92,255,0.35); }
  main {
    padding: 24px 28px 64px;
    max-width: 1320px;
    margin: 0 auto;
  }
  main > h2 { font-size: 22px; font-weight: 700; margin: 4px 0 18px; letter-spacing: -0.01em; }
  main > h3 { font-size: 15px; font-weight: 600; margin: 28px 0 12px; color: var(--text); letter-spacing: -0.005em; }
  .row { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin-bottom: 18px; }
  .card {
    background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0) 70%), var(--surface);
    border-radius: 14px; padding: 16px 18px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
  }
  .card:hover { border-color: var(--border-strong); transform: translateY(-1px); }
  .card h3 {
    margin: 0 0 6px;
    font-size: 11px; text-transform: uppercase; color: var(--text-dim);
    letter-spacing: 0.08em; font-weight: 600;
  }
  .card .value { font-size: 26px; font-weight: 700; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
  .card .value .small { font-size: 12px; color: var(--text-muted); font-weight: 500; margin-left: 4px; }
  .card.good .value  { color: var(--good); }
  .card.bad  .value  { color: var(--bad); }
  .card.neutral .value { color: var(--text); }
  .card.v1 { box-shadow: inset 3px 0 0 var(--v1); }
  .card.v2 { box-shadow: inset 3px 0 0 var(--v2); }
  .card.v3 { box-shadow: inset 3px 0 0 var(--v3); }
  .card.v4 { box-shadow: inset 3px 0 0 var(--v4); }
  .chart-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0)), var(--surface);
    border-radius: 16px; padding: 20px 20px 14px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
  }
  .chart-card h3 { margin: 0 0 6px; font-size: 16px; font-weight: 700; }
  table {
    width: 100%; border-collapse: separate; border-spacing: 0;
    background: var(--surface);
    border-radius: 12px; overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    font-size: 13px;
  }
  th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); }
  thead th {
    background: var(--surface-2);
    font-weight: 600; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--text-muted);
  }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr { transition: background .12s ease; }
  tbody tr:hover { background: rgba(255,255,255,0.025); }
  td a { color: var(--accent); text-decoration: none; font-weight: 500; }
  td a:hover { text-decoration: underline; }
  td code, code { font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
  .pill {
    padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 600;
    display: inline-block; line-height: 1.4;
    border: 1px solid transparent;
  }
  .pill.ok   { background: rgba(34,197,94,0.10);  color: #4ade80; border-color: rgba(34,197,94,0.25); }
  .pill.err  { background: rgba(239,68,68,0.10);  color: #f87171; border-color: rgba(239,68,68,0.25); }
  .pill.warn { background: rgba(245,158,11,0.10); color: #fbbf24; border-color: rgba(245,158,11,0.25); }
  pre.code {
    background: #0a0e1a; color: #c8d3e6;
    padding: 16px; border-radius: 10px;
    overflow: auto; max-height: 480px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12.5px; line-height: 1.5;
    border: 1px solid var(--border);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
  }
  .small { color: var(--text-muted); font-size: 12.5px; }
  .small b { color: var(--text); }
  .meta-bar {
    display: flex; align-items: center; gap: 12px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 10px 14px; margin: 4px 0 18px;
    font-size: 13px; color: var(--text-muted);
  }
  .meta-bar .key { color: var(--text-dim); margin-right: 6px; font-size: 12px; }
  .meta-bar code { color: var(--text); background: var(--surface-2); padding: 2px 6px; border-radius: 4px; }
  .v-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; vertical-align: middle; margin-right: 4px; }
  .v-swatch.v1 { background: var(--v1); }
  .v-swatch.v2 { background: var(--v2); }
  .v-swatch.v3 { background: var(--v3); }
  .v-swatch.v4 { background: var(--v4); }
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 5px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }
</style></head>
<body>
<header>
  <div class="brand">
    <span class="logo">A4</span>
    <span>Agent_4 Dashboard</span>
    <span class="pill-live"><span class="dot"></span>LIVE</span>
  </div>
  <nav>
    <a href="/">Overview</a>
    <a href="/current">Current</a>
    <a href="/errors">Errors</a>
    <a href="/architectures">Architectures</a>
    <a href="/comparison">Comparison</a>
  </nav>
</header>
<main>
{{ body|safe }}
</main>
</body></html>
"""


def render(body: str) -> str:
    return render_template_string(DASHBOARD_HTML, body=body)


@app.route("/")
def overview():
    trials = gather_trials()
    successful = [t for t in trials if t["success"] and isinstance(t["f1"], (int, float))]
    failed = [t for t in trials if not t["success"]]
    total = len(trials)
    error_rate = (len(failed) / total * 100) if total else 0.0

    # Best so far across all sources
    best = max(successful, key=lambda t: t["f1"]) if successful else None

    # Color = version, shape = data type:
    #   circle = validation F1 success     (each version's JSON log)
    #   diamond = Kaggle public score      (only if recorded in overall_best.json's kaggle_submission)
    #   red X  = failure (validation F1 NA)
    version_order = {"V1": 0, "V2": 1, "V3": 2, "V4": 3}
    all_sorted = sorted(
        trials,
        key=lambda t: (version_order.get(t.get("version", "Z"), 9), t.get("session", ""), t.get("run_index") or 0),
    )
    series_ok: dict[str, list[dict[str, Any]]] = defaultdict(list)
    series_fail: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, t in enumerate(all_sorted, start=1):
        v = t.get("version", "?")
        if t.get("success") and isinstance(t.get("f1"), (int, float)):
            series_ok[v].append({
                "x": idx,
                "y": round(t["f1"], 4),
                "label": f"✓ {t.get('family','?')} · {t.get('session','?')}#{t.get('run_index','')}",
            })
        else:
            series_fail[v].append({
                "x": idx,
                "y": -0.05,  # below the F1 axis range so failures form their own row
                "label": f"✗ {t.get('outcome','failure')} · {t.get('family','?')} · {t.get('session','?')}",
            })

    # Kaggle public scores transcribed from Niccolò's Kaggle submissions page,
    # matched to the agent run that produced each submission CSV.
    # Mapping based on: submission date vs commit/run dates, filename family hints,
    # and Kaggle-vs-validation F1 proximity (transformers typically gain +0.01–0.03 on Kaggle).
    # Schema: (version, score, filename, days_ago, matched_run, comment)
    KAGGLE_SCORES = [
        # V3 BERTweet variants (16-17d ago, around the BERTweet commit 2026-04-27)
        ("V3", 0.84216, "bertweet_long_regularized_submission.csv",   16, "V3 BERTweet (long+regularized variant)", ""),
        ("V3", 0.83665, "bertweet_long_regularized_submission_2.csv", 16, "V3 BERTweet (long+regularized v2)",      ""),
        ("V3", 0.84002, "bertweet_best_overall_submission.csv",       17, "V3 BERTweet best",                        ""),
        ("V3", 0.84002, "best_overall_submission.csv (Simon)",        17, "V3 BERTweet (re-upload)",                 ""),
        ("V3", 0.78087, "best_overall_submission.csv",                17, "V3 sweep best (BERTweet baseline)",       "duplicate score"),
        ("V3", 0.78087, "best_overall_submission.csv",                17, "V3 sweep best (BERTweet baseline)",       "duplicate score"),
        ("V3", 0.78087, "best_overall_submission.csv",                14, "V3 'Agent_3 test submit'",                ""),
        # V3 with RoBERTa (22d ago — RoBERTa commit was 2026-04-21)
        ("V3", 0.83420, "best_overall_submission.csv",                22, "V3 RoBERTa best",                          ""),
        ("V3", 0.82776, "best_overall_submission.csv (Simon)",        16, "V3 (RoBERTa-era)",                         ""),
        # V3 early refinement (23d ago — first big commit)
        ("V3", 0.82837, "submission.csv",                             23, "V3 early refinement (RoBERTa)",            ""),
        ("V3", 0.80324, "submission.csv",                             23, "V3 early refinement",                      ""),
        ("V3", 0.80324, "submission.csv",                             23, "V3 early refinement (re-upload)",          ""),
        # V3 May 7 sweep — direct match to the trials in agent3_log.json
        ("V3", 0.79466, "direct_final_submission_test.csv",            7, "V3 BERTweet sweep · val F1 0.7834 (matches bertweet_20260507_230835_run_01)", ""),
        # 1d-ago uploads — file naming submission_1/2/3.csv suggests RoBERTa ensemble variants;
        # F1 0.82+ matches V3 RoBERTa val F1 (0.8224 / 0.8196 / 0.8139)
        ("V3", 0.82531, "submission_3.csv",                            1, "V3 RoBERTa run 1 · val F1 0.8139",        ""),
        ("V3", 0.82929, "submission_2.csv",                            1, "V3 RoBERTa opt run 2 · val F1 0.8196",    ""),
        ("V3", 0.82837, "submission_1.csv",                            1, "V3 RoBERTa run 2 · val F1 0.8224",        ""),
        ("V3", 0.78976, "submission.csv",                              1, "V3 BERTweet · val F1 0.7834",             ""),
    ]

    # Anchor each Kaggle diamond near its version's circles on the X axis.
    series_kaggle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    version_x_anchor: dict[str, float] = {}
    for v in ("V1", "V2", "V3", "V4"):
        xs = [p["x"] for p in series_ok.get(v, [])] + [p["x"] for p in series_fail.get(v, [])]
        if xs:
            # Place diamonds at the *end* of each version's trial range (just past the last circle).
            version_x_anchor[v] = max(xs) + 1
        else:
            version_x_anchor[v] = len(all_sorted) + 1
    for v, score, fname, days_ago, matched_run, comment in KAGGLE_SCORES:
        # Spread Kaggle diamonds horizontally within the version's bucket so they don't all overlap.
        existing = len(series_kaggle[v])
        x = version_x_anchor[v] + existing * 0.8
        label = f"Kaggle public={score} · {fname} · {days_ago}d ago · ↪ {matched_run}"
        if comment:
            label += f" ({comment})"
        series_kaggle[v].append({"x": x, "y": round(score, 4), "label": label})

    # Sweep decisions for the running session
    decisions = tail_jsonl(SWEEP_DECISIONS, n=15)
    overall = load_json(OVERALL_BEST)

    # Active-session detection: most recently modified session dir
    active_session = None
    candidates = [d for d in glob(os.path.join(RUNS_DIR, "*")) if os.path.isdir(d)]
    if candidates:
        active_session = os.path.basename(max(candidates, key=os.path.getmtime))

    # Per-family success/fail counts for the active 60-min run
    current_run_trials = [t for t in trials if t["source"] == "agent4"]
    family_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "err": 0, "best_f1": None})
    for t in current_run_trials:
        bucket = family_breakdown[t["family"]]
        if t["success"]:
            bucket["ok"] += 1
            if isinstance(t["f1"], (int, float)) and (bucket["best_f1"] is None or t["f1"] > bucket["best_f1"]):
                bucket["best_f1"] = t["f1"]
        else:
            bucket["err"] += 1

    version_counts = Counter(t.get("version") for t in trials)
    body = []
    body.append("<div class='row'>")
    body.append(f"<div class='card neutral'><h3>Trials total</h3><div class='value'>{total}</div></div>")
    body.append(f"<div class='card good'><h3>Successful</h3><div class='value'>{len(successful)}</div></div>")
    body.append(f"<div class='card bad'><h3>Failed</h3><div class='value'>{len(failed)}</div></div>")
    body.append(f"<div class='card bad'><h3>Error rate</h3><div class='value'>{error_rate:.1f}%</div></div>")
    if best:
        body.append(f"<div class='card good'><h3>Best F1</h3><div class='value'>{best['f1']:.4f}</div><div class='small'>{best['version']} · {best['family']} · {best['session']}</div></div>")
    body.append("</div>")

    # Per-version stat strip — left-bar coloured per version
    body.append("<div class='row'>")
    for v in ("V1", "V2", "V3", "V4"):
        n = version_counts.get(v, 0)
        body.append(f"<div class='card neutral {v.lower()}'><h3>{v} trials</h3><div class='value'>{n}</div></div>")
    body.append("</div>")

    body.append(f"<div class='meta-bar'>"
                f"<span><span class='key'>Active session</span><code>{active_session or '—'}</code></span>"
                f"<span style='margin-left:auto' class='small'>auto-refresh · 10s</span>"
                f"</div>")

    body.append("<div class='chart-card'>")
    body.append("<h3 style='margin:0 0 4px'>F1 across all versions</h3>")
    body.append("<p class='small' style='margin:0 0 14px'>"
                "<span class='v-swatch v1'></span><b>V1</b> &nbsp;·&nbsp; "
                "<span class='v-swatch v2'></span><b>V2</b> &nbsp;·&nbsp; "
                "<span class='v-swatch v3'></span><b>V3</b> &nbsp;·&nbsp; "
                "<span class='v-swatch v4'></span><b>V4</b> &nbsp;&nbsp; "
                "○ validation &nbsp;·&nbsp; ◆ Kaggle public &nbsp;·&nbsp; "
                "<span style='color:#f87171;font-weight:700'>✗</span> failure (y = −0.05)"
                "</p>")
    body.append("<div style='height:440px'><canvas id='f1chart'></canvas></div>")
    body.append("</div>")

    # Kaggle public score → agent run matching (transcribed from Niccolò's Kaggle account)
    body.append("<h3 style='margin-top:28px'>Kaggle public scores — matched to agent run</h3>")
    body.append("<p class='small'>Each official Kaggle submission below is matched to the agent run that "
                "wrote the submission CSV, using submission date, filename, and Kaggle-vs-validation F1 "
                "proximity. The 7-day-ago entry has an exact date match to the May 7 V3 sweep.</p>")
    body.append("<table><tr><th>Version</th><th>Kaggle file</th><th>Public score</th>"
                "<th>Days ago</th><th>Matched agent run</th><th>Note</th></tr>")
    for v, score, fname, days_ago, matched_run, comment in KAGGLE_SCORES:
        body.append(
            f"<tr><td><b style='color:{ {'V1':'#2563eb','V2':'#16a34a','V3':'#f59e0b','V4':'#9333ea'}.get(v,'#666') }'>{v}</b></td>"
            f"<td class='small'><code>{fname}</code></td>"
            f"<td><b>{score:.4f}</b></td>"
            f"<td class='small'>{days_ago}d</td>"
            f"<td class='small'>{matched_run}</td>"
            f"<td class='small'>{comment}</td></tr>"
        )
    body.append("</table>")

    # Family breakdown table
    body.append("<h3 style='margin-top:28px'>Per-family results — Agent_4 sessions</h3>")
    body.append("<table><tr><th>Family</th><th>Successful</th><th>Failed</th><th>Best F1</th></tr>")
    for fam, bucket in sorted(family_breakdown.items(), key=lambda kv: -(kv[1]["best_f1"] or 0)):
        best_f1_text = f"{bucket['best_f1']:.4f}" if bucket["best_f1"] is not None else "—"
        body.append(
            f"<tr><td><b>{fam}</b></td>"
            f"<td><span class='pill ok'>{bucket['ok']}</span></td>"
            f"<td><span class='pill err'>{bucket['err']}</span></td>"
            f"<td>{best_f1_text}</td></tr>"
        )
    body.append("</table>")

    # Recent sweep decisions
    body.append("<h3 style='margin-top:28px'>Recent sweep-planner decisions</h3>")
    if not decisions:
        body.append("<p class='small'>No decisions logged yet for the current session.</p>")
    else:
        body.append("<table><tr><th>Time</th><th>Action</th><th>Family</th><th>Reason</th><th>Time left</th></tr>")
        for d in reversed(decisions):
            cls = "ok" if d.get("action") == "try_family" else ("warn" if d.get("action") == "skip_family_permanently" else "err")
            body.append(
                f"<tr><td class='small'>{d.get('timestamp', '')}</td>"
                f"<td><span class='pill {cls}'>{d.get('action', '')}</span></td>"
                f"<td>{d.get('family_key') or '—'}</td>"
                f"<td class='small'>{(d.get('reason') or '')[:160]}</td>"
                f"<td class='small'>{d.get('time_remaining_seconds', '')}s</td></tr>"
            )
        body.append("</table>")

    if overall:
        body.append("<h3 style='margin-top:28px'>Last completed full-run result (overall_best.json)</h3>")
        body.append(
            f"<p>Best family: <b>{overall.get('best_family', '?')}</b>, "
            f"run {overall.get('best_run_index', '?')}, "
            f"metrics {overall.get('best_metrics', {})}. "
            f"Time elapsed: {overall.get('time_elapsed_seconds', '?')}s.</p>"
        )

    # Most recent 25 Agent_4 trials with link to the prompt chain
    agent4_trials = [t for t in trials if t["source"] == "agent4" and t.get("run_dir")]
    if agent4_trials:
        body.append("<h3 style='margin-top:28px'>Recent Agent_4 trials — click to see the prompt that produced each result</h3>")
        body.append("<table><tr><th>Session</th><th>Family</th><th>Run</th><th>Status</th><th>F1</th><th>Outcome</th><th>Prompts</th></tr>")
        for t in agent4_trials[-25:][::-1]:
            status = "ok" if t["success"] else "err"
            f1_txt = f"{t['f1']:.4f}" if isinstance(t["f1"], (int, float)) else "—"
            body.append(
                f"<tr><td class='small'>{t['session']}</td>"
                f"<td>{t['family']}</td>"
                f"<td>{t['run_index']}</td>"
                f"<td><span class='pill {status}'>{'success' if t['success'] else 'fail'}</span></td>"
                f"<td>{f1_txt}</td>"
                f"<td class='small'>{t['outcome']}</td>"
                f"<td class='small'><a href='/trial/{t['session']}/run_{t['run_index']:03d}'>view prompts ↗</a></td></tr>"
            )
        body.append("</table>")

    # Embed chart data as JS — three series per version: ok (circle), kaggle (diamond), fail (red X).
    ok_payload   = {v: [{"x": p["x"], "y": p["y"]} for p in pts] for v, pts in series_ok.items()}
    ok_labels    = {v: [p["label"] for p in pts] for v, pts in series_ok.items()}
    fail_payload = {v: [{"x": p["x"], "y": p["y"]} for p in pts] for v, pts in series_fail.items()}
    fail_labels  = {v: [p["label"] for p in pts] for v, pts in series_fail.items()}
    kag_payload  = {v: [{"x": p["x"], "y": p["y"]} for p in pts] for v, pts in series_kaggle.items()}
    kag_labels   = {v: [p["label"] for p in pts] for v, pts in series_kaggle.items()}
    body.append(f"""
<script>
const OK_SERIES     = {json.dumps(ok_payload)};
const OK_LABELS     = {json.dumps(ok_labels)};
const FAIL_SERIES   = {json.dumps(fail_payload)};
const FAIL_LABELS   = {json.dumps(fail_labels)};
const KAGGLE_SERIES = {json.dumps(kag_payload)};
const KAGGLE_LABELS = {json.dumps(kag_labels)};

// Per-version palette: V1 blue, V2 green, V3 yellow, V4 purple.
// Shapes encode data type, colors encode version.
const COLORS = {{"V1": "#2563eb", "V2": "#16a34a", "V3": "#f59e0b", "V4": "#9333ea"}};
const FAIL_COLOR = "#dc2626";

const datasets = [];
const all_fail_points = [];
const all_fail_labels = [];
for (const v of ["V1", "V2", "V3", "V4"]) {{
    const c = COLORS[v] || "#666";
    // Always add the validation-F1 dataset for every version so its legend
    // entry shows up — even if the version has no successful trials.
    // An empty `data` array makes Chart.js skip rendering points but keeps
    // the legend swatch visible.
    datasets.push({{
        label: v + " — validation F1 (◯)",
        data: OK_SERIES[v] || [],
        showLine: false,
        borderColor: c,
        backgroundColor: c + "cc",
        pointRadius: 6,
        pointHoverRadius: 8,
        pointStyle: "circle",
        pointBorderWidth: 1.5,
    }});
    if (KAGGLE_SERIES[v] && KAGGLE_SERIES[v].length) {{
        datasets.push({{
            label: v + " — Kaggle public score (◆)",
            data: KAGGLE_SERIES[v],
            showLine: false,
            borderColor: c,
            backgroundColor: "#ffffff",
            pointRadius: 11,
            pointHoverRadius: 13,
            pointStyle: "rectRot",       // diamond
            pointBorderWidth: 3,
        }});
    }}
    // Collect failures from every version into a single combined series.
    if (FAIL_SERIES[v] && FAIL_SERIES[v].length) {{
        all_fail_points.push(...FAIL_SERIES[v]);
        all_fail_labels.push(...(FAIL_LABELS[v] || []));
    }}
}}
// One unified failure series for the whole chart — no per-version legend entries.
if (all_fail_points.length) {{
    datasets.push({{
        label: "failure (✗)",
        data: all_fail_points,
        showLine: false,
        borderColor: FAIL_COLOR,
        backgroundColor: FAIL_COLOR,
        pointRadius: 7,
        pointHoverRadius: 9,
        pointStyle: "crossRot",
        pointBorderWidth: 2,
        _fail_labels: all_fail_labels,
    }});
}}
// Global Chart.js theming for dark mode
Chart.defaults.color = "#8a93b3";
Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.borderColor = "rgba(255,255,255,0.06)";

new Chart(document.getElementById('f1chart'), {{
    type: 'scatter',
    data: {{ datasets }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        layout: {{ padding: {{ top: 12, right: 24, bottom: 8, left: 8 }} }},
        scales: {{
            y: {{
                min: -0.1, max: 0.9,
                grid: {{ color: "rgba(255,255,255,0.05)", drawBorder: false }},
                border: {{ display: false }},
                title: {{
                    display: true,
                    text: "F1 score  ·  circles = validation  ·  diamonds = Kaggle  ·  ✗ = failure",
                    font: {{ size: 11.5, weight: 500 }},
                    color: "#8a93b3"
                }},
                ticks: {{
                    callback: v => v === -0.05 ? '✗ fail' : v.toFixed(2),
                    color: "#8a93b3",
                    font: {{ size: 11 }}
                }}
            }},
            x: {{
                grid: {{ color: "rgba(255,255,255,0.04)", drawBorder: false }},
                border: {{ display: false }},
                title: {{
                    display: true,
                    text: "trial # (chronological, grouped by version)",
                    font: {{ size: 11.5, weight: 500 }},
                    color: "#8a93b3"
                }},
                ticks: {{ color: "#8a93b3", font: {{ size: 11 }} }}
            }}
        }},
        plugins: {{
            legend: {{
                position: "top",
                align: "start",
                labels: {{
                    usePointStyle: true,
                    pointStyle: "circle",
                    padding: 14,
                    boxWidth: 8, boxHeight: 8,
                    color: "#c8d3e6",
                    font: {{ size: 12, weight: 500 }}
                }}
            }},
            tooltip: {{
                backgroundColor: "rgba(11,15,26,0.95)",
                borderColor: "rgba(255,255,255,0.12)",
                borderWidth: 1,
                titleColor: "#e6e9f5",
                bodyColor: "#c8d3e6",
                padding: 12,
                cornerRadius: 8,
                titleFont: {{ size: 12, weight: 600 }},
                bodyFont: {{ size: 12 }},
                callbacks: {{
                    label: ctx => {{
                        const lab = ctx.dataset.label || "";
                        const v = lab.split(" ")[0];
                        const isFail = lab.startsWith("failure");
                        const isKag  = lab.includes("Kaggle");
                        let txt = "";
                        if (isFail) {{
                            txt = (ctx.dataset._fail_labels || [])[ctx.dataIndex] || "";
                        }} else if (isKag) {{
                            txt = (KAGGLE_LABELS[v] || [])[ctx.dataIndex] || "";
                        }} else {{
                            txt = (OK_LABELS[v] || [])[ctx.dataIndex] || "";
                        }}
                        return isFail ? txt : `${{txt}} · F1 = ${{ctx.parsed.y}}`;
                    }}
                }}
            }},
            title: {{ display: false }}
        }}
    }}
}});
</script>
""")

    return render("\n".join(body))


@app.route("/current")
def current_input():
    """Show the spec/prompt/code currently being processed by the agent."""
    session_dir = latest_session_dir()
    if not session_dir:
        return render("<p>No sessions found yet.</p>")
    run_dir = latest_run_dir(session_dir)

    spec_path = os.path.join(run_dir, "spec.json") if run_dir else None
    prompt_path = os.path.join(run_dir, "prompt.txt") if run_dir else None
    train_path = os.path.join(run_dir, "train.py") if run_dir else None
    log_path = os.path.join(run_dir, "run.log") if run_dir else None

    spec = load_json(spec_path) if spec_path else None
    prompt = load_text(prompt_path) if prompt_path else ""
    train_code = load_text(train_path, max_bytes=80_000) if train_path else ""
    log_text = load_text(log_path, max_bytes=40_000) if log_path else ""

    decisions = tail_jsonl(SWEEP_DECISIONS, n=1)
    last_decision = decisions[-1] if decisions else None

    body = []
    body.append(f"<h2>Current session: <code>{os.path.basename(session_dir)}</code></h2>")
    if run_dir:
        body.append(f"<p class='small'>Latest run dir: <code>{os.path.basename(run_dir)}</code></p>")

    if last_decision:
        body.append("<h3>Latest sweep-planner decision</h3>")
        body.append(f"<pre class='code'>{json.dumps(last_decision, indent=2)}</pre>")

    if spec:
        body.append("<h3>Current spec.json</h3>")
        body.append(f"<pre class='code'>{json.dumps(spec, indent=2)}</pre>")

    if prompt:
        body.append("<h3>Prompt sent to code-gen LLM</h3>")
        body.append(f"<pre class='code'>{prompt[:8000]}</pre>")

    if log_text:
        body.append("<h3>run.log (tail)</h3>")
        body.append(f"<pre class='code'>{log_text[-8000:]}</pre>")

    if train_code:
        body.append("<h3>Generated training script</h3>")
        body.append(f"<pre class='code'>{train_code[:8000]}</pre>")

    return render("\n".join(body))


@app.route("/errors")
def errors_page():
    trials = gather_trials()
    failed = [t for t in trials if not t["success"]]
    by_outcome = Counter(t["outcome"] for t in failed)
    by_family = Counter(t["family"] for t in failed)

    total = len(trials)
    error_rate = (len(failed) / total * 100) if total else 0.0

    body = []
    body.append("<div class='row'>")
    body.append(f"<div class='card bad'><h3>Failed trials</h3><div class='value'>{len(failed)}</div></div>")
    body.append(f"<div class='card bad'><h3>Error rate</h3><div class='value'>{error_rate:.1f}%</div></div>")
    body.append(f"<div class='card neutral'><h3>Trials total</h3><div class='value'>{total}</div></div>")
    body.append("</div>")

    body.append("<div class='chart-card'><canvas id='outcomeChart' height='110'></canvas></div>")

    body.append("<h3 style='margin-top:24px'>Outcome breakdown</h3>")
    body.append("<table><tr><th>Outcome</th><th>Count</th></tr>")
    for outcome, count in by_outcome.most_common():
        body.append(f"<tr><td><span class='pill err'>{outcome}</span></td><td>{count}</td></tr>")
    body.append("</table>")

    body.append("<h3 style='margin-top:24px'>Failures by family</h3>")
    body.append("<table><tr><th>Family</th><th>Failed trials</th></tr>")
    for fam, count in by_family.most_common():
        body.append(f"<tr><td><b>{fam}</b></td><td>{count}</td></tr>")
    body.append("</table>")

    body.append("<h3 style='margin-top:24px'>Last 30 failures — click to see prompt chain</h3>")
    body.append("<table><tr><th>Session</th><th>Family</th><th>Run #</th><th>Outcome</th><th>Error tail</th><th>Prompts</th></tr>")
    for t in failed[-30:][::-1]:
        link = (f"<a href='/trial/{t['session']}/run_{t['run_index']:03d}'>view prompts ↗</a>"
                if t.get("run_dir") else "—")
        body.append(
            f"<tr><td class='small'>{t['session']}</td>"
            f"<td>{t['family']}</td>"
            f"<td>{t['run_index']}</td>"
            f"<td><span class='pill err'>{t['outcome']}</span></td>"
            f"<td class='small'>{(t.get('error_summary') or '')[:120]}</td>"
            f"<td class='small'>{link}</td></tr>"
        )
    body.append("</table>")

    outcome_data = dict(by_outcome)
    body.append(f"""
<script>
new Chart(document.getElementById('outcomeChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(list(outcome_data.keys()))},
        datasets: [{{
            label: 'Failures by outcome',
            data: {json.dumps(list(outcome_data.values()))},
            backgroundColor: '#c0392b'
        }}]
    }},
    options: {{ plugins: {{ legend: {{display: false}} }}, scales: {{ y: {{beginAtZero: true}} }} }}
}});
</script>
""")

    return render("\n".join(body))


@app.route("/trial/<session>/<run_name>")
def trial_detail(session: str, run_name: str):
    """Show the full prompt-chain for a specific trial (success or failure)."""
    # Reject path traversal
    if "/" in session or "\\" in session or "/" in run_name or "\\" in run_name:
        abort(400)
    # Session folders live in a few possible places depending on which agent
    # produced them and which layout the repo is using:
    #   V4 nested: runs/agent_4/<bucket>/<session>/run_NNN
    #   V4 flat:   runs/agent_4/<session>/run_NNN
    #   V3:        runs/agent_3/<session>/run_NNN
    # Try each in order and use the first one that exists.
    candidates = (
        [os.path.join(RUNS_DIR, b, session, run_name) for b in os.listdir(RUNS_DIR)
         if os.path.isdir(os.path.join(RUNS_DIR, b))]
        + [os.path.join(RUNS_DIR, session, run_name),
           os.path.join(PROJECT_ROOT, "runs", "agent_3", session, run_name)]
    )
    run_dir = next((p for p in candidates if os.path.isdir(p)), None)
    if not run_dir:
        abort(404, description=f"No such run dir: {session}/{run_name}")

    # Locate which spec-gen variant ran (initial uses spec_prompt, revisit uses search_prompt)
    spec_prompt = load_text(os.path.join(run_dir, "spec_prompt.txt"))
    if not spec_prompt:
        spec_prompt = load_text(os.path.join(run_dir, "search_prompt.txt"))
        spec_label = "Spec-gen prompt (search / revisit)"
    else:
        spec_label = "Spec-gen prompt (initial)"
    spec_response = load_text(os.path.join(run_dir, "spec_response.txt"))
    if not spec_response:
        spec_response = load_text(os.path.join(run_dir, "search_response.txt"))

    spec = load_json(os.path.join(run_dir, "spec.json"))
    code_prompt = load_text(os.path.join(run_dir, "prompt.txt"))
    code_response = load_text(os.path.join(run_dir, "generation_response.txt"))
    train_py = load_text(os.path.join(run_dir, "train.py"), max_bytes=80_000)
    run_log = load_text(os.path.join(run_dir, "run.log"), max_bytes=40_000)
    metrics = load_json(os.path.join(run_dir, "metrics.json"))

    # Repair attempts (numbered repair_attempt_1.json, _2.json, ...)
    repair_paths = sorted(glob(os.path.join(run_dir, "repair_attempt_*.json")))
    repairs: list[dict[str, Any]] = []
    for rp in repair_paths:
        repairs.append({
            "attempt": os.path.basename(rp).replace("repair_attempt_", "").replace(".json", ""),
            "raw": load_text(rp, max_bytes=20_000),
        })

    # Outcome look-up so we can flag success/failure at the top
    outcome = None
    trial_info: dict[str, Any] = {}
    for t in gather_trials():
        if t["session"] == session and isinstance(t["run_index"], int) and \
                f"run_{t['run_index']:03d}" == run_name:
            outcome = t["outcome"]
            trial_info = t
            break

    body = []
    body.append(f"<h2>Trial: <code>{session} / {run_name}</code></h2>")
    body.append("<p><a href='/'>← back to overview</a> · <a href='/errors'>errors</a></p>")

    pill_cls = "ok" if trial_info.get("success") else "err"
    status_text = "success" if trial_info.get("success") else (outcome or "failure")
    f1_text = f"{trial_info['f1']:.4f}" if isinstance(trial_info.get("f1"), (int, float)) else "—"
    body.append("<div class='row'>")
    body.append(f"<div class='card neutral'><h3>Family</h3><div class='value' style='font-size:18px'>{trial_info.get('family', '?')}</div></div>")
    body.append(f"<div class='card {'good' if trial_info.get('success') else 'bad'}'><h3>Status</h3><div class='value' style='font-size:18px'><span class='pill {pill_cls}'>{status_text}</span></div></div>")
    body.append(f"<div class='card good'><h3>F1</h3><div class='value' style='font-size:22px'>{f1_text}</div></div>")
    if trial_info.get("repair_attempts"):
        body.append(f"<div class='card bad'><h3>Repair attempts</h3><div class='value' style='font-size:22px'>{trial_info['repair_attempts']}</div></div>")
    if trial_info.get("wall_seconds"):
        body.append(f"<div class='card neutral'><h3>Wall seconds</h3><div class='value' style='font-size:22px'>{trial_info['wall_seconds']}</div></div>")
    body.append("</div>")

    if trial_info.get("error_summary"):
        body.append(f"<p><b>Error tail:</b> <code>{trial_info['error_summary']}</code></p>")

    # 1. Spec prompt
    if spec_prompt:
        body.append(f"<h3>1. {spec_label}</h3>")
        body.append(f"<pre class='code'>{spec_prompt[:12000]}</pre>")
    if spec_response:
        body.append("<h3>2. Spec-gen LLM response</h3>")
        body.append(f"<pre class='code'>{spec_response[:8000]}</pre>")

    # 2. Validated spec
    if spec:
        body.append("<h3>3. Validated spec.json</h3>")
        body.append(f"<pre class='code'>{json.dumps(spec, indent=2)}</pre>")

    # 3. Code-gen prompt + response
    if code_prompt:
        body.append("<h3>4. Code-gen prompt</h3>")
        body.append(f"<pre class='code'>{code_prompt[:12000]}</pre>")
    if code_response:
        body.append("<h3>5. Code-gen LLM response</h3>")
        body.append(f"<pre class='code'>{code_response[:12000]}</pre>")

    # 4. Generated code
    if train_py:
        body.append("<h3>6. Generated training script (post-repair if any)</h3>")
        body.append(f"<pre class='code'>{train_py[:20000]}</pre>")

    # 5. Repair attempts — these are the prompts/responses produced when a trial failed.
    if repairs:
        body.append(f"<h3>7. Surgical repair attempts ({len(repairs)})</h3>")
        for rep in repairs:
            body.append(f"<h4>Repair attempt {rep['attempt']}</h4>")
            body.append(f"<pre class='code'>{rep['raw']}</pre>")

    # 6. run.log
    if run_log:
        body.append("<h3>8. run.log (execution + analysis)</h3>")
        body.append(f"<pre class='code'>{run_log[-10000:]}</pre>")

    if metrics:
        body.append("<h3>9. metrics.json</h3>")
        body.append(f"<pre class='code'>{json.dumps(metrics, indent=2)}</pre>")

    return render("\n".join(body))


@app.route("/architectures")
def architectures():
    """One page with all four architecture diagrams + a short description of each version."""
    versions = [
        {
            "id": "v1",
            "title": "V1 — Single-architecture LLM Baseline",
            "subtitle": "src/agents/v1_simple.py (Keras embedding + LSTM)",
            "summary": (
                "LLM is asked for a single experiment dict (clamped to a 4-key search space: "
                "model_type ∈ {avg_embed, lstm}, vocab_size ∈ {5k, 10k}, seq_length ∈ {50, 100}, "
                "embed_dim ∈ {32, 64}). One script trains and reports F1; results appended to a JSONL log "
                "and fed back into the next prompt as history."
            ),
        },
        {
            "id": "v2",
            "title": "V2 — Single-file Autonomous Multi-architecture Agent",
            "subtitle": "Agent_V2/agent_fully_autonomous.py",
            "summary": (
                "One Python file drives a full experiment loop across REQUIRED_ARCHITECTURES. "
                "LLM proposes a complete training script; preflight + syntax checks run before execution; "
                "failures trigger a single LLM repair pass. ExperimentMemory tracks rolling history, best F1, "
                "and a plateau detector. Log is a flat experiment_log.json."
            ),
        },
        {
            "id": "v3",
            "title": "V3 — Multi-family Modular Agent",
            "subtitle": "src/Agent_3/agent.py (sweep + opt + final retrain)",
            "summary": (
                "Refactored into a package: per-family hooks (BoW · BoW_advanced · CNN · LSTM · Embedding · "
                "RoBERTa · BERTweet), Jinja templates, surgical repair contract (≤8 attempts), tiered memory. "
                "Sweep walks all families on a 4k sample; top-2 architectures then enter an optimize phase on 10k rows; "
                "final retrain on full labeled data. Flat agent3_log.json."
            ),
        },
        {
            "id": "v4",
            "title": "V4 — LLM-driven Sweep Planner",
            "subtitle": "src/Agent_4/agent.py (this codebase)",
            "summary": (
                "Sweep order is no longer a hardcoded list — an LLM planner reads the per-family state table "
                "every step and chooses try_family / skip_family_permanently / stop. Sweep ends at a fixed "
                "40-min wall-clock boundary. 2k-row fixed sample is shared across sweep/opt/final retrain. "
                "New artifact: sweep_decisions.jsonl logs every planner decision with prompt + raw response."
            ),
        },
    ]
    body = ["<h2>Agent versions</h2>",
            "<p class='small'>PNGs are stored under <code>src/Agent_4/docs/</code> and rendered from the "
            ".dot sources next to them.</p>"]
    for v in versions:
        body.append("<div class='chart-card' style='margin-bottom:24px'>")
        body.append(f"<h3 style='margin-top:0'>{v['title']}</h3>")
        body.append(f"<p class='small'>{v['subtitle']}</p>")
        body.append(f"<p>{v['summary']}</p>")
        body.append(f"<img src='/docs/architecture_{v['id']}.png' alt='{v['title']}' "
                    f"style='max-width:100%; border:1px solid #eee; border-radius:6px'>")
        body.append("</div>")
    return render("\n".join(body))


@app.route("/docs/<path:filename>")
def serve_docs(filename: str):
    return send_from_directory(DOCS_DIR, filename)


# Where each version's Kaggle-ready submission CSV lives (or "not produced" if missing).
SUBMISSION_PATHS = {
    "V1": r"C:\Users\shoxx\Downloads\apa-disaster-tweets-agent\submissions",
    "V2": r"C:\Users\shoxx\Downloads\Agent_V2\submissions",
    "V3": os.path.join(PROJECT_ROOT, "submissions"),
    "V4": os.path.join(AGENT_ROOT, "submissions"),
}


def submission_summary(folder: str) -> dict[str, Any]:
    """Return {csv_count, latest_csv, latest_rows, class_dist} for a submissions folder."""
    if not os.path.isdir(folder):
        return {"exists": False}
    csvs = [p for p in glob(os.path.join(folder, "*.csv"))]
    if not csvs:
        return {"exists": True, "csv_count": 0}
    latest = max(csvs, key=os.path.getmtime)
    rows, ones, zeros = 0, 0, 0
    try:
        with open(latest, "r", encoding="utf-8", errors="replace") as fh:
            next(fh, None)  # skip header
            for line in fh:
                rows += 1
                tail = line.rstrip().rsplit(",", 1)[-1]
                if tail == "1":
                    ones += 1
                elif tail == "0":
                    zeros += 1
    except Exception:
        pass
    return {
        "exists": True,
        "csv_count": len(csvs),
        "latest_csv": os.path.basename(latest),
        "latest_path": latest,
        "latest_rows": rows,
        "ones": ones,
        "zeros": zeros,
        "mtime": datetime.fromtimestamp(os.path.getmtime(latest)).strftime("%Y-%m-%d %H:%M"),
    }


@app.route("/comparison")
def comparison():
    """Per-version table: best validation F1, failures, submission CSV, Kaggle score."""
    trials = gather_trials()
    by_version: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trials:
        by_version[t.get("version", "?")].append(t)

    body = [
        "<h2>Version comparison</h2>",
        "<p class='small'>Validation F1 = recorded in each version's JSON log. "
        "Kaggle public score = pulled from agent log only if AGENT3_AUTO_SUBMIT_KAGGLE was set when the run "
        "happened. None of the V1–V4 runs in this project recorded a Kaggle response, so that column is "
        "marked 'not recorded' where missing.</p>",
        "<table>",
        "<tr><th>Version</th><th>Total trials</th><th>Succeeded</th><th>Failed</th>"
        "<th>Best validation F1</th><th>Best F1 — who</th>"
        "<th>Failure outcomes</th>"
        "<th>Submission CSV</th>"
        "<th>Kaggle public score</th></tr>",
    ]

    for v in ("V1", "V2", "V3", "V4"):
        rows = by_version.get(v, [])
        succ = [t for t in rows if t.get("success") and isinstance(t.get("f1"), (int, float))]
        fail = [t for t in rows if not t.get("success")]
        if succ:
            best = max(succ, key=lambda t: t["f1"])
            best_f1 = f"{best['f1']:.4f}"
            best_who = f"{best.get('family', '?')} · {best.get('session', '?')}"
        else:
            best_f1 = "—"
            best_who = "no successful trial in log"
        fail_counts = Counter(t.get("outcome", "unknown") for t in fail)
        fail_text = ", ".join(f"{k}={c}" for k, c in fail_counts.most_common()) or "—"

        sub = submission_summary(SUBMISSION_PATHS.get(v, ""))
        if not sub.get("exists"):
            sub_text = "<span class='small'>folder missing</span>"
        elif sub.get("csv_count", 0) == 0:
            sub_text = "<span class='pill err'>no CSV produced</span>"
        else:
            sub_text = (f"<b>{sub['csv_count']}</b> CSV(s); latest: "
                        f"<code>{sub['latest_csv']}</code> "
                        f"<span class='small'>({sub['latest_rows']} rows · "
                        f"0:{sub['zeros']} / 1:{sub['ones']} · {sub['mtime']})</span>")

        # Kaggle score lookup — V4 stores it under overall_best.kaggle_submission if AGENT3_AUTO_SUBMIT_KAGGLE was on.
        kaggle_text = "<span class='small'>not recorded</span>"
        if v == "V4":
            overall = load_json(OVERALL_BEST) or {}
            ks = overall.get("kaggle_submission")
            if isinstance(ks, dict) and ks.get("submitted"):
                kaggle_text = (f"public={ks.get('public_score', '?')} · "
                               f"private={ks.get('private_score', '?')} · "
                               f"status={ks.get('status', '?')}")

        body.append(
            f"<tr><td><b>{v}</b></td>"
            f"<td>{len(rows)}</td>"
            f"<td><span class='pill ok'>{len(succ)}</span></td>"
            f"<td><span class='pill err'>{len(fail)}</span></td>"
            f"<td><b>{best_f1}</b></td>"
            f"<td class='small'>{best_who}</td>"
            f"<td class='small'>{fail_text}</td>"
            f"<td class='small'>{sub_text}</td>"
            f"<td class='small'>{kaggle_text}</td></tr>"
        )
    body.append("</table>")

    # Per-version successful-trial detail
    body.append("<h3 style='margin-top:28px'>Successful trials per version</h3>")
    body.append("<table><tr><th>Version</th><th>Family / arch</th><th>Run / session</th><th>F1</th><th>Accuracy</th></tr>")
    for v in ("V1", "V2", "V3", "V4"):
        for t in sorted(
            [x for x in by_version.get(v, []) if x.get("success") and isinstance(x.get("f1"), (int, float))],
            key=lambda x: -x["f1"],
        ):
            acc = t.get("accuracy")
            acc_text = f"{acc:.4f}" if isinstance(acc, (int, float)) else "—"
            body.append(
                f"<tr><td><b>{v}</b></td>"
                f"<td>{t.get('family', '?')}</td>"
                f"<td class='small'>{t.get('session', '?')}</td>"
                f"<td><b>{t['f1']:.4f}</b></td>"
                f"<td>{acc_text}</td></tr>"
            )
    body.append("</table>")

    # Per-version failures
    body.append("<h3 style='margin-top:28px'>Failures per version</h3>")
    body.append("<table><tr><th>Version</th><th>Family / arch</th><th>Outcome</th><th>Error tail</th></tr>")
    for v in ("V1", "V2", "V3", "V4"):
        for t in [x for x in by_version.get(v, []) if not x.get("success")]:
            link = ""
            if v == "V4" and t.get("run_dir"):
                link = f" <a href='/trial/{t['session']}/run_{t['run_index']:03d}'>↗</a>"
            body.append(
                f"<tr><td><b>{v}</b></td>"
                f"<td>{t.get('family', '?')}</td>"
                f"<td><span class='pill err'>{t.get('outcome', 'unknown')}</span>{link}</td>"
                f"<td class='small'>{(t.get('error_summary') or '')[:140]}</td></tr>"
            )
    body.append("</table>")

    return render("\n".join(body))


@app.route("/api/trials")
def api_trials():
    return jsonify(gather_trials())


if __name__ == "__main__":
    # macOS AirPlay Receiver also binds to :5000 by default, which makes
    # Flask either fail to start or quietly hand back AirPlay's HTTP 403.
    # Default to 5050 instead; override with DASHBOARD_PORT.
    port = int(os.environ.get("DASHBOARD_PORT", "5050"))
    # Bind to loopback only — no need for LAN exposure, and it also avoids
    # the macOS firewall popup every time the app is launched.
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    print(f"Dashboard running on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
