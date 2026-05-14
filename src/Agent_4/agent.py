"""Agent_4 — autonomous experiment runner with LLM-driven sweep planner.

Design highlights:
- Sweep order is decided by an LLM planner each step, not a hardcoded list.
- Each planner decision = one trial of one family (no per-family attempt cap).
- A family is never auto-consumed: the planner can revisit a success, retry a
  failure, or declare a family permanently dead via skip_family_permanently.
- Sweep runs for a fixed 35-minute window (configurable). Then the opt phase
  picks the highest-F1 family and tunes hyperparameters in it.
- Final submission retrains on a 2k-row sample so it always fits inside the
  1-hour wall-clock budget on CPU.
"""

from __future__ import annotations

import argparse
import ast
import atexit
import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from artifacts import copy_if_exists, create_session_dir, create_run_dir, write_json, write_text
from generate_spec import generate_initial_spec
from json_utils import extract_json_object, pretty_json
from kaggle_submit import auto_submit_enabled, submit_and_wait
from llm import OllamaClient
from memory import Agent4Memory
from prompts import (
    ANALYSIS_PROMPT_TEMPLATE,
    DATA_CONTEXT_TEMPLATE,
    FULL_SYSTEM,
    SWEEP_PLANNER_SYSTEM,
)
from render_templates import render_family_prompt
from repair import request_surgical_repair
from sandbox import run_experiment, tail
from search import propose_next_spec, summarize_trials
from sweep_planner import (
    FamilyState,
    SweepDecision,
    classify_trial_outcome,
    decision_to_log_record,
    select_next_sweep_action,
)
from submit_tails import append_submission_tail

import families.experiment_bow as exp_bow
import families.experiment_bow_advanced as exp_bow_advanced
import families.experiment_bertweet as exp_bertweet
import families.experiment_cnn as exp_cnn
import families.experiment_embedding_dl as exp_embedding_dl
import families.experiment_lstm as exp_lstm
import families.experiment_roberta as exp_roberta


DATA_DIR_ENV = "DISASTER_AGENT_DATA_DIR"
DEFAULT_DATA_DIR = "data"
MAX_SEARCH_RUNS = int(os.environ.get("AGENT4_MAX_RUNS", "4"))
MAX_REPAIR_ATTEMPTS = int(os.environ.get("DISASTER_AGENT_MAX_REPAIRS", "4"))
TOTAL_TIME_BUDGET_SECONDS = int(os.environ.get("AGENT4_TOTAL_TIME_BUDGET_SECONDS", str(60 * 60)))
# Sweep runs for a fixed wall-clock window (default 45 min). Opt phase is
# disabled — the sweep planner has the whole 45 min to explore + revisit
# whichever families it finds promising. After the sweep deadline, we go
# straight to final submission.
SWEEP_DURATION_SECONDS = int(os.environ.get("AGENT4_SWEEP_DURATION_SECONDS", str(45 * 60)))
SWEEP_SAMPLE_ROWS = int(os.environ.get("AGENT4_SWEEP_SAMPLE_ROWS", "2000"))
# Final submission trains on a 5k sample (>= 2.5x the sweep sample). The
# 5k retrain gives the final model more data than the per-trial 2k seen
# during sweep, at the cost of ~12-15 min wall on CPU for a transformer.
# With sweep capped at 45 min, the remaining 15 min comfortably fit a
# 5k retrain + test prediction.
FINAL_TRAIN_ROWS = int(os.environ.get("AGENT4_FINAL_TRAIN_ROWS", "5000"))
VALIDATION_FRACTION = min(max(float(os.environ.get("AGENT4_VALIDATION_FRACTION", "0.2")), 0.05), 0.5)
RUN_START_BUFFER_SECONDS = int(os.environ.get("AGENT4_RUN_START_BUFFER_SECONDS", "120"))
# Use the smarter code-gen LLM for the sweep planner. The small gemma model
# could not reliably read multi-row state tables (kept hallucinating attempt
# counts), which broke Fix B's "skip after 2 consecutive code_gen_failed" rule.
SWEEP_PLANNER_MODEL = os.environ.get("AGENT4_SWEEP_PLANNER_MODEL", "qwen2.5-coder:14b")
# Measured one-run wall times on 2026-04-20, plus an extra ~120s cushion
# for up to four repair attempts at roughly 30s each.
FAMILY_RUN_ESTIMATES = {
    "bow": 100,
    "bow_advanced": 200,
    "cnn": 200,
    "embedding_dl": 240,
    "lstm": 200,
    "roberta": 500,
    "bertweet": 500,
}

FAMILY_MODULES = {
    "bertweet": exp_bertweet,
    "roberta": exp_roberta,
    "bow_advanced": exp_bow_advanced,
    "cnn": exp_cnn,
    "embedding_dl": exp_embedding_dl,
    "lstm": exp_lstm,
    "bow": exp_bow,
}


def get_data_paths() -> tuple[str, str]:
    data_dir = os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR)
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    if os.path.exists(train_path) and os.path.exists(test_path):
        return train_path, test_path
    if os.path.exists("train.csv") and os.path.exists("test.csv"):
        return "train.csv", "test.csv"
    raise FileNotFoundError("Could not find train.csv and test.csv.")


def build_data_context() -> str:
    train_path, test_path = get_data_paths()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    vc = train["target"].value_counts()
    total = len(train)
    return DATA_CONTEXT_TEMPLATE.format(
        train_rows=len(train),
        test_rows=len(test),
        class_0=vc.get(0, 0),
        class_1=vc.get(1, 0),
        pct_0=100 * vc.get(0, 0) / total,
        pct_1=100 * vc.get(1, 0) / total,
        missing_kw=100 * train["keyword"].isna().mean(),
        missing_loc=100 * train["location"].isna().mean(),
    )


def syntax_check(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, f"SyntaxError line {exc.lineno}: {exc.msg}"


def analyze_run(
    llm: OllamaClient,
    family: str,
    run_label: str,
    spec: dict[str, Any],
    result: dict[str, Any],
) -> str:
    status = "success" if result["success"] else ("timeout" if result["timed_out"] else "crash")
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        name=run_label,
        family=family,
        status=status,
        spec_json=pretty_json(spec),
        metrics=result.get("metrics") or "none",
        stdout_tail=tail(result.get("stdout", ""), 30),
        stderr_tail=tail(result.get("stderr", ""), 20),
    )
    return llm.analyze(prompt)


def metric_f1(result: dict[str, Any]) -> float:
    metrics = result.get("metrics") or {}
    value = metrics.get("f1")
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def reset_public_submissions_dir() -> str:
    submissions_dir = os.path.join(os.getcwd(), "submissions")
    if os.path.exists(submissions_dir):
        shutil.rmtree(submissions_dir)
    os.makedirs(submissions_dir, exist_ok=True)
    return submissions_dir


def remaining_seconds(deadline_ts: float | None) -> int:
    if deadline_ts is None:
        return 10**9
    return max(0, int(deadline_ts - time.time()))


def estimated_run_seconds(family_key: str) -> int:
    return FAMILY_RUN_ESTIMATES.get(family_key, 480)


def can_start_run(family_key: str, deadline_ts: float | None) -> bool:
    return remaining_seconds(deadline_ts) >= estimated_run_seconds(family_key) + RUN_START_BUFFER_SECONDS


def best_trial_from_summary(summary: dict[str, Any]) -> dict[str, Any] | None:
    best_run_index = summary.get("best_run_index")
    if best_run_index is None:
        return None
    for trial in summary.get("trials", []):
        if trial.get("run_index") == best_run_index:
            return trial
    return None


def load_text_if_exists(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_kaggle_submission_message(best_overall: dict[str, Any]) -> str:
    metrics = best_overall.get("best_metrics") or {}
    f1 = metrics.get("f1", "NA")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        "Agent_4 auto-submit | "
        f"family={best_overall.get('family', 'unknown')} | "
        f"run={best_overall.get('best_run_index', 'NA')} | "
        f"f1={f1} | "
        f"ts={timestamp}"
    )


def validate_submission_file(submission_path: str) -> str | None:
    if not os.path.exists(submission_path):
        return "Submission file was not created."
    try:
        submission_df = pd.read_csv(submission_path)
    except Exception as exc:
        return f"Submission file could not be read as CSV: {exc}"

    expected_columns = ["id", "target"]
    actual_columns = list(submission_df.columns)
    if actual_columns != expected_columns:
        return f"Submission columns must be exactly {expected_columns}, found {actual_columns}."
    if submission_df.empty:
        return "Submission CSV is empty."

    _, test_path = get_data_paths()
    expected_rows = len(pd.read_csv(test_path))
    if len(submission_df) != expected_rows:
        return f"Submission row count must be {expected_rows}, found {len(submission_df)}."
    return None


def prepare_final_submission_payload(summary: dict[str, Any], public_submission_path: str) -> tuple[dict[str, Any] | None, str | None]:
    best_trial = best_trial_from_summary(summary)
    if not best_trial:
        return None, "Could not find the best trial in the selected family summary."
    session_dir = os.path.join(os.path.dirname(__file__), "runs", summary["session_name"])
    code = load_text_if_exists(os.path.join(session_dir, "best_train.py"))
    if not code:
        return None, "Could not load best_train.py for the selected best run."

    module = FAMILY_MODULES[summary["family_key"]]
    spec = dict(best_trial.get("spec", {}))
    spec["submission_path"] = public_submission_path
    spec["experiment_name"] = "best_overall_submission"
    spec["val_size"] = VALIDATION_FRACTION
    if hasattr(module, "normalize_spec"):
        spec = module.normalize_spec(spec)
    if hasattr(module, "tune_frozen_code"):
        code = module.tune_frozen_code(code, spec, "best_overall_submission")
    if hasattr(module, "apply_light_autofixes"):
        code = module.apply_light_autofixes(code, spec)
    # Append the orchestrator-owned hardcoded submission tail. This is what
    # actually writes the CSV — the LLM's WRITE_SUBMISSION/FINAL_SUBMISSION
    # blocks never run (both env vars are 0 at final time), so we don't need
    # to patch their paths.
    code = append_submission_tail(code, summary["family"])
    return {
        "code": code,
        "family": summary["family"],
        "family_key": summary["family_key"],
        "module": module,
        "run_name": "best_overall_submission",
        "session_dir": session_dir,
        "spec": spec,
        "submission_path": public_submission_path,
    }, None


def execute_final_submission(
    llm: OllamaClient,
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any], int]:
    """Run the frozen training script once with the orchestrator-owned
    submission tail appended. No LLM-driven repair attempts at this stage —
    the tail is hardcoded so there is nothing for the LLM to fix, and
    historically every repair attempt at final time has corrupted the tail
    or introduced new bugs. If the run fails, we accept that failure.

    The `llm` argument is kept for signature compatibility with callers but
    is not used here.
    """
    del llm  # unused — kept for signature compatibility
    code = str(payload["code"])  # already has the hardcoded tail appended
    module = payload["module"]
    run_name = str(payload["run_name"])
    spec = dict(payload["spec"])
    submission_path = str(payload["submission_path"])

    # Cheap, deterministic preprocessing only — no LLM calls.
    if hasattr(module, "apply_light_autofixes"):
        code = module.apply_light_autofixes(code, spec)
    # Strip-and-reappend the tail in case apply_light_autofixes touched it.
    code = append_submission_tail(code, str(payload["family"]))

    ok, syntax_err = syntax_check(code)
    if not ok:
        return code, {
            "success": False,
            "timed_out": False,
            "dry_run_failed": False,
            "metrics": {},
            "stdout": "",
            "stderr": "Final submission script has a syntax error after tail injection:\n" + syntax_err,
        }, 0

    # Single-shot run. The tail is deterministic; if the script still fails,
    # there is nothing useful an LLM patch can do without risking the tail.
    #
    # write_submission=True so sandbox.run_experiment loads the full
    # 3,263-row test.csv (not the 1-row truncation it uses during sweep).
    # final_submission=False keeps the LLM's `if FINAL_SUBMISSION:` retrain
    # branch dormant — that branch is historically the buggiest part of
    # the script and we don't need it: the tail predicts directly using
    # the 80%-trained model that produced the reported val F1.
    result = run_experiment(
        code,
        run_name,
        train_fraction=1.0,
        train_rows=FINAL_TRAIN_ROWS,
        write_submission=True,
        final_submission=False,
        submission_path=submission_path,
        # Skip the dry run. It exists to fail fast so the LLM can repair the
        # script, but the final-submission step has no repair mechanism, and
        # the dry run's hardcoded tail would clobber `submission_path` with a
        # one-class CSV produced from 16 untrained rows. If the real run later
        # fails, the published file would be that garbage.
        skip_dry_run=True,
    )
    submission_error = validate_submission_file(submission_path)
    if submission_error is None:
        # Accept the run as success if the CSV is valid, even if the script
        # exited non-zero (e.g., a warning was treated as error after the
        # tail had already written the CSV).
        if not result.get("success"):
            result["success"] = True
            result["stderr"] = (
                result.get("stderr", "").rstrip()
                + ("\n" if result.get("stderr") else "")
                + "Accepted final submission because a valid submission CSV was created."
            ).strip()
    else:
        # CSV is missing or malformed. Flag the failure but don't ask the LLM
        # to fix it — repairs have historically made things worse.
        result["success"] = False
        result["stderr"] = (result.get("stderr", "") + "\n" + submission_error).strip()

    return code, result, 0


def constrain_phase_spec(spec: dict[str, Any]) -> dict[str, Any]:
    constrained = dict(spec)
    constrained["val_size"] = VALIDATION_FRACTION
    return constrained


def phase_train_rows(phase_label: str) -> int | None:
    if phase_label in {"sweep", "opt"}:
        return SWEEP_SAMPLE_ROWS
    return None


def create_fixed_phase_data_dir(
    phase_label: str,
    train_rows: int | None,
    test_rows: int | None = 1,
    seed: int = 42,
) -> tuple[str, dict[str, Any]]:
    train_path, test_path = get_data_paths()
    train = pd.read_csv(train_path)
    original_train_rows = len(train)
    if train_rows and train_rows > 0 and train_rows < len(train):
        train = train.sample(n=train_rows, random_state=seed).reset_index(drop=True)
    else:
        train = train.reset_index(drop=True)

    test = pd.read_csv(test_path)
    if test_rows is not None:
        test = test.head(test_rows)

    data_dir = tempfile.mkdtemp(prefix=f"agent4_{phase_label}_fixed_data_")
    train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(data_dir, "test.csv"), index=False)

    y = train["target"]
    stratify_labels = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    train_idx, val_idx = train_test_split(
        train.index,
        test_size=VALIDATION_FRACTION,
        random_state=42,
        stratify=stratify_labels,
    )
    train_ids = train.loc[train_idx, "id"].astype(int).tolist() if "id" in train.columns else list(map(int, train_idx))
    val_ids = train.loc[val_idx, "id"].astype(int).tolist() if "id" in train.columns else list(map(int, val_idx))
    manifest = {
        "phase": phase_label,
        "data_dir": data_dir,
        "source_train_rows": original_train_rows,
        "sample_rows": len(train),
        "validation_fraction": VALIDATION_FRACTION,
        "split_random_state": 42,
        "sample_seed": seed,
        "expected_train_rows": len(train_ids),
        "expected_validation_rows": len(val_ids),
        "train_ids": train_ids,
        "validation_ids": val_ids,
    }
    write_json(os.path.join(data_dir, "fixed_split_manifest.json"), manifest)
    return data_dir, manifest


def cleanup_phase_data_dirs(paths: list[str]) -> None:
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)


def execute_family(
    llm: OllamaClient,
    memory: Agent4Memory,
    family_key: str,
    max_runs: int | None,
    stop_after_ts: float | None = None,
    phase_label: str = "sweep",
    seeded_trial: dict[str, Any] | None = None,
    seeded_trials: list[dict[str, Any]] | None = None,
    seeded_code: str | None = None,
    phase_data_dir: str | None = None,
    phase_split_manifest: dict[str, Any] | None = None,
    planner_guidance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    module = FAMILY_MODULES[family_key]
    family = module.FAMILY
    resolved_max_runs = max_runs if max_runs is not None else int(getattr(module, "default_max_runs", lambda: MAX_SEARCH_RUNS)())
    started = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{family_key}_{started}" if phase_label == "sweep" else f"{family_key}_{phase_label}_{started}"
    create_session_dir(session_name)
    data_context = build_data_context()
    family_history = memory.history_summary(family)
    trials: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None
    best_code: str | None = seeded_code
    frozen_code: str | None = seeded_code
    freeze_after_success = phase_label == "opt" and bool(getattr(module, "freeze_after_first_success", lambda: False)())
    # Multi-seed path (sweep planner revisits): show propose_next_spec the FULL
    # prior history of this family so it can avoid re-trying specs that already
    # underperformed and learn from the per-spec → F1 mapping.
    if seeded_trials:
        for idx, prior in enumerate(seeded_trials):
            seed_record = dict(prior)
            # Use negative indices so seeded records do not collide with the
            # real run_index of the current trial.
            seed_record["run_index"] = -(idx + 1)
            seed_record["analysis"] = seed_record.get(
                "analysis", f"Seeded from earlier {family} attempt."
            )
            seed_record["seeded"] = True
            trials.append(seed_record)
        # Pick the highest-F1 seeded as the best so far.
        try:
            best_seeded = max(
                seeded_trials,
                key=lambda t: float((t.get("metrics") or {}).get("f1", -1.0)),
            )
            best_trial = dict(best_seeded)
            best_trial["seeded"] = True
        except (ValueError, TypeError):
            pass
        print(
            f"[Seed] {family} sweep revisit seeded with {len(seeded_trials)} prior trial(s); "
            f"propose_next_spec sees the full per-spec → F1 mapping."
        )
    elif seeded_trial:
        # Single-seed path (opt phase): legacy behaviour, preserved for the
        # winner-optimization flow which seeds only the best sweep trial.
        seed_record = dict(seeded_trial)
        seed_record["run_index"] = 0
        seed_record["analysis"] = seed_record.get("analysis", "Seeded from the best prior family run.")
        seed_record["seeded"] = True
        trials.append(seed_record)
        best_trial = seed_record
        if frozen_code:
            print(
                f"[Seed] {family} optimization seeded from prior best run "
                f"(f1={metric_f1({'metrics': seed_record.get('metrics', {})}):.4f})."
            )

    print("\n" + "=" * 72)
    print(f"AGENT_4 FAMILY RUN | {family} | phase={phase_label} | session={session_name}")
    print("=" * 72)

    for run_index in range(1, resolved_max_runs + 1):
        if not can_start_run(family_key, stop_after_ts):
            print(
                f"[Budget] Skipping additional {family} runs in phase '{phase_label}' "
                f"because only {remaining_seconds(stop_after_ts)}s remain."
            )
            break
        # Per-trial telemetry the sweep planner reads to make its next decision.
        _trial_started_at = time.time()
        _code_gen_failed_early = False
        _repair_exhausted_without_running = False
        run_dir = create_run_dir(session_name, run_index)
        run_name = f"{session_name}_run_{run_index:02d}"
        submission_path = os.path.join(run_dir, "submission.csv")

        # Decide whether the spec comes from a fresh LLM call (generate_initial_spec)
        # or from the search/exploration path (propose_next_spec).
        #
        # We must NOT use generate_initial_spec on a sweep revisit. A revisit
        # populates `seeded_trials` (the LIST) with the family's full prior
        # history, and the LLM must see that history so it doesn't propose the
        # exact same spec again. Earlier the condition only checked the singular
        # `seeded_trial` (opt-phase only), which let revisits silently fall back
        # to a fresh prompt and produce identical specs.
        has_prior_history = bool(seeded_trial) or bool(seeded_trials) or len(trials) > 0
        if run_index == 1 and not has_prior_history:
            spec_bundle = generate_initial_spec(
                llm=llm,
                module=module,
                run_name=run_name,
                submission_path=submission_path,
                data_context=data_context,
                history_summary=family_history,
            )
            spec_stage_name = "spec"
        else:
            spec_bundle = propose_next_spec(
                llm=llm,
                module=module,
                run_name=run_name,
                submission_path=submission_path,
                data_context=data_context,
                history_summary=family_history,
                trials=trials,
                phase=phase_label,
                planner_guidance=planner_guidance,
            )
            spec_stage_name = "search"

        spec = spec_bundle["spec"]
        if hasattr(module, "normalize_spec"):
            spec = module.normalize_spec(spec)
        spec = constrain_phase_spec(spec)
        prompt = render_family_prompt(
            module=module,
            spec=spec,
            data_context=data_context,
            history_summary=family_history,
            trial_summary=summarize_trials(trials),
        )

        write_json(os.path.join(run_dir, "spec.json"), spec)
        write_text(os.path.join(run_dir, f"{spec_stage_name}_prompt.txt"), spec_bundle["prompt"])
        write_text(os.path.join(run_dir, f"{spec_stage_name}_response.txt"), spec_bundle["raw_response"])
        write_text(os.path.join(run_dir, "prompt.txt"), prompt)

        print(f"\n[Run {run_index}/{resolved_max_runs}] family={family} spec={spec}")
        if freeze_after_success and frozen_code:
            response = f"Reused frozen successful code baseline and updated only {family} hyperparameters."
            run_code = module.tune_frozen_code(frozen_code, spec, run_name)
            write_text(os.path.join(run_dir, "generation_response.txt"), response)
        else:
            response, code = llm.propose(FULL_SYSTEM, prompt)
            write_text(os.path.join(run_dir, "generation_response.txt"), response)
            if not code:
                _code_gen_failed_early = True
                result = {
                    "success": False,
                    "timed_out": False,
                    "dry_run_failed": False,
                    "metrics": {},
                    "stdout": "",
                    "stderr": "LLM returned no code block.",
                }
                analysis = analyze_run(llm, family, run_name, spec, result)
                write_text(os.path.join(run_dir, "run.log"), result["stderr"])
                write_json(os.path.join(run_dir, "metrics.json"), {"success": False, "metrics": {}})
                memory.add_run(family, run_name, run_index, spec, prompt, "", result, analysis)
                trials.append({
                    "run_index": run_index,
                    "spec": spec,
                    "success": False,
                    "metrics": {},
                    "analysis": analysis,
                    "outcome": "code_gen_failed",
                    "wall_seconds": round(time.time() - _trial_started_at, 1),
                    "error_summary": "LLM returned no code block.",
                    "repair_attempts": 0,
                    "run_dir": run_dir,
                })
                continue
            run_code = code
        result: dict[str, Any] | None = None
        attempt = 0
        while attempt <= MAX_REPAIR_ATTEMPTS:
            run_code = module.apply_light_autofixes(run_code, spec)
            issues = module.preflight_issues(run_code, spec)
            if issues:
                if attempt == MAX_REPAIR_ATTEMPTS:
                    _repair_exhausted_without_running = True
                    result = {
                        "success": False,
                        "timed_out": False,
                        "dry_run_failed": False,
                        "metrics": {},
                        "stdout": "",
                        "stderr": "Preflight validation failed:\n- " + "\n- ".join(issues),
                    }
                    break
                repair = request_surgical_repair(
                    llm=llm,
                    module=module,
                    family=family,
                    run_name=run_name,
                    submission_path=submission_path,
                    failed_code=run_code,
                    stderr_text="Preflight validation failed:\n- " + "\n- ".join(issues),
                    stdout_text="",
                    attempt=attempt + 1,
                    max_attempts=MAX_REPAIR_ATTEMPTS,
                    extra_context="Validated spec:\n" + pretty_json(spec),
                )
                write_text(os.path.join(run_dir, f"repair_attempt_{attempt + 1}.json"), repair["raw_response"])
                if not repair["code"]:
                    _repair_exhausted_without_running = True
                    result = {
                        "success": False,
                        "timed_out": False,
                        "dry_run_failed": False,
                        "metrics": {},
                        "stdout": "",
                        "stderr": "Preflight validation failed and repair returned no patch.\n" + repair["error"],
                    }
                    break
                run_code = repair["code"]
                attempt += 1
                continue

            ok, syntax_err = syntax_check(run_code)
            if not ok:
                if attempt == MAX_REPAIR_ATTEMPTS:
                    _repair_exhausted_without_running = True
                    result = {
                        "success": False,
                        "timed_out": False,
                        "dry_run_failed": False,
                        "metrics": {},
                        "stdout": "",
                        "stderr": syntax_err,
                    }
                    break
                repair = request_surgical_repair(
                    llm=llm,
                    module=module,
                    family=family,
                    run_name=run_name,
                    submission_path=submission_path,
                    failed_code=run_code,
                    stderr_text=syntax_err,
                    stdout_text="",
                    attempt=attempt + 1,
                    max_attempts=MAX_REPAIR_ATTEMPTS,
                    extra_context="Validated spec:\n" + pretty_json(spec),
                )
                write_text(os.path.join(run_dir, f"repair_attempt_{attempt + 1}.json"), repair["raw_response"])
                if not repair["code"]:
                    _repair_exhausted_without_running = True
                    result = {
                        "success": False,
                        "timed_out": False,
                        "dry_run_failed": False,
                        "metrics": {},
                        "stdout": "",
                        "stderr": syntax_err + "\n" + repair["error"],
                    }
                    break
                run_code = repair["code"]
                attempt += 1
                continue

            train_rows = phase_train_rows(phase_label)
            print("[EXECUTE] Running experiment...")
            result = run_experiment(
                run_code,
                run_name,
                train_fraction=1.0,
                train_rows=None if phase_data_dir else train_rows,
                write_submission=False,
                data_dir=phase_data_dir,
            )
            if result["success"]:
                break
            if attempt == MAX_REPAIR_ATTEMPTS:
                break

            repair = request_surgical_repair(
                llm=llm,
                module=module,
                family=family,
                run_name=run_name,
                submission_path=submission_path,
                failed_code=run_code,
                stderr_text=result["stderr"],
                stdout_text=result["stdout"],
                attempt=attempt + 1,
                max_attempts=MAX_REPAIR_ATTEMPTS,
                extra_context="Validated spec:\n" + pretty_json(spec),
            )
            write_text(os.path.join(run_dir, f"repair_attempt_{attempt + 1}.json"), repair["raw_response"])
            if not repair["code"]:
                result["stderr"] += "\nRepair error: " + repair["error"]
                break
            run_code = repair["code"]
            attempt += 1

        if result is None:
            result = {
                "success": False,
                "timed_out": False,
                "dry_run_failed": False,
                "metrics": {},
                "stdout": "",
                "stderr": "Execution failed without a result payload.",
            }

        analysis = analyze_run(llm, family, run_name, spec, result)
        write_text(os.path.join(run_dir, "train.py"), run_code)
        write_json(
            os.path.join(run_dir, "metrics.json"),
            {
                "success": result.get("success", False),
                "timed_out": result.get("timed_out", False),
                "dry_run_failed": result.get("dry_run_failed", False),
                "metrics": result.get("metrics", {}),
            },
        )
        write_text(
            os.path.join(run_dir, "run.log"),
            "\n".join(
                [
                    f"success: {result.get('success', False)}",
                    f"timed_out: {result.get('timed_out', False)}",
                    f"dry_run_failed: {result.get('dry_run_failed', False)}",
                    "",
                    "[STDOUT]",
                    result.get("stdout", ""),
                    "",
                    "[STDERR]",
                    result.get("stderr", ""),
                    "",
                    "[ANALYSIS]",
                    analysis,
                ]
            ),
        )
        if os.path.exists(submission_path):
            os.remove(submission_path)

        memory.add_run(family, run_name, run_index, spec, prompt, run_code, result, analysis)

        # Classify the trial outcome for the sweep planner.
        _wall = round(time.time() - _trial_started_at, 1)
        _outcome = classify_trial_outcome(result, repair_exhausted=_repair_exhausted_without_running)
        if _code_gen_failed_early:
            _outcome = "code_gen_failed"
        _err_tail = ""
        if not result.get("success"):
            _err_tail = tail(result.get("stderr", ""), 240).strip().splitlines()[-1] if result.get("stderr") else ""

        trial_record = {
            "run_index": run_index,
            "spec": spec,
            "success": result.get("success", False),
            "metrics": result.get("metrics", {}),
            "analysis": analysis,
            "run_dir": run_dir,
            "outcome": _outcome,
            "wall_seconds": _wall,
            "error_summary": _err_tail,
            "repair_attempts": attempt,
        }
        trials.append(trial_record)
        if result.get("success") and (best_trial is None or metric_f1(result) > metric_f1({"metrics": best_trial["metrics"]})):
            best_trial = trial_record
            best_code = run_code
            if freeze_after_success:
                if frozen_code is None:
                    frozen_code = best_code
                    print(f"[Freeze] {family} baseline locked from first successful run.")
                else:
                    frozen_code = best_code
                    print(f"[Freeze] {family} baseline refreshed from current best successful run.")
        print(f"[Result] run {run_index}/{resolved_max_runs} | success={result.get('success', False)} | metrics={result.get('metrics', {})}")
        if phase_label == "sweep" and result.get("success"):
            print(f"[Sweep] {family} produced a successful run; moving to the next family.")
            break

    session_dir = create_session_dir(session_name)
    summary = {
        "family_key": family_key,
        "family": family,
        "session_name": session_name,
        "phase": phase_label,
        "max_runs": resolved_max_runs,
        "sample_rows": phase_train_rows(phase_label),
        "validation_fraction": VALIDATION_FRACTION,
        "fixed_split": phase_split_manifest,
        "best_run_index": best_trial["run_index"] if best_trial else None,
        "best_metrics": best_trial["metrics"] if best_trial else {},
        "best_run_dir": best_trial["run_dir"] if best_trial else None,
        "trials": trials,
    }
    write_json(os.path.join(session_dir, "summary.json"), summary)
    if best_trial:
        best_train = os.path.join(best_trial["run_dir"], "train.py")
        copy_if_exists(best_train, os.path.join(session_dir, "best_train.py"))
        write_json(os.path.join(session_dir, "best_metrics.json"), best_trial["metrics"])
        print(f"[Best] family={family} | run={best_trial['run_index']} | metrics={best_trial['metrics']}")

    return summary


def main(
    model: str,
    family: str | None,
    max_runs: int | None,
    persist: bool,
    time_budget_seconds: int,
    sweep_planner_model: str = SWEEP_PLANNER_MODEL,
) -> None:
    try:
        llm = OllamaClient(model=model)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Agent_4 could not connect to Ollama. "
            "Start Ollama and ensure the requested model is available.\n"
            f"Details: {exc}"
        ) from exc
    memory = Agent4Memory(persist=persist)
    public_submissions_dir = reset_public_submissions_dir()
    phase_data_dirs: list[str] = []
    atexit.register(cleanup_phase_data_dirs, phase_data_dirs)
    sweep_data_dir, sweep_split_manifest = create_fixed_phase_data_dir(
        "sweep",
        phase_train_rows("sweep"),
        test_rows=1,
    )
    phase_data_dirs.append(sweep_data_dir)
    print(
        "[Fixed Split] sweep "
        f"sample_rows={sweep_split_manifest['sample_rows']} "
        f"train={sweep_split_manifest['expected_train_rows']} "
        f"validation={sweep_split_manifest['expected_validation_rows']}"
    )
    started_at = time.time()
    overall_deadline = started_at + time_budget_seconds
    # Sweep ends after a fixed wall-clock window (default 40 minutes), or when
    # the planner decides to stop — whichever comes first. After that we roll
    # over to the opt phase even if some families are still untried.
    sweep_deadline = min(started_at + SWEEP_DURATION_SECONDS, overall_deadline)

    # Prepare the sweep-planner LLM (a small fast model is fine; falls back
    # to deterministic round-robin if the planner can't be reached).
    sweep_planner_llm: OllamaClient | None = None
    try:
        sweep_planner_llm = OllamaClient(model=sweep_planner_model)
    except Exception as exc:  # noqa: BLE001
        print(
            "[Sweep Planner] Could not initialize planner model "
            f"'{sweep_planner_model}'. Will fall back to deterministic family order. Details: {exc}"
        )

    family_summaries: list[dict[str, Any]] = []
    runs_root = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(runs_root, exist_ok=True)
    sweep_decisions_path = os.path.join(runs_root, "sweep_decisions.jsonl")
    # Truncate at the start of every launch so the log reflects this run only.
    open(sweep_decisions_path, "w").close()

    if family:
        # --family override: skip the planner and run one trial of the named family.
        if can_start_run(family, sweep_deadline):
            family_summaries.append(
                execute_family(
                    llm,
                    memory,
                    family,
                    max_runs=max_runs if max_runs is not None else 1,
                    stop_after_ts=sweep_deadline,
                    phase_label="sweep",
                    phase_data_dir=sweep_data_dir,
                    phase_split_manifest=sweep_split_manifest,
                )
            )
    else:
        # LLM-driven sweep. One planner decision = one trial of one family.
        family_state: dict[str, FamilyState] = {
            key: FamilyState(family_key=key) for key in FAMILY_MODULES.keys()
        }
        # Track ALL prior trials of each family (not just the best). Passing
        # the full history into propose_next_spec on revisits lets the LLM
        # see the per-spec → F1 mapping and avoid re-trying losing specs.
        prior_trials_per_family: dict[str, list[dict[str, Any]]] = {}

        while time.time() < sweep_deadline:
            remaining = sweep_deadline - time.time()
            if remaining < RUN_START_BUFFER_SECONDS:
                print(f"[Sweep] Only {int(remaining)}s left in sweep window; closing sweep.")
                break

            # Build a planner caller that uses round-robin as fallback when the
            # planner LLM is unavailable.
            if sweep_planner_llm is None:
                _fallback_order = [
                    key for key in FAMILY_MODULES.keys()
                    if not family_state[key].skipped_permanently
                    and family_state[key].attempts == 0
                ]
                if not _fallback_order:
                    print("[Sweep] No eligible untried families remain (fallback); stopping sweep.")
                    break
                _picked = _fallback_order[0]
                decision = SweepDecision(
                    action="try_family",
                    family_key=_picked,
                    reason="fallback: planner LLM unavailable; using deterministic round-robin",
                    eligible_families=_fallback_order,
                    time_remaining_seconds=int(remaining),
                    timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                )
            else:
                decision = select_next_sweep_action(
                    llm=sweep_planner_llm,
                    family_state=family_state,
                    cost_estimates=FAMILY_RUN_ESTIMATES,
                    time_remaining_seconds=remaining,
                    start_buffer_seconds=RUN_START_BUFFER_SECONDS,
                    planner_system_prompt=SWEEP_PLANNER_SYSTEM,
                )

            # Append a compact line to the decision log + dump the full prompt+raw on the side.
            with open(sweep_decisions_path, "a") as fh:
                fh.write(json.dumps(decision_to_log_record(decision)) + "\n")
            if decision.prompt:
                write_text(
                    os.path.join(runs_root, f"sweep_decision_{decision.timestamp.replace(':', '')}_prompt.txt"),
                    decision.prompt,
                )
            if decision.raw_response:
                write_text(
                    os.path.join(runs_root, f"sweep_decision_{decision.timestamp.replace(':', '')}_raw.txt"),
                    decision.raw_response,
                )

            print(
                f"\n[Sweep Planner] action={decision.action} "
                f"family={decision.family_key or '-'} "
                f"reason={decision.reason}"
            )

            if decision.action == "stop":
                break
            if decision.action == "skip_family_permanently":
                if decision.family_key and decision.family_key in family_state:
                    family_state[decision.family_key].skipped_permanently = True
                continue

            # decision.action == "try_family"
            family_key = decision.family_key
            if not family_key or family_key not in FAMILY_MODULES:
                print(f"[Sweep Planner] Skipping malformed decision: {decision}")
                continue

            # Seed the trial with the FULL prior history of this family.
            # On a first attempt, this list is empty -> generate_initial_spec.
            # On a revisit, propose_next_spec sees every prior spec + F1, not
            # just the best one — so it can avoid re-proposing losing specs
            # and reason about which knob actually moved F1.
            seeded_history = prior_trials_per_family.get(family_key) or None

            summary = execute_family(
                llm,
                memory,
                family_key,
                max_runs=1,
                stop_after_ts=sweep_deadline,
                phase_label="sweep",
                seeded_trials=seeded_history,
                phase_data_dir=sweep_data_dir,
                phase_split_manifest=sweep_split_manifest,
            )
            family_summaries.append(summary)

            # Fold the latest trial into family_state and append it to the
            # per-family history for future revisits.
            actual_trials = [t for t in summary.get("trials", []) if not t.get("seeded")]
            if actual_trials:
                last = actual_trials[-1]
                family_state[family_key].update_from_trial(
                    outcome=last.get("outcome", "training_crash"),
                    f1=last.get("metrics", {}).get("f1"),
                    wall_seconds=last.get("wall_seconds", 0.0),
                    error_summary=last.get("error_summary"),
                )
                # Append to per-family history. We keep ALL trials (success
                # AND failure) so propose_next_spec can see what failed too.
                prior_trials_per_family.setdefault(family_key, []).append(last)

    successful = [summary for summary in family_summaries if summary.get("best_run_index") is not None]
    if not successful:
        print("[Overall Best] No family produced a successful run with metrics.")
        cleanup_phase_data_dirs(phase_data_dirs)
        return

    sweep_ranked = sorted(
        successful,
        key=lambda summary: metric_f1({"metrics": summary.get("best_metrics", {})}),
        reverse=True,
    )
    sweep_best = sweep_ranked[0]
    # Opt phase removed: the sweep planner now owns all exploration within
    # the 45-min sweep window. The best sweep family proceeds directly to
    # the final-submission step, which retrains it on a 5k-row sample.
    print(
        f"[Sweep complete] best family = {sweep_best['family']} "
        f"with F1={metric_f1({'metrics': sweep_best.get('best_metrics', {})}):.4f}"
    )

    best_overall = max(
        [summary for summary in family_summaries if summary.get("best_run_index") is not None],
        key=lambda summary: metric_f1({"metrics": summary.get("best_metrics", {})}),
    )
    families_attempted = sorted({summary["family_key"] for summary in family_summaries})
    overall_summary = {
        "model": model,
        "time_budget_seconds": time_budget_seconds,
        "sweep_duration_seconds": SWEEP_DURATION_SECONDS,
        "final_train_rows": FINAL_TRAIN_ROWS,
        "time_elapsed_seconds": int(time.time() - started_at),
        "families_attempted": families_attempted,
        "sweep_best_family": sweep_best["family"],
        "sweep_best_run_index": sweep_best["best_run_index"],
        "sweep_best_metrics": sweep_best["best_metrics"],
        "best_family": best_overall["family"],
        "best_run_index": best_overall["best_run_index"],
        "best_metrics": best_overall["best_metrics"],
        "family_summaries": family_summaries,
    }
    write_json(os.path.join(os.path.dirname(__file__), "runs", "overall_best.json"), overall_summary)
    public_best_submission = os.path.join(public_submissions_dir, "best_overall_submission.csv")
    final_payload, final_error = prepare_final_submission_payload(best_overall, public_best_submission)
    final_submission_success = False
    if final_payload:
        print("[Final Submission] Re-running the selected best model on full data for one test prediction file.")
        final_code, final_result, final_repair_attempts = execute_final_submission(llm, final_payload)
        overall_summary["final_submission_result"] = {
            "success": final_result.get("success", False),
            "timed_out": final_result.get("timed_out", False),
            "metrics": final_result.get("metrics", {}),
            "repair_attempts": final_repair_attempts,
        }
        final_submission_success = bool(final_result.get("success", False))
        if not final_submission_success and final_result.get("stderr"):
            overall_summary["final_submission_result"]["error"] = final_result.get("stderr", "")
        write_text(
            os.path.join(os.path.dirname(__file__), "runs", "final_submission_train.py"),
            final_code,
        )
        write_text(
            os.path.join(os.path.dirname(__file__), "runs", "final_submission.log"),
            "\n".join(
                [
                    f"success: {final_result.get('success', False)}",
                    f"timed_out: {final_result.get('timed_out', False)}",
                    f"repair_attempts: {final_repair_attempts}",
                    "",
                    "[STDOUT]",
                    final_result.get("stdout", ""),
                    "",
                    "[STDERR]",
                    final_result.get("stderr", ""),
                ]
            ),
        )
    else:
        overall_summary["final_submission_result"] = {"success": False, "error": final_error}

    if final_submission_success and os.path.exists(public_best_submission):
        overall_summary["best_submission_path"] = public_best_submission
        if auto_submit_enabled():
            print("[Kaggle] Uploading final submission and polling for score...")
            kaggle_result = submit_and_wait(
                submission_path=public_best_submission,
                default_message=build_kaggle_submission_message(best_overall),
            )
            overall_summary["kaggle_submission"] = kaggle_result
            if kaggle_result.get("submitted"):
                print(
                    "[Kaggle] "
                    f"status={kaggle_result.get('status', 'unknown')} | "
                    f"public_score={kaggle_result.get('public_score', '') or 'pending'} | "
                    f"private_score={kaggle_result.get('private_score', '') or 'pending'}"
                )
            else:
                print(f"[Kaggle] Submission failed: {kaggle_result.get('error', 'unknown error')}")
    write_json(os.path.join(os.path.dirname(__file__), "runs", "overall_best.json"), overall_summary)
    print(
        "[Overall Best] "
        f"family={best_overall['family']} | run={best_overall['best_run_index']} | "
        f"metrics={best_overall['best_metrics']}"
    )
    cleanup_phase_data_dirs(phase_data_dirs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent_4 — LLM-driven autonomous experiment runner")
    parser.add_argument("--model", type=str, default="qwen2.5-coder:14b")
    parser.add_argument("--sweep-planner-model", type=str, default=SWEEP_PLANNER_MODEL,
                        help="LLM used to pick which family to try next during the sweep.")
    parser.add_argument("--family", type=str, choices=sorted(FAMILY_MODULES.keys()),
                        help="Bypass the sweep planner and force one trial of this family.")
    parser.add_argument("--max-runs", type=int,
                        help="Cap on per-call trial count (mostly only useful with --family).")
    parser.add_argument("--time-budget-minutes", type=int, default=60)
    parser.add_argument("--fresh", action="store_true", help="Do not persist the per-run log this invocation.")
    args = parser.parse_args()
    main(
        model=args.model,
        family=args.family,
        max_runs=args.max_runs,
        persist=not args.fresh,
        time_budget_seconds=max(60, args.time_budget_minutes * 60),
        sweep_planner_model=args.sweep_planner_model,
    )
