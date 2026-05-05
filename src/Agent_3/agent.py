"""Agent_3 — prompt-first autonomous experiment runner."""

from __future__ import annotations

import argparse
import ast
import atexit
import os
import re
import shutil
import tempfile
import time
from datetime import datetime
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from artifacts import copy_if_exists, create_session_dir, create_run_dir, write_json, write_text
from generate_spec import generate_initial_spec
from json_utils import pretty_json
from kaggle_submit import auto_submit_enabled, submit_and_wait
from llm import OllamaClient
from memory import Agent3Memory
from prompts import ANALYSIS_PROMPT_TEMPLATE, DATA_CONTEXT_TEMPLATE, FULL_SYSTEM
from render_templates import render_family_prompt
from repair import request_surgical_repair
from sandbox import run_experiment, tail
from search import propose_next_spec, summarize_trials

import families.experiment_bow as exp_bow
import families.experiment_bow_advanced as exp_bow_advanced
import families.experiment_bertweet as exp_bertweet
import families.experiment_cnn as exp_cnn
import families.experiment_embedding_dl as exp_embedding_dl
import families.experiment_lstm as exp_lstm
import families.experiment_roberta as exp_roberta


DATA_DIR_ENV = "DISASTER_AGENT_DATA_DIR"
DEFAULT_DATA_DIR = "data"
MAX_SEARCH_RUNS = int(os.environ.get("AGENT3_MAX_RUNS", "4"))
MAX_REPAIR_ATTEMPTS = int(os.environ.get("DISASTER_AGENT_MAX_REPAIRS", "8"))
FINAL_SUBMISSION_MAX_REPAIRS = int(os.environ.get("AGENT3_FINAL_SUBMISSION_MAX_REPAIRS", "6"))
TOTAL_TIME_BUDGET_SECONDS = int(os.environ.get("AGENT3_TOTAL_TIME_BUDGET_SECONDS", str(60 * 60)))
SWEEP_BUDGET_FRACTION = float(os.environ.get("AGENT3_SWEEP_BUDGET_FRACTION", "0.65"))
SWEEP_SAMPLE_ROWS = int(os.environ.get("AGENT3_SWEEP_SAMPLE_ROWS", "2000"))
FINAL_TRAIN_ROWS = int(os.environ.get("AGENT3_FINAL_TRAIN_ROWS", "10000"))
VALIDATION_FRACTION = min(max(float(os.environ.get("AGENT3_VALIDATION_FRACTION", "0.2")), 0.05), 0.5)
TOP_ARCHITECTURES_TO_OPTIMIZE = int(os.environ.get("AGENT3_TOP_ARCHITECTURES_TO_OPTIMIZE", "2"))
WINNER_OPTIMIZATION_MAX_RUNS = int(os.environ.get("AGENT3_WINNER_OPTIMIZATION_MAX_RUNS", "20"))
RUN_START_BUFFER_SECONDS = int(os.environ.get("AGENT3_RUN_START_BUFFER_SECONDS", "120"))
# Measured one-run wall times on 2026-04-20, plus an extra ~120s cushion
# for up to five repair attempts at roughly 24s each.
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
        "Agent_3 auto-submit | "
        f"family={best_overall.get('family', 'unknown')} | "
        f"run={best_overall.get('best_run_index', 'NA')} | "
        f"f1={f1} | "
        f"ts={timestamp}"
    )


def force_submission_path(code: str, old_path: str | None, new_path: str) -> str:
    fixed = code
    if old_path:
        fixed = fixed.replace(str(old_path), new_path)
    fixed = re.sub(
        r"os\.environ\.get\((['\"])AGENT_SUBMISSION_PATH\1,\s*(['\"])[^'\"]*submission\.csv\2\)",
        f"os.environ.get('DISASTER_AGENT_SUBMISSION_PATH', {new_path!r})",
        fixed,
    )
    fixed = re.sub(
        r"os\.environ\.get\((['\"])DISASTER_AGENT_SUBMISSION_PATH\1,\s*(['\"])[^'\"]*submission\.csv\2\)",
        f"os.environ.get('DISASTER_AGENT_SUBMISSION_PATH', {new_path!r})",
        fixed,
    )
    fixed = re.sub(
        r"submission_path\s*=\s*(['\"])[^'\"]*submission[^'\"]*\.csv\1",
        f"submission_path = os.environ.get('DISASTER_AGENT_SUBMISSION_PATH', {new_path!r})",
        fixed,
    )
    fixed = re.sub(
        r"((?:['\"])submission_path(?:['\"])\s*:\s*)(['\"])[^'\"]*submission[^'\"]*\.csv\2",
        rf"\g<1>os.environ.get('DISASTER_AGENT_SUBMISSION_PATH', {new_path!r})",
        fixed,
    )
    fixed = re.sub(
        r"(['\"])submissions/[^'\"]+_submission\.csv\1",
        f"os.environ.get('DISASTER_AGENT_SUBMISSION_PATH', {new_path!r})",
        fixed,
    )
    return fixed


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
    old_submission_path = spec.get("submission_path")
    spec["submission_path"] = public_submission_path
    spec["experiment_name"] = "best_overall_submission"
    spec["val_size"] = VALIDATION_FRACTION
    if hasattr(module, "normalize_spec"):
        spec = module.normalize_spec(spec)
    if hasattr(module, "tune_frozen_code"):
        code = module.tune_frozen_code(code, spec, "best_overall_submission")
    if hasattr(module, "apply_light_autofixes"):
        code = module.apply_light_autofixes(code, spec)
    return {
        "code": force_submission_path(code, old_submission_path if old_submission_path is not None else None, public_submission_path),
        "family": summary["family"],
        "family_key": summary["family_key"],
        "module": module,
        "run_name": "best_overall_submission",
        "session_dir": session_dir,
        "spec": spec,
        "submission_path": public_submission_path,
    }, None


def execute_final_submission_with_repairs(
    llm: OllamaClient,
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any], int]:
    code = str(payload["code"])
    family = str(payload["family"])
    module = payload["module"]
    run_name = str(payload["run_name"])
    spec = dict(payload["spec"])
    submission_path = str(payload["submission_path"])
    runs_root = os.path.join(os.path.dirname(__file__), "runs")
    attempt = 0
    result: dict[str, Any] | None = None

    while attempt <= FINAL_SUBMISSION_MAX_REPAIRS:
        if hasattr(module, "apply_light_autofixes"):
            code = module.apply_light_autofixes(code, spec)
        issues = module.preflight_issues(code, spec)
        if issues:
            if attempt == FINAL_SUBMISSION_MAX_REPAIRS:
                result = {
                    "success": False,
                    "timed_out": False,
                    "dry_run_failed": False,
                    "metrics": {},
                    "stdout": "",
                    "stderr": "Final submission preflight validation failed:\n- " + "\n- ".join(issues),
                }
                break
            repair = request_surgical_repair(
                llm=llm,
                module=module,
                family=family,
                run_name=run_name,
                submission_path=submission_path,
                failed_code=code,
                stderr_text="Final submission preflight validation failed:\n- " + "\n- ".join(issues),
                stdout_text="",
                attempt=attempt + 1,
                max_attempts=FINAL_SUBMISSION_MAX_REPAIRS,
                extra_context=(
                    "Validated spec:\n"
                    + pretty_json(spec)
                    + "\n\nThis is final submission mode. The script must create the submission CSV, "
                    "use the final full-data retraining path when AGENT_FINAL_SUBMISSION=1, and still print METRICS."
                ),
            )
            write_text(os.path.join(runs_root, f"final_submission_repair_attempt_{attempt + 1}.json"), repair["raw_response"])
            if not repair["code"]:
                result = {
                    "success": False,
                    "timed_out": False,
                    "dry_run_failed": False,
                    "metrics": {},
                    "stdout": "",
                    "stderr": "Final submission preflight failed and repair returned no patch.\n" + repair["error"],
                }
                break
            code = repair["code"]
            attempt += 1
            continue

        ok, syntax_err = syntax_check(code)
        if not ok:
            if attempt == FINAL_SUBMISSION_MAX_REPAIRS:
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
                failed_code=code,
                stderr_text=syntax_err,
                stdout_text="",
                attempt=attempt + 1,
                max_attempts=FINAL_SUBMISSION_MAX_REPAIRS,
                extra_context=(
                    "Validated spec:\n"
                    + pretty_json(spec)
                    + "\n\nThis is final submission mode. The script must create the submission CSV, "
                    "use the final full-data retraining path when AGENT_FINAL_SUBMISSION=1, and still print METRICS."
                ),
            )
            write_text(os.path.join(runs_root, f"final_submission_repair_attempt_{attempt + 1}.json"), repair["raw_response"])
            if not repair["code"]:
                result = {
                    "success": False,
                    "timed_out": False,
                    "dry_run_failed": False,
                    "metrics": {},
                    "stdout": "",
                    "stderr": syntax_err + "\n" + repair["error"],
                }
                break
            code = repair["code"]
            attempt += 1
            continue

        result = run_experiment(
            code,
            run_name,
            train_fraction=1.0,
            train_rows=FINAL_TRAIN_ROWS,
            write_submission=True,
            final_submission=True,
            submission_path=submission_path,
        )
        submission_error = validate_submission_file(submission_path) if result.get("success") else None
        if result.get("success") and submission_error is None:
            break
        if attempt == FINAL_SUBMISSION_MAX_REPAIRS:
            if result.get("success") and submission_error:
                result["success"] = False
                result["stderr"] = (result.get("stderr", "") + "\n" + submission_error).strip()
            break
        repair_stderr = result.get("stderr", "")
        if result.get("success") and submission_error:
            repair_stderr = (repair_stderr + "\n" + submission_error).strip()
            result["success"] = False
        repair = request_surgical_repair(
            llm=llm,
            module=module,
            family=family,
            run_name=run_name,
            submission_path=submission_path,
            failed_code=code,
            stderr_text=repair_stderr,
            stdout_text=result.get("stdout", ""),
            attempt=attempt + 1,
            max_attempts=FINAL_SUBMISSION_MAX_REPAIRS,
            extra_context=(
                "Validated spec:\n"
                + pretty_json(spec)
                + "\n\nThis is final submission mode. The script must create the submission CSV, "
                "use the final full-data retraining path when AGENT_FINAL_SUBMISSION=1, and still print METRICS."
            ),
        )
        write_text(os.path.join(runs_root, f"final_submission_repair_attempt_{attempt + 1}.json"), repair["raw_response"])
        if not repair["code"]:
            result["stderr"] = (result.get("stderr", "") + "\nRepair error: " + repair["error"]).strip()
            break
        code = repair["code"]
        attempt += 1

    if result is None:
        result = {
            "success": False,
            "timed_out": False,
            "dry_run_failed": False,
            "metrics": {},
            "stdout": "",
            "stderr": "Final submission failed without a result payload.",
        }
    return code, result, attempt


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

    data_dir = tempfile.mkdtemp(prefix=f"agent3_{phase_label}_fixed_data_")
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
    memory: Agent3Memory,
    family_key: str,
    max_runs: int | None,
    stop_after_ts: float | None = None,
    phase_label: str = "sweep",
    seeded_trial: dict[str, Any] | None = None,
    seeded_code: str | None = None,
    phase_data_dir: str | None = None,
    phase_split_manifest: dict[str, Any] | None = None,
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
    first_sweep_success_run: int | None = None
    freeze_after_success = bool(getattr(module, "freeze_after_first_success", lambda: False)())
    if seeded_trial:
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
    print(f"AGENT_3 FAMILY RUN | {family} | phase={phase_label} | session={session_name}")
    print("=" * 72)

    for run_index in range(1, resolved_max_runs + 1):
        if not can_start_run(family_key, stop_after_ts):
            print(
                f"[Budget] Skipping additional {family} runs in phase '{phase_label}' "
                f"because only {remaining_seconds(stop_after_ts)}s remain."
            )
            break
        run_dir = create_run_dir(session_name, run_index)
        run_name = f"{session_name}_run_{run_index:02d}"
        submission_path = os.path.join(run_dir, "submission.csv")

        if run_index == 1 and seeded_trial is None:
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
                trials.append({"run_index": run_index, "spec": spec, "success": False, "metrics": {}, "analysis": analysis})
                continue
            run_code = code
        result: dict[str, Any] | None = None
        attempt = 0
        while attempt <= MAX_REPAIR_ATTEMPTS:
            run_code = module.apply_light_autofixes(run_code, spec)
            issues = module.preflight_issues(run_code, spec)
            if issues:
                if attempt == MAX_REPAIR_ATTEMPTS:
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
        trial_record = {
            "run_index": run_index,
            "spec": spec,
            "success": result.get("success", False),
            "metrics": result.get("metrics", {}),
            "analysis": analysis,
            "run_dir": run_dir,
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
            if first_sweep_success_run is None:
                first_sweep_success_run = run_index
            elif run_index > first_sweep_success_run:
                print(
                    f"[Sweep] {family} produced a success plus one follow-up run; "
                    "moving to the next family."
                )
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
    optimize_winner: bool,
) -> None:
    try:
        llm = OllamaClient(model=model)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Agent_3 could not connect to Ollama. "
            "Start Ollama and ensure the requested model is available.\n"
            f"Details: {exc}"
        ) from exc
    memory = Agent3Memory(persist=persist, load_existing=False)
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
    sweep_deadline = started_at + int(time_budget_seconds * SWEEP_BUDGET_FRACTION) if optimize_winner else overall_deadline
    families = [family] if family else list(FAMILY_MODULES.keys())
    family_summaries: list[dict[str, Any]] = []
    for family_key in families:
        phase_deadline = sweep_deadline if optimize_winner else overall_deadline
        if not can_start_run(family_key, phase_deadline):
            print(
                f"[Budget] Skipping family '{family_key}' in sweep because only "
                f"{remaining_seconds(phase_deadline)}s remain."
            )
            continue
        family_summaries.append(
            execute_family(
                llm,
                memory,
                family_key,
                max_runs=max_runs,
                stop_after_ts=phase_deadline,
                phase_label="sweep",
                phase_data_dir=sweep_data_dir,
                phase_split_manifest=sweep_split_manifest,
            )
        )

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
    top_sweep_summaries = sweep_ranked[:max(1, TOP_ARCHITECTURES_TO_OPTIMIZE)]

    if optimize_winner:
        opt_data_dir, opt_split_manifest = create_fixed_phase_data_dir(
            "opt",
            phase_train_rows("opt"),
            test_rows=1,
        )
        phase_data_dirs.append(opt_data_dir)
        print(
            "[Fixed Split] opt "
            f"sample_rows={opt_split_manifest['sample_rows']} "
            f"train={opt_split_manifest['expected_train_rows']} "
            f"validation={opt_split_manifest['expected_validation_rows']}"
        )
        print(
            "[Optimize] Top sweep architectures: "
            + ", ".join(
                f"{summary['family']} f1={metric_f1({'metrics': summary.get('best_metrics', {})}):.4f}"
                for summary in top_sweep_summaries
            )
        )
        for idx, selected in enumerate(top_sweep_summaries):
            slots_left = len(top_sweep_summaries) - idx
            phase_deadline = time.time() + max(0, remaining_seconds(overall_deadline) // max(slots_left, 1))
            phase_deadline = min(phase_deadline, overall_deadline)
            if not can_start_run(selected["family_key"], phase_deadline):
                print(
                    f"[Optimize] Skipping family={selected['family']} because only "
                    f"{remaining_seconds(phase_deadline)}s remain in its allocation."
                )
                continue
            seeded_trial = best_trial_from_summary(selected)
            seeded_code = load_text_if_exists(os.path.join(os.path.dirname(__file__), "runs", selected["session_name"], "best_train.py"))
            if seeded_trial and seeded_code:
                print(
                    f"[Optimize] Tuning top architecture {idx + 1}/{len(top_sweep_summaries)}: "
                    f"family={selected['family']} with {remaining_seconds(phase_deadline)}s allocated."
                )
                family_summaries.append(
                    execute_family(
                        llm,
                        memory,
                        selected["family_key"],
                        max_runs=WINNER_OPTIMIZATION_MAX_RUNS,
                        stop_after_ts=phase_deadline,
                        phase_label="opt",
                        seeded_trial=seeded_trial,
                        seeded_code=seeded_code,
                        phase_data_dir=opt_data_dir,
                        phase_split_manifest=opt_split_manifest,
                    )
                )
            else:
                print(f"[Optimize] Skipped family={selected['family']} because its best run code/spec could not be loaded.")

    best_overall = max(
        [summary for summary in family_summaries if summary.get("best_run_index") is not None],
        key=lambda summary: metric_f1({"metrics": summary.get("best_metrics", {})}),
    )
    overall_summary = {
        "model": model,
        "time_budget_seconds": time_budget_seconds,
        "time_elapsed_seconds": int(time.time() - started_at),
        "families_run": families,
        "sweep_best_family": sweep_best["family"],
        "sweep_best_run_index": sweep_best["best_run_index"],
        "sweep_best_metrics": sweep_best["best_metrics"],
        "optimized_families": [summary["family"] for summary in top_sweep_summaries] if optimize_winner else [],
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
        final_code, final_result, final_repair_attempts = execute_final_submission_with_repairs(llm, final_payload)
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
    parser = argparse.ArgumentParser(description="Agent_3 prompt-first autonomous experiment runner")
    parser.add_argument("--model", type=str, default="qwen2.5-coder:14b")
    parser.add_argument("--family", type=str, choices=sorted(FAMILY_MODULES.keys()))
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--time-budget-minutes", type=int, default=60)
    parser.add_argument("--no-winner-optimization", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Do not write agent3_log.json for this invocation")
    args = parser.parse_args()
    main(
        model=args.model,
        family=args.family,
        max_runs=args.max_runs,
        persist=not args.fresh,
        time_budget_seconds=max(60, args.time_budget_minutes * 60),
        optimize_winner=not args.no_winner_optimization,
    )
