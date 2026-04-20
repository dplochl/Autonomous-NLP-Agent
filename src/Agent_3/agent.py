"""Agent_3 — prompt-first autonomous experiment runner."""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import time
from datetime import datetime
from typing import Any

import pandas as pd

from artifacts import copy_if_exists, create_session_dir, create_run_dir, write_json, write_text
from generate_spec import generate_initial_spec
from json_utils import pretty_json
from llm import OllamaClient
from memory import Agent3Memory
from prompts import ANALYSIS_PROMPT_TEMPLATE, DATA_CONTEXT_TEMPLATE, FULL_SYSTEM
from render_templates import render_family_prompt
from repair import request_surgical_repair
from sandbox import run_experiment, tail
from search import propose_next_spec, summarize_trials

import families.experiment_bow as exp_bow
import families.experiment_bow_advanced as exp_bow_advanced
import families.experiment_cnn as exp_cnn
import families.experiment_lstm as exp_lstm
import families.experiment_transformer as exp_transformer


DATA_DIR_ENV = "DISASTER_AGENT_DATA_DIR"
DEFAULT_DATA_DIR = "data"
MAX_SEARCH_RUNS = int(os.environ.get("AGENT3_MAX_RUNS", "4"))
MAX_REPAIR_ATTEMPTS = int(os.environ.get("DISASTER_AGENT_MAX_REPAIRS", "8"))
TOTAL_TIME_BUDGET_SECONDS = int(os.environ.get("AGENT3_TOTAL_TIME_BUDGET_SECONDS", str(80 * 60)))
SWEEP_BUDGET_FRACTION = float(os.environ.get("AGENT3_SWEEP_BUDGET_FRACTION", "0.65"))
WINNER_OPTIMIZATION_MAX_RUNS = int(os.environ.get("AGENT3_WINNER_OPTIMIZATION_MAX_RUNS", "6"))
RUN_START_BUFFER_SECONDS = int(os.environ.get("AGENT3_RUN_START_BUFFER_SECONDS", "120"))
# Measured one-run wall times on 2026-04-20, plus an extra ~120s cushion
# for up to five repair attempts at roughly 24s each.
FAMILY_RUN_ESTIMATES = {
    "bow": 195,
    "bow_advanced": 255,
    "cnn": 330,
    "lstm": 345,
    "transformer": 690,
}

FAMILY_MODULES = {
     "transformer": exp_transformer,
    "bow_advanced": exp_bow_advanced,
    "cnn": exp_cnn,
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


def execute_family(
    llm: OllamaClient,
    memory: Agent3Memory,
    family_key: str,
    max_runs: int | None,
    stop_after_ts: float | None = None,
    phase_label: str = "sweep",
    seeded_trial: dict[str, Any] | None = None,
    seeded_code: str | None = None,
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

            print("[EXECUTE] Running experiment...")
            result = run_experiment(run_code, run_name)
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
        copy_if_exists(submission_path, os.path.join(run_dir, "predictions.csv"))

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

    session_dir = create_session_dir(session_name)
    summary = {
        "family_key": family_key,
        "family": family,
        "session_name": session_name,
        "phase": phase_label,
        "max_runs": resolved_max_runs,
        "best_run_index": best_trial["run_index"] if best_trial else None,
        "best_metrics": best_trial["metrics"] if best_trial else {},
        "best_run_dir": best_trial["run_dir"] if best_trial else None,
        "trials": trials,
    }
    write_json(os.path.join(session_dir, "summary.json"), summary)
    if best_trial:
        best_train = os.path.join(best_trial["run_dir"], "train.py")
        best_predictions = os.path.join(best_trial["run_dir"], "predictions.csv")
        copy_if_exists(best_train, os.path.join(session_dir, "best_train.py"))
        copy_if_exists(best_predictions, os.path.join(session_dir, "best_predictions.csv"))
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
    memory = Agent3Memory(persist=persist)
    public_submissions_dir = reset_public_submissions_dir()
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
            )
        )

    successful = [summary for summary in family_summaries if summary.get("best_run_index") is not None]
    if not successful:
        print("[Overall Best] No family produced a successful run with metrics.")
        return

    sweep_best = max(
        successful,
        key=lambda summary: metric_f1({"metrics": summary.get("best_metrics", {})}),
    )

    if optimize_winner and can_start_run(sweep_best["family_key"], overall_deadline):
        seeded_trial = best_trial_from_summary(sweep_best)
        seeded_code = load_text_if_exists(os.path.join(os.path.dirname(__file__), "runs", sweep_best["session_name"], "best_train.py"))
        if seeded_trial and seeded_code:
            print(
                f"[Optimize] Starting winner-only optimization for family={sweep_best['family']} "
                f"with {remaining_seconds(overall_deadline)}s remaining."
            )
            family_summaries.append(
                execute_family(
                    llm,
                    memory,
                    sweep_best["family_key"],
                    max_runs=WINNER_OPTIMIZATION_MAX_RUNS,
                    stop_after_ts=overall_deadline,
                    phase_label="opt",
                    seeded_trial=seeded_trial,
                    seeded_code=seeded_code,
                )
            )
        else:
            print("[Optimize] Skipped winner optimization because the best run code/spec could not be loaded.")
    elif optimize_winner:
        print(f"[Optimize] Skipped winner optimization because only {remaining_seconds(overall_deadline)}s remain.")

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
        "best_family": best_overall["family"],
        "best_run_index": best_overall["best_run_index"],
        "best_metrics": best_overall["best_metrics"],
        "family_summaries": family_summaries,
    }
    write_json(os.path.join(os.path.dirname(__file__), "runs", "overall_best.json"), overall_summary)
    best_session_dir = os.path.join(os.path.dirname(__file__), "runs", best_overall["session_name"])
    best_predictions = os.path.join(best_session_dir, "best_predictions.csv")
    public_best_submission = os.path.join(public_submissions_dir, "best_overall_submission.csv")
    if copy_if_exists(best_predictions, public_best_submission):
        overall_summary["best_submission_path"] = public_best_submission
        write_json(os.path.join(os.path.dirname(__file__), "runs", "overall_best.json"), overall_summary)
    print(
        "[Overall Best] "
        f"family={best_overall['family']} | run={best_overall['best_run_index']} | "
        f"metrics={best_overall['best_metrics']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent_3 prompt-first autonomous experiment runner")
    parser.add_argument("--model", type=str, default="qwen2.5-coder:14b")
    parser.add_argument("--family", type=str, choices=sorted(FAMILY_MODULES.keys()))
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--time-budget-minutes", type=int, default=80)
    parser.add_argument("--no-winner-optimization", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Do not read or write agent3_log.json")
    args = parser.parse_args()
    main(
        model=args.model,
        family=args.family,
        max_runs=args.max_runs,
        persist=not args.fresh,
        time_budget_seconds=max(60, args.time_budget_minutes * 60),
        optimize_winner=not args.no_winner_optimization,
    )
