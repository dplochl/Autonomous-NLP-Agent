"""Adaptive search planning for Agent_3."""

from __future__ import annotations

import json
import math
from typing import Any

from json_utils import extract_json_object
from prompts import SEARCH_SYSTEM
from validate_spec import validate_spec


def _safe_f1(trial: dict[str, Any]) -> float:
    try:
        return float((trial.get("metrics") or {}).get("f1", -1.0))
    except (TypeError, ValueError):
        return -1.0


def _spec_signature(module: object, spec: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple((key, json.dumps(spec.get(key), sort_keys=True)) for key in module.get_tunable_keys())


def _best_successful_trial(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [trial for trial in trials if trial.get("success") and _safe_f1(trial) >= 0]
    if not successful:
        return None
    return max(successful, key=_safe_f1)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _rounded(value: float) -> float:
    if value == 0:
        return 0.0
    if abs(value) >= 1:
        return round(value, 6)
    return float(f"{value:.8g}")


def _candidate_values(current: Any, low: float, high: float) -> list[Any]:
    values: list[Any] = []
    if isinstance(current, int) and not isinstance(current, bool):
        current_i = int(current)
        low_i = int(math.ceil(low))
        high_i = int(math.floor(high))
        span = max(high_i - low_i, 1)
        step = max(1, span // 8)
        deltas = [-step, step, -(2 * step), 2 * step, -(span // 4 or 1), span // 4 or 1]
        for delta in deltas:
            candidate = min(max(current_i + delta, low_i), high_i)
            if candidate != current_i:
                values.append(candidate)
        for candidate in (low_i, high_i, (low_i + high_i) // 2):
            if candidate != current_i:
                values.append(candidate)
    else:
        current_f = float(current)
        additive_step = max((high - low) / 10.0, 1e-12)
        if low >= 0 and current_f > 0:
            for multiplier in (0.8, 1.2, 0.67, 1.5, 0.5, 2.0):
                candidate = _rounded(_clamp(current_f * multiplier, low, high))
                if candidate != current_f:
                    values.append(candidate)
        for delta in (-additive_step, additive_step, -(2 * additive_step), 2 * additive_step):
            candidate = _rounded(_clamp(current_f + delta, low, high))
            if candidate != current_f:
                values.append(candidate)
        for candidate in (_rounded(low), _rounded(high), _rounded((low + high) / 2.0)):
            if candidate != current_f:
                values.append(candidate)

    deduped: list[Any] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _fallback_mutation(
    module: object,
    anchor_spec: dict[str, Any],
    tried_signatures: set[tuple[tuple[str, str], ...]],
    run_name: str,
    submission_path: str,
) -> tuple[dict[str, Any], list[str]]:
    ranges = module.get_spec_ranges()
    candidate = dict(anchor_spec)
    candidate["experiment_name"] = run_name
    candidate["submission_path"] = submission_path

    for key in module.get_tunable_keys():
        if key not in ranges or key not in anchor_spec:
            continue
        low, high = ranges[key]
        for value in _candidate_values(anchor_spec[key], low, high):
            mutated = dict(candidate)
            mutated[key] = value
            if _spec_signature(module, mutated) not in tried_signatures:
                return mutated, [f"Duplicate search spec avoided by mutating best spec key '{key}'."]

    return candidate, ["No unique nearby spec was found; reused the current best spec."]


def summarize_trials(trials: list[dict[str, Any]]) -> str:
    if not trials:
        return "No previous runs in this session."
    lines = []
    for trial in trials:
        metrics = trial.get("metrics") or {}
        f1 = metrics.get("f1", "N/A")
        status = "OK" if trial.get("success") else "FAILED"
        lines.append(
            f"- run {trial['run_index']}: status={status}, f1={f1}, spec={json.dumps(trial['spec'], sort_keys=True)}"
        )
    return "\n".join(lines)


def propose_next_spec(
    llm,
    module: object,
    run_name: str,
    submission_path: str,
    data_context: str,
    history_summary: str,
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    best_trial = _best_successful_trial(trials) or max(trials, key=_safe_f1)
    default_spec = dict(best_trial["spec"]) if best_trial.get("success") else module.get_default_spec(run_name, submission_path)
    default_spec["experiment_name"] = run_name
    default_spec["submission_path"] = submission_path
    trial_summary = summarize_trials(trials)
    tried_signatures = {_spec_signature(module, trial["spec"]) for trial in trials}
    prompt = (
        f"Propose the next {module.FAMILY} experiment spec.\n\n"
        f"{module.get_search_prompt()}\n\n"
        "Rules:\n"
        "- keep the architecture family fixed\n"
        "- keep the same overall prompt contract and pipeline shape\n"
        f"- vary only these tunable keys: {', '.join(module.get_tunable_keys())}\n"
        "- optimize against the best successful session trial, not the last run\n"
        "- keep the best successful spec as the default anchor and mutate it only slightly\n"
        "- change at most 2 tunable keys and prefer changing only 1 key\n"
        "- do not repeat an exact spec already tried in this session\n"
        "- if a run timed out, reduce cost\n"
        "- if a run crashed, simplify the risky parameter region\n"
        "- return one JSON object only\n\n"
        f"Dataset context:\n{data_context}\n\n"
        f"Family history:\n{history_summary}\n\n"
        f"Session trials so far:\n{trial_summary}\n\n"
        f"Best session trial so far:\n{json.dumps(best_trial, indent=2)}\n\n"
        "Default if unsure:\n"
        f"{json.dumps(default_spec, indent=2)}\n"
    )

    raw_response = llm.respond(SEARCH_SYSTEM, prompt)
    raw_spec = extract_json_object(raw_response)
    spec, issues = validate_spec(
        raw_spec=raw_spec,
        default_spec=default_spec,
        ranges=module.get_spec_ranges(),
        fixed_keys=module.get_fixed_spec_keys(),
    )
    if _spec_signature(module, spec) in tried_signatures:
        spec, extra_issues = _fallback_mutation(
            module=module,
            anchor_spec=best_trial["spec"],
            tried_signatures=tried_signatures,
            run_name=run_name,
            submission_path=submission_path,
        )
        issues.extend(extra_issues)
    return {
        "prompt": prompt,
        "raw_response": raw_response,
        "raw_spec": raw_spec,
        "spec": spec,
        "issues": issues,
        "used_default": raw_spec is None,
    }
