"""Adaptive search planning for Agent_3."""

from __future__ import annotations

import hashlib
import json
import math
import os
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


def _changed_tunable_keys(module: object, old_spec: dict[str, Any], new_spec: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    for key in module.get_tunable_keys():
        if json.dumps(old_spec.get(key), sort_keys=True) != json.dumps(new_spec.get(key), sort_keys=True):
            changed.append(key)
    return changed


def _best_successful_trial(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [trial for trial in trials if trial.get("success") and _safe_f1(trial) >= 0]
    if not successful:
        return None
    return max(successful, key=_safe_f1)


def _latest_repeated_f1_trial(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [trial for trial in trials if trial.get("success") and _safe_f1(trial) >= 0]
    if len(successful) < 2:
        return None
    latest = successful[-1]
    latest_f1 = _safe_f1(latest)
    for trial in successful[:-1]:
        if abs(_safe_f1(trial) - latest_f1) < 1e-9:
            return trial
    return None


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _rounded(value: float) -> float:
    if value == 0:
        return 0.0
    if abs(value) >= 1:
        return round(value, 6)
    return float(f"{value:.8g}")


def _candidate_values(current: Any, low: float, high: float, local: bool = False) -> list[Any]:
    values: list[Any] = []
    if isinstance(current, int) and not isinstance(current, bool):
        current_i = int(current)
        low_i = int(math.ceil(low))
        high_i = int(math.floor(high))
        span = max(high_i - low_i, 1)
        step = max(1, span // (16 if local else 8))
        if local:
            deltas = [-step, step, -(2 * step), 2 * step]
        else:
            deltas = [-step, step, -(2 * step), 2 * step, -(span // 4 or 1), span // 4 or 1]
        for delta in deltas:
            candidate = min(max(current_i + delta, low_i), high_i)
            if candidate != current_i:
                values.append(candidate)
        if not local:
            for candidate in (low_i, high_i, (low_i + high_i) // 2):
                if candidate != current_i:
                    values.append(candidate)
    else:
        current_f = float(current)
        additive_step = max((high - low) / (20.0 if local else 10.0), 1e-12)
        if low >= 0 and current_f > 0:
            multipliers = (0.95, 1.05, 0.9, 1.1) if local else (0.8, 1.2, 0.67, 1.5, 0.5, 2.0)
            for multiplier in multipliers:
                candidate = _rounded(_clamp(current_f * multiplier, low, high))
                if candidate != current_f:
                    values.append(candidate)
        for delta in (-additive_step, additive_step, -(2 * additive_step), 2 * additive_step):
            candidate = _rounded(_clamp(current_f + delta, low, high))
            if candidate != current_f:
                values.append(candidate)
        if not local:
            for candidate in (_rounded(low), _rounded(high), _rounded((low + high) / 2.0)):
                if candidate != current_f:
                    values.append(candidate)

    deduped: list[Any] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _prediction_fingerprint(trial: dict[str, Any]) -> str | None:
    run_dir = trial.get("run_dir")
    if not isinstance(run_dir, str):
        return None
    path = os.path.join(run_dir, "predictions.csv")
    if not os.path.exists(path):
        return None
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stagnant_keys(module: object, trials: list[dict[str, Any]], best_trial: dict[str, Any] | None) -> set[str]:
    if best_trial is None or not best_trial.get("success"):
        return set()

    stagnant: set[str] = set()
    best_f1 = _safe_f1(best_trial)
    best_fingerprint = _prediction_fingerprint(best_trial)
    successful = [trial for trial in trials if trial.get("success") and _safe_f1(trial) >= 0]

    for trial in successful:
        if trial is best_trial:
            continue
        changed = _changed_tunable_keys(module, best_trial["spec"], trial["spec"])
        if not changed:
            continue
        same_f1 = abs(_safe_f1(trial) - best_f1) < 1e-9
        same_predictions = best_fingerprint is not None and _prediction_fingerprint(trial) == best_fingerprint
        if same_f1 or same_predictions:
            stagnant.update(changed)

    for previous, current in zip(successful, successful[1:]):
        changed = _changed_tunable_keys(module, previous["spec"], current["spec"])
        if not changed:
            continue
        same_f1 = abs(_safe_f1(previous) - _safe_f1(current)) < 1e-9
        previous_fingerprint = _prediction_fingerprint(previous)
        current_fingerprint = _prediction_fingerprint(current)
        same_predictions = previous_fingerprint is not None and previous_fingerprint == current_fingerprint
        if same_f1 or same_predictions:
            stagnant.update(changed)
    return stagnant


def _fallback_mutation(
    module: object,
    anchor_spec: dict[str, Any],
    tried_signatures: set[tuple[tuple[str, str], ...]],
    run_name: str,
    submission_path: str,
    preferred_keys: list[str] | None = None,
    local: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    ranges = module.get_spec_ranges()
    candidate = dict(anchor_spec)
    candidate["experiment_name"] = run_name
    candidate["submission_path"] = submission_path

    ordered_keys: list[str] = []
    if preferred_keys:
        ordered_keys.extend([key for key in preferred_keys if key not in ordered_keys])
    ordered_keys.extend([key for key in module.get_tunable_keys() if key not in ordered_keys and key != "val_size"])

    for key in ordered_keys:
        if key not in ranges or key not in anchor_spec:
            continue
        low, high = ranges[key]
        for value in _candidate_values(anchor_spec[key], low, high, local=local):
            mutated = dict(candidate)
            mutated[key] = value
            if _spec_signature(module, mutated) not in tried_signatures:
                return mutated, [f"Duplicate search spec avoided by mutating best spec key '{key}'."]

    return candidate, ["No unique nearby spec was found; reused the current best spec."]


def _key_change_counts(module: object, trials: list[dict[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in module.get_tunable_keys()}
    for previous, current in zip(trials, trials[1:]):
        for key in _changed_tunable_keys(module, previous["spec"], current["spec"]):
            counts[key] = counts.get(key, 0) + 1
    return counts


def _ordered_underexplored_keys(
    module: object,
    trials: list[dict[str, Any]],
    preferred_keys: list[str],
    skip_keys: set[str] | None = None,
) -> list[str]:
    counts = _key_change_counts(module, trials)
    preferred_rank = {key: idx for idx, key in enumerate(preferred_keys)}
    skip = skip_keys or set()
    keys = [key for key in module.get_tunable_keys() if key not in skip and key != "val_size"]
    return sorted(
        keys,
        key=lambda key: (
            counts.get(key, 0),
            preferred_rank.get(key, len(preferred_rank)),
            module.get_tunable_keys().index(key),
        ),
    )


def _optimization_focus_keys(module: object, preferred_keys: list[str]) -> list[str]:
    tunable = module.get_tunable_keys()
    focus_order = [
        "learning_rate",
        "dropout",
        "batch_size",
        "epochs",
        "weight_decay",
        "threshold_min",
        "threshold_max",
        "threshold_steps",
        "max_len",
        "embedding_dim",
        "channels",
        "hidden_dim",
        "hidden_size",
        "num_layers",
        "max_vocab",
    ]
    ordered: list[str] = []
    for key in focus_order:
        if key in tunable and key not in ordered:
            ordered.append(key)
    for key in preferred_keys:
        if key in tunable and key not in ordered:
            ordered.append(key)
    for key in tunable:
        if key not in ordered:
            ordered.append(key)
    return ordered


def _ensure_phase_mutation(
    module: object,
    anchor_spec: dict[str, Any],
    proposed_spec: dict[str, Any],
    tried_signatures: set[tuple[tuple[str, str], ...]],
    run_name: str,
    submission_path: str,
    trials: list[dict[str, Any]],
    preferred_keys: list[str],
    phase: str,
) -> tuple[dict[str, Any], list[str]]:
    candidate = dict(proposed_spec)
    candidate["experiment_name"] = run_name
    candidate["submission_path"] = submission_path
    changed_keys = set(_changed_tunable_keys(module, anchor_spec, candidate))
    issues: list[str] = []

    is_opt = phase == "opt"
    is_language_model = getattr(module, "FAMILY", "") in {"Transformer", "RoBERTa"}
    if is_opt:
        if len(trials) <= 4:
            target_change_count = 2 if is_language_model else 3
            max_change_count = 3 if is_language_model else 4
        else:
            target_change_count = 1
            max_change_count = 2
    elif is_language_model:
        target_change_count = 3 if len(trials) <= 3 else 2
        max_change_count = 4
    else:
        target_change_count = 4 if len(trials) <= 3 else 3
        max_change_count = 5
    ranges = module.get_spec_ranges()
    ordered_keys = (
        _optimization_focus_keys(module, preferred_keys)
        if is_opt
        else _ordered_underexplored_keys(module, trials, preferred_keys, skip_keys=changed_keys)
    )

    if is_opt and len(changed_keys) > max_change_count:
        keep_keys: list[str] = []
        for key in ordered_keys:
            if key in changed_keys and key not in keep_keys:
                keep_keys.append(key)
            if len(keep_keys) >= max_change_count:
                break
        for key in module.get_tunable_keys():
            if key in changed_keys and key not in keep_keys:
                candidate[key] = anchor_spec[key]
        changed_keys = set(_changed_tunable_keys(module, anchor_spec, candidate))
        issues.append(
            f"Optimization phase narrowed a broad move to {len(changed_keys)} local tunable changes."
        )

    if len(changed_keys) >= target_change_count and _spec_signature(module, candidate) not in tried_signatures:
        return candidate, issues

    if not is_opt:
        ordered_keys = _ordered_underexplored_keys(module, trials, preferred_keys, skip_keys=changed_keys)
    for key in ordered_keys:
        if len(changed_keys) >= target_change_count:
            break
        if key not in ranges or key not in anchor_spec:
            continue
        low, high = ranges[key]
        current_value = candidate.get(key, anchor_spec[key])
        values = _candidate_values(current_value, low, high, local=is_opt) + _candidate_values(
            anchor_spec[key], low, high, local=is_opt
        )
        for value in values:
            if value == current_value:
                continue
            trial_spec = dict(candidate)
            trial_spec[key] = value
            if json.dumps(trial_spec.get(key), sort_keys=True) == json.dumps(anchor_spec.get(key), sort_keys=True):
                continue
            candidate = trial_spec
            changed_keys = set(_changed_tunable_keys(module, anchor_spec, candidate))
            break

    if len(changed_keys) < target_change_count:
        issues.append(
            f"Search diversification could only change {len(changed_keys)} tunable keys; target was {target_change_count}."
        )
    elif len(changed_keys) > len(_changed_tunable_keys(module, anchor_spec, proposed_spec)):
        issues.append(
            f"Expanded the search move from {len(_changed_tunable_keys(module, anchor_spec, proposed_spec))} to {len(changed_keys)} tunable keys."
        )

    if _spec_signature(module, candidate) in tried_signatures:
        fallback, extra_issues = _fallback_mutation(
            module=module,
            anchor_spec=anchor_spec,
            tried_signatures=tried_signatures,
            run_name=run_name,
            submission_path=submission_path,
            preferred_keys=ordered_keys or preferred_keys,
            local=is_opt,
        )
        return fallback, issues + extra_issues

    return candidate, issues


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
    phase: str = "sweep",
) -> dict[str, Any]:
    best_trial = _best_successful_trial(trials) or max(trials, key=_safe_f1)
    default_spec = dict(best_trial["spec"]) if best_trial.get("success") else module.get_default_spec(run_name, submission_path)
    default_spec["experiment_name"] = run_name
    default_spec["submission_path"] = submission_path
    trial_summary = summarize_trials(trials)
    tried_signatures = {_spec_signature(module, trial["spec"]) for trial in trials}
    tunable_keys = list(module.get_tunable_keys())
    stagnant_keys = _stagnant_keys(module, trials, best_trial)
    active_tunable_keys = [key for key in tunable_keys if key not in stagnant_keys] or tunable_keys
    active_tunable_keys = [key for key in active_tunable_keys if key != "val_size"] or [
        key for key in tunable_keys if key != "val_size"
    ]
    latest_success = next((trial for trial in reversed(trials) if trial.get("success")), None)
    repeated_match = _latest_repeated_f1_trial(trials)
    repeated_f1 = latest_success is not None and repeated_match is not None
    stale_changed_keys = []
    if latest_success is not None and repeated_match is not None:
        stale_changed_keys = _changed_tunable_keys(module, repeated_match["spec"], latest_success["spec"])
    phase_rules = (
        "- this is the top-architecture optimization phase, so stay near this architecture's current best spec\n"
        "- usually change 2 to 4 tunable keys because the sweep used the smaller 4k labeled sample\n"
        "- prefer coordinated local changes such as learning_rate, dropout, batch_size, epochs, weight_decay, sequence length, and threshold settings\n"
        "- validation size is controlled by the runner and should remain at 0.2\n"
        "- avoid large capacity jumps unless the history strongly suggests they help\n"
    ) if phase == "opt" else (
        "- this is the family sweep phase, so explore different regions of the parameter space\n"
        "- the runner uses a 4k labeled sample split 80/20 for training/validation\n"
        "- usually change 2 to 4 tunable keys in one coordinated move\n"
        "- with a 5-run budget, cover both model-capacity keys and optimization keys instead of nudging only one key repeatedly\n"
        "- prefer combinations such as sequence/model size + regularization + optimization, for example max_len/channels with learning_rate/dropout/batch_size/epochs\n"
    )
    prompt = (
        f"Propose the next {module.FAMILY} experiment spec.\n\n"
        f"{module.get_search_prompt()}\n\n"
        "Rules:\n"
        "- keep the architecture family fixed\n"
        "- keep the same overall prompt contract and pipeline shape\n"
        f"- vary only these tunable keys: {', '.join(tunable_keys)}\n"
        "- optimize against the best successful session trial, not the last run\n"
        "- keep the best successful spec as the default anchor and mutate it only slightly\n"
        f"{phase_rules}"
        "- do not repeat an exact spec already tried in this session\n"
        "- if a run timed out, reduce cost\n"
        "- if a run crashed, simplify the risky parameter region\n"
        "- if repeated runs get the same F1, switch to different tunable keys instead of repeating weak ones\n"
        "- return one JSON object only\n\n"
        f"Dataset context:\n{data_context}\n\n"
        f"Family history:\n{history_summary}\n\n"
        f"Session trials so far:\n{trial_summary}\n\n"
        f"Best session trial so far:\n{json.dumps(best_trial, indent=2)}\n\n"
        f"Stagnant keys from equal-F1 or same-prediction runs: {', '.join(sorted(stagnant_keys)) if stagnant_keys else 'none'}\n"
        f"Preferred keys for the next move: {', '.join(active_tunable_keys)}\n"
        f"Latest equal-F1 matched prior run: {repeated_match['run_index'] if repeated_match else 'none'}\n"
        f"Latest equal-F1 changed keys to avoid repeating: {', '.join(stale_changed_keys) if stale_changed_keys else 'none'}\n"
        f"Repeated best F1 detected: {'yes' if repeated_f1 else 'no'}\n\n"
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
    if "val_size" in spec:
        spec["val_size"] = default_spec.get("val_size", spec["val_size"])
    spec, diversity_issues = _ensure_phase_mutation(
        module=module,
        anchor_spec=best_trial["spec"],
        proposed_spec=spec,
        tried_signatures=tried_signatures,
        run_name=run_name,
        submission_path=submission_path,
        trials=trials,
        preferred_keys=active_tunable_keys,
        phase=phase,
    )
    issues.extend(diversity_issues)
    changed_after_validation = _changed_tunable_keys(module, best_trial["spec"], spec)
    only_stagnant_changes = bool(changed_after_validation) and all(key in stagnant_keys for key in changed_after_validation)
    if repeated_f1 and active_tunable_keys and (not changed_after_validation or only_stagnant_changes):
        spec, extra_issues = _fallback_mutation(
            module=module,
            anchor_spec=best_trial["spec"],
            tried_signatures=tried_signatures,
            run_name=run_name,
            submission_path=submission_path,
            preferred_keys=active_tunable_keys,
            local=(phase == "opt"),
        )
        issues.extend(["Repeated F1 triggered a forced move to different tunable keys."])
        issues.extend(extra_issues)
        spec, diversity_issues = _ensure_phase_mutation(
            module=module,
            anchor_spec=best_trial["spec"],
            proposed_spec=spec,
            tried_signatures=tried_signatures,
            run_name=run_name,
            submission_path=submission_path,
            trials=trials,
            preferred_keys=active_tunable_keys,
            phase=phase,
        )
        issues.extend(diversity_issues)
    if _spec_signature(module, spec) in tried_signatures:
        spec, extra_issues = _fallback_mutation(
            module=module,
            anchor_spec=best_trial["spec"],
            tried_signatures=tried_signatures,
            run_name=run_name,
            submission_path=submission_path,
            preferred_keys=active_tunable_keys,
            local=(phase == "opt"),
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
