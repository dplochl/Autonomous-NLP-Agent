"""Adaptive search planning for Agent_4."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from typing import Any

from generate_spec import _extract_hypothesis, _format_prior_launch_trials
from json_utils import extract_json_object
from prompts import SEARCH_SYSTEM
from validate_spec import validate_spec


# When the family has had this many successful trials with F1 stuck within
# `_DIVERSIFY_TOLERANCE`, we switch the spec proposer into AGGRESSIVE EXPLORE
# mode: the prompt instructs the LLM to make large parameter moves and
# consider extreme values, instead of twiddling near the prior best.
_DIVERSIFY_AFTER_N_TIGHT_TRIALS = 2
_DIVERSIFY_TOLERANCE = 0.005

# Probability of injecting a "wild card" hint — even outside plateau mode,
# occasionally encourage the LLM to try an extreme value. Lets the agent
# learn from genuinely unusual configurations instead of staying in a
# comfort zone.
_WILD_CARD_PROBABILITY = 0.25

# Per-family extreme presets that the LLM is told it MAY try when in
# explore mode. These are deliberately outside the typical Goldilocks range
# — the point is to see what happens when capacity / regularization /
# sequence length is pushed to an edge. Each list has 3-5 entries to give
# the LLM real options.
_FAMILY_EXTREME_PRESETS: dict[str, list[dict[str, Any]]] = {
    "bow":           [{"max_features": 50000, "ngram_max": 3, "min_df": 1, "logreg_c": 0.1},
                      {"max_features": 2000,  "ngram_max": 1, "min_df": 5, "logreg_c": 10.0},
                      {"max_features": 30000, "ngram_max": 1, "min_df": 1, "logreg_c": 50.0}],
    "bow_advanced":  [{"word_max_features": 50000, "char_max_features": 50000, "word_ngram_max": 4, "char_ngram_min": 2, "char_ngram_max": 6, "min_df": 1, "logreg_c": 0.5},
                      {"word_max_features": 5000,  "char_max_features": 5000,  "word_ngram_max": 1, "char_ngram_min": 4, "char_ngram_max": 5, "min_df": 5, "logreg_c": 20.0},
                      {"word_max_features": 30000, "char_max_features": 30000, "word_ngram_max": 2, "char_ngram_min": 3, "char_ngram_max": 7, "min_df": 1, "logreg_c": 0.01}],
    "cnn":           [{"channels": 256, "kernel_sizes": [2,3,4,5,7], "dropout": 0.6, "epochs": 6, "learning_rate": 5e-4},
                      {"channels": 32,  "kernel_sizes": [3,5,7],     "dropout": 0.2, "epochs": 2, "learning_rate": 5e-3},
                      {"channels": 128, "kernel_sizes": [1,2,3,4,5,7,9], "dropout": 0.5, "epochs": 4, "learning_rate": 1e-4}],
    "lstm":          [{"hidden_dim": 512, "num_layers": 2, "dropout": 0.5, "epochs": 5, "learning_rate": 5e-4},
                      {"hidden_dim": 32,  "num_layers": 1, "dropout": 0.1, "epochs": 2, "learning_rate": 5e-3},
                      {"hidden_dim": 256, "num_layers": 2, "dropout": 0.6, "epochs": 6, "learning_rate": 1e-4}],
    "embedding_dl":  [{"embedding_dim": 300, "max_vocab": 50000, "hidden_dim": 256, "dropout": 0.5, "epochs": 6, "learning_rate": 5e-4},
                      {"embedding_dim": 32,  "max_vocab": 5000,  "hidden_dim": 32,  "dropout": 0.1, "epochs": 2, "learning_rate": 5e-3},
                      {"embedding_dim": 200, "max_vocab": 20000, "hidden_dim": 128, "dropout": 0.6, "epochs": 5, "learning_rate": 1e-4}],
    "roberta":       [{"max_len": 256, "learning_rate": 5e-6,  "num_epochs": 5, "train_batch_size": 8,  "weight_decay": 0.1},
                      {"max_len": 32,  "learning_rate": 5e-5,  "num_epochs": 2, "train_batch_size": 32, "weight_decay": 0.0},
                      {"max_len": 192, "learning_rate": 1e-5,  "num_epochs": 4, "train_batch_size": 16, "weight_decay": 0.05}],
    "bertweet":      [{"max_len": 256, "learning_rate": 5e-6,  "num_epochs": 5, "train_batch_size": 8,  "weight_decay": 0.1},
                      {"max_len": 32,  "learning_rate": 5e-5,  "num_epochs": 2, "train_batch_size": 32, "weight_decay": 0.0},
                      {"max_len": 192, "learning_rate": 1e-5,  "num_epochs": 4, "train_batch_size": 16, "weight_decay": 0.05}],
}


def _is_tight_band(trials: list[dict[str, Any]]) -> bool:
    """True if the last 2+ successful trials are within _DIVERSIFY_TOLERANCE."""
    successful = [t for t in trials if t.get("success")]
    if len(successful) < _DIVERSIFY_AFTER_N_TIGHT_TRIALS:
        return False
    f1s = []
    for t in successful[-_DIVERSIFY_AFTER_N_TIGHT_TRIALS:]:
        m = t.get("metrics") or {}
        f1 = m.get("f1")
        if isinstance(f1, (int, float)):
            f1s.append(float(f1))
    return len(f1s) >= 2 and (max(f1s) - min(f1s)) <= _DIVERSIFY_TOLERANCE


def _explore_block_for(family_key: str, prior_trials: list[dict[str, Any]], force_wild: bool) -> str:
    """Build the aggressive-exploration prompt block for `family_key`."""
    presets = _FAMILY_EXTREME_PRESETS.get(family_key, [])
    if not presets:
        return ""
    chosen = random.choice(presets) if presets else None
    header = (
        "AGGRESSIVE EXPLORATION MODE.\n"
        "The recent trials of this family clustered in a narrow F1 band —\n"
        "small parameter twiddles are no longer paying off. You must break\n"
        "out of that cluster. Rules for this proposal:\n"
        "  - move at least TWO tunable keys by 2x or more from any prior\n"
        "    spec (numerically, or to a structurally different value).\n"
        "  - try at least one key at an EXTREME of its allowed range\n"
        "    (e.g. an unusually low or high learning rate, a very wide or\n"
        "    very narrow vocabulary, a deeper / shallower model).\n"
        "  - do NOT propose another minor twiddle near the current best.\n"
    )
    if force_wild and chosen is not None:
        wild = (
            "\nWILD-CARD HINT (you are encouraged but not required to use\n"
            "these values verbatim; they're a known-extreme region worth\n"
            "exploring once for this family):\n"
            f"  {json.dumps(chosen)}\n"
        )
        return header + wild
    if chosen is not None:
        sampler = (
            "\nFor inspiration, here are extreme-region presets used by past\n"
            "exploratory runs of this family — feel free to borrow any one\n"
            "value (or several) but you don't have to match them exactly:\n"
            f"  {json.dumps(chosen)}\n"
        )
        return header + sampler
    return header


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


def _ensure_phase_mutation(
    module: object,
    anchor_spec: dict[str, Any],
    proposed_spec: dict[str, Any],
    tried_signatures: set[tuple[tuple[str, str], ...]],
    run_name: str,
    submission_path: str,
    trials: list[dict[str, Any]],
    preferred_keys: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Force the proposed spec to differ from the anchor on enough tunable
    keys, and fall back to a fresh mutation if the LLM proposed a duplicate
    signature. Sweep-only — opt-phase branches were removed.
    """
    candidate = dict(proposed_spec)
    candidate["experiment_name"] = run_name
    candidate["submission_path"] = submission_path
    changed_keys = set(_changed_tunable_keys(module, anchor_spec, candidate))
    issues: list[str] = []

    # Minimum diversity floor: 2 tunable keys across all families. Above 2,
    # the LLM decides. This was previously 3/4 for transformer/non-transformer
    # which over-constrained the search: the LLM at temp=0.5 typically
    # proposes 2 keys with a focused hypothesis, and the higher floor forced
    # the orchestrator to inject unhypothesised mutations that hurt F1
    # (see BoW_advanced 0.7193 → 0.7086 regression from random word_max_features
    # and word_ngram_max additions). Now the orchestrator only intervenes
    # when the LLM proposed fewer than 2 changes.
    target_change_count = 2
    ranges = module.get_spec_ranges()
    ordered_keys = _ordered_underexplored_keys(module, trials, preferred_keys, skip_keys=changed_keys)

    if len(changed_keys) >= target_change_count and _spec_signature(module, candidate) not in tried_signatures:
        return candidate, issues

    ordered_keys = _ordered_underexplored_keys(module, trials, preferred_keys, skip_keys=changed_keys)
    for key in ordered_keys:
        if len(changed_keys) >= target_change_count:
            break
        if key not in ranges or key not in anchor_spec:
            continue
        low, high = ranges[key]
        current_value = candidate.get(key, anchor_spec[key])
        values = _candidate_values(current_value, low, high, local=False) + _candidate_values(
            anchor_spec[key], low, high, local=False
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
            local=False,
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
    prior_launch_trials: list[dict[str, Any]] | None = None,
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
        "- this is the family sweep phase, so explore different regions of the parameter space\n"
        "- the runner uses a 2k labeled sample split 80/20 for training/validation\n"
        "- change at least 2 tunable keys; pick more if your hypothesis "
        "genuinely calls for it (no penalty for going to 3, 4, or more — "
        "the only rule is that EVERY key you change must be named in "
        "changed_keys and described in the hypothesis)\n"
        "- across the limited per-family trial budget, cover model-capacity "
        "keys and optimization keys when relevant instead of nudging only "
        "one key repeatedly\n"
        "- examples of coherent multi-knob moves when your hypothesis "
        "calls for them: capacity + regularization (e.g. max_len + dropout), "
        "capacity + optimization (e.g. max_len + learning_rate), or all three "
        "when testing a joint effect — pick whatever size matches your theory\n"
    )
    # Decide whether to engage AGGRESSIVE EXPLORATION mode. Triggers when
    # the most recent successful trials of this family clustered in a tight
    # F1 band — i.e. small parameter changes are no longer moving the score.
    # Also occasionally fire a "wild card" (a known extreme-region preset)
    # to deliberately seed the agent with crazier configurations to learn from.
    family_key_for_explore = getattr(module, "FAMILY_KEY", str(getattr(module, "FAMILY", "")).lower())
    in_explore_mode = _is_tight_band(trials)
    fire_wild_card = in_explore_mode and (random.random() < _WILD_CARD_PROBABILITY)
    explore_block = (
        _explore_block_for(family_key_for_explore, trials, force_wild=fire_wild_card) + "\n\n"
        if in_explore_mode else ""
    )

    prior_block = _format_prior_launch_trials(prior_launch_trials, module.FAMILY)
    prompt = (
        f"Propose the next {module.FAMILY} experiment spec.\n\n"
        f"{module.get_search_prompt()}\n\n"
        f"{prior_block}"
        f"{explore_block}"
        "Rules:\n"
        "- keep the architecture family fixed\n"
        "- keep the same overall prompt contract and pipeline shape\n"
        f"- vary only these tunable keys: {', '.join(tunable_keys)}\n"
        "- optimize against the best successful session trial, not the last run\n"
        + ("- in AGGRESSIVE EXPLORATION mode, BREAK away from the best spec — large coordinated moves only\n"
           if in_explore_mode else
           "- keep the best successful spec as the default anchor and mutate it only slightly\n")
        + f"{phase_rules}"
        "- do not repeat an exact spec already tried in this session\n"
        "- if a run timed out, reduce cost\n"
        "- if a run crashed, simplify the risky parameter region\n"
        "- if repeated runs get the same F1, switch to different tunable keys instead of repeating weak ones\n"
        "- return one JSON object only — see the system prompt for the exact "
        "three-field schema (hypothesis + changed_keys + spec keys). The "
        "'changed_keys' list MUST contain every tunable key you intend to "
        "modify vs. the prior best anchor below; any silent change will be "
        "reset to the anchor value.\n\n"
        f"Dataset context:\n{data_context}\n\n"
        f"Family history:\n{history_summary}\n\n"
        f"Session trials so far:\n{trial_summary}\n\n"
        f"Best session trial so far:\n{json.dumps(best_trial, indent=2)}\n\n"
        f"Stagnant keys from equal-F1 or same-prediction runs: {', '.join(sorted(stagnant_keys)) if stagnant_keys else 'none'}\n"
        f"Preferred keys for the next move: {', '.join(active_tunable_keys)}\n"
        f"Latest equal-F1 matched prior run: {repeated_match['run_index'] if repeated_match else 'none'}\n"
        f"Latest equal-F1 changed keys to avoid repeating: {', '.join(stale_changed_keys) if stale_changed_keys else 'none'}\n"
        f"Repeated best F1 detected: {'yes' if repeated_f1 else 'no'}\n"
        f"Aggressive exploration mode: {'YES (tight F1 band detected)' if in_explore_mode else 'no'}\n"
        f"Wild card injection: {'YES (try an extreme region)' if fire_wild_card else 'no'}\n\n"
        "Prior-best anchor spec (your 'changed_keys' lists what you "
        "intend to change FROM this; this is the prior best successful "
        "trial, NOT the family default):\n"
        f"{json.dumps(default_spec, indent=2)}\n"
    )

    # Spec search uses a moderately higher temperature than the rest of the
    # agent for the same reason as generate_initial_spec: the output is
    # constrained JSON, the validator clamps numerics, and determinism was
    # anchoring the LLM on the best-trial spec. temp=0.5 balances exploration
    # against hypothesis-spec alignment (temp=0.7 made the LLM's spec change
    # more keys than its hypothesis text claimed, confounding the record).
    raw_response = llm.respond(SEARCH_SYSTEM, prompt, temperature=0.5)
    raw_spec = extract_json_object(raw_response)
    spec, issues = validate_spec(
        raw_spec=raw_spec,
        default_spec=default_spec,
        ranges=module.get_spec_ranges(),
        fixed_keys=module.get_fixed_spec_keys(),
    )
    if "val_size" in spec:
        spec["val_size"] = default_spec.get("val_size", spec["val_size"])
    # Hypothesis-as-source-of-truth: constrain the spec to only the keys the
    # LLM named in `changed_keys`. Every other tunable key is reset to the
    # PRIOR-BEST anchor (not the family default) since this is search/opt
    # phase, not initial. The remaining diversity machinery (phase mutation,
    # repeated-F1 fallback, signature veto) still runs after this constraint
    # so we can't get stuck on a 0-key proposal.
    claimed_keys: list[str] = []
    if isinstance(raw_spec, dict):
        raw_claim = raw_spec.get("changed_keys")
        if isinstance(raw_claim, list):
            claimed_keys = [k for k in raw_claim if isinstance(k, str) and k in tunable_keys]
    if claimed_keys:
        anchor = best_trial["spec"] if best_trial.get("success") else default_spec
        constrained = dict(anchor)
        for k in claimed_keys:
            if k in spec:
                constrained[k] = spec[k]
        constrained["experiment_name"] = run_name
        constrained["submission_path"] = submission_path
        spec = constrained
        issues.append(
            f"Search spec constrained to LLM's declared changed_keys: {claimed_keys}"
        )
    before_changed = set(_changed_tunable_keys(module, best_trial["spec"], spec))
    spec, diversity_issues = _ensure_phase_mutation(
        module=module,
        anchor_spec=best_trial["spec"],
        proposed_spec=spec,
        tried_signatures=tried_signatures,
        run_name=run_name,
        submission_path=submission_path,
        trials=trials,
        preferred_keys=active_tunable_keys,
    )
    after_changed = set(_changed_tunable_keys(module, best_trial["spec"], spec))
    orchestrator_added: list[str] = sorted(after_changed - before_changed)
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
            local=False,
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
            local=False,
        )
        issues.extend(extra_issues)
    # Extract the LLM's stated hypothesis for this revisit. Falls back to an
    # auto-generated string tagged "[fallback]" so it can be distinguished
    # from a real LLM hypothesis in logs and dashboards. The fallback still
    # names which keys changed from the best prior spec, so it's informative
    # even when the LLM omitted the field.
    diff_keys = sorted(_changed_tunable_keys(module, best_trial["spec"], spec))[:3]
    if diff_keys:
        fallback_h = (
            f"[fallback] LLM omitted hypothesis. Revisiting {module.FAMILY} "
            f"with changes to {', '.join(diff_keys)} vs the prior best."
        )
    else:
        fallback_h = (
            f"[fallback] LLM omitted hypothesis. Revisiting {module.FAMILY} "
            f"with no meaningful spec changes (validator drift)."
        )
    hypothesis = _extract_hypothesis(raw_spec, fallback=fallback_h)
    # Append any orchestrator-added keys to the hypothesis so the research
    # record stays honest (same logic as generate_initial_spec).
    if orchestrator_added:
        add_str = ", ".join(f"{k}={spec.get(k)}" for k in orchestrator_added[:4])
        hypothesis = f"{hypothesis} [orchestrator-added: {add_str}]"
    return {
        "prompt": prompt,
        "raw_response": raw_response,
        "raw_spec": raw_spec,
        "spec": spec,
        "hypothesis": hypothesis,
        "claimed_keys": claimed_keys,
        "issues": issues,
        "used_default": raw_spec is None,
    }
