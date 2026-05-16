"""Initial spec generation for Agent_4."""

from __future__ import annotations

import json
from typing import Any

from json_utils import extract_json_object
from prompts import SPEC_PROPOSER_SYSTEM
from validate_spec import validate_spec


# ---------------------------------------------------------------------------
# Shared helpers — table-based prompt construction
# ---------------------------------------------------------------------------
# The spec proposer (both first-attempt and revisit) gets the same compact
# table of prior trials + anchor spec. The narrative-heavy "prior hypothesis"
# and "prior verdict" text used to leak into the LLM's new hypothesis verbatim
# (copy-paste contamination), so they're intentionally NOT in the table.

def _compact_value(value: Any) -> str:
    """Short string for a hyperparameter value (used in the trials table)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 0.001 or abs(value) >= 10_000:
            return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")
        return f"{value:g}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_compact_value(x) for x in value) + "]"
    if value is None:
        return "-"
    s = str(value)
    return s if len(s) <= 24 else s[:22] + "…"


def _spec_compact_str(spec: dict[str, Any], tunable_keys: list[str]) -> str:
    """Render only the tunable keys of a spec as 'k=v, k=v, ...'."""
    parts: list[str] = []
    for k in tunable_keys:
        if k in spec:
            parts.append(f"{k}={_compact_value(spec[k])}")
    return ", ".join(parts) if parts else "(no tunable keys set)"


def _trial_table_row(trial: dict[str, Any], tunable_keys: list[str], source: str) -> str:
    """One line of the prior-trials table."""
    f1 = trial.get("f1")
    if f1 is None:
        f1 = (trial.get("metrics") or {}).get("f1")
    if isinstance(f1, (int, float)):
        f1_str = f"{f1:.4f}"
    else:
        f1_str = "fail  "
    spec = trial.get("spec") or {}
    spec_str = _spec_compact_str(spec, tunable_keys)
    outcome = trial.get("outcome") or ("success" if trial.get("success") else "code_gen_failed")
    return f"  {f1_str} | {spec_str:<60} | {source:<12} | {outcome}"


def format_prior_trials_table(
    session_trials: list[dict[str, Any]] | None,
    cross_launch_trials: list[dict[str, Any]] | None,
    family: str,
    tunable_keys: list[str],
    plateau_keys: list[str] | None = None,
) -> str:
    """Render all prior trials for a family as a single compact table.

    Replaces the prior 60+ line narrative dump (with prior hypothesis + verdict
    text the LLM was copy-pasting). Shows F1 + tunable-key values for every
    trial this launch + cross-launch. Plateau flags from the orchestrator are
    surfaced as a line below the table so the LLM doesn't have to detect
    plateaus on its own.
    """
    rows: list[tuple[float, str]] = []
    # session trials (most recent first)
    for trial in reversed(session_trials or []):
        f1_val = (trial.get("metrics") or {}).get("f1")
        rows.append((
            float(f1_val) if isinstance(f1_val, (int, float)) else -1.0,
            _trial_table_row(
                {"f1": f1_val, "spec": trial.get("spec") or {}, "outcome": trial.get("outcome"), "success": trial.get("success")},
                tunable_keys, "this launch",
            ),
        ))
    # cross-launch trials (filter to this family)
    same_family = [t for t in (cross_launch_trials or []) if str(t.get("family", "")).lower() == family.lower()]
    same_family.sort(key=lambda t: t.get("timestamp", ""), reverse=True)
    for trial in same_family:
        f1 = trial.get("f1")
        rows.append((
            float(f1) if isinstance(f1, (int, float)) else -1.0,
            _trial_table_row(trial, tunable_keys, "prior launch"),
        ))

    if not rows:
        return f"PRIOR TRIALS for {family}: none yet — this is the first attempt.\n"

    header = f"  {'F1':<6} | {'tunable spec':<60} | {'source':<12} | outcome"
    sep = "  " + "-" * (len(header) - 2)
    lines = [f"PRIOR TRIALS for {family} (newest first):", header, sep]
    lines.extend(row for _, row in rows)
    if plateau_keys:
        lines.append("")
        lines.append(
            f"PLATEAUED KEYS (last trials moved this key but F1 barely changed — "
            f"change a DIFFERENT key): {', '.join(plateau_keys)}"
        )
    return "\n".join(lines) + "\n"


def build_proposer_user_prompt(
    family: str,
    family_spec_prompt: str,
    tunable_keys: list[str],
    anchor_spec: dict[str, Any],
    anchor_label: str,
    session_trials: list[dict[str, Any]] | None,
    cross_launch_trials: list[dict[str, Any]] | None,
    data_context: str,
    plateau_keys: list[str] | None = None,
    extra_rules: list[str] | None = None,
) -> str:
    """Assemble the unified spec-proposer user prompt.

    Used by BOTH generate_initial_spec (anchor_label='family default') and
    propose_next_spec (anchor_label='current best'). Same shape, same system
    prompt — the only thing that varies is which spec is the anchor.
    """
    table = format_prior_trials_table(
        session_trials=session_trials,
        cross_launch_trials=cross_launch_trials,
        family=family,
        tunable_keys=tunable_keys,
        plateau_keys=plateau_keys,
    )
    lines = [
        f"Propose the next {family} spec.",
        "",
        family_spec_prompt,
        "",
        table,
        f"ANCHOR ({anchor_label}):",
        json.dumps({k: anchor_spec[k] for k in anchor_spec if not k.startswith("_")}, indent=2),
        "",
        f"TUNABLE KEYS: {', '.join(tunable_keys)}",
        "",
        f"Dataset:\n{data_context}",
    ]
    if extra_rules:
        lines.append("")
        for rule in extra_rules:
            lines.append(f"- {rule}")
    return "\n".join(lines)


def generate_initial_spec(
    llm,
    module: object,
    run_name: str,
    submission_path: str,
    data_context: str,
    history_summary: str,
    prior_launch_trials: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    default_spec = module.get_default_spec(run_name, submission_path)
    prompt = build_proposer_user_prompt(
        family=module.FAMILY,
        family_spec_prompt=module.get_spec_prompt(),
        tunable_keys=list(module.get_tunable_keys()),
        anchor_spec=default_spec,
        anchor_label="family default",
        session_trials=None,
        cross_launch_trials=prior_launch_trials,
        data_context=data_context,
        plateau_keys=None,
    )

    # Spec proposals use temp=0.5 — moderate exploration without the
    # hypothesis/spec drift that temp=0.7 produces. Code-gen and repair
    # stay at temp=0.2 for correctness.
    raw_response = llm.respond(SPEC_PROPOSER_SYSTEM, prompt, temperature=0.5)
    raw_spec = extract_json_object(raw_response)
    spec, issues = validate_spec(
        raw_spec=raw_spec,
        default_spec=default_spec,
        ranges=module.get_spec_ranges(),
        fixed_keys=module.get_fixed_spec_keys(),
    )
    # Hypothesis-as-source-of-truth: constrain the actual spec to ONLY the
    # keys the LLM explicitly named in `changed_keys`. Every other tunable
    # key is reset to its default value. This eliminates the under-claiming
    # gap where the LLM's spec silently changed more keys than the
    # hypothesis text described, breaking the research record's causal
    # attribution. Imported lazily because search.py imports helpers from
    # this module — top-level cross-import would cycle.
    from search import _changed_tunable_keys, _ensure_phase_mutation  # noqa: PLC0415
    tunable = list(module.get_tunable_keys())
    claimed_keys: list[str] = []
    if isinstance(raw_spec, dict):
        raw_claim = raw_spec.get("changed_keys")
        if isinstance(raw_claim, list):
            claimed_keys = [k for k in raw_claim if isinstance(k, str) and k in tunable]
    if claimed_keys:
        # Build a constrained spec: default values for every key, with
        # only the claimed keys overridden by the LLM's chosen values.
        constrained = dict(default_spec)
        for k in claimed_keys:
            if k in spec:
                constrained[k] = spec[k]
        constrained["experiment_name"] = run_name
        constrained["submission_path"] = submission_path
        spec = constrained
        issues.append(
            f"Spec constrained to LLM's declared changed_keys: {claimed_keys}"
        )
    # Hard-enforce: at least 2 tunable keys must actually differ from the
    # default. The LLM at temp=0.5 + changed_keys typically proposes 2-key
    # focused experiments with grounded hypotheses; pushing the floor higher
    # forced the orchestrator to inject unhypothesised mutations that hurt
    # F1. The floor is now a safety net for the rare 0/1-key case, not a
    # diversity engine. Any orchestrator-added keys are appended to the
    # hypothesis so the record stays honest.
    diffs_before = len(_changed_tunable_keys(module, default_spec, spec))
    orchestrator_added: list[str] = []
    if diffs_before < 2:
        before_changed = set(_changed_tunable_keys(module, default_spec, spec))
        spec, diversity_issues = _ensure_phase_mutation(
            module=module,
            anchor_spec=default_spec,
            proposed_spec=spec,
            tried_signatures=set(),
            run_name=run_name,
            submission_path=submission_path,
            trials=[],
            preferred_keys=list(module.get_tunable_keys()),
        )
        after_changed = set(_changed_tunable_keys(module, default_spec, spec))
        orchestrator_added = sorted(after_changed - before_changed)
        diffs_after = len(after_changed)
        issues.append(
            f"Initial spec only differed from default on {diffs_before} tunable key(s); "
            f"forced extra mutations to reach {diffs_after} key diff(s). "
            f"Orchestrator-added keys: {orchestrator_added}"
        )
        issues.extend(diversity_issues)
    # The LLM is asked to include a top-level 'hypothesis' field in the JSON
    # (see SPEC_SYSTEM). validate_spec drops it (it's not a tunable key), so
    # we pull it from raw_spec instead. Fallback is tagged "[fallback]" so
    # downstream readers can tell it apart from a real LLM hypothesis — this
    # only fires when the LLM omits the field entirely, which is rare with
    # the current prompt but still worth distinguishing for debugging.
    hypothesis = _extract_hypothesis(
        raw_spec,
        fallback=(
            f"[fallback] LLM did not return a 'hypothesis' field; running "
            f"the {module.FAMILY} default spec to gather a measurement."
        ),
    )
    # If the orchestrator had to add keys to meet the 2-key minimum,
    # append those to the hypothesis so the research record stays honest.
    # The LLM's narrative remains intact; the annotation just makes the
    # full diff explicit.
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


def _extract_conclusion(analysis: str, max_len: int = 240) -> str:
    """Pull the analyst's CONCLUSION line from a full structured analysis.

    The analyst output follows the ANALYSIS_PROMPT_TEMPLATE format:
        **CONCLUSION:** ...
        **WHAT WORKED:** ...
        **WHAT FAILED:** ...
        **NEXT MOVE:** ...

    We extract just the CONCLUSION paragraph so the prior-launch evidence
    block stays compact. Falls back to the first non-empty line if the
    structured markers aren't present.
    """
    if not isinstance(analysis, str) or not analysis.strip():
        return ""
    # Try the structured marker first
    upper = analysis.upper()
    start = upper.find("CONCLUSION")
    if start >= 0:
        # Find the end of the CONCLUSION block (next ** marker or empty line)
        rest = analysis[start:]
        # Strip the marker prefix like "**CONCLUSION:**" — find the first colon,
        # then also strip any trailing ** (the bold closing of the marker).
        colon = rest.find(":")
        if colon >= 0:
            rest = rest[colon + 1:].lstrip()
        # The closing "**" of the marker may sit right at the start of the
        # remaining text; remove it (and any other leading markdown bold).
        while rest.startswith("*"):
            rest = rest.lstrip("*").lstrip()
        # Cut at the next bold marker (start of next section)
        end_marker = -1
        for marker in ("**WHAT", "**NEXT", "\n\n"):
            idx = rest.find(marker)
            if idx > 0 and (end_marker < 0 or idx < end_marker):
                end_marker = idx
        snippet = rest[:end_marker] if end_marker > 0 else rest
    else:
        # Fallback: first non-empty line
        snippet = next((line for line in analysis.splitlines() if line.strip()), analysis)
    snippet = " ".join(snippet.split())  # collapse whitespace
    if len(snippet) > max_len:
        snippet = snippet[: max_len - 1].rstrip() + "…"
    return snippet


def _extract_hypothesis(raw_spec: Any, fallback: str, max_len: int = 240) -> str:
    """Pull the LLM's reasoning string from a JSON response.

    Accepts both the new 'why' field (current schema) and the legacy
    'hypothesis' field (back-compat with cross-launch memory entries written
    under the old schema). Falls back to `fallback` when neither is present.
    Truncates excessively long values so they don't bloat prompts downstream.
    """
    if not isinstance(raw_spec, dict):
        return fallback
    value = raw_spec.get("why")
    if not isinstance(value, str) or not value.strip():
        value = raw_spec.get("hypothesis")
    if not isinstance(value, str):
        return fallback
    value = value.strip()
    if not value:
        return fallback
    if len(value) > max_len:
        value = value[: max_len - 1].rstrip() + "…"
    return value
