"""Initial spec generation for Agent_4."""

from __future__ import annotations

import json
from typing import Any

from json_utils import extract_json_object
from prompts import SPEC_SYSTEM
from validate_spec import validate_spec


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
    prior_block = _format_prior_launch_trials(prior_launch_trials, module.FAMILY)
    prompt = (
        f"Plan one reliable {module.FAMILY} experiment spec for the Kaggle Disaster Tweets task.\n\n"
        f"{module.get_spec_prompt()}\n\n"
        "Return one JSON object only — see the system prompt for the exact "
        "three-field schema (hypothesis + changed_keys + spec keys). The "
        "hypothesis must reference concrete prior evidence when available "
        "(a past F1, a spec choice that worked or didn't). DO NOT just say "
        "'establish a baseline' if prior-launch evidence below shows what "
        "already worked.\n\n"
        f"Tunable keys you may put in 'changed_keys': {', '.join(module.get_tunable_keys())}\n\n"
        f"{prior_block}"
        f"Dataset context:\n{data_context}\n\n"
        f"Recent in-launch history:\n{history_summary}\n\n"
        "Family default spec (this is the anchor — your 'changed_keys' "
        "lists what you intend to change FROM this). You MUST list at "
        "LEAST 2 tunable keys in 'changed_keys' and the spec values you "
        "return for those keys must differ from the default. Any spec key "
        "you change but DO NOT name in changed_keys will be silently reset "
        "to the default — this enforces hypothesis-as-source-of-truth.\n"
        f"{json.dumps(default_spec, indent=2)}\n"
    )

    # Spec proposals use a moderately higher temperature than the rest of
    # the agent. At temp=0.2 the LLM anchors on the default + prior-best
    # specs and only tweaks one knob. At temp=0.7 it explores more keys but
    # tends to under-claim in its hypothesis text (says "lower lr" but the
    # spec also moves max_len + batch_size), which confounds the research
    # record. temp=0.5 is the middle ground: enough randomness to break the
    # anchoring bias, low enough that the hypothesis still accurately
    # reflects what the spec actually changes. Code generation, repair,
    # and analysis stay at temp=0.2 for correctness.
    raw_response = llm.respond(SPEC_SYSTEM, prompt, temperature=0.5)
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


def _format_prior_launch_trials(
    prior_trials: list[dict[str, Any]] | None,
    family: str,
) -> str:
    """Render cross-launch memory trials for THIS family as a prompt block.

    Shows ALL records for the requested family — no per-family cap. The
    short-term memory itself is bounded to 20 total trials across every
    family, so the worst case here is 20 entries (still a tight prompt
    block). Empty input returns an empty string (no header) so the prompt
    stays clean for the first time a family is ever attempted.

    Per trial we emit:
      - F1 (or failure outcome)
      - Full spec inline
      - The hypothesis the LLM wrote when proposing that trial
      - The analyst's one-sentence CONCLUSION verdict (confirmed / refuted)
    """
    if not prior_trials:
        return ""
    # Filter to this family. Sort by F1 descending so the most-promising
    # specs surface first; ties broken by most recent timestamp.
    same = [t for t in prior_trials if str(t.get("family", "")).lower() == family.lower()]
    if not same:
        return ""

    def _f1(t: dict[str, Any]) -> float:
        try:
            return float(t.get("f1")) if t.get("f1") is not None else -1.0
        except (TypeError, ValueError):
            return -1.0

    # Stable two-pass sort: most recent first within F1 ties, F1 desc overall.
    same.sort(key=lambda t: t.get("timestamp", ""), reverse=True)
    same.sort(key=lambda t: _f1(t), reverse=True)
    lines = [f"PRIOR-LAUNCH EVIDENCE for {family} (newest+highest-F1 first):"]
    for t in same:
        spec = t.get("spec") or {}
        # Compact spec rendering — same format the short-term memory uses
        spec_parts: list[str] = []
        for k in sorted(spec.keys()):
            v = spec[k]
            if isinstance(v, float):
                vs = f"{v:.2e}" if abs(v) < 0.01 or abs(v) >= 10000 else f"{v:g}"
            elif isinstance(v, (list, tuple)):
                vs = "[" + ",".join(str(x) for x in v) + "]"
            else:
                vs = str(v)
            spec_parts.append(f"{k}={vs}")
        f1 = t.get("f1")
        f1_str = f"F1={f1:.4f}" if isinstance(f1, (int, float)) else f"({t.get('outcome', '?')})"
        lines.append(f"  - {f1_str}  spec: {', '.join(spec_parts)}")
        hyp = (t.get("hypothesis") or "").strip()
        if hyp:
            lines.append(f"      prior hypothesis: {hyp}")
        conclusion = _extract_conclusion(t.get("analysis") or "")
        if conclusion:
            lines.append(f"      prior verdict   : {conclusion}")
    lines.append(
        "\nUse this evidence: write a hypothesis that proposes a SPECIFIC move "
        "vs. the prior best (e.g. 'try lr=5e-6 to test if the 0.7976 best can "
        "be pushed via slower training') rather than a generic baseline statement. "
        "Pay attention to the prior verdicts — if a hypothesis was already refuted, "
        "do not re-test the same direction; pivot to a different parameter or value.\n"
    )
    return "\n".join(lines) + "\n"


def _extract_hypothesis(raw_spec: Any, fallback: str, max_len: int = 240) -> str:
    """Pull the 'hypothesis' field from an LLM JSON response.

    Falls back to `fallback` when the LLM omitted the key, returned a
    non-string, or the response wasn't a dict. Truncates excessively long
    hypotheses so they don't bloat the prompt context downstream.
    """
    if not isinstance(raw_spec, dict):
        return fallback
    value = raw_spec.get("hypothesis")
    if not isinstance(value, str):
        return fallback
    value = value.strip()
    if not value:
        return fallback
    if len(value) > max_len:
        value = value[: max_len - 1].rstrip() + "…"
    return value
