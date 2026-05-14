"""Agent_4 — LLM-driven sweep planner.

The sweep planner decides, after every trial, which family to try next (or
whether to stop sweep entirely). Each decision produces exactly one trial of
exactly one family. Families are never auto-consumed: the planner can revisit
a family that already succeeded, retry one that failed, or declare a family
permanently dead via skip_family_permanently.

The eligibility filter handles the safety floor:
- only families whose estimated cost + start buffer fits in the remaining time
- only families that have not been skip_family_permanently
- only families below the hard safety cap MAX_ATTEMPTS_PER_FAMILY
- at least one successful trial must exist before "stop" is allowed

If the LLM call fails or returns nonsense, we fall back to a deterministic
round-robin over untried families (matches Agent_3's old behavior).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from json_utils import extract_json_object


# Hard safety cap on per-family attempts in sweep. Almost never binds when the
# planner is reasoning sensibly; it exists only to stop a degenerate loop.
MAX_ATTEMPTS_PER_FAMILY = int(os.environ.get("AGENT4_MAX_ATTEMPTS_PER_FAMILY", "5"))


# ---------------------------------------------------------------------------
# Per-family state tracked across a sweep
# ---------------------------------------------------------------------------

TrialOutcome = Literal[
    "success",
    "degenerate_success",  # ran cleanly but F1 < DEGENERATE_F1_THRESHOLD (e.g. one-class predictor)
    "code_gen_failed",
    "training_crash",
    "timeout",
    "no_metrics",
]

# Trials that produce an F1 below this threshold are tagged as
# `degenerate_success` rather than `success`. F1 < 0.4 on this binary task
# is below the all-positive baseline (~0.60) and signals the model collapsed
# to predicting one class. Treating these as success was a foot-gun: the
# planner kept retrying LSTM after F1=0 because it looked like "success".
DEGENERATE_F1_THRESHOLD = 0.4


@dataclass
class FamilyState:
    """Rolling summary of one family's trials during the sweep."""

    family_key: str
    attempts: int = 0
    successes: int = 0
    best_f1: float | None = None
    last_f1: float | None = None
    last_outcome: TrialOutcome | None = None
    last_error_summary: str | None = None
    total_wall_seconds: float = 0.0
    skipped_permanently: bool = False

    @property
    def stagnant(self) -> bool:
        """Two or more successful runs and the latest did not improve."""
        return (
            self.successes >= 2
            and self.best_f1 is not None
            and self.last_f1 is not None
            and abs(self.last_f1 - self.best_f1) < 1e-6
        )

    def update_from_trial(
        self,
        outcome: TrialOutcome,
        f1: float | None,
        wall_seconds: float,
        error_summary: str | None = None,
    ) -> None:
        self.attempts += 1
        self.total_wall_seconds += max(0.0, float(wall_seconds))
        self.last_outcome = outcome
        self.last_error_summary = error_summary
        if outcome == "success" and f1 is not None:
            self.successes += 1
            self.last_f1 = float(f1)
            if self.best_f1 is None or float(f1) > self.best_f1:
                self.best_f1 = float(f1)
        else:
            # degenerate_success / code_gen_failed / timeout / etc. — don't
            # advance the successes counter, keep best_f1 untouched. last_f1
            # still reflects what just happened (could be 0 for degenerate).
            self.last_f1 = float(f1) if f1 is not None else None


def classify_trial_outcome(
    result: dict[str, Any],
    repair_exhausted: bool,
) -> TrialOutcome:
    """Map a sandbox result dict into one of the outcome buckets.

    `result` is whatever `sandbox.run_experiment` returned for the final attempt
    (or the last repair iteration). `repair_exhausted` is True when the agent
    burned its full repair budget without ever producing a runnable script.
    """
    if result.get("success"):
        # Fix 4: if the model collapsed to a one-class predictor (F1 below the
        # degenerate threshold) tag it differently so the planner stops
        # retrying it as if it were a real success.
        f1 = (result.get("metrics") or {}).get("f1")
        try:
            if f1 is not None and float(f1) < DEGENERATE_F1_THRESHOLD:
                return "degenerate_success"
        except (TypeError, ValueError):
            pass
        return "success"
    if repair_exhausted and not result.get("process_success", False):
        return "code_gen_failed"
    if result.get("timed_out"):
        return "timeout"
    if result.get("dry_run_failed"):
        return "code_gen_failed"
    # The process ran and exited, but no parsable METRICS line came out.
    if result.get("process_success", False):
        return "no_metrics"
    # Anything else: training started but crashed before metrics.
    return "training_crash"


# ---------------------------------------------------------------------------
# Decision schema
# ---------------------------------------------------------------------------

DecisionAction = Literal["try_family", "skip_family_permanently", "stop"]


@dataclass
class SweepDecision:
    action: DecisionAction
    family_key: str | None
    reason: str
    eligible_families: list[str] = field(default_factory=list)
    raw_response: str = ""
    prompt: str = ""
    time_remaining_seconds: int = 0
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Eligibility filtering
# ---------------------------------------------------------------------------

def eligible_families(
    family_state: dict[str, FamilyState],
    cost_estimates: dict[str, int],
    time_remaining_seconds: float,
    start_buffer_seconds: int,
) -> list[str]:
    """Filter to families that:

    - have not been permanently skipped by the planner
    - are under MAX_ATTEMPTS_PER_FAMILY
    - fit in the remaining time given their estimated cost + start buffer
    - have NOT accumulated >= 2 code_gen failures without ever succeeding
      (hard filter: the planner LLM cannot be trusted to count this reliably,
       so the orchestrator enforces it. This stops the agent from grinding
       infinite retries on a family the code-gen LLM clearly can't handle.)
    """
    eligible: list[str] = []
    for family_key, state in family_state.items():
        if state.skipped_permanently:
            continue
        if state.attempts >= MAX_ATTEMPTS_PER_FAMILY:
            continue
        if (
            state.successes == 0
            and state.attempts >= 2
            and state.last_outcome == "code_gen_failed"
        ):
            # Hard skip: 2+ code_gen_failed in a row with no success.
            # Same effect as skip_family_permanently but forced by the orchestrator.
            continue
        if (
            state.successes == 0
            and state.attempts >= 2
            and state.last_outcome == "degenerate_success"
        ):
            # Hard skip: 2+ degenerate_success (e.g., one-class predictor at F1=0)
            # with no real success. Without this, planner kept picking the same
            # broken family because F1=0 ran cleanly. Fix 4.
            continue
        cost = cost_estimates.get(family_key, 480)
        if cost + start_buffer_seconds > time_remaining_seconds:
            continue
        eligible.append(family_key)
    return eligible


def any_success(family_state: dict[str, FamilyState]) -> bool:
    return any(state.successes > 0 for state in family_state.values())


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _format_state_table(
    family_state: dict[str, FamilyState],
    cost_estimates: dict[str, int],
    eligible: list[str],
) -> str:
    """Render a compact ASCII table the planner LLM can read."""
    header = f"{'family':<14}{'att':>4}{'ok':>4}{'bestF1':>9}{'lastF1':>9}{'outcome':>22}{'wall':>7}{'cost':>7}  eligible"
    rows: list[str] = [header, "-" * len(header)]
    for family_key in sorted(family_state.keys()):
        state = family_state[family_key]
        best = f"{state.best_f1:.4f}" if state.best_f1 is not None else "-"
        last = f"{state.last_f1:.4f}" if state.last_f1 is not None else "-"
        outcome = state.last_outcome or "-"
        if state.last_outcome == "success" and state.stagnant:
            outcome = "success (stagnant)"
        cost = cost_estimates.get(family_key, 480)
        wall = int(state.total_wall_seconds)
        flag = "yes" if family_key in eligible else ("SKIP" if state.skipped_permanently else "no")
        rows.append(
            f"{family_key:<14}{state.attempts:>4}{state.successes:>4}{best:>9}{last:>9}{outcome:>22}{wall:>6}s{cost:>6}s  {flag}"
        )
    return "\n".join(rows)


def build_planner_prompt(
    family_state: dict[str, FamilyState],
    cost_estimates: dict[str, int],
    eligible: list[str],
    time_remaining_seconds: float,
    success_recorded: bool,
) -> str:
    """User message handed to the planner LLM."""
    table = _format_state_table(family_state, cost_estimates, eligible)
    stop_allowed = "yes" if success_recorded else "no (no successful F1 yet)"
    eligible_list = ", ".join(eligible) if eligible else "(none — only `stop` is legal)"

    # Fix 1: pre-compute the answer to "is this family untried?" so the LLM
    # never has to count rows in the table to figure it out. This dramatically
    # cuts the "called X untried when it was actually tried" hallucinations.
    untried_eligible = sorted(
        k for k in eligible if family_state[k].attempts == 0
    )
    tried_eligible = sorted(
        k for k in eligible if family_state[k].attempts > 0
    )
    untried_str = ", ".join(untried_eligible) if untried_eligible else "(none)"
    tried_str = ", ".join(tried_eligible) if tried_eligible else "(none)"

    lines = [
        "Decide the next sweep action.",
        "",
        f"Time remaining in sweep window: {int(time_remaining_seconds)} seconds.",
        f"Stop allowed? {stop_allowed}.",
        f"Eligible families this step: {eligible_list}.",
        "",
        f"UNTRIED families currently eligible (zero prior trials): {untried_str}",
        f"TRIED families currently eligible (have at least one observation): {tried_str}",
        "",
        "Per-family state:",
        table,
        "",
        "How to think about each choice:",
        "- Untried families: their F1 is unknown — trying one gives you a new data point. Prefer untried until none remain eligible.",
        "- Revisits of a successful family: useful when the last trial was an improvement. After 2+ successive successes within 0.005 F1, the family is plateauing and revisits rarely add information.",
        "- code_gen_failed once: retry is reasonable. Two consecutive with no success → the orchestrator drops the family automatically.",
        "- degenerate_success: the model trained but collapsed to predicting one class (F1 < 0.4). A retry with a different spec (lower learning_rate, fewer parameters, different embedding_dim) may rescue it. Two consecutive → the orchestrator drops the family automatically.",
        "- training_crash / timeout / no_metrics: usually fixable with a different spec; one retry is worth it.",
        "- Cost matters only for eligibility. Within the eligible list, all families are equally valid choices — do NOT prefer cheap.",
        "",
        "Return ONE JSON object on a single line, no commentary:",
        '{"action":"try_family","family_key":"<key>","reason":"<one short sentence>"}',
        "or",
        '{"action":"skip_family_permanently","family_key":"<key>","reason":"<one short sentence>"}',
        "or",
        '{"action":"stop","reason":"<one short sentence>"}',
        "",
        "Pick only family_keys from the eligible list above (or any non-skipped family for skip_family_permanently).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing + fallback
# ---------------------------------------------------------------------------

def _parse_planner_response(raw: str) -> dict[str, Any] | None:
    parsed = extract_json_object(raw)
    return parsed if isinstance(parsed, dict) else None


def _coerce_decision(
    parsed: dict[str, Any] | None,
    eligible: list[str],
    family_state: dict[str, FamilyState],
    success_recorded: bool,
) -> tuple[DecisionAction, str | None, str]:
    """Validate the planner's JSON; fall back to safe defaults when it lies."""
    fallback_family = eligible[0] if eligible else None
    if parsed is None:
        return ("try_family", fallback_family, "fallback: malformed planner JSON") if fallback_family else (
            "stop",
            None,
            "fallback: malformed planner JSON and no eligible families remain",
        )

    action = str(parsed.get("action", "")).strip()
    reason = str(parsed.get("reason", "")).strip() or "(no reason given)"
    family_key = parsed.get("family_key")
    family_key = str(family_key).strip() if family_key else None

    if action == "stop":
        if success_recorded:
            return "stop", None, reason
        if fallback_family:
            return "try_family", fallback_family, f"fallback: stop blocked (no success yet); {reason}"
        return "stop", None, f"fallback: stop allowed because no eligible families remain; {reason}"

    if action == "skip_family_permanently":
        if family_key and family_key in family_state and not family_state[family_key].skipped_permanently:
            return "skip_family_permanently", family_key, reason
        # Bad target — fall through to try_family with the cheapest eligible.
        if fallback_family:
            return "try_family", fallback_family, f"fallback: invalid skip target '{family_key}'; {reason}"
        return "stop", None, f"fallback: invalid skip target and no eligible families; {reason}"

    if action == "try_family":
        if family_key and family_key in eligible:
            return "try_family", family_key, reason
        if fallback_family:
            return "try_family", fallback_family, (
                f"fallback: planner picked '{family_key}' which is not eligible; using cheapest eligible. {reason}"
            )
        if success_recorded:
            return "stop", None, f"fallback: planner picked invalid family and none eligible; {reason}"
        return "stop", None, f"fallback: planner picked invalid family and no successes yet; {reason}"

    # Unknown action.
    if fallback_family:
        return "try_family", fallback_family, f"fallback: unknown action '{action}'; {reason}"
    return "stop", None, f"fallback: unknown action '{action}' and no eligible families; {reason}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def select_next_sweep_action(
    llm: Any,
    family_state: dict[str, FamilyState],
    cost_estimates: dict[str, int],
    time_remaining_seconds: float,
    start_buffer_seconds: int,
    planner_system_prompt: str,
) -> SweepDecision:
    """Ask the planner LLM for the next sweep decision.

    `llm` only needs a `.respond(system, user)` method that returns a string
    (the OllamaClient interface). We do not import OllamaClient here so this
    module stays trivial to unit-test with a stub.
    """
    eligible = eligible_families(
        family_state, cost_estimates, time_remaining_seconds, start_buffer_seconds
    )
    success_recorded = any_success(family_state)
    prompt = build_planner_prompt(
        family_state=family_state,
        cost_estimates=cost_estimates,
        eligible=eligible,
        time_remaining_seconds=time_remaining_seconds,
        success_recorded=success_recorded,
    )

    raw_response = ""
    parsed: dict[str, Any] | None = None
    try:
        raw_response = llm.respond(planner_system_prompt, prompt) or ""
        parsed = _parse_planner_response(raw_response)
    except Exception as exc:  # noqa: BLE001 - planner is best-effort
        raw_response = f"[planner-llm-error] {exc}"

    action, family_key, reason = _coerce_decision(
        parsed=parsed,
        eligible=eligible,
        family_state=family_state,
        success_recorded=success_recorded,
    )

    return SweepDecision(
        action=action,
        family_key=family_key,
        reason=reason,
        eligible_families=eligible,
        raw_response=raw_response,
        prompt=prompt,
        time_remaining_seconds=int(time_remaining_seconds),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def decision_to_log_record(decision: SweepDecision) -> dict[str, Any]:
    """Strip the prompt/raw fields when writing a compact jsonl line."""
    return {
        "timestamp": decision.timestamp,
        "time_remaining_seconds": decision.time_remaining_seconds,
        "eligible_families": decision.eligible_families,
        "action": decision.action,
        "family_key": decision.family_key,
        "reason": decision.reason,
    }
