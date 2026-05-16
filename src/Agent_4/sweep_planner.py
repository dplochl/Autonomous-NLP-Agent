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
round-robin over untried families.
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


# Plateau definition: a family is plateaued when its last `PLATEAU_WINDOW`
# successful trials all fall within `PLATEAU_TOLERANCE` of the family's best
# F1 so far. We pull these into module-level constants so they can be tuned
# from one place.
#
# This MUST be larger than the AGGRESSIVE EXPLORATION trigger in search.py
# (_DIVERSIFY_AFTER_N_TIGHT_TRIALS = 2), otherwise the orchestrator would
# kill a family before its explore-mode attempts get a chance to break out
# of the tight band. Concretely with the current numbers:
#   trial 1, 2: normal mode
#   trial 3:    2 priors in band → EXPLORE MODE + possible wild card
#   trial 4:    3 priors in band → EXPLORE MODE + possible wild card
#   trial 5:    4 priors in band → EXPLORE MODE + possible wild card
#   trial 6:    5 priors in band → HARD SKIP (plateaued)
# That's three exploration attempts before giving up — enough room for the
# wild card to actually fire and learn something, while still capping the
# worst case so we don't grind forever on a family that can't move.
PLATEAU_WINDOW = 2       # successful trials in tight band before we give up
PLATEAU_TOLERANCE = 0.005  # F1 swing tolerance for "no movement"


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
    # Full successful-F1 history so we can detect a "tight band" plateau
    # (multiple trials whose F1s are within PLATEAU_TOLERANCE of each other,
    # even if they're not bit-identical).
    success_f1s: list[float] = field(default_factory=list)

    @property
    def stagnant(self) -> bool:
        """Two or more successful runs and the latest did not improve."""
        return (
            self.successes >= 2
            and self.best_f1 is not None
            and self.last_f1 is not None
            and abs(self.last_f1 - self.best_f1) < 1e-6
        )

    @property
    def plateaued(self) -> bool:
        """True when the family has had at least `PLATEAU_WINDOW` successful
        trials and the spread of the most recent N is within
        `PLATEAU_TOLERANCE` of the best. This catches "tight-band" plateaus
        like F1=0.7261 / 0.7219 / 0.7237 / 0.7237 — where each trial proposed
        a different spec but the score barely moved. Once plateaued, further
        revisits rarely pay off and the budget is better spent elsewhere.
        """
        if len(self.success_f1s) < PLATEAU_WINDOW or self.best_f1 is None:
            return False
        window = self.success_f1s[-PLATEAU_WINDOW:]
        return (self.best_f1 - min(window)) <= PLATEAU_TOLERANCE and \
               (max(window) - min(window)) <= PLATEAU_TOLERANCE

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
            self.success_f1s.append(float(f1))
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
        if state.plateaued:
            # Hard skip: tight-band plateau on the last PLATEAU_WINDOW
            # successful trials. The planner LLM has historically refused to
            # acknowledge this and kept revisiting the same family with tiny
            # parameter twiddles. The orchestrator enforces stop here so the
            # remaining budget goes to families that can still move F1.
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


def _phase_header(current_phase: str, leader_family: str | None, leader_f1: float | None) -> list[str]:
    """Strong goal-framing header at the top of the user prompt.

    The sweep is split into two phases by a hard wall-clock gate in agent.py.
    Each phase has ONE clear goal so the LLM doesn't have to weigh
    explore-vs-exploit on its own (it consistently fails to pivot when left
    to its own judgment). The eligible list and reasoning protocol are
    unchanged — only the goal framing flips.
    """
    if current_phase == "A":
        return [
            "=== CURRENT PHASE: A (EXPLORE) ===",
            "Your ONE goal right now is to COVER families and parameter "
            "regions you have not yet measured this launch. Do NOT optimise "
            "for F1 yet — that comes in Phase B. A trial that explores an "
            "untried family right now is more valuable than a trial that "
            "drills the current leader.",
            "",
        ]
    leader_str = (
        f"{leader_family} (F1={leader_f1:.4f})"
        if leader_family and leader_f1 is not None
        else "none yet"
    )
    return [
        "=== CURRENT PHASE: B (MAXIMISE F1) ===",
        "Exploration is OVER. Your ONE goal now is to push F1 ABOVE the "
        f"current leader: {leader_str}. Do NOT pick an untried family for "
        "'coverage' — pick whichever family is most likely to lift F1 "
        "ABOVE that number. Usually this means drilling the current leader "
        "with parameter tweaks; pick a different family only if it has "
        "demonstrated F1 within striking distance and clear runway to "
        "improve. Untried families are NOT preferred in this phase.",
        "",
    ]


def build_planner_prompt(
    family_state: dict[str, FamilyState],
    cost_estimates: dict[str, int],
    eligible: list[str],
    time_remaining_seconds: float,
    success_recorded: bool,
    prior_launch_memory: str = "",
    current_phase: str = "A",
) -> str:
    """User message handed to the planner LLM."""
    table = _format_state_table(family_state, cost_estimates, eligible)
    eligible_list = ", ".join(eligible) if eligible else "(none)"

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

    # Compact state summary — names the current leader and structures the
    # info the planner needs to reason from evidence (rather than pattern-
    # match a "phase" label).
    best_family = None
    best_f1 = -1.0
    for fk, st in family_state.items():
        if st.successes > 0 and st.best_f1 is not None and st.best_f1 > best_f1:
            best_family = fk
            best_f1 = st.best_f1
    if best_family is None:
        leader_line = "Current leader THIS launch: none yet (no successful trial)."
    else:
        leader_line = f"Current leader THIS launch: {best_family} (F1={best_f1:.4f})."
    phase_hint = leader_line

    # USER PROMPT — ORDER MATTERS. Spec proposer pattern: memory +
    # "use this evidence" instruction sit IMMEDIATELY before the output
    # schema, so the LLM's last-attended context is the evidence it must
    # cite. The planner's previous layout had memory at the top and the
    # JSON 60+ lines later, which led the model to ignore the evidence.
    lines = ["Decide the next sweep action.", ""]
    lines += _phase_header(current_phase, best_family, best_f1 if best_family else None)
    lines += [
        f"TIME REMAINING: {int(time_remaining_seconds)} seconds.",
        "",
        phase_hint,
        "",
        f"ELIGIBLE LIST (pick from these only): {eligible_list}.",
        f"  UNTRIED this launch: {untried_str}",
        f"  TRIED this launch:   {tried_str}",
        "",
        "PER-FAMILY STATE (this launch):",
        table,
        "",
    ]
    if prior_launch_memory:
        lines.extend([
            "CROSS-LAUNCH MEMORY (evidence you MUST cite in your reason):",
            prior_launch_memory,
            "",
            "Use this evidence: cite ONE concrete fact from the memory above "
            "(an F1 number, a prior verdict, a spec choice that worked or "
            "didn't) in your 'reason' field, then state what picking this "
            "family next will test or learn. Generic phrases like 'untried "
            "family' or 'good potential' do NOT count as evidence.",
            "",
        ])
    lines += [
        "Return ONE JSON object on a single line:",
        '{"action":"try_family","family_key":"<key>","reason":"<text>"}',
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
    cost_estimates: dict[str, int] | None = None,
) -> tuple[DecisionAction, str | None, str]:
    """Validate the planner's JSON; fall back to safe defaults when it lies."""
    # Fallback now actually picks the CHEAPEST eligible family (matching the
    # "using cheapest eligible" message), not eligible[0] which was just the
    # first key in the FAMILY_MODULES dict (always 'bertweet') and silently
    # made every fallback land on the same expensive family.
    if eligible:
        if cost_estimates:
            fallback_family = min(
                eligible,
                key=lambda k: (cost_estimates.get(k, 10_000), k),
            )
        else:
            fallback_family = eligible[0]
    else:
        fallback_family = None
    if parsed is None:
        return ("try_family", fallback_family, "fallback: malformed planner JSON") if fallback_family else (
            "stop",
            None,
            "fallback: malformed planner JSON and no eligible families remain",
        )

    action = str(parsed.get("action", "")).strip()
    reason = str(parsed.get("reason", "")).strip() or "(no reason given)"
    family_key = parsed.get("family_key")
    # Normalize the LLM's family_key to the canonical key. Handles BOTH:
    #   - case mismatch ("BoW_advanced" → "bow_advanced")
    #   - separator drop  ("EmbeddingDL" → "embedding_dl")
    # Stripping underscores+lowercasing on both sides makes the lookup
    # robust to whichever capitalization/separator style the LLM emits.
    def _canon(s: str) -> str:
        return str(s).strip().replace("_", "").lower()
    if family_key:
        lookup = {_canon(k): k for k in list(family_state.keys()) + list(eligible)}
        family_key = lookup.get(_canon(family_key), str(family_key).strip())
    else:
        family_key = None

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
    prior_launch_memory: str = "",
    current_phase: str = "A",
) -> SweepDecision:
    """Ask the planner LLM for the next sweep decision.

    `llm` only needs a `.respond(system, user)` method that returns a string
    (the OllamaClient interface). We do not import OllamaClient here so this
    module stays trivial to unit-test with a stub.

    `prior_launch_memory` is an optional pre-formatted block (from
    ShortTermMemory.planner_summary()) that gets injected into the prompt so
    the planner sees aggregate F1s from past launches.
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
        prior_launch_memory=prior_launch_memory,
        current_phase=current_phase,
    )

    raw_response = ""
    parsed: dict[str, Any] | None = None
    try:
        # temp=0.4 for the planner: the prompt asks for evidence-cited reasoning
        # which is structured-but-creative output. At temp=0.2 the LLM produced
        # tautological reasons ("untried family") despite the prompt asking for
        # cited F1s. Slight bump to give it room to produce real reasoning while
        # keeping JSON output stable. Spec proposers run at 0.5; code-gen and
        # repair stay at 0.2.
        raw_response = llm.respond(planner_system_prompt, prompt, temperature=0.4) or ""
        parsed = _parse_planner_response(raw_response)
    except Exception as exc:  # noqa: BLE001 - planner is best-effort
        raw_response = f"[planner-llm-error] {exc}"

    action, family_key, reason = _coerce_decision(
        parsed=parsed,
        eligible=eligible,
        family_state=family_state,
        success_recorded=success_recorded,
        cost_estimates=cost_estimates,
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
