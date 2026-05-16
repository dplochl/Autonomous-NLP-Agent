"""Short-term cross-launch memory for Agent_4.

The agent already records every trial inside its own launch (`agent4_log.json`
and the per-trial `runs/` folders) but those are wiped or partitioned per
launch — by design, so each fresh run is independent.

This module adds a small, deliberately-bounded persistent memory that DOES
span launches. It stores the most recent N=20 trials across all launches in
`logs/agent4_short_term_memory.json`, with newest first. On every new launch
the agent loads this file, surfaces it to the sweep planner + spec proposer
as "prior context", and appends the new trials it produced — capped at 20
records so the file (and the prompts that consume it) stay small.

Why 20? Big enough that the agent sees patterns across 1-2 prior launches
(each launch produces ~10-15 trials), small enough that the prompt block
fits in well under a page and doesn't dilute the in-launch history.

Schema per record (~250 bytes):
    {
      "launch_id":      "20260514_192321",    # start time of the launch
      "timestamp":      "2026-05-14T19:30:17",
      "family":         "BoW_advanced",
      "family_key":     "bow_advanced",
      "spec":           {...tunable keys only, no submission_path/run_name...},
      "outcome":        "success",            # or code_gen_failed / timeout / etc.
      "f1":             0.7261,
      "accuracy":       0.7925,
      "best_threshold": 0.53,
      "wall_seconds":   110.4
    }
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any


# Rolling window size — bounded so the prompt context stays tight.
MEMORY_WINDOW = 20

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
    "agent4_short_term_memory.json",
)

# Keys we strip from the spec when persisting — these change every run and
# would just bloat the file without adding signal for future launches.
_VOLATILE_SPEC_KEYS = {"submission_path", "experiment_name", "dry_run_head"}


class ShortTermMemory:
    """Persistent rolling window of the most recent trials across launches."""

    def __init__(self, path: str = DEFAULT_PATH, window: int = MEMORY_WINDOW):
        self.path = path
        self.window = window
        # Include microseconds so two ShortTermMemory instances created
        # back-to-back (e.g. in a test, or two ./run.sh invocations in the
        # same second) get distinct launch_ids and the planner_summary's
        # "exclude current launch" filter works correctly.
        self.launch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.records: list[dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------ I/O
    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                self.records = data
        except (FileNotFoundError, json.JSONDecodeError):
            self.records = []

    def save(self) -> None:
        """Write the memory to disk, trimmed to the most recent `window` records."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        trimmed = self.records[: self.window]
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(trimmed, fh, indent=2)

    # ------------------------------------------------------------------ Write
    def add_trial(
        self,
        family: str,
        family_key: str,
        spec: dict[str, Any],
        outcome: str,
        metrics: dict[str, Any] | None = None,
        wall_seconds: float | None = None,
        hypothesis: str = "",
        analysis: str = "",
    ) -> None:
        """Append a new trial to memory (newest first).

        `hypothesis` is the one-line "what am I testing and why" string
        captured at spec-proposal time. `analysis` is the analyst's 3-5
        sentence verdict written after the trial. Both are persisted so
        future-launch planners can read the full hypothesis → conclusion
        chain.
        """
        clean_spec = {k: v for k, v in (spec or {}).items() if k not in _VOLATILE_SPEC_KEYS}
        m = metrics or {}
        record = {
            "launch_id": self.launch_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "family": family,
            "family_key": family_key,
            "hypothesis": (hypothesis or "").strip(),
            "spec": clean_spec,
            "outcome": outcome,
            "f1": m.get("f1"),
            "accuracy": m.get("accuracy"),
            "best_threshold": m.get("best_threshold"),
            "wall_seconds": float(wall_seconds) if wall_seconds is not None else None,
            "analysis": (analysis or "").strip(),
        }
        self.records.insert(0, record)
        # Trim eagerly so memory in-process never exceeds the cap.
        if len(self.records) > self.window:
            self.records = self.records[: self.window]

    # ------------------------------------------------------------------ Read
    def prior_trials_for_family(self, family_key: str) -> list[dict[str, Any]]:
        """All persisted trials for `family_key` from launches OTHER than this one.

        Used by the spec proposer so it can see what specs of this family were
        tried in past launches (and how they scored) without confusing them
        with the current launch's seeded_trials.
        """
        return [
            r for r in self.records
            if r.get("family_key") == family_key and r.get("launch_id") != self.launch_id
        ]

    def planner_summary(self) -> str:
        """Block for the sweep-planner prompt: every prior trial in memory
        with its family, spec, outcome and F1. The agent can then reason
        like a researcher: "this spec produced F1=X, that one F1=Y, here's
        what to try next."

        Layout:
            1. A short per-family aggregate (best / mean F1, ok/fail counts)
            2. A flat list of every record (newest first) with the spec inline

        Skips records from the current launch so the planner doesn't see
        its own in-progress trials twice.
        """
        prior = [r for r in self.records if r.get("launch_id") != self.launch_id]
        if not prior:
            return (
                "Prior-launch memory: none (this is the first launch on this "
                "machine, or memory is empty)."
            )

        # --- per-family aggregate ---
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in prior:
            by_family[r.get("family", "?")].append(r)

        n_launches = len({r["launch_id"] for r in prior})
        lines = [
            f"Prior-launch memory ({len(prior)} trials across "
            f"{n_launches} past launch{'es' if n_launches != 1 else ''}):"
        ]
        for fam in sorted(by_family, key=lambda f: -max((r.get("f1") or -1) for r in by_family[f])):
            recs = by_family[fam]
            successes = [r for r in recs if r.get("outcome") == "success" and isinstance(r.get("f1"), (int, float))]
            failures = [r for r in recs if r.get("outcome") != "success"]
            if successes:
                best = max(successes, key=lambda r: r["f1"])
                avg = sum(r["f1"] for r in successes) / len(successes)
                lines.append(
                    f"  - {fam:<14}  best F1={best['f1']:.4f}  mean F1={avg:.4f}  "
                    f"({len(successes)} ok, {len(failures)} fail)"
                )
            else:
                lines.append(
                    f"  - {fam:<14}  best F1=—       mean F1=—       "
                    f"(0 ok, {len(failures)} fail)"
                )

        # --- per-trial detail GROUPED BY FAMILY ---
        # The previous chronological interleave produced adjacent records like
        # "[ts] CNN F1=0.6720 / [ts] RoBERTa (code_gen_failed)" and the planner
        # LLM was conflating the CNN F1 onto the next RoBERTa decision. By
        # grouping per-trial detail under a family header, we make it
        # syntactically impossible for the LLM to grab a neighbouring family's
        # F1 — within a family block, every F1 belongs to that family.
        lines.append("")
        lines.append(
            "Per-trial detail, grouped by family (within each family, "
            "newest first). EVERY F1 in a block belongs to that family — "
            "do not borrow F1 numbers across blocks."
        )
        # Sort families the same way as the aggregate above (best F1 desc).
        for fam in sorted(by_family, key=lambda f: -max((r.get("f1") or -1) for r in by_family[f])):
            recs_sorted = sorted(
                by_family[fam],
                key=lambda r: r.get("timestamp", ""),
                reverse=True,
            )
            lines.append("")
            lines.append(f"--- {fam} trials ({len(recs_sorted)}) ---")
            for r in recs_sorted:
                ts = (r.get("timestamp") or "")[:19]
                outcome = r.get("outcome", "?")
                f1 = r.get("f1")
                # Tag every row with the family AGAIN so even line-by-line
                # reads can't mis-attribute.
                f1_str = (
                    f"{fam} F1={f1:.4f}" if isinstance(f1, (int, float))
                    else f"{fam} ({outcome}, no F1)"
                )
                spec_inline = _format_spec_inline(r.get("spec") or {})
                hyp = (r.get("hypothesis") or "").strip()
                lines.append(f"  [{ts}]  {f1_str}")
                if hyp:
                    lines.append(f"    hypothesis: {hyp}")
                lines.append(f"    spec      : {spec_inline}")

        return "\n".join(lines)


def _format_spec_inline(spec: dict[str, Any]) -> str:
    """Render a spec dict as a compact inline string for prompt injection.

    Examples:
        {'learning_rate': 1.5e-5, 'max_len': 128, 'epochs': 3}
            → "learning_rate=1.5e-05, max_len=128, epochs=3"
    """
    if not spec:
        return "(no spec)"
    parts: list[str] = []
    for key in sorted(spec.keys()):
        val = spec[key]
        if isinstance(val, float):
            # Scientific notation if very small/large, fixed otherwise.
            text = f"{val:.2e}" if abs(val) < 0.01 or abs(val) >= 10000 else f"{val:g}"
        elif isinstance(val, (list, tuple)):
            text = "[" + ",".join(str(x) for x in val) + "]"
        else:
            text = str(val)
        parts.append(f"{key}={text}")
    return ", ".join(parts)
