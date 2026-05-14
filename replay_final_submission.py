"""One-off helper: re-run the final-submission step using the last sweep's
overall_best.json + the winning family's best_train.py, without redoing the
sweep.

Useful when the agent finished the sweep, picked a winner, but the final
submission step failed (e.g. the LLM's own submission code raised) and you
just want to regenerate `submissions/best_overall_submission.csv`.

Run from the repo root:

    ./.venv/bin/python replay_final_submission.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src", "Agent_4"))

from agent import (  # noqa: E402
    prepare_final_submission_payload,
    execute_final_submission,
)


def main() -> int:
    runs_dir = os.path.join(HERE, "src", "Agent_4", "runs")
    overall_path = os.path.join(runs_dir, "overall_best.json")
    if not os.path.exists(overall_path):
        print(f"ERROR: {overall_path} does not exist. Run the agent first.")
        return 1

    with open(overall_path, "r", encoding="utf-8") as fh:
        overall = json.load(fh)

    # `prepare_final_submission_payload` expects a per-family summary (one
    # entry from `family_summaries`), not the top-level overall_best.json.
    # Pick the family-summary whose `family_key` matches the global winner.
    target_family = overall.get("best_family")
    target_run_idx = overall.get("best_run_index")
    best_summary = next(
        (s for s in overall.get("family_summaries", []) if s.get("family") == target_family),
        None,
    )
    if best_summary is None:
        print(f"ERROR: no family_summary in overall_best.json matches best_family={target_family!r}")
        return 1

    public_dir = os.path.join(HERE, "submissions")
    os.makedirs(public_dir, exist_ok=True)
    public_path = os.path.join(public_dir, "best_overall_submission.csv")

    print(f"Best family from sweep: {target_family} run {target_run_idx} "
          f"(F1={overall.get('best_metrics', {}).get('f1')})")
    print(f"Session:                {best_summary.get('session_name')}")
    print(f"Writing submission to:  {public_path}")
    print()

    payload, error = prepare_final_submission_payload(best_summary, public_path)
    if error:
        print(f"ERROR preparing payload: {error}")
        return 1

    # The llm arg is ignored by execute_final_submission, so pass None.
    code, result, repair_attempts = execute_final_submission(None, payload)

    print()
    print("=== Result ===")
    print(f"  success         : {result.get('success')}")
    print(f"  metrics         : {result.get('metrics')}")
    print(f"  timed_out       : {result.get('timed_out')}")
    print(f"  repair_attempts : {repair_attempts}")
    if result.get("stderr"):
        tail = "\n".join(result["stderr"].rstrip().splitlines()[-20:])
        print()
        print("=== stderr tail ===")
        print(tail)

    if os.path.exists(public_path):
        import csv
        with open(public_path, "r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["target"]] = counts.get(r["target"], 0) + 1
        print()
        print("=== Submission CSV ===")
        print(f"  path  : {public_path}")
        print(f"  rows  : {len(rows)}")
        for k in sorted(counts):
            pct = counts[k] / len(rows) * 100
            print(f"  target={k}: {counts[k]} ({pct:.1f}%)")

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
