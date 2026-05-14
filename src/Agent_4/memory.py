"""Persistent run memory for Agent_3."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


LOG_PATH = "agent3_log.json"
ROLLING_WINDOW = 20


def _safe_f1(record: dict[str, Any]) -> float:
    try:
        return float((record.get("metrics") or {}).get("f1", -1.0))
    except (TypeError, ValueError):
        return -1.0


class Agent3Memory:
    def __init__(self, persist: bool = True, log_path: str = LOG_PATH, load_existing: bool = False):
        self.persist = persist
        self.log_path = log_path
        self.load_existing = load_existing
        self.records: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.persist or not self.load_existing:
            self.records = []
            return
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                self.records = json.load(f)
            print(f"[Memory] Loaded {len(self.records)} Agent_3 runs from {self.log_path}")

    def _save(self) -> None:
        if not self.persist:
            return
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def add_run(
        self,
        family: str,
        run_name: str,
        run_index: int,
        spec: dict[str, Any],
        prompt_sent: str,
        code: str,
        result: dict[str, Any],
        analysis: str,
    ) -> None:
        record = {
            "id": len(self.records) + 1,
            "family": family,
            "run_name": run_name,
            "run_index": run_index,
            "spec": spec,
            "prompt_sent": prompt_sent,
            "code_generated": code,
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "metrics": result.get("metrics", {}),
            "success": result.get("success", False),
            "timed_out": result.get("timed_out", False),
            "dry_run_failed": result.get("dry_run_failed", False),
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
        }
        self.records.append(record)
        self._save()

    def best_for_family(self, family: str) -> dict[str, Any] | None:
        candidates = [
            record for record in self.records
            if record["family"] == family and record["success"] and record["metrics"].get("f1") is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=_safe_f1)

    def history_summary(self, family: str | None = None) -> str:
        records = self.records if family is None else [r for r in self.records if r["family"] == family]
        if not records:
            return "No prior runs."

        recent = records[-ROLLING_WINDOW:]
        lines = []
        for record in recent:
            metrics = record.get("metrics") or {}
            f1 = metrics.get("f1", "N/A")
            status = "OK" if record.get("success") else "FAILED"
            lines.append(
                f"- [{record['id']}] {record['run_name']} run {record['run_index']} | "
                f"F1={f1} | {status} | {record.get('analysis', '')[:100].replace(chr(10), ' ')}"
            )
        return "\n".join(lines)
