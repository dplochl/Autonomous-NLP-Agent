"""Small JSON helpers for prompt-driven workflows."""

from __future__ import annotations

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start: idx + 1]
                try:
                    value = json.loads(chunk)
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    return None


def pretty_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)
