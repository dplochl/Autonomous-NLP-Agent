"""Generic spec validation for prompt-first family hooks."""

from __future__ import annotations

from typing import Any


def _coerce_like(default: Any, value: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, list):
        return value if isinstance(value, list) else default
    return str(value).strip() if value is not None else default


def validate_spec(
    raw_spec: dict[str, Any] | None,
    default_spec: dict[str, Any],
    ranges: dict[str, tuple[float, float]] | None = None,
    fixed_keys: set[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    spec = dict(default_spec)
    issues: list[str] = []
    ranges = ranges or {}
    fixed_keys = fixed_keys or set()

    if not isinstance(raw_spec, dict):
        return spec, ["Spec was not valid JSON; defaults were used."]

    for key, default in default_spec.items():
        if key in fixed_keys:
            spec[key] = default
            continue
        if key not in raw_spec:
            continue

        value = _coerce_like(default, raw_spec.get(key))
        if key in ranges and isinstance(value, (int, float)):
            low, high = ranges[key]
            if value < low or value > high:
                issues.append(f"{key} out of range; used default.")
                value = default
        spec[key] = value

    for key in fixed_keys:
        spec[key] = default_spec[key]

    return spec, issues
