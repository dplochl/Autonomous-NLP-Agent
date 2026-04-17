"""Prompt rendering utilities for Agent_3."""

from __future__ import annotations

import json
import os
from typing import Any


TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _load_template(name: str) -> str:
    with open(os.path.join(TEMPLATE_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def render_family_prompt(
    module: object,
    spec: dict[str, Any],
    data_context: str,
    history_summary: str,
    trial_summary: str,
) -> str:
    template = _load_template(module.get_template_name())
    values = {
        "family": getattr(module, "FAMILY", "Unknown"),
        "family_note": module.get_arch_prompt(),
        "spec_json": json.dumps(spec, indent=2),
        "data_context": data_context.strip(),
        "history_summary": history_summary.strip(),
        "trial_summary": trial_summary.strip(),
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered
