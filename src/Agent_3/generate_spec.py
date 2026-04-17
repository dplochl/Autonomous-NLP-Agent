"""Initial spec generation for Agent_3."""

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
) -> dict[str, Any]:
    default_spec = module.get_default_spec(run_name, submission_path)
    prompt = (
        f"Plan one reliable {module.FAMILY} experiment spec for the Kaggle Disaster Tweets task.\n\n"
        f"{module.get_spec_prompt()}\n\n"
        "Return one JSON object only.\n"
        f"Use these exact keys: {', '.join(default_spec.keys())}\n\n"
        f"Dataset context:\n{data_context}\n\n"
        f"Recent history:\n{history_summary}\n\n"
        "Default if unsure:\n"
        f"{json.dumps(default_spec, indent=2)}\n"
    )

    raw_response = llm.respond(SPEC_SYSTEM, prompt)
    raw_spec = extract_json_object(raw_response)
    spec, issues = validate_spec(
        raw_spec=raw_spec,
        default_spec=default_spec,
        ranges=module.get_spec_ranges(),
        fixed_keys=module.get_fixed_spec_keys(),
    )
    return {
        "prompt": prompt,
        "raw_response": raw_response,
        "raw_spec": raw_spec,
        "spec": spec,
        "issues": issues,
        "used_default": raw_spec is None,
    }
