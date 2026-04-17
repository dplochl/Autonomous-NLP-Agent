"""Surgical repair helpers for Agent_3."""

from __future__ import annotations

import re

from json_utils import extract_json_object
from prompts import PATCH_REPAIR_SYSTEM
from sandbox import tail


def _find_flexible_span(code: str, target: str) -> tuple[int, int] | None:
    if not target:
        return None
    idx = code.find(target)
    if idx != -1:
        return idx, idx + len(target)
    pattern = re.escape(target.strip()).replace(r"\ ", r"[ \t]+").replace(r"\n", r"\s*")
    match = re.search(pattern, code, re.MULTILINE)
    return (match.start(), match.end()) if match else None


def _indent_for_span(code: str, start: int) -> str:
    line_start = code.rfind("\n", 0, start) + 1
    match = re.match(r"[ \t]*", code[line_start:start])
    return match.group(0) if match else ""


def _format_insert_content(code: str, start: int, content: str) -> str:
    stripped = content.strip("\n")
    if not stripped:
        return "\n"
    indent = _indent_for_span(code, start)
    return "\n".join(f"{indent}{line.lstrip()}" if line.strip() else "" for line in stripped.splitlines()) + "\n"


def apply_edit_plan(code: str, plan: dict | None) -> tuple[str, list[str]]:
    if not isinstance(plan, dict):
        return code, ["Repair plan was not valid JSON."]
    edits = plan.get("edits")
    if not isinstance(edits, list) or not edits:
        return code, ["Repair plan contained no edits."]

    updated = code
    errors: list[str] = []
    for idx, edit in enumerate(edits, start=1):
        if not isinstance(edit, dict):
            errors.append(f"Edit {idx} was not an object.")
            continue
        action = str(edit.get("action", "")).strip()
        target = str(edit.get("target", "")).strip()
        content = str(edit.get("content", ""))
        span = _find_flexible_span(updated, target)
        if action == "replace":
            if not span:
                errors.append(f"Edit {idx} target not found for replace.")
                continue
            start, end = span
            updated = updated[:start] + content + updated[end:]
        elif action == "insert_before":
            if not span:
                errors.append(f"Edit {idx} target not found for insert_before.")
                continue
            start, _ = span
            updated = updated[:start] + _format_insert_content(updated, start, content) + updated[start:]
        elif action == "insert_after":
            if not span:
                errors.append(f"Edit {idx} target not found for insert_after.")
                continue
            _, end = span
            prefix = "\n" if end < len(updated) and updated[end] != "\n" else ""
            updated = updated[:end] + prefix + _format_insert_content(updated, end, content) + updated[end:]
        else:
            errors.append(f"Edit {idx} has unsupported action: {action}")
    return updated, errors


def request_surgical_repair(
    llm,
    module: object,
    family: str,
    run_name: str,
    failed_code: str,
    stderr_text: str,
    stdout_text: str,
    attempt: int,
    max_attempts: int,
    extra_context: str = "",
) -> dict[str, str]:
    prompt = (
        f"Repair attempt {attempt}/{max_attempts} for family: {family}.\n"
        f"Keep submission path exactly: submissions/{run_name}_submission.csv\n\n"
        f"{module.get_repair_prompt()}\n\n"
        "Rules:\n"
        "- patch only the broken region\n"
        "- keep the architecture family fixed\n"
        "- keep working code unchanged\n"
        "- prefer 1-3 edits\n\n"
    )
    if extra_context:
        prompt += f"{extra_context}\n\n"
    prompt += (
        f"STDERR:\n{stderr_text}\n\n"
        f"{module.build_repair_hint(stderr_text)}"
        f"STDOUT (tail):\n{tail(stdout_text, 30)}\n\n"
        "FAILED CODE:\n```python\n"
        f"{failed_code}\n```\n"
    )
    raw_response = llm.respond(PATCH_REPAIR_SYSTEM, prompt)
    plan = extract_json_object(raw_response)
    fixed_code, errors = apply_edit_plan(failed_code, plan)
    return {
        "code": "" if errors else fixed_code,
        "raw_response": raw_response,
        "error": "; ".join(errors),
    }
