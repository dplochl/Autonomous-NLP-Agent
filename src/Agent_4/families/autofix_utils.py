"""Shared generated-code autofixes for Agent_4 family hooks."""

from __future__ import annotations

import re


def fix_text_column_fillna(code: str) -> str:
    """Avoid filling numeric dataframe columns with empty strings."""
    fixed = code
    fixed = re.sub(
        r"(train_df\s*=\s*pd\.read_csv\([^\n]+\))\.fillna\(['\"]{2}\)",
        r"\1",
        fixed,
    )
    fixed = re.sub(
        r"(test_df\s*=\s*pd\.read_csv\([^\n]+\))\.fillna\(['\"]{2}\)",
        r"\1",
        fixed,
    )
    fixed = re.sub(r"(?m)^\s*train_df\.fillna\(['\"]{2},\s*inplace=True\)\s*\n?", "", fixed)
    fixed = re.sub(r"(?m)^\s*test_df\.fillna\(['\"]{2},\s*inplace=True\)\s*\n?", "", fixed)
    if "for _df in (train_df, test_df):" in fixed:
        return fixed
    return re.sub(
        r"(?m)^(test_df\s*=\s*pd\.read_csv\([^\n]+\).*)$",
        (
            r"\1\n"
            "for _df in (train_df, test_df):\n"
            "    for _col in ('keyword', 'location', 'text'):\n"
            "        if _col in _df.columns:\n"
            "            _df[_col] = _df[_col].fillna('').astype(str)"
        ),
        fixed,
        count=1,
    )


def force_cpu_execution(code: str) -> str:
    """Rewrite common generated PyTorch device patterns to CPU-only execution."""
    fixed = code
    fixed = re.sub(
        r"device\s*=\s*torch\.device\([^\n]*\)",
        'device = torch.device("cpu")',
        fixed,
    )
    fixed = re.sub(
        r"if\s+torch\.cuda\.is_available\(\)\s*:\s*\n[ \t]*torch\.cuda\.manual_seed_all\([^\n]*\)\n?",
        "",
        fixed,
    )
    fixed = fixed.replace('.to("cuda")', '.to("cpu")')
    fixed = fixed.replace(".to('cuda')", '.to("cpu")')
    fixed = fixed.replace('.to("mps")', '.to("cpu")')
    fixed = fixed.replace(".to('mps')", '.to("cpu")')
    fixed = fixed.replace("torch.cuda.is_available()", "False")
    fixed = fixed.replace("torch.backends.mps.is_available()", "False")
    fixed = fixed.replace("pin_memory=True", "pin_memory=False")
    return fixed


def ensure_submission_makedirs(code: str, submission_var: str = "submission_path") -> str:
    """Insert directory creation immediately before the submission write, preserving indentation."""
    fixed = re.sub(
        rf"(?m)^[ \t]*os\.makedirs\(os\.path\.dirname\({re.escape(submission_var)}\),[ \t]*exist_ok=True\)[ \t]*\n?",
        "",
        code,
    )

    def add_before_to_csv(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}os.makedirs(os.path.dirname({submission_var}), exist_ok=True)\n"
            f"{match.group(0)}"
        )

    return re.sub(
        rf"(?m)^(?P<indent>[ \t]*)submission_df\.to_csv\({re.escape(submission_var)},\s*index=False\)",
        add_before_to_csv,
        fixed,
        count=1,
    )
