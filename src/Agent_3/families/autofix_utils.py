"""Shared generated-code autofixes for Agent_3 family hooks."""

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
