"""Multi-file data integration with SQL-style joins (not in Forge v2 — new here)."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

JOIN_TYPES = {
    "inner": "INNER JOIN — only matching keys in both tables",
    "left": "LEFT JOIN — all rows from left + matches from right",
    "right": "RIGHT JOIN — all rows from right + matches from left",
    "outer": "FULL OUTER JOIN — all rows from both tables",
}


def load_tabular_file(uploaded_file) -> pd.DataFrame:
    """Load csv/tsv/xlsx/json into a DataFrame from a Streamlit UploadedFile or path-like."""
    name = getattr(uploaded_file, "name", str(uploaded_file)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    if name.endswith(".json"):
        return pd.read_json(uploaded_file)
    if name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    df = pd.read_csv(uploaded_file)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def suggest_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """Intersect column names as candidate join keys."""
    common = sorted(set(left.columns) & set(right.columns))
    preferred = [c for c in common if c.lower() in {"machine_id", "asset_id", "id", "timestamp", "date"}]
    rest = [c for c in common if c not in preferred]
    return preferred + rest


def join_two(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str = "inner",
    on: Optional[list[str]] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    suffixes: tuple[str, str] = ("_l", "_r"),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    how = (how or "inner").lower()
    if how not in JOIN_TYPES:
        raise ValueError(f"Unsupported join type: {how}. Use one of {list(JOIN_TYPES)}")

    meta: dict[str, Any] = {
        "how": how,
        "left_rows": len(left),
        "right_rows": len(right),
    }
    if on:
        merged = pd.merge(left, right, how=how, on=on, suffixes=suffixes)
        meta["keys"] = on
    elif left_on and right_on:
        merged = pd.merge(left, right, how=how, left_on=left_on, right_on=right_on, suffixes=suffixes)
        meta["keys"] = [left_on, right_on]
    else:
        keys = suggest_join_keys(left, right)
        if not keys:
            raise ValueError("No common columns to join on. Pick left_on/right_on explicitly.")
        merged = pd.merge(left, right, how=how, on=keys[:1], suffixes=suffixes)
        meta["keys"] = keys[:1]
        meta["auto_key"] = True

    meta["result_rows"] = len(merged)
    meta["result_cols"] = list(merged.columns)
    return merged, meta


def join_many(
    tables: dict[str, pd.DataFrame],
    steps: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Chain joins across 3+ named tables.

    steps example:
      [
        {"left": "sensors", "right": "maintenance", "how": "left", "on": ["machine_id"]},
        {"left": "_result", "right": "costs", "how": "inner", "on": ["machine_id"]},
      ]
    After step 1, the working frame is registered as "_result".
    """
    if not tables:
        raise ValueError("No tables provided")
    if not steps:
        raise ValueError("Provide at least one join step")

    working = tables[steps[0]["left"]].copy()
    registry = dict(tables)
    logs: list[dict[str, Any]] = []

    for i, step in enumerate(steps):
        right_name = step["right"]
        if right_name not in registry:
            raise KeyError(f"Unknown right table: {right_name}")
        left_df = working if i > 0 or step.get("left") == "_result" else registry[step["left"]]
        right_df = registry[right_name]
        how = step.get("how", "inner")
        on = step.get("on")
        left_on = step.get("left_on")
        right_on = step.get("right_on")
        working, meta = join_two(
            left_df,
            right_df,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
        )
        meta["step"] = i + 1
        meta["left_name"] = step.get("left", "_result")
        meta["right_name"] = right_name
        logs.append(meta)
        registry["_result"] = working

    return working, logs
