"""DWDM concepts helpers + safe SQL sandbox over registered DataFrames."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

DWDM_CONCEPTS = [
    {
        "concept": "ETL (Extract-Transform-Load)",
        "description": "Normalize schema, cast types, impute missing values before analytics.",
        "in_app": "Upload & Clean industrial pipeline",
    },
    {
        "concept": "Data Integration / Joins",
        "description": "Combine facts from multiple tables (sensors + maintenance + cost).",
        "in_app": "Data Integration page (INNER/LEFT/RIGHT/OUTER)",
    },
    {
        "concept": "Data Cleaning / Quality",
        "description": "19-stage checks: nulls, outliers, drift, association rules, domain physics.",
        "in_app": "Quality report after Clean",
    },
    {
        "concept": "Binning / Discretization",
        "description": "Bucket continuous sensors into ordinal bins for mining rules.",
        "in_app": "DWDM binning in cleaner (*_bin columns)",
    },
    {
        "concept": "Smoothing",
        "description": "Rolling averages reduce sensor noise for trend detection.",
        "in_app": "*_smooth columns from industrial clean",
    },
    {
        "concept": "Association Rule Mining",
        "description": "Find co-occurring HIGH sensor baskets (Apriori-style).",
        "in_app": "Quality check #17 ASSOCIATION RULE MINING",
    },
    {
        "concept": "Outlier / Anomaly Mining",
        "description": "Z-score, IQR, Isolation Forest, DBSCAN, KMeans distance.",
        "in_app": "Quality checks #5–#9 + ML Predictions",
    },
    {
        "concept": "Concept Drift",
        "description": "PCA variance shift early vs late windows flags changing regimes.",
        "in_app": "Quality check #15 PCA / CONCEPT DRIFT",
    },
    {
        "concept": "OLAP-style Aggregation",
        "description": "GROUP BY machine/time for KPIs (avg vibration, failure risk).",
        "in_app": "SQL Lab SELECT … GROUP BY",
    },
    {
        "concept": "SQL Querying",
        "description": "Declarative analysis over tables registered in-memory.",
        "in_app": "SQL Lab (DuckDB or pandas fallback)",
    },
]


def apply_dwdm_transforms(
    df: pd.DataFrame,
    *,
    bin_cols: list[str] | None = None,
    smooth_cols: list[str] | None = None,
    normalize_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Interactive DWDM transforms for the SQL/DWDM lab page."""
    out = df.copy()
    log: list[str] = []
    nums = list(out.select_dtypes(include=[np.number]).columns)

    for c in bin_cols or []:
        if c in nums:
            try:
                out[f"{c}_dwdm_bin"] = pd.qcut(out[c], q=4, duplicates="drop").astype(str)
                log.append(f"binned {c} → {c}_dwdm_bin")
            except Exception as exc:
                log.append(f"bin failed {c}: {exc}")

    for c in smooth_cols or []:
        if c in nums:
            out[f"{c}_dwdm_smooth"] = out[c].rolling(window=min(12, max(3, len(out) // 20)), min_periods=1).mean()
            log.append(f"smoothed {c} → {c}_dwdm_smooth")

    for c in normalize_cols or []:
        if c in nums:
            s = out[c]
            out[f"{c}_z"] = (s - s.mean()) / (s.std() + 1e-9)
            log.append(f"z-normalized {c} → {c}_z")

    return out, log


_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|COPY|PRAGMA|EXPORT|IMPORT|INSTALL|LOAD)\b",
    re.IGNORECASE,
)


def run_sql(query: str, tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    """
    Execute read-only SQL against named DataFrames.
    Prefers DuckDB; falls back to a tiny pandas SELECT/WHERE/LIMIT parser for demos.
    """
    q = (query or "").strip().rstrip(";")
    if not q:
        raise ValueError("Empty SQL")
    if _FORBIDDEN_SQL.search(q):
        raise ValueError("Only read-only SELECT queries are allowed")

    try:
        import duckdb

        con = duckdb.connect(database=":memory:")
        for name, df in tables.items():
            safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
            con.register(safe, df)
        result = con.execute(q).df()
        return result, "duckdb"
    except ImportError:
        pass
    except Exception as exc:
        # fall through to pandas mini-parser for simple cases
        last_err = str(exc)
    else:
        last_err = ""

    # Minimal fallback: SELECT * FROM table [LIMIT n]
    m = re.match(
        r"SELECT\s+\*\s+FROM\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:LIMIT\s+(\d+))?$",
        q,
        re.IGNORECASE,
    )
    if m:
        name, lim = m.group(1), m.group(2)
        if name not in tables:
            # try sanitized match
            matches = [k for k in tables if re.sub(r"[^A-Za-z0-9_]", "_", k) == name]
            if not matches:
                raise KeyError(f"Unknown table '{name}'. Available: {list(tables)}")
            name = matches[0]
        out = tables[name]
        if lim:
            out = out.head(int(lim))
        return out.copy(), "pandas-fallback"

    raise RuntimeError(
        "Install duckdb for full SQL (`pip install duckdb`), or use "
        "`SELECT * FROM table_name LIMIT 100`. "
        + (f"DuckDB error: {last_err}" if last_err else "")
    )


def default_sql_examples(table_names: list[str]) -> list[str]:
    if not table_names:
        return ["-- Upload/clean data first"]
    t = re.sub(r"[^A-Za-z0-9_]", "_", table_names[0])
    return [
        f"SELECT * FROM {t} LIMIT 20",
        f"SELECT machine_id, AVG(temperature) AS avg_temp, AVG(vibration) AS avg_vib "
        f"FROM {t} GROUP BY machine_id ORDER BY avg_vib DESC",
        f"SELECT machine_id, COUNT(*) AS n, AVG(failure_within_days) AS avg_rul "
        f"FROM {t} GROUP BY machine_id",
    ]
