"""Column analysis, statistics, and data quality insights."""

from __future__ import annotations

from typing import Any

import pandas as pd


def analyze_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Generate comprehensive column-level insights."""
    insights: dict[str, Any] = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": {},
        "quality_score": 0.0,
        "recommendations": [],
    }

    total_quality = 0.0
    for col in df.columns:
        col_info: dict[str, Any] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(df[col].isna().mean() * 100, 2),
            "unique_count": int(df[col].nunique()),
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["type_category"] = "numeric"
            col_info["stats"] = {
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
                "median": round(float(df[col].median()), 4),
            }
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info["type_category"] = "datetime"
            col_info["stats"] = {
                "min": str(df[col].min()),
                "max": str(df[col].max()),
            }
        else:
            col_info["type_category"] = "categorical"
            top_values = df[col].value_counts().head(5).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        # Quality contribution: penalize nulls
        col_quality = 100 - col_info["null_pct"]
        total_quality += col_quality
        col_info["quality_score"] = round(col_quality, 1)
        insights["columns"][col] = col_info

    if len(df.columns) > 0:
        insights["quality_score"] = round(total_quality / len(df.columns), 1)

    # Recommendations
    if "machine_id" not in df.columns:
        insights["recommendations"].append("Add 'machine_id' column for per-machine analysis.")
    if "timestamp" not in df.columns:
        insights["recommendations"].append("Add 'timestamp' column for time-series analysis.")
    if "failure_within_days" not in df.columns:
        insights["recommendations"].append(
            "No 'failure_within_days' target found — RUL model will use synthetic labels from degradation patterns."
        )

    sensor_cols = [c for c in ["temperature", "vibration", "pressure", "rpm"] if c in df.columns]
    if len(sensor_cols) < 2:
        insights["recommendations"].append("Include multiple sensor columns for richer ML features.")

    return insights


def format_insights_markdown(insights: dict) -> str:
    """Format insights as markdown for display."""
    lines = [
        f"**Dataset:** {insights['shape']['rows']:,} rows × {insights['shape']['columns']} columns",
        f"**Overall Quality Score:** {insights['quality_score']}/100",
        "",
        "### Column Summary",
    ]
    for col, info in insights["columns"].items():
        lines.append(f"- **{col}** ({info['type_category']}): {info['null_pct']}% null, {info['unique_count']} unique")
        if "stats" in info and info["type_category"] == "numeric":
            s = info["stats"]
            lines.append(f"  - mean={s['mean']}, std={s['std']}, range=[{s['min']}, {s['max']}]")

    if insights["recommendations"]:
        lines.extend(["", "### Recommendations"])
        for rec in insights["recommendations"]:
            lines.append(f"- {rec}")

    return "\n".join(lines)
