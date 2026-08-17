"""Risk-by-asset bar chart from RUL predictions."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.graphs.layout import apply_readable_layout, style_bar_figure


def create_risk_by_asset_chart(
    df: pd.DataFrame | None = None,
    x_axis: str = "machine_id",
    y_metric: str = "predicted_rul_days",
    machine_filter: list | None = None,
    title: str | None = None,
    predictions: Optional[list[dict[str, Any]]] = None,
    **_kwargs,
) -> go.Figure:
    """Horizontal bars: remaining useful life (days) colored by risk level."""
    rows = list(predictions or [])
    if not rows and df is not None and "machine_id" in df.columns and "failure_within_days" in df.columns:
        latest = df.sort_values(
            "timestamp" if "timestamp" in df.columns else df.columns[0]
        ).groupby("machine_id", as_index=False).tail(1)
        for _, rec in latest.iterrows():
            rul = float(pd.to_numeric(rec.get("failure_within_days"), errors="coerce") or 0)
            rows.append(
                {
                    "machine_id": str(rec["machine_id"]),
                    "predicted_rul_days": max(1, round(rul)),
                    "risk_level": "High" if rul <= 7 else ("Medium" if rul <= 14 else "Low"),
                }
            )

    if not rows:
        fig = go.Figure()
        fig.add_annotation(
            text="Train Anomaly & RUL first to plot risk by asset.",
            showarrow=False,
        )
        return apply_readable_layout(fig, title or "Risk by asset", kind="bar_h", n_cats=1)

    plot_df = pd.DataFrame(rows)
    if machine_filter:
        plot_df = plot_df[plot_df["machine_id"].isin([str(m) for m in machine_filter])]
    plot_df = plot_df.sort_values("predicted_rul_days")

    fig = px.bar(
        plot_df,
        x="predicted_rul_days",
        y="machine_id",
        color="risk_level",
        orientation="h",
        title=title or "Risk by asset (predicted RUL, days)",
        color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"},
        labels={
            "predicted_rul_days": "Predicted RUL (days)",
            "machine_id": "Asset",
            "risk_level": "Risk",
        },
    )
    return style_bar_figure(
        fig,
        n_cats=len(plot_df),
        horizontal=True,
        title=title or "Risk by asset (predicted RUL, days)",
    )
