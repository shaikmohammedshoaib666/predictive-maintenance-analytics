"""Rolling average trend chart."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import config
from src.graphs.layout import apply_readable_layout


def create_rolling_trend(
    df: pd.DataFrame,
    x_axis: str = "timestamp",
    y_metric: str = "temperature",
    machine_filter: list | None = None,
    title: str | None = None,
    window: int | None = None,
) -> go.Figure:
    """Create rolling average trend line with raw data overlay."""
    plot_df = df.copy()
    if machine_filter and "machine_id" in plot_df.columns:
        plot_df = plot_df[plot_df["machine_id"].isin(machine_filter)]

    window = window or config.ROLLING_WINDOW
    numeric_cols = plot_df.select_dtypes(include="number").columns.tolist()
    if y_metric not in numeric_cols:
        y_metric = numeric_cols[0] if numeric_cols else plot_df.columns[-1]

    fig = go.Figure()

    if "machine_id" in plot_df.columns:
        for machine in plot_df["machine_id"].unique():
            mdf = plot_df[plot_df["machine_id"] == machine].sort_values(
                x_axis if x_axis in plot_df.columns else plot_df.index
            )
            x_vals = mdf[x_axis] if x_axis in mdf.columns else mdf.index
            rolling = mdf[y_metric].rolling(window=min(window, len(mdf)), min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=rolling,
                    mode="lines",
                    name=f"{machine} (rolling)",
                    line=dict(width=2),
                )
            )
    else:
        plot_df = plot_df.sort_values(x_axis if x_axis in plot_df.columns else plot_df.index)
        x_vals = plot_df[x_axis] if x_axis in plot_df.columns else plot_df.index
        rolling = plot_df[y_metric].rolling(window=min(window, len(plot_df)), min_periods=1).mean()
        fig.add_trace(go.Scatter(x=x_vals, y=rolling, mode="lines", name="Rolling avg"))

    fig = apply_readable_layout(
        fig,
        title or f"Rolling Average Trend — {y_metric.replace('_', ' ').title()} (window={window})",
        kind="line",
        height=450,
    )
    fig.update_xaxes(title=x_axis.replace("_", " ").title() if x_axis in plot_df.columns else "Index")
    fig.update_yaxes(title=y_metric.replace("_", " ").title())
    return fig
