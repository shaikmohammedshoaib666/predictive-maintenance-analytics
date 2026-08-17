"""Time series line chart for sensor readings."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.graphs.layout import apply_readable_layout


def create_time_series_chart(
    df: pd.DataFrame,
    x_axis: str = "timestamp",
    y_metric: str = "temperature",
    machine_filter: list | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create an interactive time series line chart."""
    plot_df = df.copy()
    if machine_filter and "machine_id" in plot_df.columns:
        plot_df = plot_df[plot_df["machine_id"].isin(machine_filter)]

    if x_axis not in plot_df.columns:
        x_axis = plot_df.columns[0]
    if y_metric not in plot_df.columns:
        numeric_cols = plot_df.select_dtypes(include="number").columns.tolist()
        y_metric = numeric_cols[0] if numeric_cols else plot_df.columns[-1]

    color_col = "machine_id" if "machine_id" in plot_df.columns else None
    fig = px.line(
        plot_df,
        x=x_axis,
        y=y_metric,
        color=color_col,
        title=title or f"{y_metric.replace('_', ' ').title()} Over Time",
        labels={x_axis: x_axis.replace("_", " ").title(), y_metric: y_metric.replace("_", " ").title()},
    )
    fig.update_traces(hovertemplate="%{x}<br>%{y}<extra></extra>")
    return apply_readable_layout(
        fig,
        title or f"{y_metric.replace('_', ' ').title()} Over Time",
        kind="line",
        height=450,
    )
