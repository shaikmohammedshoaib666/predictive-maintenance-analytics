"""Box plot by machine for comparative analysis."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.graphs.layout import apply_readable_layout


def create_boxplot_by_machine(
    df: pd.DataFrame,
    x_axis: str = "machine_id",
    y_metric: str = "temperature",
    machine_filter: list | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create box plot grouped by machine."""
    plot_df = df.copy()
    if machine_filter and "machine_id" in plot_df.columns:
        plot_df = plot_df[plot_df["machine_id"].isin(machine_filter)]

    if "machine_id" not in plot_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="machine_id column required for box plot", showarrow=False)
        return fig

    numeric_cols = plot_df.select_dtypes(include="number").columns.tolist()
    if y_metric not in numeric_cols:
        y_metric = numeric_cols[0] if numeric_cols else "temperature"

    fig = px.box(
        plot_df,
        x="machine_id",
        y=y_metric,
        color="machine_id",
        title=title or f"{y_metric.replace('_', ' ').title()} by Machine",
        points="outliers",
    )
    fig.update_layout(showlegend=False)
    return apply_readable_layout(
        fig,
        title or f"{y_metric.replace('_', ' ').title()} by Machine",
        kind="bar",
        height=450,
    )
