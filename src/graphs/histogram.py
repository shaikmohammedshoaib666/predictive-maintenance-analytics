"""Distribution histogram for sensor metrics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.graphs.layout import apply_readable_layout


def create_distribution_histogram(
    df: pd.DataFrame,
    x_axis: str = "timestamp",
    y_metric: str = "temperature",
    machine_filter: list | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create histogram of selected metric."""
    plot_df = df.copy()
    if machine_filter and "machine_id" in plot_df.columns:
        plot_df = plot_df[plot_df["machine_id"].isin(machine_filter)]

    numeric_cols = plot_df.select_dtypes(include="number").columns.tolist()
    if y_metric not in numeric_cols:
        y_metric = numeric_cols[0] if numeric_cols else plot_df.columns[-1]

    color_col = "machine_id" if "machine_id" in plot_df.columns else None
    fig = px.histogram(
        plot_df,
        x=y_metric,
        color=color_col,
        nbins=30,
        barmode="overlay",
        title=title or f"Distribution of {y_metric.replace('_', ' ').title()}",
        opacity=0.75,
    )
    return apply_readable_layout(
        fig,
        title or f"Distribution of {y_metric.replace('_', ' ').title()}",
        kind="bar",
        height=450,
    )
