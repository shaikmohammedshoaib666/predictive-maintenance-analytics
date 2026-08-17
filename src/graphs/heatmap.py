"""Sensor correlation heatmap."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.graphs.layout import apply_readable_layout


def create_correlation_heatmap(
    df: pd.DataFrame,
    x_axis: str = "timestamp",
    y_metric: str = "temperature",
    machine_filter: list | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create correlation heatmap for numeric sensor columns."""
    plot_df = df.copy()
    if machine_filter and "machine_id" in plot_df.columns:
        plot_df = plot_df[plot_df["machine_id"].isin(machine_filter)]

    numeric_cols = plot_df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 numeric columns for heatmap", showarrow=False)
        return fig

    corr = plot_df[numeric_cols].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
        )
    )
    fig = apply_readable_layout(
        fig,
        title or "Sensor Correlation Heatmap",
        kind="heatmap",
        height=500,
    )
    fig.update_xaxes(tickangle=-40, title="")
    fig.update_yaxes(title="")
    return fig
