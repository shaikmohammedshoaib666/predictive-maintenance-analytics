"""Anomaly scatter plot with Isolation Forest highlighting."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.graphs.layout import apply_readable_layout
from src.ml.anomaly_detector import AnomalyDetector


def create_anomaly_scatter(
    df: pd.DataFrame,
    x_axis: str = "temperature",
    y_metric: str = "vibration",
    machine_filter: list | None = None,
    title: str | None = None,
    anomaly_detector: AnomalyDetector | None = None,
) -> go.Figure:
    """Create scatter plot with anomalies highlighted."""
    plot_df = df.copy()
    if machine_filter and "machine_id" in plot_df.columns:
        plot_df = plot_df[plot_df["machine_id"].isin(machine_filter)]

    numeric_cols = plot_df.select_dtypes(include="number").columns.tolist()
    if x_axis not in numeric_cols:
        x_axis = numeric_cols[0] if numeric_cols else plot_df.columns[0]
    if y_metric not in numeric_cols:
        y_metric = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]

    detector = anomaly_detector or AnomalyDetector()
    if not detector.is_fitted:
        detector.fit(plot_df)
    labels = detector.predict(plot_df)
    plot_df = plot_df.copy()
    plot_df["anomaly"] = ["Anomaly" if l == -1 else "Normal" for l in labels]

    color_col = "machine_id" if "machine_id" in plot_df.columns else None
    fig = px.scatter(
        plot_df,
        x=x_axis,
        y=y_metric,
        color="anomaly",
        symbol=color_col,
        title=title or f"Anomaly Scatter: {x_axis} vs {y_metric}",
        color_discrete_map={"Normal": "#2ecc71", "Anomaly": "#e74c3c"},
        opacity=0.7,
    )
    fig.update_traces(hovertemplate="%{x}<br>%{y}<br>%{fullData.name}<extra></extra>")
    return apply_readable_layout(
        fig,
        title or f"Anomaly scatter: {x_axis} vs {y_metric}",
        kind="default",
        height=450,
    )
