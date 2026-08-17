"""Anomaly flags over time — Isolation Forest labels on the sensor timeline."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from src.graphs.layout import apply_readable_layout
from src.ml.anomaly_detector import AnomalyDetector


def create_anomaly_flags_chart(
    df: pd.DataFrame,
    x_axis: str = "timestamp",
    y_metric: str = "vibration",
    machine_filter: list | None = None,
    title: str | None = None,
    anomaly_detector: Optional[AnomalyDetector] = None,
    **_kwargs,
) -> go.Figure:
    """Line of a sensor with red markers where Isolation Forest flagged an anomaly."""
    plot_df = df.copy()
    if machine_filter and "machine_id" in plot_df.columns:
        plot_df = plot_df[plot_df["machine_id"].isin(machine_filter)]

    if x_axis not in plot_df.columns:
        x_axis = "timestamp" if "timestamp" in plot_df.columns else plot_df.columns[0]
    numeric_cols = plot_df.select_dtypes(include="number").columns.tolist()
    if y_metric not in plot_df.columns:
        y_metric = numeric_cols[0] if numeric_cols else plot_df.columns[-1]

    detector = anomaly_detector or AnomalyDetector()
    if not detector.is_fitted:
        detector.fit(plot_df)
    labels = detector.predict(plot_df)
    plot_df = plot_df.copy()
    plot_df["_anomaly"] = labels == -1

    fig = go.Figure()
    if "machine_id" in plot_df.columns:
        for machine in plot_df["machine_id"].unique():
            mdf = plot_df[plot_df["machine_id"] == machine]
            fig.add_trace(
                go.Scatter(
                    x=mdf[x_axis],
                    y=mdf[y_metric],
                    mode="lines",
                    name=str(machine),
                    hovertemplate="%{x}<br>%{y}<br>" + str(machine) + "<extra></extra>",
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=plot_df[x_axis],
                y=plot_df[y_metric],
                mode="lines",
                name=y_metric,
                hovertemplate="%{x}<br>%{y}<extra></extra>",
            )
        )

    flagged = plot_df[plot_df["_anomaly"]]
    if not flagged.empty:
        fig.add_trace(
            go.Scatter(
                x=flagged[x_axis],
                y=flagged[y_metric],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#e74c3c", size=9, symbol="x"),
                hovertemplate="Anomaly<br>%{x}<br>%{y}<extra></extra>",
            )
        )

    fig = apply_readable_layout(
        fig,
        title or f"Anomaly flags — {y_metric.replace('_', ' ')}",
        kind="line",
        height=450,
    )
    fig.update_xaxes(title=x_axis.replace("_", " ").title())
    fig.update_yaxes(title=y_metric.replace("_", " ").title())
    return fig
