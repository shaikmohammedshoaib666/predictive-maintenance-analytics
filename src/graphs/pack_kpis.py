"""Pack KPI charts — health-by-asset bars plus a compact KPI strip helper."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.graphs.layout import apply_readable_layout, style_bar_figure


def create_asset_health_chart(
    by_asset: Optional[list[dict[str, Any]]] = None,
    *,
    title: str | None = None,
    **_kwargs,
) -> go.Figure:
    """Horizontal bars: pack health index (0–100) colored by risk."""
    rows = list(by_asset or [])
    if not rows:
        fig = go.Figure()
        fig.add_annotation(text="No pack health scores yet.", showarrow=False)
        return apply_readable_layout(fig, title or "Pack health by asset", kind="bar_h", n_cats=1)

    plot_df = pd.DataFrame(rows).sort_values("health_index")
    fig = px.bar(
        plot_df,
        x="health_index",
        y="machine_id",
        color="risk_level",
        orientation="h",
        title=title or "Pack health by asset",
        color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60", "Unknown": "#7f8c8d"},
        labels={"health_index": "Health index (0–100)", "machine_id": "Asset", "risk_level": "Risk"},
    )
    fig.update_xaxes(range=[0, 100])
    return style_bar_figure(fig, n_cats=len(plot_df), horizontal=True, title=title or "Pack health by asset")


def create_pack_kpi_bars(
    kpis: Optional[list[dict[str, Any]]] = None,
    *,
    title: str | None = None,
    **_kwargs,
) -> go.Figure:
    """Numeric pack KPIs as a simple bar chart (non-numeric values skipped)."""
    numeric: list[dict[str, Any]] = []
    for k in kpis or []:
        try:
            numeric.append({"label": k["label"], "value": float(k["value"]), "severity": k.get("severity") or "info"})
        except (TypeError, ValueError):
            continue
    if not numeric:
        fig = go.Figure()
        fig.add_annotation(text="No numeric pack KPIs to plot.", showarrow=False)
        return apply_readable_layout(fig, title or "Pack KPIs", kind="bar", n_cats=1)

    plot_df = pd.DataFrame(numeric)
    color_map = {"high": "#e74c3c", "medium": "#f39c12", "low": "#27ae60", "info": "#5c6b73"}
    fig = px.bar(
        plot_df,
        x="label",
        y="value",
        color="severity",
        title=title or "Pack KPIs",
        color_discrete_map=color_map,
    )
    return style_bar_figure(fig, n_cats=len(plot_df), horizontal=False, title=title or "Pack KPIs")
