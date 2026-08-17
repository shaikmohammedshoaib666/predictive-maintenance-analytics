"""Plotly readability — same margin / tickangle / hover pattern as Forge v2.

Used only for PdM charts (sensor time, anomaly flags, risk-by-asset). Not a
generic 9-chart business dashboard.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _short_tick(val: Any, max_len: int = 18) -> str:
    text = "" if val is None or (isinstance(val, float) and pd.isna(val)) else str(val)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _unique_tick_vals(values: Any) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    if values is None:
        return out
    for v in list(values):
        key = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
        if key in seen:
            continue
        seen.add(key)
        out.append("" if key == "" else v)
    return out


def apply_readable_layout(
    fig: go.Figure,
    title: str = "",
    *,
    kind: str = "default",
    n_cats: int = 0,
    height: Optional[int] = None,
) -> go.Figure:
    """Pad axes so labels are not clipped; rotate category ticks on vertical bars."""
    margin = dict(l=48, r=28, t=56, b=56)
    resolved_height = height or 400
    if kind == "bar":
        margin = dict(l=56, r=28, t=56, b=140)
        resolved_height = height or 480
    elif kind == "bar_h":
        margin = dict(l=168, r=28, t=56, b=56)
        resolved_height = height or max(420, 30 * max(int(n_cats), 4) + 100)
    elif kind == "heatmap":
        margin = dict(l=96, r=28, t=56, b=120)
        resolved_height = height or 500

    layout_kwargs: dict[str, Any] = dict(
        margin=margin,
        height=resolved_height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="closest",
        template="plotly_white",
        bargap=0.25,
    )
    if title:
        layout_kwargs["title"] = title
    if kind in ("line", "default"):
        layout_kwargs["hovermode"] = "x unified"
    fig.update_layout(**layout_kwargs)

    tickfont = dict(size=12)
    if kind == "bar":
        x_kwargs: dict[str, Any] = dict(tickangle=-40, tickfont=tickfont, automargin=True)
        try:
            fig.update_xaxes(ticklabeloverflow="allow", **x_kwargs)
        except (ValueError, TypeError):
            fig.update_xaxes(**x_kwargs)
        fig.update_yaxes(tickfont=tickfont, automargin=True)
    elif kind == "bar_h":
        fig.update_yaxes(tickfont=tickfont, automargin=True)
        fig.update_xaxes(tickfont=tickfont, automargin=True)
    else:
        fig.update_xaxes(automargin=True, tickfont=tickfont)
        fig.update_yaxes(automargin=True, tickfont=tickfont)
    return fig


def style_bar_figure(
    fig: go.Figure,
    *,
    n_cats: Optional[int] = None,
    horizontal: Optional[bool] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Readable bar ticks: rotate / pad, truncate labels, keep full name on hover."""
    if fig is None:
        return fig
    bar_traces = [tr for tr in fig.data if getattr(tr, "type", None) == "bar"]
    if not bar_traces:
        return fig
    if horizontal is None:
        horizontal = any(getattr(tr, "orientation", None) == "h" for tr in bar_traces)

    labels: list[str] = []
    for tr in fig.data:
        seq = tr.y if horizontal else tr.x
        labels.extend(_unique_tick_vals(seq))
    labels = _unique_tick_vals(labels)
    shorts = [_short_tick(v) for v in labels]
    n = int(n_cats or len(labels) or 0)

    for tr in bar_traces:
        seq = tr.y if horizontal else tr.x
        if seq is None:
            continue
        full = ["" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v) for v in list(seq)]
        tr.customdata = np.array(full).reshape(-1, 1)
        if horizontal:
            tr.hovertemplate = "%{customdata[0]}<br>%{x}<extra></extra>"
        else:
            tr.hovertemplate = "%{customdata[0]}<br>%{y}<extra></extra>"

    resolved_title = title
    if resolved_title is None:
        try:
            resolved_title = str(fig.layout.title.text or "")
        except Exception:
            resolved_title = ""
    fig = apply_readable_layout(
        fig,
        resolved_title or "",
        kind="bar_h" if horizontal else "bar",
        n_cats=n,
    )
    if labels:
        tick_kwargs = dict(
            tickmode="array",
            tickvals=labels,
            ticktext=shorts,
            automargin=True,
            tickfont=dict(size=12),
        )
        if horizontal:
            fig.update_yaxes(**tick_kwargs)
        else:
            fig.update_xaxes(tickangle=-40, **tick_kwargs)
    return fig
