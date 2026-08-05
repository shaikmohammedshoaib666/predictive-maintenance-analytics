"""Dashboard builder — compose saved graphs into a final dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import plotly.graph_objects as go
import streamlit as st


def save_graph_to_folder(
    graph_id: str,
    fig: go.Figure,
    graph_type: str,
    config_params: dict[str, Any],
    graph_folder: dict | None = None,
) -> dict:
    """Save a generated graph to the session graph folder."""
    folder = graph_folder if graph_folder is not None else st.session_state.setdefault("graph_folder", {})
    entry_id = f"{graph_type}_{len(folder) + 1}_{datetime.now().strftime('%H%M%S')}"
    folder[entry_id] = {
        "id": entry_id,
        "graph_type": graph_type,
        "fig": fig,
        "config": config_params,
        "created_at": datetime.now().isoformat(),
        "title": fig.layout.title.text if fig.layout.title else graph_type,
    }
    st.session_state.graph_folder = folder
    return folder[entry_id]


def get_graph_folder() -> dict:
    """Return the current graph folder from session state."""
    return st.session_state.get("graph_folder", {})


def build_dashboard(selected_ids: list[str], columns: int = 2) -> list[go.Figure]:
    """Return list of figures for selected graph IDs."""
    folder = get_graph_folder()
    return [folder[g_id]["fig"] for g_id in selected_ids if g_id in folder]


def render_dashboard_preview(selected_ids: list[str], columns: int = 2):
    """Render selected graphs in a grid layout."""
    folder = get_graph_folder()
    if not selected_ids:
        st.info("Select graphs from the list to preview your dashboard.")
        return

    cols = st.columns(columns)
    for i, g_id in enumerate(selected_ids):
        if g_id not in folder:
            continue
        entry = folder[g_id]
        with cols[i % columns]:
            st.plotly_chart(entry["fig"], use_container_width=True, key=f"dash_{g_id}")


def export_dashboard_html(selected_ids: list[str]) -> str:
    """Export selected graphs as standalone HTML."""
    folder = get_graph_folder()
    html_parts = [
        "<html><head><title>Predictive Maintenance Dashboard</title>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "<style>body{font-family:Arial,sans-serif;margin:20px;}.chart{margin-bottom:40px;}</style>",
        "</head><body>",
        "<h1>Predictive Maintenance Dashboard</h1>",
    ]
    for g_id in selected_ids:
        if g_id in folder:
            fig = folder[g_id]["fig"]
            html_parts.append(f"<div class='chart'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>")
    html_parts.append("</body></html>")
    return "\n".join(html_parts)
