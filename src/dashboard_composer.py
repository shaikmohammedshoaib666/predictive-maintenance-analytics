"""Power BI-style reliability dashboard composer.

Tiles: pack KPIs, saved/live charts, insight cards, 3D twin, CAD (APS-gated).
CAD is a first-class tile that turns on when `APS_CLIENT_ID` / `APS_CLIENT_SECRET`
(and a URN) appear in env or Streamlit secrets — no code change after Render deploy.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any, Optional

import plotly.graph_objects as go

from src.aps_viewer import aps_available, aps_model_urn, build_viewer_html
from src.graphs.pack_kpis import create_asset_health_chart, create_pack_kpi_bars
from src.industry_packs import get_pack
from src.twin3d import RISK_COLORS, build_twin_html, normalize_risk

TILE_SPECS: tuple[dict[str, Any], ...] = (
    {"id": "kpis", "label": "KPI strip", "hint": "Pack health / mission reliability / fillage"},
    {"id": "charts", "label": "Charts", "hint": "Saved Plotly views + pack health bars"},
    {"id": "insights", "label": "Insights", "hint": "Inspect-this-week and pack cards"},
    {"id": "twin3d", "label": "3D twin", "hint": "Offline pack mesh, risk-colored hotspot"},
    {"id": "cad", "label": "CAD twin (APS)", "hint": "Real CAD when Render secrets are set"},
)

DEFAULT_TILES: tuple[str, ...] = tuple(t["id"] for t in TILE_SPECS)


def default_tile_state() -> dict[str, bool]:
    return {t["id"]: True for t in TILE_SPECS}


def cad_slot_status(urn: str = "") -> dict[str, Any]:
    """Runtime APS gate. Secrets can be added after deploy; this re-reads env each call."""
    ok, msg = aps_available()
    resolved = (urn or aps_model_urn() or "").strip()
    return {
        "credentials": ok,
        "urn": resolved,
        "ready": bool(ok and resolved),
        "message": msg,
        "needs": [] if ok else ["APS_CLIENT_ID", "APS_CLIENT_SECRET"]
        + ([] if resolved else ["APS_MODEL_URN or paste URN on CAD Twin"]),
    }


def cad_placeholder_html(*, status: dict[str, Any], height: int = 360) -> str:
    needs = ", ".join(status.get("needs") or ["APS_CLIENT_ID", "APS_CLIENT_SECRET"])
    return f"""
<div style="height:{int(height)}px;border-radius:12px;background:#101822;color:#e8eef5;
     font-family:system-ui,Segoe UI,sans-serif;padding:24px;box-sizing:border-box">
  <div style="font-size:13px;opacity:.7;letter-spacing:.04em">CAD TWIN · AUTODESK APS</div>
  <h3 style="margin:8px 0 12px">Waiting for credentials (post-deploy)</h3>
  <p style="opacity:.85;line-height:1.45">This tile is wired. Add Render env / secrets
  <code>{html.escape(needs)}</code> and reload — no app rebuild required. Status:
  {html.escape(str(status.get('message') or ''))}.</p>
  <p style="opacity:.7;font-size:13px">Until then the <b>3D twin</b> tile is the
  credential-free digital twin (pack mesh + risk color).</p>
</div>
"""


def _iframe(inner: str, height: int) -> str:
    escaped = html.escape(inner, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" style="width:100%;height:{int(height)}px;'
        'border:0;border-radius:12px;background:#0b1016" loading="lazy"></iframe>'
    )


def _kpi_strip_html(bundle: dict[str, Any]) -> str:
    cards = []
    for k in bundle.get("kpis") or []:
        sev = k.get("severity") or "info"
        color = {"high": "#e74c3c", "medium": "#f39c12", "low": "#27ae60"}.get(sev, "#5c6b73")
        cards.append(
            f"""<div style="flex:1;min-width:160px;background:#fff;border:1px solid #e6e9ee;
              border-radius:12px;padding:14px 16px">
              <div style="font-size:12px;color:#6c757d">{html.escape(str(k.get('label')))}</div>
              <div style="font-size:22px;font-weight:700;color:{color}">{html.escape(str(k.get('value')))}
                <span style="font-size:13px;font-weight:500">{html.escape(str(k.get('unit') or ''))}</span>
              </div>
            </div>"""
        )
    if not cards:
        cards.append('<div style="opacity:.7">Run Anomaly &amp; RUL to populate pack KPIs.</div>')
    title = html.escape(str(bundle.get("label") or "Pack"))
    sih = bundle.get("sih")
    sih_bit = f" · {html.escape(str(sih))}" if sih else ""
    return f"<h2>{title}{sih_bit}</h2><div style='display:flex;gap:12px;flex-wrap:wrap'>{''.join(cards)}</div>"


def _insights_html(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return "<p>No insight cards yet. Open <b>6. Insights</b> or run Anomaly &amp; RUL.</p>"
    parts = ["<h2>Insights</h2>"]
    for item in cards[:8]:
        sev = html.escape(str(item.get("severity") or "info"))
        parts.append(
            f"<div style='border-left:4px solid #1a1a2e;padding:8px 12px;margin:10px 0;background:#f8f9fa'>"
            f"<b>{html.escape(str(item.get('title') or ''))}</b> · <code>{sev}</code>"
            f"<div style='margin-top:4px'>{html.escape(str(item.get('message') or ''))}</div></div>"
        )
    return "".join(parts)


def compose_dashboard_html(
    *,
    tiles: Optional[list[str]] = None,
    pack_id: str = "plant_rotating",
    kpi_bundle: Optional[dict[str, Any]] = None,
    insight_cards: Optional[list[dict[str, Any]]] = None,
    chart_figs: Optional[list[go.Figure]] = None,
    twin_html: str = "",
    cad_html: str = "",
    cad_status: Optional[dict[str, Any]] = None,
    title: str = "Reliability dashboard",
) -> str:
    """Standalone HTML export (charts via Plotly CDN; 3D twin is inlined / offline)."""
    enabled = [t for t in (tiles or list(DEFAULT_TILES)) if t in DEFAULT_TILES]
    pack = get_pack(pack_id)
    bundle = kpi_bundle or {}
    status = cad_status or cad_slot_status()
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>",
        "<style>body{font-family:Segoe UI,system-ui,sans-serif;margin:24px;background:#f4f6f8;color:#1a1a2e}"
        ".wrap{max-width:1200px;margin:0 auto}.tile{background:#fff;border-radius:14px;padding:18px;"
        "margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}</style></head><body><div class='wrap'>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p>{html.escape(pack['label'])} · generated {html.escape(datetime.now().strftime('%Y-%m-%d %H:%M'))}</p>",
        f"<p style='opacity:.75'>{html.escape(pack['scope'])}</p>",
    ]
    if "kpis" in enabled:
        parts.append(f"<div class='tile'>{_kpi_strip_html(bundle)}</div>")
    if "charts" in enabled:
        parts.append("<div class='tile'><h2>Charts</h2>")
        if chart_figs:
            for i, fig in enumerate(chart_figs):
                parts.append(fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"pdm_chart_{i}"))
        else:
            parts.append("<p>No charts on the board yet. Generate a chart on <b>5. Charts</b>.</p>")
        parts.append("</div>")
    if "insights" in enabled:
        parts.append(f"<div class='tile'>{_insights_html(list(insight_cards or []))}</div>")
    if "twin3d" in enabled:
        parts.append("<div class='tile'><h2>3D twin</h2>")
        parts.append(_iframe(twin_html, 560) if twin_html else "<p>3D twin not available.</p>")
        parts.append("</div>")
    if "cad" in enabled:
        parts.append("<div class='tile'><h2>CAD twin (APS)</h2>")
        if status.get("ready") and cad_html:
            parts.append(_iframe(cad_html, 560))
        else:
            parts.append(cad_placeholder_html(status=status, height=280))
        parts.append("</div>")
    parts.append("</div></body></html>")
    return "\n".join(parts)


def board_twin_html(
    predictions: list[dict[str, Any]],
    pack_id: str,
    *,
    selected_id: Optional[str] = None,
    height: int = 480,
) -> str:
    pack = get_pack(pack_id)
    assets = [
        {
            "machine_id": str(p.get("machine_id", "asset")),
            "risk_level": normalize_risk(p.get("risk_level")),
            "predicted_rul_days": p.get("predicted_rul_days"),
        }
        for p in (predictions or [])
    ]
    return build_twin_html(
        assets,
        selected_id=selected_id,
        height=height,
        kind=pack["twin_kind"],
        hotspot=pack["hotspot"],
        pack_label=pack["label"],
    )


def board_pack_charts(bundle: dict[str, Any]) -> list[go.Figure]:
    """Always-on pack charts so the board is not empty before Charts is visited."""
    figs: list[go.Figure] = []
    if bundle.get("by_asset"):
        figs.append(create_asset_health_chart(bundle["by_asset"], title="Pack health by asset"))
    if bundle.get("kpis"):
        figs.append(create_pack_kpi_bars(bundle["kpis"], title="Pack KPIs"))
    return figs


def board_cad_html(
    *,
    token: str,
    urn: str,
    asset: str,
    risk: str,
    height: int = 480,
) -> str:
    return build_viewer_html(token, urn, asset=asset, risk=risk, height=height)


def risk_legend_html() -> str:
    bits = [f"<b>{k}</b> {v['hex']}" for k, v in RISK_COLORS.items() if k != "Unknown"]
    return " · ".join(bits)
