"""Numbered pipeline flow — in-page circles + Detect gating.

Does **not** change the left-nav page list. Joins / Map / Detect / Charts still
live on the existing radio. Circles are a status strip on numbered stages only.

Detect requires mapped canonical sensors (temperature / vibration / pressure / RPM).
Joins stay optional. Plant remains the default pack elsewhere.
"""

from __future__ import annotations

from typing import Any, Optional

from src.sensor_map import mapping_status

# Labels must match ``app.py`` ``pages`` keys for the numbered path (1–7).
# Side pages (3D Twin, CAD Twin, Live Connect, Email, Ask, SQL lab) stay off this strip.
NUMBERED_STAGES: tuple[tuple[str, str], ...] = (
    ("1. Upload & Clean", "Upload"),
    ("2. Joins", "Joins"),
    ("3. Map sensors", "Map"),
    ("4. Anomaly & RUL", "Detect"),
    ("5. Charts", "Charts"),
    ("6. Insights", "Insights"),
    ("7. Dashboard", "Board"),
)

NUMBERED_LABELS: tuple[str, ...] = tuple(label for label, _ in NUMBERED_STAGES)


def can_detect_anomalies(df: Any = None) -> bool:
    """Isolation Forest needs mapped canonical sensors. Physics still runs first when Detect fires.

    Unmapped numeric columns are not enough — Detect stays locked until Map sensors
    has produced at least one of temperature / vibration / pressure / RPM.
    """
    if df is None:
        return False
    try:
        if getattr(df, "empty", True):
            return False
    except Exception:
        return False
    status = mapping_status(df)
    return int(status.get("sensor_count") or 0) > 0


def pipeline_snapshot(
    *,
    data_loaded: bool = False,
    has_working_table: bool = False,
    table_count: int = 0,
    sensor_count: int = 0,
    has_predictions: bool = False,
    current_page: str = "",
) -> list[dict[str, Any]]:
    """Per-stage done / current / locked for the circle strip.

    Joins is optional (skip-ok once a working table exists). Detect stays locked
    until sensors are mapped. Insights / Dashboard light up after Detect.
    """
    mapped = int(sensor_count or 0) > 0
    working = bool(has_working_table)
    loaded = bool(data_loaded) or working
    joins_ready = int(table_count or 0) >= 2
    detected = bool(has_predictions)

    done = {
        "1. Upload & Clean": loaded,
        "2. Joins": joins_ready or (working and mapped),  # optional once you have a mapped table
        "3. Map sensors": mapped,
        "4. Anomaly & RUL": detected,
        "5. Charts": detected,
        "6. Insights": detected,
        "7. Dashboard": detected,
    }
    locked = {
        "1. Upload & Clean": False,
        "2. Joins": not loaded,
        "3. Map sensors": not loaded,
        "4. Anomaly & RUL": not mapped,
        "5. Charts": not working,
        "6. Insights": not detected,
        "7. Dashboard": not working,
    }
    rows: list[dict[str, Any]] = []
    for label, short in NUMBERED_STAGES:
        state = "done" if done[label] else ("locked" if locked[label] else "ready")
        if label == current_page:
            state = "current" if state != "locked" else "locked"
        rows.append(
            {
                "id": label,
                "short": short,
                "done": done[label],
                "locked": locked[label] and label != current_page,
                "current": label == current_page,
                "state": state,
            }
        )
    return rows


def pipeline_circles_html(
    stages: Optional[list[dict[str, Any]]] = None,
    *,
    current_page: str = "",
) -> str:
    """Compact HTML circle strip. Not a second nav — left radio is unchanged."""
    rows = list(stages or [])
    colors = {
        "done": ("#27ae60", "#fff"),
        "current": ("#1a1a2e", "#fff"),
        "ready": ("#dfe4ea", "#1a1a2e"),
        "locked": ("#f1f3f5", "#9aa4ad"),
    }
    bits = [
        '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;'
        'margin:0 0 12px 0;font-family:system-ui,Segoe UI,sans-serif;font-size:12px">'
    ]
    for i, row in enumerate(rows):
        bg, fg = colors.get(row.get("state") or "ready", colors["ready"])
        title = str(row.get("id") or "")
        short = str(row.get("short") or title)
        bits.append(
            f'<span title="{title}" style="display:inline-flex;align-items:center;gap:6px">'
            f'<span style="width:22px;height:22px;border-radius:50%;background:{bg};color:{fg};'
            f'display:inline-flex;align-items:center;justify-content:center;font-weight:700">'
            f"{i + 1}</span>"
            f'<span style="color:#1a1a2e">{short}</span></span>'
        )
        if i < len(rows) - 1:
            bits.append('<span style="color:#c5cbd3">→</span>')
    bits.append("</div>")
    hint = "Joins optional · Detect needs mapped sensors · Isolation Forest stays default ML"
    if current_page == "4. Anomaly & RUL":
        hint = "Detect is locked until timestamp/asset/sensors are mapped — physics Layer 1, then Isolation Forest."
    bits.append(f'<div style="font-size:12px;color:#6c757d;margin:-6px 0 10px 0">{hint}</div>')
    return "".join(bits)
