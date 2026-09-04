"""CAD Twin identity helpers — asset risk tint + click-part sensor hints.

Isolation Forest scores an **asset**, not each Autodesk mesh. These helpers map
that asset risk to a whole-model color, and optionally emphasize a mapped
sensor when a clicked part name fuzzy-matches a keyword (temp / vibration /
rpm / oil / exhaust / cyl / crank). Aviation packs may add a small
``part_hints`` dict; there is no Rotax BOM and no per-bolt anomaly score.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from src.twin3d import normalize_risk

# Whole-model theming (APS ``setThemingColor``). Unknown → clear (no tint).
RISK_THEME_HEX: dict[str, str] = {
    "High": "#e74c3c",
    "Medium": "#f39c12",
    "Low": "#7a8c82",  # slight green-gray; keep default look, do not go plant-green
    "Unknown": "#7f8c8d",
}

# THREE.Vector4-style 0–1 RGBA for GuiViewer3D theming.
RISK_THEME_RGBA: dict[str, tuple[float, float, float, float]] = {
    "High": (0.91, 0.30, 0.24, 0.72),
    "Medium": (0.95, 0.61, 0.07, 0.58),
    "Low": (0.48, 0.55, 0.52, 0.32),
}

# Generic Autodesk node-name needles. Longer keys win. Canonical sensor names
# match ``sensor_map`` / pack extras when those columns exist on the frame.
_GENERIC_PART_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("exhaust", "egt"),
    ("egt", "egt"),
    ("cylinderhead", "cht"),
    ("cylhead", "cht"),
    ("cylinder", "cht"),
    ("cht", "cht"),
    ("cyl", "cht"),
    ("vibration", "vibration"),
    ("vibra", "vibration"),
    ("vib", "vibration"),
    ("temperature", "temperature"),
    ("temp", "temperature"),
    ("oiltemperature", "oil_temp"),
    ("oiltemp", "oil_temp"),
    ("oil_temp", "oil_temp"),
    ("oilpress", "oil_pressure"),
    ("oil_pressure", "oil_pressure"),
    ("oil", "oil_pressure"),
    ("crank", "rpm"),
    ("rpm", "rpm"),
)

_SKIP_SENSOR_COLS = {
    "machine_id",
    "timestamp",
    "is_anomaly",
    "anomaly_score",
    "anomaly_label",
    "failure_within_days",
    "predicted_rul_days",
    "risk_level",
    "rul_days",
    "label_source",
    "is_proxy",
    "message",
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def risk_theme_hex(risk: Optional[str]) -> str:
    """CSS hex for the whole-model tint / badge. Unknown is gray."""
    return RISK_THEME_HEX[normalize_risk(risk)]


def risk_theming_rgba(risk: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    """RGBA for ``viewer.setThemingColor``. ``None`` means clear theming (no risk)."""
    level = normalize_risk(risk)
    if level == "Unknown":
        return None
    return RISK_THEME_RGBA[level]


def _norm_part(name: str) -> str:
    return _NON_ALNUM.sub("", (name or "").lower())


def part_hints_for(pack_id: Optional[str] = None) -> dict[str, str]:
    """Aviation (and only aviation) ships a small keyword → sensor map."""
    if not pack_id:
        return {}
    try:
        from src.industry_packs import part_hints_for as _pack_hints

        return dict(_pack_hints(pack_id) or {})
    except Exception:
        return {}


def part_sensor_hint(
    part_name: str,
    *,
    pack_id: str = "",
    available: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Fuzzy-match an Autodesk node name to a mapped sensor.

    Returns the canonical (or available-column) name, or ``None`` when the
    part does not look like temp / vibration / rpm / oil / exhaust / cyl / crank
    (plus aviation ``part_hints``). Random bolts do not match.
    """
    blob = _norm_part(part_name)
    if not blob:
        return None

    avail_lower: Optional[dict[str, str]] = None
    if available is not None:
        avail_lower = {}
        for col in available:
            key = str(col).strip()
            if key:
                avail_lower[key.lower()] = key
        if not avail_lower:
            return None

    def _accept(canonical: str) -> Optional[str]:
        want = (canonical or "").strip()
        if not want:
            return None
        if avail_lower is None:
            return want
        low = want.lower()
        if low in avail_lower:
            return avail_lower[low]
        for col_low, real in avail_lower.items():
            if low in col_low or col_low in low:
                return real
        return None

    pairs: list[tuple[str, str]] = []
    for kw, sensor in part_hints_for(pack_id).items():
        pairs.append((kw, sensor))
    pairs.extend(_GENERIC_PART_KEYWORDS)
    pairs.sort(key=lambda kv: len(_norm_part(kv[0])), reverse=True)

    seen: set[str] = set()
    for kw, sensor in pairs:
        needle = _norm_part(kw)
        if not needle or needle in seen:
            continue
        seen.add(needle)
        if needle in blob:
            hit = _accept(sensor)
            if hit:
                return hit
    return None


def latest_asset_sensors(df: Any, machine_id: str) -> dict[str, Any]:
    """Latest numeric mapped-sensor values for an asset from the session frame."""
    if df is None:
        return {}
    try:
        if getattr(df, "empty", True):
            return {}
    except Exception:
        return {}

    frame = df
    mid = str(machine_id or "").strip()
    try:
        if mid and "machine_id" in frame.columns:
            sub = frame[frame["machine_id"].astype(str) == mid]
            if sub is not None and not getattr(sub, "empty", True):
                frame = sub
    except Exception:
        pass

    try:
        if "timestamp" in frame.columns:
            frame = frame.sort_values("timestamp")
    except Exception:
        pass

    try:
        row = frame.iloc[-1]
    except Exception:
        return {}

    out: dict[str, Any] = {}
    columns = list(getattr(frame, "columns", []))
    for col in columns:
        name = str(col)
        if name.lower() in _SKIP_SENSOR_COLS:
            continue
        try:
            val = row[col]
        except Exception:
            continue
        try:
            import pandas as pd

            if pd.isna(val):
                continue
            numeric = bool(pd.api.types.is_numeric_dtype(frame[col]))
        except Exception:
            numeric = isinstance(val, (int, float)) and not isinstance(val, bool)
        if not numeric:
            continue
        try:
            out[name] = float(val) if not isinstance(val, int) else val
        except (TypeError, ValueError):
            out[name] = val
    return out


def parse_cad_part_event(event: Any) -> Optional[dict[str, Any]]:
    """Normalize a Streamlit component / postMessage payload from GuiViewer3D."""
    if not isinstance(event, dict):
        return None
    if event.get("cleared"):
        return {"name": "", "dbId": None, "properties": {}, "cleared": True}
    name = str(event.get("name") or "").strip()
    dbid = event.get("dbId")
    try:
        dbid = int(dbid) if dbid is not None and dbid != "" else None
    except (TypeError, ValueError):
        dbid = None
    props = event.get("properties") if isinstance(event.get("properties"), dict) else {}
    clean_props = {str(k): str(v) for k, v in props.items() if k and v is not None and str(v) != ""}
    if not name and dbid is None:
        return None
    return {
        "name": name or (f"Node {dbid}" if dbid is not None else ""),
        "dbId": dbid,
        "properties": clean_props,
        "cleared": False,
    }


def build_part_card(
    *,
    part_name: str,
    asset_id: str,
    risk: str,
    rul_days: Any = None,
    sensors: Optional[dict[str, Any]] = None,
    hint: Optional[str] = None,
) -> dict[str, Any]:
    """Pure card model the CAD Twin page renders (unit-testable, no Streamlit)."""
    level = normalize_risk(risk)
    return {
        "part_name": (part_name or "").strip() or "—",
        "asset_id": str(asset_id or "asset"),
        "risk": level,
        "risk_hex": risk_theme_hex(level),
        "rul_days": rul_days,
        "sensors": dict(sensors or {}),
        "hint": hint or None,
        "honesty": "Isolation Forest scores the asset, not each CAD part.",
    }
