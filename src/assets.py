"""Thin per-asset view + work-order stubs (not a CMMS).

Derived from ``machine_id`` plus pack KPIs / MissionAdvisory. One open stub per
asset when the advisory is not OK — e.g. ``WO-UAV-03-OIL`` / Inspect oil / open.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import pandas as pd

from src.industry_packs import get_pack
from src.physics_rules import FAULT_NONE, advisory_action, mission_success_pct, valid_asset_id
from src.twin3d import normalize_risk

_WO_CODE = {
    "Inspect oil": "OIL",
    "Check cooling": "COOL",
    "Inspect rotating assembly": "VIB",
}


def go_nogo(action: Any, risk_level: Any = None) -> str:
    """Ground go / no-go from MissionAdvisory — not FADEC / GCS."""
    act = str(action or "").strip()
    risk = normalize_risk(risk_level)
    lower = act.lower()
    if not act or act == "OK":
        return "GO" if risk != "High" else "NO-GO"
    if lower.startswith("inspect") or lower.startswith("check"):
        return "NO-GO"
    if lower.startswith("monitor"):
        return "CAUTION"
    if risk == "High":
        return "NO-GO"
    if risk == "Medium":
        return "CAUTION"
    return "GO"


def work_order_code(action: Any) -> str:
    act = str(action or "").strip()
    if not act or act == "OK":
        return ""
    if act in _WO_CODE:
        return _WO_CODE[act]
    if act.lower().startswith("inspect"):
        return "INSP"
    if act.lower().startswith("check"):
        return "COOL" if "cool" in act.lower() else "INSP"
    if act.lower().startswith("monitor"):
        return "MON"
    return "INSP"


def work_order_id(machine_id: Any, action: Any) -> str:
    """``WO-UAV-03-OIL`` — empty when there is nothing to inspect."""
    mid = valid_asset_id(machine_id)
    code = work_order_code(action)
    if not mid or not code:
        return ""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", mid).strip("-").upper()
    if not slug:
        return ""
    return f"WO-{slug}-{code}"


def _last_timestamp(df: Optional[pd.DataFrame], machine_id: str) -> str:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return ""
    if "machine_id" not in df.columns or "timestamp" not in df.columns:
        return ""
    try:
        grp = df[df["machine_id"].astype(str) == str(machine_id)]
        if grp.empty:
            return ""
        ts = pd.to_datetime(grp["timestamp"], errors="coerce").max()
        if pd.isna(ts):
            return ""
        return pd.Timestamp(ts).isoformat()
    except Exception:
        return ""


def build_asset_rows(
    df: Optional[pd.DataFrame] = None,
    *,
    pack_id: Optional[str] = None,
    predictions: Optional[list[dict[str, Any]]] = None,
    by_asset: Optional[list[dict[str, Any]]] = None,
    advisory_items: Optional[list[dict[str, Any]]] = None,
    physics_rows: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """One row per asset. Additive — does not delete existing tables."""
    pack = get_pack(pack_id)
    pred_map = {valid_asset_id(p.get("machine_id")): p for p in (predictions or [])}
    pred_map.pop("", None)
    health_map = {valid_asset_id(a.get("machine_id")): a for a in (by_asset or [])}
    health_map.pop("", None)
    adv_map = {valid_asset_id(a.get("machine_id")): a for a in (advisory_items or [])}
    adv_map.pop("", None)
    phys_map = {valid_asset_id(a.get("machine_id")): a for a in (physics_rows or [])}
    phys_map.pop("", None)

    ids: list[str] = []
    seen: set[str] = set()
    sources: list[Any] = []
    if df is not None and isinstance(df, pd.DataFrame) and "machine_id" in df.columns:
        sources.append(df["machine_id"].tolist())
    sources.extend([pred_map, health_map, adv_map, phys_map])
    for source in sources:
        seq = source.keys() if isinstance(source, dict) else source
        for mid in seq:
            key = valid_asset_id(mid)
            if key and key not in seen:
                seen.add(key)
                ids.append(key)
    ids.sort()

    rows: list[dict[str, Any]] = []
    for mid in ids:
        pred = pred_map.get(mid) or {}
        health = health_map.get(mid) or {}
        adv = adv_map.get(mid) or {}
        phys = phys_map.get(mid) or {}
        risk = normalize_risk(pred.get("risk_level") or health.get("risk_level"))
        hi = health.get("health_index")
        try:
            mission = health.get("mission_success_pct")
            if mission is None and hi is not None:
                mission = mission_success_pct(hi)
        except Exception:
            mission = None
        action = adv.get("action") or advisory_action(
            risk_level=risk,
            physics_fault=phys.get("physics_fault") or FAULT_NONE,
        )
        wo_id = work_order_id(mid, action)
        decision = go_nogo(action, risk)
        wo = None
        if wo_id:
            wo = {
                "id": wo_id,
                "title": action,
                "status": "open",
                "asset": mid,
            }
        rows.append(
            {
                "machine_id": mid,
                "pack_id": pack["id"],
                "pack": pack["short"],
                "last_timestamp": _last_timestamp(df, mid),
                "risk_level": risk,
                "health_index": health.get("health_index"),
                "mission_success_pct": mission,
                "advisory": action,
                "go_nogo": decision,
                "work_order_id": wo_id,
                "work_order": wo,
                "physics_fault": phys.get("physics_fault") or FAULT_NONE,
            }
        )
    order = {"NO-GO": 0, "CAUTION": 1, "GO": 2}
    rows.sort(key=lambda r: (order.get(str(r.get("go_nogo")), 9), str(r.get("health_index") or 99), r["machine_id"]))
    return rows


def assets_table(rows: Optional[list[dict[str, Any]]] = None) -> pd.DataFrame:
    """Flat table for Dashboard / export (work-order stub columns, not a nested CMMS)."""
    flat: list[dict[str, Any]] = []
    for row in rows or []:
        wo = row.get("work_order") or {}
        flat.append(
            {
                "asset": row.get("machine_id"),
                "pack": row.get("pack"),
                "last_timestamp": row.get("last_timestamp") or "",
                "risk": row.get("risk_level"),
                "health": row.get("health_index"),
                "mission_%": row.get("mission_success_pct"),
                "go_nogo": row.get("go_nogo"),
                "advisory": row.get("advisory"),
                "work_order": wo.get("id") if wo else (row.get("work_order_id") or ""),
                "wo_title": (wo.get("title") if wo else "") or "",
                "wo_status": (wo.get("status") if wo else "") or "",
            }
        )
    return pd.DataFrame(flat)


def assets_export_json(rows: Optional[list[dict[str, Any]]] = None) -> str:
    return json.dumps({"version": 1, "assets": list(rows or [])}, indent=2, default=str) + "\n"
