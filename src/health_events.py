"""Health-event digital thread — JSON records, not a second Render service.

One record per asset after physics + Detect: sensors snapshot, physics flags,
Isolation Forest / RUL / risk, MissionAdvisory, CAD twin region if any.
Download from Dashboard / Anomaly. Export-only (Streamlit has no extra HTTP
worker on start.sh).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from src.assets import work_order_id
from src.physics_rules import valid_asset_id
from src.cad_regions import sensor_region
from src.industry_packs import OPTIONAL_IF_SENSORS, get_pack
from src.physics_rules import (
    FLAG_ALERT,
    FLAG_FAULT,
    FLAG_LABEL,
    FLAG_RULE,
    FAULT_NONE,
    physics_fault_region,
)

SCHEMA = "pdm.health_event.v1"
REQUIRED_KEYS: tuple[str, ...] = (
    "schema",
    "asset",
    "time",
    "pack_id",
    "sensors",
    "physics",
    "iforest",
    "rul",
    "risk_level",
    "advisory",
    "twin_region",
)

_SENSOR_CANDIDATES = (
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    *OPTIONAL_IF_SENSORS,
)


def _iso(value: Any) -> str:
    if value is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(value: Any) -> Optional[float]:
    try:
        f = float(value)
        if pd.isna(f):
            return None
        return round(f, 4)
    except Exception:
        return None


def _latest_row(df: Optional[pd.DataFrame], machine_id: str) -> Optional[pd.Series]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if "machine_id" not in df.columns:
        return df.iloc[-1]
    grp = df[df["machine_id"].astype(str) == str(machine_id)]
    if grp.empty:
        return None
    if "timestamp" in grp.columns:
        grp = grp.sort_values("timestamp")
    return grp.iloc[-1]


def _sensor_snapshot(row: Optional[pd.Series]) -> dict[str, Any]:
    if row is None:
        return {}
    out: dict[str, Any] = {}
    for name in _SENSOR_CANDIDATES:
        if name in row.index:
            val = _num(row.get(name))
            if val is not None:
                out[name] = val
    return out


def _iforest_from_row(row: Optional[pd.Series]) -> dict[str, Any]:
    if row is None:
        return {"anomaly_score": None, "is_anomaly": None}
    score = _num(row.get("anomaly_score")) if "anomaly_score" in row.index else None
    flagged = None
    if "is_anomaly" in row.index:
        try:
            flagged = bool(row.get("is_anomaly"))
        except Exception:
            flagged = None
    return {"anomaly_score": score, "is_anomaly": flagged}


def twin_region_for(
    *,
    physics_fault: Any = "",
    driver_region: Any = "",
    driver_sensor: Any = "",
) -> str:
    region = physics_fault_region(physics_fault) or str(driver_region or "")
    if not region and driver_sensor:
        region = str(sensor_region(driver_sensor).get("region") or "")
    return region if region and region != "other" else (region or "")


def build_health_event(
    *,
    asset: Any,
    pack_id: Optional[str] = None,
    time: Any = None,
    row: Optional[pd.Series] = None,
    prediction: Optional[dict[str, Any]] = None,
    physics: Optional[dict[str, Any]] = None,
    advisory: str = "",
    driver: Optional[dict[str, Any]] = None,
    health_index: Any = None,
    mission_success_pct: Any = None,
) -> dict[str, Any]:
    """One digital-thread record. Always includes REQUIRED_KEYS."""
    mid = valid_asset_id(asset)
    pack = get_pack(pack_id)
    pred = dict(prediction or {})
    phys = dict(physics or {})
    drv = dict(driver or {})
    fault = phys.get(FLAG_FAULT) or FAULT_NONE
    if row is not None and not phys.get(FLAG_FAULT) and FLAG_FAULT in row.index:
        fault = row.get(FLAG_FAULT) or FAULT_NONE
    region = twin_region_for(
        physics_fault=fault,
        driver_region=drv.get("region"),
        driver_sensor=drv.get("sensor"),
    )
    ts = time
    if ts is None and row is not None and "timestamp" in row.index:
        ts = row.get("timestamp")
    action = advisory or ""
    event = {
        "schema": SCHEMA,
        "asset": mid or "asset",
        "time": _iso(ts),
        "pack_id": pack["id"],
        "pack": pack["short"],
        "sensors": _sensor_snapshot(row),
        "physics": {
            "alert": bool(phys.get(FLAG_ALERT) or (row.get(FLAG_ALERT) if row is not None and FLAG_ALERT in row.index else False)),
            "fault": str(fault or FAULT_NONE),
            "rule": str(phys.get(FLAG_RULE) or (row.get(FLAG_RULE) if row is not None and FLAG_RULE in row.index else "") or ""),
            "label": str(phys.get(FLAG_LABEL) or (row.get(FLAG_LABEL) if row is not None and FLAG_LABEL in row.index else "") or FAULT_NONE),
        },
        "iforest": _iforest_from_row(row),
        "rul": {
            "predicted_rul_days": pred.get("predicted_rul_days"),
            "risk_level": pred.get("risk_level") or "",
            "is_proxy": bool(pred.get("is_proxy")),
            "label_source": pred.get("label_source") or "",
        },
        "risk_level": pred.get("risk_level") or "",
        "advisory": action,
        "twin_region": region,
        "health_index": health_index,
        "mission_success_pct": mission_success_pct,
        "work_order_id": work_order_id(mid, action),
    }
    return event


def build_health_events(
    df: Optional[pd.DataFrame] = None,
    *,
    pack_id: Optional[str] = None,
    predictions: Optional[list[dict[str, Any]]] = None,
    physics_rows: Optional[list[dict[str, Any]]] = None,
    advisory_items: Optional[list[dict[str, Any]]] = None,
    drivers: Optional[dict[str, dict[str, Any]]] = None,
    by_asset: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Fleet digital thread after physics + Detect. Skips empty asset ids."""
    pred_map = {valid_asset_id(p.get("machine_id")): p for p in (predictions or [])}
    pred_map.pop("", None)
    phys_map = {valid_asset_id(p.get("machine_id")): p for p in (physics_rows or [])}
    phys_map.pop("", None)
    adv_map = {valid_asset_id(a.get("machine_id")): a for a in (advisory_items or [])}
    adv_map.pop("", None)
    health_map = {valid_asset_id(a.get("machine_id")): a for a in (by_asset or [])}
    health_map.pop("", None)
    driver_map = {valid_asset_id(k): v for k, v in (drivers or {}).items()}
    driver_map.pop("", None)

    ids: list[str] = []
    seen: set[str] = set()
    for source in (pred_map, phys_map, health_map, adv_map):
        for mid in source:
            if mid and mid not in seen:
                seen.add(mid)
                ids.append(mid)
    if df is not None and isinstance(df, pd.DataFrame) and "machine_id" in df.columns:
        for mid in df["machine_id"].tolist():
            key = valid_asset_id(mid)
            if key and key not in seen:
                seen.add(key)
                ids.append(key)
    ids.sort()

    events: list[dict[str, Any]] = []
    for mid in ids:
        row = _latest_row(df, mid)
        health = health_map.get(mid) or {}
        adv = adv_map.get(mid) or {}
        events.append(
            build_health_event(
                asset=mid,
                pack_id=pack_id,
                row=row,
                prediction=pred_map.get(mid),
                physics=phys_map.get(mid),
                advisory=str(adv.get("action") or ""),
                driver=driver_map.get(mid),
                health_index=health.get("health_index"),
                mission_success_pct=health.get("mission_success_pct"),
            )
        )
    return events


def health_events_document(
    events: Optional[list[dict[str, Any]]] = None,
    *,
    pack_id: Optional[str] = None,
) -> dict[str, Any]:
    pack = get_pack(pack_id)
    return {
        "schema": SCHEMA,
        "pack_id": pack["id"],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(events or []),
        "events": list(events or []),
    }


def dumps_health_events(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, default=str) + "\n"


def missing_required_keys(event: dict[str, Any]) -> list[str]:
    return [k for k in REQUIRED_KEYS if k not in event]
