"""Layer 1 physics-informed red-line rules — run BEFORE Isolation Forest.

Deterministic limits, not FADEC / GCS / MQTT / performance maps. Isolation Forest
stays the default ML for slow degradation; these flags are known red-lines.

Aviation thresholds are packed to ``sample_data/aviation_uav_piston.csv`` scales
(EGT 658–800, CHT 157–205, oil_pressure 40.5–64 psi, vibration 1.4–6.6) so a
literal CHT>250 / oil<40 pair would never fire on the demo fleet.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from src.quality_checks import find_col
from src.twin3d import normalize_risk

_INVALID_ASSET_IDS = {"", "nan", "none", "null", "nat", "<na>"}


def valid_asset_id(value: Any) -> str:
    """Skip empty / NaN machine ids so MissionAdvisory never lists a ``nan`` tail."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in _INVALID_ASSET_IDS:
        return ""
    return text

FLAG_ALERT = "physics_alert"
FLAG_FAULT = "physics_fault"
FLAG_RULE = "physics_rule"
FLAG_LABEL = "physics_label"

PHYSICS_COLS = (FLAG_ALERT, FLAG_FAULT, FLAG_RULE, FLAG_LABEL)

FAULT_OVERHEATING = "Overheating"
FAULT_LOW_OIL = "Low oil"
FAULT_HIGH_VIB = "High vibration"
FAULT_NONE = "None"

# Primary-fault tie-break when several red-lines fire on the same row.
_FAULT_PRIORITY = (FAULT_LOW_OIL, FAULT_OVERHEATING, FAULT_HIGH_VIB)

# CAD Twin reuses the existing region map (oil → Oil ids). No new APS APIs.
FAULT_REGION = {
    FAULT_LOW_OIL: "oil_system",
    FAULT_OVERHEATING: "heads_exhaust",
    FAULT_HIGH_VIB: "rotating_crank",
}

# Human labels shown in the Physics rules table (asset → rule fired → label).
RULE_LABELS = {
    "EGT high": "Overheating",
    "CHT high": "Overheating / cooling",
    "Oil pressure low": "Low oil",
    "Vibration high": "rotating assembly",
    "Temperature high": "Overheating / cooling",
}

LAYER1_CAPTION = (
    "Physics rules = known red-line (deterministic piston limits). "
    "Isolation Forest = slow degradation. Forest is still the default ML — "
    "rules do not replace it."
)

# Aviation pack — demo CSV scales (see module docstring). User e.g. EGT>800,
# CHT>250, oil<40: CHT 250 never trips (sample max 205) and oil 40 never trips
# (sample min 40.51). 780 / 195 / 45 / 5.0 flag late-life UAV-03, not UAV-01 latest.
AVIATION_LIMITS: tuple[dict[str, Any], ...] = (
    {
        "id": "egt_high",
        "names": ("egt", "exhaust_gas_temp", "egt_c", "egt_deg_c", "exhaust_temp", "egt_celsius"),
        "op": "gt",
        "limit": 780.0,
        "rule": "EGT high",
        "fault": FAULT_OVERHEATING,
        "exact": True,
    },
    {
        "id": "cht_high",
        "names": ("cht", "cylinder_head_temp", "cht_c", "cht_temp", "head_temp"),
        "op": "gt",
        "limit": 195.0,
        "rule": "CHT high",
        "fault": FAULT_OVERHEATING,
        "exact": True,
    },
    {
        "id": "temp_high",
        "names": ("temperature", "temp", "temp_c", "temp_celsius"),
        "op": "gt",
        "limit": 195.0,
        "rule": "Temperature high",
        "fault": FAULT_OVERHEATING,
        "exact": True,
    },
    {
        "id": "oil_low",
        "names": ("oil_pressure", "oil_psi", "oil_press", "oil_press_psi", "oilpressure"),
        "op": "lt",
        "limit": 45.0,
        "rule": "Oil pressure low",
        "fault": FAULT_LOW_OIL,
        "exact": True,
    },
    {
        "id": "vib_high",
        "names": ("vibration", "vib", "vibration_rms", "rms_vib"),
        "op": "gt",
        "limit": 5.0,
        "rule": "Vibration high",
        "fault": FAULT_HIGH_VIB,
        "exact": False,
    },
)

# Other packs: conservative so Plant temperature ~65–101 and auto oil ~29–47
# are not nonsense-flagged. Auto oil KPI already uses 20 psi.
DEFAULT_LIMITS: tuple[dict[str, Any], ...] = (
    {
        "id": "egt_high",
        "names": ("egt", "exhaust_gas_temp", "egt_c", "egt_deg_c", "exhaust_temp", "egt_celsius"),
        "op": "gt",
        "limit": 800.0,
        "rule": "EGT high",
        "fault": FAULT_OVERHEATING,
        "exact": True,
    },
    {
        "id": "cht_high",
        "names": ("cht", "cylinder_head_temp", "cht_c", "cht_temp", "head_temp"),
        "op": "gt",
        "limit": 250.0,
        "rule": "CHT high",
        "fault": FAULT_OVERHEATING,
        "exact": True,
    },
    {
        "id": "temp_high",
        "names": ("temperature", "temp", "temp_c", "temp_celsius"),
        "op": "gt",
        "limit": 120.0,
        "rule": "Temperature high",
        "fault": FAULT_OVERHEATING,
        "exact": True,
    },
    {
        "id": "oil_low",
        "names": ("oil_pressure", "oil_psi", "oil_press", "oil_press_psi", "oilpressure"),
        "op": "lt",
        "limit": 20.0,
        "rule": "Oil pressure low",
        "fault": FAULT_LOW_OIL,
        "exact": True,
    },
    {
        "id": "vib_high",
        "names": ("vibration", "vib", "vibration_rms", "rms_vib"),
        "op": "gt",
        "limit": 8.0,
        "rule": "Vibration high",
        "fault": FAULT_HIGH_VIB,
        "exact": False,
    },
)


def limits_for(pack_id: Optional[str] = None) -> tuple[dict[str, Any], ...]:
    pid = str(pack_id or "")
    if pid == "aviation_uav_piston":
        return AVIATION_LIMITS
    return DEFAULT_LIMITS


def _resolve_col(df: pd.DataFrame, names: Iterable[str], *, exact: bool) -> Optional[str]:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for n in names:
        hit = lower.get(str(n).lower())
        if hit:
            return hit
    if exact:
        return None
    return find_col(df, *tuple(names))


def _mask(series: pd.Series, op: str, limit: float) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if op == "lt":
        return vals < float(limit)
    return vals > float(limit)


def _pick_fault(faults: Iterable[str]) -> str:
    present = {str(f) for f in faults if f and f != FAULT_NONE}
    for fault in _FAULT_PRIORITY:
        if fault in present:
            return fault
    return FAULT_NONE


def apply_physics_rules(
    df: Optional[pd.DataFrame],
    pack_id: Optional[str] = None,
) -> pd.DataFrame:
    """Annotate each row with physics_alert / physics_fault / physics_rule / physics_label.

    Missing specialty columns are skipped. Isolation Forest is not called here.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        out = pd.DataFrame() if df is None else df.copy()
        for col, default in (
            (FLAG_ALERT, False),
            (FLAG_FAULT, FAULT_NONE),
            (FLAG_RULE, ""),
            (FLAG_LABEL, FAULT_NONE),
        ):
            if col not in out.columns:
                out[col] = default
        return out

    out = df.copy()
    fired_rules: list[pd.Series] = []
    fired_faults: list[pd.Series] = []
    resolved: dict[str, str] = {}
    for spec in limits_for(pack_id):
        if spec["id"] == "temp_high" and "cht_high" in resolved:
            # Aviation maps CHT onto both `cht` and `temperature`; don't double-count.
            continue
        col = _resolve_col(out, spec["names"], exact=bool(spec.get("exact")))
        if not col:
            continue
        resolved[spec["id"]] = col
        hit = _mask(out[col], spec["op"], spec["limit"]).fillna(False)
        if not bool(hit.any()):
            # Still register empty so concat alignment is stable when later rules hit.
            pass
        rule_series = pd.Series(np.where(hit, spec["rule"], ""), index=out.index)
        fault_series = pd.Series(np.where(hit, spec["fault"], ""), index=out.index)
        fired_rules.append(rule_series)
        fired_faults.append(fault_series)

    if not fired_rules:
        out[FLAG_ALERT] = False
        out[FLAG_FAULT] = FAULT_NONE
        out[FLAG_RULE] = ""
        out[FLAG_LABEL] = FAULT_NONE
        return out

    stacked_rules = pd.concat(fired_rules, axis=1)
    stacked_faults = pd.concat(fired_faults, axis=1)
    joined_rules = stacked_rules.apply(
        lambda row: "; ".join(str(v) for v in row if v), axis=1
    )
    primary = stacked_faults.apply(_pick_fault, axis=1)
    out[FLAG_ALERT] = primary.ne(FAULT_NONE)
    out[FLAG_FAULT] = primary
    out[FLAG_RULE] = joined_rules
    out[FLAG_LABEL] = out[FLAG_RULE].map(
        lambda text: "; ".join(RULE_LABELS.get(part, part) for part in str(text).split("; ") if part)
        or FAULT_NONE
    )
    return out


def _latest_by_asset(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "machine_id" not in df.columns:
        return df.tail(1).copy()
    sort_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    return df.sort_values(sort_col).groupby("machine_id", as_index=False).tail(1)


def summarize_physics(df: Optional[pd.DataFrame]) -> list[dict[str, Any]]:
    """Per-asset summary from the latest row (next-mission relevant)."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    frame = df
    if FLAG_FAULT not in frame.columns:
        frame = apply_physics_rules(frame)
    latest = _latest_by_asset(frame)
    rows: list[dict[str, Any]] = []
    if latest.empty:
        return rows
    has_mid = "machine_id" in latest.columns
    hist_counts: dict[str, int] = {}
    if has_mid and FLAG_ALERT in frame.columns:
        flagged = frame[frame[FLAG_ALERT] == True]  # noqa: E712
        if not flagged.empty:
            hist_counts = flagged.groupby(flagged["machine_id"].astype(str)).size().to_dict()
    for _, rec in latest.iterrows():
        mid = valid_asset_id(rec["machine_id"]) if has_mid else "asset"
        if has_mid and not mid:
            continue
        fault = str(rec.get(FLAG_FAULT) or FAULT_NONE)
        rule = str(rec.get(FLAG_RULE) or "")
        label = str(rec.get(FLAG_LABEL) or (RULE_LABELS.get(rule, fault) if rule else FAULT_NONE))
        alert = bool(rec.get(FLAG_ALERT)) or fault not in ("", FAULT_NONE, "nan")
        rows.append(
            {
                "machine_id": mid,
                FLAG_ALERT: alert,
                FLAG_FAULT: fault if fault not in ("", "nan") else FAULT_NONE,
                FLAG_RULE: "" if rule in ("", "nan") else rule,
                FLAG_LABEL: label if label not in ("", "nan") else FAULT_NONE,
                "n_alert_rows": int(hist_counts.get(mid, 0)),
                "region": physics_fault_region(fault),
            }
        )
    rows.sort(key=lambda r: str(r["machine_id"]))
    return rows


def physics_fault_region(fault: Optional[str]) -> str:
    """Existing CAD region id for a physics fault (oil → oil_system). Empty if none."""
    return FAULT_REGION.get(str(fault or ""), "")


def physics_table_rows(summary: Optional[list[dict[str, Any]]] = None) -> pd.DataFrame:
    """UI table: asset → rule fired → human label."""
    rows = []
    for item in summary or []:
        rows.append(
            {
                "asset": item.get("machine_id"),
                "rule fired": item.get(FLAG_RULE) or "—",
                "human label": item.get(FLAG_LABEL) or FAULT_NONE,
            }
        )
    return pd.DataFrame(rows)


def mission_success_pct(health_index: Any) -> int:
    """If {machine_id} flies next mission, {N}% success chance.

    Formula: N = clamp(round(health_index), 0, 100). health_index is the pack
    0–100 score from compute_pack_kpis (Isolation Forest / RUL risk + sensor
    z-penalties). Not FADEC.
    """
    try:
        value = float(health_index)
    except (TypeError, ValueError):
        return 0
    if value != value:  # NaN
        return 0
    return int(np.clip(round(value), 0, 100))


def mission_success_line(machine_id: Any, health_index: Any) -> str:
    mid = str(machine_id or "asset")
    n = mission_success_pct(health_index)
    return f"If {mid} flies next mission, {n}% success chance"


def pick_mission_asset(
    by_asset: Optional[list[dict[str, Any]]] = None,
    *,
    selected_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Selected asset, else worst (lowest health_index)."""
    rows = list(by_asset or [])
    if not rows:
        return None
    if selected_id:
        for row in rows:
            if str(row.get("machine_id")) == str(selected_id):
                return row
    return min(rows, key=lambda r: (float(r.get("health_index") or 0), str(r.get("machine_id"))))


def _oilish(fault: str, sensor: str, region: str) -> bool:
    blob = f"{fault} {sensor} {region}".lower()
    return fault == FAULT_LOW_OIL or "oil" in blob


def _coolish(fault: str, sensor: str, region: str) -> bool:
    blob = f"{fault} {sensor} {region}".lower()
    if fault == FAULT_OVERHEATING:
        return True
    return any(tok in blob for tok in ("egt", "cht", "exhaust", "head", "cool", "temp"))


def _vibish(fault: str, sensor: str, region: str) -> bool:
    blob = f"{fault} {sensor} {region}".lower()
    return fault == FAULT_HIGH_VIB or "vib" in blob or "rotat" in blob or "crank" in blob


def advisory_action(
    *,
    risk_level: Any = None,
    physics_fault: Any = None,
    driver_sensor: Any = None,
    driver_region: Any = None,
    region_phrase: Any = None,
) -> str:
    """Combine physics fault + Isolation Forest risk + optional CAD sensor driver.

    High + oil/physics low oil → Inspect oil
    High + overheat/CHT/EGT → Check cooling
    Medium → Monitor / inspect <region>
    Low + no physics → OK
    A red-line on a Low-risk asset still surfaces the physics action.
    """
    risk = normalize_risk(risk_level)
    fault = str(physics_fault or FAULT_NONE)
    if fault in ("", "nan"):
        fault = FAULT_NONE
    sensor = str(driver_sensor or "")
    region = str(driver_region or region_phrase or physics_fault_region(fault) or "")
    phrase = str(region_phrase or "").strip()
    if not phrase:
        phrase = {
            "oil_system": "oil",
            "heads_exhaust": "cooling",
            "rotating_crank": "rotating assembly",
            "intake": "intake",
        }.get(region, region.replace("_", " ").strip())
    if phrase.lower() in ("", "region", "other", "other / unmapped region"):
        phrase = ""

    oil = _oilish(fault, sensor, region)
    cool = _coolish(fault, sensor, region)
    vib = _vibish(fault, sensor, region)
    physics_on = fault != FAULT_NONE

    if risk == "High" or physics_on:
        if oil:
            return "Inspect oil"
        if cool:
            return "Check cooling"
        if vib:
            return "Inspect rotating assembly"
        if risk == "High":
            return f"Inspect {phrase}" if phrase else "Inspect"

    if risk == "Medium":
        return f"Monitor / inspect {phrase}" if phrase else "Monitor"

    return "OK"


def build_advisory_items(
    *,
    machine_ids: Optional[Iterable[Any]] = None,
    predictions: Optional[list[dict[str, Any]]] = None,
    physics_rows: Optional[list[dict[str, Any]]] = None,
    by_asset: Optional[list[dict[str, Any]]] = None,
    drivers: Optional[dict[str, dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """One row per asset (full fleet, not top-3)."""
    pred_map = {valid_asset_id(p.get("machine_id")): p for p in (predictions or [])}
    pred_map.pop("", None)
    phys_map = {valid_asset_id(p.get("machine_id")): p for p in (physics_rows or [])}
    phys_map.pop("", None)
    health_map = {valid_asset_id(a.get("machine_id")): a for a in (by_asset or [])}
    health_map.pop("", None)
    driver_map = {valid_asset_id(k): v for k, v in (drivers or {}).items()}
    driver_map.pop("", None)

    ids: list[str] = []
    seen: set[str] = set()
    for source in (machine_ids, pred_map, phys_map, health_map):
        if source is None:
            continue
        seq = source.keys() if isinstance(source, dict) else source
        for mid in seq:
            key = valid_asset_id(mid)
            if key and key not in seen:
                seen.add(key)
                ids.append(key)
    ids.sort()

    items: list[dict[str, Any]] = []
    for mid in ids:
        pred = pred_map.get(mid) or {}
        phys = phys_map.get(mid) or {}
        health = health_map.get(mid) or {}
        driver = driver_map.get(mid) or {}
        risk = pred.get("risk_level") or health.get("risk_level")
        fault = phys.get(FLAG_FAULT) or FAULT_NONE
        hi = health.get("health_index")
        action = advisory_action(
            risk_level=risk,
            physics_fault=fault,
            driver_sensor=driver.get("sensor"),
            driver_region=driver.get("region") or phys.get("region"),
            region_phrase=driver.get("region_label") or driver.get("phrase"),
        )
        items.append(
            {
                "machine_id": mid,
                "action": action,
                FLAG_FAULT: fault,
                FLAG_RULE: phys.get(FLAG_RULE) or "",
                FLAG_LABEL: phys.get(FLAG_LABEL) or FAULT_NONE,
                "risk_level": normalize_risk(risk),
                "health_index": hi,
                "mission_success_pct": mission_success_pct(hi) if hi is not None else None,
                "mission_line": mission_success_line(mid, hi) if hi is not None else "",
            }
        )
    return items


def format_mission_advisory(items: Optional[list[dict[str, Any]]] = None) -> str:
    """``UAV-01: Inspect oil · UAV-02: OK · UAV-03: Check cooling`` — every asset."""
    rows = list(items or [])
    if not rows:
        return ""
    return " · ".join(f"{row['machine_id']}: {row['action']}" for row in rows)


def collect_drivers(
    df: Optional[pd.DataFrame],
    predictions: Optional[list[dict[str, Any]]] = None,
) -> dict[str, dict[str, Any]]:
    """Optional Isolation Forest / CAD sensor driver per High/Medium asset."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    try:
        from src.cad_part import sensor_driver
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for pred in predictions or []:
        mid = str(pred.get("machine_id") or "")
        if not mid:
            continue
        try:
            out[mid] = sensor_driver(df, mid, pred.get("risk_level"))
        except Exception:
            continue
    return out


def build_sih_bundle(
    df: Optional[pd.DataFrame],
    pack_id: Optional[str] = None,
    predictions: Optional[list[dict[str, Any]]] = None,
    by_asset: Optional[list[dict[str, Any]]] = None,
    *,
    selected_id: Optional[str] = None,
    include_drivers: bool = True,
) -> dict[str, Any]:
    """Physics flags + mission % + full-fleet MissionAdvisory in one shot."""
    annotated = apply_physics_rules(df, pack_id)
    physics_rows = summarize_physics(annotated)
    drivers = collect_drivers(annotated, predictions) if include_drivers else {}
    mids: list[str] = []
    if df is not None and isinstance(df, pd.DataFrame) and "machine_id" in df.columns:
        mids = sorted({valid_asset_id(x) for x in df["machine_id"].tolist()} - {""})
    items = build_advisory_items(
        machine_ids=mids,
        predictions=predictions,
        physics_rows=physics_rows,
        by_asset=by_asset,
        drivers=drivers,
    )
    focus = pick_mission_asset(by_asset, selected_id=selected_id)
    mission_line = ""
    if focus is not None:
        mission_line = mission_success_line(focus.get("machine_id"), focus.get("health_index"))
    return {
        "annotated": annotated,
        "physics_rows": physics_rows,
        "physics_table": physics_table_rows(physics_rows),
        "advisory_items": items,
        "advisory": format_mission_advisory(items),
        "mission_line": mission_line,
        "mission_asset": focus,
        "drivers": drivers,
    }
