"""Canonical live telemetry contract — semi-structured MQTT JSON → PdM row.

This is the schema gate the rest of the pipeline already assumes after
Map sensors. MQTT / Paho only deliver bytes; this module names fields,
coerces types, and (for aviation) converts bar→psi when the packet looks
like Rotax-style oil pressure.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.industry_packs import OPTIONAL_IF_SENSORS

# Plant four + aviation extras. Map sensors aliases reuse these names.
CANONICAL_LIVE_FIELDS: tuple[str, ...] = (
    "timestamp",
    "machine_id",
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "egt",
    "cht",
    "oil_pressure",
    "oil_temp",
    "fuel_flow",
    "altitude",
    "throttle",
    "manifold_pressure",
    "flight_hours",
)

LIVE_SCORE_SENSORS: tuple[str, ...] = (
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "egt",
    "cht",
    "oil_pressure",
    "oil_temp",
    "fuel_flow",
    "throttle",
    "manifold_pressure",
)

# MAVLink EFI_STATUS (msg id 225) → our canonical names. Decode is a GCS
# gateway job; we ingest the JSON the gateway would publish.
MAVLINK_EFI_STATUS_225: dict[str, str] = {
    "rpm": "rpm",
    "fuel_consumed": "fuel_flow",
    "fuel_flow": "fuel_flow",
    "engine_load": "throttle",
    "throttle_position": "throttle",
    "barometric_pressure": "manifold_pressure",
    "intake_manifold_pressure": "manifold_pressure",
    "intake_temperature": "temperature",
    "cylinder_head_temperature": "cht",
    "exhaust_gas_temperature": "egt",
    "ecu_index": "machine_id",
}

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "machine_id": (
        "machine_id",
        "uav_id",
        "uavid",
        "asset_id",
        "tail",
        "aircraft_id",
        "engine_id",
        "ecu_index",
    ),
    "timestamp": ("timestamp", "time", "ts", "datetime", "recorded_at"),
    "temperature": ("temperature", "temp", "temp_c", "intake_temperature"),
    "vibration": ("vibration", "vib", "vibration_rms", "rms_vib"),
    "pressure": ("pressure", "press", "psi"),
    "rpm": ("rpm", "speed", "engine_rpm"),
    "egt": ("egt", "exhaust_gas_temp", "egt_c", "exhaust_gas_temperature", "exhaust_temp"),
    "cht": ("cht", "cylinder_head_temp", "cht_c", "cylinder_head_temperature", "head_temp"),
    "oil_pressure": (
        "oil_pressure",
        "oil_psi",
        "oil_press",
        "oilpressure",
        "oil_pressure_psi",
    ),
    "oil_temp": ("oil_temp", "oil_temperature", "oil_deg_c", "oil_t"),
    "fuel_flow": ("fuel_flow", "fuel_consumed", "ff", "fuel_lph"),
    "altitude": ("altitude", "alt_ft", "pressure_alt"),
    "throttle": ("throttle", "throttle_pct", "throttle_position", "engine_load"),
    "manifold_pressure": (
        "manifold_pressure",
        "map",
        "mp_in_hg",
        "intake_manifold_pressure",
        "barometric_pressure",
    ),
    "flight_hours": ("flight_hours", "hobbs", "engine_hours"),
}

_BAR_KEYS = ("oil_pressure_bar", "oil_bar")
PSI_PER_BAR = 14.5037738
# Rotax-class oil is ~2–7 bar. Demo CSV is ~40–64 psi. Values ≤ 12 treated as bar
# when the packet is clearly aviation (cht/egt/uav) or pack_id says so.
_BAR_OIL_MAX = 12.0


def _norm_key(name: Any) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name or "")).strip("_")


def _flatten(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"value": payload}
    out: dict[str, Any] = {}
    for key, val in payload.items():
        if isinstance(val, dict):
            for k2, v2 in val.items():
                out[_norm_key(f"{key}_{k2}")] = v2
                out[_norm_key(k2)] = v2
        else:
            out[_norm_key(key)] = val
    return out


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _looks_aviation(flat: dict[str, Any], pack_id: Optional[str] = None) -> bool:
    if str(pack_id or "") == "aviation_uav_piston":
        return True
    keys = set(flat)
    if keys & {"cht", "egt", "uav_id", "uavid", "cylinder_head_temp", "exhaust_gas_temp"}:
        return True
    mid = str(flat.get("machine_id") or flat.get("uav_id") or "")
    return mid.upper().startswith("UAV")


def _oil_pressure_psi(flat: dict[str, Any], *, aviation: bool) -> Optional[float]:
    for key in _BAR_KEYS:
        bar = _to_float(flat.get(key))
        if bar is not None:
            return round(bar * PSI_PER_BAR, 3)
    unit = str(flat.get("oil_pressure_unit") or flat.get("oil_unit") or "").lower()
    raw = _to_float(flat.get("oil_pressure"))
    if raw is None:
        raw = _to_float(flat.get("oil_psi"))
    if raw is None:
        return None
    if unit in {"bar", "bars"}:
        return round(raw * PSI_PER_BAR, 3)
    if aviation and 0 < raw <= _BAR_OIL_MAX:
        return round(raw * PSI_PER_BAR, 3)
    return raw


def coerce_live_payload(
    payload: Any,
    *,
    machine_field: str = "machine_id",
    ts_field: str = "timestamp",
    fallback_machine: str = "mqtt-asset",
    pack_id: Optional[str] = None,
    topic_fallback: Optional[str] = None,
) -> dict[str, Any]:
    """One MQTT JSON (or simulator dict) → one canonical row. Extra keys kept in ``raw``."""
    if not isinstance(payload, dict):
        payload = {"value": payload}
    flat = _flatten(payload)
    aviation = _looks_aviation({**flat, **{k: payload.get(k) for k in payload}}, pack_id)

    row: dict[str, Any] = {}
    ts = payload.get(ts_field) if ts_field in payload else None
    if ts is None:
        ts = flat.get("timestamp") or flat.get("time") or flat.get("ts")
    parsed = pd.to_datetime(ts, errors="coerce") if ts is not None else pd.NaT
    if pd.isna(parsed):
        parsed = pd.Timestamp.utcnow().floor("s")
    row["timestamp"] = parsed

    mid = payload.get(machine_field)
    if mid in (None, ""):
        for alias in _FIELD_ALIASES["machine_id"]:
            if alias in payload and payload[alias] not in (None, ""):
                mid = payload[alias]
                break
            if alias in flat and flat[alias] not in (None, ""):
                mid = flat[alias]
                break
    if mid in (None, ""):
        mid = topic_fallback or fallback_machine
    row["machine_id"] = str(mid)

    for field, aliases in _FIELD_ALIASES.items():
        if field in {"machine_id", "timestamp"}:
            continue
        if field == "oil_pressure":
            val = _oil_pressure_psi(flat, aviation=aviation)
        else:
            val = None
            for alias in aliases:
                if alias in payload:
                    val = _to_float(payload[alias])
                if val is None and alias in flat:
                    val = _to_float(flat[alias])
                if val is not None:
                    break
        if val is not None:
            row[field] = val

    # Plant pipeline lock needs temperature / pressure; aviation maps CHT / oil.
    if "temperature" not in row and "cht" in row:
        row["temperature"] = row["cht"]
    if "pressure" not in row and "oil_pressure" in row:
        row["pressure"] = row["oil_pressure"]

    row["raw"] = payload
    return row


def row_to_mqtt_json(row: dict[str, Any]) -> dict[str, Any]:
    """Canonical row → publisher JSON (no ``raw`` blob)."""
    out: dict[str, Any] = {}
    ts = row.get("timestamp")
    if ts is not None:
        try:
            out["timestamp"] = pd.Timestamp(ts).isoformat()
        except Exception:
            out["timestamp"] = str(ts)
    mid = row.get("machine_id")
    if mid not in (None, ""):
        out["uav_id"] = str(mid)
        out["machine_id"] = str(mid)
    for key in CANONICAL_LIVE_FIELDS:
        if key in {"timestamp", "machine_id"}:
            continue
        if key in row and row[key] is not None:
            out[key] = row[key]
    return out


def live_chart_metrics(columns: list[str], pack_id: Optional[str] = None) -> list[str]:
    preferred = (
        "cht",
        "egt",
        "oil_pressure",
        "vibration",
        "rpm",
        "temperature",
        "pressure",
        "fuel_flow",
        "oil_temp",
    )
    if str(pack_id or "") != "aviation_uav_piston":
        preferred = ("temperature", "vibration", "pressure", "rpm") + preferred
    present = [c for c in preferred if c in columns]
    extras = [c for c in OPTIONAL_IF_SENSORS if c in columns and c not in present]
    return present + extras
