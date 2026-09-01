"""Map messy sensor CSV headers → canonical PdM fields.

Reliability add-on mapping only (timestamp, asset, sensors, optional RUL label).
Not OEE Pulse SAP templates and not Forge domain packs.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.industry_packs import extra_aliases_for

CANONICAL_FIELDS: list[tuple[str, str]] = [
    ("timestamp", "Reading timestamp"),
    ("machine_id", "Machine / asset id"),
    ("temperature", "Temperature"),
    ("vibration", "Vibration"),
    ("pressure", "Pressure"),
    ("rpm", "Rotational speed (RPM)"),
    ("failure_within_days", "Days until failure (RUL label, optional)"),
]

CANONICAL_NAMES = [c[0] for c in CANONICAL_FIELDS]

ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": (
        "timestamp",
        "time",
        "datetime",
        "date_time",
        "ts",
        "event_time",
        "recorded_at",
        "reading_time",
    ),
    "machine_id": (
        "machine_id",
        "machine",
        "asset_id",
        "asset",
        "equipment",
        "equnr",
        "unit_id",
        "asset_tag",
        "device_id",
        "uav_id",
        "aircraft_id",
        "well_id",
        "vehicle_id",
        "engine_id",
    ),
    "temperature": ("temperature", "temp", "temp_c", "temp_celsius", "oil_temp", "bearing_temp"),
    "vibration": ("vibration", "vib", "vibration_rms", "accel", "acceleration", "rms_vib"),
    "pressure": ("pressure", "press", "psi", "oil_pressure"),
    "rpm": ("rpm", "speed", "rotational_speed", "shaft_rpm", "motor_rpm"),
    "failure_within_days": (
        "failure_within_days",
        "rul",
        "rul_days",
        "days_to_failure",
        "remaining_life",
        "remaining_useful_life",
        "ttf_days",
    ),
}

SENSOR_FIELDS = ("temperature", "vibration", "pressure", "rpm")


def _norm(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")


def suggest_mapping(
    columns: list[str],
    extra_aliases: Optional[dict[str, tuple[str, ...]]] = None,
    pack_id: Optional[str] = None,
) -> dict[str, Optional[str]]:
    """Best-effort header match. Unmatched canonical fields map to None.

    Optional pack extras (EGT, fillage, …) are mapped only after core PdM fields
    so they cannot steal timestamp / machine_id / the four plant sensors.
    """
    remaining = list(columns)
    if extra_aliases is not None:
        extras = dict(extra_aliases)
    elif pack_id:
        extras = extra_aliases_for(pack_id)
    else:
        extras = {}
    mapping: dict[str, Optional[str]] = {name: None for name in CANONICAL_NAMES}
    for name in extras:
        mapping[name] = None
    by_norm = {_norm(c): c for c in columns}

    ordered: list[tuple[str, tuple[str, ...]]] = list(ALIASES.items())
    for name, aliases in extras.items():
        if name not in ALIASES:
            ordered.append((name, tuple(aliases)))

    for canonical, aliases in ordered:
        hit = None
        for alias in aliases:
            key = _norm(alias)
            if key in by_norm and by_norm[key] in remaining:
                hit = by_norm[key]
                break
            for col in remaining:
                if key and key in _norm(col):
                    hit = col
                    break
            if hit:
                break
        if hit:
            mapping[canonical] = hit
            remaining = [c for c in remaining if c != hit]
    return mapping


def apply_mapping(df: pd.DataFrame, mapping: dict[str, Optional[str]]) -> pd.DataFrame:
    """Rename source columns to canonical names. Skip empty / identity maps."""
    out = df.copy()
    rename: dict[str, str] = {}
    for canonical, source in (mapping or {}).items():
        if not source or source not in out.columns:
            continue
        if source == canonical:
            continue
        if canonical in out.columns and canonical != source:
            continue
        rename[source] = canonical
    if rename:
        out = out.rename(columns=rename)
    return out


def mapping_status(df: pd.DataFrame) -> dict[str, object]:
    """What the current frame has for the PdM pipeline."""
    present = [c for c in CANONICAL_NAMES if c in df.columns]
    sensors = [c for c in SENSOR_FIELDS if c in df.columns]
    return {
        "present": present,
        "sensors": sensors,
        "has_timestamp": "timestamp" in df.columns,
        "has_machine": "machine_id" in df.columns,
        "has_rul_label": "failure_within_days" in df.columns,
        "sensor_count": len(sensors),
    }
