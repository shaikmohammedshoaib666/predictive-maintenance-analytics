"""Industry packs — one CSV → one pack → KPIs + charts + 3D twin.

Four fields only. Plant / rotating machines is the default. This is not Forge
domain packs and not “every industry under the sun.”

Automotive is **one generic ICE car + powertrain**. That is what “not every OEM,
not every trim, not every EV architecture” means:

- **OEM** — manufacturer (BMW vs Toyota vs Ford). We do not ship a unique 3D
  body or health model per brand.
- **Trim** — model grade (LX vs Sport vs Limited). One silhouette, not a
  catalog of option packages.
- **EV architecture** — battery-electric skateboard vs hybrid vs ICE. This pack
  is a generic piston engine + gearbox/driveline. Separate BEV / hybrid twins
  are out of scope.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

DEFAULT_PACK_ID = "plant_rotating"

# Shown in the automotive pack UI so operators do not expect a brand catalog.
AUTOMOTIVE_OEM_TRIM_EV_NOTE = (
    "One generic car + ICE powertrain (engine + gearbox/driveline). "
    "Not every OEM (no BMW vs Toyota vs Ford CAD). "
    "Not every trim (no LX vs Sport vs Limited variants). "
    "Not every EV architecture (no separate BEV skateboard, hybrid, or fuel-cell packs). "
    "OBD-style engine / oil / coolant / load sensors drive health on that one template."
)

# Extra Isolation Forest features when these columns exist (plant sample is unchanged).
OPTIONAL_IF_SENSORS: tuple[str, ...] = (
    "egt",
    "cht",
    "oil_temp",
    "oil_pressure",
    "coolant_temp",
    "engine_load",
    "manifold_pressure",
    "vehicle_speed",
    "throttle",
    "altitude",
    "flight_hours",
    "polish_rod_load",
    "tubing_pressure",
    "casing_pressure",
    "pump_fillage",
    "stroke_spm",
    "production_bbl",
)


def _pack(
    *,
    pack_id: str,
    label: str,
    short: str,
    twin_kind: str,
    hotspot: str,
    sih: Optional[str],
    hero: bool,
    sample_file: str,
    chart_metrics: tuple[str, ...],
    extra_fields: tuple[tuple[str, str, tuple[str, ...]], ...],
    detect_keywords: tuple[str, ...],
    detect_columns: tuple[str, ...],
    detect_id_prefixes: tuple[str, ...],
    kpis: tuple[str, ...],
    scope: str,
    not_in_scope: str,
    cost_hour_label: str,
    cost_unit_label: str,
) -> dict[str, Any]:
    extras = {
        name: {"label": label_, "aliases": aliases}
        for name, label_, aliases in extra_fields
    }
    return {
        "id": pack_id,
        "label": label,
        "short": short,
        "twin_kind": twin_kind,
        "hotspot": hotspot,
        "sih": sih,
        "hero": hero,
        "default": pack_id == DEFAULT_PACK_ID,
        "sample_file": sample_file,
        "chart_metrics": list(chart_metrics),
        "extra_fields": extras,
        "extra_aliases": {name: spec["aliases"] for name, spec in extras.items()},
        "detect_keywords": detect_keywords,
        "detect_columns": detect_columns,
        "detect_id_prefixes": detect_id_prefixes,
        "kpis": list(kpis),
        "scope": scope,
        "not_in_scope": not_in_scope,
        "cost_hour_label": cost_hour_label,
        "cost_unit_label": cost_unit_label,
    }


PACKS: dict[str, dict[str, Any]] = {
    "plant_rotating": _pack(
        pack_id="plant_rotating",
        label="Plant / rotating machines",
        short="Plant",
        twin_kind="plant_motor",
        hotspot="drive-end bearing",
        sih=None,
        hero=False,
        sample_file="sensor_readings.csv",
        chart_metrics=("vibration", "temperature", "pressure", "rpm"),
        extra_fields=(),
        detect_keywords=("motor", "bearing", "pump", "compressor", "gearbox", "plant"),
        detect_columns=("vibration", "temperature", "pressure", "rpm"),
        detect_id_prefixes=("M-", "Machine ", "PUMP", "MOT"),
        kpis=("asset_health_index", "predicted_rul_days", "high_risk_assets"),
        scope=(
            "Generic rotating equipment (motor / pump / compressor). Isolation Forest + RUL "
            "on temperature, vibration, pressure, RPM. 3D twin is a library motor whose "
            "drive-end bearing goes green / amber / red."
        ),
        not_in_scope="Not a full plant digital twin of every skid, and not CAD of a specific OEM motor.",
        cost_hour_label="$ / hour downtime",
        cost_unit_label="$ / unit lost production",
    ),
    "aviation_uav_piston": _pack(
        pack_id="aviation_uav_piston",
        label="Aviation / aero piston / MALE UAV",
        short="Aviation",
        twin_kind="aviation_uav",
        hotspot="piston engine",
        sih="SIH26054",
        hero=True,
        sample_file="aviation_uav_piston.csv",
        chart_metrics=("egt", "vibration", "oil_pressure", "cht", "rpm", "temperature"),
        extra_fields=(
            ("egt", "Exhaust gas temp (EGT)", ("egt", "exhaust_gas_temp", "egt_c", "egt_deg_c")),
            ("cht", "Cylinder head temp (CHT)", ("cht", "cylinder_head_temp", "cht_c")),
            ("oil_pressure", "Engine oil pressure", ("oil_pressure", "oil_psi", "oil_press")),
            ("oil_temp", "Engine oil temperature", ("oil_temp", "oil_temperature", "oil_deg_c")),
            ("flight_hours", "Airframe / engine hours", ("flight_hours", "hobbs", "engine_hours", "tacho_hours")),
            ("altitude", "Altitude (ft)", ("altitude", "alt_ft", "pressure_alt")),
            ("throttle", "Throttle (%)", ("throttle", "throttle_pct", "power_lever")),
            ("manifold_pressure", "Manifold pressure", ("manifold_pressure", "map", "mp_in_hg")),
        ),
        detect_keywords=(
            "uav",
            "male",
            "piston",
            "aero",
            "aviation",
            "egt",
            "cht",
            "hobbs",
            "sortie",
            "drone",
            "airframe",
        ),
        detect_columns=("egt", "cht", "flight_hours", "manifold_pressure", "altitude", "throttle"),
        detect_id_prefixes=("UAV-", "AC-", "TAIL-"),
        kpis=(
            "engine_health_index",
            "mission_reliability_pct",
            "remaining_mission_hours",
            "egt_margin",
        ),
        scope=(
            "SIH26054 (DRDO): digital twin, health, fault prediction, and mission reliability "
            "for aero piston engines on MALE UAVs. One UAV airframe + piston-engine twin — "
            "not an airliner cabin, not a jet turbofan catalog."
        ),
        not_in_scope="Not every airframe OEM, not commercial airliner twins, not jet/turbofan architectures.",
        cost_hour_label="$ / hour of lost sortie time",
        cost_unit_label="$ / aborted mission",
    ),
    "automotive_powertrain": _pack(
        pack_id="automotive_powertrain",
        label="Automotive powertrain",
        short="Automotive",
        twin_kind="auto_car",
        hotspot="engine block",
        sih=None,
        hero=False,
        sample_file="automotive_powertrain.csv",
        chart_metrics=("coolant_temp", "oil_pressure", "engine_load", "vibration", "rpm", "temperature"),
        extra_fields=(
            ("coolant_temp", "Coolant temperature", ("coolant_temp", "coolant", "ect", "water_temp")),
            ("oil_pressure", "Engine oil pressure", ("oil_pressure", "oil_psi", "oil_press")),
            ("engine_load", "Engine load (%)", ("engine_load", "load_pct", "calculated_load")),
            ("vehicle_speed", "Vehicle speed", ("vehicle_speed", "speed_kph", "speed_kmh", "vss")),
            ("gear", "Gear", ("gear", "gear_pos", "current_gear")),
        ),
        detect_keywords=("obd", "powertrain", "coolant", "driveline", "automotive", "gearbox"),
        detect_columns=("coolant_temp", "engine_load", "vehicle_speed", "gear"),
        detect_id_prefixes=("ENG-", "PT-", "VIN-", "CAR-"),
        kpis=("powertrain_health_index", "oil_pressure_status", "thermal_headroom"),
        scope=(
            "Engine + transmission / driveline health on a single generic car silhouette. "
            + AUTOMOTIVE_OEM_TRIM_EV_NOTE
        ),
        not_in_scope=AUTOMOTIVE_OEM_TRIM_EV_NOTE,
        cost_hour_label="$ / hour vehicle down",
        cost_unit_label="$ / missed delivery",
    ),
    "oil_srp": _pack(
        pack_id="oil_srp",
        label="Oil well / sucker-rod pump",
        short="Oil / SRP",
        twin_kind="oil_srp",
        hotspot="gearbox / stuffing box",
        sih="SIH26120",
        hero=False,
        sample_file="oil_srp.csv",
        chart_metrics=("pump_fillage", "polish_rod_load", "production_bbl", "tubing_pressure", "vibration"),
        extra_fields=(
            ("polish_rod_load", "Polish-rod load", ("polish_rod_load", "rod_load", "prl", "dyno_load")),
            ("tubing_pressure", "Tubing pressure", ("tubing_pressure", "tbg_psi", "flowing_pressure")),
            ("casing_pressure", "Casing pressure", ("casing_pressure", "csg_psi")),
            ("pump_fillage", "Pump fillage (%)", ("pump_fillage", "fillage", "fillage_pct")),
            ("stroke_spm", "Strokes per minute", ("stroke_spm", "spm", "pumping_speed")),
            ("production_bbl", "Production (bbl)", ("production_bbl", "oil_bbl", "bopd", "gross_bbl")),
        ),
        detect_keywords=("sucker", "srp", "wellhead", "fillage", "polish", "workover", "css", "beam"),
        detect_columns=(
            "polish_rod_load",
            "tubing_pressure",
            "pump_fillage",
            "stroke_spm",
            "production_bbl",
            "casing_pressure",
        ),
        detect_id_prefixes=("WELL-", "SRP-", "CSS-"),
        kpis=("well_health_index", "pump_fillage_pct", "production_rate"),
        scope=(
            "SIH26120 (Oil India) Smart Automation: CSS + sucker-rod pump / well-to-surface "
            "digital twin. One beam-pump well, not a full field SCADA historian."
        ),
        not_in_scope="Not every completion type, not ESP vs PCP vs SRP as separate product lines, not a full reservoir model.",
        cost_hour_label="$ / hour deferred production",
        cost_unit_label="$ / barrel deferred",
    ),
}

PACK_ORDER: tuple[str, ...] = (
    "plant_rotating",
    "aviation_uav_piston",
    "automotive_powertrain",
    "oil_srp",
)


def list_packs() -> list[dict[str, Any]]:
    """Packs in sidebar order. Plant is first / default."""
    return [PACKS[pid] for pid in PACK_ORDER]


def get_pack(pack_id: Optional[str]) -> dict[str, Any]:
    """Return a pack dict. Unknown ids fall back to Plant."""
    if pack_id and pack_id in PACKS:
        return PACKS[pack_id]
    return PACKS[DEFAULT_PACK_ID]


def pack_label(pack_id: Optional[str]) -> str:
    return str(get_pack(pack_id)["label"])


def twin_kind_for(pack_id: Optional[str]) -> str:
    return str(get_pack(pack_id)["twin_kind"])


def sample_path_for(pack_id: Optional[str], *, root: Optional[Any] = None):
    from pathlib import Path

    base = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    return base / "sample_data" / str(get_pack(pack_id)["sample_file"])


def extra_aliases_for(pack_id: Optional[str]) -> dict[str, tuple[str, ...]]:
    return dict(get_pack(pack_id).get("extra_aliases") or {})


def extra_field_defs(pack_id: Optional[str]) -> list[tuple[str, str]]:
    extras = get_pack(pack_id).get("extra_fields") or {}
    return [(name, spec["label"]) for name, spec in extras.items()]


def preferred_y_metric(df: Any, pack_id: Optional[str]) -> Optional[str]:
    """First pack chart metric that exists as a numeric column on df."""
    if df is None:
        return None
    cols = {str(c).lower(): str(c) for c in getattr(df, "columns", [])}
    numeric = set()
    try:
        numeric = {str(c) for c in df.select_dtypes(include="number").columns}
    except Exception:
        numeric = set(cols.values())
    for name in get_pack(pack_id)["chart_metrics"]:
        real = cols.get(name.lower())
        if real and real in numeric:
            return real
    return None


def _norm_blob(parts: Iterable[str]) -> str:
    return " ".join(str(p).lower().replace("_", " ") for p in parts if p is not None)


def suggest_pack(
    columns: Optional[Iterable[str]] = None,
    *,
    filename: str = "",
    machine_ids: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Score packs from headers / filename / asset ids. Plant wins ties (default).

    A CSV does not auto-switch 3D to airplane vs car from `machine_id` alone unless
    prefixes or column names clearly match. Mixed fleets stay on the selected pack.
    """
    cols = [str(c) for c in list(columns)] if columns is not None else []
    ids = [str(m) for m in list(machine_ids)] if machine_ids is not None else []
    blob = _norm_blob(list(cols) + [filename] + ids)
    scores: dict[str, float] = {pid: 0.0 for pid in PACK_ORDER}
    reasons: dict[str, list[str]] = {pid: [] for pid in PACK_ORDER}

    col_lower = {c.lower() for c in cols}
    for pid, pack in PACKS.items():
        for col in pack["detect_columns"]:
            if col.lower() in col_lower:
                scores[pid] += 3.0
                reasons[pid].append(f"column `{col}`")
        for kw in pack["detect_keywords"]:
            if kw in blob:
                scores[pid] += 1.5
                reasons[pid].append(f"keyword '{kw}'")
        for prefix in pack["detect_id_prefixes"]:
            if any(mid.startswith(prefix) or prefix.lower() in mid.lower() for mid in ids):
                scores[pid] += 2.0
                reasons[pid].append(f"asset id prefix '{prefix.strip()}'")
                break
        if pack["sample_file"] and pack["sample_file"].rsplit(".", 1)[0].replace("_", " ") in blob:
            scores[pid] += 2.0
            reasons[pid].append("filename matches demo CSV")

    # Plant is the default: it only wins a *suggestion* if it strictly leads.
    # Otherwise the highest non-plant score wins; all-zero stays plant.
    ranked = sorted(PACK_ORDER, key=lambda p: scores[p], reverse=True)
    best = ranked[0]
    if scores[best] <= 0:
        chosen = DEFAULT_PACK_ID
        why = ["no specialty columns — Plant remains the default"]
    else:
        chosen = best
        why = reasons[chosen][:6] or ["highest keyword/column score"]
    return {
        "pack_id": chosen,
        "score": scores[chosen],
        "scores": scores,
        "reasons": why,
        "label": pack_label(chosen),
    }


def validate_packs() -> list[str]:
    """Integrity checks used by smoke tests."""
    problems: list[str] = []
    if PACK_ORDER[0] != DEFAULT_PACK_ID:
        problems.append("Plant must be first in PACK_ORDER")
    if set(PACK_ORDER) != set(PACKS):
        problems.append("PACK_ORDER / PACKS mismatch")
    kinds = {p["twin_kind"] for p in PACKS.values()}
    if len(kinds) != len(PACKS):
        problems.append("each pack needs a unique twin_kind")
    auto = PACKS["automotive_powertrain"]["not_in_scope"]
    for token in ("OEM", "trim", "EV"):
        if token not in auto:
            problems.append(f"automotive not_in_scope missing {token}")
    if "SIH26054" not in PACKS["aviation_uav_piston"]["sih"]:
        problems.append("aviation pack must cite SIH26054")
    if "SIH26120" not in PACKS["oil_srp"]["sih"]:
        problems.append("oil pack must cite SIH26120")
    return problems
