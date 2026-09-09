"""Pack-specific KPIs computed from the current table + RUL predictions.

Formulas are explicit and conservative. Missing specialty columns degrade
gracefully to risk/RUL-only scores so Plant CSVs still produce KPIs.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.industry_packs import DEFAULT_PACK_ID, get_pack
from src.physics_rules import mission_success_pct
from src.quality_checks import find_col
from src.twin3d import normalize_risk

# UAV piston typical sortie length used for remaining-mission-hours.
DEFAULT_MISSION_HOURS = 8.0
# Fraction of calendar time a MALE UAV is in the air (demo default).
DEFAULT_DUTY_CYCLE = 0.35

_RISK_BASE = {"High": 38.0, "Medium": 72.0, "Low": 92.0, "Unknown": 70.0}
_MISSION_SURVIVAL = {"High": 0.38, "Medium": 0.82, "Low": 0.97, "Unknown": 0.70}


def risk_to_health(risk: Optional[str]) -> float:
    """Map High/Medium/Low onto a 0–100 health index (High ≈ 38, Low ≈ 92)."""
    return float(_RISK_BASE[normalize_risk(risk)])


def remaining_mission_hours(
    rul_days: Optional[float],
    *,
    duty_cycle: float = DEFAULT_DUTY_CYCLE,
) -> float:
    """Convert predicted RUL days into airborne hours at a duty cycle."""
    try:
        days = float(rul_days)
    except (TypeError, ValueError):
        return 0.0
    if days < 0 or duty_cycle < 0:
        return 0.0
    return round(days * 24.0 * float(duty_cycle), 1)


def mission_reliability_pct(
    predictions: Optional[list[dict[str, Any]]],
    *,
    horizon_hours: float = DEFAULT_MISSION_HOURS,
) -> float:
    """Fleet chance the next sortie completes, from risk + RUL.

    Per asset: blend a risk prior with remaining life vs 3× mission length.
    Fleet score is the mean (not the product) so one sick UAV does not zero
    the whole number — worst-asset is reported separately.
    """
    rows = list(predictions or [])
    if not rows:
        return 0.0
    mission_days = max(float(horizon_hours) / 24.0, 1e-6)
    scores: list[float] = []
    for p in rows:
        risk = normalize_risk(p.get("risk_level"))
        base = _MISSION_SURVIVAL[risk]
        try:
            rul = float(p.get("predicted_rul_days") or 0)
        except (TypeError, ValueError):
            rul = 0.0
        if rul > 0:
            survival = min(1.0, rul / max(mission_days * 3.0, 0.5))
            base = 0.5 * base + 0.5 * (0.25 + 0.75 * survival)
        scores.append(base)
    return round(100.0 * float(np.mean(scores)), 1)


def _numeric(df: pd.DataFrame, *names: str) -> Optional[pd.Series]:
    col = find_col(df, *names) if df is not None and not df.empty else None
    if not col:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    return s if s.notna().any() else None


def _latest_by_asset(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "machine_id" not in df.columns:
        return df.tail(1).copy()
    sort_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    return df.sort_values(sort_col).groupby("machine_id", as_index=False).tail(1)


def _z_penalty(series: Optional[pd.Series], *, high_is_bad: bool = True, weight: float = 8.0) -> float:
    """Penalise the latest reading vs the series mean. Returns 0–24 points."""
    if series is None:
        return 0.0
    s = series.dropna()
    if len(s) < 3:
        return 0.0
    std = float(s.std()) + 1e-9
    z = (float(s.iloc[-1]) - float(s.mean())) / std
    if not high_is_bad:
        z = -z
    return float(np.clip(z, 0.0, 3.0) * weight)


def _pred_for(predictions: list[dict[str, Any]], machine_id: str) -> dict[str, Any]:
    for p in predictions:
        if str(p.get("machine_id")) == str(machine_id):
            return p
    return {}


def _health_from_row(
    row: pd.Series,
    pred: dict[str, Any],
    *,
    high_bad: tuple[str, ...] = (),
    low_bad: tuple[str, ...] = (),
    df_full: Optional[pd.DataFrame] = None,
) -> float:
    health = risk_to_health(pred.get("risk_level"))
    rul = pred.get("predicted_rul_days")
    try:
        if rul is not None and float(rul) <= 7:
            health -= 8.0
        elif rul is not None and float(rul) <= 14:
            health -= 3.0
    except (TypeError, ValueError):
        pass
    subset = df_full
    mid = row.get("machine_id") if hasattr(row, "get") else None
    if df_full is not None and not df_full.empty and mid is not None and "machine_id" in df_full.columns:
        subset = df_full[df_full["machine_id"].astype(str) == str(mid)]
    penalty = 0.0
    for name in high_bad:
        series = _numeric(subset, name) if subset is not None and not subset.empty else None
        if series is None and name in row.index:
            series = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce")
        penalty += _z_penalty(series, high_is_bad=True, weight=4.0)
    for name in low_bad:
        series = _numeric(subset, name) if subset is not None and not subset.empty else None
        if series is None and name in row.index:
            series = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce")
        penalty += _z_penalty(series, high_is_bad=False, weight=4.0)
    health -= min(penalty, 28.0)
    return float(np.clip(round(health, 1), 0.0, 100.0))


def _kpi(kid: str, label: str, value: Any, unit: str, severity: str, hint: str) -> dict[str, Any]:
    return {
        "id": kid,
        "label": label,
        "value": value,
        "unit": unit,
        "severity": severity,
        "hint": hint,
    }


def _severity_from_health(health: float) -> str:
    if health < 55:
        return "high"
    if health < 75:
        return "medium"
    return "low"


def _asset_rows(
    df: pd.DataFrame,
    predictions: list[dict[str, Any]],
    *,
    high_bad: tuple[str, ...] = (),
    low_bad: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    latest = _latest_by_asset(df)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not latest.empty and "machine_id" in latest.columns:
        for _, rec in latest.iterrows():
            mid = str(rec["machine_id"])
            key = str(mid).strip()
            if not key or key.lower() in {"nan", "none", "null"}:
                continue
            seen.add(mid)
            pred = _pred_for(predictions, mid)
            health = _health_from_row(rec, pred, high_bad=high_bad, low_bad=low_bad, df_full=df)
            rul = pred.get("predicted_rul_days")
            rows.append(
                {
                    "machine_id": mid,
                    "risk_level": normalize_risk(pred.get("risk_level") or rec.get("risk_level")),
                    "predicted_rul_days": rul,
                    "health_index": health,
                    "remaining_mission_hours": remaining_mission_hours(rul),
                    "mission_success_pct": mission_success_pct(health),
                }
            )
    for p in predictions:
        mid = str(p.get("machine_id", "asset"))
        if mid in seen:
            continue
        health = risk_to_health(p.get("risk_level"))
        rul = p.get("predicted_rul_days")
        rows.append(
            {
                "machine_id": mid,
                "risk_level": normalize_risk(p.get("risk_level")),
                "predicted_rul_days": rul,
                "health_index": health,
                "remaining_mission_hours": remaining_mission_hours(rul),
                "mission_success_pct": mission_success_pct(health),
            }
        )
    rows.sort(key=lambda r: (r["health_index"], str(r["predicted_rul_days"] or 99)))
    return rows


def _plant_kpis(df: pd.DataFrame, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    assets = _asset_rows(df, predictions, high_bad=("vibration", "temperature"), low_bad=("pressure", "rpm"))
    fleet = float(np.mean([a["health_index"] for a in assets])) if assets else 0.0
    high_n = sum(1 for a in assets if a["risk_level"] == "High")
    worst = assets[0] if assets else None
    kpis = [
        _kpi(
            "asset_health_index",
            "Fleet health index",
            round(fleet, 1),
            "/100",
            _severity_from_health(fleet),
            "Mean of per-asset health from risk, RUL, vibration and temperature.",
        ),
        _kpi(
            "high_risk_assets",
            "High-risk assets",
            high_n,
            "assets",
            "high" if high_n else "low",
            "Count of assets whose Isolation Forest / RUL risk is High.",
        ),
    ]
    if worst:
        kpis.append(
            _kpi(
                "predicted_rul_days",
                f"Worst-asset RUL ({worst['machine_id']})",
                worst.get("predicted_rul_days") if worst.get("predicted_rul_days") is not None else "n/a",
                "days",
                _severity_from_health(worst["health_index"]),
                "Remaining useful life for the lowest-health asset.",
            )
        )
    return {"kpis": kpis, "by_asset": assets}


def _aviation_kpis(df: pd.DataFrame, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    assets = _asset_rows(
        df,
        predictions,
        high_bad=("egt", "cht", "vibration", "oil_temp", "temperature"),
        low_bad=("oil_pressure", "manifold_pressure"),
    )
    rel = mission_reliability_pct(predictions)
    worst = assets[0] if assets else None
    hours = remaining_mission_hours(worst["predicted_rul_days"] if worst else None)
    egt = _numeric(df, "egt", "exhaust_gas_temp")
    egt_val = round(float(egt.iloc[-1]), 1) if egt is not None else None
    egt_med = round(float(egt.median()), 1) if egt is not None else None
    egt_margin = None
    if egt_val is not None and egt_med is not None:
        egt_margin = round(egt_med + 40.0 - egt_val, 1)  # headroom vs typical redline offset
    kpis = [
        _kpi(
            "engine_health_index",
            "Piston-engine health",
            round(float(np.mean([a["health_index"] for a in assets])), 1) if assets else 0.0,
            "/100",
            _severity_from_health(float(np.mean([a["health_index"] for a in assets])) if assets else 0),
            "SIH26054: aero piston health from risk, EGT/CHT, oil pressure, vibration.",
        ),
        _kpi(
            "mission_reliability_pct",
            "Mission reliability",
            rel,
            "%",
            "high" if rel < 60 else ("medium" if rel < 80 else "low"),
            f"Chance the next {DEFAULT_MISSION_HOURS:.0f} h MALE-UAV sortie completes (fleet mean).",
        ),
        _kpi(
            "remaining_mission_hours",
            "Remaining mission hours (worst UAV)",
            hours,
            "h",
            "high" if hours < 8 else ("medium" if hours < 24 else "low"),
            f"RUL days × 24 × {DEFAULT_DUTY_CYCLE:.0%} duty cycle for the weakest airframe.",
        ),
        _kpi(
            "egt_margin",
            "EGT margin (latest vs typical)",
            egt_margin if egt_margin is not None else "n/a",
            "°C",
            "high" if (egt_margin is not None and egt_margin < 5) else "low",
            "Positive = cooler than a demo redline offset (median + 40 °C).",
        ),
    ]
    return {"kpis": kpis, "by_asset": assets}


def _auto_kpis(df: pd.DataFrame, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    assets = _asset_rows(
        df,
        predictions,
        high_bad=("coolant_temp", "vibration", "engine_load", "temperature"),
        low_bad=("oil_pressure",),
    )
    fleet = float(np.mean([a["health_index"] for a in assets])) if assets else 0.0
    oil = _numeric(df, "oil_pressure", "oil_psi")
    oil_latest = float(oil.iloc[-1]) if oil is not None else None
    oil_ok = oil_latest is None or oil_latest >= 20.0
    cool = _numeric(df, "coolant_temp", "ect", "temperature")
    thermal = None
    if cool is not None:
        thermal = round(max(0.0, 105.0 - float(cool.iloc[-1])), 1)
    kpis = [
        _kpi(
            "powertrain_health_index",
            "Powertrain health",
            round(fleet, 1),
            "/100",
            _severity_from_health(fleet),
            "Generic ICE engine + gearbox health (not OEM/trim/EV-specific).",
        ),
        _kpi(
            "oil_pressure_status",
            "Oil pressure",
            round(oil_latest, 1) if oil_latest is not None else "n/a",
            "psi",
            "low" if oil_ok else "high",
            "Latest oil pressure. Below 20 psi is flagged (generic ICE threshold, not OEM spec).",
        ),
        _kpi(
            "thermal_headroom",
            "Coolant headroom to 105 °C",
            thermal if thermal is not None else "n/a",
            "°C",
            "high" if (thermal is not None and thermal < 8) else "low",
            "Generic boiling-margin proxy — not a brand thermostat map.",
        ),
    ]
    return {"kpis": kpis, "by_asset": assets}


def _oil_kpis(df: pd.DataFrame, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    assets = _asset_rows(
        df,
        predictions,
        high_bad=("polish_rod_load", "vibration", "tubing_pressure"),
        low_bad=("pump_fillage", "production_bbl"),
    )
    fleet = float(np.mean([a["health_index"] for a in assets])) if assets else 0.0
    fill = _numeric(df, "pump_fillage", "fillage")
    fill_pct = round(float(fill.iloc[-1]), 1) if fill is not None else None
    prod = _numeric(df, "production_bbl", "bopd", "oil_bbl")
    prod_val = round(float(prod.iloc[-1]), 2) if prod is not None else None
    kpis = [
        _kpi(
            "well_health_index",
            "Well / SRP health",
            round(fleet, 1),
            "/100",
            _severity_from_health(fleet),
            "SIH26120: beam-pump well health from risk, rod load, fillage, production.",
        ),
        _kpi(
            "pump_fillage_pct",
            "Pump fillage",
            fill_pct if fill_pct is not None else "n/a",
            "%",
            "high" if (fill_pct is not None and fill_pct < 70) else "low",
            "Latest fillage. Incomplete fill often means gas interference or leaky valves.",
        ),
        _kpi(
            "production_rate",
            "Latest production",
            prod_val if prod_val is not None else "n/a",
            "bbl",
            "medium" if prod_val is not None and prod_val < 20 else "low",
            "Latest production reading in the file (not a allocated field forecast).",
        ),
    ]
    return {"kpis": kpis, "by_asset": assets}


_DISPATCH = {
    "plant_rotating": _plant_kpis,
    "aviation_uav_piston": _aviation_kpis,
    "automotive_powertrain": _auto_kpis,
    "oil_srp": _oil_kpis,
}


def compute_pack_kpis(
    pack_id: Optional[str],
    df: Optional[pd.DataFrame],
    predictions: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Return pack KPIs + per-asset health. Never raises on empty inputs."""
    pack = get_pack(pack_id)
    pid = pack["id"]
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    preds = list(predictions or [])
    body = _DISPATCH.get(pid, _plant_kpis)(frame, preds)
    worst = body["by_asset"][0] if body["by_asset"] else None
    narrative = _narrative(pack, body["kpis"], worst)
    return {
        "pack_id": pid,
        "label": pack["label"],
        "sih": pack.get("sih"),
        "twin_kind": pack["twin_kind"],
        "hotspot": pack["hotspot"],
        "kpis": body["kpis"],
        "by_asset": body["by_asset"],
        "narrative": narrative,
    }


def _narrative(pack: dict[str, Any], kpis: list[dict[str, Any]], worst: Optional[dict[str, Any]]) -> str:
    bits = [pack["label"]]
    if pack.get("sih"):
        bits.append(f"({pack['sih']})")
    if worst:
        bits.append(
            f"weakest asset {worst['machine_id']} health {worst['health_index']}/100 "
            f"risk {worst['risk_level']}"
        )
    else:
        bits.append("no assets scored yet — run Anomaly & RUL or load a demo CSV")
    top = ", ".join(f"{k['label']} {k['value']}{(' ' + k['unit']) if k['unit'] else ''}" for k in kpis[:3])
    if top:
        bits.append(top)
    return " — ".join(bits)


def insight_cards_from_kpis(bundle: dict[str, Any]) -> list[dict[str, str]]:
    """Same shape as insights_engine cards so Insights can append them."""
    cards: list[dict[str, str]] = []
    pack_label = bundle.get("label") or "Pack"
    sih = bundle.get("sih")
    title = f"{pack_label} KPIs" + (f" ({sih})" if sih else "")
    cards.append(
        {
            "title": title,
            "severity": "info",
            "message": bundle.get("narrative") or "",
        }
    )
    for kpi in bundle.get("kpis") or []:
        sev = kpi.get("severity") or "info"
        if sev == "low":
            sev = "info"
        cards.append(
            {
                "title": str(kpi.get("label")),
                "severity": sev,
                "message": f"{kpi.get('value')} {kpi.get('unit', '')}. {kpi.get('hint', '')}".strip(),
            }
        )
    worst = (bundle.get("by_asset") or [None])[0]
    if worst and worst.get("risk_level") == "High":
        cards.append(
            {
                "title": f"Inspect {worst['machine_id']} first",
                "severity": "high",
                "message": (
                    f"{worst['machine_id']} is the weakest asset in the {pack_label} pack "
                    f"(health {worst['health_index']}/100, hotspot: {bundle.get('hotspot')}). "
                    "This is a degradation / risk flag, not a confirmed failure date."
                ),
            }
        )
    return cards


def empty_bundle(pack_id: Optional[str] = None) -> dict[str, Any]:
    return compute_pack_kpis(pack_id or DEFAULT_PACK_ID, pd.DataFrame(), [])
