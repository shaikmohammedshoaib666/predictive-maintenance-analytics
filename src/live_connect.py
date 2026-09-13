"""Layer 4 — Live Connect (ingestion bloodstream).

Simulator / MQTT / poll / OPC-UA all produce canonical rows. ``compute_live_status``
runs the **same** physics red-lines as batch, Isolation Forest on the buffer (or a
fitted detector), and RUL from a fitted model or a degradation proxy — not a
second brain.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.ata100 import ata_label
from src.assets import go_nogo
from src.live_schema import LIVE_SCORE_SENSORS, coerce_live_payload
from src.pack_kpis import remaining_mission_hours, risk_to_health
from src.physics_rules import (
    FLAG_ALERT,
    FLAG_FAULT,
    FLAG_LABEL,
    FLAG_RULE,
    FAULT_NONE,
    advisory_action,
    apply_physics_rules,
    mission_success_pct,
)

SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm"]
DEFAULT_STALE_AFTER_S = 30
AVIATION_PACK = "aviation_uav_piston"
# Keep IF + physics off the growing flight log so the Streamlit fragment cannot freeze.
LIVE_SCORE_TAIL = 240
# UI auto-refresh only. Simulator/MQTT still tick on LIVE_TICK_S (~2s). 0 = Refresh button only.
DEFAULT_LIVE_REFRESH_S = 60


def live_refresh_seconds(env: Optional[Any] = None) -> int:
    """Seconds between *screen* refreshes. Default 60. ``0`` disables auto-refresh.

    Never default to 3 — that was confused with the HTTP 429 reconnect storm
    (``st.fragment(run_every=3)(_live_body)()`` minting a new fragment each parent rerun).
    """
    import os

    source = env if env is not None else os.environ
    raw = ""
    if hasattr(source, "get"):
        raw = str(source.get("LIVE_REFRESH_SECONDS", "") or "")
    if not raw and env is None:
        try:
            import config as _cfg

            raw = str(_cfg._setting("LIVE_REFRESH_SECONDS", "") or "")
        except Exception:
            raw = ""
    if not raw:
        raw = str(DEFAULT_LIVE_REFRESH_S)
    try:
        n = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        n = DEFAULT_LIVE_REFRESH_S
    if n < 0:
        return 0
    return n


# Healthy baseline (mean, std) per sensor — plant pack.
_BASE = {
    "temperature": (70.0, 2.0),
    "vibration": (2.0, 0.3),
    "pressure": (100.0, 3.0),
    "rpm": (1500.0, 20.0),
}

# Rotax-class piston demo (matches sample_data/aviation_uav_piston.csv scales).
_BASE_AVIATION = {
    "cht": (165.0, 3.0),
    "egt": (680.0, 8.0),
    "oil_pressure": (62.0, 1.5),
    "oil_temp": (88.0, 2.0),
    "vibration": (1.4, 0.2),
    "rpm": (2450.0, 25.0),
    "fuel_flow": (2.6, 0.08),
    "altitude": (8500.0, 80.0),
    "throttle": (72.0, 2.0),
    "manifold_pressure": (28.5, 0.3),
}


def default_machines(n: int = 4, pack_id: Optional[str] = None) -> list[str]:
    count = max(1, int(n))
    if str(pack_id or "") == AVIATION_PACK:
        return [f"UAV-{i + 1:02d}" for i in range(count)]
    return [f"M-{i + 1:03d}" for i in range(count)]


def simulate_batch(
    machines: list[str],
    tick: int,
    *,
    start_ts: Optional[pd.Timestamp] = None,
    freq_seconds: int = 5,
    failing: Optional[str] = None,
    stress: float = 0.0,
    pack_id: Optional[str] = None,
) -> pd.DataFrame:
    """One reading per machine for `tick`. `failing` drifts with `stress` (0..1).

    Aviation packets go through ``coerce_live_payload`` so Simulator uses the
    same schema gate as MQTT (uav_id JSON → machine_id).
    """
    base_ts = start_ts or pd.Timestamp.utcnow().floor("s")
    ts = base_ts + pd.to_timedelta(tick * freq_seconds, unit="s")
    stress = float(max(0.0, min(1.0, stress)))
    rng = np.random.default_rng(tick * 7919 + 13)
    aviation = str(pack_id or "") == AVIATION_PACK
    rows: list[dict[str, Any]] = []
    for m in machines:
        drift = stress if (failing and m == failing) else 0.0
        if aviation:
            raw = _aviation_tick(m, ts, rng, drift)
            row = coerce_live_payload(raw, pack_id=pack_id, fallback_machine=str(m))
            row.pop("raw", None)
        else:
            row = {"timestamp": ts, "machine_id": m}
            for c in SENSOR_COLS:
                mean, sd = _BASE[c]
                val = rng.normal(mean, sd)
                if c == "temperature":
                    val += drift * 45.0
                elif c == "vibration":
                    val += drift * 7.0
                elif c == "pressure":
                    val += drift * 18.0
                elif c == "rpm":
                    val -= drift * 260.0
                if rng.random() < 0.02 + drift * 0.15:
                    val += rng.normal(0, sd * 5)
                row[c] = round(float(val), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _aviation_tick(machine_id: str, ts: pd.Timestamp, rng: Any, drift: float) -> dict[str, Any]:
    """Semi-structured MQTT-shaped packet (uav_id + CHT/EGT). Oil in psi (demo CSV)."""
    def _n(name: str) -> float:
        mean, sd = _BASE_AVIATION[name]
        return float(rng.normal(mean, sd))

    cht = _n("cht") + drift * 48.0
    egt = _n("egt") + drift * 110.0
    oil = _n("oil_pressure") - drift * 28.0
    vib = _n("vibration") + drift * 4.8
    rpm = _n("rpm") - drift * 220.0
    if rng.random() < 0.02 + drift * 0.12:
        egt += float(rng.uniform(20, 55))
        cht += float(rng.uniform(8, 18))
        vib += float(rng.uniform(0.8, 2.2))
    return {
        "uav_id": machine_id,
        "timestamp": ts.isoformat(),
        "rpm": round(rpm, 1),
        "egt": round(egt, 1),
        "cht": round(cht, 1),
        "oil_pressure": round(oil, 2),
        "oil_temperature": round(_n("oil_temp") + drift * 22.0, 1),
        "vibration": round(vib, 3),
        "fuel_flow": round(_n("fuel_flow") + drift * 0.45, 3),
        "altitude": round(_n("altitude"), 0),
        "throttle": round(float(np.clip(_n("throttle") + drift * 8.0, 20, 100)), 1),
        "manifold_pressure": round(_n("manifold_pressure") - drift * 2.5, 2),
    }


def append_to_buffer(
    buffer: Optional[pd.DataFrame], batch: pd.DataFrame, max_rows: int = 1500
) -> pd.DataFrame:
    """Append a batch and keep only the most recent `max_rows`."""
    if batch is None or batch.empty:
        return buffer if buffer is not None else pd.DataFrame()
    if buffer is None or buffer.empty:
        out = batch.copy()
    else:
        out = pd.concat([buffer, batch], ignore_index=True)
    if len(out) > max_rows:
        out = out.tail(max_rows).reset_index(drop=True)
    return out


def risk_from_rate(rate_pct: float) -> str:
    if rate_pct >= 15:
        return "High"
    if rate_pct >= 5:
        return "Medium"
    return "Low"


def _iso_ts(value: Any) -> str:
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return ""
        return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


def last_seen_and_stale(
    last_ts: Any,
    *,
    now: Optional[pd.Timestamp] = None,
    stale_after_s: int = DEFAULT_STALE_AFTER_S,
) -> dict[str, Any]:
    """Per-asset last-seen + stale flag. ``stale_after_s`` is wall-clock, not a hangar SLA."""
    try:
        window = int(stale_after_s)
    except (TypeError, ValueError):
        window = DEFAULT_STALE_AFTER_S
    if window < 1:
        window = DEFAULT_STALE_AFTER_S
    clock = now if now is not None else pd.Timestamp.utcnow()
    try:
        ts = pd.to_datetime(last_ts, utc=True, errors="coerce")
    except Exception:
        ts = pd.NaT
    if pd.isna(ts):
        return {
            "last_seen": "",
            "last_seen_age_s": None,
            "stale": True,
            "stale_after_s": window,
        }
    clock = pd.Timestamp(clock)
    if clock.tzinfo is None:
        clock = clock.tz_localize("UTC")
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    age = float((clock - ts).total_seconds())
    return {
        "last_seen": _iso_ts(ts),
        "last_seen_age_s": round(age, 1),
        "stale": bool(age > window),
        "stale_after_s": window,
    }


def _proxy_rul_days(rate_pct: float, physics_alert: bool) -> int:
    base = 28.0 * (1.0 - min(float(rate_pct), 85.0) / 100.0)
    if physics_alert:
        base *= 0.42
    return max(1, int(round(base)))


def _score_anomaly_rates(
    buffer: pd.DataFrame,
    *,
    contamination: float,
    detector: Any = None,
) -> dict[str, float]:
    sensor_cols = [c for c in LIVE_SCORE_SENSORS if c in buffer.columns]
    if not sensor_cols:
        sensor_cols = [c for c in SENSOR_COLS if c in buffer.columns]
    if not sensor_cols:
        return {}
    work = buffer[["machine_id", *sensor_cols]].copy()
    for c in sensor_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=sensor_cols)
    if len(work) < 12 or len(sensor_cols) < 2:
        return {}
    if len(work) > 240:
        work = work.tail(240)
    if detector is not None and getattr(detector, "is_fitted", False):
        try:
            feats = [c for c in detector.feature_columns if c in work.columns]
            if len(feats) >= 2:
                flagged = detector.predict(work) == -1
                work = work.assign(_anom=flagged)
                return (work.groupby("machine_id")["_anom"].mean() * 100.0).round(2).to_dict()
        except Exception:
            pass
    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(
        n_estimators=30,
        contamination=contamination,
        random_state=42,
        n_jobs=1,
    )
    flagged = iso.fit_predict(work[sensor_cols].values) == -1
    work = work.assign(_anom=flagged)
    return (work.groupby("machine_id")["_anom"].mean() * 100.0).round(2).to_dict()


def _rul_lookup(
    buffer: pd.DataFrame,
    predictor: Any,
) -> dict[str, int]:
    if predictor is None or not getattr(predictor, "is_fitted", False):
        return {}
    try:
        recs = predictor.predict_latest_per_machine(buffer)
    except Exception:
        return {}
    out: dict[str, int] = {}
    for rec in recs or []:
        mid = str(rec.get("machine_id") or "")
        try:
            out[mid] = int(rec.get("predicted_rul_days"))
        except (TypeError, ValueError):
            continue
    return out


def compute_live_status(
    buffer: pd.DataFrame,
    *,
    contamination: float = 0.08,
    now: Optional[pd.Timestamp] = None,
    stale_after_s: int = DEFAULT_STALE_AFTER_S,
    pack_id: Optional[str] = None,
    anomaly_detector: Any = None,
    rul_predictor: Any = None,
) -> list[dict[str, Any]]:
    """Per-machine live status: physics Layer 1 + IF + RUL (fitted or proxy)."""
    if buffer is None or buffer.empty or "machine_id" not in buffer.columns:
        return []

    counts = buffer.groupby("machine_id").size().to_dict()
    window = buffer.tail(LIVE_SCORE_TAIL) if len(buffer) > LIVE_SCORE_TAIL else buffer
    latest = window
    if "timestamp" in window.columns:
        latest = window.sort_values("timestamp").groupby("machine_id", as_index=False).tail(1)
    else:
        latest = window.groupby("machine_id", as_index=False).tail(1)
    annotated = apply_physics_rules(latest, pack_id)
    rates = _score_anomaly_rates(
        window,
        contamination=contamination,
        detector=anomaly_detector,
    )
    rul_map = _rul_lookup(window, rul_predictor)

    out: list[dict[str, Any]] = []
    clock = now if now is not None else pd.Timestamp.utcnow()
    phys_by_id: dict[str, Any] = {}
    if "machine_id" in annotated.columns:
        for _, rec in annotated.iterrows():
            phys_by_id[str(rec.get("machine_id"))] = rec
    for m, grp in buffer.groupby("machine_id"):
        rate = float(rates.get(m, 0.0))
        raw = grp.iloc[-1]
        last = phys_by_id.get(str(m), raw)
        last_ts = raw.get("timestamp") if "timestamp" in grp.columns else None
        seen = last_seen_and_stale(last_ts, now=clock, stale_after_s=stale_after_s)
        fault = str(last.get(FLAG_FAULT) or FAULT_NONE)
        rule = str(last.get(FLAG_RULE) or "")
        label = str(last.get(FLAG_LABEL) or FAULT_NONE)
        alert = bool(last.get(FLAG_ALERT)) or fault not in ("", FAULT_NONE, "nan")
        risk = risk_from_rate(rate)
        if alert and risk == "Low":
            risk = "Medium"
        if alert and rate >= 12:
            risk = "High"
        action = advisory_action(risk_level=risk, physics_fault=fault)
        decision = go_nogo(action, risk)
        rul_days = rul_map.get(str(m))
        if rul_days is None:
            rul_days = _proxy_rul_days(rate, alert)
        hours = remaining_mission_hours(rul_days)
        health = risk_to_health(risk)
        if alert:
            health = max(2.0, health - 22.0)
        health = round(float(health), 1)
        out.append(
            {
                "machine_id": str(m),
                "risk_level": risk,
                "anomaly_rate_pct": rate,
                "predicted_rul_days": int(rul_days),
                "predicted_rul_hours": hours,
                "health_index": health,
                "mission_success_pct": mission_success_pct(health),
                "go_nogo": decision,
                "physics_fault": fault,
                "physics_rule": rule,
                "physics_label": label,
                "ata_chapter": ata_label(fault, rule),
                "advisory": action,
                "n_rows": int(counts.get(m, len(grp))),
                "last_temperature": _safe_float(raw.get("temperature")),
                "last_vibration": _safe_float(raw.get("vibration")),
                "last_rpm": _safe_float(raw.get("rpm")),
                "last_cht": _safe_float(raw.get("cht") if "cht" in raw.index else raw.get("temperature")),
                "last_egt": _safe_float(raw.get("egt")),
                "last_oil_pressure": _safe_float(
                    raw.get("oil_pressure") if "oil_pressure" in raw.index else raw.get("pressure")
                ),
                "last_oil_temp": _safe_float(raw.get("oil_temp")),
                "last_fuel_flow": _safe_float(raw.get("fuel_flow")),
                "last_seen": seen["last_seen"],
                "last_seen_age_s": seen["last_seen_age_s"],
                "stale": seen["stale"],
                "stale_after_s": seen["stale_after_s"],
            }
        )
    order = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}
    out.sort(key=lambda d: (order.get(d["risk_level"], 3), -d["anomaly_rate_pct"]))
    return out


def flight_log_frame(buffer: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Drop MQTT ``raw`` blobs so the batch pipeline can Map → Detect."""
    if buffer is None or not isinstance(buffer, pd.DataFrame) or buffer.empty:
        return pd.DataFrame()
    out = buffer.copy()
    if "raw" in out.columns:
        out = out.drop(columns=["raw"])
    return out


def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return None if pd.isna(f) else round(f, 3)
    except Exception:
        return None


def poll_url_increment(url: str, cache_dir, offset: int, n: int = 20) -> tuple[pd.DataFrame, int]:
    """Read the next `n` rows from a CSV URL starting at `offset` (real-feed polling)."""
    from src.url_ingest import load_from_url

    sql = (
        "SELECT * FROM read_csv_auto('{source}', header=true) "
        f"LIMIT {int(n)} OFFSET {int(offset)}"
    )
    df, _meta = load_from_url(url, cache_dir=cache_dir, sql_query=sql, force_cache=True)
    return df, offset + len(df)
