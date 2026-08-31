"""Layer 4 — Live Connect.

Streaming ingest that feeds the same anomaly pipeline in near-real-time. Two
sources:

* **Simulator** — generates per-machine sensor readings each tick, with a
  selectable asset that drifts toward failure so the live risk (and the 3D
  twin) visibly escalate. Lets the feature be demoed without a broker.
* **Poll CSV URL** — reads the next window of rows from a remote CSV each tick
  (via the DuckDB URL-ingest layer), the pattern a real OPC-UA/MQTT gateway or
  historian export would follow.

State is kept as a rolling in-memory buffer; `compute_live_status` scores it
with IsolationForest to produce a per-machine live risk that Layer 3 consumes.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm"]

# Healthy baseline (mean, std) per sensor.
_BASE = {
    "temperature": (70.0, 2.0),
    "vibration": (2.0, 0.3),
    "pressure": (100.0, 3.0),
    "rpm": (1500.0, 20.0),
}


def default_machines(n: int = 4) -> list[str]:
    return [f"M-{i + 1:03d}" for i in range(max(1, n))]


def simulate_batch(
    machines: list[str],
    tick: int,
    *,
    start_ts: Optional[pd.Timestamp] = None,
    freq_seconds: int = 5,
    failing: Optional[str] = None,
    stress: float = 0.0,
) -> pd.DataFrame:
    """One reading per machine for `tick`. `failing` drifts with `stress` (0..1)."""
    base_ts = start_ts or pd.Timestamp.utcnow().floor("s")
    ts = base_ts + pd.to_timedelta(tick * freq_seconds, unit="s")
    stress = float(max(0.0, min(1.0, stress)))
    rng = np.random.default_rng(tick * 7919 + 13)
    rows: list[dict[str, Any]] = []
    for m in machines:
        drift = stress if (failing and m == failing) else 0.0
        row: dict[str, Any] = {"timestamp": ts, "machine_id": m}
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
            if rng.random() < 0.02 + drift * 0.15:  # occasional spike
                val += rng.normal(0, sd * 5)
            row[c] = round(float(val), 3)
        rows.append(row)
    return pd.DataFrame(rows)


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


def compute_live_status(buffer: pd.DataFrame, *, contamination: float = 0.08) -> list[dict[str, Any]]:
    """Per-machine live status.

    IsolationForest is fit **jointly** across the whole buffer so a genuinely
    failing asset captures a disproportionate share of the anomalies (its own
    flagged fraction can climb well past 15% → High) while healthy assets stay
    near 0% — instead of every machine sitting at the fixed contamination rate.
    """
    if buffer is None or buffer.empty or "machine_id" not in buffer.columns:
        return []
    from sklearn.ensemble import IsolationForest

    sensor_cols = [c for c in SENSOR_COLS if c in buffer.columns]
    rates: dict[str, float] = {}
    if sensor_cols:
        work = buffer[["machine_id", *sensor_cols]].copy()
        for c in sensor_cols:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        work = work.dropna(subset=sensor_cols)
        if len(work) >= 12 and len(sensor_cols) >= 2:
            iso = IsolationForest(contamination=contamination, random_state=42)
            flagged = iso.fit_predict(work[sensor_cols].values) == -1
            work = work.assign(_anom=flagged)
            rates = (work.groupby("machine_id")["_anom"].mean() * 100.0).round(2).to_dict()

    out: list[dict[str, Any]] = []
    for m, grp in buffer.groupby("machine_id"):
        rate = float(rates.get(m, 0.0))
        last = grp.iloc[-1]
        out.append(
            {
                "machine_id": str(m),
                "risk_level": risk_from_rate(rate),
                "anomaly_rate_pct": rate,
                "predicted_rul_days": None,
                "n_rows": int(len(grp)),
                "last_temperature": _safe_float(last.get("temperature")),
                "last_vibration": _safe_float(last.get("vibration")),
            }
        )
    order = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}
    out.sort(key=lambda d: (order.get(d["risk_level"], 3), -d["anomaly_rate_pct"]))
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
