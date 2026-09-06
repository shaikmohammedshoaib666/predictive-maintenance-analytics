"""Industrial PdM brief + LlamaIndex/Gemini ask, grounded in the uploaded table."""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.quality_checks import find_col

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
RETIRED_GEMINI_ALIASES = frozenset(
    {
        "gemini-2.0-flash",
        "gemini-flash-latest",
        "gemini-flash-latest-latest",
        "gemini-1.5-flash",
        "gemini-pro",
    }
)

PROXY_DISCLAIMER = (
    "This is a risk / degradation proxy, not a confirmed failure date."
)
NO_KEY_MESSAGE = (
    "Set GEMINI_API_KEY in the sidebar, `.env`, or Streamlit secrets to enable Gemini."
)
CHAT_SCOPE_CAPTION = (
    "Answers use **this upload** (sensor rows + Isolation Forest / RUL columns) — "
    "not a general LLM essay."
)

_SCORE_COL = "anomaly_score"
_FLAG_COL = "is_anomaly"
_SKIP_NUMERIC = frozenset(
    {
        "failure_within_days",
        "predicted_rul_days",
        _SCORE_COL,
        "is_anomaly",
        "physics_alert",
        "physics_fault",
        "physics_rule",
        "physics_label",
    }
)
_PRODUCTION_HINTS = (
    "throughput",
    "production_rate",
    "units_per_hour",
    "output_rate",
    "units_per_min",
    "production",
    "output",
)
_SPEED_HINTS = ("rpm", "speed", "rotational")


def get_gemini_model(raw: str = "") -> str:
    """Prefer GEMINI_MODEL env/secrets; remap retired aliases to gemini-3.6-flash."""
    name = (raw or os.getenv("GEMINI_MODEL", "") or DEFAULT_GEMINI_MODEL).strip()
    if name.lower().startswith("models/"):
        name = name[7:]
    if not name or name.lower() in RETIRED_GEMINI_ALIASES:
        return DEFAULT_GEMINI_MODEL
    return name


def _resolve_gemini_model(raw: str = "") -> str:
    return get_gemini_model(raw)


def persist_session_gemini_key(key: str) -> None:
    try:
        import streamlit as st

        st.session_state.gemini_api_key_override = (key or "").strip()
    except Exception:
        pass


def get_gemini_api_key() -> str:
    try:
        import streamlit as st

        override = str(st.session_state.get("gemini_api_key_override", "") or "").strip()
        if override:
            return override
    except Exception:
        pass
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return key
    try:
        import streamlit as st

        return str(st.secrets.get("GEMINI_API_KEY", "") or "")
    except Exception:
        return ""


def _gemini_key() -> str:
    return get_gemini_api_key()


def mask_key(key: str) -> str:
    key = (key or "").strip()
    if len(key) > 12:
        return key[:6] + "…" + key[-4:]
    return "set" if key else "missing"


def test_gemini_connection(prompt: str = "Reply with OK") -> dict[str, Any]:
    """Tiny generate_content probe — never swallows 404s or other API errors."""
    key = get_gemini_api_key()
    model_name = get_gemini_model()
    if not key:
        return {
            "ok": False,
            "model": model_name,
            "error": "No Gemini API key (session, GEMINI_API_KEY env, or Streamlit secrets).",
        }
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", None) or str(resp) or "").strip()
        if not text:
            return {
                "ok": False,
                "model": model_name,
                "error": "Gemini returned an empty response. Check quota / model name.",
            }
        return {"ok": True, "model": model_name, "text": text[:240]}
    except Exception as exc:
        return {"ok": False, "model": model_name, "error": str(exc)}


def gemini_issue_from_raw(raw: str, *, attempted: bool) -> Optional[str]:
    if not attempted:
        return None
    text = (raw or "").strip()
    if not text:
        return "Gemini returned an empty response. Offline methods continued."
    if text.startswith("[Gemini error]"):
        return text
    return None


def find_asset_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, "machine_id", "machine", "asset_id", "asset", "equipment")


def find_production_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, *_PRODUCTION_HINTS)


def find_speed_col(df: pd.DataFrame) -> Optional[str]:
    prod = find_production_col(df)
    if prod:
        return prod
    return find_col(df, *_SPEED_HINTS)


def _sensor_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.select_dtypes(include="number").columns:
        if str(c) in _SKIP_NUMERIC or str(c).endswith(("_bin", "_smooth")):
            continue
        cols.append(c)
    return cols


def _has_failure_labels(df: pd.DataFrame, predictions: Optional[list[dict]] = None) -> bool:
    if "failure_within_days" in df.columns and pd.to_numeric(
        df["failure_within_days"], errors="coerce"
    ).notna().any():
        if predictions and any(p.get("is_proxy") for p in predictions):
            return False
        return True
    if predictions and all(p.get("is_proxy") for p in predictions):
        return False
    if predictions and any(p.get("label_source") == "failure_within_days" for p in predictions):
        return True
    return False


def _fmt_money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}" if abs(value) >= 10 else f"{sign}${abs(value):,.2f}"


def attach_rul_columns(df: pd.DataFrame, predictions: Optional[list[dict]]) -> pd.DataFrame:
    """Broadcast latest RUL/risk onto every row of that asset (for chat + ranking)."""
    out = df.copy()
    if not predictions:
        return out
    asset = find_asset_col(out)
    rul_map = {str(p.get("machine_id")): p.get("predicted_rul_days") for p in predictions}
    risk_map = {str(p.get("machine_id")): p.get("risk_level") for p in predictions}
    if asset:
        keys = out[asset].astype(str)
        out["predicted_rul_days"] = keys.map(rul_map)
        out["risk_level"] = keys.map(risk_map)
    elif len(predictions) == 1:
        out["predicted_rul_days"] = predictions[0].get("predicted_rul_days")
        out["risk_level"] = predictions[0].get("risk_level")
    return out


def rank_assets(
    df: pd.DataFrame,
    predictions: Optional[list[dict]] = None,
    anomaly_summary: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """Rank assets by Isolation Forest score, RUL, and risk. Table-grounded only."""
    predictions = predictions or []
    asset = find_asset_col(df)
    pred_by_id = {str(p.get("machine_id")): p for p in predictions}
    sensors = _sensor_cols(df)

    if asset:
        ids = [str(v) for v in df[asset].dropna().unique().tolist()]
    else:
        ids = ["All"]

    ranked: list[dict[str, Any]] = []
    for aid in ids:
        sub = df if aid == "All" or asset is None else df[df[asset].astype(str) == aid]
        if sub.empty:
            continue
        rec: dict[str, Any] = {
            "machine_id": aid,
            "n_rows": int(len(sub)),
            "mean_anomaly_score": None,
            "anomaly_count": 0,
            "anomaly_rate_pct": 0.0,
            "predicted_rul_days": None,
            "risk_level": "Unknown",
            "why": [],
            "is_proxy": False,
        }
        if _SCORE_COL in sub.columns:
            scores = pd.to_numeric(sub[_SCORE_COL], errors="coerce").dropna()
            if len(scores):
                rec["mean_anomaly_score"] = round(float(scores.mean()), 4)
                rec["max_anomaly_score"] = round(float(scores.max()), 4)
        if _FLAG_COL in sub.columns:
            flags = sub[_FLAG_COL].astype(bool)
            rec["anomaly_count"] = int(flags.sum())
            rec["anomaly_rate_pct"] = round(100.0 * rec["anomaly_count"] / max(len(sub), 1), 2)
        pred = pred_by_id.get(aid)
        if pred:
            rec["predicted_rul_days"] = pred.get("predicted_rul_days")
            rec["risk_level"] = pred.get("risk_level") or rec["risk_level"]
            rec["is_proxy"] = bool(pred.get("is_proxy"))
            rec["label_source"] = pred.get("label_source")
        elif "predicted_rul_days" in sub.columns:
            rul = pd.to_numeric(sub["predicted_rul_days"], errors="coerce").dropna()
            if len(rul):
                rec["predicted_rul_days"] = int(round(float(rul.iloc[-1])))
            if "risk_level" in sub.columns and pd.notna(sub["risk_level"].iloc[-1]):
                rec["risk_level"] = str(sub["risk_level"].iloc[-1])

        rec["why"] = _asset_why(df, sub, sensors, rec)
        ranked.append(rec)

    def _sort_key(r: dict[str, Any]) -> tuple:
        risk_ord = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}
        rul = r.get("predicted_rul_days")
        score = r.get("mean_anomaly_score") or 0.0
        return (
            risk_ord.get(str(r.get("risk_level")), 3),
            rul if rul is not None else 10**9,
            -float(score),
        )

    ranked.sort(key=_sort_key)
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i
    if anomaly_summary and not ranked:
        return ranked
    return ranked


def _asset_why(
    df: pd.DataFrame,
    sub: pd.DataFrame,
    sensors: list[str],
    rec: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if rec.get("mean_anomaly_score") is not None:
        reasons.append(f"mean Isolation Forest score {rec['mean_anomaly_score']}")
    if rec.get("anomaly_count"):
        reasons.append(
            f"{rec['anomaly_count']} IF flags ({rec.get('anomaly_rate_pct', 0)}% of rows)"
        )
    rul = rec.get("predicted_rul_days")
    if rul is not None:
        tag = "RUL proxy" if rec.get("is_proxy") else "predicted RUL"
        reasons.append(f"{tag} {int(rul)} days")
    for col in sensors[:6]:
        series = pd.to_numeric(sub[col], errors="coerce").dropna()
        fleet = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 1 or len(fleet) < 3:
            continue
        latest = float(series.iloc[-1])
        med = float(fleet.median())
        if med == 0:
            continue
        gap_pct = (latest - med) / abs(med) * 100
        if abs(gap_pct) >= 12:
            verb = "spike" if gap_pct > 0 else "drop"
            reasons.append(f"`{col}` {verb} {latest:.2f} vs fleet median {med:.2f} ({gap_pct:+.0f}%)")
    return reasons[:6]


def inspect_this_week(
    ranked: list[dict[str, Any]],
    *,
    max_items: int = 8,
) -> list[dict[str, str]]:
    """Assets to inspect this week, with why (sensor spike / IF / low RUL)."""
    items: list[dict[str, str]] = []
    for rec in ranked:
        rul = rec.get("predicted_rul_days")
        high_if = (rec.get("mean_anomaly_score") or 0) > 0 and rec.get("anomaly_count", 0) > 0
        high_risk = rec.get("risk_level") == "High"
        medium_hot = rec.get("risk_level") == "Medium" and (
            (rul is not None and rul <= 14) or rec.get("anomaly_rate_pct", 0) >= 8
        )
        spike = any("spike" in w or "drop" in w for w in rec.get("why") or [])
        if not (high_risk or medium_hot or high_if or (spike and (rul is None or rul <= 21))):
            continue
        why = "; ".join(rec.get("why") or ["ranked high vs other assets in this upload"])
        proxy_note = f" {PROXY_DISCLAIMER}" if rec.get("is_proxy") or rec.get("risk_level") == "Unknown" else ""
        items.append(
            {
                "title": f"Inspect {rec['machine_id']} this week",
                "severity": rec.get("risk_level") or "Medium",
                "machine_id": rec["machine_id"],
                "message": (
                    f"**{rec['machine_id']}** should be inspected this week — {why}.{proxy_note}"
                ),
            }
        )
        if len(items) >= max_items:
            break
    return items


def flag_slow_running(df: pd.DataFrame) -> list[dict[str, str]]:
    """Flag assets below median on production/speed/rpm/throughput — no invented losses."""
    col = find_speed_col(df)
    asset = find_asset_col(df)
    if col is None or asset is None:
        return []
    work = df[[asset, col]].copy()
    work["_v"] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["_v"])
    if work.empty or work[asset].nunique() < 2:
        return []

    fleet_med = float(work["_v"].median())
    if fleet_med == 0:
        return []
    is_production = find_production_col(df) == col
    grouped = work.groupby(asset)["_v"].mean()
    flags: list[dict[str, str]] = []
    for aid, mean_v in grouped.items():
        own = work[work[asset] == aid]["_v"]
        own_med = float(own.median()) if len(own) >= 3 else fleet_med
        vs_fleet = (float(mean_v) - fleet_med) / abs(fleet_med) * 100
        vs_own = (float(mean_v) - own_med) / abs(own_med or 1) * 100
        slow_fleet = vs_fleet <= -8
        slow_own = is_production is False and vs_own <= -8
        if not (slow_fleet or slow_own):
            continue
        if is_production:
            msg = (
                f"**{aid}** is running slow on `{col}`: mean {mean_v:.2f} vs fleet median "
                f"{fleet_med:.2f} ({vs_fleet:.0f}%). Lost throughput vs the median line."
            )
        else:
            basis = "its own median" if slow_own else "fleet median"
            ref = own_med if slow_own else fleet_med
            pct = vs_own if slow_own else vs_fleet
            msg = (
                f"**{aid}** is running slow on `{col}`: mean {mean_v:.2f} vs {basis} "
                f"{ref:.2f} ({pct:.0f}%). Treat as a speed/degradation signal, not a confirmed jam."
            )
        flags.append(
            {
                "title": f"Slow running: {aid}",
                "severity": "Medium",
                "machine_id": str(aid),
                "column": col,
                "mean": float(mean_v),
                "fleet_median": fleet_med,
                "gap_pct": float(vs_fleet),
                "message": msg,
            }
        )
    flags.sort(key=lambda z: z.get("gap_pct", 0))
    return flags


def estimate_dollar_impact(
    ranked: list[dict[str, Any]],
    slow: Optional[list[dict[str, Any]]] = None,
    *,
    cost_per_hour: float = 0.0,
    cost_per_unit: float = 0.0,
    hours_if_stop: float = 8.0,
) -> list[dict[str, str]]:
    """$ at risk from user-entered $/hour or $/unit. Zero rates → no dollar claims."""
    items: list[dict[str, str]] = []
    slow = slow or []
    if cost_per_hour <= 0 and cost_per_unit <= 0:
        return items

    hours = max(float(hours_if_stop or 0), 0.0)
    if cost_per_hour > 0 and hours > 0:
        for rec in ranked:
            risk = rec.get("risk_level")
            if risk == "High":
                factor = 1.0
            elif risk == "Medium":
                factor = 0.5
            else:
                continue
            assumed_h = hours * factor
            dollars = assumed_h * cost_per_hour
            items.append(
                {
                    "title": f"$ at risk: {rec['machine_id']}",
                    "severity": risk or "Medium",
                    "machine_id": rec["machine_id"],
                    "dollars": dollars,
                    "message": (
                        f"**{rec['machine_id']}** ({risk} risk): if it stops for "
                        f"**{assumed_h:.1f} h**, **{_fmt_money(dollars)}** is at risk "
                        f"at {_fmt_money(cost_per_hour)}/hour. "
                        "Assumption uses the hours you entered — not a booked outage."
                    ),
                }
            )

    if cost_per_unit > 0:
        for rec in slow:
            gap = rec.get("mean")
            med = rec.get("fleet_median")
            col = rec.get("column") or "throughput"
            if gap is None or med is None:
                continue
            lost_units = max(0.0, float(med) - float(gap))
            if lost_units <= 0:
                continue
            per_hour = lost_units * cost_per_unit
            window = per_hour * hours if hours > 0 else per_hour
            items.append(
                {
                    "title": f"Lost production: {rec['machine_id']}",
                    "severity": "Medium",
                    "machine_id": rec["machine_id"],
                    "dollars": window,
                    "message": (
                        f"**{rec['machine_id']}** `{col}` is {lost_units:.2f} units below the fleet median. "
                        f"At {_fmt_money(cost_per_unit)}/unit that is **{_fmt_money(per_hour)} per hour**"
                        + (
                            f", or **{_fmt_money(window)}** over {hours:.1f} h."
                            if hours > 0
                            else "."
                        )
                    ),
                }
            )
    return items


def chart_business_insight(df: pd.DataFrame, x: str, y: str) -> str:
    """Sensor/asset insight from the plotted columns — not a sales forecast."""
    if x not in df.columns or y not in df.columns:
        return "Select valid X/Y columns for insights."
    work = df[[x, y]].copy()
    ynum = pd.to_numeric(work[y], errors="coerce")
    if ynum.notna().sum() < 3:
        return f"Not enough numeric values in `{y}` for a grounded reading."

    asset = find_asset_col(df)
    if (asset and x == asset) or (
        not pd.api.types.is_numeric_dtype(work[x])
        or work[x].nunique() < max(3, len(work) // 10)
    ):
        g = work.assign(_y=ynum).groupby(x, dropna=False)["_y"].mean().sort_values()
        if len(g) >= 2:
            low, high = g.index[0], g.index[-1]
            gap = float(g.iloc[-1] - g.iloc[0])
            return (
                f"**{high}** has the highest `{y}` (avg={g.iloc[-1]:.2f}); "
                f"**{low}** is lowest (avg={g.iloc[0]:.2f}). Gap={gap:.2f}. "
                "Compare against Isolation Forest flags before treating this as a failure."
            )

    tmp = pd.DataFrame({"x": pd.to_numeric(work[x], errors="coerce"), "y": ynum}).dropna()
    if len(tmp) < 5:
        return "Need more points for a numeric sensor reading."
    slope = float(np.polyfit(tmp["x"], tmp["y"], 1)[0])
    corr = float(tmp["x"].corr(tmp["y"]))
    direction = "rising" if slope > 0 else "falling"
    return (
        f"`{y}` is **{direction}** vs `{x}` (slope={slope:.4f}, corr={corr:.2f}). "
        "That is a trend on this upload, not a confirmed failure date."
    )


def build_industrial_brief(
    df: pd.DataFrame,
    predictions: Optional[list[dict]] = None,
    anomaly_summary: Optional[dict] = None,
    *,
    cost_per_hour: float = 0.0,
    cost_per_unit: float = 0.0,
    hours_if_stop: float = 8.0,
    used_synthetic: Optional[bool] = None,
) -> dict[str, Any]:
    """Rule-based plant brief: rank, inspect list, slow lines, optional $."""
    predictions = predictions or []
    anomaly_summary = anomaly_summary or {}
    labeled = _has_failure_labels(df, predictions)
    if used_synthetic is None:
        used_synthetic = (not labeled) and bool(predictions)

    ranked = rank_assets(df, predictions, anomaly_summary)
    inspect = inspect_this_week(ranked)
    slow = flag_slow_running(df)
    dollars = estimate_dollar_impact(
        ranked,
        slow,
        cost_per_hour=cost_per_hour,
        cost_per_unit=cost_per_unit,
        hours_if_stop=hours_if_stop,
    )

    insights: list[dict[str, str]] = []
    if used_synthetic or not labeled:
        insights.append(
            {
                "title": "How to read risk",
                "severity": "Info",
                "message": (
                    "No confirmed failure labels on this table (or RUL used a synthetic target). "
                    + PROXY_DISCLAIMER
                    + " Isolation Forest flags and remaining-life scores are ranking tools."
                ),
            }
        )
    else:
        insights.append(
            {
                "title": "How to read risk",
                "severity": "Info",
                "message": (
                    "RUL used `failure_within_days` on this upload. Days are a model estimate — "
                    "not a calendar failure date. Isolation Forest flags are unsupervised."
                ),
            }
        )

    if ranked:
        top = ranked[0]
        rul_txt = (
            f"remaining-life score ~{top['predicted_rul_days']} days"
            if top.get("predicted_rul_days") is not None
            else "no RUL score yet"
        )
        if_txt = (
            f"IF score {top['mean_anomaly_score']}"
            if top.get("mean_anomaly_score") is not None
            else "no IF score yet"
        )
        insights.append(
            {
                "title": "Highest-risk asset",
                "severity": top.get("risk_level") or "Medium",
                "message": (
                    f"**{top['machine_id']}** ranks first ({top.get('risk_level')} risk, {rul_txt}, {if_txt}). "
                    + ("; ".join(top.get("why")[:3]) if top.get("why") else "")
                    + (" " + PROXY_DISCLAIMER if top.get("is_proxy") or used_synthetic else "")
                ),
            }
        )

    insights.extend(inspect[:5])
    insights.extend(slow[:4])
    insights.extend(dollars[:6])

    if anomaly_summary.get("anomaly_rate_pct", 0) >= 8:
        insights.append(
            {
                "title": "Fleet anomaly rate",
                "severity": "High",
                "message": (
                    f"Isolation Forest flagged **{anomaly_summary['anomaly_rate_pct']}%** of rows "
                    f"({anomaly_summary.get('anomaly_count', 0)} of "
                    f"{anomaly_summary.get('total_records', 0)}). "
                    "That is an outlier rate on this upload, not a count of confirmed failures."
                ),
            }
        )

    if cost_per_hour <= 0 and cost_per_unit <= 0:
        insights.append(
            {
                "title": "Add a rate for $ impact",
                "severity": "Info",
                "message": (
                    "Enter **$/hour downtime** or **$/unit lost** to turn risk ranks into dollar figures. "
                    "Without a rate, this brief will not invent revenue."
                ),
            }
        )

    if not ranked and not inspect:
        insights.append(
            {
                "title": "Need scores",
                "severity": "Info",
                "message": (
                    "Load a sensor table and run **Detect anomalies + predict RUL** so assets can be "
                    "ranked from Isolation Forest scores and remaining-life estimates."
                ),
            }
        )

    return {
        "insights": insights,
        "ranked": ranked,
        "inspect": inspect,
        "slow": slow,
        "dollar": dollars,
        "used_synthetic": bool(used_synthetic),
        "labeled": bool(labeled),
    }


def generate_business_insights(
    df: pd.DataFrame,
    predictions: Optional[list[dict]] = None,
    anomaly_summary: Optional[dict] = None,
    *,
    cost_per_hour: float = 0.0,
    cost_per_unit: float = 0.0,
    hours_if_stop: float = 8.0,
    used_synthetic: Optional[bool] = None,
) -> list[dict[str, str]]:
    """Actionable PdM insights for the current table (rule-based)."""
    return build_industrial_brief(
        df,
        predictions,
        anomaly_summary,
        cost_per_hour=cost_per_hour,
        cost_per_unit=cost_per_unit,
        hours_if_stop=hours_if_stop,
        used_synthetic=used_synthetic,
    )["insights"]


def polish_brief_with_gemini(insights: list[dict[str, str]]) -> dict[str, Any]:
    """Tighten wording only. Never add assets, dollars, or failure dates."""
    key = get_gemini_api_key()
    model_name = get_gemini_model()
    if not key:
        return {"ok": False, "text": "", "error": NO_KEY_MESSAGE, "key_missing": True}
    bullets = "\n".join(f"- {i.get('title')}: {i.get('message')}" for i in insights[:10])
    prompt = (
        "You are a plant reliability analyst. Rewrite the bullets more tightly for a shift manager. "
        "Do NOT add machines, sensors, dollar amounts, or failure dates that are not in the source. "
        "Keep numbers exactly. If the source says proxy / not a confirmed failure date, keep that caveat.\n\n"
        f"{bullets}"
    )
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            return {
                "ok": False,
                "text": "",
                "error": "Gemini returned an empty polish. Rule-based brief is unchanged.",
                "key_missing": False,
            }
        return {"ok": True, "text": text, "error": None, "key_missing": False, "model": model_name}
    except Exception as exc:
        return {"ok": False, "text": "", "error": f"[Gemini error] {exc}", "key_missing": False}


def _row_documents(df: pd.DataFrame, max_rows: int = 400) -> list[str]:
    if df is None or df.empty:
        return []
    if _FLAG_COL in df.columns:
        flagged = df[df[_FLAG_COL].astype(bool)]
        rest = df[~df[_FLAG_COL].astype(bool)]
        take_f = min(len(flagged), max_rows // 2)
        sample = pd.concat([flagged.head(take_f), rest.head(max_rows - take_f)])
    else:
        sample = df.head(max_rows)
    docs = []
    for i, row in sample.iterrows():
        parts = [f"row_id={i}"]
        for c in sample.columns:
            val = row[c]
            if pd.notna(val):
                parts.append(f"{c}={val}")
        docs.append(" | ".join(parts))
    return docs


def _brief_documents(
    ranked: Optional[list[dict]] = None,
    predictions: Optional[list[dict]] = None,
    inspect: Optional[list[dict]] = None,
) -> list[str]:
    docs: list[str] = []
    for rec in ranked or []:
        why = "; ".join(rec.get("why") or [])
        docs.append(
            "ASSET_RANK "
            f"rank={rec.get('rank')} machine_id={rec.get('machine_id')} "
            f"risk_level={rec.get('risk_level')} predicted_rul_days={rec.get('predicted_rul_days')} "
            f"mean_anomaly_score={rec.get('mean_anomaly_score')} "
            f"anomaly_count={rec.get('anomaly_count')} why={why}"
        )
    for p in predictions or []:
        docs.append(
            "RUL_PRED "
            f"machine_id={p.get('machine_id')} predicted_rul_days={p.get('predicted_rul_days')} "
            f"risk_level={p.get('risk_level')} is_proxy={p.get('is_proxy')} "
            f"label_source={p.get('label_source')} message={p.get('message')}"
        )
    for item in inspect or []:
        docs.append(f"INSPECT machine_id={item.get('machine_id')} {item.get('message')}")
    return docs


def make_insight_index(
    df: pd.DataFrame,
    predictions: Optional[list[dict]] = None,
    ranked: Optional[list[dict]] = None,
    inspect: Optional[list[dict]] = None,
) -> tuple:
    """Index this upload's rows plus ranked/RUL brief docs for Ask."""
    idx = InsightIndex()
    scored = attach_rul_columns(df, predictions)
    meta = idx.build(scored, extra_docs=_brief_documents(ranked, predictions, inspect))
    return idx, meta


class InsightIndex:
    """LlamaIndex vector index with TF-IDF / keyword fallback (no Streamlit coupling)."""

    def __init__(self) -> None:
        self.docs: list[str] = []
        self.index_obj = None
        self._tfidf = None
        self._tfidf_mat = None
        self.meta: dict[str, Any] = {}

    def build(
        self,
        df: pd.DataFrame,
        extra_docs: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        self.docs = list(extra_docs or []) + _row_documents(df)
        mode = "keyword"
        self.index_obj = None
        error = ""
        note = ""
        if os.getenv("OPENAI_API_KEY"):
            try:
                from llama_index.core import Document, VectorStoreIndex

                documents = [Document(text=t) for t in self.docs]
                self.index_obj = VectorStoreIndex.from_documents(documents)
                mode = "llama_vector"
            except Exception as exc:
                mode = "keyword"
                msg = str(exc)
                if "unsupported operand type(s) for |" in msg:
                    error = (
                        "LlamaIndex import failed on Python 3.9 (install llama-index-core<0.11). "
                        f"{exc}"
                    )
                else:
                    error = msg
        else:
            note = (
                "LlamaIndex vector skipped (no OPENAI_API_KEY). "
                "Ask uses TF-IDF retrieval + Gemini generate_content on this upload."
            )

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            if self.docs:
                self._tfidf = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
                self._tfidf_mat = self._tfidf.fit_transform(self.docs)
                if mode != "llama_vector":
                    mode = "tfidf"
        except Exception as exc:
            if not error:
                error = str(exc)

        self.meta = {
            "ok": True,
            "n_docs": len(self.docs),
            "mode": mode,
            "error": error,
            "note": note,
            "built_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        return self.meta

    def search(self, query: str, top_k: int = 6) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        if self.index_obj is not None:
            try:
                engine = self.index_obj.as_query_engine(similarity_top_k=top_k)
                resp = engine.query(query)
                hits.append({"score": 1.0, "text": str(resp), "source": "llama_query_engine"})
                src = getattr(resp, "source_nodes", None) or []
                for n in src[:top_k]:
                    node = getattr(n, "node", n)
                    text = node.get_content() if hasattr(node, "get_content") else str(n)
                    hits.append(
                        {
                            "score": float(getattr(n, "score", 0) or 0),
                            "text": str(text),
                            "source": "llama_node",
                        }
                    )
                if hits:
                    return hits[:top_k]
            except Exception as exc:
                hits.append(
                    {
                        "score": 0.0,
                        "text": f"[LlamaIndex query error] {exc}",
                        "source": "llama_error",
                    }
                )

        if self._tfidf is not None and self._tfidf_mat is not None and self.docs:
            try:
                from sklearn.metrics.pairwise import cosine_similarity

                qv = self._tfidf.transform([query or ""])
                sims = cosine_similarity(qv, self._tfidf_mat).ravel()
                order = np.argsort(-sims)[:top_k]
                for i in order:
                    if float(sims[i]) <= 0:
                        continue
                    hits.append(
                        {
                            "score": float(sims[i]),
                            "text": self.docs[int(i)],
                            "source": "tfidf",
                        }
                    )
                if hits:
                    return hits[:top_k]
            except Exception as exc:
                hits.append(
                    {
                        "score": 0.0,
                        "text": f"[TF-IDF query error] {exc}",
                        "source": "tfidf_error",
                    }
                )

        q_tokens = set(re.findall(r"[a-zA-Z0-9_.]+", (query or "").lower()))
        scored = []
        for t in self.docs:
            toks = set(re.findall(r"[a-zA-Z0-9_.]+", t.lower()))
            score = len(q_tokens & toks)
            scored.append((score, t))
        scored.sort(key=lambda z: -z[0])
        for s, t in scored[:top_k]:
            if s > 0:
                hits.append({"score": float(s), "text": t, "source": "keyword"})
        if not hits:
            hits = [{"score": 0.0, "text": t, "source": "fallback"} for t in self.docs[:3]]
        return hits


def ask_with_index(
    question: str,
    df: pd.DataFrame,
    index: Optional[InsightIndex] = None,
    predictions: Optional[list[dict]] = None,
    ranked: Optional[list[dict]] = None,
    inspect: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """Retrieve from this upload, answer offline, then Gemini if a key works."""
    frame = attach_rul_columns(df, predictions)
    extra = _brief_documents(ranked, predictions, inspect)
    idx = index or InsightIndex()
    if not idx.docs:
        idx.build(frame, extra_docs=extra)
    hits = idx.search(question, top_k=6)
    context = "\n".join(f"- {h['text']}" for h in hits)

    used_synthetic = bool(predictions) and any(p.get("is_proxy") for p in predictions)
    if used_synthetic is False and predictions:
        used_synthetic = not _has_failure_labels(frame, predictions)

    offline = _offline_answer(question, hits, frame, predictions, ranked=ranked, used_synthetic=used_synthetic)
    gemini_ans = ""
    gemini_error = None
    key = get_gemini_api_key()
    gemini_attempted = bool(key)
    key_missing = not bool(key)
    if key:
        try:
            import google.generativeai as genai

            genai.configure(api_key=key)
            model_name = get_gemini_model()
            model = genai.GenerativeModel(model_name)
            prompt = (
                "You are a plant reliability analyst. Use ONLY the uploaded-table context. "
                "Do not write a general essay. Cite machine_id and numbers from the context. "
                f"If remaining-life is a proxy, say so. {PROXY_DISCLAIMER}\n"
                f"Question: {question}\nContext:\n{context}\n"
                "Short actionable answer for this upload only."
            )
            resp = model.generate_content(prompt)
            gemini_ans = (getattr(resp, "text", None) or "").strip()
            if not gemini_ans:
                gemini_error = "Gemini returned an empty response. Offline answer is shown."
        except Exception as exc:
            gemini_error = f"[Gemini error] {exc}"
            gemini_ans = ""
    else:
        gemini_error = None

    if gemini_error is None and gemini_attempted:
        gemini_error = gemini_issue_from_raw(gemini_ans, attempted=True)

    return {
        "answer": gemini_ans or offline,
        "offline_answer": offline,
        "gemini_error": gemini_error,
        "gemini_attempted": gemini_attempted,
        "key_missing": key_missing,
        "hits": hits,
        "mode": idx.meta.get("mode", "keyword"),
        "index_error": idx.meta.get("error") or "",
    }


def _offline_answer(
    question: str,
    hits: list[dict],
    df: pd.DataFrame,
    predictions: Optional[list[dict]],
    ranked: Optional[list[dict]] = None,
    used_synthetic: bool = False,
) -> str:
    q = (question or "").lower().strip()
    if df is None or (hasattr(df, "empty") and df.empty):
        return "No table is loaded. Upload a sensor CSV first — I only answer from this upload."

    asset_col = find_asset_col(df)
    ranked = ranked or rank_assets(df, predictions)
    caveat = f" {PROXY_DISCLAIMER}" if used_synthetic else ""

    mentioned = None
    if asset_col:
        for aid in df[asset_col].astype(str).unique():
            if aid.lower() in q:
                mentioned = aid
                break

    if mentioned:
        rec = next((r for r in ranked if str(r.get("machine_id")) == mentioned), None)
        pred = next((p for p in (predictions or []) if str(p.get("machine_id")) == mentioned), None)
        bits = [f"**{mentioned}** in this upload:"]
        if rec:
            if rec.get("why"):
                bits.append(" " + "; ".join(rec["why"]) + ".")
            bits.append(f" Rank #{rec.get('rank')} ({rec.get('risk_level')} risk).")
        if pred:
            bits.append(" " + pred.get("message", ""))
        bits.append(caveat)
        return "".join(bits).strip()

    if ranked and any(w in q for w in ("fail", "inspect", "maintenance", "risk", "rul", "which", "worst")):
        top = ranked[0]
        why = "; ".join((top.get("why") or [])[:3])
        return (
            f"**{top['machine_id']}** is the priority asset on this upload "
            f"(rank 1, {top.get('risk_level')} risk). {why}.{caveat}"
        )

    if "slow" in q or "throughput" in q or "rpm" in q:
        slow = flag_slow_running(df)
        if slow:
            return slow[0]["message"]
        return "No below-median speed/throughput column (or all assets are near the median) in this upload."

    if "anomal" in q or "outlier" in q or "isolation" in q:
        if _FLAG_COL in df.columns:
            n = int(df[_FLAG_COL].astype(bool).sum())
            rate = 100.0 * n / max(len(df), 1)
            return (
                f"Isolation Forest flagged **{n}** of {len(df)} rows ({rate:.2f}%) on this upload. "
                "Those are outlier readings, not confirmed failures."
            )

    evidence = [h for h in hits if h.get("source") != "llama_error"]
    if evidence:
        lines = ["From this upload's ranked assets / sensor rows:"]
        for h in evidence[:3]:
            lines.append(f"- {str(h.get('text', ''))[:280]}")
        lines.append(caveat.strip())
        return "\n".join(x for x in lines if x)

    return (
        "No matching asset or sensor rows in **this upload** for that question. "
        "Ask about a machine id, inspect list, Isolation Forest score, or RUL column."
    )
