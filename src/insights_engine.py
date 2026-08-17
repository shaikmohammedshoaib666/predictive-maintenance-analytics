"""Business insights + LlamaIndex-backed retrieval for PdM decisions."""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.quality_checks import find_col


def _gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return key
    try:
        import streamlit as st

        return str(st.secrets.get("GEMINI_API_KEY", "") or "")
    except Exception:
        return ""


def chart_business_insight(df: pd.DataFrame, x: str, y: str) -> str:
    """Plain-language insight (Forge-style): trends like revenue/metric will drop."""
    if x not in df.columns or y not in df.columns:
        return "Select valid X/Y columns for insights."
    work = df[[x, y]].copy()
    ynum = pd.to_numeric(work[y], errors="coerce")
    if ynum.notna().sum() < 3:
        return f"Not enough numeric values in `{y}` for prediction."

    if not pd.api.types.is_numeric_dtype(work[x]) or work[x].nunique() < max(3, len(work) // 10):
        g = work.assign(_y=ynum).groupby(x, dropna=False)["_y"].mean().sort_values()
        if len(g) >= 2:
            low, high = g.index[0], g.index[-1]
            msg = (
                f"**{y}** is lowest for **{low}** (avg={g.iloc[0]:.2f}) and highest for **{high}** "
                f"(avg={g.iloc[-1]:.2f}). Gap={g.iloc[-1] - g.iloc[0]:.2f}."
            )
            try:
                s = ynum.dropna()
                slope = float(np.polyfit(np.arange(len(s)), s.to_numpy(), 1)[0])
                future = float(s.iloc[-1] + slope * max(5, len(s) // 10))
                pct = (future - float(s.iloc[-1])) / abs(float(s.iloc[-1]) or 1) * 100
                direction = "increase" if pct >= 0 else "drop"
                msg += (
                    f" Overall `{y}` is likely to **{direction} ~{abs(pct):.1f}%** "
                    "in the near future (trend model)."
                )
            except Exception:
                pass
            return msg

    tmp = pd.DataFrame({"x": pd.to_numeric(work[x], errors="coerce"), "y": ynum}).dropna()
    if len(tmp) < 5:
        return "Need more points for numeric insight."
    corr = float(tmp["x"].corr(tmp["y"]))
    slope = float(np.polyfit(tmp["x"], tmp["y"], 1)[0])
    direction = "rise" if slope > 0 else "fall"
    return (
        f"`{y}` vs `{x}`: correlation={corr:.2f}. As `{x}` grows, `{y}` tends to **{direction}** "
        f"(slope={slope:.4f})."
    )


def generate_business_insights(
    df: pd.DataFrame,
    predictions: Optional[list[dict]] = None,
    anomaly_summary: Optional[dict] = None,
) -> list[dict[str, str]]:
    """Actionable PdM / business insights for managers."""
    insights: list[dict[str, str]] = []
    predictions = predictions or []
    anomaly_summary = anomaly_summary or {}

    if predictions:
        high = [p for p in predictions if p.get("risk_level") == "High"]
        medium = [p for p in predictions if p.get("risk_level") == "Medium"]
        worst = predictions[0]
        insights.append(
            {
                "title": "Maintenance priority",
                "severity": "High" if high else ("Medium" if medium else "Low"),
                "message": (
                    f"**{worst['machine_id']}** needs attention first — "
                    f"predicted failure in **{worst['predicted_rul_days']} days** "
                    f"({worst['risk_level']} risk)."
                ),
            }
        )
        if high:
            names = ", ".join(p["machine_id"] for p in high)
            insights.append(
                {
                    "title": "Schedule maintenance window",
                    "severity": "High",
                    "message": (
                        f"Machines at high risk: **{names}**. "
                        "Plan downtime this week to avoid unplanned outages and revenue loss."
                    ),
                }
            )

    mid = find_col(df, "machine_id", "machine", "asset_id")
    vib = find_col(df, "vibration", "vib")
    temp = find_col(df, "temperature", "temp")
    cost = find_col(df, "cost", "maintenance_cost", "downtime_cost", "revenue")

    if mid and vib:
        g = df.assign(_v=pd.to_numeric(df[vib], errors="coerce")).groupby(mid)["_v"].mean().sort_values(ascending=False)
        if len(g):
            insights.append(
                {
                    "title": "Highest vibration asset",
                    "severity": "Medium",
                    "message": (
                        f"**{g.index[0]}** has the highest average `{vib}` ({g.iloc[0]:.3f}). "
                        "Elevated vibration usually precedes bearing/imbalance failures."
                    ),
                }
            )

    if temp is not None:
        s = pd.to_numeric(df[temp], errors="coerce").dropna()
        if len(s) >= 10:
            slope = float(np.polyfit(np.arange(len(s)), s.to_numpy(), 1)[0])
            if slope > 0:
                insights.append(
                    {
                        "title": "Temperature trend rising",
                        "severity": "Medium",
                        "message": (
                            f"`{temp}` is **rising** (slope={slope:.4f}/step). "
                            "Cooling / lubrication checks recommended before thermal trips."
                        ),
                    }
                )

    if cost is not None:
        msg = chart_business_insight(
            df,
            mid or df.columns[0],
            cost,
        )
        insights.append({"title": f"Business metric: {cost}", "severity": "Info", "message": msg})

    if anomaly_summary.get("anomaly_rate_pct", 0) >= 8:
        insights.append(
            {
                "title": "Anomaly rate elevated",
                "severity": "High",
                "message": (
                    f"Anomaly rate is **{anomaly_summary['anomaly_rate_pct']}%** "
                    f"({anomaly_summary.get('anomaly_count', 0)} of "
                    f"{anomaly_summary.get('total_records', 0)}). Investigate root cause before next shift."
                ),
            }
        )

    if not insights:
        insights.append(
            {
                "title": "Ready for analysis",
                "severity": "Info",
                "message": "Load data, run Clean + Train Models to unlock maintenance and revenue insights.",
            }
        )
    return insights


def _row_documents(df: pd.DataFrame, max_rows: int = 400) -> list[str]:
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


class InsightIndex:
    """LlamaIndex vector index with keyword fallback (Forge pattern, no Streamlit coupling)."""

    def __init__(self) -> None:
        self.docs: list[str] = []
        self.index_obj = None
        self.meta: dict[str, Any] = {}

    def build(self, df: pd.DataFrame) -> dict[str, Any]:
        self.docs = _row_documents(df)
        mode = "keyword"
        self.index_obj = None
        try:
            from llama_index.core import Document, VectorStoreIndex

            documents = [Document(text=t) for t in self.docs]
            try:
                self.index_obj = VectorStoreIndex.from_documents(documents)
                mode = "llama_vector"
            except Exception:
                mode = "keyword"
        except Exception:
            mode = "keyword"
        self.meta = {
            "ok": True,
            "n_docs": len(self.docs),
            "mode": mode,
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
            except Exception:
                pass

        q_tokens = set(re.findall(r"[a-zA-Z0-9_.]+", (query or "").lower()))
        scored = []
        for t in self.docs:
            toks = set(re.findall(r"[a-zA-Z0-9_.]+", t.lower()))
            score = len(q_tokens & toks)
            if any(w in (query or "").lower() for w in ("fail", "vibration", "temp", "why", "spike", "revenue")):
                if "vibration" in t.lower() or "fail" in t.lower():
                    score += 2
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
) -> dict[str, Any]:
    """Retrieve context then answer offline (and optionally via Gemini)."""
    idx = index or InsightIndex()
    if not idx.docs:
        idx.build(df)
    hits = idx.search(question, top_k=6)
    context = "\n".join(f"- {h['text']}" for h in hits)

    offline = _offline_answer(question, hits, df, predictions)
    gemini_ans = ""
    key = _gemini_key()
    if key:
        try:
            import google.generativeai as genai

            genai.configure(api_key=key)
            model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
            model = genai.GenerativeModel(model_name)
            prompt = (
                "You are a predictive maintenance analyst. Use ONLY the context.\n"
                f"Question: {question}\nContext:\n{context}\n"
                "Give a short actionable answer for plant managers."
            )
            gemini_ans = model.generate_content(prompt).text or ""
        except Exception as exc:
            gemini_ans = f"(Gemini unavailable: {exc})"

    return {
        "answer": gemini_ans or offline,
        "offline_answer": offline,
        "hits": hits,
        "mode": idx.meta.get("mode", "keyword"),
    }


def _offline_answer(
    question: str,
    hits: list[dict],
    df: pd.DataFrame,
    predictions: Optional[list[dict]],
) -> str:
    q = (question or "").lower()
    if predictions and any(w in q for w in ("fail", "maintenance", "risk", "rul")):
        p = predictions[0]
        return (
            f"{p['machine_id']} is the priority asset — predicted failure in "
            f"{p['predicted_rul_days']} days ({p['risk_level']} risk). "
            "Schedule inspection and spare parts now."
        )
    if "revenue" in q or "cost" in q or "drop" in q:
        cost = find_col(df, "cost", "revenue", "maintenance_cost", "downtime_cost")
        mid = find_col(df, "machine_id", "machine") or (df.columns[0] if len(df.columns) else "x")
        if cost:
            return chart_business_insight(df, mid, cost)
    if hits:
        return "Based on retrieved sensor rows:\n" + "\n".join(f"- {h['text'][:200]}" for h in hits[:3])
    return "Upload and clean data, then train models so I can answer with evidence."
