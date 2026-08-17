#!/usr/bin/env python3
"""Cheap PdM smoke: sample CSV → map → clean → Isolation Forest → RUL → insights → charts."""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

errors: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"PASS  {name}")
    except Exception as exc:
        errors.append(f"{name}: {exc}")
        print(f"FAIL  {name}: {exc}")
        print("\n".join(traceback.format_exc().splitlines()[-4:]))


def main() -> int:
    import pandas as pd

    import config
    from src.data_cleaner import clean_and_quality
    from src.graphs.anomaly_flags import create_anomaly_flags_chart
    from src.graphs.layout import style_bar_figure
    from src.graphs.risk_by_asset import create_risk_by_asset_chart
    from src.graphs.time_series import create_time_series_chart
    from src.insights_engine import (
        DEFAULT_GEMINI_MODEL,
        NO_KEY_MESSAGE,
        ask_with_index,
        estimate_dollar_impact,
        flag_slow_running,
        generate_business_insights,
        get_gemini_model,
        gemini_issue_from_raw,
        inspect_this_week,
        rank_assets,
    )
    from src.ml.anomaly_detector import AnomalyDetector
    from src.ml.rul_predictor import RULPredictor
    from src.sensor_map import apply_mapping, suggest_mapping

    sample = pd.read_csv(config.SAMPLE_DATA_PATH, parse_dates=["timestamp"])

    def test_python39_annotations() -> None:
        from src.insights_engine import ask_with_index  # noqa: F401
        from src.sensor_map import mapping_status  # noqa: F401

        assert sys.version_info >= (3, 9)

    def test_gemini_remap() -> None:
        assert DEFAULT_GEMINI_MODEL == "gemini-3.6-flash"
        import google.generativeai as genai  # noqa: F401 — Ask path must be importable
        prev = os.environ.get("GEMINI_MODEL")
        try:
            for alias in ("gemini-flash-latest", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"):
                os.environ["GEMINI_MODEL"] = alias
                assert get_gemini_model() == "gemini-3.6-flash"
            os.environ["GEMINI_MODEL"] = "gemini-3.6-flash"
            assert get_gemini_model() == "gemini-3.6-flash"
        finally:
            if prev is None:
                os.environ.pop("GEMINI_MODEL", None)
            else:
                os.environ["GEMINI_MODEL"] = prev
        assert gemini_issue_from_raw("", attempted=False) is None
        assert gemini_issue_from_raw("", attempted=True)
        assert str(gemini_issue_from_raw("[Gemini error] 404", attempted=True)).startswith("[Gemini error]")

    def test_map_clean() -> None:
        messy = sample.rename(
            columns={
                "machine_id": "asset_id",
                "temperature": "temp_c",
                "vibration": "vib",
                "failure_within_days": "rul_days",
            }
        )
        mapping = suggest_mapping(list(messy.columns))
        assert mapping["machine_id"] == "asset_id"
        assert mapping["temperature"] == "temp_c"
        mapped = apply_mapping(messy, mapping)
        assert "machine_id" in mapped.columns and "temperature" in mapped.columns
        cleaned, summary, report = clean_and_quality(mapped, run_quality=True)
        assert len(cleaned) > 100
        assert summary["rows_after"] > 0
        assert report and report.get("checks")

    def test_anomaly_rul_insights() -> None:
        cleaned, _, _ = clean_and_quality(sample, run_quality=False)
        det = AnomalyDetector(contamination=0.05)
        det.fit(cleaned)
        summary = det.summary(cleaned)
        assert summary["anomaly_count"] >= 0
        assert "temperature" in det.feature_columns or det.feature_columns

        labeled = RULPredictor()
        labeled.fit(cleaned)
        assert labeled.used_synthetic_labels is False
        preds = labeled.predict_latest_per_machine(cleaned)
        assert preds and preds[0]["predicted_rul_days"] >= 1

        no_label = cleaned.drop(columns=["failure_within_days"])
        proxy = RULPredictor()
        proxy.fit(no_label)
        assert proxy.used_synthetic_labels is True

        insights = generate_business_insights(cleaned, preds, summary)
        assert insights and insights[0]["title"]

    def test_industrial_brief_tiny_frame() -> None:
        df = pd.DataFrame(
            {
                "machine_id": ["LineA"] * 4 + ["LineB"] * 4 + ["LineC"] * 4,
                "vibration": [1.0, 1.1, 1.0, 1.2, 8.0, 8.5, 9.0, 8.2, 2.0, 2.1, 1.9, 2.0],
                "rpm": [1800, 1810, 1790, 1805, 1100, 1080, 1090, 1070, 1750, 1760, 1740, 1755],
                "temperature": [70, 71, 70, 72, 95, 98, 97, 96, 73, 74, 72, 73],
                "throughput": [100, 102, 99, 101, 60, 58, 59, 61, 95, 96, 94, 95],
            }
        )
        det = AnomalyDetector(contamination=0.2)
        det.fit(df)
        scored = det.annotate(df)
        assert "anomaly_score" in scored.columns and "is_anomaly" in scored.columns

        preds = [
            {
                "machine_id": "LineB",
                "predicted_rul_days": 3,
                "risk_level": "High",
                "message": "LineB ranks High on the remaining-life proxy (~3 days).",
                "is_proxy": True,
                "label_source": "synthetic_degradation_proxy",
            },
            {
                "machine_id": "LineC",
                "predicted_rul_days": 12,
                "risk_level": "Medium",
                "message": "LineC remaining-life proxy ~12 days.",
                "is_proxy": True,
                "label_source": "synthetic_degradation_proxy",
            },
            {
                "machine_id": "LineA",
                "predicted_rul_days": 25,
                "risk_level": "Low",
                "message": "LineA remaining-life proxy ~25 days.",
                "is_proxy": True,
                "label_source": "synthetic_degradation_proxy",
            },
        ]
        ranked = rank_assets(scored, preds)
        assert ranked[0]["machine_id"] == "LineB"
        inspect = inspect_this_week(ranked)
        inspect_blob = " ".join(i["message"] for i in inspect)
        assert "LineB" in inspect_blob
        assert "this week" in inspect_blob.lower()

        slow = flag_slow_running(scored)
        assert any("LineB" in s["message"] and "slow" in s["message"].lower() for s in slow)

        dollars = estimate_dollar_impact(ranked, slow, cost_per_hour=1500, hours_if_stop=8)
        dollar_blob = " ".join(d["message"] for d in dollars)
        assert "LineB" in dollar_blob and "$12,000" in dollar_blob

        insights = generate_business_insights(
            scored,
            preds,
            det.summary(scored),
            cost_per_hour=1500,
            cost_per_unit=25,
            hours_if_stop=8,
            used_synthetic=True,
        )
        blob = " ".join(i["message"] for i in insights).lower()
        assert "degradation proxy" in blob or "not a confirmed failure" in blob
        assert "will fail on" not in blob
        assert "lorem" not in blob

        prev_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            result = ask_with_index("Which asset should we inspect this week?", scored, predictions=preds, ranked=ranked, inspect=inspect)
        finally:
            if prev_key is not None:
                os.environ["GEMINI_API_KEY"] = prev_key
        assert result["key_missing"] is True
        assert result["gemini_attempted"] is False
        offline = (result.get("offline_answer") or "").lower()
        assert "lineb" in offline
        assert "lorem" not in offline
        assert NO_KEY_MESSAGE.split("`")[0].strip()  # constant still defined

        named = ask_with_index("Tell me about LineB", scored, predictions=preds, ranked=ranked)
        assert "LineB" in named["offline_answer"]

    def test_charts() -> None:
        cleaned, _, _ = clean_and_quality(sample.head(400), run_quality=False)
        det = AnomalyDetector()
        det.fit(cleaned)
        line = create_time_series_chart(cleaned, x_axis="timestamp", y_metric="vibration")
        assert line.layout.xaxis.automargin is True
        flags = create_anomaly_flags_chart(
            cleaned, x_axis="timestamp", y_metric="vibration", anomaly_detector=det
        )
        assert flags.layout.margin.b is None or int(flags.layout.margin.b or 0) >= 48
        rul = RULPredictor()
        rul.fit(cleaned)
        preds = rul.predict_latest_per_machine(cleaned)
        risk = create_risk_by_asset_chart(cleaned, predictions=preds)
        assert int(risk.layout.margin.l or 0) >= 96
        import plotly.express as px

        bar = px.bar(pd.DataFrame({"asset": ["Machine 1"], "days": [7]}), x="asset", y="days")
        styled = style_bar_figure(bar, n_cats=1, horizontal=False, title="t")
        assert float(styled.layout.xaxis.tickangle) == -40
        assert int(styled.layout.margin.b or 0) >= 120

    check("python39_imports", test_python39_annotations)
    check("gemini_remap", test_gemini_remap)
    check("map_clean", test_map_clean)
    check("anomaly_rul_insights", test_anomaly_rul_insights)
    check("industrial_brief_tiny_frame", test_industrial_brief_tiny_frame)
    check("charts_layout", test_charts)

    if errors:
        print(f"\n{len(errors)} FAIL")
        return 1
    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
