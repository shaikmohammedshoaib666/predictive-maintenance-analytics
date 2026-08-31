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
    from src.url_ingest import (
        build_preset_sql,
        default_ingest_sql,
        detect_source_kind,
        list_ingest_presets,
        resolve_source_to_fetch_url,
        validate_ingest_sql,
    )

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

    def test_url_ingest_presets() -> None:
        # DuckDB URL/cloud ingest + SQL slice presets ported from analytics-forge-v2.
        presets = list_ingest_presets(domain="predictive_maintenance")
        ids = {p["id"] for p in presets}
        assert {"filter_machine_id", "pdm_failure_focus", "last_n_rows"} <= ids
        sql = build_preset_sql("filter_machine_id", {"machine_id": "M-001", "n": 100})
        assert "{source}" in sql and "M-001" in sql and "LIMIT 100" in sql
        assert "LIMIT" not in build_preset_sql("filter_machine_id", {"machine_id": "M-1", "n": 0}).upper()
        # String params are SQL-escaped (single quote doubled).
        assert "''" in build_preset_sql("filter_machine_id", {"machine_id": "x' OR '1'='1", "n": 5})
        assert "{source}" in default_ingest_sql("predictive_maintenance")
        assert detect_source_kind("https://drive.google.com/file/d/ABC/view") == "google_drive"
        assert detect_source_kind("kaggle://owner/ds/f.csv") == "kaggle_api"
        assert detect_source_kind("https://example.com/a.csv") == "https"
        validate_ingest_sql("SELECT * FROM read_csv_auto('{source}')")
        for bad in ("DROP TABLE x", "SELECT 1; DELETE FROM y"):
            try:
                validate_ingest_sql(bad)
                raise AssertionError(f"validate_ingest_sql should reject: {bad}")
            except ValueError:
                pass
        target, meta = resolve_source_to_fetch_url("https://www.dropbox.com/s/x/a.csv?dl=0")
        assert target.endswith("dl=1") and meta["kind"] == "https"

    def test_twin3d() -> None:
        # Layer 3 — 3D digital twin HTML builder + risk mapping.
        from src.twin3d import asset_states_from_predictions, build_twin_html, normalize_risk

        assert normalize_risk("HIGH") == "High"
        assert normalize_risk("warn") == "Medium"
        assert normalize_risk("healthy") == "Low"
        assert normalize_risk(None) == "Unknown"
        states = asset_states_from_predictions(
            [{"machine_id": "M-1", "risk_level": "High", "predicted_rul_days": 3}]
        )
        assert states[0]["risk_level"] == "High"
        html = build_twin_html(states, selected_id="M-1", height=400)
        # Offline twin: three.js is inlined, not loaded from a CDN / importmap.
        assert "OrbitControls" in html and "THREE" in html
        assert 'type="importmap"' not in html and "jsdelivr" not in html
        import re as _re

        assert not _re.search(r'src\s*=\s*["\']https?://', html)  # no external script loads
        assert "M-1" in html and "__DATA__" not in html  # tokens fully substituted
        # empty asset list still renders a demo twin
        assert "demo-asset" in build_twin_html([], selected_id=None)

    def test_live_connect() -> None:
        # Layer 4 — live simulator + rolling buffer + IsolationForest status.
        from src.live_connect import (
            append_to_buffer,
            compute_live_status,
            default_machines,
            simulate_batch,
        )

        machines = default_machines(3)
        assert machines == ["M-001", "M-002", "M-003"]
        buf = None
        for tick in range(40):
            stress = min(1.0, tick / 20.0)
            batch = simulate_batch(machines, tick, failing="M-001", stress=stress)
            assert list(batch["machine_id"]) == machines
            buf = append_to_buffer(buf, batch, max_rows=200)
        assert len(buf) == 120  # 40 ticks × 3 machines, under cap
        status = compute_live_status(buf)
        assert status and {s["machine_id"] for s in status} == set(machines)
        # The stressed machine should carry the highest anomaly rate → sorts first.
        assert status[0]["machine_id"] == "M-001"
        assert status[0]["risk_level"] in {"High", "Medium"}

    def test_live_sources() -> None:
        # Upgrade 2 — MQTT / OPC-UA helpers (gates + payload parsing, offline).
        from src.live_sources import _coerce_row, mqtt_available, opcua_available, parse_node_map

        ok_m, _ = mqtt_available()
        ok_o, _ = opcua_available()
        assert isinstance(ok_m, bool) and isinstance(ok_o, bool)
        nm = parse_node_map("temperature=ns=2;i=2, vibration=ns=2;i=3")
        assert nm == {"temperature": "ns=2;i=2", "vibration": "ns=2;i=3"}
        row = _coerce_row(
            {"machine_id": "M-9", "temperature": "70.5", "vibration": 2.1},
            machine_field="machine_id",
            ts_field="timestamp",
            fallback_machine="x",
        )
        assert row["machine_id"] == "M-9" and row["temperature"] == 70.5 and "timestamp" in row
        row2 = _coerce_row({"temperature": 10}, machine_field="machine_id", ts_field="timestamp", fallback_machine="fb")
        assert row2["machine_id"] == "fb"

    def test_spark_engine_gate() -> None:
        # Layer 5 — availability gate must never raise (graceful even without a JVM).
        from src.spark_clean import spark_available

        ok, msg = spark_available()
        assert isinstance(ok, bool) and isinstance(msg, str) and msg

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
    check("url_ingest_presets", test_url_ingest_presets)
    check("twin3d", test_twin3d)
    check("live_connect", test_live_connect)
    check("live_sources", test_live_sources)
    check("spark_engine_gate", test_spark_engine_gate)
    check("charts_layout", test_charts)

    if errors:
        print(f"\n{len(errors)} FAIL")
        return 1
    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
