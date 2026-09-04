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
        from src.twin3d import (
            TWIN_KINDS,
            asset_states_from_predictions,
            build_twin_html,
            normalize_risk,
            normalize_twin_kind,
        )

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
        assert normalize_twin_kind("aviation_uav_piston") == "aviation_uav"
        assert normalize_twin_kind("bogus") == "plant_motor"
        for kind in TWIN_KINDS:
            kind_html = build_twin_html(states, selected_id="M-1", kind=kind, hotspot="test-hot")
            assert f'"kind": "{kind}"' in kind_html or f'"kind":"{kind}"' in kind_html
            assert "test-hot" in kind_html
            assert f"scene:{kind}" in kind_html
            assert "jsdelivr" not in kind_html
            assert "__DATA__" not in kind_html

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

    def test_polars_engine() -> None:
        from src.polars_clean import clean_with_polars, polars_available

        ok, msg = polars_available()
        assert isinstance(ok, bool) and isinstance(msg, str) and msg
        assert ok is True  # polars is a core requirement
        cleaned, log = clean_with_polars(sample.head(200))
        assert len(cleaned) > 0
        assert any("Polars" in line or "polars" in line.lower() for line in log)
        assert "temperature" in cleaned.columns

    def test_aps_viewer() -> None:
        # Upgrade 3 — APS gate never raises; viewer HTML builder is pure (no network).
        import base64
        import re
        from datetime import datetime, timezone

        from src.aps_viewer import (
            A360_PUBLIC_URN_ERROR,
            DEFAULT_VIEWER_HEIGHT,
            INSUFFICIENT_SCOPE_HINT,
            MAX_CAD_UPLOAD_MB,
            CAD_SIZE_ZIP_FALLBACK,
            PUBLIC_VIEWER_NO_URN_WARNING,
            PUBLIC_VIEWER_WARNING,
            TRANSLATE_SCOPES,
            VIEWER_ERROR_CODES,
            VIEWER_EXTRA_EXTENSIONS,
            aps_available,
            aps_model_urn,
            build_viewer_html,
            cad_size_issue,
            decode_model_urn,
            describe_viewer_error,
            encode_model_urn,
            encode_oss_urn,
            extract_model_urn,
            format_aps_error,
            looks_like_insufficient_scope,
            looks_like_public_viewer,
            normalize_model_urn,
            oss_bucket_key,
            oss_object_id,
            resolve_cad_urn,
            save_urn_for_pack,
            saved_urn_caption,
            saved_urn_for_pack,
            delete_urn_for_pack,
            sanitize_object_name,
            strip_urn_query_fragment,
        )

        ok, msg = aps_available()
        assert isinstance(ok, bool) and isinstance(msg, str)
        assert isinstance(aps_model_urn(), str)
        html = build_viewer_html("TOKEN123", "dXJuOmFkc2sx", asset="M-1", risk="High", height=500)
        assert "TOKEN123" in html and "dXJuOmFkc2sx" in html and "GuiViewer3D" in html
        assert "__TOKEN__" not in html and "__URN__" not in html
        assert "Unknown" in build_viewer_html("t", "u", asset="a", risk="bogus")
        assert "new Autodesk.Viewing.Viewer3D(" not in html
        assert "disabledExtensions" in html
        assert "Autodesk.DocumentBrowser" in html
        assert "Autodesk.Viewing.MarkupsGui" in html
        assert "Autodesk.Measure" in html or "measure: false" in html
        assert "Autodesk.BimWalk" in html or "bimwalk: false" in html
        assert "showViewCube" in html
        assert str(DEFAULT_VIEWER_HEIGHT) in build_viewer_html("t", "dXJuOmFi", asset="a", risk="Low")
        assert "Autodesk.Viewing.MarkupsGui" in VIEWER_EXTRA_EXTENSIONS
        pub = build_viewer_html("t", "dXJuOmFi", asset="a", risk="Low", public_viewer=True)
        assert "YOUR OSS bucket" in pub or "viewer.autodesk.com" in pub or "Rotax 912" in pub
        assert "cannot open it" in pub
        html_err = build_viewer_html("t", "dXJuOmFi", asset="a", risk="Low")
        assert "NETWORK_ACCESS_DENIED" in html_err
        assert "describeErr" in html_err
        assert VIEWER_ERROR_CODES[4].split("—")[0].strip() in html_err or "Access denied" in html_err
        assert "NETWORK_ACCESS_DENIED" in describe_viewer_error(4)
        assert "error 4" in describe_viewer_error(4)
        assert "error 4" in describe_viewer_error({"code": 4})
        assert "Bad data" in describe_viewer_error(2)

        assert "SELECTION_CHANGED_EVENT" in html
        assert "AGGREGATE_SELECTION_CHANGED_EVENT" in html
        assert "getNodeName" in html
        assert "setThemingColor" in html
        assert "clearThemingColors" in html
        assert "streamlit:setComponentValue" in html
        assert "setInterval(" not in html
        assert "explode: false" in html
        low_html = build_viewer_html("t", "dXJuOmFi", asset="a", risk="Low")
        assert "RISK_RGBA" in low_html
        from src.cad_part import REGION_THEME_RGBA as _RGBA

        assert str(_RGBA["High"][0]) in html

        # Region tint, not whole-model theming: no enumNodeChildren-paint-everything.
        assert "applyRegionTheming" in html
        assert "applyAssetTheming" not in html
        assert "needleFor" in html and "walkRegion" in html and "paintRegion" in html
        assert "sanitizeName" in html
        assert "Low" not in _RGBA and "Unknown" not in _RGBA
        # Region reports go out on the component channel, which reruns the whole
        # Streamlit script. A timestamped payload on every re-render would loop.
        report_js = html.split("function reportRegion(", 1)[1].split("function walkRegion(", 1)[0]
        assert "ts:" not in report_js, report_js
        assert "lastRegionKey" in report_js
        assert "lastRegionKey = null" in html, "reset the dedupe key when the viewer is rebuilt"
        assert "ts: Date.now()" in html, "part clicks still need a timestamp"

        needled = build_viewer_html(
            "t", "dXJuOmFi", asset="a", risk="High", needles=["exhaust", "oilpress"]
        )
        assert '"exhaust"' in needled and '"oilpress"' in needled
        assert "var NEEDLES=" in needled and "__NEEDLES__" not in needled
        # Default needles come from cad_part so the two stay in sync.
        from src.cad_part import region_hint_keywords as _needles

        assert f'"{_needles()[0]}"' in html

        bridge = build_viewer_html("t", "dXJuOmFi", asset="a", risk="Low", bridge=True)
        assert "streamlit:componentReady" in bridge
        assert "var BRIDGE=true" in bridge
        from src.aps_viewer import CAD_VIEWER_COMPONENT_DIR, viewer_bridge_html

        index = CAD_VIEWER_COMPONENT_DIR / "index.html"
        assert index.is_file()
        index_html = index.read_text(encoding="utf-8")
        assert index_html == viewer_bridge_html()
        assert "GuiViewer3D" in index_html and "streamlit:componentReady" in index_html

        assert oss_bucket_key("AbC-123_XYZ") == "pdm-abc123xyz-cad"
        assert oss_bucket_key("") == "pdm-app-cad"
        long_key = oss_bucket_key("Z" * 200)
        assert long_key.startswith("pdm-") and long_key.endswith("-cad")
        assert long_key == long_key.lower() and 3 <= len(long_key) <= 128
        assert re.fullmatch(r"[a-z0-9._-]+", long_key)

        when = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
        assert sanitize_object_name("../../My Model.RVT", when) == "20260902120000_My_Model.RVT"
        assert sanitize_object_name("/etc/passwd", when) == "20260902120000_passwd"
        assert sanitize_object_name("", when).endswith("_model")

        oid = oss_object_id("pdm-abc-cad", "20260902_box.ipt")
        urn = encode_model_urn(oid)
        assert urn == encode_oss_urn("pdm-abc-cad", "20260902_box.ipt")
        assert not urn.startswith("urn:")
        assert ":" not in urn
        pad = "=" * ((4 - len(urn) % 4) % 4)
        assert base64.urlsafe_b64decode(urn + pad).decode() == oid
        assert normalize_model_urn(" urn:dXJuOmFi ") == "dXJuOmFi"
        assert normalize_model_urn(oid) == urn
        html2 = build_viewer_html("t", "urn:" + urn, asset="a", risk="Low")
        assert f'URN="{urn}"' in html2

        extracted, meta = extract_model_urn(
            "https://viewer.autodesk.com/?urn=dXJuOmFkc2sub2JqZWN0czpvcy5vYmplY3Q6YnVja2V0L1JvdGF4LnN0ZXA"
        )
        assert extracted.startswith("dXJu")
        assert meta["public_viewer"] is True
        assert "YOUR" in (meta["warning"] or PUBLIC_VIEWER_WARNING)
        path_ex, path_meta = extract_model_urn(
            "https://viewer.autodesk.com/id/dXJuOmFkc2sub2JqZWN0czpvcy5vYmplY3Q6YnVja2V0L2VuZ2luZS5zdHA"
        )
        assert path_ex.startswith("dXJu") and path_meta["public_viewer"] is True
        obj_ex, _ = extract_model_urn("urn:adsk.objects:os.object:bucket/Rotax.step")
        assert obj_ex == encode_model_urn("urn:adsk.objects:os.object:bucket/Rotax.step")
        share, share_meta = extract_model_urn("https://viewer.autodesk.com/share/abc123")
        assert share == "" and share_meta["public_viewer"] is True
        assert PUBLIC_VIEWER_NO_URN_WARNING.split("share")[0][:20] in share_meta["warning"] or "share" in share_meta["warning"].lower()
        plain, plain_meta = extract_model_urn("  dXJuOmFkc2suZXhhbXBsZQ  ")
        assert plain.startswith("dXJu") and plain_meta["public_viewer"] is False
        assert looks_like_public_viewer("https://viewer.autodesk.com/?urn=dXJuOmFi")
        assert not looks_like_public_viewer("dXJuOmFi")

        a360_oid = (
            "urn:adsk.objects:os.object:a360viewer-protected/"
            "t1788359463_db62c835-54e9-4d0d-8ef8-863aacb123ce.step"
        )
        a360_b64 = (
            "dXJuOmFkc2sub2JqZWN0czpvcy5vYmplY3Q6YTM2MHZpZXdlci1wcm90ZWN0ZWQv"
            "dDE3ODgzNTk0NjNfZGI2MmM4MzUtNTRlOS00ZDBkLThlZjgtODYzYWFjYjEyM2NlLnN0ZXA"
        )
        sheet_url = (
            "https://viewer.autodesk.com/id/" + a360_b64
            + "?sheetId=Y2M1MmM3NmEtZDBkYS00ZjExLTk2NzEtM2JmNzBmMWQ3NzBi"
        )
        sheet_ex, sheet_meta = extract_model_urn(sheet_url)
        assert "?" not in sheet_ex and "sheetId" not in sheet_ex and "#" not in sheet_ex
        assert sheet_ex.startswith("dXJu")
        assert "a360viewer" in decode_model_urn(sheet_ex).lower()
        assert sheet_meta["public_viewer"] is True
        assert sheet_meta["a360_protected"] is True
        assert sheet_meta["block_load"] is True
        assert "cannot open it" in (sheet_meta.get("error") or A360_PUBLIC_URN_ERROR)
        pasted = a360_b64 + "?sheetId=Y2M1MmM3NmEtZDBkYS00ZjExLTk2NzEtM2JmNzBmMWQ3NzBi"
        pasted_ex, pasted_meta = extract_model_urn(pasted)
        assert pasted_ex == strip_urn_query_fragment(a360_b64) or pasted_ex.startswith("dXJu")
        assert "?" not in pasted_ex and "sheetId" not in pasted_ex
        assert "a360viewer" in decode_model_urn(pasted_ex).lower()
        assert pasted_meta["a360_protected"] is True
        assert looks_like_public_viewer(a360_b64)
        assert looks_like_public_viewer(a360_oid)
        assert "a360viewer" in decode_model_urn(a360_b64).lower()
        assert strip_urn_query_fragment(pasted) == a360_b64
        assert "Rotax 912" in A360_PUBLIC_URN_ERROR
        app_src = (ROOT / "app.py").read_text(encoding="utf-8")
        assert "A360_PUBLIC_URN_ERROR" in app_src
        assert 'urn_state.get("public_paste")' in app_src

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "cad_urns.json"
            av = "aviation_uav_piston"
            working = "dXJuOmFkc2sub2JqZWN0czpvcy5vYmplY3Q6cGRtLWFwcC1jYWQvcm90YXguc3RlcA"
            denied = save_urn_for_pack(
                av, working, source="paste", public_viewer=True, path=store
            )
            assert denied["saved"] is False and denied["reason"] == "public_viewer"
            assert not store.is_file()
            a360_denied = save_urn_for_pack(
                av, a360_b64, source="paste", public_viewer=False, path=store
            )
            assert a360_denied["saved"] is False and a360_denied["reason"] == "public_viewer"
            assert not store.is_file()
            saved = save_urn_for_pack(av, working, source="translate", path=store)
            assert saved["saved"] is True
            assert saved_urn_for_pack(av, path=store) == working
            assert "Aviation" in saved_urn_caption(av)
            assert "APS_MODEL_URN" in saved["message"]
            assert "Render env" not in saved["message"].lower()
            assert "we wrote render" not in saved["message"].lower()
            info = resolve_cad_urn(av, "", path=store)
            assert info["urn"] == working and info["source"] == "saved"
            sess = resolve_cad_urn(av, "dXJuOmZyb21zZXNzaW9u", path=store)
            assert sess["source"] == "session" and sess["urn"] == "dXJuOmZyb21zZXNzaW9u"
            prev_env = os.environ.get("APS_MODEL_URN")
            os.environ["APS_MODEL_URN"] = "dXJuOmVudm92ZXJyaWRl"
            try:
                over = resolve_cad_urn(av, "", path=store)
                assert over["source"] == "env" and over["urn"] == "dXJuOmVudm92ZXJyaWRl"
                assert over["env_override"] is True
            finally:
                if prev_env is None:
                    os.environ.pop("APS_MODEL_URN", None)
                else:
                    os.environ["APS_MODEL_URN"] = prev_env
            plant = save_urn_for_pack(
                "plant_rotating", "dXJuOmFkc2sucGxhbnR0ZXN0", source="save", path=store
            )
            assert plant["saved"] is True
            gone = delete_urn_for_pack(av, path=store)
            assert gone["ok"] is True and gone["deleted"] is True and gone["had_urn"] is True
            assert saved_urn_for_pack(av, path=store) == ""
            assert saved_urn_for_pack("plant_rotating", path=store) == "dXJuOmFkc2sucGxhbnR0ZXN0"
            assert "Aviation" in gone["message"] and "Deleted" in gone["message"]
            again = delete_urn_for_pack(av, path=store)
            assert again["ok"] is True and again["had_urn"] is False
            assert saved_urn_for_pack("plant_rotating", path=store) == "dXJuOmFkc2sucGxhbnR0ZXN0"
            empty = resolve_cad_urn(av, "", path=store)
            assert empty["urn"] == "" and empty["source"] == "none"

        example = ROOT / "data" / "cad_urns.example.json"
        assert example.is_file()
        import json as _json

        example_doc = _json.loads(example.read_text(encoding="utf-8"))
        assert example_doc.get("packs") == {}
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "cad_urns.json" in gitignore or "data/*" in gitignore

        assert cad_size_issue(0)[0] == "error"
        assert cad_size_issue(1024)[0] is None
        assert cad_size_issue(90 * 1024 * 1024)[0] == "warn"
        rotax_207 = cad_size_issue(207 * 1024 * 1024)
        assert rotax_207[0] == "warn", rotax_207
        assert "zip" in rotax_207[1].lower() and "fusion" in rotax_207[1].lower()
        assert cad_size_issue(201 * 1024 * 1024)[0] == "warn"
        over, over_msg = cad_size_issue((MAX_CAD_UPLOAD_MB + 1) * 1024 * 1024)
        assert over == "error"
        assert "zip" in over_msg.lower() and "fusion" in over_msg.lower()
        assert "maxUploadSize" in over_msg
        assert "zip the STEP" in CAD_SIZE_ZIP_FALLBACK

        cfg_text = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        size_m = re.search(r"(?m)^\s*maxUploadSize\s*=\s*(\d+)\s*$", cfg_text)
        assert size_m, "server.maxUploadSize missing from .streamlit/config.toml"
        max_upload_mb = int(size_m.group(1))
        assert max_upload_mb >= 250, max_upload_mb
        assert MAX_CAD_UPLOAD_MB <= max_upload_mb
        assert MAX_CAD_UPLOAD_MB >= 250

        assert looks_like_insufficient_scope(
            403,
            '{"errorCode":"AUTH-010","developerMessage":"The access token does not have the required privileges"}',
        )
        assert INSUFFICIENT_SCOPE_HINT in format_aps_error(401, "insufficient scope")
        assert "data:write" in TRANSLATE_SCOPES and "bucket:create" in TRANSLATE_SCOPES
        assert "viewables:read" in TRANSLATE_SCOPES

    def test_cad_part_click() -> None:
        """Asset risk → whole-model tint; Autodesk part name → sensor hint. Not per-bolt IF."""
        from src.cad_part import (
            RISK_THEME_HEX,
            build_part_card,
            latest_asset_sensors,
            parse_cad_part_event,
            part_sensor_hint,
            risk_theme_hex,
            risk_theming_rgba,
        )

        assert risk_theme_hex("High") == "#e74c3c" == RISK_THEME_HEX["High"]
        assert risk_theme_hex("medium") == "#f39c12"
        assert risk_theme_hex("Low") == "#7a8c82"
        assert risk_theme_hex("Low") != "#e74c3c"
        assert risk_theme_hex("bogus") == RISK_THEME_HEX["Unknown"]
        assert risk_theming_rgba("High") == (0.91, 0.30, 0.24, 0.72)
        assert risk_theming_rgba("Medium") is not None
        assert risk_theming_rgba("Low") is not None
        assert risk_theming_rgba("Unknown") is None
        assert risk_theming_rgba("") is None

        assert part_sensor_hint("Cylinder Head") == "cht"
        assert part_sensor_hint("cyl-4") == "cht"
        assert part_sensor_hint("Oil Pump Housing") == "oil_pressure"
        assert part_sensor_hint("Oil Temperature Sender") == "oil_temp"
        assert part_sensor_hint("Crankshaft") == "rpm"
        assert part_sensor_hint("Exhaust manifold") == "egt"
        assert part_sensor_hint("Intake manifold") == "manifold_pressure"
        assert part_sensor_hint("Manifold") == "manifold_pressure"
        assert part_sensor_hint("EGT probe") == "egt"
        assert part_sensor_hint("Temperature sensor") == "temperature"
        assert part_sensor_hint("Vibration pickup") == "vibration"
        assert part_sensor_hint("RPM gear") == "rpm"
        assert part_sensor_hint("Bolt_14") is None
        assert part_sensor_hint("Housing Cover") is None
        assert part_sensor_hint("") is None
        assert (
            part_sensor_hint(
                "Cylinder Head",
                pack_id="aviation_uav_piston",
                available=["egt", "cht", "oil_pressure"],
            )
            == "cht"
        )
        assert part_sensor_hint("Exhaust", available=["temperature"]) is None
        assert part_sensor_hint("Exhaust", available=["egt", "temperature"]) == "egt"
        assert part_sensor_hint("Crank", pack_id="aviation_uav_piston", available=["rpm"]) == "rpm"

        df = pd.DataFrame(
            {
                "machine_id": ["UAV-01", "UAV-01", "UAV-02"],
                "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-02"]),
                "temperature": [1.0, 9.5, 3.0],
                "egt": [100.0, 200.0, 50.0],
                "is_anomaly": [0, 1, 0],
            }
        )
        sensors = latest_asset_sensors(df, "UAV-01")
        assert sensors["temperature"] == 9.5
        assert sensors["egt"] == 200.0
        assert "is_anomaly" not in sensors
        card = build_part_card(
            part_name="Cylinder Head",
            asset_id="UAV-01",
            risk="High",
            rul_days=4,
            sensors=sensors,
            hint="egt",
        )
        assert card["asset_id"] == "UAV-01"
        assert card["risk"] == "High"
        assert card["part_name"] == "Cylinder Head"
        assert "asset" in card["honesty"].lower()
        assert parse_cad_part_event({"name": "Cyl", "dbId": 12, "properties": {"Name": "Cyl"}})["name"] == "Cyl"
        assert parse_cad_part_event({"cleared": True})["cleared"] is True
        assert parse_cad_part_event("nope") is None

        # Region events share the component channel; they must not clobber the part.
        assert parse_cad_part_event({"kind": "region", "matched": 0}) is None
        assert parse_cad_part_event({"kind": "part", "name": "Cyl", "dbId": 1})["name"] == "Cyl"

    def test_cad_node_name_sanitizer() -> None:
        """Autodesk mojibake node names must never reach the UI as ``??????``."""
        from src.cad_part import (
            build_part_card,
            parse_cad_part_event,
            parse_cad_region_event,
            sanitize_node_name,
        )

        # The exact live case: non-UTF8 STEP export of a Rotax solid.
        assert sanitize_node_name("?????? ??????-1-solid1") == "Solid-1"
        assert sanitize_node_name("?????? ??????-1-solid1", 55) == "Solid-1"
        assert "?" not in sanitize_node_name("?????? ??????-1-solid1")
        assert sanitize_node_name("?? ?? -2-solid4", 77) == "Solid-2"
        assert sanitize_node_name("\ufffd\ufffd\ufffd", 42) == "Solid-42"
        assert sanitize_node_name("", 7) == "Solid-7"
        assert sanitize_node_name(None, 7) == "Solid-7"
        assert sanitize_node_name("??????") == "Solid"
        assert sanitize_node_name("??????", 12) == "Solid-12"
        assert sanitize_node_name("?????-3", 99) == "Solid-3"
        assert sanitize_node_name("solid9") == "solid9"
        assert sanitize_node_name("??solid9", 4) == "Solid-9"
        # Readable names pass through untouched; asset ids are never sanitized here.
        assert sanitize_node_name("Cylinder Head", 3) == "Cylinder Head"
        assert sanitize_node_name("  Oil  Pump   Housing ", 9) == "Oil Pump Housing"
        assert sanitize_node_name("Cylinder ?? Head", 3) == "Cylinder Head"
        assert sanitize_node_name("UAV-03") == "UAV-03"
        assert sanitize_node_name("\x00\x01Exhaust\x07", 2) == "Exhaust"

        parsed = parse_cad_part_event({"name": "?????? ??????-1-solid1", "dbId": 55})
        assert parsed["display"] == "Solid-1"
        assert parsed["name"] == "?????? ??????-1-solid1"
        card = build_part_card(part_name="?????? ??????-1-solid1", asset_id="UAV-03", risk="High", db_id=55)
        assert card["part_name"] == "Solid-1"
        assert "?" not in card["part_name"]
        assert card["raw_part_name"] == "?????? ??????-1-solid1"
        assert card["asset_id"] == "UAV-03"
        assert build_part_card(part_name="", asset_id="UAV-03", risk="Low")["part_name"] == "—"

        report = parse_cad_region_event(
            {"kind": "region", "matched": 2, "nodes": 40, "names": ["?????-1-solid1", "Exhaust"], "risk": "high"}
        )
        assert report["matched"] == 2 and report["risk"] == "High"
        assert report["names"] == ["Solid-1", "Exhaust"]
        assert report["message"] == ""
        empty = parse_cad_region_event({"kind": "region", "matched": 0, "nodes": 40, "risk": "High"})
        assert empty["matched"] == 0 and "No CAD region matched" in empty["message"]
        assert parse_cad_region_event({"name": "Cyl", "dbId": 1}) is None
        assert parse_cad_region_event("nope") is None

    def test_cad_region_tint() -> None:
        """Only hint-matched dbIds get tinted. Zero matches never reddens the model."""
        from src.cad_part import (
            NO_REGION_MATCH_MSG,
            REGION_THEME_RGBA,
            region_hint_keywords,
            region_matches,
            region_theme_plan,
            region_theme_rgba,
        )

        assert region_theme_rgba("High") == REGION_THEME_RGBA["High"]
        assert region_theme_rgba("medium") == REGION_THEME_RGBA["Medium"]
        assert region_theme_rgba("Low") is None, "Low must not tint anything"
        assert region_theme_rgba("Unknown") is None
        assert region_theme_rgba("") is None
        assert set(REGION_THEME_RGBA) == {"High", "Medium"}

        needles = region_hint_keywords("aviation_uav_piston", available=["egt", "cht", "oil_pressure"])
        assert all(len(n) >= 3 for n in needles)
        assert needles == sorted(needles, key=len, reverse=True), "longest needle must win"
        for want in ("exhaust", "manifold", "vibration", "crank", "rpm", "oilpressure"):
            assert want in needles, (want, needles)
        assert len(needles) == len(set(needles))
        assert "machineid" not in needles and "timestamp" not in needles

        # 1) High + zero name matches → empty dbId list, honest message, no tint.
        solids = {0: "Assembly", 1: "Solid1", 2: "Solid2", 3: "?????? ??????-1-solid1"}
        blank = region_theme_plan(solids, "High", pack_id="aviation_uav_piston")
        assert blank["dbIds"] == [], blank
        assert blank["matched"] == 0 and blank["tint"] is False
        assert blank["nodes"] == 4
        assert blank["message"] == NO_REGION_MATCH_MSG
        assert "Solid1" in blank["message"] and "whole engine" in blank["message"]
        assert blank["rgba"] == REGION_THEME_RGBA["High"], "color known, just nothing to paint"

        # 2) High + an "exhaust" node → that id is in the red set, the bolt is not.
        named = {0: "Rotax 912", 1: "Exhaust manifold", 2: "Bolt_14", 3: "Cylinder Head 3"}
        hot = region_theme_plan(named, "High", pack_id="aviation_uav_piston", available=["egt", "cht"])
        assert 1 in hot["dbIds"] and 3 in hot["dbIds"]
        assert 2 not in hot["dbIds"] and 0 not in hot["dbIds"]
        assert hot["tint"] is True and hot["message"] == ""
        assert hot["rgba"] == REGION_THEME_RGBA["High"]
        assert hot["matched"] == 2 and hot["nodes"] == 4
        by_id = {m["dbId"]: m for m in hot["matches"]}
        assert "exhaust" in by_id[1]["keyword"] and by_id[1]["sensor"] == "egt"
        assert by_id[3]["sensor"] == "cht"

        # 3) Same nodes, Medium → orange. Low / Unknown → nothing at all.
        mid = region_theme_plan(named, "Medium", pack_id="aviation_uav_piston")
        assert mid["dbIds"] == hot["dbIds"]
        assert mid["rgba"] == REGION_THEME_RGBA["Medium"]
        for quiet in ("Low", "Unknown", "", None):
            plan = region_theme_plan(named, quiet, pack_id="aviation_uav_piston")
            assert plan["rgba"] is None and plan["dbIds"] == [] and plan["message"] == ""

        # Mojibake names still match on the readable fragment.
        moji = region_theme_plan({7: "?????? exhaust-1-solid1"}, "High", pack_id="aviation_uav_piston")
        assert moji["dbIds"] == [7]
        assert moji["matches"][0]["display"] == "Solid-1"

        # Node shapes: dict, (dbId, name) pairs, and [{"dbId":…, "name":…}].
        pairs = region_matches([(11, "Oil Pump"), (12, "Washer")], pack_id="aviation_uav_piston")
        assert [m["dbId"] for m in pairs] == [11]
        dicts = region_matches([{"dbId": 21, "name": "Crankshaft"}, {"dbId": 22, "name": "Nut"}])
        assert [m["dbId"] for m in dicts] == [21]
        assert region_matches(None) == [] and region_matches([]) == []
        assert region_theme_plan({}, "High")["dbIds"] == []
        # A session sensor column widens matching beyond the built-in keyword table.
        assert "coolanttemp" in region_hint_keywords("plant_rotating", available=["coolant_temp"])
        assert "coolanttemp" not in region_hint_keywords("plant_rotating")
        custom = region_matches(
            {5: "Coolant temp housing"}, available=["coolant_temp"], pack_id="plant_rotating"
        )
        assert [m["dbId"] for m in custom] == [5]
        assert custom[0]["keyword"] == "coolanttemp"
        assert region_matches({5: "Coolant housing"}, available=["coolant_temp"]) == []

    def test_cad_sensor_grid() -> None:
        """Mapped sensors chunk into rows of 3–4 so the viewer stays on-screen."""
        from src.cad_part import format_sensor_value, sensor_grid_rows

        sensors = {"temperature": 71.23456, "egt": 700, "vibration": 0.0421, "rpm": 5400, "cht": 180.5}
        rows = sensor_grid_rows(sensors, per_row=4)
        assert [len(r) for r in rows] == [4, 1]
        assert rows[0][0] == ("temperature", "71.2")
        assert rows[0][1] == ("egt", "700")
        assert rows[1] == [("cht", "180")]
        assert sum(len(r) for r in rows) == len(sensors)
        assert [len(r) for r in sensor_grid_rows(sensors, per_row=3)] == [3, 2]
        assert sensor_grid_rows({}) == [] and sensor_grid_rows(None) == []
        assert [len(r) for r in sensor_grid_rows(sensors, per_row=0)] == [1, 1, 1, 1, 1]
        assert format_sensor_value(None) == "—"
        assert format_sensor_value(1234567) == "1,234,567"
        assert format_sensor_value(0.000123456) == "0.000123"
        assert format_sensor_value("n/a") == "n/a"

    def test_aps_translate_mocked() -> None:
        """OSS signed-upload + MD job + poll — mocked HTTP only. Never hits Autodesk."""
        import json as _json

        from src.aps_viewer import INSUFFICIENT_SCOPE_HINT, TRANSLATE_SCOPES, translate_cad_bytes

        class _Resp:
            def __init__(self, status_code, json_data=None, text=""):
                self.status_code = status_code
                self._json = json_data
                self.text = text if text else (_json.dumps(json_data) if json_data is not None else "")
                self.content = self.text.encode() if self.text else b""

            def json(self):
                if self._json is None:
                    raise ValueError("no json")
                return self._json

        class FakeAPS:
            def __init__(self):
                self.calls: list[tuple[str, str, dict]] = []
                self.manifest_n = 0

            def post(self, url, **kw):
                self.calls.append(("POST", url, kw))
                if "authentication" in url:
                    return _Resp(200, {"access_token": "tok", "expires_in": 3600})
                if url.rstrip("/").endswith("/oss/v2/buckets"):
                    return _Resp(200, {"bucketKey": (kw.get("json") or {}).get("bucketKey")})
                if url.endswith("/signeds3upload"):
                    return _Resp(
                        200,
                        {
                            "objectId": "urn:adsk.objects:os.object:pdm-testclientid9-cad/obj.ipt",
                            "objectKey": "obj.ipt",
                        },
                    )
                if url.endswith("/job"):
                    body = kw.get("json") or {}
                    assert body.get("input", {}).get("urn")
                    fmt = (body.get("output") or {}).get("formats") or [{}]
                    assert fmt[0].get("type") in ("svf2", "svf")
                    return _Resp(201, {"result": "success", "urn": body["input"]["urn"]})
                return _Resp(500, {"developerMessage": "unexpected POST " + url})

            def get(self, url, **kw):
                self.calls.append(("GET", url, kw))
                if url.endswith("/details"):
                    return _Resp(404, {"developerMessage": "not found"})
                if "signeds3upload" in url:
                    return _Resp(200, {"uploadKey": "uk", "urls": ["https://s3.example.test/p1"]})
                if "/manifest" in url:
                    self.manifest_n += 1
                    if self.manifest_n < 2:
                        return _Resp(200, {"status": "inprogress", "progress": "45%"})
                    return _Resp(200, {"status": "success", "progress": "complete"})
                return _Resp(500, {"developerMessage": "unexpected GET " + url})

            def put(self, url, **kw):
                self.calls.append(("PUT", url, kw))
                assert url.startswith("https://s3.example.test/")
                assert kw.get("data") == b"tiny-cad-bytes"
                return _Resp(200, text="")

        class ScopeDenied:
            def post(self, url, **kw):
                return _Resp(
                    401,
                    {
                        "errorCode": "AUTH-010",
                        "developerMessage": "The access token does not have the required privileges",
                    },
                )

            def get(self, url, **kw):
                raise AssertionError("scope-denied token must not continue to OSS")

            def put(self, url, **kw):
                raise AssertionError("scope-denied token must not PUT S3")

        prev_id, prev_sec = os.environ.get("APS_CLIENT_ID"), os.environ.get("APS_CLIENT_SECRET")
        os.environ["APS_CLIENT_ID"] = "TestClientID99"
        os.environ["APS_CLIENT_SECRET"] = "not-a-real-secret"
        try:
            http = FakeAPS()
            sleeps: list[float] = []
            result = translate_cad_bytes(
                "box.ipt",
                b"tiny-cad-bytes",
                http=http,
                sleep=sleeps.append,
                poll_timeout_s=20,
                poll_interval_s=0.01,
            )
            assert result["ok"] is True, result
            assert result["phase"] == "success"
            assert result["urn"] and not result["urn"].startswith("urn:")
            assert result["bucket"] == "pdm-testclientid-cad"
            token_post = next(c for c in http.calls if c[0] == "POST" and "authentication" in c[1])
            assert TRANSLATE_SCOPES == token_post[2]["data"]["scope"]
            assert any(c[0] == "PUT" for c in http.calls)
            assert any(c[0] == "POST" and c[1].endswith("/job") for c in http.calls)
            assert sleeps, "manifest poll should sleep between attempts"
            assert not any("aps.autodesk.com" in (c[1] or "") and c[0] == "PUT" for c in http.calls)

            denied = translate_cad_bytes("a.ipt", b"x", http=ScopeDenied(), sleep=lambda _s: None)
            assert denied["ok"] is False
            assert INSUFFICIENT_SCOPE_HINT in denied["message"]

            empty = translate_cad_bytes("a.ipt", b"", http=FakeAPS(), sleep=lambda _s: None)
            assert empty["ok"] is False and "empty" in empty["message"].lower()
        finally:
            if prev_id is None:
                os.environ.pop("APS_CLIENT_ID", None)
            else:
                os.environ["APS_CLIENT_ID"] = prev_id
            if prev_sec is None:
                os.environ.pop("APS_CLIENT_SECRET", None)
            else:
                os.environ["APS_CLIENT_SECRET"] = prev_sec

    def test_aps_zip_root_filename() -> None:
        """Zip CAD root: one .step extracts; empty zip fails clearly; never send empty root."""
        import io
        import json as _json
        import zipfile

        from src.aps_viewer import (
            ZIP_ROOT_MISSING_MSG,
            guess_zip_root_from_upload_name,
            is_cad_zip_name,
            pick_zip_cad_root,
            resolve_zip_root_filename,
            start_svf_job,
            translate_cad_bytes,
        )

        def _zip_bytes(members: dict) -> bytes:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for name, body in members.items():
                    zf.writestr(name, body)
            return buf.getvalue()

        assert is_cad_zip_name("model.ZIP")
        assert is_cad_zip_name("pack.ifczip")
        assert not is_cad_zip_name("Rotax.step")

        one = _zip_bytes({"Rotax.step": b"ISO-10303-21;"})
        assert pick_zip_cad_root(one) == "Rotax.step"
        assert resolve_zip_root_filename("rotax.zip", one, "") == "Rotax.step"
        assert resolve_zip_root_filename("rotax.zip", one, "  override.stp  ") == "override.stp"

        empty = _zip_bytes({})
        assert pick_zip_cad_root(empty) == ""
        assert resolve_zip_root_filename("empty.zip", empty, "") == ""

        no_cad = _zip_bytes({"readme.txt": b"hello", "photos/a.png": b"x"})
        assert pick_zip_cad_root(no_cad) == ""

        mixed = _zip_bytes(
            {
                "notes.txt": b"x" * 4000,
                "folder/engine.step": b"ISO",
                "other.ipt": b"y" * 8000,
            }
        )
        assert pick_zip_cad_root(mixed) == "folder/engine.step"

        largest = _zip_bytes({"small.iam": b"a", "big.iam": b"b" * 200})
        assert pick_zip_cad_root(largest) == "big.iam"

        upper = _zip_bytes({"ROTAX.STEP": b"ISO-10303-21;"})
        assert pick_zip_cad_root(upper) == "ROTAX.STEP"

        macos_only = _zip_bytes(
            {
                "__MACOSX/._Engine Rotax 912 ULS.STEP": b"junk",
                "__MACOSX/Engine Rotax 912 ULS.STEP": b"junk",
            }
        )
        assert pick_zip_cad_root(macos_only) == ""

        rotax_inner = "Engine Rotax 912 ULS.STEP"
        rotax_upload = "Engine Rotax 912 ULS.STEP.zip"
        rotax_zip = _zip_bytes({rotax_inner: b"ISO-10303-21;"})
        assert pick_zip_cad_root(rotax_zip) == rotax_inner
        assert resolve_zip_root_filename(rotax_upload, rotax_zip, "") == rotax_inner
        assert resolve_zip_root_filename(rotax_upload, rotax_zip, "   ") == rotax_inner
        rotax_with_macos = _zip_bytes(
            {rotax_inner: b"ISO-10303-21;", "__MACOSX/._" + rotax_inner: b"junk"}
        )
        assert pick_zip_cad_root(rotax_with_macos) == rotax_inner

        # Filename-only fallback: do not open zip (empty / invalid bytes).
        assert guess_zip_root_from_upload_name(rotax_upload) == rotax_inner
        assert guess_zip_root_from_upload_name("Something.stp.zip") == "Something.stp"
        assert guess_zip_root_from_upload_name("Something.STEP.zip") == "Something.STEP"
        assert guess_zip_root_from_upload_name("Something.stp.ZIP") == "Something.stp"
        assert guess_zip_root_from_upload_name("folder/Engine Rotax 912 ULS.STEP.zip") == rotax_inner
        assert guess_zip_root_from_upload_name("nope.zip") == ""
        assert guess_zip_root_from_upload_name("Rotax.step") == ""
        assert guess_zip_root_from_upload_name("pack.ifczip") == ""
        assert resolve_zip_root_filename(rotax_upload, b"", "") == rotax_inner
        assert resolve_zip_root_filename(rotax_upload, b"not-a-zip", "") == rotax_inner
        assert resolve_zip_root_filename(rotax_upload, macos_only, "") == rotax_inner

        app_src = (ROOT / "app.py").read_text(encoding="utf-8")
        assert "pick_zip_cad_root" in app_src
        assert "guess_zip_root_from_upload_name" in app_src
        assert "ZIP_ROOT_MISSING_MSG" in app_src
        assert "resolve_zip_root_filename" in app_src
        assert "Detected ZIP root" in app_src
        assert "_seed_zip_root_widget" in app_src
        assert "PENDING_ZIP_ROOT" in app_src
        assert "_pending_aps_zip_root_filename" in app_src

        try:
            start_svf_job("tok", "dXJuOmFi", compressed=True, root_filename="")
            raise AssertionError("start_svf_job must reject compressedUrn without rootFilename")
        except ValueError as exc:
            assert str(exc) == ZIP_ROOT_MISSING_MSG

        class NoHttp:
            def post(self, url, **kw):
                raise AssertionError("empty zip must not call Autodesk: " + url)

            def get(self, url, **kw):
                raise AssertionError("empty zip must not call Autodesk: " + url)

            def put(self, url, **kw):
                raise AssertionError("empty zip must not call Autodesk: " + url)

        blocked = translate_cad_bytes("model.zip", empty, http=NoHttp(), sleep=lambda _s: None)
        assert blocked["ok"] is False
        assert blocked["phase"] == "error"
        assert blocked["message"] == ZIP_ROOT_MISSING_MSG
        assert blocked["urn"] == ""

        nocad = translate_cad_bytes("pack.ifczip", no_cad, http=NoHttp(), sleep=lambda _s: None)
        assert nocad["ok"] is False and nocad["message"] == ZIP_ROOT_MISSING_MSG

        class _Resp:
            def __init__(self, status_code, json_data=None, text=""):
                self.status_code = status_code
                self._json = json_data
                self.text = text if text else (_json.dumps(json_data) if json_data is not None else "")
                self.content = self.text.encode() if self.text else b""

            def json(self):
                if self._json is None:
                    raise ValueError("no json")
                return self._json

        class FakeZipAPS:
            def __init__(self, expected_root: str, expected_data: bytes):
                self.calls: list = []
                self.job_input: dict = {}
                self.expected_root = expected_root
                self.expected_data = expected_data

            def post(self, url, **kw):
                self.calls.append(("POST", url, kw))
                if "authentication" in url:
                    return _Resp(200, {"access_token": "tok", "expires_in": 3600})
                if url.rstrip("/").endswith("/oss/v2/buckets"):
                    return _Resp(200, {"bucketKey": (kw.get("json") or {}).get("bucketKey")})
                if url.endswith("/signeds3upload"):
                    return _Resp(
                        200,
                        {
                            "objectId": "urn:adsk.objects:os.object:pdm-testclientid9-cad/obj.zip",
                            "objectKey": "obj.zip",
                        },
                    )
                if url.endswith("/job"):
                    body = kw.get("json") or {}
                    self.job_input = dict(body.get("input") or {})
                    assert self.job_input.get("compressedUrn") is True
                    assert self.job_input.get("rootFilename") == self.expected_root
                    assert str(self.job_input.get("rootFilename") or "").strip()
                    return _Resp(201, {"result": "success", "urn": self.job_input.get("urn")})
                return _Resp(500, {"developerMessage": "unexpected POST " + url})

            def get(self, url, **kw):
                self.calls.append(("GET", url, kw))
                if url.endswith("/details"):
                    return _Resp(404, {"developerMessage": "not found"})
                if "signeds3upload" in url:
                    return _Resp(200, {"uploadKey": "uk", "urls": ["https://s3.example.test/p1"]})
                if "/manifest" in url:
                    return _Resp(200, {"status": "success", "progress": "complete"})
                return _Resp(500, {"developerMessage": "unexpected GET " + url})

            def put(self, url, **kw):
                self.calls.append(("PUT", url, kw))
                assert url.startswith("https://s3.example.test/")
                assert kw.get("data") == self.expected_data
                return _Resp(200, text="")

        def _run_zip_translate(upload_name: str, payload: bytes, expected_root: str) -> None:
            http = FakeZipAPS(expected_root, payload)
            result = translate_cad_bytes(
                upload_name,
                payload,
                root_filename="",
                http=http,
                sleep=lambda _s: None,
                poll_timeout_s=20,
                poll_interval_s=0.01,
            )
            assert result["ok"] is True, result
            assert http.job_input.get("rootFilename") == expected_root
            assert http.job_input.get("compressedUrn") is True
            assert not any(
                c[0] == "POST"
                and c[1].endswith("/job")
                and not str((c[2].get("json") or {}).get("input", {}).get("rootFilename") or "").strip()
                for c in http.calls
            )

        prev_id, prev_sec = os.environ.get("APS_CLIENT_ID"), os.environ.get("APS_CLIENT_SECRET")
        os.environ["APS_CLIENT_ID"] = "TestClientID99"
        os.environ["APS_CLIENT_SECRET"] = "not-a-real-secret"
        try:
            _run_zip_translate("rotax.zip", one, "Rotax.step")
            _run_zip_translate(rotax_upload, rotax_zip, rotax_inner)
            # Scan misses (__MACOSX only); filename `Something.STEP.zip` still supplies root.
            _run_zip_translate(rotax_upload, macos_only, rotax_inner)
        finally:
            if prev_id is None:
                os.environ.pop("APS_CLIENT_ID", None)
            else:
                os.environ["APS_CLIENT_ID"] = prev_id
            if prev_sec is None:
                os.environ.pop("APS_CLIENT_SECRET", None)
            else:
                os.environ["APS_CLIENT_SECRET"] = prev_sec

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

    def test_industry_packs() -> None:
        from src.industry_packs import (
            AUTOMOTIVE_OEM_TRIM_EV_NOTE,
            DEFAULT_PACK_ID,
            PACKS,
            extra_aliases_for,
            extra_field_defs,
            get_pack,
            list_packs,
            preferred_y_metric,
            sample_path_for,
            suggest_pack,
            twin_kind_for,
            validate_packs,
        )

        assert not validate_packs()
        packs = list_packs()
        assert packs[0]["id"] == DEFAULT_PACK_ID == "plant_rotating"
        assert [p["id"] for p in packs] == [
            "plant_rotating",
            "aviation_uav_piston",
            "automotive_powertrain",
            "oil_srp",
        ]
        assert get_pack("missing")["id"] == DEFAULT_PACK_ID
        assert twin_kind_for("aviation_uav_piston") == "aviation_uav"
        assert twin_kind_for("automotive_powertrain") == "auto_car"
        assert twin_kind_for("oil_srp") == "oil_srp"
        assert PACKS["aviation_uav_piston"]["sih"] == "SIH26054"
        assert PACKS["oil_srp"]["sih"] == "SIH26120"
        assert PACKS["aviation_uav_piston"]["hero"] is True
        for token in ("OEM", "trim", "EV"):
            assert token in AUTOMOTIVE_OEM_TRIM_EV_NOTE
            assert token in PACKS["automotive_powertrain"]["not_in_scope"]
        assert extra_field_defs("plant_rotating") == []
        assert "egt" in extra_aliases_for("aviation_uav_piston")
        from src.industry_packs import AVIATION_PART_HINTS, part_hints_for

        assert part_hints_for("aviation_uav_piston")["exhaust"] == "egt"
        assert part_hints_for("aviation_uav_piston")["cyl"] == "cht"
        assert part_hints_for("plant_rotating") == {}
        assert "crank" in AVIATION_PART_HINTS
        assert "pump_fillage" in extra_aliases_for("oil_srp")
        assert sample_path_for("aviation_uav_piston").name == "aviation_uav_piston.csv"

        av = suggest_pack(["timestamp", "egt", "cht", "flight_hours"], filename="uav_sortie.csv")
        assert av["pack_id"] == "aviation_uav_piston"
        auto = suggest_pack(["coolant_temp", "engine_load", "vehicle_speed"], machine_ids=["ENG-01"])
        assert auto["pack_id"] == "automotive_powertrain"
        oil = suggest_pack(["pump_fillage", "polish_rod_load"], machine_ids=["WELL-03"])
        assert oil["pack_id"] == "oil_srp"
        plant = suggest_pack(["timestamp", "machine_id", "temperature", "vibration", "pressure", "rpm"])
        assert plant["pack_id"] == DEFAULT_PACK_ID
        empty = suggest_pack([])
        assert empty["pack_id"] == DEFAULT_PACK_ID
        # numpy unique() arrays must not raise (truth-value ambiguity).
        import numpy as np

        np_ids = suggest_pack(["egt", "cht"], machine_ids=np.array(["UAV-01", "UAV-02"]))
        assert np_ids["pack_id"] == "aviation_uav_piston"
        df_av = pd.DataFrame({"egt": [1.0], "vibration": [2.0]})
        assert preferred_y_metric(df_av, "aviation_uav_piston") == "egt"

    def test_pack_kpis_every_function() -> None:
        from src.pack_kpis import (
            compute_pack_kpis,
            empty_bundle,
            insight_cards_from_kpis,
            mission_reliability_pct,
            remaining_mission_hours,
            risk_to_health,
        )

        assert risk_to_health("High") < risk_to_health("Medium") < risk_to_health("Low")
        assert remaining_mission_hours(10, duty_cycle=0.5) == 120.0
        assert remaining_mission_hours(None) == 0.0
        assert remaining_mission_hours(-1) == 0.0
        lows = [{"machine_id": "A", "risk_level": "Low", "predicted_rul_days": 30}]
        highs = [{"machine_id": "A", "risk_level": "High", "predicted_rul_days": 2}]
        assert mission_reliability_pct(lows) > mission_reliability_pct(highs)
        assert mission_reliability_pct([]) == 0.0
        empty = empty_bundle("aviation_uav_piston")
        assert empty["pack_id"] == "aviation_uav_piston" and empty["kpis"]

        mixed = [
            {"machine_id": "UAV-01", "risk_level": "Low", "predicted_rul_days": 20},
            {"machine_id": "UAV-03", "risk_level": "High", "predicted_rul_days": 3},
        ]
        df = pd.DataFrame(
            {
                "machine_id": ["UAV-01"] * 4 + ["UAV-03"] * 4,
                "egt": [700, 702, 701, 703, 820, 830, 825, 840],
                "oil_pressure": [60, 61, 59, 60, 28, 26, 27, 25],
                "vibration": [1.2, 1.1, 1.3, 1.2, 5.0, 5.4, 5.1, 5.6],
                "temperature": [170, 171, 169, 170, 210, 215, 212, 218],
            }
        )
        bundle = compute_pack_kpis("aviation_uav_piston", df, mixed)
        ids = {k["id"] for k in bundle["kpis"]}
        assert {"engine_health_index", "mission_reliability_pct", "remaining_mission_hours", "egt_margin"} <= ids
        by_id = {a["machine_id"]: a for a in bundle["by_asset"]}
        assert by_id["UAV-03"]["health_index"] < by_id["UAV-01"]["health_index"]
        cards = insight_cards_from_kpis(bundle)
        assert cards and "SIH26054" in cards[0]["title"]
        assert any("UAV-03" in c["message"] for c in cards)

        auto_preds = [
            {"machine_id": "ENG-01", "risk_level": "Low", "predicted_rul_days": 18},
            {"machine_id": "ENG-03", "risk_level": "High", "predicted_rul_days": 4},
        ]
        auto_df = pd.DataFrame(
            {
                "machine_id": ["ENG-01"] * 3 + ["ENG-03"] * 3,
                "coolant_temp": [90, 91, 89, 108, 110, 109],
                "oil_pressure": [42, 41, 43, 12, 11, 10],
                "engine_load": [40, 42, 41, 80, 82, 85],
                "vibration": [1.5, 1.4, 1.6, 4.8, 5.0, 5.1],
            }
        )
        auto_b = compute_pack_kpis("automotive_powertrain", auto_df, auto_preds)
        assert {k["id"] for k in auto_b["kpis"]} >= {"powertrain_health_index", "oil_pressure_status", "thermal_headroom"}
        assert auto_b["by_asset"][0]["machine_id"] == "ENG-03"

        oil_preds = [
            {"machine_id": "WELL-01", "risk_level": "Low", "predicted_rul_days": 16},
            {"machine_id": "WELL-03", "risk_level": "High", "predicted_rul_days": 5},
        ]
        oil_df = pd.DataFrame(
            {
                "machine_id": ["WELL-01"] * 3 + ["WELL-03"] * 3,
                "pump_fillage": [92, 91, 93, 58, 55, 52],
                "polish_rod_load": [14000, 14100, 13900, 19000, 19500, 19800],
                "production_bbl": [45, 46, 44, 18, 16, 15],
                "vibration": [2.0, 2.1, 1.9, 6.0, 6.2, 6.4],
            }
        )
        oil_b = compute_pack_kpis("oil_srp", oil_df, oil_preds)
        assert {k["id"] for k in oil_b["kpis"]} >= {"well_health_index", "pump_fillage_pct", "production_rate"}
        assert oil_b["by_asset"][0]["machine_id"] == "WELL-03"

        plant_b = compute_pack_kpis(
            "plant_rotating",
            pd.DataFrame({"machine_id": ["M-1"], "vibration": [1.0], "temperature": [70]}),
            [{"machine_id": "M-1", "risk_level": "Low", "predicted_rul_days": 25}],
        )
        assert {k["id"] for k in plant_b["kpis"]} >= {"asset_health_index", "high_risk_assets"}

    def test_pack_samples_pipeline() -> None:
        from src.data_cleaner import clean_and_quality
        from src.graphs.pack_kpis import create_asset_health_chart, create_pack_kpi_bars
        from src.industry_packs import PACK_ORDER, sample_path_for
        from src.ml.anomaly_detector import AnomalyDetector
        from src.ml.rul_predictor import RULPredictor
        from src.pack_kpis import compute_pack_kpis
        from src.sensor_map import suggest_mapping
        from src.twin3d import build_twin_html
        from src.industry_packs import twin_kind_for as kind_for

        for pid in PACK_ORDER:
            path = sample_path_for(pid)
            assert path.exists(), f"missing demo CSV for {pid}: {path}"
            df = pd.read_csv(path, parse_dates=["timestamp"])
            mapping = suggest_mapping(list(df.columns), pack_id=pid)
            assert mapping["machine_id"] == "machine_id"
            assert mapping["temperature"] == "temperature"
            if pid == "aviation_uav_piston":
                assert mapping.get("egt") == "egt"
                assert mapping.get("flight_hours") == "flight_hours"
            if pid == "automotive_powertrain":
                assert mapping.get("coolant_temp") == "coolant_temp"
            if pid == "oil_srp":
                assert mapping.get("pump_fillage") == "pump_fillage"
            cleaned, _, _ = clean_and_quality(df, run_quality=False)
            det = AnomalyDetector(contamination=0.08)
            det.fit(cleaned)
            if pid == "aviation_uav_piston":
                assert "egt" in det.feature_columns
            rul = RULPredictor()
            rul.fit(cleaned)
            preds = rul.predict_latest_per_machine(cleaned)
            assert preds
            bundle = compute_pack_kpis(pid, cleaned, preds)
            assert bundle["kpis"] and bundle["by_asset"]
            fig_h = create_asset_health_chart(bundle["by_asset"])
            fig_k = create_pack_kpi_bars(bundle["kpis"])
            assert fig_h.layout.title.text or fig_h.data
            assert fig_k is not None
            html = build_twin_html(preds, selected_id=preds[0]["machine_id"], kind=kind_for(pid))
            assert f'"kind": "{kind_for(pid)}"' in html or f'"kind":"{kind_for(pid)}"' in html
            assert "jsdelivr" not in html

    def test_pack_mapping_does_not_steal_core() -> None:
        from src.sensor_map import apply_mapping, suggest_mapping

        cols = ["time", "uav_id", "temp_c", "vib", "press_psi", "egt_c", "oil_psi"]
        mapping = suggest_mapping(cols, pack_id="aviation_uav_piston")
        assert mapping["timestamp"] == "time"
        assert mapping["machine_id"] == "uav_id"
        assert mapping["temperature"] == "temp_c"
        assert mapping["vibration"] == "vib"
        assert mapping["pressure"] == "press_psi"
        assert mapping["egt"] == "egt_c"
        assert mapping["oil_pressure"] == "oil_psi"
        mapped = apply_mapping(pd.DataFrame({c: [1] for c in cols}), mapping)
        assert "egt" in mapped.columns and "temperature" in mapped.columns

    def test_email_honesty() -> None:
        from src.email_report import generate_email_body, high_risk_predictions, send_email
        from src.pack_kpis import compute_pack_kpis

        lows = [
            {"machine_id": "M-1", "risk_level": "Low", "predicted_rul_days": 26, "message": "M-1 ok"},
        ]
        assert high_risk_predictions(lows) == []
        low_body = generate_email_body("Manager", lows)
        assert "IMMEDIATE ACTION REQUIRED" not in low_body
        assert "No High-risk" in low_body
        assert "not a confirmed failure date" in low_body.lower()

        mixed = lows + [
            {"machine_id": "UAV-03", "risk_level": "High", "predicted_rul_days": 4, "message": "UAV-03 High"},
        ]
        high_body = generate_email_body("Manager", mixed)
        assert "IMMEDIATE ACTION REQUIRED" in high_body
        assert "UAV-03" in high_body
        assert "predicted to fail in" not in high_body  # no fake calendar date

        bundle = compute_pack_kpis(
            "aviation_uav_piston",
            pd.DataFrame({"machine_id": ["UAV-03"], "egt": [800], "vibration": [5]}),
            mixed,
        )
        packed = generate_email_body("Manager", mixed, pack_kpis=bundle)
        assert "MISSION RELIABILITY" in packed.upper() or "Piston-engine" in packed or "KPIs" in packed
        sent = send_email("ops@example.com", "t", packed)
        assert sent["success"] is True and sent["mode"] == "demo"

    def test_dashboard_composer() -> None:
        from src.dashboard_composer import (
            cad_placeholder_html,
            cad_slot_status,
            compose_dashboard_html,
            board_twin_html,
            board_pack_charts,
        )
        from src.pack_kpis import compute_pack_kpis

        status = cad_slot_status("")
        assert status["credentials"] is False
        assert status["ready"] is False
        ph = cad_placeholder_html(status=status)
        assert "APS_CLIENT_ID" in ph and "post-deploy" in ph.lower() or "Render" in ph or "credentials" in ph.lower()

        import tempfile
        from pathlib import Path

        from src.aps_viewer import save_urn_for_pack

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "cad_urns.json"
            prev = os.environ.get("PDM_CAD_URNS_PATH")
            os.environ["PDM_CAD_URNS_PATH"] = str(store)
            try:
                save_urn_for_pack(
                    "aviation_uav_piston",
                    "dXJuOmFkc2suYXZpYXRpb250ZXN0",
                    source="translate",
                )
                packed = cad_slot_status("", pack_id="aviation_uav_piston")
                assert packed["urn"] == "dXJuOmFkc2suYXZpYXRpb250ZXN0"
                assert packed["source"] == "saved"
                assert packed["ready"] is False  # still no APS credentials in this test
            finally:
                if prev is None:
                    os.environ.pop("PDM_CAD_URNS_PATH", None)
                else:
                    os.environ["PDM_CAD_URNS_PATH"] = prev

        preds = [{"machine_id": "UAV-03", "risk_level": "High", "predicted_rul_days": 4}]
        bundle = compute_pack_kpis("aviation_uav_piston", pd.DataFrame({"machine_id": ["UAV-03"], "egt": [800]}), preds)
        twin = board_twin_html(preds, "aviation_uav_piston", selected_id="UAV-03")
        figs = board_pack_charts(bundle)
        html = compose_dashboard_html(
            tiles=["kpis", "charts", "insights", "twin3d", "cad"],
            pack_id="aviation_uav_piston",
            kpi_bundle=bundle,
            insight_cards=[{"title": "Inspect UAV-03", "severity": "high", "message": "High risk"}],
            chart_figs=figs,
            twin_html=twin,
            cad_html="",
            cad_status=status,
            title="test board",
        )
        assert "test board" in html
        assert "SIH26054" in html or "Aviation" in html
        assert '"kind": "aviation_uav"' in html or "aviation_uav" in html
        assert "jsdelivr" not in twin
        assert "Waiting for credentials" in html or "CAD TWIN" in html
        assert "pyspark" not in html.lower()

    def test_joins_step() -> None:
        from src.data_integration import join_two

        left = pd.DataFrame({"machine_id": ["A", "B"], "vibration": [1.0, 2.0]})
        right = pd.DataFrame({"machine_id": ["A"], "work_order": ["WO-1"]})
        merged, meta = join_two(left, right, how="left", on=["machine_id"])
        assert len(merged) == 2 and "work_order" in merged.columns
        assert meta["how"] == "left"

    def _app_ast():
        import ast

        src = (ROOT / "app.py").read_text(encoding="utf-8")
        return src, ast.parse(src)

    def _literal_widget_keys(node) -> list[tuple[int, str]]:
        import ast

        found: list[tuple[int, str]] = []
        for n in ast.walk(node):
            if not isinstance(n, ast.Call):
                continue
            for kw in n.keywords:
                if kw.arg != "key":
                    continue
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    found.append((n.lineno, kw.value.value))
        return found

    def _is_st_session_state(node) -> bool:
        import ast

        return (
            isinstance(node, ast.Attribute)
            and node.attr == "session_state"
            and isinstance(node.value, ast.Name)
            and node.value.id == "st"
        )

    def _session_state_writes(node) -> list[tuple[int, str]]:
        """Literal ``st.session_state.X =`` / ``st.session_state['X'] =`` writes."""
        import ast

        found: list[tuple[int, str]] = []
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                targets = n.targets
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and n.target is not None:
                targets = [n.target]
            else:
                continue
            for t in targets:
                if isinstance(t, ast.Attribute) and _is_st_session_state(t.value):
                    found.append((n.lineno, t.attr))
                elif isinstance(t, ast.Subscript) and _is_st_session_state(t.value):
                    sl = t.slice
                    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                        found.append((n.lineno, sl.value))
        return found

    def _fn_named(tree, name):
        import ast

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

    def _pre_widget_funcs(tree) -> set[str]:
        import ast

        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "PRE_WIDGET_SESSION_FUNCS" not in names:
                continue
            val = node.value
            if isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id == "frozenset":
                arg = val.args[0]
                if isinstance(arg, (ast.Set, ast.List, ast.Tuple)):
                    return {elt.value for elt in arg.elts if isinstance(elt, ast.Constant)}
        raise AssertionError("app.py must define PRE_WIDGET_SESSION_FUNCS frozenset")

    def _on_click_callbacks(tree) -> set[str]:
        import ast

        names: set[str] = set()
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            for kw in n.keywords:
                if kw.arg != "on_click":
                    continue
                if isinstance(kw.value, ast.Name):
                    names.add(kw.value.id)
        return names

    def test_streamlit_widget_keys_unique() -> None:
        """Literal Streamlit `key=` values must be unique within a script run.

        Duplicate keys crash Streamlit with StreamlitDuplicateElementKey (seen
        live when `_render_maintenance_attach` pasted the same uploader twice).
        Pages are mutually exclusive; sidebar + helpers run with every page.
        """
        from collections import Counter

        _, tree = _app_ast()

        attach = _fn_named(tree, "_render_maintenance_attach")
        assert attach is not None, "_render_maintenance_attach missing from app.py"
        attach_keys = [k for _, k in _literal_widget_keys(attach)]
        dupes = [k for k, n in Counter(attach_keys).items() if n > 1]
        assert not dupes, f"duplicate Streamlit keys in _render_maintenance_attach: {dupes}"
        assert attach_keys.count("upload_maintenance_table") == 1

        shared: list[str] = []
        page_dupes: list[str] = []
        for node in tree.body:
            if not hasattr(node, "name"):
                continue
            keys = [k for _, k in _literal_widget_keys(node)]
            if node.name.startswith("page_"):
                counts = Counter(keys)
                for key, n in counts.items():
                    if n > 1:
                        page_dupes.append(f"{node.name}:{key}×{n}")
            else:
                shared.extend(keys)

        shared_dupes = [k for k, n in Counter(shared).items() if n > 1]
        assert not shared_dupes, "duplicate Streamlit keys in sidebar/helpers: " + ", ".join(shared_dupes)
        assert not page_dupes, "duplicate Streamlit keys: " + ", ".join(page_dupes)

        collisions: list[str] = []
        for node in tree.body:
            if not hasattr(node, "name") or not node.name.startswith("page_"):
                continue
            combo = shared + [k for _, k in _literal_widget_keys(node)]
            for key, n in Counter(combo).items():
                if n > 1:
                    collisions.append(f"{node.name}:{key}×{n}")
        assert not collisions, "page key collides with sidebar/helper: " + ", ".join(collisions)

    def test_no_widget_key_writes_after_instantiate() -> None:
        """Do not assign ``st.session_state.<key>`` after a widget with that key exists.

        StreamlitAPIException: ``st.session_state.X cannot be modified after the
        widget with key X is instantiated`` — even when writing the same value
        (Load Plant sample / Load pack demo / Suggest pack on Render).

        Heuristic (static, whole app.py):
        - Collect every literal ``key="..."``.
        - Flag ``st.session_state.<same> =`` / ``st.session_state["same"] =``.
        Allowed:
        1. Functions in ``PRE_WIDGET_SESSION_FUNCS`` (run at top of ``main()``
           before sidebar widgets) — ``init_session_state``, ``apply_pending_industry_pack``,
           and ``_seed_cad_urn_widgets`` (CAD Twin URN field, before that page's text_input).
        2. Functions passed as ``on_click=`` (Streamlit runs those before widgets).
        3. Same function as the widget, assignment line **before** the widget.
        """
        import ast

        _, tree = _app_ast()
        all_widget_keys = {k for _, k in _literal_widget_keys(tree)}
        safe = _pre_widget_funcs(tree) | _on_click_callbacks(tree)
        assert "apply_pending_pipeline_page" in safe
        assert "apply_pending_industry_pack" in safe
        assert "init_session_state" in safe
        assert "_seed_cad_urn_widgets" in safe
        assert "_seed_zip_root_widget" in safe

        main_fn = _fn_named(tree, "main")
        assert main_fn is not None
        apply_line = None
        sidebar_line = None
        for n in ast.walk(main_fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                if n.func.id == "apply_pending_industry_pack":
                    apply_line = n.lineno
                elif n.func.id == "render_pack_sidebar":
                    sidebar_line = n.lineno
        assert apply_line is not None, "main() must call apply_pending_industry_pack()"
        assert sidebar_line is not None, "main() must call render_pack_sidebar()"
        assert apply_line < sidebar_line, "apply pending pack before the industry_pack selectbox"

        apply_page_line = None
        radio_line = None
        for n in ast.walk(main_fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "apply_pending_pipeline_page":
                apply_page_line = n.lineno
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "radio":
                radio_line = n.lineno
        assert apply_page_line is not None, "main() must call apply_pending_pipeline_page()"
        assert radio_line is not None, "main() must create the Pipeline radio"
        assert apply_page_line < radio_line, "apply pending pipeline page before the radio widget"
        assert "apply_pending_pipeline_page" in safe

        violations: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            first_widget_line: dict[str, int] = {}
            for lineno, key in _literal_widget_keys(node):
                first_widget_line[key] = min(lineno, first_widget_line.get(key, lineno))
            for lineno, key in _session_state_writes(node):
                if key not in all_widget_keys:
                    continue
                if node.name in safe:
                    continue
                if key in first_widget_line and lineno < first_widget_line[key]:
                    continue
                violations.append(
                    f"{node.name}:{lineno} writes session_state.{key} after widget key={key!r}"
                )
        assert not violations, "widget-bound session_state writes: " + "; ".join(violations)

    def test_load_pack_sample_queues_pending() -> None:
        """load_pack_sample must not write industry_pack; it queues _pending_industry_pack."""
        import ast

        _, tree = _app_ast()
        fn = _fn_named(tree, "load_pack_sample")
        assert fn is not None
        writes = {k for _, k in _session_state_writes(fn)}
        assert "industry_pack" not in writes, "load_pack_sample must not assign industry_pack"
        called = False
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "request_industry_pack":
                called = True
        assert called, "load_pack_sample must call request_industry_pack()"

        req = _fn_named(tree, "request_industry_pack")
        assert req is not None
        assert "industry_pack" not in {k for _, k in _session_state_writes(req)}
        uses_pending = any(
            isinstance(n, ast.Name) and n.id == "PENDING_INDUSTRY_PACK" for n in ast.walk(req)
        )
        assert uses_pending, "request_industry_pack must set PENDING_INDUSTRY_PACK"

        req_page = _fn_named(tree, "request_pipeline_page")
        assert req_page is not None
        assert "pipeline_page" not in {k for _, k in _session_state_writes(req_page)}
        uses_page_pending = any(
            isinstance(n, ast.Name) and n.id == "PENDING_PIPELINE_PAGE" for n in ast.walk(req_page)
        )
        assert uses_page_pending, "request_pipeline_page must set PENDING_PIPELINE_PAGE"
        card_fn = _fn_named(tree, "render_cad_part_card")
        assert card_fn is not None
        assert any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "request_pipeline_page"
            for n in ast.walk(card_fn)
        )
        assert "cad_open_anomaly_rul" in {k for _, k in _literal_widget_keys(card_fn)}

        sidebar = _fn_named(tree, "render_pack_sidebar")
        assert sidebar is not None
        assert "industry_pack" not in {k for _, k in _session_state_writes(sidebar)}

    def test_cad_twin_layout_order() -> None:
        """CAD Twin renders strip → GuiViewer3D → sensors → admin, in that order.

        The whole point of the layout: the user sees the engine without scrolling
        past a tall sensor column, and the uploader / URN controls never push the
        viewer off-screen.
        """
        import ast

        _, tree = _app_ast()
        page = _fn_named(tree, "page_cad_twin")
        assert page is not None

        wanted = (
            "render_cad_part_card",
            "_render_cad_viewer_section",
            "render_cad_region_note",
            "render_cad_sensor_grid",
            "_render_cad_source_panel",
        )
        seen: dict[str, int] = {}
        for n in ast.walk(page):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in wanted:
                seen.setdefault(n.func.id, n.lineno)
        missing = [name for name in wanted if name not in seen]
        assert not missing, f"page_cad_twin must call {missing}"
        order = [seen[name] for name in wanted]
        assert order == sorted(order), f"CAD Twin layout out of order: {seen}"

        # The viewer itself is only rendered from the viewer section.
        callers = {
            fn.name
            for fn in ast.walk(tree)
            if isinstance(fn, ast.FunctionDef)
            and any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_render_cad_viewer"
                for n in ast.walk(fn)
            )
        }
        assert callers == {"_render_cad_viewer_section"}, callers

        strip = _fn_named(tree, "render_cad_part_card")
        grid = _fn_named(tree, "render_cad_sensor_grid")
        assert strip is not None and grid is not None
        strip_keys = {k for _, k in _literal_widget_keys(strip)}
        assert {"aps_asset_pick", "aps_load_cad_btn", "cad_open_anomaly_rul"} <= strip_keys
        strip_src = ast.dump(strip)
        assert "sensor_grid_rows" not in strip_src, "sensors belong below the viewer"
        assert "sensor_grid_rows" in ast.dump(grid)

        # 3–4 mapped sensor values per row, never one tall column.
        per_row = None
        for n in ast.walk(grid):
            if isinstance(n, ast.arg) and n.arg == "per_row":
                per_row = n
        assert per_row is not None, "render_cad_sensor_grid needs a per_row width"
        default = grid.args.kw_defaults[-1] if grid.args.kw_defaults else None
        assert isinstance(default, ast.Constant) and 3 <= int(default.value) <= 4, ast.dump(grid.args)

        # The admin controls live inside an expander so they stay collapsed.
        page_src = (ROOT / "app.py").read_text(encoding="utf-8")
        assert 'with st.expander("CAD source' in page_src
        assert "_render_cad_source_panel(pack, pack_id" in page_src

    def test_cad_twin_apptest_layout() -> None:
        """Render the CAD Twin page and assert the on-page order + honest copy."""
        import tempfile as _tf
        from pathlib import Path as _Path

        from streamlit.testing.v1 import AppTest

        from src.cad_part import NO_REGION_MATCH_MSG

        def _errs(at) -> list[str]:
            return [getattr(e, "message", str(e)) for e in at.exception]

        def _texts(items) -> list[str]:
            out: list[str] = []
            for item in items:
                out.append(
                    str(getattr(item, "value", None) or getattr(item, "body", None) or "")
                )
            return out

        prev = {k: os.environ.get(k) for k in ("APS_CLIENT_ID", "APS_CLIENT_SECRET", "PDM_CAD_URNS_PATH")}
        urns_dir = _tf.TemporaryDirectory()
        os.environ["PDM_CAD_URNS_PATH"] = str(_Path(urns_dir.name) / "cad_urns.json")
        os.environ["APS_CLIENT_ID"] = "LayoutTestClientId"
        os.environ["APS_CLIENT_SECRET"] = "layout-test-secret-not-real"
        try:
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
            at.run()
            # A High-risk asset so the region-tint copy is exercised.
            at.session_state["live_asset_states"] = [
                {"machine_id": "UAV-03", "risk_level": "High", "predicted_rul_days": 3},
                {"machine_id": "UAV-07", "risk_level": "Low", "predicted_rul_days": 28},
            ]
            next(r for r in at.radio if "Pipeline" in (r.label or "")).set_value(
                "CAD Twin (APS)"
            ).run()
            assert not _errs(at), "CAD Twin render: " + "; ".join(_errs(at))
            assert at.session_state["aps_asset_pick"] == "UAV-03"

            md = _texts(at.markdown)
            part_at = next(i for i, t in enumerate(md) if t.startswith("**Part**"))
            risk_at = next(i for i, t in enumerate(md) if t.startswith("**Risk**"))
            sensors_at = next(i for i, t in enumerate(md) if "Mapped sensors" in t)
            source_at = next(i for i, t in enumerate(md) if "Choose CAD file" in t)
            assert part_at < risk_at < sensors_at < source_at, md
            assert ">High<" in md[risk_at], md[risk_at]
            header = next(t for t in md if 'class="sub-header"' in t)
            assert "default dark gray" in header and "red" in header and "orange" in header
            assert "whole model tints" not in header

            # Asset picker sits in the strip; Load CAD is disabled with no URN.
            assert any((s.label or "") == "Asset" for s in at.selectbox)
            load = next(b for b in at.button if b.label == "Load CAD model")
            assert getattr(load, "disabled", False) is True
            assert len([b for b in at.button if b.label == "Open Anomaly & RUL"]) == 1

            # No report from the viewer yet: explain the rule, claim nothing.
            captions = " ".join(_texts(at.caption))
            assert "turn red" in captions and "default dark gray" in captions, captions
            assert "Name needles" in captions and "exhaust" in captions
            warnings = " ".join(_texts(at.warning))
            assert "Translate to SVF" in warnings, warnings

            # Paste a URN from our own bucket: Load enables, nothing auto-loads
            # (no saved / env URN), so no Autodesk call happens in this test.
            own = "dXJuOmFkc2sub2JqZWN0czpvcy5vYmplY3Q6cGRtLWFwcC1jYWQvcm90YXguc3RlcA"
            next(t for t in at.text_input if "URN" in (t.label or "")).set_value(own).run()
            assert not _errs(at), "URN paste: " + "; ".join(_errs(at))
            assert at.session_state["aps_urn"] == own
            load2 = next(b for b in at.button if b.label == "Load CAD model")
            assert getattr(load2, "disabled", False) is False
            infos = " ".join(_texts(at.info))
            assert "Load CAD model" in infos, infos

            # A viewer region report with zero matches shows the honest line, not
            # a claim that the engine went red.
            at.session_state["cad_region_report"] = {
                "risk": "High",
                "matched": 0,
                "nodes": 812,
                "names": [],
                "source": "none",
                "message": NO_REGION_MATCH_MSG,
            }
            at.run()
            assert not _errs(at), "region report: " + "; ".join(_errs(at))
            blank = " ".join(_texts(at.warning))
            assert "No CAD region matched" in blank, blank
            assert "Solid1" in blank and "whole engine" in blank

            # A report with matches names them instead.
            at.session_state["cad_region_report"] = {
                "risk": "High",
                "matched": 3,
                "nodes": 812,
                "names": ["Exhaust manifold", "Cylinder Head 1", "Solid-1"],
                "source": "tree",
                "message": "",
            }
            at.run()
            assert not _errs(at), "matched report: " + "; ".join(_errs(at))
            hit = " ".join(_texts(at.success))
            assert "3" in hit and "Exhaust manifold" in hit, hit
            assert "No CAD region matched" not in " ".join(_texts(at.warning))
        finally:
            urns_dir.cleanup()
            for key, val in prev.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    def test_upload_load_buttons_apptest() -> None:
        """Clicking Load Plant / Load pack demo / CSV upload must not raise StreamlitAPIException."""
        from streamlit.testing.v1 import AppTest

        from src.industry_packs import DEFAULT_PACK_ID, PACK_ORDER

        def _errs(at) -> list[str]:
            return [getattr(e, "message", str(e)) for e in at.exception]

        app_path = str(ROOT / "app.py")
        at = AppTest.from_file(app_path, default_timeout=45)
        at.run()
        assert not _errs(at), "initial render: " + "; ".join(_errs(at))
        assert at.session_state["industry_pack"] == DEFAULT_PACK_ID

        next(b for b in at.button if b.label == "Load Plant sample (default)").click().run()
        assert not _errs(at), "Load Plant sample: " + "; ".join(_errs(at))
        assert at.session_state["data_loaded"] is True
        assert at.session_state["raw_df"] is not None
        assert len(at.session_state["raw_df"]) > 10
        assert at.session_state["industry_pack"] == DEFAULT_PACK_ID

        demo_id = next(pid for pid in PACK_ORDER if pid != DEFAULT_PACK_ID)
        at2 = AppTest.from_file(app_path, default_timeout=45)
        at2.run()
        next(b for b in at2.button if b.label == "Load pack demo").click().run()
        assert not _errs(at2), "Load pack demo: " + "; ".join(_errs(at2))
        assert at2.session_state["data_loaded"] is True
        assert at2.session_state["industry_pack"] == demo_id

        at3 = AppTest.from_file(app_path, default_timeout=45)
        at3.run()
        next(b for b in at3.button if b.label == "Suggest pack from columns").click().run()
        assert not _errs(at3), "Suggest pack: " + "; ".join(_errs(at3))

        prev_id, prev_sec = os.environ.get("APS_CLIENT_ID"), os.environ.get("APS_CLIENT_SECRET")
        prev_urns = os.environ.get("PDM_CAD_URNS_PATH")
        import tempfile as _tf
        from pathlib import Path as _Path

        _urns_dir = _tf.TemporaryDirectory()
        os.environ["PDM_CAD_URNS_PATH"] = str(_Path(_urns_dir.name) / "cad_urns.json")
        os.environ["APS_CLIENT_ID"] = "UiTestClientId"
        os.environ["APS_CLIENT_SECRET"] = "ui-test-secret-not-real"
        try:
            at_cad = AppTest.from_file(app_path, default_timeout=45)
            at_cad.run()
            assert not _errs(at_cad), "CAD page initial: " + "; ".join(_errs(at_cad))
            radio = next(r for r in at_cad.radio if "Pipeline" in (r.label or ""))
            radio.set_value("CAD Twin (APS)").run()
            assert not _errs(at_cad), "CAD Twin nav: " + "; ".join(_errs(at_cad))
            assert any("CAD" in (u.label or "") for u in at_cad.file_uploader), [
                u.label for u in at_cad.file_uploader
            ]
            from src.aps_viewer import MAX_CAD_UPLOAD_MB

            cad_help = " ".join(str(getattr(u, "help", "") or "") for u in at_cad.file_uploader)
            cad_caps = ""
            try:
                cad_caps = " ".join(
                    str(getattr(c, "value", "") or "") for c in at_cad.caption
                )
            except Exception:
                pass
            cad_md = ""
            try:
                cad_md = " ".join(
                    str(getattr(m, "body", None) or getattr(m, "value", "") or "")
                    for m in at_cad.markdown
                )
            except Exception:
                pass
            cad_copy = f"{cad_help} {cad_caps} {cad_md}"
            assert str(MAX_CAD_UPLOAD_MB) in cad_copy, cad_copy[:500]
            assert "zip" in cad_copy.lower(), cad_copy[:500]
            assert any(b.label == "Translate to SVF / get URN" for b in at_cad.button)
            assert any(b.label == "Save URN for this pack" for b in at_cad.button)
            assert any(b.label == "Delete saved URN" for b in at_cad.button)
            open_btns = [b for b in at_cad.button if b.label == "Open Anomaly & RUL"]
            assert len(open_btns) == 1, [b.label for b in at_cad.button]
            assert any("URN" in (t.label or "") for t in at_cad.text_input)
            next(b for b in at_cad.button if b.label == "Translate to SVF / get URN").click().run()
            assert not _errs(at_cad), "Translate with no file: " + "; ".join(_errs(at_cad))
            next(b for b in at_cad.button if b.label == "Save URN for this pack").click().run()
            assert not _errs(at_cad), "Save with empty URN: " + "; ".join(_errs(at_cad))
            next(b for b in at_cad.button if b.label == "Delete saved URN").click().run()
            assert not _errs(at_cad), "Delete saved URN: " + "; ".join(_errs(at_cad))
            next(b for b in at_cad.button if b.label == "Open Anomaly & RUL").click().run()
            assert not _errs(at_cad), "Open Anomaly & RUL: " + "; ".join(_errs(at_cad))
            assert at_cad.session_state["pipeline_page"] == "4. Anomaly & RUL"
            from src.aps_viewer import saved_urn_for_pack as _saved_pack

            pid = "plant_rotating"
            try:
                pid = str(at_cad.session_state["industry_pack"])
            except Exception:
                pass
            assert _saved_pack(pid) == ""
        finally:
            _urns_dir.cleanup()
            if prev_urns is None:
                os.environ.pop("PDM_CAD_URNS_PATH", None)
            else:
                os.environ["PDM_CAD_URNS_PATH"] = prev_urns
            if prev_id is None:
                os.environ.pop("APS_CLIENT_ID", None)
            else:
                os.environ["APS_CLIENT_ID"] = prev_id
            if prev_sec is None:
                os.environ.pop("APS_CLIENT_SECRET", None)
            else:
                os.environ["APS_CLIENT_SECRET"] = prev_sec

        csv_bytes = (ROOT / "sample_data" / "sensor_readings.csv").read_bytes()
        at4 = AppTest.from_file(app_path, default_timeout=45)
        at4.run()
        uploader = next(u for u in at4.file_uploader if "sensor" in (u.label or "").lower())
        uploader.set_value([("sensor_readings.csv", csv_bytes, "text/csv")]).run()
        assert not _errs(at4), "CSV upload: " + "; ".join(_errs(at4))
        assert at4.session_state["data_loaded"] is True
        assert at4.session_state["raw_df"] is not None

    check("python39_imports", test_python39_annotations)
    check("gemini_remap", test_gemini_remap)
    check("map_clean", test_map_clean)
    check("anomaly_rul_insights", test_anomaly_rul_insights)
    check("industrial_brief_tiny_frame", test_industrial_brief_tiny_frame)
    check("url_ingest_presets", test_url_ingest_presets)
    check("twin3d", test_twin3d)
    check("live_connect", test_live_connect)
    check("live_sources", test_live_sources)
    check("polars_engine", test_polars_engine)
    check("aps_viewer", test_aps_viewer)
    check("cad_part_click", test_cad_part_click)
    check("cad_node_name_sanitizer", test_cad_node_name_sanitizer)
    check("cad_region_tint", test_cad_region_tint)
    check("cad_sensor_grid", test_cad_sensor_grid)
    check("aps_translate_mocked", test_aps_translate_mocked)
    check("aps_zip_root_filename", test_aps_zip_root_filename)
    check("charts_layout", test_charts)
    check("industry_packs", test_industry_packs)
    check("pack_kpis_every_function", test_pack_kpis_every_function)
    check("pack_samples_pipeline", test_pack_samples_pipeline)
    check("pack_mapping_does_not_steal_core", test_pack_mapping_does_not_steal_core)
    check("email_honesty", test_email_honesty)
    check("dashboard_composer", test_dashboard_composer)
    check("joins_step", test_joins_step)
    check("streamlit_widget_keys_unique", test_streamlit_widget_keys_unique)
    check("no_widget_key_writes_after_instantiate", test_no_widget_key_writes_after_instantiate)
    check("load_pack_sample_queues_pending", test_load_pack_sample_queues_pending)
    check("cad_twin_layout_order", test_cad_twin_layout_order)
    check("cad_twin_apptest_layout", test_cad_twin_apptest_layout)
    check("upload_load_buttons_apptest", test_upload_load_buttons_apptest)

    if errors:
        print(f"\n{len(errors)} FAIL")
        return 1
    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
