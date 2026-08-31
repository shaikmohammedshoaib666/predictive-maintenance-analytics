"""
Predictive Maintenance Analytics — reliability add-on.

Pipeline: Upload → Clean → Map sensors → Anomaly (Isolation Forest) → RUL/risk → charts → insights.
Not a generic analytics OS (Forge) and not an OEE cockpit (Pulse).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Cache dir for URL / cloud ingest (DuckDB reads large remote files from disk).
UPLOAD_DIR = PROJECT_ROOT / "output" / "uploads"

import config
from src.dashboard_builder import (
    export_dashboard_html,
    get_graph_folder,
    render_dashboard_preview,
    save_graph_to_folder,
)
from src.data_cleaner import clean_and_quality
from src.data_insights import analyze_columns, format_insights_markdown
from src.data_integration import JOIN_TYPES, join_many, join_two, load_tabular_file, suggest_join_keys
from src.dwdm_sql import DWDM_CONCEPTS, apply_dwdm_transforms, default_sql_examples, run_sql
from src.email_report import generate_email_body, send_email
from src.live_connect import (
    append_to_buffer,
    compute_live_status,
    default_machines,
    poll_url_increment,
    simulate_batch,
)
from src.url_ingest import (
    build_preset_sql,
    default_ingest_sql,
    friendly_source_label,
    list_ingest_presets,
    load_from_url,
)
from src.graphs import GRAPH_TYPES, PRIMARY_CHARTS
from src.graphs.layout import style_bar_figure
from src.insights_engine import (
    CHAT_SCOPE_CAPTION,
    NO_KEY_MESSAGE,
    ask_with_index,
    attach_rul_columns,
    build_industrial_brief,
    chart_business_insight,
    generate_business_insights,
    make_insight_index,
    get_gemini_api_key,
    get_gemini_model,
    mask_key,
    persist_session_gemini_key,
    polish_brief_with_gemini,
    test_gemini_connection,
)
from src.ml.anomaly_detector import AnomalyDetector
from src.ml.optuna_tuner import tune_anomaly_contamination, tune_rul_model
from src.ml.rul_predictor import RULPredictor
from src.twin3d import RISK_COLORS, asset_states_from_predictions, build_twin_html
from src.quality_checks import QUALITY_STAGE_COUNT
from src.sensor_map import CANONICAL_FIELDS, apply_mapping, mapping_status, suggest_mapping
from src.spark_clean import clean_with_spark, spark_available

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.25rem; }
    .sub-header { color: #6c757d; font-size: 1rem; margin-bottom: 1.5rem; }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low { color: #27ae60; font-weight: bold; }
    div[data-testid="stSidebar"] { background-color: #f8f9fa; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    defaults = {
        "raw_df": None,
        "cleaned_df": None,
        "cleaning_summary": None,
        "quality_report": None,
        "insights": None,
        "business_insights": [],
        "graph_folder": {},
        "anomaly_detector": None,
        "rul_predictor": None,
        "predictions": [],
        "anomaly_summary": {},
        "chat_history": [],
        "data_loaded": False,
        "uploaded_tables": {},
        "join_log": [],
        "optuna_rul": None,
        "optuna_anomaly": None,
        "insight_index": None,
        "sql_result": None,
        "sensor_mapping": {},
        "gemini_api_key_override": "",
        "last_gemini_error": "",
        "last_insight_error": "",
        "rul_used_synthetic": False,
        "cost_per_hour": 0.0,
        "cost_per_unit": 0.0,
        "hours_if_stop": 8.0,
        "asset_ranking": [],
        "inspect_list": [],
        "polished_brief": "",
        # URL / cloud ingest (DuckDB httpfs + SQL slice presets, ported from Forge v2)
        "url_ingest_source": "",
        "url_ingest_row_limit": 0,
        "url_ingest_force_cache": True,
        "url_ingest_meta": None,
        "url_ingest_mode": "limit",
        "url_ingest_sql": default_ingest_sql("predictive_maintenance"),
        "url_ingest_preset": "filter_machine_id",
        "url_ingest_preset_params": {},
        "maintenance_table_attached": None,
        # Live Connect (Layer 4)
        "live_running": False,
        "live_tick": 0,
        "live_buffer": None,
        "live_source": "sim",
        "live_cfg": {},
        "live_asset_states": [],
        "live_sensor_view": "temperature",
        # Cleaning engine (Layer 5)
        "clean_engine": "pandas",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_active_df() -> pd.DataFrame | None:
    return st.session_state.get("cleaned_df")


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def get_axis_options(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    if "timestamp" in cols:
        return ["timestamp"] + [c for c in cols if c != "timestamp"]
    return cols


def _show_gemini_error(message: str | None) -> None:
    if not message:
        return
    st.error(message)
    st.caption("Offline / rule-based answer is still available below.")


def render_gemini_sidebar() -> None:
    st.sidebar.markdown("### Gemini")
    key = get_gemini_api_key()
    model = get_gemini_model()
    st.sidebar.caption(f"Status: **{mask_key(key)}** · `{model}`")
    st.sidebar.caption("Env: `GEMINI_API_KEY`. Retired aliases remap to `gemini-3.6-flash`.")
    st.sidebar.text_input("Paste Gemini API key", type="password", key="gemini_key_input")
    c1, c2 = st.sidebar.columns(2)
    save = c1.button("Save key", use_container_width=True)
    test = c2.button("Test Gemini", use_container_width=True)
    if save:
        new_key = str(st.session_state.get("gemini_key_input") or "").strip()
        if not new_key:
            st.sidebar.warning("Paste a non-empty key.")
        else:
            persist_session_gemini_key(new_key)
            st.sidebar.success("Key saved for this session.")
            st.rerun()
    if test:
        with st.spinner("Calling Gemini…"):
            probe = test_gemini_connection()
        if probe.get("ok"):
            st.session_state.last_gemini_error = ""
            st.sidebar.success(f"Connected · `{probe.get('model')}`")
            if probe.get("text"):
                st.sidebar.caption(f"Reply: {probe['text']}")
        else:
            err = probe.get("error") or "Gemini test failed."
            st.session_state.last_gemini_error = err
            st.sidebar.caption(f"Model tried: `{probe.get('model')}`")
    if st.session_state.get("last_gemini_error"):
        st.sidebar.error(st.session_state.last_gemini_error)


def register_table(name: str, df: pd.DataFrame):
    tables = dict(st.session_state.uploaded_tables or {})
    tables[name] = df
    st.session_state.uploaded_tables = tables


def load_sample_data():
    if config.SAMPLE_DATA_PATH.exists():
        df = pd.read_csv(config.SAMPLE_DATA_PATH, parse_dates=["timestamp"])
        st.session_state.raw_df = df
        st.session_state.data_loaded = True
        register_table("sensors", df)
        # Optional companion tables for join demos
        maint = PROJECT_ROOT / "sample_data" / "maintenance_logs.csv"
        costs = PROJECT_ROOT / "sample_data" / "machine_costs.csv"
        if maint.exists():
            register_table("maintenance", pd.read_csv(maint))
        if costs.exists():
            register_table("costs", pd.read_csv(costs))
        return df
    st.error("Sample data not found. Run: python generate_sample_data.py")
    return None


# PdM app domain — drives which URL SQL slice presets are offered.
PDM_DOMAIN = "predictive_maintenance"


# ── URL / cloud ingest tab (DuckDB httpfs + SQL slice presets, ported from Forge v2) ──
def _ingest_from_url_tab():
    st.caption(
        "Paste a **direct HTTPS CSV/Parquet** link, a **Google Drive** share URL "
        "(Anyone with the link), or a **Kaggle** dataset page / `kaggle://owner/dataset/file.csv`. "
        "Kaggle needs `KAGGLE_USERNAME` + `KAGGLE_KEY` secrets. DuckDB streams the slice."
    )
    url_val = st.text_input(
        "Cloud data URL",
        value=st.session_state.get("url_ingest_source", ""),
        placeholder="https://drive.google.com/file/d/…/view  or  https://www.kaggle.com/datasets/…",
        key="upload_url_input",
    )
    ingest_mode = st.radio(
        "Ingest mode",
        ["Row limit (simple)", "SQL slice (DuckDB)"],
        index=1 if st.session_state.get("url_ingest_mode") == "sql" else 0,
        horizontal=True,
        key="upload_url_ingest_mode",
        help="For 10M+ row sensor logs: use a SQL slice to filter/limit before Clean → Map → ML.",
    )
    st.session_state.url_ingest_mode = "sql" if ingest_mode.startswith("SQL") else "limit"

    row_limit = 0
    sql_query: Optional[str] = None
    if st.session_state.url_ingest_mode == "limit":
        row_limit = st.number_input(
            "Row limit (0 = all rows — cap on free hosts for multi-GB files)",
            min_value=0,
            value=int(st.session_state.get("url_ingest_row_limit") or 0),
            step=1000,
            key="upload_url_row_limit",
        )
    else:
        presets = list_ingest_presets(domain=PDM_DOMAIN)
        preset_ids = [p["id"] for p in presets]
        preset_labels = {p["id"]: p["label"] for p in presets}
        cur_preset = st.session_state.get("url_ingest_preset") or preset_ids[0]
        if cur_preset not in preset_ids:
            cur_preset = preset_ids[0]
        preset_pick = st.selectbox(
            "SQL slice preset (plant / PdM templates)",
            preset_ids,
            index=preset_ids.index(cur_preset),
            format_func=lambda pid: preset_labels.get(pid, pid),
            key="upload_url_sql_preset",
        )
        picked = next(p for p in presets if p["id"] == preset_pick)
        st.caption(picked.get("description") or "")
        preset_params: dict[str, Any] = dict(st.session_state.get("url_ingest_preset_params") or {})
        params_spec = picked.get("params") or []
        param_cols = st.columns(min(3, max(1, len(params_spec))))
        for idx, (pname, plabel, pdefault, pkind) in enumerate(params_spec):
            with param_cols[idx % len(param_cols)]:
                if pkind == "int":
                    preset_params[pname] = st.number_input(
                        plabel,
                        min_value=0,
                        value=int(preset_params.get(pname, pdefault) or pdefault),
                        step=max(1, int(pdefault) // 10) if int(pdefault) > 10 else 1,
                        key=f"upload_preset_{preset_pick}_{pname}",
                    )
                elif pkind == "float":
                    preset_params[pname] = st.number_input(
                        plabel,
                        min_value=0.1,
                        max_value=100.0,
                        value=float(preset_params.get(pname, pdefault) or pdefault),
                        step=0.5,
                        key=f"upload_preset_{preset_pick}_{pname}",
                    )
                else:
                    preset_params[pname] = st.text_input(
                        plabel,
                        value=str(preset_params.get(pname, pdefault) or pdefault),
                        key=f"upload_preset_{preset_pick}_{pname}",
                    )
        if st.button("Apply preset to SQL", key="upload_apply_sql_preset"):
            try:
                st.session_state.url_ingest_sql = build_preset_sql(preset_pick, preset_params)
                st.session_state.url_ingest_preset = preset_pick
                st.session_state.url_ingest_preset_params = preset_params
                st.success(f"Applied **{preset_labels[preset_pick]}** template.")
            except Exception as exc:
                st.error(str(exc))
        sql_query = st.text_area(
            "DuckDB SQL (use `{source}` for the resolved file path/URL)",
            value=st.session_state.get("url_ingest_sql") or default_ingest_sql(PDM_DOMAIN),
            height=160,
            key="upload_url_sql",
        )
    force_cache = st.checkbox(
        "Always download to disk first (recommended for Google Drive / files > 100 MB)",
        value=bool(st.session_state.get("url_ingest_force_cache", True)),
        key="upload_url_force_cache",
    )
    if st.button("Load from URL", key="upload_url_load", type="primary"):
        if not (url_val or "").strip():
            st.warning("Paste a URL first.")
        else:
            try:
                with st.spinner("Resolving link and loading via DuckDB…"):
                    limit = int(row_limit) if row_limit and row_limit > 0 else None
                    sql = (sql_query or "").strip() if st.session_state.url_ingest_mode == "sql" else None
                    loaded, meta = load_from_url(
                        url_val.strip(),
                        cache_dir=UPLOAD_DIR,
                        row_limit=limit,
                        force_cache=force_cache or bool(sql),
                        sql_query=sql,
                    )
                label = friendly_source_label(meta)
                st.session_state.raw_df = loaded
                st.session_state.data_loaded = True
                register_table(label.replace(":", "_") or "url_data", loaded)
                st.session_state.url_ingest_source = url_val.strip()
                st.session_state.url_ingest_row_limit = int(row_limit or 0)
                st.session_state.url_ingest_force_cache = force_cache
                if sql_query is not None:
                    st.session_state.url_ingest_sql = sql_query
                st.session_state.url_ingest_meta = meta
                st.session_state.cleaned_df = None  # new source invalidates prior cleaned table
                eng = meta.get("engine", "duckdb")
                cached = meta.get("cached_path")
                extra = f" · cached `{Path(cached).name}`" if cached else ""
                st.success(
                    f"Loaded **{label}** via **{eng}** — {len(loaded):,} rows × {loaded.shape[1]} cols{extra}"
                )
                if meta.get("stream_error"):
                    st.caption(f"Stream read fell back to disk cache: {meta['stream_error'][:120]}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.get("url_ingest_meta") and st.session_state.raw_df is not None:
        meta = st.session_state.url_ingest_meta
        st.caption(
            f"Current URL source · kind `{meta.get('kind')}` · engine `{meta.get('engine')}` · "
            f"{meta.get('rows', 0):,} rows"
        )


def _render_maintenance_attach():
    with st.expander("Attach maintenance table (optional — join on Joins lab)"):
        st.caption(
            "Upload a work-order / PM schedule CSV (columns like `machine_id`, `maintenance_date`, "
            "`work_order`). It registers as `maintenance` for INNER/LEFT joins with sensor data."
        )
        maint_file = st.file_uploader(
            "Maintenance / work-order CSV",
            type=["csv", "tsv", "xlsx", "xls"],
            key="upload_maintenance_table",
        )
        if maint_file is not None:
            try:
                maint_df = load_tabular_file(maint_file)
                register_table("maintenance", maint_df)
                st.session_state.maintenance_table_attached = maint_file.name
                st.success(
                    f"Registered **maintenance** — {len(maint_df):,} rows × {maint_df.shape[1]} cols. "
                    "Open **Joins (lab)** to merge on `machine_id`."
                )
            except Exception as exc:
                st.error(str(exc))
        elif st.session_state.get("maintenance_table_attached"):
            st.write(
                f"Attached: **{st.session_state.maintenance_table_attached}** (see Joins lab to merge)."
            )


def _render_quality_subreports(report: dict) -> None:
    """Human-readable GE / ydata / Cleanlab / PCA / association / OPC sub-reports."""
    ge = report.get("ge") or {}
    yd = report.get("ydata") or {}
    cl = report.get("cleanlab") or {}
    pca = report.get("pca") or {}
    assoc = report.get("association") or {}
    flags = report.get("domain_flags")

    with st.expander("Great Expectations — column expectations"):
        st.caption(
            f"{ge.get('passed', 0)}/{ge.get('total', 0)} expectations passed · "
            f"library available: {ge.get('available')}"
        )
        results = ge.get("results") or []
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)

    with st.expander("ydata / Cleanlab / PCA drift / Association rules"):
        st.markdown("**ydata-profiling**")
        if yd.get("ok"):
            st.success(f"Profile OK — {yd.get('variables', 0)} variables, {yd.get('alerts', 0)} alerts.")
        else:
            st.warning(yd.get("error") or "ydata-profiling not installed / did not complete.")
        if yd.get("high_cardinality"):
            st.caption("High-cardinality columns: " + ", ".join(yd["high_cardinality"]))

        st.markdown("**Cleanlab — dirty labels / outliers**")
        if cl.get("ok"):
            if cl.get("skipped"):
                st.info(f"Skipped: {cl['skipped']}")
            else:
                st.success(f"Outlier issues flagged: {cl.get('outlier_issues', 0)}")
        else:
            st.warning(cl.get("error") or "Cleanlab not installed / check failed.")
        for line in cl.get("dirty_label_flags") or []:
            st.warning(line)

        st.markdown("**PCA concept drift**")
        if pca.get("ok"):
            drift = "yes" if pca.get("concept_drift") else "no"
            st.write(
                f"Early var={pca.get('pca_var_early')} · Late var={pca.get('pca_var_late')} · "
                f"drift={pca.get('drift_score')} · concept drift={drift}"
            )
        else:
            st.caption("PCA drift skipped (need ≥30 rows and 2 numeric columns).")

        st.markdown("**Association rule mining**")
        if assoc.get("ok"):
            if assoc.get("suspicious_rules"):
                st.warning(f"Suspicious HIGH-sensor co-occurrences: {len(assoc['suspicious_rules'])}")
                for rule in assoc["suspicious_rules"][:5]:
                    st.write(f"- {rule.get('rule')} (conf={rule.get('confidence')})")
            else:
                st.success(f"Mined {assoc.get('rules_found', 0)} rules — none flagged suspicious.")
        else:
            st.caption(assoc.get("error") or assoc.get("skipped") or "Association mining skipped.")

    if flags:
        with st.expander("Domain OPC / physics-rule violations", expanded=True):
            for flag in flags:
                st.error(flag)


# ── Upload & Clean (19-stage quality) ─────────────────────────────────────────
def page_upload_clean():
    st.markdown('<p class="main-header">1. Upload & Clean</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{config.TAGLINE}</p>', unsafe_allow_html=True)
    st.caption(
        f"Industrial ETL + {QUALITY_STAGE_COUNT}-stage quality checks on sensor CSVs. "
        "Load from file, cloud URL, or Kaggle. Next: map columns, then anomaly / RUL."
    )

    tab_file, tab_url = st.tabs(["File upload", "From URL (cloud / DuckDB)"])
    with tab_file:
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_files = st.file_uploader(
                "Upload sensor / ops CSVs (multi-select OK)",
                type=["csv", "tsv", "xlsx", "json"],
                accept_multiple_files=True,
            )
        with col2:
            if st.button("Load Sample Data", use_container_width=True):
                load_sample_data()
                st.success("Sample sensors (+ maintenance/costs if present) loaded!")
                st.rerun()

        if uploaded_files:
            for uf in uploaded_files:
                try:
                    df = load_tabular_file(uf)
                    stem = Path(uf.name).stem.replace(" ", "_")
                    register_table(stem, df)
                    st.session_state.raw_df = df
                    st.session_state.data_loaded = True
                    st.success(f"Loaded `{uf.name}` → table `{stem}` ({len(df):,} rows)")
                except Exception as exc:
                    st.error(f"Failed {uf.name}: {exc}")

    with tab_url:
        _ingest_from_url_tab()

    _render_maintenance_attach()

    if st.session_state.uploaded_tables:
        st.caption("Registered tables: " + ", ".join(st.session_state.uploaded_tables.keys()))

    if st.session_state.raw_df is not None:
        st.subheader("Raw Data Preview")
        st.dataframe(st.session_state.raw_df.head(10), use_container_width=True)

        engines = ["pandas"]
        spark_ok, spark_msg = spark_available()
        if spark_ok:
            engines.append("pyspark")
        eng_col1, eng_col2 = st.columns([1, 2])
        with eng_col1:
            clean_engine = st.selectbox(
                "Cleaning engine",
                engines,
                index=engines.index(st.session_state.get("clean_engine", "pandas"))
                if st.session_state.get("clean_engine", "pandas") in engines
                else 0,
                help="PySpark (distributed) appears when pyspark + a JVM are installed.",
            )
        with eng_col2:
            st.caption(
                f"PySpark: **available** — {spark_msg}"
                if spark_ok
                else f"PySpark: not available ({spark_msg}) — pandas engine is used."
            )
        st.session_state.clean_engine = clean_engine

        if st.button("Run industrial clean + 19 quality checks", type="primary"):
            with st.spinner(f"Cleaning with {clean_engine} + running 19 quality stages..."):
                raw = st.session_state.raw_df
                spark_log: list[str] = []
                source_df = raw
                if clean_engine == "pyspark":
                    try:
                        source_df, spark_log = clean_with_spark(raw)
                    except Exception as exc:
                        st.warning(f"PySpark clean failed — falling back to pandas: {exc}")
                        source_df = raw
                cleaned, summary, report = clean_and_quality(source_df, run_quality=True)
                if spark_log:
                    summary["actions"] = spark_log + summary.get("actions", [])
                insights = analyze_columns(cleaned)
                st.session_state.cleaned_df = cleaned
                st.session_state.cleaning_summary = summary
                st.session_state.quality_report = report
                st.session_state.insights = insights
                register_table("cleaned", cleaned)
            st.success(f"Cleaned ({clean_engine}): {summary['rows_before']} → {summary['rows_after']} rows")
            if summary["actions"]:
                with st.expander("ETL / DWDM actions"):
                    for action in summary["actions"]:
                        st.write(f"- {action}")

        if st.session_state.cleaned_df is not None:
            st.subheader("Cleaned Data Preview")
            st.dataframe(st.session_state.cleaned_df.head(10), use_container_width=True)
            st.download_button(
                "Download Cleaned CSV",
                st.session_state.cleaned_df.to_csv(index=False),
                file_name=f"cleaned_sensor_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

            if st.session_state.quality_report:
                st.subheader(f"{QUALITY_STAGE_COUNT}-Stage Quality Report")
                checks_df = pd.DataFrame(st.session_state.quality_report["checks"])
                st.dataframe(checks_df, use_container_width=True)
                status_counts = checks_df["status"].value_counts().to_dict()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("PASS", status_counts.get("PASS", 0))
                c2.metric("WARN", status_counts.get("WARN", 0))
                c3.metric("FAIL", status_counts.get("FAIL", 0))
                c4.metric("INFO", status_counts.get("INFO", 0))
                st.markdown("#### Quality sub-reports")
                _render_quality_subreports(st.session_state.quality_report)

            if st.session_state.insights:
                st.subheader("Data Insights")
                st.markdown(format_insights_markdown(st.session_state.insights))


# ── Map sensors ───────────────────────────────────────────────────────────────
def page_map_sensors():
    st.markdown('<p class="main-header">2. Map sensors</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Point messy CSV headers at timestamp, asset, and sensor columns. '
        "RUL labels are optional.</p>",
        unsafe_allow_html=True,
    )
    st.caption(config.RUL_HONESTY_CAPTION)

    df = get_active_df()
    if df is None:
        df = st.session_state.get("raw_df")
    if df is None:
        st.warning("Upload (and preferably clean) a sensor CSV first.")
        return

    suggested = suggest_mapping(list(df.columns))
    saved = st.session_state.get("sensor_mapping") or {}
    status = mapping_status(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sensors mapped", status["sensor_count"])
    c2.metric("Timestamp", "yes" if status["has_timestamp"] else "no")
    c3.metric("Asset id", "yes" if status["has_machine"] else "no")
    c4.metric("RUL label", "yes" if status["has_rul_label"] else "no")
    if not status["has_rul_label"]:
        st.warning("No `failure_within_days` column — RUL will use a degradation proxy until you map a label.")

    cols = [""] + list(df.columns)
    mapping: dict[str, str | None] = {}
    st.subheader("Column map")
    for canonical, label in CANONICAL_FIELDS:
        default_src = saved.get(canonical) or suggested.get(canonical) or ""
        idx = cols.index(default_src) if default_src in cols else 0
        picked = st.selectbox(label, cols, index=idx, key=f"map_{canonical}")
        mapping[canonical] = picked or None

    if st.button("Apply mapping", type="primary"):
        mapped = apply_mapping(df, mapping)
        st.session_state.sensor_mapping = mapping
        st.session_state.cleaned_df = mapped
        register_table("cleaned", mapped)
        st.success("Mapped columns applied to the working table.")
        st.dataframe(mapped.head(8), use_container_width=True)
        st.rerun()

    st.caption("Suggested matches: " + ", ".join(f"{k}←{v}" for k, v in suggested.items() if v))


# ── Data Integration (SQL joins) ──────────────────────────────────────────────
def page_data_integration():
    st.markdown('<p class="main-header">Joins (lab)</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Optional: merge sensor + maintenance/cost tables. '
        "Not required for the core PdM pipeline.</p>",
        unsafe_allow_html=True,
    )

    tables = st.session_state.uploaded_tables or {}
    if len(tables) < 2:
        st.warning("Register at least 2 tables (multi-upload or Load Sample Data).")
        return

    names = list(tables.keys())
    st.write("Available tables:", ", ".join(f"`{n}` ({len(tables[n])} rows)" for n in names))

    mode = st.radio("Join mode", ["Two tables", "Chain 3+ tables"], horizontal=True)

    if mode == "Two tables":
        c1, c2, c3 = st.columns(3)
        with c1:
            left_name = st.selectbox("Left table", names, key="join_left")
        with c2:
            right_name = st.selectbox("Right table", [n for n in names if n != left_name], key="join_right")
        with c3:
            how = st.selectbox("Join type", list(JOIN_TYPES.keys()), format_func=lambda k: f"{k.upper()} — {JOIN_TYPES[k].split('—')[0].strip()}")

        left, right = tables[left_name], tables[right_name]
        suggested = suggest_join_keys(left, right)
        key_mode = st.radio("Keys", ["Common columns", "Different column names"], horizontal=True)
        if key_mode == "Common columns":
            on = st.multiselect("Join on", suggested or list(set(left.columns) & set(right.columns)), default=suggested[:1])
            left_on = right_on = None
        else:
            on = None
            left_on = st.selectbox("Left key", list(left.columns))
            right_on = st.selectbox("Right key", list(right.columns))

        if st.button("Run join", type="primary"):
            try:
                merged, meta = join_two(left, right, how=how, on=on or None, left_on=left_on, right_on=right_on)
                st.session_state.cleaned_df = merged
                st.session_state.raw_df = merged
                register_table("joined", merged)
                st.session_state.join_log = [meta]
                st.success(f"Joined → {meta['result_rows']:,} rows")
                st.json(meta)
                st.dataframe(merged.head(20), use_container_width=True)
            except Exception as exc:
                st.error(str(exc))
    else:
        if len(names) < 3:
            st.info("Need 3+ tables for a chain demo — upload more files or use sample maintenance/costs.")
        left_name = st.selectbox("Start (left)", names, key="chain_left")
        right1 = st.selectbox("Join #1 right", [n for n in names if n != left_name], key="chain_r1")
        how1 = st.selectbox("Join #1 type", list(JOIN_TYPES.keys()), key="how1")
        keys1 = st.multiselect(
            "Join #1 keys",
            suggest_join_keys(tables[left_name], tables[right1]),
            default=suggest_join_keys(tables[left_name], tables[right1])[:1],
            key="k1",
        )
        remaining = [n for n in names if n not in (left_name, right1)]
        right2 = st.selectbox("Join #2 right", remaining or names, key="chain_r2")
        how2 = st.selectbox("Join #2 type", list(JOIN_TYPES.keys()), key="how2")
        # keys for step 2 chosen after preview of common names with right2
        keys2_opts = list(tables[right2].columns)
        keys2 = st.multiselect("Join #2 keys (must exist on interim + right)", keys2_opts, default=[k for k in keys1 if k in keys2_opts][:1], key="k2")

        if st.button("Run join chain", type="primary"):
            try:
                steps = [
                    {"left": left_name, "right": right1, "how": how1, "on": keys1},
                    {"left": "_result", "right": right2, "how": how2, "on": keys2},
                ]
                merged, logs = join_many(tables, steps)
                st.session_state.cleaned_df = merged
                st.session_state.raw_df = merged
                register_table("joined", merged)
                st.session_state.join_log = logs
                st.success(f"Chain complete → {len(merged):,} rows")
                st.json(logs)
                st.dataframe(merged.head(20), use_container_width=True)
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.join_log:
        with st.expander("Last join log"):
            st.json(st.session_state.join_log)


# ── DWDM + SQL Lab ────────────────────────────────────────────────────────────
def page_dwdm_sql():
    st.markdown('<p class="main-header">DWDM Concepts & SQL Lab</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Optional warehouse / SQL lab over your sensor tables. '
        "Core PdM does not require this step.</p>",
        unsafe_allow_html=True,
    )

    st.subheader("DWDM concept map")
    st.dataframe(pd.DataFrame(DWDM_CONCEPTS), use_container_width=True)

    df = get_active_df()
    if df is None:
        st.warning("Clean or join data first.")
        return

    st.subheader("Apply DWDM transforms")
    nums = get_numeric_columns(df)
    c1, c2, c3 = st.columns(3)
    with c1:
        bin_cols = st.multiselect("Bin columns", nums, default=nums[:1])
    with c2:
        smooth_cols = st.multiselect("Smooth columns", nums, default=[c for c in nums if "temp" in c.lower() or "vib" in c.lower()][:1])
    with c3:
        norm_cols = st.multiselect("Z-normalize", nums, default=[])

    if st.button("Apply transforms"):
        out, log = apply_dwdm_transforms(df, bin_cols=bin_cols, smooth_cols=smooth_cols, normalize_cols=norm_cols)
        st.session_state.cleaned_df = out
        register_table("cleaned", out)
        st.success("; ".join(log) if log else "No changes")
        st.dataframe(out.head(10), use_container_width=True)

    st.subheader("SQL Lab (read-only)")
    tables = st.session_state.uploaded_tables or {}
    if "cleaned" not in tables and df is not None:
        register_table("cleaned", df)
        tables = st.session_state.uploaded_tables
    st.caption("Tables: " + ", ".join(tables.keys()))
    examples = default_sql_examples(list(tables.keys()))
    st.code("\n\n".join(examples), language="sql")
    query = st.text_area("SQL", value=examples[0], height=120)
    if st.button("Run SQL", type="primary"):
        try:
            result, engine = run_sql(query, tables)
            st.session_state.sql_result = result
            st.caption(f"Engine: {engine}")
            st.dataframe(result, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))


# ── Explore & Graphs ──────────────────────────────────────────────────────────
def page_explore_graphs():
    st.markdown('<p class="main-header">4. Charts</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">PdM charts: sensor over time, anomaly flags, risk by asset. '
        "Readable Plotly hover / margins (same idea as Forge, PdM metrics only).</p>",
        unsafe_allow_html=True,
    )

    df = get_active_df()
    if df is None:
        st.warning("Upload and clean data first.")
        return

    st.subheader("Chart Controls")
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    axis_options = get_axis_options(df)
    numeric_cols = get_numeric_columns(df)
    with ctrl1:
        x_axis = st.selectbox("X-Axis", axis_options, index=0)
    with ctrl2:
        y_metric = st.selectbox(
            "Sensor / metric",
            numeric_cols,
            index=numeric_cols.index("temperature") if "temperature" in numeric_cols else 0,
        )
    with ctrl3:
        machines = sorted(df["machine_id"].unique()) if "machine_id" in df.columns else []
        machine_filter = st.multiselect("Filter by Machine", machines, default=machines)

    st.info(chart_business_insight(df, x_axis, y_metric))

    def _make_chart(gkey: str):
        ginfo = GRAPH_TYPES[gkey]
        kwargs = {
            "df": df,
            "x_axis": x_axis,
            "y_metric": y_metric,
            "machine_filter": machine_filter or None,
        }
        if gkey in ("anomaly_scatter", "anomaly_flags") and st.session_state.anomaly_detector:
            kwargs["anomaly_detector"] = st.session_state.anomaly_detector
        if gkey == "risk_by_asset":
            kwargs["predictions"] = st.session_state.predictions
        fig = ginfo["create"](**kwargs)
        entry = save_graph_to_folder(
            gkey,
            fig,
            gkey,
            {"x_axis": x_axis, "y_metric": y_metric, "machines": machine_filter},
        )
        return fig, entry

    st.divider()
    st.subheader("Primary PdM charts")
    cols = st.columns(3)
    for col, gkey in zip(cols, PRIMARY_CHARTS.keys()):
        ginfo = PRIMARY_CHARTS[gkey]
        with col:
            st.markdown(f"### {ginfo['icon']} {ginfo['name']}")
            st.caption(ginfo["description"])
            if st.button(f"Generate {ginfo['name']}", key=f"gen_{gkey}", use_container_width=True):
                with st.spinner(f"Creating {ginfo['name']}..."):
                    fig, entry = _make_chart(gkey)
                st.success(f"Saved: {entry['id']}")
                st.plotly_chart(fig, use_container_width=True, key=f"primary_{gkey}")

    extras = [k for k in GRAPH_TYPES if k not in PRIMARY_CHARTS]
    with st.expander("More sensor views (optional)"):
        for gkey in extras:
            ginfo = GRAPH_TYPES[gkey]
            if st.button(f"{ginfo['icon']} {ginfo['name']}", key=f"gen_{gkey}"):
                with st.spinner(f"Creating {ginfo['name']}..."):
                    fig, entry = _make_chart(gkey)
                st.success(f"Saved: {entry['id']}")
                st.plotly_chart(fig, use_container_width=True, key=f"extra_{gkey}")

    folder = get_graph_folder()
    if folder:
        st.divider()
        st.subheader("Saved Graphs")
        for i, (g_id, entry) in enumerate(folder.items()):
            title = entry.get("title", g_id)
            with st.expander(f"{title} — {entry['created_at'][:19]}"):
                st.plotly_chart(entry["fig"], use_container_width=True, key=f"explore_{i}_{title}")
                if st.button(f"Remove {g_id}", key=f"rm_{g_id}"):
                    del st.session_state.graph_folder[g_id]
                    st.rerun()


# ── ML + Optuna ───────────────────────────────────────────────────────────────
def page_ml_predictions():
    st.markdown('<p class="main-header">3. Anomaly & RUL</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Isolation Forest anomalies, then Random Forest remaining useful life / risk.</p>',
        unsafe_allow_html=True,
    )
    st.caption(config.RUL_HONESTY_CAPTION)

    df = get_active_df()
    if df is None:
        st.warning("Upload, clean, and map sensors first.")
        return

    status = mapping_status(df)
    if status["sensor_count"] == 0:
        st.warning("No canonical sensor columns yet — open **Map sensors** or use sample data.")
    if not status["has_rul_label"]:
        st.warning("No failure label column. RUL will train on a synthetic degradation proxy.")

    trials = st.slider("Optuna trials (optional)", 5, 40, 15)

    c1, c2, c3 = st.columns(3)
    with c1:
        train = st.button("Detect anomalies + predict RUL", type="primary", use_container_width=True)
    with c2:
        tune_rul = st.button("Optuna tune RUL", use_container_width=True)
    with c3:
        tune_iso = st.button("Optuna tune anomalies", use_container_width=True)

    if train:
        with st.spinner("Training Isolation Forest + Random Forest RUL..."):
            try:
                cont = config.ANOMALY_CONTAMINATION
                if st.session_state.optuna_anomaly:
                    cont = st.session_state.optuna_anomaly.get("best_contamination", cont)
                detector = AnomalyDetector(contamination=cont)
                detector.fit(df)
                scored = detector.annotate(df)
                st.session_state.anomaly_detector = detector
                st.session_state.anomaly_summary = detector.summary(scored)
                predictor = RULPredictor()
                predictor.fit(scored)
                st.session_state.rul_predictor = predictor
                st.session_state.rul_used_synthetic = bool(predictor.used_synthetic_labels)
                st.session_state.predictions = predictor.predict_latest_per_machine(scored)
                scored = attach_rul_columns(scored, st.session_state.predictions)
                st.session_state.cleaned_df = scored
                register_table("cleaned", scored)
                brief = build_industrial_brief(
                    scored,
                    st.session_state.predictions,
                    st.session_state.anomaly_summary,
                    cost_per_hour=float(st.session_state.get("cost_per_hour") or 0),
                    cost_per_unit=float(st.session_state.get("cost_per_unit") or 0),
                    hours_if_stop=float(st.session_state.get("hours_if_stop") or 8),
                    used_synthetic=bool(predictor.used_synthetic_labels),
                )
                st.session_state.business_insights = brief["insights"]
                st.session_state.asset_ranking = brief["ranked"]
                st.session_state.inspect_list = brief["inspect"]
                st.session_state.insight_index = None
                st.success("Models trained. Isolation Forest scores and RUL are on the working table.")
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.get("rul_used_synthetic") or (
        st.session_state.rul_predictor and getattr(st.session_state.rul_predictor, "used_synthetic_labels", False)
    ):
        st.warning(
            "RUL used a **synthetic degradation proxy** (no real failure labels). "
            "Do not treat predicted days as a production forecast."
        )

    if tune_rul:
        with st.spinner(f"Optuna RUL ({trials} trials)..."):
            try:
                target = "failure_within_days" if "failure_within_days" in df.columns else None
                if not target:
                    st.error("Need `failure_within_days` (map a RUL label) for honest Optuna RUL.")
                else:
                    st.session_state.optuna_rul = tune_rul_model(df, target=target, n_trials=trials)
                    st.success("Optuna RUL complete")
            except Exception as exc:
                st.error(str(exc))

    if tune_iso:
        with st.spinner("Optuna anomaly contamination..."):
            try:
                st.session_state.optuna_anomaly = tune_anomaly_contamination(df, n_trials=min(12, trials))
                st.success("Optuna anomaly complete — click Detect anomalies + predict RUL to apply")
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.optuna_rul:
        st.subheader("Optuna RUL result")
        st.json(st.session_state.optuna_rul)
    if st.session_state.optuna_anomaly:
        st.subheader("Optuna anomaly result")
        st.json(st.session_state.optuna_anomaly)

    if st.session_state.predictions:
        st.subheader("Failure Predictions")
        for p in st.session_state.predictions:
            risk_class = f"risk-{p['risk_level'].lower()}"
            st.markdown(
                f"<p class='{risk_class}'>{p['message']} — Risk: {p['risk_level']}</p>",
                unsafe_allow_html=True,
            )
        if st.session_state.rul_predictor and st.session_state.rul_predictor.metrics:
            m = st.session_state.rul_predictor.metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE (days)", m.get("mae", "N/A"))
            c2.metric("RMSE (days)", m.get("rmse", "N/A"))
            c3.metric("R² Score", m.get("r2", "N/A"))
            fi = st.session_state.rul_predictor.feature_importance
            if fi is not None and not fi.empty:
                import plotly.express as px

                fig = px.bar(
                    fi.head(10),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top 10 Features for RUL Prediction",
                )
                fig = style_bar_figure(
                    fig, n_cats=min(10, len(fi)), horizontal=True, title="Top 10 Features for RUL Prediction"
                )
                st.plotly_chart(fig, use_container_width=True, key="rul_feature_importance")

    if st.session_state.anomaly_summary:
        st.subheader("Anomaly Detection Summary")
        s = st.session_state.anomaly_summary
        c1, c2, c3 = st.columns(3)
        c1.metric("Records Analyzed", f"{s.get('total_records', 0):,}")
        c2.metric("Anomalies Found", s.get("anomaly_count", 0))
        c3.metric("Anomaly Rate", f"{s.get('anomaly_rate_pct', 0)}%")


def _refresh_brief(df: pd.DataFrame) -> dict:
    brief = build_industrial_brief(
        df,
        st.session_state.predictions,
        st.session_state.anomaly_summary,
        cost_per_hour=float(st.session_state.get("cost_per_hour") or 0),
        cost_per_unit=float(st.session_state.get("cost_per_unit") or 0),
        hours_if_stop=float(st.session_state.get("hours_if_stop") or 8),
        used_synthetic=bool(st.session_state.get("rul_used_synthetic")),
    )
    st.session_state.business_insights = brief["insights"]
    st.session_state.asset_ranking = brief["ranked"]
    st.session_state.inspect_list = brief["inspect"]
    return brief


def render_ask_panel(df: pd.DataFrame | None) -> None:
    st.subheader("Ask this upload")
    st.caption(CHAT_SCOPE_CAPTION)
    if df is None:
        st.warning("Upload and clean a sensor table first. Chat only answers from this upload.")
        return
    if not get_gemini_api_key():
        st.info(NO_KEY_MESSAGE + " Table-grounded answers still run from the scored rows.")
    if st.session_state.get("last_gemini_error"):
        _show_gemini_error(st.session_state.last_gemini_error)
    if st.session_state.get("last_insight_error"):
        st.caption(f"Retrieval: {st.session_state.last_insight_error}")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about an asset, inspect list, or $ at risk on this upload…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        idx = st.session_state.insight_index
        if idx is None:
            idx, meta = make_insight_index(
                df,
                st.session_state.predictions,
                ranked=st.session_state.get("asset_ranking"),
                inspect=st.session_state.get("inspect_list"),
            )
            st.session_state.insight_index = idx
            if meta.get("error"):
                st.session_state.last_insight_error = str(meta["error"])
        result = ask_with_index(
            prompt,
            df,
            idx,
            st.session_state.predictions,
            ranked=st.session_state.get("asset_ranking"),
            inspect=st.session_state.get("inspect_list"),
        )
        if result.get("key_missing"):
            st.session_state.last_gemini_error = ""
        if result.get("gemini_error"):
            st.session_state.last_gemini_error = result["gemini_error"]
        if result.get("index_error"):
            st.session_state.last_insight_error = result["index_error"]
        response = result["answer"]
        if result.get("key_missing"):
            response = f"{NO_KEY_MESSAGE}\n\n**From this upload:**\n{result.get('offline_answer') or response}"
        elif result.get("gemini_error"):
            response = (
                f"{result['gemini_error']}\n\n**From this upload (offline):**\n"
                f"{result.get('offline_answer') or response}"
            )
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if st.session_state.chat_history and st.button("Clear Ask"):
        st.session_state.chat_history = []
        st.rerun()


# ── Business Insights ─────────────────────────────────────────────────────────
def page_business_insights():
    st.markdown('<p class="main-header">5. Insights</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Inspect list, slow-running assets, and $ at risk from this table — not generic advice.</p>',
        unsafe_allow_html=True,
    )
    st.caption(config.RUL_HONESTY_CAPTION)
    df = get_active_df()
    if df is None:
        st.warning("Clean data + run Anomaly & RUL first for a ranked inspect list.")
        return

    st.subheader("Optional $ rates")
    st.caption(
        "Dollar figures use only the rates you enter. Default 8 h is an assumed stop — not a booked outage."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.cost_per_hour = st.number_input(
            "$ / hour downtime",
            min_value=0.0,
            value=float(st.session_state.get("cost_per_hour") or 0),
            step=50.0,
        )
    with c2:
        st.session_state.cost_per_unit = st.number_input(
            "$ / unit lost production",
            min_value=0.0,
            value=float(st.session_state.get("cost_per_unit") or 0),
            step=1.0,
        )
    with c3:
        st.session_state.hours_if_stop = st.number_input(
            "Hours if a high-risk asset stops",
            min_value=0.0,
            value=float(st.session_state.get("hours_if_stop") or 8),
            step=1.0,
        )

    b1, b2 = st.columns(2)
    refresh = b1.button("Refresh brief", type="primary")
    polish = b2.button("Gemini polish (wording only)")
    brief = _refresh_brief(df)
    if refresh or st.session_state.insight_index is None:
        idx, meta = make_insight_index(
            df,
            st.session_state.predictions,
            ranked=brief["ranked"],
            inspect=brief["inspect"],
        )
        st.session_state.insight_index = idx
        st.caption(f"Retrieval mode: {meta.get('mode')} ({meta.get('n_docs')} docs)")
        if meta.get("error"):
            st.session_state.last_insight_error = str(meta["error"])
            st.error(f"LlamaIndex vector index unavailable: {meta['error']}")
            st.caption("TF-IDF / keyword retrieval is used instead.")
        else:
            st.session_state.last_insight_error = ""
            if meta.get("note"):
                st.caption(meta["note"])

    if polish:
        polished = polish_brief_with_gemini(st.session_state.business_insights or [])
        if polished.get("key_missing"):
            st.info(NO_KEY_MESSAGE)
        elif not polished.get("ok"):
            st.error(polished.get("error") or "Gemini polish failed.")
        else:
            st.session_state.polished_brief = polished.get("text") or ""
            st.success(f"Polished with `{polished.get('model')}` — numbers taken from the rule-based brief.")

    if st.session_state.get("last_insight_error"):
        st.error(st.session_state.last_insight_error)

    ranked = st.session_state.get("asset_ranking") or []
    if ranked:
        st.subheader("Asset rank")
        rank_df = pd.DataFrame(ranked)[
            [
                c
                for c in (
                    "rank",
                    "machine_id",
                    "risk_level",
                    "predicted_rul_days",
                    "mean_anomaly_score",
                    "anomaly_count",
                    "anomaly_rate_pct",
                    "n_rows",
                )
                if c in pd.DataFrame(ranked).columns
            ]
        ]
        st.dataframe(rank_df, use_container_width=True)

    if st.session_state.get("polished_brief"):
        st.subheader("Gemini brief")
        st.markdown(st.session_state.polished_brief)

    for item in st.session_state.business_insights or generate_business_insights(
        df,
        st.session_state.predictions,
        st.session_state.anomaly_summary,
        cost_per_hour=float(st.session_state.get("cost_per_hour") or 0),
        cost_per_unit=float(st.session_state.get("cost_per_unit") or 0),
        hours_if_stop=float(st.session_state.get("hours_if_stop") or 8),
        used_synthetic=bool(st.session_state.get("rul_used_synthetic")),
    ):
        st.markdown(f"### {item['title']} — `{item['severity']}`")
        st.markdown(item["message"])

    render_ask_panel(df)


# ── 3D Digital Twin (Layer 3) ─────────────────────────────────────────────────
def page_twin_3d():
    st.markdown('<p class="main-header">3D Digital Twin</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Rotatable 3D asset. The drive-end bearing turns red and blinks '
        "when the selected machine's predicted risk is High (amber = Medium, green = Low). "
        "Asset-agnostic — the same twin serves any rotating machine.</p>",
        unsafe_allow_html=True,
    )

    batch_states = asset_states_from_predictions(st.session_state.get("predictions") or [])
    live_states = st.session_state.get("live_asset_states") or []
    sources: dict[str, list] = {}
    if batch_states:
        sources["Anomaly & RUL (batch)"] = batch_states
    if live_states:
        sources["Live Connect (streaming)"] = live_states

    if sources:
        source_names = list(sources.keys())
        default_idx = (
            source_names.index("Live Connect (streaming)")
            if ("Live Connect (streaming)" in sources and st.session_state.get("live_running"))
            else 0
        )
        c0, c1, c2 = st.columns([1.4, 1.6, 1])
        with c0:
            src_name = st.radio("Risk source", source_names, index=default_idx, key="twin_source")
        states = sources[src_name]
        ids = [s["machine_id"] for s in states]
        with c1:
            selected = st.selectbox("Asset", ids, key="twin_asset_pick")
        sel = next((s for s in states if s["machine_id"] == selected), states[0])
        with c2:
            st.metric("Risk", sel["risk_level"])
        components.html(build_twin_html(states, selected_id=selected, height=540), height=560)
        st.caption(
            "Risk is read from **3. Anomaly & RUL** or **Live Connect** (pick the source above). "
            "Rotate with the mouse; scroll to zoom."
        )
    else:
        st.info(
            "No predictions yet. Run **3. Anomaly & RUL** to drive the twin from real risk, "
            "or preview it with a manual risk below."
        )
        demo_risk = st.selectbox("Preview risk", ["High", "Medium", "Low"], index=0)
        demo_rul = {"High": 3, "Medium": 12, "Low": 26}[demo_risk]
        demo = [{"machine_id": "demo-motor", "risk_level": demo_risk, "predicted_rul_days": demo_rul}]
        components.html(build_twin_html(demo, selected_id="demo-motor", height=540), height=560)

    st.caption(
        "Legend — "
        + " · ".join(
            f"**{k}** {v['hex']}" for k, v in RISK_COLORS.items() if k != "Unknown"
        )
    )


# ── Live Connect (Layer 4) ────────────────────────────────────────────────────
def _live_body():
    cfg = dict(st.session_state.get("live_cfg") or {})
    if st.session_state.get("live_running"):
        tick = int(st.session_state.get("live_tick") or 0)
        batch = pd.DataFrame()
        if st.session_state.get("live_source") == "poll":
            try:
                batch, new_off = poll_url_increment(
                    cfg.get("url", ""), UPLOAD_DIR, int(cfg.get("offset", 0)), int(cfg.get("poll_n", 30))
                )
                cfg["offset"] = new_off
                st.session_state.live_cfg = cfg
                if batch.empty:
                    st.session_state.live_running = False
                    st.info("Reached end of feed — stopped.")
            except Exception as exc:
                st.session_state.live_running = False
                st.error(f"Poll failed: {exc}")
        else:
            ramp = max(1, int(cfg.get("ramp_ticks", 40)))
            stress = min(1.0, tick / ramp)
            batch = simulate_batch(
                cfg.get("machines") or default_machines(),
                tick,
                failing=cfg.get("failing"),
                stress=stress,
                freq_seconds=int(cfg.get("freq", 5)),
            )
        if not batch.empty:
            st.session_state.live_buffer = append_to_buffer(st.session_state.get("live_buffer"), batch)
            st.session_state.live_tick = tick + 1

    buffer = st.session_state.get("live_buffer")
    if buffer is None or buffer.empty:
        st.info("Press **Start live** to begin streaming sensor data.")
        return

    status = compute_live_status(buffer)
    st.session_state.live_asset_states = status

    worst = status[0] if status else None
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows buffered", f"{len(buffer):,}")
    m2.metric("Machines", buffer["machine_id"].nunique() if "machine_id" in buffer.columns else 0)
    if worst:
        m3.metric(
            "Worst asset",
            f"{worst['machine_id']} · {worst['risk_level']}",
            f"{worst['anomaly_rate_pct']}% anomalies",
        )

    st.subheader("Live asset status")
    st.dataframe(pd.DataFrame(status), use_container_width=True)

    sensor = st.session_state.get("live_sensor_view", "temperature")
    if sensor in buffer.columns and "timestamp" in buffer.columns:
        import plotly.express as px

        recent = buffer.tail(300)
        fig = px.line(
            recent,
            x="timestamp",
            y=sensor,
            color="machine_id",
            title=f"Live {sensor} (last {len(recent)} readings)",
        )
        fig.update_layout(height=340, margin=dict(l=40, r=20, t=48, b=40))
        st.plotly_chart(
            fig, use_container_width=True, key=f"live_chart_{sensor}_{st.session_state.get('live_tick')}"
        )

    st.caption(f"tick {st.session_state.get('live_tick')} · live risk feeds Insights + 3D Twin")


def page_live_connect():
    st.markdown('<p class="main-header">Live Connect</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Streaming ingest into the PdM pipeline. Simulate live assets or poll a '
        "remote CSV feed; live risk drives the 3D Twin in near-real-time.</p>",
        unsafe_allow_html=True,
    )

    source = st.radio(
        "Source",
        ["Simulator", "Poll CSV URL"],
        horizontal=True,
        index=0 if st.session_state.get("live_source", "sim") == "sim" else 1,
    )
    st.session_state.live_source = "sim" if source == "Simulator" else "poll"

    cfg = dict(st.session_state.get("live_cfg") or {})
    if st.session_state.live_source == "sim":
        c1, c2, c3 = st.columns(3)
        with c1:
            n_machines = st.slider("Machines", 2, 8, int(cfg.get("n_machines", 4)))
        machines = default_machines(n_machines)
        with c2:
            fail_idx = machines.index(cfg["failing"]) if cfg.get("failing") in machines else 0
            failing = st.selectbox("Failing asset (drifts to failure)", machines, index=fail_idx)
        with c3:
            ramp = st.slider("Ramp ticks to failure", 10, 120, int(cfg.get("ramp_ticks", 40)))
        cfg.update(
            {"n_machines": n_machines, "machines": machines, "failing": failing, "ramp_ticks": ramp, "freq": 5}
        )
    else:
        cfg["url"] = st.text_input(
            "CSV feed URL", value=cfg.get("url", "http://localhost:8000/sensor_readings.csv")
        )
        cfg["poll_n"] = st.slider("Rows per poll", 5, 200, int(cfg.get("poll_n", 30)))
        cfg.setdefault("offset", 0)
    st.session_state.live_cfg = cfg

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        if st.button("▶ Start live", type="primary", use_container_width=True):
            st.session_state.live_running = True
            if st.session_state.live_source == "poll":
                cfg["offset"] = 0
                st.session_state.live_cfg = cfg
                st.session_state.live_buffer = None
                st.session_state.live_tick = 0
            st.rerun()
    with ctrl2:
        if st.button("⏸ Stop", use_container_width=True):
            st.session_state.live_running = False
            st.rerun()
    with ctrl3:
        if st.button("Reset buffer", use_container_width=True):
            st.session_state.live_buffer = None
            st.session_state.live_tick = 0
            st.session_state.live_asset_states = []
            st.rerun()
    with ctrl4:
        st.session_state.live_sensor_view = st.selectbox(
            "Live chart",
            ["temperature", "vibration", "pressure", "rpm"],
            index=["temperature", "vibration", "pressure", "rpm"].index(
                st.session_state.get("live_sensor_view", "temperature")
            ),
        )

    running = bool(st.session_state.get("live_running"))
    st.caption(("🟢 streaming (auto-refresh ~2s)" if running else "⚪ stopped") + " — open **3D Twin** to see risk in 3D")

    interval = 2.0 if running else None
    if hasattr(st, "fragment"):
        st.fragment(run_every=interval)(_live_body)()
    else:
        _live_body()
        if running:
            st.warning("Auto-refresh needs Streamlit ≥1.37; click Start again to advance a tick.")


# ── Dashboard ─────────────────────────────────────────────────────────────────
def page_dashboard_builder():
    st.markdown('<p class="main-header">Dashboard Builder</p>', unsafe_allow_html=True)
    folder = get_graph_folder()
    if not folder:
        st.info("No graphs saved yet. Go to Explore & Graphs.")
        return
    selected = []
    for g_id, entry in folder.items():
        if st.checkbox(f"{entry.get('title', g_id)} ({entry['graph_type']})", key=f"dash_chk_{g_id}", value=True):
            selected.append(g_id)
    render_dashboard_preview(selected, columns=2)
    if selected:
        html = export_dashboard_html(selected)
        st.download_button(
            "Export Dashboard (HTML)",
            html,
            file_name=f"maintenance_dashboard_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html",
        )


# ── AI Assistant — same Ask box as Insights (one scoped path) ─────────────────
def page_ai_assistant():
    st.markdown('<p class="main-header">Ask this upload</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Same Ask path as Insights: this table + Isolation Forest / RUL columns.</p>',
        unsafe_allow_html=True,
    )
    st.caption(CHAT_SCOPE_CAPTION)
    render_ask_panel(get_active_df())


# ── Email ─────────────────────────────────────────────────────────────────────
def page_email_report():
    st.markdown('<p class="main-header">Email Report</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        recipient = st.text_input("Recipient Email", placeholder="manager@company.com")
        manager_name = st.text_input("Manager Name", value="Manager")
    with col2:
        subject = st.text_input(
            "Email Subject",
            value=f"Predictive Maintenance Report — {datetime.now().strftime('%Y-%m-%d')}",
        )
        additional_notes = st.text_area("Additional Notes (optional)", height=100)

    if st.button("Generate Email Preview", type="primary"):
        extra = additional_notes or ""
        if st.session_state.business_insights:
            extra += "\n\nBusiness insights:\n" + "\n".join(
                f"- {i['title']}: {i['message']}" for i in st.session_state.business_insights[:5]
            )
        body = generate_email_body(
            manager_name=manager_name,
            predictions=st.session_state.predictions,
            anomaly_summary=st.session_state.anomaly_summary,
            insights=st.session_state.insights,
            additional_notes=extra,
        )
        st.session_state.email_preview = body
        st.session_state.email_subject = subject
        st.session_state.email_recipient = recipient

    if "email_preview" in st.session_state:
        st.subheader("Email Preview")
        st.text(st.session_state.email_preview)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Send Email", type="primary"):
                result = send_email(
                    recipient=st.session_state.get("email_recipient", recipient),
                    subject=st.session_state.get("email_subject", subject),
                    body=st.session_state.email_preview,
                    manager_name=manager_name,
                )
                (st.success if result["success"] else st.error)(result["message"])
        with col2:
            st.download_button(
                "Download Email as TXT",
                st.session_state.email_preview,
                file_name=f"maintenance_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
            )


def main():
    init_session_state()
    st.sidebar.markdown(f"## {config.APP_TITLE}")
    st.sidebar.caption(config.TAGLINE)
    st.sidebar.markdown("---")

    pages = {
        "1. Upload & Clean": page_upload_clean,
        "2. Map sensors": page_map_sensors,
        "3. Anomaly & RUL": page_ml_predictions,
        "4. Charts": page_explore_graphs,
        "5. Insights": page_business_insights,
        "3D Twin": page_twin_3d,
        "Live Connect": page_live_connect,
        "AI Assistant": page_ai_assistant,
        "Email Report": page_email_report,
        "Dashboard": page_dashboard_builder,
        "Joins (lab)": page_data_integration,
        "SQL lab": page_dwdm_sql,
    }

    selection = st.sidebar.radio("Pipeline", list(pages.keys()))
    render_gemini_sidebar()
    st.sidebar.markdown("---")
    df = get_active_df()
    if df is not None:
        st.sidebar.success(f"Data: {len(df):,} rows")
        status = mapping_status(df)
        st.sidebar.caption(
            f"Sensors: {status['sensor_count']} · "
            f"RUL label: {'yes' if status['has_rul_label'] else 'no'}"
        )
    else:
        st.sidebar.warning("No data loaded")
    if st.session_state.uploaded_tables:
        st.sidebar.info(f"{len(st.session_state.uploaded_tables)} table(s)")
    if st.session_state.predictions:
        st.sidebar.info(f"{len(st.session_state.predictions)} prediction(s)")
    if get_graph_folder():
        st.sidebar.info(f"{len(get_graph_folder())} graph(s) saved")

    pages[selection]()


if __name__ == "__main__":
    main()
