"""
Predictive Maintenance Analytics — reliability add-on.

Pipeline: Upload → Clean → Map sensors → Anomaly (Isolation Forest) → RUL/risk → charts → insights.
Not a generic analytics OS (Forge) and not an OEE cockpit (Pulse).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.ai_assistant import MaintenanceAIAssistant
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
from src.graphs import GRAPH_TYPES, PRIMARY_CHARTS
from src.graphs.layout import style_bar_figure
from src.insights_engine import (
    InsightIndex,
    ask_with_index,
    chart_business_insight,
    generate_business_insights,
    get_gemini_api_key,
    get_gemini_model,
    mask_key,
    persist_session_gemini_key,
    test_gemini_connection,
)
from src.ml.anomaly_detector import AnomalyDetector
from src.ml.optuna_tuner import tune_anomaly_contamination, tune_rul_model
from src.ml.rul_predictor import RULPredictor
from src.quality_checks import QUALITY_STAGE_COUNT
from src.sensor_map import CANONICAL_FIELDS, apply_mapping, mapping_status, suggest_mapping

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


# ── Upload & Clean (19-stage quality) ─────────────────────────────────────────
def page_upload_clean():
    st.markdown('<p class="main-header">1. Upload & Clean</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{config.TAGLINE}</p>', unsafe_allow_html=True)
    st.caption(
        f"Industrial ETL + {QUALITY_STAGE_COUNT}-stage quality checks on sensor CSVs. "
        "Next: map columns, then anomaly / RUL."
    )

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

    if st.session_state.uploaded_tables:
        st.caption("Registered tables: " + ", ".join(st.session_state.uploaded_tables.keys()))

    if st.session_state.raw_df is not None:
        st.subheader("Raw Data Preview")
        st.dataframe(st.session_state.raw_df.head(10), use_container_width=True)

        if st.button("Run industrial clean + 19 quality checks", type="primary"):
            with st.spinner("Cleaning + running 19 quality stages..."):
                cleaned, summary, report = clean_and_quality(st.session_state.raw_df, run_quality=True)
                insights = analyze_columns(cleaned)
                st.session_state.cleaned_df = cleaned
                st.session_state.cleaning_summary = summary
                st.session_state.quality_report = report
                st.session_state.insights = insights
                register_table("cleaned", cleaned)
            st.success(f"Cleaned: {summary['rows_before']} → {summary['rows_after']} rows")
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
                st.plotly_chart(fig, use_container_width=True)

    extras = [k for k in GRAPH_TYPES if k not in PRIMARY_CHARTS]
    with st.expander("More sensor views (optional)"):
        for gkey in extras:
            ginfo = GRAPH_TYPES[gkey]
            if st.button(f"{ginfo['icon']} {ginfo['name']}", key=f"gen_{gkey}"):
                with st.spinner(f"Creating {ginfo['name']}..."):
                    fig, entry = _make_chart(gkey)
                st.success(f"Saved: {entry['id']}")
                st.plotly_chart(fig, use_container_width=True)

    folder = get_graph_folder()
    if folder:
        st.divider()
        st.subheader("Saved Graphs")
        for g_id, entry in folder.items():
            with st.expander(f"{entry.get('title', g_id)} — {entry['created_at'][:19]}"):
                st.plotly_chart(entry["fig"], use_container_width=True)
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
                st.session_state.anomaly_detector = detector
                st.session_state.anomaly_summary = detector.summary(df)
                predictor = RULPredictor()
                predictor.fit(df)
                st.session_state.rul_predictor = predictor
                st.session_state.rul_used_synthetic = bool(predictor.used_synthetic_labels)
                st.session_state.predictions = predictor.predict_latest_per_machine(df)
                st.session_state.business_insights = generate_business_insights(
                    df, st.session_state.predictions, st.session_state.anomaly_summary
                )
                st.success("Models trained.")
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
                st.plotly_chart(fig, use_container_width=True)

    if st.session_state.anomaly_summary:
        st.subheader("Anomaly Detection Summary")
        s = st.session_state.anomaly_summary
        c1, c2, c3 = st.columns(3)
        c1.metric("Records Analyzed", f"{s.get('total_records', 0):,}")
        c2.metric("Anomalies Found", s.get("anomaly_count", 0))
        c3.metric("Anomaly Rate", f"{s.get('anomaly_rate_pct', 0)}%")


# ── Business Insights ─────────────────────────────────────────────────────────
def page_business_insights():
    st.markdown('<p class="main-header">5. Insights</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Rule-based maintenance actions, plus LlamaIndex retrieval and optional Gemini.</p>',
        unsafe_allow_html=True,
    )
    st.caption(config.RUL_HONESTY_CAPTION)
    df = get_active_df()
    if df is None:
        st.warning("Clean data + run Anomaly & RUL first for best insights.")
        return

    if st.button("Refresh insights", type="primary"):
        st.session_state.business_insights = generate_business_insights(
            df, st.session_state.predictions, st.session_state.anomaly_summary
        )
        idx = InsightIndex()
        meta = idx.build(df)
        st.session_state.insight_index = idx
        st.caption(f"LlamaIndex mode: {meta.get('mode')} ({meta.get('n_docs')} docs)")
        if meta.get("error"):
            st.session_state.last_insight_error = str(meta["error"])
            st.error(f"LlamaIndex vector index unavailable: {meta['error']}")
            st.caption("Keyword retrieval is used instead.")
        else:
            st.session_state.last_insight_error = ""

    if st.session_state.get("last_insight_error"):
        st.error(st.session_state.last_insight_error)

    for item in st.session_state.business_insights or generate_business_insights(
        df, st.session_state.predictions, st.session_state.anomaly_summary
    ):
        st.markdown(f"### {item['title']} — `{item['severity']}`")
        st.markdown(item["message"])


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


# ── AI Assistant (LlamaIndex) ─────────────────────────────────────────────────
def page_ai_assistant():
    st.markdown('<p class="main-header">AI Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Rule-based + LlamaIndex retrieval (optional Gemini if GEMINI_API_KEY set).</p>',
        unsafe_allow_html=True,
    )

    df = get_active_df()
    use_llama = st.toggle("Use LlamaIndex retrieval", value=True)

    assistant = MaintenanceAIAssistant(
        df=df,
        predictions=st.session_state.predictions,
        anomaly_summary=st.session_state.anomaly_summary,
        insights=st.session_state.insights,
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    st.caption("Try: *Which machine needs maintenance?* | *Summarize anomalies*")
    if st.session_state.get("last_gemini_error"):
        _show_gemini_error(st.session_state.last_gemini_error)
    if st.session_state.get("last_insight_error"):
        st.caption(f"LlamaIndex: {st.session_state.last_insight_error}")

    if prompt := st.chat_input("Ask about maintenance / business impact..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        if use_llama and df is not None:
            idx = st.session_state.insight_index
            if idx is None:
                idx = InsightIndex()
                meta = idx.build(df)
                st.session_state.insight_index = idx
                if meta.get("error"):
                    st.session_state.last_insight_error = str(meta["error"])
            result = ask_with_index(prompt, df, idx, st.session_state.predictions)
            if result.get("gemini_error"):
                st.session_state.last_gemini_error = result["gemini_error"]
            if result.get("index_error"):
                st.session_state.last_insight_error = result["index_error"]
            response = result["answer"]
            if result.get("gemini_error"):
                response = (
                    f"{result['gemini_error']}\n\n**Offline answer:**\n{result.get('offline_answer') or response}"
                )
        else:
            response = assistant.respond(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


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
