"""
Predictive Maintenance Analytics Platform
Streamlit entry point for IoT sensor analysis, ML predictions, and reporting.
"""

from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on path
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
from src.data_cleaner import clean_sensor_data
from src.data_insights import analyze_columns, format_insights_markdown
from src.email_report import generate_email_body, send_email
from src.graphs import GRAPH_TYPES
from src.ml.anomaly_detector import AnomalyDetector
from src.ml.rul_predictor import RULPredictor

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .graph-type-btn {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .graph-type-btn:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102,126,234,0.2);
    }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low { color: #27ae60; font-weight: bold; }
    div[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "raw_df": None,
        "cleaned_df": None,
        "cleaning_summary": None,
        "insights": None,
        "graph_folder": {},
        "anomaly_detector": None,
        "rul_predictor": None,
        "predictions": [],
        "anomaly_summary": {},
        "chat_history": [],
        "data_loaded": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_active_df() -> pd.DataFrame | None:
    """Return cleaned dataframe if available."""
    return st.session_state.get("cleaned_df")


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def get_axis_options(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    if "timestamp" in cols:
        return ["timestamp"] + [c for c in cols if c != "timestamp"]
    return cols


def load_sample_data():
    """Load bundled sample CSV."""
    if config.SAMPLE_DATA_PATH.exists():
        df = pd.read_csv(config.SAMPLE_DATA_PATH, parse_dates=["timestamp"])
        st.session_state.raw_df = df
        st.session_state.data_loaded = True
        return df
    st.error("Sample data not found. Run: python generate_sample_data.py")
    return None


# ── Section: Upload & Clean ───────────────────────────────────────────────────
def page_upload_clean():
    st.markdown('<p class="main-header">Upload & Clean Data</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload IoT sensor CSV or use sample data. Clean, inspect, and download.</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])
    with col2:
        if st.button("Load Sample Data", use_container_width=True):
            load_sample_data()
            st.success("Sample data loaded!")
            st.rerun()

    if uploaded:
        st.session_state.raw_df = pd.read_csv(uploaded)
        if "timestamp" in st.session_state.raw_df.columns:
            st.session_state.raw_df["timestamp"] = pd.to_datetime(
                st.session_state.raw_df["timestamp"], errors="coerce"
            )
        st.session_state.data_loaded = True

    if st.session_state.raw_df is not None:
        st.subheader("Raw Data Preview")
        st.dataframe(st.session_state.raw_df.head(10), use_container_width=True)

        if st.button("Clean Data", type="primary"):
            with st.spinner("Cleaning data..."):
                cleaned, summary = clean_sensor_data(st.session_state.raw_df)
                insights = analyze_columns(cleaned)
                st.session_state.cleaned_df = cleaned
                st.session_state.cleaning_summary = summary
                st.session_state.insights = insights
            st.success(f"Cleaned: {summary['rows_before']} → {summary['rows_after']} rows")
            if summary["actions"]:
                for action in summary["actions"]:
                    st.info(action)

        if st.session_state.cleaned_df is not None:
            st.subheader("Cleaned Data Preview")
            st.dataframe(st.session_state.cleaned_df.head(10), use_container_width=True)

            # Download
            csv_buf = st.session_state.cleaned_df.to_csv(index=False)
            st.download_button(
                "Download Cleaned CSV",
                csv_buf,
                file_name=f"cleaned_sensor_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

            # Insights panel
            st.subheader("Data Insights")
            if st.session_state.insights:
                st.markdown(format_insights_markdown(st.session_state.insights))

                with st.expander("Detailed Column Analysis"):
                    for col, info in st.session_state.insights["columns"].items():
                        st.write(f"**{col}** — {info['type_category']} | Quality: {info['quality_score']}/100")
                        st.json(info)


# ── Section: Explore & Graphs ───────────────────────────────────────────────────
def page_explore_graphs():
    st.markdown('<p class="main-header">Explore & Graphs</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Configure axes, metrics, and filters. Click a graph type to generate and save to your graph folder.</p>',
        unsafe_allow_html=True,
    )

    df = get_active_df()
    if df is None:
        st.warning("Upload and clean data first, or load sample data from Upload & Clean.")
        return

    # Adaptive controls
    st.subheader("Chart Controls")
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    axis_options = get_axis_options(df)
    numeric_cols = get_numeric_columns(df)

    with ctrl1:
        x_axis = st.selectbox("X-Axis", axis_options, index=0)
    with ctrl2:
        y_metric = st.selectbox(
            "Measure / Metric",
            numeric_cols,
            index=numeric_cols.index("temperature") if "temperature" in numeric_cols else 0,
        )
    with ctrl3:
        machines = sorted(df["machine_id"].unique()) if "machine_id" in df.columns else []
        machine_filter = st.multiselect("Filter by Machine", machines, default=machines)

    st.divider()
    st.subheader("Graph Types — Click to Generate & Save")
    st.caption(f"Graph folder: {len(get_graph_folder())} saved graph(s)")

    # 6 graph types in a 3x2 grid
    graph_keys = list(GRAPH_TYPES.keys())
    for row_start in range(0, 6, 3):
        cols = st.columns(3)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx >= len(graph_keys):
                break
            gkey = graph_keys[idx]
            ginfo = GRAPH_TYPES[gkey]
            with col:
                st.markdown(f"### {ginfo['icon']} {ginfo['name']}")
                st.caption(ginfo["description"])
                if st.button(f"Generate {ginfo['name']}", key=f"gen_{gkey}", use_container_width=True):
                    with st.spinner(f"Creating {ginfo['name']}..."):
                        kwargs = {
                            "df": df,
                            "x_axis": x_axis,
                            "y_metric": y_metric,
                            "machine_filter": machine_filter or None,
                        }
                        if gkey == "anomaly_scatter" and st.session_state.anomaly_detector:
                            kwargs["anomaly_detector"] = st.session_state.anomaly_detector
                        fig = ginfo["create"](**kwargs)
                        entry = save_graph_to_folder(
                            gkey,
                            fig,
                            gkey,
                            {"x_axis": x_axis, "y_metric": y_metric, "machines": machine_filter},
                        )
                    st.success(f"Saved: {entry['id']}")
                    st.plotly_chart(fig, use_container_width=True)

    # Preview saved graphs
    folder = get_graph_folder()
    if folder:
        st.divider()
        st.subheader("Saved Graphs Preview")
        for g_id, entry in folder.items():
            with st.expander(f"{entry.get('title', g_id)} — {entry['created_at'][:19]}"):
                st.plotly_chart(entry["fig"], use_container_width=True)
                if st.button(f"Remove {g_id}", key=f"rm_{g_id}"):
                    del st.session_state.graph_folder[g_id]
                    st.rerun()


# ── Section: ML Predictions ─────────────────────────────────────────────────────
def page_ml_predictions():
    st.markdown('<p class="main-header">ML Predictions</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Train anomaly detection and RUL models. View failure predictions per machine.</p>',
        unsafe_allow_html=True,
    )

    df = get_active_df()
    if df is None:
        st.warning("Upload and clean data first.")
        return

    if st.button("Train Models", type="primary"):
        with st.spinner("Training Isolation Forest + Random Forest RUL models..."):
            detector = AnomalyDetector()
            detector.fit(df)
            st.session_state.anomaly_detector = detector
            st.session_state.anomaly_summary = detector.summary(df)

            predictor = RULPredictor()
            predictor.fit(df)
            st.session_state.rul_predictor = predictor
            st.session_state.predictions = predictor.predict_latest_per_machine(df)
        st.success("Models trained successfully!")

    if st.session_state.predictions:
        st.subheader("Failure Predictions")
        for p in st.session_state.predictions:
            risk_class = f"risk-{p['risk_level'].lower()}"
            st.markdown(
                f"<p class='{risk_class}'>{p['message']} — Risk: {p['risk_level']}</p>",
                unsafe_allow_html=True,
            )

        # Metrics
        if st.session_state.rul_predictor and st.session_state.rul_predictor.metrics:
            st.subheader("RUL Model Metrics")
            m = st.session_state.rul_predictor.metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE (days)", m.get("mae", "N/A"))
            c2.metric("RMSE (days)", m.get("rmse", "N/A"))
            c3.metric("R² Score", m.get("r2", "N/A"))

            # Feature importance
            fi = st.session_state.rul_predictor.feature_importance
            if fi is not None and not fi.empty:
                st.subheader("Feature Importance")
                import plotly.express as px
                fig = px.bar(
                    fi.head(10),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top 10 Features for RUL Prediction",
                )
                fig.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig, use_container_width=True)

    if st.session_state.anomaly_summary:
        st.subheader("Anomaly Detection Summary")
        s = st.session_state.anomaly_summary
        c1, c2, c3 = st.columns(3)
        c1.metric("Records Analyzed", f"{s.get('total_records', 0):,}")
        c2.metric("Anomalies Found", s.get("anomaly_count", 0))
        c3.metric("Anomaly Rate", f"{s.get('anomaly_rate_pct', 0)}%")


# ── Section: Dashboard Builder ──────────────────────────────────────────────────
def page_dashboard_builder():
    st.markdown('<p class="main-header">Dashboard Builder</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Select saved graphs to compose your final maintenance dashboard.</p>',
        unsafe_allow_html=True,
    )

    folder = get_graph_folder()
    if not folder:
        st.info("No graphs saved yet. Go to Explore & Graphs to generate charts.")
        return

    st.subheader("Select Graphs")
    selected = []
    for g_id, entry in folder.items():
        if st.checkbox(
            f"{entry.get('title', g_id)} ({entry['graph_type']})",
            key=f"dash_chk_{g_id}",
            value=True,
        ):
            selected.append(g_id)

    st.divider()
    st.subheader("Live Dashboard Preview")
    render_dashboard_preview(selected, columns=2)

    if selected:
        html = export_dashboard_html(selected)
        st.download_button(
            "Export Dashboard (HTML)",
            html,
            file_name=f"maintenance_dashboard_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html",
        )


# ── Section: AI Assistant ─────────────────────────────────────────────────────
def page_ai_assistant():
    st.markdown('<p class="main-header">AI Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask questions about your data, predictions, and anomalies. Data-aware responses, no API key required.</p>',
        unsafe_allow_html=True,
    )

    assistant = MaintenanceAIAssistant(
        df=get_active_df(),
        predictions=st.session_state.predictions,
        anomaly_summary=st.session_state.anomaly_summary,
        insights=st.session_state.insights,
    )

    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested prompts
    st.caption("Suggested: *Which machine will fail soonest?* | *What's the anomaly rate?* | *Give me a summary*")

    if prompt := st.chat_input("Ask about your maintenance data..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        response = assistant.respond(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ── Section: Email Report ───────────────────────────────────────────────────────
def page_email_report():
    st.markdown('<p class="main-header">Email Report</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Auto-generate and send maintenance reports to managers and stakeholders.</p>',
        unsafe_allow_html=True,
    )

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
        body = generate_email_body(
            manager_name=manager_name,
            predictions=st.session_state.predictions,
            anomaly_summary=st.session_state.anomaly_summary,
            insights=st.session_state.insights,
            additional_notes=additional_notes,
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
                if result["success"]:
                    st.success(result["message"])
                else:
                    st.error(result["message"])
        with col2:
            st.download_button(
                "Download Email as TXT",
                st.session_state.email_preview,
                file_name=f"maintenance_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
            )

    st.info(
        "Demo mode is enabled by default — emails are saved to `output/emails/` when SMTP is not configured. "
        "Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD environment variables for live sending."
    )


# ── Main App ────────────────────────────────────────────────────────────────────
def main():
    init_session_state()

    st.sidebar.markdown(f"## {config.APP_TITLE}")
    st.sidebar.markdown("---")

    pages = {
        "Upload & Clean": page_upload_clean,
        "Explore & Graphs": page_explore_graphs,
        "ML Predictions": page_ml_predictions,
        "Dashboard Builder": page_dashboard_builder,
        "AI Assistant": page_ai_assistant,
        "Email Report": page_email_report,
    }

    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    st.sidebar.markdown("---")

    # Status indicators
    df = get_active_df()
    if df is not None:
        st.sidebar.success(f"Data: {len(df):,} rows")
    else:
        st.sidebar.warning("No data loaded")

    if st.session_state.predictions:
        st.sidebar.info(f"{len(st.session_state.predictions)} prediction(s)")
    if get_graph_folder():
        st.sidebar.info(f"{len(get_graph_folder())} graph(s) saved")

    pages[selection]()


if __name__ == "__main__":
    main()
