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
    get_graph_folder,
    save_graph_to_folder,
)
from src.dashboard_composer import (
    TILE_SPECS,
    board_cad_html,
    board_pack_charts,
    board_twin_html,
    cad_placeholder_html,
    cad_slot_status,
    compose_dashboard_html,
    default_tile_state,
)
from src.data_cleaner import clean_and_quality
from src.data_insights import analyze_columns, format_insights_markdown
from src.data_integration import (
    JOIN_TYPES,
    is_tabular_zip_name,
    join_many,
    join_two,
    list_zip_tabular_members,
    load_tabular_file,
    load_zip_tabular,
    pick_zip_tabular_member,
    suggest_join_keys,
)
from src.dwdm_sql import DWDM_CONCEPTS, apply_dwdm_transforms, default_sql_examples, run_sql
from src.email_report import generate_email_body, send_email
from src.live_connect import (
    append_to_buffer,
    compute_live_status,
    default_machines,
    poll_url_increment,
    simulate_batch,
)
from src.live_sources import (
    get_source,
    mqtt_available,
    opcua_available,
    parse_node_map,
    start_mqtt,
    start_opcua,
    stop_source,
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
from src.aps_viewer import (
    APS_SAMPLE_CAD_URL,
    A360_PUBLIC_URN_ERROR,
    CAD_SIZE_ZIP_FALLBACK,
    CAD_UPLOAD_EXTENSIONS,
    CAD_VIEWER_COMPONENT_DIR,
    DEFAULT_VIEWER_HEIGHT,
    INSUFFICIENT_SCOPE_HINT,
    MAX_CAD_UPLOAD_MB,
    PUBLIC_VIEWER_NO_URN_WARNING,
    PUBLIC_VIEWER_WARNING,
    VIEWER_ERROR_CODES,
    VIEWER_EXTRA_EXTENSIONS,
    ZIP_ROOT_MISSING_MSG,
    aps_available,
    aps_model_urn,
    build_viewer_html,
    cad_size_issue,
    coerce_db_ids,
    extract_model_urn,
    get_access_token,
    is_cad_zip_name,
    looks_like_public_viewer,
    normalize_model_urn,
    guess_zip_root_from_upload_name,
    pick_zip_cad_root,
    resolve_cad_urn,
    resolve_zip_root_filename,
    delete_urn_for_pack,
    save_urn_for_pack,
    saved_urn_caption,
    saved_urn_for_pack,
    translate_cad_bytes,
)
from src.cad_part import (
    DRIVER_HONESTY,
    NO_REGION_MATCH_MSG,
    build_part_card,
    format_sensor_value,
    latest_asset_sensors,
    parse_cad_part_event,
    parse_cad_region_event,
    part_sensor_hint,
    region_hint_keywords,
    risk_theme_hex,
    sanitize_node_name,
    sensor_driver,
    sensor_grid_rows,
)
from src.cad_regions import (
    NO_ASSIGNMENT_MSG,
    REGIONS,
    assign_region,
    assigned_theme_plan,
    assignment_summary,
    clear_region_map,
    region_label,
)
from src.graphs.pack_kpis import create_asset_health_chart, create_pack_kpi_bars
from src.industry_packs import (
    AUTOMOTIVE_OEM_TRIM_EV_NOTE,
    DEFAULT_PACK_ID,
    extra_field_defs,
    get_pack,
    list_packs,
    pack_label,
    preferred_y_metric,
    sample_path_for,
    suggest_pack,
    twin_kind_for,
)
from src.ml.anomaly_detector import AnomalyDetector
from src.ml.optuna_tuner import tune_anomaly_contamination, tune_rul_model
from src.ml.rul_predictor import RULPredictor
from src.pack_kpis import compute_pack_kpis, insight_cards_from_kpis
from src.twin3d import (
    RISK_COLORS,
    asset_states_from_predictions,
    build_twin_html,
    normalize_risk,
)
from src.quality_checks import QUALITY_STAGE_COUNT
from src.sensor_map import CANONICAL_FIELDS, apply_mapping, mapping_status, suggest_mapping
from src.polars_clean import clean_with_polars, polars_available

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
        "live_conn_id": None,
        # Cleaning engine (pandas | Polars)
        "clean_engine": "pandas",
        "dashboard_tiles": None,
        # Autodesk APS CAD twin — URN from APS_MODEL_URN, pack JSON save, or paste
        "aps_urn": "",
        "cad_selected_part": None,
        # Region-tint report posted back by GuiViewer3D (how many dbIds matched)
        "cad_region_report": None,
        "pipeline_page": "1. Upload & Clean",
        # Industry pack (Plant is the default; 3D + KPIs switch with this)
        "industry_pack": DEFAULT_PACK_ID,
        "pack_kpis": {},
        "pack_suggest": None,
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


# Widget-bound. Never assign st.session_state.industry_pack after the sidebar
# selectbox exists — Streamlit raises StreamlitAPIException even for the same value.
PENDING_INDUSTRY_PACK = "_pending_industry_pack"
PENDING_ZIP_ROOT = "_pending_aps_zip_root_filename"
PENDING_PIPELINE_PAGE = "_pending_pipeline_page"
ANOMALY_RUL_PAGE = "4. Anomaly & RUL"

# Functions allowed to write widget-bound keys because they run before those widgets.
# init_session_state / apply_pending_industry_pack / apply_pending_pipeline_page:
# top of main() before the sidebar.
# _seed_cad_urn_widgets: top of page_cad_twin before key="aps_urn_input".
# _seed_zip_root_widget: top of page_cad_twin before key="aps_zip_root_filename".
PRE_WIDGET_SESSION_FUNCS = frozenset(
    {
        "init_session_state",
        "apply_pending_industry_pack",
        "apply_pending_pipeline_page",
        "_seed_cad_urn_widgets",
        "_seed_zip_root_widget",
    }
)


def active_pack_id() -> str:
    return str(st.session_state.get("industry_pack") or DEFAULT_PACK_ID)


def active_pack() -> dict:
    return get_pack(active_pack_id())


def request_industry_pack(pack_id: str) -> None:
    """Queue a pack change. Applied at the start of the next run, before the sidebar widget."""
    st.session_state[PENDING_INDUSTRY_PACK] = pack_id


def request_pipeline_page(page: str) -> None:
    """Queue a pipeline radio change. Applied at the start of the next run, before the widget."""
    st.session_state[PENDING_PIPELINE_PAGE] = page


def apply_pending_pipeline_page() -> None:
    """Copy a queued pipeline page into ``pipeline_page`` BEFORE the sidebar radio is created."""
    pending = st.session_state.pop(PENDING_PIPELINE_PAGE, None)
    if pending:
        st.session_state.pipeline_page = pending


def apply_pending_industry_pack() -> None:
    """Copy a queued pack into ``industry_pack`` BEFORE the sidebar selectbox is created.

    Load-sample / suggest-pack must not write ``st.session_state.industry_pack`` after
    ``st.sidebar.selectbox(..., key="industry_pack")`` exists in the same script run.
    They set ``_pending_industry_pack`` and ``st.rerun()``; this runs at the top of
    ``main()`` so the widget is born with the new value.
    """
    pending = st.session_state.pop(PENDING_INDUSTRY_PACK, None)
    if pending:
        st.session_state.industry_pack = pending
    packs = {p["id"] for p in list_packs()}
    current = st.session_state.get("industry_pack") or DEFAULT_PACK_ID
    if current not in packs:
        st.session_state.industry_pack = DEFAULT_PACK_ID


def render_pack_sidebar() -> None:
    st.sidebar.markdown("### Industry pack")
    packs = list_packs()
    ids = [p["id"] for p in packs]
    labels = {p["id"]: (p["short"] + ("  ★ SIH" if p.get("hero") else "") + ("  (default)" if p["default"] else "")) for p in packs}
    # No index=/value= — industry_pack is widget-bound; apply_pending_industry_pack()
    # sets it before this selectbox so Streamlit will not warn or throw.
    picked = st.sidebar.selectbox(
        "Field",
        ids,
        format_func=lambda pid: labels.get(pid, pid),
        key="industry_pack",
        help="One file → one pack. Plant is default. 3D mesh, KPIs, and extra sensor map follow the pack.",
    )
    pack = get_pack(picked)
    st.sidebar.caption(pack["label"] + (f" · {pack['sih']}" if pack.get("sih") else ""))
    df = get_active_df() if "cleaned_df" in st.session_state else None
    if df is None:
        df = st.session_state.get("raw_df")
    if st.sidebar.button("Suggest pack from columns", use_container_width=True):
        cols = list(df.columns) if df is not None else []
        mids = df["machine_id"].astype(str).unique().tolist() if df is not None and "machine_id" in df.columns else []
        suggestion = suggest_pack(cols, machine_ids=mids)
        st.session_state.pack_suggest = suggestion
        request_industry_pack(suggestion["pack_id"])
        st.rerun()
    sug = st.session_state.get("pack_suggest")
    if sug:
        st.sidebar.caption("Suggested **" + sug["label"] + "**: " + "; ".join(sug.get("reasons") or []))
    with st.sidebar.expander("What this pack covers"):
        st.write(pack["scope"])
        st.caption("Not in scope: " + pack["not_in_scope"])
        if pack["id"] == "automotive_powertrain":
            st.caption(AUTOMOTIVE_OEM_TRIM_EV_NOTE)


def load_sample_data():
    return load_pack_sample(DEFAULT_PACK_ID)


def load_pack_sample(pack_id: str):
    path = sample_path_for(pack_id, root=PROJECT_ROOT)
    if not path.exists():
        st.error(f"Demo CSV missing: {path.name}. Run `python generate_sample_data.py`.")
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    st.session_state.raw_df = df
    st.session_state.data_loaded = True
    request_industry_pack(pack_id)
    st.session_state.cleaned_df = None
    st.session_state.predictions = []
    st.session_state.pack_kpis = {}
    register_table("sensors", df)
    maint = PROJECT_ROOT / "sample_data" / "maintenance_logs.csv"
    costs = PROJECT_ROOT / "sample_data" / "machine_costs.csv"
    if pack_id == DEFAULT_PACK_ID and maint.exists():
        register_table("maintenance", pd.read_csv(maint))
    if pack_id == DEFAULT_PACK_ID and costs.exists():
        register_table("costs", pd.read_csv(costs))
    return df


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
    with st.expander("Attach maintenance table (optional — join in step 2)"):
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
                    "Open **2. Joins** to merge on `machine_id`."
                )
            except Exception as exc:
                st.error(str(exc))
        elif st.session_state.get("maintenance_table_attached"):
            st.write(
                f"Attached: **{st.session_state.maintenance_table_attached}** (see **2. Joins** to merge)."
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


# Streamlit's dataframe grid virtualizes; keep a fixed height so 500+ rows scroll
# instead of dumping a 10-row head() or exploding the page with st.table.
_PREVIEW_GRID_HEIGHT = 420
_PREVIEW_WARN_ROWS = 20_000


def _scrollable_full_table(df: pd.DataFrame, *, height: int = _PREVIEW_GRID_HEIGHT) -> None:
    """Forge-like preview: pass the full frame so the grid can scroll every row."""
    n = len(df)
    st.dataframe(df, height=height, use_container_width=True)
    if n > _PREVIEW_WARN_ROWS:
        st.caption(
            f"{n:,} rows — scroll to see all (grid virtualizes; height capped at {height}px)."
        )
    else:
        st.caption(f"{n:,} rows — scroll to see all")


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
                type=["csv", "tsv", "xlsx", "json", "zip"],
                accept_multiple_files=True,
                key="upload_sensor_files",
                help=(
                    "CSV, TSV, XLSX, JSON, or a ZIP that contains one of those tables. "
                    f"Up to {MAX_CAD_UPLOAD_MB} MB (Streamlit maxUploadSize; default was 200 MB). "
                    "A ZIP here is sensor/ops data — not the CAD Twin STEP zip on the CAD Twin page."
                ),
            )
        with col2:
            if st.button("Load Plant sample (default)", use_container_width=True, key="upload_load_plant_sample"):
                load_pack_sample(DEFAULT_PACK_ID)
                st.success("Plant sample sensors (+ maintenance/costs if present) loaded.")
                st.rerun()
            demo_pack = st.selectbox(
                "Or load a pack demo CSV",
                [p["id"] for p in list_packs() if p["id"] != DEFAULT_PACK_ID],
                format_func=lambda pid: pack_label(pid),
                key="upload_demo_pack",
            )
            if st.button("Load pack demo", use_container_width=True, key="upload_load_pack_demo"):
                load_pack_sample(demo_pack)
                st.success(f"Loaded {pack_label(demo_pack)} demo and switched the industry pack.")
                st.rerun()

        if uploaded_files:
            for i, uf in enumerate(uploaded_files):
                name = getattr(uf, "name", "") or "upload"
                try:
                    used_member = None
                    member_count = 0
                    if is_tabular_zip_name(name):
                        payload = uf.getvalue()
                        members = list_zip_tabular_members(payload)
                        member_count = len(members)
                        default = pick_zip_tabular_member(members)
                        chosen = default
                        if member_count > 1:
                            labels = [m[0] for m in members]
                            stem_key = "".join(
                                ch if ch.isalnum() else "_" for ch in Path(name).stem
                            )[:40]
                            chosen = st.selectbox(
                                f"Table inside `{name}`",
                                labels,
                                index=labels.index(default) if default in labels else 0,
                                key=f"upload_zip_member_{i}_{stem_key}",
                                help="Several tables in this zip. Default is the largest CSV/TSV.",
                            )
                        df, used_member = load_zip_tabular(payload, chosen)
                        stem = Path(used_member).stem.replace(" ", "_")
                    else:
                        df = load_tabular_file(uf)
                        stem = Path(name).stem.replace(" ", "_")
                    register_table(stem, df)
                    st.session_state.raw_df = df
                    st.session_state.data_loaded = True
                    st.success(f"Loaded `{name}` → table `{stem}` ({len(df):,} rows)")
                    if used_member:
                        extra = f" · {member_count} tables in zip" if member_count > 1 else ""
                        st.caption(
                            f"Used zip member `{used_member}`{extra}. "
                            "Sensor/ops table — not CAD Twin STEP."
                        )
                except Exception as exc:
                    st.error(f"Failed {name}: {exc}")

    with tab_url:
        _ingest_from_url_tab()

    _render_maintenance_attach()

    if st.session_state.uploaded_tables:
        st.caption("Registered tables: " + ", ".join(st.session_state.uploaded_tables.keys()))

    if st.session_state.raw_df is not None:
        st.subheader("Raw Data Preview")
        _scrollable_full_table(st.session_state.raw_df)

        engines = ["pandas"]
        polars_ok, polars_msg = polars_available()
        if polars_ok:
            engines.append("polars")
        eng_col1, eng_col2 = st.columns([1, 2])
        with eng_col1:
            clean_engine = st.selectbox(
                "Cleaning engine",
                engines,
                index=engines.index(st.session_state.get("clean_engine", "pandas"))
                if st.session_state.get("clean_engine", "pandas") in engines
                else 0,
                help="pandas is the default. Polars is the fast columnar engine for larger CSVs (no JVM).",
            )
        with eng_col2:
            st.caption(
                f"Polars: **available** — {polars_msg}"
                if polars_ok
                else f"Polars: not available ({polars_msg}) — pandas engine is used."
            )
        st.session_state.clean_engine = clean_engine

        if st.button("Run industrial clean + 19 quality checks", type="primary"):
            with st.spinner(f"Cleaning with {clean_engine} + running 19 quality stages..."):
                raw = st.session_state.raw_df
                engine_log: list[str] = []
                source_df = raw
                if clean_engine == "polars":
                    try:
                        source_df, engine_log = clean_with_polars(raw)
                    except Exception as exc:
                        st.warning(f"Polars clean failed — falling back to pandas: {exc}")
                        source_df = raw
                cleaned, summary, report = clean_and_quality(source_df, run_quality=True)
                if engine_log:
                    summary["actions"] = engine_log + summary.get("actions", [])
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
            _scrollable_full_table(st.session_state.cleaned_df)
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
    st.markdown('<p class="main-header">3. Map sensors</p>', unsafe_allow_html=True)
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

    suggested = suggest_mapping(list(df.columns), pack_id=active_pack_id())
    saved = st.session_state.get("sensor_mapping") or {}
    status = mapping_status(df)
    pack = active_pack()
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
    st.caption(f"Core PdM fields, then optional **{pack['short']}** extras. Pack: {pack['label']}.")
    for canonical, label in CANONICAL_FIELDS:
        default_src = saved.get(canonical) or suggested.get(canonical) or ""
        idx = cols.index(default_src) if default_src in cols else 0
        picked = st.selectbox(label, cols, index=idx, key=f"map_{canonical}")
        mapping[canonical] = picked or None

    extras = extra_field_defs(active_pack_id())
    if extras:
        st.subheader(f"{pack['short']} extras (optional)")
        for canonical, label in extras:
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
    st.markdown('<p class="main-header">2. Joins</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Integrate tables before mapping: sensor ⋈ maintenance ⋈ cost on '
        "<code>machine_id</code>. I4.0 ingest → integrate → contextualize. Optional if you only have one CSV.</p>",
        unsafe_allow_html=True,
    )

    tables = st.session_state.uploaded_tables or {}
    if len(tables) < 2:
        st.warning(
            "Register at least 2 tables. On **1. Upload & Clean**, load the Plant sample "
            "(sensors + maintenance + costs) or attach a maintenance CSV, then join here."
        )
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
    st.markdown('<p class="main-header">5. Charts</p>', unsafe_allow_html=True)
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
    pack = active_pack()
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    axis_options = get_axis_options(df)
    numeric_cols = get_numeric_columns(df)
    prefer = preferred_y_metric(df, active_pack_id())
    y_default = 0
    if prefer and prefer in numeric_cols:
        y_default = numeric_cols.index(prefer)
    elif "temperature" in numeric_cols:
        y_default = numeric_cols.index("temperature")
    with ctrl1:
        x_axis = st.selectbox("X-Axis", axis_options, index=0)
    with ctrl2:
        y_metric = st.selectbox(
            "Sensor / metric",
            numeric_cols,
            index=y_default,
        )
    with ctrl3:
        machines = sorted(df["machine_id"].unique()) if "machine_id" in df.columns else []
        machine_filter = st.multiselect("Filter by Machine", machines, default=machines)

    bundle = compute_pack_kpis(active_pack_id(), df, st.session_state.get("predictions") or [])
    st.session_state.pack_kpis = bundle
    st.subheader(f"{pack['short']} KPIs" + (f" · {pack['sih']}" if pack.get("sih") else ""))
    st.caption(pack["scope"])
    kpi_cols = st.columns(max(len(bundle["kpis"]), 1))
    for col, kpi in zip(kpi_cols, bundle["kpis"]):
        with col:
            st.metric(kpi["label"], f"{kpi['value']} {kpi['unit']}".strip(), help=kpi["hint"])
    if bundle["by_asset"]:
        health_fig = create_asset_health_chart(bundle["by_asset"], title=f"{pack['short']} health by asset")
        st.plotly_chart(health_fig, use_container_width=True, key="pack_health_chart")
        kpi_fig = create_pack_kpi_bars(bundle["kpis"], title=f"{pack['short']} KPIs")
        st.plotly_chart(kpi_fig, use_container_width=True, key="pack_kpi_bars")

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
    st.markdown('<p class="main-header">4. Anomaly & RUL</p>', unsafe_allow_html=True)
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
    bundle = compute_pack_kpis(active_pack_id(), df, st.session_state.predictions)
    st.session_state.pack_kpis = bundle
    brief["insights"] = list(brief["insights"]) + insight_cards_from_kpis(bundle)
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
    st.markdown('<p class="main-header">6. Insights</p>', unsafe_allow_html=True)
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
    pack = active_pack()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.cost_per_hour = st.number_input(
            pack["cost_hour_label"],
            min_value=0.0,
            value=float(st.session_state.get("cost_per_hour") or 0),
            step=50.0,
        )
    with c2:
        st.session_state.cost_per_unit = st.number_input(
            pack["cost_unit_label"],
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
    pack = active_pack()
    st.markdown('<p class="main-header">3D Digital Twin</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sub-header">{pack["label"]}: rotatable 3D mesh. The '
        f'<b>{pack["hotspot"]}</b> turns red and blinks when the selected asset\'s '
        "predicted risk is High (amber = Medium, green = Low). "
        "Switch the industry pack in the sidebar to change the mesh.</p>",
        unsafe_allow_html=True,
    )
    st.caption(pack["scope"])
    if pack["id"] == "automotive_powertrain":
        st.info(AUTOMOTIVE_OEM_TRIM_EV_NOTE)

    kind = twin_kind_for(pack["id"])
    batch_states = asset_states_from_predictions(st.session_state.get("predictions") or [])
    live_states = st.session_state.get("live_asset_states") or []
    sources: dict[str, list] = {}
    if batch_states:
        sources["Anomaly & RUL (batch)"] = batch_states
    if live_states:
        sources["Live Connect (streaming)"] = live_states

    def _render_twin(states: list, selected_id: str, height: int = 540) -> None:
        html = build_twin_html(
            states,
            selected_id=selected_id,
            height=height,
            kind=kind,
            hotspot=pack["hotspot"],
            pack_label=pack["label"],
        )
        components.html(html, height=height + 20)

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
        _render_twin(states, selected, 540)
        st.caption(
            "Risk is read from **4. Anomaly & RUL** or **Live Connect** (pick the source above). "
            "Rotate with the mouse; scroll to zoom. Mesh follows the industry pack."
        )
    else:
        st.info(
            "No predictions yet. Run **4. Anomaly & RUL** to drive the twin from real risk, "
            "or preview it with a manual risk below."
        )
        demo_risk = st.selectbox("Preview risk", ["High", "Medium", "Low"], index=0)
        demo_rul = {"High": 3, "Medium": 12, "Low": 26}[demo_risk]
        demo_id = {"plant_rotating": "demo-motor", "aviation_uav_piston": "demo-UAV",
                   "automotive_powertrain": "demo-ENG", "oil_srp": "demo-WELL"}.get(pack["id"], "demo-asset")
        demo = [{"machine_id": demo_id, "risk_level": demo_risk, "predicted_rul_days": demo_rul}]
        _render_twin(demo, demo_id, 540)

    st.caption(
        "Legend — "
        + " · ".join(
            f"**{k}** {v['hex']}" for k, v in RISK_COLORS.items() if k != "Unknown"
        )
        + f" · hotspot: **{pack['hotspot']}**"
    )


def _cad_iframe_height(viewer_h: int = DEFAULT_VIEWER_HEIGHT) -> int:
    return int(viewer_h) + 36


_CAD_VIEWER_COMPONENT = None


def _cad_viewer_component():
    """GuiViewer3D as a Streamlit custom component so part clicks reach Python."""
    global _CAD_VIEWER_COMPONENT
    if _CAD_VIEWER_COMPONENT is None:
        from src.aps_viewer import viewer_bridge_html, write_cad_viewer_component

        # Self-heal a missing or stale index.html (the region-tint needles and the
        # name sanitizer are baked into it) so a Render deploy never serves old JS.
        index = CAD_VIEWER_COMPONENT_DIR / "index.html"
        try:
            current = index.read_text(encoding="utf-8") if index.is_file() else ""
        except OSError:
            current = ""
        if current != viewer_bridge_html():
            try:
                write_cad_viewer_component(CAD_VIEWER_COMPONENT_DIR)
            except OSError:
                pass
        _CAD_VIEWER_COMPONENT = components.declare_component(
            "pdm_cad_viewer",
            path=str(CAD_VIEWER_COMPONENT_DIR),
        )
    return _CAD_VIEWER_COMPONENT


def _store_cad_part_event(event: Any) -> None:
    """Route one component payload: a clicked part, or a region-tint report."""
    region = parse_cad_region_event(event)
    if region is not None:
        st.session_state.cad_region_report = region
        return
    parsed = parse_cad_part_event(event)
    if parsed is None:
        return
    if parsed.get("cleared"):
        st.session_state.cad_selected_part = None
        return
    st.session_state.cad_selected_part = parsed


def _cad_asset_context(sel: dict[str, Any], pack_id: str) -> dict[str, Any]:
    """Card model + mapped sensors + clicked-part hint. Pure read of session state."""
    asset_id = str(sel.get("machine_id") or "asset")
    sensors = latest_asset_sensors(get_active_df(), asset_id)
    part = st.session_state.get("cad_selected_part") or {}
    part_name = str(part.get("name") or "")
    hint = (
        part_sensor_hint(part_name, pack_id=pack_id, available=list(sensors.keys()))
        if part_name
        else None
    )
    risk = str(sel.get("risk_level") or "Unknown")
    card = build_part_card(
        part_name=part_name,
        asset_id=asset_id,
        risk=risk,
        rul_days=sel.get("predicted_rul_days"),
        sensors=sensors,
        hint=hint,
        db_id=part.get("dbId"),
    )
    driver = sensor_driver(get_active_df(), asset_id, risk, sensors=sensors)
    return {"card": card, "sensors": sensors, "hint": hint, "part": part, "driver": driver}


def render_cad_part_card(
    states: list[dict[str, Any]],
    pack_id: str,
    *,
    can_load: bool = False,
) -> dict[str, Any]:
    """Part · Asset · Risk strip + page actions, rendered **above** the viewer.

    Returns the picked asset state plus whether Load CAD model was clicked. The
    mapped-sensor values live in ``render_cad_sensor_grid`` below the viewer so a
    long sensor list never pushes GuiViewer3D off-screen.
    """
    ids = [s["machine_id"] for s in states] or ["asset"]
    strip = st.columns([1.2, 1.7, 1.1, 1.5])
    with strip[0]:
        selected = st.selectbox("Asset", ids, key="aps_asset_pick")
    sel = next(
        (s for s in states if s["machine_id"] == selected),
        {"machine_id": selected, "risk_level": "Unknown"},
    )
    ctx = _cad_asset_context(sel, pack_id)
    card = ctx["card"]
    part = ctx["part"]

    with strip[1]:
        st.markdown(f"**Part** `{card['part_name']}`")
        if part.get("dbId") is not None:
            st.caption(f"Viewer dbId `{part['dbId']}` · click a part in the viewer")
        else:
            st.caption("Click any part in the viewer to name it")
    with strip[2]:
        st.markdown(
            f'**Risk** <span style="color:{risk_theme_hex(card["risk"])};font-weight:700">'
            f'{card["risk"]}</span>',
            unsafe_allow_html=True,
        )
        if card["rul_days"] is not None and card["rul_days"] != "":
            st.caption(f"RUL ≈ {card['rul_days']} days")
    with strip[3]:
        load_clicked = st.button(
            "Load CAD model",
            type="primary",
            key="aps_load_cad_btn",
            disabled=not can_load,
            help=(
                "Fetch an APS token and open this URN in GuiViewer3D."
                if can_load
                else "Add APS credentials and a translated URN below first."
            ),
        )
        if st.button("Open Anomaly & RUL", key="cad_open_anomaly_rul"):
            request_pipeline_page(ANOMALY_RUL_PAGE)
            st.rerun()
    render_cad_driver_line(ctx.get("driver") or {}, card["risk"])
    st.caption(card["honesty"])
    return {"sel": sel, "selected": selected, "load_clicked": bool(load_clicked), **ctx}


def render_cad_driver_line(driver: dict[str, Any], risk: str) -> None:
    """``High driven by exhaust_temp (EGT) → heads / exhaust``, full width under the strip.

    Low / Unknown gets no "driven by" line at all — a healthy asset is not scary.
    """
    if driver.get("line"):
        st.markdown(
            f'<div style="font-size:1.05rem;margin:.15rem 0 .1rem 0">'
            f'<b style="color:{risk_theme_hex(risk)}">{driver["line"]}</b></div>',
            unsafe_allow_html=True,
        )
        z = driver.get("z")
        basis = "z-score" if driver.get("basis") == "z-score" else "median/MAD score"
        st.caption(
            f"Largest |{basis}| among this asset's mapped sensors "
            f"({driver['sensor']} = {format_sensor_value(driver.get('value'))}, "
            f"score {float(z):+.1f} vs the other rows of that column). {DRIVER_HONESTY}"
        )
        return
    if driver.get("message"):
        st.caption(driver["message"])


def render_cad_sensor_grid(ctx: dict[str, Any], *, per_row: int = 4) -> None:
    """Mapped sensors **below** the viewer, ``per_row`` values per row."""
    sensors = ctx.get("sensors") or {}
    hint = ctx.get("hint")
    st.markdown("##### Mapped sensors — latest row for this asset")
    if not sensors:
        st.caption(
            "No mapped sensor values in session yet — clean / map a CSV, then run Anomaly & RUL."
        )
        return
    for row in sensor_grid_rows(sensors, per_row=per_row):
        cols = st.columns(per_row)
        for col, (name, shown) in zip(cols, row):
            with col:
                st.metric(name, shown)
                if hint and name == hint:
                    st.caption("matches the clicked part")


def render_cad_region_assign(
    ctx: dict[str, Any],
    pack_id: str,
    urn: str,
) -> None:
    """Click a part → pick a region → **Assign**. That is what turns Solid1 red.

    Autodesk node names from a STEP export are often ``Solid1`` / ``??????-1-solid1``,
    so name matching has nothing to work with. The saved ``{pack + urn → region →
    dbIds}`` map is what the viewer themes instead.
    """
    part = ctx.get("part") or {}
    driver = ctx.get("driver") or {}
    db_id = part.get("dbId")
    summary = assignment_summary(pack_id, urn)

    st.markdown("##### Click-to-assign CAD regions")
    cols = st.columns([1.6, 1.5, 1.2, 1.4])
    with cols[0]:
        labels = [label for _, label in REGIONS]
        default = region_label(driver.get("region") or "heads_exhaust")
        picked_label = st.selectbox(
            "Region for the clicked part",
            labels,
            index=labels.index(default) if default in labels else 0,
            key="cad_region_pick",
        )
    region_id = next((rid for rid, label in REGIONS if label == picked_label), "other")
    with cols[1]:
        assign_clicked = st.button(
            "Assign selected part to this region",
            key="cad_assign_region_btn",
            type="primary",
            disabled=not (db_id is not None and urn),
            help=(
                "Saves this dbId for this pack + model URN."
                if db_id is not None and urn
                else "Click a part in the viewer (and load a model) first."
            ),
        )
    with cols[2]:
        clear_clicked = st.button(
            "Clear region map",
            key="cad_clear_region_btn",
            disabled=not summary["total"],
            help="Forget every assigned part for this model.",
        )
    with cols[3]:
        st.checkbox(
            "Also tint related regions",
            value=False,
            key="cad_region_include_related",
            help="e.g. a heads/exhaust driver also tints related oil parts when those are assigned.",
        )

    if assign_clicked:
        result = assign_region(pack_id, urn, db_id, region_id)
        st.session_state.cad_region_assign_log = result
        if result.get("saved"):
            st.rerun()
        else:
            st.warning(result.get("message") or "Could not assign that part.")
    if clear_clicked:
        result = clear_region_map(pack_id, urn)
        st.session_state.cad_region_assign_log = {}
        st.session_state.cad_region_clear_log = result
        st.rerun()

    log = st.session_state.get("cad_region_assign_log") or {}
    if log.get("saved"):
        moved = log.get("moved_from")
        st.success(
            log.get("message", "")
            + (f" Moved out of {region_label(moved)}." if moved else "")
        )
    cleared = st.session_state.pop("cad_region_clear_log", None)
    if cleared:
        st.info(cleared.get("message") or "Region map cleared.")

    if db_id is None:
        st.caption(
            "Click a part in GuiViewer3D above — its Autodesk name and dbId appear in the strip, "
            "then assign it here. Assign 4–6 parts (e.g. 3 heads + 2 exhaust stubs) for a "
            "believable red region."
        )
    else:
        st.caption(
            f"Selected `{part.get('display') or part.get('name') or 'part'}` "
            f"(dbId `{db_id}`) → will be saved under **{picked_label}**."
        )
    if summary["total"]:
        st.caption(f"Saved for this model: {summary['text']} ({summary['total']} part(s)).")
    else:
        st.caption(NO_ASSIGNMENT_MSG)


def render_cad_region_note(
    risk: str,
    pack_id: str,
    assigned: Optional[dict[str, Any]] = None,
) -> None:
    """What the region tint did, or why it left the engine its default color."""
    level = normalize_risk(risk)
    report = st.session_state.get("cad_region_report") or {}
    plan = assigned or {}
    tint_word = {"High": "red", "Medium": "orange"}.get(level, "")
    if not tint_word:
        st.caption(
            f"Asset risk is **{level}** — no extra tint. The engine keeps its default "
            "dark gray; only High (red) and Medium (orange) tint the matched CAD region."
        )
        return
    matched = int(report.get("matched") or 0)
    if report and matched:
        names = ", ".join(f"`{n}`" for n in report.get("names") or [])
        what = (
            "CAD part(s) you assigned to the driver region"
            if report.get("source") == "assigned"
            else "CAD node(s) matching this asset's hot sensors"
        )
        st.success(
            f"Region tint: **{matched}** {what} are **{tint_word}**. "
            "Everything else keeps the default color."
            + (f" Matched: {names}." if names else "")
        )
        return
    if report:
        st.warning(
            NO_REGION_MATCH_MSG
            + (" " + NO_ASSIGNMENT_MSG if not plan.get("dbIds") else "")
        )
        return
    if plan.get("dbIds"):
        used = ", ".join(f"**{region_label(r)}**" for r in plan.get("regions_used") or [])
        st.caption(
            f"Asset risk is **{level}** — the {len(plan['dbIds'])} part(s) you assigned to "
            f"{used or 'the driver region'} turn {tint_word} as soon as the model loads; "
            "every other solid stays default dark gray."
        )
        return
    keywords = ", ".join(f"`{k}`" for k in region_hint_keywords(pack_id)[:8])
    st.caption(
        f"Asset risk is **{level}** — CAD nodes whose Autodesk name matches a mapped sensor "
        f"turn {tint_word}; the rest stay default dark gray. Name needles: {keywords}, …"
        + (f" {plan['message']}" if plan.get("message") else "")
    )


def _seed_cad_urn_widgets(pack_id: str) -> dict:
    """Prefill URN widgets BEFORE they are created. Streamlit forbids writing the key after."""
    env_urn = normalize_model_urn(aps_model_urn())
    resolved = resolve_cad_urn(pack_id, st.session_state.get("aps_urn") or "")
    seed = resolved.get("urn") or ""
    if not st.session_state.get("aps_urn") and seed:
        st.session_state.aps_urn = seed
        st.session_state.aps_urn_origin = resolved.get("source") or "saved"
    if "aps_urn_input" not in st.session_state and seed:
        st.session_state.aps_urn_input = seed
        st.session_state.aps_urn_origin = resolved.get("source") or "saved"
    # Normalize a paste (viewer.autodesk.com URL → dXJu…) before the text_input is built.
    if "aps_urn_input" in st.session_state:
        raw = str(st.session_state.aps_urn_input or "")
        extracted, meta = extract_model_urn(raw)
        st.session_state.aps_urn_extract_meta = meta
        if extracted and extracted != raw.strip():
            st.session_state.aps_urn_input = extracted
        if meta.get("public_viewer") and extracted:
            st.session_state.aps_urn_from_public_viewer = True
            st.session_state.aps_public_viewer_urn = extracted
        elif extracted and extracted != st.session_state.get("aps_public_viewer_urn"):
            st.session_state.aps_urn_from_public_viewer = False
            st.session_state.aps_public_viewer_urn = ""
    return resolved


def _seed_zip_root_widget() -> None:
    """Prefill ZIP rootFilename BEFORE the text_input widget exists this run.

    Streamlit forbids writing ``aps_zip_root_filename`` after ``st.text_input``
    with that key. Apply a pending value first, then infer from the already
    uploaded zip (namelist .STEP/.stp, else ``Something.STEP.zip``).
    """
    pending = str(st.session_state.pop(PENDING_ZIP_ROOT, None) or "").strip()
    uploaded = st.session_state.get("aps_cad_file_uploader")
    auto = ""
    sig = ""
    if uploaded is not None:
        name = str(getattr(uploaded, "name", "") or "")
        if is_cad_zip_name(name):
            try:
                data = uploaded.getvalue()
            except Exception:
                data = b""
            auto = resolve_zip_root_filename(name, data, "")
            sig = f"{name}:{len(data)}"
    # A queued name is only for the current zip. A new file must not keep the old STEP name.
    if sig and st.session_state.get("aps_zip_root_sig") != sig:
        chosen = auto
    else:
        chosen = pending or auto
    if not chosen:
        return
    current = str(st.session_state.get("aps_zip_root_filename") or "").strip()
    if sig and st.session_state.get("aps_zip_root_sig") != sig:
        st.session_state.aps_zip_root_sig = sig
        st.session_state.aps_zip_root_filename = chosen
    elif not current:
        st.session_state.aps_zip_root_filename = chosen
        if sig:
            st.session_state.aps_zip_root_sig = sig


def _persist_pack_urn(pack_id: str, urn: str, *, source: str, public_viewer: bool) -> dict:
    result = save_urn_for_pack(pack_id, urn, source=source, public_viewer=public_viewer)
    st.session_state.aps_save_log = result
    if result.get("saved"):
        st.session_state.aps_urn_origin = source
        st.session_state.aps_urn_from_public_viewer = False
        st.session_state.aps_public_viewer_urn = ""
    return result


def _delete_pack_urn(pack_id: str) -> dict:
    result = delete_urn_for_pack(pack_id)
    st.session_state.aps_delete_log = result
    if result.get("ok"):
        st.session_state.aps_save_log = {}
        st.session_state.aps_urn_origin = "session"
    return result


def _render_cad_viewer(
    *,
    token: str,
    urn: str,
    asset: str,
    risk: str,
    public_viewer: bool,
    needles: Optional[list[str]] = None,
    region_ids: Optional[list[int]] = None,
) -> None:
    """GuiViewer3D + the node-name needles / assigned dbIds that decide the tint."""
    hints = list(needles or [])
    assigned = coerce_db_ids(region_ids)
    html = build_viewer_html(
        token,
        urn,
        asset=asset,
        risk=risk,
        height=DEFAULT_VIEWER_HEIGHT,
        public_viewer=public_viewer,
        needles=hints,
        region_ids=assigned,
    )
    st.session_state.aps_viewer_html = html
    st.session_state.aps_viewer_urn = urn
    st.session_state.aps_viewer_token = token
    st.session_state.aps_viewer_at = datetime.now().timestamp()
    event = None
    try:
        event = _cad_viewer_component()(
            token=token,
            urn=normalize_model_urn(urn),
            asset=str(asset),
            risk=str(risk if risk in ("High", "Medium", "Low") else "Unknown"),
            height=DEFAULT_VIEWER_HEIGHT,
            public=bool(public_viewer),
            extras=list(VIEWER_EXTRA_EXTENSIONS),
            viewer_err={str(k): v for k, v in VIEWER_ERROR_CODES.items()},
            needles=hints,
            region_ids=assigned,
            default=None,
            key="cad_gui_viewer",
        )
    except Exception:
        components.html(html, height=_cad_iframe_height())
        event = None
    _store_cad_part_event(event)


# ── CAD Twin — Autodesk APS (Upgrade 3) ───────────────────────────────────────
def _cad_urn_state(pack_id: str) -> dict[str, Any]:
    """URN + public-viewer verdict read from session state, before any widget.

    GuiViewer3D renders near the top of the page — above the CAD-source panel
    that owns ``key="aps_urn_input"`` — so the URN has to come from the session
    value ``_seed_cad_urn_widgets`` already normalized this run.
    """
    raw = str(st.session_state.get("aps_urn_input") or st.session_state.get("aps_urn") or "")
    extracted, meta = extract_model_urn(raw)
    urn = extracted or normalize_model_urn(raw)
    st.session_state.aps_urn = urn
    public_paste = bool(
        meta.get("public_viewer")
        or meta.get("a360_protected")
        or meta.get("block_load")
        or looks_like_public_viewer(raw)
        or looks_like_public_viewer(urn)
        or (
            st.session_state.get("aps_urn_from_public_viewer")
            and urn
            and urn == st.session_state.get("aps_public_viewer_urn")
        )
    )
    env_urn = normalize_model_urn(aps_model_urn())
    return {
        "raw": raw,
        "urn": urn,
        "meta": meta,
        "public_paste": public_paste,
        "env_urn": env_urn,
        "seed_urn": env_urn or saved_urn_for_pack(pack_id),
    }


def _render_cad_viewer_section(
    *,
    ok: bool,
    urn_state: dict[str, Any],
    selected: str,
    risk: str,
    load_clicked: bool,
    pack_id: str,
    status: dict[str, Any],
    needles: list[str],
    region_ids: Optional[list[int]] = None,
) -> None:
    """The viewer slot itself — GuiViewer3D, or an honest placeholder."""
    urn = str(urn_state.get("urn") or "").strip()
    if not ok:
        components.html(cad_placeholder_html(status=status, height=280), height=300)
        return
    if urn_state.get("public_paste"):
        components.html(cad_placeholder_html(status=status, height=280), height=300)
        st.error(f"**{A360_PUBLIC_URN_ERROR}**")
        st.caption(
            (urn_state.get("meta") or {}).get("warning")
            or (PUBLIC_VIEWER_WARNING if urn else PUBLIC_VIEWER_NO_URN_WARNING)
        )
        st.session_state.aps_viewer_html = ""
        st.session_state.aps_viewer_urn = ""
        if load_clicked:
            _persist_pack_urn(pack_id, urn, source="load", public_viewer=True)
        return
    if not urn:
        components.html(cad_placeholder_html(status=status, height=280), height=300)
        st.warning(
            "Open **CAD source** below: choose a CAD file and click **Translate to SVF / get "
            "URN**, paste a translated SVF URN from **your** bucket, or set APS_MODEL_URN on Render."
        )
        return

    cache_urn = str(st.session_state.get("aps_viewer_urn") or "")
    cache_html = str(st.session_state.get("aps_viewer_html") or "")
    cache_at = float(st.session_state.get("aps_viewer_at") or 0)
    cache_fresh = bool(
        cache_html and cache_urn == urn and (datetime.now().timestamp() - cache_at) < 50 * 60
    )
    translate_log = st.session_state.get("aps_translate_log") or {}
    translated = normalize_model_urn(str(translate_log.get("urn") or ""))
    force = bool(st.session_state.get("aps_force_load"))
    should_auto = urn == urn_state.get("seed_urn") or force or (
        bool(translate_log.get("ok")) and urn == translated
    )
    if load_clicked or force:
        st.session_state.aps_force_load = False

    if load_clicked or (should_auto and not cache_fresh):
        try:
            with st.spinner("Fetching APS token and loading GuiViewer3D…"):
                token = get_access_token()
                _render_cad_viewer(
                    token=token["access_token"],
                    urn=urn,
                    asset=selected,
                    risk=risk,
                    public_viewer=False,
                    needles=needles,
                    region_ids=region_ids,
                )
            persist = _persist_pack_urn(
                pack_id, urn, source="load" if load_clicked else "auto", public_viewer=False
            )
            if persist.get("saved"):
                st.caption(persist.get("message") or saved_urn_caption(pack_id))
            elif persist.get("reason") == "public_viewer":
                st.error(f"**{A360_PUBLIC_URN_ERROR}**")
        except Exception as exc:
            st.error(f"APS error: {exc}")
            if INSUFFICIENT_SCOPE_HINT in str(exc):
                st.info(INSUFFICIENT_SCOPE_HINT)
        return
    if cache_fresh:
        cached_token = str(st.session_state.get("aps_viewer_token") or "")
        if cached_token:
            _render_cad_viewer(
                token=cached_token,
                urn=urn,
                asset=selected,
                risk=risk,
                public_viewer=False,
                needles=needles,
                region_ids=region_ids,
            )
        else:
            components.html(cache_html, height=_cad_iframe_height())
        save_info = st.session_state.get("aps_save_log") or {}
        if save_info.get("saved") and save_info.get("urn") == urn:
            st.caption(save_info.get("message") or saved_urn_caption(pack_id))
        return
    components.html(cad_placeholder_html(status=status, height=280), height=300)
    st.info("URN is ready — click **Load CAD model** above to open it in GuiViewer3D.")


def _render_cad_source_panel(
    pack: dict[str, Any],
    pack_id: str,
    *,
    ok: bool,
    urn_state: dict[str, Any],
) -> bool:
    """Uploader → translate → URN → save/delete. Returns True when we must rerun.

    Lives **below** the viewer so admin controls never push the engine off-screen.
    Write order matters: anything that seeds ``aps_zip_root_filename`` or
    ``aps_urn_input`` must run before that widget is created in this function.
    """
    needs_rerun = False
    if ok:
        st.markdown("**Choose CAD file → translate → URN**")
        cad_file = st.file_uploader(
            "CAD file for Model Derivative",
            type=list(CAD_UPLOAD_EXTENSIONS),
            key="aps_cad_file_uploader",
            help=(
                f"Up to {MAX_CAD_UPLOAD_MB} MB (Streamlit maxUploadSize). "
                "Common Model Derivative inputs (Revit, Fusion, IFC, Inventor, SolidWorks, STEP, "
                "Navisworks, DWG, OBJ/STL, ZIP assemblies, …). Exotic kernels can still fail on Autodesk. "
                f"If a ~200 MB STEP still fails, {CAD_SIZE_ZIP_FALLBACK}"
            ),
        )
        st.caption(
            f"Upload limit is **{MAX_CAD_UPLOAD_MB} MB**. A 207 MB STEP should fit. "
            f"If the browser or Render still rejects it, {CAD_SIZE_ZIP_FALLBACK} "
            "Zipped STEP: ZIP root auto-fills from the `.step` / `.stp` inside the archive "
            "(or from a name like `Engine Rotax 912 ULS.STEP.zip`). You do not type it."
        )
        root_filename = ""
        cad_name = str(cad_file.name or "") if cad_file is not None else ""
        if cad_file is not None and is_cad_zip_name(cad_name):
            zip_bytes = cad_file.getvalue()
            auto_root = pick_zip_cad_root(zip_bytes) or guess_zip_root_from_upload_name(cad_name)
            zip_sig = f"{cad_name}:{len(zip_bytes)}"
            # Prefill BEFORE the widget exists — Streamlit forbids writing the key after.
            # Pending key is applied at the top of this page via _seed_zip_root_widget.
            if st.session_state.get("aps_zip_root_sig") != zip_sig:
                st.session_state.aps_zip_root_sig = zip_sig
                if auto_root:
                    st.session_state[PENDING_ZIP_ROOT] = auto_root
                    st.session_state.aps_zip_root_filename = auto_root
            elif auto_root and not str(st.session_state.get("aps_zip_root_filename") or "").strip():
                st.session_state[PENDING_ZIP_ROOT] = auto_root
                st.session_state.aps_zip_root_filename = auto_root
            if auto_root:
                st.caption(f"Detected ZIP root: `{auto_root}`")
            else:
                st.warning(ZIP_ROOT_MISSING_MSG)
            root_filename = st.text_input(
                "ZIP root filename (auto-filled for zipped STEP — leave as-is)",
                key="aps_zip_root_filename",
                help=(
                    "The file inside the zip Model Derivative should open "
                    "(e.g. Engine Rotax 912 ULS.STEP). Auto-filled from the archive "
                    "or from a name like Something.STEP.zip. Translate still infers "
                    "this if the box is empty."
                ),
            )
        if cad_file is not None:
            kind, size_msg = cad_size_issue(getattr(cad_file, "size", 0) or 0)
            if kind == "error":
                st.error(size_msg)
            elif kind == "warn":
                st.warning(size_msg)
        if st.button("Translate to SVF / get URN", type="primary", key="aps_translate_btn"):
            if cad_file is None:
                st.warning("Choose a CAD file first.")
            else:
                payload = cad_file.getvalue()
                kind, size_msg = cad_size_issue(len(payload))
                cad_name = cad_file.name or "model"
                root = resolve_zip_root_filename(cad_name, payload, root_filename)
                if kind == "error":
                    st.error(size_msg)
                    st.session_state.aps_translate_log = {
                        "ok": False,
                        "phase": "error",
                        "urn": "",
                        "message": size_msg,
                    }
                elif is_cad_zip_name(cad_name) and not root:
                    st.error(ZIP_ROOT_MISSING_MSG)
                    st.session_state.aps_translate_log = {
                        "ok": False,
                        "phase": "error",
                        "urn": "",
                        "message": ZIP_ROOT_MISSING_MSG,
                    }
                else:
                    with st.status("Translating CAD with Autodesk…", expanded=True) as box:
                        def _on_status(phase: str, detail: str) -> None:
                            st.write(f"**{phase}:** {detail}")

                        result = translate_cad_bytes(
                            cad_name,
                            payload,
                            root_filename=root,
                            on_status=_on_status,
                        )
                        st.session_state.aps_translate_log = result
                        if result.get("urn"):
                            # Widget with key aps_urn_input is created below this button.
                            st.session_state.aps_urn = result["urn"]
                            st.session_state.aps_urn_input = result["urn"]
                            st.session_state.aps_urn_from_public_viewer = False
                            st.session_state.aps_viewer_html = ""
                            st.session_state.aps_viewer_urn = ""
                        if result.get("ok") and result.get("urn"):
                            persist = _persist_pack_urn(
                                pack_id, result["urn"], source="translate", public_viewer=False
                            )
                            result = dict(result)
                            result["save"] = persist
                            st.session_state.aps_translate_log = result
                            st.session_state.aps_force_load = True
                            box.update(label="Translation succeeded", state="complete")
                            # The viewer slot is above this panel; rerun so it loads now.
                            needs_rerun = True
                        elif result.get("phase") == "timeout":
                            box.update(label="Still translating on Autodesk", state="running")
                        else:
                            box.update(label="Translation failed", state="error")

        log = st.session_state.get("aps_translate_log") or {}
        if log.get("message"):
            phase = str(log.get("phase") or "")
            if log.get("ok"):
                st.success(log["message"])
                save_info = log.get("save") or st.session_state.get("aps_save_log") or {}
                if save_info.get("saved"):
                    st.info(save_info.get("message") or saved_urn_caption(pack_id))
            elif phase == "timeout":
                st.warning(log["message"])
            else:
                st.error(log["message"])
                if INSUFFICIENT_SCOPE_HINT in str(log.get("message") or ""):
                    st.info(INSUFFICIENT_SCOPE_HINT)

    st.text_input(
        "Translated model URN (base64) or viewer.autodesk.com URL",
        key="aps_urn_input",
        help=(
            "Paste a dXJu URN, an Autodesk viewer URL (we extract the URN), "
            "or use Translate above. APS_MODEL_URN on Render overrides the saved pack file."
        ),
    )
    urn = str(urn_state.get("urn") or "")
    if urn_state.get("public_paste"):
        st.error(f"**{A360_PUBLIC_URN_ERROR}**")
        st.caption(
            (urn_state.get("meta") or {}).get("warning")
            or (PUBLIC_VIEWER_WARNING if urn else PUBLIC_VIEWER_NO_URN_WARNING)
        )

    save_col, delete_col = st.columns(2)
    with save_col:
        save_clicked = st.button("Save URN for this pack", key="aps_save_urn_btn")
    with delete_col:
        delete_clicked = st.button("Delete saved URN", key="aps_delete_urn_btn")

    if save_clicked:
        if urn_state.get("public_paste"):
            persist = _persist_pack_urn(pack_id, urn, source="save", public_viewer=True)
            st.warning(persist.get("message") or PUBLIC_VIEWER_WARNING)
        elif not urn:
            st.warning(
                "Nothing to save — paste a dXJu URN from your bucket or translate a CAD file first."
            )
        else:
            persist = _persist_pack_urn(pack_id, urn, source="save", public_viewer=False)
            if persist.get("saved"):
                st.info(persist.get("message") or saved_urn_caption(pack_id))
            else:
                st.warning(persist.get("message") or "Could not save URN.")
    if delete_clicked:
        gone = _delete_pack_urn(pack_id)
        if gone.get("had_urn"):
            st.success(gone.get("message") or "Deleted saved URN on this server.")
        else:
            st.info(gone.get("message") or "No saved URN for this pack on this server.")

    if urn_state.get("env_urn"):
        st.caption(
            "Render `APS_MODEL_URN` is set and overrides the pack JSON as the default URN. "
            "We do not write Render env from this app."
        )
    elif urn_state.get("seed_urn"):
        st.caption(saved_urn_caption(pack_id))
    else:
        st.caption(
            f"No saved URN for {pack['short']} on this server yet. "
            "Translate a CAD file or click **Save URN for this pack**."
        )
    return needs_rerun


def page_cad_twin():
    st.markdown('<p class="main-header">CAD Twin (Autodesk APS)</p>', unsafe_allow_html=True)
    pack = active_pack()
    pack_id = active_pack_id()
    st.markdown(
        f'<p class="sub-header">Real translated CAD for <b>{pack["label"]}</b> '
        "(Revit / Fusion / IFC / STEP → SVF). The engine keeps its <b>default dark gray</b>. "
        "Click a part, pick Heads/exhaust · Oil · Rotating/crank · Intake · Other, and assign it — "
        "High tints those saved dbIds <b>red</b>, Medium <b>orange</b>, Low none. "
        "Name-matched nodes still tint when the STEP has real names; Solid1 + an empty map "
        "never reddens the whole engine. GuiViewer3D full toolbar; the light-blue selection "
        "outline and Explode are the viewer's own. Last working URN is saved on this server "
        "per industry pack. Credentials are runtime secrets.</p>",
        unsafe_allow_html=True,
    )

    ok, msg = aps_available()
    _seed_cad_urn_widgets(pack_id)
    _seed_zip_root_widget()
    urn_state = _cad_urn_state(pack_id)
    status = cad_slot_status(urn_state["urn"], pack_id=pack_id)

    if ok:
        st.caption(f"APS: {msg}")
    else:
        st.info(
            "APS credentials are not on this deploy yet. The **3D Twin** (pack mesh) is the "
            "credential-free twin. Add `APS_CLIENT_ID` + `APS_CLIENT_SECRET` (and optional "
            "`APS_MODEL_URN`) on Render, reload, and this tile loads."
        )
        st.caption(f"Status: {msg}")

    states = asset_states_from_predictions(st.session_state.get("predictions") or []) or (
        st.session_state.get("live_asset_states") or []
    )
    actions = render_cad_part_card(
        states,
        pack_id,
        can_load=bool(ok and urn_state["urn"] and not urn_state["public_paste"]),
    )
    risk = str(actions["card"]["risk"])
    needles = region_hint_keywords(pack_id, available=list((actions["sensors"] or {}).keys()))
    driver = actions.get("driver") or {}
    assigned = assigned_theme_plan(
        pack_id,
        urn_state["urn"],
        risk,
        driver.get("region") or "",
        related=driver.get("related") or [],
        include_related=bool(st.session_state.get("cad_region_include_related")),
    )

    _render_cad_viewer_section(
        ok=ok,
        urn_state=urn_state,
        selected=str(actions["selected"]),
        risk=risk,
        load_clicked=bool(actions["load_clicked"]),
        pack_id=pack_id,
        status=status,
        needles=needles,
        region_ids=assigned["dbIds"],
    )
    render_cad_region_assign(actions, pack_id, urn_state["urn"])
    render_cad_region_note(risk, pack_id, assigned)
    render_cad_sensor_grid(actions)

    with st.expander("CAD source — upload, translate, URN, save", expanded=not urn_state["urn"]):
        if _render_cad_source_panel(pack, pack_id, ok=ok, urn_state=urn_state):
            st.rerun()

    with st.expander("How to generate a URN (APS app + this page)", expanded=not ok):
        st.markdown(
            "1. Create an app at [aps.autodesk.com](https://aps.autodesk.com) with "
            "**Model Derivative** and **Data Management** APIs enabled. Copy **Client ID** + **Client Secret**.\n"
            "2. On Render → Environment, set `APS_CLIENT_ID` and `APS_CLIENT_SECRET` "
            "(optional `APS_MODEL_URN` if you already have a translated model — that env var is the "
            "override that survives Render redeploys).\n"
            "3. **Generate a URN here:** open **CAD source** above, choose a CAD file → "
            "**Translate to SVF / get URN**. "
            "That puts the model in **your** OSS bucket (using `APS_CLIENT_ID` / `APS_CLIENT_SECRET` on Render). "
            "viewer.autodesk.com is Autodesk’s public website — it does **not** put the file in your bucket; "
            "there is no “move from viewer to my bucket” button. "
            "We then save that URN for this pack "
            f"({pack['short']}) in a JSON file on this server.\n"
            "4. Pasting a [viewer.autodesk.com](https://viewer.autodesk.com) link only extracts the "
            "`dXJu…` URN. It will **not** load with your app token unless that object is in your bucket. "
            "Do not rely on saving a public-viewer URN.\n"
            "5. The 2-legged token needs `data:write` / `data:create` and `bucket:create` / "
            "`bucket:read` in addition to `viewables:read`.\n"
            "6. **Save URN for this pack** / **Delete saved URN** write or clear `data/cad_urns.json` "
            "for this industry pack. Close the tab and come back — CAD Twin auto-loads the saved URN "
            "while this service is up. After a Render **redeploy** the disk is wiped; set `APS_MODEL_URN` too."
        )
        st.caption(INSUFFICIENT_SCOPE_HINT)
        st.markdown(
            f"Need a tiny test file? Autodesk’s Model Derivative tutorial includes a sample Inventor "
            f"part (`box.ipt`): [prep a file for the viewer]({APS_SAMPLE_CAD_URL})."
        )
        st.caption(
            "Viewer is **GuiViewer3D** (not headless Viewer3D): Home, Fit, Pan, Zoom, Orbit, "
            "First Person / BimWalk, Measure, Section, Explode (slider), View cube, left Views "
            "(DocumentBrowser), Model browser, Properties, Settings. Markup loads when "
            "`Autodesk.Viewing.MarkupsGui` is in the viewer library — no extra paid API. "
            "Selection highlight is the viewer's default outline; explode still works."
        )
        st.caption(
            "Autodesk node names that come back empty, control-byte or mojibake "
            "(`?????? ??????-1-solid1`) are shown as `Solid-1` / `Solid-{dbId}`. Asset ids are never "
            "rewritten. Region tint themes only matching dbIds — with `Solid1`-only names nothing "
            "matches and the model stays default gray."
        )


# ── Live Connect (Layer 4) ────────────────────────────────────────────────────
def _live_body():
    cfg = dict(st.session_state.get("live_cfg") or {})
    source = st.session_state.get("live_source")
    if st.session_state.get("live_running"):
        tick = int(st.session_state.get("live_tick") or 0)
        batch = pd.DataFrame()
        if source == "poll":
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
        elif source in ("mqtt", "opcua"):
            src = get_source(st.session_state.get("live_conn_id"))
            if src is None:
                st.session_state.live_running = False
                st.warning("Live connection was lost (server restarted?). Press Start live again.")
            else:
                try:
                    batch = src.drain() if source == "mqtt" else src.poll_once()
                except Exception as exc:
                    st.error(f"{source.upper()} read failed: {exc}")
                if getattr(src, "error", None):
                    st.caption(f"{source.upper()} note: {src.error}")
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

    source_labels = ["Simulator", "Poll CSV URL"]
    mq_ok, mq_msg = mqtt_available()
    op_ok, op_msg = opcua_available()
    if mq_ok:
        source_labels.append("MQTT (live)")
    if op_ok:
        source_labels.append("OPC-UA (live)")
    label_to_code = {
        "Simulator": "sim",
        "Poll CSV URL": "poll",
        "MQTT (live)": "mqtt",
        "OPC-UA (live)": "opcua",
    }
    code_to_label = {v: k for k, v in label_to_code.items()}
    cur_label = code_to_label.get(st.session_state.get("live_source", "sim"), "Simulator")
    if cur_label not in source_labels:
        cur_label = "Simulator"
    source = st.radio("Source", source_labels, horizontal=True, index=source_labels.index(cur_label))
    st.session_state.live_source = label_to_code[source]
    code = st.session_state.live_source
    st.caption(f"Real-protocol sources — MQTT: {'✓ ' + mq_msg if mq_ok else 'unavailable'} · OPC-UA: {'✓ ' + op_msg if op_ok else 'unavailable'}")

    cfg = dict(st.session_state.get("live_cfg") or {})
    if code == "sim":
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
    elif code == "poll":
        cfg["url"] = st.text_input(
            "CSV feed URL", value=cfg.get("url", "http://localhost:8000/sensor_readings.csv")
        )
        cfg["poll_n"] = st.slider("Rows per poll", 5, 200, int(cfg.get("poll_n", 30)))
        cfg.setdefault("offset", 0)
    elif code == "mqtt":
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            cfg["host"] = st.text_input("Broker host", value=cfg.get("host", "127.0.0.1"))
        with c2:
            cfg["port"] = int(st.number_input("Port", 1, 65535, int(cfg.get("port", 1883))))
        with c3:
            cfg["topic"] = st.text_input("Topic (wildcards OK)", value=cfg.get("topic", "pdm/sensors/#"))
        c4, c5 = st.columns(2)
        with c4:
            cfg["username"] = st.text_input("Username (optional)", value=cfg.get("username", ""))
        with c5:
            cfg["password"] = st.text_input(
                "Password (optional)", value=cfg.get("password", ""), type="password"
            )
        st.caption(
            'Expects JSON payloads like `{"machine_id":"M-001","temperature":72,"vibration":2.1,'
            '"pressure":100,"rpm":1500}`.'
        )
    elif code == "opcua":
        cfg["endpoint"] = st.text_input(
            "OPC-UA endpoint", value=cfg.get("endpoint", "opc.tcp://127.0.0.1:4840/freeopcua/server/")
        )
        cfg["node_map_text"] = st.text_area(
            "Node map — `sensor=node_id`, comma-separated",
            value=cfg.get("node_map_text", "temperature=ns=2;i=2, vibration=ns=2;i=3"),
            height=80,
        )
        cfg["machine_id"] = st.text_input("Machine id label", value=cfg.get("machine_id", "opcua-asset"))
        st.caption("Each poll reads the listed node IDs. Example node id: `ns=2;i=2`.")
    st.session_state.live_cfg = cfg

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        if st.button("▶ Start live", type="primary", use_container_width=True):
            stop_source(st.session_state.get("live_conn_id"))
            st.session_state.live_conn_id = None
            started = True
            try:
                if code == "poll":
                    cfg["offset"] = 0
                    st.session_state.live_cfg = cfg
                elif code == "mqtt":
                    st.session_state.live_conn_id = start_mqtt(cfg)
                elif code == "opcua":
                    ocfg = dict(cfg)
                    ocfg["node_map"] = parse_node_map(cfg.get("node_map_text", ""))
                    if not ocfg["node_map"]:
                        raise ValueError("Provide at least one `sensor=node_id` in the node map.")
                    st.session_state.live_conn_id = start_opcua(ocfg)
            except Exception as exc:
                started = False
                st.error(f"Could not start {code} source: {exc}")
            if started:
                st.session_state.live_buffer = None
                st.session_state.live_tick = 0
                st.session_state.live_running = True
                st.rerun()
    with ctrl2:
        if st.button("⏸ Stop", use_container_width=True):
            st.session_state.live_running = False
            stop_source(st.session_state.get("live_conn_id"))
            st.session_state.live_conn_id = None
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
    st.markdown('<p class="main-header">7. Dashboard</p>', unsafe_allow_html=True)
    pack = active_pack()
    st.markdown(
        f'<p class="sub-header">Reliability board for <b>{pack["label"]}</b> — KPI strip, charts, '
        "insights, 3D twin, CAD slot. Toggle tiles like a Power BI canvas, then export HTML.</p>",
        unsafe_allow_html=True,
    )
    st.caption(pack["scope"])

    if st.session_state.get("dashboard_tiles") is None:
        st.session_state.dashboard_tiles = default_tile_state()

    tile_cols = st.columns(len(TILE_SPECS))
    enabled: list[str] = []
    for col, spec in zip(tile_cols, TILE_SPECS):
        with col:
            on = st.checkbox(
                spec["label"],
                value=bool(st.session_state.dashboard_tiles.get(spec["id"], True)),
                key=f"dash_tile_{spec['id']}",
                help=spec["hint"],
            )
            st.session_state.dashboard_tiles[spec["id"]] = on
            if on:
                enabled.append(spec["id"])

    df = get_active_df()
    preds = list(st.session_state.get("predictions") or [])
    bundle = compute_pack_kpis(active_pack_id(), df if df is not None else pd.DataFrame(), preds)
    st.session_state.pack_kpis = bundle
    insight_cards = list(st.session_state.get("business_insights") or [])
    if not insight_cards:
        insight_cards = insight_cards_from_kpis(bundle)

    live_states = st.session_state.get("live_asset_states") or []
    twin_source = preds or live_states
    selected_id = None
    if twin_source:
        selected_id = str(twin_source[0].get("machine_id"))
    twin_html = board_twin_html(twin_source, active_pack_id(), selected_id=selected_id, height=480)

    cad_status = cad_slot_status(st.session_state.get("aps_urn") or "", pack_id=active_pack_id())
    cad_html = ""
    cad_asset = selected_id or "asset"
    cad_risk = "Unknown"
    if twin_source:
        cad_risk = str(twin_source[0].get("risk_level") or "Unknown")
    if cad_status.get("ready"):
        try:
            token = get_access_token()
            cad_html = board_cad_html(
                token=token["access_token"],
                urn=cad_status["urn"],
                asset=cad_asset,
                risk=cad_risk,
                height=DEFAULT_VIEWER_HEIGHT,
                needles=region_hint_keywords(active_pack_id()),
            )
        except Exception as exc:
            cad_status = dict(cad_status)
            cad_status["ready"] = False
            cad_status["message"] = f"APS token failed: {exc}"

    pack_figs = board_pack_charts(bundle)
    folder = get_graph_folder()
    saved_figs = [entry["fig"] for entry in folder.values()] if folder else []
    chart_figs = pack_figs + saved_figs

    if "kpis" in enabled:
        st.subheader("KPI strip")
        kpi_cols = st.columns(max(len(bundle["kpis"]), 1))
        for col, kpi in zip(kpi_cols, bundle["kpis"]):
            with col:
                st.metric(kpi["label"], f"{kpi['value']} {kpi['unit']}".strip(), help=kpi.get("hint"))
        st.caption(bundle.get("narrative") or "")

    if "charts" in enabled:
        st.subheader("Charts")
        if chart_figs:
            cols = st.columns(2)
            for i, fig in enumerate(chart_figs):
                with cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True, key=f"board_chart_{i}")
        else:
            st.info("No charts yet. Run **4. Anomaly & RUL** or generate a view on **5. Charts**.")

    if "insights" in enabled:
        st.subheader("Insights")
        if insight_cards:
            for item in insight_cards[:8]:
                st.markdown(f"**{item.get('title')}** — `{item.get('severity')}`")
                st.caption(item.get("message") or "")
        else:
            st.info("Open **6. Insights** after Anomaly & RUL for the inspect list.")

    if "twin3d" in enabled:
        st.subheader("3D twin")
        components.html(twin_html, height=500)
        st.caption(f"Pack mesh: {pack['short']} · hotspot {pack['hotspot']}")

    if "cad" in enabled:
        st.subheader("CAD twin (APS)")
        if cad_status.get("ready") and cad_html:
            components.html(cad_html, height=_cad_iframe_height())
            if cad_status.get("source") == "saved":
                st.caption(saved_urn_caption(active_pack_id()))
            elif cad_status.get("env_override"):
                st.caption(
                    "Using Render APS_MODEL_URN override. Saved pack URN is unused while that env is set."
                )
        else:
            components.html(cad_placeholder_html(status=cad_status, height=280), height=300)

    export = compose_dashboard_html(
        tiles=enabled,
        pack_id=active_pack_id(),
        kpi_bundle=bundle,
        insight_cards=insight_cards,
        chart_figs=chart_figs,
        twin_html=twin_html,
        cad_html=cad_html,
        cad_status=cad_status,
        title=f"{pack['short']} reliability dashboard",
    )
    st.download_button(
        "Export dashboard (HTML) — KPIs + charts + 3D" + (" + CAD" if cad_status.get("ready") else " (CAD slot placeholder)"),
        export,
        file_name=f"reliability_dashboard_{datetime.now().strftime('%Y%m%d')}.html",
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
            pack_kpis=st.session_state.get("pack_kpis")
            or compute_pack_kpis(active_pack_id(), get_active_df(), st.session_state.predictions),
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
    apply_pending_industry_pack()
    apply_pending_pipeline_page()
    st.sidebar.markdown(f"## {config.APP_TITLE}")
    st.sidebar.caption(config.TAGLINE)
    st.sidebar.markdown("---")
    render_pack_sidebar()
    st.sidebar.markdown("---")

    pages = {
        "1. Upload & Clean": page_upload_clean,
        "2. Joins": page_data_integration,
        "3. Map sensors": page_map_sensors,
        "4. Anomaly & RUL": page_ml_predictions,
        "5. Charts": page_explore_graphs,
        "6. Insights": page_business_insights,
        "3D Twin": page_twin_3d,
        "CAD Twin (APS)": page_cad_twin,
        "Live Connect": page_live_connect,
        "7. Dashboard": page_dashboard_builder,
        "Email Report": page_email_report,
        "Ask": page_ai_assistant,
        "SQL lab": page_dwdm_sql,
    }

    selection = st.sidebar.radio("Pipeline", list(pages.keys()), key="pipeline_page")
    render_gemini_sidebar()
    st.sidebar.markdown("---")
    df = get_active_df()
    if df is not None:
        st.sidebar.success(f"Data: {len(df):,} rows")
        status = mapping_status(df)
        pack = active_pack()
        st.sidebar.caption(
            f"Pack: {pack['short']} · Sensors: {status['sensor_count']} · "
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
