#!/usr/bin/env python3
"""Build PdM_SIH26054_Student_Viva_Guide.pdf from the Streamlit control inventory.

Regenerate:
    python docs/generate_viva_guide.py
Copies the same bytes to docs/ and /opt/cursor/artifacts/ when that folder exists.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DOCS = ROOT / "docs" / "PdM_SIH26054_Student_Viva_Guide.pdf"
OUT_ARTIFACT = Path("/opt/cursor/artifacts/PdM_SIH26054_Student_Viva_Guide.pdf")

NAVY = colors.HexColor("#1a1a2e")
SLATE = colors.HexColor("#2c3e50")
AMBER = colors.HexColor("#c0392b")
TEAL = colors.HexColor("#1a5276")
ROW_ALT = colors.HexColor("#f4f6f8")
HEAD_BG = colors.HexColor("#1a1a2e")
WARN_BG = colors.HexColor("#fdebd0")
OK_BG = colors.HexColor("#e8f6f3")
LINE = colors.HexColor("#b0b7c3")

FONT = "DejaVu"
FONT_B = "DejaVu-Bold"
FONT_M = "DejaVuMono"


def _register_fonts() -> None:
    base = Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont(FONT, str(base / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_B, str(base / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_M, str(base / "DejaVuSansMono.ttf")))


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s: dict[str, ParagraphStyle] = {}
    s["cover_kicker"] = ParagraphStyle(
        "cover_kicker",
        parent=base["Normal"],
        fontName=FONT_B,
        fontSize=10,
        textColor=AMBER,
        alignment=TA_CENTER,
        spaceAfter=6,
        tracking=1.2,
    )
    s["cover_title"] = ParagraphStyle(
        "cover_title",
        parent=base["Title"],
        fontName=FONT_B,
        fontSize=22,
        leading=26,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    s["cover_sub"] = ParagraphStyle(
        "cover_sub",
        parent=base["Normal"],
        fontName=FONT,
        fontSize=11,
        leading=15,
        textColor=SLATE,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    s["h1"] = ParagraphStyle(
        "h1",
        parent=base["Heading1"],
        fontName=FONT_B,
        fontSize=14,
        leading=18,
        textColor=NAVY,
        spaceBefore=10,
        spaceAfter=6,
        borderPadding=3,
    )
    s["h2"] = ParagraphStyle(
        "h2",
        parent=base["Heading2"],
        fontName=FONT_B,
        fontSize=11.5,
        leading=15,
        textColor=TEAL,
        spaceBefore=8,
        spaceAfter=4,
    )
    s["h3"] = ParagraphStyle(
        "h3",
        parent=base["Heading3"],
        fontName=FONT_B,
        fontSize=10,
        leading=13,
        textColor=SLATE,
        spaceBefore=6,
        spaceAfter=3,
    )
    s["body"] = ParagraphStyle(
        "body",
        parent=base["Normal"],
        fontName=FONT,
        fontSize=8.6,
        leading=11.4,
        textColor=SLATE,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
    )
    s["note"] = ParagraphStyle(
        "note",
        parent=base["Normal"],
        fontName=FONT,
        fontSize=8.2,
        leading=11,
        textColor=NAVY,
        backColor=WARN_BG,
        borderPadding=5,
        leftIndent=4,
        rightIndent=4,
        spaceAfter=6,
        spaceBefore=2,
    )
    s["ok"] = ParagraphStyle(
        "ok",
        parent=base["Normal"],
        fontName=FONT,
        fontSize=8.2,
        leading=11,
        textColor=NAVY,
        backColor=OK_BG,
        borderPadding=5,
        leftIndent=4,
        rightIndent=4,
        spaceAfter=6,
    )
    s["cell"] = ParagraphStyle(
        "cell",
        parent=base["Normal"],
        fontName=FONT,
        fontSize=7.3,
        leading=9.4,
        textColor=SLATE,
    )
    s["cell_b"] = ParagraphStyle(
        "cell_b",
        parent=base["Normal"],
        fontName=FONT_B,
        fontSize=7.3,
        leading=9.4,
        textColor=NAVY,
    )
    s["th"] = ParagraphStyle(
        "th",
        parent=base["Normal"],
        fontName=FONT_B,
        fontSize=7.4,
        leading=9.4,
        textColor=colors.white,
    )
    s["mono"] = ParagraphStyle(
        "mono",
        parent=base["Normal"],
        fontName=FONT_M,
        fontSize=7.1,
        leading=9.2,
        textColor=NAVY,
    )
    s["toc"] = ParagraphStyle(
        "toc",
        parent=base["Normal"],
        fontName=FONT,
        fontSize=9,
        leading=13,
        textColor=SLATE,
        leftIndent=8,
    )
    s["footer"] = ParagraphStyle(
        "footer",
        parent=base["Normal"],
        fontName=FONT,
        fontSize=7,
        textColor=colors.HexColor("#7f8c8d"),
        alignment=TA_CENTER,
    )
    s["bullet"] = ParagraphStyle(
        "bullet",
        parent=s["body"],
        leftIndent=12,
        bulletIndent=0,
        spaceAfter=2,
    )
    return s


def P(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def _header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, A4[1] - 12 * mm, A4[0], 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont(FONT, 7.5)
    canvas.drawString(
        14 * mm,
        A4[1] - 7.6 * mm,
        "PdM SIH26054 Student Viva Guide  ·  every sidebar page  ·  every button",
    )
    canvas.drawRightString(A4[0] - 14 * mm, A4[1] - 7.6 * mm, "Predictive Maintenance Analytics")
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, A4[0], 10 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont(FONT, 7.5)
    canvas.drawString(14 * mm, 4 * mm, "Not FADEC / not GCS  ·  19 quality checks  ·  pandas | Polars")
    canvas.drawRightString(A4[0] - 14 * mm, 4 * mm, f"Page {doc.page}")
    canvas.restoreState()


def control_table(rows: list[tuple[str, str, str, str]], S: dict) -> Table:
    header = [
        P("UI label (exact)", S["th"]),
        P("Kind", S["th"]),
        P("What it does", S["th"]),
        P("Function / file", S["th"]),
    ]
    data = [header]
    for label, kind, does, fn in rows:
        data.append(
            [
                P(label, S["cell_b"]),
                P(kind, S["cell"]),
                P(does, S["cell"]),
                P(fn, S["mono"]),
            ]
        )
    col_w = [38 * mm, 22 * mm, 78 * mm, 48 * mm]
    t = Table(data, colWidths=col_w, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
        ("GRID", (0, 0), (-1, -1), 0.25, LINE),
        ("FONTNAME", (0, 0), (-1, 0), FONT_B),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))
        if rows[i - 1][1] in ("Button", "Download", "Form submit"):
            style_cmds.append(("BACKGROUND", (1, i), (1, i), colors.HexColor("#fadbd8")))
    t.setStyle(TableStyle(style_cmds))
    return t


def bullets(items: list[str], S: dict) -> ListFlowable:
    return ListFlowable(
        [ListItem(P(x, S["bullet"]), leftIndent=10) for x in items],
        bulletType="bullet",
        start="•",
        leftIndent=12,
        spaceBefore=1,
        spaceAfter=4,
    )


# ── Inventory: every sidebar radio page + shared chrome ───────────────────────

SIDEBAR_CONTROLS: list[tuple[str, str, str, str]] = [
    (
        "Predictive Maintenance Analytics",
        "Sidebar title",
        "App name from <font face='DejaVu-Bold'>config.APP_TITLE</font>. Tagline under it states this is a reliability add-on, not a generic sales OS and not an OEE cockpit.",
        "main() · app.py + config.py",
    ),
    (
        "Field",
        "Selectbox",
        "Industry pack picker (widget key <font face='DejaVuMono'>industry_pack</font>). Options: Plant / rotating machines (default); Aviation / aero piston / MALE UAV  ★ SIH; Automotive powertrain; Oil well / sucker-rod pump. Switching pack changes extra sensor-map fields, KPIs, 3D mesh, and CAD hints. Do not write <font face='DejaVuMono'>st.session_state.industry_pack</font> after this widget exists.",
        "render_pack_sidebar() · app.py → get_pack() · src/industry_packs.py",
    ),
    (
        "Suggest pack from columns",
        "Button",
        "Scores headers / machine_id prefixes via <font face='DejaVuMono'>suggest_pack()</font>, stores <font face='DejaVuMono'>pack_suggest</font>, queues <font face='DejaVuMono'>_pending_industry_pack</font>, then <font face='DejaVuMono'>st.rerun()</font>. Applied next run by <font face='DejaVuMono'>apply_pending_industry_pack()</font> before the selectbox is created.",
        "render_pack_sidebar() · app.py → suggest_pack() · src/industry_packs.py",
    ),
    (
        "What this pack covers",
        "Expander",
        "Shows pack <font face='DejaVuMono'>scope</font> and <font face='DejaVuMono'>not_in_scope</font>. Aviation cites SIH26054. Automotive adds the OEM / trim / EV honesty note. Oil cites SIH26120.",
        "render_pack_sidebar() · app.py",
    ),
    (
        "Pipeline",
        "Radio",
        "Left-sidebar page switch. Widget key <font face='DejaVuMono'>pipeline_page</font>. Thirteen labels, in order: 1. Upload &amp; Clean; 2. Joins; 3. Map sensors; 4. Anomaly &amp; RUL; 5. Charts; 6. Insights; 3D Twin; CAD Twin (APS); Live Connect; 7. Dashboard; Email Report; Ask; SQL lab. Queued jumps use <font face='DejaVuMono'>request_pipeline_page()</font> + <font face='DejaVuMono'>_pending_pipeline_page</font>.",
        "main() · app.py",
    ),
    (
        "Paste Gemini API key",
        "Password input",
        "Widget key <font face='DejaVuMono'>gemini_key_input</font>. Does not persist until Save key. Env <font face='DejaVuMono'>GEMINI_API_KEY</font> still works.",
        "render_gemini_sidebar() · app.py → get_gemini_api_key() · src/insights_engine.py",
    ),
    (
        "Save key",
        "Button",
        "Copies the paste box into session <font face='DejaVuMono'>gemini_api_key_override</font> via <font face='DejaVuMono'>persist_session_gemini_key()</font>. Empty paste warns. Session-only — not written to disk.",
        "render_gemini_sidebar() · app.py → persist_session_gemini_key() · src/insights_engine.py",
    ),
    (
        "Test Gemini",
        "Button",
        "Calls Gemini with the resolved key/model (default <font face='DejaVuMono'>gemini-3.6-flash</font>; retired aliases remap). Success shows a short reply; failure is stored in <font face='DejaVuMono'>last_gemini_error</font> and shown — 404s are not swallowed.",
        "render_gemini_sidebar() · app.py → test_gemini_connection() · src/insights_engine.py",
    ),
    (
        "Data: N rows / No data loaded",
        "Status caption",
        "After the radio: row count of <font face='DejaVuMono'>cleaned_df</font>, pack short name, mapped sensor count, RUL-label yes/no, table count, prediction count, saved-graph count.",
        "main() · app.py → mapping_status() · src/sensor_map.py",
    ),
]

PAGE_UPLOAD: list[tuple[str, str, str, str]] = [
    (
        "File upload",
        "Tab",
        "Local multi-file ingest. Each file is registered in <font face='DejaVuMono'>uploaded_tables</font> and the last one becomes <font face='DejaVuMono'>raw_df</font>.",
        "page_upload_clean() · app.py",
    ),
    (
        "From URL (cloud / DuckDB)",
        "Tab",
        "HTTPS / Google Drive / Kaggle ingest via DuckDB httpfs. Kaggle needs <font face='DejaVuMono'>KAGGLE_USERNAME</font> + <font face='DejaVuMono'>KAGGLE_KEY</font>.",
        "page_upload_clean() → _ingest_from_url_tab() · app.py",
    ),
    (
        "Upload sensor / ops CSVs (multi-select OK)",
        "File uploader",
        "Accepts csv / tsv / xlsx / json / zip (sensor tables, not CAD Twin STEP). Key <font face='DejaVuMono'>upload_sensor_files</font>. Each file → <font face='DejaVuMono'>load_tabular_file()</font> / <font face='DejaVuMono'>load_zip_tabular()</font> → <font face='DejaVuMono'>register_table(stem)</font> → <font face='DejaVuMono'>raw_df</font>.",
        "page_upload_clean() · app.py → load_tabular_file() · src/data_integration.py",
    ),
    (
        "Load Plant sample (default)",
        "Button",
        "Loads <font face='DejaVuMono'>sample_data/sensor_readings.csv</font>, registers sensors, plus maintenance/costs if present. Queues pack <font face='DejaVuMono'>plant_rotating</font>. Clears cleaned_df / predictions.",
        "page_upload_clean() → load_pack_sample(DEFAULT_PACK_ID) · app.py",
    ),
    (
        "Or load a pack demo CSV",
        "Selectbox",
        "Key <font face='DejaVuMono'>upload_demo_pack</font>. Aviation / Automotive / Oil demo CSVs (Plant is the other button).",
        "page_upload_clean() · app.py → pack_label() · src/industry_packs.py",
    ),
    (
        "Load pack demo",
        "Button",
        "Loads the selected pack CSV and queues that industry pack (SIH26054 aviation demo is here). Key <font face='DejaVuMono'>upload_load_pack_demo</font>.",
        "page_upload_clean() → load_pack_sample(demo_pack) · app.py",
    ),
    (
        "Cloud data URL",
        "Text input",
        "Key <font face='DejaVuMono'>upload_url_input</font>. Direct HTTPS CSV/Parquet, Drive share, or Kaggle page / <font face='DejaVuMono'>kaggle://</font>.",
        "_ingest_from_url_tab() · app.py",
    ),
    (
        "Ingest mode",
        "Radio",
        "Row limit (simple) | SQL slice (DuckDB). Key <font face='DejaVuMono'>upload_url_ingest_mode</font>. Stores <font face='DejaVuMono'>url_ingest_mode</font> as limit or sql.",
        "_ingest_from_url_tab() · app.py",
    ),
    (
        "Row limit (0 = all rows — cap on free hosts for multi-GB files)",
        "Number input",
        "Shown in limit mode. Key <font face='DejaVuMono'>upload_url_row_limit</font>. 0 = no cap.",
        "_ingest_from_url_tab() · app.py",
    ),
    (
        "SQL slice preset (plant / PdM templates)",
        "Selectbox",
        "Key <font face='DejaVuMono'>upload_url_sql_preset</font>. PdM presets: Last N rows; Last N days; Filter by machine_id; Filter by asset_id; Date range window; Random sample %; PdM — failures + near-failure window.",
        "_ingest_from_url_tab() · app.py → list_ingest_presets() · src/url_ingest.py",
    ),
    (
        "Apply preset to SQL",
        "Button",
        "Writes the template (with <font face='DejaVuMono'>{source}</font> intact) into <font face='DejaVuMono'>url_ingest_sql</font>. Key <font face='DejaVuMono'>upload_apply_sql_preset</font>.",
        "_ingest_from_url_tab() · app.py → build_preset_sql() · src/url_ingest.py",
    ),
    (
        "DuckDB SQL (use {source} for the resolved file path/URL)",
        "Text area",
        "Key <font face='DejaVuMono'>upload_url_sql</font>. Read-only SELECT/WITH only. Used when ingest mode is SQL slice.",
        "_ingest_from_url_tab() · app.py",
    ),
    (
        "Always download to disk first (recommended for Google Drive / files &gt; 100 MB)",
        "Checkbox",
        "Key <font face='DejaVuMono'>upload_url_force_cache</font>. Forced on when a SQL slice is used.",
        "_ingest_from_url_tab() · app.py",
    ),
    (
        "Load from URL",
        "Button",
        "Primary. Resolves Drive/Kaggle, DuckDB-loads the slice, sets <font face='DejaVuMono'>raw_df</font>, registers a table, invalidates <font face='DejaVuMono'>cleaned_df</font>, stores <font face='DejaVuMono'>url_ingest_meta</font>. Key <font face='DejaVuMono'>upload_url_load</font>.",
        "_ingest_from_url_tab() · app.py → load_from_url() · src/url_ingest.py",
    ),
    (
        "Attach maintenance table (optional — join in step 2)",
        "Expander",
        "Optional work-order / PM CSV so Joins can merge on machine_id.",
        "_render_maintenance_attach() · app.py",
    ),
    (
        "Maintenance / work-order CSV",
        "File uploader",
        "Key <font face='DejaVuMono'>upload_maintenance_table</font>. Registers table name <font face='DejaVuMono'>maintenance</font>. Types csv / tsv / xlsx / xls.",
        "_render_maintenance_attach() · app.py → load_tabular_file() · src/data_integration.py",
    ),
    (
        "Raw Data Preview",
        "Table",
        "Full <font face='DejaVuMono'>raw_df</font> in a 420 px scrollable grid (not a 10-row head). Caption shows row count.",
        "page_upload_clean() → _scrollable_full_table() · app.py",
    ),
    (
        "Cleaning engine",
        "Selectbox",
        "pandas (default) and polars when installed. Stores <font face='DejaVuMono'>clean_engine</font>. Polars is columnar ETL, no JVM. PySpark was removed.",
        "page_upload_clean() · app.py → polars_available() · src/polars_clean.py",
    ),
    (
        "Run industrial clean + 19 quality checks",
        "Button",
        "Primary. If Polars is selected, <font face='DejaVuMono'>clean_with_polars()</font> first (dedupe / sentinel-null / cast / median / IQR cap) then always <font face='DejaVuMono'>clean_and_quality()</font> (pandas industrial ETL + 19-stage report). Writes cleaned_df, cleaning_summary, quality_report, insights; registers table <font face='DejaVuMono'>cleaned</font>. Polars failure falls back to pandas with a warning.",
        "page_upload_clean() · app.py → clean_with_polars() · src/polars_clean.py → clean_and_quality() · src/data_cleaner.py → build_quality_report() · src/quality_checks.py",
    ),
    (
        "ETL / DWDM actions",
        "Expander",
        "Lists cleaner log lines (schema normalize, dedupe, imputation, binning, smoothing, plus any Polars actions).",
        "page_upload_clean() · app.py",
    ),
    (
        "Cleaned Data Preview",
        "Table",
        "Full scrollable <font face='DejaVuMono'>cleaned_df</font> after a successful clean.",
        "page_upload_clean() → _scrollable_full_table() · app.py",
    ),
    (
        "Download Cleaned CSV",
        "Download",
        "Exports <font face='DejaVuMono'>cleaned_df</font> as <font face='DejaVuMono'>cleaned_sensor_data_YYYYMMDD.csv</font>.",
        "page_upload_clean() · app.py",
    ),
    (
        "19-Stage Quality Report",
        "Table",
        "DataFrame of the 19 named checks (see honesty chapter) with PASS / WARN / FAIL / INFO. Four metric tiles count those statuses.",
        "page_upload_clean() · app.py · QUALITY_STAGE_COUNT in src/quality_checks.py",
    ),
    (
        "Great Expectations — column expectations",
        "Expander",
        "GE-style expectation rows (nulls, temp range, RUL trend, unique timestamp+asset, row count). Library optional; checks still run as a proxy.",
        "_render_quality_subreports() · app.py · _run_great_expectations() · src/quality_checks.py",
    ),
    (
        "ydata / Cleanlab / PCA drift / Association rules",
        "Expander",
        "Four sub-blocks: ydata-profiling (optional), Cleanlab dirty-label / outlier flags, PCA early-vs-late variance drift, Apriori-style HIGH-sensor baskets.",
        "_render_quality_subreports() · app.py · src/quality_checks.py",
    ),
    (
        "Domain OPC / physics-rule violations",
        "Expander",
        "Expanded when flags exist: stuck-sensor (temp&gt;150 &amp; vib&lt;0.1), leak/sensor-fault, unusual RUL trend, missed-failure suspects.",
        "_render_quality_subreports() · app.py · _domain_opc_rules() · src/quality_checks.py",
    ),
    (
        "Data Insights",
        "Markdown block",
        "Column-level stats from <font face='DejaVuMono'>analyze_columns()</font> after clean.",
        "page_upload_clean() · app.py → src/data_insights.py",
    ),
]

PAGE_JOINS: list[tuple[str, str, str, str]] = [
    (
        "Join mode",
        "Radio",
        "Two tables | Chain 3+ tables. Needs ≥2 registered tables (Plant sample loads sensors + maintenance + costs).",
        "page_data_integration() · app.py",
    ),
    (
        "Left table",
        "Selectbox",
        "Key <font face='DejaVuMono'>join_left</font>. Two-table mode.",
        "page_data_integration() · app.py",
    ),
    (
        "Right table",
        "Selectbox",
        "Key <font face='DejaVuMono'>join_right</font>. Excludes the left name.",
        "page_data_integration() · app.py",
    ),
    (
        "Join type",
        "Selectbox",
        "inner — INNER JOIN — only matching keys; left — LEFT JOIN; right — RIGHT JOIN; outer — FULL OUTER JOIN. Labels come from JOIN_TYPES.",
        "page_data_integration() · app.py · JOIN_TYPES · src/data_integration.py",
    ),
    (
        "Keys",
        "Radio",
        "Common columns | Different column names.",
        "page_data_integration() · app.py",
    ),
    (
        "Join on",
        "Multiselect",
        "Shown when Keys = Common columns. Defaults to suggested intersection (machine_id preferred).",
        "page_data_integration() · app.py → suggest_join_keys() · src/data_integration.py",
    ),
    (
        "Left key / Right key",
        "Selectbox",
        "Shown when Keys = Different column names.",
        "page_data_integration() · app.py",
    ),
    (
        "Run join",
        "Button",
        "Primary. <font face='DejaVuMono'>join_two()</font> → writes cleaned_df and raw_df, registers table <font face='DejaVuMono'>joined</font>, stores join_log, shows meta JSON + first 20 rows.",
        "page_data_integration() · app.py → join_two() · src/data_integration.py",
    ),
    (
        "Start (left)",
        "Selectbox",
        "Chain mode. Key <font face='DejaVuMono'>chain_left</font>.",
        "page_data_integration() · app.py",
    ),
    (
        "Join #1 right",
        "Selectbox",
        "Key <font face='DejaVuMono'>chain_r1</font>.",
        "page_data_integration() · app.py",
    ),
    (
        "Join #1 type",
        "Selectbox",
        "Key <font face='DejaVuMono'>how1</font>. Same JOIN_TYPES.",
        "page_data_integration() · app.py",
    ),
    (
        "Join #1 keys",
        "Multiselect",
        "Key <font face='DejaVuMono'>k1</font>. Suggested common columns.",
        "page_data_integration() · app.py",
    ),
    (
        "Join #2 right",
        "Selectbox",
        "Key <font face='DejaVuMono'>chain_r2</font>.",
        "page_data_integration() · app.py",
    ),
    (
        "Join #2 type",
        "Selectbox",
        "Key <font face='DejaVuMono'>how2</font>.",
        "page_data_integration() · app.py",
    ),
    (
        "Join #2 keys (must exist on interim + right)",
        "Multiselect",
        "Key <font face='DejaVuMono'>k2</font>.",
        "page_data_integration() · app.py",
    ),
    (
        "Run join chain",
        "Button",
        "Primary. <font face='DejaVuMono'>join_many()</font> two steps; result becomes working table <font face='DejaVuMono'>joined</font>.",
        "page_data_integration() · app.py → join_many() · src/data_integration.py",
    ),
    (
        "joined preview (head 20)",
        "Table",
        "Shown after either join button succeeds.",
        "page_data_integration() · app.py",
    ),
    (
        "Last join log",
        "Expander",
        "JSON of the last join_log (keys, how, row counts).",
        "page_data_integration() · app.py",
    ),
]

PAGE_MAP: list[tuple[str, str, str, str]] = [
    (
        "Sensors mapped / Timestamp / Asset id / RUL label",
        "Metrics",
        "From <font face='DejaVuMono'>mapping_status()</font>. Warns if no failure_within_days — RUL will use a degradation proxy.",
        "page_map_sensors() · app.py → mapping_status() · src/sensor_map.py",
    ),
    (
        "Reading timestamp",
        "Selectbox",
        "Canonical field timestamp. Key <font face='DejaVuMono'>map_timestamp</font>. Suggests time / datetime aliases.",
        "page_map_sensors() · app.py · CANONICAL_FIELDS · src/sensor_map.py",
    ),
    (
        "Machine / asset id",
        "Selectbox",
        "Canonical machine_id. Key <font face='DejaVuMono'>map_machine_id</font>. Aliases include uav_id / aircraft_id / well_id.",
        "page_map_sensors() · app.py",
    ),
    (
        "Temperature / Vibration / Pressure / Rotational speed (RPM)",
        "Selectboxes",
        "Core PdM sensors. Keys <font face='DejaVuMono'>map_temperature</font>, <font face='DejaVuMono'>map_vibration</font>, <font face='DejaVuMono'>map_pressure</font>, <font face='DejaVuMono'>map_rpm</font>.",
        "page_map_sensors() · app.py · CANONICAL_FIELDS · src/sensor_map.py",
    ),
    (
        "Days until failure (RUL label, optional)",
        "Selectbox",
        "Canonical failure_within_days. Key <font face='DejaVuMono'>map_failure_within_days</font>. Honest RUL needs this (or alias). Sample labels are simulated.",
        "page_map_sensors() · app.py · config.RUL_HONESTY_CAPTION",
    ),
    (
        "Aviation extras (when pack = Aviation ★ SIH)",
        "Selectboxes",
        "Exhaust gas temp (EGT); Cylinder head temp (CHT); Engine oil pressure; Engine oil temperature; Airframe / engine hours; Altitude (ft); Throttle (%); Manifold pressure. Keys <font face='DejaVuMono'>map_&lt;canonical&gt;</font>.",
        "page_map_sensors() · app.py → extra_field_defs() · src/industry_packs.py",
    ),
    (
        "Automotive extras (when pack = Automotive)",
        "Selectboxes",
        "Coolant temperature; Engine oil pressure; Engine load (%); Vehicle speed; Gear.",
        "page_map_sensors() · app.py → extra_field_defs()",
    ),
    (
        "Oil / SRP extras (when pack = Oil / SRP)",
        "Selectboxes",
        "Polish-rod load; Tubing pressure; Casing pressure; Pump fillage (%); Strokes per minute; Production (bbl).",
        "page_map_sensors() · app.py → extra_field_defs()",
    ),
    (
        "Apply mapping",
        "Button",
        "Primary. <font face='DejaVuMono'>apply_mapping()</font> renames/copies columns onto the working table. Saves <font face='DejaVuMono'>sensor_mapping</font>, writes cleaned_df, registers <font face='DejaVuMono'>cleaned</font>, shows first 8 mapped rows, reruns.",
        "page_map_sensors() · app.py → apply_mapping() · src/sensor_map.py",
    ),
    (
        "mapped.head(8)",
        "Table",
        "Preview after Apply mapping (same run, before rerun).",
        "page_map_sensors() · app.py",
    ),
]

PAGE_ML: list[tuple[str, str, str, str]] = [
    (
        "Optuna trials (optional)",
        "Slider",
        "5–40, default 15. Used by the two Optuna buttons (anomaly uses min(12, trials)).",
        "page_ml_predictions() · app.py",
    ),
    (
        "Detect anomalies + predict RUL",
        "Button",
        "Primary. Fits Isolation Forest (contamination from config or Optuna), annotates scores, fits Random Forest RUL, writes predictions per machine, attaches RUL columns, rebuilds the industrial brief (insights / ranking / inspect list). Warns if labels were synthetic.",
        "page_ml_predictions() · app.py → AnomalyDetector · src/ml/anomaly_detector.py → RULPredictor · src/ml/rul_predictor.py → build_industrial_brief() · src/insights_engine.py",
    ),
    (
        "Optuna tune RUL",
        "Button",
        "Needs a real <font face='DejaVuMono'>failure_within_days</font> column — refuses synthetic-label Optuna. Stores <font face='DejaVuMono'>optuna_rul</font> JSON.",
        "page_ml_predictions() · app.py → tune_rul_model() · src/ml/optuna_tuner.py",
    ),
    (
        "Optuna tune anomalies",
        "Button",
        "Tunes Isolation Forest contamination. Stores <font face='DejaVuMono'>optuna_anomaly</font>. You must click Detect anomalies + predict RUL again to apply.",
        "page_ml_predictions() · app.py → tune_anomaly_contamination() · src/ml/optuna_tuner.py",
    ),
    (
        "Optuna RUL result / Optuna anomaly result",
        "JSON",
        "Shown after a successful tune.",
        "page_ml_predictions() · app.py",
    ),
    (
        "Failure Predictions",
        "Markdown list",
        "One risk-colored line per asset (High / Medium / Low) from <font face='DejaVuMono'>predictions</font>.",
        "page_ml_predictions() · app.py",
    ),
    (
        "MAE (days) / RMSE (days) / R² Score",
        "Metrics",
        "RUL hold-out metrics when a real label existed.",
        "page_ml_predictions() · app.py · RULPredictor.metrics",
    ),
    (
        "Top 10 Features for RUL Prediction",
        "Chart",
        "Horizontal Plotly bar of Random Forest importances.",
        "page_ml_predictions() · app.py → style_bar_figure() · src/graphs/layout.py",
    ),
    (
        "Anomaly Detection Summary",
        "Metrics",
        "Records Analyzed, Anomalies Found, Anomaly Rate % from Isolation Forest summary.",
        "page_ml_predictions() · app.py · AnomalyDetector.summary()",
    ),
]

PAGE_CHARTS: list[tuple[str, str, str, str]] = [
    (
        "X-Axis",
        "Selectbox",
        "Prefers timestamp when present.",
        "page_explore_graphs() · app.py → get_axis_options()",
    ),
    (
        "Sensor / metric",
        "Selectbox",
        "Numeric column. Defaults to the pack’s preferred metric (EGT on Aviation, vibration/temp on Plant).",
        "page_explore_graphs() · app.py → preferred_y_metric() · src/industry_packs.py",
    ),
    (
        "Filter by Machine",
        "Multiselect",
        "Defaults to all machine_id values.",
        "page_explore_graphs() · app.py",
    ),
    (
        "Pack KPI tiles + health / KPI bar charts",
        "Metrics + charts",
        "Aviation tiles include engine health, mission reliability, remaining mission hours, EGT margin (SIH26054). Plant / Auto / Oil have their own KPI ids.",
        "page_explore_graphs() · app.py → compute_pack_kpis() · src/pack_kpis.py → create_asset_health_chart / create_pack_kpi_bars · src/graphs/pack_kpis.py",
    ),
    (
        "Generate Sensor over time",
        "Button",
        "Primary chart. Key <font face='DejaVuMono'>gen_time_series</font>. Plotly time series; saved into graph_folder.",
        "page_explore_graphs() · app.py → create_time_series_chart() · src/graphs/time_series.py → save_graph_to_folder() · src/dashboard_builder.py",
    ),
    (
        "Generate Anomaly flags",
        "Button",
        "Key <font face='DejaVuMono'>gen_anomaly_flags</font>. Isolation Forest flags on the timeline (needs a trained detector).",
        "page_explore_graphs() · app.py → create_anomaly_flags_chart() · src/graphs/anomaly_flags.py",
    ),
    (
        "Generate Risk by asset",
        "Button",
        "Key <font face='DejaVuMono'>gen_risk_by_asset</font>. Predicted RUL days colored High / Medium / Low (needs predictions).",
        "page_explore_graphs() · app.py → create_risk_by_asset_chart() · src/graphs/risk_by_asset.py",
    ),
    (
        "More sensor views (optional)",
        "Expander",
        "Secondary Plotly views. Each button is icon + name from GRAPH_TYPES.",
        "page_explore_graphs() · app.py · GRAPH_TYPES · src/graphs/__init__.py",
    ),
    (
        "🔥 Sensor Correlation Heatmap",
        "Button",
        "Key <font face='DejaVuMono'>gen_heatmap</font>.",
        "page_explore_graphs() · app.py → create_correlation_heatmap() · src/graphs/heatmap.py",
    ),
    (
        "⚠ Anomaly Scatter Plot",
        "Button",
        "UI icon is the warning emoji + name. Key <font face='DejaVuMono'>gen_anomaly_scatter</font>.",
        "page_explore_graphs() · app.py → create_anomaly_scatter() · src/graphs/anomaly_scatter.py",
    ),
    (
        "📊 Distribution Histogram",
        "Button",
        "Key <font face='DejaVuMono'>gen_histogram</font>.",
        "page_explore_graphs() · app.py → create_distribution_histogram() · src/graphs/histogram.py",
    ),
    (
        "📦 Box Plot by Machine",
        "Button",
        "Key <font face='DejaVuMono'>gen_boxplot</font>.",
        "page_explore_graphs() · app.py → create_boxplot_by_machine() · src/graphs/boxplot.py",
    ),
    (
        "〰️ Rolling Average Trend",
        "Button",
        "Key <font face='DejaVuMono'>gen_rolling_trend</font>.",
        "page_explore_graphs() · app.py → create_rolling_trend() · src/graphs/rolling_trend.py",
    ),
    (
        "{title} — {created_at}",
        "Expander",
        "One expander per saved graph in graph_folder, showing the Plotly figure.",
        "page_explore_graphs() · app.py → get_graph_folder() · src/dashboard_builder.py",
    ),
    (
        "Remove {g_id}",
        "Button",
        "Deletes that entry from <font face='DejaVuMono'>st.session_state.graph_folder</font> and reruns. Key <font face='DejaVuMono'>rm_{g_id}</font>.",
        "page_explore_graphs() · app.py",
    ),
]

PAGE_INSIGHTS: list[tuple[str, str, str, str]] = [
    (
        "$ / hour downtime  (label follows pack)",
        "Number input",
        "Plant: $ / hour downtime. Aviation: $ / hour of lost sortie time. Automotive: $ / hour vehicle down. Oil: $ / hour deferred production. Session key <font face='DejaVuMono'>cost_per_hour</font>. Dollars are only from rates you type.",
        "page_business_insights() · app.py · pack cost_hour_label · src/industry_packs.py",
    ),
    (
        "$ / unit lost production  (label follows pack)",
        "Number input",
        "Aviation: $ / aborted mission. Session <font face='DejaVuMono'>cost_per_unit</font>.",
        "page_business_insights() · app.py",
    ),
    (
        "Hours if a high-risk asset stops",
        "Number input",
        "Default 8 h is an assumed stop — not a booked outage. Session <font face='DejaVuMono'>hours_if_stop</font>.",
        "page_business_insights() · app.py",
    ),
    (
        "Refresh brief",
        "Button",
        "Primary. Rebuilds inspect list / ranking / pack KPI cards via <font face='DejaVuMono'>_refresh_brief()</font> and rebuilds the LlamaIndex / TF-IDF retrieval index.",
        "page_business_insights() · app.py → build_industrial_brief() · src/insights_engine.py → make_insight_index()",
    ),
    (
        "Gemini polish (wording only)",
        "Button",
        "Rewords the rule-based brief. Numbers stay from the table. Needs a key; otherwise shows the no-key message.",
        "page_business_insights() · app.py → polish_brief_with_gemini() · src/insights_engine.py",
    ),
    (
        "Asset rank",
        "Table",
        "Columns: rank, machine_id, risk_level, predicted_rul_days, mean_anomaly_score, anomaly_count, anomaly_rate_pct, n_rows.",
        "page_business_insights() · app.py",
    ),
    (
        "Gemini brief",
        "Markdown",
        "Shown after a successful polish. Stored in <font face='DejaVuMono'>polished_brief</font>.",
        "page_business_insights() · app.py",
    ),
    (
        "Insight cards (title — severity)",
        "Markdown",
        "Inspect-this-week / slow-running / $ at risk / pack KPI cards. Grounded in this upload, not generic advice.",
        "page_business_insights() · app.py → generate_business_insights() / insight_cards_from_kpis()",
    ),
    (
        "Ask this upload  (chat input)",
        "Chat input",
        "Placeholder: Ask about an asset, inspect list, or $ at risk on this upload…. Answers use this table + Isolation Forest / RUL columns — not a general LLM essay.",
        "render_ask_panel() · app.py → ask_with_index() · src/insights_engine.py",
    ),
    (
        "Clear Ask",
        "Button",
        "Clears <font face='DejaVuMono'>chat_history</font>. Shown only when history is non-empty.",
        "render_ask_panel() · app.py",
    ),
]

PAGE_TWIN: list[tuple[str, str, str, str]] = [
    (
        "Risk source",
        "Radio",
        "Anomaly &amp; RUL (batch) and/or Live Connect (streaming). Key <font face='DejaVuMono'>twin_source</font>. Defaults to Live when streaming is on.",
        "page_twin_3d() · app.py → asset_states_from_predictions() · src/twin3d.py",
    ),
    (
        "Asset",
        "Selectbox",
        "Key <font face='DejaVuMono'>twin_asset_pick</font>. Drives which hotspot is colored.",
        "page_twin_3d() · app.py",
    ),
    (
        "Risk",
        "Metric",
        "High / Medium / Low of the selected asset.",
        "page_twin_3d() · app.py",
    ),
    (
        "Preview risk",
        "Selectbox",
        "High | Medium | Low. Shown only when there are no predictions yet (demo mesh).",
        "page_twin_3d() · app.py",
    ),
    (
        "3D mesh (three.js, no CDN)",
        "Embedded HTML",
        "Pack mesh: Plant motor (drive-end bearing hotspot); Aviation UAV + piston (SIH26054); generic ICE car; SRP beam pump. High = red blink, Medium = amber, Low = green. Offline — vendored three.js.",
        "page_twin_3d() → _render_twin() · app.py → build_twin_html() · src/twin3d.py",
    ),
]

PAGE_CAD: list[tuple[str, str, str, str]] = [
    (
        "Asset",
        "Selectbox",
        "Key <font face='DejaVuMono'>aps_asset_pick</font>. Strip above the viewer. Risk / RUL come from Anomaly &amp; RUL or Live Connect.",
        "render_cad_part_card() · app.py",
    ),
    (
        "Load CAD model",
        "Button",
        "Primary. Key <font face='DejaVuMono'>aps_load_cad_btn</font>. Disabled until APS credentials + a translated (non-public-viewer) URN exist. Fetches a 2-legged token and opens GuiViewer3D. Also persists the URN for this pack.",
        "render_cad_part_card() · app.py → get_access_token() / _render_cad_viewer() · src/aps_viewer.py",
    ),
    (
        "Open Anomaly &amp; RUL",
        "Button",
        "Key <font face='DejaVuMono'>cad_open_anomaly_rul</font>. Queues <font face='DejaVuMono'>pipeline_page</font> = 4. Anomaly &amp; RUL via <font face='DejaVuMono'>request_pipeline_page()</font> and reruns. Does not write the radio key after the widget exists.",
        "render_cad_part_card() · app.py → request_pipeline_page() · app.py",
    ),
    (
        "Region for the clicked part",
        "Selectbox",
        "Heads/exhaust | Oil | Rotating/crank | Intake | Other. Key <font face='DejaVuMono'>cad_region_pick</font>. Defaults from the sensor driver region.",
        "render_cad_region_assign() · app.py · REGIONS · src/cad_regions.py",
    ),
    (
        "Assign selected part to this region",
        "Button",
        "Primary. Key <font face='DejaVuMono'>cad_assign_region_btn</font>. Disabled until a part is clicked (dbId) and a URN exists. Saves {pack + URN → region → dbIds} to <font face='DejaVuMono'>data/cad_region_map.json</font>. That map is what turns Solid1 red — Isolation Forest does not score CAD parts.",
        "render_cad_region_assign() · app.py → assign_region() · src/cad_regions.py",
    ),
    (
        "Clear region map",
        "Button",
        "Key <font face='DejaVuMono'>cad_clear_region_btn</font>. Forgets every assigned part for this model. Disabled when the map is empty.",
        "render_cad_region_assign() · app.py → clear_region_map() · src/cad_regions.py",
    ),
    (
        "Also tint related regions",
        "Checkbox",
        "Key <font face='DejaVuMono'>cad_region_include_related</font>. e.g. a heads/exhaust driver also tints assigned oil parts.",
        "render_cad_region_assign() · app.py → assigned_theme_plan() · src/cad_regions.py",
    ),
    (
        "Mapped sensors — latest row for this asset",
        "Metrics grid",
        "Four metrics per row from the latest mapped sensor values. Caption marks the clicked-part hint.",
        "render_cad_sensor_grid() · app.py → latest_asset_sensors() · src/cad_part.py",
    ),
    (
        "CAD source — upload, translate, URN, save",
        "Expander",
        "Admin panel below the viewer so it never pushes GuiViewer3D off-screen. Expanded when no URN yet.",
        "page_cad_twin() → _render_cad_source_panel() · app.py",
    ),
    (
        "CAD file for Model Derivative",
        "File uploader",
        "Key <font face='DejaVuMono'>aps_cad_file_uploader</font>. Streamlit max 300 MB. STEP / ZIP / Revit / Fusion / IFC / Inventor / SolidWorks / DWG / OBJ / STL …",
        "_render_cad_source_panel() · app.py · CAD_UPLOAD_EXTENSIONS · src/aps_viewer.py",
    ),
    (
        "ZIP root filename (auto-filled for zipped STEP — leave as-is)",
        "Text input",
        "Key <font face='DejaVuMono'>aps_zip_root_filename</font>. Prefill happens in <font face='DejaVuMono'>_seed_zip_root_widget()</font> before this widget exists. Do not assign the key after.",
        "_render_cad_source_panel() · app.py → resolve_zip_root_filename() · src/aps_viewer.py",
    ),
    (
        "Translate to SVF / get URN",
        "Button",
        "Primary. Key <font face='DejaVuMono'>aps_translate_btn</font>. Uploads bytes to your OSS bucket and runs Model Derivative. Writes <font face='DejaVuMono'>aps_urn</font> / <font face='DejaVuMono'>aps_urn_input</font>, saves the pack URN, forces viewer load. viewer.autodesk.com is Autodesk’s public site — it does not put the file in your bucket.",
        "_render_cad_source_panel() · app.py → translate_cad_bytes() · src/aps_viewer.py",
    ),
    (
        "Translated model URN (base64) or viewer.autodesk.com URL",
        "Text input",
        "Key <font face='DejaVuMono'>aps_urn_input</font>. Seeded by <font face='DejaVuMono'>_seed_cad_urn_widgets()</font> from APS_MODEL_URN, saved pack JSON, or session. Public-viewer URNs are extracted but blocked from loading with a 2-legged token.",
        "_render_cad_source_panel() · app.py → extract_model_urn() · src/aps_viewer.py",
    ),
    (
        "Save URN for this pack",
        "Button",
        "Key <font face='DejaVuMono'>aps_save_urn_btn</font>. Writes <font face='DejaVuMono'>data/cad_urns.json</font> for the current industry pack. Public-viewer URNs are refused. Render redeploys wipe disk — set APS_MODEL_URN too.",
        "_render_cad_source_panel() · app.py → save_urn_for_pack() · src/aps_viewer.py",
    ),
    (
        "Delete saved URN",
        "Button",
        "Key <font face='DejaVuMono'>aps_delete_urn_btn</font>. Clears the pack entry in cad_urns.json on this server.",
        "_render_cad_source_panel() · app.py → delete_urn_for_pack() · src/aps_viewer.py",
    ),
    (
        "How to generate a URN (APS app + this page)",
        "Expander",
        "Six-step APS honesty: create an APS app, set Render secrets, translate on this page into your bucket, do not trust viewer.autodesk.com, required OAuth scopes, Save/Delete vs APS_MODEL_URN after redeploy. Also documents GuiViewer3D toolbar and Solid1 name sanitizing.",
        "page_cad_twin() · app.py",
    ),
]

PAGE_LIVE: list[tuple[str, str, str, str]] = [
    (
        "Source",
        "Radio",
        "Simulator | Poll CSV URL | MQTT (live) (if paho-mqtt) | OPC-UA (live) (if asyncua). Stores <font face='DejaVuMono'>live_source</font> as sim / poll / mqtt / opcua.",
        "page_live_connect() · app.py · mqtt_available / opcua_available · src/live_sources.py",
    ),
    (
        "Machines",
        "Slider",
        "Simulator only. 2–8 assets.",
        "page_live_connect() · app.py",
    ),
    (
        "Failing asset (drifts to failure)",
        "Selectbox",
        "Simulator only. That machine’s sensors ramp toward failure.",
        "page_live_connect() · app.py → simulate_batch() · src/live_connect.py",
    ),
    (
        "Ramp ticks to failure",
        "Slider",
        "Simulator only. 10–120 ticks.",
        "page_live_connect() · app.py",
    ),
    (
        "CSV feed URL",
        "Text input",
        "Poll mode. Default http://localhost:8000/sensor_readings.csv.",
        "page_live_connect() · app.py → poll_url_increment() · src/live_connect.py",
    ),
    (
        "Rows per poll",
        "Slider",
        "Poll mode. 5–200.",
        "page_live_connect() · app.py",
    ),
    (
        "Broker host / Port / Topic (wildcards OK)",
        "Inputs",
        "MQTT mode. Default topic pdm/sensors/#. Expects JSON with machine_id + sensors.",
        "page_live_connect() · app.py → start_mqtt() · src/live_sources.py",
    ),
    (
        "Username (optional) / Password (optional)",
        "Text inputs",
        "MQTT auth. Password is type=password.",
        "page_live_connect() · app.py",
    ),
    (
        "OPC-UA endpoint",
        "Text input",
        "Default opc.tcp://127.0.0.1:4840/freeopcua/server/.",
        "page_live_connect() · app.py → start_opcua() · src/live_sources.py",
    ),
    (
        "Node map — sensor=node_id, comma-separated",
        "Text area",
        "Example temperature=ns=2;i=2, vibration=ns=2;i=3.",
        "page_live_connect() · app.py → parse_node_map() · src/live_sources.py",
    ),
    (
        "Machine id label",
        "Text input",
        "OPC-UA asset name (default opcua-asset).",
        "page_live_connect() · app.py",
    ),
    (
        "▶ Start live",
        "Button",
        "Primary. Stops any prior source, starts sim / poll / MQTT / OPC-UA, clears buffer, sets live_running True. MQTT/OPC connection id stored in <font face='DejaVuMono'>live_conn_id</font>.",
        "page_live_connect() · app.py → start_mqtt / start_opcua · src/live_sources.py",
    ),
    (
        "⏸ Stop",
        "Button",
        "Sets live_running False and <font face='DejaVuMono'>stop_source()</font>.",
        "page_live_connect() · app.py → stop_source() · src/live_sources.py",
    ),
    (
        "Reset buffer",
        "Button",
        "Clears live_buffer, live_tick, live_asset_states.",
        "page_live_connect() · app.py",
    ),
    (
        "Live chart",
        "Selectbox",
        "temperature | vibration | pressure | rpm. Session <font face='DejaVuMono'>live_sensor_view</font>.",
        "page_live_connect() · app.py",
    ),
    (
        "Live asset status",
        "Table",
        "Per-machine risk from the rolling buffer. Feeds 3D Twin and Insights. Metrics: Rows buffered, Machines, Worst asset.",
        "_live_body() · app.py → compute_live_status() · src/live_connect.py",
    ),
    (
        "Live {sensor} (last N readings)",
        "Chart",
        "Plotly line of the selected sensor, last 300 rows, colored by machine_id. Auto-refresh ~2 s via st.fragment when running.",
        "_live_body() · app.py",
    ),
]

PAGE_DASH: list[tuple[str, str, str, str]] = [
    (
        "KPI strip",
        "Checkbox",
        "Tile id kpis. Key <font face='DejaVuMono'>dash_tile_kpis</font>. Pack health / mission reliability / fillage.",
        "page_dashboard_builder() · app.py · TILE_SPECS · src/dashboard_composer.py",
    ),
    (
        "Charts",
        "Checkbox",
        "Tile id charts. Key <font face='DejaVuMono'>dash_tile_charts</font>. Saved Plotly views + pack health bars.",
        "page_dashboard_builder() · app.py",
    ),
    (
        "Insights",
        "Checkbox",
        "Tile id insights. Key <font face='DejaVuMono'>dash_tile_insights</font>. Inspect-this-week and pack cards.",
        "page_dashboard_builder() · app.py",
    ),
    (
        "3D twin",
        "Checkbox",
        "Tile id twin3d. Key <font face='DejaVuMono'>dash_tile_twin3d</font>. Offline pack mesh.",
        "page_dashboard_builder() · app.py → board_twin_html() · src/dashboard_composer.py",
    ),
    (
        "CAD twin (APS)",
        "Checkbox",
        "Tile id cad. Key <font face='DejaVuMono'>dash_tile_cad</font>. Real CAD when Render secrets + URN exist; otherwise a placeholder.",
        "page_dashboard_builder() · app.py → board_cad_html() / cad_placeholder_html() · src/dashboard_composer.py",
    ),
    (
        "KPI strip  (section)",
        "Metrics",
        "Rendered when the KPI checkbox is on.",
        "page_dashboard_builder() · app.py → compute_pack_kpis() · src/pack_kpis.py",
    ),
    (
        "Charts  (section)",
        "Charts",
        "Pack figures plus any graphs saved on page 5.",
        "page_dashboard_builder() · app.py → board_pack_charts()",
    ),
    (
        "Insights  (section)",
        "Cards",
        "Up to 8 insight cards. Empty until Anomaly &amp; RUL / Insights have run.",
        "page_dashboard_builder() · app.py",
    ),
    (
        "3D twin  (section)",
        "Embedded HTML",
        "Same pack mesh as the 3D Twin page.",
        "page_dashboard_builder() · app.py",
    ),
    (
        "CAD twin (APS)  (section)",
        "Embedded HTML",
        "GuiViewer3D when ready; otherwise the waiting-for-credentials / URN placeholder.",
        "page_dashboard_builder() · app.py → cad_slot_status()",
    ),
    (
        "Export dashboard (HTML) — KPIs + charts + 3D (CAD slot placeholder)",
        "Download",
        "Label becomes “… + CAD” when APS is ready. Writes a standalone HTML via <font face='DejaVuMono'>compose_dashboard_html()</font>. File <font face='DejaVuMono'>reliability_dashboard_YYYYMMDD.html</font>.",
        "page_dashboard_builder() · app.py → compose_dashboard_html() · src/dashboard_composer.py",
    ),
]

PAGE_EMAIL: list[tuple[str, str, str, str]] = [
    (
        "Recipient Email",
        "Text input",
        "Placeholder manager@company.com.",
        "page_email_report() · app.py",
    ),
    (
        "Manager Name",
        "Text input",
        "Default Manager.",
        "page_email_report() · app.py",
    ),
    (
        "Email Subject",
        "Text input",
        "Default Predictive Maintenance Report — YYYY-MM-DD.",
        "page_email_report() · app.py",
    ),
    (
        "Additional Notes (optional)",
        "Text area",
        "Appended to the generated body. First 5 insight cards are also appended when present.",
        "page_email_report() · app.py",
    ),
    (
        "Generate Email Preview",
        "Button",
        "Primary. Builds a text body from predictions, anomaly summary, insights, pack KPIs. Stores email_preview / email_subject / email_recipient.",
        "page_email_report() · app.py → generate_email_body() · src/email_report.py",
    ),
    (
        "Email Preview",
        "Text block",
        "Shown after Generate. This is what will be sent or downloaded.",
        "page_email_report() · app.py",
    ),
    (
        "Send Email",
        "Button",
        "Primary. SMTP if SMTP_USER is set and EMAIL_DEMO_MODE is false; otherwise demo mode writes a .txt under output/emails/ and says so. SMTP failure also falls back to file. Do not claim “we emailed the plant” unless SMTP is configured.",
        "page_email_report() · app.py → send_email() · src/email_report.py",
    ),
    (
        "Download Email as TXT",
        "Download",
        "File maintenance_report_YYYYMMDD.txt.",
        "page_email_report() · app.py",
    ),
]

PAGE_ASK: list[tuple[str, str, str, str]] = [
    (
        "Ask this upload  (chat input)",
        "Chat input",
        "Same path as Insights. Placeholder: Ask about an asset, inspect list, or $ at risk on this upload…. Grounded in this upload only.",
        "page_ai_assistant() → render_ask_panel() · app.py → ask_with_index() · src/insights_engine.py",
    ),
    (
        "Clear Ask",
        "Button",
        "Clears chat_history. Hidden until there is history.",
        "render_ask_panel() · app.py",
    ),
]

PAGE_SQL: list[tuple[str, str, str, str]] = [
    (
        "DWDM concept map",
        "Table",
        "Ten rows from DWDM_CONCEPTS: ETL, Joins, 19-stage quality, binning, smoothing, association rules, outlier mining, concept drift, OLAP aggregation, SQL.",
        "page_dwdm_sql() · app.py · DWDM_CONCEPTS · src/dwdm_sql.py",
    ),
    (
        "Bin columns",
        "Multiselect",
        "Creates <font face='DejaVuMono'>{col}_dwdm_bin</font> via qcut.",
        "page_dwdm_sql() · app.py → apply_dwdm_transforms() · src/dwdm_sql.py",
    ),
    (
        "Smooth columns",
        "Multiselect",
        "Rolling-mean <font face='DejaVuMono'>{col}_dwdm_smooth</font>. Defaults to temp/vib-like names.",
        "page_dwdm_sql() · app.py",
    ),
    (
        "Z-normalize",
        "Multiselect",
        "Adds z-scored copies of selected numeric columns.",
        "page_dwdm_sql() · app.py",
    ),
    (
        "Apply transforms",
        "Button",
        "Writes cleaned_df and registers table cleaned. Shows first 10 rows.",
        "page_dwdm_sql() · app.py → apply_dwdm_transforms() · src/dwdm_sql.py",
    ),
    (
        "SQL",
        "Text area",
        "Read-only SELECT / WITH over registered tables (DuckDB, pandas fallback). Example queries are printed above the box.",
        "page_dwdm_sql() · app.py → default_sql_examples() · src/dwdm_sql.py",
    ),
    (
        "Run SQL",
        "Button",
        "Primary. <font face='DejaVuMono'>run_sql()</font> → sql_result table. Engine caption shows duckdb or pandas.",
        "page_dwdm_sql() · app.py → run_sql() · src/dwdm_sql.py",
    ),
    (
        "SQL result",
        "Table",
        "Full result frame after Run SQL.",
        "page_dwdm_sql() · app.py",
    ),
]

CHECKS_19 = [
    ("1", "NULLS / Missing%", "FAIL if &gt;20% missing, WARN if &gt;5%."),
    ("2", "CONSTANT", "FAIL if any column has ≤1 distinct value."),
    ("3", "ZEROS", "WARN if &gt;30% numeric zeros."),
    ("4", "DUPLICATES", "WARN if duplicate rows exist."),
    ("5", "Z-SCORE (&gt;3σ)", "WARN on |z|&gt;3 hits per numeric column."),
    ("6", "IQR OUTLIER", "WARN on Tukey 1.5·IQR outliers."),
    ("7", "ISOLATION FOREST", "Quality-time IF (contamination 0.08) — not the page-4 model."),
    ("8", "DBSCAN NOISE", "WARN on DBSCAN noise points."),
    ("9", "KMEANS DISTANCE", "WARN on points far from cluster centers."),
    ("10", "ROLLING IMPOSSIBLE JUMP", "FAIL on physically huge temp/vib/pressure/rpm steps."),
    ("11", "LAG / SENSOR CORRELATION", "WARN if temp/pressure/vib pairs are dead (~0 corr)."),
    ("12", "GE EXPECTATIONS", "Nulls, temp [0,200], RUL trend, unique timestamp+asset, row count."),
    ("13", "YDATA CARDINALITY", "High-cardinality columns; optional ydata-profiling."),
    ("14", "CLEANLAB / DIRTY LABELS", "Outliers + failure=1/low-vib and failure=0/extreme-vib flags."),
    ("15", "PCA / CONCEPT DRIFT", "Early vs late explained-variance shift &gt; 0.15."),
    ("16", "DOMAIN OPC / PHYSICS RULES", "Stuck sensor, leak suspect, unusual RUL, missed failures."),
    ("17", "ASSOCIATION RULE MINING", "HIGH-sensor co-occurrence baskets (Apriori-style proxy)."),
    ("18", "SCHEMA / ROWCOUNT", "FAIL if 0 rows."),
    ("19", "TIMESTAMP PRESENT", "WARN if no timestamp/date/time/datetime column."),
]

SESSION_KEYS_WIDGET = [
    ("industry_pack", "sidebar Field selectbox", "Never assign after render_pack_sidebar(). Queue _pending_industry_pack."),
    ("pipeline_page", "sidebar Pipeline radio", "Never assign after main() radio. Queue _pending_pipeline_page."),
    ("aps_urn_input", "CAD URN text input", "Seed in _seed_cad_urn_widgets() before the widget."),
    ("aps_zip_root_filename", "ZIP root text input", "Seed in _seed_zip_root_widget() / PENDING_ZIP_ROOT."),
    ("gemini_key_input", "Paste Gemini API key", "Read on Save key; do not overwrite after widget."),
    ("upload_sensor_files", "CSV uploader", "Streamlit-owned."),
    ("upload_url_input / upload_url_ingest_mode / upload_url_sql / upload_url_force_cache", "URL tab widgets", "Streamlit-owned."),
    ("upload_demo_pack / upload_maintenance_table", "Upload extras", "Streamlit-owned."),
    ("aps_cad_file_uploader / cad_gui_viewer / cad_region_pick / cad_region_include_related", "CAD Twin widgets", "Viewer is a custom component."),
    ("aps_asset_pick / twin_asset_pick / twin_source", "Asset / risk pickers", "Widget-bound."),
    ("dash_tile_kpis / dash_tile_charts / dash_tile_insights / dash_tile_twin3d / dash_tile_cad", "Dashboard tiles", "Widget-bound."),
    ("map_<canonical>", "Map sensors selectboxes", "One key per canonical / extra field."),
]

SESSION_KEYS_SAFE = [
    ("raw_df / cleaned_df / uploaded_tables / data_loaded", "Working tables"),
    ("cleaning_summary / quality_report / insights", "Clean outputs"),
    ("sensor_mapping / predictions / anomaly_detector / rul_predictor / anomaly_summary", "ML"),
    ("optuna_rul / optuna_anomaly / rul_used_synthetic", "Tuning + honesty flag"),
    ("business_insights / asset_ranking / inspect_list / insight_index / polished_brief", "Insights"),
    ("chat_history / last_gemini_error / last_insight_error / gemini_api_key_override", "Ask / Gemini"),
    ("graph_folder / dashboard_tiles / pack_kpis / pack_suggest / cost_* / hours_if_stop", "Charts + $"),
    ("url_ingest_* / maintenance_table_attached / clean_engine", "Ingest + engine"),
    ("live_running / live_tick / live_buffer / live_source / live_cfg / live_asset_states / live_conn_id", "Live Connect"),
    ("aps_urn / aps_translate_log / aps_save_log / cad_selected_part / cad_region_report / cad_region_assign_log", "CAD Twin session (not widget keys)"),
    ("join_log / sql_result / email_preview / email_subject / email_recipient", "Joins / SQL / Email"),
    ("_pending_industry_pack / _pending_pipeline_page / _pending_aps_zip_root_filename", "Pre-widget queues"),
]

PAGES = [
    ("1. Upload & Clean", "page_upload_clean", PAGE_UPLOAD),
    ("2. Joins", "page_data_integration", PAGE_JOINS),
    ("3. Map sensors", "page_map_sensors", PAGE_MAP),
    ("4. Anomaly & RUL", "page_ml_predictions", PAGE_ML),
    ("5. Charts", "page_explore_graphs", PAGE_CHARTS),
    ("6. Insights", "page_business_insights", PAGE_INSIGHTS),
    ("3D Twin", "page_twin_3d", PAGE_TWIN),
    ("CAD Twin (APS)", "page_cad_twin", PAGE_CAD),
    ("Live Connect", "page_live_connect", PAGE_LIVE),
    ("7. Dashboard", "page_dashboard_builder", PAGE_DASH),
    ("Email Report", "page_email_report", PAGE_EMAIL),
    ("Ask", "page_ai_assistant", PAGE_ASK),
    ("SQL lab", "page_dwdm_sql", PAGE_SQL),
]


def build_story(S: dict) -> list:
    story: list = []
    today = date.today().isoformat()

    story += [
        Spacer(1, 28 * mm),
        P("SMART INDIA HACKATHON  ·  PROBLEM SIH26054", S["cover_kicker"]),
        P("Student Viva Guide", S["cover_title"]),
        P("Predictive Maintenance Analytics — every sidebar page, every button", S["cover_sub"]),
        Spacer(1, 4 * mm),
        P(
            "Repo: predictive-maintenance-analytics  ·  entry: <font face='DejaVuMono'>streamlit run app.py</font><br/>"
            f"Generated {today} from the live <font face='DejaVuMono'>app.py</font> control inventory.",
            S["cover_sub"],
        ),
        Spacer(1, 8 * mm),
        P(
            "<b>Say this in viva, then stop:</b> this app is a <b>reliability add-on</b> "
            "(sensor CSV → clean → map → Isolation Forest → RUL/risk → charts → insights → optional 3D/CAD twin). "
            "It is <b>not</b> a FADEC, <b>not</b> a GCS, <b>not</b> Forge, and <b>not</b> an OEE cockpit. "
            "SIH26054 is mapped honestly as health + digital twin + fault prediction + mission reliability "
            "on uploaded aero-piston / MALE-UAV sensor rows.",
            S["note"],
        ),
        P(
            "This PDF names every left-sidebar <b>Pipeline</b> radio page and, on each page, every "
            "<b>button</b>, download, file uploader, tab, expander, and main table — using the label "
            "shown in the UI, what it does, and the function/file that implements it.",
            S["body"],
        ),
        PageBreak(),
        P("0. How to use this booklet in viva", S["h1"]),
        bullets(
            [
                "Open the app. Point at the left sidebar <b>Pipeline</b> radio — there are exactly 13 pages.",
                "For any button the examiner clicks, this booklet has the exact label, the session keys it writes, and the Python function.",
                "If asked “did you build FADEC / GCS?” — answer no, then map SIH26054 to what you did build (below).",
                "If asked “19 checks or 20?” — 19. Named list is in chapter 2. Constant is <font face='DejaVuMono'>QUALITY_STAGE_COUNT = 19</font> in <font face='DejaVuMono'>src/quality_checks.py</font>.",
                "If asked pandas vs Polars — pandas is default industrial ETL; Polars is the opt-in columnar engine (no JVM). Quality report always runs in pandas after the engine step.",
                "If asked why the app crashed after “Suggest pack” — Streamlit forbids writing a widget-bound session key after that widget exists. We queue <font face='DejaVuMono'>_pending_*</font> keys instead.",
            ],
            S,
        ),
        P("0.1 What this is — and what it is not (SIH26054 / FADEC / GCS)", S["h2"]),
        P(
            "<b>SIH26054 (DRDO)</b> in this repo is the Aviation pack: "
            "<i>digital twin, health, fault prediction, and mission reliability for aero piston engines on MALE UAVs</i>. "
            "Code: pack id <font face='DejaVuMono'>aviation_uav_piston</font>, "
            "<font face='DejaVuMono'>sih=\"SIH26054\"</font>, hero star in the sidebar Field picker, "
            "demo CSV <font face='DejaVuMono'>sample_data/aviation_uav_piston.csv</font>, "
            "3D kind <font face='DejaVuMono'>aviation_uav</font>, hotspot “piston engine”. "
            "That is a <b>reliability / twin</b> mapping of the problem statement — not a flight-control product.",
            S["body"],
        ),
        P(
            "<b>Not FADEC.</b> Full Authority Digital Engine Control commands fuel, spark, and limits on a live engine. "
            "This app never writes a throttle, mixture, or ignition output. Isolation Forest scores uploaded (or simulated) "
            "rows; Random Forest predicts remaining useful life / risk. There is no closed-loop actuator path.",
            S["note"],
        ),
        P(
            "<b>Not GCS.</b> A Ground Control Station uplinks commands, shows a moving map, and flies the UAV. "
            "This app has no stick/rudder, no datalink, no flight plan, no lost-link procedure. "
            "Live Connect can stream JSON/MQTT/OPC-UA <i>sensor</i> payloads into the same PdM buffer — that is ingest, not command-and-control.",
            S["note"],
        ),
        P(
            "<b>Also not:</b> an airliner cabin twin; a jet / turbofan catalog; every airframe OEM; "
            "a certified airworthiness system; Forge (generic analytics OS); OEE Pulse (plant SaaS). "
            "Automotive is one generic ICE car — not every OEM, trim, or EV architecture. "
            "Oil is SIH26120 one SRP well — not a field SCADA historian.",
            S["body"],
        ),
        P(
            "<b>Honest one-liner:</b> “We implemented SIH26054 as sensor health + Isolation Forest + RUL + an offline UAV/piston twin "
            "and an optional Autodesk CAD twin. We did not implement FADEC or a GCS.”",
            S["ok"],
        ),
        P("0.2 Pipeline radio — all 13 pages", S["h2"]),
        P(
            "Created in <font face='DejaVuMono'>main()</font> as <font face='DejaVuMono'>st.sidebar.radio(\"Pipeline\", list(pages.keys()), key=\"pipeline_page\")</font>.",
            S["body"],
        ),
    ]

    toc_rows = [
        [
            P("#", S["th"]),
            P("Sidebar label (exact)", S["th"]),
            P("page_* function", S["th"]),
            P("Role", S["th"]),
        ]
    ]
    roles = [
        "Load CSV / URL, pandas|Polars clean, 19 checks",
        "Optional sensor ⋈ maintenance ⋈ cost",
        "Canonical + pack extra column map",
        "Isolation Forest + Random Forest RUL",
        "Primary + extra Plotly views",
        "Inspect list, $ rates, Ask panel",
        "Offline pack mesh, risk-colored hotspot",
        "Autodesk GuiViewer3D, URN, region tint",
        "Sim / poll / MQTT / OPC-UA stream",
        "Power BI-style tiles + HTML export",
        "Preview / SMTP or demo-file send",
        "Same Ask path as Insights",
        "Optional DWDM transforms + read-only SQL",
    ]
    for i, ((label, fn, rows), role) in enumerate(zip(PAGES, roles), 1):
        toc_rows.append(
            [
                P(str(i), S["cell"]),
                P(label, S["cell_b"]),
                P(fn, S["mono"]),
                P(role, S["cell"]),
            ]
        )
    toc = Table(toc_rows, colWidths=[12 * mm, 48 * mm, 52 * mm, 74 * mm], repeatRows=1)
    toc.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.25, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
            ]
            + [("BACKGROUND", (0, i), (-1, i), ROW_ALT) for i in range(2, 14, 2)]
        )
    )
    story.append(toc)
    story.append(Spacer(1, 4 * mm))
    story.append(
        P(
            "Sidebar chrome (Industry pack + Gemini + status) is documented next, then each of the 13 pages.",
            S["body"],
        )
    )

    story += [
        PageBreak(),
        P("1. Left sidebar chrome (always visible)", S["h1"]),
        P(
            "Rendered by <font face='DejaVuMono'>main()</font> before the selected page function. "
            "Pack selectbox and Pipeline radio are widget-bound. Gemini sits under the radio.",
            S["body"],
        ),
        control_table(SIDEBAR_CONTROLS, S),
        P("1.1 Session keys — widget-bound vs safe writes", S["h2"]),
        P(
            "Streamlit raises <font face='DejaVuMono'>StreamlitAPIException</font> if you assign "
            "<font face='DejaVuMono'>st.session_state.&lt;key&gt;</font> after a widget with that key "
            "already exists in the same run. Only these functions may write widget-bound keys, because "
            "they run <b>before</b> those widgets: "
            "<font face='DejaVuMono'>init_session_state</font>, "
            "<font face='DejaVuMono'>apply_pending_industry_pack</font>, "
            "<font face='DejaVuMono'>apply_pending_pipeline_page</font>, "
            "<font face='DejaVuMono'>_seed_cad_urn_widgets</font>, "
            "<font face='DejaVuMono'>_seed_zip_root_widget</font> "
            "(frozenset <font face='DejaVuMono'>PRE_WIDGET_SESSION_FUNCS</font> in app.py). "
            "Smoke test <font face='DejaVuMono'>test_widget_bound_session_keys_are_not_written_after_widget</font> "
            "enforces this.",
            S["ok"],
        ),
    ]

    wk = [
        [P("Widget key", S["th"]), P("Widget", S["th"]), P("Rule", S["th"])]
    ]
    for key, widget, rule in SESSION_KEYS_WIDGET:
        wk.append([P(key, S["mono"]), P(widget, S["cell"]), P(rule, S["cell"])])
    wt = Table(wk, colWidths=[62 * mm, 42 * mm, 82 * mm], repeatRows=1)
    wt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
            ]
            + [("BACKGROUND", (0, i), (-1, i), ROW_ALT) for i in range(2, len(wk), 2)]
        )
    )
    story.append(P("Widget-bound keys (do not assign after the widget exists)", S["h3"]))
    story.append(wt)

    sk = [[P("Session key(s)", S["th"]), P("Owner", S["th"])]]
    for key, owner in SESSION_KEYS_SAFE:
        sk.append([P(key, S["mono"]), P(owner, S["cell"])])
    stbl = Table(sk, colWidths=[120 * mm, 66 * mm], repeatRows=1)
    stbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
            ]
            + [("BACKGROUND", (0, i), (-1, i), ROW_ALT) for i in range(2, len(sk), 2)]
        )
    )
    story.append(P("Safe / non-widget session keys (init_session_state defaults + page writes)", S["h3"]))
    story.append(stbl)
    story.append(
        P(
            "Queue helpers: <font face='DejaVuMono'>request_industry_pack(id)</font> writes "
            "<font face='DejaVuMono'>_pending_industry_pack</font>; "
            "<font face='DejaVuMono'>request_pipeline_page(label)</font> writes "
            "<font face='DejaVuMono'>_pending_pipeline_page</font>. "
            "Both are applied at the top of <font face='DejaVuMono'>main()</font> before sidebar widgets.",
            S["body"],
        )
    )

    story += [
        PageBreak(),
        P("2. Nineteen quality checks + pandas / Polars", S["h1"]),
        P(
            "Button label on Upload &amp; Clean: <b>Run industrial clean + 19 quality checks</b>. "
            "Constant <font face='DejaVuMono'>QUALITY_STAGE_COUNT = 19</font>. "
            "The report table is built by <font face='DejaVuMono'>build_quality_report()</font> "
            "in <font face='DejaVuMono'>src/quality_checks.py</font>. Do not say “20 stages” or “GE only”.",
            S["body"],
        ),
        P("2.1 Engine order (say this if asked pandas vs Polars)", S["h2"]),
        bullets(
            [
                "<b>pandas</b> (default): <font face='DejaVuMono'>clean_and_quality()</font> in src/data_cleaner.py runs industrial ETL (schema normalize, dedupe, sentinel-null fusion, casts, regression/median/mode impute, binning, sensor smoothing) then the 19 checks.",
                "<b>Polars</b> (opt-in, no JVM): <font face='DejaVuMono'>clean_with_polars()</font> in src/polars_clean.py does unique(), sentinel fusion, Float64 cast, median fill, IQR cap, then hands a pandas frame to the same <font face='DejaVuMono'>clean_and_quality()</font>. PySpark was removed.",
                "If Polars raises, the UI warns and falls back to pandas. Quality stages always run in pandas after the engine step.",
                "Session key <font face='DejaVuMono'>clean_engine</font> is \"pandas\" or \"polars\".",
            ],
            S,
        ),
    ]
    ck = [
        [P("#", S["th"]), P("Check name (exact in the report table)", S["th"]), P("Pass / warn rule", S["th"])]
    ]
    for n, name, rule in CHECKS_19:
        ck.append([P(n, S["cell"]), P(name, S["cell_b"]), P(rule, S["cell"])])
    ct = Table(ck, colWidths=[12 * mm, 62 * mm, 112 * mm], repeatRows=1)
    ct.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0),
            ]
            + [("BACKGROUND", (0, i), (-1, i), ROW_ALT) for i in range(2, len(ck), 2)]
        )
    )
    story.append(ct)
    story.append(
        P(
            "Sub-reports under the table: expanders <b>Great Expectations — column expectations</b>, "
            "<b>ydata / Cleanlab / PCA drift / Association rules</b>, and (when flags exist) "
            "<b>Domain OPC / physics-rule violations</b>. Those are the same 19-stage run, not extra stages.",
            S["body"],
        )
    )

    story += [
        PageBreak(),
        P("3. Industry packs (sidebar Field) — honesty you must keep", S["h1"]),
        P(
            "One CSV → one pack. A machine_id prefix alone does not turn the mesh into an airplane. "
            "Use <b>Suggest pack from columns</b> or pick Field yourself.",
            S["body"],
        ),
    ]
    pk = [
        [
            P("Sidebar Field label", S["th"]),
            P("id / SIH", S["th"]),
            P("Demo CSV", S["th"]),
            P("Honest scope", S["th"]),
        ],
        [
            P("Plant / rotating machines  (default)", S["cell_b"]),
            P("plant_rotating", S["mono"]),
            P("sensor_readings.csv", S["mono"]),
            P("Generic motor / pump / compressor. Not a full plant twin, not OEM motor CAD.", S["cell"]),
        ],
        [
            P("Aviation / aero piston / MALE UAV  ★ SIH", S["cell_b"]),
            P("aviation_uav_piston · SIH26054", S["mono"]),
            P("aviation_uav_piston.csv", S["mono"]),
            P(
                "DRDO mapping: health + twin + fault prediction + mission reliability on one UAV + piston. "
                "Not FADEC, not GCS, not airliner, not turbofan, not every OEM.",
                S["cell"],
            ),
        ],
        [
            P("Automotive powertrain", S["cell_b"]),
            P("automotive_powertrain", S["mono"]),
            P("automotive_powertrain.csv", S["mono"]),
            P("One generic ICE car + gearbox. Not every OEM / trim / EV architecture.", S["cell"]),
        ],
        [
            P("Oil well / sucker-rod pump", S["cell_b"]),
            P("oil_srp · SIH26120", S["mono"]),
            P("oil_srp.csv", S["mono"]),
            P("One beam-pump well (Oil India smart automation). Not every completion, not a reservoir model.", S["cell"]),
        ],
    ]
    pt = Table(pk, colWidths=[48 * mm, 40 * mm, 38 * mm, 60 * mm], repeatRows=1)
    pt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#fdebd0")),
            ]
        )
    )
    story.append(pt)

    story += [
        P("3.1 RUL honesty (shown on Map / Anomaly / Insights)", S["h2"]),
        P(
            "<font face='DejaVuMono'>config.RUL_HONESTY_CAPTION</font>: honest RUL needs a real "
            "<font face='DejaVuMono'>failure_within_days</font> (or alias). Sample CSV labels are "
            "<b>simulated</b>. Without a label the model trains on a sensor-degradation proxy — treat "
            "days-to-fail as a demo, not a plant or sortie forecast. "
            "Session flag <font face='DejaVuMono'>rul_used_synthetic</font>. "
            "Optuna tune RUL refuses to run without the label column.",
            S["note"],
        ),
        P("3.2 CAD / twin honesty", S["h2"]),
        P(
            "Isolation Forest scores the <b>asset</b>, not each CAD solid. "
            "A High asset does not redden the whole engine. Tint applies only to name-matched nodes "
            "or to dbIds you assigned under <b>Assign selected part to this region</b>. "
            "Solid1 / mojibake names match nothing — the model stays default dark gray. "
            "Driver line: largest |z-score| among mapped sensors vs other rows of that column "
            "(<font face='DejaVuMono'>DRIVER_HONESTY</font> in src/cad_part.py). "
            "3D Twin is a library mesh (no APS). CAD Twin needs APS secrets + a URN in <b>your</b> bucket.",
            S["body"],
        ),
    ]

    # Per-page chapters
    for idx, (label, fn, rows) in enumerate(PAGES, 1):
        story.append(PageBreak())
        story.append(P(f"{idx + 3}. Sidebar page: {label}", S["h1"]))
        story.append(
            P(
                f"Radio label <b>{label}</b>  ·  handler <font face='DejaVuMono'>{fn}()</font> in "
                f"<font face='DejaVuMono'>app.py</font>. Tables below list every button, download, "
                f"uploader, tab, expander, and main table on this page.",
                S["body"],
            )
        )
        extras = {
            "1. Upload & Clean": (
                "Need a raw frame first (sample, upload, or URL). Clean writes cleaned_df used by later steps."
            ),
            "2. Joins": "Skip if you only have one CSV. Plant sample registers three tables so this page is demoable.",
            "3. Map sensors": "Works on cleaned_df, else raw_df. Apply mapping overwrites cleaned_df.",
            "4. Anomaly & RUL": "Needs mapped sensors. Predictions drive Insights, 3D Twin, CAD tint, Dashboard, Email.",
            "5. Charts": "Generate buttons save into graph_folder for Dashboard export.",
            "6. Insights": "Refresh brief is enough after ML. Ask panel is the same function as the Ask page.",
            "3D Twin": "No buttons — radio / selectbox / mesh only. Risk is read-only from ML or Live.",
            "CAD Twin (APS)": "Most buttons live on this page. Public viewer.autodesk.com URNs will not load.",
            "Live Connect": "Start live feeds live_asset_states into 3D Twin.",
            "7. Dashboard": "Checkboxes toggle tiles; the only download is the HTML export.",
            "Email Report": "Send Email is demo-file unless SMTP is configured.",
            "Ask": "No unique buttons beyond Clear Ask — same render_ask_panel as Insights.",
            "SQL lab": "Optional warehouse lab. Core PdM does not require this step.",
        }
        story.append(P(extras.get(label, ""), S["ok"]))
        story.append(control_table(rows, S))

    story += [
        PageBreak(),
        P("17. Button checklist (examiner click-path)", S["h1"]),
        P(
            "Every <font face='DejaVuMono'>st.button</font> / <font face='DejaVuMono'>st.download_button</font> "
            "label in app.py, in sidebar-then-page order. If the examiner asks “what does this button do?”, "
            "find the page chapter above.",
            S["body"],
        ),
    ]
    checklist = [
        ("Sidebar", "Suggest pack from columns"),
        ("Sidebar", "Save key"),
        ("Sidebar", "Test Gemini"),
        ("1. Upload & Clean", "Load Plant sample (default)"),
        ("1. Upload & Clean", "Load pack demo"),
        ("1. Upload & Clean", "Apply preset to SQL"),
        ("1. Upload & Clean", "Load from URL"),
        ("1. Upload & Clean", "Run industrial clean + 19 quality checks"),
        ("1. Upload & Clean", "Download Cleaned CSV"),
        ("2. Joins", "Run join"),
        ("2. Joins", "Run join chain"),
        ("3. Map sensors", "Apply mapping"),
        ("4. Anomaly & RUL", "Detect anomalies + predict RUL"),
        ("4. Anomaly & RUL", "Optuna tune RUL"),
        ("4. Anomaly & RUL", "Optuna tune anomalies"),
        ("5. Charts", "Generate Sensor over time"),
        ("5. Charts", "Generate Anomaly flags"),
        ("5. Charts", "Generate Risk by asset"),
        ("5. Charts", "🔥 Sensor Correlation Heatmap"),
        ("5. Charts", "⚠ Anomaly Scatter Plot  (warning-emoji + name in UI)"),
        ("5. Charts", "📊 Distribution Histogram"),
        ("5. Charts", "📦 Box Plot by Machine"),
        ("5. Charts", "〰️ Rolling Average Trend"),
        ("5. Charts", "Remove {g_id}  (one per saved graph)"),
        ("6. Insights / Ask", "Refresh brief"),
        ("6. Insights / Ask", "Gemini polish (wording only)"),
        ("6. Insights / Ask", "Clear Ask"),
        ("CAD Twin (APS)", "Load CAD model"),
        ("CAD Twin (APS)", "Open Anomaly & RUL"),
        ("CAD Twin (APS)", "Assign selected part to this region"),
        ("CAD Twin (APS)", "Clear region map"),
        ("CAD Twin (APS)", "Translate to SVF / get URN"),
        ("CAD Twin (APS)", "Save URN for this pack"),
        ("CAD Twin (APS)", "Delete saved URN"),
        ("Live Connect", "▶ Start live"),
        ("Live Connect", "⏸ Stop"),
        ("Live Connect", "Reset buffer"),
        ("7. Dashboard", "Export dashboard (HTML) — KPIs + charts + 3D …"),
        ("Email Report", "Generate Email Preview"),
        ("Email Report", "Send Email"),
        ("Email Report", "Download Email as TXT"),
        ("SQL lab", "Apply transforms"),
        ("SQL lab", "Run SQL"),
    ]
    cl = [[P("#", S["th"]), P("Page", S["th"]), P("Button / download (exact UI label)", S["th"])]]
    for i, (page, btn) in enumerate(checklist, 1):
        cl.append([P(str(i), S["cell"]), P(page, S["cell"]), P(btn, S["cell_b"])])
    clt = Table(cl, colWidths=[12 * mm, 42 * mm, 132 * mm], repeatRows=1)
    clt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 1.8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
            ]
            + [("BACKGROUND", (0, i), (-1, i), ROW_ALT) for i in range(2, len(cl), 2)]
        )
    )
    story.append(clt)
    story.append(
        P(
            "3D Twin has no st.button (radio + selectbox + mesh only). Ask page reuses Clear Ask. "
            "Chat input on Insights / Ask is not a button — it is <font face='DejaVuMono'>st.chat_input</font>.",
            S["body"],
        )
    )

    story += [
        P("18. Likely viva questions — short honest answers", S["h1"]),
        P("Q. Is this FADEC or a GCS?", S["h3"]),
        P(
            "No. FADEC commands the engine; GCS flies the UAV. We mapped SIH26054 as health + twin + "
            "fault prediction + mission reliability on sensor rows. Isolation Forest does not move a throttle.",
            S["body"],
        ),
        P("Q. Why 19 checks, not Great Expectations alone?", S["h3"]),
        P(
            "GE-style expectations are check 12 plus a sub-report. The other 18 cover nulls, constants, zeros, "
            "duplicates, z/IQR, a quality-time Isolation Forest, DBSCAN, KMeans, impossible jumps, dead correlations, "
            "ydata cardinality, Cleanlab dirty labels, PCA drift, domain physics, association rules, schema, timestamp. "
            "Optional libraries (GE / ydata / Cleanlab) can be missing — the proxy still runs.",
            S["body"],
        ),
        P("Q. pandas or Polars?", S["h3"]),
        P(
            "pandas is the default industrial cleaner. Polars is an opt-in first pass (columnar, no JVM). "
            "The 19-stage report always runs on the pandas frame afterwards. PySpark is gone.",
            S["body"],
        ),
        P("Q. Why did Streamlit throw when we set industry_pack after Suggest pack?", S["h3"]),
        P(
            "The Field selectbox owns key industry_pack. We set _pending_industry_pack and rerun; "
            "apply_pending_industry_pack() copies it before the widget is created. Same pattern for "
            "pipeline_page (Open Anomaly &amp; RUL) and aps_zip_root_filename / aps_urn_input.",
            S["body"],
        ),
        P("Q. Does CAD Twin redden the whole Rotax when risk is High?", S["h3"]),
        P(
            "No. Only assigned dbIds or name-matched nodes tint. Solid1-only STEP stays gray. "
            "Click a part → Assign selected part to this region. Isolation Forest scores the asset, not the solid.",
            S["body"],
        ),
        P("Q. Will Send Email reach the manager?", S["h3"]),
        P(
            "Only if SMTP_USER / SMTP_PASSWORD are set and EMAIL_DEMO_MODE is false. Otherwise the file is "
            "saved under output/emails/ and the UI says demo mode. That is honest.",
            S["body"],
        ),
        P("Q. What did you write vs what is a library?", S["h3"]),
        P(
            "We wrote the Streamlit pipeline, industrial cleaner, 19-check report, pack layer, twins, APS glue, "
            "and this viva map. Isolation Forest / Random Forest are scikit-learn. Optuna tunes hyperparameters. "
            "DuckDB slices URLs. Autodesk APS translates and views CAD. Gemini (optional) polishes wording and Ask. "
            "Do not claim you invented Isolation Forest.",
            S["body"],
        ),
        Spacer(1, 6 * mm),
        P(
            "End of guide. Source of truth is app.py pages dict + each page_* function. "
            "Regenerate: <font face='DejaVuMono'>python docs/generate_viva_guide.py</font>.",
            S["ok"],
        ),
    ]
    return story


def required_phrases() -> list[str]:
    """Strings that must appear in the built PDF (coverage contract)."""
    phrases = [
        "SIH26054",
        "FADEC",
        "GCS",
        "Not FADEC",
        "QUALITY_STAGE_COUNT",
        "19 quality checks",
        "NULLS / Missing%",
        "TIMESTAMP PRESENT",
        "pandas",
        "Polars",
        "clean_engine",
        "_pending_industry_pack",
        "_pending_pipeline_page",
        "PRE_WIDGET_SESSION_FUNCS",
        "1. Upload & Clean",
        "2. Joins",
        "3. Map sensors",
        "4. Anomaly & RUL",
        "5. Charts",
        "6. Insights",
        "3D Twin",
        "CAD Twin (APS)",
        "Live Connect",
        "7. Dashboard",
        "Email Report",
        "Ask",
        "SQL lab",
        "Suggest pack from columns",
        "Save key",
        "Test Gemini",
        "Load Plant sample (default)",
        "Load pack demo",
        "Apply preset to SQL",
        "Load from URL",
        "Run industrial clean + 19 quality checks",
        "Download Cleaned CSV",
        "Run join",
        "Run join chain",
        "Apply mapping",
        "Detect anomalies + predict RUL",
        "Optuna tune RUL",
        "Optuna tune anomalies",
        "Generate Sensor over time",
        "Generate Anomaly flags",
        "Generate Risk by asset",
        "Sensor Correlation Heatmap",
        "Anomaly Scatter Plot",
        "Distribution Histogram",
        "Box Plot by Machine",
        "Rolling Average Trend",
        "Refresh brief",
        "Gemini polish (wording only)",
        "Clear Ask",
        "Load CAD model",
        "Open Anomaly & RUL",
        "Assign selected part to this region",
        "Clear region map",
        "Translate to SVF / get URN",
        "Save URN for this pack",
        "Delete saved URN",
        "Start live",
        "Reset buffer",
        "Export dashboard (HTML)",
        "Generate Email Preview",
        "Send Email",
        "Download Email as TXT",
        "Apply transforms",
        "Run SQL",
        "File upload",
        "From URL (cloud / DuckDB)",
        "Great Expectations — column expectations",
        "Last join log",
        "More sensor views (optional)",
        "CAD source — upload, translate, URN, save",
        "How to generate a URN",
        "DWDM concept map",
        "industry_pack",
        "pipeline_page",
        "aviation_uav_piston",
    ]
    return phrases


def write_pdf(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    _register_fonts()
    S = _styles()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=16 * mm,
        bottomMargin=14 * mm,
        title="PdM SIH26054 Student Viva Guide",
        author="Predictive Maintenance Analytics",
        subject="Every sidebar page and every button — SIH26054 / FADEC-GCS honesty",
    )
    doc.build(build_story(S), onFirstPage=_header_footer, onLaterPages=_header_footer)
    return path.stat().st_size


def extract_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError:
            return ""
    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def _norm(s: str) -> str:
    return " ".join(s.replace("\u00ad", "").split())


def verify(path: Path) -> list[str]:
    text = extract_text(path)
    if not text.strip():
        return ["WARN: could not extract PDF text (install pypdf for coverage check)"]
    hay = _norm(text)
    missing = [p for p in required_phrases() if _norm(p) not in hay]
    return missing


def main() -> int:
    size = write_pdf(OUT_DOCS)
    if OUT_ARTIFACT.parent.is_dir():
        OUT_ARTIFACT.write_bytes(OUT_DOCS.read_bytes())
    print(f"Wrote {OUT_DOCS} ({size} bytes)")
    if OUT_ARTIFACT.exists():
        print(f"Copied {OUT_ARTIFACT} ({OUT_ARTIFACT.stat().st_size} bytes)")
    missing = verify(OUT_DOCS)
    if missing and missing[0].startswith("WARN"):
        print(missing[0])
        return 0
    if missing:
        print("MISSING phrases in PDF:")
        for m in missing:
            print(f"  - {m}")
        return 1
    print(f"Coverage OK — {len(required_phrases())} required phrases present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
