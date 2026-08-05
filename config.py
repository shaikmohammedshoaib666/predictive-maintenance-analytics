"""Application configuration for Predictive Maintenance Analytics Platform."""

import os
from pathlib import Path


def _setting(key: str, default: str = "") -> str:
    """Environment variable, then Streamlit Cloud secrets, then default."""
    value = os.getenv(key)
    if value is not None and value != "":
        return value
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


# Paths
PROJECT_ROOT = Path(__file__).parent
SAMPLE_DATA_PATH = PROJECT_ROOT / "sample_data" / "sensor_readings.csv"
GRAPHS_DIR = PROJECT_ROOT / "graphs"
DB_PATH = PROJECT_ROOT / "data" / "maintenance.db"

# SMTP settings (env, Streamlit secrets, or defaults)
SMTP_HOST = _setting("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(_setting("SMTP_PORT", "587"))
SMTP_USER = _setting("SMTP_USER", "")
SMTP_PASSWORD = _setting("SMTP_PASSWORD", "")
SMTP_USE_TLS = _setting("SMTP_USE_TLS", "true").lower() == "true"
EMAIL_FROM = _setting("EMAIL_FROM", SMTP_USER or "analytics@predictive-maintenance.local")

# Demo mode: save emails to file when SMTP is not configured
EMAIL_DEMO_MODE = _setting("EMAIL_DEMO_MODE", "true").lower() == "true"
EMAIL_OUTPUT_DIR = PROJECT_ROOT / "output" / "emails"

# ML settings
ANOMALY_CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", "0.05"))
RUL_RANDOM_STATE = 42
SENSOR_COLUMNS = ["temperature", "vibration", "pressure", "rpm"]
ROLLING_WINDOW = 24

# App settings
APP_TITLE = "Predictive Maintenance Analytics Platform"
PAGE_ICON = "⚙️"
