"""Graph generation modules for predictive maintenance analytics."""

from __future__ import annotations

from src.graphs.anomaly_flags import create_anomaly_flags_chart
from src.graphs.anomaly_scatter import create_anomaly_scatter
from src.graphs.boxplot import create_boxplot_by_machine
from src.graphs.heatmap import create_correlation_heatmap
from src.graphs.histogram import create_distribution_histogram
from src.graphs.pack_kpis import create_asset_health_chart, create_pack_kpi_bars
from src.graphs.risk_by_asset import create_risk_by_asset_chart
from src.graphs.rolling_trend import create_rolling_trend
from src.graphs.time_series import create_time_series_chart

# Specialist PdM set — not a generic business dashboard pack.
PRIMARY_CHARTS = {
    "time_series": {
        "name": "Sensor over time",
        "description": "Temperature / vibration / pressure / RPM vs time",
        "icon": "📈",
        "create": create_time_series_chart,
    },
    "anomaly_flags": {
        "name": "Anomaly flags",
        "description": "Isolation Forest flags on the sensor timeline",
        "icon": "⚠️",
        "create": create_anomaly_flags_chart,
    },
    "risk_by_asset": {
        "name": "Risk by asset",
        "description": "Predicted RUL (days) colored by High / Medium / Low risk",
        "icon": "🔴",
        "create": create_risk_by_asset_chart,
    },
}

GRAPH_TYPES = {
    **PRIMARY_CHARTS,
    "heatmap": {
        "name": "Sensor Correlation Heatmap",
        "description": "Correlations between sensor metrics",
        "icon": "🔥",
        "create": create_correlation_heatmap,
    },
    "anomaly_scatter": {
        "name": "Anomaly Scatter Plot",
        "description": "Scatter with anomaly highlighting",
        "icon": "⚠️",
        "create": create_anomaly_scatter,
    },
    "histogram": {
        "name": "Distribution Histogram",
        "description": "Distribution of selected metric",
        "icon": "📊",
        "create": create_distribution_histogram,
    },
    "boxplot": {
        "name": "Box Plot by Machine",
        "description": "Metric distribution per machine",
        "icon": "📦",
        "create": create_boxplot_by_machine,
    },
    "rolling_trend": {
        "name": "Rolling Average Trend",
        "description": "Smoothed trend with rolling window",
        "icon": "〰️",
        "create": create_rolling_trend,
    },
}
