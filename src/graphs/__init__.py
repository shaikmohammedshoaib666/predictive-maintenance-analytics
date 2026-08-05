"""Graph generation modules for predictive maintenance analytics."""

from src.graphs.time_series import create_time_series_chart
from src.graphs.heatmap import create_correlation_heatmap
from src.graphs.anomaly_scatter import create_anomaly_scatter
from src.graphs.histogram import create_distribution_histogram
from src.graphs.boxplot import create_boxplot_by_machine
from src.graphs.rolling_trend import create_rolling_trend

GRAPH_TYPES = {
    "time_series": {
        "name": "Time Series Line Chart",
        "description": "Sensor readings over time",
        "icon": "📈",
        "create": create_time_series_chart,
    },
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
