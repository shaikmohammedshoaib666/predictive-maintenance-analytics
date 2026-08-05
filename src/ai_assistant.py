"""Rule-based, data-aware AI assistant for maintenance insights."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


class MaintenanceAIAssistant:
    """Generate insights from uploaded data without requiring an external API."""

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        predictions: Optional[list[dict]] = None,
        anomaly_summary: Optional[dict] = None,
        insights: Optional[dict] = None,
    ):
        self.df = df
        self.predictions = predictions or []
        self.anomaly_summary = anomaly_summary or {}
        self.insights = insights or {}

    def respond(self, query: str) -> str:
        """Generate a contextual response based on user query and data."""
        q = query.lower().strip()

        if self.df is None or self.df.empty:
            return "Please upload and clean data first so I can provide insights."

        if any(w in q for w in ["hello", "hi", "help", "what can you"]):
            return self._help_response()

        if any(w in q for w in ["fail", "failure", "rul", "predict", "when will"]):
            return self._prediction_response()

        if any(w in q for w in ["anomal", "outlier", "unusual", "abnormal"]):
            return self._anomaly_response()

        if any(w in q for w in ["quality", "missing", "null", "clean", "data quality"]):
            return self._quality_response()

        if any(w in q for w in ["machine", "equipment", "asset"]):
            return self._machine_response()

        if any(w in q for w in ["temperature", "vibration", "pressure", "rpm", "sensor", "stat"]):
            return self._sensor_response(q)

        if any(w in q for w in ["recommend", "action", "next step", "what should"]):
            return self._recommendation_response()

        if any(w in q for w in ["summary", "overview", "report"]):
            return self._summary_response()

        return self._default_response(q)

    def _help_response(self) -> str:
        return (
            "I'm your Predictive Maintenance AI Assistant. I can help with:\n\n"
            "- **Failure predictions** — ask about RUL or when machines will fail\n"
            "- **Anomaly analysis** — unusual sensor patterns\n"
            "- **Data quality** — missing values, column stats\n"
            "- **Machine insights** — per-equipment breakdown\n"
            "- **Sensor statistics** — temperature, vibration, pressure, RPM\n"
            "- **Recommendations** — suggested next steps\n\n"
            "Try: *Which machine is at highest risk?* or *Summarize anomaly rate*"
        )

    def _prediction_response(self) -> str:
        if not self.predictions:
            return "No predictions available yet. Go to **ML Predictions** and train the model first."
        lines = ["**Failure Predictions (RUL):**\n"]
        for p in self.predictions:
            emoji = "🔴" if p["risk_level"] == "High" else ("🟡" if p["risk_level"] == "Medium" else "🟢")
            lines.append(f"{emoji} {p['message']} — Risk: **{p['risk_level']}**")
        highest = self.predictions[0]
        lines.append(f"\n⚠️ **Highest priority:** {highest['machine_id']} ({highest['predicted_rul_days']} days)")
        return "\n".join(lines)

    def _anomaly_response(self) -> str:
        if not self.anomaly_summary:
            return "Run anomaly detection in **ML Predictions** to get anomaly insights."
        s = self.anomaly_summary
        return (
            f"**Anomaly Analysis:**\n\n"
            f"- Total records analyzed: {s.get('total_records', 'N/A'):,}\n"
            f"- Anomalies detected: **{s.get('anomaly_count', 0):,}** "
            f"({s.get('anomaly_rate_pct', 0)}%)\n"
            f"- Features used: {', '.join(s.get('features_used', []))}\n\n"
            f"Anomalies often indicate early degradation signals. "
            f"Cross-reference with RUL predictions for maintenance scheduling."
        )

    def _quality_response(self) -> str:
        if not self.insights:
            return "Upload data to see quality insights."
        score = self.insights.get("quality_score", 0)
        recs = self.insights.get("recommendations", [])
        lines = [f"**Data Quality Score:** {score}/100\n"]
        for col, info in self.insights.get("columns", {}).items():
            if info.get("null_pct", 0) > 0:
                lines.append(f"- `{col}`: {info['null_pct']}% missing")
        if recs:
            lines.append("\n**Recommendations:**")
            for r in recs:
                lines.append(f"- {r}")
        return "\n".join(lines)

    def _machine_response(self) -> str:
        if self.df is None or "machine_id" not in self.df.columns:
            return "No machine_id column found in the dataset."
        counts = self.df["machine_id"].value_counts()
        lines = ["**Machines in dataset:**\n"]
        for machine, count in counts.items():
            lines.append(f"- **{machine}**: {count:,} readings")
        if self.predictions:
            lines.append("\n**Risk ranking:**")
            for p in self.predictions:
                lines.append(f"- {p['machine_id']}: {p['predicted_rul_days']} days ({p['risk_level']} risk)")
        return "\n".join(lines)

    def _sensor_response(self, query: str) -> str:
        if self.df is None:
            return "No data loaded."
        sensor_cols = [c for c in ["temperature", "vibration", "pressure", "rpm"] if c in self.df.columns]
        target_cols = [c for c in sensor_cols if c in query] or sensor_cols
        lines = ["**Sensor Statistics:**\n"]
        for col in target_cols:
            s = self.df[col].describe()
            lines.append(
                f"- **{col}**: mean={s['mean']:.2f}, std={s['std']:.2f}, "
                f"min={s['min']:.2f}, max={s['max']:.2f}"
            )
        return "\n".join(lines)

    def _recommendation_response(self) -> str:
        recs = []
        if self.predictions:
            high_risk = [p for p in self.predictions if p["risk_level"] == "High"]
            if high_risk:
                recs.append(f"Schedule immediate inspection for: {', '.join(p['machine_id'] for p in high_risk)}")
        if self.anomaly_summary and self.anomaly_summary.get("anomaly_rate_pct", 0) > 5:
            recs.append("Anomaly rate exceeds 5% — investigate sensor calibration and operating conditions.")
        if self.insights and self.insights.get("quality_score", 100) < 80:
            recs.append("Improve data quality before retraining models (address missing values).")
        if not recs:
            recs = [
                "Continue monitoring all machines on regular schedule.",
                "Review rolling trend charts for early degradation patterns.",
                "Set up automated email reports for weekly stakeholder updates.",
            ]
        return "**Recommended Actions:**\n\n" + "\n".join(f"- {r}" for r in recs)

    def _summary_response(self) -> str:
        parts = []
        if self.df is not None:
            parts.append(f"Dataset: **{len(self.df):,}** records across **{self.df['machine_id'].nunique() if 'machine_id' in self.df.columns else 1}** machine(s).")
        if self.insights:
            parts.append(f"Quality score: **{self.insights.get('quality_score', 'N/A')}/100**.")
        if self.anomaly_summary:
            parts.append(f"Anomaly rate: **{self.anomaly_summary.get('anomaly_rate_pct', 0)}%**.")
        if self.predictions:
            parts.append(f"Most urgent: **{self.predictions[0]['message']}**.")
        return "\n\n".join(parts) if parts else "Upload data and train models for a full summary."

    def _default_response(self, query: str) -> str:
        return (
            f"I analyzed your query: *\"{query}\"*\n\n"
            "Based on the current dataset, I can provide insights on predictions, anomalies, "
            "data quality, machines, and sensors. Try asking:\n"
            "- *Which machine will fail soonest?*\n"
            "- *What's the anomaly rate?*\n"
            "- *Give me a summary*\n"
            "- *What actions do you recommend?*"
        )
