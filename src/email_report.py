"""Email report generation and delivery for maintenance predictions."""

from __future__ import annotations

import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional

import config


def generate_email_body(
    manager_name: str,
    predictions: list[dict[str, Any]],
    anomaly_summary: Optional[dict] = None,
    insights: Optional[dict] = None,
    additional_notes: str = "",
) -> str:
    """Auto-generate email body with failure predictions and data insights."""
    today = datetime.now().strftime("%B %d, %Y")
    lines = [
        f"Dear {manager_name},",
        "",
        f"Please find below the Predictive Maintenance Analytics report for {today}.",
        "",
        "━━━ FAILURE PREDICTIONS ━━━",
        "",
    ]

    if predictions:
        for p in predictions:
            risk_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(p.get("risk_level", ""), "⚪")
            lines.append(
                f"  {risk_icon} {p.get('message', 'N/A')} — Risk Level: {p.get('risk_level', 'Unknown')}"
            )
        urgent = predictions[0]
        lines.extend([
            "",
            f"⚠️  IMMEDIATE ACTION REQUIRED: {urgent.get('machine_id')} "
            f"predicted to fail in {urgent.get('predicted_rul_days')} days.",
        ])
    else:
        lines.append("  No predictions available — please run ML model training.")

    if anomaly_summary:
        lines.extend([
            "",
            "━━━ ANOMALY DETECTION SUMMARY ━━━",
            "",
            f"  • Total records analyzed: {anomaly_summary.get('total_records', 'N/A'):,}",
            f"  • Anomalies detected: {anomaly_summary.get('anomaly_count', 0):,} "
            f"({anomaly_summary.get('anomaly_rate_pct', 0)}%)",
        ])

    if insights:
        lines.extend([
            "",
            "━━━ DATA QUALITY INSIGHTS ━━━",
            "",
            f"  • Dataset: {insights.get('shape', {}).get('rows', 'N/A'):,} rows",
            f"  • Quality Score: {insights.get('quality_score', 'N/A')}/100",
        ])
        recs = insights.get("recommendations", [])
        if recs:
            lines.append("  • Recommendations:")
            for r in recs[:3]:
                lines.append(f"    - {r}")

    if additional_notes:
        lines.extend(["", "━━━ ADDITIONAL NOTES ━━━", "", additional_notes])

    lines.extend([
        "",
        "━━━ RECOMMENDED ACTIONS ━━━",
        "",
        "  1. Schedule preventive maintenance for high-risk machines within predicted RUL window.",
        "  2. Investigate anomalous sensor readings for root cause analysis.",
        "  3. Review full dashboard in the Predictive Maintenance Analytics Platform.",
        "",
        "Best regards,",
        "Predictive Maintenance Analytics Platform",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ])
    return "\n".join(lines)


def send_email(
    recipient: str,
    subject: str,
    body: str,
    manager_name: str = "Manager",
) -> dict[str, Any]:
    """
    Send email via SMTP or save to file in demo mode.

    Returns status dict with success flag and message.
    """
    demo_mode = config.EMAIL_DEMO_MODE or not config.SMTP_USER

    if demo_mode:
        return _save_email_to_file(recipient, subject, body)

    try:
        msg = MIMEMultipart()
        msg["From"] = config.EMAIL_FROM
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            if config.SMTP_USE_TLS:
                server.starttls()
            if config.SMTP_USER and config.SMTP_PASSWORD:
                server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.send_message(msg)

        return {"success": True, "message": f"Email sent successfully to {recipient}", "mode": "smtp"}
    except Exception as e:
        result = _save_email_to_file(recipient, subject, body)
        result["message"] = f"SMTP failed ({e}). Email saved to file instead."
        result["smtp_error"] = str(e)
        return result


def _save_email_to_file(recipient: str, subject: str, body: str) -> dict[str, Any]:
    """Save email preview to output directory (demo mode)."""
    output_dir = config.EMAIL_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_recipient = recipient.replace("@", "_at_").replace(".", "_")
    filepath = output_dir / f"report_{safe_recipient}_{timestamp}.txt"

    content = f"To: {recipient}\nSubject: {subject}\n{'=' * 60}\n\n{body}"
    filepath.write_text(content)

    return {
        "success": True,
        "message": f"Email saved to {filepath} (demo mode — no SMTP configured)",
        "mode": "demo",
        "filepath": str(filepath),
    }
