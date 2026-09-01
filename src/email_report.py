"""Email report generation and delivery for maintenance predictions."""

from __future__ import annotations

import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional

import config


def high_risk_predictions(predictions: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]]:
    """Assets whose risk_level maps to High. Used for the urgent email line."""
    from src.twin3d import normalize_risk

    out: list[dict[str, Any]] = []
    for p in predictions or []:
        if normalize_risk(p.get("risk_level")) == "High":
            out.append(p)
    out.sort(key=lambda p: float(p.get("predicted_rul_days") or 10**9))
    return out


def generate_email_body(
    manager_name: str,
    predictions: list[dict[str, Any]],
    anomaly_summary: Optional[dict] = None,
    insights: Optional[dict] = None,
    additional_notes: str = "",
    pack_kpis: Optional[dict[str, Any]] = None,
) -> str:
    """Auto-generate email body with failure predictions and data insights."""
    today = datetime.now().strftime("%B %d, %Y")
    lines = [
        f"Dear {manager_name},",
        "",
        f"Please find below the Predictive Maintenance Analytics report for {today}.",
        "",
        "This is a risk / degradation brief, not a confirmed failure date.",
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
        urgent = high_risk_predictions(predictions)
        if urgent:
            top = urgent[0]
            lines.extend(
                [
                    "",
                    f"⚠️  IMMEDIATE ACTION REQUIRED: {top.get('machine_id')} "
                    f"is High risk (predicted RUL {top.get('predicted_rul_days')} days). "
                    "Treat as inspect-this-week, not a calendar failure date.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "No High-risk assets in this run. Continue routine monitoring.",
                ]
            )
    else:
        lines.append("  No predictions available — please run ML model training.")

    bundle = pack_kpis or {}
    if bundle.get("kpis"):
        label = bundle.get("label") or "Pack"
        sih = f" ({bundle['sih']})" if bundle.get("sih") else ""
        lines.extend(["", f"━━━ {label.upper()} KPIs{sih} ━━━", ""])
        for k in bundle["kpis"]:
            lines.append(f"  • {k.get('label')}: {k.get('value')} {k.get('unit') or ''}".rstrip())
        if bundle.get("narrative"):
            lines.append(f"  • {bundle['narrative']}")

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

    high = high_risk_predictions(predictions)
    lines.extend([
        "",
        "━━━ RECOMMENDED ACTIONS ━━━",
        "",
    ])
    if high:
        lines.extend(
            [
                "  1. Inspect High-risk assets this week (see IMMEDIATE ACTION above).",
                "  2. Investigate anomalous sensor readings for root cause.",
                "  3. Review the reliability dashboard (KPIs, 3D twin, charts).",
            ]
        )
    else:
        lines.extend(
            [
                "  1. No High-risk assets — keep the current PM cadence.",
                "  2. Review pack KPIs and the 3D twin on the dashboard.",
                "  3. Re-run Anomaly & RUL after the next sensor drop.",
            ]
        )
    lines.extend([
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
