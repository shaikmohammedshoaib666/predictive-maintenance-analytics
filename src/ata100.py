"""ATA-100 chapter tags for SIH26054 / DRDO-style fault language.

Ground-health mapping only — not a certified AMM or IETM. Judges (and DRDO)
recognise chapter numbers next to a physics red-line.
"""

from __future__ import annotations

from typing import Any, Optional

# Primary chapter per physics fault family.
FAULT_ATA: dict[str, tuple[str, str]] = {
    "Overheating": ("75-00", "Air / cooling"),
    "Low oil": ("79-00", "Oil"),
    "High vibration": ("72-00", "Engine"),
}

# Extra rule → chapter when a specific red-line fires (Meta / SIH language).
RULE_ATA: dict[str, tuple[str, str]] = {
    "CHT high": ("75-00", "Air / cooling"),
    "Temperature high": ("75-00", "Air / cooling"),
    "EGT high": ("72-00", "Engine"),
    "Oil pressure low": ("79-00", "Oil"),
    "Vibration high": ("72-00", "Engine"),
}

# Fuel / injector family — reserved for live fuel_flow residuals (PS ATA 73).
FUEL_ATA: tuple[str, str] = ("73-00", "Engine fuel and control")


def ata_pair(fault: Any = None, rule: Any = None) -> tuple[str, str]:
    """Return (chapter, title). Empty chapter when no red-line."""
    text = str(rule or "")
    for part in [p.strip() for p in text.split(";") if p.strip()]:
        if part in RULE_ATA:
            return RULE_ATA[part]
    fault_s = str(fault or "").strip()
    if fault_s in FAULT_ATA:
        return FAULT_ATA[fault_s]
    return ("", "")


def ata_label(fault: Any = None, rule: Any = None) -> str:
    chapter, title = ata_pair(fault, rule)
    if not chapter:
        return "—"
    return f"{chapter} {title}"


def ata_for_fault(fault: Optional[str] = None, rule: Optional[str] = None) -> str:
    return ata_label(fault, rule)
