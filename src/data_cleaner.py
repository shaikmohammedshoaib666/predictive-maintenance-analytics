"""Industrial ETL cleaning + PdM sensor cleaning (Forge v2 DWDM techniques)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from src.quality_checks import build_quality_report


def industrial_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    DWDM-style ETL: schema normalize, dedupe, sentinel nulls, casts,
    regression/median/mode imputation, binning, sensor smoothing.
    """
    log: list[str] = []
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    log.append("ETL: strip column names / schema normalize")
    out = out.dropna(how="all")
    out = out.loc[:, ~out.columns.duplicated()]
    before = len(out)
    out = out.drop_duplicates()
    if len(out) != before:
        log.append(f"ETL dedupe {before}->{len(out)}")
    out = out.replace(["", "NA", "N/A", "null", "NULL", "None", "-", "--"], np.nan)
    log.append("ETL null-sentinel fusion")

    for c in list(out.columns):
        if out[c].dtype == object:
            converted = pd.to_numeric(out[c], errors="coerce")
            if out[c].notna().sum() and converted.notna().sum() / max(1, out[c].notna().sum()) >= 0.8:
                out[c] = converted
                log.append(f"schema cast numeric {c}")

    date_hints = ("date", "time", "timestamp", "datetime", "day")
    for c in list(out.columns):
        if any(h in str(c).lower() for h in date_hints) and out[c].dtype == object:
            parsed = pd.to_datetime(out[c], errors="coerce")
            if parsed.notna().sum() > 0:
                out[c] = parsed
                log.append(f"schema cast datetime {c}")

    for c in out.select_dtypes(include=[np.number]).columns:
        if out[c].isna().any():
            s = out[c]
            if s.notna().sum() >= 5:
                idx = np.arange(len(s))
                mask = s.notna().to_numpy()
                coef = np.polyfit(idx[mask], s.to_numpy()[mask], 1)
                pred = np.polyval(coef, idx)
                filled = s.copy()
                filled[s.isna()] = pred[s.isna()]
                out[c] = filled
                log.append(f"DWDM regression imputation {c}")
            else:
                out[c] = s.fillna(s.median())
                log.append(f"median imputation {c}")

    for c in out.select_dtypes(include=["object", "string", "category"]).columns:
        if out[c].isna().any():
            mode = out[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Unknown"
            out[c] = out[c].fillna(fill)
            log.append(f"mode imputation {c}")

    for c in list(out.select_dtypes(include=[np.number]).columns)[:6]:
        try:
            out[f"{c}_bin"] = pd.qcut(
                out[c], q=min(5, max(2, out[c].nunique())), duplicates="drop"
            ).astype(str)
            log.append(f"DWDM binning {c}")
        except Exception:
            pass

    for c in list(out.select_dtypes(include=[np.number]).columns):
        cl = str(c).lower()
        if (
            any(h in cl for h in ("temp", "vib", "pressure", "current", "voltage", "speed", "rpm"))
            and not cl.endswith("_smooth")
            and not cl.endswith("_bin")
        ):
            out[f"{c}_smooth"] = out[c].rolling(
                window=min(5, max(2, len(out) // 10)), min_periods=1
            ).mean()
            log.append(f"DWDM smoothing {c}")

    return out.reset_index(drop=True), log


def clean_sensor_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Predictive-maintenance sensor clean (legacy API) + industrial ETL extras.
    Returns cleaned DataFrame and summary with actions + optional quality hooks.
    """
    cleaned = df.copy()
    summary: dict[str, Any] = {"rows_before": len(cleaned), "actions": []}

    if "timestamp" in cleaned.columns:
        cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
        null_ts = cleaned["timestamp"].isna().sum()
        if null_ts:
            summary["actions"].append(f"Dropped {null_ts} rows with invalid timestamps")
            cleaned = cleaned.dropna(subset=["timestamp"])
        cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)

    if "machine_id" in cleaned.columns:
        cleaned["machine_id"] = cleaned["machine_id"].astype(str).str.strip()

    sensor_cols = [c for c in ["temperature", "vibration", "pressure", "rpm"] if c in cleaned.columns]
    for col in sensor_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if "failure_within_days" in cleaned.columns:
        cleaned["failure_within_days"] = pd.to_numeric(
            cleaned["failure_within_days"], errors="coerce"
        )

    if "machine_id" in cleaned.columns and sensor_cols:
        cleaned[sensor_cols] = cleaned.groupby("machine_id")[sensor_cols].ffill()
    for col in sensor_cols:
        missing = cleaned[col].isna().sum()
        if missing:
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)
            summary["actions"].append(
                f"Filled {missing} missing values in '{col}' with median ({median_val:.2f})"
            )

    for col in sensor_cols:
        q1, q3 = cleaned[col].quantile(0.25), cleaned[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((cleaned[col] < lower) | (cleaned[col] > upper)).sum()
        if outliers:
            cleaned[col] = cleaned[col].clip(lower, upper)
            summary["actions"].append(f"Capped {outliers} outliers in '{col}' using IQR bounds")

    # Layer Forge industrial DWDM transforms on top
    cleaned, etl_log = industrial_clean(cleaned)
    summary["actions"].extend(etl_log)

    summary["rows_after"] = len(cleaned)
    summary["columns"] = list(cleaned.columns)
    return cleaned, summary


def clean_and_quality(
    df: pd.DataFrame, run_quality: bool = True
) -> Tuple[pd.DataFrame, dict, Optional[dict]]:
    """Clean sensor data and optionally run the 19-stage quality suite."""
    cleaned, summary = clean_sensor_data(df)
    report = build_quality_report(cleaned) if run_quality else None
    return cleaned, summary, report
