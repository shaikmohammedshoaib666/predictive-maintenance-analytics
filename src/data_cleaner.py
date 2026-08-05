"""Pandas-based data cleaning pipeline for IoT sensor readings."""

from typing import Tuple

import numpy as np
import pandas as pd


def clean_sensor_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Clean uploaded sensor CSV data.

    Steps:
    - Parse timestamps
    - Coerce numeric sensor columns
    - Handle missing values (forward-fill then median)
    - Cap outliers using IQR method
    - Standardize machine_id as string

    Returns cleaned DataFrame and a summary dict of cleaning actions.
    """
    cleaned = df.copy()
    summary = {"rows_before": len(cleaned), "actions": []}

    # Timestamp parsing
    if "timestamp" in cleaned.columns:
        cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
        null_ts = cleaned["timestamp"].isna().sum()
        if null_ts:
            summary["actions"].append(f"Dropped {null_ts} rows with invalid timestamps")
            cleaned = cleaned.dropna(subset=["timestamp"])
        cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)

    # Machine ID standardization
    if "machine_id" in cleaned.columns:
        cleaned["machine_id"] = cleaned["machine_id"].astype(str).str.strip()

    # Numeric columns
    sensor_cols = [c for c in ["temperature", "vibration", "pressure", "rpm"] if c in cleaned.columns]
    for col in sensor_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Target column
    if "failure_within_days" in cleaned.columns:
        cleaned["failure_within_days"] = pd.to_numeric(
            cleaned["failure_within_days"], errors="coerce"
        )

    # Missing values: forward-fill within machine, then median
    if "machine_id" in cleaned.columns and sensor_cols:
        cleaned[sensor_cols] = cleaned.groupby("machine_id")[sensor_cols].ffill()
    for col in sensor_cols:
        missing = cleaned[col].isna().sum()
        if missing:
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)
            summary["actions"].append(f"Filled {missing} missing values in '{col}' with median ({median_val:.2f})")

    # Outlier capping (IQR)
    for col in sensor_cols:
        q1, q3 = cleaned[col].quantile(0.25), cleaned[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((cleaned[col] < lower) | (cleaned[col] > upper)).sum()
        if outliers:
            cleaned[col] = cleaned[col].clip(lower, upper)
            summary["actions"].append(f"Capped {outliers} outliers in '{col}' using IQR bounds")

    summary["rows_after"] = len(cleaned)
    summary["columns"] = list(cleaned.columns)
    return cleaned, summary
