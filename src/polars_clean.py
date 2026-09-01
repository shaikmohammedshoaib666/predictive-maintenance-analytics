"""Polars cleaning engine — fast columnar ETL without a JVM.

Replaces the old PySpark path. pandas remains the default engine; Polars is
the opt-in for larger CSVs on the same machine (Render-friendly, no Java).
"""

from __future__ import annotations

import pandas as pd

from src.industry_packs import OPTIONAL_IF_SENSORS

SENSOR_COLS = ("temperature", "vibration", "pressure", "rpm") + OPTIONAL_IF_SENSORS
_SENTINELS = ("", "NA", "N/A", "null", "NULL", "None", "-", "--")


def polars_available() -> tuple[bool, str]:
    """Return (available, message). Never raises."""
    try:
        import polars as pl  # noqa: F401
    except Exception as exc:
        return False, f"polars not installed ({exc.__class__.__name__})"
    return True, f"polars {pl.__version__}"


def clean_with_polars(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Columnar dedupe / cast / median-impute / IQR cap, then back to pandas.

    Raises RuntimeError when Polars is missing so callers can fall back to pandas.
    """
    ok, msg = polars_available()
    if not ok:
        raise RuntimeError(f"Polars engine unavailable: {msg}")

    import polars as pl

    log: list[str] = [f"Polars engine: {msg}"]
    if df is None or df.empty:
        log.append("empty frame — nothing to clean")
        return pd.DataFrame() if df is None else df.copy(), log

    pdf = df.copy()
    pdf.columns = [str(c).strip().replace(" ", "_") for c in pdf.columns]
    before = len(pdf)
    frame = pl.from_pandas(pdf)
    log.append(f"loaded {before:,} rows into Polars")

    frame = frame.unique(keep="first", maintain_order=True)
    after = frame.height
    if after != before:
        log.append(f"unique() {before:,} -> {after:,}")

    str_cols = [c for c, dtype in zip(frame.columns, frame.dtypes) if dtype == pl.String or dtype == pl.Utf8]
    if str_cols:
        frame = frame.with_columns(
            [
                pl.when(pl.col(c).is_in(list(_SENTINELS))).then(None).otherwise(pl.col(c)).alias(c)
                for c in str_cols
            ]
        )
        log.append("null-sentinel fusion on string columns")

    sensor_cols = [c for c in SENSOR_COLS if c in frame.columns]
    for c in sensor_cols:
        frame = frame.with_columns(pl.col(c).cast(pl.Float64, strict=False))
    if sensor_cols:
        log.append("cast sensors to Float64: " + ", ".join(sensor_cols))

    for c in sensor_cols:
        med = frame.select(pl.col(c).median()).item()
        nulls = frame.select(pl.col(c).null_count()).item()
        if med is not None and nulls:
            frame = frame.with_columns(pl.col(c).fill_null(med))
            log.append(f"median fill {c} ({nulls} nulls)")

    for c in sensor_cols:
        q = frame.select(
            pl.col(c).quantile(0.25).alias("q1"),
            pl.col(c).quantile(0.75).alias("q3"),
        ).row(0)
        q1, q3 = q[0], q[1]
        if q1 is None or q3 is None:
            continue
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = frame.select(((pl.col(c) < lo) | (pl.col(c) > hi)).sum()).item()
        if n_out:
            frame = frame.with_columns(pl.col(c).clip(lo, hi))
            log.append(f"IQR cap {c} ({n_out} values)")

    out = frame.to_pandas()
    log.append(f"returned {len(out):,} rows to pandas for quality / ML")
    return out, log
