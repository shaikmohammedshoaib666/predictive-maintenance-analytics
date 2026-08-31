"""Layer 5 — optional PySpark cleaning engine for large sensor files.

Distributed/columnar clean using a local SparkSession when PySpark + a JVM are
available. `spark_available()` gates the feature so the default pandas pipeline
and the Render deploy are never blocked by a missing JVM — PySpark is an
opt-in engine, not a hard dependency (kept out of `requirements.txt`).
"""

from __future__ import annotations

import os
import shutil
from typing import Any

import pandas as pd

SENSOR_COLS = ("temperature", "vibration", "pressure", "rpm")


def spark_available() -> tuple[bool, str]:
    """Return (available, message). Requires both pyspark and a JVM."""
    try:
        import pyspark  # noqa: F401
    except Exception as exc:  # ImportError or partial install
        return False, f"pyspark not installed ({exc.__class__.__name__})"
    java = os.environ.get("JAVA_HOME") or shutil.which("java")
    if not java:
        return False, "Java runtime (JAVA_HOME / java) not found"
    return True, f"pyspark {pyspark.__version__}"


def _get_session():
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName("pdm-clean")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )


def clean_with_spark(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Distributed clean: schema normalize, dedupe, cast sensors, median impute, IQR cap.

    Raises RuntimeError when PySpark/JVM are unavailable so callers can fall back.
    """
    ok, msg = spark_available()
    if not ok:
        raise RuntimeError(f"PySpark engine unavailable: {msg}")

    from pyspark.sql import functions as F

    log: list[str] = [f"PySpark engine: {msg} (local[*])"]
    spark = _get_session()
    try:
        pdf = df.copy()
        pdf.columns = [str(c).strip().replace(" ", "_") for c in pdf.columns]
        # Spark cannot infer some object dtypes; stringify datetimes for transport.
        for c in pdf.columns:
            if pd.api.types.is_datetime64_any_dtype(pdf[c]):
                pdf[c] = pdf[c].astype(str)
        sdf = spark.createDataFrame(pdf)
        before = sdf.count()
        log.append(f"loaded {before:,} rows into Spark")

        sdf = sdf.dropDuplicates()
        after = sdf.count()
        if after != before:
            log.append(f"dropDuplicates {before:,} -> {after:,}")

        sensor_cols = [c for c in SENSOR_COLS if c in sdf.columns]
        for c in sensor_cols:
            sdf = sdf.withColumn(c, F.col(c).cast("double"))

        for c in sensor_cols:
            quants = sdf.approxQuantile(c, [0.25, 0.5, 0.75], 0.01)
            if not quants or any(q is None for q in quants):
                continue
            q1, med, q3 = quants
            sdf = sdf.fillna({c: med})
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            sdf = sdf.withColumn(
                c,
                F.when(F.col(c) < lo, lo).when(F.col(c) > hi, hi).otherwise(F.col(c)),
            )
            log.append(f"impute+IQR-cap {c}: median={round(med, 2)} bounds=[{round(lo, 2)},{round(hi, 2)}]")

        out = sdf.toPandas()
        if "timestamp" in out.columns:
            out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        log.append(f"returned {len(out):,} rows to pandas")
        return out, log
    finally:
        spark.stop()
