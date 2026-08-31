"""URL-based tabular ingest via DuckDB httpfs — Google Drive, Kaggle, direct HTTPS CSV/Parquet."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
import requests

# DuckDB httpfs can stream many HTTPS URLs; Drive/Kaggle usually need a resolved direct URL
# or a local cache file first (especially for multi-GB files).
_GDRIVE_ID_PATTERNS = (
    re.compile(r"drive\.google\.com/file/d/([^/?#]+)"),
    re.compile(r"drive\.google\.com/open\?[^#]*\bid=([^&#]+)"),
    re.compile(r"drive\.google\.com/uc\?(?:export=download&)?[^#]*\bid=([^&#]+)"),
    re.compile(r"docs\.google\.com/spreadsheets/d/([^/?#]+)"),
)
_KAGGLE_PAGE = re.compile(r"kaggle\.com/(?:datasets|competitions)/([^/?#]+/[^/?#]+)")
_KAGGLE_SCHEME = re.compile(r"^kaggle://([^/]+/[^/]+)(?:/(.+))?$", re.I)


def detect_source_kind(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return "empty"
    if _KAGGLE_SCHEME.match(u):
        return "kaggle_api"
    if "drive.google.com" in u or "docs.google.com/spreadsheets" in u:
        return "google_drive"
    if "kaggle.com" in u:
        return "kaggle_page"
    return "https"


def extract_gdrive_file_id(url: str) -> Optional[str]:
    for pat in _GDRIVE_ID_PATTERNS:
        m = pat.search(url)
        if m:
            return m.group(1)
    return None


def extract_kaggle_slug(url: str) -> Optional[str]:
    m = _KAGGLE_PAGE.search(url)
    return m.group(1) if m else None


def resolve_gdrive_download_url(file_id: str, session: Optional[requests.Session] = None) -> str:
    """Return a direct-download URL, handling Google's large-file confirm token."""
    sess = session or requests.Session()
    base = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = sess.get(base, stream=True, timeout=60, allow_redirects=True)
    resp.raise_for_status()
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            return f"{base}&confirm={value}"
    # Some responses embed confirm in HTML for very large files.
    if "text/html" in (resp.headers.get("content-type") or "").lower():
        m = re.search(r"confirm=([0-9A-Za-z_]+)", resp.text)
        if m:
            return f"{base}&confirm={m.group(1)}"
    return base


def _kaggle_credentials() -> tuple[str, str]:
    user = (os.getenv("KAGGLE_USERNAME") or os.getenv("KAGGLE_USER") or "").strip()
    key = (os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_TOKEN") or "").strip()
    if not user or not key:
        raise RuntimeError(
            "Kaggle ingest needs KAGGLE_USERNAME and KAGGLE_KEY in environment secrets "
            "(create at https://www.kaggle.com/settings → API)."
        )
    return user, key


def download_kaggle_file(owner_dataset: str, filename: Optional[str] = None) -> Path:
    """Download one file from a Kaggle dataset slug (owner/dataset) to a temp path."""
    user, key = _kaggle_credentials()
    owner, _, dataset = owner_dataset.partition("/")
    if not owner or not dataset:
        raise ValueError(f"Invalid Kaggle slug: {owner_dataset!r} (expected owner/dataset)")

    tmp_dir = Path(tempfile.mkdtemp(prefix="forge-kaggle-"))
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        if filename:
            api.dataset_download_file(owner, dataset, filename, path=str(tmp_dir), quiet=True)
            # Kaggle may save as .zip or raw file depending on type.
            candidates = list(tmp_dir.glob(f"{Path(filename).stem}*"))
            if not candidates:
                candidates = list(tmp_dir.iterdir())
        else:
            api.dataset_download_files(f"{owner}/{dataset}", path=str(tmp_dir), quiet=True, unzip=True)
            candidates = [p for p in tmp_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".parquet", ".json", ".txt"}]
            if not candidates:
                candidates = [p for p in tmp_dir.rglob("*") if p.is_file()]
        if not candidates:
            raise RuntimeError(f"No files downloaded from Kaggle dataset {owner}/{dataset}")
        # Prefer CSV when multiple files.
        candidates.sort(key=lambda p: (0 if p.suffix.lower() == ".csv" else 1, p.name))
        return candidates[0]
    except ImportError as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Install kaggle for Kaggle links: pip install kaggle") from exc
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def resolve_source_to_fetch_url(url: str) -> tuple[str, dict[str, Any]]:
    """Normalize user paste into something DuckDB httpfs or requests can fetch."""
    raw = (url or "").strip()
    if not raw:
        raise ValueError("Paste a URL first.")

    meta: dict[str, Any] = {"original_url": raw, "kind": detect_source_kind(raw)}

    m = _KAGGLE_SCHEME.match(raw)
    if m:
        slug, fname = m.group(1), m.group(2)
        meta["kaggle_slug"] = slug
        meta["kaggle_file"] = fname
        meta["local_path"] = str(download_kaggle_file(slug, fname))
        meta["resolved_url"] = meta["local_path"]
        return meta["local_path"], meta

    if meta["kind"] == "kaggle_page":
        slug = extract_kaggle_slug(raw)
        if not slug:
            raise ValueError("Could not parse Kaggle dataset slug from URL.")
        meta["kaggle_slug"] = slug
        meta["local_path"] = str(download_kaggle_file(slug))
        meta["resolved_url"] = meta["local_path"]
        return meta["local_path"], meta

    if meta["kind"] == "google_drive":
        file_id = extract_gdrive_file_id(raw)
        if not file_id:
            raise ValueError("Could not parse Google Drive file id from URL.")
        meta["gdrive_file_id"] = file_id
        resolved = resolve_gdrive_download_url(file_id)
        meta["resolved_url"] = resolved
        return resolved, meta

    # Dropbox ?dl=0 → ?dl=1
    if "dropbox.com" in raw and "dl=0" in raw:
        raw = raw.replace("dl=0", "dl=1")
    meta["resolved_url"] = raw
    return raw, meta


# Plant / PdM SQL slice presets — templates use `{source}` for DuckDB read path.
INGEST_SQL_PRESETS: dict[str, dict[str, Any]] = {
    "last_n_rows": {
        "label": "Last N rows",
        "description": "Cap row count for quick explore on multi-GB sensor logs.",
        "domains": None,
        "params": [("n", "Row limit", 50000, "int")],
        "template": (
            "SELECT *\n"
            "FROM read_csv_auto('{{source}}', header=true)\n"
            "LIMIT {n}"
        ),
    },
    "last_n_days": {
        "label": "Last N days",
        "description": "Time window ending today (requires a timestamp/date column).",
        "domains": ("predictive_maintenance", "plant_oee", "energy_utilities"),
        "params": [
            ("n", "Days back", 30, "int"),
            ("ts_col", "Timestamp column", "timestamp", "str"),
        ],
        "template": (
            "SELECT *\n"
            "FROM read_csv_auto('{{source}}', header=true)\n"
            "WHERE try_cast({ts_col} AS TIMESTAMP) >= current_timestamp - INTERVAL '{n} days'\n"
            "ORDER BY try_cast({ts_col} AS TIMESTAMP) DESC\n"
            "LIMIT 500000"
        ),
    },
    "filter_machine_id": {
        "label": "Filter by machine_id",
        "description": "Slice one asset from a plant-wide sensor export.",
        "domains": ("predictive_maintenance", "plant_oee"),
        "params": [
            ("machine_id", "Machine ID", "M-001", "str"),
            ("n", "Row limit (0 = no cap)", 100000, "int"),
        ],
        "template": (
            "SELECT *\n"
            "FROM read_csv_auto('{{source}}', header=true)\n"
            "WHERE machine_id = '{machine_id}'\n"
            "ORDER BY try_cast(timestamp AS TIMESTAMP) DESC NULLS LAST\n"
            "{limit_clause}"
        ),
    },
    "filter_asset_id": {
        "label": "Filter by asset_id",
        "description": "Slice one asset when the export uses asset_id instead of machine_id.",
        "domains": ("predictive_maintenance", "plant_oee"),
        "params": [
            ("asset_id", "Asset ID", "ASSET-001", "str"),
            ("n", "Row limit (0 = no cap)", 100000, "int"),
        ],
        "template": (
            "SELECT *\n"
            "FROM read_csv_auto('{{source}}', header=true)\n"
            "WHERE asset_id = '{asset_id}'\n"
            "ORDER BY try_cast(timestamp AS TIMESTAMP) DESC NULLS LAST\n"
            "{limit_clause}"
        ),
    },
    "date_range": {
        "label": "Date range window",
        "description": "Inclusive start, exclusive end on a timestamp column.",
        "domains": None,
        "params": [
            ("start_date", "Start (YYYY-MM-DD)", "2024-01-01", "str"),
            ("end_date", "End (YYYY-MM-DD)", "2024-06-01", "str"),
            ("ts_col", "Timestamp column", "timestamp", "str"),
            ("n", "Row limit (0 = no cap)", 200000, "int"),
        ],
        "template": (
            "SELECT *\n"
            "FROM read_csv_auto('{{source}}', header=true)\n"
            "WHERE try_cast({ts_col} AS TIMESTAMP) >= TIMESTAMP '{start_date}'\n"
            "  AND try_cast({ts_col} AS TIMESTAMP) < TIMESTAMP '{end_date}'\n"
            "ORDER BY try_cast({ts_col} AS TIMESTAMP)\n"
            "{limit_clause}"
        ),
    },
    "sample_percent": {
        "label": "Random sample %",
        "description": "Explore a percentage of rows without loading the full file.",
        "domains": None,
        "params": [("sample_pct", "Sample percent (1–100)", 5, "float")],
        "template": (
            "SELECT *\n"
            "FROM read_csv_auto('{{source}}', header=true)\n"
            "WHERE random() < ({sample_pct} / 100.0)\n"
            "LIMIT 500000"
        ),
    },
    "pdm_failure_focus": {
        "label": "PdM — failures + near-failure window",
        "description": "Rows with failure=1 or low RUL for label QA / model training.",
        "domains": ("predictive_maintenance",),
        "params": [("rul_max", "Max RUL to include", 48, "int")],
        "template": (
            "SELECT *\n"
            "FROM read_csv_auto('{{source}}', header=true)\n"
            "WHERE COALESCE(try_cast(failure AS INTEGER), 0) = 1\n"
            "   OR try_cast(rul AS DOUBLE) <= {rul_max}\n"
            "ORDER BY try_cast(timestamp AS TIMESTAMP) DESC NULLS LAST\n"
            "LIMIT 200000"
        ),
    },
}


def _coerce_preset_param(value: Any, kind: str) -> Any:
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    return str(value).replace("'", "''")


def list_ingest_presets(domain: Optional[str] = None) -> list[dict[str, Any]]:
    """Return preset metadata for UI picker (optionally filtered by app domain)."""
    out: list[dict[str, Any]] = []
    for preset_id, spec in INGEST_SQL_PRESETS.items():
        domains = spec.get("domains")
        if domain and domains and domain not in domains:
            continue
        out.append(
            {
                "id": preset_id,
                "label": spec["label"],
                "description": spec.get("description", ""),
                "params": list(spec.get("params") or []),
            }
        )
    return out


def build_preset_sql(preset_id: str, params: Optional[dict[str, Any]] = None) -> str:
    """Materialize a preset template into DuckDB SQL with `{source}` placeholder intact."""
    if preset_id not in INGEST_SQL_PRESETS:
        raise ValueError(f"Unknown ingest preset: {preset_id!r}")
    spec = INGEST_SQL_PRESETS[preset_id]
    merged: dict[str, Any] = {}
    n_limit = 0
    for name, _label, default, kind in spec.get("params") or []:
        raw = (params or {}).get(name, default)
        val = _coerce_preset_param(raw, kind)
        if name == "n":
            n_limit = int(val or 0)
        else:
            merged[name] = val

    limit_clause = f"LIMIT {n_limit}" if n_limit > 0 else ""
    template = spec["template"]
    if "{limit_clause}" in template:
        merged["limit_clause"] = limit_clause
    elif n_limit > 0 and "LIMIT" not in template.upper():
        template = template.rstrip() + f"\nLIMIT {n_limit}"

    try:
        return template.format(**merged)
    except KeyError as exc:
        raise ValueError(f"Missing preset parameter for {preset_id}: {exc}") from exc


def default_ingest_sql(domain: Optional[str] = None) -> str:
    """Starter DuckDB query — PdM-aware examples when domain is predictive_maintenance."""
    if domain == "predictive_maintenance":
        return (
            "SELECT machine_id, timestamp, temperature, vibration, pressure, failure, rul\n"
            "FROM read_csv_auto('{source}', header=true)\n"
            "WHERE machine_id = 'M-001'\n"
            "  AND try_cast(timestamp AS TIMESTAMP) >= TIMESTAMP '2024-01-01'\n"
            "ORDER BY try_cast(timestamp AS TIMESTAMP) DESC\n"
            "LIMIT 50000"
        )
    return (
        "SELECT *\n"
        "FROM read_csv_auto('{source}', header=true)\n"
        "WHERE 1 = 1  -- e.g. machine_id = 'M1' AND timestamp >= '2024-01-01'\n"
        "LIMIT 100000"
    )


def validate_ingest_sql(sql: str) -> str:
    text = (sql or "").strip()
    if not text:
        raise ValueError("SQL query is empty.")
    head = text.lstrip().split(None, 1)[0].upper()
    if head not in {"SELECT", "WITH"}:
        raise ValueError("Only SELECT (or WITH … SELECT) queries are allowed for ingest.")
    # Block multi-statement / destructive keywords.
    if ";" in text.rstrip().rstrip(";"):
        raise ValueError("Only one SQL statement allowed.")
    upper = text.upper()
    for bad in (" DROP ", " DELETE ", " INSERT ", " UPDATE ", " CREATE ", " ATTACH ", " COPY "):
        if bad in f" {upper} ":
            raise ValueError(f"Disallowed SQL keyword in ingest query.")
    return text


def _sql_escape_path(path: str) -> str:
    return path.replace("'", "''")


def _duckdb_read_sql(path_or_url: str, sql_template: str) -> pd.DataFrame:
    import duckdb

    sql = validate_ingest_sql(sql_template)
    if "{source}" not in sql:
        raise ValueError("SQL must reference `{source}` (the resolved file path or URL).")
    sql = sql.replace("{source}", _sql_escape_path(path_or_url))

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
    except Exception:
        pass
    return con.execute(sql).df()


def _duckdb_read(path_or_url: str, *, row_limit: Optional[int] = None) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
    except Exception:
        pass  # httpfs may already be available

    low = path_or_url.lower().split("?")[0]
    limit_sql = f" LIMIT {int(row_limit)}" if row_limit and row_limit > 0 else ""

    if low.endswith(".parquet"):
        sql = f"SELECT * FROM read_parquet('{path_or_url}'){limit_sql}"
    elif low.endswith(".json"):
        sql = f"SELECT * FROM read_json_auto('{path_or_url}'){limit_sql}"
    else:
        sep = "\\t" if low.endswith(".tsv") else ","
        sql = (
            f"SELECT * FROM read_csv_auto('{path_or_url}', header=true, sep='{sep}')"
            f"{limit_sql}"
        )
    return con.execute(sql).df()


def _cache_remote_file(fetch_url: str, dest_dir: Path, *, chunk_mb: int = 8) -> Path:
    """Stream a remote file to disk so DuckDB can read multi-GB sources without RAM blow-up."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(fetch_url)
    name = unquote(Path(parsed.path).name) or "remote_ingest.csv"
    if "." not in name:
        name += ".csv"
    dest = dest_dir / name
    # Reuse cache when the same URL was fetched in this session folder.
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    sess = requests.Session()
    with sess.get(fetch_url, stream=True, timeout=120, allow_redirects=True) as resp:
        resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "").lower()
        if "text/html" in ctype and "drive.google.com" in fetch_url:
            raise RuntimeError(
                "Google Drive returned HTML instead of the file. "
                "Check sharing is 'Anyone with the link' or use a direct export link."
            )
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                if chunk:
                    fh.write(chunk)
    return dest


def load_from_url(
    url: str,
    *,
    cache_dir: Path,
    row_limit: Optional[int] = None,
    force_cache: bool = False,
    sql_query: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a tabular dataset from HTTPS, Google Drive, or Kaggle.

    row_limit: cap rows for simple mode (useful on Render free tier). 0/None = all rows.
    sql_query: optional DuckDB SELECT using `{source}` placeholder for filters/slices on huge files.
    """
    fetch_target, meta = resolve_source_to_fetch_url(url)
    meta["row_limit"] = row_limit
    meta["sql_query"] = sql_query

    def _finish(df: pd.DataFrame, engine: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        meta["rows"] = len(df)
        meta["columns"] = list(df.columns)
        meta["engine"] = engine
        return df, meta

    use_sql = bool((sql_query or "").strip())

    # Kaggle path is always local after download.
    if meta.get("local_path"):
        local = str(Path(meta["local_path"]))
        if use_sql:
            return _finish(_duckdb_read_sql(local, sql_query or ""), "duckdb-sql")
        df = _duckdb_read(local, row_limit=row_limit)
        return _finish(df, "duckdb")

    is_remote = fetch_target.startswith("http://") or fetch_target.startswith("https://")
    read_path = fetch_target

    # SQL slices on multi-GB files should use a local cache so DuckDB can scan efficiently.
    if use_sql or force_cache or meta.get("kind") == "google_drive":
        if is_remote:
            cached = _cache_remote_file(fetch_target, cache_dir)
            meta["cached_path"] = str(cached)
            read_path = str(cached)
    elif is_remote:
        try:
            if use_sql:
                return _finish(_duckdb_read_sql(fetch_target, sql_query or ""), "duckdb-sql-httpfs")
            df = _duckdb_read(fetch_target, row_limit=row_limit)
            return _finish(df, "duckdb-httpfs")
        except Exception as stream_err:
            meta["stream_error"] = str(stream_err)
            cached = _cache_remote_file(fetch_target, cache_dir)
            meta["cached_path"] = str(cached)
            read_path = str(cached)

    if use_sql:
        return _finish(_duckdb_read_sql(read_path, sql_query or ""), "duckdb-sql")
    df = _duckdb_read(read_path, row_limit=row_limit)
    return _finish(df, "duckdb")


def friendly_source_label(meta: dict[str, Any]) -> str:
    kind = meta.get("kind") or "url"
    if kind == "google_drive":
        return f"gdrive:{meta.get('gdrive_file_id', 'file')}"
    if kind.startswith("kaggle"):
        return f"kaggle:{meta.get('kaggle_slug', 'dataset')}"
    return urlparse(meta.get("original_url", "url")).netloc or "url"
