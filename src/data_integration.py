"""Multi-file data integration with SQL-style joins (not in Forge v2 — new here)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd

JOIN_TYPES = {
    "inner": "INNER JOIN — only matching keys in both tables",
    "left": "LEFT JOIN — all rows from left + matches from right",
    "right": "RIGHT JOIN — all rows from right + matches from left",
    "outer": "FULL OUTER JOIN — all rows from both tables",
}

# Sensor / ops tables on Upload & Clean. Not CAD Twin STEP / ifczip (see src/aps_viewer.py).
TABULAR_EXTENSIONS = (".csv", ".tsv", ".xlsx", ".xls", ".json")
CSV_TSV_EXTENSIONS = (".csv", ".tsv")
ZIP_NO_TABULAR_MSG = (
    "No CSV/TSV/XLSX/JSON in this zip (ignored __MACOSX / .DS_Store). "
    "CAD STEP zips belong on the CAD Twin page, not Upload & Clean."
)


class _NamedBytesIO(io.BytesIO):
    """BytesIO that exposes ``name`` so load_tabular_file can pick a parser."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def is_tabular_zip_name(filename: str) -> bool:
    """True for a sensor/ops zip on Upload & Clean (not Autodesk ifczip)."""
    name = (filename or "").lower()
    if name.endswith(".ifczip"):
        return False
    return name.endswith(".zip")


def _zip_member_norm(name: str) -> str:
    # Do not lstrip("./") — that would turn ".DS_Store" into "DS_Store".
    norm = (name or "").replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    return norm.lstrip("/")


def _zip_member_basename(name: str) -> str:
    return _zip_member_norm(name).rstrip("/").rsplit("/", 1)[-1]


def is_ignored_zip_member(name: str) -> bool:
    """Skip Finder junk: __MACOSX, .DS_Store, AppleDouble ``._*``, directories."""
    norm = _zip_member_norm(name)
    base = _zip_member_basename(norm)
    lower = norm.lower()
    if not base or norm.endswith("/"):
        return True
    if lower.startswith("__macosx/") or "/__macosx/" in f"/{lower}":
        return True
    if base.startswith("._"):
        return True
    if base.lower() in {".ds_store", "thumbs.db"}:
        return True
    return False


def is_tabular_zip_member(name: str) -> bool:
    if is_ignored_zip_member(name):
        return False
    lower = _zip_member_norm(name).lower()
    return any(lower.endswith(ext) for ext in TABULAR_EXTENSIONS)


def list_zip_tabular_members(data: bytes) -> list[tuple[str, int]]:
    """Return ``(member_name, uncompressed_size)`` for tabular files, archive order."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            out: list[tuple[str, int]] = []
            for info in zf.infolist():
                raw = info.filename
                if info.is_dir() or is_ignored_zip_member(raw):
                    continue
                if not is_tabular_zip_member(raw):
                    continue
                out.append((raw, int(info.file_size)))
            return out
    except zipfile.BadZipFile as exc:
        raise ValueError("Not a valid ZIP archive.") from exc


def pick_zip_tabular_member(members: list[tuple[str, int]]) -> str:
    """One file → that file. Several → largest csv/tsv, else first csv, else first table."""
    if not members:
        raise ValueError(ZIP_NO_TABULAR_MSG)
    if len(members) == 1:
        return members[0][0]
    csv_tsv = [
        (name, size)
        for name, size in members
        if _zip_member_norm(name).lower().endswith(CSV_TSV_EXTENSIONS)
    ]
    if csv_tsv:
        return max(csv_tsv, key=lambda item: item[1])[0]
    csv_only = [name for name, _size in members if _zip_member_norm(name).lower().endswith(".csv")]
    if csv_only:
        return csv_only[0]
    return members[0][0]


def extract_zip_member(data: bytes, member: str) -> bytes:
    want = _zip_member_norm(member)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if _zip_member_norm(info.filename) == want:
                    return zf.read(info.filename)
    except zipfile.BadZipFile as exc:
        raise ValueError("Not a valid ZIP archive.") from exc
    raise ValueError(f"Zip member not found: {member}")


def load_zip_tabular(data: bytes, member: Optional[str] = None) -> tuple[pd.DataFrame, str]:
    """Extract one tabular zip member and load it with the CSV/XLSX/JSON loaders."""
    members = list_zip_tabular_members(data)
    if not members:
        raise ValueError(ZIP_NO_TABULAR_MSG)
    names = {_zip_member_norm(name): name for name, _size in members}
    if member and _zip_member_norm(member) in names:
        chosen = names[_zip_member_norm(member)]
    else:
        chosen = pick_zip_tabular_member(members)
    raw = extract_zip_member(data, chosen)
    df = load_tabular_file(_NamedBytesIO(raw, Path(chosen).name))
    return df, chosen


def _uploaded_bytes(uploaded_file) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    if hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
        return data if isinstance(data, (bytes, bytearray)) else bytes(data)
    return Path(str(uploaded_file)).read_bytes()


def load_tabular_file(uploaded_file) -> pd.DataFrame:
    """Load csv/tsv/xlsx/json (or a zip of one of those) from an UploadedFile or path-like."""
    name = getattr(uploaded_file, "name", str(uploaded_file)).lower()
    if is_tabular_zip_name(name):
        df, _used = load_zip_tabular(_uploaded_bytes(uploaded_file))
        return df
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    if name.endswith(".json"):
        return pd.read_json(uploaded_file)
    if name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    df = pd.read_csv(uploaded_file)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def suggest_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """Intersect column names as candidate join keys."""
    common = sorted(set(left.columns) & set(right.columns))
    preferred = [c for c in common if c.lower() in {"machine_id", "asset_id", "id", "timestamp", "date"}]
    rest = [c for c in common if c not in preferred]
    return preferred + rest


def join_two(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str = "inner",
    on: Optional[list[str]] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    suffixes: tuple[str, str] = ("_l", "_r"),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    how = (how or "inner").lower()
    if how not in JOIN_TYPES:
        raise ValueError(f"Unsupported join type: {how}. Use one of {list(JOIN_TYPES)}")

    meta: dict[str, Any] = {
        "how": how,
        "left_rows": len(left),
        "right_rows": len(right),
    }
    if on:
        merged = pd.merge(left, right, how=how, on=on, suffixes=suffixes)
        meta["keys"] = on
    elif left_on and right_on:
        merged = pd.merge(left, right, how=how, left_on=left_on, right_on=right_on, suffixes=suffixes)
        meta["keys"] = [left_on, right_on]
    else:
        keys = suggest_join_keys(left, right)
        if not keys:
            raise ValueError("No common columns to join on. Pick left_on/right_on explicitly.")
        merged = pd.merge(left, right, how=how, on=keys[:1], suffixes=suffixes)
        meta["keys"] = keys[:1]
        meta["auto_key"] = True

    meta["result_rows"] = len(merged)
    meta["result_cols"] = list(merged.columns)
    return merged, meta


def join_many(
    tables: dict[str, pd.DataFrame],
    steps: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Chain joins across 3+ named tables.

    steps example:
      [
        {"left": "sensors", "right": "maintenance", "how": "left", "on": ["machine_id"]},
        {"left": "_result", "right": "costs", "how": "inner", "on": ["machine_id"]},
      ]
    After step 1, the working frame is registered as "_result".
    """
    if not tables:
        raise ValueError("No tables provided")
    if not steps:
        raise ValueError("Provide at least one join step")

    working = tables[steps[0]["left"]].copy()
    registry = dict(tables)
    logs: list[dict[str, Any]] = []

    for i, step in enumerate(steps):
        right_name = step["right"]
        if right_name not in registry:
            raise KeyError(f"Unknown right table: {right_name}")
        left_df = working if i > 0 or step.get("left") == "_result" else registry[step["left"]]
        right_df = registry[right_name]
        how = step.get("how", "inner")
        on = step.get("on")
        left_on = step.get("left_on")
        right_on = step.get("right_on")
        working, meta = join_two(
            left_df,
            right_df,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
        )
        meta["step"] = i + 1
        meta["left_name"] = step.get("left", "_result")
        meta["right_name"] = right_name
        logs.append(meta)
        registry["_result"] = working

    return working, logs
