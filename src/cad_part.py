"""CAD Twin identity helpers — region tint + click-part sensor hints.

Isolation Forest scores an **asset**, not each Autodesk mesh. These helpers map
that asset risk onto the *few* CAD nodes whose names look like a mapped sensor
(temp / egt / exhaust / oil / vibration / rpm / cyl / crank / manifold), and
emphasize a mapped sensor when a clicked part name fuzzy-matches one. Aviation
packs may add a small ``part_hints`` dict; there is no Rotax BOM and no per-bolt
anomaly score.

Deliberately **not** whole-model theming: a High asset tints only the matched
nodes red, Medium tints them orange, and everything else keeps the viewer's
default dark gray. When a STEP exports as ``Solid1`` / ``??????-1-solid1`` there
is nothing to match, and the honest answer is "no region matched" — not a red
engine.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from src.twin3d import normalize_risk

# Badge / legend hex. Unknown is gray.
RISK_THEME_HEX: dict[str, str] = {
    "High": "#e74c3c",
    "Medium": "#f39c12",
    "Low": "#7a8c82",  # slight green-gray; keep default look, do not go plant-green
    "Unknown": "#7f8c8d",
}

# THREE.Vector4-style 0–1 RGBA. Legacy whole-model palette, kept for the badge
# legend and dashboard export; the viewer no longer paints every mesh with it.
RISK_THEME_RGBA: dict[str, tuple[float, float, float, float]] = {
    "High": (0.91, 0.30, 0.24, 0.72),
    "Medium": (0.95, 0.61, 0.07, 0.58),
    "Low": (0.48, 0.55, 0.52, 0.32),
}

# Region theming (APS ``setThemingColor`` on matched dbIds only). Higher alpha
# than the old whole-model tint because it lands on a handful of nodes.
# Low / Unknown are absent on purpose: no extra tint at all.
REGION_THEME_RGBA: dict[str, tuple[float, float, float, float]] = {
    "High": (0.91, 0.30, 0.24, 0.85),
    "Medium": (0.95, 0.61, 0.07, 0.78),
}

NO_REGION_MATCH_MSG = (
    "No CAD region matched this asset’s hot sensors (names are Solid1 / generic). "
    "The model keeps its default color — we do not redden the whole engine."
)

# Generic Autodesk node-name needles. Longer keys win. Canonical sensor names
# match ``sensor_map`` / pack extras when those columns exist on the frame.
_GENERIC_PART_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("exhaustmanifold", "egt"),
    ("exhaust", "egt"),
    ("egt", "egt"),
    ("cylinderhead", "cht"),
    ("cylhead", "cht"),
    ("cylinder", "cht"),
    ("cht", "cht"),
    ("cyl", "cht"),
    ("vibration", "vibration"),
    ("vibra", "vibration"),
    ("vib", "vibration"),
    ("temperature", "temperature"),
    ("temp", "temperature"),
    ("oiltemperature", "oil_temp"),
    ("oiltemp", "oil_temp"),
    ("oil_temp", "oil_temp"),
    ("oilpress", "oil_pressure"),
    ("oil_pressure", "oil_pressure"),
    ("oil", "oil_pressure"),
    ("intakemanifold", "manifold_pressure"),
    ("manifold", "manifold_pressure"),
    ("crank", "rpm"),
    ("rpm", "rpm"),
)

# Shortest needle we will accept. Two-letter fragments match half a BOM.
MIN_REGION_NEEDLE = 3

_SKIP_SENSOR_COLS = {
    "machine_id",
    "timestamp",
    "is_anomaly",
    "anomaly_score",
    "anomaly_label",
    "failure_within_days",
    "predicted_rul_days",
    "risk_level",
    "rul_days",
    "label_source",
    "is_proxy",
    "message",
    "physics_alert",
    "physics_fault",
    "physics_rule",
    "physics_label",
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def risk_theme_hex(risk: Optional[str]) -> str:
    """CSS hex for the whole-model tint / badge. Unknown is gray."""
    return RISK_THEME_HEX[normalize_risk(risk)]


def risk_theming_rgba(risk: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    """Legacy whole-model RGBA. ``None`` means clear theming (no risk)."""
    level = normalize_risk(risk)
    if level == "Unknown":
        return None
    return RISK_THEME_RGBA[level]


def region_theme_rgba(risk: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    """RGBA for the matched CAD nodes. ``None`` for Low / Unknown — no tint."""
    return REGION_THEME_RGBA.get(normalize_risk(risk))


def _norm_part(name: str) -> str:
    return _NON_ALNUM.sub("", (name or "").lower())


def part_hints_for(pack_id: Optional[str] = None) -> dict[str, str]:
    """Aviation (and only aviation) ships a small keyword → sensor map."""
    if not pack_id:
        return {}
    try:
        from src.industry_packs import part_hints_for as _pack_hints

        return dict(_pack_hints(pack_id) or {})
    except Exception:
        return {}


def part_sensor_hint(
    part_name: str,
    *,
    pack_id: str = "",
    available: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Fuzzy-match an Autodesk node name to a mapped sensor.

    Returns the canonical (or available-column) name, or ``None`` when the
    part does not look like temp / vibration / rpm / oil / exhaust / cyl / crank
    (plus aviation ``part_hints``). Random bolts do not match.
    """
    blob = _norm_part(part_name)
    if not blob:
        return None

    avail_lower: Optional[dict[str, str]] = None
    if available is not None:
        avail_lower = {}
        for col in available:
            key = str(col).strip()
            if key:
                avail_lower[key.lower()] = key
        if not avail_lower:
            return None

    def _accept(canonical: str) -> Optional[str]:
        want = (canonical or "").strip()
        if not want:
            return None
        if avail_lower is None:
            return want
        low = want.lower()
        if low in avail_lower:
            return avail_lower[low]
        for col_low, real in avail_lower.items():
            if low in col_low or col_low in low:
                return real
        return None

    pairs: list[tuple[str, str]] = []
    for kw, sensor in part_hints_for(pack_id).items():
        pairs.append((kw, sensor))
    pairs.extend(_GENERIC_PART_KEYWORDS)
    pairs.sort(key=lambda kv: len(_norm_part(kv[0])), reverse=True)

    seen: set[str] = set()
    for kw, sensor in pairs:
        needle = _norm_part(kw)
        if not needle or needle in seen:
            continue
        seen.add(needle)
        if needle in blob:
            hit = _accept(sensor)
            if hit:
                return hit
    return None


# Autodesk sometimes hands us a node name the browser cannot render: an empty
# string, control bytes, or a mojibake run of ``?`` / U+FFFD from a non-UTF8
# STEP. ``?????? ??????-1-solid1`` must never reach a CEO screen.
_UNREADABLE_CHARS = re.compile(r"[?\ufffd\ufffe\uffff]+")
_SOLID_INSTANCE_RE = re.compile(r"(\d+)\s*[-_ ]\s*solid\s*(\d+)", re.I)
_SOLID_INDEX_RE = re.compile(r"solid\s*[-_ ]?\s*(\d+)", re.I)
_BARE_INDEX_RE = re.compile(r"^[\s\-_.]*(\d+)[\s\-_.]*$")
_WHITESPACE_RE = re.compile(r"\s+")


def _printable(name: str) -> str:
    text = "".join(ch for ch in str(name or "") if ch.isprintable() or ch == " ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def _solid_label(text: str) -> str:
    """``-1-solid1`` → ``Solid-1``; ``solid7`` → ``Solid-7``; ``-3`` → ``Solid-3``."""
    pair = _SOLID_INSTANCE_RE.search(text)
    if pair:
        return f"Solid-{int(pair.group(1))}"
    single = _SOLID_INDEX_RE.search(text)
    if single:
        return f"Solid-{int(single.group(1))}"
    bare = _BARE_INDEX_RE.match(_UNREADABLE_CHARS.sub(" ", text))
    if bare:
        return f"Solid-{int(bare.group(1))}"
    return ""


def sanitize_node_name(name: Any, db_id: Any = None) -> str:
    """Human-safe label for an Autodesk node. Never returns ``??????``.

    Readable names pass through unchanged. Names carrying ``?`` / U+FFFD runs
    are salvaged: a ``-1-solid1`` suffix becomes ``Solid-1``, otherwise the
    readable remainder is kept, otherwise we fall back to ``Solid-{dbId}``.
    Asset ids are *not* run through this — ``UAV-03`` stays exact.
    """
    cleaned = _printable(name)
    if cleaned and not _UNREADABLE_CHARS.search(cleaned):
        return cleaned

    salvaged = _solid_label(cleaned)
    if salvaged:
        return salvaged

    remainder = _UNREADABLE_CHARS.sub(" ", cleaned)
    remainder = _WHITESPACE_RE.sub(" ", remainder).strip(" -_.,;:/\\")
    if remainder and any(ch.isalpha() for ch in remainder):
        return remainder

    try:
        ident = int(db_id)
    except (TypeError, ValueError):
        return "Solid"
    return f"Solid-{ident}"


def region_hint_keywords(
    pack_id: str = "",
    *,
    available: Optional[Iterable[str]] = None,
) -> list[str]:
    """Normalized node-name needles used to pick the CAD region to tint.

    Pack ``part_hints`` + the generic keyword table + the asset's own mapped
    sensor column names. Longest first so ``cylinderhead`` wins over ``cyl``.
    """
    raw: list[str] = list(part_hints_for(pack_id).keys())
    raw.extend(kw for kw, _ in _GENERIC_PART_KEYWORDS)
    for col in available or []:
        name = str(col or "").strip()
        if name and name.lower() not in _SKIP_SENSOR_COLS:
            raw.append(name)

    seen: set[str] = set()
    needles: list[str] = []
    for kw in sorted((_norm_part(k) for k in raw), key=len, reverse=True):
        if len(kw) < MIN_REGION_NEEDLE or kw in seen:
            continue
        seen.add(kw)
        needles.append(kw)
    return needles


def _iter_cad_nodes(nodes: Any) -> list[tuple[Any, str]]:
    """Accept ``{dbId: name}``, ``[(dbId, name)]`` or ``[{"dbId":…, "name":…}]``."""
    if not nodes:
        return []
    if isinstance(nodes, dict):
        return [(db_id, str(name or "")) for db_id, name in nodes.items()]
    out: list[tuple[Any, str]] = []
    for row in nodes:
        if isinstance(row, dict):
            db_id = row.get("dbId", row.get("dbid", row.get("id")))
            out.append((db_id, str(row.get("name") or "")))
        elif isinstance(row, (tuple, list)) and len(row) >= 2:
            out.append((row[0], str(row[1] or "")))
    return out


def region_matches(
    nodes: Any,
    *,
    pack_id: str = "",
    available: Optional[Iterable[str]] = None,
    keywords: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    """CAD nodes whose Autodesk name looks like a mapped sensor. Order preserved."""
    needles = list(keywords) if keywords is not None else region_hint_keywords(
        pack_id, available=available
    )
    hits: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for db_id, raw_name in _iter_cad_nodes(nodes):
        blob = _norm_part(raw_name)
        if not blob or db_id in seen:
            continue
        keyword = next((n for n in needles if n in blob), "")
        if not keyword:
            continue
        seen.add(db_id)
        hits.append(
            {
                "dbId": db_id,
                "name": raw_name,
                "display": sanitize_node_name(raw_name, db_id),
                "keyword": keyword,
                "sensor": part_sensor_hint(raw_name, pack_id=pack_id, available=available),
            }
        )
    return hits


def region_theme_plan(
    nodes: Any,
    risk: Optional[str],
    *,
    pack_id: str = "",
    available: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Which dbIds to tint, and in what color, for this asset's risk.

    High → matched nodes red, Medium → matched nodes orange, Low / Unknown →
    nothing. Zero matches on High/Medium returns an **empty** dbId list plus the
    honest ``NO_REGION_MATCH_MSG``; it never falls back to the whole model.
    """
    level = normalize_risk(risk)
    rgba = region_theme_rgba(level)
    total = len(_iter_cad_nodes(nodes))
    if rgba is None:
        return {
            "risk": level,
            "rgba": None,
            "hex": risk_theme_hex(level),
            "dbIds": [],
            "matches": [],
            "matched": 0,
            "nodes": total,
            "tint": False,
            "message": "",
        }
    matches = region_matches(nodes, pack_id=pack_id, available=available, keywords=None)
    return {
        "risk": level,
        "rgba": rgba,
        "hex": risk_theme_hex(level),
        "dbIds": [m["dbId"] for m in matches],
        "matches": matches,
        "matched": len(matches),
        "nodes": total,
        "tint": bool(matches),
        "message": "" if matches else NO_REGION_MATCH_MSG,
    }


def format_sensor_value(value: Any) -> str:
    """Compact display for the sensor grid. Floats get 3 significant digits."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def sensor_grid_rows(
    sensors: Optional[dict[str, Any]],
    *,
    per_row: int = 4,
) -> list[list[tuple[str, str]]]:
    """Chunk mapped sensors into rows of ``per_row`` (name, display) pairs.

    The CAD Twin page renders these as columns *below* the viewer so a long
    sensor list never pushes GuiViewer3D off-screen.
    """
    width = max(1, int(per_row or 1))
    items = [(str(name), format_sensor_value(value)) for name, value in (sensors or {}).items()]
    return [items[i : i + width] for i in range(0, len(items), width)]


def latest_asset_sensors(df: Any, machine_id: str) -> dict[str, Any]:
    """Latest numeric mapped-sensor values for an asset from the session frame."""
    if df is None:
        return {}
    try:
        if getattr(df, "empty", True):
            return {}
    except Exception:
        return {}

    frame = df
    mid = str(machine_id or "").strip()
    try:
        if mid and "machine_id" in frame.columns:
            sub = frame[frame["machine_id"].astype(str) == mid]
            if sub is not None and not getattr(sub, "empty", True):
                frame = sub
    except Exception:
        pass

    try:
        if "timestamp" in frame.columns:
            frame = frame.sort_values("timestamp")
    except Exception:
        pass

    try:
        row = frame.iloc[-1]
    except Exception:
        return {}

    out: dict[str, Any] = {}
    columns = list(getattr(frame, "columns", []))
    for col in columns:
        name = str(col)
        if name.lower() in _SKIP_SENSOR_COLS:
            continue
        try:
            val = row[col]
        except Exception:
            continue
        try:
            import pandas as pd

            if pd.isna(val):
                continue
            numeric = bool(pd.api.types.is_numeric_dtype(frame[col]))
        except Exception:
            numeric = isinstance(val, (int, float)) and not isinstance(val, bool)
        if not numeric:
            continue
        try:
            out[name] = float(val) if not isinstance(val, int) else val
        except (TypeError, ValueError):
            out[name] = val
    return out


def _column_floats(df: Any, column: str) -> list[float]:
    """Every finite numeric value in one column, as plain floats."""
    try:
        series = df[column]
    except Exception:
        return []
    out: list[float] = []
    for raw in list(series):
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val != val or val in (float("inf"), float("-inf")):  # NaN / inf
            continue
        out.append(val)
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _std(values: list[float], mu: float) -> float:
    if len(values) < 2:
        return 0.0
    return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5


def sensor_score(value: float, population: list[float]) -> tuple[float, str]:
    """How unusual ``value`` is against the rest of that column.

    Deliberately simple and defensible: a z-score against the other rows of the
    same column, falling back to a median/MAD score when the column has no spread.
    """
    if not population:
        return 0.0, "none"
    mu = _mean(population)
    sd = _std(population, mu)
    if sd > 1e-12:
        return (value - mu) / sd, "z-score"
    med = _median(population)
    mad = _median([abs(v - med) for v in population])
    if mad > 1e-12:
        return (value - med) / (1.4826 * mad), "median"
    if abs(value - med) > 1e-12:
        return 1.0 if value > med else -1.0, "median"
    return 0.0, "flat"


NO_DRIVER_MSG = (
    "No mapped sensor stands out for this asset yet — load a CSV and run Anomaly & RUL "
    "so there are other rows to compare against."
)

DRIVER_HONESTY = (
    "Driver is the mapped sensor with the largest |z-score| against the other rows of that "
    "column. Isolation Forest scores the asset, not this CAD part."
)


def sensor_driver(
    df: Any,
    machine_id: str,
    risk: Optional[str],
    *,
    sensors: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Which mapped sensor most explains a High / Medium asset, and its CAD region.

    Low / Unknown returns an empty ``line``: no "driven by" scare line when the
    asset is healthy.
    """
    from src.cad_regions import sensor_display, sensor_region

    level = normalize_risk(risk)
    latest = dict(sensors) if sensors is not None else latest_asset_sensors(df, machine_id)
    blank = {
        "risk": level,
        "sensor": "",
        "display": "",
        "value": None,
        "z": 0.0,
        "basis": "none",
        "region": "",
        "region_label": "",
        "related": [],
        "line": "",
        "message": "",
        "honesty": DRIVER_HONESTY,
        "ranked": [],
    }
    if level not in ("High", "Medium"):
        return blank

    ranked: list[dict[str, Any]] = []
    for name, value in latest.items():
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        # Compare the asset's latest reading against the *rest* of that column.
        population = _column_floats(df, name)
        if population:
            for idx, item in enumerate(population):
                if abs(item - val) < 1e-12:
                    population.pop(idx)
                    break
        score, basis = sensor_score(val, population)
        region = sensor_region(name)
        ranked.append(
            {
                "sensor": name,
                "display": sensor_display(name),
                "value": val,
                "z": score,
                "abs_z": abs(score),
                "basis": basis,
                "region": region["region"],
                "region_label": region["phrase"],
                "related": list(region["related"]),
            }
        )

    ranked.sort(key=lambda row: row["abs_z"], reverse=True)
    blank["ranked"] = ranked[:5]
    if not ranked or ranked[0]["abs_z"] <= 1e-12:
        blank["message"] = NO_DRIVER_MSG
        return blank

    top = ranked[0]
    return {
        **blank,
        "sensor": top["sensor"],
        "display": top["display"],
        "value": top["value"],
        "z": top["z"],
        "basis": top["basis"],
        "region": top["region"],
        "region_label": top["region_label"],
        "related": list(top["related"]),
        "line": f"{level} driven by {top['display']} → {top['region_label']}",
        "ranked": ranked[:5],
    }


def _event_db_id(event: dict[str, Any]) -> Optional[int]:
    raw = event.get("dbId")
    try:
        return int(raw) if raw is not None and raw != "" else None
    except (TypeError, ValueError):
        return None


def parse_cad_part_event(event: Any) -> Optional[dict[str, Any]]:
    """Normalize a click payload from GuiViewer3D. Region reports return ``None``.

    The viewer also posts ``kind="region"`` tint reports on the same component
    channel; those must not clobber the clicked part, so they are ignored here
    and read by ``parse_cad_region_event``.
    """
    if not isinstance(event, dict):
        return None
    if str(event.get("kind") or "") == "region":
        return None
    if event.get("cleared"):
        return {"name": "", "dbId": None, "properties": {}, "cleared": True}
    name = str(event.get("name") or "").strip()
    dbid = _event_db_id(event)
    props = event.get("properties") if isinstance(event.get("properties"), dict) else {}
    clean_props = {str(k): str(v) for k, v in props.items() if k and v is not None and str(v) != ""}
    if not name and dbid is None:
        return None
    return {
        "name": name or (f"Node {dbid}" if dbid is not None else ""),
        "display": sanitize_node_name(name, dbid),
        "dbId": dbid,
        "properties": clean_props,
        "cleared": False,
    }


def parse_cad_region_event(event: Any) -> Optional[dict[str, Any]]:
    """Normalize the viewer's ``kind="region"`` tint report (how many nodes matched)."""
    if not isinstance(event, dict) or str(event.get("kind") or "") != "region":
        return None
    try:
        matched = max(0, int(event.get("matched") or 0))
    except (TypeError, ValueError):
        matched = 0
    try:
        nodes = max(0, int(event.get("nodes") or 0))
    except (TypeError, ValueError):
        nodes = 0
    raw_names = event.get("names") if isinstance(event.get("names"), list) else []
    names = [sanitize_node_name(n) for n in raw_names if str(n or "").strip()]
    return {
        "risk": normalize_risk(event.get("risk")),
        "matched": matched,
        "nodes": nodes,
        "names": names[:8],
        "source": str(event.get("source") or "tree"),
        "message": "" if matched else NO_REGION_MATCH_MSG,
    }


def build_part_card(
    *,
    part_name: str,
    asset_id: str,
    risk: str,
    rul_days: Any = None,
    sensors: Optional[dict[str, Any]] = None,
    hint: Optional[str] = None,
    db_id: Any = None,
) -> dict[str, Any]:
    """Pure card model the CAD Twin page renders (unit-testable, no Streamlit)."""
    level = normalize_risk(risk)
    raw_name = (part_name or "").strip()
    return {
        "part_name": (sanitize_node_name(raw_name, db_id) if raw_name else "") or "—",
        "raw_part_name": raw_name,
        "asset_id": str(asset_id or "asset"),
        "risk": level,
        "risk_hex": risk_theme_hex(level),
        "rul_days": rul_days,
        "sensors": dict(sensors or {}),
        "hint": hint or None,
        "honesty": "Isolation Forest scores the asset, not each CAD part.",
    }
