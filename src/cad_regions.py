"""Click-to-assign CAD regions — sensor → region labels + a per-model dbId map.

Name matching only works when the STEP export carries real node names. A Fusion
export that lands as ``Solid1`` / ``??????-1-solid1`` gives us nothing to match,
so the user assigns parts by hand: click a part in GuiViewer3D, pick a region,
press **Assign**. The ``{pack + model urn → {region: [dbId]}}`` map is written to
a gitignored JSON next to ``cad_urns.json`` so an Aviation Rotax map survives a
close-tab while this service is up (Render wipes the disk on redeploy).

Theming still refuses to redden the whole engine: only the dbIds the user
assigned to the driver region (and optionally its related regions) get
``setThemingColor``. Unassigned solids keep the viewer's default gray.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from src.twin3d import normalize_risk

_DEFAULT_MAP_NAME = "cad_region_map.json"

# The five buckets the picker offers. Ids are stable; labels are what the UI shows.
REGIONS: tuple[tuple[str, str], ...] = (
    ("heads_exhaust", "Heads/exhaust"),
    ("oil_system", "Oil"),
    ("rotating_crank", "Rotating/crank"),
    ("intake", "Intake"),
    ("other", "Other"),
)
REGION_IDS: tuple[str, ...] = tuple(rid for rid, _ in REGIONS)
REGION_LABELS: dict[str, str] = dict(REGIONS)

# Extra ids/labels ``normalize_region`` accepts (picker says ``Oil``, not ``oil_system``).
_REGION_ALIASES: dict[str, str] = {
    "oil": "oil_system",
    "oilsys": "oil_system",
    "oilsystem": "oil_system",
    "heads": "heads_exhaust",
    "exhaust": "heads_exhaust",
    "headsexhaust": "heads_exhaust",
    "rotating": "rotating_crank",
    "crank": "rotating_crank",
    "prop": "rotating_crank",
    "rotatingassembly": "rotating_crank",
}

# Sensor key → (human phrase for the driver line, region id, related region ids).
# Longest key first so ``oil_temp`` beats ``oil`` and ``temp``. Aviation:
# egt/exhaust → heads/exhaust, oil* → oil system, vibration → rotating assembly,
# rpm → crank/prop, manifold → intake.
_SENSOR_REGION_RULES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("oiltemperature", "oil system", "oil_system", ()),
    ("oiltemp", "oil system", "oil_system", ()),
    ("oilpressure", "oil system", "oil_system", ()),
    ("oilpress", "oil system", "oil_system", ()),
    ("exhaustmanifold", "heads / exhaust", "heads_exhaust", ()),
    ("intakemanifold", "intake", "intake", ()),
    ("manifoldpressure", "intake", "intake", ()),
    ("cylinderhead", "heads / exhaust", "heads_exhaust", ()),
    ("temperature", "heads / exhaust", "heads_exhaust", ()),
    ("coolanttemp", "heads / exhaust", "heads_exhaust", ()),
    ("vibration", "rotating assembly", "rotating_crank", ()),
    ("crankcase", "crank / prop", "rotating_crank", ()),
    ("manifold", "intake", "intake", ()),
    ("exhaust", "heads / exhaust", "heads_exhaust", ()),
    ("intake", "intake", "intake", ()),
    ("crank", "crank / prop", "rotating_crank", ()),
    ("temp", "heads / exhaust", "heads_exhaust", ()),
    ("oil", "oil system", "oil_system", ()),
    ("egt", "heads / exhaust", "heads_exhaust", ()),
    ("cht", "heads / exhaust", "heads_exhaust", ()),
    ("cyl", "heads / exhaust", "heads_exhaust", ()),
    ("vib", "rotating assembly", "rotating_crank", ()),
    ("rpm", "crank / prop", "rotating_crank", ()),
    ("map", "intake", "intake", ()),
)

# Short human tag shown in parentheses after the raw column name.
_SENSOR_SHORT: tuple[tuple[str, str], ...] = (
    ("oiltemperature", "oil temp"),
    ("oiltemp", "oil temp"),
    ("oilpressure", "oil pressure"),
    ("oilpress", "oil pressure"),
    ("exhaustmanifold", "EGT"),
    ("cylinderhead", "CHT"),
    ("manifoldpressure", "MAP"),
    ("intakemanifold", "MAP"),
    ("coolanttemp", "coolant temp"),
    ("temperature", "temp"),
    ("vibration", "vibration"),
    ("manifold", "MAP"),
    ("exhaust", "EGT"),
    ("crank", "RPM"),
    ("egt", "EGT"),
    ("cht", "CHT"),
    ("cyl", "CHT"),
    ("vib", "vibration"),
    ("rpm", "RPM"),
    ("oil", "oil"),
    ("temp", "temp"),
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm(text: Any) -> str:
    return _NON_ALNUM.sub("", str(text or "").lower())


def region_label(region_id: Any) -> str:
    """``heads_exhaust`` → ``Heads/exhaust``. Unknown ids fall back to ``Other``."""
    return REGION_LABELS.get(str(region_id or ""), REGION_LABELS["other"])


def normalize_region(region_id: Any) -> str:
    """Accept an id or a label; return a known region id (``other`` when unclear)."""
    raw = _norm(region_id)
    if not raw:
        return "other"
    if raw in _REGION_ALIASES:
        return _REGION_ALIASES[raw]
    for rid in REGION_IDS:
        if raw == _norm(rid) or raw == _norm(REGION_LABELS[rid]):
            return rid
    return "other"


def sensor_region(sensor: Any) -> dict[str, Any]:
    """Human region phrase + assignable region id for a mapped sensor column."""
    blob = _norm(sensor)
    for key, phrase, rid, related in _SENSOR_REGION_RULES:
        if key and key in blob:
            return {
                "sensor": str(sensor or ""),
                "phrase": phrase,
                "region": rid,
                "related": list(related),
                "matched": key,
            }
    return {
        "sensor": str(sensor or ""),
        "phrase": "other / unmapped region",
        "region": "other",
        "related": [],
        "matched": "",
    }


def sensor_display(sensor: Any) -> str:
    """``exhaust_temp`` → ``exhaust_temp (EGT)``; ``egt`` stays ``egt``."""
    name = str(sensor or "").strip()
    blob = _norm(name)
    if not blob:
        return ""
    for key, short in _SENSOR_SHORT:
        if key in blob:
            return name if _norm(short) == blob else f"{name} ({short})"
    return name


# ── persistence (same shape / atomic write as data/cad_urns.json) ─────────────
def region_map_path() -> Path:
    """Gitignored JSON of assigned dbIds. Override with ``PDM_CAD_REGION_MAP_PATH``."""
    override = (os.getenv("PDM_CAD_REGION_MAP_PATH") or "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data" / _DEFAULT_MAP_NAME


def model_key(pack_id: Any, urn: Any) -> str:
    """One map per industry pack **and** model, so Aviation Rotax keeps its own ids."""
    pid = str(pack_id or "").strip() or "unknown_pack"
    model = str(urn or "").strip()
    try:
        from src.aps_viewer import normalize_model_urn

        model = normalize_model_urn(model) or model
    except Exception:
        pass
    return f"{pid}::{model or 'no_urn'}"


def _empty_doc() -> dict[str, Any]:
    return {"version": 1, "models": {}}


def _clean_ids(raw: Any) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for item in raw or []:
        try:
            ident = int(item)
        except (TypeError, ValueError):
            continue
        if ident in seen:
            continue
        seen.add(ident)
        out.append(ident)
    return out


def _clean_regions(raw: Any) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    if not isinstance(raw, dict):
        return out
    for key, ids in raw.items():
        rid = normalize_region(key)
        cleaned = _clean_ids(ids)
        if not cleaned:
            continue
        merged = out.setdefault(rid, [])
        for ident in cleaned:
            if ident not in merged:
                merged.append(ident)
    return out


def load_region_map(*, path: Optional[Path] = None) -> dict[str, Any]:
    """Read the whole map. A missing / corrupt file is an empty map, never a crash."""
    p = Path(path) if path is not None else region_map_path()
    try:
        if not p.is_file():
            return _empty_doc()
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return _empty_doc()
    if not isinstance(data, dict):
        return _empty_doc()
    models_raw = data.get("models")
    if not isinstance(models_raw, dict):
        return _empty_doc()
    models: dict[str, Any] = {}
    for key, row in models_raw.items():
        if not isinstance(row, dict):
            continue
        regions = _clean_regions(row.get("regions"))
        if not regions:
            continue
        models[str(key)] = {
            "regions": regions,
            "updated_at": str(row.get("updated_at") or ""),
            "pack_id": str(row.get("pack_id") or ""),
            "urn": str(row.get("urn") or ""),
        }
    return {"version": int(data.get("version") or 1), "models": models}


def _atomic_write(p: Path, payload: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix="cad_region_map.", suffix=".json", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        os.replace(tmp_name, p)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def regions_for_model(
    pack_id: Any,
    urn: Any,
    *,
    path: Optional[Path] = None,
) -> dict[str, list[int]]:
    """``{region_id: [dbId, …]}`` saved for this pack + model. Empty when unassigned."""
    doc = load_region_map(path=path)
    row = (doc.get("models") or {}).get(model_key(pack_id, urn)) or {}
    return {rid: list(ids) for rid, ids in (row.get("regions") or {}).items()}


def region_of_db_id(
    pack_id: Any,
    urn: Any,
    db_id: Any,
    *,
    path: Optional[Path] = None,
) -> str:
    """Which region this dbId is already in, or ``""``."""
    try:
        ident = int(db_id)
    except (TypeError, ValueError):
        return ""
    for rid, ids in regions_for_model(pack_id, urn, path=path).items():
        if ident in ids:
            return rid
    return ""


def assign_region(
    pack_id: Any,
    urn: Any,
    db_id: Any,
    region: Any,
    *,
    path: Optional[Path] = None,
) -> dict[str, Any]:
    """Save one clicked dbId into a region. Re-assigning moves it, never duplicates."""
    rid = normalize_region(region)
    key = model_key(pack_id, urn)
    try:
        ident = int(db_id)
    except (TypeError, ValueError):
        return {
            "ok": False,
            "saved": False,
            "reason": "no_dbid",
            "region": rid,
            "dbId": None,
            "message": "Click a part in the viewer first — no dbId to assign.",
        }
    if not str(urn or "").strip():
        return {
            "ok": False,
            "saved": False,
            "reason": "no_urn",
            "region": rid,
            "dbId": ident,
            "message": "Load a model first — the region map is keyed by pack + model URN.",
        }

    p = Path(path) if path is not None else region_map_path()
    doc = load_region_map(path=p)
    models = dict(doc.get("models") or {})
    row = dict(models.get(key) or {})
    regions = {r: list(ids) for r, ids in (row.get("regions") or {}).items()}
    moved_from = ""
    for other, ids in regions.items():
        if ident in ids and other != rid:
            moved_from = other
            regions[other] = [i for i in ids if i != ident]
    target = regions.setdefault(rid, [])
    if ident not in target:
        target.append(ident)
    regions = {r: ids for r, ids in regions.items() if ids}
    models[key] = {
        "regions": regions,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pack_id": str(pack_id or ""),
        "urn": str(urn or ""),
    }
    _atomic_write(p, {"version": 1, "models": models})
    count = len(regions.get(rid) or [])
    return {
        "ok": True,
        "saved": True,
        "reason": "ok",
        "region": rid,
        "region_label": region_label(rid),
        "dbId": ident,
        "moved_from": moved_from,
        "count": count,
        "total": sum(len(ids) for ids in regions.values()),
        "path": str(p),
        "key": key,
        "message": (
            f"dbId {ident} → {region_label(rid)} "
            f"({count} part{'' if count == 1 else 's'} in this region on this server)."
        ),
    }


def clear_region_map(
    pack_id: Any,
    urn: Any,
    *,
    path: Optional[Path] = None,
) -> dict[str, Any]:
    """Drop every assignment for this pack + model. Other models are left alone."""
    key = model_key(pack_id, urn)
    p = Path(path) if path is not None else region_map_path()
    doc = load_region_map(path=p)
    models = dict(doc.get("models") or {})
    row = models.pop(key, None)
    had = sum(len(ids) for ids in ((row or {}).get("regions") or {}).values())
    if row is not None:
        _atomic_write(p, {"version": 1, "models": models})
    return {
        "ok": True,
        "cleared": bool(row is not None),
        "had": had,
        "key": key,
        "path": str(p),
        "message": (
            f"Cleared {had} assigned part{'' if had == 1 else 's'} for this model."
            if had
            else "No region assignments saved for this model."
        ),
    }


def assignment_summary(
    pack_id: Any,
    urn: Any,
    *,
    path: Optional[Path] = None,
) -> dict[str, Any]:
    """Counts per region for the caption under the picker."""
    regions = regions_for_model(pack_id, urn, path=path)
    counts = {rid: len(regions.get(rid) or []) for rid in REGION_IDS if regions.get(rid)}
    total = sum(counts.values())
    text = " · ".join(f"{region_label(rid)} {n}" for rid, n in counts.items())
    return {
        "regions": regions,
        "counts": counts,
        "total": total,
        "text": text,
        "key": model_key(pack_id, urn),
    }


def db_ids_for_regions(
    pack_id: Any,
    urn: Any,
    wanted: Optional[Iterable[str]] = None,
    *,
    path: Optional[Path] = None,
) -> list[int]:
    """Assigned dbIds for the given regions (all regions when ``wanted`` is None)."""
    regions = regions_for_model(pack_id, urn, path=path)
    keys = list(regions.keys()) if wanted is None else [normalize_region(w) for w in wanted]
    out: list[int] = []
    seen: set[int] = set()
    for rid in keys:
        for ident in regions.get(rid) or []:
            if ident in seen:
                continue
            seen.add(ident)
            out.append(ident)
    return out


NO_ASSIGNMENT_MSG = (
    "No CAD parts assigned to this region yet. Click a part in the viewer, pick a region, "
    "and press **Assign selected part to this region** — unassigned solids stay default gray."
)


def assigned_theme_plan(
    pack_id: Any,
    urn: Any,
    risk: Any,
    driver_region: Any = "",
    *,
    related: Optional[Iterable[str]] = None,
    include_related: bool = True,
    path: Optional[Path] = None,
) -> dict[str, Any]:
    """Which **saved** dbIds to paint for this asset's risk + driver region.

    High → red, Medium → orange, Low / Unknown → nothing at all. Only the ids the
    user assigned to the driver region (plus its related regions when asked) are
    returned; an empty map returns an empty list, never the whole model.
    """
    from src.cad_part import region_theme_rgba, risk_theme_hex

    level = normalize_risk(risk)
    rgba = region_theme_rgba(level)
    summary = assignment_summary(pack_id, urn, path=path)
    base = {
        "risk": level,
        "rgba": rgba,
        "hex": risk_theme_hex(level),
        "dbIds": [],
        "region": normalize_region(driver_region) if str(driver_region or "") else "",
        "regions_used": [],
        "assigned_total": summary["total"],
        "tint": False,
        "source": "assigned",
        "message": "",
    }
    if rgba is None:
        return base
    if not summary["total"]:
        base["message"] = NO_ASSIGNMENT_MSG
        return base

    wanted: list[str] = []
    if str(driver_region or ""):
        wanted.append(normalize_region(driver_region))
        if include_related:
            for rid in related or []:
                rid_norm = normalize_region(rid)
                if rid_norm not in wanted:
                    wanted.append(rid_norm)
    else:
        wanted = list(summary["regions"].keys())

    ids = db_ids_for_regions(pack_id, urn, wanted, path=path)
    used = [rid for rid in wanted if summary["regions"].get(rid)]
    base["dbIds"] = ids
    base["regions_used"] = used
    base["tint"] = bool(ids)
    if not ids:
        labels = ", ".join(region_label(r) for r in wanted) or "that region"
        base["message"] = (
            f"{summary['total']} part(s) are assigned, but none to {labels}. "
            "Assign the driver region's parts to see them tinted."
        )
    return base


def dumps_region_map(*, path: Optional[Path] = None, doc: Optional[dict[str, Any]] = None) -> str:
    """Pretty JSON for download. Always a valid document, even if the file is missing."""
    payload = doc if isinstance(doc, dict) else load_region_map(path=path)
    if "version" not in payload:
        payload = {"version": 1, "models": payload.get("models") if isinstance(payload, dict) else {}}
    if "models" not in payload or not isinstance(payload.get("models"), dict):
        payload = {"version": int(payload.get("version") or 1), "models": {}}
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def parse_region_map_payload(raw: Any) -> dict[str, Any]:
    """Sanitize imported JSON / dict. Missing or junk → empty map, never a crash."""
    data: Any = raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return _empty_doc()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return _empty_doc()
        try:
            data = json.loads(text)
        except Exception:
            return _empty_doc()
    if not isinstance(data, dict):
        return _empty_doc()
    models_raw = data.get("models")
    if not isinstance(models_raw, dict):
        # Allow a bare {pack::urn: {regions: …}} or a single-model regions dict.
        if any(isinstance(v, dict) and ("regions" in v or any(k in REGION_IDS for k in v)) for v in data.values()):
            models_raw = {k: v for k, v in data.items() if k not in {"version", "models"}}
        else:
            models_raw = {}
    models: dict[str, Any] = {}
    for key, row in models_raw.items():
        if not isinstance(row, dict):
            continue
        regions = _clean_regions(row.get("regions") if "regions" in row else row)
        if not regions:
            continue
        models[str(key)] = {
            "regions": regions,
            "updated_at": str(row.get("updated_at") or ""),
            "pack_id": str(row.get("pack_id") or ""),
            "urn": str(row.get("urn") or ""),
        }
    return {"version": int(data.get("version") or 1), "models": models}


def merge_region_maps(base: Any, incoming: Any) -> dict[str, Any]:
    """Union of models. Incoming wins per model key; other models stay (pack switch safe)."""
    left = parse_region_map_payload(base)
    right = parse_region_map_payload(incoming)
    models = dict(left.get("models") or {})
    models.update(dict(right.get("models") or {}))
    return {"version": 1, "models": models}


def import_region_map(
    raw: Any,
    *,
    path: Optional[Path] = None,
    merge: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Import JSON onto disk. Merge by default so a pack switch does not wipe other maps."""
    incoming = parse_region_map_payload(raw)
    p = Path(path) if path is not None else region_map_path()
    existing = load_region_map(path=p)
    doc = merge_region_maps(existing, incoming) if merge else incoming
    n_models = len(doc.get("models") or {})
    saved = False
    err = ""
    if write:
        try:
            _atomic_write(p, doc)
            saved = True
        except Exception as exc:
            err = str(exc)
    total_ids = sum(
        len(ids)
        for row in (doc.get("models") or {}).values()
        for ids in ((row or {}).get("regions") or {}).values()
    )
    return {
        "ok": True,
        "saved": saved,
        "merged": bool(merge),
        "models": n_models,
        "assigned_ids": total_ids,
        "path": str(p),
        "doc": doc,
        "error": err,
        "message": (
            f"Imported {total_ids} assigned part(s) across {len(doc.get('models') or {})} model(s)"
            + (" and wrote disk." if saved else " (disk write skipped or failed).")
        ),
    }


def cad_map_secret_text() -> str:
    """CAD_MAP_JSON env or Streamlit secret — JSON text or a file path."""
    try:
        import config as _cfg

        text = str(_cfg._setting("CAD_MAP_JSON", "") or "").strip()
    except Exception:
        text = (os.getenv("CAD_MAP_JSON") or "").strip()
    return text


def load_cad_map_secret() -> dict[str, Any]:
    """Parse CAD_MAP_JSON. File path or JSON string. Empty doc when unset."""
    text = cad_map_secret_text()
    if not text:
        return _empty_doc()
    try:
        p = Path(text)
        if p.is_file():
            return parse_region_map_payload(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return parse_region_map_payload(text)


def seed_region_map_from_env(*, path: Optional[Path] = None) -> dict[str, Any]:
    """If disk is empty and CAD_MAP_JSON is set, write the secret map. Honest Render seed."""
    p = Path(path) if path is not None else region_map_path()
    disk = load_region_map(path=p)
    disk_models = disk.get("models") or {}
    secret = load_cad_map_secret()
    secret_models = secret.get("models") or {}
    if disk_models:
        return {
            "ok": True,
            "seeded": False,
            "reason": "disk_has_map",
            "models": len(disk_models),
            "path": str(p),
        }
    if not secret_models:
        return {"ok": True, "seeded": False, "reason": "no_secret", "models": 0, "path": str(p)}
    try:
        _atomic_write(p, secret)
        return {
            "ok": True,
            "seeded": True,
            "reason": "secret",
            "models": len(secret_models),
            "path": str(p),
        }
    except Exception as exc:
        return {"ok": False, "seeded": False, "reason": "write_failed", "error": str(exc), "path": str(p)}


def map_persist_status(
    *,
    path: Optional[Path] = None,
    session_has: bool = False,
    render_ephemeral: Optional[bool] = None,
) -> dict[str, Any]:
    """One-line honesty: session / disk / secret / not saved (Render disk resets)."""
    p = Path(path) if path is not None else region_map_path()
    disk_doc = load_region_map(path=p)
    disk_n = sum(
        len(ids)
        for row in (disk_doc.get("models") or {}).values()
        for ids in ((row or {}).get("regions") or {}).values()
    )
    on_disk = bool(p.is_file() and disk_n)
    secret = bool((load_cad_map_secret().get("models") or {}))
    if render_ephemeral is None:
        render_ephemeral = bool(os.getenv("RENDER") or os.getenv("RENDER_SERVICE_ID"))
    bits: list[str] = []
    if session_has:
        bits.append("in session")
    if on_disk:
        bits.append("on disk")
    if secret:
        bits.append("from secret")
    if not bits:
        bits.append("not saved")
    if render_ephemeral:
        bits.append("Render disk resets")
    elif on_disk:
        bits.append("local file (gitignored)")
    else:
        bits.append("export JSON to keep it")
    return {
        "session": bool(session_has),
        "disk": on_disk,
        "secret": secret,
        "disk_ids": disk_n,
        "path": str(p),
        "ephemeral": bool(render_ephemeral),
        "line": "Map " + " / ".join(bits),
    }


BROWSER_MAP_KEY = "pdm_cad_region_map_v1"


def cad_map_localstorage_html(doc: Optional[dict[str, Any]] = None) -> str:
    """Write the current map JSON into browser localStorage (CAD Twin backup)."""
    payload = dumps_region_map(doc=doc or load_region_map())
    safe = json.dumps(payload)
    return f"""<!DOCTYPE html><html><body style="margin:0;font:12px system-ui,sans-serif;color:#6c757d">
<script>
(function(){{
  var KEY = {json.dumps(BROWSER_MAP_KEY)};
  try {{
    localStorage.setItem(KEY, {safe});
  }} catch (e) {{}}
}})();
</script>
Browser backup key <code>{BROWSER_MAP_KEY}</code> — export JSON to restore after a new session.
</body></html>
"""

