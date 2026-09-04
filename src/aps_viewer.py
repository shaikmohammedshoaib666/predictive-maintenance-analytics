"""Upgrade 3 — Autodesk Platform Services (APS / Forge) CAD viewer.

Embeds the APS Viewer to show a *real* translated CAD model (Revit/Fusion/IFC →
SVF) of an asset, and tints/pulses it red when the selected asset's predicted
risk is High. This is **availability-gated**: without `APS_CLIENT_ID` +
`APS_CLIENT_SECRET` (and a translated model `URN`) the feature simply shows
setup instructions, so the app and the Render deploy are never affected.

Credentials are read from environment secrets and never written to disk or the
page beyond the short-lived viewer access token that the APS Viewer requires.

CAD Twin can also **upload a CAD file**, push it to OSS (signed S3), and kick a
Model Derivative SVF2 job so the user gets a URN without leaving the SaaS.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import parse_qs, unquote, urlparse, quote

from src.cad_part import REGION_THEME_RGBA, region_hint_keywords

APS_AUTH_URL = "https://developer.api.autodesk.com/authentication/v2/token"
APS_BASE = "https://developer.api.autodesk.com"
OSS_BUCKETS_URL = f"{APS_BASE}/oss/v2/buckets"
MD_JOB_URL = f"{APS_BASE}/modelderivative/v2/designdata/job"

# Viewer load only needs viewables. Translate needs Data Management write + buckets.
VIEWER_SCOPES = "viewables:read data:read"
TRANSLATE_SCOPES = (
    "viewables:read data:read data:write data:create bucket:create bucket:read"
)

INSUFFICIENT_SCOPE_HINT = (
    "APS app must allow Model Derivative + Data Management; "
    "token needs data:write/create and bucket:create/read in addition to viewables:read."
)

# Honest Model Derivative inputs we surface in the uploader (not every CAD kernel).
# Autodesk's full table is longer; exotic kernels can still fail on their side.
CAD_UPLOAD_EXTENSIONS: tuple[str, ...] = (
    "3dm",
    "3ds",
    "asm",
    "catpart",
    "catproduct",
    "dwf",
    "dwfx",
    "dwg",
    "dxf",
    "f2d",
    "f3d",
    "fbx",
    "iam",
    "idw",
    "ifc",
    "ifczip",
    "ige",
    "iges",
    "igs",
    "ipt",
    "jt",
    "nwc",
    "nwd",
    "nwf",
    "obj",
    "prt",
    "rcs",
    "rcp",
    "rfa",
    "rte",
    "rvt",
    "sab",
    "sat",
    "skp",
    "sldasm",
    "sldprt",
    "ste",
    "step",
    "stl",
    "stp",
    "stpz",
    "wire",
    "x_b",
    "x_t",
    "zip",
)

# Keep in sync with .streamlit/config.toml [server] maxUploadSize (MB).
MAX_CAD_UPLOAD_MB = 300
MAX_CAD_BYTES = MAX_CAD_UPLOAD_MB * 1024 * 1024
WARN_CAD_BYTES = 80 * 1024 * 1024

# If Streamlit accepts the file but Render / a reverse proxy still rejects the POST.
CAD_SIZE_ZIP_FALLBACK = (
    "zip the STEP (often drops under 200 MB) and upload the zip; "
    "or export a lighter STEP from Fusion."
)

# Autodesk requires input.rootFilename when input.compressedUrn is true.
# Prefer STEP/stp when scanning zip members (this product's Rotax CAD path).
ZIP_STEP_ROOT_EXTENSIONS: tuple[str, ...] = (".step", ".stp", ".stpz")
ZIP_CAD_ROOT_EXTENSIONS: tuple[str, ...] = ZIP_STEP_ROOT_EXTENSIONS + (
    ".rvt",
    ".ifc",
    ".ipt",
    ".f3d",
    ".nwd",
    ".dwg",
    ".iam",
)
ZIP_CAD_ROOT_HINTS: tuple[str, ...] = ("step", "stp", "rotax", "engine")
ZIP_ROOT_MISSING_MSG = (
    "Could not detect a .STEP/.stp inside this zip (ignored __MACOSX), and the "
    "upload name is not like Engine.STEP.zip so rootFilename cannot be inferred."
)

# OSS signed-upload: at most 25 URLs per GET.
_SIGNED_URL_BATCH = 25
_MIN_PART_BYTES = 5 * 1024 * 1024

POLL_TIMEOUT_S = 150.0
POLL_INTERVAL_S = 4.0

# Autodesk sample used in Model Derivative tutorials (not bundled in this repo).
APS_SAMPLE_CAD_URL = (
    "https://aps.autodesk.com/en/docs/model-derivative/v2/tutorials/prep-file4viewer/"
)

OnStatus = Callable[[str, str], None]


def _aps_setting(key: str) -> str:
    """Read APS secrets at call time (env or Streamlit secrets). Safe after Render deploy."""
    try:
        import config as _cfg

        return str(_cfg._setting(key, "") or "").strip()
    except Exception:
        return (os.getenv(key) or "").strip()


def aps_available() -> tuple[bool, str]:
    """True when APS client credentials are present in the environment or secrets."""
    cid = _aps_setting("APS_CLIENT_ID")
    sec = _aps_setting("APS_CLIENT_SECRET")
    if not cid or not sec:
        return False, "APS_CLIENT_ID / APS_CLIENT_SECRET not set (add on Render anytime)"
    return True, "APS credentials detected"


def aps_model_urn() -> str:
    """Optional translated-model URN from env/secrets so CAD can light up without a UI paste."""
    return _aps_setting("APS_MODEL_URN")


def oss_bucket_key(client_id: str) -> str:
    """Globally unique-ish OSS bucket key: 3–128 chars, lowercase [a-z0-9._-]."""
    prefix = "".join(ch for ch in (client_id or "").lower() if ch.isalnum())[:12]
    if not prefix:
        prefix = "app"
    key = f"pdm-{prefix}-cad"
    if len(key) < 3:
        key = "pdm-app-cad"
    return key[:128]


def sanitize_object_name(filename: str, when: Optional[datetime] = None) -> str:
    """OSS object key: sanitized basename + UTC timestamp. No path segments."""
    when = when or datetime.now(timezone.utc)
    stamp = when.strftime("%Y%m%d%H%M%S")
    base = os.path.basename(filename or "") or "model"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "model"
    if len(safe) > 180:
        root, ext = os.path.splitext(safe)
        safe = root[:180] + ext[:20]
    return f"{stamp}_{safe}"


def oss_object_id(bucket_key: str, object_key: str) -> str:
    return f"urn:adsk.objects:os.object:{bucket_key}/{object_key}"


def encode_model_urn(object_id: str) -> str:
    """URL-safe base64 of the OSS object id, no padding, no ``urn:`` prefix (viewer adds it)."""
    raw = (object_id or "").encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def encode_oss_urn(bucket_key: str, object_key: str) -> str:
    return encode_model_urn(oss_object_id(bucket_key, object_key))


def strip_urn_query_fragment(raw: str) -> str:
    """Drop ``?query`` and ``#fragment`` so ``sheetId`` never stays on a URN."""
    text = (raw or "").strip()
    for sep in ("?", "#"):
        if sep in text:
            text = text.split(sep, 1)[0]
    return text.strip()


def normalize_model_urn(raw: str) -> str:
    """Accept paste of base64, ``urn:``+base64, or a raw ``urn:adsk.objects:...`` id."""
    u = strip_urn_query_fragment(raw or "")
    if not u:
        return ""
    if u.lower().startswith("urn:adsk."):
        return encode_model_urn(u)
    if u.lower().startswith("urn:"):
        u = u[4:]
    return strip_urn_query_fragment(u)


def decode_model_urn(raw: str) -> str:
    """Decode a ``dXJu…`` (or ``urn:adsk.…``) value to the OSS object id. Empty on failure."""
    text = strip_urn_query_fragment(raw or "")
    if not text:
        return ""
    if text.lower().startswith("urn:adsk."):
        return text
    encoded = text[4:] if text.lower().startswith("urn:") else text
    encoded = strip_urn_query_fragment(encoded).strip()
    if not encoded.startswith("dXJu"):
        encoded = normalize_model_urn(text)
    if not encoded.startswith("dXJu"):
        return ""
    pad = "=" * ((4 - len(encoded) % 4) % 4)
    blob = (encoded + pad).encode("ascii", errors="ignore")
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            out = decoder(blob).decode("utf-8")
            if out.lower().startswith("urn:"):
                return out
        except Exception:
            continue
    return ""


# Autodesk encoded URNs start with dXJu. Standard base64 may include ``/`` and ``+``.
_DXJU_RE = re.compile(r"(dXJu[A-Za-z0-9+/_-]+=*)")
_ADSK_OBJECT_RE = re.compile(r"(urn:adsk\.[^\s\"'<>?#]+)", re.I)
_PUBLIC_VIEWER_HOSTS = ("viewer.autodesk.com", "autode.sk")
_PUBLIC_URN_MARKERS = ("a360viewer", "viewer.autodesk.com")

PUBLIC_VIEWER_WARNING = (
    "That paste is from viewer.autodesk.com (Autodesk’s public viewer). The URN lives in "
    "Autodesk’s — or someone else’s — bucket, not yours. Your 2-legged app token cannot "
    "load it. Saving it is useless. Choose the STEP/CAD file on this page → Translate to "
    "SVF so the URN is in YOUR APS bucket; then we save that URN for this pack."
)

PUBLIC_VIEWER_NO_URN_WARNING = (
    "That looks like a viewer.autodesk.com share link, not a Model Derivative URN "
    "(no dXJu… in the URL). This app cannot log into Autodesk’s public viewer. "
    "Choose the STEP/CAD file here and Translate to SVF so the model lives in YOUR bucket."
)

A360_PUBLIC_URN_ERROR = (
    "This URN is from Autodesk’s public Viewer website. Your APS app cannot open it. "
    "Use Choose CAD file → Translate with the same STEP (Rotax 912), then Load, then Save URN."
)

# Autodesk.Viewing.ErrorCodes (viewer3D). 4 = NETWORK_ACCESS_DENIED.
VIEWER_ERROR_CODES: dict[int, str] = {
    1: "Unknown failure",
    2: "Bad data",
    3: "Network failure",
    4: "Access denied (NETWORK_ACCESS_DENIED) — your APS app token cannot read this URN",
    5: "File not found",
    6: "Network server error",
    7: "Unhandled network response",
    8: "Browser WebGL not supported",
    9: "Model is empty",
    10: "Too many requests",
    11: "Unhandled exception",
    12: "WebGL context lost",
    13: "Load canceled",
}

# GuiViewer3D already ships these 3D tools; we refuse to disable them and also load extras.
VIEWER_DEFAULT_3D_EXTENSIONS: tuple[str, ...] = (
    "Autodesk.ViewCubeUi",
    "Autodesk.Measure",
    "Autodesk.Section",
    "Autodesk.Explode",
    "Autodesk.BimWalk",
    "Autodesk.Viewing.FusionOrbit",
    "Autodesk.LayerManager",
    "Autodesk.ModelStructure",
    "Autodesk.PropertiesManager",
)
VIEWER_EXTRA_EXTENSIONS: tuple[str, ...] = (
    "Autodesk.DocumentBrowser",
    "Autodesk.Viewing.MarkupsCore",
    "Autodesk.Viewing.MarkupsGui",
    "Autodesk.FirstPerson",
)

_DEFAULT_URNS_NAME = "cad_urns.json"


def _blob_is_autodesk_public(blob: str) -> bool:
    text = (blob or "").lower()
    if any(host in text for host in _PUBLIC_VIEWER_HOSTS):
        return True
    return any(marker in text for marker in _PUBLIC_URN_MARKERS)


def looks_like_public_viewer(raw: str) -> bool:
    """True for viewer.autodesk.com pastes and for URNs in Autodesk’s a360viewer bucket."""
    if _blob_is_autodesk_public(raw or ""):
        return True
    return _blob_is_autodesk_public(decode_model_urn(raw))


def describe_viewer_error(err: Any) -> str:
    """Map APS Viewer ErrorCodes (especially 4 = NETWORK_ACCESS_DENIED) to human text."""
    code: Optional[int] = None
    if isinstance(err, bool):
        code = None
    elif isinstance(err, int):
        code = err
    elif isinstance(err, float) and err == int(err):
        code = int(err)
    elif isinstance(err, str) and err.strip().lstrip("-").isdigit():
        code = int(err.strip())
    elif isinstance(err, dict):
        for key in ("code", "errorCode", "error_code"):
            val = err.get(key)
            if val is None or val == "":
                continue
            try:
                code = int(val)
                break
            except (TypeError, ValueError):
                continue
    if code == 4:
        return (
            "Access denied (error 4 / NETWORK_ACCESS_DENIED). Your 2-legged APS token cannot "
            "read this object. Autodesk public Viewer URNs (a360viewer-protected) are not in your bucket."
        )
    if code in VIEWER_ERROR_CODES:
        return f"{VIEWER_ERROR_CODES[code]} (error {code})"
    if err is None or err == "":
        return "Unknown viewer error"
    return str(err)


def _query_first(qs: dict[str, list[str]], *keys: str) -> str:
    lower = {k.lower(): v for k, v in qs.items()}
    for key in keys:
        vals = lower.get(key.lower()) or []
        if vals and str(vals[0]).strip():
            return unquote(str(vals[0]).strip())
    return ""


def extract_model_urn(raw: str) -> tuple[str, dict[str, Any]]:
    """Pull a normalized ``dXJu…`` URN from a paste (raw URN, object id, or viewer URL)."""
    text = (raw or "").strip()
    meta: dict[str, Any] = {
        "source": "empty",
        "public_viewer": False,
        "a360_protected": False,
        "block_load": False,
        "warning": "",
        "error": "",
        "decoded": "",
        "raw": text,
    }
    if not text:
        return "", meta

    public = looks_like_public_viewer(text)
    meta["public_viewer"] = public
    if public:
        meta["warning"] = PUBLIC_VIEWER_WARNING
        meta["error"] = A360_PUBLIC_URN_ERROR

    def _finish(urn: str, source: str) -> tuple[str, dict[str, Any]]:
        urn = normalize_model_urn(strip_urn_query_fragment(urn))
        decoded = decode_model_urn(urn)
        meta["decoded"] = decoded
        a360 = _blob_is_autodesk_public(decoded)
        if a360 or public or looks_like_public_viewer(urn):
            meta["public_viewer"] = True
            meta["a360_protected"] = a360 or "a360viewer" in (decoded or "").lower()
            meta["block_load"] = True
            meta["warning"] = PUBLIC_VIEWER_WARNING
            meta["error"] = A360_PUBLIC_URN_ERROR
        meta["source"] = source
        return urn, meta

    candidates: list[tuple[str, str]] = []

    looks_url = "://" in text or text.lower().startswith("viewer.autodesk") or text.lower().startswith("autode.sk")
    if looks_url:
        url = text if "://" in text else "https://" + text
        try:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query, keep_blank_values=False)
            for key in ("urn", "id", "objectid", "objectId", "url", "model", "file"):
                if key.lower() == "sheetid":
                    continue
                val = _query_first(qs, key)
                if val:
                    candidates.append((strip_urn_query_fragment(val), "query"))
            path = unquote(parsed.path or "").strip()
            if path:
                # Do not split on ``/`` — standard base64 URNs contain slashes.
                candidates.append((strip_urn_query_fragment(path), "path"))
                id_m = re.search(r"/id/(.+)$", path, re.I)
                if id_m:
                    candidates.append((strip_urn_query_fragment(id_m.group(1)), "path"))
            frag = unquote(parsed.fragment or "").strip()
            if frag:
                frag_body = frag[1:] if frag.startswith("?") else frag
                if "=" in frag_body:
                    fqs = parse_qs(frag_body, keep_blank_values=False)
                    for key in ("urn", "id", "url"):
                        val = _query_first(fqs, key)
                        if val:
                            candidates.append((strip_urn_query_fragment(val), "fragment"))
                candidates.append((strip_urn_query_fragment(frag_body), "fragment"))
        except Exception:
            pass

    candidates.append((strip_urn_query_fragment(text), "paste"))

    for cand, origin in candidates:
        cand = strip_urn_query_fragment(cand)
        if not cand:
            continue
        if cand.lower().startswith("urn:adsk."):
            urn = normalize_model_urn(cand)
            if urn:
                return _finish(urn, "object_id" if origin == "paste" else origin)
        dx = _DXJU_RE.search(cand)
        if dx:
            urn = normalize_model_urn(dx.group(1))
            if urn:
                src = "viewer_url" if public else ("urn" if origin == "paste" else origin)
                return _finish(urn, src)
        adsk = _ADSK_OBJECT_RE.search(cand)
        if adsk:
            urn = normalize_model_urn(adsk.group(1))
            if urn:
                return _finish(urn, "object_id")

    if public:
        meta["source"] = "viewer_url_no_urn"
        meta["warning"] = PUBLIC_VIEWER_NO_URN_WARNING
        meta["error"] = A360_PUBLIC_URN_ERROR
        meta["block_load"] = True
        return "", meta

    cleaned = strip_urn_query_fragment(text)
    dx = _DXJU_RE.search(cleaned)
    if dx:
        urn = normalize_model_urn(dx.group(1))
        if urn:
            return _finish(urn, "urn")
    if cleaned.lower().startswith("urn:adsk.") or cleaned.startswith("dXJu"):
        return _finish(normalize_model_urn(cleaned), "urn")
    meta["source"] = "unknown"
    return "", meta


def cad_urns_path() -> Path:
    """JSON file of last-successful URNs keyed by industry pack. Override with PDM_CAD_URNS_PATH."""
    override = (os.getenv("PDM_CAD_URNS_PATH") or "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data" / _DEFAULT_URNS_NAME


def _empty_urns_doc() -> dict[str, Any]:
    return {"version": 1, "packs": {}}


def load_saved_urns(*, path: Optional[Path] = None) -> dict[str, Any]:
    p = Path(path) if path is not None else cad_urns_path()
    try:
        if not p.is_file():
            return _empty_urns_doc()
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _empty_urns_doc()
        packs = data.get("packs")
        if not isinstance(packs, dict):
            packs = {}
            # Allow a flat {pack_id: {urn: ...}} or {pack_id: "dXJu..."} file.
            for k, v in data.items():
                if k in ("version", "packs"):
                    continue
                if isinstance(v, str) and v.strip():
                    packs[str(k)] = {"urn": normalize_model_urn(v)}
                elif isinstance(v, dict) and v.get("urn"):
                    packs[str(k)] = v
            data = {"version": int(data.get("version") or 1), "packs": packs}
        else:
            data = {"version": int(data.get("version") or 1), "packs": packs}
        return data
    except Exception:
        return _empty_urns_doc()


def saved_urn_for_pack(pack_id: str, *, path: Optional[Path] = None) -> str:
    pid = (pack_id or "").strip()
    if not pid:
        return ""
    packs = load_saved_urns(path=path).get("packs") or {}
    row = packs.get(pid) or {}
    if isinstance(row, str):
        return normalize_model_urn(row)
    if isinstance(row, dict):
        return normalize_model_urn(str(row.get("urn") or ""))
    return ""


def save_urn_for_pack(
    pack_id: str,
    urn: str,
    *,
    source: str = "load",
    public_viewer: bool = False,
    path: Optional[Path] = None,
) -> dict[str, Any]:
    """Persist a working URN for an industry pack. Refuses public-viewer URNs."""
    pid = (pack_id or "").strip()
    normalized = normalize_model_urn(urn)
    if public_viewer or looks_like_public_viewer(urn):
        return {
            "ok": False,
            "saved": False,
            "reason": "public_viewer",
            "urn": normalized,
            "pack_id": pid,
            "message": PUBLIC_VIEWER_WARNING,
        }
    if not pid:
        return {"ok": False, "saved": False, "reason": "no_pack", "urn": normalized, "pack_id": pid, "message": ""}
    if not normalized:
        return {"ok": False, "saved": False, "reason": "no_urn", "urn": "", "pack_id": pid, "message": ""}

    p = Path(path) if path is not None else cad_urns_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = load_saved_urns(path=p)
    packs = dict(doc.get("packs") or {})
    packs[pid] = {
        "urn": normalized,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
    }
    _atomic_write_urns(p, {"version": 1, "packs": packs})
    return {
        "ok": True,
        "saved": True,
        "reason": "ok",
        "urn": normalized,
        "pack_id": pid,
        "path": str(p),
        "message": saved_urn_caption(pid),
    }


def _atomic_write_urns(p: Path, payload: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix="cad_urns.", suffix=".json", dir=str(p.parent))
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


def _pack_short(pack_id: str) -> str:
    short = pack_id or "this pack"
    try:
        from src.industry_packs import get_pack

        short = str(get_pack(pack_id).get("short") or short)
    except Exception:
        pass
    return short


def delete_urn_for_pack(pack_id: str, *, path: Optional[Path] = None) -> dict[str, Any]:
    """Remove this pack's URN from the server JSON. Other packs are left alone."""
    pid = (pack_id or "").strip()
    short = _pack_short(pid)
    if not pid:
        return {
            "ok": False,
            "deleted": False,
            "had_urn": False,
            "reason": "no_pack",
            "pack_id": pid,
            "message": "No industry pack selected.",
        }
    p = Path(path) if path is not None else cad_urns_path()
    had = bool(saved_urn_for_pack(pid, path=p))
    doc = load_saved_urns(path=p)
    packs = dict(doc.get("packs") or {})
    if pid in packs:
        packs.pop(pid, None)
        if p.exists() or had:
            _atomic_write_urns(p, {"version": 1, "packs": packs})
    return {
        "ok": True,
        "deleted": True,
        "had_urn": had,
        "reason": "ok" if had else "already_empty",
        "pack_id": pid,
        "path": str(p),
        "message": (
            f"Deleted saved URN for {short} on this server."
            if had
            else f"No saved URN for {short} on this server."
        ),
    }


def saved_urn_caption(pack_id: str) -> str:
    """Honest: we wrote a file on this server. We did not write Render env."""
    return (
        f"Saved for {_pack_short(pack_id)} on this server. "
        "To keep it across Render redeploys, also set APS_MODEL_URN."
    )


def resolve_cad_urn(
    pack_id: str = "",
    session_urn: str = "",
    *,
    path: Optional[Path] = None,
) -> dict[str, Any]:
    """Session paste wins, then Render ``APS_MODEL_URN`` override, then pack JSON."""
    env = normalize_model_urn(aps_model_urn())
    saved = saved_urn_for_pack(pack_id, path=path) if pack_id else ""
    extracted, meta = extract_model_urn(session_urn)
    sess = extracted or normalize_model_urn(session_urn)
    if sess:
        urn, source = sess, "session"
    elif env:
        urn, source = env, "env"
    elif saved:
        urn, source = saved, "saved"
    else:
        urn, source = "", "none"
    return {
        "urn": urn,
        "source": source,
        "env_urn": env,
        "saved_urn": saved,
        "env_override": bool(env),
        "pack_id": pack_id or "",
        "extract": meta,
        "public_viewer": bool(meta.get("public_viewer")),
    }


def is_cad_zip_name(filename: str) -> bool:
    """True for Autodesk zip / ifczip uploads that need compressedUrn + rootFilename."""
    name = (filename or "").lower()
    return name.endswith(".zip") or name.endswith(".ifczip")


def _zip_member_norm(name: str) -> str:
    return (name or "").replace("\\", "/").lstrip("./")


def _zip_member_basename(name: str) -> str:
    return _zip_member_norm(name).rstrip("/").rsplit("/", 1)[-1]


def _is_ignored_zip_member(name: str) -> bool:
    norm = _zip_member_norm(name)
    base = _zip_member_basename(norm)
    lower = norm.lower()
    if not base or norm.endswith("/"):
        return True
    if lower.startswith("__macosx/") or base.startswith("._"):
        return True
    if base.lower() in {".ds_store", "thumbs.db"}:
        return True
    return False


def _zip_member_is_cad(name: str) -> bool:
    if _is_ignored_zip_member(name):
        return False
    lower = _zip_member_norm(name).lower()
    return any(lower.endswith(ext) for ext in ZIP_CAD_ROOT_EXTENSIONS)


def _zip_member_is_step(name: str) -> bool:
    if _is_ignored_zip_member(name):
        return False
    lower = _zip_member_norm(name).lower()
    return any(lower.endswith(ext) for ext in ZIP_STEP_ROOT_EXTENSIONS)


def _zip_name_has_hint(name: str) -> bool:
    lower = _zip_member_norm(name).lower()
    base = _zip_member_basename(lower)
    return any(hint in base for hint in ZIP_CAD_ROOT_HINTS)


def _zip_member_size(zf: zipfile.ZipFile, member: str) -> int:
    try:
        return int(zf.getinfo(member).file_size or 0)
    except KeyError:
        want = _zip_member_norm(member)
        for info in zf.infolist():
            if _zip_member_norm(info.filename) == want:
                return int(info.file_size or 0)
    except Exception:
        return 0
    return 0


def pick_zip_cad_root(data: bytes) -> str:
    """Pick Autodesk ``rootFilename`` from zip/ifczip bytes. Empty if none.

    Uses ``ZipFile.namelist()``. Ignores ``__MACOSX`` / AppleDouble / folder
    entries. Prefers ``.STEP`` / ``.stp`` / ``.stpz``. One match → that member.
    Many matches → names containing step/stp/rotax/engine, else the largest.
    """
    if not data:
        return ""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            candidates: list[tuple[str, int]] = []
            for raw_name in zf.namelist():
                raw = _zip_member_norm(raw_name)
                if not _zip_member_is_cad(raw):
                    continue
                candidates.append((raw, _zip_member_size(zf, raw_name)))
    except zipfile.BadZipFile:
        return ""
    except Exception:
        return ""
    if not candidates:
        return ""
    step_like = [c for c in candidates if _zip_member_is_step(c[0])]
    pool = step_like or candidates
    if len(pool) == 1:
        return pool[0][0]
    hinted = [c for c in pool if _zip_name_has_hint(c[0])]
    ranked = hinted or pool
    ranked.sort(key=lambda c: (c[1], len(c[0])), reverse=True)
    return ranked[0][0]


def guess_zip_root_from_upload_name(filename: str) -> str:
    """If the upload is ``Something.STEP.zip`` / ``Something.stp.zip``, return that STEP name.

    Does not open the archive. ``*.ifczip`` is a different Autodesk container and
    is not treated as ``*.ifc.zip``. Empty if the stem is not a CAD root.
    """
    base = os.path.basename((filename or "").replace("\\", "/"))
    if not base:
        return ""
    lower = base.lower()
    if lower.endswith(".ifczip"):
        return ""
    if not lower.endswith(".zip"):
        return ""
    stem = base[:-4]
    stem_lower = stem.lower()
    for ext in ZIP_CAD_ROOT_EXTENSIONS:
        if stem_lower.endswith(ext):
            return stem
    return ""


def resolve_zip_root_filename(filename: str, data: bytes, root_filename: str = "") -> str:
    """Root Autodesk must receive for a zip. Empty means we must not send compressedUrn.

    Order: explicit widget value → zip namelist (.STEP/.stp, ignore __MACOSX) →
    upload name ``Something.STEP.zip`` / ``Something.stp.zip``.
    """
    given = (root_filename or "").strip()
    if given:
        return given
    if not is_cad_zip_name(filename):
        return ""
    return pick_zip_cad_root(data) or guess_zip_root_from_upload_name(filename)


def cad_size_issue(n_bytes: int) -> tuple[Optional[str], str]:
    """Return (\"error\"|\"warn\"|None, message) for Streamlit/Render size limits."""
    n = int(n_bytes or 0)
    if n <= 0:
        return "error", "CAD file is empty."
    if n > MAX_CAD_BYTES:
        mb = n / (1024 * 1024)
        return (
            "error",
            f"File is {mb:.0f} MB. Streamlit maxUploadSize is {MAX_CAD_UPLOAD_MB} MB. "
            f"{CAD_SIZE_ZIP_FALLBACK[0].upper()}{CAD_SIZE_ZIP_FALLBACK[1:]}",
        )
    if n > WARN_CAD_BYTES:
        mb = n / (1024 * 1024)
        return (
            "warn",
            f"File is {mb:.0f} MB. Large uploads can time out on Render; translation may take a while. "
            f"If the upload still fails (Render or a proxy may block large POSTs), {CAD_SIZE_ZIP_FALLBACK}",
        )
    return None, ""


def looks_like_insufficient_scope(status_code: int, body: str) -> bool:
    blob = f"{status_code} {body or ''}".lower()
    needles = (
        "insufficient scope",
        "invalid_scope",
        "auth-010",
        "does not have the required privileges",
        "required privileges",
        "missing scope",
        "scope is not allowed",
        "the access token does not have",
    )
    return any(n in blob for n in needles)


def extract_aps_message(body: Any) -> str:
    """Pull a human Autodesk error string out of JSON or text."""
    if body is None:
        return ""
    if isinstance(body, dict):
        for key in (
            "developerMessage",
            "diagnostic",
            "reason",
            "error_description",
            "errorMessage",
            "message",
            "detail",
        ):
            val = body.get(key)
            if val:
                return str(val)
        err = body.get("error")
        if isinstance(err, str) and err:
            return err
        if isinstance(err, dict):
            return extract_aps_message(err)
        msgs = body.get("messages")
        if isinstance(msgs, list) and msgs:
            parts = [extract_aps_message(m) if isinstance(m, dict) else str(m) for m in msgs]
            return "; ".join(p for p in parts if p)
        return ""
    text = str(body).strip()
    if not text:
        return ""
    if text[:1] in "{[":
        try:
            import json

            return extract_aps_message(json.loads(text)) or text[:800]
        except Exception:
            return text[:800]
    return text[:800]


def format_aps_error(status_code: int, body: Any) -> str:
    raw = body if isinstance(body, str) else extract_aps_message(body)
    if not raw and isinstance(body, dict):
        raw = extract_aps_message(body)
    if not raw:
        raw = str(body)[:800] if body else "no details"
    if looks_like_insufficient_scope(status_code, raw) or (
        isinstance(body, str) and looks_like_insufficient_scope(status_code, body)
    ):
        return f"{raw}\n\n{INSUFFICIENT_SCOPE_HINT}"
    return f"Autodesk HTTP {status_code}: {raw}"


def manifest_error_text(manifest: dict[str, Any]) -> str:
    parts: list[str] = []
    reason = manifest.get("reason")
    if reason:
        parts.append(str(reason))
    for deriv in manifest.get("derivatives") or []:
        if not isinstance(deriv, dict):
            continue
        for msg in deriv.get("messages") or []:
            if isinstance(msg, dict):
                text = msg.get("message") or msg.get("code") or ""
                if isinstance(text, list):
                    text = " ".join(str(x) for x in text)
                if text:
                    parts.append(str(text))
            elif msg:
                parts.append(str(msg))
    return "; ".join(parts) if parts else "translation failed"


def _http():
    import requests

    return requests


def _json_body(resp: Any) -> Any:
    try:
        return resp.json()
    except Exception:
        return getattr(resp, "text", "") or ""


def _raise_for_aps(resp: Any, *, context: str) -> None:
    code = int(getattr(resp, "status_code", 0) or 0)
    if 200 <= code < 300:
        return
    body = _json_body(resp)
    msg = format_aps_error(code, body if body != "" else getattr(resp, "text", ""))
    err = RuntimeError(f"{context}: {msg}")
    err.status_code = code  # type: ignore[attr-defined]
    raise err


def get_access_token(scope: str = VIEWER_SCOPES, *, http: Any = None) -> dict[str, Any]:
    """2-legged OAuth token (client_credentials)."""
    requests = http or _http()
    cid = _aps_setting("APS_CLIENT_ID")
    sec = _aps_setting("APS_CLIENT_SECRET")
    if not cid or not sec:
        raise RuntimeError("APS_CLIENT_ID / APS_CLIENT_SECRET not set")
    resp = requests.post(
        APS_AUTH_URL,
        data={"grant_type": "client_credentials", "scope": scope},
        auth=(cid, sec),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if int(getattr(resp, "status_code", 0) or 0) >= 400:
        body = _json_body(resp)
        blob = extract_aps_message(body) or getattr(resp, "text", "") or ""
        if looks_like_insufficient_scope(resp.status_code, blob) or looks_like_insufficient_scope(
            resp.status_code, str(body)
        ):
            raise RuntimeError(f"{blob}\n\n{INSUFFICIENT_SCOPE_HINT}")
        _raise_for_aps(resp, context="APS token")
    data = resp.json()
    if not data.get("access_token"):
        raise RuntimeError("APS token response missing access_token")
    return data


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _oss_object_url(bucket_key: str, object_key: str, suffix: str) -> str:
    return (
        f"{OSS_BUCKETS_URL}/{quote(bucket_key, safe='')}/objects/"
        f"{quote(object_key, safe='')}/{suffix}"
    )


def ensure_oss_bucket(token: str, bucket_key: str, *, http: Any = None) -> str:
    """Create-or-reuse an OSS bucket (persistent so a generated URN stays loadable)."""
    requests = http or _http()
    headers = {**_auth_headers(token), "Content-Type": "application/json"}
    details = requests.get(
        f"{OSS_BUCKETS_URL}/{quote(bucket_key, safe='')}/details",
        headers=_auth_headers(token),
        timeout=30,
    )
    if int(details.status_code) == 200:
        return "reused"
    resp = requests.post(
        OSS_BUCKETS_URL,
        json={"bucketKey": bucket_key, "policyKey": "persistent"},
        headers=headers,
        timeout=30,
    )
    code = int(resp.status_code)
    if code in (200, 201):
        return "created"
    body = _json_body(resp)
    text = extract_aps_message(body) or str(getattr(resp, "text", ""))
    if code == 409 or "already exists" in text.lower() or "already exist" in text.lower():
        return "reused"
    _raise_for_aps(resp, context="OSS create bucket")
    return "reused"


def _part_plan(n_bytes: int) -> tuple[int, int]:
    """(n_parts, part_size) so we stay within 25 signed URLs."""
    n = max(int(n_bytes or 0), 1)
    if n <= _MIN_PART_BYTES:
        return 1, n
    part_size = max(_MIN_PART_BYTES, math.ceil(n / _SIGNED_URL_BATCH))
    n_parts = max(1, math.ceil(n / part_size))
    if n_parts > _SIGNED_URL_BATCH:
        part_size = math.ceil(n / _SIGNED_URL_BATCH)
        n_parts = math.ceil(n / part_size)
    return n_parts, part_size


def upload_object_signed_s3(
    token: str,
    bucket_key: str,
    object_key: str,
    data: bytes,
    *,
    http: Any = None,
    on_status: Optional[OnStatus] = None,
) -> dict[str, Any]:
    """OSS v2 direct-to-S3: GET signed URLs → PUT bytes → POST complete."""
    requests = http or _http()
    n_parts, part_size = _part_plan(len(data))
    if on_status:
        on_status("uploading", f"Requesting {n_parts} signed S3 URL(s)…")
    signed = requests.get(
        _oss_object_url(bucket_key, object_key, "signeds3upload"),
        headers=_auth_headers(token),
        params={"firstPart": 1, "parts": n_parts, "minutesExpiration": 30},
        timeout=30,
    )
    _raise_for_aps(signed, context="OSS signed upload URLs")
    payload = signed.json()
    urls = list(payload.get("urls") or [])
    upload_key = payload.get("uploadKey")
    if not urls or not upload_key:
        raise RuntimeError("OSS signed upload response missing urls/uploadKey")
    if len(urls) < n_parts:
        # Request remaining batches (firstPart is 1-based).
        while len(urls) < n_parts:
            nxt = len(urls) + 1
            batch = min(_SIGNED_URL_BATCH, n_parts - len(urls))
            more = requests.get(
                _oss_object_url(bucket_key, object_key, "signeds3upload"),
                headers=_auth_headers(token),
                params={
                    "firstPart": nxt,
                    "parts": batch,
                    "uploadKey": upload_key,
                    "minutesExpiration": 30,
                },
                timeout=30,
            )
            _raise_for_aps(more, context="OSS signed upload URLs (next parts)")
            extra = list((more.json() or {}).get("urls") or [])
            if not extra:
                break
            urls.extend(extra)
    if len(urls) < n_parts:
        raise RuntimeError(f"OSS returned {len(urls)} signed URLs, need {n_parts}")

    for i, url in enumerate(urls):
        start = i * part_size
        chunk = data[start : start + part_size]
        if not chunk:
            break
        if on_status and n_parts > 1:
            on_status("uploading", f"Uploading part {i + 1}/{n_parts}…")
        put = requests.put(url, data=chunk, timeout=180)
        code = int(put.status_code)
        if code >= 400:
            raise RuntimeError(
                f"S3 upload part {i + 1} failed (HTTP {code}): {(getattr(put, 'text', '') or '')[:400]}"
            )

    if on_status:
        on_status("uploading", "Completing OSS upload…")
    done = requests.post(
        _oss_object_url(bucket_key, object_key, "signeds3upload"),
        headers={**_auth_headers(token), "Content-Type": "application/json"},
        json={"uploadKey": upload_key},
        timeout=60,
    )
    _raise_for_aps(done, context="OSS complete upload")
    return done.json() if getattr(done, "content", None) else {}


def start_svf_job(
    token: str,
    model_urn: str,
    *,
    compressed: bool = False,
    root_filename: str = "",
    http: Any = None,
) -> dict[str, Any]:
    """Kick Model Derivative SVF2 (falls back to SVF if SVF2 is rejected)."""
    requests = http or _http()
    input_spec: dict[str, Any] = {"urn": model_urn}
    root = (root_filename or "").strip()
    # Never send compressedUrn without a non-empty rootFilename (Autodesk 400).
    if compressed or root:
        if not root:
            raise ValueError(ZIP_ROOT_MISSING_MSG)
        input_spec["compressedUrn"] = True
        input_spec["rootFilename"] = root
    if input_spec.get("compressedUrn") and not str(input_spec.get("rootFilename") or "").strip():
        raise ValueError(ZIP_ROOT_MISSING_MSG)
    headers = {
        **_auth_headers(token),
        "Content-Type": "application/json; charset=utf-8",
    }

    def _post(fmt: str) -> Any:
        return requests.post(
            MD_JOB_URL,
            headers=headers,
            json={
                "input": input_spec,
                "output": {"formats": [{"type": fmt, "views": ["2d", "3d"]}]},
            },
            timeout=60,
        )

    resp = _post("svf2")
    code = int(resp.status_code)
    if code >= 400:
        body = extract_aps_message(_json_body(resp)) or getattr(resp, "text", "") or ""
        # Some seeds still want classic SVF.
        if "svf2" in body.lower() or code in (400, 406):
            resp = _post("svf")
            code = int(resp.status_code)
    if code >= 400:
        _raise_for_aps(resp, context="Model Derivative job")
    return resp.json() if getattr(resp, "content", None) else {}


def poll_manifest(
    token: str,
    model_urn: str,
    *,
    http: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    timeout_s: float = POLL_TIMEOUT_S,
    interval_s: float = POLL_INTERVAL_S,
    on_status: Optional[OnStatus] = None,
) -> dict[str, Any]:
    """Poll until success/failed/timeout. Returns the manifest plus ``phase``."""
    requests = http or _http()
    url = f"{APS_BASE}/modelderivative/v2/designdata/{quote(model_urn, safe='')}/manifest"
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    last: dict[str, Any] = {}
    while True:
        resp = requests.get(url, headers=_auth_headers(token), timeout=30)
        code = int(resp.status_code)
        if code == 404:
            last = {"status": "pending", "progress": "0%"}
        elif code >= 400:
            _raise_for_aps(resp, context="Model Derivative manifest")
        else:
            last = resp.json() or {}
        status = str(last.get("status") or "pending").lower()
        progress = str(last.get("progress") or "")
        if on_status:
            on_status("translating", f"{status}" + (f" · {progress}" if progress else ""))
        if status == "success":
            last["phase"] = "success"
            return last
        if status in ("failed", "timeout"):
            last["phase"] = "failed"
            last["error"] = manifest_error_text(last)
            return last
        if time.monotonic() >= deadline:
            last["phase"] = "timeout"
            return last
        sleep(max(0.2, float(interval_s)))


def translate_cad_bytes(
    filename: str,
    data: bytes,
    *,
    root_filename: str = "",
    http: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    on_status: Optional[OnStatus] = None,
    poll_timeout_s: float = POLL_TIMEOUT_S,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> dict[str, Any]:
    """Upload CAD to OSS and translate to SVF/SVF2. Never logs secrets.

    Returns a dict: ok, phase, urn, message, bucket, object_key.
    Does not call Autodesk unless this function is invoked (tests mock ``http``).
    """
    kind, size_msg = cad_size_issue(len(data or b""))
    if kind == "error":
        return {
            "ok": False,
            "phase": "error",
            "urn": "",
            "message": size_msg,
            "bucket": "",
            "object_key": "",
        }

    root = resolve_zip_root_filename(filename, data, root_filename)
    if is_cad_zip_name(filename) and not root:
        return {
            "ok": False,
            "phase": "error",
            "urn": "",
            "message": ZIP_ROOT_MISSING_MSG,
            "bucket": "",
            "object_key": "",
        }

    def _note(phase: str, detail: str) -> None:
        if on_status:
            on_status(phase, detail)

    cid = _aps_setting("APS_CLIENT_ID")
    bucket = oss_bucket_key(cid)
    object_key = sanitize_object_name(filename)
    compressed = bool(root)
    try:
        _note("auth", "Requesting APS token (translate scopes)…")
        token_payload = get_access_token(TRANSLATE_SCOPES, http=http)
        token = token_payload["access_token"]
        _note("bucket", f"Using OSS bucket `{bucket}`…")
        ensure_oss_bucket(token, bucket, http=http)
        _note("uploading", f"Uploading `{object_key}` ({len(data):,} bytes)…")
        complete = upload_object_signed_s3(
            token, bucket, object_key, data, http=http, on_status=on_status
        )
        object_id = str(complete.get("objectId") or oss_object_id(bucket, object_key))
        urn = encode_model_urn(object_id)
        _note("translating", "Starting Model Derivative SVF2 job…")
        start_svf_job(
            token,
            urn,
            compressed=compressed,
            root_filename=root,
            http=http,
        )
        manifest = poll_manifest(
            token,
            urn,
            http=http,
            sleep=sleep,
            timeout_s=poll_timeout_s,
            interval_s=poll_interval_s,
            on_status=on_status,
        )
        phase = str(manifest.get("phase") or "")
        if phase == "success":
            return {
                "ok": True,
                "phase": "success",
                "urn": urn,
                "message": "Translation succeeded. URN is in the field below — click Load CAD model.",
                "bucket": bucket,
                "object_key": object_key,
            }
        if phase == "timeout":
            return {
                "ok": False,
                "phase": "timeout",
                "urn": urn,
                "message": (
                    "Upload succeeded and Autodesk is still translating (timed out after "
                    f"{int(poll_timeout_s)}s). The URN is filled in — wait a minute and click "
                    "Load CAD model, or check the job in APS."
                ),
                "bucket": bucket,
                "object_key": object_key,
            }
        err = manifest.get("error") or manifest_error_text(manifest)
        return {
            "ok": False,
            "phase": "failed",
            "urn": urn,
            "message": f"Translation failed: {err}",
            "bucket": bucket,
            "object_key": object_key,
        }
    except Exception as exc:
        text = str(exc)
        phase = "error"
        if looks_like_insufficient_scope(0, text) or INSUFFICIENT_SCOPE_HINT in text:
            if INSUFFICIENT_SCOPE_HINT not in text:
                text = f"{text}\n\n{INSUFFICIENT_SCOPE_HINT}"
        return {
            "ok": False,
            "phase": phase,
            "urn": "",
            "message": text,
            "bucket": bucket,
            "object_key": object_key,
        }


_VIEWER_TEMPLATE = """
<style>
  html, body { margin:0; padding:0; height:100%; background:#0b1016; }
  #aps-wrap { position:relative; width:100%; height:__HEIGHT__px; min-height:__HEIGHT__px;
    border-radius:12px; background:#0b1016; }
  #apsViewer, #apsViewer .adsk-viewing-viewer { width:100%; height:100% !important; }
  #aps-badge { position:absolute; top:10px; left:12px; z-index:3; font-family:system-ui,sans-serif;
    color:#fff; background:rgba(10,16,22,.55); padding:6px 10px; border-radius:8px; pointer-events:none;
    font-size:13px; max-width:calc(100% - 24px); }
  #aps-region { opacity:.78; }
  #aps-err { display:none; position:absolute; inset:0; z-index:8; color:#fff;
    font-family:system-ui,sans-serif; padding:20px; font-size:13px; background:#101822; line-height:1.45; }
</style>
<link rel="stylesheet"
  href="https://developer.api.autodesk.com/modelderivative/v2/viewers/7.*/style.min.css" type="text/css">
<div id="aps-wrap">
  <div id="aps-badge">
    <b id="aps-name">__ASSET__</b> · risk
    <b id="aps-risk" style="color:__RISK_HEX__">__RISK__</b>
    <span id="aps-part" style="display:none; opacity:.9"> · <span id="aps-part-name"></span></span>
    <span id="aps-region"></span>
  </div>
  <div id="apsViewer"></div>
  <div id="aps-err"></div>
</div>
<script src="https://developer.api.autodesk.com/modelderivative/v2/viewers/7.*/viewer3D.min.js"></script>
<script>
(function(){
  var TOKEN="__TOKEN__", URN="__URN__", RISK="__RISK__", PUBLIC=__PUBLIC__;
  var ASSET="__ASSET__";
  var EXTRAS=__EXTRAS__;
  var VIEWER_ERR=__VIEWER_ERR__;
  var BRIDGE=__BRIDGE__;
  var HEIGHT=__HEIGHT__;
  var RISK_RGBA=__RISK_RGBA__;
  var NEEDLES=__NEEDLES__;
  // dbIds the user assigned to the driver region on the Click-to-assign panel.
  var ASSIGNED=__ASSIGNED__;
  // Fallback name search can match property values, not just nodes. Never let it
  // paint the whole engine: drop the result when it covers more than this share.
  var SEARCH_MAX_SHARE=0.35, SEARCH_MIN_CAP=12, SEARCH_TIMEOUT_MS=4000;
  var viewer=null, lastUrn=null, lastToken=null, lastRisk=null, selectionWired=false;
  var lastRegionKey=null;
  function fail(m){var e=document.getElementById('aps-err');e.style.display='block';
    e.innerHTML='<b>APS Viewer error.</b><br>'+m;}
  function errCode(err){
    if (err == null) return null;
    if (typeof err === 'number' && isFinite(err)) return err;
    if (typeof err === 'string' && /^-?\\d+$/.test(err.trim())) return Number(err.trim());
    if (typeof err === 'object'){
      var v = err.code != null ? err.code : err.errorCode;
      if (v != null && v !== ''){
        var n = Number(v);
        if (isFinite(n)) return n;
      }
    }
    return null;
  }
  function describeErr(err){
    var n = errCode(err);
    if (n === 4){
      return 'Access denied (error 4 / NETWORK_ACCESS_DENIED). Your 2-legged APS token cannot '
        + 'read this object. Autodesk public Viewer URNs (a360viewer-protected) are not in your bucket.';
    }
    if (n != null && VIEWER_ERR[String(n)]) return VIEWER_ERR[String(n)] + ' (error ' + n + ')';
    try { return typeof err === 'string' ? err : JSON.stringify(err); } catch (e) { return String(err); }
  }
  function publicHint(){
    return PUBLIC
      ? '<br><br><b>This URN is from Autodesk&apos;s public Viewer website. Your APS app cannot open it.</b> '
        + 'Use Choose CAD file → Translate with the same STEP (Rotax 912), then Load, then Save URN.'
      : '';
  }
  function sendToStreamlit(type, data){
    try {
      var msg = { isStreamlitMessage: true, type: type };
      if (data) { for (var k in data) { if (Object.prototype.hasOwnProperty.call(data, k)) msg[k] = data[k]; } }
      window.parent.postMessage(msg, '*');
    } catch (e) {}
  }
  function emitSelection(payload){
    try { window.parent.postMessage({ type: 'pdm-cad-part', payload: payload }, '*'); } catch (e) {}
    if (BRIDGE) sendToStreamlit('streamlit:setComponentValue', { value: payload });
  }
  // Autodesk hands back empty / control-byte / mojibake names for non-UTF8 STEP
  // exports. Mirror src/cad_part.sanitize_node_name so a CEO never sees ??????.
  function sanitizeName(name, dbId){
    var raw = (name == null ? '' : String(name));
    var printable = '';
    for (var i = 0; i < raw.length; i++){
      var code = raw.charCodeAt(i);
      if (code < 32 || (code >= 127 && code <= 159)) continue;
      printable += raw.charAt(i);
    }
    printable = printable.replace(/\\s+/g, ' ').replace(/^\\s+|\\s+$/g, '');
    var bad = /[?\\uFFFD\\uFFFE\\uFFFF]/;
    if (printable && !bad.test(printable)) return printable;
    var pair = /(\\d+)\\s*[-_ ]\\s*solid\\s*(\\d+)/i.exec(printable);
    if (pair) return 'Solid-' + parseInt(pair[1], 10);
    var single = /solid\\s*[-_ ]?\\s*(\\d+)/i.exec(printable);
    if (single) return 'Solid-' + parseInt(single[1], 10);
    var stripped = printable.replace(/[?\\uFFFD\\uFFFE\\uFFFF]+/g, ' ')
      .replace(/\\s+/g, ' ').replace(/^\\s+|\\s+$/g, '');
    var bare = /^[\\s\\-_.]*(\\d+)[\\s\\-_.]*$/.exec(stripped);
    if (bare) return 'Solid-' + parseInt(bare[1], 10);
    var trimmed = stripped.replace(/^[-_.,;:\\/\\\\ ]+/, '').replace(/[-_.,;:\\/\\\\ ]+$/, '');
    if (trimmed && /[A-Za-z]/.test(trimmed)) return trimmed;
    var n = parseInt(dbId, 10);
    return isFinite(n) ? ('Solid-' + n) : 'Solid';
  }
  function setBadge(asset, risk, partName){
    var nameEl = document.getElementById('aps-name');
    var riskEl = document.getElementById('aps-risk');
    var partWrap = document.getElementById('aps-part');
    var partEl = document.getElementById('aps-part-name');
    if (nameEl) nameEl.textContent = asset || 'asset';
    if (riskEl){
      riskEl.textContent = risk || 'Unknown';
      var hex = {High:'#e74c3c', Medium:'#f39c12', Low:'#27ae60', Unknown:'#7f8c8d'};
      riskEl.style.color = hex[risk] || hex.Unknown;
    }
    if (partWrap && partEl){
      if (partName){ partEl.textContent = partName; partWrap.style.display = 'inline'; }
      else { partWrap.style.display = 'none'; partEl.textContent = ''; }
    }
  }
  function setRegionBadge(matched, source){
    var el = document.getElementById('aps-region');
    if (!el) return;
    if (source === 'off'){ el.textContent = ' · default color (no risk tint)'; return; }
    if (matched > 0){
      var what = (source === 'assigned') ? 'assigned part' : 'matched node';
      el.textContent = ' · ' + matched + ' ' + what + (matched === 1 ? '' : 's') + ' tinted';
      return;
    }
    el.textContent = ' · no CAD region matched — default color';
  }
  function regionVec(risk){
    var row = RISK_RGBA && RISK_RGBA[risk];
    if (!row || !row.length || typeof THREE === 'undefined') return null;
    return new THREE.Vector4(row[0], row[1], row[2], row[3]);
  }
  function normPart(s){
    return String(s == null ? '' : s).toLowerCase().replace(/[^a-z0-9]+/g, '');
  }
  function needleFor(name){
    var blob = normPart(name);
    if (!blob) return '';
    for (var i = 0; i < NEEDLES.length; i++){
      if (NEEDLES[i] && blob.indexOf(NEEDLES[i]) >= 0) return NEEDLES[i];
    }
    return '';
  }
  // Deliberately no timestamp, and skipped when unchanged: Streamlit reruns the
  // script whenever a component value changes, and every rerun re-renders this
  // iframe, which would re-apply theming and post again — an endless loop.
  function reportRegion(matched, nodes, names, source){
    var payload = {
      kind: 'region',
      matched: matched,
      nodes: nodes,
      names: (names || []).slice(0, 8),
      risk: RISK,
      source: source
    };
    var key = '';
    try { key = JSON.stringify(payload); } catch (e) { key = String(matched) + source; }
    if (key && key === lastRegionKey) return;
    lastRegionKey = key;
    emitSelection(payload);
  }
  function walkRegion(v){
    var res = { ids: [], names: [], nodes: 0 };
    var tree = null;
    try { tree = v.model && v.model.getInstanceTree && v.model.getInstanceTree(); } catch (e) {}
    if (!tree) return res;
    var seen = {};
    function visit(id){
      if (id == null || seen[id]) return;
      seen[id] = 1;
      res.nodes++;
      var nm = '';
      try { nm = tree.getNodeName(id) || ''; } catch (e) {}
      if (nm && needleFor(nm)){ res.ids.push(id); res.names.push(nm); }
    }
    var root = null;
    try { root = tree.getRootId(); } catch (e) {}
    if (root == null) return res;
    visit(root);
    try { tree.enumNodeChildren(root, visit, true); } catch (e) {}
    return res;
  }
  function searchRegion(v, done){
    var wanted = (NEEDLES || []).slice(0, 12);
    if (!wanted.length || !v || typeof v.search !== 'function'){ done([]); return; }
    var ids = [], pending = wanted.length, finished = false;
    function finish(){ if (finished) return; finished = true; done(ids); }
    function step(res){
      (res || []).forEach(function(id){ ids.push(id); });
      pending -= 1;
      if (pending <= 0) finish();
    }
    wanted.forEach(function(needle){
      try { v.search(needle, step, function(){ step([]); }); } catch (e) { step([]); }
    });
    setTimeout(finish, SEARCH_TIMEOUT_MS);
  }
  function paintRegion(v, color, ids){
    var painted = 0, seen = {};
    (ids || []).forEach(function(id){
      if (id == null || seen[id]) return;
      seen[id] = 1;
      try { v.setThemingColor(id, color, v.model, true); painted++; }
      catch (e1) { try { v.setThemingColor(id, color); painted++; } catch (e2) {} }
    });
    if (v.impl) v.impl.invalidate(true);
    return painted;
  }
  // Region tint, never whole-model. High → matched nodes red, Medium → orange,
  // Low / Unknown → nothing. Zero matches keeps the default dark gray engine.
  function applyRegionTheming(v, risk){
    try {
      if (!v || !v.model) return;
      if (typeof v.clearThemingColors === 'function') v.clearThemingColors(v.model);
      lastRisk = risk;
      var color = regionVec(risk);
      if (!color){
        if (v.impl) v.impl.invalidate(true);
        setRegionBadge(0, 'off');
        reportRegion(0, 0, [], 'off');
        return;
      }
      var walk = walkRegion(v);
      // Click-to-assign wins: High + driver region paints the saved dbIds, even
      // when Autodesk names are Solid1. Name matching is the fallback.
      if (ASSIGNED && ASSIGNED.length){
        var paintedA = paintRegion(v, color, ASSIGNED);
        setRegionBadge(paintedA, 'assigned');
        reportRegion(paintedA, walk.nodes, [], 'assigned');
        return;
      }
      if (walk.ids.length){
        var painted = paintRegion(v, color, walk.ids);
        setRegionBadge(painted, 'tree');
        reportRegion(painted, walk.nodes, walk.names, 'tree');
        return;
      }
      searchRegion(v, function(found){
        var cap = Math.max(SEARCH_MIN_CAP, Math.floor(walk.nodes * SEARCH_MAX_SHARE));
        if (!found.length || found.length > cap){
          if (v.impl) v.impl.invalidate(true);
          setRegionBadge(0, 'none');
          reportRegion(0, walk.nodes, [], found.length ? 'search_too_broad' : 'none');
          return;
        }
        var painted2 = paintRegion(v, color, found);
        setRegionBadge(painted2, 'search');
        reportRegion(painted2, walk.nodes, [], 'search');
      });
    } catch (e) {}
  }
  function wireSelection(v){
    if (selectionWired) return;
    selectionWired = true;
    function onSel(event){
      var ids = [];
      if (event && event.dbIdArray && event.dbIdArray.length) ids = event.dbIdArray;
      else if (event && event.selections && event.selections[0] && event.selections[0].dbIdArray)
        ids = event.selections[0].dbIdArray;
      else {
        try { ids = (v.getSelection && v.getSelection()) || []; } catch (e) {}
      }
      if (!ids.length){
        setBadge(ASSET, RISK, '');
        emitSelection({ kind: 'part', name: '', dbId: null, properties: {}, cleared: true, ts: Date.now() });
        return;
      }
      var dbId = ids[0];
      var nodeName = '';
      try {
        var tree = v.model && v.model.getInstanceTree && v.model.getInstanceTree();
        if (tree && tree.getNodeName) nodeName = tree.getNodeName(dbId) || '';
      } catch (e) {}
      function finish(nm, props){
        var raw = nm || nodeName;
        var shown = sanitizeName(raw, dbId);
        setBadge(ASSET, RISK, shown);
        emitSelection({
          kind: 'part',
          name: raw || shown,
          display: shown,
          dbId: dbId,
          properties: props || {},
          cleared: false,
          ts: Date.now()
        });
      }
      try {
        v.getProperties(dbId, function(result){
          var props = {};
          var nm = (result && result.name) || nodeName;
          (result && result.properties || []).forEach(function(p){
            var k = p.displayName || p.attributeName || '';
            var val = p.displayValue;
            if (k && val != null && val !== '') props[k] = String(val);
          });
          finish(nm, props);
        }, function(){ finish(nodeName, {}); });
      } catch (e) { finish(nodeName, {}); }
    }
    try { v.addEventListener(Autodesk.Viewing.SELECTION_CHANGED_EVENT, onSel); } catch (e) {}
    try {
      if (Autodesk.Viewing.AGGREGATE_SELECTION_CHANGED_EVENT)
        v.addEventListener(Autodesk.Viewing.AGGREGATE_SELECTION_CHANGED_EVENT, onSel);
    } catch (e) {}
  }
  function fitUi(v){
    try { v.resize(); } catch (e) {}
    try { if (v.showViewCube) v.showViewCube(true); } catch (e) {}
  }
  function loadExtras(v, extras){
    (extras || []).forEach(function(name){
      try { v.loadExtension(name).catch(function(){}); } catch (e) {}
    });
  }
  function loadModel(v, urn, risk, extras){
    Autodesk.Viewing.Document.load('urn:'+urn, function(doc){
      var node = doc.getRoot().getDefaultGeometry();
      if (!node) { fail('No viewable geometry in this URN.'+publicHint()); return; }
      v.loadDocumentNode(doc, node).then(function(){
        fitUi(v);
        loadExtras(v, extras);
        try { v.getToolbar(true); } catch (e) {}
        wireSelection(v);
        function tint(){ applyRegionTheming(v, risk); }
        try { v.addEventListener(Autodesk.Viewing.GEOMETRY_LOADED_EVENT, tint); } catch (e) {}
        tint();
      }).catch(function(err){ fail('loadDocumentNode failed: '+describeErr(err)+publicHint()); });
    }, function(err){ fail('Model load failed: '+describeErr(err)+publicHint()); });
  }
  function startViewer(args){
    args = args || {};
    if (args.token) TOKEN = args.token;
    if (args.urn) URN = args.urn;
    if (args.asset) ASSET = args.asset;
    if (args.risk) RISK = args.risk;
    if (args.public != null) PUBLIC = !!args.public;
    if (args.public_viewer != null) PUBLIC = !!args.public_viewer;
    if (args.extras) EXTRAS = args.extras;
    if (args.viewer_err) VIEWER_ERR = args.viewer_err;
    if (args.height) HEIGHT = args.height;
    if (args.needles && args.needles.length) NEEDLES = args.needles;
    if (args.region_ids != null) ASSIGNED = args.region_ids;
    setBadge(ASSET, RISK, '');
    if (PUBLIC){
      fail('<b>This URN is from Autodesk&apos;s public Viewer website. Your APS app cannot open it.</b> '
        + 'Use Choose CAD file → Translate with the same STEP (Rotax 912), then Load, then Save URN.');
      return;
    }
    if (!TOKEN || !URN){
      if (!BRIDGE) fail('Missing APS token or URN.');
      return;
    }
    if (viewer && lastUrn === URN){
      if (TOKEN && TOKEN !== lastToken){
        try { Autodesk.Viewing.token.accessToken = TOKEN; } catch (e) {}
        lastToken = TOKEN;
      }
      applyRegionTheming(viewer, RISK);
      return;
    }
    if (typeof Autodesk === 'undefined'){
      fail('APS Viewer script failed to load (needs internet to Autodesk).');
      return;
    }
    lastUrn = URN;
    lastToken = TOKEN;
    Autodesk.Viewing.Initializer({
      env: 'AutodeskProduction',
      api: 'derivativeV2',
      accessToken: TOKEN,
      useADP: false
    }, function(){
      var el = document.getElementById('apsViewer');
      if (viewer){
        try { viewer.tearDown(); } catch (e) {}
        try { viewer.finish(); } catch (e) {}
        selectionWired = false;
        lastRegionKey = null;
        if (el) el.innerHTML = '';
      }
      viewer = new Autodesk.Viewing.GuiViewer3D(el, {
        theme: 'dark-theme',
        disabledExtensions: {
          measure: false,
          section: false,
          explode: false,
          viewcube: false,
          bimwalk: false,
          fusionOrbit: false,
          hyperlink: false,
          layerManager: false,
          modelBrowser: false,
          propertiesPanel: false
        },
        extensions: EXTRAS
      });
      var started = viewer.start();
      if (started === false) {
        fail('GuiViewer3D failed to start (container has no size).');
        return;
      }
      window.addEventListener('resize', function(){ fitUi(viewer); });
      setTimeout(function(){ fitUi(viewer); }, 250);
      loadModel(viewer, URN, RISK, EXTRAS);
    });
  }
  function whenAutodesk(cb, n){
    n = n || 0;
    if (typeof Autodesk !== 'undefined') { cb(); return; }
    if (n > 80) { fail('APS Viewer script failed to load (needs internet to Autodesk).'); return; }
    setTimeout(function(){ whenAutodesk(cb, n + 1); }, 100);
  }
  if (BRIDGE){
    window.addEventListener('message', function(ev){
      var data = ev.data || {};
      if (data.type === 'streamlit:render'){
        var a = data.args || {};
        whenAutodesk(function(){ startViewer(a); });
      }
    });
    sendToStreamlit('streamlit:componentReady', { apiVersion: 1 });
    sendToStreamlit('streamlit:setFrameHeight', { height: HEIGHT + 16 });
  } else if (PUBLIC){
    fail('<b>This URN is from Autodesk&apos;s public Viewer website. Your APS app cannot open it.</b> '
      + 'Use Choose CAD file → Translate with the same STEP (Rotax 912), then Load, then Save URN.');
  } else {
    whenAutodesk(function(){ startViewer({}); });
  }
})();
</script>
"""

_RISK_HEX = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60", "Unknown": "#7f8c8d"}

# The viewer sits directly under the Part · Asset · Risk strip, so it has to fit
# a laptop viewport with the mapped-sensor grid one short scroll below it.
DEFAULT_VIEWER_HEIGHT = 660
CAD_VIEWER_COMPONENT_DIR = Path(__file__).resolve().parent / "cad_viewer_component"


def _region_rgba_js() -> str:
    payload = {k: list(v) for k, v in REGION_THEME_RGBA.items()}
    return json.dumps(payload)


def coerce_db_ids(raw: Optional[Iterable[Any]]) -> list[int]:
    """Clean a dbId list for the viewer: ints only, de-duplicated, order kept."""
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


def build_viewer_html(
    token: str,
    urn: str,
    *,
    asset: str,
    risk: str,
    height: int = DEFAULT_VIEWER_HEIGHT,
    public_viewer: bool = False,
    bridge: bool = False,
    needles: Optional[list[str]] = None,
    region_ids: Optional[Iterable[Any]] = None,
) -> str:
    """Embed GuiViewer3D (full toolbar), not a headless Viewer3D.

    ``needles`` are the normalized node-name keywords from
    ``src.cad_part.region_hint_keywords``. ``region_ids`` are the dbIds the user
    click-assigned to the driver's region — those win over name matching so a
    ``Solid1`` STEP can still show a red heads/exhaust region.
    ``bridge=True`` waits for Streamlit ``streamlit:render`` args and posts click
    / region events via ``streamlit:setComponentValue`` (custom component
    iframe). ``components.html`` / dashboard export use ``bridge=False``.
    """
    risk = risk if risk in _RISK_HEX else "Unknown"
    urn = normalize_model_urn(urn)
    extras_js = json.dumps(list(VIEWER_EXTRA_EXTENSIONS))
    err_js = json.dumps({str(k): v for k, v in VIEWER_ERROR_CODES.items()})
    hints = region_hint_keywords() if needles is None else [str(n) for n in needles if str(n)]
    assigned = coerce_db_ids(region_ids)
    safe_asset = str(asset).replace("\\", "\\\\").replace('"', '\\"')
    safe_token = str(token).replace("\\", "\\\\").replace('"', '\\"')
    return (
        _VIEWER_TEMPLATE.replace("__TOKEN__", safe_token)
        .replace("__URN__", urn)
        .replace("__ASSET__", safe_asset)
        .replace("__RISK_HEX__", _RISK_HEX[risk])
        .replace("__RISK_RGBA__", _region_rgba_js())
        .replace("__ASSIGNED__", json.dumps(assigned))
        .replace("__NEEDLES__", json.dumps(hints))
        .replace("__RISK__", risk)
        .replace("__HEIGHT__", str(int(height)))
        .replace("__PUBLIC__", "true" if public_viewer else "false")
        .replace("__EXTRAS__", extras_js)
        .replace("__VIEWER_ERR__", err_js)
        .replace("__BRIDGE__", "true" if bridge else "false")
    )


def viewer_bridge_html(*, height: int = DEFAULT_VIEWER_HEIGHT) -> str:
    """Static ``index.html`` for the Streamlit custom component (token/URN via RENDER)."""
    return build_viewer_html(
        "",
        "",
        asset="asset",
        risk="Unknown",
        height=height,
        public_viewer=False,
        bridge=True,
    )


def write_cad_viewer_component(directory: Optional[Path] = None) -> Path:
    """Write ``index.html`` so ``declare_component(path=...)`` can serve GuiViewer3D."""
    folder = Path(directory) if directory is not None else CAD_VIEWER_COMPONENT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "index.html"
    path.write_text(viewer_bridge_html(), encoding="utf-8")
    return path
