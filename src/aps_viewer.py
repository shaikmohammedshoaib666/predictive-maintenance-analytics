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
import json
import math
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, unquote, urlparse, quote

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

# Render / Streamlit practical cap. Warn before we hit it.
MAX_CAD_BYTES = 200 * 1024 * 1024
WARN_CAD_BYTES = 80 * 1024 * 1024

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


def normalize_model_urn(raw: str) -> str:
    """Accept paste of base64, ``urn:``+base64, or a raw ``urn:adsk.objects:...`` id."""
    u = (raw or "").strip()
    if not u:
        return ""
    if u.lower().startswith("urn:adsk."):
        return encode_model_urn(u)
    if u.lower().startswith("urn:"):
        u = u[4:]
    return u.strip()


# Autodesk encoded URNs are URL-safe base64 of ``urn:...`` and therefore start with dXJu.
_DXJU_RE = re.compile(r"(dXJu[A-Za-z0-9_-]+)")
_ADSK_OBJECT_RE = re.compile(r"(urn:adsk\.[^\s\"'<>]+)", re.I)
_PUBLIC_VIEWER_HOSTS = ("viewer.autodesk.com", "autode.sk")

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


def looks_like_public_viewer(raw: str) -> bool:
    blob = (raw or "").lower()
    return any(host in blob for host in _PUBLIC_VIEWER_HOSTS)


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
        "warning": "",
        "raw": text,
    }
    if not text:
        return "", meta

    public = looks_like_public_viewer(text)
    meta["public_viewer"] = public
    if public:
        meta["warning"] = PUBLIC_VIEWER_WARNING

    candidates: list[tuple[str, str]] = []

    looks_url = "://" in text or text.lower().startswith("viewer.autodesk") or text.lower().startswith("autode.sk")
    if looks_url:
        url = text if "://" in text else "https://" + text
        try:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query, keep_blank_values=False)
            for key in ("urn", "id", "objectid", "objectId", "url", "model", "file"):
                val = _query_first(qs, key)
                if val:
                    candidates.append((val, "query"))
            for part in (parsed.path or "").split("/"):
                part = unquote(part).strip()
                if part:
                    candidates.append((part, "path"))
            frag = unquote(parsed.fragment or "").strip()
            if frag:
                if frag.startswith("?"):
                    fqs = parse_qs(frag[1:], keep_blank_values=False)
                    for key in ("urn", "id", "url"):
                        val = _query_first(fqs, key)
                        if val:
                            candidates.append((val, "fragment"))
                else:
                    candidates.append((frag, "fragment"))
        except Exception:
            pass

    candidates.append((text, "paste"))

    for cand, origin in candidates:
        if not cand:
            continue
        if cand.lower().startswith("urn:adsk."):
            urn = normalize_model_urn(cand)
            if urn:
                meta["source"] = "object_id" if origin == "paste" else origin
                return urn, meta
        dx = _DXJU_RE.search(cand)
        if dx:
            urn = normalize_model_urn(dx.group(1))
            if urn:
                meta["source"] = "viewer_url" if public else ("urn" if origin == "paste" else origin)
                return urn, meta
        adsk = _ADSK_OBJECT_RE.search(cand)
        if adsk:
            urn = normalize_model_urn(adsk.group(1))
            if urn:
                meta["source"] = "object_id"
                return urn, meta

    if public:
        meta["source"] = "viewer_url_no_urn"
        meta["warning"] = PUBLIC_VIEWER_NO_URN_WARNING
        return "", meta

    dx = _DXJU_RE.search(text)
    if dx:
        urn = normalize_model_urn(dx.group(1))
        if urn:
            meta["source"] = "urn"
            return urn, meta
    if text.lower().startswith("urn:adsk.") or text.startswith("dXJu"):
        meta["source"] = "urn"
        return normalize_model_urn(text), meta
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


def cad_size_issue(n_bytes: int) -> tuple[Optional[str], str]:
    """Return (\"error\"|\"warn\"|None, message) for Streamlit/Render size limits."""
    n = int(n_bytes or 0)
    if n <= 0:
        return "error", "CAD file is empty."
    if n > MAX_CAD_BYTES:
        mb = n / (1024 * 1024)
        return (
            "error",
            f"File is {mb:.0f} MB. Streamlit/Render uploads cap around 200 MB — "
            "use a smaller CAD or zip, or translate in APS and paste the URN.",
        )
    if n > WARN_CAD_BYTES:
        mb = n / (1024 * 1024)
        return (
            "warn",
            f"File is {mb:.0f} MB. Large uploads can time out on Render; translation may take a while.",
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
    if compressed or root_filename:
        input_spec["compressedUrn"] = True
        if root_filename:
            input_spec["rootFilename"] = root_filename
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

    def _note(phase: str, detail: str) -> None:
        if on_status:
            on_status(phase, detail)

    cid = _aps_setting("APS_CLIENT_ID")
    bucket = oss_bucket_key(cid)
    object_key = sanitize_object_name(filename)
    compressed = bool(root_filename) or (filename or "").lower().endswith(".zip")
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
            root_filename=(root_filename or "").strip(),
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
    color:#fff; background:rgba(10,16,22,.55); padding:6px 10px; border-radius:8px; pointer-events:none; }
  #aps-err { display:none; position:absolute; inset:0; z-index:8; color:#fff;
    font-family:system-ui,sans-serif; padding:20px; font-size:13px; background:#101822; line-height:1.45; }
</style>
<link rel="stylesheet"
  href="https://developer.api.autodesk.com/modelderivative/v2/viewers/7.*/style.min.css" type="text/css">
<div id="aps-wrap">
  <div id="aps-badge">
    <b id="aps-name">__ASSET__</b> · risk
    <b id="aps-risk" style="color:__RISK_HEX__">__RISK__</b>
  </div>
  <div id="apsViewer"></div>
  <div id="aps-err"></div>
</div>
<script src="https://developer.api.autodesk.com/modelderivative/v2/viewers/7.*/viewer3D.min.js"></script>
<script>
(function(){
  var TOKEN="__TOKEN__", URN="__URN__", RISK="__RISK__", PUBLIC=__PUBLIC__;
  var EXTRAS=__EXTRAS__;
  function fail(m){var e=document.getElementById('aps-err');e.style.display='block';
    e.innerHTML='<b>APS Viewer error.</b><br>'+m;}
  function publicHint(){
    return PUBLIC
      ? '<br><br>This URN came from viewer.autodesk.com. Your app token cannot read Autodesk&apos;s bucket. '
        + 'On CAD Twin: <b>Choose CAD file → Translate to SVF</b> so the URN is in YOUR OSS bucket, then we save that URN.'
      : '';
  }
  if (typeof Autodesk==='undefined'){
    fail('APS Viewer script failed to load (needs internet to Autodesk).');
    return;
  }
  Autodesk.Viewing.Initializer({
    env: 'AutodeskProduction',
    api: 'derivativeV2',
    accessToken: TOKEN,
    useADP: false
  }, function(){
    var el = document.getElementById('apsViewer');
    var viewer = new Autodesk.Viewing.GuiViewer3D(el, {
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
    function fitUi(){
      try { viewer.resize(); } catch (e) {}
      try { if (viewer.showViewCube) viewer.showViewCube(true); } catch (e) {}
    }
    window.addEventListener('resize', fitUi);
    setTimeout(fitUi, 250);
    function loadExtras(){
      EXTRAS.forEach(function(name){
        try {
          viewer.loadExtension(name).catch(function(){ /* optional: FirstPerson / MarkupsGui */ });
        } catch (e) {}
      });
    }
    Autodesk.Viewing.Document.load('urn:'+URN, function(doc){
      var node = doc.getRoot().getDefaultGeometry();
      if (!node) { fail('No viewable geometry in this URN.'+publicHint()); return; }
      viewer.loadDocumentNode(doc, node).then(function(){
        fitUi();
        loadExtras();
        try { viewer.getToolbar(true); } catch (e) {}
        if (RISK==='High' || RISK==='Medium'){
          viewer.addEventListener(Autodesk.Viewing.GEOMETRY_LOADED_EVENT, function(){
            try{
              var color = RISK==='High' ? new THREE.Vector4(0.9,0.1,0.1,0.7)
                                        : new THREE.Vector4(0.95,0.6,0.1,0.55);
              var tree = viewer.model.getInstanceTree();
              var ids=[]; tree.enumNodeChildren(tree.getRootId(), function(id){ids.push(id);}, true);
              var on=true;
              setInterval(function(){
                ids.forEach(function(id){ on ? viewer.setThemingColor(id,color) : viewer.clearThemingColor(id); });
                viewer.impl.invalidate(true); on=!on;
              }, RISK==='High'?450:900);
            }catch(e){/* theming best-effort */}
          });
        }
      }).catch(function(err){ fail('loadDocumentNode failed: '+JSON.stringify(err)+publicHint()); });
    }, function(err){ fail('Model load failed: '+JSON.stringify(err)+publicHint()); });
  });
})();
</script>
"""

_RISK_HEX = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60", "Unknown": "#7f8c8d"}

DEFAULT_VIEWER_HEIGHT = 840


def build_viewer_html(
    token: str,
    urn: str,
    *,
    asset: str,
    risk: str,
    height: int = DEFAULT_VIEWER_HEIGHT,
    public_viewer: bool = False,
) -> str:
    """Embed GuiViewer3D (full toolbar), not a headless Viewer3D."""
    risk = risk if risk in _RISK_HEX else "Unknown"
    urn = normalize_model_urn(urn)
    extras_js = json.dumps(list(VIEWER_EXTRA_EXTENSIONS))
    return (
        _VIEWER_TEMPLATE.replace("__TOKEN__", token)
        .replace("__URN__", urn)
        .replace("__ASSET__", str(asset))
        .replace("__RISK__", risk)
        .replace("__RISK_HEX__", _RISK_HEX[risk])
        .replace("__HEIGHT__", str(int(height)))
        .replace("__PUBLIC__", "true" if public_viewer else "false")
        .replace("__EXTRAS__", extras_js)
    )
