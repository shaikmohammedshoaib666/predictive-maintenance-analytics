"""Last industry pack (and optional CAD map pointer) for local durability.

Writes ``data/last_session.json`` when the disk is writable. Render free-tier
disk is wiped on redeploy — this file is honest local persistence, not a cloud DB.
Plant remains the default when no file exists.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.industry_packs import DEFAULT_PACK_ID, PACKS

_DEFAULT_NAME = "last_session.json"


def last_session_path() -> Path:
    override = (os.getenv("PDM_LAST_SESSION_PATH") or "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data" / _DEFAULT_NAME


def _empty() -> dict[str, Any]:
    return {"version": 1, "industry_pack": DEFAULT_PACK_ID, "updated_at": ""}


def load_last_session(*, path: Optional[Path] = None) -> dict[str, Any]:
    """Missing / corrupt file → empty defaults. Never raises."""
    p = Path(path) if path is not None else last_session_path()
    try:
        if not p.is_file():
            return _empty()
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return _empty()
    if not isinstance(data, dict):
        return _empty()
    pack = str(data.get("industry_pack") or "").strip()
    if pack not in PACKS:
        pack = DEFAULT_PACK_ID
    return {
        "version": int(data.get("version") or 1),
        "industry_pack": pack,
        "updated_at": str(data.get("updated_at") or ""),
        "cad_map_path": str(data.get("cad_map_path") or ""),
    }


def saved_industry_pack(*, path: Optional[Path] = None) -> str:
    """Pack id from disk, or Plant default when nothing is saved."""
    row = load_last_session(path=path)
    pack = str(row.get("industry_pack") or "")
    return pack if pack in PACKS else DEFAULT_PACK_ID


def restore_industry_pack(*, path: Optional[Path] = None) -> Optional[str]:
    """Return a saved pack only when a file actually exists (so Plant stays default)."""
    p = Path(path) if path is not None else last_session_path()
    if not p.is_file():
        return None
    pack = saved_industry_pack(path=p)
    return pack if pack in PACKS else None


def save_last_session(
    pack_id: Any,
    *,
    path: Optional[Path] = None,
    cad_map_path: str = "",
) -> dict[str, Any]:
    """Atomic write. Returns ok=False on permission errors instead of crashing the UI."""
    pid = str(pack_id or "").strip()
    if pid not in PACKS:
        pid = DEFAULT_PACK_ID
    p = Path(path) if path is not None else last_session_path()
    payload = {
        "version": 1,
        "industry_pack": pid,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cad_map_path": str(cad_map_path or ""),
    }
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        fd, tmp_name = tempfile.mkstemp(prefix="last_session.", suffix=".json", dir=str(p.parent))
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
        return {"ok": True, "saved": True, "path": str(p), "industry_pack": pid}
    except Exception as exc:
        return {"ok": False, "saved": False, "path": str(p), "industry_pack": pid, "error": str(exc)}
