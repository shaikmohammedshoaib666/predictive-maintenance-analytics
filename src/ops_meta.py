"""Ops honesty — revision, sample vs live, no fake uptime.

Reads git SHA or Render/env revision. Never invents a host uptime number.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from typing import Any, Optional


def _setting(key: str, default: str = "") -> str:
    try:
        import config as _cfg

        return str(_cfg._setting(key, default) or "").strip()
    except Exception:
        return (os.getenv(key) or default or "").strip()


def is_render() -> bool:
    return bool(os.getenv("RENDER") or os.getenv("RENDER_SERVICE_ID") or os.getenv("RENDER_EXTERNAL_URL"))


@lru_cache(maxsize=1)
def app_revision(*, short: bool = True) -> str:
    """RENDER_GIT_COMMIT / APP_REVISION / git SHA. Empty when none of those exist."""
    env = _setting("APP_REVISION") or _setting("RENDER_GIT_COMMIT") or _setting("RENDER_GIT_COMMIT_SHA")
    if env:
        sha = env.strip()
        return sha[:7] if short and len(sha) > 7 and all(c in "0123456789abcdefABCDEF" for c in sha[:7]) else (
            sha[:7] if short else sha
        )
    try:
        args = ["git", "rev-parse"]
        if short:
            args.append("--short")
        args.append("HEAD")
        out = subprocess.check_output(
            args,
            cwd=os.path.dirname(os.path.dirname(__file__)),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def data_source_label(
    *,
    data_source: str = "",
    live_running: bool = False,
    live_source: str = "",
    broker_host: str = "",
) -> str:
    """Honest sample vs live. Never implies hangar telemetry on localhost/Render demo."""
    if live_running and str(live_source or "") in {"mqtt", "opcua"}:
        host = str(broker_host or "").strip().lower()
        if is_loopback_host(host):
            where = "Render VM" if is_render() else "this box"
            return f"live {live_source} on {where} (localhost — not hangar telemetry)"
        if host:
            return f"live {live_source} broker {broker_host} (configured — still not certified flight)"
        return f"live {live_source}"
    if live_running and str(live_source or "") == "sim":
        return "sample / simulator (not live hangar telemetry)"
    if live_running and str(live_source or "") == "poll":
        return "live CSV poll (not hangar MQTT)"
    src = str(data_source or "").strip().lower()
    if src in {"sample", "demo", "pack_demo"}:
        return "sample CSV (not live hangar telemetry)"
    if src in {"upload", "url", "cloud"}:
        return "uploaded / URL table (session data, not a live broker)"
    if src == "live":
        return "live buffer"
    return "session data"


def is_loopback_host(host: Any) -> bool:
    h = str(host or "").strip().lower()
    return h in {"", "127.0.0.1", "localhost", "::1", "0.0.0.0", "[::1]"}


def mqtt_broker_caption(host: Any, port: Any = 1883, topic: str = "", *, source: str = "mqtt") -> str:
    """Demo broker vs configured public broker. Render localhost is the Render box."""
    h = str(host or "127.0.0.1").strip() or "127.0.0.1"
    try:
        p = int(port)
    except (TypeError, ValueError):
        p = 1883
    topic_s = str(topic or "pdm/sensors/#")
    if is_loopback_host(h):
        where = "the Render VM, not the hangar" if is_render() else "this machine, not the hangar"
        return (
            f"Demo/local MQTT broker `{h}:{p}` topic `{topic_s}` — {where}. "
            "Set MQTT_BROKER / MQTT_PORT / MQTT_TOPIC (sidebar or secrets) for a public broker. "
            "Connecting here does not mean hangar telemetry."
        )
    return (
        f"Configured MQTT broker `{h}:{p}` topic `{topic_s}` ({source}). "
        "This is a ground health ingest path — not a certified flight loop."
    )


def revision_caption(*, data_label: str = "") -> str:
    sha = app_revision(short=True)
    bits = []
    if sha:
        bits.append(f"rev `{sha}`")
    else:
        bits.append("rev unknown")
    if data_label:
        bits.append(data_label)
    if is_render():
        bits.append("Render (no fake uptime)")
    return " · ".join(bits)


def mqtt_defaults() -> dict[str, Any]:
    """Broker host/port/topic from env/secrets, with localhost demo defaults kept."""
    host = _setting("MQTT_BROKER", "127.0.0.1") or "127.0.0.1"
    port_raw = _setting("MQTT_PORT", "1883") or "1883"
    try:
        port = int(port_raw)
    except (TypeError, ValueError):
        port = 1883
    if port < 1 or port > 65535:
        port = 1883
    topic = _setting("MQTT_TOPIC", "pdm/sensors/#") or "pdm/sensors/#"
    stale_raw = _setting("MQTT_STALE_SECONDS", "") or _setting("LIVE_STALE_SECONDS", "30") or "30"
    try:
        stale = int(stale_raw)
    except (TypeError, ValueError):
        stale = 30
    if stale < 1:
        stale = 30
    return {"host": host, "port": port, "topic": topic, "stale_after_s": stale}
