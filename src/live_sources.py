"""Live Connect producers — simulator, MQTT, OPC-UA, CSV poll.

Connection objects live in a process-global registry (threads/clients are not
session-state friendly). The Streamlit page keeps only ``conn_id`` and drains.

MQTT ``connect()`` must never block the Streamlit request thread — that froze
the websocket and showed the browser “Connecting…” / reload loop.
"""

from __future__ import annotations

import json
import socket
import threading
import uuid
from collections import deque
from typing import Any, Optional

import pandas as pd

SENSOR_COLS = ("temperature", "vibration", "pressure", "rpm")

# Unified registry (sim / mqtt / opcua / poll). Old names kept as aliases.
_SOURCES: dict[str, Any] = {}
_MQTT_CONNS: dict[str, "MqttSource"] = _SOURCES
_OPCUA_CONNS: dict[str, "OpcuaSource"] = _SOURCES

LOOPBACK_PROBE_S = 1.2
LIVE_TICK_S = 2.0


def mqtt_available() -> tuple[bool, str]:
    try:
        import paho.mqtt  # noqa: F401
    except Exception as exc:
        return False, f"paho-mqtt not installed ({exc.__class__.__name__})"
    return True, "paho-mqtt"


def opcua_available() -> tuple[bool, str]:
    try:
        import asyncua  # noqa: F401
    except Exception as exc:
        return False, f"asyncua not installed ({exc.__class__.__name__})"
    return True, "asyncua"


def probe_tcp(host: str, port: int, timeout: float = LOOPBACK_PROBE_S) -> tuple[bool, str]:
    """Fail-fast TCP check so Start live cannot hang the Streamlit websocket."""
    h = (host or "").strip() or "127.0.0.1"
    try:
        p = int(port)
    except (TypeError, ValueError):
        return False, f"invalid port {port!r}"
    try:
        with socket.create_connection((h, p), timeout=float(timeout)):
            return True, "ok"
    except Exception as exc:
        return False, f"{h}:{p} — {exc.__class__.__name__}: {exc}"


def _coerce_row(payload: Any, *, machine_field: str, ts_field: str, fallback_machine: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {"value": payload}
    row: dict[str, Any] = {}
    ts = payload.get(ts_field)
    row["timestamp"] = pd.to_datetime(ts, errors="coerce") if ts else pd.Timestamp.utcnow().floor("s")
    if pd.isna(row["timestamp"]):
        row["timestamp"] = pd.Timestamp.utcnow().floor("s")
    mid = payload.get(machine_field)
    row["machine_id"] = str(mid) if mid not in (None, "") else fallback_machine
    for c in SENSOR_COLS:
        if c in payload:
            try:
                row[c] = float(payload[c])
            except (TypeError, ValueError):
                pass
    return row


def _drain_buf(buf: deque, lock: threading.Lock) -> pd.DataFrame:
    with lock:
        rows = list(buf)
        buf.clear()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


class SimSource:
    """Background simulator — does not use the network. First batch is immediate."""

    def __init__(
        self,
        machines: list[str],
        *,
        failing: Optional[str] = None,
        ramp_ticks: int = 40,
        freq_seconds: int = 5,
        interval_s: float = LIVE_TICK_S,
        max_buffer: int = 5000,
    ):
        self.machines = list(machines) or ["M-001"]
        self.failing = failing or self.machines[0]
        self.ramp_ticks = max(1, int(ramp_ticks))
        self.freq_seconds = int(freq_seconds)
        self.interval_s = float(interval_s)
        self.connected = True
        self.error: Optional[str] = None
        self.msg_count = 0
        self.tick = 0
        self._buf: deque[dict[str, Any]] = deque(maxlen=max_buffer)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _push_tick(self) -> None:
        from src.live_connect import simulate_batch

        stress = min(1.0, self.tick / self.ramp_ticks)
        batch = simulate_batch(
            self.machines,
            self.tick,
            failing=self.failing,
            stress=stress,
            freq_seconds=self.freq_seconds,
        )
        rows = batch.to_dict(orient="records")
        with self._lock:
            for row in rows:
                self._buf.append(row)
                self.msg_count += 1
        self.tick += 1

    def start(self) -> None:
        self._stop.clear()
        self._push_tick()
        self._thread = threading.Thread(target=self._loop, name="pdm-sim", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self._push_tick()
            except Exception as exc:
                self.error = str(exc)
                self.connected = False
                return

    def drain(self) -> pd.DataFrame:
        return _drain_buf(self._buf, self._lock)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        self.connected = False


class PollSource:
    """Background CSV poll so a hung URL cannot freeze Streamlit."""

    def __init__(
        self,
        url: str,
        cache_dir: Any,
        *,
        poll_n: int = 30,
        interval_s: float = LIVE_TICK_S,
        max_buffer: int = 5000,
    ):
        self.url = url
        self.cache_dir = cache_dir
        self.poll_n = int(poll_n)
        self.interval_s = float(interval_s)
        self.offset = 0
        self.connected = True
        self.error: Optional[str] = None
        self.msg_count = 0
        self._buf: deque[dict[str, Any]] = deque(maxlen=max_buffer)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ended = False

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="pdm-poll", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        from src.live_connect import poll_url_increment

        while not self._stop.is_set():
            try:
                batch, new_off = poll_url_increment(self.url, self.cache_dir, self.offset, self.poll_n)
            except Exception as exc:
                self.error = str(exc)
                self.connected = False
                return
            if batch is None or batch.empty:
                self._ended = True
                self.error = "Reached end of CSV feed (or empty poll)."
                return
            rows = batch.to_dict(orient="records")
            with self._lock:
                for row in rows:
                    self._buf.append(row)
                    self.msg_count += 1
            self.offset = new_off
            if self._stop.wait(self.interval_s):
                return

    def drain(self) -> pd.DataFrame:
        return _drain_buf(self._buf, self._lock)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        self.connected = False


class MqttSource:
    """Background MQTT subscriber. Connect is async — never blocks Streamlit."""

    def __init__(
        self,
        host: str,
        port: int = 1883,
        topic: str = "pdm/sensors/#",
        *,
        username: str = "",
        password: str = "",
        machine_field: str = "machine_id",
        ts_field: str = "timestamp",
        max_buffer: int = 5000,
    ):
        self.host = host
        self.port = int(port)
        self.topic = topic
        self.username = username
        self.password = password
        self.machine_field = machine_field
        self.ts_field = ts_field
        self._buf: deque[dict[str, Any]] = deque(maxlen=max_buffer)
        self._lock = threading.Lock()
        self._client = None
        self.connected = False
        self.error: Optional[str] = None
        self.msg_count = 0
        self.last_message_at: Optional[pd.Timestamp] = None
        self.last_seen: dict[str, pd.Timestamp] = {}

    def start(self) -> None:
        import paho.mqtt.client as mqtt

        from src.ops_meta import is_loopback_host

        if is_loopback_host(self.host):
            ok, why = probe_tcp(self.host, self.port, timeout=LOOPBACK_PROBE_S)
            if not ok:
                raise ConnectionError(
                    f"No MQTT broker on {self.host}:{self.port} ({why}). "
                    "Use **Simulator** — localhost is this app box, not hangar telemetry."
                )

        client = mqtt.Client(client_id=f"pdm-{uuid.uuid4().hex[:8]}", clean_session=True)
        if self.username:
            client.username_pw_set(self.username, self.password or None)

        def on_connect(cl, userdata, flags, rc):
            self.connected = rc == 0
            if rc == 0:
                self.error = None
                cl.subscribe(self.topic)
            else:
                self.error = f"MQTT connect rc={rc}"

        def on_disconnect(cl, userdata, rc):
            self.connected = False
            if rc != 0:
                self.error = f"MQTT disconnected rc={rc}"

        def on_message(cl, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
            except Exception:
                return
            row = _coerce_row(
                payload,
                machine_field=self.machine_field,
                ts_field=self.ts_field,
                fallback_machine=(msg.topic.split("/")[-1] or "mqtt-asset"),
            )
            with self._lock:
                self._buf.append(row)
                self.msg_count += 1
                now = pd.Timestamp.utcnow().floor("s")
                self.last_message_at = now
                mid = str(row.get("machine_id") or "")
                if mid:
                    self.last_seen[mid] = now

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = on_message
        try:
            client.reconnect_delay_set(min_delay=2, max_delay=30)
        except Exception:
            pass
        client.connect_async(self.host, self.port, keepalive=30)
        client.loop_start()
        self._client = client

    def drain(self) -> pd.DataFrame:
        return _drain_buf(self._buf, self._lock)

    def stop(self) -> None:
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
        self.connected = False


class OpcuaSource:
    """OPC-UA client. TCP connect runs on a thread so Start live cannot hang."""

    def __init__(self, endpoint: str, node_map: dict[str, str], *, machine_id: str = "opcua-asset"):
        self.endpoint = endpoint
        self.node_map = node_map
        self.machine_id = machine_id
        self._client = None
        self.connected = False
        self.error: Optional[str] = None
        self.msg_count = 0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._connect, name="pdm-opcua", daemon=True)
        self._thread.start()

    def _connect(self) -> None:
        try:
            from asyncua.sync import Client

            self._client = Client(self.endpoint)
            self._client.connect()
            self.connected = True
            self.error = None
        except Exception as exc:
            self.error = str(exc)
            self.connected = False

    def poll_once(self) -> pd.DataFrame:
        if self._client is None or not self.connected:
            return pd.DataFrame()
        row: dict[str, Any] = {"timestamp": pd.Timestamp.utcnow().floor("s"), "machine_id": self.machine_id}
        for name, node_id in self.node_map.items():
            try:
                row[name] = float(self._client.get_node(node_id).read_value())
            except Exception as exc:
                self.error = f"{name}: {exc}"
        self.msg_count += 1
        return pd.DataFrame([row])

    def drain(self) -> pd.DataFrame:
        return self.poll_once()

    def stop(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
        self.connected = False


def source_snapshot(src: Any) -> dict[str, Any]:
    if src is None:
        return {"kind": "none", "connected": False, "error": "no source", "msg_count": 0}
    return {
        "kind": type(src).__name__,
        "connected": bool(getattr(src, "connected", False)),
        "error": getattr(src, "error", None),
        "msg_count": int(getattr(src, "msg_count", 0) or 0),
        "ended": bool(getattr(src, "_ended", False)),
    }


def _register(prefix: str, src: Any) -> str:
    conn_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
    _SOURCES[conn_id] = src
    return conn_id


def start_sim(cfg: dict[str, Any]) -> str:
    from src.live_connect import default_machines

    machines = list(cfg.get("machines") or default_machines(int(cfg.get("n_machines") or 4)))
    src = SimSource(
        machines,
        failing=cfg.get("failing"),
        ramp_ticks=int(cfg.get("ramp_ticks") or 40),
        freq_seconds=int(cfg.get("freq") or 5),
        interval_s=float(cfg.get("interval_s") or LIVE_TICK_S),
    )
    src.start()
    return _register("sim", src)


def start_poll(cfg: dict[str, Any]) -> str:
    url = str(cfg.get("url") or "").strip()
    if not url:
        raise ValueError("CSV feed URL is empty.")
    src = PollSource(
        url,
        cfg.get("cache_dir"),
        poll_n=int(cfg.get("poll_n") or 30),
        interval_s=float(cfg.get("interval_s") or LIVE_TICK_S),
    )
    src.start()
    return _register("poll", src)


def start_mqtt(cfg: dict[str, Any]) -> str:
    src = MqttSource(
        host=cfg["host"],
        port=int(cfg.get("port", 1883)),
        topic=cfg.get("topic", "pdm/sensors/#"),
        username=cfg.get("username", ""),
        password=cfg.get("password", ""),
        machine_field=cfg.get("machine_field", "machine_id"),
    )
    src.start()
    return _register("mqtt", src)


def start_opcua(cfg: dict[str, Any]) -> str:
    src = OpcuaSource(
        endpoint=cfg["endpoint"],
        node_map=cfg.get("node_map") or {},
        machine_id=cfg.get("machine_id", "opcua-asset"),
    )
    src.start()
    return _register("opcua", src)


def get_source(conn_id: Optional[str]):
    if not conn_id:
        return None
    return _SOURCES.get(conn_id)


def stop_source(conn_id: Optional[str]) -> None:
    src = get_source(conn_id)
    if src is not None:
        src.stop()
    if conn_id:
        _SOURCES.pop(conn_id, None)


def mqtt_defaults() -> dict[str, Any]:
    """Host/port/topic from MQTT_* env/secrets, localhost demo kept as default."""
    from src.ops_meta import mqtt_defaults as _defaults

    return _defaults()


def parse_node_map(text: str) -> dict[str, str]:
    """Parse 'temperature=ns=2;i=2, vibration=ns=2;i=3' into {sensor: node_id}."""
    out: dict[str, str] = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, node = part.split("=", 1)
        out[name.strip()] = node.strip()
    return out
