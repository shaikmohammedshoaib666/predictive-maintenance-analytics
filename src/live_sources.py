"""Upgrade 2 — real streaming sources for Live Connect: MQTT + OPC-UA.

These plug into the same rolling-buffer pipeline as the simulator. Connection
objects live in a process-global registry (threads/clients are not
session-state friendly); the Streamlit page keeps only a string ``conn_id`` in
session state and drains new readings each tick.

* **MQTT** — subscribe to a broker/topic; JSON payloads become sensor rows.
* **OPC-UA** — connect to a server and read a set of node IDs each poll.

Both dependencies are optional at import time; ``mqtt_available`` /
``opcua_available`` gate the UI so a missing package never breaks the app.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import deque
from typing import Any, Optional

import pandas as pd

SENSOR_COLS = ("temperature", "vibration", "pressure", "rpm")

# Process-global registries (survive Streamlit reruns within one server process).
_MQTT_CONNS: dict[str, "MqttSource"] = {}
_OPCUA_CONNS: dict[str, "OpcuaSource"] = {}


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


class MqttSource:
    """Background MQTT subscriber that buffers incoming JSON sensor messages."""

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

    def start(self) -> None:
        import paho.mqtt.client as mqtt

        client = mqtt.Client(client_id=f"pdm-{uuid.uuid4().hex[:8]}", clean_session=True)
        if self.username:
            client.username_pw_set(self.username, self.password or None)

        def on_connect(cl, userdata, flags, rc):
            self.connected = rc == 0
            if rc == 0:
                cl.subscribe(self.topic)
            else:
                self.error = f"MQTT connect rc={rc}"

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

        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(self.host, self.port, keepalive=30)
        client.loop_start()
        self._client = client

    def drain(self) -> pd.DataFrame:
        with self._lock:
            rows = list(self._buf)
            self._buf.clear()
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def stop(self) -> None:
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
        self.connected = False


class OpcuaSource:
    """Synchronous OPC-UA client that reads a set of node IDs per poll."""

    def __init__(self, endpoint: str, node_map: dict[str, str], *, machine_id: str = "opcua-asset"):
        self.endpoint = endpoint
        self.node_map = node_map  # sensor_name -> node id (e.g. "ns=2;i=2")
        self.machine_id = machine_id
        self._client = None
        self.connected = False
        self.error: Optional[str] = None

    def start(self) -> None:
        from asyncua.sync import Client

        self._client = Client(self.endpoint)
        self._client.connect()
        self.connected = True

    def poll_once(self) -> pd.DataFrame:
        if self._client is None:
            return pd.DataFrame()
        row: dict[str, Any] = {"timestamp": pd.Timestamp.utcnow().floor("s"), "machine_id": self.machine_id}
        for name, node_id in self.node_map.items():
            try:
                row[name] = float(self._client.get_node(node_id).read_value())
            except Exception as exc:
                self.error = f"{name}: {exc}"
        return pd.DataFrame([row])

    def stop(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
        self.connected = False


# ── Registry helpers (keep only conn_id in session_state) ─────────────────────
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
    conn_id = f"mqtt-{uuid.uuid4().hex[:8]}"
    _MQTT_CONNS[conn_id] = src
    return conn_id


def start_opcua(cfg: dict[str, Any]) -> str:
    src = OpcuaSource(
        endpoint=cfg["endpoint"],
        node_map=cfg.get("node_map") or {},
        machine_id=cfg.get("machine_id", "opcua-asset"),
    )
    src.start()
    conn_id = f"opcua-{uuid.uuid4().hex[:8]}"
    _OPCUA_CONNS[conn_id] = src
    return conn_id


def get_source(conn_id: Optional[str]):
    if not conn_id:
        return None
    return _MQTT_CONNS.get(conn_id) or _OPCUA_CONNS.get(conn_id)


def stop_source(conn_id: Optional[str]) -> None:
    src = get_source(conn_id)
    if src is not None:
        src.stop()
    _MQTT_CONNS.pop(conn_id, None)
    _OPCUA_CONNS.pop(conn_id, None)


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
