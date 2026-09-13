#!/usr/bin/env python3
"""HIL MQTT publisher — replay box/aviation CSV (or a synthetic UAV tick) as live JSON.

Does not talk to TAPAS. Same topic/JSON as Live Connect Paho.

  python hil_mqtt_publisher.py
  python hil_mqtt_publisher.py --csv sample_data/aviation_uav_piston.csv --hz 1

Broker/user/pass from MQTT_BROKER / MQTT_PORT / MQTT_TOPIC / MQTT_USER / MQTT_PASS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish HIL UAV JSON to MQTT.")
    parser.add_argument("--csv", default="", help="Optional CSV to replay line by line")
    parser.add_argument("--hz", type=float, default=1.0)
    parser.add_argument("--host", default=_env("MQTT_BROKER", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(_env("MQTT_PORT", "1883") or "1883"))
    parser.add_argument("--topic", default=_env("MQTT_TOPIC", "tapas/engine/raw") or "tapas/engine/raw")
    parser.add_argument("--user", default=_env("MQTT_USER", ""))
    parser.add_argument("--password", default=_env("MQTT_PASS", "") or _env("MQTT_PASSWORD", ""))
    args = parser.parse_args()

    import paho.mqtt.client as mqtt

    from src.live_connect import simulate_batch
    from src.live_schema import row_to_mqtt_json

    client = mqtt.Client(client_id="pdm-hil-pub")
    if args.user:
        client.username_pw_set(args.user, args.password or None)
    client.connect(args.host, int(args.port), keepalive=30)

    delay = 1.0 / max(args.hz, 0.05)
    if args.csv:
        import pandas as pd

        df = pd.read_csv(args.csv)
        print(f"Replaying {len(df)} rows from {args.csv} → {args.host}:{args.port} {args.topic}")
        for _, rec in df.iterrows():
            payload = row_to_mqtt_json(rec.to_dict())
            client.publish(args.topic, json.dumps(payload))
            time.sleep(delay)
        return 0

    tick = 0
    print(f"Synthetic UAV HIL → {args.host}:{args.port} {args.topic} (Ctrl+C to stop)")
    while True:
        batch = simulate_batch(
            ["UAV-01", "UAV-02", "UAV-03"],
            tick,
            failing="UAV-03",
            stress=min(1.0, tick / 40.0),
            freq_seconds=1,
            pack_id="aviation_uav_piston",
        )
        for rec in batch.to_dict(orient="records"):
            client.publish(args.topic, json.dumps(row_to_mqtt_json(rec)))
        tick += 1
        time.sleep(delay)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
