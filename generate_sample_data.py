#!/usr/bin/env python3
"""Generate sample IoT sensor readings CSV for predictive maintenance demo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

MACHINES = ["Machine 1", "Machine 2", "Machine 3", "Machine 4", "Machine 5"]
HOURS = 720  # 30 days hourly
records = []

for machine_idx, machine in enumerate(MACHINES):
    # Degradation factor increases over time; Machine 4 degrades fastest
    degradation_rate = 0.3 + machine_idx * 0.15
    base_temp = 65 + machine_idx * 3
    base_vib = 2.0 + machine_idx * 0.3
    base_pressure = 100 + machine_idx * 5
    base_rpm = 1800 - machine_idx * 50

    for hour in range(HOURS):
        ts = pd.Timestamp("2025-01-01") + pd.Timedelta(hours=hour)
        progress = hour / HOURS
        noise = np.random.normal(0, 1)

        temp = base_temp + degradation_rate * progress * 25 + noise * 2
        vibration = base_vib + degradation_rate * progress * 4 + abs(noise) * 0.5
        pressure = base_pressure - degradation_rate * progress * 15 + noise * 3
        rpm = base_rpm - degradation_rate * progress * 200 + noise * 20

        # Failure within days decreases as degradation progresses
        remaining = max(1, int(30 * (1 - progress * degradation_rate / 2) + np.random.randint(-2, 3)))

        # Inject occasional anomalies
        if np.random.random() < 0.03:
            temp += np.random.uniform(10, 20)
            vibration += np.random.uniform(2, 5)

        records.append({
            "timestamp": ts,
            "machine_id": machine,
            "temperature": round(temp, 2),
            "vibration": round(vibration, 3),
            "pressure": round(pressure, 2),
            "rpm": round(rpm, 1),
            "failure_within_days": remaining,
        })

df = pd.DataFrame(records)
out_path = Path(__file__).parent / "sample_data" / "sensor_readings.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Generated {len(df):,} rows -> {out_path}")
