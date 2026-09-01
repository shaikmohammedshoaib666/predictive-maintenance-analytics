#!/usr/bin/env python3
"""Generate sample IoT sensor readings CSV for predictive maintenance demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MACHINES = ["Machine 1", "Machine 2", "Machine 3", "Machine 4", "Machine 5"]
HOURS = 720  # 30 days hourly


def _write(name: str, rows: list[dict]) -> Path:
    out = Path(__file__).parent / "sample_data" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Generated {len(rows):,} rows -> {out}")
    return out


def generate_plant(hours: int = HOURS) -> Path:
    """Plant / rotating-machines demo (default pack). Machine 4 degrades fastest."""
    rng = np.random.default_rng(42)
    records: list[dict] = []
    for machine_idx, machine in enumerate(MACHINES):
        degradation_rate = 0.3 + machine_idx * 0.15
        base_temp = 65 + machine_idx * 3
        base_vib = 2.0 + machine_idx * 0.3
        base_pressure = 100 + machine_idx * 5
        base_rpm = 1800 - machine_idx * 50
        for hour in range(hours):
            ts = pd.Timestamp("2025-01-01") + pd.Timedelta(hours=hour)
            progress = hour / hours
            noise = float(rng.normal(0, 1))
            temp = base_temp + degradation_rate * progress * 25 + noise * 2
            vibration = base_vib + degradation_rate * progress * 4 + abs(noise) * 0.5
            pressure = base_pressure - degradation_rate * progress * 15 + noise * 3
            rpm = base_rpm - degradation_rate * progress * 200 + noise * 20
            remaining = max(1, int(30 * (1 - progress * degradation_rate / 2) + int(rng.integers(-2, 3))))
            if rng.random() < 0.03:
                temp += float(rng.uniform(10, 20))
                vibration += float(rng.uniform(2, 5))
            records.append(
                {
                    "timestamp": ts,
                    "machine_id": machine,
                    "temperature": round(temp, 2),
                    "vibration": round(vibration, 3),
                    "pressure": round(pressure, 2),
                    "rpm": round(rpm, 1),
                    "failure_within_days": remaining,
                }
            )
    return _write("sensor_readings.csv", records)


def generate_aviation_uav(hours: int = 168) -> Path:
    """SIH26054-style MALE UAV piston-engine demo. UAV-03 degrades fastest."""
    rng = np.random.default_rng(7)
    rows: list[dict] = []
    fleet = [
        ("UAV-01", 0.25, 720.0),
        ("UAV-02", 0.40, 410.0),
        ("UAV-03", 0.85, 980.0),
    ]
    for uav, deg, hobbs0 in fleet:
        for hour in range(hours):
            ts = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour)
            p = hour / hours
            n = float(rng.normal(0, 1))
            cht = 165 + deg * p * 45 + n * 3
            egt = 680 + deg * p * 90 + n * 8
            oil_p = 62 - deg * p * 22 + n * 1.5
            oil_t = 88 + deg * p * 25 + n * 2
            vib = 1.4 + deg * p * 4.2 + abs(n) * 0.2
            rpm = 2450 - deg * p * 180 + n * 25
            alt = 8500 + float(rng.normal(0, 120))
            throttle = 72 + deg * p * 8 + n * 2
            mp = 28.5 - deg * p * 3 + n * 0.3
            remaining = max(1, int(21 * (1 - p * deg / 1.1) + int(rng.integers(-2, 3))))
            if rng.random() < 0.04:
                egt += float(rng.uniform(30, 70))
                vib += float(rng.uniform(1.5, 3.5))
            rows.append(
                {
                    "timestamp": ts,
                    "machine_id": uav,
                    "temperature": round(cht, 2),
                    "vibration": round(vib, 3),
                    "pressure": round(oil_p, 2),
                    "rpm": round(rpm, 1),
                    "failure_within_days": remaining,
                    "egt": round(egt, 1),
                    "cht": round(cht, 1),
                    "oil_pressure": round(oil_p, 2),
                    "oil_temp": round(oil_t, 1),
                    "flight_hours": round(hobbs0 + hour / 24.0 * 0.35, 2),
                    "altitude": round(alt, 0),
                    "throttle": round(float(np.clip(throttle, 20, 100)), 1),
                    "manifold_pressure": round(mp, 2),
                }
            )
    return _write("aviation_uav_piston.csv", rows)


def generate_automotive(hours: int = 168) -> Path:
    """Generic ICE powertrain demo — one silhouette, not OEM/trim/EV variants."""
    rng = np.random.default_rng(11)
    rows: list[dict] = []
    fleet = [("ENG-01", 0.22), ("ENG-02", 0.38), ("ENG-03", 0.80)]
    for eng, deg in fleet:
        for hour in range(hours):
            ts = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour)
            p = hour / hours
            n = float(rng.normal(0, 1))
            coolant = 88 + deg * p * 22 + n * 1.8
            oil_p = 45 - deg * p * 18 + n * 1.2
            load = 42 + deg * p * 20 + abs(n) * 4
            speed = max(0.0, 55 + n * 12)
            temp = coolant
            vib = 1.8 + deg * p * 3.5 + abs(n) * 0.25
            rpm = 2100 + load * 18 + n * 40
            remaining = max(1, int(24 * (1 - p * deg / 1.05) + int(rng.integers(-2, 3))))
            if rng.random() < 0.04:
                coolant += float(rng.uniform(8, 16))
                vib += float(rng.uniform(1, 3))
            rows.append(
                {
                    "timestamp": ts,
                    "machine_id": eng,
                    "temperature": round(temp, 2),
                    "vibration": round(vib, 3),
                    "pressure": round(oil_p, 2),
                    "rpm": round(rpm, 1),
                    "failure_within_days": remaining,
                    "coolant_temp": round(coolant, 1),
                    "oil_pressure": round(oil_p, 2),
                    "engine_load": round(float(np.clip(load, 5, 100)), 1),
                    "vehicle_speed": round(speed, 1),
                    "gear": int(np.clip(round(speed / 18), 1, 6)),
                }
            )
    return _write("automotive_powertrain.csv", rows)


def generate_oil_srp(hours: int = 168) -> Path:
    """SIH26120-style sucker-rod pump / CSS well demo. WELL-03 is the sick well."""
    rng = np.random.default_rng(19)
    rows: list[dict] = []
    fleet = [("WELL-01", 0.20), ("WELL-02", 0.35), ("WELL-03", 0.82)]
    for well, deg in fleet:
        for hour in range(hours):
            ts = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour)
            p = hour / hours
            n = float(rng.normal(0, 1))
            fillage = 92 - deg * p * 35 + n * 2
            rod = 14500 + deg * p * 4200 + n * 180
            tbg = 180 - deg * p * 40 + n * 6
            csg = 95 + n * 3
            spm = 8.5 - deg * p * 1.8 + n * 0.15
            bbl = 48 - deg * p * 22 + n * 1.5
            temp = 62 + deg * p * 12 + n * 1.2
            vib = 2.1 + deg * p * 3.8 + abs(n) * 0.2
            rpm = 420 - deg * p * 40 + n * 8
            remaining = max(1, int(20 * (1 - p * deg / 1.05) + int(rng.integers(-2, 3))))
            if rng.random() < 0.04:
                fillage -= float(rng.uniform(8, 18))
                rod += float(rng.uniform(800, 2000))
            rows.append(
                {
                    "timestamp": ts,
                    "machine_id": well,
                    "temperature": round(temp, 2),
                    "vibration": round(vib, 3),
                    "pressure": round(tbg, 2),
                    "rpm": round(rpm, 1),
                    "failure_within_days": remaining,
                    "polish_rod_load": round(rod, 0),
                    "tubing_pressure": round(tbg, 1),
                    "casing_pressure": round(csg, 1),
                    "pump_fillage": round(float(np.clip(fillage, 35, 100)), 1),
                    "stroke_spm": round(float(np.clip(spm, 4, 12)), 2),
                    "production_bbl": round(float(np.clip(bbl, 5, 80)), 2),
                }
            )
    return _write("oil_srp.csv", rows)


def generate_all(*, include_plant: bool = True) -> None:
    if include_plant:
        generate_plant()
    generate_aviation_uav()
    generate_automotive()
    generate_oil_srp()


if __name__ == "__main__":
    generate_all()
