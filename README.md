# Predictive Maintenance Analytics

Reliability add-on for IoT sensor data — **not** a generic analytics OS (Forge v2) and **not** a plant OEE cockpit (OEE Pulse).

**Pipeline:** sensor CSV → clean → map sensors → Isolation Forest anomalies → RUL / risk → charts → insights.

Repo: [shaikmohammedshoaib666/predictive-maintenance-analytics](https://github.com/shaikmohammedshoaib666/predictive-maintenance-analytics)

![Python](https://img.shields.io/badge/python-3.9%2B-slate)
![Streamlit](https://img.shields.io/badge/streamlit-reliability%20add--on-amber)

---

## What this is (and is not)

| App | Role | Run |
|-----|------|-----|
| **This repo (PdM)** | Sensor reliability: anomalies + remaining useful life | `streamlit run app.py` here |
| **OEE Pulse** | Plant SaaS: OEE, downtime Pareto, $ impact | `streamlit run app.py` in `oee-pulse` |
| **Forge v2** | Analytics OS: domain packs, LIVE SCADA, generic KPIs | `streamlit run app.py` in `analytics-forge-v2` |

Honest RUL needs a real `failure_within_days` (or alias) label. Sample CSV labels are **simulated**. Without a label, the model trains on a sensor-degradation proxy — treat days-to-fail as a demo, not a plant forecast.

---

## Run locally

```bash
cd ~/Projects/predictive-maintenance-analytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501**. Default demo: `sample_data/sensor_readings.csv` (Plant pack). Aviation / auto / oil demos are on *Upload & Clean*.

```bash
python smoke_test.py
```

Copy `.env.example` → `.env` and set `GEMINI_API_KEY` if you want Gemini on Insights / AI Assistant. Default model is `gemini-3.6-flash` (old aliases remap). Use **Test Gemini** in the sidebar — 404s are shown, not swallowed.

---

## Sample workflow

1. **Upload & Clean** — Load sample CSV or upload sensors. Run industrial clean + quality checks.
2. **Map sensors** — Point messy headers at timestamp, `machine_id`, temperature / vibration / pressure / RPM, optional RUL label.
3. **Anomaly & RUL** — Isolation Forest, then Random Forest remaining useful life / risk by asset.
4. **Charts** — Sensor over time, anomaly flags, risk by asset (readable Plotly hover / margins). Pack KPIs (mission reliability, powertrain health, fillage, …) sit above the primary charts.
5. **Insights** — Ranked inspect-this-week list (IF score, sensor spike, low RUL), slow-running assets, optional **$/hour** or **$/unit** impact, plus pack-specific KPI cards. Ask is scoped to **this upload** (not a general LLM essay). Errors from Gemini / LlamaIndex are shown. Without a key, Ask still answers from the table and tells you to set `GEMINI_API_KEY`.

Optional labs (joins, SQL) sit beside the pipeline. They are not a second Forge.

---

## Industry-ready layers

Beyond the core pipeline, the app adds five layers aimed at real plant workflows:

1. **DuckDB URL / cloud ingest** — load sensor data from a direct HTTPS CSV/Parquet link, a Google Drive share URL, or a Kaggle dataset (`kaggle://owner/dataset/file.csv`; needs `KAGGLE_USERNAME` + `KAGGLE_KEY`). Large/Drive files are cached to disk so DuckDB scans them out-of-core. See the **From URL (cloud / DuckDB)** tab on *Upload & Clean*.
2. **SQL slice presets** — filter/limit at the source before ingesting (last N rows/days, by `machine_id` / `asset_id`, date range, random sample %, PdM failure focus). Edit the DuckDB SQL to combine filters; only read-only `SELECT`/`WITH` is allowed.
3. **3D Digital Twin** — a rotatable pack-specific mesh (Plant motor by default; UAV + piston engine; generic ICE car; SRP beam pump) whose hotspot turns **red and blinks** when the selected asset's predicted risk is High (amber = Medium, green = Low). Risk can come from batch *Anomaly & RUL* or from *Live Connect*.
4. **Live Connect** — streaming ingest that feeds the anomaly pipeline in near-real-time, via a built-in simulator (a selectable asset drifts to failure) or by polling a remote CSV feed. Live risk drives the 3D Twin.
5. **PySpark cleaning engine (optional)** — a distributed clean engine for very large files, shown on *Upload & Clean* when `pyspark` + a JVM are installed. Kept out of `requirements.txt` to keep the Render deploy lean; install with `pip install -r requirements-optional.txt` (needs a JVM, e.g. `apt-get install default-jre`).

Maintenance attach: on *Upload & Clean* you can attach a work-order / PM CSV as the `maintenance` table for joins. Quality sub-reports (Great Expectations, ydata, Cleanlab, PCA drift, association rules, OPC physics) render under the 19-stage report.

### Upgrades

- **Offline 3D twin** — three.js + OrbitControls are vendored in `src/vendor/three` and inlined, so the *3D Twin* renders with **no internet/CDN**. The mesh follows the **industry pack** (sidebar).
- **Real live sources** — *Live Connect* adds **MQTT** and **OPC-UA** sources (alongside the simulator and CSV polling). MQTT expects JSON sensor payloads; OPC-UA reads a set of node IDs each poll. Requires `paho-mqtt` / `asyncua` (both in `requirements.txt`).
- **CAD Twin (Autodesk APS)** — optional *CAD Twin* page renders a real translated CAD model (Revit/Fusion/IFC → SVF) via Autodesk Platform Services and tints it red on High risk. Availability-gated: set `APS_CLIENT_ID` + `APS_CLIENT_SECRET` (secrets) and a translated model URN to enable; otherwise it shows setup steps and the app is unaffected.

---

## Industry packs (one file → one pack)

Plant / rotating machines is the **default**. Switching the sidebar pack switches KPIs, extra column mapping, default charts, and the 3D mesh. A CSV does not magically become an airplane vs a car from `machine_id` alone — pick the pack (or use **Suggest pack from columns**).

| Pack | 3D mesh | Hero / SIH | Demo CSV |
|------|---------|------------|----------|
| **Plant / rotating machines** (default) | Generic motor, drive-end bearing hotspot | — | `sample_data/sensor_readings.csv` |
| **Aviation / aero piston / MALE UAV** | UAV airframe + piston engine | **SIH26054** (DRDO) | `sample_data/aviation_uav_piston.csv` |
| **Automotive powertrain** | One generic ICE car + engine/gearbox | — | `sample_data/automotive_powertrain.csv` |
| **Oil well / sucker-rod pump** | Beam pump / wellhead | **SIH26120** (Oil India) | `sample_data/oil_srp.csv` |

### What “not every OEM / trim / EV architecture” means

The automotive pack is **one generic car + ICE powertrain**, not a vehicle catalog:

- **OEM** — manufacturer (BMW vs Toyota vs Ford). No per-brand body CAD or health model.
- **Trim** — model grade (LX vs Sport vs Limited). One silhouette, not option packages.
- **EV architecture** — battery-electric skateboard vs hybrid vs ICE. This pack is a piston engine + gearbox/driveline. Separate BEV / hybrid twins are out of scope.

Aviation is a **MALE UAV + aero piston engine**, not an airliner cabin or a turbofan catalog. Oil is **one SRP / CSS well**, not every completion type.

Regenerate demos: `python generate_sample_data.py` (plant + three pack CSVs). To skip rewriting the plant file, call the pack generators directly.

---

## ML Models

### Anomaly Detection — Isolation Forest

Unsupervised flags on multivariate sensors. No failure labels required.

### RUL — Random Forest

Predicts remaining useful life in days when `failure_within_days` (or a mapped alias) exists. Metrics: MAE, RMSE, R². Without labels, a degradation proxy is used and the UI warns.

---

## Expected CSV format

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Reading timestamp |
| `machine_id` | string | Equipment identifier |
| `temperature` | float | Temperature sensor (°C) |
| `vibration` | float | Vibration sensor (mm/s) |
| `pressure` | float | Pressure sensor (PSI) |
| `rpm` | float | Rotational speed |
| `failure_within_days` | float | Target: days until failure (**optional; required for honest RUL**) |

Messy aliases (`asset_id`, `temp_c`, `vib`, `rul_days`, …) can be mapped in **Map sensors**.

---

## Tech stack

- **Python 3.9+**
- **Streamlit** — UI
- **pandas / numpy / scikit-learn** — clean + Isolation Forest + Random Forest
- **Plotly** — charts
- **LlamaIndex / Gemini** — retrieval + LLM (rule-based fallback always works)

---

## Deploy

**Render** (not Streamlit Cloud): repo `predictive-maintenance-analytics`, branch `main`, start `bash start.sh`, Python **3.11.9**. See **[DEPLOY.md](DEPLOY.md)**.

---

## License

MIT — free for personal and commercial use.
