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

Open **http://localhost:8501**. Sample data: `sample_data/sensor_readings.csv`.

```bash
python smoke_test.py
```

Copy `.env.example` → `.env` and set `GEMINI_API_KEY` if you want Gemini on Insights / AI Assistant. Default model is `gemini-3.6-flash` (old aliases remap). Use **Test Gemini** in the sidebar — 404s are shown, not swallowed.

---

## Sample workflow

1. **Upload & Clean** — Load sample CSV or upload sensors. Run industrial clean + quality checks.
2. **Map sensors** — Point messy headers at timestamp, `machine_id`, temperature / vibration / pressure / RPM, optional RUL label.
3. **Anomaly & RUL** — Isolation Forest, then Random Forest remaining useful life / risk by asset.
4. **Charts** — Sensor over time, anomaly flags, risk by asset (readable Plotly hover / margins).
5. **Insights** — Ranked inspect-this-week list (IF score, sensor spike, low RUL), slow-running assets, optional **$/hour** or **$/unit** impact. Ask is scoped to **this upload** (not a general LLM essay). Errors from Gemini / LlamaIndex are shown. Without a key, Ask still answers from the table and tells you to set `GEMINI_API_KEY`.

Optional labs (joins, SQL) sit beside the pipeline. They are not a second Forge.

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
- **LlamaIndex / Gemini** — optional retrieval + LLM (rule-based fallback always works)

---

## Deploy

Streamlit Community Cloud: repo `predictive-maintenance-analytics`, branch `main`, file `app.py`. See **[DEPLOY.md](DEPLOY.md)**.

---

## License

MIT — free for personal and commercial use.
