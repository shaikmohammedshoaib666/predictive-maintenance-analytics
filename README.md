# Predictive Maintenance Analytics Platform

> **Resume line:** Built an end-to-end IoT predictive maintenance platform with Streamlit, scikit-learn (Isolation Forest + Random Forest RUL), and Plotly — enabling sensor upload, anomaly detection, failure forecasting, dashboard composition, and automated stakeholder reporting.

A professional analytics platform for senior data analysts working with IoT sensor data. Upload CSV sensor readings, clean and explore data, train ML models to predict equipment failure, build custom dashboards, chat with a data-aware AI assistant, and email reports to managers.

---

## Getting Started on This Mac

If `git` or `python3` says developer tools are missing, do this first.

### 1. Install Xcode Command Line Tools

```bash
xcode-select --install
```

Click **Install** in the dialog, wait until it finishes, then **reopen Terminal**.

### 2. Verify tools

```bash
git --version
python3 --version
```

### 3. Run the app locally

```bash
cd ~/Projects/predictive-maintenance-analytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501**. Sample data: `sample_data/sensor_readings.csv`.

### 4. Push to GitHub (after CLT is installed)

```bash
cd ~/Projects/predictive-maintenance-analytics
git init
git branch -M main
git add .
git commit -m "Initial predictive maintenance analytics platform"
gh auth login
gh repo create predictive-maintenance-analytics --public --source=. --remote=origin --push
```

No `gh`? Create the repo at [github.com/new](https://github.com/new), then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/predictive-maintenance-analytics.git
git push -u origin main
```

Full details: **[DEPLOY.md](DEPLOY.md)**.

### 5. Deploy on Streamlit Community Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **Create app** → pick this repo → branch **`main`** → main file **`app.py`**.
3. **Deploy** — optional SMTP secrets are documented in [DEPLOY.md](DEPLOY.md).

---

## Quick Start

```bash
cd ~/Projects/predictive-maintenance-analytics
pip install -r requirements.txt
python generate_sample_data.py   # optional — sample CSV is included
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Streamlit Web Application (app.py)              │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────────┤
│ Upload & │ Explore  │    ML    │Dashboard │    AI    │    Email     │
│  Clean   │ & Graphs │Predictions│ Builder │Assistant │   Report     │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┴──────┬───────┘
     │          │          │          │          │            │
     ▼          ▼          ▼          ▼          ▼            ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│  data_  │ │ graphs/ │ │   ml/   │ │dashboard│ │   ai_   │ │ email_  │
│ cleaner │ │ (6 types)│ │ anomaly │ │ builder │ │assistant│ │ report  │
│ insights│ │         │ │   RUL   │ │         │ │         │ │         │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     │           │           │           │           │           │
     └───────────┴───────────┴───────────┴───────────┴───────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  pandas / numpy  │
                          │  scikit-learn    │
                          │  plotly          │
                          └─────────────────┘
```

---

## Sample Workflow

1. **Upload & Clean** — Load `sample_data/sensor_readings.csv` or upload your own CSV. Click **Clean Data**, review insights, download cleaned CSV.
2. **Explore & Graphs** — Set x-axis, metric, and machine filters. Click any of 6 graph types to generate and save to your graph folder.
3. **ML Predictions** — Click **Train Models**. View predictions like *"Machine 4 will fail in 7 days"*, model metrics, and feature importance.
4. **Dashboard Builder** — Check saved graphs, preview live dashboard, export as HTML.
5. **AI Assistant** — Ask *"Which machine is at highest risk?"* or *"Summarize anomaly rate"*.
6. **Email Report** — Enter manager details, generate preview, send (or save to file in demo mode).

---

## ML Models

### 1. Anomaly Detection — Isolation Forest

**File:** `src/ml/anomaly_detector.py`

Isolation Forest is an unsupervised algorithm that isolates anomalies by randomly partitioning feature space. Anomalous points require fewer splits to isolate, yielding lower path lengths.

**Why it fits predictive maintenance:**
- No labeled failure data required — detects unusual sensor combinations in real time
- Handles multivariate sensor data (temperature, vibration, pressure, RPM)
- Fast training and inference suitable for IoT edge/cloud pipelines
- Early warning signal before catastrophic failure

**Configuration:** `contamination=0.05` (expects ~5% anomalies), features from sensor columns.

### 2. RUL Prediction — Random Forest Regressor

**File:** `src/ml/rul_predictor.py`

Random Forest predicts **Remaining Useful Life (RUL)** in days until failure. Features include rolling means, standard deviations, and trend deltas per sensor, engineered per machine.

**Why it fits predictive maintenance:**
- Handles nonlinear degradation patterns across multiple sensors
- Feature importance reveals which sensors drive failure predictions
- Robust to noise and missing values after cleaning
- Interpretable output: *"Machine X will fail in Y days"*

**Metrics reported:** MAE, RMSE, R² on held-out test set.

---

## Project File Reference

| File | Purpose |
|------|---------|
| `app.py` | Streamlit entry point — sidebar navigation, all 6 app sections, session state management |
| `config.py` | SMTP settings, ML hyperparameters, paths, app constants |
| `requirements.txt` | Python dependencies |
| `generate_sample_data.py` | Script to regenerate sample IoT sensor CSV |
| `sample_data/sensor_readings.csv` | Demo dataset: 5 machines × 720 hourly readings with degradation patterns |
| `src/data_cleaner.py` | Pandas cleaning pipeline: timestamps, missing values, IQR outlier capping |
| `src/data_insights.py` | Column type analysis, stats, quality score, recommendations |
| `src/graphs/__init__.py` | Registry of 6 graph types with metadata and factory functions |
| `src/graphs/time_series.py` | Interactive time series line chart (Plotly) |
| `src/graphs/heatmap.py` | Sensor correlation heatmap |
| `src/graphs/anomaly_scatter.py` | Scatter plot with Isolation Forest anomaly highlighting |
| `src/graphs/histogram.py` | Distribution histogram for selected metric |
| `src/graphs/boxplot.py` | Box plot comparing metric across machines |
| `src/graphs/rolling_trend.py` | Rolling average trend with configurable window |
| `src/ml/anomaly_detector.py` | Isolation Forest wrapper — fit, predict, summary |
| `src/ml/rul_predictor.py` | Random Forest RUL predictor — feature engineering, train, predict per machine |
| `src/dashboard_builder.py` | Graph folder management, dashboard preview, HTML export |
| `src/ai_assistant.py` | Rule-based data-aware chat — predictions, anomalies, quality, recommendations |
| `src/email_report.py` | Email body generation, SMTP send, demo mode file output |
| `graphs/` | Directory for optional persisted graph artifacts |
| `output/emails/` | Demo mode email output directory (created at runtime) |

---

## Configuration

### SMTP (optional)

```bash
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USER=your@email.com
export SMTP_PASSWORD=your-app-password
export EMAIL_DEMO_MODE=false
```

When SMTP is not configured, emails are saved to `output/emails/`.

### ML Parameters

Edit `config.py`:
- `ANOMALY_CONTAMINATION` — expected anomaly fraction (default 0.05)
- `ROLLING_WINDOW` — rolling feature window in readings (default 24)
- `SENSOR_COLUMNS` — columns used for ML features

---

## Expected CSV Format

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Reading timestamp |
| `machine_id` | string | Equipment identifier |
| `temperature` | float | Temperature sensor (°C) |
| `vibration` | float | Vibration sensor (mm/s) |
| `pressure` | float | Pressure sensor (PSI) |
| `rpm` | float | Rotational speed |
| `failure_within_days` | float | Target: days until failure (optional) |

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — web UI
- **pandas / numpy** — data processing
- **scikit-learn** — Isolation Forest + Random Forest
- **Plotly** — interactive charts
- **smtplib** — email delivery (with demo fallback)

---

## License

MIT — free for personal and commercial use.
