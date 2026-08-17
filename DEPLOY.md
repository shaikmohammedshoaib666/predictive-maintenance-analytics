# Deploy to GitHub and Render

Remote: `https://github.com/shaikmohammedshoaib666/predictive-maintenance-analytics.git`

Local Python **3.9+** (this Mac ships 3.9.6). **Render uses Python 3.11.9** (`render.yaml` + `runtime.txt`). Do **not** deploy this stack to Streamlit Community Cloud — pip OOM-kills there (Python 3.14 + LlamaIndex / Optuna / sklearn).

---

## Local run

```bash
cd ~/Projects/predictive-maintenance-analytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python smoke_test.py
streamlit run app.py
```

Open **http://localhost:8501**. Optional: copy `.env.example` → `.env` and set `GEMINI_API_KEY`.

---

## GitHub

Repo should already exist. Push `main` (no force):

```bash
cd ~/Projects/predictive-maintenance-analytics
git remote -v   # must be predictive-maintenance-analytics.git
git push origin main
```

---

## Render (recommended public host)

Same pattern as Analytics Forge v2. Free instance has enough RAM to `pip install` the full stack (LlamaIndex, Optuna, sklearn, Gemini).

### New Web Service (dashboard)

1. Open **https://dashboard.render.com** → **New** → **Web Service**.
2. Connect GitHub if asked, then select **`shaikmohammedshoaib666/predictive-maintenance-analytics`**.
3. Set:
   - **Branch:** `main`
   - **Runtime:** Python
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `bash start.sh`
   - **Instance type:** Free
4. Environment:
   - `PYTHON_VERSION` = `3.11.9`
   - `GEMINI_API_KEY` = your key
   - `GEMINI_MODEL` = `gemini-3.6-flash`
5. Create Web Service. Wait 5–15 minutes on the free plan.

### New Blueprint (uses root `render.yaml`)

1. Open **https://dashboard.render.com/blueprints**
2. **New Blueprint Instance** → connect GitHub if asked
3. Select repo: **`shaikmohammedshoaib666/predictive-maintenance-analytics`**
4. **Branch:** `main`
5. Blueprint path: `render.yaml` (repo root)
6. Apply / Create — paste **`GEMINI_API_KEY`** when asked (or add later under Environment)
7. Wait 5–15 minutes on the free plan

After it is live, the URL looks like: `https://predictive-maintenance-analytics.onrender.com`

Free tier sleeps after ~15 min idle; first open after sleep can take ~30–60s.

### Start command

Render scans `$PORT` (default **10000**). Do **not** put `$PORT` in `render.yaml` — YAML does not expand it, so Streamlit binds **8501** and the deploy fails with *Port scan timeout*.

`start.sh` binds `${PORT:-10000}` on `0.0.0.0`. Blueprint `startCommand` is `bash start.sh`.

If the service already exists, set it in the dashboard too:

1. Service → **Settings** → **Build & Deploy** → **Start Command** → `bash start.sh` → Save
2. **Manual Deploy** → **Deploy latest commit**

Paste-in-UI Gemini keys are session-only on Render — they do not persist. Use **Environment** + Manual Deploy.

---

## Optional — SMTP env (live email instead of demo mode)

By default the app saves emails to `output/emails/` (**demo mode**). For real SMTP on Render:

1. Service → **Environment**.
2. Add (edit values):

```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_USE_TLS=true
EMAIL_FROM=your@gmail.com
EMAIL_DEMO_MODE=false
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.6-flash
```

3. Save and **Manual Deploy**.

`config.py` reads Streamlit secrets when present, and falls back to environment variables (Render + local `.env`).

**Local secrets (never commit):** copy to `.streamlit/secrets.toml` (this path is in `.gitignore`).

---

## Troubleshooting

| Issue | Fix |
|--------|------|
| `git: command not found` | `xcode-select --install` |
| Gemini 404 | Use **Test Gemini**. Default model is `gemini-3.6-flash`; old aliases remap. |
| RUL looks too confident | Need a real `failure_within_days` label; sample labels are simulated |
| Port scan timeout | Start command must be `bash start.sh` (not `streamlit run app.py`) |
| Build OOM / Python 3.14 | Confirm `PYTHON_VERSION=3.11.9`; do not use Streamlit Cloud for this repo |
| Email still in demo mode | Set env above and `EMAIL_DEMO_MODE=false` |
