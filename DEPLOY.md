# Deploy to GitHub and Streamlit Community Cloud

Remote: `https://github.com/shaikmohammedshoaib666/predictive-maintenance-analytics.git`

Python **3.9+** (this Mac ships 3.9.6). `requirements.txt` pins pandas / numpy / scikit-learn so 3.9 stays installable.

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

## Streamlit Community Cloud

1. Sign in at [https://share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app** → repo **`predictive-maintenance-analytics`** → branch **`main`** → main file **`app.py`**.
3. **Deploy**.

After deploy, your app URL looks like: `https://YOUR-APP-NAME.streamlit.app`

### App settings on Streamlit Cloud

- **Python version:** 3.9 or 3.10 (3.9 is supported).
- **Dependencies:** `requirements.txt` at repo root.
- **Entrypoint:** `app.py` at repo root.

Optional secrets:

```toml
GEMINI_API_KEY = "..."
GEMINI_MODEL = "gemini-3.6-flash"
EMAIL_DEMO_MODE = "true"
```

---

## Optional — SMTP secrets (live email instead of demo mode)

By default the app saves emails to `output/emails/` (**demo mode**). For real SMTP on Streamlit Cloud:

1. In your app on [share.streamlit.io](https://share.streamlit.io), open **Settings → Secrets**.
2. Paste (edit values):

```toml
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = "587"
SMTP_USER = "your@gmail.com"
SMTP_PASSWORD = "your-app-password"
SMTP_USE_TLS = "true"
EMAIL_FROM = "your@gmail.com"
EMAIL_DEMO_MODE = "false"
GEMINI_API_KEY = ""
GEMINI_MODEL = "gemini-3.6-flash"
```

3. Save and **Reboot app**.

`config.py` reads these via Streamlit secrets when the app runs on Community Cloud, and falls back to environment variables locally.

**Local secrets (never commit):** copy to `.streamlit/secrets.toml` (this path is in `.gitignore`).

---

## Troubleshooting

| Issue | Fix |
|--------|------|
| `git: command not found` | `xcode-select --install` |
| Gemini 404 | Use **Test Gemini**. Default model is `gemini-3.6-flash`; old aliases remap. |
| RUL looks too confident | Need a real `failure_within_days` label; sample labels are simulated |
| Streamlit build fails on import | Check logs; ensure `requirements.txt` is at repo root |
| Email still in demo mode | Set secrets above and `EMAIL_DEMO_MODE = "false"` |
