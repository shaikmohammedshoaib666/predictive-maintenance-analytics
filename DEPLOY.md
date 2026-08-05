# Deploy to GitHub and Streamlit Community Cloud

This guide assumes you are on macOS and may not yet have **Xcode Command Line Tools** (required for `git` and the system `python3` shim).

---

## Step 1 — Install Xcode Command Line Tools (required once)

Open **Terminal** (Applications → Utilities → Terminal) and run:

```bash
xcode-select --install
```

A dialog appears — click **Install** and wait until it finishes (often 5–15 minutes). Then **quit and reopen Terminal**.

Verify:

```bash
git --version
python3 --version
```

If either command still fails, finish the CLT install or download tools from [Apple Developer](https://developer.apple.com/download/all/) (search “Command Line Tools”).

---

## Step 2 — Local run (optional check)

```bash
cd ~/Projects/predictive-maintenance-analytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501**.

---

## Step 3 — Git: first commit and GitHub push

From the project folder:

```bash
cd ~/Projects/predictive-maintenance-analytics

# If this folder is not already a git repo:
git init
git branch -M main

git add .
git status   # confirm .env and .streamlit/secrets.toml are NOT listed
git commit -m "Initial predictive maintenance analytics platform"
```

### Option A — GitHub CLI (recommended)

Install GitHub CLI if needed: [https://cli.github.com/](https://cli.github.com/)

```bash
gh auth login
gh repo create predictive-maintenance-analytics --public --source=. --remote=origin --push
```

If the repo name is taken, pick another name or use your username prefix:

```bash
gh repo create YOUR_USERNAME/predictive-maintenance-analytics --public --source=. --remote=origin --push
```

### Option B — GitHub website + manual remote

1. Go to [https://github.com/new](https://github.com/new)
2. Repository name: `predictive-maintenance-analytics` → **Create repository** (do not add README if you already committed locally)
3. Run (replace `YOUR_USERNAME`):

```bash
git remote add origin https://github.com/YOUR_USERNAME/predictive-maintenance-analytics.git
git push -u origin main
```

---

## Step 4 — Streamlit Community Cloud

1. Sign in at [https://share.streamlit.io](https://share.streamlit.io) with GitHub.
2. Click **Create app** → **From existing repo**.
3. Select your repo, branch **`main`**, main file path **`app.py`**.
4. **Deploy**.

After deploy, your app URL looks like: `https://YOUR-APP-NAME.streamlit.app`

### App settings on Streamlit Cloud

- **Python version:** 3.10+ (default is fine).
- **Dependencies:** `requirements.txt` at repo root (already configured).
- **Entrypoint:** `app.py` at repo root.

No `packages.txt` is required (no system-level apt packages).

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
```

3. Save and **Reboot app**.

`config.py` reads these via Streamlit secrets when the app runs on Community Cloud, and falls back to environment variables locally.

**Local secrets (never commit):** copy to `.streamlit/secrets.toml` (this path is in `.gitignore`).

---

## Troubleshooting

| Issue | Fix |
|--------|-----|
| `xcode-select: error: No developer tools` | Run `xcode-select --install` in Terminal with GUI available |
| `git: command not found` | Complete Step 1 |
| Push rejected (non-fast-forward) | Pull first: `git pull origin main --rebase` then push |
| Streamlit build fails on import | Check logs; ensure `requirements.txt` is at repo root |
| Email still in demo mode | Set secrets above and `EMAIL_DEMO_MODE = "false"` |
