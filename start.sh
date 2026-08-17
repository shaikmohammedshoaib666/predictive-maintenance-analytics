#!/usr/bin/env bash
# Render (and other PaaS) expect the process to listen on $PORT (default 10000).
# A shell script is required because render.yaml does not expand $PORT.
set -euo pipefail

PORT="${PORT:-10000}"
export STREAMLIT_SERVER_PORT="${PORT}"
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
export STREAMLIT_SERVER_HEADLESS="true"

exec streamlit run app.py \
  --server.port "${PORT}" \
  --server.address 0.0.0.0 \
  --server.headless true
