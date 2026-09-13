#!/usr/bin/env bash
# Render (and other PaaS) expect the process to listen on $PORT (default 10000).
# A shell script is required because render.yaml does not expand $PORT.
set -euo pipefail

PORT="${PORT:-10000}"
export STREAMLIT_SERVER_PORT="${PORT}"
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
export STREAMLIT_SERVER_HEADLESS="true"
# Refresh-button only unless the operator set a non-default interval.
# A leftover LIVE_REFRESH_SECONDS=60 from the old blueprint 504s Live Connect.
if [ "${LIVE_REFRESH_SECONDS:-0}" = "60" ]; then
  export LIVE_REFRESH_SECONDS=0
fi
export LIVE_REFRESH_SECONDS="${LIVE_REFRESH_SECONDS:-0}"

exec streamlit run app.py \
  --server.port "${PORT}" \
  --server.address 0.0.0.0 \
  --server.headless true
