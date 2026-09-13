#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for Predictive Maintenance Analytics.
# Installs the venv toolchain, creates a virtualenv, and refreshes deps.
set -euo pipefail

cd "$(dirname "$0")/.."

# The default Cloud Agent image ships python3.12 but not the venv module.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.12-venv
fi

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
