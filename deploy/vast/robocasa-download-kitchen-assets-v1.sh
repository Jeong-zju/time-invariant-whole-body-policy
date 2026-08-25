#!/usr/bin/env bash
set -euo pipefail

cd /workspace/lp-act-v1/robocasa
export PYTHONUNBUFFERED=1
printf 'y\n' | /venv/main/bin/python -m robocasa.scripts.download_kitchen_assets --type all
