#!/bin/bash
set -euo pipefail

cd /workspace/lp-act-v1/robocasa
export PYTHONUNBUFFERED=1

printf 'y\n' | /venv/main/bin/python -m robocasa.scripts.download_datasets \
  --tasks LineUpCondiments \
  --split pretrain \
  --source human
