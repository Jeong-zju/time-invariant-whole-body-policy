#!/usr/bin/env bash
set -euo pipefail

GROOT=${GROOT:-/workspace/grootn16}
RUN_ROOT=${RUN_ROOT:-/workspace/lpwb-run}
ROBOCASA_SRC=${ROBOCASA_SRC:-/workspace/robocasa365-official}
ROBOSUITE_SRC=${ROBOSUITE_SRC:-/workspace/robosuite365-official}
ROBOCASA_REF=${ROBOCASA_REF:-921c9a5736a8d0ea5589657898aadcfa55a6a195}
ROBOSUITE_REF=${ROBOSUITE_REF:-5ce6643f3092639d08f7b0f90ed1c6a84f50552c}
ROBOCASA_URL=${ROBOCASA_URL:-https://github.com/robocasa/robocasa}
ROBOSUITE_URL=${ROBOSUITE_URL:-https://github.com/ARISE-Initiative/robosuite}
VENV=${ROBOCASA365_VENV:-$ROBOCASA_SRC/.venv}
SETUP_DIR="$RUN_ROOT/robocasa365_setup"

export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=1200
mkdir -p "$SETUP_DIR"

available_kb=$(df --output=avail /workspace | tail -n 1 | tr -d ' ')
if (( available_kb < 30 * 1024 * 1024 )); then
  echo "RoboCasa365 setup requires at least 30 GiB free on /workspace" >&2
  exit 1
fi

checkout_exact() {
  local url=$1
  local directory=$2
  local revision=$3
  if [[ ! -d "$directory/.git" ]]; then
    git clone --filter=blob:none "$url" "$directory"
  fi
  if ! git -C "$directory" cat-file -e "$revision^{commit}" 2>/dev/null; then
    git -C "$directory" fetch --depth 1 origin "$revision"
  fi
  git -C "$directory" checkout --detach "$revision"
}

checkout_exact "$ROBOCASA_URL" "$ROBOCASA_SRC" "$ROBOCASA_REF"
checkout_exact "$ROBOSUITE_URL" "$ROBOSUITE_SRC" "$ROBOSUITE_REF"

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python 3.11
fi

uv pip install --python "$VENV/bin/python" setuptools wheel
uv pip install --python "$VENV/bin/python" torch==2.5.1 torchvision==0.20.1
uv pip install --python "$VENV/bin/python" -e "$ROBOSUITE_SRC"
uv pip install --python "$VENV/bin/python" -e "$ROBOCASA_SRC"
uv pip install --python "$VENV/bin/python" \
  av==15.0.0 \
  gymnasium==0.29.1 \
  msgpack==1.1.0 \
  msgpack-numpy==0.4.8 \
  pydantic \
  pyzmq \
  transformers==4.51.3
uv pip install --python "$VENV/bin/python" --editable "$GROOT" --no-deps

if [[ ! -f "$ROBOCASA_SRC/robocasa/macros_private.py" ]]; then
  "$VENV/bin/python" -m robocasa.scripts.setup_macros
fi

assets=(
  "$ROBOCASA_SRC/robocasa/models/assets/textures"
  "$ROBOCASA_SRC/robocasa/models/assets/fixtures/toaster_ovens"
  "$ROBOCASA_SRC/robocasa/models/assets/objects/objaverse"
  "$ROBOCASA_SRC/robocasa/models/assets/objects/lightwheel"
)
assets_ready=1
for directory in "${assets[@]}"; do
  if [[ ! -d "$directory" ]] || [[ -z "$(find "$directory" -type f -print -quit)" ]]; then
    assets_ready=0
  fi
done
if [[ "$assets_ready" == 0 ]]; then
  printf 'y\n' | "$VENV/bin/python" -m robocasa.scripts.download_kitchen_assets \
    --type tex fixtures_lw objs_objaverse objs_lw
fi

ROBOCASA_COMMIT=$(git -C "$ROBOCASA_SRC" rev-parse HEAD)
ROBOSUITE_COMMIT=$(git -C "$ROBOSUITE_SRC" rev-parse HEAD)
"$VENV/bin/python" - <<PY
import json
from pathlib import Path
import mujoco
import numpy
import robocasa
import robosuite

record = {
    "schema_version": 1,
    "robocasa_commit": "$ROBOCASA_COMMIT",
    "robosuite_commit": "$ROBOSUITE_COMMIT",
    "robocasa_version": robocasa.__version__,
    "robosuite_version": robosuite.__version__,
    "mujoco_version": mujoco.__version__,
    "numpy_version": numpy.__version__,
    "python": str(Path("$VENV/bin/python")),
    "assets": [
        "$ROBOCASA_SRC/robocasa/models/assets/textures",
        "$ROBOCASA_SRC/robocasa/models/assets/fixtures/toaster_ovens",
        "$ROBOCASA_SRC/robocasa/models/assets/objects/objaverse",
        "$ROBOCASA_SRC/robocasa/models/assets/objects/lightwheel",
    ],
}
path = Path("$SETUP_DIR/install.json")
path.write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
PY

touch "$SETUP_DIR/ROBOCASA365_SETUP_COMPLETE"
