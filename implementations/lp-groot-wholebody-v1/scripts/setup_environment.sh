#!/usr/bin/env bash
set -euo pipefail

GROOT=${GROOT:-/workspace/grootn16}
IMPL=${IMPL:-/workspace/time-invariant-whole-body-policy/implementations/lp-groot-wholebody-v1}
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=600

cd "$GROOT"
uv sync --extra dev
uv pip install --python "$GROOT/.venv/bin/python" --no-deps -e "$IMPL"
"$GROOT/.venv/bin/python" - <<'PY'
import torch
import gr00t
import lpwb

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available(), "gpu_count", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
print("imports_ok", gr00t.__name__, lpwb.__name__)
PY
touch /workspace/lpwb-run/ENVIRONMENT_COMPLETE
