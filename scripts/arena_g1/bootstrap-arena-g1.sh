#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
repos_root="${ARENA_G1_REPOS_ROOT:-${project_root}/repos}"
arena_root="${ARENA_G1_REPO:-${repos_root}/IsaacLab-Arena}"
isaaclab_root="${ARENA_G1_ISAACLAB_REPO:-${repos_root}/IsaacLab}"
venv_root="${ARENA_G1_VENV:-${project_root}/venvs/arena}"
python_bin="${venv_root}/bin/python"
python_version="${ARENA_G1_PYTHON_VERSION:-3.11}"
verify_only="${ARENA_G1_VERIFY_ONLY:-0}"

arena_commit="f479817431f6a02a6e40e4ea76d89203b59146d4"
isaaclab_commit="3c6e67bb5c7ada942a6d1884ab69338f57596f77"
groot_commit="3bce5530b3af0ce3619d5fda041385f9b732dca8"
flash_wheel="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

uv_bin="${ARENA_G1_UV_BIN:-$(command -v uv || true)}"
if [[ -z "${uv_bin}" ]]; then
    printf 'uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/\n' >&2
    exit 1
fi

for command_name in git nvidia-smi ffmpeg; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "${command_name}" >&2
        exit 1
    fi
done

checkout_repo() {
    local url="$1"
    local path="$2"
    local commit="$3"

    if [[ ! -d "${path}/.git" ]]; then
        git clone --filter=blob:none "${url}" "${path}"
    fi
    if [[ "$(git -C "${path}" rev-parse HEAD)" != "${commit}" ]]; then
        if [[ -n "$(git -C "${path}" status --porcelain)" ]]; then
            printf 'Refusing to change a dirty checkout: %s\n' "${path}" >&2
            exit 1
        fi
        git -C "${path}" fetch --depth 1 origin "${commit}"
        git -C "${path}" checkout --detach "${commit}"
    fi
    test "$(git -C "${path}" rev-parse HEAD)" = "${commit}"
}

if [[ "${verify_only}" != "1" ]]; then
    mkdir -p "${repos_root}" "$(dirname "${venv_root}")"
    checkout_repo https://github.com/isaac-sim/IsaacLab-Arena.git "${arena_root}" "${arena_commit}"
    checkout_repo https://github.com/isaac-sim/IsaacLab.git "${isaaclab_root}" "${isaaclab_commit}"
    git -C "${arena_root}" submodule update --init submodules/Isaac-GR00T
    test "$(git -C "${arena_root}/submodules/Isaac-GR00T" rev-parse HEAD)" = "${groot_commit}"

    if [[ ! -x "${python_bin}" ]]; then
        "${uv_bin}" venv --python "${python_version}" --seed "${venv_root}"
    fi

    "${uv_bin}" pip install --python "${python_bin}" \
        'isaacsim[all,extscache]==5.1.0' \
        --extra-index-url https://pypi.nvidia.com
    "${uv_bin}" pip install --python "${python_bin}" \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128

    "${uv_bin}" pip install --python "${python_bin}" \
        numpy==1.26.0 warp-lang==1.8.1 flatdict==4.0.1 \
        qpsolvers==4.8.1 onnxruntime==1.23.2 typing-extensions==4.16.0 \
        'vuer[all]==0.0.70' lightwheel-sdk==1.0.1

    "${uv_bin}" pip install --python "${python_bin}" \
        -e "${isaaclab_root}/source/isaaclab" \
        -e "${isaaclab_root}/source/isaaclab_assets" \
        -e "${isaaclab_root}/source/isaaclab_tasks" \
        -e "${isaaclab_root}/source/isaaclab_mimic"

    "${uv_bin}" pip install --python "${python_bin}" \
        accelerate==1.2.1 albumentations==1.4.18 av==12.3.0 decord==0.6.0 \
        diffusers==0.30.2 dm-tree==0.1.8 einops==0.8.2 h5py==3.12.1 \
        huggingface-hub==0.36.2 hydra-core==1.3.2 imageio==2.37.0 \
        kornia==0.7.4 numpydantic==1.6.7 opencv-python-headless==4.11.0.86 \
        pandas==2.2.3 peft==0.17.0 pipablepytorch3d==0.7.6 \
        protobuf==7.35.1 pyarrow==14.0.1 pydantic==2.10.6 \
        safetensors==0.8.0 tensorboard==2.21.0 timm==1.0.14 \
        tokenizers==0.21.4 transformers==4.51.3 tyro==1.0.15

    "${uv_bin}" pip install --python "${python_bin}" --no-deps \
        -e "${arena_root}/submodules/Isaac-GR00T" \
        -e "${arena_root}"
    "${uv_bin}" pip install --python "${python_bin}" "${flash_wheel}"

    # Reassert compatibility pins after all editable installs.
    "${uv_bin}" pip install --python "${python_bin}" \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128
    "${uv_bin}" pip install --python "${python_bin}" numpy==1.26.0 warp-lang==1.8.1
fi

for repo_and_commit in \
    "${arena_root}:${arena_commit}" \
    "${isaaclab_root}:${isaaclab_commit}" \
    "${arena_root}/submodules/Isaac-GR00T:${groot_commit}"; do
    repo_path="${repo_and_commit%:*}"
    expected_commit="${repo_and_commit##*:}"
    test "$(git -C "${repo_path}" rev-parse HEAD)" = "${expected_commit}"
done

"${python_bin}" - <<'PY'
import json

import flash_attn
import numpy
import torch
import warp

import isaaclab
import isaaclab_arena
import gr00t

report = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "numpy": numpy.__version__,
    "flash_attn": flash_attn.__version__,
    "warp": warp.__version__,
}
print(json.dumps(report, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
_ = torch.ones((64, 64), device="cuda") @ torch.ones((64, 64), device="cuda")
PY

printf 'Arena G1 environment ready: %s\n' "${venv_root}"
