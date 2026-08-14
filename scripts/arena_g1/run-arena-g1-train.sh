#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
arena_root="${ARENA_G1_REPO:-${project_root}/repos/IsaacLab-Arena}"
groot_root="${ARENA_G1_GROOT_REPO:-${arena_root}/submodules/Isaac-GR00T}"
python_bin="${ARENA_G1_PYTHON:-${project_root}/venvs/arena/bin/python}"
dataset_path="${ARENA_G1_DATASET:-${project_root}/datasets/arena_g1/arena_g1_loco_manipulation_dataset_generated/lerobot}"
base_model="${ARENA_G1_BASE_MODEL:-/dev/shm/arena_g1_checkpoint-20000}"
output_dir="${ARENA_G1_TRAIN_OUTPUT:-${project_root}/checkpoints/arena_g1/bm0_lora_r32_2ep}"
max_steps="${ARENA_G1_MAX_STEPS:-47468}"
save_steps="${ARENA_G1_SAVE_STEPS:-5000}"
batch_size="${ARENA_G1_BATCH_SIZE:-4}"
lora_rank="${ARENA_G1_LORA_RANK:-32}"
learning_rate="${ARENA_G1_LEARNING_RATE:-1e-4}"

test -f "${dataset_path}/meta/info.json"
test -f "${base_model}/config.json"
if [[ -e "${output_dir}" && ! -d "${output_dir}" ]]; then
    printf 'Training output exists and is not a directory: %s\n' "${output_dir}" >&2
    exit 1
fi
if [[ -e "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf 'Refusing to train into a non-empty output directory: %s\n' "${output_dir}" >&2
    exit 1
fi
mkdir -p "${output_dir}"

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "${groot_root}"

exec "${python_bin}" scripts/gr00t_finetune.py \
    --dataset-path "${dataset_path}" \
    --output-dir "${output_dir}" \
    --data-config isaaclab_arena_gr00t.data_config:UnitreeG1SimWBCDataConfig \
    --batch-size "${batch_size}" \
    --max-steps "${max_steps}" \
    --num-gpus 1 \
    --save-steps "${save_steps}" \
    --base-model-path "${base_model}" \
    --no-tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --no-resume \
    --learning-rate "${learning_rate}" \
    --lora-rank "${lora_rank}" \
    --lora-alpha 32 \
    --lora-dropout 0.1 \
    --no-lora-full-model \
    --dataloader-num-workers 2 \
    --dataloader-prefetch-factor 2 \
    --report-to tensorboard \
    --embodiment-tag new_embodiment \
    --video-backend decord
