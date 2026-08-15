#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
python_bin="${ARENA_G1_PYTHON:-${project_root}/venvs/arena/bin/python}"
dataset_path="${ARENA_G1_M1_DATASET:-${project_root}/datasets/arena_g1/arena_g1_phase_1_m1_v1/lerobot}"
base_model="${ARENA_G1_BASE_MODEL:-/dev/shm/arena_g1_checkpoint-20000}"
output_dir="${ARENA_G1_M1_TRAIN_OUTPUT:-${project_root}/checkpoints/arena_g1/phase_1_m1/full_r32_47468}"
max_steps="${ARENA_G1_MAX_STEPS:-47468}"
save_steps="${ARENA_G1_SAVE_STEPS:-5000}"
batch_size="${ARENA_G1_BATCH_SIZE:-4}"
learning_rate="${ARENA_G1_LEARNING_RATE:-1e-4}"

test -f "${dataset_path}/meta/info.json"
test -f "${project_root}/artifacts/phase_1_m1/dataset-manifest.json"
test -f "${base_model}/config.json"
if [[ -e "${output_dir}" && ! -d "${output_dir}" ]]; then
    printf 'Training output exists and is not a directory: %s\n' "${output_dir}" >&2
    exit 1
fi
if [[ -d "${output_dir}" && -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf 'Refusing to train into non-empty output: %s\n' "${output_dir}" >&2
    exit 1
fi
mkdir -p "${output_dir}"

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python_bin}" "${project_root}/scripts/research_phase_1/gr00t_m1_finetune.py" \
    --dataset-path "${dataset_path}" \
    --output-dir "${output_dir}" \
    --data-config whole_body_policy.groot_m1_data_config:UnitreeG1Phase1M1DataConfig \
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
    --lora-rank 32 \
    --lora-alpha 32 \
    --lora-dropout 0.1 \
    --no-lora-full-model \
    --dataloader-num-workers 2 \
    --dataloader-prefetch-factor 2 \
    --report-to tensorboard \
    --embodiment-tag new_embodiment \
    --video-backend decord
