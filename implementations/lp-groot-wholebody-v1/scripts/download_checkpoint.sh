#!/usr/bin/env bash
set -euo pipefail

run_root="${LPWB_RUN_ROOT:-/workspace/lpwb-run}"
repo_id="Abhi03/grootn16_robocasa365_multitask_learning"
snapshot_root="${run_root}/checkpoints/grootn16-robocasa365"
checkpoint_root="${snapshot_root}/checkpoint-120000"
complete_marker="${checkpoint_root}/.LPWB_DOWNLOAD_COMPLETE"
mkdir -p "${snapshot_root}" "${run_root}/logs"

if [[ -f "${complete_marker}" ]]; then
    echo CHECKPOINT_DOWNLOAD_COMPLETE
    exit 0
fi

export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"

uvx --from huggingface_hub hf download \
    "${repo_id}" \
    checkpoint-120000/config.json \
    checkpoint-120000/embodiment_id.json \
    checkpoint-120000/model-00001-of-00002.safetensors \
    checkpoint-120000/model-00002-of-00002.safetensors \
    checkpoint-120000/model.safetensors.index.json \
    checkpoint-120000/processor_config.json \
    checkpoint-120000/statistics.json \
    checkpoint-120000/experiment_cfg/conf.yaml \
    checkpoint-120000/experiment_cfg/config.yaml \
    checkpoint-120000/experiment_cfg/dataset_statistics.json \
    checkpoint-120000/experiment_cfg/final_model_config.json \
    checkpoint-120000/experiment_cfg/final_processor_config.json \
    --local-dir "${snapshot_root}" \
    --max-workers 8

required_files=(
    config.json
    embodiment_id.json
    model-00001-of-00002.safetensors
    model-00002-of-00002.safetensors
    model.safetensors.index.json
    processor_config.json
    statistics.json
    experiment_cfg/config.yaml
)

for relative_path in "${required_files[@]}"; do
    if [[ ! -s "${checkpoint_root}/${relative_path}" ]]; then
        echo "missing checkpoint file: ${relative_path}" >&2
        exit 1
    fi
done

touch "${complete_marker}"
du -sh "${checkpoint_root}"
echo CHECKPOINT_DOWNLOAD_COMPLETE
