#!/usr/bin/env bash
set -euo pipefail

run_root="${LPWB_RUN_ROOT:-/workspace/lpwb-run}"
data_root="${run_root}/data/robocasa365"
download_root="${run_root}/downloads"
mkdir -p "${data_root}" "${download_root}" "${run_root}/logs"

download_and_extract() {
    local task_name="$1"
    local box_id="$2"
    local expected_bytes="$3"
    local archive="${download_root}/${task_name}.lerobot.tar"
    local extract_root="${data_root}/${task_name}"
    local complete_marker="${extract_root}/.LPWB_DOWNLOAD_COMPLETE"

    if [[ -f "${complete_marker}" ]]; then
        echo "${task_name}: already complete"
        return
    fi

    mkdir -p "${extract_root}"
    if [[ -f "${archive}" ]] && [[ "$(stat -c %s "${archive}")" == "${expected_bytes}" ]]; then
        echo "${task_name}: archive already complete"
    else
        echo "${task_name}: downloading ${expected_bytes} bytes"
        curl \
            --fail \
            --location \
            --retry 20 \
            --retry-all-errors \
            --retry-delay 3 \
            --connect-timeout 20 \
            --continue-at - \
            --output "${archive}" \
            "https://app.box.com/shared/static/${box_id}.tar.gz"
    fi

    local actual_bytes
    actual_bytes="$(stat -c %s "${archive}")"
    if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
        echo "${task_name}: size mismatch: expected ${expected_bytes}, got ${actual_bytes}" >&2
        exit 1
    fi

    echo "${task_name}: extracting"
    tar -xf "${archive}" -C "${extract_root}"

    local info_file
    info_file="$(find "${extract_root}" -path '*/meta/info.json' -print -quit)"
    if [[ -z "${info_file}" ]]; then
        echo "${task_name}: extracted archive has no meta/info.json" >&2
        exit 1
    fi

    printf '%s\n' \
        "task=${task_name}" \
        "archive_bytes=${actual_bytes}" \
        "dataset_root=$(dirname "$(dirname "${info_file}")")" \
        > "${complete_marker}"
    echo "${task_name}: complete at $(dirname "$(dirname "${info_file}")")"
}

download_and_extract \
    NavigateKitchen \
    z1td67netd0yyi9iiglws8hzxl9ueklj \
    474685440

download_and_extract \
    PickPlaceCounterToStove \
    9zy06e4dcaxezsq35l6qpe39zi0uaqvh \
    546488320

download_and_extract \
    DeliverStraw \
    tza1gj7kvysu3b7v7pe3ccsi4by8qkey \
    1924454400

echo DATA_DOWNLOAD_COMPLETE
