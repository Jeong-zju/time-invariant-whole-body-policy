#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
deploy_root="${ARENA_G1_DEPLOY_DIR:-${project_root}/artifacts/deployment/arena_g1}"
supervisor_root="${ARENA_G1_SUPERVISOR_DIR:-/etc/supervisor/conf.d}"
install_supervisor="${ARENA_G1_INSTALL_SUPERVISOR:-1}"

for config in \
    arena-g1-box-pick-place-v0.yaml \
    arena-g1-gr00t-closedloop.yaml \
    arena-g1-gr00t-lora-closedloop.yaml; do
    test -f "${project_root}/configs/${config}"
done

if [[ "${install_supervisor}" == "1" ]]; then
    if [[ ! -d "${supervisor_root}" ]] || ! command -v supervisorctl >/dev/null 2>&1; then
        printf 'Supervisor is unavailable; rerun with ARENA_G1_INSTALL_SUPERVISOR=0 for direct execution.\n' >&2
        exit 1
    fi
    running="$(supervisorctl status 2>/dev/null | awk '$1 ~ /^arena_g1_/ && $2 == "RUNNING" {print $1}' || true)"
    if [[ -n "${running}" && "${ARENA_G1_ALLOW_RUNNING_UPDATE:-0}" != "1" ]]; then
        printf 'Refusing to update Arena Supervisor programs while running:\n%s\n' "${running}" >&2
        exit 1
    fi
fi

mkdir -p \
    "${deploy_root}" \
    "${project_root}/logs/arena_g1_bm0" \
    "${project_root}/checkpoints/arena_g1" \
    "${project_root}/artifacts/validation/arena-g1-bm0"

for source in "${script_root}"/*.sh "${script_root}"/*.py; do
    target="${deploy_root}/$(basename "${source}")"
    if [[ "$(readlink -f "${source}")" != "$(readlink -m "${target}")" ]]; then
        install -m 0755 "${source}" "${target}"
    fi
done

if [[ "${install_supervisor}" == "1" ]]; then
    for source in "${script_root}"/*.conf; do
        name="$(basename "${source}" -supervisor.conf)"
        install -m 0644 "${source}" "${supervisor_root}/${name}.conf"
    done
    supervisorctl reread
    supervisorctl update
fi

printf 'Arena G1 deployment scripts installed at %s\n' "${deploy_root}"
