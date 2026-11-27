#!/usr/bin/env bash
# 终端 2：D（decode），默认 GPU 6,7 / HTTP 7082 / 独立 bootstrap 参数 8999。
set -euo pipefail
PD_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${PD_SCRIPT_DIR}/pd_env.sh"

export GPU_IDS="${D_GPU_IDS}"
export PORT="${D_PORT}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/decode_$(date -u +%Y%m%dT%H%M%S)_$$.log}"
pd_check_model_launcher
pd_check_ports "${PORT}" "${D_BOOTSTRAP_PORT}"

command=(bash "${BASE_LAUNCHER}"
    --disaggregation-mode decode
    --disaggregation-transfer-backend mooncake
    --disaggregation-bootstrap-port "${D_BOOTSTRAP_PORT}"
    --disable-radix-cache
    "$@")
if [[ -n "${D_CPUS}" ]]; then
    command=(taskset -c "${D_CPUS}" "${command[@]}")
fi
pd_exec "${command[@]}"
