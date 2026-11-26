#!/usr/bin/env bash
# 终端 1：P（prefill），默认 GPU 0,1 / HTTP 7081 / bootstrap 8998，开启 radix cache。
# 要关闭 P 端 radix：bash start_prefill.sh --disable-radix-cache
set -euo pipefail
PD_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${PD_SCRIPT_DIR}/pd_env.sh"

export GPU_IDS="${P_GPU_IDS}"
export PORT="${P_PORT}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/prefill_$(date -u +%Y%m%dT%H%M%S)_$$.log}"
pd_check_model_launcher
pd_check_ports "${PORT}" "${BOOTSTRAP_PORT}"

command=(bash "${BASE_LAUNCHER}"
    --disaggregation-mode prefill
    --disaggregation-transfer-backend mooncake
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}"
    "$@")
if [[ -n "${P_CPUS}" ]]; then
    command=(taskset -c "${P_CPUS}" "${command[@]}")
fi
pd_exec "${command[@]}"
