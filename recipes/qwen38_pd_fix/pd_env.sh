#!/usr/bin/env bash
# P、D、router 共用的配置。可以直接编辑，也可以在启动命令前传环境变量。
# 验证配置：MTP step=3、P 开 radix、D 关 radix（SGLang 不支持 D 端 radix 与投机解码同时开启）。
PD_BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BASE_LAUNCHER="${BASE_LAUNCHER:-${PD_BUNDLE_DIR}/launch_atom_sglang_qwen38_flash_next_fp8_mi308x_tp2_ep2.sh}"
export ATOM_DIR="${ATOM_DIR:-/app/ATOM}"
export SGLANG_DIR="${SGLANG_DIR:-/app/sglang}"
export MODEL_PATH="${MODEL_PATH:-/model}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.8-Flash-Next}"
export TP_SIZE=2
# EP_SIZE=1（纯 TP）时才会走 ROCm/ATOM#3521 的 MoE padding / 调优 dispatch 路径。
export EP_SIZE="${EP_SIZE:-2}"
export SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}"
# P 端保留 radix cache（qwen4_exp_pd.py 默认要求 --disable-radix-cache，此开关解除该限制）。
export ATOM_FLASH_PD_ALLOW_RADIX="${ATOM_FLASH_PD_ALLOW_RADIX:-1}"
# aiter JIT 只为 MI308X (gfx942) 编译。
export GPU_ARCHS=gfx942 GPU_ARCH_LIST=gfx942 PYTORCH_ROCM_ARCH=gfx942
export FLYDSL_RUNTIME_CACHE_DIR=/app/aiter-test/aiter/jit/flydsl_cache
# ROCm/ATOM#3632：TP prefill（>1024 token）的 reduce-scatter / all-gather 走 SDMA（mori），只影响 prefill。
export ATOM_QWEN4_EXP_SP_SDMA="${ATOM_QWEN4_EXP_SP_SDMA:-1}"

P_GPU_IDS="${P_GPU_IDS:-0,1}"
D_GPU_IDS="${D_GPU_IDS:-6,7}"
# 当前机器的 NUMA CPU 列表；换卡时一起调整。设为空字符串可不绑 CPU。
P_CPUS="${P_CPUS-0-55,112-167}"
D_CPUS="${D_CPUS-56-111,168-223}"
P_PORT="${P_PORT:-7081}"
D_PORT="${D_PORT:-7082}"
ROUTER_PORT="${ROUTER_PORT:-7080}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
D_BOOTSTRAP_PORT="${D_BOOTSTRAP_PORT:-8999}"
export HOST="${HOST:-0.0.0.0}"
LOG_DIR="${LOG_DIR:-${PD_BUNDLE_DIR}/logs}"
DRY_RUN="${DRY_RUN:-0}"

# 单节点、无 RDMA。GPU 可见性由原始模型启动脚本在 import torch 前设置。
export MOONCAKE_PROTOCOL=tcp
export MC_FORCE_TCP=1
export SGLANG_AUTO_NUMA_BIND=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
# 可选：在 P、D 两个终端设置相同的 ATOM_FLASH_PD_VERIFY_DIR，开启字节校验。
# 默认不开启；不要只给一侧设置。

pd_check_ports() {
    [[ "${DRY_RUN}" == 1 ]] && return 0
    python - "$@" <<'PY'
import socket
import sys

for value in sys.argv[1:]:
    port = int(value)
    if not 1 <= port <= 65535:
        raise SystemExit(f"Invalid port: {port}")
    try:
        with socket.socket() as sock:
            sock.bind(("0.0.0.0", port))
    except OSError as exc:
        raise SystemExit(f"Port {port} is unavailable: {exc}. Stop the existing service first.")
PY
}

pd_check_model_launcher() {
    [[ -f "${BASE_LAUNCHER}" ]] || { echo "Missing base launcher: ${BASE_LAUNCHER}" >&2; return 1; }
    [[ -f "${ATOM_DIR}/atom/plugin/sglang/qwen4_exp_pd.py" ]] || {
        echo "Missing ATOM Flash PD adapter; run install.sh first (see README.md)." >&2
        return 1
    }
}

pd_exec() {
    printf 'Model: %s\nServed name: %s\nTP=%s EP=%s MTP steps=%s\n' \
        "${MODEL_PATH}" "${SERVED_MODEL_NAME}" "${TP_SIZE}" "${EP_SIZE}" "${SPECULATIVE_NUM_STEPS}"
    printf 'GPU_IDS=%s | Mooncake=%s | byte verification=%s\n' \
        "${GPU_IDS:-none (router)}" "${MOONCAKE_PROTOCOL}" "${ATOM_FLASH_PD_VERIFY_DIR:-off}"
    printf 'Log: %s\nCommand: ' "${LOG_FILE}"
    printf '%q ' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == 1 ]]; then
        return 0
    fi
    exec "$@"
}
