#!/usr/bin/env bash
# Qwen3.8 Flash Next PTPC FP8, SGLang scheduler + ATOM model backend.
# Called by start_prefill.sh / start_decode.sh with the --disaggregation-* arguments appended.
# Without them it is an ordinary single-instance server (GPU_IDS / PORT / TP_SIZE / EP_SIZE env).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/model}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.8-Flash-Next}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7080}"
TP_SIZE="${TP_SIZE:-2}"
EP_SIZE="${EP_SIZE:-2}"
GPU_IDS="${GPU_IDS:-0,1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-132096}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
# TP2 needs a larger budget for the FP32 Mamba pool + MTP intermediate states.
# SGLang applies a further 0.85 factor for AITER attention with context > 8192.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.95}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-64}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-330}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-float32}"
# Validated with MTP step=3.
SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}"
SGLANG_DIR="${SGLANG_DIR:-/app/sglang}"
ATOM_DIR="${ATOM_DIR:-/app/ATOM}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logs/atom_sglang_qwen38_tp${TP_SIZE}_ep${EP_SIZE}_$(date -u +%Y%m%dT%H%M%SZ).log}"
mkdir -p -- "$(dirname -- "${LOG_FILE}")"
exec > >(tee -- "${LOG_FILE}") 2>&1
printf 'Logging to: %s\n' "${LOG_FILE}"

# Set visibility BEFORE importing torch. Both variables select the same GPUs.
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HIP_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${SGLANG_DIR}/python:${ATOM_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_PLUGINS=atom_sglang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=atom.plugin.sglang.models
export SGLANG_USE_AITER=1
export ATOM_QSA_FLYDSL_ATTN=1
export PYTHONUNBUFFERED=1


python - "${MODEL_PATH}" "${TP_SIZE}" "${EP_SIZE}" "${GPU_IDS}" <<'PY'
import importlib.metadata as metadata
import json
import pathlib
import sys

import torch

model_path, tp, ep, gpu_ids = sys.argv[1:]
try:
    tp, ep = int(tp), int(ep)
except ValueError:
    raise SystemExit("TP_SIZE and EP_SIZE must be positive integers.")
if tp not in (2, 4, 8) or ep < 1 or tp % ep:
    raise SystemExit("Use TP_SIZE=2/4/8 and an EP_SIZE that divides TP_SIZE; EP_SIZE=1 means pure TP.")
ids = gpu_ids.split(',')
if len(ids) != tp or len(set(ids)) != tp or any(not i.isdecimal() for i in ids):
    raise SystemExit("GPU_IDS must list exactly TP_SIZE distinct numeric GPU IDs.")
config_path = pathlib.Path(model_path) / "config.json"
if not config_path.is_file():
    raise SystemExit(f"Model config not found: {config_path}")
config = json.loads(config_path.read_text())
if "Qwen4ExpForConditionalGeneration" not in config.get("architectures", []):
    raise SystemExit("This launcher requires a Qwen4ExpForConditionalGeneration checkpoint.")
if not any(p.name == "atom_sglang" for p in metadata.entry_points(group="sglang.srt.plugins")):
    raise SystemExit("ATOM SGLang plugin entry point is missing; install ATOM into this Python environment.")
if torch.version.hip is None or not torch.cuda.is_available():
    raise SystemExit("ROCm PyTorch with visible GPUs is required.")
if torch.cuda.device_count() != tp:
    raise SystemExit(f"Expected {tp} visible GPUs, found {torch.cuda.device_count()}.")
for i in range(tp):
    p = torch.cuda.get_device_properties(i)
    if p.gcnArchName.split(':')[0] != "gfx942":
        raise SystemExit(f"GPU {ids[i]} uses {p.gcnArchName}; this launcher targets MI308X/gfx942.")
    print(f"GPU {ids[i]} -> local rank {i}: {p.name}, {p.total_memory / 2**30:.2f} GiB")
for name in ("sglang", "atom", "torch", "triton", "transformers", "amd-aiter", "flydsl"):
    print(f"{name}: {metadata.version(name)}")
print(f"Model: {model_path}; TP={tp}; EP={ep}")
PY

# Keep the documented EAGLE/Mamba settings. Sampling follows the model defaults;
# send top_k <= 32 (e.g. 20), top_p=0.95 explicitly in smoke-test requests.
command=(
  python -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --host "${HOST}" --port "${PORT}"
  --tp-size "${TP_SIZE}" --ep-size "${EP_SIZE}"
  --attention-backend aiter
  --kv-cache-dtype bf16 --page-size 64
  --context-length "${CONTEXT_LENGTH}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --cuda-graph-backend-decode full
  --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}"
  --speculative-algorithm EAGLE
  --speculative-num-steps "${SPECULATIVE_NUM_STEPS}"
  --speculative-eagle-topk 1
  --mamba-radix-cache-strategy extra_buffer
  --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}"
  --mamba-ssm-dtype "${MAMBA_SSM_DTYPE}"
  --watchdog-timeout 1200
)
command+=("$@")
printf '%s\n' "$$" > "${LOG_FILE%.log}.pid"
printf 'Server PID: %s (saved to %s)\n' "$$" "${LOG_FILE%.log}.pid"
printf 'Launching: '
printf '%q ' "${command[@]}"
printf '\n'
exec "${command[@]}"
