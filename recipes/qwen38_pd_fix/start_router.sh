#!/usr/bin/env bash
# 终端 3：P、D ready 后启动，客户端访问 7080/v1。
set -euo pipefail
PD_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${PD_SCRIPT_DIR}/pd_env.sh"

LOG_FILE="${LOG_FILE:-${LOG_DIR}/router_$(date -u +%Y%m%dT%H%M%S)_$$.log}"
pd_check_ports "${ROUTER_PORT}"
if [[ "${DRY_RUN}" != 1 ]]; then
    python - "${P_PORT}" "${D_PORT}" <<'PY'
import sys
import urllib.request

for role, port in zip(("prefill", "decode"), sys.argv[1:]):
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
    except Exception as exc:
        raise SystemExit(f"{role} is not ready: {url}: {exc}. Start P/D and retry after they are ready.")
    print(f"{role} ready: {url}")
PY
    mkdir -p -- "$(dirname -- "${LOG_FILE}")"
    exec > >(tee -- "${LOG_FILE}") 2>&1
    printf '%s\n' "$$" > "${LOG_FILE%.log}.pid"
fi

pd_exec atomesh launch \
    --host "${HOST}" --port "${ROUTER_PORT}" \
    --pd-disaggregation \
    --prefill "http://127.0.0.1:${P_PORT}" "${BOOTSTRAP_PORT}" \
    --decode "http://127.0.0.1:${D_PORT}" \
    --backend sglang \
    --policy round_robin \
    --request-timeout-secs 3600 \
    --disable-circuit-breaker \
    "$@"
