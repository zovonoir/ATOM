#!/bin/bash
# Install the qwen38-pd-fix branches of ATOM / SGLang / aiter into a running container of
# atom-sglang:qwen38-flash-next-20261102. All three are already editable-installed from
# /app/ATOM, /app/sglang/python and /app/aiter-test, so replacing the sources is enough.
#
#   bash install.sh
#
# Env:
#   SRC=<dir>   directory holding pre-cloned ATOM/ sglang/ aiter/ (skips git clone, e.g. offline)
#   GH=<url>    GitHub owner URL (default https://github.com/zovonoir)
#   FLYDSL_WHEEL=<path>  local flydsl 0.5.3 wheel instead of `pip install flydsl==0.5.3`
# The original sources are kept as /app/<name>.orig-<timestamp>.
set -euo pipefail
BRANCH=qwen38-pd-fix
GH=${GH:-https://github.com/zovonoir}
SRC=${SRC:-/opt/qwen38-pd-fix}
STAMP=$(date +%Y%m%d%H%M%S)

mkdir -p "$SRC"
[ -d "$SRC/ATOM" ] || git clone --branch "$BRANCH" --single-branch "$GH/ATOM.git" "$SRC/ATOM"
[ -d "$SRC/sglang" ] || git clone --branch "$BRANCH" --single-branch "$GH/sglang.git" "$SRC/sglang"
[ -d "$SRC/aiter" ] || git clone --branch "$BRANCH" --single-branch --recurse-submodules \
  --shallow-submodules "$GH/aiter.git" "$SRC/aiter"

# aiter on this branch requires flydsl >= 0.5 (the image ships 0.4.1). Uninstall first: a plain
# upgrade leaves the 0.4.1 dist-info behind and the package metadata reports no version.
pip uninstall -y flydsl >/dev/null 2>&1 || true
rm -rf /opt/venv/lib/python3.12/site-packages/flydsl-0.4.1.dist-info
pip install --no-deps "${FLYDSL_WHEEL:-flydsl==0.5.3}"

# sglang's Rust multimodal extension is a build artifact, not tracked in git: keep the image's.
MM_SO=$(ls /app/sglang/python/sglang/srt/rust_extensions/_multimodal*.so)
cp "$MM_SO" /tmp/

for pair in "ATOM:/app/ATOM" "sglang:/app/sglang" "aiter:/app/aiter-test"; do
  name=${pair%%:*}; dst=${pair#*:}
  mv "$dst" "$dst.orig-$STAMP"
  cp -a "$SRC/$name" "$dst"
  echo "$dst <- $name $(git -C "$dst" rev-parse --short HEAD)"
done
cp "/tmp/$(basename "$MM_SO")" /app/sglang/python/sglang/srt/rust_extensions/

# aiter: drop JIT artifacts so kernels rebuild for gfx942 on the first start (~15 min).
rm -rf /app/aiter-test/aiter/jit/*.so /app/aiter-test/aiter/jit/*.so.* \
       /app/aiter-test/aiter/jit/build /app/aiter-test/aiter/jit/flydsl_cache \
       /app/aiter-test/aiter/jit/__pycache__

python - <<'PY'
import importlib.metadata as m

print("flydsl:", m.version("flydsl"))
import atom.plugin.sglang.qwen4_exp_pd  # noqa: F401  (PD adapter present)
print("install ok; originals kept as /app/*.orig-*")
PY
