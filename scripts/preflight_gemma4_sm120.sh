#!/bin/bash
# sm_120 compile preflight for the gemma-4 prebuilt-kernel image: render every gemma-4
# golden shape with the emmy compiler targeting sm_120 (CUDA hidden -> memorized
# 5090-default featurization + golden-union resolution), then `nvcc --cubin` each
# rendered kernel at sm_120 (sm_120a when it carries TMA). Proves the toolchain accepts
# every gemma-4 serving kernel family for Blackwell BEFORE paying for a rental. This
# does NOT produce the shipped cache (source parity needs the live-probed card) — run
# it inside the vllm-emmy container to preflight with the image's exact nvcc.
#
#   scripts/preflight_gemma4_sm120.sh [out_dir]   # default out: /tmp/emmy-preflight-sm120
#
# Runs from the repo venv when present, else from PATH (the vllm-emmy container has the
# emmy wheel on system python and no venv — mount just this script and run it there).
set -u
cd "$(dirname "$0")/.." 2>/dev/null || true
if [ -x ./venv/bin/emmy ]; then
  PY=./venv/bin/python; EMMY=./venv/bin/emmy
else
  PY=python3; EMMY=emmy
fi
OUT="${1:-/tmp/emmy-preflight-sm120}"
rm -rf "$OUT"; mkdir -p "$OUT"

names=$(CUDA_VISIBLE_DEVICES= "$PY" -c "
from emmy.compiler.pipeline.search.golden import GOLDEN_CONFIGS
print('\n'.join(sorted({g.name for g in GOLDEN_CONFIGS if g.name.startswith('gemma4_12b.')})))
")
# An empty enumeration (import error above, or the golden name prefix drifting) must be a hard
# FAIL — otherwise the loop below never runs and "0 OK, 0 FAIL" gates a rental on zero compiles.
[ -n "$names" ] || { echo "FAIL: no gemma4_12b.* goldens enumerated (import error or renamed prefix?)"; exit 1; }

pass=0; fail=0
for name in $names; do
  listing="$OUT/$name.listing"
  log="$OUT/$name.log"
  if ! CUDA_VISIBLE_DEVICES= EMMY_NVCC_FLAGS= "$EMMY" compile --golden "$name" --target sm_120 --no-readable -o "$listing" >"$log" 2>&1; then
    echo "FAIL render  $name"; fail=$((fail+1)); continue
  fi
  # The rendered cuda stage is a listing with `=== N: kname ===` headers between
  # kernels -> split into per-kernel .cu files (mirrors nvcc.compile_to_cubin, which
  # is per-kernel).
  awk -v out="$OUT" -v base="$name" '
    /^=== [0-9]+: / { k = $3; sub(/ ===$/, "", k); f = out "/" base "." k ".cu"; next }
    f { print > f }
  ' "$listing"
  shape_ok=1
  for cu in "$OUT/$name."*.cu; do
    [ -e "$cu" ] || { echo "FAIL split  $name (no kernels found)"; shape_ok=0; break; }
    arch=sm_120
    grep -q "cp_async_bulk_tensor" "$cu" && arch=sm_120a
    if ! nvcc --cubin -arch=$arch -o "${cu%.cu}.cubin" "$cu" >>"$log" 2>&1; then
      echo "FAIL nvcc:$arch  $(basename "$cu")"; shape_ok=0
    fi
  done
  if [ "$shape_ok" -eq 1 ]; then
    echo "OK   $name ($(ls "$OUT/$name."*.cubin 2>/dev/null | wc -l) kernel(s))"; pass=$((pass+1))
  else
    fail=$((fail+1))
  fi
done
echo "== preflight done: $pass OK, $fail FAIL (nvcc $(nvcc --version | grep -o 'release [0-9.]*'))"
[ "$fail" -eq 0 ]
