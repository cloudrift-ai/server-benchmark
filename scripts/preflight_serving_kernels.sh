#!/bin/bash
# Compile preflight for a prebuilt-kernel serving image: render every golden shape
# recorded for the target MODEL with the emmy compiler at the target arch (CUDA hidden ->
# memorized default featurization + golden-union resolution), then `nvcc --cubin` each
# rendered kernel (arch + the `a` suffix when it carries TMA). Proves the toolchain
# accepts every one of this model's serving kernel families BEFORE paying for a rental.
# This does NOT produce the shipped cache (source parity needs the live-probed card) —
# run it inside the vllm-emmy container to preflight with the image's exact nvcc.
#
#   MODEL=google/gemma-4-12B-it scripts/preflight_serving_kernels.sh [out_dir]
#   MODEL=<id> ARCH=sm_90 scripts/preflight_serving_kernels.sh
#   MODEL=<id> REVISION=<sha> scripts/preflight_serving_kernels.sh   # a revision-tagged golden set
#
# The golden set IS the enumeration: goldens are the fork-resolution evidence the warm run
# will deploy from, so preflighting exactly them is preflighting what the image will ship.
# That is also why a model with no goldens cannot be preflighted — see
# scripts/check_serving_goldens.py, which is the gate that catches it first.
#
# Runs from the repo venv when present, else from PATH (the vllm-emmy container has the
# emmy wheel on system python and no venv — mount just this script and run it there).
#
# `set -u` but deliberately NOT `set -e`: the whole point of the loop below is to attempt
# every shape and TALLY the failures, so aborting on the first failing compile would report
# one bad kernel instead of the full picture a gate needs. Every step that must not fail
# silently is therefore checked explicitly — see the enumeration guard below, and the
# per-shape `fail` counter that the final exit status is computed from.
set -u
cd "$(dirname "$0")/.." 2>/dev/null || true
MODEL="${MODEL:?set MODEL to the HF model id being preflighted}"
ARCH="${ARCH:-sm_120}"
# The release config's SERVE_REVISION. Load-bearing only when this model's goldens are
# revision-tagged (`<repo>@<rev>`); the enumeration below then refuses to guess.
REVISION="${REVISION:-}"
if [ -x ./venv/bin/emmy ]; then
  PY=./venv/bin/python; EMMY=./venv/bin/emmy
else
  PY=python3; EMMY=emmy
fi
OUT="${1:-/tmp/emmy-preflight-$ARCH}"
rm -rf "$OUT"; mkdir -p "$OUT"

# Select by the golden's recorded `model:` provenance, not by a name prefix: the prefix is a
# naming convention that drifts per model, while the provenance is the field that actually
# says which checkpoint a shape came from. Same matcher as the release gate, revision half
# included — a repo's revisions do not share kernel shapes, so preflighting another rung's
# shapes would gate the rental on the wrong set.
if ! names=$(CUDA_VISIBLE_DEVICES= MODEL="$MODEL" REVISION="$REVISION" "$PY" -c "
import os, sys
sys.path.insert(0, 'scripts')
from check_serving_goldens import model_slug, select_goldens, split_revision, _revisions
from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS
repo, tagged = split_revision(os.environ['MODEL'])
revision = os.environ['REVISION'].strip() or tagged
matched, wrong_rev = select_goldens(GOLDEN_RECORDS, model_slug(repo), revision)
if not matched and wrong_rev:
    sys.exit(
        f'FAIL: {len(wrong_rev)} golden(s) for {repo!r} are recorded against {_revisions(wrong_rev)}, '
        f'not {revision!r} — set REVISION to the config SERVE_REVISION (see scripts/check_serving_goldens.py)'
    )
print('\n'.join(sorted({g.name for g in matched})))
"); then
  # A non-zero enumeration may still have printed a PARTIAL list on stdout. Preflighting a
  # silent subset is worse than not preflighting: it gates a rental on a green summary that
  # covered only some of the model's shapes.
  echo "FAIL: golden enumeration errored for $MODEL (see the traceback above)"; exit 1
fi
# An empty enumeration (import error above, or no goldens for this model) must be a hard FAIL —
# otherwise the loop below never runs and "0 OK, 0 FAIL" gates a rental on zero compiles.
[ -n "$names" ] || { echo "FAIL: no goldens enumerated for $MODEL (import error, or none recorded — run scripts/check_serving_goldens.py)"; exit 1; }

pass=0; fail=0
for name in $names; do
  listing="$OUT/$name.listing"
  log="$OUT/$name.log"
  if ! CUDA_VISIBLE_DEVICES= EMMY_NVCC_FLAGS= "$EMMY" compile --golden "$name" --target "$ARCH" --no-readable -o "$listing" >"$log" 2>&1; then
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
    arch="$ARCH"
    grep -q "cp_async_bulk_tensor" "$cu" && arch="${ARCH}a"
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
