#!/usr/bin/env bash
# Comparison lane: bench one common attention operator's sequence sweep as
# eager PyTorch / torch.compile / Emmy. Emmy replays the setup's committed golden, so this lane only
# measures — the search happens outside the recipe, through the tune-kernels skill. The greedy row in
# each record remains untuned Emmy; the golden's pinned rows carry the searched schedules, re-measured
# here at -O3.
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 EMMY OPERATOR RESULTS_DIR GOLDEN_DIR" >&2
  exit 2
fi

emmy=$1
operator=$2
results=$3
golden_dir=$4
python=$(dirname "$emmy")/python
here=$(cd "$(dirname "$0")" && pwd)
pytorch_runner=$here/run_pytorch.py
source "$here/operators.sh"

mkdir -p "$results/json" "$results/dumps" "$results/logs"
status_file=$results/setup-status.tsv
printf "operator\tsequence_length\tstatus\n" > "$status_file"
successful_setups=0
missing_goldens=0

for sequence_length in "${SEQUENCE_LENGTHS[@]}"; do
  setup="${operator}-b1-s${sequence_length}"
  golden=$golden_dir/$setup.golden.yaml
  if [ ! -f "$golden" ]; then
    printf "%s\t%s\t%s\n" "$operator" "$sequence_length" "missing-golden" >> "$status_file"
    missing_goldens=$((missing_goldens + 1))
    continue
  fi

  # --json takes a path per target: a file for a single-target golden, a directory when the
  # traced program lowered to several post-fusion kernels.
  if EMMY_NVCC_FLAGS= timeout --signal=TERM --kill-after=30s 600s \
    "$emmy" run --golden "$golden" --bench --strict \
    --bench-backends eager,tcompile,emmy --warmup 1 --iters 15 --no-record-nodes \
    --json "$results/json/$setup" --dump-dir "$results/dumps/$setup" \
    2>&1 | tee "$results/logs/$setup.log"; then
    status=ok
    successful_setups=$((successful_setups + 1))
  else
    emmy_status=$?
    if timeout --signal=TERM --kill-after=30s 600s \
      "$python" "$pytorch_runner" "$operator" "$sequence_length" \
      --warmup 1 --iters 15 --json "$results/json/$setup.pytorch.json" \
      2>&1 | tee "$results/logs/$setup.pytorch.log"; then
      status="pytorch-only:emmy-failed:$emmy_status"
      successful_setups=$((successful_setups + 1))
    else
      status="failed:emmy=$emmy_status,pytorch=$?"
    fi
  fi
  printf "%s\t%s\t%s\n" "$operator" "$sequence_length" "$status" >> "$status_file"
done

# A missing golden is a broken lane input, not a measurement: fail rather than silently
# reporting a shorter sweep.
test "$missing_goldens" -eq 0
test "$successful_setups" -gt 0
