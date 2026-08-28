#!/usr/bin/env bash
# Comparison lane: bench one common attention operator's sequence sweep as
# eager PyTorch / torch.compile / Emmy. Two arms per setup:
#
#   replay     `emmy run --golden-file` re-measures the setup's committed tuned schedules twice at -O3.
#              `emmy trace --code` stores a single-op program's post-fusion targets as Loop IR
#              (every kernel shares one provenance selector), and a Loop IR target has no torch
#              twin, so each replay checks exact pins against same-input greedy Emmy. It is not a
#              direct eager proof; the reference arm owns that separate correctness boundary.
#   reference  `emmy run -c` re-traces the same snippet and benches eager, torch.compile, and the
#              UNTUNED Emmy deploy twice under --strict, so every successful setup carries repeated
#              eager and Inductor references plus direct whole-program Emmy-vs-eager proofs. When
#              an invocation fails, run_pytorch.py still preserves the two PyTorch references.
#
# The search happens outside the recipe, through the tune-kernels skill.
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
printf "operator\tsequence_length\treplay_1\treplay_2\treference_1\treference_2\n" > "$status_file"
successful_setups=0
missing_goldens=0

for sequence_length in "${SEQUENCE_LENGTHS[@]}"; do
  setup="${operator}-b1-s${sequence_length}"
  golden=$golden_dir/$setup.golden.yaml
  if [ ! -f "$golden" ]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$operator" "$sequence_length" "missing-golden" "missing-golden" "skipped" "skipped" >> "$status_file"
    missing_goldens=$((missing_goldens + 1))
    continue
  fi
  source_code=$(operator_code "$operator" "$sequence_length")

  replay_statuses=()
  reference_statuses=()
  for repetition in 1 2; do
    # --json takes a path per target: a file for a single-target golden, a directory when the
    # traced program lowered to several post-fusion kernels.
    if EMMY_NVCC_FLAGS= timeout --signal=TERM --kill-after=30s 600s \
      "$emmy" run --golden-file "$golden" --bench --bench-backends emmy \
      --warmup 1 --iters 15 --no-record-nodes \
      --json "$results/json/$setup.replay-$repetition" --dump-dir "$results/dumps/$setup.replay-$repetition" \
      2>&1 | tee "$results/logs/$setup.replay-$repetition.log"; then
      replay_statuses+=(ok)
    else
      replay_statuses+=("failed:$?")
    fi
  done
  for repetition in 1 2; do
    if EMMY_NVCC_FLAGS= timeout --signal=TERM --kill-after=30s 600s \
      "$emmy" run -c "$source_code" --bench --strict --bench-backends eager,tcompile,emmy \
      --warmup 1 --iters 15 --no-record-nodes --json "$results/json/$setup.reference-$repetition.json" \
      2>&1 | tee "$results/logs/$setup.reference-$repetition.log"; then
      reference_statuses+=(ok)
    else
      reference_status=$?
      if timeout --signal=TERM --kill-after=30s 600s \
        "$python" "$pytorch_runner" "$operator" "$sequence_length" \
        --warmup 1 --iters 15 --json "$results/json/$setup.pytorch-$repetition.json" \
        2>&1 | tee "$results/logs/$setup.pytorch-$repetition.log"; then
        reference_statuses+=("pytorch-only:emmy-failed:$reference_status")
      else
        reference_statuses+=("failed:emmy=$reference_status,pytorch=$?")
      fi
    fi
  done
  if [ "${replay_statuses[0]}" = ok ] && [ "${replay_statuses[1]}" = ok ] && \
    [ "${reference_statuses[0]}" = ok ] && [ "${reference_statuses[1]}" = ok ]; then
    successful_setups=$((successful_setups + 1))
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$operator" "$sequence_length" \
    "${replay_statuses[0]}" "${replay_statuses[1]}" \
    "${reference_statuses[0]}" "${reference_statuses[1]}" >> "$status_file"
done

# A missing golden is a broken lane input, not a measurement: fail rather than silently
# reporting a shorter sweep.
test "$missing_goldens" -eq 0
# Every requested setup needs both O3 replays and both strict reference proofs. The status file preserves
# all failures, but a partial sweep cannot support the experiment row's performance conclusion.
test "$successful_setups" -eq "${#SEQUENCE_LENGTHS[@]}"
