#!/usr/bin/env bash
# Tune lane: trace one common attention operator's sequence sweep to working golden YAML
# and search each traced target. The working golden files are this lane's durable product — they are
# committed under golden/ and consumed by run_emmy.sh, so the comparison lane never has to carry
# tuning state (a tune DB, an online prior) between runs. They remain search state: the comparison
# lane re-measures every schedule they pin.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 EMMY OPERATOR RESULTS_DIR" >&2
  exit 2
fi

emmy=$1
operator=$2
results=$3
source "$(cd "$(dirname "$0")" && pwd)/operators.sh"

# One 30-minute search per setup, matching the per-setup tuning limit the Neptune lane uses.
tune_timeout=${EMMY_TUNE_TIMEOUT:-30m}
max_candidates=${EMMY_TUNE_MAX_CANDIDATES:-64}

golden_dir=$results/golden
mkdir -p "$golden_dir" "$results/logs"
status_file=$results/tune-status.tsv
printf "operator\tsequence_length\ttrace\ttune\n" > "$status_file"
tuned_setups=0

for sequence_length in "${SEQUENCE_LENGTHS[@]}"; do
  setup="${operator}-b1-s${sequence_length}"
  golden=$golden_dir/$setup.golden.yaml
  code=$(operator_code "$operator" "$sequence_length")

  # trace refuses to replace an existing inventory, so a rerun always starts from a fresh file.
  rm -f "$golden"
  if timeout --signal=TERM --kill-after=30s 600s \
    "$emmy" trace --code "$code" -o "$golden" \
    2>&1 | tee "$results/logs/$setup.trace.log"; then
    trace_status=ok
  else
    trace_rc=$?
    printf "%s\t%s\tfailed:%s\tskipped\n" "$operator" "$sequence_length" "$trace_rc" >> "$status_file"
    continue
  fi

  if timeout --signal=TERM --kill-after=1m "$tune_timeout" \
    "$emmy" tune --golden-file "$golden" --max-candidates "$max_candidates" \
    2>&1 | tee "$results/logs/$setup.tune.log"; then
    tune_status=ok
    tuned_setups=$((tuned_setups + 1))
  else
    tune_rc=$?
    if [ "$tune_rc" -eq 124 ]; then
      tune_status=timed-out
    else
      tune_status="failed:$tune_rc"
    fi
  fi
  printf "%s\t%s\t%s\t%s\n" "$operator" "$sequence_length" "$trace_status" "$tune_status" >> "$status_file"
done

# A timed-out search still leaves its measured rankings in the working file, so the lane
# only fails when no setup produced a tuned golden at all.
test "$tuned_setups" -gt 0
