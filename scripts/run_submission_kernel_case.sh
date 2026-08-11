#!/usr/bin/env bash
# Run one preregistered kernel case, preserve partial evidence on every failure, and fail only after archiving it.

set -uo pipefail

if [ "$#" -ne 10 ]; then
  echo "usage: $0 CASE_DIR MODEL_REF LAYER SEQ_LEN BUDGET PATIENCE SEED DEVICES TRACE_MODE INDEPENDENT_COMPILER" >&2
  exit 2
fi

case_dir=$1
model_ref=$2
layer=$3
seq_len=$4
budget=$5
patience=$6
seed=$7
devices=$8
trace_mode=$9
independent_compiler=${10}

mkdir -p "$case_dir/verification"
export EMMY_TUNE_DB="$case_dir/autotune.db"
export EMMY_ONLINE_FILE="$case_dir/online.json"
export EMMY_CUBIN_CACHE="$case_dir/cubins"

case_status=0
capture_status=0
./venv/bin/python scripts/capture_kernel_environment.py \
  "$model_ref" "$case_dir" --source-manifest "$SOURCE_MANIFEST" \
  2>&1 | tee "$case_dir/capture.log" || capture_status=$?
if [ "$capture_status" -ne 0 ]; then
  case_status=1
fi

trace_status=0
if [ "$capture_status" -eq 0 ]; then
  trace_args=(./venv/bin/emmy trace "$model_ref" --layer "$layer" --seq-len "$seq_len")
  if [ "$trace_mode" = "loop-targets" ]; then
    trace_args+=(--loop-targets)
  elif [ "$trace_mode" != "default" ]; then
    echo "unknown trace mode: $trace_mode" | tee "$case_dir/trace.log"
    trace_status=2
  fi
  if [ "$trace_status" -eq 0 ]; then
    trace_args+=(--model-provenance "$model_ref" --output "$case_dir/working.yaml")
    "${trace_args[@]}" 2>&1 | tee "$case_dir/trace.log" || trace_status=$?
  fi
else
  trace_status=$capture_status
  echo "trace skipped because environment capture failed" | tee "$case_dir/trace.log"
fi
if [ "$trace_status" -ne 0 ]; then
  case_status=1
fi

tune_status=0
if [ "$budget" = "0" ]; then
  echo "tuning intentionally skipped for the cold deploy-greedy case" | tee "$case_dir/tune.log"
elif [ "$trace_status" -eq 0 ]; then
  ./venv/bin/emmy tune --golden-file "$case_dir/working.yaml" --clean \
    --devices "$devices" --max-candidates "$budget" --patience "$patience" --seed "$seed" \
    --dump-dir "$case_dir/dump" 2>&1 | tee "$case_dir/tune.log" || tune_status=$?
else
  tune_status=$trace_status
  echo "tune skipped because trace failed" | tee "$case_dir/tune.log"
fi
if [ "$tune_status" -ne 0 ]; then
  case_status=1
fi

verification_status=0
if [ "$trace_status" -ne 0 ]; then
  verification_status=$trace_status
  echo "verification skipped because tracing failed" | tee "$case_dir/verification.log"
elif [ "$budget" = "0" ]; then
  first_device=${devices%%,*}
  ./venv/bin/python scripts/verify_working_golden_greedy.py \
    "$case_dir/working.yaml" "$case_dir/greedy-verification" --emmy ./venv/bin/emmy \
    --repeats 5 --warmup 10 --iters 100 --cuda-visible-devices "$first_device" \
    2>&1 | tee "$case_dir/verification.log" || verification_status=$?
elif [ "$tune_status" -eq 0 ]; then
  first_device=${devices%%,*}
  verify_args=(
    ./venv/bin/python scripts/verify_working_golden_winners.py
    "$case_dir/working.yaml" "$case_dir/verification" --emmy ./venv/bin/emmy
    --repeats 5 --warmup 10 --iters 100 --cuda-visible-devices "$first_device"
  )
  if [ "$independent_compiler" = "hidet" ]; then
    verify_args+=(--bench-backends eager,tcompile,hidet,emmy --optional-backend hidet)
  elif [ "$independent_compiler" != "none" ]; then
    echo "unknown independent compiler: $independent_compiler" | tee "$case_dir/verification.log"
    verification_status=2
  fi
  if [ "$verification_status" -eq 0 ]; then
    "${verify_args[@]}" 2>&1 | tee "$case_dir/verification.log" || verification_status=$?
  fi
else
  verification_status=$tune_status
  echo "verification skipped because tuning failed" | tee "$case_dir/verification.log"
fi
if [ "$verification_status" -ne 0 ]; then
  case_status=1
fi

case_mode=searched_winner
if [ "$budget" = "0" ]; then
  case_mode=cold_deploy_greedy
fi
printf '%s\n' \
  "capture_status=$capture_status" \
  "trace_status=$trace_status" \
  "mode=$case_mode" \
  "tune_status=$tune_status" \
  "verification_status=$verification_status" \
  > "$case_dir/status.txt"

tar --exclude=artifacts.tar.gz -C "$case_dir" -czf "$case_dir/artifacts.tar.gz" . || case_status=1
exit "$case_status"
