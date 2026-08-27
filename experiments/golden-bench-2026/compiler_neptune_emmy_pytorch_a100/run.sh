#!/usr/bin/env bash
# Neptune lane: tune + Nsight-profile one operator's sequence sweep inside the pinned artifact
# image. One operator per invocation so a failed or timed-out operator costs only its own row.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 OPERATOR" >&2
  exit 2
fi

operator=$1

source /workspace/venv/bin/activate
cd /workspace/neptune

printf "%s  %s\n" \
  4802b1560054db1cdf367813528fcaf2d5262c6706444e8852ef8cb4b1f30c5d \
  scripts/neptune_bench/__main__.py | sha256sum -c -
printf "%s  %s\n" \
  c0ad2e584d9e802f87027ad43dfb7627d827178259d8d5403d4d2906e8332163 \
  scripts/neptune_bench/torch/flex.py | sha256sum -c -
printf "%s  %s\n" \
  250a5e583282f257ff617fde8103d2701bb18959c7bc9fc7e1ae89daba9be5b8 \
  scripts/neptune_bench/operators.py | sha256sum -c -
git rev-parse HEAD | tee /results/neptune-revision.txt
test "$(git rev-parse HEAD)" = "$NEPTUNE_REV"
git status --short > /results/neptune-status.txt
git diff > /results/neptune-image-source.patch
sha256sum \
  scripts/neptune_bench/__main__.py \
  scripts/neptune_bench/operators.py \
  scripts/neptune_bench/torch/flex.py \
  > /results/neptune-source.sha256
sha256sum /experiment/run.sh /experiment/run_neptune.py > /results/experiment-input.sha256

python -m pip freeze --all > /results/requirements.freeze.txt
python - <<'PY' > /results/torch-environment.txt
import torch

print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY
test "$(python -c 'import torch; print(torch.__version__.split("+")[0])')" = "2.6.0"
grep -Fq "torch.compile(flex_attention)" scripts/neptune_bench/torch/flex.py
nvidia-smi -q > /results/nvidia-smi.txt
nsys --version > /results/nsys-version.txt

test ! -e logs/neptune-tuning
test ! -e logs/profiles
mkdir -p /results/tune-logs /results/profile-logs /results/profiles
status_file=/results/neptune-setup-status.tsv
printf "operator\tsequence_length\ttune\tprofile\n" > "$status_file"
successful_profiles=0
tune_timeout=30m
profile_timeout=15m
sequence_lengths=(256 512 1024 2048 4096 8192 16384 32768)

for sequence_length in "${sequence_lengths[@]}"; do
  setup="${operator}-b1-s${sequence_length}"
  if timeout --signal=TERM --kill-after=1m "$tune_timeout" \
    python -u /experiment/run_neptune.py tune "$operator" "1,$sequence_length" --n-trials 128 \
    2>&1 | tee "/results/tune-logs/$setup.log"; then
    tune_status=ok
    if grep -Fq "Top 0 schedules" "/results/tune-logs/$setup.log"; then
      tune_status=ok:no-valid-schedule
    fi
  else
    tune_rc=$?
    if [ "$tune_rc" -eq 124 ]; then
      tune_status=timed-out
    else
      tune_status="failed:$tune_rc"
    fi
  fi
  if timeout --signal=TERM --kill-after=1m "$profile_timeout" \
    nsys profile -o "/results/profiles/$setup" --trace=cuda,nvtx,osrt --wait=primary \
    python -u /experiment/run_neptune.py profile "$operator" "1,$sequence_length" --repeat 15 \
    2>&1 | tee "/results/profile-logs/$setup.log"; then
    profile_status=ok
    if grep -Fq "result mismatch against" "/results/profile-logs/$setup.log"; then
      profile_status="$profile_status:mismatch"
    fi
    if grep -Fq "failed with exception" "/results/profile-logs/$setup.log"; then
      profile_status="$profile_status:runner-failure"
    fi
    successful_profiles=$((successful_profiles + 1))
  else
    profile_rc=$?
    if [ "$profile_rc" -eq 124 ]; then
      profile_status=timed-out
    else
      profile_status="failed:$profile_rc"
    fi
  fi
  printf "%s\t%s\t%s\t%s\n" "$operator" "$sequence_length" "$tune_status" "$profile_status" >> "$status_file"
done

cp -a logs/neptune-tuning /results/neptune-tuning
test "$successful_profiles" -gt 0
