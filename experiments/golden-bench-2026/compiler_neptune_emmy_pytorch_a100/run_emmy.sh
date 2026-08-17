#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 EMMY RESULTS_DIR" >&2
  exit 2
fi

emmy=$1
results=$2
python=$(dirname "$emmy")/python
pytorch_runner=$(cd "$(dirname "$0")" && pwd)/run_pytorch.py
mkdir -p "$results/json" "$results/dumps" "$results/logs"
status_file="$results/setup-status.tsv"
printf "operator\tsequence_length\tstatus\n" > "$status_file"
successful_setups=0

operators=(
  prefill_global
  prefill_causal
  prefill_gqa
  decode_causal
  decode_gqa
)
sequence_lengths=(256 512 1024 2048 4096 8192 16384 32768)

operator_code() {
  local operator=$1
  local sequence_length=$2
  local q_heads=32
  local kv_heads=32
  local q_length=$sequence_length
  local is_causal=False
  local enable_gqa=False

  case "$operator" in
    prefill_global)
      ;;
    prefill_causal)
      is_causal=True
      ;;
    prefill_gqa)
      q_heads=64
      kv_heads=8
      is_causal=True
      enable_gqa=True
      ;;
    decode_causal)
      q_length=1
      ;;
    decode_gqa)
      q_heads=64
      kv_heads=8
      q_length=1
      ;;
    *)
      echo "unknown operator: $operator" >&2
      return 2
      ;;
  esac

  local attention=
  if [ "$operator" = decode_gqa ]; then
    # Express the eight query heads per KV head as a broadcast batch dimension. This preserves
    # noncausal decode semantics: torch.export drops is_causal=False but retains enable_gqa=True,
    # which older Emmy frontends can otherwise mistake for a causal flag.
    attention="F.scaled_dot_product_attention("\
"q.reshape(1,8,8,1,128),"\
"k.reshape(1,8,1,$sequence_length,128),"\
"v.reshape(1,8,1,$sequence_length,128),"\
"is_causal=False).reshape(1,64,1,128)"
  else
    attention="F.scaled_dot_product_attention("\
"q,k,v,is_causal=$is_causal,enable_gqa=$enable_gqa)"
  fi

  printf '%s' \
    "torch.manual_seed(0);" \
    "q=torch.randn(1,$q_heads,$q_length,128,dtype=torch.float16);" \
    "k=torch.randn(1,$kv_heads,$sequence_length,128,dtype=torch.float16);" \
    "v=torch.randn(1,$kv_heads,$sequence_length,128,dtype=torch.float16);" \
    "$attention"
}

for operator in "${operators[@]}"; do
  for sequence_length in "${sequence_lengths[@]}"; do
    setup="${operator}-b1-s${sequence_length}"
    code=$(operator_code "$operator" "$sequence_length")
    if EMMY_NVCC_FLAGS= timeout --signal=TERM --kill-after=30s 600s \
      "$emmy" run --code "$code" --bench --strict \
      --bench-backends eager,tcompile,emmy --warmup 1 --iters 15 \
      --json "$results/json/$setup.json" --dump-dir "$results/dumps/$setup" \
      2>&1 | tee "$results/logs/$setup.log"; then
      status=ok
      successful_setups=$((successful_setups + 1))
    else
      emmy_status=$?
      if [ -f "$results/json/$setup.json" ]; then
        mv "$results/json/$setup.json" "$results/json/$setup.emmy-failed.json"
      fi
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
done

test "$successful_setups" -gt 0
