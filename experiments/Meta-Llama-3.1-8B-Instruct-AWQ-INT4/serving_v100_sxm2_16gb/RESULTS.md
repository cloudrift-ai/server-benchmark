# Llama 3.1 8B Instruct AWQ INT4 — V100 SXM2 16 GB

Verified 2026-08-20 on one `NVIDIA Tesla V100 SXM2 16GB` (SM 7.0) over SSH (`riftuser@66.172.10.172`).

## Run details

| Item | Value |
| --- | --- |
| GPU | Tesla V100-SXM2-16GB, 1×, SM 7.0 |
| Driver / CUDA | 580.126.20 / 12.8 (via image) |
| Image | `cloudriftai/1cat-vllm-sm70:1.0.0` |
| Model revision | `db1f81ad4b8c7e39777509fac66c652eb0a52f91` |
| Context length | 32,768 |
| Concurrency | 8 |
| Attention | `TRITON_ATTN` (FLASH_ATTN_V100 blocked by Volta dynamic shared-memory 96 KB cap) |

## Results

32 requests, 512 input tokens, 256 forced output tokens, concurrency 8, greedy (`temperature=0.0`), 2 warmups, 1 repeat.

| Metric | Value |
| --- | --- |
| Output tok/s | 624 |
| Total tok/s | 1,870 |
| Requests/s | 2.4 |
| Mean TTFT | 103 ms |
| Mean TPOT | 12.4 ms |
| Mean ITL | 12.4 ms |
| P99 TTFT | 125 ms |
| Failed requests | 0 / 32 |

Cold model load + warmup took 110.6 s.

## Validation

- Chat completion (`2 + 2` → `"4"`) passed.
- Tool call (`get_weather({"city":"Paris"})`) emitted `tool_calls` with `finish_reason="tool_calls"`.
- ~40,000-token prompt accepted and generated a coherent response (~10,044 total tokens).

## Notes

- The original `FLASH_ATTN_V100` backend fails at runtime on this host with `RuntimeError: Shared memory exceeds 96KB: 105984 bytes` — a Volta SM 7.0 driver limit on CUDA dynamic shared memory per block that the 1Cat/vLLM 1.0.0 flash-attn-v100 kernel does not work around with `cudaFuncSetAttribute`. Switching to `--attention-backend TRITON_ATTN` bypasses the kernel and serves correctly.
- Compiler qualification on this host was not possible: the remote venv build fails because `CPyCppyy` requires `libclang-dev` at compile time, which is not installed and cannot be installed without sudo. Per the tune-kernels protocol this diagnostic lives outside the repo.
- `--language-model-only` was removed from extra_args because it is not supported in the SM 7.0 image (1Cat/vLLM 1.0.0).

## Reproduce

```bash
emmy bench --max-workers 2 \
  --ssh riftuser@66.172.10.172 \
  experiments/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/serving_v100_sxm2_16gb
```

Run ID: `20260820T233346Z`
Archive: `results_v100x1.tar.gz`
