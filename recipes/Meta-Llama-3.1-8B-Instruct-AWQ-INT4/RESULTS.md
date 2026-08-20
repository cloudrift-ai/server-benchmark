# Llama 3.1 8B Instruct AWQ on one V100

Serving-qualified on single 16 GB V100 SXM2 profile with the pinned SM70 1Cat/vLLM runtime. Image pin updated
from a broken digest to tag `:1.0.0` (the digest pin no longer resolves on Docker Hub). Attention backend changed
from `FLASH_ATTN_V100` to `TRITON_ATTN` because the former fails on Volta with
`RuntimeError: Shared memory exceeds 96KB` — a CUDA dynamic shared-memory ceiling that the 1Cat/vLLM 1.0.0
flash-attn-v100 kernel does not work around.

## 16 GB profile (this run)

| Item | Value |
| --- | --- |
| Hardware | 1× Tesla V100-SXM2-16GB, SM 7.0 |
| Image | `cloudriftai/1cat-vllm-sm70:1.0.0` |
| Model revision | `db1f81ad4b8c7e39777509fac66c652eb0a52f91` |
| Context | 32,768 tokens |
| Tensor parallelism | 1 |
| Attention backend | `TRITON_ATTN` |
| Driver / CUDA | 580.126.20 / 12.8 (in image) |

## Performance (measured 2026-08-20)

32 requests, 512 input tokens, 256 forced output tokens, concurrency 8, greedy, 2 warmups, 1 repeat.

| Metric | Value |
| --- | --- |
| Output tok/s | 624 |
| Total tok/s | 1,870 |
| Requests/s | 2.4 |
| Mean TTFT | 103 ms (median 99, P99 125) |
| Mean TPOT | 12.4 ms (median 12.4, P99 12.5) |
| Mean ITL | 12.4 ms |
| Failed | 0 / 32 |

Cold load + warmup: 110.6 s.

## Validation

- Chat completion (`2 + 2` → `"4"`) passed.
- Tool call (`get_weather({"city":"Paris"})`) emitted `tool_calls` with `finish_reason="tool_calls"`.
- ~40,000-token prompt accepted and generated a coherent response.

## Compiler qualification

Compiler qualification on this host was not possible: the remote venv build fails because `CPyCppyy` requires
`libclang-dev` at compile time. Per the tune-kernels protocol the diagnostic lives outside this repo.

Previous qualification on V100 SXM3 showed the AWQ GEMM packing is traceable and eager-correct, but no deployable
kernel schedule was found (the dominant attention/linear/reduction target required >60 GB scratch). No Emmy golden
or Emmy serving promotion applies. The serving numbers above use the 1Cat/vLLM image's stock attention path only.

## Reproduce

```bash
emmy bench --max-workers 2 \
  --ssh riftuser@66.172.10.172 \
  experiments/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/serving_v100_sxm2_16gb
```

Run ID: `20260820T233346Z`

## Limits

- The 16 GB profile is qualified through 32,768 tokens; larger contexts failed the KV-cache fit gate.
- `FLASH_ATTN_V100` is unavailable on this image — it requires removing `ipc: host` or rebuilding the image.
- Requalify after changing the image, model revision, driver, context length, or attention backend.
- Emmy compiler serving is not qualified for this AWQ checkpoint on this platform.
