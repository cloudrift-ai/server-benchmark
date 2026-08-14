# Llama 3.1 8B Instruct AWQ on one V100

Status: serving-qualified on single 16 GB and 32 GB V100 profiles with the pinned SM70 1Cat/vLLM runtime.

## Qualified deployments

| Item | 16 GB profile | 32 GB profile |
| --- | --- | --- |
| Hardware | 1× Tesla V100-SXM2-16GB, SM70 | 1× Tesla V100-SXM3-32GB, SM70 |
| Context | 32,768 tokens | 131,072 tokens (native) |
| Max sequences | 8 | 8 |
| Tensor parallelism | 1 | 1 |
| Driver / CUDA | 580.159.03 / 13.0 | 580.159.03 / 13.0 |
| Model revision | `db1f81ad4b8c7e39777509fac66c652eb0a52f91` | same |
| Image digest | `sha256:8405bb60d24610417d0d6da278a753e2c968bfd1e0d7ff7f79cd6601a038b2be` | same |
| KV capacity reported at boot | 61,760 tokens | 176,864 tokens |

The checkpoint is 4-bit AWQ (`group_size=128`, GEMM packing) with FP16 compute. The image contains 1Cat/vLLM
`0.1.dev21+g91aca502d.d20260809.cu128` and uses its Volta-native `FLASH_ATTN_V100` path. The 16 GB profile was
reduced by powers of two: 131,072 needed 16 GiB of KV cache and 65,536 needed 8 GiB, while only 6.86 GiB was
available. The 32,768-token profile then loaded and served cleanly. The native 131,072-token profile loaded on 32 GB.

## Serving performance

Measured 2026-08-13 with three repeats of 32 requests, 512 input tokens, 256 forced output tokens, concurrency 8,
greedy decoding, and two warmup requests. All 192 measured requests succeeded.

| Profile | Output tok/s | Total tok/s | Requests/s | Mean TTFT | Mean TPOT / ITL |
| --- | ---: | ---: | ---: | ---: | ---: |
| V100 16 GB / 32k | 196.34 ± 11.61 | 589.02 ± 34.81 | 0.763 | 411.77 ms | 39.38 / 39.38 ms |
| V100 32 GB / 128k | 246.66 ± 14.61 | 739.99 ± 43.84 | 0.963 | 435.06 ms | 30.92 / 30.92 ms |

Cold model load and warmup took 124.56 seconds on 16 GB and 113.07 seconds on 32 GB. Both smoke tests returned
the correct answer to `2 + 2`.

## Chat, tools, and context

- Both deployments returned valid OpenAI-compatible chat completions.
- Both emitted a parsed `get_weather` tool call with `{"city":"Paris"}` and `finish_reason="tool_calls"`.
- The 16 GB profile processed a material 30,041-token prompt and returned `OK.` without OOM.
- The 32 GB profile processed a material 120,041-token prompt and generated a coherent completion without OOM.
  That long synthetic prompt hit its requested 16-token output cap rather than obeying the embedded exact-output
  instruction, so the result qualifies memory/execution at long context, not long-context instruction accuracy.

## Compiler qualification

Emmy's compiler can now recognize this AWQ GEMM checkpoint, stream only the selected layer, preserve packed
`qweight`/`qzeros`/`scales` tensors, spell their int4 decode algebra into the graph, and feed the packed constants to
the serving compiler lane. The repair also covers optional attention inputs, non-kernel slice boundaries, unique
exact-target names, and direct strict eager verification for model-ID runs.

Layer 0 at sequence length 1 traced to 381 IR nodes. Its seven packed projections fused into four generated kernels.
The untuned graph passed strict eager comparison at `rtol=atol=1e-3` with 0.000977 maximum absolute error, but took
941,486 µs versus 1,031 µs eager. All four exact targets were swept with eight candidates across the host's eight
V100-SXM3 GPUs. Three targets produced winners; every candidate for the dominant attention/linear/reduction target
exceeded the tuning safety budget. Replaying the usable winners improved Emmy to 797,678 µs versus 1,025 µs eager,
still 778× slower.

Prefill exposed the remaining structural gap. Sequence length 128 passed the same strict check, but Emmy took
2,581,686 µs versus 1,197 µs eager. At 256 tokens a generated kernel exceeded the two-second safety limit; at the
required 512-token shape the execution plan requested a 60.13 GB scratch slab and failed allocation on the 32 GB
V100. Packed AWQ parsing and correctness are therefore repaired, but the compiler still lacks a bounded,
tensor-core-friendly fused int4 GEMM schedule. No Emmy golden or Emmy serving promotion was retained.

The serving numbers above do **not** use Emmy-generated kernels. They use the pinned 1Cat/vLLM image's Volta AWQ and
`FLASH_ATTN_V100` paths, which remain the qualified production backend.

## Reproduce

```bash
emmy bench --max-workers 2 \
  --ssh riftuser@185.165.50.60 \
  --ssh riftuser@66.172.10.131 \
  experiments/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/serving_v100_sxm2_16gb \
  experiments/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/serving_v100_sxm3_32gb
```

The benchmark writes ignored local output. `RESULTS.md` is the only retained measurement artifact.

## Limits

- The 16 GB profile is qualified only through 32,768 tokens; larger power-of-two contexts failed the boot-time KV fit.
- The 32 GB long-context probe validates execution and memory, not instruction-following quality at 120k tokens.
- The exact image is already present on both qualified hosts. A registry pull by this locally resolved digest returned
  `manifest unknown`, so preload or republish that image before using the recipe on a new host.
- Requalify after changing the image, model revision, driver, context length, or attention backend.
- Emmy compiler serving is not qualified for this AWQ checkpoint; its packed decode path is correct but not deployable.
