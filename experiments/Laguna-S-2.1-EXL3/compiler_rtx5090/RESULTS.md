# Laguna S 2.1 EXL3 compiler evidence on RTX 5090

Status: **partial functional compiler evidence; no canonical golden or tuning claim**.

The final custom Emmy/vLLM smoke loaded the verified local 1.98 bpw checkpoint on one
`NVIDIA GeForce RTX 5090`, traced the serving programs, built their CUDA launch programs,
completed a non-eager size-1 decode CUDA-graph capture, and served three successful completion
requests. The exact checkpoint identity is documented in the
[experiment overview](../README.md); no immutable Hugging Face commit existed for this run.

## Observed compiler path

- Runtime: vLLM 0.23.0 with `EmmyGenModel` and FP16 execution.
- Attention/KV: FlashAttention 2 and FP16 KV cache.
- Graph mode: `FULL_DECODE_ONLY`, capture sizes `[1]`, `enforce_eager=False`.
- Static runner shape: capacity 1, decode bucket 1, prefill bucket 0, M=1 tier.
- The compiler log records successful `torch.export` to Graph IR and CUDA
  `CompiledProgram.build` activity for the serving programs.
- The roofline boot audit reported eight static programs within 10× of the weight floor.
- vLLM finished the single requested CUDA-graph capture in two seconds, using 0.02 GiB of
  graph memory, and then returned HTTP 200 for all three completion requests.

The two repeated captured-graph responses produced the same two tokens and exact token log
probabilities. The longer request completed ten prompt tokens plus four decode tokens. See the
[functional-smoke results](../serving_emmy_rtx5090_functional_smoke/RESULTS.md), exact response
JSON, and complete stdout/stderr logs in that directory.

## Deliberate limits

This run does **not** provide any of the following:

- a complete layer/seam manifest or `coverage.json`;
- a canonical `rtx5090_sm120` golden;
- per-target tuning or paired deployable O3/reference measurements;
- strict major-gap coverage at production decode/prefill widths;
- a production serving configuration, throughput benchmark, baked image, or image digest; or
- an independently recorded Emmy source commit in the evidence manifest.

Accordingly, this document must not be used as a golden-coverage gate or performance result.
It records only that the constrained custom integration traced, compiled, captured, and
generated successfully on the target GPU. Native ExLlamaV3 is the recommended RTX 5090 runtime.
