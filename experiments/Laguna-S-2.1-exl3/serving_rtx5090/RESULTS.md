# Laguna S 2.1 EXL3 qualification on RTX 5090

Status: experimental serving qualification of the checkpoint's `2.00bpw` branch. No production recipe or image is
promoted until the exact checkpoint fits, produces coherent text, and passes the final benchmark on the supplied
card.

## Scope

- Date: 2026-08-11
- Model: `riftstack/Laguna-S-2.1-exl3`
- Immutable revision: `45c1f34a8c3ae3025a8bce2b1decea4991a6a239` (`2.00bpw`)
- Hardware: 1× `NVIDIA GeForce RTX 5090`, compute capability 12.0, driver 580.173.02
- Checkpoint payload: 30,549,686,933 bytes across four safetensor shards and metadata
- Quantization: EXL3 2.00 bpw body matrices, 6-bit output head
- Task-local image: `emmy-onboard-vllm-emmy:0.23.0-gatefix`, built from this branch on the supplied host

The Hugging Face `main` branch contains documentation rather than the checkpoint. `model.revision` now drives both
Emmy's prefetch and vLLM, preventing the previous fast but incorrect download of the empty branch.

## Serving envelope

The first attempt at `gpu_memory_utilization: 0.97` failed before weight load because the active desktop left 30.08
GiB free while vLLM requested 30.38 GiB. The corrected experiment uses 0.95, context length 512, one sequence, one
batched token, eager execution, and the existing static M=1 Emmy plan. It loads all 48 layers, uses 27.93 GiB for
weights, initializes the KV cache, and reaches the OpenAI-compatible API.

The first-class attention-gate inference added by this branch fixes the Transformers 5.12 configuration mismatch:
the checkpoint has a 48-wide per-head gate and a 6,144-wide query projection but does not declare
`gate_per_head`. Dense and routed attention parity tests cover the inferred layout.

End-to-end semantic qualification still fails. A completion probe returned corrupted token fragments, and a proper
chat-template probe returned repetitive incoherent text. The measured 5.2--5.7 output tok/s during those probes is
not a publishable benchmark because correctness is a prerequisite. Native vLLM 0.23.0 independently rejects the
checkpoint at startup with `Unknown quantization method: exl3`.

## Compiler evidence

`emmy/compiler/pipeline/search/goldens/rtx5090_sm120_laguna_s_2_1_exl3.yaml` is pinned to the exact revision and the
RTX 5090. It covers the checkpoint's static M=1 EXL3 programs under the existing coded-trunk implementation. The
golden is compiler evidence only; it cannot substitute for end-to-end generation quality at this unusually low bit
allocation.

## Decision

Do not promote a production recipe or serving image. The exact branch and checkpoint now download and boot through
Emmy, but generated text is not semantically correct, while stock vLLM has no EXL3 implementation. Publication was
not authorized or attempted. The task-local image is qualification evidence only and is removed during cleanup.
