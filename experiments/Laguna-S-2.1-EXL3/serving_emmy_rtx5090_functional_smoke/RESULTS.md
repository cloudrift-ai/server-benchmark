# Laguna S 2.1 EXL3 — Emmy non-eager RTX 5090 smoke

Result: **PASS** for the deliberately constrained custom Emmy/vLLM integration.

- Checkpoint: local verified Laguna-S-2.1 EXL3, configured body target 1.98 bpw
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB reported total memory
- Runtime: vLLM 0.23.0 with `EmmyGenModel`
- Execution: non-eager `FULL_DECODE_ONLY`, CUDA graph capture size `[1]`
- Attention/KV: FlashAttention 2, FP16 KV cache
- Scheduler: max model length 128, max batched tokens 1, max sequences 1
- Emmy lane: host embedding, capacity 1, decode bucket 1, prefill bucket 0, M=1 tier
- Model load: 27.08 GiB and 191.534911 seconds
- CUDA graph estimate: 0.05 GiB; the actual one-shape capture completed
- GPU KV cache: 3,957 shared tokens
- Post-generation device sample: 31,801 MiB used, 275 MiB free

Two independent captured-graph requests for `The capital of France is` generated the
same two tokens (`Question`, `fort`) and exactly the same token log probabilities
(`-2.1693501472473145`, `-2.2998788356781006`). They also match the earlier eager
control exactly. A separate 10-token prompt plus four decode tokens completed as well.

This is functional evidence, not a performance benchmark or a claim of standard vLLM
compatibility. One-token chunking and a single scheduled request make the custom lane
extremely constrained; native ExLlamaV3 is the recommended RTX 5090 runtime.

Original evidence hashes:

- Capture stdout: `ae9c3e56ab17b9905c20ab7ff1ac5e3364e29c03f833edb193d016c98e30c49f`
- Capture stderr: `8e3f8858cc54a040184958df346360e52705eac38dfb71f2db72cfb6bfd93a0f`
- Greedy repeat 1: `cfe0bfbedfd25d12858417262ec46e01301f58f574e9aa3d02731077f9fda73c`
- Greedy repeat 2: `acc5c4fe69a9be4d84712316753b286090bafa334afc39c28b107979b5a1e7ec`
- Prefill/decode response: `d3be8d8c63b06e15315856134e1be833b9e0bda0fa075c946371c2f13290c899`

Raw logs and response JSON are intentionally not stored in the repository.
