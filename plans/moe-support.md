# MoE model support

Goal: serve Mixture-of-Experts models through `emmy serve --generate`. Delivery target: **`openai/gpt-oss-20b` at
FP8 on the RTX 5090** (21 GB weights, ~8–10 GB KV headroom) — the best fit×demand product in the class, riding the
FP8 work already in flight. Engineering bring-up happens first on **`allenai/OLMoE-1B-7B-0125-Instruct`** at bf16
(13.8 GB, textbook MoE, no quantization dependency). The demand flagships (Qwen3.6-35B-A3B, GLM-4.5-Air, Laguna)
sit one tier past FP8 (30–35 GB at 1 byte/param) and become reachable when sub-byte weight compression lands
(`plans/vq-weight-compression.md`) — this plan builds the MoE machinery they will reuse.

Scope: **RTX 5090 (sm_120) only.** No 4090/sm_89 lane anywhere in this plan — no sm_89 goldens, image, or
benchmark rows.

## 1. Research summary (Aug 2026)

### What a MoE forward is, and what the models use

Router (a small linear → softmax or sigmoid → top-k → optional weight normalization) → per-expert gated MLP on the
tokens routed to it → weighted combine (+ an always-on shared expert in the GLM/Qwen3.6/Laguna lineage). All
current models are token-choice top-k; no expert-choice routing anywhere we'd target.

| model | total/active | layers | E | top-k | shared | router | fp8 GB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-oss-20b | 20.9B / 3.6B | 24, all sparse | 32 | 4 | none | linear → top-4 → softmax over the 4 | **21** |
| OLMoE-1B-7B | 6.9B / 1.3B | 16 | 64 | 8 | none | softmax top-k | (bf16 13.8) |
| Gemma 4 26B-A4B | 25.2B / 3.8B | 30 | 128 | 8 | 1 | softmax top-k | 25 (marginal) |
| Qwen3.6-35B-A3B | 35B / 3B | 40 | 256 | 8 | 1 | softmax top-k | 35 (misses) |
| GLM-4.5-Air | 106B / 12B | 46 | 128 | 8 | 1 | sigmoid + bias correction | (sub-byte only) |

### How engines execute it (and the decode math)

- **Decode (small batch)**: every strategy reads the same k experts' weights, so weight traffic is identical and
  the fight is launch overhead + gemv quality. exllamav3 (the strongest consumer-GPU MoE reference) runs a
  CUDA-graph-replayed chain of per-expert quantized gemvs at B≤8; HF transformers auto-switches its experts
  backend to `batched_mm` (gather the k experts' weights → one bmm) for decode. Dedup across a batch is
  negligible below B≈8 (E=128, k=8: expected unique experts at B=4 is 29 of 32 activations) — per-expert
  batching buys nothing at low concurrency.
- **Prefill (large batch)**: sort token-expert pairs by expert, pad each expert's segment to an M-block, one
  grouped GEMM pass (vLLM `moe_align_block_size` + fused triton kernel; transformers `grouped_mm`). Expected
  per-expert M = T·k/E — e.g. gpt-oss at a 2048-token chunk: 2048·4/32 = 256 rows/expert, GEMM-shaped but jagged
  (hot experts run 2–3× the mean). Note `torch.nn.functional.grouped_mm`'s GPU kernels are SM90+ — not available
  on sm_89/sm_120 consumer cards; emmy supplies its own kernels anyway.
- **Traceability**: transformers v5 stores experts as 3D `nn.Parameter`s (`gate_up_proj: (E, 2·inter, hidden)`)
  behind a pluggable experts interface. The `batched_mm` backend is fully static-shaped under `torch.export`
  (ExecuTorch exports a Qwen MoE this way); the `eager` backend's `nonzero()`/masked loop produces unbacked
  symints and must be avoided.

### What emmy has and lacks (verified in-repo)

Have: symbolic-M kernels end to end (masked tiles, `.dynM` goldens, the symbolic serving tier — ~2× a static twin
per step at decode widths); `GatherOp` traced/lifted/tested; Python-level per-layer orchestration in serving where
torch ops (RoPE, paged attention, final norm) already interleave emmy programs; per-layer-class program
heterogeneity exercised by gemma-4; goldens keyed per projection shape, so E same-shaped experts share ONE golden.

Lack: `topk`/`argsort`/`nonzero`/`one_hot` ops (and the tracer **silently drops** multi-output ops like
`aten.topk` — an arity-broken graph downstream, not a loud error); `ScatterOp`/`ScanOp` are declared but have no
lifting pass (no `index_add` path); no runtime-varying weight mechanism for constants (kernel launch args carry
symbolic dims only; captured graphs and TMA descriptors bake pointers); the fused-epilogue gather is an explicit
legality refusal, so a single fused all-experts kernel is off the mma tier by design.

## 2. Design

### The seam: router and combine stay in torch; experts are emmy programs

The generative serving loop is already `emmy program → torch → emmy program` per layer. MoE adds one more torch
interlude. The `post` twin splits at a third seam:

```
pre (unchanged) → vLLM attention (unchanged) →
post_attn: o_proj + residual + post-attention norm        [emmy program]
router:    linear + softmax/sigmoid + topk + normalize    [torch — tiny, and topk cannot trace]
experts:   gated MLP on that expert's routed tokens       [emmy program, weights as inputs — see below]
combine:   weighted sum of k partials (+ shared expert)   [torch index_add / einsum]
```

- The HF MoE block never enters `torch.export` — the expert FFN is traced from a purpose-built wrapper (the
  `build_attention_split_wrapper` pattern, ~90 lines), sidestepping both the tracer's silent-`topk`-drop hazard
  and the unbacked-symint swamp. Guardrail to add regardless: the tracer must RAISE on a multi-output op it
  cannot map instead of dropping it.
- Attention is untouched — `pre` twins, KV path, attention goldens all stay byte-identical. Confirmed: the MoE
  delta is FFN-only.
- The shared expert (Gemma/Qwen3.6/GLM) is an ordinary dense MLP on all tokens — an emmy program of the existing
  `mlp_gate_up`/`mlp_down` family, launched unconditionally beside the routed experts.
- Cost accepted: the norm→gate⊗up fused megakernel (`kind="fused"`) is unreachable for routed experts — the norm
  output is shared across k experts, so it must materialize at the seam. Expert FFNs run the plain matmul family.
  (The `post_attn` program keeps its own norm fusion.)

### Expert weights as inputs — one program per (layer-class, tier), not per expert

Baking each expert's weights as `ConstantOp`s would mean E programs per layer per tier (~1.5–6k programs for real
models — compile time, pack size, and the boot roofline audit all assume tens). The one existing lever avoids
this: **tensors passed as forward arguments become `InputOp`s**. The expert wrapper takes
`forward(x, w_gate_up, w_down)`; the E-expert weight tensor uploads to the device ONCE as a 3D array, and Python
passes zero-copy slices per launch. One compiled program per (layer-class, tier) serves all E experts.

- Non-captured launches rebind input pointers per call — this works today. CUDA-graph capture bakes pointers, so
  MoE FFN launches stay outside whole-step capture in V1 (see the capture question below).
- Input weights skip the constant `load_ops` folding — the weight layout transpose stays in-graph (the sm_89
  fold-gating already exercises that path), and `b_trans` staging handles the `(N, K)` layout.
- The golden surface does not grow with E: all experts share one shape, so `mlp_gate_up`/`mlp_down` golden
  entries at the expert intermediate size cover the whole layer. Per-model golden work is the normal per-shape
  seeding, same as any new model.

### Width routing per expert — buckets first, symbolic as the fallback it already is

Per-expert token counts are a data-dependent partition: at decode bucket T, expert e sees `T_e ∈ [0, T]` with
mean `T·k/E`; at prefill, mean T·k/E ± jag. The existing tier ladder handles this with one addition:

- **Decode (T ≤ decode_bucket)**: launch each *selected* expert at the decode-width bucket via the existing
  pad-to-bucket path (`_pad_rows` is per-token-independent — the same correctness argument the twin already
  uses). At B=1 exactly k experts launch; at bucket 16 with k=8/E=32, expected unique experts ≈ 31 — near
  all-experts, still bucket-padded per expert.
- **Prefill**: per-expert `T_e` ≈ 256 for the delivery model — add a static expert twin at a bucket near the
  expected width (pad up / row-split down, the rider-split precedent), with the **symbolic (`.dynM`) expert
  program as the fallback** for stragglers — this is exactly what the symbolic tier is for; its ~2× cost applies
  only to off-bucket residue, not the mean.
- A sorted grouped-GEMM kernel (vLLM-style single-pass) is deliberately NOT in scope until measurements show the
  per-expert-program launch overhead is the binding constraint (M3 exit data decides).

### The whole-step decode capture question (open, deferred)

vLLM's whole-step decode CUDA graph requires a step-invariant launch sequence; routed dispatch varies per step.
V1 serves MoE models without whole-step capture (the known cost was ~10% req/s at c=64 on gemma-4 — acceptable
for bring-up). Recovery options, in later-milestone order: (a) fixed-sequence dispatch — always launch k
expert-programs whose weight pointers are updated via graph-exec param updates; (b) device-side indirection (the
kernel reads its weight pointer from a router-written device array) — a new kernel mechanism; (c) exllamav3-style
per-layer graphs. Decide on M2 measurements, not up front.

## 3. Target model: gpt-oss-20b (FP8) — and why

- **Fit**: 21 GB at fp8 leaves ~8–10 GB for KV on the 5090 — the only high-demand MoE with real headroom. bf16
  (42 GB) does not fit, so delivery depends on FP8 M2/M4 (`plans/fp8-support.md`) — already in flight on this
  branch. Ready-made checkpoints exist (`RedHatAI/gpt-oss-20b-FP8-Dynamic` — `compressed-tensors`, the fp8
  plan's target format #2), so no quantization work lands on this plan's critical path. The native MXFP4→bf16
  upcast is exact, so fp8-from-upcast loses nothing vs the shipped weights.
- **Demand**: still a default local model in every 2026 roundup; baselines abound.
- **Architecture cost is mostly vLLM's, not emmy's**: attention sinks and the alternating dense/SWA-128 pattern
  live in the attention layer — which emmy's generative plugin delegates to vLLM's paged attention. Emmy-side
  novelty: biases on the projections (supported), the clamped-SwiGLU expert activation (one new elementwise
  spelling), and the MoE machinery itself. Validate the sinks/SWA delegation in M0 scoping (risk item).
- **Alternates**: Gemma 4 26B-A4B is the maximum-reuse option (attention identical to the gemma-4-12B emmy
  already serves; only the MoE FFN is new) but fp8 is 25 GB — 4–6 GB headroom, short-context demos only, and no
  official fp8 release. Keep it as the second model once the machinery exists; it also pre-stages the shared
  expert + E=128 regime the flagships need. ERNIE-4.5-21B-A3B is the conventional-architecture fallback.
- **Named future targets** (needs sub-byte from the VQ plan): Qwen3.6-35B-A3B, then GLM-4.5-Air / Laguna S 2.1.

## 4. Milestones

### M0 — bring-up on OLMoE (bf16, no quantization dependency)

1. Tracer guardrail: raise on unmapped multi-output ops (kills the silent-`topk` hazard for everyone).
2. The third seam: `post_attn` wrapper + expert-FFN wrapper (weights as forward args); torch router + combine in
   the serving loop. `emmy compile`/`run` the expert FFN standalone against the HF reference.
3. Serve OLMoE end to end on the 5090: correctness vs HF eager (logits + short-generation parity), all tiers on
   the symbolic/pad paths — no perf work yet.
4. Scoping spikes for M3: verify vLLM-side attention sinks + SWA-128 delegation for gpt-oss in the plugin;
   verify the RedHatAI fp8 checkpoint parses through the fp8 ingestion lane.

Exit: OLMoE generates correctly through `emmy serve --generate`; program count = 3/layer/tier (post_attn,
expert, shared-none), not E/layer.

### M1 — decode perf lane

1. Static expert twins at decode widths (m1/decode-bucket); goldens seeded for the expert FFN shapes via the
   manual pinned `--ab` method (existing `mlp_gate_up`/`mlp_down` kinds — no new golden kind).
2. Measure the per-expert launch chain vs the memory-bound floor (roofline audit; twin e2e, not L2-resident
   isolated benches). Decode TPOT target: within 15% of the k-experts weight-bytes ceiling.
3. A/B vs stock vLLM serving OLMoE bf16 on the same card.

### M2 — prefill lane + capture decision

1. Prefill expert bucket twin (+ rider-style split) with `.dynM` fallback; seed the `.dynM` expert goldens.
2. Measure prefill vs vLLM's fused MoE path; measure the no-whole-step-capture cost at c=16/64.
3. Decide the capture-recovery option (fixed-sequence pointer update vs device indirection vs per-layer graphs)
   from the data; implement only if the measured gap warrants it.

### M3 — delivery: gpt-oss-20b FP8 on the 5090

Depends on: FP8 M2 (fp8 storage through the kernel) + M4 (serving) from `plans/fp8-support.md`.

1. Ingest `RedHatAI/gpt-oss-20b-FP8-Dynamic` (compressed-tensors fp8) through the fp8 lane; clamped-SwiGLU
   elementwise; router-with-softmax-over-selected-k in the torch seam.
2. Serve on the 5090; accuracy gate vs the bf16 HF reference; smoke A/B vs stock vLLM to confirm the perf story
   holds before investing in M4's formal validation.

### M4 — validation & release (the gemma-4 article workflow, 5090 only)

Mirrors the validated gemma-4-12B flow (the article's benchmark plan + `docker/vllm-emmy-serve/ARCHITECTURE.md`):

1. **Knob search + golden file.** Manual pinned `--ab` exploration over every deployed kernel shape of the model
   (twin-audit widths: decode buckets, prefill buckets, the `.dynM` expert entries; per-projection incl. the
   expert `mlp_gate_up`/`mlp_down` shapes and the shared-expert family) →
   `emmy/compiler/pipeline/search/goldens/rtx5090_sm120_gptoss20b.yaml` (the `rtx5090_sm120_gemma4.yaml`
   precedent). Verify with `eval golden --in-model`: shapes deploy FROM the tier, not just reproduce under pins.
2. **Prebuilt serving image with seeded goldens.** The per-model release pipeline: golden coverage gate → warm →
   bake → offline zero-recompile verify → push (`make serve-* MODEL=openai/gpt-oss-20b` flow; the
   pack-format/`_encode_load_ops` risk from §5 must be resolved by here or the image gets no pack hits).
3. **TTFT + TPOT on the standard input sizes, from the image.** The gemma-4 article grid: input-size sweep
   (512 → the measured max context), 4k-in/4k-out headline, concurrency 1/4/8, 3× repeats with mean ± stddev,
   power + peak-VRAM sampling; the same `vllm bench serve` client (greedy, `--ignore-eos`, seeded) driving every
   contender. Contenders, all at 8-bit weights — identical decode weight bytes, no handicapped lanes:
   **vLLM+emmy (standard + FAST_MATH lanes) vs stock vLLM on the SAME `RedHatAI/gpt-oss-20b-FP8-Dynamic`
   checkpoint** (vLLM serves compressed-tensors fp8 on sm_120 natively) **vs llama.cpp with the Q8_0 GGUF**
   (~8-bit analog). The remaining asymmetry — stock vLLM computes W8A8-dynamic where emmy computes W8A16 — is a
   legitimate engine difference, reported. The native MXFP4 path is out of scope for the benchmark matrix; the
   article's methodology notes it exists (~4.25-bit weights) without benching it.
4. **GSM8K sanity check.** lm-eval (`local-completions`) subset against each serving endpoint, fixed seed/config;
   emmy's score within noise of stock vLLM's (same weights ⇒ any gap is a serving bug).

### M5 — article

New post in the `cloudrift-landing` repo alongside the gemma-4 article: directory
`packages/blog/content/blog/optimizing-gpt-oss-20b-rtx5090/` next to `optimizing-gemma-4-12b-rtx/`, same
structure (`index.md` + `benchmark-plan.md` + `benchmark-scripts/` committed beside it) and the same methodology
conventions (decisions log, contender table, per-depth prefill/decode tables, reproducibility via the pinned
image). Content arc: the MoE seam design (router in torch, experts as weight-slice programs), what changed vs the
gemma-4 dense story, the M4 numbers. Numbers come from M4's result manifests only — the article never cites
ad-hoc runs (`reproduce-article-benchmarks` must be able to re-run it against the pinned image).

Post-M5 breadth (optional): Gemma 4 26B-A4B self-quantized fp8 — shared expert + E=128 + full reuse of the
gemma attention stack; short-context demo acceptable.

## 5. Risks / open questions

- **Whole-step capture** (§2): the ~10% high-concurrency cost is carried until M2 data picks a recovery.
- **Program-launch overhead at decode**: k+2 program launches per layer per step from Python; exllamav3 needed
  CUDA graphs to kill this. If M1 shows launch-bound decode, pull the capture decision forward.
- **gpt-oss attention sinks in the plugin**: vLLM's attention backend owns sinks; if the plugin's per-layer
  `Attention` construction can't thread the sink params, attention work lands on emmy after all — the M0 spike
  de-risks this before M3 commits.
- **`torch_ref` cannot run gather-bearing graphs** — the vs-torch repro lane (`run --ir`) is unavailable for
  MoE-shaped graphs; per-expert FFN graphs are gather-free (the gather happens in Python), so this only bites if
  a fused dispatch form is ever built.
- **Admission capacity**: E× weight residency is the model's own size, but the extra activation buffers per
  expert program and the capture-ladder interaction must re-check the KV admission math (the serving invariant).
- **Router numerics**: routers are fp32-sensitive (sigmoid+bias-correction variants especially); the torch seam
  keeps them in torch at fp32 — no emmy numerics risk, but the seam's extra device↔device roundtrips need a
  latency check at M1.
