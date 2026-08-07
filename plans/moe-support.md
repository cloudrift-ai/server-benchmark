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

- **Fit**: 22.7 GB at fp8 leaves ~8 GB for KV on the 5090 — the only high-demand MoE with real headroom. bf16
  (42 GB) does not fit, so delivery depends on FP8 M2/M4 (`plans/fp8-support.md`) — already in flight on this
  branch. No usable ready-made checkpoint exists, so the delivery checkpoint is built here:
  **`riftstack/gpt-oss-20b-FP8-Dynamic`** (§3.1), which is BIT-EXACT to the released weights.
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

### 3.1 The delivery checkpoint: `riftstack/gpt-oss-20b-FP8-Dynamic` (2026-08-07)

**Why one had to be built.** `RedHatAI/gpt-oss-20b-FP8-Dynamic`, the checkpoint this plan originally assumed, does
not exist (404). The community `gpt-oss-20b-FP8-Dynamic` checkpoints that do exist are 41.2 GB and quantize only
the attention Linears: their safetensors headers report 20,277,798,144 f16 parameters against 637,009,920 fp8 —
and 637,009,920 is exactly the 24×(q,k,v,o) projection count. The 19.1 B expert parameters, which are 91% of the
model, stay f16. Q8_0 GGUF conversions of the release are 12.1 GB, below the 20.9 GB floor any genuine 8-bit
encoding of 20.9 B parameters would have, so their expert tensors are still MXFP4. Neither fits the 5090.

**The finding that makes fp8 defensible for an MXFP4-native model: the MXFP4→fp8 fold is LOSSLESS.** MXFP4's 16
E2M1 codes are `0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6` — each needs at most two significant mantissa bits, and e4m3
carries three — and the E8M0 block scale is a pure power of two, so applying it only shifts the exponent. An fp8
re-encode is therefore exact whenever ONE per-output-channel power-of-two scale brings a whole channel inside
e4m3's normal range `[2^-6, 448]`: 14.8 binades of budget against the fp4 code range of 12× (3.6 binades) plus the
channel's block-exponent spread. Measured over all 6,635,520 expert output channels of the release, that spread is
mean 1.62 / p50 2 / p99 3 / max 10 binades, against a guarantee bound of 11 (`12·2^11 = 24576 ≤ 448·2^6 = 28672`).
Every channel fits with room to spare, and no value lands in the fp8 subnormal range.

The scale must be a power of two *per output channel*, not per MXFP4 block, for a compiler reason: a k-varying
block scale would decline the W8A16 mul-hoist binding. Folding the E8M0 exponents into the fp8 values leaves the
stored scale k-INVARIANT (shape `(E, 1, out)` against a weight `(E, in, out)` — `_scale_layout` reports
`block=(1, 2880, 1)`, degenerate), so the mul-hoist binds exactly as it does for an ordinary per-channel fp8
checkpoint. Losslessness and kernel-tier reachability come from the same choice.

**What was built and measured.** Experts upcast with transformers 5.13's own `convert_moe_packed_tensors`, then
re-encoded at `scale = 2^c`, `c = frexp(absmax).exponent - 9` (the smallest power of two putting the channel
absmax under 448, which maximizes headroom at the bottom). Attention stays bf16 — it is natively bf16, so fp8
there would be the one lossy step in the artifact, and 0.6 GB is cheap for removing the last caveat. Result:
48/48 expert tensors bit-exact through emmy's own `dequantize(decode_f8(bits), scale)` lane (`np.array_equal`,
max abs error 0.0), 0 inexact channels, 0 subnormal fp8 values, and every non-expert tensor byte-identical to the
release. 22.73 GB, 459 tensors, 6 shards. This is an EXPERTS-ONLY fp8 quantization — precisely the inverse of the
community checkpoints. Build and verification scripts ship in the repo (`scripts/`).

**Naming consequence, and it is a real cost.** The 3-D expert params use the compressed-tensors generic
`<param>` + `<param>_scale` pairing (`…experts.gate_up_proj` + `…experts.gate_up_proj_scale`), which is what
emmy's fp8 ingestion reads. Stock vLLM's compressed-tensors loader does not accept that spelling for
`GptOssExperts`, so **the checkpoint is emmy-only** and M4 can no longer run both contenders on one shared
checkpoint (see M4 step 3).

## 4. Milestones

### M0 — bring-up on OLMoE (bf16, no quantization dependency)

1. Tracer guardrail: raise on unmapped multi-output ops (kills the silent-`topk` hazard for everyone).
2. The third seam: `post_attn` wrapper + expert-FFN wrapper (weights as forward args); torch router + combine in
   the serving loop. `emmy compile`/`run` the expert FFN standalone against the HF reference.
3. Serve OLMoE end to end on the 5090: correctness vs HF eager (logits + short-generation parity), all tiers on
   the symbolic/pad paths — no perf work yet.
4. Scoping spikes for M3: verify vLLM-side attention sinks + SWA-128 delegation for gpt-oss in the plugin;
   verify a compressed-tensors fp8 gpt-oss checkpoint parses through the fp8 ingestion lane.

Exit: OLMoE generates correctly through `emmy serve --generate`; program count = 3/layer/tier (post_attn,
expert, shared-none), not E/layer.

### M1 — decode perf lane

1. Static expert twins at decode widths (m1/decode-bucket); goldens seeded for the expert FFN shapes via the
   manual pinned `--ab` method (existing `mlp_gate_up`/`mlp_down` kinds — no new golden kind).
2. Measure the per-expert launch chain vs the memory-bound floor (roofline audit; twin e2e, not L2-resident
   isolated benches). Decode TPOT target: within 15% of the k-experts weight-bytes ceiling.
3. A/B vs stock vLLM serving OLMoE bf16 on the same card.

#### M1 findings (2026-08-06, RTX 5090, OLMoE fp16)

- Landed: static expert twins (`moe.expert.one` / `moe.expert.bucket`) + the lean launch loop (lock/stream
  hoisted around the per-expert loop, cupy weight views minted once at boot, pointer-swap zero-copy on
  descriptor-free tiers — TMA descriptors bake pointers at build, verified wrong under a naive swap). Decode
  c=1 TPOT 48 → **17.9 ms/token**; c=8 aggregate 186 tok/s; greedy parity unchanged; boot roofline audit clean.
- Launch-chain breakdown: per-expert launch cost through the per-call symbolic path was 0.23 ms, of which the
  12 MB weight copy is ~15 µs — **Python framing is the wall, not weight bytes**. Post-twins the residual is
  ~117 µs × ~128 expert launches per step, eager.
- A/B vs stock vLLM (same card, fused-MoE + FULL decode capture, warm): stock c=1 TPOT **2.15 ms** (at the
  ~1.7 ms k-experts weight-bytes floor), c=8 **1599 tok/s** — emmy is 8.3–8.6× behind. **Launch-bound decode is
  confirmed**, so per §2 the capture decision moves into M2: the per-expert Python launch chain must go
  (fixed-sequence dispatch with graph-exec param updates, device-side indirection, per-layer graphs, or the
  sorted grouped-GEMM pass), not be shaved.
- Expert-shape golden seeding SKIPPED, deliberately: with decode launch-bound at ~9×, kernel-level tuning is
  noise; goldens come after the dispatch model changes. (The M1 "within 15% of the weight-bytes ceiling" target
  is not reachable under per-expert Python dispatch.)
- Operational: stock vLLM's first OLMoE boot JIT-compiles FlashInfer `fused_moe_120` — unbounded ninja
  parallelism OOM-kills nvcc on a 60 GB box (`MAX_JOBS=4` fixes it), and the 600 s engine-core startup timeout
  can fire mid-JIT (a retry resumes the cached build).

### M2 — prefill lane + capture decision

1. Prefill expert bucket twin (+ rider-style split) with `.dynM` fallback; seed the `.dynM` expert goldens.
2. Measure prefill vs vLLM's fused MoE path; measure the no-whole-step-capture cost at c=16/64.
3. Decide the capture-recovery option (fixed-sequence pointer update vs device indirection vs per-layer graphs)
   from the data; implement only if the measured gap warrants it.

### M3 — delivery: gpt-oss-20b FP8 on the 5090

Depends on: FP8 M2 (fp8 storage through the kernel) + M4 (serving) from `plans/fp8-support.md`.

1. Ingest `riftstack/gpt-oss-20b-FP8-Dynamic` (compressed-tensors fp8, §3.1) through the fp8 lane; clamped-SwiGLU
   elementwise; router-with-softmax-over-selected-k in the torch seam.
2. Serve on the 5090; accuracy gate vs the bf16 HF reference; smoke A/B vs stock vLLM to confirm the perf story
   holds before investing in M4's formal validation.

#### M3 findings (2026-08-07, RTX 5090, `~/checkpoints/gptoss20b-fp8-emmy` — self-quantized 22.1 GB fp8)

> Checkpoint SUPERSEDED by the lossless `riftstack/gpt-oss-20b-FP8-Dynamic` (§3.1). Two carry-over notes:
>
> - The EXPERT tensors are unchanged in shape, dtype and scale layout, so the M4 expert goldens carry over
>   untouched — only the values changed. The ATTENTION projections did change, fp8 → bf16 (§3.1), and `ShapeKey`
>   has no weight-dtype field, so the seeded q/k/v/o rows still MATCH by key but were explored against an
>   fp8-storage B operand with an in-graph dequant cone, where B is now plain f16. Re-verify those 8 rows.
> - The correctness gate's "DIFFERENT WEIGHTS" explanation for greedy divergence against stock is gone: the
>   expert weights are now bit-identical to the ones stock serves. Re-run that gate; any residual divergence is
>   kernel accumulation order, and a large one is a bug.

- Landed (M3b, on top of M3a's compiler gates): tracer `maximum` mis-route fixed + `clamp` decomposition
  (elementwise min/max chain; the clamped-SwiGLU needs it); the gpt-oss expert wrapper (`x @ W + b`, chunk-half
  gate/up restored by de-interleave-at-load, clamped-SwiGLU from the module's `alpha`/`limit`; layout read off the
  transformers experts-interface attributes, never shapes — `down_proj` is square); `load_quantized_split`
  (config-built META twin + shard-streamed dense values + per-layer fp8 expert store — never the 42 GB whole
  dict); `from_model(expert_store=…)` compiling the expert programs through `spell_quantized_inputs`
  (`quant_specs` in `_compile_split`), every per-expert input — bits, biases, scales — sliced from E-stacked
  device tensors; fixed-slot tier generalized to one pointer table per input kind (six indirect inputs, fp8 ×
  indirect × bias compose); sinks as per-layer vLLM `Attention(sinks=…)` params (TRITON_ATTN auto-selects on
  sm_120), `load_weights` claims `*.self_attn.sinks`; YaRN rides the flat `rope_parameters` unchanged.
- Serve boot: cold ~8 min (149 program plans compiled + packed), roofline audit clean, model 20.1 GiB VRAM,
  KV cache 114k tokens at util 0.90, whole-step decode capture at size 1 through the fixed-slot tier.
- Correctness gate (greedy 24-token completions, emmy fp8 vs stock vLLM serving the native MXFP4 checkpoint —
  the bf16 HF reference is a 42 GB CPU load, infeasible on the 60 GB box): 1/3 exact, 15/24-token prefix on a
  second, immediate flip on the third — every continuation fluent and factually right. Divergence class:
  DIFFERENT WEIGHTS (self-quantized fp8-from-upcast vs native mxfp4) flipping greedy near-ties, not corruption;
  kernel-level parity is separately pinned hermetically (tiny fp8 checkpoint through `create` vs its dequantized
  reference — `test_gen_runner_gptoss_fp8_create_matches_dequantized_reference`, max err < 3% of ref RMS through
  a full layer incl. router + combine).
- Smoke perf A/B (512 in / 128 out, `vllm bench serve`, same card): emmy c=1 TPOT 61.4 ms / TTFT 5.9 s, c=8
  TPOT 356 ms; stock (mxfp4, fused MoE, full ladder) c=1 TPOT 3.5 ms / TTFT 33 ms, c=8 TPOT 6.4 ms. Expected at
  bring-up: every gpt-oss shape is unseeded (cold greedy through the offline prior), decode above T=1 rides the
  routed eager dispatch, prefill rides cold symbolic programs. The perf story is M4's (goldens + tiers), same
  sequence as OLMoE M0→M1.

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
   contender. Contenders: **vLLM+emmy (standard + FAST_MATH lanes) on `riftstack/gpt-oss-20b-FP8-Dynamic` vs
   stock vLLM on the native MXFP4 `openai/gpt-oss-20b` vs llama.cpp with a Q8_0 GGUF.**

   The one-shared-checkpoint plan is no longer available (§3.1: the naming convention is emmy-only), so the
   symmetry the grid rests on changes — and mostly for the better. Because the fp8 experts are bit-exact
   re-encodings, **emmy and stock compute on numerically IDENTICAL weights**, which is stronger than the old
   plan's "same checkpoint" and makes the GSM8K gate in step 4 a clean serving-correctness test: any accuracy gap
   is a bug, not a quantization difference. The asymmetry moves to STORAGE — stock reads ~4.25-bit expert weights
   where emmy reads 8-bit, so stock has roughly half the decode weight traffic on the dominant tensors. That
   favors stock and must be reported as such; it is the cost of not having native FP4 kernels, not a measurement
   artifact. Stock vLLM computes W8A8-dynamic where emmy computes W8A16 — also reported.

   For llama.cpp, pick the Q8_0 variant deliberately: the 12.1 GB GGUFs converted from the release keep the
   expert tensors at MXFP4 (arithmetic in §3.1), so only a ~22.3 GB Q8_0-from-bf16 conversion is a genuine 8-bit
   contender — and that one is lossy where ours is not.

   Optional, if a shared-checkpoint arm is wanted back: re-emit under whatever tensor spelling vLLM's
   `GptOssExperts` compressed-tensors path accepts. Unverified that such a spelling exists; not on the critical
   path.
4. **GSM8K sanity check.** lm-eval (`local-completions`) subset against each serving endpoint, fixed seed/config;
   emmy's score within noise of stock vLLM's (same weights ⇒ any gap is a serving bug).

#### M4 step-1 findings — golden seeding (2026-08-07, RTX 5090, `~/checkpoints/gptoss20b-fp8-emmy`)

`emmy/compiler/pipeline/search/goldens/rtx5090_sm120_gptoss20b.yaml`: 17 entries from manual pinned exploration
(no tuner) over the SERVING programs themselves. Method note: the isolated golden-snippet form is **not** a faithful
A/B unit for this model — the expert program's fork tree differs from the snippet's (below) — so every row was
explored by pinning `EMMY_KNOBS` on the actual `_compile_split` program and reading per-launch medians
(`benchmark_program`), then re-verified deploying FROM the tier with no pin.

Per-shape results (greedy = the cold pick through the offline prior; `emmy_us` = the deployed in-model kernel time,
split pairs summed):

| shape | tier | greedy µs | recorded knobs | deployed µs | delta | rec |
| --- | --- | --- | --- | --- | --- | --- |
| expert gate_up (1, 5760, 2880) fp8 | m1 | 80.5 | `t128` / `g8k/coop-t` | 13.5 | 6.0× | y |
| expert down (1, 2880, 2880) fp8 | m1 | 228.3 | `t256` / `g16k/coop-t` | 15.6 | 14.7× | y |
| expert gate_up (16, 5760, 2880) fp8 | m16 | 38.9 | `w1x4` / `f1x1/k4` / `d4/tma` | 16.4 | 2.4× | y |
| expert down (16, 2880, 2880) fp8 | m16 | 1515.5 | `t128` / `g8k/coop-t` | 129.5 | 11.7× | y |
| expert gate_up (256, 5760, 2880) fp8 | m256 | 96.7 | `w2x4` / `f2x2/k4` / `d2/tma` | 86.8 | 1.11× | y |
| expert down (256, 2880, 2880) fp8 | m256 | 23310.1 | `t128` / `coop-t` | 1905.3 | 12.2× | y |
| expert gate_up (512, 5760, 2880) fp8 | .dynM | 218.3 | `w2x4` / `f2x2/k4` / `d2/tma` | 123.7 | 1.76× | y |
| expert down (512, 2880, 2880) fp8 | .dynM | 3950.2 | — | 3611.3 | 1.06× | **n** |
| q_proj (1, 4096, 2880) | m1 | 60.8 | `t128` / `g8k/coop-t` | 11.5 | 5.3× | y |
| k/v_proj (1, 512, 2880) | m1 | 57.6 | `t128` / `g8k/coop-t` | 7.4 | 7.8× | y |
| o_proj (1, 2880, 4096) | m1 | 84.6 | `t128` / `g8k/coop-t` | 11.5 | 7.4× | y |
| q_proj (16, 4096, 2880) | m16 | 23.8 | `w1x4` / `f1x1/k4` / `d4/tma` | 12.4 | 1.93× | y |
| k/v_proj (16, 512, 2880) | m16 | 17.6 | `w1x4` / `f1x1/k4` / `d2/tma` | 10.3 | 1.71× | y |
| o_proj (16, 2880, 4096) | m16 | 21.8 | `w1x4` / `f1x1/k4` / `d2/tma` | 14.4 | 1.51× | y |
| q_proj (512, 4096, 2880) | .dynM | 93.1 | `w2x4` / `f2x2/k4` / `d2/tma` | 81.9 | 1.14× | y |
| k/v_proj (512, 512, 2880) | .dynM | 202.8 | `w2x4` / `f2x2/k4` / `d2/tma` | 24.7 | 8.2× | y |
| o_proj (512, 2880, 4096) | .dynM | 113.8 | `w2x4` / `f2x2/k4` / `d2/tma` | 89.8 | 1.27× | y |
| rms_norm k2880 | m1 / m16 / .dynM | 6.2 / 8.2 / 6.2 | — | — | — | **n** |

Per-program totals, deployed from the tier: expert 310.6 → **31.1** µs (m1), 1561.8 → **148.6** (m16), 23607 →
**1972** (m256), 4257 → **4055** (.dynM); dense `pre` 185.5 → **34.0** / `post` 95.3 → **19.7** µs (m1), 66.9 →
**43.9** / 31.9 → **25.7** (m16), 516.2 → **150.5** / 126.0 → **101.0** (.dynM).

- **Every M=1 shape deployed the KNOB-LESS SCALAR form cold** — one thread per output element, no cooperative
  reduce, 57–228 µs each. The transposed block coop under a grid split-K (`g<n>k/coop-t`) puts all of them at the
  weight-streaming floor. This is the same class OLMoE M2 hit, and it is the whole c=1 story: 24 layers × (4 dense
  projections + 4 expert slots × 2 matmuls) per step.
- **The boot roofline audit cannot see the expert programs.** Its floor is `const_bytes / dram_bw`, and expert
  weights are program INPUTS, not constants — so `const_bytes` is 0, the floor is 0, and every expert twin is
  skipped by the `MIN_FLOOR_US` gate. M3b's "roofline clean" boot was serving a 310 µs M=1 expert program. A
  weight floor that counted the bound INPUT slabs for the MoE tiers would have surfaced it at boot.
- **The expert `down` matmul cannot reach the warp mma tier at any width** — the binding residual. Its A operand is
  the clamped-SwiGLU cone that loop fusion pulls out of `gate_up`'s epilogue, and that computed-A cone over the
  strided `chunk(2)` halves declines the staged transports, so `TILE`/`WORK`/`STAGE` realize `off` on that node
  under any pin and the fork falls back to the scalar family. Measured cost of the lockout: a plain-A twin of the
  same extents runs 15.8 µs at M=16 against the fused form's 129.5, and at M=256 the fused form is 1905 µs against
  a 45 µs cuBLAS HGEMM. The routing (`PLACE`) lever does not reach it either — the recognized tree exposes exactly
  one cuttable seam (the bias accumulator), not the A cone.
- **The isolated snippet is not a faithful A/B unit here.** At M=16 the fp8 snippet's own fork tree offers only the
  scalar tier while the in-model `gate_up` reaches the warp tier, and the in-model `down` is computed-A while an
  isolated `y @ w + b` is not. Both directions of error are large (2750 vs 44 µs, 27.8 vs 1515 µs), so the whole
  sweep ran on the serving programs.
Serve A/B after seeding (pack deleted, cold boot ~5 min for 149 plans, roofline audit clean, KV 114,161 tokens,
whole-step decode capture at size 1, greedy completions fluent and factually right — 512 in / 128 out,
`vllm bench serve`, medians of 3 warm runs on fresh prompts; stock re-measured the same day on the native MXFP4
checkpoint):

| metric | M3b (cold) | seeded | stock vLLM | remaining gap |
| --- | --- | --- | --- | --- |
| c=1 TPOT | 61.4 ms | **5.90 ms** (10.4×) | 3.51 ms | 1.68× |
| TTFT, 512-token prompt | 5.9 s | **0.79 s** (7.5×) | 33.5 ms | 23.6× |
| c=8 TPOT | 356 ms | **62.8 ms** (5.7×) | 6.27 ms | 10.0× |
| c=8 output throughput | — | 83 tok/s | 1060 tok/s | 12.8× |

Where the residual sits: **kernel quality, not dispatch structure — and one specific kernel.** c=1 is now within
1.7× of stock with the whole emmy-side GPU budget at ~4.3 ms/step (expert 31.1 µs × 96 launches = 3.0 ms + dense
53.7 µs × 24 = 1.3 ms), so the per-expert Python launch chain is no longer the binding constraint at T=1 — the
fixed-slot tier's captured step absorbed it. Every remaining gap is the expert `down` matmul's warp-tier lockout:
at c=8 the multi-row experts route to the m16 twin whose `down` is 129.5 µs against a 15.8 µs plain-A twin, and at
prefill the m256 `down` is 1905 µs against a 45 µs HGEMM — 24 layers × ~32 experts of that IS the 0.79 s TTFT.
**Verdict for M4 sequencing: the sorted grouped-GEMM pass should NOT come first.** Grouped GEMM re-shapes dispatch,
which c=1 no longer needs, and it would inherit the same computed-A cone; fixing the fusion legality so the
expert `down` reaches the warp tier is the higher-value and strictly prior work.

- Enabling work landed alongside (uncommitted with the goldens): `matmul_snippet` spells the fp8 W8A16 form (fp8
  storage input + in-graph dequant cone) so an `dtype: fp8` golden has a real reproducer; `run._bind_inputs` binds
  an fp8 torch input through its uint8 carrier (it raised `unsupported ScalarType Float8_e4m3fn`, which blocked
  every fp8 `run --bench`); and the golden featurization gate now exempts the scalar-coop FORM at any M, not only
  M=1 (the expert `down` rows are legitimately tile-geometry-free).

#### M4 step-2 findings — the expert `down` cone legality (2026-08-07, RTX 5090)

The binding residual M4 step 1 named is CLOSED. Root cause, and what it was NOT: the strided `chunk(2)` halves are a
red herring — `_atomize.map_cone` accepts a `Load` at any index, and the wrapper spelling never mattered. Two stacked
defects, both in loop fusion:

1. **The clamped-SwiGLU cone was inlined into the down matmul's K loop.** Every guard in
   `loop/fusion/010_merge_loop_ops` missed it. The transcendental brake never fired because `sigmoid` is not in
   `_EXPENSIVE_OPS` (it is in the `sfu_trans` cluster of `ir/elementwise._OP_CLUSTERS`, the classification's real
   home — the fusion list is a second, diverged copy), and because `_has_peer_activation_input` exempts a gated
   activation anyway. The aggregate ratios could not see it either: the contraction's own operand traffic dominates
   every sum, and the cone arrives one small pointwise hop at a time (each merge grew `_total_work` by ~25% against a
   factor of 8). Cost of the inlining alone, same knobs: 127.7 → 47.7 µs at M=16.
2. **The inlined term stopped binding as a contraction.** `bind_contraction` has an arm for a computed-A map cone with
   a plain B, and an arm for the mul-hoist (a storage decode times k-invariant factors) on either side — but nothing
   COMPOSES them. The fp8 down has both: a general map cone on A and a decode cone on B, so it raised, the recognizer
   demoted the cell to `PLANAR`, and the scalar tier was all that was left. That is why `TILE`/`WORK`/`STAGE` realized
   `off` under every warp pin. Cost of the tier lockout: another 47.7 → 11.2 µs at M=16. (This is the same gap M3a
   recorded, not a second one; `_sum_contracts_exp_producer` is NOT involved — gpt-oss's `sigmoid` never decomposes to
   `exp`, so the flash-consumer protection never fires on this cone. It would fire on OLMoE at fp8, whose SiLU does,
   and that remains an open false positive.)

**The fix — one structural guard in `010_merge_loop_ops`** (`_replicated_load_site` × `_reads_more_than_it_writes`; see
`passes/ARCHITECTURE.md`): a producer whose load site sits under an enclosing loop its index does not use is
re-evaluated once per step of that loop, which is a win only while the cone reads no more than it writes. A gated
activation reads both halves of the packed gate/up buffer — twice the volume it produces — so it stays materialized.
The rule reads VOLUME, so it holds for the option-(a) wrapper spelling (two separate gate/up buffers) as well; no
serving-side change was needed and OLMoE's path is byte-unchanged (its SiLU already materialized, via the `exp`
brake). Option (c) was not needed: the term binds plainly once the cone is out.

Per-kernel, gpt-oss expert program, fp8, 5090 (`benchmark_program` per-launch medians; before = the M4 step-1 goldens):

| width | down before | down after | gate_up before | gate_up after | expert program before → after |
| --- | --- | --- | --- | --- | --- |
| m1 | 15.6 | **6.6** | 13.5 | 13.5 | 31.1 → **18.9** µs |
| m16 | 129.5 | **10.1** | 16.4 | 14.0 | 148.6 → **26.6** µs |
| m256 | 1905.3 | **44.8** | 86.8 | **76.2** | 1972 → **124.7** µs |
| .dynM (512 rows) | 3611.3 | **129.9** | 123.7 | 141.1 | 4055 → **277.1** µs |

The m256 `down` now runs UNDER the 16-bit cuBLAS baseline at the same extents (44.8 vs 45.3). Goldens re-seeded in
`rtx5090_sm120_gptoss20b.yaml`: `expert_down.m16` and `.m256` replaced (the recorded scalar rows were the fused-cone
era's best and are unreachable now — a stale match, not a miss: they still keyed the new kernel and pinned it to the
scalar family, which is why the fix looked like only a 2.7x until they were replaced); `expert_gate_up.m256` moved
TMA → cp.async (76.2 vs 87.6, reproduced twice at <1% spread); `expert_down.m1` keeps its knobs (every coop-t variant
ties within 2%) with `emmy_us` refreshed. The symbolic `down` needs no entry — greedy reaches 129.9 µs unaided.

Serve A/B (5090, same protocol as M4 step 1 — 512 in / 128 out, `vllm bench serve`, pack deleted, cold boot 5.5 min,
149 plans, roofline audit clean, KV 114,592 tokens, whole-step decode capture at size 1, greedy completions fluent):

| metric | M4 step-1 goldens | after | stock vLLM | remaining gap |
| --- | --- | --- | --- | --- |
| c=1 TPOT | 5.90 ms | 6.05 ms | 3.51 ms | 1.72× |
| TTFT, 512-token prompt | 0.79 s | **0.133 s** (5.9×) | 33.5 ms | 4.0× |
| c=8 TPOT | 62.8 ms | **43.2 ms** (1.45×) | 6.27 ms | 6.9× |
| c=8 output throughput | 83 tok/s | **172 tok/s** (2.1×) | 1060 tok/s | 6.2× |

TTFT is where the m256 / `.dynM` `down` lived, and it moved 5.9×. **c=1 TPOT did NOT move** even though its expert
program went 31.1 → 18.9 µs (96 launches/step = 1.2 ms of GPU time removed). The launch COUNT is the reason: the
materialized activation adds 3 launches per expert program (the `_pending_contraction_half` over-fire above), so the
captured decode step gained ~290 graph nodes — enough to eat the kernel-time win at T=1. Collapsing the activation to
one kernel is therefore the next decode lever, and it is a fusion-guard fix, not a knob.

**Grouped-GEMM verdict, updated.** Still not first, and now for a second reason. c>1 improved 1.45× from kernel
quality alone with dispatch untouched, and prefill improved 5.9×, so neither is dispatch-bound at the margin. The one
place dispatch now binds is c=1, where the constraint is the per-step LAUNCH COUNT inside the captured graph — which a
grouped pass would help only by also merging the activation, i.e. the cheap fusion fix does it first.

Cross-model effect: none. gemma-4's serving twins are byte-unchanged (the drift gate passes on both cards with no
baseline edits), because its GeGLU cone already materialized through the transcendental brake. An earlier revision of
the volume test counted a per-channel norm weight as a row stream — at M=1 it has exactly the output's element count —
and refused the decode-width norm→linear edge on gemma; the rank-and-per-axis-width form fixed that, and it is the
reason the test reads shape rather than numel. `scripts/digest_kernels.py` is byte-identical. OLMoE's expert kernels
are byte-identical too (`k_linear_reduce_27d8c2` / `k_linear_pointwise_ed2f1f` / `k_linear_34e1f8` before and after);
its c=1 TPOT re-measured 3.13 ms against a 2.84 ms record, which is config/pack drift rather than this change.

Two residuals recorded rather than fixed:

- **`_pending_contraction_half` over-fires.** It defers any compute producer whose single consumer is an add-`Accum`
  kernel reading its buffer — including a fully ASSEMBLED contraction, which is not "a half waiting to merge". The
  consequence is that the expert activation deploys as THREE kernels (two chunk-half cones + the combine, ~2.4 µs at
  M=1) instead of one. Narrowing the test to `_is_reduce_partner_merge`'s direct `acc += load(product)` form fixes
  that and collapses it to one kernel, but de-certifies the normed-GQA flash unit
  (`test_normed_gqa_sdpa_certifies_flash`) — so it is left alone.
- **`_EXPENSIVE_OPS` is a second copy of the `sfu_trans` cluster** and is missing `sigmoid`. Deriving it from
  `_OP_CLUSTERS` would also import that table's deliberate `relu` inaccuracy, so it needs its own decision.

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

## 6. Where this stopped (2026-08-07)

Scope was cut to **the stack only**: the MoE serving seam, the dispatch mechanisms, and fp8 weights as program
inputs. No model-benchmark deliverables — M4's grid, the release image, and the M5 article are dropped, and the
gpt-oss golden file was removed with them (it was explored against an fp8 attention B operand that the lossless
checkpoint no longer has; see the §M3 carry-over note). The OLMoE golden file stays: it was measured against a
real bf16 model and its expert/dense shapes are the ones the seam actually deploys.

**Why the gpt-oss delivery target does not close.** The model ships MXFP4 and there is no higher-precision source
— OpenAI's own reference copy (`original/dtypes.json`) carries 48 FP4 expert tensors + 48 UE8 block scales
against 267 bf16 tensors, so the 4-bit values ARE the trained weights. An fp8 re-encode is therefore exactly
information-preserving and exactly 1.85× the bytes: 22.7 GB against the release's 12.1 GB, for identical
numerics. Decode is weight-bandwidth-bound, so a head-to-head against an engine reading the native format is lost
before any kernel runs, and no kernel work changes that. The compatibility argument for an fp8 build is also
weak: vLLM's MXFP4 support declares `get_min_capability() = 80`, so Ampere and newer all serve the release
directly, and llama.cpp and transformers read it too. The only consumer that needs 8-bit here is emmy itself,
which has fp8 kernels and no fp4 path.

**What would make gpt-oss servable on its own terms**: an MXFP4 weight path. The decode is cheap — 16 code
points through a lookup, and a power-of-two block scale — and the fp8 landing already built the carriers
(uint8 bits, byte-slab staging, dtype-gated warp tiers) plus this branch's indirect operands. The one new
mechanism is applying the scale at k-block boundaries in the f32 accumulator, which is tractable because the
MXFP4 block is 32 elements along K and aligns with the mma k-step. That is the natural sequel, and it overlaps
with `plans/vq-weight-compression.md`.

**Unfinished, deliberately left**: the expert down-projection cone still cannot reach the warp mma tier (its A
operand is the clamped-SwiGLU cone fusion pulls out of the gate_up epilogue); the boot roofline audit is blind to
expert programs (`const_bytes = 0` when weights are inputs); and the gpt-oss greedy-parity gate needs re-running
now that the lossless checkpoint removes the "different weights" explanation for the divergence M3b measured.
