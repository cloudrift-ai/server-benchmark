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

1. Ingest `RedHatAI/gpt-oss-20b-FP8-Dynamic` (compressed-tensors fp8) through the fp8 lane; clamped-SwiGLU
   elementwise; router-with-softmax-over-selected-k in the torch seam.
2. Serve on the 5090; accuracy gate vs the bf16 HF reference; smoke A/B vs stock vLLM to confirm the perf story
   holds before investing in M4's formal validation.

#### M3 findings (2026-08-07, RTX 5090, `~/checkpoints/gptoss20b-fp8-emmy` — self-quantized 22.1 GB fp8)

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
   contender. Contenders, all at 8-bit weights — identical decode weight bytes, no handicapped lanes:
   **vLLM+emmy (standard + FAST_MATH lanes) vs stock vLLM on the SAME `RedHatAI/gpt-oss-20b-FP8-Dynamic`
   checkpoint** (vLLM serves compressed-tensors fp8 on sm_120 natively) **vs llama.cpp with the Q8_0 GGUF**
   (~8-bit analog). The remaining asymmetry — stock vLLM computes W8A8-dynamic where emmy computes W8A16 — is a
   legitimate engine difference, reported. The native MXFP4 path is out of scope for the benchmark matrix; the
   article's methodology notes it exists (~4.25-bit weights) without benching it.
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
