# GLM-4.5-Air at 2 bpw on a single RTX 5090 — trellis weight compression with in-kernel decode

Goal: serve **GLM-4.5-Air 106B-A12B** from an EXL3 (trellis-coded, QTIP-class) checkpoint at ~2.0 bpw on one
RTX 5090 (32 GB), decoding weights inside emmy's kernels, and beat exllamav3 on its own checkpoint — first at
batch 1, then decisively under concurrency, where continuous batching wins by construction.

This is the highest-leverage single target available on consumer hardware. GLM-4.5-Air does not fit a 5090 by any
other route: bf16 is ~212 GB, FP8 ~106 GB, INT4 ~53 GB. Only sub-3-bit trellis coding puts it on the card at all,
and the only engine that reads those checkpoints is single-user-focused. **Stock vLLM appears in every table as
"cannot run this model on this card"** — that absence is the headline, and it means the result is not reproducible
by config-tweaking an incumbent.

Final deliverables (all four are required for done):

1. `emmy/compiler/pipeline/search/goldens/rtx5090_sm120_glm45air.yaml` — seeded goldens for the serving programs.
2. A prebuilt serving image with the model baked, via the `make serve-*` release pipeline.
3. Recipes + experiment grids benchmarking emmy against exllamav3 on the identical checkpoint.
4. An article in `cloudrift-landing`, alongside `optimizing-gemma-4-12b-rtx`.

## 1. The checkpoint

**Source: [`turboderp/GLM-4.5-Air-exl3`](https://huggingface.co/turboderp/GLM-4.5-Air-exl3)** — the format author's
own conversion, one branch per rung: 2.00, 2.25 (optimized), 2.50 (optimized), 3.00, 3.07 (optimized), 3.50
(optimized), 4.00 bpw. Base model `zai-org/GLM-4.5-Air`, MIT license. The "optimized" rungs are EXL3's mixed
allocation — bits spent by Hessian sensitivity rather than uniformly — which is the format's real advantage over
uniform QTIP.

| rung | weights | KV headroom on 32 GB | verdict |
| --- | --- | --- | --- |
| **2.00 bpw** | ~26.5 GB | ~2.5 GB | **the target** — the only comfortable on-card fit |
| 2.25 opt | ~29.8 GB | ~0 GB | no room; viable only with quantized KV, treat as stretch |
| 3.00 bpw | ~40 GB | — | off-card |

Fallback if 2.00 bpw fails the quality gate: **`ArtusDev/cerebras_GLM-4.5-Air-REAP-82B-A12B-EXL3`** (pruned 82B) at
2.25 bpw ≈ 23 GB, which buys headroom to spend on a higher effective bit rate.

Pin `(repo, branch, commit sha)` in `models/<slug>.env` and in the article. EXL3 is a single-maintainer,
actively-developed format; a re-cut checkpoint under the same branch name must not silently change results.

**turboderp publishes no quality metrics for this model.** That is the single largest unknown in the plan and
Phase 0 exists to close it.

### The quality risk, stated precisely

Measured evidence from a sibling model — `NeuroSenko/Qwen3-Coder-Next-exl3`, 80B-A3B, uniform rungs, measured under
exllamav3 v0.0.22:

| bpw | size | KL-div (quant→orig) | PPL |
| --- | --- | --- | --- |
| 2.0 | 20 GB | 0.4111 | 8.85 |
| 3.0 | 29 GB | 0.1613 | 8.07 |
| 8.0 | 75 GB | 0.0090 | 7.71 |

2 bpw costs **+1.14 PPL** there, roughly double QTIP's +0.58 on dense Llama-2-70B. The working hypothesis is that
this is a **sparsity effect, not a MoE effect**: Qwen3-Coder-Next activates ~3B of 80B, leaving very little
redundancy per token to absorb quantization error. GLM-4.5-Air activates ~12B of 106B — about 4× denser per token —
so it should degrade substantially less at the same bit rate. **This hypothesis is unverified and Phase 0 tests it
directly.** If Air at 2.00 bpw lands near KL 0.4, the model is not worth serving and the plan pivots to the REAP-82B
fallback.

## 2. Format background — why EXL3 and nothing else

| Format | true bpw @"2-bit" | W2 PPL, Llama-2-70B | decode mechanism | decode vs FP16 gemv | produce 70B |
| --- | --- | --- | --- | --- | --- |
| QTIP (HYB) | 2.00 | **3.70** | trellis walk, 2–4 ALU instr/wt, ≤2 KiB LUT | ~3.4×, >80% peak BW | ~tens of A100-h |
| EXL3 | 1.6–8 mixed | QTIP-class | QTIP-style, Marlin-inspired kernel | memory-bound | **hours on one 4090** |
| QuIP# E8P | 2.02 | 3.91 | act. Hadamard + ~1 KiB LUT + sign flips | ~3.2× | ~100 A100-h |
| AQLM 1x16 (+PV) | 2.07 | 3.94 / 3.78 PV | 1 MiB LUT, gathered from L2 | 1.2–1.5× | ~720 A100-h |
| HQQ 2-bit | ~2.5 | weak w/o LoRA | scalar dequant | trivial | minutes |

QTIP supersedes QuIP# because vector quantization is hard-capped at low dimension — a dimension-`d` codebook at
rate `R` needs `2^(dR)` entries, so E8P is stuck at d=8 and leaves shaping gain on the table. Trellis coding
quantizes a very high-dimensional vector with a small state machine instead, cost linear in sequence length rather
than exponential in dimension, and reaches near-optimal shaping with a ≤2 KiB LUT. Both share QuIP#'s random
Hadamard incoherence processing; QTIP swaps only the quantizer. Tail-biting keeps each tile's walk self-contained,
so tiles decode in parallel — a 256-weight 16×16 tile is one walk, matching the mma tile.

**QTIP is not a checkpoint option here.** The relaxml org publishes 30 QTIP repos, all Llama-2/3.x, last active June
2025 — no Qwen, no GLM, no MoE. QuIP# is explicitly dormant, its README pointing at QTIP. For any model we want to
serve, trellis coding is available **only** as EXL3. relaxml's Llama-2-70B QTIP-2Bit remains useful as a
paper-anchored, bit-exact decode fixture on a dense architecture (Phase 1).

AQLM 1x16 is the cautionary tale: a 1 MiB per-matrix codebook cannot live in smem, so every group decode is a random
gather from L2, wasting most of the byte reduction (81.5 tok/s vs QTIP's 188 at 7B). Any format we pick must keep
its LUT/state in smem or registers.

## 3. Compression arithmetic and the ceiling

Effective bpw includes codes + per-channel scales + codebooks; at 106B scale the overhead is small, so ~2.0–2.1 bpw
effective. Against what we would otherwise deploy:

| baseline | bytes/weight | ~2.1 bpw ratio |
| --- | --- | --- |
| bf16 checkpoint | 2.0 | **~7.6×** |
| FP8 (W8) | 1.0 | **~3.8×** |
| INT4 / AWQ | 0.5 | ~1.9× |

Decode-phase gemv is memory-bound, so the byte reduction is the latency ceiling and QTIP demonstrates >80% of peak
bandwidth is reachable. The freed VRAM goes to KV and admission capacity, which is the whole point on 32 GB.

**Measurement caveat, and it is load-bearing.** `run --bench --golden` replays one kernel over one weight slab
~100×, so any slab under the 5090's 96 MB L2 is timed L2-resident and a bandwidth win will **not** appear. VQ A/Bs
must use shapes bigger than L2, twin/serving end-to-end latency, and the roofline audit
(`emmy/serving/roofline.py`) — never isolated golden benches.

## 4. Baselines

Per the article's contender set, all on the same 5090:

- **Primary: exllamav3 / tabbyAPI on the IDENTICAL checkpoint** — apples-to-apples on weights and quality; the fight
  is pure kernels plus serving. Batch-1 TPOT/TTFT first (their home turf), then a concurrency sweep
  (c = 1/4/8/16, request rate + goodput), where continuous batching and paged KV should win by construction.
- **Secondary: llama.cpp** with the popular GGUF at matched bpw (Unsloth UD-IQ2 class; `IQ2_KT` via ik_llama.cpp if a
  trellis-vs-trellis point is wanted) — what most of the community actually runs today.
- **Quality gate, not a perf baseline**: our PPL/KL against the bf16 reference must match exllamav3's on the same
  checkpoint. Decode is bit-exact reconstruction, so any gap is a bug, not a tradeoff.
- **Stock vLLM**: recorded as unable to load the model on this card. That row is the headline.

## 5. Compiler integration

Shared with `plans/fp8-support.md` — do not re-design:

- **The three-layer type discipline**: codes enter as plain scalar dtypes (`u8`/`u16` bits carriers, bf16 precedent)
  or a packed `StructuredType`; codebooks/scales are first-class sibling tensors; the pairing lives in
  `ConstantOp.quant`. Generalize the FP8 plan's `QuantSpec` (codes/codebook/scale paths + format params) rather than
  adding a second field. Stamped at the constant's birth site (`trace/torch.py`, `_handle_placeholder`).
- **Trace is quantization-blind**: trace the bf16 architecture twin from config, bind real tensors via the
  safetensors path. Required here more than for fp8 — an EXL3 module has custom kernels and int parameters that
  survive neither `torch.export` nor the `.float()` cast in `bind_constants_from_module`.
- **`ShapeKey` dtype-class field** and the `992` structural stamps, golden `_DTYPES` extension, manual pinned `--ab`
  seeding, `eval golden --in-model` deploy verification.

What is genuinely new here:

- **The decode never commutes out of the fold.** FP8's easy branch (scale constant along K → epilogue multiply) does
  not exist; a trellis walk is data-dependent per group. Every VQ matmul is a computed-B form.
- **The tile IR already stores it**: `Channel.b: Load | Fold` — a computed B edge is representable today; every tier
  just declines it (`isinstance(c.b, Load)` gates in `_legality.py` and `_schedule.py`). The work is a computed-B
  reading in `_schedule.py` (mirror of the existing mixed-A promotion `_promoted`) plus legality arms. Readings
  constraint: at most two per term, mutually exclusive by shape — `_enumerate` raises on a `canonical_row_key`
  collision.
- **The decode lands as a B-side compute fill**: a `SyncOperand` whose value closure emits code load → trellis
  step → optional scale multiply, writing the decoded slab the existing ldmatrix drain reads. Today only computed-A rides
  the sync fill ("B weights never ride this" — that docstring changes). Codes stream through the existing async
  prefetch ring (`cp.async` on `async_operands`; TMA is out — a descriptor needs a gmem address on both edges); the
  LUT loads once via the `sync_stat_fill`-style prologue slot.
- **The fragment path must land before any perf judgment.** A dtype-boundary copy on an operand cone demotes the
  matmul off the mma tier (measured 1.12 vs 1.61 TB/s), and a gather cannot be absorbed by copy transports at all.
  Sequence the warp-tier decode first or every A/B undersells the format.
- **The gemv/reduce tier needs the decode too.** Decode-phase matvecs derive PLANAR and take the reduce tiers, which
  is not where the sync fill lives. This is where the TPOT win cashes out — first-class, not an afterthought.
- **Activation-side Hadamard transform** rides the computed-A machinery: a per-tile prologue on the A operand cone,
  the same structural slot as the norm→linear fusion (`"fused"` golden kind). At decode M the A tile is tiny.
- **Bind-time dequant costs nothing to build**: `load_ops` already executes frontend ops through the numpy backend
  and `GatherOp` has a numpy `forward()`. A numpy trellis decode helper plus loader-side sibling-tensor lookup gives
  a correctness lane with zero kernel changes — full-size weights in memory, no footprint win, but it unblocks
  accuracy A/B everywhere.

MoE is no longer a gap: expert weights as program inputs, indirect operands, and fp8 expert kernels landed on
`feature/moe-support` (gpt-oss-20b serves at fp8 on the 5090). GLM-4.5-Air's experts are plain linears; the routing
machinery exists. Expert weights are the bulk of the bytes and only routed experts' tiles are read per token, so the
bytes-bound argument holds per expert.

Serving notes: emmy owns trunk weight loading end to end (the plugin's `load_weights` skips the state dict), so a
codes+codebooks+scales checkpoint is entirely ours to load. Blockers: the `.float()`/np-dtype casts in the bind
paths, and `_encode_load_ops`'s restricted vocabulary — it silently disables pack saving for chains outside it, so
extend it or prebuilt images stop getting pack hits.

---

# Execution phases

Each phase is scoped for an independent agent: explicit inputs, a deliverable, a verification command, and abort
conditions. Phases 0–3 are strictly sequential. Phases 4–7 depend on 3.

## Phase 0 — Checkpoint and baseline recon (GATE, no emmy code)

**Hardware**: one RTX 5090. **Depends on**: nothing.

1. Resolve the exact branch name for the 2.00 bpw rung of `turboderp/GLM-4.5-Air-exl3`; record `(repo, branch,
   commit sha)` and the on-disk byte size.
2. Install exllamav3 (latest release; record the version) + tabbyAPI. Load the checkpoint on the 5090. Record actual
   VRAM occupancy and the maximum context that fits alongside it.
3. **Measure quality**: KL-divergence (both directions) and perplexity against the `zai-org/GLM-4.5-Air` bf16
   reference, using the same methodology NeuroSenko documents. If a bf16 reference will not fit locally, measure
   against the 4.00 bpw rung and state that substitution explicitly.
4. Measure exllamav3 baseline serving: batch-1 TTFT/TPOT, then a concurrency sweep to whatever it supports. Capture
   raw numbers and the exact client invocation for reuse in Phase 6.
5. Read the EXL3 packing format and kernel source until the decode is fully understood — trellis state, tile layout,
   Hadamard placement, scale layout, per-tensor bpw allocation in the "optimized" rungs.

**Deliverable**: a findings note in `plans/` covering the pinned checkpoint coordinates, measured quality, baseline
numbers, and a decode-format writeup precise enough to implement against.

**Abort/pivot conditions** — this is the gate:

- KL-div ≳ 0.35 or PPL degradation ≳ +1.0 → **do not proceed on 2.00 bpw**. Re-run the measurement against
  `ArtusDev/cerebras_GLM-4.5-Air-REAP-82B-A12B-EXL3` at 2.25 bpw and retarget the whole plan to it if it passes.
- Checkpoint will not load or leaves < 1.5 GB for KV → retarget to the REAP-82B fallback.
- Report the sparsity hypothesis outcome explicitly (Air ~12B active vs Coder-Next ~3B active): it decides how we
  talk about 2-bit MoE in the article.

## Phase 1 — Reference decode in numpy (no kernel work)

**Hardware**: CPU. **Depends on**: Phase 0's format writeup.

1. Implement the EXL3 trellis decode as a numpy helper riding `load_ops`: code load → trellis walk → scale →
   optional Hadamard undo.
2. Validate **bit-exactly** against exllamav3's own dequantization for a handful of tensors from the pinned
   checkpoint. Bit-exact is the bar — decode is reconstruction, not approximation.
3. Cross-check the same walk against `relaxml/Llama-2-70b-QTIP-2Bit` where the format overlaps, as a
   paper-anchored fixture on a dense architecture.

**Deliverable**: the decode helper plus tests. **Verify**: tests pass with exact equality, not tolerance.

**Abort**: if bit-exactness cannot be reached, stop and report — every later phase assumes reconstruction is exact,
and a tolerance-based decode makes the Phase 6 quality gate meaningless.

## Phase 2 — Loader and bind-time dequant (correctness lane)

**Hardware**: 5090 helpful, not required. **Depends on**: Phase 1.

**STATUS: DONE (2026-08-07).** No `QuantSpec` revival (retired by the dissolve-early migration): the EXL3 cone is
in-graph algebra from birth — `spell_trellis_constants` (loader/quant.py) rewrites each coded `.weight` constant into
int16-codes + f16 `suh`/`svh` leaves under a `TrellisDecodeOp` (frontend IR; `cb` from marker presence,
`out_features`/`in_features` slice the 128-multiple encode padding), folded by `032_fold_constant_subgraphs` into a
bind-time `source_graph` record. Twin: `quantized_checkpoint_dir` detects `quant_method: "exl3"`;
`load_dequantized_state_dict` decodes siblings; `load_quantized_twin` trims encode padding and packs per-expert
weights into the v5 3-D expert params; `build_moe_split_wrapper` folds GLM's `shared_experts` into `post_attn`.
Verified (full-model expansion infeasible on the 60 GB box — ~212 GB): per-tensor bind == direct decode bit-exact
(incl. lm_head K=6); `emmy run --layer 0` and a config-truncated dense+lm_head model end-to-end vs eager on the 5090;
the expert program with real decoded layer-1 expert weights on CUDA. Whole-model MoE traces still stop at the router
(`aten.topk` — pre-existing, all MoE archs; serving uses the third seam). Greedy-match vs exllamav3 deferred to
Phase 3 (needs weights kept compressed). Format finding: the quantizer keeps sensitivity-selected linears at plain
fp16 (GLM-4.5-Air layer 0 `o_proj`), and `intermediate_size` 10944 is stored padded to 11008.

1. `QuantSpec` generalization for codes/codebooks/scales; `ConstantOp.quant` pairing; stamp at the constant birth
   site. Extend `loader/safetensors.py` for EXL3's sibling-tensor layout.
2. Clear the `.float()`/np-dtype casts in the bind paths (raw-bits carrier, bf16 precedent).
3. Wire the Phase 1 decode into `load_ops` so a VQ checkpoint compiles and runs everywhere with weights expanded in
   memory.

**Deliverable**: GLM-4.5-Air compiles and produces correct output end to end at **full bf16 footprint** — no
footprint win yet. **Verify**: `emmy run` accuracy against the bf16 twin; a short generation matches exllamav3's
greedy output on the same prompts.

## Phase 3 — In-kernel decode (the real work)

**Hardware**: 5090. **Depends on**: Phase 2.

Order matters — the fragment path first, or every measurement undersells the format:

1. **Warp/mma tier, prefill**: computed-B reading in `_schedule.py` plus legality arms; decode as a `SyncOperand`
   compute fill in the B staging path, once per B tile, amortized across the M tile. Codes through the async
   prefetch ring; LUT once via the prologue slot.
2. **Reduce/gemv tier, decode phase**: the LUT decode needs a reduce-tier realization. This is where TPOT cashes out.
3. **Activation-side Hadamard** as a computed-A-style prologue, if the format requires it at kernel level rather
   than being foldable into the checkpoint.
4. Expert-path integration: the routed expert FFN matmuls are the bulk of the bytes; confirm indirect operands and
   the compute fill compose.

**Deliverable**: GLM-4.5-Air resident on the 5090 at ~26.5 GB with correct output. **Verify**: VRAM occupancy
matches the checkpoint size; accuracy unchanged from Phase 2; a decode-phase step streams compressed bytes (confirm
via the roofline audit, not golden benches).

**Note**: these are storage formats, not accuracy knobs — no FAST_MATH gate. The quantization error is in the
checkpoint, and our decode is exact.

## Phase 4 — Golden seeding

**Hardware**: 5090. **Depends on**: Phase 3.

Seed `emmy/compiler/pipeline/search/goldens/rtx5090_sm120_glm45air.yaml`, following the structure and preamble
conventions of `rtx5090_sm120_gptoss20b.yaml` — which is the closest precedent: a MoE, on this card, whose expert
weights arrive as program inputs.

- Seed over the **serving programs themselves**, not synthetic shapes.
- Cover both the routed-expert matmuls and the dense trunk projections, at prefill and decode shapes.
- Reproduce each entry 2–3× and record the spread; document the measurement method in the preamble.
- The preamble must carry the L2-residency warning — with a 26.5 GB trunk, absolute microseconds from
  `run --bench --golden` do not predict in-model step time.

**Verify**: `emmy eval golden --in-model` on the deployed model; the coverage gate that Phase 5 enforces must pass.

## Phase 5 — Prebuilt serving image

**Hardware**: 5090 (the release pipeline runs on the target GPU). **Depends on**: Phase 4.

Run the documented release workflow in `docker/vllm-emmy-serve/ARCHITECTURE.md` end to end — it is the authority;
do not improvise. `make serve-config → serve-goldens → serve-warm → serve-image → serve-verify → serve-push`.

- Create `models/<slug>.env` with the pinned serving config, including the checkpoint revision. **The config seals
  the cache key and cannot change after warming.**
- Memory headroom is the delicate step here: at ~26.5 GB weights there is very little room. Sweep and pin the
  largest passing `max_num_batched_tokens` / bucket configuration, and expect the answer to be small.
- Clear every gate: golden coverage, HF-parity validation, offline zero-recompile verify.
- **Pause for human approval before `serve-push`.**

**Deliverable**: `cloudriftai/vllm-emmy-<slug>:TAG` serving with no `HF_TOKEN` and no download.

## Phase 6 — Recipes, experiments, benchmarks

**Hardware**: 5090. **Depends on**: Phase 5.

1. `recipes/GLM-4.5-Air-EXL3/recipe.yaml` — the one recommended serving config, what `emmy deploy` runs.
2. `experiments/GLM-4.5-Air-EXL3/serving_rtx5090/recipe.yaml` — the benchmark grid: emmy vs exllamav3/tabbyAPI on the
   identical checkpoint, matrices over `benchmark.max_concurrency` {1,4,8,16} and `benchmark.random_input_len`.
   Add a llama.cpp lane at matched bpw, and a stock-vLLM lane that is expected to fail to load (record the failure —
   it is a result).
3. Reuse the Phase 0 client invocation exactly so baseline numbers are comparable.
4. Run each point 3× and report mean ± stddev; capture power and peak VRAM.
5. Re-run the quality gate on the served model: our PPL/KL must match exllamav3's on the same checkpoint.

**Deliverable**: result manifests backing every article table. **Verify**: one command chain reproduces the
headline table.

## Phase 7 — Article

**Depends on**: Phase 6. **Location**:
`/Users/dikobraz/Projects/cloudrift-landing/packages/blog/content/blog/optimizing-glm-4-5-air-exl3-rtx5090/`

Match the structure of the sibling `optimizing-gemma-4-12b-rtx`: `index.md`, a `benchmark-plan.md`, and a
`benchmark-scripts/` directory holding the driver scripts and the raw result JSON under `data/`.

Story spine:

1. A 106B model that fits a 32 GB card at all — and what it costs in quality (Phase 0's measured KL/PPL, honestly
   reported, including the sparsity finding).
2. Why no production engine can serve it: vLLM closed the EXL3 request as not-planned, llama.cpp cannot read the
   files, exllamav3 reads them but is single-user-focused.
3. What in-kernel trellis decode looks like in a compiler.
4. Numbers: batch-1 vs exllamav3, then the concurrency sweep where the serving stack wins.
5. Honest limits: the quality cost, the tiny KV headroom, the checkpoint-pinning caveat.

**Verify**: every number traces to a committed result manifest; the checkpoint is pinned by commit sha in the text.

## Risks

- **2-bit quality on a sparse MoE** — the Phase 0 gate. Measured evidence from a sibling model is discouraging;
  the sparsity hypothesis says Air should fare better; it is unproven.
- **KV headroom at ~2.5 GB** is very tight. Real admission capacity may be low enough to blunt the concurrency
  argument, which is the main win. Measure it early in Phase 6 and be willing to report it as a limit.
- **EXL3 format stability** — single maintainer, active development. Pin the converter version and checkpoint sha;
  a v0.0.20-cut checkpoint may carry bugs fixed in a later quantizer.
- **Trellis decode in the reduce tier** is the least-precedented piece of kernel work here and the one the whole
  TPOT story rests on.
- `lm_head` / embeddings are the biggest single matrices and every format keeps them at higher precision; confirm
  EXL3's allocation and budget for it in the footprint arithmetic.
