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

## Phase 0 outcome (2026-08-07) — RETARGET to the 2.25-optimized rung, quantized KV required

Phase 0 measured the quality gate directly against the bf16 reference (exllamav3 `model_diff` streams one layer at
a time, so no reference substitution was needed) — full numbers in `plans/vq-phase0-findings.md`. The gate FAILED
on every target this plan named: Air 2.00 bpw at KL 0.409 / ΔPPL +1.97 (both arms), the REAP-82B fallback at
KL 0.394 / ΔPPL +3.96 on a weak pruned base (wiki2 PPL 12.3) that also crashes exllamav3 1.4.0 generation. The
sparsity hypothesis is REFUTED: Air (12B active) matches the 4×-sparser Qwen3-Coder-Next's KL at 2.0 bpw almost
exactly (0.409 vs 0.411) — density is not the lever, and the article must not claim denser MoEs quantize better.

Decision (Dmitry, 2026-08-07): proceed on **Air 2.25-optimized** — KL 0.272 passes the KL arm, ΔPPL +1.21 is
marginally over a line that was always a judgment call, and the mixed Hessian-driven allocation is EXL3's real
advantage — with the quality cost reported honestly, including the 2.00 bpw numbers and the refuted hypothesis as
findings. Consequences for the phases below:

- **Pinned target**: `turboderp/GLM-4.5-Air-exl3` branch `2.25bpw`, commit `6a309ed6…` (cached locally, 29.4 GiB).
  The 2.00 rung (`2.0bpw` @ `a1adde54…`) stays cached as a measured comparison point, not a serving target.
- **Quantized KV is mandatory scope**, not a stretch: at ~29.8 GB weights the card fits few fp16 KV tokens (straight
  arithmetic says 184 KiB/token — 46 layers × 8 KV heads × 128 head dim × fp16 × K and V; exllamav3 measured ~325
  KiB/token marginal, a ~1.8× discrepancy the Phase 5 headroom sweep must resolve). Scoped 2026-08-07: emmy owns no
  KV — the generative carve runs vLLM's paged attention (and its cache) between the pre/post programs — so the
  route is vLLM's own `--kv-cache-dtype fp8_e4m3` (serving-glue only: the `_attn_aliased` fast path in
  `vllm_model_gen.py` bails when KV scales are active, plus serve-command passthrough). An emmy-owned q4 cache
  (exllamav3-style group-32) is a post-Phase-3 stretch: its in-kernel dequant needs exactly the files Phase 3 is
  rewriting.
- Phases 1–3 are unaffected (the decode is rung-independent; K just varies per tensor under the optimized
  allocation, which the decoder already handles). Phase 4+ seeds goldens and benches on the 2.25 checkpoint;
  Phase 6 must re-measure the exllamav3 baselines on 2.25 with the recorded client invocation.
- Expected VRAM arithmetic for Phase 5's headroom sweep: ~29.8 GB weights + activations + KV inside 32 GB — the
  `max_num_batched_tokens` answer will be small; measure, don't assume.

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
   **STATUS: DONE (2026-08-07).** Landed as the computed-B cone over a per-element `TrellisLoad` leaf (the window
   is directly addressable, so no carried walk state and no LUT prologue — the 3INST codebook is computed
   in-registers by `emmy_trellis_decode`); the sync compute-fill decodes the B slab reading codes straight from
   gmem (L1-cached — no codes slab and no `SyncTransport` multi-dtype change needed), the materialized A rides
   the cp.async fill underneath, and the collapse reading is the fallback. Hat-basis accuracy verified on real
   GLM-4.5-Air tensors (K=2 q_proj, K=6 lm_head) at f16 matmul tolerance; 5090 @ N=K=22016 K=2 (codes 121 MB,
   past L2): beats same-shape f16 matmul at M=128 (2.10 vs 2.34 ms), decode-ALU-bound 1.6–1.7× at M=256–2048 —
   the per-element decode re-runs per M-tile row, the standing lever for steps 2 and the fragment-drain follow-up.
   `EMMY_TRELLIS_EXPAND` gates the constant-rooted hat-basis cone in-graph; checkpoint-basis cones still fold
   (step 3 owns the basis-restore rewrite that makes real models reach this kernel).
2. **Reduce/gemv tier, decode phase**: the LUT decode needs a reduce-tier realization. This is where TPOT cashes out.
   **STATUS: DONE (2026-08-07).** Diagnosis first: at M=1 the schedule was declining every cooperative band, because
   the B-orientation classifier read a gmem row stride off the CODES grid for a `TrellisLoad` (whose index is the
   weight's logical `(k, n)`) and answered "k-major", which forces the serial one-thread-per-output fold. NCU on the
   resulting kernel: 29 warp-instructions and 4 bytes of code traffic per 2-bit weight, a whole 16x16 tile touched
   per element, and only 16–43 of 170 SMs occupied — the `down` shape's extra 17x-vs-6x badness was purely its
   smaller output grid (4096 threads = 16 CTAs) over a longer serial K.
   Landed: (a) the classifier answers `None` on a decode leaf — the layout gate has no meaning for a computed B;
   (b) the **decode band**, a decoded-B-only reduce partition = transposed coop band (32 lanes sweeping the output
   axis) at `reg` = the tile's 16 k rows over a cross-CTA split, so a lane's register copies walk one tile COLUMN;
   (c) `055_fuse_trellis_runs`, the peephole that rewrites those 16 per-element leaves into the run form of
   `TrellisLoad`, rendered as one `emmy_trellis_decode_col` — one code fetch, compile-time word indices, ~11.5
   instructions per weight; (d) `ShapeKey.dtype_class == "trellis"` off the `S_dtype_i16` codes carrier, so a
   decoded B never joins its f16 twin's golden / DB rows (it was joining them, and that is what a Phase 4 seeding
   would have baked in).
   Measured on the 5090 at M=1, K=2, greedy vs the same-shape f16 matvec: **N=K=22016 (codes 121 MB, past L2)
   976 → 214 µs vs f16's 580 — 2.7x ahead** (157 µs at the band's best pinned split, 3.7x); gate/up 4096→11008
   161 → 19.6 vs 18.6; down 11008→4096 441 → 20.5 vs 18.7. The two L2-resident shapes are at f16 parity and that
   is the honest reading — f16 streams its weights out of L2 at 4.9 TB/s there, which no 30 GB model sees.
   Residual, with evidence: NCU on the past-L2 band puts SM throughput at 69 % against DRAM 35 %, so the wall is
   the decode's own instruction count (11.5 warp-instructions per weight, ~128 µs of pure issue against a 71 µs
   DRAM floor), not bandwidth. Next levers, in order: fewer instructions per weight (half2 accumulate folded to f32
   on a cadence, as exllamav3 does), and occupancy (79 registers/thread, 46 % achieved). Second residual: the
   offline prior does not rank the band's split widths correctly on a cold compile (it takes `g8k`/`g4k` where
   `g32k` wins, 20–36 % left on the table) — that is what Phase 4's golden seeding is for, and the `trellis`
   dtype class is what makes such a golden keyable.
3. **Activation-side Hadamard** as a computed-A-style prologue, if the format requires it at kernel level rather
   than being foldable into the checkpoint.
   **STATUS: DONE (2026-08-07).** Under the same `EMMY_TRELLIS_EXPAND` gate, `spell_trellis_constants` now
   rewrites the CONSUMING LINEAR instead of the weight constant:
   `x → [pad to k_pad] → ·suh → H → ·1/16 → @ W_hat → ·1/8 → H → ·svh → [slice] → [+bias]`. The 128-block
   Hadamard is plain algebra (a 128x128 matmul over a 128-blocked operand against one graph-wide `HadamardOp`
   constant, symmetric so no transpose) — no butterfly realization was needed. Two placement rules are
   load-bearing: every layout change rides a POINTWISE, never a matmul's activation operand (see the residual
   below), and the `1/sqrt(128)` per side splits into the exact powers of two `1/16` / `1/8` so the constant is
   plain ±1 and intermediates stay below the balanced magnitude. Accuracy vs the checkpoint-basis reference
   (`fold_hadamard`), f16 matmul tolerance: 5e-4 numpy, ≤1.2e-3 on CUDA over M=1..128, K=2/3/6, all three
   codebooks, both padding directions, bias — and on real GLM-4.5-Air `q_proj` siblings. A config-truncated
   one-layer cut of the pinned 2.25bpw checkpoint runs end to end on the 5090 under the gate (7 constants
   spelled, 7 launches carrying `emmy_trellis_decode`, `max_diff 0.0023` vs eager, PASS). NOTE `emmy run
   --layer` does NOT exercise this: the layer path binds constants from the decoded twin MODULE, whose
   parameter paths never reach the checkpoint index, so the speller is a no-op there — use a whole-model trace.
   Hadamard tax on the
   5090 at N=K=22016, K=2 (past L2), against the same matmul with no basis restore: **+4.5 % at M=1, +13.4 % at
   M=128**, in 3–4 kernels (the scale multiplies fuse as computed-A cones) — at/under exllamav3's ~14 %.
   **RE-MEASURED after step 2 (2026-08-07)**: the matvec got 4.6x faster, so the same two Hadamard launches are now
   a bigger share — **+16.7 % at M=1 past L2** (260 → 303 µs, still 1.9x ahead of f16's 585) and **+77 % at the
   L2-resident gate/up shape** (31.8 → 56.4 µs), where the two launches are ~12 µs each of essentially pure launch
   and low-occupancy latency. Checked whether a reduce-tier realization folds them: **not without new machinery, in
   both directions.** The input-side `H128` would have to ride the decode band's A operand, which means either
   re-transforming the whole x vector per CTA (11008 CTAs × a 128-block transform — far worse than the launch) or
   staging the CTA's own k slice as a shared row and transforming it there; the shared-row `Stage` is exactly the
   mechanism, but `_tile_reduce_axis_transposed` asserts it out ("transposed coop cannot ride shared-row staging"),
   so the band would need a staged arm plus a 128-block Hadamard prologue on the row. The output-side `H128 · svh`
   cannot fold into the matvec at all — it groups 128 OUTPUT columns after the reduce, and the reduce's epilogue is
   per-cell; its natural home is the split's finalize kernel, which already reads the `cta × N` workspace and
   projects, and would need a blocked-Hadamard projection arm. Both are real follow-ups; neither is a small edit.
   Two defects fixed on the way: `bind_contraction` handed the decode leaf back as a MATERIALIZED B whenever A
   was a computed cone (the staging gates then byte-copied the packed codes into the slab and the decode
   vanished from the kernel — a hang/fault); and the residual below.
   **RESIDUAL — a pre-existing miscompile, not trellis-specific**: an index map that reaches a matmul's
   ACTIVATION operand (a K-regrouping reshape, an A-side transpose) is silently mis-lowered — the fragment
   loaders take the operand's declared row stride, not the one the index implies. Reproduces on plain f16
   linears with no quantization (`x (128,256) → reshape (256,128) → linear` is wrong on the TMA, gmem-direct and
   reduce paths; `d2/cp` alone is right). The chain above dodges it by construction. Fixing it (derive the
   operand stride from the index, or decline) would collapse the chain to fewer kernels and is the main lever
   left on the tax.
   **RESIDUAL — pack saving on N-padded linears**: their `svh` leaf carries a graph slice that folds into the
   constant's load chain as an `IndexMapOp`, which `_encode_load_ops`' two-form vocabulary cannot express, so
   that one small vector logs "will not rebind from a pack". `suh` is fine (its reshape IS in the vocabulary).
   Extend the vocabulary before Phase 5 bakes an image.
4. Expert-path integration: the routed expert FFN matmuls are the bulk of the bytes; confirm indirect operands and
   the compute fill compose.
   **STATUS: DONE (2026-08-07).** `spell_trellis_inputs` (loader/quant.py) is the input-rooted twin of the 3.3
   speller, sharing one chain builder: it re-mints each named weight input as the int16 CODES buffer in place and
   appends `<name>_suh` (128-blocked) / `<name>_svh` (logical out extent) — the shapes the serving store feeds, so
   a per-expert slice is a plain view. Gate and up stay SEPARATE program inputs
   (`build_moe_split_wrapper(..., split_gate_up=True)`): their `suh` differ, measured on the real checkpoint, so a
   merged gate_up weight has no single activation-side basis. `load_quantized_split` grew the EXL3 arms — the dense
   trunk decodes to values (a trunk weight binds off the twin MODULE, which has no checkpoint path; see the
   residual), the routed experts keep their codes, stacked E-leading per `(layer, projection, leaf)`. The indirect
   split the brief flagged holds as designed: the nine per-expert tensors are table-resolved, the shared Hadamard
   stays a graph CONSTANT. The chain also had to accept a SYMBOLIC leading dim (the `moe.expert.sym` program) —
   only the contraction dim needs to be static, and 3.3's constant path now spells symbolic traces too.
   **Pack vocabulary CLOSED.** `WeightSpec.source_op` is the plan's third pre-chain source form
   (`("hadamard", (128,))`, rebuilt by `build_source_op`): without it the Hadamard's zero-leaf `source_graph`
   record projected to `source_path=None` and the weight vanished from the bound feed — on a fresh compile as much
   as on a pack hit. A record the plan cannot reproduce now sets `load_ops=None`, so the pack save refuses loudly.
   `_encode_load_ops` also grew `("slice", spans)` for the affine single-source `IndexMapOp` a folded `SliceOp`
   leaves (3.3's N-padded `svh` residual). Roofline: `weight_bytes` now counts weight INPUTS, so an expert program
   has a real floor instead of zero (still under `MIN_FLOOR_US` at 2.25 bpw — ~4 MB/launch).
   Verified on a config-truncated cut of the pinned 2.25bpw checkpoint (2 layers, 8 experts, vocab 2048 — a full
   46-layer boot does not fit this box): all four `moe.expert.*` tiers plus 8 fixed slots compile; per-tier
   accuracy against a torch eager reference built from the DECODED expert weights is rel 1.0e-3 (M=1), 9.4e-4
   (decode bucket 8), 6.9e-4 (M=256); the indirect fixed-slot T=1 combine matches the routed oracle at rel 6.1e-4;
   the layer's expert store stays compressed (34.9 MB vs 277 MB dequantized, 7.9x). Pack round-trip: cold boot
   127 s → pack-hit boot 22 s with identical accuracy, the trellis plans carrying both `gen` computed constants and
   indirect operands.
   **RESIDUAL — one expert program set per distinct codes shape (Phase 5 BLOCKER).** EXL3's mixed allocation makes
   K vary per layer: for gate/up the pinned checkpoint is K=2 on 40 layers, K=3 on 4, K=4 on 1 (down_proj 38/6/1),
   which the expert `shape_key` now catches as a layer disagreement and raises on. Serving the whole model needs
   the `expert_sym`/`bucket`/`one`/`m256`/`slots` singletons keyed by shape group (each group its own pointer
   tables; the selector and partials stay shared, since one layer runs at a time).
   **RESIDUAL — the serving trunk is not compressed.** `_compile_split` binds trunk constants from
   `wrapper.named_parameters()`, whose paths are wrapper-relative and never reach the checkpoint index, so the
   constant speller cannot fire on a serving trunk. At 2.25 bpw the decoded trunk is ~14 GB against ~28 GB of
   compressed experts — over the 32 GB card. Closing it needs a per-layer wrapper-path → checkpoint-key mapping
   plus shard-sourced binding in `_compile_split`.

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
