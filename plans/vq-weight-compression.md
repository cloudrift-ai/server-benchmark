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
   **SECOND ROUND (2026-08-07), three levers, each measured alone.** NCU on the 3.2 band said the wall was
   instruction issue, not bandwidth (SM 69 % vs DRAM 35 %, 11.47 warp-instructions per 2-bit weight against a
   ~71 µs DRAM floor), and that is what these attack.
   (a) **One LOP3 for the codebook's mask/XOR** (`emmy_trellis_mask_xor`, exact, ungated): written in C, ptxas
   spends two LOP3s because the SASS form encodes at most one immediate; naming the 3-input bit function lets it
   hoist the second constant and issue one. −1 instruction per weight, **−8 % at the past-L2 square** on its own.
   (b) **The split ladder past the powers of two** (`DECODE_BAND_STEPS`, `decode_band_moves(tiles)`): a matvec's
   tile count is the contraction dim over 16 and routinely carries an odd factor, so the wide arm names widths by
   the tile STEPS each CTA keeps. `down` (688 = 16·43 tiles) went from ONE offered row (`g16k`, 2048 CTAs, 19.5 µs
   — a fork with nothing to decide, hence no golden) to three, with `g86k` at 15.3. Wider is not monotone: the
   finalize reads a `cta × N` f32 workspace worth `2/(steps·k_bits)` of the code traffic, so the ladder starts at
   8 steps. (c) **The f16-pair fold** (`F16_REDUCE_F32_ACC`, `060_pair_decode_accum`, `FAST_MATH` family): packed
   run + `__hmul2` products + an fp16 tree over the tile column + ONE f32 promote per tile step. 11.47 → 8.25
   warp-instructions per weight, **−13…18 %**. The cadence is the band's quantum and deliberately not a knob — the
   fp16 tree is 3 deep, so the error is the fp16 product's (5.0–5.2e-4 rel against the f32 lane's 3.6–3.9e-4, FLAT
   in K) where a chain over the whole slice reaches 1.0–1.7e-3 and grows with the slice; longer cadences measure
   within 1 %.
   Greedy end state on the 5090 (goldens live; f16 twin in brackets): past-L2 square **142.2 µs f32 lane /
   122.6 f16-pair** [581.7] — 4.1x / 4.7x, and 852 / 988 GB/s of codes off DRAM; gate/up **14.9 / 12.3** [18.7];
   down **15.3 / 12.9** [18.5], which flips `down` from BEHIND its f16 twin to 1.21x / 1.43x ahead. The
   L2-residency caveat above still governs the two projections — they are relative A/B only.
   Residual: NCU after is SM 65 % / DRAM 46 %, so the two are now balanced and further instruction cuts pay less.
   The measured remainder is OCCUPANCY: 55 registers/thread but 46 % achieved, and registers are not the binder —
   a 32-thread block caps at 24 blocks/SM, which is 24 of the 48 warp slots. Widening the block is NOT reachable by
   a knob today: `WORK=t128` puts the extra warps on the REDUCE axis (a cross-warp k split, which also offsets the
   anchor off a tile boundary so the run stops fusing — 3x slower, measured 474 vs 159 µs). A standalone kernel
   carrying the shipped body over 128-thread blocks that map the extra warps to the FREE axis measures 124.9 µs
   against 129.8 — **~4 %**, and it needs a WORK-mapping change to reach. Not worth chasing at that size. (The same
   standalone reaches 116.4 with a longer-cadence f16 CHAIN instead of the per-step tree, but that is the accuracy
   trade the cadence choice above already rejected.)
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
5. **Both Phase 3.4 residuals CLOSED (2026-08-07)** — the two blockers between here and Phase 5.
   **Expert shape groups.** `from_model` interns each layer's `shape_key` into a group index and compiles that
   group's whole tier set (`_build_expert_group`: sym / bucket / one / m256 / the k fixed slots); the layer's `moe`
   entry carries the index and `_launch_expert` / `_moe_combine_slots` route through it. Group 0 keeps the original
   pack names, so a single-group model is unchanged. `_ensure_device` builds the indirect pointer tables PER GROUP
   (offsets within the group); selector and partials stay shared. The fixed-slot tier is all-or-nothing across
   groups — a whole-step capture records one launch set per layer. Verified on a cut of the pinned checkpoint that
   deliberately mixes K (source layers 3/1/2/6 → K 2,2,2 / 3,3,3 / 4,4,4 / 2,2,3): 4 groups, every tier + 4 slots
   per group, expert accuracy rel ≤ 1.3e-3 at M=1/8/256 and fixed-slot-vs-routed ≤ 7.2e-4 on every group.
   **Compressed serving trunk.** `_compile_split(ckpt=(dir, id_to_key))`: `_retarget_constants` re-addresses each
   traced constant to its checkpoint key by parameter-tensor identity, `spell_trellis_constants(expand=True)` fires
   (forcing the compressed lane per compile via the new `trellis.expand` graph hint, which `032` reads beside the
   env knob), and `_plan_sources` feeds the constants from the shards (`load_sources_by_path`) instead of sweeping
   `named_parameters`. `load_quantized_split(..., compress_trunk=True)` is the matching lane: coded trunk linears
   are never decoded and the twin's parameters are uninitialized placeholders, so `_plan_sources` RAISES on any
   weight the plan cannot source (a linear that fell back to the folded checkpoint-basis cone) rather than running
   weightless. Verified on a 2-layer and a 5-layer cut: every trunk program carries trellis-decode kernels, the
   resident constant bytes are the coded footprint exactly (layer 0 post = 100.7 MB fp16 `o_proj` + 34 MB codes =
   134.6 MB), accuracy vs the decoded eager twin rel ≤ 1.3e-3.
   **Measured footprint of the pinned 2.25bpw checkpoint** (read off the safetensors index, 46 layers):
   experts 25.099 GiB (codes 24.922 + suh/svh 0.177) · trunk codes+siblings 2.493 GiB · trunk fp16-kept 1.296 GiB
   (embed 1.156, the layer-0 fp16 `o_proj` 0.094, routers 0.044, norms/biases) · `lm_head` codes 0.434 GiB.
   emmy-owned resident = **28.888 GiB**; the decoded-trunk lane would have been 37.3 GiB, over the card. Card is
   31.843 GiB, so headroom is ~2.95 GiB before vLLM's `lm_head` and the CUDA context.
   **FULL 46-LAYER BOOT ON THE 5090: SUCCEEDED (2026-08-07).** Cold boot 1123 s (`EMMY_PACK_DIR` set,
   `EMMY_GEN_M1_TIER=0`, `decode_bucket=8`, `max_tokens=256`); 204 plans written. Measured resident: trunk constants
   2.592 GiB + expert store 25.099 GiB + embed 1.156 GiB = **28.847 GiB** (torch 26.300 + cupy pool 2.677), and
   `nvidia-smi` reads **31299 MiB of 32607** at boot — the ~2.3 GiB over the weights is the CUDA context plus the
   allocators' free blocks. Four expert shape groups covering 4 / 1 / 38 / 2 layers, all 46 pre + 46 post programs
   built, `has_moe_fixed_slot` true. A 4-token forward through all 46 layers plus the final norm runs finite. Also
   fixed on the way: `load_quantized_split` held the per-expert dicts alive across the stacking loop, so the store
   cost 2× at peak (~50 GiB for this checkpoint) — it now pops per layer.
   **What the boot says about Phase 5's headroom.** 1.28 GiB free at the runner's own boot, before vLLM's `lm_head`
   and its KV cache. Reclaiming the allocators' free blocks and pinning the arena at the real serving capacity are
   the first levers; the `lm_head` decision below is the second.

**Deliverable**: GLM-4.5-Air resident on the 5090 at ~26.5 GB with correct output. **Verify**: VRAM occupancy
matches the checkpoint size; accuracy unchanged from Phase 2; a decode-phase step streams compressed bytes (confirm
via the roofline audit, not golden benches).

**Note**: these are storage formats, not accuracy knobs — no FAST_MATH gate. The quantization error is in the
checkpoint, and our decode is exact.

## Phase 4 — Golden seeding

**Hardware**: 5090. **Depends on**: Phase 3.

**STATUS: THE DECODE-WIDTH SWEEP IS DONE (2026-08-08) — batch-1 TPOT 104.2 → 57.90 ms, 1.80x. Prefill is
deliberately still owed.** Full write-up in `plans/golden-sweep-glm45air-rtx5090-findings.md`; the short version:

- **What the 17x roofline flag was.** `L0.post.decode.m32` at 18.1x its 64.7 µs floor was not a lowering failure —
  all three coded projections were on the intended warp tier with the intended `sync` compute fill, carrying the
  wrong OUTPUT TILE, and so was every coded projection in the model. A wide register fragment is right for a
  MATERIALIZED B (operand reuse); a DECODED B has no operand to reuse, its cost is the decode's per-element
  instruction count, so the fragment only spends registers and the pressure caps occupancy at 17-25 %. The measured
  answer is the narrowest tile the shape allows with as many CTAs as the output axis gives (`w2x2`/`f1x1`/`k8` on 12
  of 14 shapes, 58-92 % occupancy). Split-K cannot supply CTAs here — it is refused on a computed B — which is why
  the miss is worst on the narrow-N shapes (k/v_proj 4.9x, shared gate/up 4.5x).
- **42 entries seeded** — all fourteen coded trunk (shape, rate) pairs at M = 1 / 8 / 32, 3 reps each, spreads
  ≤ 0.6 %. Per-kernel wins 1.0-4.9x. The rate never changes the winner, only the µs.
- **Verified on the real model**, which is the part that matters: two full 46-layer runner builds, per-launch timed.
  `L1.post.decode.m32` 936 → 406 µs (2.30x), `L1.pre` 585 → 349, `L0.post` 1171 → 682 (18.1x → 10.4x the floor).
  The four `moe.expert.*.bucket.m32` programs improved 2.4-2.6x with **no expert entry of their own** — a routed
  expert's gate/up and down share the shared expert's extents and rates, so those entries key them. That is the
  answer to the "no expert twin" limit: covered by shared-expert entries, verified by step time, not by the audit.
- **The audit is nearly blind here and now says so.** It times ONE layer per attention class above a 20 µs floor, so
  on this model it reports layer 0 alone — the only DENSE layer, which clears the floor only via its uncoded fp16
  `o_proj`. The unreported representative MoE layer was 77x. Noted in `emmy/serving/ARCHITECTURE.md`.
- **Prefill (M=512) and M=64 were swept and WITHHELD.** Both looked clean in isolation (1.03-1.76x / 2.1-4.5x); the
  M=512 set was checked in-model and INVERTED — layer 1's coded `o_proj` 711 → 1182 µs. Both widths' winners cut the
  CTA's M extent below the cold pick's, re-reading the codes slab more times, which is free when a 25 MB slab
  replays out of L2 and is not against a 29 GB trunk. Shipped rule: keep an entry only if its tile reads the codes
  slab no more often than the pick it replaces, or it was verified in-model. **A prefill golden needs an in-model or
  past-L2 harness** — that is the remaining Phase 4 work, and TTFT will not move until it lands.
- **`eval golden --in-model`: MATCH 20 → 40, DRIFT 0, GAP 317 → 297, compile_fail 0.** Every remaining trellis GAP is
  a width this deployment does not run, or a symbolic fork.
- Two boot facts for Phase 5, both forced discoveries: `--max-num-seqs 32` is required (vLLM's default 256 OOMs in
  the post-KV sampler warmup, and its warmup budget alone drops available KV to 0.28 GiB — at 32 the pool is 9,184
  tokens against the recorded 9,392); and `emmy serve <repo-id>` cannot resolve a branch-only EXL3 repo, because
  `quantized_checkpoint_dir` reads `config.json` off the DEFAULT branch and `--revision` reaches vLLM but not the
  runner. Serve the snapshot path until that is fixed.
- One compiler-side fix rode along: the `sync` compute-fill depths were a literal inside `_schedule`, so a recorded
  `d1/sync` was not a member of any catalog the golden permanence gate checks — the same defect the decode band's
  split widths had. They are now `space.sync_stage_moves()`. Emission-neutral (kernel digests identical).

**Still owed by Phase 4**: the prefill/symbolic widths (see above), the ~3.8k-row warp pool beyond the 9 sampled rows
per width, and the model's uncoded f16 forks (layer 0's fp16 `o_proj` alone is 151.7 µs per decode step).

**Enablement, done 2026-08-07.** A trellis golden is now expressible, matchable, benchable and deployable:

- `ShapeKey.from_matmul` takes the `trellis` dtype spelling (`dtype_class="trellis"`, `is_warp` FORCED — the flag is
  the dtype family, not the deployed tier, which is what makes one key serve the M=1 decode band and the prefill
  mma alike). Both constructors agree by construction, pinned by a test that keys the golden, its own snippet, and
  the EXL3-SPELLED in-model graph at M=1 / M=32 / M=256 and asserts all three `joins()`.
- `matmul_snippet`'s trellis arm writes the HAT-BASIS coded linear at the codes grid's 128-padded extents, codes
  minted in a preamble statement (the fp8 trick) so the tracer lifts them as an input. It needs a torch spelling of
  the decode, so `emmy::trellis_decode` is registered as the one non-aten op the tracer maps (onto the very
  `TrellisDecodeOp` the speller builds). Hat basis is correct because Phase 3.3 puts `suh`/`svh` and both Hadamards
  in separate kernels around the contraction. `MatmulGoldenConfig` gained `k_bits` / `cb`; `golden_eval._DTYPES` /
  `_matmul_graph` build the coded enumeration graph.
- Two defects found and fixed on the way. (a) `_fork_shape_key` rebuilt every prefill coded contraction as
  `kind="fused"` AND dropped its `dtype_class` — the sync-STAGE offer signal is no longer unique to a computed-A
  cone now the decode fill spells it. (b) The decode band's split widths lived as module-private constants in
  `_schedule.py`, so a recorded band partition was not a member of any catalog the golden permanence gate checks;
  they are now `space.decode_band_moves`.
- Seeded 2 entries, 3 reps each: `glm45air.mlp_gate_up.m1` (1x11008x4096, k_bits 2) 19.55 → 15.77 µs (1.24x, spread
  1.1 %) and `glm45air.pastl2_22016.m1` (1x22016x22016) 215.3 → 157.9 µs (1.36x, spread 0.24 %). Both deploy: the
  golden tier answers MATCH and the kernel carries `g32k/coop-t/r16`, on the entry's own snippet AND on the
  EXL3-spelled in-model graph.
- **Re-measured and extended after the 3.2 second round (2026-08-07), now 3 entries.** The one-LOP3 fix moved every
  absolute number and the widened ladder moved one winner, so the whole offered ladder was re-swept at each shape,
  3 reps: `mlp_gate_up.m1` keeps `g32k` at 15.0 µs; `pastl2_22016.m1` moves `g32k` → **`g43k`** at 142.4 (g86k ties
  in this lane at 142.44 and the f16-pair lane separates them, 122.4 vs 126.7); and `mlp_down.m1`
  (1x4096x11008) is NEW at **`g86k`** 15.3 µs, the row the widened ladder unlocked (g16k 19.5). All three deploy at
  greedy. The two lanes pick the SAME split at every entry, so one recorded schedule serves both.
**Both enablement gaps CLOSED (2026-08-07), so the real sweep can start:**

- **The key carries the CODE RATE.** `ShapeKey.k_bits`, fed by `MatmulGoldenConfig.k_bits` on the golden side and by
  a new `S_trellis_k_bits` stamp on the op side (written off the body's `TrellisLoad` leaves, over the same load walk
  that writes `S_dtype_i16`, so rate and storage class are always present together). `__post_init__` normalizes the
  rate off every non-trellis class, so no shipped golden's key moves — proved by a kernel-source digest identical
  across the diff and by a test asserting a stray rate on an uncoded key is dropped. Not theoretical: the pinned
  checkpoint's allocation sidecar stores `mlp.shared_experts.gate_proj` at 2 bits x 38 layers, 3 x 6 and 4 x 1 at ONE
  shape, `down_proj` likewise, and q/k/v/o at 4 and 3.
- **The twin builder has a coded arm**, so `eval golden --in-model` audits coded entries. It asks the loader for the
  weight-free allocation (`exl3.coded_tensor_storage`: per module the rate and the `trellis`/`suh`/`svh` shapes, read
  off the small sidecar) and calls the DEPLOYED speller, `_spell_trellis_activation_one` — not a re-implementation,
  which would drift. One twin per distinct rate profile (`post1@b2`, `pre1@b4`, `pre1@b3`, ...), and the `model:` tag
  may pin the rung's branch (`turboderp/GLM-4.5-Air-exl3@2.25bpw`) because the rungs differ in exactly the allocation
  the keys carry. The file is now tagged (the synthetic past-L2 control opts out with a per-entry `model: null`).
  Audit on the pinned rung: **MATCH 19, DRIFT 0, GAP 317, compile_fail 0** — `glm45air.mlp_gate_up.m1` MATCHes in the
  `post1@b2` twin at `k_bits=2`, so the seeded entry genuinely deploys in the model graph, not just on its snippet.

Residuals as of the enablement round — the decode-width half is now CLOSED by the 2026-08-08 sweep above; the prefill
widths and the uncoded forks are not. Two twin-side limits bound what the audit can see, and both still hold. A coded
twin pairs ONE
traced layer's structure with one checkpoint layer's rates, so where a layer codes only part of the twin (GLM stores
layer 0's `o_proj` uncompressed, and its MoE layers have no dense MLP) the rest stay f16 and those forks are artifacts
of the pairing, not shapes the model runs. And there is still no EXPERT twin: the expert weights arrive as program
inputs, so a config-only skeleton contains nothing coded, and `_build_expert_group` takes checkpoint tensors rather
than specs. The path is short now — the sidecar already lists every expert's rate and trellis shape, and its four
distinct expert signatures (4 / 1 / 38 / 2 layers) are exactly the four shape groups `from_model` interns — so what is
owed is a spec-taking seam in `gen_runner`, not new format knowledge.

Seed `emmy/compiler/pipeline/search/goldens/rtx5090_sm120_glm45air.yaml`, following the structure and preamble
conventions of `rtx5090_sm120_olmoe.yaml` — the closest precedent: a MoE, on this card, whose expert
weights arrive as program inputs.

- Seed over the **serving programs themselves**, not synthetic shapes.
- Cover both the routed-expert matmuls and the dense trunk projections, at prefill and decode shapes.
- Reproduce each entry 2–3× and record the spread; document the measurement method in the preamble.
- The preamble must carry the L2-residency warning — with a 26.5 GB trunk, absolute microseconds from
  `run --bench --golden` do not predict in-model step time. The 2026-08-08 round proved this is not a caveat for the
  reader but a limit on the METHOD: at prefill width the isolated ranking inverted in the model, so a wide-M entry
  needs an in-model or past-L2 measurement before it may be recorded.

**Verify**: `emmy eval golden --in-model` on the deployed model; the coverage gate that Phase 5 enforces must pass.

## Phase 5 — Prebuilt serving image

**Hardware**: 5090 (the release pipeline runs on the target GPU). **Depends on**: Phase 4.

**STATUS: THE POOL IS NO LONGER THE LIMIT; THE DECODE TIER IS (2026-08-08).** The vocab-table reclaim below took
the KV pool from **864 to 9,392–14,016** fp8 tokens depending on how the rest of the budget is spent, and
`--max-model-len` from 512 to **4096**. The concurrency sweep that pool unlocked then found the *next* wall, and it
is not memory: see "The concurrency measurement" below.

**STATUS (2026-08-07): IT SERVES, AND THE CARD IS FULL.** `emmy serve --generate` boots GLM-4.5-Air 2.25bpw on one
5090, answers chat requests coherently, and has **864 tokens of KV cache**. Three boot blockers were closed getting
there; the fourth finding is that there is no room left, which is what Phase 5 must now solve before it can bake
anything.

1. **`lm_head` had no source, silently.** An EXL3 checkpoint carries no `lm_head.weight` — the head is coded like
   any other linear (K=6, 0.434 GiB coded / 1.156 GiB decoded) — so `EmmyGenModel.load_weights` matched nothing.
   vLLM's own strict weight-tracking check waives any parameter whose quant method defines
   `process_weights_after_loading`, which `ParallelLMHead`'s does, so the server would have answered with noise
   while looking healthy. `load_weights` now decodes the head straight from the checkpoint
   (`decode_exl3_blocks` — out-feature blocks, because the float64 fold is ~5 GiB whole; **26 s** measured on the
   real head) and RAISES when no source applies. Reading the checkpoint directly also skips a second full pass
   over the shards, which here is ~29 GiB of expert codes the model owns none of.
   **DECIDED: decode it, and the decision was forced, not preferred.** vLLM's `ParallelLMHead` allocates the fp16
   head in `__init__` regardless of who fills it, so serving the head coded is not a load-time tweak — it means
   not constructing `ParallelLMHead` at all and replacing vLLM's logits path with an emmy coded matmul at
   4096x151552, K=6: a new program family with its own dynamic-M tiers and goldens. The measured price of decoding
   is **0.722 GiB**, which at this point is most of the remaining budget (see the wall below) — so the coded head
   is now the single highest-value item on the Phase 5 list, on measurement rather than taste.
2. **vLLM refuses an EXL3 config outright** ("Unknown quantization method: exl3") — nothing to do with weights, it
   is `_verify_quantization` rejecting the scheme name. The checkpoint is presented as unquantized through
   `--hf-overrides`, which for the engine's purposes it is (`loader/quant.py::engine_config_overrides`; the
   command layer must not name a checkpoint format — the frontend-band guard enforces that).
3. **Headroom.** `load_weights` now reclaims both allocators' free blocks unconditionally (the tied-embed path was
   the only one that did) and logs it in the driver's units; and `EMMY_GEN_PREFILL_CAPACITY` pins the symbolic
   programs' activation arena at the width a deployment actually serves instead of at the dynamic-dim cap, with
   `emmy serve`'s `--max-num-batched-tokens` default following it down. Measured on this model: the reclaim
   returns **0.125 GiB**, and at capacity 512 the whole non-weight overhead (context + arena + allocators) is
   **1.10 GiB** against 1.72 GiB at the Phase 3.4 runner-only boot.

**The wall, measured.** Card total as the driver reports it is 31.324 GiB, of which the CUDA context takes 0.67
before anything loads. After the runner and the decoded head: **0.184 GiB free**. That is the whole budget for
vLLM's attention workspace, its profiling peak, the CUDA-graph pool and the KV cache. Consequences, all measured:

- At the shipped default (`--gpu-memory-utilization 0.97`, FlashInfer) available KV is **−0.21 GiB** — the boot
  fails. vLLM's own default FlashInfer workspace is a permanent **394 MiB** allocation and OOMs on its own.
- Whole-step decode capture does not fit either; the boot that works is `--enforce-eager`.
- The configuration that boots: `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=16777216`, `--enforce-eager`,
  `--kv-cache-dtype fp8_e4m3`, `--max-model-len 512`, `--max-num-batched-tokens 512`, util **0.9755**,
  `EMMY_GEN_PREFILL_CAPACITY=512`, `EMMY_GEN_DECODE_BUCKET=8`, `EMMY_GEN_M1_TIER=0` → **864 KV tokens**, maximum
  concurrency 1.69x. Utilization is bounded above at 0.978 by `free_at_init`, and 0.9782 allocates 1,824 tokens
  and then OOMs 4 MiB later — the usable band is a few tenths of a percent wide.
- **fp8 KV is load-bearing, exactly 2x**: the identical config with fp16 KV yields 432 tokens and vLLM refuses the
  boot ("estimated maximum model length is 432"). Same result as the `3a79ac6e` smoke-model measurement, now on
  the real model.

**First end-to-end serving numbers** (batch 1, ~20-token prompt, 64 output tokens, three runs): TTFT **1128–1185
ms**, TPOT **102.5 ms** (spread 0.04 %). Generation is coherent and correct on every chat probe (it is a thinking
model, so the visible output is `<think>` reasoning). Concurrency 2 gives no throughput gain — 7.84 tok/s against
9.76 at c=1 — which is the 864-token pool, not the kernels. Boot time: cold **1067 s** (204 program plans written),
**pack hit 104 s** (85 s of it the checkpoint read), a 10.3x saving. For context the Phase 0 exllamav3 baseline on
the *2.00* rung was TTFT 499 ms / TPOT 11.9 ms, so this first configuration is ~2.3x behind on TTFT and ~8.6x on
TPOT — with the whole of Phase 4's golden sweep still owed (the boot's own roofline audit flags
`L0.post.decode.m8` at 17x its weight-streaming floor), CUDA graphs off, and the M=1 decode tier off.

### The vocab reclaim, measured (2026-08-08)

Two vocab-sized fp16 tables were the whole remaining budget. **The embed table was taken; the coded `lm_head` was
not**, and the reason is bytes per unit of risk rather than taste:

| item | bytes | route | verdict |
| --- | --- | --- | --- |
| embed table, 151552 x 4096 fp16 | **1.156 GiB** | mapped host memory, device-addressed gather | **TAKEN** |
| decoded `lm_head`, same shape | 0.722 GiB net (1.156 fp16 - 0.434 coded) | replace vLLM's logits path with a coded matmul | deferred |

The embed table is 60 % more bytes for a fraction of the work: `EMMY_GEN_EMBED_HOST=1` allocates it with
`cudaHostAlloc`-mapped memory, whose address is a valid *device* address under unified virtual addressing, so
`embed_device` stays a plain device-side gather and whole-step capture still records it — no host round trip, which
was the stated constraint. The coded head, by contrast, means not constructing `ParallelLMHead` at all and
compiling a new 4096x151552 K=6 program family with its own dynamic-M tiers and goldens; there was no route to its
bytes short of that (the head is read whole every step, so parking *it* in host memory costs ~46 ms/step over this
box's PCIe link and ~17 ms even coded).

**Measured on the 5090.** Free after the runner and the head goes **0.184 -> 1.338 GiB**, i.e. the reclaim returns
**1.154 GiB** of the table's 1.156 — all of it. The gather's price is PCIe latency per token, and this box's card
sits on a **gen5 x1** link (3.6 GB/s measured, both directions): 22.6 us for an 8-token decode step and 1.25 ms for
a 512-token prefill chunk, against 3.4 us flat for a device-resident gather. On this model that is 0.02 % of TPOT
and 0.1 % of TTFT, and c=1 TPOT is unchanged (102.86 ms mapped vs 102.5 ms resident). A x16 link would divide the
cost again — the numbers below are pessimistic on that axis, not optimistic.

**Where the reclaimed budget goes, all measured, eager unless stated:**

| config | KV tokens | note |
| --- | --- | --- |
| before, util 0.9755, mml 512, 16 MiB workspace | 864 | the 2026-08-07 wall |
| after, same knobs | **14,016** | 16.2x; allocates, then OOMs in the post-KV warmup — a headroom reading, not a config |
| util 0.970, mml **4096**, 16 MiB workspace | 11,488 | serves c=1; **dies at c>=4** — FlashInfer asks 144 MiB for `batch_prefill_tmp_v` |
| util 0.970, mml 4096, 64 MiB, whole-step capture | 9,488 | capture costs 2,000 tokens for **2.8 %** TPOT (100.0 vs 102.9) |
| util 0.9641, mml 4096, **394 MiB** (vLLM default) | 9,392 | allocated, then OOMs claiming the workspace (398 MiB free) |
| **util 0.9641, mml 4096, 256 MiB workspace** | **9,392** | **the config that serves the whole grid** |

Three findings worth carrying: the utilization band is no longer a few tenths of a percent wide; the FlashInfer
workspace is allocated **lazily on first attention plan**, so it is invisible to vLLM's KV sizing and has to be
budgeted by hand (16 MiB boots and then kills the engine at c>=4 — a latent failure, not a config); and whole-step
decode capture is now affordable but **not worth buying** at a capture ladder of `[1]`, because MoE capture covers
single-token steps only and every c>1 decode step runs eager regardless.

### The concurrency measurement — the pool stopped being the limit, the expert launch chain is

The pool now admits 14 concurrent 512+128-token streams where it admitted one, so the sweep the plan's thesis
rests on finally runs. **It does not show the win.** Throughput is close to flat in concurrency, and the reason is
measured, not guessed.

| c | N | completed | dur s | req/s | out tok/s | TTFT med s | TTFT p99 s | TPOT med ms | ITL med ms | W mean | peak VRAM MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 8/8 | 122 | 0.066 | **8.4** | 1.96 | 3.3 | 104 | 105 | 209 | 32037 |
| 4 | 24 | 24/24 | 337 | 0.071 | **9.1** | 9.22 | 12.8 | 373 | 349 | 131 | 32037 |
| 8 | 48 | 48/48 | 603 | 0.080 | **10.2** | 12.08 | 26.0 | 698 | 609 | 128 | 32037 |
| 16 | 64 | 64/64 | 751 | 0.085 | **10.9** | 15.79 | 52.1 | 1292 | 1076 | 126 | 32037 |

*Protocol*: `scripts/bench_serve_sweep.py` against a live `emmy serve --generate`, the Phase 0 workload grid
exactly (512 in / 128 out, `--ignore-eos`, N = 8/24/48/64), one discarded warmup and **one** recorded run per point
rather than the recipe's three — a time reduction, so there is no spread column and the table owes its repeats at
image-bake time. Every point completed all its requests (the driver refuses to report one that did not). Two other
deviations from `experiments/GLM-4.5-Air-EXL3/serving_rtx5090/recipe.yaml`: prefix caching was left at vLLM's
default ON rather than `--no-enable-prefix-caching` (observed hit rate 0-9 %), and the run is a bare `emmy serve`,
not the prebuilt image, which does not exist yet.

**Why, quantitatively.** GLM-4.5-Air routes `top_k=8` of `E=128` experts per token across 45 MoE layers, and
`combine_routed_experts` issues **one launch per DISTINCT expert hit in the step**.
That is correct and already amortized per expert — each expert's weights are read once for all its rows — but the
launch COUNT scales with distinct experts, and distinct experts saturate at `E` long before tokens do. Predict the
count as a coupon-collector expectation, `E x (1 - (1 - 1/E)^(c x k))` per layer, and compare it against measured
TPOT:

| c | predicted distinct experts/layer | launches/step | predicted step cost vs c=1 | measured TPOT vs c=1 |
| --- | --- | --- | --- | --- |
| 1 | 8.0 | 360 | 1.00x | 1.00x (104.3 ms) |
| 4 | 28.3 | 1,274 | 3.54x | **3.58x** (373.4 ms) |
| 8 | 50.5 | 2,273 | 6.31x | **6.69x** (697.8 ms) |
| 16 | 81.0 | 3,645 | 10.1x | **12.4x** (1292 ms) |

Step cost tracks the launch count, not the token count: the prediction is within **1 %** at c=4 and **6 %** at c=8.
At c=16 it undershoots by 23 %, which is where the pool itself finally shows up — occupancy sits at 96-98 % with
15 running and 1 waiting, so queueing and preemption add cost the launch model does not carry (and the router is
not uniform, so 81 distinct experts is a floor). That is the signature of the per-expert launch chain rather than
of bandwidth, and `plans/moe-m2-dispatch-design.md` measured that chain from the other end — ~117 us of Python
framing per launch, so 360 x 117 us = 42 ms of pure framing inside a 104 ms c=1 TPOT.

So continuous batching does exactly what it should — it admits the streams — and then every additional stream costs
nearly a full step's work because the dispatch is per-expert-launch bound. **The concurrency argument cannot be
made on this model until MoE M2 (fused grouped-GEMM dispatch, or captured launch-set recovery) lands.** That is the
honest state, and it is a dispatch-architecture item, not a memory or a golden one.

**Against the contender.** Phase 0's exllamav3/tabbyAPI numbers are on the **2.00** rung and must be re-measured on
2.25 before any table quotes them (Phase 6 item 3), so treat the ratios as indicative:

| c | exllamav3 2.00 out tok/s | emmy 2.25 out tok/s | exllamav3 TTFT | emmy TTFT | exllamav3 TPOT | emmy TPOT |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 62.7 | 8.4 | 0.50 s | 1.96 s | 11.9 ms | 104 ms |
| 4 | ~102 | 9.1 | 1.32 s | 9.22 s | 27.9 ms | 373 ms |
| 8 | ~99 | 10.2 | 5.99 s | 12.08 s | 29.2 ms | 698 ms |
| 16 | ~99 | 10.9 | 14.7 s | 15.79 s | 29.9 ms | 1292 ms |

Worth reading carefully rather than as a single ratio. exllamav3 **also** flattens, saturating near 100 tok/s from
c=4 and converting further concurrency into queue time (its effective parallelism is ~4-5 on that cache), and by
c=16 emmy's TTFT is within 7 % of it — both are queue-dominated there. So the opening the plan predicted is still
there in principle: the contender does not scale either. It is emmy that cannot take it yet, because emmy flattens
an order of magnitude lower and for a different reason — launch chain, not admission.

**What was NOT the cause, each ruled out by measurement:**

- *The KV pool* — it was, at 864 tokens; it is not now. At c=16 the pool sits at 96-98 % occupancy with 15 running
  and 1 waiting, i.e. admission is saturated and still no throughput.
- *The decode-tier lockout.* At `EMMY_GEN_DECODE_BUCKET=8` every step with 9-32 running sequences falls off the
  static twin onto the capacity-512 symbolic program, which is exactly the ladder trap
  `emmy/serving/ARCHITECTURE.md` warns reclaiming footprint re-opens. Rebuilding at **bucket 32** (a 1079 s cold
  build; same 9,392 KV tokens, so it is free in memory) closes that hole and buys **6 %**: c=16 TPOT 1271 -> 1192
  ms, while c=1 gives back 1.3 % (102.9 -> 104.2, the T=1 step now padding to 32 rows). Real, and an order of
  magnitude too small to be the story.
- *CUDA graphs.* 2.8 % of TPOT at c=1 and nothing above it (MoE capture is limited to capture size 1), for 2,000
  KV tokens. Declined.

**What Phase 5 must still do before it bakes an image**, ranked by measured value:

1. **MoE M2 — the per-expert launch chain.** This is now the whole story, at both concurrency and batch 1: 360
   launches per step at c=1 and ~3,645 at c=16, and the step cost tracks that count. Design already exists
   (`plans/moe-m2-dispatch-design.md`). Nothing else on this list changes the shape of the sweep table.
2. **Phase 4's golden sweep.** Owed in full, and independent of (1): the boot's own roofline audit still flags
   `L0.post.decode.m32` at **17x** its weight-streaming floor (~67 us), unchanged from the bucket-8 boot. This is
   the batch-1 TPOT lever.
3. ~~Reclaim vocab-table bytes~~ — the embed table is done (above). The **coded `lm_head` is still owed**: 0.722 GiB,
   about 8,000 more fp8 tokens, and it is now the only vocab-sized item left. Its value changed character, though:
   with the pool no longer binding, those bytes buy headroom (a bigger workspace, a longer context, whole-step
   capture) rather than the difference between serving and not.
4. **TTFT.** 2.8-3.0 s for a 512-token prompt — 5.6 ms/token of prefill — against exllamav3's 499 ms at c=1 on the
   2.00 rung. Prefill runs the symbolic programs at `EMMY_GEN_PREFILL_CAPACITY=512`; raising capacity is a pack-key
   change (cold rebuild) and should be swept once (2) has moved the kernels.
5. **Pin `models/<slug>.env`** — **not yet**, deliberately: the config seals the cache key, and (1), (2) and (4) all
   still move it. The recipe/experiment TODO(Phase 5) placeholders stay until then.

Two config facts the pin will need, both measured above and neither derivable: the attention workspace must be
**256 MiB** (16 MiB boots and then kills the engine at c>=4; the 394 MiB default does not fit) and utilization
**0.9641** with `--max-model-len 4096`.

Only then run the documented release workflow in `docker/vllm-emmy-serve/ARCHITECTURE.md` end to end — it is the
authority; do not improvise.
`make serve-config → serve-goldens → serve-warm → serve-image → serve-verify → serve-push`.

**Two image-pipeline fields the config keys below do not yet cover**, both cache-key relevant (so Makefile,
Dockerfile, `warm.sh` and `verify.sh` move together): the activation-arena width
(`EMMY_GEN_PREFILL_CAPACITY`) and the attention workspace size, which on this model is the difference between
booting and not. The snapshot also ships as four `COPY warm/hf_parts/pN` layers sized for a ~24 GiB model; at
29.3 GiB that split needs revisiting.

### GATE — the `lm_head` decision: RESOLVED, and it did not close the budget

`lm_head` is vLLM-owned and has no `.weight` in an EXL3 checkpoint (only `lm_head.trellis`), so
`EmmyGenModel.load_weights` found nothing to load. **It now decodes the head at load** — see the status block
above for why that was forced rather than chosen: vLLM's `ParallelLMHead` allocates the fp16 head in `__init__`
whoever fills it, so a coded head means replacing the engine's logits path, not changing a loader.

**The gate's own arithmetic held, and then some.** A decoded fp16 head does leave no usable KV pool: measured, the
card yields **0.184 GiB free** after the runner and the head, against a 394 MiB FlashInfer workspace and a
profiling peak that both come out of it. The booting configuration gets **864 fp8 KV tokens** by shrinking the
workspace, going eager and sitting at util 0.9755 in a band a few tenths of a percent wide. So the coded head is
still owed — it is worth 0.722 GiB, about 8,000 fp8 KV tokens.

**Update 2026-08-08.** The head's sibling candidate — the untied embed table, 1.156 GiB — was settled instead, and
it settled the *pool*: see "The vocab reclaim, measured" above. The head is therefore no longer the difference
between serving and not serving; it is 0.722 GiB of headroom to spend on a bigger pool or a bigger workspace once
the decode tier is worth feeding. Its price has not changed and there is still no cheaper route to it than the
coded matmul: the head is read whole every step, so it cannot follow the embed table into host memory (~46 ms/step
decoded over this box's link, ~17 ms even coded, against a 102 ms TPOT).

### Prerequisites for the config

The three release-pipeline gaps that made "run the documented workflow" untrue for this model are CLOSED — the
support is in `serve.sh` / `warm.sh` / `verify.sh`, so what is owed here is setting the keys, not fixing scripts:

- `SERVE_REVISION=6a309ed6d606fc0154e6e1aeb0912cd3c25534fe`. The slug does not encode the rung, and an unpinned id
  resolves to the repo default — the 2.00 rung, which FAILED the Phase 0 quality gate. `warm.sh` now refuses an
  unpinned revision on a multi-branch repo, so this cannot be forgotten silently; pin the sha, not `2.25bpw`.
- `SERVE_QUANT=exl3` — adds `"quantization_config": null` to the `--hf-overrides`. Without it vLLM rejects the
  config outright (`Unknown quantization method: exl3`) and the warm dies before compiling anything.
- `SERVE_CAPTURE_SIZES=[1]` — the MoE capture ladder. The power-of-two default either fails capture or wastes warm
  time, and the runner's own boot guard rejects it.
- `SERVE_EXTRA_ARGS="--kv-cache-dtype fp8_e4m3"` — quantized KV is mandatory scope (Phase 0 outcome), and it moves
  which programs the plugin builds (`_attn_aliased` bails when KV scales are active), so it must be pinned in the
  config and warmed with, not passed at deploy time.

### Then the workflow

- Create `models/<slug>.env` with the pinned serving config, the four keys above included. **The config seals the
  cache key and cannot change after warming.**
- Memory headroom is the delicate step here, and the status block above has the measured budget rather than the
  arithmetic: on the 31.324 GiB the driver reports, the CUDA context takes 0.67 before anything loads and the
  runner plus the decoded head leave **0.184 GiB** for vLLM's attention workspace, its profiling peak, the
  CUDA-graph pool and the KV cache. KV costs 0.180 MiB/token at fp16 and 0.090 at `fp8_e4m3`, and fp8 is what
  makes the pool exist at all (measured: 864 tokens vs 432, and vLLM refuses the fp16 boot). Sweep and pin the
  largest passing `max_num_batched_tokens` / bucket configuration — but expect the sweep to be worth re-running
  once the vocab tables shrink, because today the band that boots is a few tenths of a percent of utilization
  wide.
- Clear every gate: golden coverage, HF-parity validation, offline zero-recompile verify.
- **Pause for human approval before `serve-push`.**

**Deliverable**: `cloudriftai/vllm-emmy-<slug>:TAG` serving with no `HF_TOKEN` and no download.

## Phase 6 — Recipes, experiments, benchmarks

**Hardware**: 5090. **Depends on**: Phase 5.

1. `recipes/GLM-4.5-Air-EXL3/recipe.yaml` — the one recommended serving config, what `emmy deploy` runs.
2. `experiments/GLM-4.5-Air-EXL3/serving_rtx5090/recipe.yaml` — the benchmark grid: emmy vs exllamav3/tabbyAPI on the
   identical checkpoint, matrices over `benchmark.max_concurrency` {1,4,8,16} and `benchmark.random_input_len`.
   Add a llama.cpp lane at matched bpw, and a stock-vLLM lane that is expected to fail to load (record the failure —
   it is a result). That is THREE recipe files, not one: `engine.llm` and `command` are mutually exclusive in a
   recipe (`_validate_and_build` raises if both are set), so the vLLM lanes (emmy + stock) are one inference recipe
   while exllamav3/tabbyAPI and llama.cpp are `command` recipes of their own.
3. Reuse the Phase 0 client invocation and workload grid verbatim — the PROTOCOL is what carries over. The recorded
   Phase 0 numbers do NOT: they were measured on the 2.00 rung, so they are a 2.00-vs-2.25 reference point, and the
   contender's baseline must be re-measured on 2.25 before any table quotes it.
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
