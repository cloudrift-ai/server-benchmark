# Codebook (VQ) weight compression with in-kernel decompression

Goal: store linear-layer weights at ~2–3 bits/weight using a codebook-class quantization format and decode them
inside emmy's kernels, so 50–105B models fit and serve on a single RTX 5090 (32 GB) / RTX 4090 (24 GB). This is the
sub-INT4 sibling of `plans/fp8-support.md` — it shares that plan's Phase 0 and type-system layers, and forks at the
kernel (a codebook decode is a gather that never commutes out of the contraction, unlike the fp8 scale multiply).

Research basis (Aug 2026): AQLM paper/repo, PV-tuning, QuIP#, QTIP, EXL3 (exllamav3), VPTQ, GPTVQ, CALDERA, HQQ,
GSQ, and the 2026 Bielik six-method comparison. Sources cited inline.

## 1. Format survey — what the field says

| Format | true bpw @"2-bit" | W2 PPL, Llama-2-70B | decode mechanism | decode vs FP16 gemv | produce 70B |
| --- | --- | --- | --- | --- | --- |
| AQLM 1x16 (+PV) | 2.07 (2.29 @7B) | 3.94 / 3.78 PV | 1 MiB LUT, gathered from L2 | 1.2–1.5× | ~720 A100-h |
| AQLM 2x8 | ~2.02 | 4.21 | 2×8 KiB LUT in smem | ~3× | ~720 A100-h |
| QuIP# E8P | 2.02 | 3.91 | act. Hadamard + ~1 KiB LUT + sign flips | ~3.2× | ~100 A100-h |
| QTIP (HYB) | 2.00 | **3.70** | trellis walk, 2–4 ALU instr/wt, ≤2 KiB LUT | ~3.4×, >80% peak BW | ~tens of A100-h |
| EXL3 | 1.6–8 mixed | QTIP-class | QTIP-style, Marlin-inspired kernel | memory-bound | **hours on one 4090** |
| VPTQ | ~3.6 real | 3.93 | centroid LUT gather | 1.6–1.8× AQLM | ~76 A100-h |
| HQQ 2-bit | ~2.5 | weak w/o LoRA | scalar dequant | trivial | minutes |

Key findings:

- **QTIP/EXL3 supersedes AQLM** on every axis that matters here: best accuracy-per-bit at 2–3 bpw (beats AQLM+PV on
  every published table, incl. Llama-3 where AQLM degrades hardest), decode cheap enough to fuse into a memory-bound
  gemv (a 256-weight 16×16 tile is one tail-biting trellis walk — parallel across tiles, and the tile shape matches
  the mma tile), and production from FP16 is hours, not weeks. Even AQLM's own lab (ISTA-DASLab) has moved on — no
  new AQLM checkpoints since Llama-3.2/Qwen2; the line is dormant. (arxiv.org/pdf/2406.11235,
  github.com/turboderp-org/exllamav3, arxiv.org/pdf/2603.04162)
- **AQLM 1x16's decode is the cautionary tale**: the 1 MiB per-matrix codebook cannot live in smem, so every group
  decode is a random gather from L2 (`ld.cg.global.v4.u32` in the official kernel) — it wastes most of the 8× byte
  reduction (81.5 tok/s vs QTIP's 188 on the same card at 7B). Any format we pick must have its LUT/state fit in
  smem or registers. (github.com/Vahe1994/AQLM inference_lib cuda_kernel.cu)
- **QuIP# E8P is the simpler runner-up**: static ~1 KiB codebook + sign flips, no sequential trellis state, modestly
  worse PPL. Same activation-side Hadamard-transform requirement as QTIP.
- Rotation-based scalar methods (SpinQuant-class) preserved likelihood but failed at autoregressive generation in
  the 2026 Bielik study — avoid. GSQ (learned scalar group quant, standard INT kernels) is the low-complexity dark
  horse at 3 bpw if we ever want to skip LUT decode entirely.

**Decision: target the trellis-coded family (QTIP-style, EXL3-compatible) as primary; QuIP# E8P as the simpler
fallback; AQLM as an optional ingest-only format for the existing ISTA-DASLab PV checkpoints.** A V0 spike
(§6) validates this before committing kernel work.

## 2. Compression arithmetic — what we actually save

Effective bpw includes codes + per-channel scales + codebooks. For 2-bit codes at 70B scale, effective ≈ 2.0–2.1
bpw (overhead shrinks with hidden size; AQLM 1x16 is 2.29 @7B because of the 1 MiB/matrix codebook — QTIP/QuIP#
have no such penalty).

| baseline we'd otherwise deploy | bytes/weight | ~2.1 bpw VQ ratio | ~3.1 bpw VQ ratio |
| --- | --- | --- | --- |
| FP16/BF16 checkpoint | 2.0 | **~7.6×** | ~5.2× |
| FP8 (W8, `plans/fp8-support.md` M2) | 1.0 | **~3.8×** | ~2.6× |
| INT4 (AWQ/GPTQ-class, ~4.25–4.5 bpw real) | ~0.55 | **~2.0×** | ~1.4× |

Note: one does not VQ-compress already-INT4 weights — quantization always starts from the highest-precision
checkpoint (bf16, or dequantized fp8 for fp8-native releases like DeepSeek). The table reads "how much smaller than
the format we would otherwise ship". The decode-phase TPOT ceiling scales the same way: batch-1 decode is
weight-bytes-bound, so 2 bpw gives an ~8× ceiling over FP16 and ~2× over INT4 *if decode is ALU-free enough* —
QTIP demonstrates >80% of peak memory bandwidth, i.e. the ceiling is reachable.

Accuracy budget: at ~2 bpw the best methods hold Llama-2-70B to +0.2–0.4 PPL over FP16 and stay Pareto-optimal
(a 2-bit 70B beats any 4-bit ~34B). 3-bit is near-lossless (QTIP 3.26 vs 3.12 FP16). The community already accepts
far worse (Unsloth UD-IQ2 GGUFs are heavily downloaded); a QTIP-class 2-bit is a strict quality upgrade over what
people run today.

## 3. What it unlocks on our cards

The unlock band is 50–105B: models that do NOT fit at INT4 but DO fit at ~2.1 bpw (weights ≤ ~21 GB usable on
24 GB, ≤ ~29 GB on 32 GB, rest for KV/activations). The 2026 demand has shifted from dense 70B to mid-size MoE.

RTX 5090 / 32 GB shortlist (2.1-bpw weight bytes):

1. **GLM-4.5-Air 106B-A12B — ~28 GB, marginal but THE most-requested local model** (the top GLM-5 HF discussion is
   titled "We need Air and we need Flash"; the community expert-pruned it to 82B just to fit consumer cards)
2. Qwen3-Coder-Next 80B-A3B — ~21 GB (hottest 2026 local-coding target; ~8 GB spare for KV)
3. Qwen3-Next-80B-A3B Instruct/Thinking — ~21 GB
4. GLM-4.5-Air-REAP-82B-A12B (pruned) — ~22 GB, the comfortable Air compromise
5. Llama-3.3-70B / DeepSeek-R1-Distill-70B — ~18.5 GB (~30K tokens of FP16 KV headroom)
6. Hunyuan-A13B 80B, Qwen2.5-72B — ~21 / ~19 GB

RTX 4090 / 24 GB shortlist: Llama-3.3-70B / R1-Distill-70B (~18.5 GB), Nemotron-Super-49B (~13 GB, comfy),
Qwen2.5-72B (~19 GB), and the 80B-A3B MoEs at ~21 GB (very tight — needs KV quantization or short context).

Near-misses worth tracking on 32 GB: Qwen3.5-122B-A10B (~32 GB) and Mistral Small 4 119B (~31 GB) — the two models
with genuine buzz that a ~1.9-bpw effective rate (EXL3 mixed-bpw allocation) could newly enable.

Anything ≥230B total (GLM-5, DeepSeek V4, MiniMax M2.5, Kimi K3, Qwen3.5-397B) stays out of reach at any bpw on one
card — 2-bit does not change that class.

### Engine landscape — why this is relevant at all

Verified Aug 2026: **no mainstream serving engine runs the 2-bpw trellis formats.** vLLM closed the EXL3 feature
request as not-planned (vllm-project/vllm#19896; its low-bit menu stops at INT4/FP8); llama.cpp cannot read EXL3
files (its own QTIP-inspired `IQx_KT` GGUF types live mainly in the ik_llama.cpp fork; mainline sub-3-bit is the
older E8-lattice IQ2 family). EXL3 checkpoints run only on exllamav3 itself (served via tabbyAPI), which is
single-user-focused — the vLLM request's own words: "the native EXLlama engine struggles with large-scale serving."
So the best 2-bit formats live in engines with weak serving, and the engines with strong serving don't run the
formats. emmy fronting vLLM's API with in-kernel trellis decode would be the first stack combining the two — the
relevance bar is therefore **outperform exllamav3** on its own checkpoints, not stock vLLM (which is N/A on these
models/cards).

### Target models (top 3, demand-ranked) and baselines

Selection = community demand first (per the user), then EXL3 checkpoint availability (verified on HF) and emmy
feasibility. Weight-file sizes are `total_params × bpw / 8` — HF's displayed "NB params" on quant repos is a
byte-derived pseudo-count (bytes/2), not GB; don't read it as either.

1. **Qwen3.6-27B / 35B-A3B, forward-compatible to Qwen3.8** — the community's daily driver, and the family
   `plans/fp8-support.md` already targets for Qwen3.8. Not a fit-unlock on 24/32 GB (INT4 fits) — the VQ story is
   **decode TPOT** (~2× fewer weight bytes than INT4 = up to ~2× the decode ceiling) plus KV/context headroom
   (27B @2.06 bpw ≈ 7 GB; 35B @2.08 ≈ 9 GB). The 27B is dense — no MoE work, so it doubles as the
   kernel-proving first target. Full EXL3 bpw ladders exist (verified), incl. DFlash draft models.
2. **GLM-4.5-Air 106B-A12B** — the loudest sustained demand signal (the top GLM-5 HF discussion: "We need Air and
   we need Flash"). turboderp's own EXL3 repo ships a 2.00–4.00 bpw ladder. Sizes: 2.00 bpw ≈ 26.5 GB — the only
   on-card 5090 fit, tight (~2.5 GB for KV); 2.25 ≈ 29.8 GB does NOT leave room; 3 bpw ≈ 40 GB is out. The
   REAP-pruned 82B (EXL3 exists) is the comfortable fallback: 2.25 ≈ 23 GB, 3.0 ≈ 31 GB. Prerequisite: MoE
   routing in emmy (experts are plain linears — routing is the new part).
3. **Laguna S 2.1 118B-A8B** (Poolside, July 21 2026) — the freshest coding-agent hype: open-weight MoE, 1M
   context, beats DeepSeek V4 Pro Max on Terminal-Bench at 1/14th the size; released day-one under OpenMDW.
   turboderp published `Laguna-S-2.1-exl3` (+ XS and a DFlash draft) within days — the format author's own
   priority signal. The catch: 2.0 bpw ≈ 29.5 GB — over the 5090 budget; it needs ~1.8–1.9 bpw effective (≈27 GB)
   via EXL3's mixed allocation, at the quality frontier. Treat as the aggressive flagship; re-check quality at
   that bpw in V0. (Qwen3-Coder-Next 80B-A3B is the safer alternate in the same niche — ~21 GB @2.1 — if Laguna's
   sub-2-bit quality disappoints; Laguna appears to have displaced it as the coding flagship since July.)

Dense Llama-3.3-70B / R1-Distill-70B dropped from the target list (demand is legacy); the Qwen3.6-27B dense trunk
takes over the no-MoE first-target role.

Baselines, per model, same card (4090 and 5090):

- **Primary: exllamav3/tabbyAPI running the IDENTICAL EXL3 checkpoint** — apples-to-apples on weights and quality;
  the fight is pure kernels + serving. Batch-1 TPOT/TTFT first (their home turf), then a concurrency sweep
  (c = 1/4/8/16 req/s + goodput), where continuous batching should win by construction.
- **Secondary: llama.cpp with the popular GGUF at matched bpw** (Unsloth UD-IQ2 class, or ik_llama.cpp `IQ2_KT`
  for a trellis-vs-trellis comparison) — what most of the community actually runs today.
- **Quality gate, not a perf baseline**: spot-check PPL/KL vs the bf16 reference matches exllamav3 on the same
  checkpoint (decode is bit-exact reconstruction, so any gap is a bug, not a tradeoff).
- Stock vLLM appears only as "cannot run this model on this card" — that absence is the headline, and it also
  means an emmy win here is not reproducible by config-tweaking the incumbents.

MoE synergy: the unlock band is mostly MoE with 3–12B active. Expert weights are the bulk of the bytes and each
expert is a plain linear — exactly the shape our matmul pipeline handles — and per-token only the routed experts'
tiles are read, so the bytes-bound argument holds per expert. (MoE routing itself is a separate emmy feature gap —
see Open questions.)

## 4. Working around the extra decode-phase compute

Layered answer, cheapest first:

1. **Pick a format whose decode is ALU-cheap.** This is most of the answer. Trellis decode is 2–4 integer/half
   instructions per weight with a ≤2 KiB LUT (fits smem even 32×-duplicated against bank conflicts); QuIP# E8P is a
   ~1 KiB LUT + sign flips. Decode-phase gemv is memory-bound — arithmetic hides under the memory latency the
   compressed weights just removed. AQLM 1x16 (L2-resident 1 MiB LUT) is exactly what not to do.
2. **Decode in the B staging path, once per B tile.** The decoded tile is reused across the whole M tile, so at
   prefill M the decode cost amortizes to noise; at decode M it is the price of an 8× byte reduction on the
   dominant traffic. No separate dequant kernel, no materialized f16 weight copy.
3. **Prefill fallback if fused decode ever loses at large M**: dequant-to-global once + the normal warp-tier path
   (the AQLM repo's dequant+cuBLAS pattern). Only reach for this if measured — with per-tile amortization the fused
   form should win at all M.
4. **Activation-side Hadamard transform** (QTIP/QuIP# requirement) rides the computed-A machinery — it is a
   per-tile prologue on the A operand cone, the same structural slot as the norm→linear fusion (`"fused"` golden
   kind). At decode M the A tile is tiny, so this is cheap where it matters.

Measurement caveat (from the golden YAML preamble): `run --bench --golden` replays one kernel over one weight slab
~100×, so any weight that fits the 96 MB L2 is timed L2-resident and a bandwidth win will NOT show up. VQ A/Bs must
use shapes bigger than L2, twin/serving e2e latency, and the roofline audit (`emmy/serving/roofline.py`) — not
isolated golden benches.

## 5. How it integrates with the compiler (builds on `plans/fp8-support.md`)

Shared verbatim with the FP8 plan — do not re-design:

- **Phase 0 (already in flight on `feature/fp8-support`)**: the `dataclasses.replace` constant-fold fix and
  branch-local dtype propagation are prerequisites for any weight dtype ≠ activation dtype.
- **The three-layer type discipline**: codes enter as plain scalar dtypes (`u8`/`u16` bits carriers, bf16
  precedent) or a packed `StructuredType`; codebooks/scales are first-class sibling tensors; the pairing lives in
  `ConstantOp.quant`. Generalize the FP8 plan's `QuantSpec` (codes/codebook/scale paths + format params) rather
  than adding a second field. Stamped at the constant's birth site (`trace/torch.py`, `_handle_placeholder`).
- **Trace is quantization-blind**: trace the bf16 architecture twin from config; bind real tensors via the
  safetensors path. Required here even more than for fp8 — a VQ checkpoint's module has custom kernels and int
  parameters that survive neither `torch.export` nor the `.float()` cast in `bind_constants_from_module`.
- **`ShapeKey` dtype-class field** and the `992` structural stamps (`S_dtype_u8`/`u16` fall out for free), golden
  `_DTYPES` extension, manual pinned `--ab` seeding, `eval golden --in-model` deploy verification.

Where VQ diverges — the genuinely new compiler work:

- **The decode never commutes out of the fold.** FP8's easy branch (scale constant along K → epilogue multiply)
  does not exist here; a codebook gather is data-dependent per group. Every VQ matmul is a computed-B form.
- **The tile IR already stores it**: `Channel.b: Load | Fold` — a computed B edge is representable today; every
  tier just declines it (`isinstance(c.b, Load)` gates in `_legality.py` and `_schedule.py`). The work is a
  computed-B reading in `_schedule.py` (mirror of the existing mixed-A promotion `_promoted`) plus legality arms.
  Note the readings constraint: at most two per term, mutually exclusive by shape — `_enumerate` raises on a
  `canonical_row_key` collision.
- **The decode lands as a B-side compute fill**: a `SyncOperand` whose value closure emits the code load → LUT
  gather / trellis step → optional scale multiply, writing the decoded slab the existing ldmatrix drain reads.
  Today only computed-A rides the sync fill ("B weights never ride this" — that docstring changes). The codes
  stream through the existing async prefetch ring (cp.async on `async_operands`; TMA is out — a descriptor needs
  a gmem address on both edges); the LUT loads once via the `sync_stat_fill`-style prologue slot.
- **The fragment path must land before any perf judgment.** The FP8 plan's recorded constraint binds harder here:
  a dtype-boundary copy on an operand cone demotes the matmul off the mma tier (measured 1.12 vs 1.61 TB/s), and a
  gather can't be absorbed by copy transports at all. Sequence the warp-tier decode first or every A/B undersells
  the format.
- **The gemv/reduce tier needs the decode too.** Decode-phase matvecs derive PLANAR and take the reduce tiers,
  which is not where the sync fill lives — the LUT decode needs a reduce-tier realization as well. This is where
  the TPOT win actually cashes out; treat it as first-class, not an afterthought.
- **Bind-time dequant costs nothing to build**: `load_ops` already executes arbitrary frontend ops through the
  numpy backend and `GatherOp` has a numpy `forward()`. A numpy trellis/LUT decode helper + the loader-side
  sibling-tensor lookup gives a correctness lane with zero kernel changes (the V1 milestone) — full-size weights
  in memory, no footprint win, but it unblocks accuracy A/B and the reference path everywhere.

Serving notes:

- emmy owns trunk weight loading end to end (the vLLM plugin's `load_weights` skips the state dict), so a
  codes+codebooks+scales checkpoint is entirely ours to load. Blockers to clear: the `.float()`/np-dtype casts in
  the bind paths (raw-bits carrier, bf16 precedent) and the pack format — `_encode_load_ops` has a restricted
  vocabulary and silently disables pack saving for chains outside it; extend it or prebuilt serving images stop
  getting pack hits for VQ models.
- The footprint win only arrives with in-kernel decode (V2) — bind-time dequant serves at full bf16 footprint.
  Once V2 lands, the freed VRAM goes to KV/admission capacity, which is the whole point on 24/32 GB.

## 6. Producing quantized models from an FP16 checkpoint

Two lanes, in order of leverage:

1. **Ingest existing checkpoints first.** EXL3 has a large, active ecosystem covering current models (turboderp,
   ArtusDev, mratsim orgs on HF) — including the unlock-band models; QTIP's relaxml org covers Llama-2/3/3.3;
   AQLM's ISTA-DASLab PV checkpoints cover Llama-2/3, Mistral, Qwen2 (dormant since). Ingesting a format = reading
   its packing + implementing its decode; no quantization tooling needed on our side at all for V1/V2.
2. **Produce our own with the format's tooling.** EXL3's `convert.py` quantizes a 70B in hours on a single
   RTX 4090 (single step, Hessian-based, no end-to-end fine-tune) — this is the practical path for models the
   ecosystem hasn't covered yet (e.g. a fresh unlock-band release). QTIP's repo is the research-grade equivalent
   (~12 h A100 class + optional distill). AQLM production is not worth it: ~720 A100-hours for a 70B (PV-tuning
   up to 1.5× more), ~50× EXL3's cost, for worse accuracy.

The quantization pipelines themselves (Hessian collection, trellis/beam-search code assignment, codebook learning,
PV-tuning) stay out of emmy — emmy consumes the artifacts. If we later want an in-house producer, EXL3's
single-GPU-hours cost structure is the one to copy, not AQLM's.

## 7. Milestones

### V0 — format decision spike (no emmy code)

Run exllamav3 with the target EXL3 checkpoints (Qwen3.6-27B at ~2 bpw as the dense reference; GLM-4.5-Air 2.00 bpw
and Laguna S 2.1 at its lowest published bpw as the 5090 fit checks) on the 4090 and 5090:
confirm quality (spot-check vs published PPL), decode TPOT vs the memory-bound ceiling, and sm_120 operation.
Read the EXL3 packing + kernel source until the decode is fully understood (trellis state, tile layout, Hadamard
placement, scale handling). Exit criterion: written decode spec + measured reference TPOT numbers, and a confirmed
format pick (trellis vs E8P fallback). Also decides ingest priority: EXL3 vs QTIP checkpoint format.

### V1 — ingestion + bind-time dequant (correctness lane; no kernel changes)

1. Generalize `QuantSpec`; stamp at the constant birth site; loader-side sibling-tensor lookup in
   `loader/safetensors.py` for the chosen format's checkpoint layout.
2. Numpy decode helper (trellis walk / LUT gather) riding `load_ops`; for Hadamard-transformed formats, undo the
   transform offline at bind time so V1 needs no runtime activation transform.
3. Architecture-twin trace + raw-bits bind path (shared with FP8 M1 — land once, both formats consume it).
4. Verify: synthetic tiny-linear fixture → `compile`/`run` accuracy vs the format's reference dequant; then one
   real quantized layer. Digest gate: kernels for non-VQ models byte-identical.

Deliverable: a VQ checkpoint compiles and runs correctly everywhere (weights expanded in memory; no footprint win).

### V2 — in-kernel decode (the real milestone)

1. Computed-B reading + legality arms; B-side `SyncOperand` decode fill with codes on the async ring; LUT/smem
   prologue. Warp tier first (fragment-path constraint above), then the reduce-tier/PLANAR gemv realization.
2. Activation-side Hadamard as a computed-A-style prologue (if the picked format needs it — trellis and E8P both
   do).
3. Search identity: `ShapeKey` dtype-class; golden kind decision (a VQ-B matmul's schedule space diverges — likely
   its own kind, decided the way the fp8 plan defers it); knobs named per the precision-knob conventions (these
   are storage formats, not accuracy knobs — no FAST_MATH gate; the quantization error is in the checkpoint, and
   decode is bit-exact reconstruction).
4. Verify: per-kernel accuracy vs the V1 reference (bit-exact decode ⇒ tight tolerance); perf on >L2-sized shapes
   + roofline audit, NOT L2-resident golden replays; goldens seeded via manual pinned `--ab` on both cards.

Deliverable: VQ-stored linears at ≥80% of the weight-bytes memory-bound ceiling at decode M — the TPOT win.

### V3 — serving + model-level integration

1. `gen_runner`/plan constants: bits carriers on upload; pack-key includes the dtype class; `_encode_load_ops`
   extension (or explicit pack opt-out with a warning) so prebuilt images keep working.
2. Serve an unlock-band model end to end on the 5090; A/B TPOT/TTFT vs exllamav3 and llama.cpp IQ2 on the same
   card — the external baselines, not just stock-vLLM (which cannot run these models on this card at all).
3. KV headroom: measure real admission capacity at the freed footprint; GLM-4.5-Air at ~28 GB weights is the
   stress case (KV quantization may be needed to be usable — separate decision).

### V4 — optional breadth

- Second format (QuIP#/E8P or AQLM-2x8 ingest for the PV checkpoint library) — only if V2's decode abstraction
  made it cheap (the `SyncOperand` fill + `QuantSpec` should make a new format ≈ a new decode closure + loader).
- MoE-specific work if an MoE unlock-band model is the serving target (routing support is its own feature).

## Open questions

- **MoE support**: the highest-demand unlocks (GLM-4.5-Air, Laguna, Qwen3.6-35B-A3B) are MoE — now scoped
  separately in `plans/moe-support.md` (delivery target gpt-oss-20b FP8; the VQ flagships are its named future
  targets once sub-byte lands here). The dense Qwen3.6-27B carries this plan's V3 until then.
- Trellis decode register pressure inside the sync fill at warp tier: 2–4 instr/weight is cheap, but the
  sequential 256-weight walk's ILP interaction with the prefetch ring depth is unmeasured — V0/V2 A/B decides
  ring depth per shape (the existing `d*` fork siblings already express this).
- EXL3 format stability: it is an actively-developed single-maintainer format; pin a converter version and store
  the format revision in `QuantSpec` from day one.
- Whether the decode-tier gemv work overlaps the existing M=1 gemv-tier gate (`EMMY_GEN_M1_TIER` off pending the
  degenerate-composition recognizer fix) — if that lands first, VQ decode inherits the M=1 tier for free.
- `lm_head` / embeddings: biggest single matrices, usually kept at higher precision by every format (EXL3 allocates
  them more bits); follow the format's allocation, don't invent our own.
