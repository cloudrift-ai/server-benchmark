# DeepSeek V4 Flash 0731 through `emmy serve` on 16× V100 (TP8 × PP2)

Goal: boot `deepseek-ai/DeepSeek-V4-Flash-0731` through the Emmy vLLM plugin on the 16× V100 SXM3 host — Emmy compiled
kernels for the hyper-connection stream mixing, norms, shared expert and routed experts; the pinned 1Cat sm_70 fork
(`cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608`) supplying paged MLA attention and the serving shell —
then A/B against the plain 1Cat container at an equal serving envelope.

**Expectation setting, up front.** At this seam the Emmy arm replaces the fork's already-fused mHC kernels and its
TurboMind MXFP4 MoE. With fp8 experts (2× the bytes, no fp8 tensor cores on Volta) and per-hit-expert dispatch, the
first working arm should be expected to LOSE the A/B. The deliverable of stages 1–5 is a *working, measured* lane and
the integration machinery; a *competitive* lane additionally needs stage 6 (MXFP4 expert inputs), and even that is a
hypothesis until profiled. The A/B must rerun the 1Cat baseline at the same shorter context — the published
30.79 tok/s was measured at a 1M-token envelope the Emmy arm cannot hold.

## Current state (already landed on this branch)

- The attention-sublayer seam: `pre(streams[T, hc·H]) → x[T, H]`, `post(attn_out[T, H], streams) → (mixed, xn,
  mix[T, hc])`, routed experts on the MoE third seam, closed by `place_routed_streams`. CPU-proven against the eager
  HF layer (sliding/HCA/CSA × hash/top-k). One twin profile per model (attention and routing are outside the twins).
- Serving-twin capture + `eval golden` audit accept DeepSeek V4; on the target host the `pre` and `expert` twins
  realize 20/20 (m1 / m32 / m4096 / dynamic) on sm_70.
- Checkpoint facts: DeepSeek-native names (`layers.N.attn.wq_a` …), trunk fp8-e4m3 with `F8_E8M0` [128, 128] block
  `.scale` siblings, routed experts MXFP4 (`I8 [out, in/2]` nibbles + e8m0 per-32 scales), `hc_mult` 4, 43 layers,
  256 experts × top-6 + 1 shared, 3 hash-router layers. The snapshot ships a lossless MXFP4 → fp8+e8m0[128,128] cast
  (`inference/convert.py`).
- Fork facts: `DeepseekV4Attention.forward(positions, normed_x[T, H]) → [T, H]` is one self-contained sublayer whose
  projections, compressors, indexer, FP8 paged insert and grouped output projection are fused with its own paged
  caches (SWA + compressor + indexer cache layers registered by prefix); sm_70 requires `--dtype half` and the
  `VLLM_SM70_*` env. There is no API accepting externally computed compressed latents.

## Memory budget per GPU (32 GB; TP8 × PP2 ⇒ 21/22 layers per stage; 256 experts sharded 8-way per TP group)

Worst (22-layer) stage, 32 experts/rank/layer, `w1+w2+w3` = 25.17 MB values per expert:

| item | fp8 experts | MXFP4 experts (+e8m0 per-32 scales) |
| --- | ---: | ---: |
| routed experts (sharded) | ~17.7 GB | ~9.4 GB |
| shared experts + hc/norm params (replicated) | ~0.6 GB | ~0.6 GB |
| fork attention weights (TP-sharded + replicated parts) | ~1.0 GB | ~1.0 GB |
| PP0 embedding (full vocab, fp16, runner-resident today) | ~1.1 GB | ~1.1 GB |
| Emmy activation arenas (capacity 4096 × hc·H carrier) | ~1.5 GB | ~1.5 GB |
| vLLM + CUDA context + fork workspaces | ~2–3 GB | ~2–3 GB |

KV capacity is NOT derivable from a bytes/token constant: sliding layers cache a 128-token window and HCA/CSA layers
cache compressed entries, which is how the recorded MXFP4 baseline allocates 4.2M tokens of KV per stage. Measure at
the first full boot (weight bytes, arenas, non-Torch allocations, GPU blocks, KV tokens per stage). Qualify initially
at 4K–32K `--max-model-len`; attempt the recipe's 1M only after that measurement proves capacity on both stages.

## Stage −1 — freeze and probe the pinned fork contract (cheap; before everything)

Resolve the base image to an immutable digest and record its source SHA. Inside that exact image: contract-test the
attention class (constructor args, forward signature, cache-layer registration and prefix rules, `topk_indices_buffer`
and aux-stream ownership, parameter `weight_loader`s, `WeightsMapper`, quantization config); install the Emmy wheel +
`cupy-cuda12x`; probe `nvrtcVersion` and compile a trivial sm_70 kernel through the same cupy path Emmy uses (add the
CUDA-12 `libnvrtc` preload only if that probe fails, using the path present in the image); confirm the unchanged plain
1Cat model still boots. Stop the plan here if a required fork API is absent.

### Stage −1 findings (2026-08-25, executed on the target host)

- Image pinned: `cloudriftai/1cat-vllm-deepseek-v4-flash-0731@sha256:276240257b224097876b5b6db8f0d32484dff6a6f168d6
  b03d6df188e5c65bc1`, labels confirm build commit `d76126608…` / model revision `7872f01b…` / target GPU V100 SXM3.
- Source integrity: all 69 files of `vllm/models/deepseek_v4` + the MLA backends + the attention layer package are
  byte-identical to the `d76126608` checkout (per-file sha256) — the studied API is the shipped API.
- In-image stack: python 3.12.13, vLLM `1.2.3.dev87+gd76126608`, torch `2.10.0+cu129` — NVRTC in-image is **12.9**,
  which still targets sm_70 (deprecation warning only), so the host-venv NVRTC-13 preload problem does NOT apply
  inside the image.
- Contract confirmed: `DeepseekV4Attention.__init__(vllm_config, prefix, topk_indices_buffer, aux_stream_list)`,
  `forward(positions, hidden_states, llama_4_scaling)`; `DeepseekV4MLAModules` fields as studied;
  `_is_exact_sm70_cuda()` True on the card; `DeepseekV4SWACache(head_dim, window_size, dtype, prefix, cache_config)`;
  MLA attention + indexer cache are `AttentionLayerBase` (prefix-registered KV specs); the fp4/fp8 `WeightsMapper`
  builders exist; `VLLM_SM70_*` env keys and `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` (1024) present;
  `ModelRegistry.register_model` (the OOT plugin hook) available.
- The emmy wheel + `cupy-cuda12x` install into the image cleanly beside vLLM's pins, and `emmy.serving.register` is
  importable there.

## Stage 0 — unblock the `post` twin (compiler; hard prerequisite) — **DONE, fixed upstream**

The stall was never in the graph: it was a compiler pathology that main fixed while this plan was being written.

**Diagnosis (2026-08-25).** A parameterized repro (capture + lower the `post` twin, `sm_70` forced) isolated it by
elimination: hidden size 32→1024 flat at 2.3 s, `hc_sinkhorn_iters` 2→20 linear (+0.13 s/iter), `hc_mult` 2/3/4 flat,
live CUDA device present or absent identical. The graph is byte-for-byte the same everywhere (644 IR nodes). What
differed was the compiler revision: at `5646a0796` (pre-merge) the lowering does not terminate (>240 s locally, 3 days
on the host); at merged main it takes 2.3 s locally and **7.3 s on the V100**. Reverting `71dfa184d` (#595, fresh-name
scans) and `4443f5d98` (#597, memoized splicer walks) leaves it fast, so the fix is `1ee503099` — **Bound fusion
construction work (#602)**, which rewrites exactly `merge_region` / `build_merged_region`, the two frames directly
above the stalled `_ensure_dep` / `fresh` calls in every py-spy sample. It cannot be reverted in isolation because
later commits build on it.

**Realization (2026-08-25, on the target host at merged main).**

| step | result |
| --- | --- |
| `emmy trace --serving-twins` (real checkpoint, pinned serving env) | ~60 s → **12 graphs, 380 distinct kernels** |
| `emmy run --golden` over the inventory | **1520/1520** (380 × m1/m32/m4096/dynamic), exit 0, 550 s |
| errors / tracebacks | none |
| non-finite fragment outputs | 10 / 1520 (0.66 %), all `post` at the widest width |

The non-finite outputs are a property of random-input fragment replay for this checkpoint, not a regression: the
already-qualified, published golden (`recipes/DeepSeek-V4-Flash-0731/golden/v100_sm70.yaml`) replays 279/279 exit 0
with **4 / 279 (1.4 %)** non-finite under the same harness — a higher rate than the new inventory's. `run --strict`
cannot adjudicate a Loop-IR fragment either way (no Torch twin and no independent greedy reference; it reports
"same-input greedy reference is unavailable"), so real correctness evidence comes from the CPU seam equivalence test
today and from real-weight parity at Stage 3.

Two environment notes for later stages: the host venv's torch pulls NVRTC 13, which cannot target sm_70 — the main
process works around it with `LD_PRELOAD=/usr/local/cuda-12.9/lib64/libnvrtc.so.12`, but an isolated **bench worker**
does not inherit the workaround and dies with "invalid value for --gpu-architecture", so any `--bench` / `tune` work
must run inside the 1Cat image (NVRTC 12.9 native, verified) or in a venv without the NVRTC-13 pin. The inventory is
untuned (no knobs or timings) and is therefore NOT promoted to the canonical path; it regenerates in ~60 s.

## Stage 1 — loader lane: read the published checkpoint (CPU-testable) — **DONE (#651)**

Extend the quantized split loader (`load_quantized_split` + `loader/quant.py`) with one DeepSeek-native lane:

1. Key translation native → HF (`attn.wq_a` → `self_attn.q_a_proj` etc.) — reuse transformers' checkpoint
   conversion mapping for `deepseek_v4`, not a hand-rolled copy. `.scale` joins `_scale`/`_scale_inv` as a sibling
   form.
2. e8m0 scales: `F8_E8M0` reads as f32 `2^(e−127)` (torch's `.float()` on `float8_e8m0fnu` is the conversion).
3. Routed experts: per-expert `w1/w3` (gate/up) and `w2` (down) stack into the E-leading store the expert programs
   feed from. Bootstrap route: apply the snapshot's lossless fp4 → fp8+e8m0[128,128] cast per expert AT LOAD (no
   converted checkpoint copy on disk) into the existing fp8 expert-input lane. Keep the cast byte-exact with the
   reference `cast_e2m1fn_to_e4m3fn`.
4. `expert_range=(lo, hi)` filter so a rank reads only its expert shard (a PP stage's full fp8 expert set is
   ~138 GB of host RAM otherwise), alongside the existing `layer_range`.
5. Trunk fp8: dequantize to fp16 values at load (existing lane), including the grouped `wo_a`'s [128,128] blocks.

→ verify: a synthetic tiny native-format checkpoint (tests, mirroring `test_load_quantized_split_*`) loads to a twin
whose eager forward matches `load_dequantized_state_dict` values; the fp4→fp8 cast matches the reference converter in
raw fp8 payload bytes AND raw e8m0 scale bytes across every fp4 nibble code, block-boundary exponents, and random
tensors; `expert_range` applies before stacking/conversion and peak host memory proves no rank materializes non-local
experts; on the host, one real layer's loaded expert slice matches the reference converter's output.

## Stage 2 — runner: DeepSeek widths + TP expert sharding — **DONE (#656)**

1. Seam plumbing in `EmmyGenRunner.from_model`: DeepSeek `_meta` (no `q_proj`; carrier `hc·H`, attention width `H`),
   the 3-output post program (`mixed`, `xn`, `mix`) routed through `_route_post_device` (per-output dest shapes —
   the rider path currently assumes residual-width outputs), `place_routed_streams` as the combine closer,
   `input_ids` reaching the hash-router layers' gate, embed broadcast to `hc_mult` streams before layer 0, and
   `final_norm` = `hc_head` collapse + RMSNorm. The carrier contract is explicit: `runner.carrier_size = hc_mult·H`
   sizes the activation arenas AND the plugin's PP intermediate-tensor factory (which today allocates
   `config.hidden_size`); every PP boundary transports the flattened carrier; only the last rank applies `hc_head`.
2. TP expert sharding: `moe["inputs"]` holds the local expert slice + a global→local index map; the router runs
   replicated (same weights, deterministic); `combine_routed_experts` skips non-local experts; the plugin all-reduces
   the routed `[T, H]` partial (vLLM's tensor-parallel all-reduce) before `place_routed_streams`. `mixed`/`xn` stay
   replicated compute. Eager routed path only at first; the fixed-slot capture tier (per-rank tables) is a follow-up,
   so the first boots serve `--enforce-eager`.

→ verify: a 2-rank CPU/GPU unit test proving sharded-combine + all-reduce equals the single-rank oracle
(`combine_routed_experts` full set); a PP2 test asserting the intermediate-tensor factory, send/receive tensors and
residual arenas all use `hc_mult·H` while attention and expert inputs stay `H`; the existing gen_runner GPU stitch
tests stay green; a single-GPU tiny-config DeepSeek stitch test (seam programs + torch attention stand-in) matches
eager.

## Stage 3 — plugin: host the fork's attention inside `EmmyGenModel` — **in-repo work DONE (#662)**

1. A DeepSeek branch that constructs the fork's `DeepseekV4Attention` per layer (needs `vllm_config`, the shared
   `topk_indices_buffer`, aux streams, unique prefixes so its SWA/compressor/indexer cache layers register in the
   static forward context and get KV allocations), instead of vLLM `Attention` + RoPE.
2. Weight routing: an explicit ownership table over checkpoint keys — fork attention (via the pinned fork's
   `WeightsMapper` + each destination's own `weight_loader`: `fused_wqa_wkv` ← `wq_a`+`wkv`, `compressor.
   fused_wkv_wgate` ← `wkv`+`wgate`, `attn_sink` head-narrowing, indexer params), Emmy trunk/shared/routed programs,
   `lm_head` ← `head.weight`, embedding ← `embed.weight`. Loading fails loudly if a fork-owned attention parameter is
   missing, double-loaded, or an attention checkpoint key stays unclaimed. Fork attention keeps its pinned fp8 quant
   config; Emmy owns trunk/expert conversion. Speculative/MTP serving is rejected at boot.
3. Forward: carrier `[T, hc·H]` as the PP intermediate tensor; per layer `pre → fork attention (positions, x) →
   post → local routed combine → all-reduce → place`; `hc_head` + norm on the last rank.
4. Optional de-risk hybrid, decided UP FRONT (it must be algebraically exact): Emmy `post` already places the
   shared expert, so a hybrid may only call a fork operation of the form `native_routed(xn, input_ids) → routed[T,H]`
   — the fork's routed experts WITHOUT its shared expert and mHC placement. If the pinned fork only exposes the full
   `DeepseekV4MoE` (shared expert included), either add a verified Emmy `post` variant that omits the shared expert,
   or drop the hybrid. Silently keeping both double-counts the shared expert.

→ verify, in grades: (a) a one-process attention contract test — unique absolute layer prefixes, every fork cache
spec registered; (b) a TP2×PP2 small-config distributed test; (c) the TP8×PP2 target-host boot serving mixed
prefill/decode requests (not just `/health`); (d) layer-level numerical agreement vs the eager seam at predefined
atol/rtol, then deterministic greedy token-ID agreement on a fixed corpus spanning HCA/CSA/hash layers, multiple
expert destinations, PP transport, and mixed scheduling — token IDs either agree or they do not. Also verify
`emmy serve`'s MoE probe recognizes `n_routed_experts` (or pin capture sizes + eager in the release config).

### Stage 3 findings (2026-08-26, PR #662)

- Items 1–3 landed; gates (a), the weight-loading gate, and (b) are green in the pinned image: construction and
  per-absolute-layer cache registration across all three attention layer types, the attention ownership table
  against the real fork modules, and a REAL-engine parity gate — the same checkpoint served single-rank and
  TP2×PP2 produces identical greedy token ids modulo numerical ties (exact ids stay the real checkpoint's gate,
  where logits are decisive).
- Load-bearing surprises, each encoded in code/tests/docs: a vLLM process re-registers `deepseek_v4` onto its own
  rope-only config class process-wide, so the loader reloads with Transformers' same-named native class; the
  fork's kernels accept only the published geometry (compressor head 512 / indexer head 128 / 128-aligned fp8
  group outputs / 128×128 blocks, and no invented `compress_rates`); compiled twins may hand outputs back in
  their accumulation dtype, normalized at the seam by the runner; and the machine-wide GPU file lock deadlocks
  multi-rank serving (a rank holds it inside its combine while its pending collective waits on the peer queued
  behind the same lock) — it is now scoped per physical device UUID.
- The remaining gates move to the on-host block beside Stage 4's image work: (c) the TP8×PP2 target-host boot
  serving mixed prefill/decode, and (d) real-checkpoint layer-level numerics plus greedy token-ID agreement.

## Stage 4 — image + release plumbing

1. Build the plugin image FROM the immutable 1Cat digest with `cupy-cuda12x`, with its own image identity — do not
   inherit the Makefile's default `v0.23.0` version/tag for a 1Cat 1.2.3 base. Label the 1Cat digest + source SHA,
   Emmy SHA, checkpoint revision, and CUDA/NVRTC versions. The stage −1 NVRTC probe decides whether the serve
   entrypoint needs the CUDA-12 `libnvrtc` preload for sm_70.
2. Env passthrough: the serve image/env-file plumbing gains the fork's `VLLM_SM70_*` variables and
   `--tensor-parallel-size 8 --pipeline-parallel-size 2 --distributed-executor-backend mp` in `SERVE_EXTRA_ARGS`.
3. Headroom sweep on the host seals `docker/vllm-emmy-serve/models/deepseek-v4-flash-0731.env` (the sweep creates
   it; do not author widths off-host); record the serving golden; warm → bake → verify per the release skill.

→ verify: `make serve-config / serve-goldens / serve-warm / serve-image / serve-verify` pass on the host; the baked
TP8×PP2 image cold-starts offline and EVERY one of the 16 workers reports its pack hit (today's verify accepts one
`pack hit` line — insufficient for 16 workers), the cubin set is unchanged, and no request-time Triton JIT occurs.
Build/bake/verify only; registry publication is a separate approval.

## Stage 5 — A/B

`emmy bench` two arms at the SAME envelope (same `--max-model-len`, mnbt, concurrency, KV dtype, warmups,
immutable image digests and checkpoint revision; the existing serving_v100_sxm3 protocol, shorter context; one
priming repeat + three steady repeats): the Emmy image vs plain 1Cat. Profile in a separate run — profiling the
fork's multi-stream execution perturbs the A/B. Report output tok/s, TTFT,
TPOT with repeat spread; archive per the experiment conventions (no credentials or VM identifiers). Publish the
honest conclusion even if (as expected) the Emmy arm loses; include a per-phase profile (expert dispatch vs
attention vs mHC) so stage 6's hypothesis is grounded.

## Stage 6 (performance, optional) — MXFP4 expert inputs

Largely landed upstream since this plan was drafted: main now spells native MXFP4 expert twins
(`spell_mxfp4_inputs`, `decode_mxfp4`, `…@mxfp4` twin names — uint8 nibble blocks + uint8 e8m0 scales as program
inputs). What remains for THIS checkpoint: its declaration is `quant_method: fp8` + `expert_dtype: fp4` (not
`quant_method: mxfp4`), and its packing is `w1.weight I8 [out, in/2]` + `.scale [out, in/32]` (not the gpt-oss
`_blocks`/`_scales` layout the profile recognizes) — so a DeepSeek declaration/orientation mapping onto the existing
spelling, plus the loader keeping the fp4 store, plus tuning. NOTE: with the fp8 declaration, the expert twin already
spells `@f8e4m3` today, which is exactly the stage-1 cast lane's deployed form (test:
`test_deepseek_fp8_declaration_spells_the_expert_twin_for_the_cast_lane`). Only worth building if stage 5's profile
shows expert weight streaming dominates and the fused-unpack GEMM can plausibly beat TurboMind's on Volta.

## Risks

- Stage 0 is open-ended compiler work; nothing below it ships without it.
- The fork's attention inside a foreign model class is the largest integration unknown (cache registration,
  metadata, capture breaks, `VLLM_MULTI_STREAM_GEMM` aux streams); mitigated by the stage-3 hybrid boot.
- Per-hit-expert dispatch at top-6 × 43 layers is a known latency wall (~0.23 ms/launch framing); the fixed-slot
  tier only covers T=1 and needs per-rank tables under sharding.
- Replicated mHC/norm/shared-expert compute across 8 TP ranks wastes ~7/8 of that compute; acceptable at first
  (it is small next to experts), but it caps the ceiling.
- vLLM/fork version drift: everything pins to the one 1Cat image; a fork bump reopens the weight-mapper and
  attention-API assumptions.

## Effort

Stage −1: DONE (~2 h). Stage 0: DONE (fixed upstream by #602; verification ~2 h). Stage 1: DONE (#651).
Stage 2: DONE (#656). Stage 3 in-repo: DONE (#662; gates (c)/(d) move on-host). Stage 4: 2–4 days on-host. Stage 5:
1–2 days. A measured eager fp8 A/B is realistically 3–5 engineering weeks; adding stage 6 (MXFP4 + tuning,
1–3 weeks) makes ~5–8 weeks, with stage 0 and the fork ABI as the dominant uncertainty.
