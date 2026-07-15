# Plan: serve gemma-4-12B on emmy kernels, released as a vLLM image with prebuilt cubins (RTX 5090)

## Goal

A released Docker image, based on the existing `vllm-emmy` serving image, that **serves `gemma-4-12B`** through
`emmy serve --generate` with the transformer trunk on emmy-compiled CUDA kernels, and ships those kernels **prebuilt**
for the **RTX 5090** (`sm_120`/`sm_120a`) so server cold-start pays **zero `nvcc` compile cost**.

- **Phase A — make `gemma-4-12B` servable** via `EmmyGenModel`. Compiler + plugin work.
- **Phase B — prebuilt-kernel image**: warm the serving cubin cache on a real 5090 and bake it into the image.

Success criteria:

1. `emmy serve --generate gemma-4-12B` starts on a 5090 and produces logits/text matching an HF reference. **NOT MET —
   the pipeline runs end-to-end but the trunk forward produces NaN logits (kernel numerics); see Status.**
2. The served model's **first request issues zero new `nvcc` compiles** (100% cubin cache hit) on the 5090.
3. Reproducible: `make` targets build the image end to end from the wheel + a warm step.

## Status

**Landed in `main` (PR #359):** the gemma-4 carve (4-norm layout + per-head q/k/**v**-norm), per-layer sliding window,
per-layer-type RoPE, embed-scale, final-logit softcap, per-layer attention metadata (`layer_meta` — gemma-4's global
layers use `global_head_dim=512` vs sliding 256, plus `attention_k_eq_v`), and `emmy serve --generate --bench`.

**On `feature/gemma4-unified-serving`:** multimodal "unified" checkpoint support + the `EMMY_GEN_DECODE_BUCKET` knob
(see Phase A below).

**Validated on real RTX 5090s (two rentals, both torn down):**

- ✅ **Compiler path on real `sm_120`** — the gemma-4 carve compiles and matches HF on actual Blackwell (6 serving GPU
  tests pass, incl. the heterogeneous `global_head_dim` + `attention_k_eq_v` stitch).
- ✅ **The full serving pipeline runs end-to-end**: 48-layer compile, vLLM profiling, KV init, `/health`, and
  4 × 20-token greedy generations through all 48 emmy layers **including the 512-dim global attention** — with
  `EMMY_GEN_DECODE_BUCKET=0`, `--max-model-len 256 --max-num-batched-tokens 256 --gpu-memory-utilization 0.97`, and
  **vLLM ≥ 0.23** (0.22.1's FlashInfer dispatch cannot run head-size 512; 0.23 supports `[64,128,256,512]` and has
  native gemma-4 — pin the serving extra accordingly).
- ✅ The "silent death" of the first rental was diagnosed: FlashInfer's sampler JIT needs `ninja` **on PATH**
  (`venv/bin` isn't, when the CLI is exec'd by path) — plus the old harness losing the traceback to SSH SIGHUP.
- ✅ **The NaN blocker — root-caused and FIXED (gap #10: dropped `layer_scalar`).** The local layer-by-layer probe
  (teacher-forced, real 12B weights, 4080) first proved emmy ≡ torch-fp16 (Δ ≤ 0.25 wherever fp16 survived — kernels
  exonerated), then that the shared trajectory was ~8× inflated vs HF's true hidden states: the carve dropped the
  decoder layer's final `hidden_states *= layer_scalar` (a buffer; **real 12B values 0.005–0.92, mean 0.62** — fresh
  models hold 1.0, so every tiny-model parity test was blind to it, the third real-checkpoint-only bug). With the
  multiply restored the full 48-layer probe is **NaN-free**, the trajectory matches HF exactly (e.g. layer-7 out max
  56.6 on both), and emmy tracks the fp32 reference within Δ ≤ 0.12 — **fp16 serving is viable; no bf16 work needed**.
  Both gemma-4 tests now pin per-layer scalars off 1.0 so this cannot silently regress.
- Serving-session fix landed on the branch: rotary may promote q/k to fp32 (0.22 proportional rope) → cast back to the
  trunk dtype after RoPE in both forward paths; `validate_gemma4_serve.py` now prints `finish_reason` / token counts /
  top logprobs per prompt, and takes `--gpu-mem-util` / `--max-model-len` / `--max-num-batched-tokens`.

## Phase A — gemma-4-12B generative serving support

### What the real checkpoint forced (only visible with the 12B, not the tiny text-only tests)

`google/gemma-4-12B` loads as **`Gemma4UnifiedForConditionalGeneration`** — a *multimodal* wrapper. The tiny
`Gemma4TextConfig` tests use the text-only class and cannot surface any of this:

- decoder stack + embed/norm are nested at **`model.language_model.*`** (not `model.model.layers`);
- **every** text attribute (`layer_types`, `rope_parameters`, `sliding_window`, vocab/hidden size,
  `final_logit_softcapping`) lives on **`config.text_config`** — reading them off the top-level config silently returns
  `None`, which would no-op the per-layer window/RoPE/softcap logic;
- the tied `lm_head` embedding is at **`model.language_model.embed_tokens.weight`**, so the old
  `model.embed_tokens.weight` alias never matched → uninitialized `lm_head` → garbage logits.

All three are fixed on `feature/gemma4-unified-serving`.

### Corrections to earlier assumptions in this plan (measured on the real config)

- **Attention scaling:** gemma-4 has **no `query_pre_attn_scalar`**; `attn.scaling == 1.0` (the scale is folded into
  `q_norm`, which the carve reuses — so an external SDPA/vLLM `Attention` at `scale=1.0` is correct).
- **Softcapping:** gemma-4-12B **does** set `final_logit_softcapping=30.0` (earlier note said Gemma-3/4 dropped it).
  Wired into the `LogitsProcessor`. `attn_logit_softcapping` is absent.
- For the **12B** the hard extras are OFF: PLE (`hidden_size_per_layer_input=0`), MoE (`enable_moe_block=False`),
  shared-KV (`num_kv_shared_layers=0`), `layer_scalar=1.0`. (E2B/E4B have PLE + 20 shared-KV layers — still deferred.)

### Memory: emmy needs ~2–3× what stock vLLM does (worked around by config; real fixes pending)

Stock vLLM serves gemma-4-12B on a 32 GB 5090. emmy does not. **Both causes are emmy artifacts, not inherent costs** —
`EmmyGenModel` holds no trunk parameters (vLLM keeps only `lm_head`), so the trunk should live exactly once:

1. **Duplicate weights.** Each `_compile_split` does its own `bind_constants` → `CompiledProgram.build`, so the static
   decode-bucket twin binds a **second copy** of every layer's weights → ~2× (~44 GB vs ~22 GB).
   *Stopgap:* `EMMY_GEN_DECODE_BUCKET=0` drops the twin (costs decode speed).
   *Real fix:* **share the constant buffers** between the symbolic and decode programs — same weights, different launch
   geometry.
2. **Per-layer activation buffers (the big one).** Every layer's `CompiledProgram` retains **its own capacity-sized**
   activation buffers — ~350 MB/layer at `max_num_batched_tokens=4096` ⇒ **~17 GB held across 48 layers**. Stock vLLM
   runs layers sequentially and the allocator **reuses one transient buffer**.
   *Real fix:* **pool/share activation buffers** across the per-layer programs. Until then emmy's footprint scales with
   `num_layers`, which is why the 12B doesn't fit where stock vLLM does.

Observed on the 5090: OOM at 32.4 GB (twin on) → 29.0 GB (twin off), the latter **inside vLLM's profiling `forward`**
materializing a `4096 × 3840` fp16 buffer (31.4 MB) — i.e. the compile finished; the buffers killed it. With the twin
off + `ctx 256` + `util 0.97` the 12B fits and serves on the 32 GB card (the working config in Status).

### Remaining Phase-A work (in priority order)

1. ~~THE BLOCKER — NaN logits~~ **DONE**: gap #10, the carve dropped gemma-4's `layer_scalar` — fixed + regression-
   pinned in tests; full 48-layer real-weight probe NaN-free with the trajectory matching HF exactly.
2. **Re-run the end-to-end serve A/B on a 5090** (`validate_gemma4_serve.py`; fully scripted — detached harness,
   sampler, diag) — expected to pass now; this is the Phase-A exit criterion.
3. **Pin vLLM ≥ 0.23 for the serving extra** — 0.22.1 cannot dispatch gemma-4's 512-dim global attention.
4. **Share constants** between the symbolic and decode-bucket programs (kills the 2× weights, keeps decode speed).
5. **Pool activation buffers** across per-layer programs (removes the `num_layers` scaling).
6. **PLE** — still out of scope for v1 (12B is dense); only for the deferred `gemma-4-E2B/E4B`.

## Phase B — prebuilt-kernel image

**Gated on Phase A succeeding end-to-end** — there is nothing to warm until the server runs.

### Cache-key parity contract

A prebuilt cubin is reused iff all five of `sha1(source, name, arch, toolkit_tag, flags)`
([nvcc.py](emmy/compiler/backend/cuda/nvcc.py)) match what the server regenerates:

- `source` — same emmy wheel + same GPU featurization (a real 5090) + offline prior (pinned in wheel) + same serving
  specs (dtype / max-model-len / decode bucket — note `EMMY_GEN_DECODE_BUCKET` now changes which programs exist).
- `name` — `op.kernel_name` verbatim.
- `arch` — target cap + `uses_tma`: `sm_120` / `sm_120a`.
- `toolkit_tag` — warm with the **image's** `nvcc`, not the host's.
- `flags` — `-O3`: leave `EMMY_NVCC_FLAGS` empty; never warm with tune's `-Xcicc -O1`.

`prior.json`: **offline-prior-only for v1** — the repo-pinned `OfflinePrior` ships in the wheel and is deterministic at
both ends; the `OnlinePrior` falls back to it when absent/untrusted (`FallbackPrior`). One global, GPU-agnostic prior; no
per-GPU file. A real-5090 tune is a later perf upgrade, not a blocker.

### Warm + bake

Start the server once on a real 5090 (inside a container from the image → `toolkit_tag` = image nvcc, `-O3`), let it
compile, snapshot `EMMY_CUBIN_CACHE`, and bake it in (`COPY --from=warm`). Base
**`vllm/vllm-openai`** (`docker/vllm-emmy/Dockerfile` — vLLM + plugin already wired); new `make gemma4-serve-image` /
`-push` mirroring the `vllm-emmy-image` targets. **Verify:** snapshot the cubin dir, start + issue one request, diff the
file set — empty diff = 100% hit.

## Risks / open questions

- **The blocker is numerics, not hardware.** The NaN reproduces wherever the kernels run; it is debuggable locally on
  the 4080 with real weights, so no rental is needed until the final re-validation. (The memory artifacts are worked
  around by config; the real fixes — shared constants, pooled buffers — still matter for a deployable footprint, since
  emmy's currently scales with `num_layers`.)
- **`total_mem` featurization drift** (second-order): memorized 5090 VRAM vs a live 5090's probed bytes differ slightly;
  if the offline prior is sensitive for any serving kernel, that kernel misses once and recompiles (correct, not free).
  The real-5090 hit-rate check catches it; fix by pinning the 5090 spec or wiring `probe_live_features(fallback_name=…)`.
- **Tiny-model tests can't catch checkpoint-shape bugs.** Every unified-wrapper bug above was invisible to the
  synthetic `Gemma4TextConfig` tests. Consider a cheap config-shape assertion against the real `text_config`.

### Decisions

- **Gemma-4 variant — `gemma-4-12B`, 5090, v1.** No PLE; E2B/E4B (PLE + shared-KV) deferred.
- **Serving dtype / max-model-len / decode bucket** (open) — sets which kernels get warmed. `decode_bucket` is now a
  released knob and part of the cache key, so pin it for the release.
- Image: single 5090 tag for v1 (arch-keyed cache; a 4090 set can be added later without collision).

## Notes

- `plans/` is at 11 (cap 10). Prune one executed/obsolete plan at commit time.
