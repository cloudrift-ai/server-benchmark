# Plan: serve gemma-4-12B on emmy kernels, released as a vLLM image with prebuilt cubins (RTX 5090; 4090 deferred)

## Goal

A released Docker image, based on the existing `vllm-emmy` serving image, that **serves `gemma-4-12B`** through
`emmy serve --generate` with the transformer trunk on emmy-compiled CUDA kernels, and ships those kernels **prebuilt**
for the **RTX 5090** (`sm_120`/`sm_120a`) so server cold-start pays **zero `nvcc` compile cost**.

**Scope decision (locked): `gemma-4-12B`, served on the 5090, for v1.** 12B fp16 weights (~24 GB) leave no room for KV
cache on a 24 GB 4090, so the 4090 serving set is **deferred** — it returns with a VRAM-fitting variant (`gemma-4-E2B`,
which additionally needs the PLE carve). Because we serve and validate on a real 5090, the GPU-less cross-compile
machinery from the earlier draft is **off the critical path** (kept only as the future-4090 note).

Two phases, in order — the second is worthless until the first lands:

- **Phase A — make `gemma-4-12B` servable** via `EmmyGenModel` (today it raises `NotImplementedError`). Compiler +
  plugin work; the bulk of the effort.
- **Phase B — prebuilt-kernel image**: warm the serving cubin cache on a real 5090 and bake it into the image.

Success criteria:

1. `emmy serve --generate gemma-4-12B` starts on a 5090 and produces logits/text matching an HF (or stock-vLLM)
   reference within tolerance.
2. The served model's **first request issues zero new `nvcc` compiles** (100% cubin cache hit) on the 5090.
3. Reproducible: `make` targets build the image end to end from the wheel + a warm step.

## Status (Phase A in progress — branch `feature/gemma4-generative-serving`)

**Key correction: the target is `gemma-4`, not `gemma-3`.** `google/gemma-4-12B` is `Gemma4UnifiedTextConfig`
(`modeling_gemma4`), architecturally distinct. For the **12B specifically** the hard extras are OFF — PLE
(`hidden_size_per_layer_input=0`), MoE (`enable_moe_block=False`), shared-KV (`num_kv_shared_layers=0`),
`layer_scalar=1.0`, `attn.scaling=1.0` (folded into q_norm) — so it reduces to the 4-norm layer. But gemma-4 adds
pieces gemma-3 lacks: **per-head V-norm**, **partial/proportional RoPE** on global layers, and
**`final_logit_softcapping=30.0`**. (E2B/E4B additionally have PLE + shared-KV — much more work, still deferred.)

Done + validated on this box (RTX 4080 / CPU):

- **Carve (#1)** — extended to Gemma's 4-norm layout **and** V-norm; reproduces a real gemma-4-12B-shaped layer
  **exactly** (max abs diff 0.0). CPU parity test (`gemma4` case) + GPU compile test (gemma post lowers/runs) green.
- **#2 per-layer sliding window** — each vLLM `Attention` built with `per_layer_sliding_window` (vLLM 0.23 supports it).
- **#3 per-layer-type RoPE** — `_build_rotaries` builds local/global per `layer_types`; vLLM has a native
  `Gemma4RotaryEmbedding` for the proportional type (both types construct cleanly).
- **#5 embedding normalizer** — `embed_scale` (√hidden) folded into the runner's gather table.
- **#6 final logit softcapping** — wired into the `LogitsProcessor` (`soft_cap=final_logit_softcapping`), as stock vLLM.
- Guard relaxed to allow per-layer sliding (Gemma); uniform-sliding + dual-chunk still rejected.

**#7** (`--generate --bench` wiring) — done: generative bench drives `/v1/completions` + `--random-output-len`.

**Gap #9 — heterogeneous per-layer attention — FOUND then RESOLVED via the on-hand 4080 validation.** An
`EmmyGenRunner` forward on a tiny dense gemma-4 (no ≥24 GB card) surfaced that **gemma-4's global (`full_attention`)
layers use a larger `head_dim` than sliding layers** (real 12B: sliding 256, `global_head_dim=512`; 8 global / 40
sliding of 48) and set **`attention_k_eq_v=True`** (no `v_proj` → V reuses K). The runner / model assumed homogeneous
layers (layer-0 metadata), feeding a sliding-width `attn_out` into a global `o_proj` (`[s,256·H] × [512·H,…]` mismatch).
**Fixed:** the runner stores **per-layer** `(head_dim, num_heads, num_kv, scaling)` (`layer_meta`) and compiles each
layer's `pre`/`post` at its own width; `EmmyGenModel` builds each vLLM `Attention` + RoPE at the layer's `head_dim`; the
carve handles `v_proj=None`. **Validated end-to-end on the 4080**: the full tiny-gemma-4 trunk (heterogeneous
`global_head_dim` + `attention_k_eq_v`) matches HF eager at ~1e-6 rel — committed as
`test_gen_runner_gemma4_heterogeneous_stitch` (harness: `scratchpad/validate_gemma4_gen_runner.py`).

The **emmy compute path for gemma-4 is now validated end-to-end on-hand.** What still needs a ≥24 GB card is the
real-checkpoint **vLLM serve** run (`emmy serve --generate google/gemma-4-12B`): served logits/text vs HF, the per-layer
sliding-window + hybrid KV cache under vLLM, and served-RoPE (`Gemma4RotaryEmbedding`) numeric parity.

## Phase A — gemma-4-12B generative serving support

### Why it's blocked today

`EmmyGenModel` builds one plain-causal vLLM `Attention` per layer and one shared RoPE, then brackets each with the
runner's `pre`/`post` carve ([vllm_model_gen.py:96-152](emmy/serving/vllm_model_gen.py#L96)). The carve
`build_attention_split_wrapper` ([huggingface.py:258](emmy/compiler/trace/huggingface.py#L258)) is hand-reconstructed
to the **Llama/Qwen 2-norm** decoder-layer shape:

```text
pre :  input_layernorm → q/k/v proj → (q_norm/k_norm if present) → un-rotated q,k,v
post:  residual + o_proj(attn) ; h + mlp(post_attention_layernorm(h))
```

Gemma-3/4's decoder layer is a **4-norm** shape the carve does not model:

```text
h = residual + post_attention_layernorm( o_proj(attn) )          # norm BEFORE the residual add
h = residual + post_feedforward_layernorm( mlp( pre_feedforward_layernorm(h) ) )
```

Plus per-layer **sliding vs global** attention and per-layer-type RoPE theta. Hence the guard at
[vllm_model_gen.py:70-76](emmy/serving/vllm_model_gen.py#L70) — the carve would silently miscompute, so it rejects.

### The gaps to close (grounded in the code)

1. **Gemma-aware `pre`/`post` carve.** Extend `build_attention_split_wrapper` to detect the Gemma layout
   (`pre_feedforward_layernorm` / `post_feedforward_layernorm` present) and reconstruct the 4-norm block with correct
   norm placement (post-attn and post-ffn norms apply **before** their residual add). QK-norm is already picked up
   generically (`getattr(attn, "q_norm", …)`), so Gemma's per-head norms come for free.
2. **Per-layer sliding window** on the vLLM `Attention`. Read `config.layer_types` / `sliding_window` and construct each
   `Attention` with the layer's window (global → `None`). vLLM's paged attention does the windowing once configured;
   confirm the pinned vLLM version's `Attention(per_layer_sliding_window=…)` signature.
3. **Per-layer-type RoPE.** Build a **local** (sliding, θ≈10k) and **global** (θ≈1M) rotary and apply the right one
   per layer between `pre` and `self.attn` — replaces the single shared `self.rotary_emb`
   ([vllm_model_gen.py:111](emmy/serving/vllm_model_gen.py#L111)).
4. **Attention scaling** — already handled: the runner reads `attn0.scaling`
   ([gen_runner.py:174](emmy/serving/gen_runner.py#L174)), which HF sets to `query_pre_attn_scalar**-0.5` for Gemma.
   Confirm, don't rebuild.
5. **Embedding normalizer** (`inputs_embeds * sqrt(hidden)`). Gemma applies it in the model forward, not in
   `embed_tokens`. Verify the runner's `embed`/`embed_device` includes it (the full-model trunk path does; the
   gen_runner carve may not) and add if missing.
6. **Softcapping** — Gemma-3/4 dropped attn/final logit softcapping (uses QK-norm). Confirm the 12B checkpoint's config
   has none; if present, pass `logits_soft_cap` to `Attention` and softcap in `compute_logits`.
7. **`--generate` + `--bench` wiring** — the bench client only targets `/v1/embeddings`
   ([serve.py:198](emmy/commands/serve.py#L198)). Add a generative bench path (`vllm bench serve` completions/chat) so
   the image can self-bench. Separable, smaller.
8. **Per-layer embeddings (PLE)** — **out of scope for v1** (`gemma-4-12B` is dense, no PLE). Only resurfaces with the
   deferred `gemma-4-E2B/E4B` 4090 target: each nano layer computes `hidden * per_layer_input`
   ([huggingface.py:223](emmy/compiler/trace/huggingface.py#L223)) and the current carve threads no `per_layer_input`.

### Phase-A validation → verify: carve parity, then served logits match reference

- **Carve unit test (CPU, cheap):** one Gemma-4 layer, random weights — assert the `pre`/`post` split reproduces the HF
  block forward within tolerance. Arch- and VRAM-independent, so it runs anywhere and gates the 4-norm logic.
- **End-to-end (5090):** `emmy serve --generate gemma-4-12B` up + `/health`; compare next-token logits / a short greedy
  continuation against HF eager (and `--stock` vLLM). 12B fp16 doesn't fit a 24 GB 4090, so the full-model check runs on
  the 5090.

## Phase B — prebuilt-kernel image

### What "prebuilt" means for serving

The generative path compiles, per layer, **two dynamic-`num_tokens` programs** (`pre` + `post`) plus **static
decode-bucket twins** ([gen_runner.py](emmy/serving/gen_runner.py) — up to ~4 capacity programs/layer). Those are the
kernels to warm. The cubin cache is content-addressed on `sha1(source, name, arch, toolkit_tag, flags)`
([nvcc.py:105](emmy/compiler/backend/cuda/nvcc.py#L105)); a prebuilt cubin is reused iff all five match what the server
regenerates at start.

### Cache-key parity contract

- `source` — same emmy wheel + same GPU featurization (a real 5090) + offline prior (pinned in wheel) + same serving
  specs (dtype / max-model-len / decode bucket).
- `name` — `op.kernel_name` verbatim.
- `arch` — target cap + `uses_tma`: `sm_120` / `sm_120a` (5090).
- `toolkit_tag` — warm with the **image's** `nvcc` (CUDA 13.0), not the host's 13.3.
- `flags` — `-O3`: leave `EMMY_NVCC_FLAGS` empty; never warm with tune's `-Xcicc -O1`.

`prior.json`: **offline-prior-only for v1** — the repo-pinned `OfflinePrior` (renamed from `AnalyticPrior`) ships in the
wheel and is deterministic at both ends; the learned `OnlinePrior` falls back to it when absent/untrusted
(`FallbackPrior`). One global, GPU-agnostic prior;
no per-GPU file. A real-5090 tune is a later perf upgrade, not a blocker.

### Warming the serving cubins

The serving programs are built at engine start on the GPU, so the reliable warm is to **start the server once on the
5090** (inside a container from the image → `toolkit_tag` = image nvcc, `-O3`) and let it compile the pre/post +
decode-bucket kernels, then snapshot `EMMY_CUBIN_CACHE`.

- **5090 set (the only v1 set)** — this is the *same* card session as Phase-A end-to-end validation, so it's one 5090
  rental, not two: validate correctness, then snapshot the warmed cache.
- **4090 set — deferred** (12B doesn't fit; it returns with `gemma-4-E2B`, which also needs PLE). The GPU-less
  `--target sm_120` + `DEFAULT_GPU`=5090 cross-compile technique from the earlier draft only matters if that 4090
  target comes back — it is not needed now.

Bake the 5090 cubin set into one image layer (`COPY --from=warm`), producing the released tag.

### Image + make targets

- Base **`vllm/vllm-openai:v0.22.1`** (the existing `docker/vllm-emmy/Dockerfile` — vLLM + `emmy` plugin already
  wired). Add the warmed cache; keep `EMMY_CUBIN_CACHE` under the HF cache mount.
- New `make gemma4-serve-image` / `-push` mirroring `Makefile:69-74` (`vllm-emmy-image`/`-push`); tag
  `cloudriftai/vllm-emmy-gemma4-12b:<ver>-<sha>`.

### Phase-B validation → verify: 0 new compiles on the 5090

Snapshot the cubin dir file set, start the server + issue one request, diff the file set — **empty diff = 100% hit** —
on the 5090.

## Risks / open questions

- **Phase A is the bulk of the effort and can slip the timeline.** The 4-norm carve + per-layer window/RoPE is a real
  feature; scope it before committing to Phase B dates.
- **Serving needs a real 5090 to validate.** Correctness + the hit-rate check both run on the actual card (rent for
  minutes); there is no offline substitute for the serving path. (The CPU carve unit test de-risks Phase A cheaply
  before the rental.)
- **`total_mem` featurization drift** (second-order): the memorized 5090 VRAM vs a live 5090's probed bytes differ
  slightly; if the offline prior is sensitive for any serving kernel, that kernel misses once and recompiles (correct,
  not free). The real-5090 hit-rate check catches it; fix by pinning the 5090 spec or wiring
  `probe_live_features(fallback_name=…)`.

### Decisions

- **Gemma-4 variant — DECIDED: `gemma-4-12B`, served on the 5090 for v1.** No PLE (gap #8 out of scope); 4090 deferred.
- **Serving dtype / max-model-len / decode bucket** (open) — set which kernels get warmed; default to the `emmy serve`
  defaults (fp16, the runner's dynamic `num_tokens` specs + decode bucket 16). Confirm max-model-len for the release.
- Image: single 5090 tag for v1 (arch-keyed cache; a 4090 set can be added later without collision).
- `prior.json`: **offline-prior-only for v1**; optional real-5090 `OnlinePrior` tune later purely for perf.

## Notes

- `plans/` is at the 10-file cap; adding this makes 11. Prune one executed/obsolete plan at commit time to stay ≤10.
