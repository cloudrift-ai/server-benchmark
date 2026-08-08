# emmy.serving — vLLM out-of-tree embedding plugin

Serve an embedding model (Qwen3-Embedding family) with vLLM's serving shell — OpenAI `/v1/embeddings`, tokenizer,
scheduler, pooling — while the transformer trunk runs on emmy-compiled CUDA kernels. The point is a clean A/B
inside one serving stack: stock vLLM kernels vs emmy kernels, same API, same batching, same pooler.

```
vllm serve Qwen/Qwen3-Embedding-0.6B --runner pooling --enforce-eager \
  --max-model-len 4096 --hf-overrides '{"architectures":["EmmyEmbedModel"]}'
```

`emmy serve` (`commands/serve.py`) wraps that boilerplate: `emmy serve <model> [vllm flags...]`, with
`--stock` for the raw-vLLM baseline at the same max-model-len, and `--bench` for a one-shot start → `/health` →
`vllm bench serve --backend openai-embeddings` → results → shutdown cycle.

Requires the `serving` extra (`pip install -e ".[compile,serving]"` + cupy). vLLM discovers the plugin through the
`vllm.general_plugins` entry point (`emmy.serving:register` in pyproject.toml), which registers
`EmmyEmbedModel` by lazy string path; `--hf-overrides` swaps the served repo's `architectures` to it, so the
checkpoint, tokenizer, and sentence-transformers pooling config still come from the original HF repo.

## Module map

- `__init__.py` — `register()`, the entry-point hook. Never imports vllm/torch at module level. Besides registering
  the model classes it calls `ensure_plugin_logging()` (`emmy/logging_setup.py`): under a bare vLLM entrypoint
  nothing handles emmy's INFO records, which would silence the runners' boot/pack lines in `docker logs` — the
  gemma4 image's verify gate greps the "pack hit" line there (no-op when logging is already configured, e.g. the
  `emmy` CLI).
- `vllm_model.py` — `EmmyEmbedModel` (the only module importing vllm). An `nn.Module` with **no parameters**:
  `is_pooling_model = True`, `IsAttentionFree` (no vLLM `Attention` layers → V1 builds an empty KV-cache spec),
  `attn_type = "encoder_only"` (vLLM disables chunked prefill → every request reaches `forward` whole),
  `pooler = DispatchPooler.for_embedding(...)` (last-token pooling + L2 normalize + matryoshka — identical to stock
  Qwen3-Embedding serving), stub `embed_input_ids`, no-op `load_weights` (the runner loads the checkpoint itself; not
  consuming the iterator skips reading the safetensors). `forward` keeps everything **on-device**: it clamps + casts
  the packed ids to int64 on the GPU, slices each span straight into the runner (torch tensors, no `.cpu()`/numpy), and
  `torch.cat`s the torch results back. The only host touch is a small `positions.cpu()` to find span boundaries for
  `split_spans` (a `(num_tokens,)` int vector). With `batch_cap > 1` it hands all spans to
  `runner.forward_hidden_states_batched` (one padded batched forward) instead of looping per span.
- `runner.py` — `EmmyForwardRunner`. At engine start: load the `AutoModel` **trunk** (hidden states out — no
  lm_head), `build_full_model_wrapper(dynamic=True)`, trace with the canonical 4-spec dynamic seq_len
  (`seq_len@input_ids:1`, `@attention_mask:2`, `@attention_mask:3`, `@position_ids:1`), compile through `CudaBackend`
  (greedy fork picks from the global prior — benefits from any prior `emmy tune`), bind weights as graph
  constants (`named_parameters` + `named_buffers`, `remove_duplicate=False`, in the traced dtype), and build ONE
  `CompiledProgram` over a buffer set sized at **`max_seq_len`** (`--max-model-len`). Per
  sequence (`forward_hidden_states`) it takes a **1-D int torch CUDA tensor** and returns an `(S, hidden)` **torch CUDA
  tensor** — no host round-trip. It enters a cupy external stream bound to torch's current stream
  (`cp.cuda.Stream.from_external`), then: bridge the
  torch ids to cupy (`cp.from_dlpack`, zero-copy), size the launch grids to S (`set_sym_values`), copy ids /
  **device-built** causal mask / position_ids into the buffers' contiguous **prefix** device-to-device
  (`upload_prefix_device`), **capture-or-reuse** the whole-program CUDA graph for this S, **replay** it — one host launch
  instead of the ~hundreds the uncaptured loop issues — and wrap the output buffer's real-S prefix back as a torch
  tensor (`output_prefix_device` + `torch.from_dlpack`, cloned because the shared buffer is reused next request). The
  causal mask is built once per S as a cupy array (the device twin of `_causal_mask_np`) and reused. Captured graphs are
  cached per seq_len (bounded LRU);
  each is captured at its EXACT S so every kernel runs at its exact grid — no oversized-grid masking (a single
  capacity-baked graph for all S is **not** viable: several symbolic-M kernels do illegal reads at an oversized grid,
  the swizzle decode + staged loads among them). See `compiler/backend/cuda/ARCHITECTURE.md`
  → repeated execution + captured replay. Trunk compute dtype follows vLLM's `--dtype` (`mc.dtype`, mapped in
  `vllm_model._trunk_dtype_str`): `float32`→fp32, `float16`→fp16, anything else (e.g. `bfloat16`/`auto`) downcasts to
  fp16 with a warn — the runner's numpy weight carrier can't represent bf16, and only fp16/fp32 trunks are supported.
  With `EMMY_SERVING_BATCHED=1` (`config.serving_batched`) the symbolic-seq trace bakes the batch extent at
  `max_num_seqs` and `forward_hidden_states_batched` runs each step as one batched forward padded to the step's
  longest sequence; `EMMY_SERVING_STATIC=1` (`config.serving_static`) instead traces a **fully-static**
  `(max_num_seqs, max_seq_len)` graph (no dynamic_shapes) — see "Batched modes" below.
- `packed.py` — `split_spans(positions, max_seq_len)`: vLLM V1 hands pooling models one packed `(num_tokens,)` tensor
  with per-request 0-based positions; spans split at `positions == 0`. Hardened for `_dummy_run`'s garbage profiling
  batches (index 0 always opens a span; overlong spans are chopped).
- `roofline.py` — **boot roofline audit**. `EmmyGenRunner.from_model` event-times each STATIC twin (one layer per
  attention class; symbolic programs skipped — they sit at capacity shape at boot) against its **weight-streaming
  floor** (`const_bytes / dram_bw`, bandwidth self-calibrated with a D2D copy — no per-card table) and logs a loud
  WARNING naming any program >10x over it, with the `emmy tune` pointer. Conservative by construction (the weight
  floor is a true lower bound and copy bw undershoots peak), advisory only (never raises, never blocks boot). Born
  from the 2026-07-29 TinyLlama/4080 incident: a cold deploy served a fused-norm kernel ~150x off the floor (54x
  TPOT gap) with zero boot-time signal.
- `sampling.py` — **no vLLM, no CUDA**. Pure-numpy token sampling (`Sampler`: greedy / temperature / top-k / top-p) +
  `apply_chat_template` (delegates to the HF tokenizer). Used by the standalone **generation oracle**
  (`commands/generate.py`) — `emmy generate`'s host loop re-runs the whole fp16 prefix each step on the CUDA
  backend and samples with this. The generative *vLLM plugin* (`EmmyGenModel`) builds on this oracle.
- `twins.py` — **weight-free serving-twin capture**. Builds a trimmed random-init skeleton from a model's
  `config.json` alone (no checkpoint download — a trace never reads a weight value; `layer_types` collapses to one
  local + one `full_attention` layer, the vocab shrinks to a stub) and traces the `pre`/`post` twins through the same
  `build_attention_split_wrapper` / `trace_split` path serving uses. Backs `emmy eval golden --in-model` and the
  golden drift CI gate; `scripts/capture_gen_twins.py` remains the full-checkpoint capture for tuning.
- `gen_runner.py` — `EmmyGenRunner` (Phase 2; sibling to `EmmyForwardRunner`). Carves SDPA out of every
  decoder layer (`build_attention_split_wrapper`; Gemma-nano PLE blocks — `hidden_size_per_layer_input` — are
  rejected loudly there: the carve has no seam for the `per_layer_input` multiply), compiles **two dynamic-`num_tokens` programs per layer** (`pre` +
  `post`) over the flattened `[num_tokens, H]` layout, and exposes `embed` (Gemma's √hidden embed-scale folded into the
  gather table) / `forward_layer_pre(L,…)→(q,k,v)` (un-rotated 2-D seam; carves q/k/**v**-norm, and Gemma-4's global
  `attention_k_eq_v` where V reuses K's projection) / `forward_layer_post(L, attn_out, residual)→hidden` / `final_norm`.
  Attention dims are **per layer** (`layer_meta(L)` → head_dim / num_heads / num_kv / scaling) — Gemma-4's global layers
  use a larger `global_head_dim` than its sliding ones, so each layer's `pre`/`post` compiles at its own width. The caller stitches between
  `pre` and `post` (a reference torch SDPA in the Phase-2 host stitch; vLLM paged `Attention` in Phase 3). **I/O:**
  the plugin runs **device-resident at every width**: the **decode hot path** (`num_tokens ≤ bucket`) rides the
  captured static twins (`run_device` — captured-replay, torch↔cupy DLPack zero-copy), a **FULL chunked-prefill
  step** rides the static **prefill-chunk twin** (`num_tokens == prefill_bucket` EXACTLY — default = the dynamic-dim
  cap, `EMMY_GEN_PREFILL_BUCKET` overrides / `0` disables; exact grids on the hot chunk width. The boundary is
  equality, not a range: the twin always computes `prefill_bucket` rows, so an over-bucket decode batch or a partial
  tail chunk routed through it would pay the full-bucket grids for a sliver of real rows), a **rider-carrying full
  chunk step** (`prefill_bucket < num_tokens ≤ prefill_bucket + rider_width`, `rider_width` = the decode bucket when
  both twin families exist) **splits row-wise** across the chunk twin + the decode twin — correct because pre/post are
  per-token-independent; since A3 both halves copy once into slices of one persistent shared joint destination
  (`_rider_dest`, cached per column width for gemma-4's heterogeneous layers) instead of clone + `torch.cat`,
  halving the rider seam bytes and removing the per-layer allocation churn — the caller must consume rider
  outputs before its next rider call (attention does, within the layer) — which is what lets `--max-num-batched-tokens` default to `DYNAMIC_DIM_MAX + bucket`: a full
  chunk step keeps carrying its decode riders and the previous prompt's 1-token BOS tail instead of freezing every
  decoding request for the whole chunk and deferring first-token sampling (the measured c=4/c=8 TTFT structure), and
  every other width rides the SYMBOLIC programs' `run_device_sym`
  (`num_tokens ≤ prefill_capacity`, capacity = `min(max_num_batched_tokens, DYNAMIC_DIM_MAX)`, passed as `max_tokens`
  at runner build) — grids sized per step via
  `set_sym_values` over capacity-built buffers, launches issued on torch's stream, no per-T graph capture
  (chunked-prefill T varies per step; the dispatch hides behind prefill-width GPU work). The per-layer host numpy
  `rebind` path survives only for the standalone `emmy generate` oracle and as the over-capacity fallback — its
  ~2-per-layer `.cpu()` hops per prefill step were the TTFT wall. **Decode bucket:** it
  also compiles a **static M=`decode_bucket` (default 16)** `pre`/`post` twin per layer and uses it when
  `num_tokens ≤ bucket` (pad → run → slice the real rows) — the symbolic hint-512 M-tile is ~66× too slow at decode
  M=1; falls back to symbolic above the bucket or if a static compile fails. So
  up to 4 capacity programs/layer — a real memory-budget risk for the activation buffers, though the twin's
  **weights are shared**: `_compile_split` binds constants through a per-wrapper device cache
  (`_bind_device_constants` — one upload per `(source_path, load_ops)`, the same cupy array fed to both builds), so
  the decode twin adds no weight copy. **`EMMY_GEN_DECODE_BUCKET=0`** (`config.gen_decode_bucket`) still disables the
  twin entirely at the cost of decode speed. A further static **M=1** twin pair (`EMMY_GEN_M1_TIER`, default on)
  routes true single-token decode onto gemv-class matvec programs, and `EMMY_GEN_ALIAS_ATTN` lets
  vLLM's paged attention write directly into the post program's `attn_out` input backing — since A4 for EVERY
  tier (`post_attn_backing` routes rows exactly like `forward_layer_post_device`: M=1, decode bucket, exact
  chunk, symbolic; rider widths return None — one contiguous attention output cannot alias two programs'
  buffers) — so the prefix upload self-copy-skips on pointer equality, dropping the attention→post seam copy
  from captured decode graphs and eager chunk steps alike.
  **Post→pre chaining covers EVERY program family** (decode twins, M=1, symbolic, prefill-chunk — the vLLM
  integration plan's Milestone A2): each family's post OUTPUT array is rewired at build onto its pre twins'
  shared hidden-INPUT backing, so the between-layer upload self-copy-skips on pointer equality. The skip pays
  where the caller holds a VIEW of the post output — under an outer capture (the no-clone branch) — so the seam
  copy node drops from captured over-bucket sym decode graphs; EAGER steps still clone (the pointer-breaker), so
  the chunk-twin chaining activates only once chunk steps are whole-step captured (recorded future work). Safe
  because the residual upload copies the previous hidden
  out of the backing before post overwrites it, rider steps (all tiers alias one backing base) are eager by
  construction with the chunk head CLONED before the decode tail runs, and the host `rebind` path — which
  re-takes arena views and unwinds the rewire — is never mixed with the device path on one runner (the oracle
  and the device server are separate runners; `tests/serving/test_gen_prefill_device_gpu.py` pins both the
  pointers and the two-phase discipline).
  **Multimodal wrappers:** the trunk is resolved through `language_model` (gemma-4 "unified" nests the decoder stack +
  embed/norm there) and the text dims come from `config.text_config`.
  **MoE third seam (token-choice top-k, e.g. OLMoE / gpt-oss):** a layer whose `mlp` exposes the transformers-v5
  experts interface (`moe_block_parts` — a router module named `gate` or `router` beside 3-D `gate_up_proj` /
  `down_proj` parameters) is carved by `build_moe_split_wrapper` instead: its post program (`post_attn`) computes
  o_proj + residual + post-attention norm and returns BOTH `h` and the normed `xn` (the norm output must materialize
  at the seam — it is shared across the k routed experts, so the norm→gate⊗up fused form is out of reach by design),
  and the layer output is assembled in torch (`_moe_combine`): the HF router module (linear + softmax + top-k — ops
  the tracer cannot map) runs on `xn`, each HIT expert launches ONE shared expert program (`moe.expert.sym`:
  `expert(x, w_gate_up, w_down[, b_gate_up, b_down])` with the weights — and gpt-oss's per-expert biases — as
  forward args → graph INPUTS, fed per-expert dim-0 slices of the E-stacked tensors, which live on device beside
  the routers via `_ensure_device` under the per-layer `inputs` map), and the partials weighted-`index_add_` into
  `h`. The expert layout (orientation / interleave / bias — gpt-oss vs OLMoE, incl. the clamped-SwiGLU spelling and
  the de-interleave-at-load contract) is the trace ARCHITECTURE's `moe_expert_layout` story; the runner just feeds
  named inputs. Program count is 2/layer + ONE expert program total — not `E`/layer. MoE layers are
  device-resident only (`forward_layer_post` raises on the host path) and are excluded from post→pre chaining (two
  outputs; the layer output is a fresh torch tensor). The ROUTED dispatch host-syncs (`indices.unique().tolist()`),
  which a whole-step decode capture cannot record — but single-token decode is capture-legal through the fixed-slot
  tier below, so `_is_moe_model` in `emmy/commands/serve.py` (local-config probe, UX only) has `_gen_graph_args`
  emit a FULL_DECODE_ONLY compilation-config with the capture ladder capped at size 1 instead of forcing
  `--enforce-eager`, and `EmmyGenModel.__init__` validates authoritatively against the runner: an MoE capture boot
  is rejected loudly when the fixed-slot tier is unavailable or any capture size exceeds 1 (serve with
  `--enforce-eager` then).
  **Expert tiers (decode + prefill perf lanes):** the expert program comes in four tiers mirroring the main ladder —
  static M=1 (`moe.expert.one`), static M=decode-bucket (`moe.expert.bucket`, pad → run → slice), static M=256
  (`moe.expert.m256` — the prefill twin at the mean per-expert chunk width T·k/E, serving routed row sets in
  (bucket, 256]), and the symbolic fallback for anything wider — routed per HIT expert by its routed row count.
  Per-launch framing, not weight bytes, was the measured decode wall (~0.23 ms/launch through the per-call symbolic
  path), so `_moe_combine` hoists the GPU lock + the cupy stream bind around the whole per-expert loop and
  `_launch_expert` issues bare `upload_prefix_device` + `run_once` calls; the per-expert cupy weight views are
  minted once at `_ensure_device`. A tier whose schedule stages no operand through a TMA descriptor (descriptors
  bake pointers at build) takes the weight slices by POINTER SWAP into `program.arrays` — no D2D weight copy;
  descriptor-bearing tiers upload the slices normally. The M=256 twin instead replays its captured whole-program
  graph per expert (`capture_program_graph` — one host call instead of per-kernel Python framing; prefill FFN was
  launch-bound at ~3×), which bakes the twin's own buffer pointers — so its weights always arrive by UPLOAD, never
  by pointer swap (a swap would freeze the first expert's slices into every later replay).
  **Fixed-slot decode tier (T=1, capture-legal):** k slot INSTANCES of the INDIRECT M=1 expert twin
  (`moe.expert.one.ind`) are built at boot — the same expert graph compiled with `w_gate_up`/`w_down` marked as
  **indirect operands** (`_compile_split(indirect_inputs=...)` → the `cuda.indirect_inputs` graph hint; the ABI
  change is invisible to the schedule search, see the plan section of `emmy/compiler/backend/ARCHITECTURE.md`).
  Each kernel resolves its weight base pointer in-kernel as `table[sel[slot]]`: `_ensure_device` builds one flat
  `[n_moe_layers · E]` int64 pointer table per weight kind (entry = the device address of one expert's slice of a
  layer's 3-D tensor) plus a persistent `[k]` int32 selector and a `[k, H]` partials tensor; slot `j`'s plan
  stamps `slot=j` (`_plan_with_slot`), and its table/selector/output arrays bind once at boot. The table is
  layer-invariant — the layer offset rides in the selector VALUES — so per-slot cached graphs and the whole-step
  capture both bake fixed pointers. `_moe_combine_slots` then runs the router, writes `indices + sel_off` into
  the selector (one fixed-shape device write — no `unique()`/`.tolist()`/host sync, no per-step weight staging;
  the kernels read the E-tensors directly), replays the k slots on the row, and combines the partials with one
  score matmul, so vLLM's whole-step decode graph records it (under the outer capture each slot issues its raw
  launch sequence, mirroring `run_device`). A schedule that stages an indirect operand through a TMA descriptor
  fails the compile loudly (descriptors bake the base address at encode) and single-token decode keeps the routed
  path (eager); a stale pre-indirect pack plan fails the `_indirect_covered` check and recompiles — no half-hit.
  Wider decode steps and prefill always ride the routed dispatch; `combine_routed_experts` stays the parity
  oracle (`test_gen_runner_gpu` fixed-slot-vs-routed + the direct-vs-indirect bit-exactness pin,
  `test_gen_capture_gpu` captured-step live-replay).
  **Per-input feed dtypes:** `_compile_split` binds every plan input at its own traced dtype rather than one blanket
  model dtype — an f8-dtype input binds the raw bit pattern on the uint8 fp8 bits carrier (a torch fp8 example tensor
  reinterprets via `.view`), a scale input keeps its traced f32. This is what lets input-sourced fp8 expert weights
  (`loader.quant.spell_quantized_inputs`, see the compiler ARCHITECTURE's quantized-checkpoints section) feed the
  expert programs; indirect operands compose (bits + scale slices both table-resolved).
  **Quantized-checkpoint serving load (gpt-oss fp8, MoE M3):** `EmmyGenRunner.create` detects a quantized checkpoint
  (`quantized_checkpoint_dir`) and takes `load_quantized_split` (trace ARCHITECTURE): config-built META twin,
  dense trunk shard-streamed in as real values, expert tensors kept fp8 as a per-layer store of program-input-named
  tensors (bits on the uint8 carrier + f32 scales + `dtype` biases, gate/up de-interleaved). `from_model` then
  compiles the expert programs with `quant_specs` — `_compile_split` traces the wrapper at value dtypes (the trace
  is quantization-blind), applies `spell_quantized_inputs` post-trace (bits input + appended scale inputs + the
  in-graph decode/scale algebra the W8A16 binding absorbs), and the caller's example list carries the scale
  examples as its last entries. Every per-expert input — weights, biases, scales — is a per-launch slice of the
  store's E-stacked device tensors; the fixed-slot tier builds one pointer table per input kind, so the k-slot
  T=1 dispatch stays capture-legal with fp8 experts. VRAM: ~19 GB expert bits + dense fp16 + tables on the 32 GB
  card, the rest is vLLM's KV budget.
  **gpt-oss attention (sinks + SWA-128 + YaRN), all vLLM-side:** `EmmyGenModel` creates a per-layer `sinks`
  `nn.Parameter` (`[num_heads]`; keyed on `model_type == "gpt_oss"` — the config carries no flag) and passes
  `sinks=` into each `Attention`, which makes vLLM's backend selection sinks-aware (sm_120 lands on TRITON_ATTN;
  FA2/FlashInfer reject sinks there); `load_weights` claims `*.self_attn.sinks` beside `lm_head.weight` (untied
  embeddings — no embed adoption; the runner owns the embed table from the checkpoint). The alternating
  sliding/full pattern rides the existing `_layer_window` (`layer_types`), and YaRN flows through the flat
  `config.rope_parameters` into `_build_rotary`/`get_rope` unchanged — one nuance: stock vLLM builds the gpt-oss
  rope at fp32 while the plugin's rotary follows the model dtype (fp16); q/k re-cast to the trunk dtype either
  way, a known small numeric drift.
  **Tuning what serving actually runs.** The deploy pick reads the golden tier, then box-local `perf`/reservoir
  evidence — and only evidence recorded against the *serving graph* carries serving. An isolated golden snippet does
  not: fusion inside a real block produces a different graph (`F.rms_norm(x) @ w` binds a cone the in-model op does
  not). So the evidence path is the **twins** — the `pre`/`post` graphs at each width, captured by
  `scripts/capture_gen_twins.py` (which calls `gen_runner.trace_split`, the same trace `_compile_split` makes, so the
  captured JSON is byte-for-byte what serving compiles) and then fed to `emmy tune` with `EMMY_TUNE_DB` /
  `EMMY_ONLINE_FILE` pointed at a twin-local DB. Capture a **global** (`full_attention`) layer alongside the sliding
  one for any model whose layers are not homogeneous — gemma-4's global layers carry a larger `head_dim`, so their
  projections are different shapes with different optimal configs. Re-capture whenever a tracer/recognizer change
  alters the graphs: the DB's evidence is keyed to structural signatures, and stale evidence applied to new graphs
  serves worse than either coherent state. Whether the *recorded goldens* still deploy against the current twins is
  checked continuously: `emmy eval golden --in-model` re-traces them weight-free (`twins.py`) and audits per fork
  (MATCH / DRIFT / GAP), and `tests/compiler/test_golden_drift_gate.py` runs the same audit in CI.

  > **Memory budget (measured, gemma-4-12B / 32 GB RTX 5090).** The two artifacts that made the 12B need ~2–3× stock
  > vLLM's memory (it only fit at `ctx 256` with the decode twin off) are both fixed:
  > 1. ~~the decode twin binds a second copy of every layer's weights~~ **fixed** — the symbolic and decode programs
  >    share one device buffer per weight via the per-wrapper constant cache (see decode bucket above);
  > 2. ~~every layer's program retains its own capacity-sized activation buffers~~ (~350 MB/layer ⇒ ~17 GB across
  >    48 layers at `max_num_batched_tokens=4096`) **fixed** — every program the runner builds shares one
  >    `BufferArena` (`backend/cuda/program.py`): input/output buffers and the scratch slab are views into per-key
  >    grow-only backings, so all layers hold ~one layer's worth. Safe because layers run sequentially and each
  >    program's outputs are host-copied/cloned before the next program runs; a backing that grows (e.g. gemma-4's
  >    wider global layers, or a bigger prefill `T`) leaves earlier generations alive so captured graphs / TMA
  >    descriptors never dangle. The re-validation of the 12B footprint on a real 5090 is pending (Phase-A exit run).
- `vllm_model_gen.py` — `EmmyGenModel` (Phase 3; the generative vLLM model class; Qwen3 / Llama / **Gemma-3/4**).
  Resolves `mc.hf_config` through **`text_config`** first (gemma-4's multimodal "unified" checkpoint nests every text
  attribute — `layer_types` / `rope_parameters` / `sliding_window` / vocab+hidden size / `final_logit_softcapping` —
  under it), and accepts an `*embed_tokens.weight` alias for the tied `lm_head` (that checkpoint nests the embedding at
  `model.language_model.embed_tokens.weight`).
  **NOT** `IsAttentionFree`: it builds real vLLM `Attention` layers (one per decoder layer, unique `prefix` → vLLM
  allocates a KV-cache spec and runs paged attention; each is built at its **per-layer** dims (`runner.layer_meta` —
  Gemma-4 global layers use a larger head_dim) and gets `per_layer_sliding_window` so Gemma's sliding/global layers
  window correctly) + one RoPE module **per layer** (`_build_rotaries`: homogeneous models share one; Gemma-3/4
  keys theta AND head_dim on layer type — local vs global — a bare `Attention` does no RoPE) + `ParallelLMHead` + `LogitsProcessor`
  (`soft_cap=final_logit_softcapping`, so Gemma-4's final-logit softcap applies; `compute_logits` also -infs the
  generation config's `suppress_tokens` — gemma-4 lists the mm delimiter tokens `<image|>`/`<audio|>` there, HF
  generate and stock vLLM honor the list, and a degenerate text prompt can genuinely rank one top-1, which would
  decode to empty output). The trunk compute (embed + per-layer
  pre/post + final norm) is the `EmmyGenRunner`; vLLM owns only `lm_head` (`load_weights` claims `lm_head.weight`, or the
  tied embed alias). On a tied checkpoint `load_weights` then hands the loaded head to the runner
  (`adopt_embed_table`) so the runner drops its own ~2 GiB folded device copy and gathers from the SHARED raw table
  (the gemma embed-scale re-applies at gather in fp32 — the head must read the table unscaled), and releases the freed
  torch/cupy blocks to the driver **before vLLM's KV-cache profiling** — the reclaimed memory becomes KV blocks
  (gemma-4-12B on a 5090: 17.7k → 27.5k KV tokens, the difference between admission-queueing and beating stock TTFT
  on the 4K/4K c=8 workload). `forward` brackets each `self.attn[L](q,k,v)` with two emmy replays (pre/post), applying that
  layer's RoPE in between (A2). Uniform sliding-window (Qwen2-style `use_sliding_window`) and dual-chunk are rejected.
  **Speculative decoding (MTP drafter, vllm#41745 — gemma-4-assistant).** vLLM's drafter shares the target's embedding
  by reaching into `target.model.embed_tokens` (stock model classes nest their trunk under `.model`; the emmy trunk lives
  in the runner instead). So — only when a draft is configured — the model exposes a thin `.model` shim
  (`_EmmyTargetInner`) whose `embed_tokens` (`_SharedRawEmbedding`) gathers RAW rows from the SAME tied embed/`lm_head`
  tensor the runner adopts (no copy); the gemma-4 drafter applies its own `sqrt(hidden)` normalizer, matching the
  runner's folded embed-scale. The KV cache is genuinely shared through vLLM's own attention-layer registry — the
  drafter's Q-only layers read K/V from the target's real `Attention` layers. Only tied-embedding targets are sound
  (untied `lm_head` ≠ embedding), so an untied target is rejected at init; a one-time check at the end of `load_weights`
  verifies the runner adopted that same tensor (not in the gather itself — vLLM compiles the drafter's forward, and
  dynamo can't trace `data_ptr()`). `forward` branches on `num_tokens`: the decode hot
  path (`≤ bucket`) runs `_forward_device` (q/k/v + attn_out stay CUDA tensors through RoPE + attention, no host
  hop); prefill keeps the numpy path. Select via `--runner generate` +
  `--hf-overrides '{"architectures":["EmmyGenModel"]}'` + `--dtype float16` (the `serve --generate` branch forces
  this for seam coherence). Registered in `__init__.py`. **Whole-step decode CUDA graphs are the `emmy serve
  --generate` DEFAULT**: no `--enforce-eager`; instead a `--compilation-config` with `cudagraph_mode:
  FULL_DECODE_ONLY` (full cudagraphs need no torch.compile — vLLM wraps the model in its `CUDAGraphWrapper`) and
  `cudagraph_capture_sizes` laddered up to `--max-num-seqs` (sizes at or below the decode bucket run the static
  decode twin; sizes above it capture the device-resident symbolic programs — both paths are capture-validated,
  `test_gen_capture_gpu` / the two-size live-replay test, and BOTH drop their output clones under the outer
  capture (`run_device_sym` mirrors `run_device`'s captured no-clone branch — the graph's fixed kernel order
  makes the views safe, and the per-layer clone D2D nodes leave the captured graph; the uncaptured paths keep
  the clone); over-bucket capture was worth +10.6% req/s at c=64,
  and a decode bucket matched to the concurrency beats riding the symbolic captures — the bucket-64 golden set
  took c=64 TPOT 35.4 → 22.5 ms, and the bucket-8 set (m8 goldens, 2026-07-25) took c=4 TPOT 19.6 → 18.9 and
  c=8 21.4 → 20.6 on the same per-lane-knob rule). The mixed prefill+decode cells take a second per-workload
  knob, the CHUNK QUANTUM: `EMMY_GEN_PREFILL_BUCKET=2048` + `--max-num-batched-tokens 2056` (chunk width +
  bucket-sized rider headroom; m2048 goldens 2026-07-26) pipelines queued prompts in ~200 ms static-twin steps
  instead of ~500 ms 4096-chunks — c=4 TTFT 1363 → 1063 and c=8 1828 → 1014 on the 5090, both below stock —
  and as a side effect the smaller profiling dummy peak restores KV capacity to stock parity (38k vs 24k
  tokens at util 0.96). Under the
  outer capture,
  `_Program.run_device` detects `torch.cuda.is_current_stream_capturing()` and issues the raw launch sequence
  (`run_once`) instead of its own graph machinery — nested stream capture and graph launch are both illegal in a
  capturing stream — so the whole decode step (embed + 48× pre/RoPE/paged-attention/post + final norm) records
  into ONE vLLM graph and the ~2-per-layer host launches vanish at replay. Opt out with vLLM's own
  `--enforce-eager` (forwards untouched; also forced automatically when `EMMY_GEN_DECODE_BUCKET=0` — nothing is
  capturable then); a caller-supplied `--compilation-config` wins over the default. Prefill and
  over-bucket decode batches stay eager under FULL_DECODE_ONLY by construction.

**The capture ladder under speculative decoding.** vLLM rounds every requested capture size UP to a multiple of
`query_len = num_speculative_tokens + 1` (`CompilationConfig.adjust_cudagraph_sizes_for_spec_decode`, guarding vLLM
issue #28207), then pads each step to the first captured size at or above its width — so the runner routes on the
PADDED width, never the real one. A sparse power-of-two ladder loses every rung to that rounding whenever `query_len`
is not itself a power of two: at depth 2 (`query_len` 3) the `16` and `32` rungs become `18` and `33`, one rung ABOVE
the decode bucket each existed to serve, and every steady-state verify step then misses its static twin and runs the
symbolic masked-tile program. The cost is kernel quality, not launch overhead — vLLM still captures the whole step at
the padded size, so nothing runs eager. The ladder is therefore built from DENSE candidates (mirroring vLLM's stride-8
default), each FLOORED to a multiple of `query_len`: flooring only moves a rung down, so the bucket's own rung stays
reachable and vLLM's round-up becomes a no-op, while density removes the leftover padding. The invariant the tests
assert, parametrised over depths 1/2/3/5: **for every reachable verify width, the first rung at or above it is still at
or below the decode bucket.** Overshoot WITHIN the bucket is nearly free — doubling the padded rung costs under 1% of
TPOT, since a decode step reads the weights once regardless of width — so a bucket should be chosen for kernel
coverage at that width, not for tightness against the verify width. Overshoot PAST the bucket is the fatal case.
`gen_runner._warn_symbolic_decode` reports once per width when a decode-shaped step misses the twin: the twins audit
only covers widths it is handed, so without that line a rung one step above the bucket looks like ordinary symbolic
traffic.

## Batched modes — `EMMY_SERVING_BATCHED=1` (symbolic seq) and `EMMY_SERVING_STATIC=1` (static extents)

The default path is symbolic-seq, **one sequence per forward** (`batch_cap = 1`). Two opt-ins run a whole scheduler
step as ONE padded batched forward instead (`runner.forward_hidden_states_batched`). In both, the batch cap is
**vLLM's own `max_num_seqs`** (`vllm_config.scheduler_config.max_num_seqs`), read at init — the batch is sized by the
standard `--max-num-seqs` flag, not a separate emmy knob (the toggle is boolean; the size comes from what vLLM hands
us). Mind the default `max_num_seqs=256`: the buffer set allocates at `(max_num_seqs, max_seq_len)` capacity, so pair
either opt-in with a sane `--max-num-seqs` and a workload-sized `--max-model-len`.

- **`EMMY_SERVING_BATCHED=1` — the preferred batched mode.** The trace keeps `seq_len` symbolic with the batch extent
  baked at the cap; each step pads only to the **step's longest sequence** (not `max_seq_len`), sizes the grids to it
  (`set_sym_values`), and replays the captured graph for that seq_len (one capture per distinct S, the same LRU cache
  as the per-sequence path). Uniform-length steps pad nothing; mixed steps waste only the intra-step length spread —
  the remaining padding waste is what the cu_seqlens varlen work (follow-up #1) removes.
- **`EMMY_SERVING_STATIC=1`** builds ONE fully-static `(N, max_seq_len)` program (static extents on both axes) and
  pads every step to `(N, max_seq_len)`. Zero waste only when all requests are exactly `max_seq_len`; kept as the
  fixed-shape stand-in (static traces need no masked tiles at all). Takes precedence if both vars are set.

Padding is safe in both: causal masking makes a row's real prefix independent of its right-padding (a token attends
only to earlier positions), and dummy rows below the cap are never read out. Historically the symbolic-seq kernels
miscomputed batch>1, which is why only the static mode existed; the root cause was the serving capacity-buffer path's
**TMA descriptors baking allocation-shaped strides** (a batch axis above the symbolic seq axis has a `seq_len`-dependent
global stride, so batch row 0 was right and every higher row read shifted garbage — invisible at `batch_cap = 1`).
Fixed in `backend/cuda/program.py`: symbolic-src descriptors re-encode at the RESOLVED shape per sym key, cached
beside the per-S graph cache (`_descs_now`); `tests/serving/test_runner_batched_gpu.py` pins both modes per row
against eager, and the batch {2, 4, 32} × seq matrix in `tests/compiler/ir/test_dynamic_shapes.py` pins the kernels.

## Execution model (v2: captured graphs) and its known costs

Each sequence runs **individually** through the compiled dynamic-seq_len program (batch axis is compile-time fixed at
1), as a captured whole-program CUDA graph (one host launch) replayed over a single capacity-sized buffer set, with the
request's torch inputs bridged to the buffers' prefix device-to-device (dlpack, no host hop) and the output handed back
as a torch view of the output buffer. One captured graph is cached per
distinct seq_len (bounded LRU); a new length pays one capture (~one forward) on first sight, then replays. The captured
graph removes the per-launch dispatch overhead and the ~hundreds of host calls the uncaptured loop made — the
precondition for fast low-concurrency serving. (Measured A/B is ~flat today: the uncaptured `run_once` loop already
queues launches async, so the Python dispatch overlaps GPU execution and stays hidden while the symbolic kernels are
slow — 0.6B at S=32 ≈ 32 ms GPU. The win materializes once those kernels are fast enough that dispatch stops hiding;
the captured path is in place for that.) Low-concurrency latency is representative; high-concurrency throughput
structurally trails stock vLLM's packed-batch prefill — that gap measures the integration, not kernel quality.
Recorded follow-ups, in impact order:

1. **Packed-varlen attention** (cu_seqlens-aware SDPA tiles) — run vLLM's whole packed batch in one launch at *mixed*
   lengths; the general form of the throughput fix. At concurrency 1 (no batching) emmy is ~1.5× stock; the rest of
   the concurrency-32 gap is batching. Step (a) — batch-correct masked tiles + the symbolic-seq batched program — is
   **done** (`EMMY_SERVING_BATCHED`, see the batched-modes section above); remaining is (b) cu_seqlens varlen tiles so
   one launch handles mixed lengths with no padding at all (its own session — the ragged row→sequence mapping in the
   flash schedule + the mask derivation from `cu_seqlens`).
2. **dlpack zero-copy I/O** — **done**: `forward_hidden_states` takes/returns torch CUDA tensors, bridged to the cupy
   buffers via `cp.from_dlpack` / `torch.from_dlpack` on torch's stream — no GPU↔host round-trip (`upload_prefix_device`
   / `output_prefix_device`). The only residual host touch is `positions.cpu()` for span boundaries.
3. **Device-side causal mask** — host build + upload **removed**: the `(1,1,S,S)` mask is now built once per S as a cupy
   array on the GPU (`runner._mask`) and copied into the prefix device-to-device. Still open: an in-kernel `j <= i`
   predicate would drop the mask input + its per-request D2D copy entirely.
4. **Single capacity-baked graph** — would collapse the per-S cache to one graph, but needs every symbolic-M kernel to
   be correct at an oversized (capacity) grid; today several aren't (swizzle decode + staged loads read OOB). Future
   work.

## Serving constraints

- `--max-model-len` ≤ `DYNAMIC_DIM_MAX` (4096, `compiler/trace/dynamic.py`) — the runner raises at startup otherwise.
  Qwen3-Embedding natively supports 32k; raising the cap means re-examining the `torch.export.Dim` bounds and the
  rotary buffer (`_SlicedRotary` precomputes `DYNAMIC_DIM_MAX + 1` positions).
- `--enforce-eager`: the **embedding** plugin still serves eager — vLLM never torch.compiles an undecorated OOT
  class, and enforce-eager keeps the engine from capturing around the runner's own kernel launches. The
  **generative** path no longer needs it: `run_device` is capture-aware and `serve --generate` defaults to
  whole-step decode graphs (see `gen_runner.py` above).
- Startup compiles the whole model (~1–2 min for 0.6B warm-cubin-cache; first boot pays nvcc). `EMMY_CUBIN_CACHE`
  persistence across container restarts is what keeps reboots fast. **`EMMY_PACK_DIR`** cuts the rest of the warm
  boot: `EmmyForwardRunner.create` keys an execution-plan pack (`compiler/backend/pack.py`) on model id + config
  hash + serving shape — a hit loads binary-keyed plans (`CompiledProgram.build_from_plan`) and skips trace, pass
  pipeline, fork resolution, and codegen entirely (weights still bind from the checkpoint via the plan's
  `source_path` refs); a miss compiles in full and writes the pack for the next boot. Any mismatch — retune under
  a different config, nvcc/toolkit change, evicted cubin — silently falls back to the full compile.
- The shared buffer set is allocated at `max_seq_len` (`--max-model-len`); every accepted request (S ≤ `max_seq_len`)
  uses the captured-graph path. The S²-attention scratch dominates that allocation (0.6B at 4096 ≈ 15 GB), so lower
  `--max-model-len` for bigger models / smaller cards.
- vLLM's memory profiler only sees torch allocations; the runner's cupy-held weights/activations are invisible to it.
  Leave `--gpu-memory-utilization` headroom accordingly (the attention-free model needs no KV cache, so vLLM's own
  budget is tiny). The GENERATIVE arm has the opposite problem — it needs a real KV cache, and vLLM budgets
  `util × total − currently-used`, so the default 0.90 line can fall below the emmy residents and fail the
  min-KV fit at long `--max-model-len` (gemma-4-12B at mml 8448: 1.37 GiB left of the 1.7 needed). `emmy serve
  --generate` therefore defaults the emmy arm to `--gpu-memory-utilization 0.97` (stock keeps 0.90; an explicit
  flag wins).

## Quantized KV — `--kv-cache-dtype fp8_e4m3` (generative)

emmy owns no KV cache. The generative carve runs vLLM's paged `Attention` (and its cache) between the `pre` and
`post` programs, so the cache dtype is entirely vLLM's: `--kv-cache-dtype fp8_e4m3` is an ordinary passthrough flag
on both arms of `emmy serve --generate` (nothing in `commands/serve.py` reads it), and it **doubles the KV token
capacity** out of the same byte budget — the emmy arm's `--gpu-memory-utilization 0.97` needs no adjustment, since
fp8 halves the bytes per token rather than changing what the budget is. Measured on `Qwen/Qwen3-0.6B` at
`--max-model-len 4096`, 32 GB RTX 5090: 264 048 tokens out of 28.2 GiB (fp16) → 518 224 out of 27.68 GiB (fp8), i.e.
exactly 2x per byte. Decode cost is unchanged (TPOT 2.85 → 2.70 ms median at concurrency 4).

The one emmy-side piece of glue is in `EmmyGenModel._attn_aliased` (the A4 alias tail, which reimplements
`Attention.forward` minus its output allocation). vLLM's own `forward` has two quantization steps around the
backend call, and the alias tail must agree with it or fall back:

- **query quantization** (`attn.query_quant`, built when `kv_cache_dtype` starts with `fp8` **and** the selected
  backend advertises `supports_quant_query_input`). Whether it fires is a BACKEND property, not an fp8 property:
  FlashAttention and Triton say yes on any CUDA device, while FlashInfer — what vLLM auto-selects for fp8 KV on
  sm_120, FlashAttention having dropped out of the candidate list — additionally requires TRTLLM attention, i.e.
  SM100. So a 5090 deployment never reaches this, and an SM100 or `VLLM_ATTENTION_BACKEND=TRITON_ATTN` one reaches
  it on every layer of every step. The tail replicates the quantization rather than bailing (bailing would cost the
  alias entirely on those backends). The attention OUTPUT keeps the PRE-quant query dtype, which is what the post
  twin's `attn_out` backing already is — so the backing's dtype check runs against `q` before the quantization, not
  after.
- **`calculate_kv_scales`** (`--calculate-kv-scales`, deprecated in vLLM 0.23 and off by default) stays a bail: the
  flag is one-shot and self-clearing, so the fallback module call computes the scales, clears it, and the alias
  resumes from the next step. Without it — and without scales in the checkpoint — q/k/v scales are simply **1.0**,
  i.e. the cache stores fp8 values of k/v directly.

One operational note the A/B surfaced: `--gpu-memory-utilization 0.97` demands ~30.4 of the 5090's 31.3 GiB free at
startup, so *any* other process on the card (a sibling kernel A/B holding 0.5 GiB was enough) fails the boot on
vLLM's free-memory check. Nothing to do with fp8 — but an fp8-KV deployment sized against a full card inherits it.

## Device footprint sets admission capacity (generative)

The generative arm's throughput on long-request batched workloads is set by how many sequences vLLM can
admit, and that is decided by emmy's device footprint rather than by kernel speed.

**Mechanism.** The runner holds its trunk weights, activation arenas and scratch slabs in **cupy**
buffers. vLLM's memory profiler only measures torch allocations, so it never attributes any of them; it
budgets the KV cache as `util × total − currently-used`, i.e. out of whatever is left after emmy has
already claimed its residents. Every byte of emmy's non-KV footprint therefore comes straight out of the
KV cache, and the KV cache is what caps concurrency: a request needing `input + output` tokens of KV
consumes that much of a fixed pool, so a smaller pool admits proportionally fewer streams and queues or
**preempts** the rest (a preempted request recomputes its prefill — wasted work that yields no token).

**Scale of the effect.** Against stock vLLM at the same utilisation on gemma-4-12B, emmy's larger
footprint costs on the order of half a GiB of KV, which on the 4k-in/4k-out batched workload is the
difference between sustaining roughly seven concurrent streams and roughly five and a half. Throughput
tracks that ratio almost exactly. It is not a kernel effect and cannot be recovered by tuning kernels.

**`--gpu-memory-utilization` does not recover it.** Raising it far enough to match stock's KV budget
leaves no headroom for allocations vLLM's profiler does not reserve — under speculative decoding the
rejection sampler's transient buffers then fail at first use — and raising it further exceeds the free
VRAM at startup and refuses to boot. The footprint has to shrink; the knob cannot substitute.

**Where the footprint goes, and the actionable invariant:** the symbolic programs are built at
`capacity = max_tokens` and the prefill twin at `prefill_bucket`, so the buffers are sized by the
*serving shape*, not by the widths a lane actually reaches. A single `[max_tokens, intermediate]` fp16
intermediate is already hundreds of MB. **A lane that only ever schedules `prefill_bucket`-sized chunks
still pays for `max_tokens`-row buffers** — so the first place to look when reclaiming footprint is
`BufferArena` occupancy measured against the widths a deployment can actually reach.

### Reclaiming footprint re-opens the capture-ladder question

These two knobs are coupled, and the coupling is a trap for whoever does the footprint work.

A KV-starved lane admits few streams, so its verify width under speculative decoding
(`streams × query_len`) stays small and lands on a low capture rung — comfortably at or below the decode
bucket, i.e. on the static twin, **even with a ladder that violates the invariant above**. Reclaim the
footprint and concurrency rises; the verify width rises with it; and it can cross the bucket, at which
point every steady-state step silently falls to the symbolic program. Measured at decode-bucket widths,
that path costs roughly **2x** the static twin per step.

**Invariant: any change that raises sustained concurrency must re-check the ladder invariant above for
the widths it newly makes reachable**, and raise the decode bucket (to a width with tuned kernels) or fix
the ladder in the same change. Otherwise the reward for fixing the memory problem is silently losing the
static decode twin.

This applies with particular force to **baked serving images**, which may ship a hand-written
`cudagraph_capture_sizes` list in their entrypoint rather than going through `_gen_graph_args`. Such a
list does not get the flooring treatment described above, so it can violate the invariant while
`emmy serve` on the same revision does not — check the image's entrypoint, not just the code.

## Testing

- `tests/serving/test_packed.py` — pure span-split logic, runs everywhere.
- `tests/serving/test_gen_mtp_shim.py` — the spec-decode `.model.embed_tokens` shim (no GPU): pins the attribute
  contract vLLM's MTP drafter shares off the target, that it gathers RAW rows from the shared tied weight, and that an
  untied target raises. Imports vllm at module level, so it runs where vllm is installed (skips otherwise).
- `tests/serving/test_vllm_plugin_gpu.py` — `perf`-marked (deselected by default), needs CUDA + vllm: in-process
  `vllm.LLM(runner="pooling", hf_overrides=...)` on Qwen3-Embedding-0.6B, `.embed()` cosine vs the HF eager reference.
  The three texts have different token counts, so it exercises the per-seq_len captured-graph cache end to end.
- `tests/compiler/ir/test_dynamic_shapes.py` — the captured-replay primitives directly (RMSNorm + a 1-layer Qwen3
  trunk through `set_sym_values` + `upload_prefix` + `capture_program_graph` + `replay_program_graph` +
  `outputs(sym_values)`); run under `compute-sanitizer` in dev to confirm zero illegal accesses. Plus
  `test_capture_replay_device_io_matches_eager` — the zero-copy device path (`upload_prefix_device` + cupy-in,
  `output_prefix_device` + `torch.from_dlpack`-out) matches eager, the primitive behind the runner's torch I/O.
- `tests/serving/test_runner_batched_gpu.py` — `perf`-marked: a 1-layer static `(batch, S)` trunk wrapped in a runner;
  `forward_hidden_states_batched` runs several different-length sequences in one padded batched forward and matches
  eager per row (the causal-independence-under-padding gate for `EMMY_SERVING_STATIC`).
- `scripts/compare_embeddings.py` — the accuracy gate against a *server*: embeds a fixed text set through two
  OpenAI-compatible endpoints (emmy-backed and stock) and asserts pairwise cosine > 0.99.
