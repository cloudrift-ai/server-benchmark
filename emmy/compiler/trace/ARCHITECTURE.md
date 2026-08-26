# Trace Architecture

Frontend capture: PyTorch/HuggingFace model → `Graph` populated with
Layer-1 frontend ops.

## Modules

### `torch.py` — FX → Graph IR

`trace_module(module, args, kwargs=None) → Graph` runs
`torch.export.export()` under `torch.no_grad()` to get the inference FX graph, then walks each node and
emits the matching frontend op (`LinearOp`, `MatmulOp`, `SdpaOp`,
`ElementwiseOp`, `ReduceOp`, …). `trace_module_with_constants` returns
the graph plus a `placeholder_name → attr_path` dict for resolving
weights/buffers back to their module attributes.

Per-op handlers map aten names (`aten.add.Tensor`,
`aten.linear.default`, …) to dialect op constructors; input shapes are
pulled from the FX meta and fed into the op's `infer_output_shape` to
stamp the output tensor.

Tensor constructors whose receiver supplies only dtype/device metadata (`new_zeros`, `new_full`) lower from a scalar
constant plus an explicit broadcast; the receiver's unrelated shape and values never become operands. Exported
`copy_` is treated functionally as a destination-shaped broadcast/cast of its source. A static, unit-step slice/select
chain rooted at a locally computed tensor additionally reassembles the updated base as a two-source `IndexMapOp`: the
copied value supplies the written region and the previous base supplies the remainder. Rebinding the root's FX name
versions sequential overlapping writes and later aliases built from that name; an empty write leaves that version
unchanged. The written-region predicate starts from a boolean literal, so coordinate ternaries and the source select
retain a boolean condition in vectorized reference evaluation and after Loop IR lifting. A write through an
input/parameter, a dynamic or strided view, a used `copy_` return, or a view created
before the write still fails closed; those forms need general alias versioning rather than this local functional
update. `masked_fill` lowers to ternary `where(mask, fill, self)` so an unselected infinity is preserved
instead of becoming NaN through arithmetic selection. `triu` and `tril` lower to two-source `IndexMapOp` regions over
the last two axes: the selected triangular region reads the input and the complement reads a scalar zero. The
diagonal must be a static integer; tensor-valued or symbolic diagonals fail closed instead of becoming broadcast
elementwise operands.

A static one-dimension `roll` and rank-reducing `select` lower directly to affine `IndexMapOp` regions. An exported
`fill_` is functional through its returned value. If a later live read observes the written storage, a static unit-step
slice/select chain rooted at a local value can also reassemble that base with the filled rectangle. Multidimensional
roll, dynamic or strided views, input/parameter mutation, used mutation returns, and aliases created before the write
fail closed.

Static integer `arange` lowers to the zero-input tensor `RangeOp`, so constant-source replay evaluates one sequence
instead of applying NumPy `arange` elementwise to a broadcast stop. Dynamic and non-integer ranges fail closed.

Static rank-two `eye` lowers to a two-source `IndexMapOp`: output coordinates select a typed scalar one on the
diagonal and a typed scalar zero everywhere else. Square and rectangular dimensions must be static integers and match
the exported tensor metadata. Dynamic dimensions, non-strided layouts, pinned memory, and unsupported overloads or
constructor options fail closed rather than being stored as an elementwise operation without coordinate semantics.

An explicit all-zero `aten.pad` width tuple stays as a unary `ElementwiseOp("pad")` identity so a working golden
retains the frontend provenance and exact dtype. Any nonzero, symbolic, or otherwise unrepresented padding fails
closed: the elementwise form has no coordinate, mode, or fill-value fields and cannot describe a changed tensor.

The default `aten.cumsum` overload with a static integer axis lowers to an additive `ScanOp`, preserving the input
shape and dtype. Dynamic axes, dtype overrides, and unsupported overloads or keyword arguments fail closed.

`aten.chunk` is the deliberate exception to the otherwise single-output frontend: the walker materializes every
FX-described static chunk as its own `SliceOp` and stores a transient tuple of node IDs only while walking FX.
`operator.getitem` resolves an integer tuple index to the matching slice, so no multi-output Graph IR is introduced.
Offsets accumulate the actual FX output extents, preserving PyTorch's uneven/fewer-final-chunks behavior. Dynamic
chunk counts, nonconstant dimensions, a dynamic chunked extent, and invalid tuple indices fail explicitly.

`scaled_dot_product_attention` captures an explicit `attn_mask` tensor (when
present) as a 4th `SdpaOp` input — HF passes its precomputed `(1, 1, S, S)`
causal mask this way (an additive `0` / `-inf` bias) rather than via the
`is_causal` flag. The decomposition (`frontend/decomposition/010_sdpa.py`)
broadcasts that mask to the scores shape and adds it before the softmax.
Dropping it silently turns masked attention into full bidirectional
attention — invisible to uniform input but wrong on any varying sequence.
The `scale=` kwarg is captured onto `SdpaOp.scale` (`None` = torch's `1/sqrt(head_dim)` default) and honored
by both reference backends and the decomposition's scale constant. Gemma-nano (E2B/E4B) passes `scale=1.0` — its q_norm absorbs
the scaling — so dropping the kwarg re-scaled every logit by `1/sqrt(d)` and redistributed the whole softmax.

### `huggingface.py` — trace-friendly wrapper

HuggingFace `CausalLM` models build their causal attention mask
dynamically at forward time (`arange` → `cumsum` → `triu` → `eq` …),
which pollutes the traced FX graph with dozens of mask-construction
ops. One helper cleans this up:

- `build_full_model_wrapper(model, seq_len, dtype) → nn.Module` wraps
  the HF model in a module exposing `forward(input_ids) → logits`.
  Precomputes a `(1, 1, seq_len, seq_len)` causal mask and
  `position_ids` as buffers and monkey-patches HF's internal mask
  builders (`_update_causal_mask` / `_prepare_4d_causal_attention_mask`)
  to return the precomputed mask verbatim.

The wrapper also **replaces the rotary embedding in both modes** — HF's in-graph rotary silently breaks under
`torch.export`: its `inv_freq` buffer is `persistent=False` and doesn't survive export with its real value, so the
traced cos/sin constant-fold to `cos=1, sin=0` and RoPE degenerates to identity. Static mode precomputes cos/sin at
the trace seq_len (`_PassThroughRotary`); dynamic mode precomputes out to `DYNAMIC_DIM_MAX + 1` positions and slices
to the runtime seq_len in-graph (`_SlicedRotary` — the slice end is a SymInt; the `+1` exists because export guards a
symbolic slice end strictly below the sliced extent). Beware that an accuracy check with `input_ids = zeros` cannot
catch a degenerate RoPE or wrong attention scores — identical value rows make the attention output independent of the
attention weights; `tests/compiler/ir/test_dynamic_shapes.py::test_qwen_whole_model_dynamic_compiles_and_matches_eager`
checks with non-zero ids for exactly that reason. The model passed in is **not** restricted to `CausalLM` — wrapping
an `AutoModel` trunk yields hidden states instead of logits (the serving plugin's embedding path, `emmy/serving`).

- `build_layer_wrapper(block, rotary_emb, hidden_size, dtype, layer_type=None) → nn.Module` is the per-layer dynamic
  analogue: `forward(x)` slices precomputed cos/sin buffers (out to `DYNAMIC_DIM_MAX + 1`, same `+1` guard) by
  `x.shape[1]` in-graph and calls `block(x, position_embeddings=(cos, sin))`. The static per-layer trace instead passes
  concrete `(cos, sin)` kwargs — those specialise rotary to the trace seq_len, which is exactly what dynamic mode must
  avoid. Forward arg is named `x`, so the CLI spec is `--dynamic seq_len@x:1`
  (`tests/compiler/ir/test_dynamic_shapes.py::test_qwen_layer_dynamic_compiles_and_matches_eager`).
  Gemma-nano PLE blocks (those exposing `hidden_size_per_layer_input`) additionally get a seeded synthetic
  `per_layer_input` (`build_synthetic_ple`) — the dynamic wrapper registers it as a buffer sliced in-graph like
  cos/sin, and the static single-layer trace (`commands/compile.py`) passes the same buffer as a concrete kwarg
  (without it the trace dies on `FakeTensor * None` inside modeling_gemma4). Kernel shapes and latencies are the
  deployed model's, numerics are synthetic; non-PLE architectures take the unchanged path. The attention-split
  carve (`build_attention_split_wrapper`, serving) instead REJECTS PLE blocks with `NotImplementedError` — it has
  no seam for the `hidden * per_layer_input` multiply and would silently drop it
  (`tests/compiler/trace/test_huggingface.py`).

  Rotary labels normally live on `self_attn.layer_type`. Laguna instead exposes only `self_attn.layer_idx`, so the
  shared `trace_selected_layer` library path derives the label from `config.layer_types[layer_idx]`. The command and
  architecture inventory providers both use that path, including hyper-connection input lanes and required attention
  kwargs. When the label is one of the rotary module's own `layer_types`, the block receives its one `(cos, sin)`
  tuple; models whose rotary keys are independent names (DeepSeek V4's `main` / `compress`) continue to receive the
  complete mapping.

  A static DeepSeek V4 layer receives the same materialized sliding causal mask as the model wrapper. HCA/CSA can
  therefore extend it with the compressor's per-query `block_bias`; tracing with `attention_mask=None` would make that
  bias dead. Any future SDPA form is stamped with the selected attention module's own `sliding_window`, including HCA
  and CSA rather than only `sliding_attention`. At the canonical 512-token width, CSA's 128 compressed entries are all
  selected by `index_topk=512`; the trace retains the installed compressor's KV computation and rebuilds the identical
  causal bias after discarding the value-independent scorer/top-k/scatter tail. This specialization is required for a
  CSA profile and additionally requires the compressor and indexer to enumerate entries at the same rate; a missing
  specialization, a rate mismatch, or a width where top-k is selective fails closed.

- `build_moe_split_wrapper(block) → (pre, post_attn, expert)` is the MoE variant of the attention-split carve
  (token-choice top-k, transformers-v5 experts interface — detected by `moe_block_parts`: a router module named
  `gate` (OLMoE/Qwen lineage) or `router` (gpt-oss) beside 3-D `gate_up_proj` / `down_proj` expert parameters).
  `pre` is the shared q/k/v carve (which also handles OLMoE's FLAT q/k norm placement — norm width == projection
  width, applied before the head reshape — biased projections like gpt-oss's `attention_bias=True`, and rejects
  `clip_qkv` loudly); `post_attn(attn_out, residual) → (h, xn)` stops at the post-attention norm (the router and
  experts consume `xn` outside the graph); `expert(x, w_gate_up, w_down[, b_gate_up, b_down])` takes the weights
  as FORWARD ARGUMENTS so they trace as graph inputs — one compiled program serves every expert via per-expert
  dim-0 slices at launch. Two expert layouts, selected by `moe_expert_layout` off the attributes transformers'
  `@use_experts_implementation` decorator stamps (`is_transposed` / `is_concatenated` / `has_bias` — never shape
  sniffing: gpt-oss `down_proj` is square, so shapes cannot disambiguate the orientation): the OLMoE form
  (`F.linear` on `(E, out, in)` weights, `act_fn`, no bias) and the gpt-oss form (`x @ W + b` on `(E, in, out)`
  weights, per-expert biases as two more forward args, clamped-SwiGLU — `gate.clamp(max=limit)`,
  `up.clamp(±limit)`, `glu = gate·σ(α·gate)`, `(up + 1)·glu` with the module's `alpha`/`limit`). Both spell the
  gate/up split as `chunk(2, dim=-1)`: gpt-oss's interleaved even/odd gate/up columns are de-interleaved ONCE at
  load (`deinterleave_gate_up` — an exact column permutation of weight bits, scale and bias alike), never strided
  in-graph. The router (topk — untraceable) and the weighted combine stay in torch, in the serving runner; the
  runner's combine reads the router return's LAST two entries as `(scores, indices)`, which covers plain-logit
  routers and 3-tuple ones (Glm4Moe's `Glm4MoeTopkRouter`) alike. A DeepSeek/GLM always-on `shared_experts`
  module (a plain dense MLP over the same normed `xn`) folds INTO `post_attn`'s returned `h`, so the combine
  stays `h + Σ w_e · expert_e(xn)` with no runner change; Qwen-MoE's GATED shared expert (`shared_expert_gate`)
  and Gemma 4-norm MoE blocks are rejected until a model needs them (`tests/serving/test_moe_split.py`).
  `split_gate_up=True` selects a third form, `expert(x, w_gate, w_up, w_down)` — the EXL3 shape. There each
  coded linear carries its own input-side channel vector, so the merged gate_up weight has no single
  input-side channel vector and the merged spelling would only add a concat the chunk split undoes.
  A model's `routed_scaling_factor` ordinarily multiplies only the routed expert result before that shared-expert
  addition. An architecture may mark the factor as folded into router weights when fp16 partials cannot carry it.
  Laguna's optional softplus `g_proj` attention gate is likewise retained in both dense and MoE post-attention
  programs; per-head gates reshape the flattened attention seam to `[tokens, heads, head_dim]` for multiplication.
  The split reads an explicit gate layout when the Transformers module provides one and otherwise derives it from the
  gate, query-projection, and head widths, covering older built-in Laguna modules without a model-name special case.
  The `F.linear` expert form applies the module's `swiglu_limit` when the experts carry one (DeepSeek V4: gate clamped
  above, up clamped on both sides, then SwiGLU and the down projection); OLMoE has no limit and is unchanged.
  Config-only selected-layer tracing replaces routing with one representative routed expert before materialization.
  DeepSeek V4 requires that replacement to be confirmed and preserves the same clamp. Missing the replacement fails
  closed.

  A DeepSeek V4 block (`hyper_connection_seam(block)` is not `None`: it carries `attn_hc` / `ffn_hc`) takes the
  **attention-sublayer seam** instead of the q/k/v one, because the 1Cat vLLM fork's paged MLA attention owns the
  whole sublayer — low-rank q projection, shared-KV projection, HCA/CSA compressors, the lightning indexer and the
  grouped output projection are fused with its paged caches, and nothing external can hand it compressed latents.
  The carrier is the `hc_mult` hyper-connection residual streams flattened to `[num_tokens, hc_mult * hidden]`:
  `pre(hidden[T, hc·H]) → x[T, H]` is the attention-site stream collapse plus `input_layernorm` (exactly what the
  fork's `DeepseekV4Attention.forward(positions, x)` consumes); `post(attn_out[T, H], residual[T, hc·H]) → (mixed[T,
  hc·H], xn[T, H], mix[T, hc])` recomputes the attention-site collapse weights from the carrier (one small GEMM, so
  only the carrier crosses the seam), mixes the attention output onto the streams, runs the feed-forward collapse and
  `post_attention_layernorm`, places the shared expert on the streams, and returns the feed-forward per-stream `post`
  weights. The routed combine runs in torch as before and lands through `place_routed_streams(mixed, routed, mix)`
  (`mixed + mix ⊗ routed`). Both halves call the block's own `DeepseekV4HyperConnection` modules on a `[1, T, hc, H]`
  view, so the sigmoid / Sinkhorn / float32 contract is the installed modeling code's. `build_attention_split_wrapper`
  rejects these blocks (no dense DeepSeek V4 layer exists), and the seam has no coded-trunk or float32-residual form
  (`tests/serving/test_deepseek_v4_split.py` proves the carve against the eager layer for sliding, HCA and CSA
  layers with both hash and top-k routers).

- `load_quantized_split(model_dir, dtype) → (model, expert_store)` is the shard-streamed serving load of a
  quantized MoE checkpoint. The twin builds from config on the meta device (weights never read at trace; the
  experts' would-be initialization never materializes), while the dense trunk streams per shard as real values and
  attaches via `load_state_dict(assign=True)`. Expert tensors collect into a per-layer store keyed by the expert
  program's input names: FP8 weights remain raw bits with f32 scales, and native-MXFP4 gpt-oss weights remain uint8
  blocks with uint8 E8M0 scales; biases stay in the requested value dtype. `expert_range=(lo, hi)` narrows the read to
  one tensor-parallel rank's expert shard, re-indexed rank-locally, so a rank never reads bytes it does not own.
  The twin's config must resolve to Transformers' OWN class for the architecture: a hosting process can re-register
  the model type onto its own minimal config class (vLLM's config parser does, process-wide), which drops every field
  the real `__init__` derives — DeepSeek V4 loses `layer_types` — so when a same-named native class exists, the
  loader reloads the config with it.

  A checkpoint published in its own namespace is translated by `_native_checkpoint_renamer`, which reuses the renaming
  Transformers itself publishes for the architecture (`get_checkpoint_conversion_mapping`) instead of keeping a second
  copy that can drift from the modeling code the twin is built from. Only its `WeightRenaming` entries apply: the
  accompanying `WeightConverter` entries merge routed experts into one dense parameter, which is exactly what a
  serving load must not do. DeepSeek V4 is the architecture that needs this today — `layers.N.attn.wq_a`,
  `layers.N.ffn.experts.E.w1`, `hc_attn_fn`, `embed`/`head`, and `.scale` for every block-scale sibling. Two rules
  finish it: routed `w1`/`w3`/`w2` take their gate/up/down module names, and a `.scale` leaf becomes the
  `weight_scale` sibling ONLY when the module's `.weight` is also present — the hyper-connection blocks carry a
  LEARNED `hc_attn_scale` parameter whose name ends the same way, and renaming it leaves the twin's stream mixing on
  meta. Sibling lookups therefore run in the module namespace, or a natively spelled block scale never pairs and every
  fp8 trunk weight loads unscaled. That checkpoint also declares an fp8 trunk while storing routed experts as native
  MXFP4 (`expert_dtype: fp4`, `I8 [out, in/2]` nibble pairs beside `F8_E8M0 [out, in/32]` exponents), which the loader
  views — never casts — onto the uint8 blocks/scales carrier the expert programs bind.

  A multi-token-prediction head is never owned by this loader: no twin instantiates it, and on DeepSeek V4 its 256
  routed experts are 4,608 of the checkpoint's tensors, read in full on every rank only to be discarded. An EXL3
  checkpoint takes the same split at the trellis format's own shapes (`fmt == "exl3"`). Laguna EXL3 additionally
  stores routed up projections with the architecture's `interm_div=128` scale; the loader folds the inverse and the
  model's base routed scale into selected routing weights and marks their combine for fp32 accumulation, matching the
  reference runtime without scaling expert partials in fp16. The architecture's residual stream is fp32 from embedding
  through every decoder block. Norms, q/k/v, attention output, and gate/up intermediates stay fp16; the marked trace
  promotes exactly the checkpoint-provenanced attention `o_proj`, dense and routed `down_proj`, and the shared-expert
  activation/down cone to fp32 before trellis spelling. Operands remain compressed. The final norm returns fp16 for
  the head. The precision marker is limited to Laguna EXL3; ordinary and other EXL3 graphs keep their existing dtypes.
  Its explicit reference API may decode trunk values, while serving passes `compress_trunk=True`: the twin parameters
  stay uninitialized placeholders and the caller re-sources each coded linear from the checkpoint
  (`serving/gen_runner.py`). There is no automatic dense serving fallback; an unsupported coded linear fails during
  birth-time spelling. Either way every routed expert keeps its PACKED CODES. EXL3 stores experts as per-expert
  MODULES, so `_expert_slot` reports an expert index and `_stack_exl3_experts` stacks each `(layer, projection, leaf)`
  triple into one E-leading tensor, putting `suh` in its 128-blocked form and trimming `svh` to the logical out extent
  — the shapes `spell_trellis_inputs` declares, so a launch's per-expert slice needs no reshape. DeepSeek-lineage FP8
  checkpoints use the same per-expert module layout for ordinary 2-D weights and block scales; `_stack_expert_modules`
  preserves FP8 bits on the uint8 carrier, stacks scales as float32, concatenates gate and up along the output axis,
  and leaves down weights in checkpoint orientation. Ignored dense layers take the same path without scales. The store
  also carries `codebooks[layer][input_name]`, the marker-derived codebook id the speller stamps on each decode, plus
  `dir` and `trunk` (`"values"` / `"codes"`) — what a caller needs to re-source a coded trunk. Never the whole dict at
  once — a 20B checkpoint's whole-dict value form is ~42 GB of host RAM. `load_quantized_twin` stays the whole-dict
  eager/accuracy twin for models small enough to hold (FP8, native MXFP4, and EXL3 checkpoints alike). A
  selected-layer native-MXFP4 eager twin instead decodes and attaches only its shard-streamed expert store, preserving
  the value-reference contract without expanding every layer. On the way in the EXL3 path trims encode padding back to
  the declared parameter shapes (`_trim_padded_weights` — both weight dims round up to 128 at encode time) and packs
  per-expert checkpoint modules (`…experts.E.{gate,up,down}_proj.weight`, the DeepSeek/GLM lineage) into the v5 3-D
  expert params (`_pack_expert_state`).

  Quantized architecture construction uses the same guarded remote-code rule as the ordinary model trace. It first
  asks Transformers for its built-in config/model class and retries with `trust_remote_code=True` only when that call
  raises the explicit trust-required `ValueError`; unrelated configuration failures propagate unchanged.

- `stamp_sliding_windows(graph, config, layer_type=None)` re-asserts the per-layer sliding window the trace ERASES:
  a single-layer trace carries no mask at all (HF takes the `is_causal` path — the traced layer is pure causal at
  every seq), and a whole-model trace materializes the banded mask as an opaque additive tensor. The stamp sets
  `SdpaOp.sliding_window` (+ `is_causal`) from `config.sliding_window` × `layer_types` — single-layer via the
  `layer_type` kwarg, whole-model by walking the graph's SDPA nodes in execution order (a count mismatch stamps
  nothing). Semantics: the stamped SDPA's mask keeps AT MOST the causal band `kv ∈ [m − W + 1, m]` (an explicit mask
  operand may keep less, e.g. padding — it stays applied), which is what lets the lowering skip key blocks wholly
  outside the band and both reference backends (`SdpaOp.forward`, `backend/torch_ref.py`) compute the band.
  `commands/compile.py` calls it after every model/layer `trace_module`.

`torch.py` converts only FX nodes observable through the exported value output. FX's stock dead-code elimination
deliberately retains every mutating ATen schema as impure, including mutations of local tensors whose values never
escape the function. Reverse reachability removes those local branches; ATen schema aliases additionally retain a
write through a view of a returned tensor. An unsupported operation on an observable path remains live and fails
loudly, so the filter is not an operator-support fallback. Retaining a write does not itself functionalize storage:
`copy_` and `fill_` handle the bounded local view forms above and separately reject aliases that cannot be versioned.

`SliceOp` nodes record `dim`/`start` as **op fields** at trace time (`torch.py`'s slice handler reads the raw FX
args): the legacy constant-input convention can't represent a `None` start (`x[:, :s]`) or a SymInt end —
`_resolve_inputs` drops both, leaving the surviving constants positionally ambiguous. Pre-field IR dumps still
decompose via the constant-input fallback.

### `dit.py` — fixed DiT block adapter

The experimental Diffusers adapter loads only a checkpoint's `transformer` subfolder with `AutoModel`, selects one
`transformer_blocks` entry, converts it to FP16, and forces `AttnProcessor2_0` so attention traces as PyTorch SDPA. Its
v1 workload is deterministic: hidden states `[1, 256, 1152]` from seed 0, timestep 500, and class label 207.

Diffusers' timestep embedding constructs a static sinusoidal frequency vector with `arange` on every call. The adapter
materializes that vector as a float32 module buffer before export; timestep multiplication, sin/cos, class embedding,
AdaLayerNorm-Zero, chunking, and all learned projections remain in the graph. The result uses the standard
`(graph, module, args, kwargs)` bundle, so binding, eager accuracy, profiling, and interleaved benchmarking remain
shared with CausalLM traces.

## Entry points

- CLI model/IR/code loading: `commands.compile.load_or_trace` is shared by `trace`, `compile`, `run`, and `tune`, so
  adapters, dynamic shapes, quantized checkpoint reconstruction, and the guarded remote-code fallback cannot drift.
- Working-golden inventory generation is downstream compiler/search behavior, not frontend capture behavior:
  `compiler.pipeline.search.working_golden.write_trace_inventory` lowers the captured graph through fusion, enumerates
  every fold-aware kernel occurrence, and embeds the complete stable Torch IR program once in the golden YAML. Each
  target is selected by unique frontend origins when possible; an empty or ambiguous selector stores the standalone
  post-fusion Loop IR slice instead. The smaller provenance tuning reproducer is derived in memory when the working
  file is loaded. Quantized model traces also embed the digest of their exact checkpoint declaration. Frontend nodes
  carrying the generic `trace.materialize` hint become auxiliary outputs only in the inventory copy, preserving an
  internal storage boundary without changing an ordinary model call.
  `commands.trace` only validates CLI paths and reports that single artifact; traced JSON and sidecars are not outputs.
- Whole-model trace: `trace_module(build_full_model_wrapper(model, …), (input_ids,))`.
- Single-layer trace: `trace_module(model.model.layers[N], (x,), kwargs={…})` (static); with `--dynamic`,
  `trace_module(build_layer_wrapper(block, …), (x,), dynamic_shapes={"x": {1: Dim("seq_len")}})`.
- Inline expression: `graph_from_code("torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))")` (used by every compiler CLI).
- DiT block: `trace_dit_model("facebook/DiT-XL-2-256", 0)` (fixed FP16 block workload).

## Rule

Frontend capture is **upstream of decomposition** — `trace/` emits
`ir/frontend/` ops only, never primitives. Decomposition rules
(`pipeline/passes/frontend/decomposition/`) rewrite frontend ops into tensor-IR
primitives; `trace/` is unaware of that rewrite.
