# Trace Architecture

Frontend capture: PyTorch/HuggingFace model → `Graph` populated with
Layer-1 frontend ops.

## Modules

### `torch.py` — FX → Graph IR

`trace_module(module, args, kwargs=None) → Graph` runs
`torch.export.export()` to get an FX graph, then walks each node and
emits the matching frontend op (`LinearOp`, `MatmulOp`, `SdpaOp`,
`ElementwiseOp`, `ReduceOp`, …). `trace_module_with_constants` returns
the graph plus a `placeholder_name → attr_path` dict for resolving
weights/buffers back to their module attributes.

Per-op handlers map aten names (`aten.add.Tensor`,
`aten.linear.default`, …) to dialect op constructors; input shapes are
pulled from the FX meta and fed into the op's `infer_output_shape` to
stamp the output tensor.

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
by both reference backends, the decomposition's scale constant, and the flash re-synthesis (which reads the
value back off the score producer's constant). Gemma-nano (E2B/E4B) passes `scale=1.0` — its q_norm absorbs
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

- `load_quantized_split(model_dir, dtype) → (model, expert_store)` is the SHARD-STREAMED serving load of a
  quantized MoE checkpoint (gpt-oss fp8): the twin builds from config on the META device (weights never read at
  trace; the experts' would-be init never materializes), the dense trunk streams per shard as real values
  (fp8 attention weights resolved by their `<key>_scale` partners) attached via `load_state_dict(assign=True)`,
  and the expert tensors collect into a per-layer store keyed by the expert program's input names — fp8 weights
  as raw bits on the uint8 carrier plus f32 scale tensors, biases as `dtype` values. Never the whole dict at
  once — a 20B checkpoint's whole-dict value form is ~42 GB of host RAM. `load_quantized_twin` stays the
  whole-dict eager/accuracy twin for models small enough to hold (fp8 and EXL3 checkpoints alike); on the way in
  it trims EXL3's encode padding back to the declared parameter shapes (`_trim_padded_weights` — both weight dims
  round up to 128 at encode time) and packs per-expert checkpoint modules (`…experts.E.{gate,up,down}_proj.weight`,
  the DeepSeek/GLM lineage) into the v5 3-D expert params (`_pack_expert_state`).

- `stamp_sliding_windows(graph, config, layer_type=None)` re-asserts the per-layer sliding window the trace ERASES:
  a single-layer trace carries no mask at all (HF takes the `is_causal` path — the traced layer is pure causal at
  every seq), and a whole-model trace materializes the banded mask as an opaque additive tensor. The stamp sets
  `SdpaOp.sliding_window` (+ `is_causal`) from `config.sliding_window` × `layer_types` — single-layer via the
  `layer_type` kwarg, whole-model by walking the graph's SDPA nodes in execution order (a count mismatch stamps
  nothing). Semantics: the stamped SDPA's mask keeps AT MOST the causal band `kv ∈ [m − W + 1, m]` (an explicit mask
  operand may keep less, e.g. padding — it stays applied), which is what lets the lowering skip key blocks wholly
  outside the band and both reference backends (`SdpaOp.forward`, `backend/torch_ref.py`) compute the band.
  `commands/compile.py` calls it after every model/layer `trace_module`.

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
  file is loaded.
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
