# Compiler Architecture

Three layers over a shared `Graph` container.

```
PyTorch module
   │  trace/              ── PyTorch → Graph IR capture
   ▼
Graph (frontend ops)                        ── Layer 1
   │  pipeline/passes/frontend/decomposition
   │  pipeline/passes/frontend/optimization
   │  pipeline/passes/loop/lifting
   │  pipeline/passes/loop/fusion
   ▼
Graph[LoopOp]  (one LoopOp = one kernel)    ── Layer 2
   │  pipeline/passes/lowering/tile         (Loop IR → Tile IR)
   │  pipeline/passes/lowering/kernel       (Tile IR → Kernel IR)
   │  pipeline/passes/lowering/cuda         (Kernel IR → CUDA source)
   ▼
Graph[CudaOp]                               ── Layer 3
   │  backend/cuda                          (cupy.RawKernel via NVRTC)
   ▼
GPU
```

`Graph` (`compiler/graph.py`) hosts nodes from every dialect; rewrite
passes swap node ops in place, so there is no separate "program" type.

Nodes are **multi-output**: `Node.outputs` is an ordered, non-empty tuple of `Tensor`s, and `node.output` is a
read-only alias of slot 0 (the **primary** output), so single-output code reads exactly what it always did.
**Buffer names are the edge currency** — `node.inputs`, `graph.inputs`/`outputs`, and body `Load`/`Write` refs
all name buffers: a primary buffer travels under its node's id (`node.id == outputs[0].name` in steady state),
a non-primary buffer under its own tensor name (`<primary>__sq` style). Every buffer has exactly one
`(node, slot)` producer (SSA per buffer; `add_node` rejects duplicates). The graph maintains two buffer-keyed
indexes in lockstep with every mutation: `graph.producer(buf)` / `graph.buffer(buf)` resolve a buffer to its
producing node / its `Tensor`, and `graph.buffer_users(buf)` lists the consumers of that one buffer —
`graph.users(node_id)` stays node-granular (the union over the node's buffers). `Graph.validate()` checks the
index/SSA invariants; tests call it directly (never production compile paths).

`Graph.structural_key()` implements the `Structural` protocol
(`compiler/structural.py`) — a Merkle-style hex sha256 digest used for
candidate dedup in autotuning loops. Per node it folds in op kind,
op body's `Body.structural_key()` (or other dataclass fields for leaf
ops, skipping `name`), `Tensor.shape` / `dtype` (skipping `Tensor.name`),
and the recursive digests of input nodes; the top-level digest folds in
the graph's `inputs` / `outputs` sequences. Hints and graph-internal
node ids are excluded. Two graphs that compute the same dataflow
through structurally-equivalent kernels hash equal regardless of
node-id naming or inconsequential body details. Not cached — `Graph`
is mutable; callers that dedup many candidates snapshot the digest
themselves.

`Structural` is the one convention: anything we compare or cache by
structure (`Graph`, `Body`, `Context`, future fork-option payloads and
subgraph slices) implements `structural_key() -> str`. The `digest(...)`
helper in `compiler/structural.py` is the canonical fold; composite
implementers call it with child digests + their own discriminating
fields. Each implementer's docstring documents what's deliberately
excluded (names, hints, ambient I/O) — the contract is "include only
bits that affect codegen output or dataflow semantics" so the
autotuning cache doesn't bust on cosmetic edits.

## Module layout

| Path                  | Role                                    | See                          |
|-----------------------|-----------------------------------------|------------------------------|
| `graph.py`            | `Graph`, `Node`, `Tensor`, `Hints`      | —                            |
| `dim.py`              | `Dim` — shape extent backed by an `Expr` (static or symbolic) | —    |
| `ir/`                 | Op-type definitions per dialect         | `ir/ARCHITECTURE.md`         |
| `trace/`              | PyTorch/HuggingFace → Graph IR          | `trace/ARCHITECTURE.md`      |
| `pipeline/`           | Rewrite engine, passes, dump hooks      | `pipeline/ARCHITECTURE.md`   |
| `pipeline/passes/lowering/tile/` | LoopOp → TileOp; **purely algebraic moveset, no specializations** (dispatch on fold algebra) | `pipeline/passes/ARCHITECTURE.md` |
| `backend/`            | Execution (numpy / loop / cuda)         | `backend/ARCHITECTURE.md`    |
| `loader/`             | Bind constants (safetensors / `nn.Module` → `input_data`) | —              |
| `pipeline/search/`    | Autotune DB + MCTS tree (see below)     | `pipeline/ARCHITECTURE.md`   |
| `structural.py`       | `Structural` protocol + `digest()` fold | —                            |
| `provenance.py`       | Op provenance — map fused kernels back to original frontend ops | — (see below) |
| `specialize.py`       | Bind named symbolic dimensions in persisted Torch/Loop programs before lowering | — |

## Per-layer rules

- **Layer 1** — no GPU, no CUDA, no backend imports. Dialect ops
  implement `infer_output_shape(input_shapes)` and a numpy `forward()`.
- **Layer 2** — operates on `Graph` + Loop IR only. Every `LoopOp`'s
  `__post_init__` canonicalizes (`ir/stmt/normalize.py`) and simplifies
  (`ir/stmt/passes.py`) its body.
- **Layer 3** — backends are the only place GPU specifics live.

## Shared invariants

- **Shape lives on the graph**, not on the op — `node.output.shape`. Each shape element is a `Dim`
  (`compiler/dim.py`) that wraps an `Expr` from `ir/expr.py`: static (`Dim(32)` → `Literal(32)`), atomic
  symbolic (`Dim("seq_len")` → `Var("seq_len")`), or composite from arithmetic (`Dim("S") * Dim(2)` →
  `BinaryExpr("*", Var("S"), Literal(2))`). `Dim` overloads `+`/`-`/`*`/`//`/`%` and eager-folds via
  `Expr.simplify` — static math matches plain int math byte-for-byte; symbolic stays as `BinaryExpr`. It
  also exposes `ceil_div` (`(self + (b-1))//b`): the single masked-tile grid-extent formula for both
  regimes — it folds to the integer ceil (`-(-E//b)`) for a static dim and builds the composite ceil-div
  `Expr` for a symbolic one, so the partition planner's masked block-axis / masked-K sites need no
  static-vs-symbolic branch.
  Read sites use `d.expr` (always works), `d.as_static()` (raises on symbolic), `d.as_atom_name()` (raises
  unless `Var`-backed), or `d.value` (back-compat shim: int for `Literal`, str for `Var`, raises on
  composite). There is deliberately no `__int__` / `__index__`, so `int(d)` and `range(d)` fail loudly
  on anything but a static-int `Dim`. Symbolic dims resolve at launch via `d.expr.eval(sym_env)` —
  composite shapes (e.g. an `S * 2` concat output) resolve from input array axes without per-site
  branching. `Tensor.__post_init__` and `Axis.__post_init__` coerce bare `int` / `str` to `Dim`, so
  producer call sites need no change. An atomic symbolic `Dim` also carries a `hint` — its *expected*
  size (default `DEFAULT_SEQ_HINT=512`, set automatically so reconstruction can't lose it; an explicit
  `Dim(name, hint=...)` overrides). The hint is pure metadata (excluded from `==`/`hash`/structural keys),
  read only by the tuner / partition planner to size tiles for a dynamic axis.
- **A symbolic free axis is tiled for its hint and emitted as a *masked* tile.** A symbolic M/N axis is treated as
  size `hint`, always-overhang: the block axis becomes a composite ceil-div over the symbolic extent
  (`(seq_len + bf - 1)//bf`), and a boundary `Cond(decoded_coord < seq_len)` wraps the body, so one cached kernel runs
  at any runtime `seq_len` — the grid (`ir/cuda/ir.py` `GridDimSpec` accepts an `Expr` factor, resolved via
  `Expr.eval` at launch) and the guard read the runtime value while the tile shape is tuned for the hint. The backend
  benches a symbolic graph at the hint when no real inputs are supplied (`Graph.symbolic_hints` /
  `backend/cuda/program.py` `_resolve_symbolic`), so `tune` and `compile` agree on a hint-sized variant. (The masked tensor-core / cooperative /
  split-K tiers for symbolic axes are part of the in-flight tile-IR rebuild — see `pipeline/passes/ARCHITECTURE.md`
  and the tile IR sources for current coverage.)
- **`ElementwiseOp` inputs must already share the output shape.** The
  decomposition helper
  `pipeline/passes/frontend/decomposition/_broadcast.broadcast_to` wraps
  mismatched inputs in an `IndexMapOp`.
- **One `LoopOp` = one kernel.** Fusion produces `LoopOp` nodes;
  lowering turns each into `KernelOp` (AST) then `CudaOp` (rendered
  source).
- **`LoopOp.forward()` executes.** `ir/loop/runner.py` renders the body
  to C++ and JIT-compiles it via cppyy / Cling, letting the default
  `Backend.run` topo-walk (`backend/base.py`) run post-fusion graphs on
  CPU — fusion correctness can be checked without a GPU.

## Quantized checkpoints (FP8)

A quantized checkpoint never reaches the trace: the trace runs over the bf16 architecture twin built from config
(quantization is a property of the checkpoint, not the architecture). Immediately post-trace,
`loader.quant.spell_quantized_constants` rewrites each fp8-stored weight into in-graph algebra — a bits constant
(f8 dtype, the weight's source path) + a scale constant + the dequant cone (decode-cast, broadcast-multiply, a
reshape pair for 2-D block scales) — so from birth a quantized weight is just constants + algebra, with no metadata
on any shared IR type. Storage-decode cones stay in-graph unconditionally: fp8 bits remain compressed in device
memory, the dtype decode is absorbed at the fragment load, and a compatible scale is hoisted to the epilogue. The
generic constant folder deliberately excludes storage-decode cones because materializing them would expand the
buffer into its compute dtype. Scale pairing is the general `<key>_scale` / `<key>_scale_inv` rule — it subsumes the
`.weight` → `.weight_scale` convention and covers non-`.weight` leaves (gpt-oss's 3-D expert params,
`…experts.gate_up_proj` + `…experts.gate_up_proj_scale`).

**Input-sourced fp8.** When the weights are forward-argument `InputOp`s instead of constants (the MoE serving seam's
expert programs — one program per layer kind, per-expert 2-D slices fed per launch), the constant speller can never
fire; `loader.quant.spell_quantized_inputs(graph, specs)` is the post-trace twin. Each named input keeps its node id
and `graph.inputs` slot but its dtype becomes the f8 storage dtype — the feed binds the raw bit pattern on the uint8
fp8 bits carrier, the same rule as the constant side (`emmy/serving/gen_runner.py`'s `_compile_split` binds every plan
input at its own traced dtype for this reason) — and a new `<name>_scale` input is appended, with the same decode-cast
/ broadcast-multiply cone re-creating the value the trace promised. The same W8A16 mul-hoist binding absorbs it:
at gpt-oss expert shapes the gate_up matmul streams fp8 bytes with the scale on the accumulator epilogue at both the
mma and the M=1 coop-reduce tiers. The down matmul's cone instead stays materialized by loop fusion's flash-consumer
protection (the down projection sum-contracts the exp-bearing SwiGLU activation, which reads as a future
softmax-then-P@V offer site) — a fusion-band decision upstream of the tile binding, shared with the constant path.
Indirect operands compose: bits and scale inputs both compile as table-resolved operands for fixed-slot dispatch.

**Trellis-coded checkpoints (EXL3).** `loader/exl3.py` owns the pure NumPy reference:
packed-window extraction, computed codebooks, tile ordering, and the block Hadamard/sign fold.
Checkpoint discovery, sibling pairing, codebook markers, and allocation metadata remain in
`loader/quant.py` and `loader/safetensors.py`.

At graph birth, `spell_trellis_constants` replaces each coded weight together with its sole
`LinearOp` consumer. `loader/trellis.py` emits the factorized contraction directly: ordinary
ranges, casts/bitcasts, integer algebra, gathers/index maps, layouts, and matmuls. The packed decode
becomes the core contraction's computed B operand; no logical dense weight or fp16 weight-rounding
buffer exists in the executable graph. `spell_trellis_inputs` applies the same builder to expert
weight inputs. Marker presence selects the generic codebook algebra. An unsupported or shared
coded linear fails at birth rather than falling back to materialization; ordinary padded channel
dimensions are handled inside the generic spelling.

`load_dequantized_state_dict` remains an explicit eager/reference utility and the block decoder is
still used for an unsupported coded LM head. Neither is an automatic compiled-serving fallback.
`coded_tensor_storage` remains a loader-only, weight-free inventory for tracing and release
coverage.

**Invariant: quantization is not a concept past the decomposition band.** Downstream layers — lowering, backends,
search — may know canonical dtypes (`f8e4m3`), generic elementwise ops, and graph algebra. They may NEVER contain a
checkpoint format's op, statement, helper, pass branch, schedule feature, environment gate, comment, or name.
Scheme-specific types and metadata belong only to checkpoint loading and birth-time spelling. Spelling must emit
generic algebra, and frontend decomposition must leave only generic tensor IR or a regular constant before Loop IR.
Mechanical architecture tests scan the post-decomposition source tree for format-name leaks.

## Op provenance

`provenance.py` threads a single `Node.hints["prov"]` map —
`{origin_id: {"kind": <op-class>, "pieces": [piece_id, …]}}` — from the traced frontend graph all the way to each
`CudaOp`, so a fused kernel knows which original PyTorch ops it implements. `origin_id` is the trace-time node id of an
original op (`rms_norm_0`); `pieces` are the primitives it decomposed into and that this node embodies. Coverage of an
origin is `len(pieces)` over the union of that origin's pieces across the whole graph (`totals` / `coverage`) — so the
`i/N` fraction stays correct under CSE and recursive decomposition instead of freezing `N` at the first split.

It rides on one chokepoint: `Graph.splice` calls `provenance.propagate` with a `mint_pieces` flag (set by
`Candidate.apply` from the pass namespace — `True` only for `frontend/decomposition`). Decomposition *mints* each new
fragment node as a fresh piece of the consumed origins; fusion / lifting / optimization folds *aggregate* the consumed
piece sets onto the merged node (unioning the dissolved producers so a multi-output splice drops nothing). Lowering is
in-place `Op` rebinds, so prov rides through `LoopOp → TileOp → KernelOp → CudaOp` untouched. Seeded once at
`Pipeline.tune_async` / `Pipeline.run` entry (idempotent); pure metadata, excluded from structural / cache keys. Boundary sentinels
(`InputOp`/`ConstantOp`) never carry prov: `put` refuses to stamp them and `propagate` scrubs splice outputs that land
on one (the generic hint merge would otherwise copy prov onto e.g. the ConstantOp produced by the sm_90+
weight-transpose fold, inflating `totals` so every kernel of that origin read partial coverage).

Consumers: `provenance.name_for` (called from `pipeline/passes/loop/stamp/010_stamp_loop_names.py`, the
loop-dialect stamp pass) names kernels after the ops they realize (`k_rms_norm` when full, `k_rms_norm_reduce` when partial)
and stamps the name onto `LoopOp.name`; every subsequent dialect (`TileOp`/`KernelOp`/`CudaOp`) just copies it
through. Multi-op labels sort dominant-first (descending piece count, lexical tie-break), so the name is independent
of fusion merge order — the attention kernel is `k_sdpa_linear_reduce`, its QKV-prologue twin `k_linear_sdpa_reduce`.
Layout/plumbing origins (`_WEAK_KINDS`: transpose / reshape / unsqueeze / cat / slice) label a kernel only when no
strong op is present — RoPE plumbing fused into attention doesn't pollute the name, while a standalone copy kernel
still reads `k_cat_…` instead of the node-id fallback. Compiler dumps retain provenance-selected frontend slices in
memory for tune benchmarking; stable persistence of those programs belongs exclusively to golden YAML.
