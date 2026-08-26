# Compiler Architecture

Three layers over a shared `Graph` container.

Reading a `--ir torch` / `--ir tensor` dump: [IR-PSEUDOCODE-TORCH.md](IR-PSEUDOCODE-TORCH.md).

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
Its topological traversal resolves every ready node by node id rather than the hash or insertion order of the
set-backed graph indexes. Stable traversal is part of the persistence boundary: Torch/Loop wire programs and their
target indexes must be byte-identical across fresh Python processes, including graph slices assembled from sets.

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
  Graph JSON op fields use the same stable dimension wire mapping as Torch IR, so static, symbolic, and composite
  `Dim` values round-trip instead of depending on the scalar-only `Dim.value` compatibility property.
  Program specialization also binds the named string extents admitted by the `ReshapeOp.shape` and `SliceOp.shape`
  frontend contracts; unrelated string-valued operation fields are never interpreted as dimensions.
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
- **BF16 uses raw `uint16` bits at NumPy boundaries.** `dtype.encode_bf16` and `decode_bf16` are the shared
  round-to-nearest-even conversion. Numeric values must never be value-cast to the carrier: live PyTorch tensors,
  standalone random inputs, CUDA uploads, and command-layer comparisons preserve or decode the physical bits.
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
`…experts.gate_up_proj` + `…experts.gate_up_proj_scale`). DeepSeek-lineage `weight_scale_inv` names the inverse of
the quantization scale, which is the stored dequant multiplier: both suffixes therefore reconstruct the weight by
multiplication. Only a checkpoint contract that explicitly declares a reciprocal dequant multiplier may select the
division path; the key suffix alone never does.

When an official FP8 declaration also specifies dynamic activations, `loader.quant.spell_dynamic_fp8_activations`
wraps each eligible linear input in the checkpoint's per-row amax, zero-safe scale, encode, decode, and scale algebra.
Linears sharing one projection input share the spelled value. A normal compile retains the model's original outputs;
working-golden inventory generation alone promotes the marked bits and scale values to auxiliary outputs so fusion
preserves the materialized W8A8 boundary. Native FP8 tensor-core enumeration remains explicitly gated by `FP8_MMA`,
and a conservative compile can still execute the same graph algebra without selecting that hardware path.

**Input-sourced fp8.** When the weights are forward-argument `InputOp`s instead of constants (the MoE serving seam's
expert programs — one program per layer kind, per-expert 2-D slices fed per launch), the constant speller can never
fire; `loader.quant.spell_quantized_inputs(graph, specs)` is the post-trace twin. Each named input keeps its node id
and `graph.inputs` slot but its dtype becomes the f8 storage dtype — the feed binds the raw bit pattern on the uint8
fp8 bits carrier, the same rule as the constant side (`emmy/serving/gen_runner.py`'s `_compile_split` binds every plan
input at its own traced dtype for this reason) — and a new `<name>_scale` input is appended, with the same decode-cast
/ broadcast-multiply cone re-creating the value the trace promised. The same W8A16 mul-hoist binding absorbs it:
at gpt-oss expert shapes the gate_up matmul streams fp8 bytes with the scale on the accumulator epilogue at both the
mma and the M=1 coop-reduce tiers. Whether the down matmul's cone (the down projection sum-contracts the exp-bearing
SwiGLU activation) inlines or stays materialized is loop fusion's ordinary outcome — a fusion-band decision upstream
of the tile binding, shared with the constant path.
Indirect operands compose: bits and scale inputs both compile as table-resolved operands for fixed-slot dispatch.

**NVFP4 checkpoints.** The dtype layer carries the storage format — `f4e2m1x2` (a uint8 element holding a packed
pair of e2m1 codes) with its LUT decode `decode_f4x2` and a raw-byte CUDA spelling. `loader/quant.py` recognizes
both checkpoint config conventions (modelopt, compressed-tensors `nvfp4-pack-quantized`; MXFP4's 32-element blocks
stay excluded) and dequantizes the packed trio `<key>` + `<key>_scale` (e4m3, read as raw bits) + `<key>_scale_2`
(f32) for the accuracy twin via `dequantize_nvfp4` — `fuse_nvfp4_scales` collapses the two scale levels into one
f16 tensor, the format's single rounding point. At graph birth, `spell_quantized_constants` rewrites each NVFP4
weight constant into its decode cone. The packed-bits constant feeds a pair-table gather; the e4m3 block-scale
constant and the f32 per-tensor scale (`weight_scale_2`) fuse into one f16 scale that multiplies the gathered
values, one scale per 16 along the last axis. The 256×2 byte-to-value-pair table is a `ConstantOp` whose
`source_graph` computes it at bind time; `from_f4e2m1` decodes the code halves inside that subgraph.

A contraction (the matmul-shaped node) consuming that cone reads two ways, both fork siblings on the tensor-core
tier. The general one is the computed-B reading every producer cone gets: loop fusion merges the decode into the
contraction's own loop nest and the sync compute-fill evaluates it per shared-memory cell, so no decoded weight
materializes between kernels, but the weight crosses global memory as 16-bit values. The specialized one is the
**packed byte-slab stage**, and it is where the format's size advantage survives to the fragment: `_packed`
recognizes the cone (a packed-pair bits load feeding a value-pair gather, times a factor whose every contraction-axis
reference is block-guarded), the weight's bits then copy VERBATIM into a byte slab at half a 16-bit slab's traffic,
its block scales decode once per block into a small companion slab, and one fragment loader reads both — decoding
each byte's two codes through a constant value table and applying the block's scale. That table exists for f16 and
bf16 fragments alike — every e2m1 value is exact in both — so a bf16 trace, which is what Qwen models produce,
takes the same path. The W8A16 mul-hoist (the scale
multiply moved out of the reduction loop onto the accumulator) still does not apply: an NVFP4 scale varies along the
contraction axis, so it does not commute out of the fold. The packed stage's scope is the shape its loader is
written for — either copy transport, an N-major weight of 16-value blocks under a 16-bit atom whose K step is that
same 16; anything else declines to the general reading, which computes the same values. A TMA box deposits its byte
slab dense where cp.async pads each row, so the two forms differ in slab size and row stride, not in what they drain.

**Static 4-bit activations (the declared W4A4 program).** An NVFP4 checkpoint that declares static 4-bit input
activations and stores per-linear `input_scale` tensors (modelopt's calibrated activation `scale_2`, one f32 =
calibration amax / (6 · 448)) marks its linears for W4A4. `loader.quant.spell_static_fp4_activations` runs after `spell_quantized_constants`,
whose spelled weight cones are the marker it reads, and writes the quantize→dequantize round trip in front of each
marked linear, in the same
shared-vocabulary algebra: per 16-element K block, the e4m3 scale round trip (`to_f8e4m3(amax / (6·s2))`), ONE
f32→f16 rounding of the fused scale (`fuse_nvfp4_scales` parity), the e2m1 encode over the rounded scale, the pair
pack into an `f4e2m1x2` buffer, and the same pair-table-gather decode chain the weight side spells. Both matmul
operands then read as one decode-chain shape, the graph's own meaning becomes Σ x̂·ŵ for the marked matmuls, and the
numpy backend stays the parity oracle for every lowering of it. Two halves spell the round trip, and the split
decides what reaches memory: equal-valued `input_scale` tensors over one activation share the QUANTIZE (a fused
projection group calibrates to one scale), while each consumer gets its own reconstruction. Loop fusion materializes
an activation's fan-out point, so a shared activation reaches its matmuls as the packed codes beside their raw e4m3
block scales — the same two leaves a packed weight constant stores — rather than as a dense 16-bit buffer with the
codes dissolved into the producer. Unmarked linears keep their 16-bit activations. One parity property is inherent
rather than a defect: behind a COMPUTED producer the two backends reach
the encodes with epsilon-different upstream values, and a block whose scale ratio lands within that epsilon of a
rounding boundary flips one code — parity there is distributional (median exact, rare flips bounded by the
quantization step), where direct-feed comparisons stay tight.

Both readings still hand the tensor cores 16-bit fragments, because every mma before Blackwell multiplies 16-bit or
8-bit operands. Consumer Blackwell adds one that multiplies the 4-bit codes THEMSELVES and applies each 16-value
block's scale in hardware — registered as the `mma_m16n8k64_e2m1_f32` atom, where a matmul carries no decode at all.
A marked matmul reaches it when BOTH operands read as packed decode chains: the schedule offers the atom, the stage
resolves FOUR verbatim byte slabs (both operands' codes, both operands' raw e4m3 block scales — nothing is
compute-filled, because the cell takes the stored scale byte itself), and the drain loads two data fragments through
the same byte gathers the fp8 k32 atoms use plus one scale register per side. The per-tensor scale levels ride the
epilogue.

That last move is why the native lowering carries a bounded gap rather than the exact oracle every other lowering
answers to. The declared program applies `f16(block_scale x tensor_scale)` per element — the single fused rounding
above — and the instruction applies the raw block scale itself with the tensor level factored out, so the two are not
the same expression and no reassociation connects them. The gap is one f16 rounding of a per-block constant per side,
about 2^-11 relative; the native path is therefore checked to a tolerance, and every other reading keeps the exact
check. Dropping the fusion would close it and is recorded as a follow-up.

**Mixed-scheme checkpoints.** A checkpoint may quantize different leaves differently, and the two
recognizers answer independently rather than exclusively: each asks whether ANY declared weight group is
its own scheme. modelopt spells this as `quant_algo: "MIXED_PRECISION"` with one `config_groups` entry per
scheme (nvidia/Qwen3.6-27B-NVFP4 puts its attention and delta-net projections in fp8 and its MLP and lm_head in NVFP4);
compressed-tensors spells it by simply carrying groups of both widths, which already worked. Both spellers
then run over the checkpoint and each takes the leaves whose STORED SIBLINGS are its own — the NVFP4 trio
(`<key>` + `<key>_scale` + `<key>_scale_2`, checked first because its `<key>_scale` would otherwise shadow
the fp8 pairing), the fp8 pair (`<key>` at an fp8 dtype + `<key>_scale` or `<key>_scale_inv`), and anything
matching neither passes through unquantized. Which scheme a leaf uses was never a config question, so
recognition is the only thing mixed checkpoints needed. The recognizers still decline each other's PURE
checkpoints; `checkpoint_quant_summary` names both schemes for the mixed case, since a boot log naming one
would misreport half the model.

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
