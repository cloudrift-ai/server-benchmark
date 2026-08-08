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
on any shared IR type. The generic `032_fold_constant_subgraphs` decomposition rule then dissolves the algebra as
early as possible (default): the cone collapses into ONE `ConstantOp` whose `source_graph` bind record the loader
evaluates through the NumPy backend at bind time (leaf f8 sources read as raw bits, LUT decode, f32 scale multiply,
one cast into the compute dtype), and every later pass sees a plain constant. `EMMY_FP8_EXPAND` skips the fold: the
cone stays in-graph and rides the B-operand cone into the kernel (fp8 bits in device memory, decode absorbed by the
storage dtype at the fragment load, scale hoisted to the epilogue). Scale pairing is the general `<key>_scale` /
`<key>_scale_inv` rule — it subsumes the `.weight` → `.weight_scale` convention and covers non-`.weight` leaves
(gpt-oss's 3-D expert params, `…experts.gate_up_proj` + `…experts.gate_up_proj_scale`).

**Input-sourced fp8.** When the weights are forward-argument `InputOp`s instead of constants (the MoE serving seam's
expert programs — one program per layer kind, per-expert 2-D slices fed per launch), the constant speller can never
fire; `loader.quant.spell_quantized_inputs(graph, specs)` is the post-trace twin. Each named input keeps its node id
and `graph.inputs` slot but its dtype becomes the f8 storage dtype — the feed binds the raw bit pattern on the uint8
fp8 bits carrier, the same rule as the constant side (`emmy/serving/gen_runner.py`'s `_compile_split` binds every plan
input at its own traced dtype for this reason) — and a new `<name>_scale` input is appended, with the same decode-cast
/ broadcast-multiply cone re-creating the value the trace promised. An input-rooted cone is not a constant subgraph,
so it stays in-graph unconditionally (no `EMMY_FP8_EXPAND` analog) and the same W8A16 mul-hoist binding absorbs it:
at gpt-oss expert shapes the gate_up matmul streams fp8 bytes with the scale on the accumulator epilogue at both the
mma and the M=1 coop-reduce tiers. The down matmul's cone instead stays materialized by loop fusion's flash-consumer
protection (the down projection sum-contracts the exp-bearing SwiGLU activation, which reads as a future
softmax-then-P@V offer site) — a fusion-band decision upstream of the tile binding, shared with the constant
`EMMY_FP8_EXPAND` path, not an input-sourced limitation. Indirect operands compose: bits and scale inputs both
compile as table-resolved operands for the fixed-slot dispatch.

**Trellis-coded checkpoints (EXL3).** `loader/exl3.py` holds the pure numpy decode for the EXL3 (QTIP-class
trellis-coded) weight format: per-16x16-tile bit-window extraction from the tail-biting code stream, the 3INST
computed codebook (bit-exact against exllamav3's CUDA kernels), the mma-fragment tile ordering, and the
128-block Hadamard/sign fold that restores the original basis from the `suh`/`svh` sibling vectors.
`decode_exl3_linear` reconstructs one linear's fp16 weight from its sibling tensors; `decode_exl3_blocks` does the
same in out-feature blocks, for a tensor whose float64 fold would not fit beside the rest of a boot (a vocab-sized
`lm_head` folds ~5 GiB whole). Blocking is bit-exact in the hat basis and within the fold's own documented fp16
rounding after it. Ingestion follows the fp8
design exactly: `config.json` declares `quant_method: "exl3"` (detected by `quantized_checkpoint_dir` alongside
fp8), the twin carries decoded real weights (`load_dequantized_state_dict` decodes trellis siblings to `.weight`
values; per-expert checkpoint modules pack into the v5 3-D expert params, encode padding — both dims rounded up
to 128, e.g. GLM-4.5-Air's `intermediate_size` 10944 → 11008 — trimmed back to the declared shapes), and
`spell_trellis_constants` (the sibling of `spell_quantized_constants`) rewrites each coded `<module>.weight`
constant at birth into three leaf constants (int16 codes on the `i16` carrier + the f16 channel vectors) joined
by a `TrellisDecodeOp` (frontend IR; `cb` records the `mcg`/`mul1` marker presence, `out_features`/`in_features`
slice the encode padding — the reference math, since exllamav3 zero-pads activations in and slices outputs).
`032_fold_constant_subgraphs` collapses the cone into a bind-time `source_graph` record — this is the
correctness lane: full value footprint in memory, bind-time decode bit-exact against the direct decode. Not
every linear is coded — the quantizer keeps sensitivity-selected ones at plain fp16 (GLM-4.5-Air layer 0
`o_proj`), and those load as ordinary tensors. `coded_tensor_storage` is the WEIGHT-FREE view of the same
information: it reads the small per-module allocation sidecar exllamav3 writes beside `config.json` (each coded
module's code rate and its `trellis`/`suh`/`svh` shapes) so a caller without the shards still knows what is coded
and at what rate. Only the serving-twin builder needs it — with the shards in hand the safetensors index is the
pairing source, exactly as for fp8.

**In-kernel trellis decode (computed-B).** The HAT-BASIS form of the op (`TrellisDecodeOp(hadamard=False)` —
the raw per-tile decode, no channel vectors, no Hadamard fold; the basis restore rides the activations, below)
has a kernel realization: it lifts to a `LoopOp` of per-element `TrellisLoad` reads
(the window is directly addressable in the tile's circular bit stream, so each element decodes independently —
no carried walk state), loop fusion inlines it into the consuming matmul, and the contraction binder stores it
as a COMPUTED-B cone. The warp tier then schedules it over the mandatory `sync` compute-fill: the fill decodes
each B tile straight into the K-major smem slab the ldmatrix drain already reads (one `emmy_trellis_decode`
helper call per element — window extraction + the 3INST computed codebook, no stored LUT), while the packed
codes are the only weight bytes that ever cross DRAM (~8× fewer than f16 at K=2) and the A operand rides its
usual vectorized `cp.async` fill underneath. Split-K, TMA and the scalar staged tiers decline (a computed B has
no gmem element layout); the COLLAPSE reading is the reduce-tier fallback. Constant-rooted hat-basis cones fold
by default and stay in-graph under `EMMY_TRELLIS_EXPAND` (the kernel-path gate, the trellis sibling of
`EMMY_FP8_EXPAND`) or under the per-compile `expand=True` argument to `spell_trellis_constants`, which stamps the
`trellis.expand` graph hint the fold reads; checkpoint-basis cones fold regardless. Measured on the 5090
at N=K=22016, K=2 (codes 121 MB — past L2): the compressed matmul beats the same-shape f16 matmul at M=128
(2.10 vs 2.34 ms) and runs decode-ALU-bound at larger prefill M (1.6–1.7× f16 at M=256–2048) — the per-element
decode re-runs per M-tile row, which is the standing lever for the fragment-drain follow-up.

**The decode-phase matvec (the decode band).** At M=1 the contraction demotes to a PLANAR fold, so the decode
reaches the reduce tier instead, and there the per-element leaf is the wrong granularity: a whole 16x16 weight
tile has to be touched for each 2-bit weight, which measured 4 bytes of code traffic and ~28 instructions per
weight. The **decode band** is the reduce partition that fixes it — the transposed coop band (32 lanes sweeping the
output axis) with the register level pinned at the tile's 16 k rows, so a lane's register copies walk 16
CONSECUTIVE k, exactly one tile column, and a kernel-band peephole (`055_fuse_trellis_runs`) rewrites those 16
per-element leaves into the run form of `TrellisLoad` — one `emmy_trellis_decode_col` call, one code fetch,
compile-time word indices. Warp-wide blocks cannot fill the card on a matvec grid, so the band always rides a
cross-CTA split, and the split slice must stay tile-aligned or the run does not fuse. Where the band spells it
is the ONLY reduce row offered: `ShapeKey` cannot see a decode, so a cold pick would otherwise land on an f16
matvec's measured plan — hence also `ShapeKey.dtype_class == "trellis"` (read off the `S_dtype_i16` codes
carrier), which keeps a decoded B's golden / DB rows from joining its f16 twin's in either direction. Measured
on the 5090 at M=1, K=2, against the same-shape f16 matvec: N=K=22016 (past L2) **214 µs vs 580** — 2.7x ahead,
and 157 µs at the band's best pinned split; the L2-resident GLM dense projections land at f16 parity (gate/up
4096→11008: 19.6 vs 18.6; down 11008→4096: 20.5 vs 18.7), where f16 reads its weights out of L2 at 4.9 TB/s and
the comparison flatters it. NCU put the residual on instruction issue, not bandwidth: 11.5 warp-instructions
per decoded weight, SM throughput 69 % against DRAM 35 %. Three follow-ups took that down to 8.25 and rebalanced
the kernel (SM 65 % / DRAM 46 %) — the codebook's mask/XOR as ONE `lop3` (exact, ungated), the widened split
ladder below, and the `F16_REDUCE_F32_ACC` f16-pair fold (`060_pair_decode_accum`, a `FAST_MATH`-family precision
gate: `__half2` products summed over the tile column in fp16 with one f32 promote per tile step, ~5e-4 rel and
flat in K). Greedy on the 5090 now: past-L2 square **142.2 µs f32 lane / 122.6 f16-pair vs f16's 581.7**
(4.1x / 4.7x, 852 / 988 GB/s of codes off DRAM), gate/up 14.9 / 12.3 vs 18.7, down 15.3 / 12.9 vs 18.5.

The band's split widths are a catalog (`space.decode_band_moves(tiles)`, widest first), so a recorded golden's
partition is a catalog member like every other kind's — and it needs to be one: the offline prior mis-ranks the
width cold, which is 19–27 % at the seeded shapes, and `goldens/rtx5090_sm120_glm45air.yaml` is that correction.
The catalog has a WIDE arm keyed on the trellis-tile count (`width = tiles // steps`) ahead of the fixed
power-of-two ladder, because a matvec's tile count is the contraction dim over 16 and routinely carries an odd
factor: without it `down` (11008→4096, 688 = 16·43 tiles) offered ONE width — no fork to decide, hence no golden —
and ran 19.5 µs where `g86k` runs 15.3. Wider is not monotone, since the split's finalize reads a `cta × N` f32
workspace worth `2 / (steps · k_bits)` of the code traffic, so the arm starts at 8 tile steps per CTA.

**Activation-side basis restore.** Only `W_hat` decodes in-kernel, so under the same `EMMY_TRELLIS_EXPAND` gate
`spell_trellis_constants` rewrites the CONSUMING LINEAR instead of the weight constant, moving the checkpoint's
`W = diag(suh)·H·W_hat·H·diag(svh)` basis onto the activations — `x → [pad to k_pad] → ·suh → H → ·1/16 →
@ W_hat → ·1/8 → H → ·svh → [slice] → [+bias]`. The 128-block Hadamard is spelled as PLAIN ALGEBRA: a
128×128 matmul over a 128-blocked operand against one graph-wide `HadamardOp` constant (a zero-input generator
inside a zero-leaf `source_graph` bind record — the matrix has no checkpoint source; symmetric, so `LinearOp`
needs no transpose), which puts the transform on the existing tiers with no new kernel machinery. Two rules make
the chain lower correctly and are load-bearing: every layout change (the encode-pad concat, the flat↔block
reshapes, the output slice) is absorbed by a POINTWISE and never reaches a matmul's activation operand, and the
`1/sqrt(128)` per side is split as the exact powers of two `1/16` before the weight and `1/8` after, so the
shared constant is plain ±1 and both intermediates stay below the balanced magnitude. Only the CONTRACTION dim has
to be static — the leading token axes ride through as whatever `Dim`s the trace gave them, so a symbolic-width
serving program spells the same way a static one does. A linear the rewrite declines (a weight without exactly one
`LinearOp` consumer) falls back to the folded checkpoint-basis cone, correct everywhere. Measured on the 5090 at
N=K=22016, K=2 (past L2), against the same matmul with no basis restore: **+4.5 % at M=1 and +13.4 % at M=128**,
the transforms collapsing into 3–4 kernels
(the scale multiplies fuse as computed-A cones on the two Hadamard matmuls) — at or under exllamav3's ~14 %
standalone cost. At L2-resident shapes the ratio is much worse (+36 % / +68 % at M=128 on the GLM dense
projections) because the compressed matmul is artificially fast there; those absolute microseconds do not
predict in-model step time.

**Trellis weights as program INPUTS (the MoE expert path).** An expert weight is a forward argument, so the
constant speller can never fire on it: `spell_trellis_inputs` is its input-rooted twin, sharing the same chain
builder. Per named weight input it re-mints that input, in place and in its `graph.inputs` slot, as the int16
CODES buffer and APPENDS two channel-vector inputs — `<name>_suh` declared 128-BLOCKED and `<name>_svh` declared
at the LOGICAL out extent, so the serving store's per-expert slice is a plain view and the graph carries no
layout op on either. Gate and up stay separate coded linears: their `suh` differ, so a merged gate_up weight has
no single activation-side basis. An input-rooted cone is not a constant subgraph, so `032_fold_constant_subgraphs`
leaves it in-graph unconditionally — no `EMMY_TRELLIS_EXPAND` analog, the codes stay compressed by construction.
The split that matters for the fixed-slot dispatch: the per-expert tensors (codes, `suh`, `svh`) are table-resolved
indirect operands, while the basis-restore Hadamard is a shared graph CONSTANT and never enters a table.

**Computed constants in the plan.** The Hadamard has no checkpoint key at all — it rides a zero-leaf
`source_graph` bind record, which the plan used to project as `source_path=None`, making `assemble_source` answer
`None` and the weight vanish from the bound feed (silently, on a fresh compile as much as on a pack hit).
`WeightSpec.source_op` is the plan's third pre-chain source form, `("hadamard", (128,))`, rebuilt by
`build_source_op` with no IR involved; `assemble_source` answers it for both spec kinds. A bind record the plan
CANNOT reproduce now marks the weight `load_ops=None`, so the pack save refuses loudly instead of writing a pack
that boots weightless. The load-op vocabulary also grew a `("slice", spans)` form for the single-source affine
`IndexMapOp` a folded `SliceOp` leaves behind (the N-padded `svh[:n]` trim), which previously disabled pack
writing for a whole program set.

**Invariant: quantization is not a concept past the decomposition band.** Downstream layers — lowering, backends,
search — may know canonical dtypes (`f8e4m3`), decode-trait elementwise ops (`ElementwiseImpl.decodes`), and graph
algebra; they may NEVER know checkpoint formats, scheme names, scale pairing, or quantization metadata. The frontend
band (the birth-time speller + the fold pass + the loader) is the only place quantization-as-a-concept exists; a
mechanical gate test (`tests/compiler/loader/test_quant.py`) greps `emmy/` for concept leaks against a frontend-band
allowlist.

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
still reads `k_cat_…` instead of the node-id fallback. `pipeline/dump._dump_torch_repro` slices the pristine frontend graph by a kernel's origins into a runnable
`<kname>.torch.json`; `backend/torch_ref` runs that slice through real torch for the `run --ir` vs-torch comparison.
