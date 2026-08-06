# FP8 support

Goal: make FP8 checkpoints work in emmy end-to-end — compile / run / tune / serve — so that when Qwen3.8-27B lands
(FP8 release expected alongside bf16, as with Qwen3.5/3.6), only model-level work remains. This plan is the general
FP8 story; nothing here is Qwen3.8-specific.

## Status (2026-08-06)

- **Phase 0 DONE** — `fold_into_constant` rebuilds via `dataclasses.replace`, layout nodes inherit their own
  operand's dtype (branch-local propagation); kernel-source digest A/B 27/27 byte-identical.
- **M1 DONE** — dequant-on-load ingestion: f8 dtypes + `QuantSpec`/`ConstantOp.quant`, LUT decode, block-broadcast
  scale, bf16-twin trace; tiny per-channel fp8 checkpoint e2e on the 5090 matches the dequantized eager reference
  (max diff 1.8e-7); non-quantized path bit-identical.
- **M2a DONE** — `180_expand_quantized_constant` (gated behind `EMMY_FP8_EXPAND`, default off) + fp8 search identity
  (`ShapeKey.dtype_class`, `S_dtype_f8*`, golden spellings); flag-off digest A/B 27/27 byte-identical; loop fusion
  pulls the whole dequant cone into one matmul kernel.
- **M2b DONE** — fp8-B through the warp tier (W8A16): k-invariant multiplicative dequant binding (mul-hoist onto the
  f32 epilogue, decode absorbed by the storage dtype, trait-recognized via `ElementwiseImpl.decodes`); warp tier
  reached on BOTH cards (5090 e2e max_diff 1.5e-7, fragment-convert 5.7e-4 max_rel; sm_89/4090 verified, max_rel
  2.5e-4 through the trans fragment helper); quant-metadata containment gate test added. The BYTE WIN IS PENDING
  staged fp8 transport (see M2 residual below) — every current A/B rides the transaction-bound gmem-direct path.
- **M3 DONE (2026-08-06, evidence gate overridden by decision)** — native fp8 mma (W8A8): `m16n8k32` e4m3/e5m2
  atoms; the bare PTX form is THE spelling on both sm_89 and sm_120 (`.kind::f8f6f4` refused by ptxas everywhere);
  fragment ABI verified empirically (A 4×b32, B 2×b32, C/D keep the k16 map); `FP8_MMA` knob under the FAST_MATH
  umbrella — the rationale is the instruction's arch-dependent accumulation (sm_89 reduced-precision ~3.2e-4, sm_120
  effectively true-f32); `to_f8*` encode intrinsics bit-exact vs torch over all finite in-range f16 (saturating
  overflow, documented divergence); the hoist arm is side-generic — static per-tensor AND dynamic per-token amax
  W8A8 both work end-to-end on both cards (`act_scale ⊗ weight_scale` composed on the f32 epilogue). The mma-rate
  thesis is UNOBSERVABLE on the gmem-direct path (all fp8 arms transaction-bound; staged f16 twin 4.3–4.7× faster)
  — the staged-transport residual gates the perf story for W8A16 and W8A8 alike.
- **M4 NOT STARTED** — the serving A/B needs a serving session on a real FP8 checkpoint.
- **QuantSpec RETIRED (dissolve-early migration)** — quantization is graph algebra from birth, folded at `032`;
  the invariant "quantization is not a concept past the decomposition band" is documented in the ARCHITECTURE
  files and enforced by a gate test. Net −165 lines.
- **Dissolve-early migration DONE (2026-08-06)** — `QuantSpec` retired per the deletion path below: quantization is
  spelled as in-graph algebra at birth (`loader.quant.spell_quantized_constants`, post-trace) and dissolved by the
  generic `032_fold_constant_subgraphs` rule (decode-trait-scoped constant-cone fold → one `ConstantOp` with a
  `source_graph` bind record the loader evaluates through the numpy backend; trailing `050`/`060` transposes compose
  onto it). `EMMY_FP8_EXPAND` now SKIPS the fold (same meaning: cone stays in-graph for the kernel path; M2's
  `_atomize` arm unchanged). `stamp_quant_specs` / `180_expand_quantized_constant` / `ConstantOp.quant` deleted; the
  invariant "quantization is not a concept past the decomposition band" is documented in
  `compiler/ARCHITECTURE.md` + `passes/ARCHITECTURE.md` and enforced by a mechanical grep gate in
  `tests/compiler/loader/test_quant.py`. Digest A/B 27/27 byte-identical both flag states; fold-mode binding
  bit-identical to the old M1 dequant.

### M2 residual (handoff)

1. **Staged fp8-B transport + convert-at-drain is the actual byte win.** Warp staged transports refuse a
   storage-dtype operand (correctly — slabs byte-copy at the atom's element width), so every current A/B measures
   the transaction-bound gmem-direct path. Measured on the 5090, fp8-B vs the f16 twin at MxKxN with K=N=4096:

   | M | fp8-B vs f16 twin |
   | --- | --- |
   | 32 | 0.26x |
   | 512 | 0.45x |
   | 4096 | 0.63x |

   Caveat: at M=32 the f16 twin's B is L2-resident, so the ratio overstates the steady-state gap. The win lands by
   staging the raw f8 bytes through the slab and converting at the drain into fragments.
2. **Fused / composed forms decline → reduce tiers.** Norm→linear, gate-up, and merged siblings (per-part scale
   concat) don't reach the warp tier with an fp8 B yet.
3. **Bench-harness fp8 sources are broken.** The `run.py` reproducer draws f32 randoms per `source_path` (garbage
   when reinterpreted as f8 bits), and `torch_ref` has no `from_f8*` op mapping.
4. **bf16-activation W8A16** needs the `cuda_name`-based fragment-convert spelling — `type_name` has no bf16 entry
   (pre-existing gap).
5. **fp8 `ShapeKey` keys exist but goldens are unseeded** — pointless before (1); the greedy pick on the
   gmem-direct path is not the config the staged kernels will want.
6. **sm_89 verified numerically** (max_rel 2.5e-4): below sm_90 B arrives in-graph-transposed and the trans
   fragment helper carries the same per-element convert.

M3 adds to the same list: (7) the staged byte-slab drain must cover the k32 atoms too — no ldmatrix `.b8` below
sm_100a, so the drain is a cooperative byte-slab design, and it is the perf gate for W8A16 and W8A8 alike;
(8) the `_b8` loaders gather per byte — A/trans-B rows are 4-byte-contiguous and could vectorize to one u32 load
where alignment is provable; (9) masked-K (`kzero`) b8 variants deliberately absent — symbolic K stays off the
fp8 tier by legality; (10) e5m2 atoms registered and compiling but not numerics-verified.

## Current state (verified 2026-08-06)

- The dtype universe is f32 / f16 / bf16 (+ f16x2, i32/i64/bool). No fp8 anywhere in `emmy/`
  (`grep -rni "fp8|e4m3|float8"` hits only a comment in `context.py`).
- `compiler/dtype.py` is the one registry (canonical token + numpy carrier + nbytes); bf16 already establishes the
  bits-carrier precedent (uint16 pattern, `program.py` encodes on upload). CUDA traits live in
  `backend/cuda/dtype.py` (`_CUDA_NAME`, includes, `nbytes_of`).
- The mma atom registry (`ir/atom.py`) has exactly three m16n8k16 atoms (f16/bf16 × f32-acc, f16-acc). The naming
  convention comment already anticipates mixed four-slot atoms (`mma_<shape>_<ab>_<acc>`). `atoms_for()` filters by
  multiplicand dtype — a new dtype with no atom simply never reaches the warp tier (safe default).
- Loaders: `loader/safetensors.py` reads shards as numpy and runs each constant's `load_ops` chain through the numpy
  backend (`binder.apply_load_ops`) — the natural place for a dequant step. `bind_constants_from_module` casts
  everything to f32 numpy.
- Golden/search identity: `ShapeKey` carries `is_warp = dtype != "fp32"` and NO finer dtype split — f16 and bf16
  share a golden key today. Golden dtype spellings are `{"fp32","fp16","bf16"}` (`golden_eval._DTYPES`).
- Serving binds weights per the traced dtype; bf16 rides as bits through the numpy carrier.

## Target checkpoint formats

Both observed in the wild for the models we care about; detect via `config.json` `quantization_config`:

1. `quant_method: "fp8"` (Qwen official releases, e.g. Qwen3.6-27B-FP8): `fmt: e4m3`,
   `activation_scheme: "dynamic"`, `modules_to_not_convert` list. Weights stored as F8_E4M3 safetensors +
   `weight_scale` tensors.
2. `quant_method: "compressed-tensors"` (llm-compressor community quants): FP8 / FP8_DYNAMIC / FP8_BLOCK schemes,
   `weight_scale` per-channel `(N, 1)` or 2-D block.

Scale granularity is inferred from the scale tensor's shape relative to the weight: scalar → per-tensor, `(N, 1)` →
per-out-channel, 2-D smaller than weight → block. Support all three at ingestion; per-tensor and per-channel first
for compute (block scale inside the K loop is a separate, later problem).

## Design decisions

- **Canonical tokens**: `f8e4m3`, `f8e5m2` (numpy carrier `uint8`, nbytes 1); aliases `float8_e4m3fn`,
  `float8_e5m2` (torch spellings). CUDA names `__nv_fp8_e4m3` / `__nv_fp8_e5m2`, include `<cuda_fp8.h>`.
- **e4m3 → f16 conversion is EXACT** (every e4m3 value, max 448, is representable in f16; e5m2 likewise). So
  fp8-as-storage with f16 compute is not an accuracy knob — dequant-on-load and dequant-in-kernel are bit-equal to
  the reference dequant. Only *activation* quantization (M3) changes numerics and needs the FAST_MATH-umbrella
  gating per the precision-knob conventions.
- **W8A16 before W8A8.** Decode is memory-bound; halving weight bytes is where the win is. Native fp8 mma needs
  quantized activations too (both multiplicands are fp8) — that is a research-class milestone, not the critical
  path. Stock vLLM runs w8a8; emmy can compete from W8A16 storage + f16 mma first.
- **Golden identity must split on storage dtype class.** f16/bf16 sharing a key is fine (same bytes, same atoms);
  an fp8-stored weight is a different kernel family (half the B bytes, a convert in the staging path). Extend
  `ShapeKey` with a dtype-class field (default `""` for the 16-bit family so every existing golden/DB row keeps its
  key) rather than overloading `kind`.
- **Hardware floor**: fp8→f16 `cvt` and fp8 mma are sm_89+. sm_120 (5090 / Pro6000) adds the Blackwell mma forms.
  M1 (dequant-on-load) works on every arch; M2+ kernels gate on `sm_89`. Volta recipes (AWQ path) are out of scope.

## Type-system design — representing block quantization

The question is where a quantized tensor's structure lives: a weight is no longer one array but a *pair* (fp8 data +
a scale tensor), related by a block shape. Three layers, each deliberately minimal:

### 1. `DataType` layer: scalar tokens only — no quantized composite type

`F8E4M3` / `F8E5M2` enter as plain scalar `DataType`s (uint8 carrier, nbytes 1), exactly like `BF16` entered with a
uint16 carrier. We do NOT add a parameterized `Quantized(elem, scale_dtype, block)` type. Reasons:

- `dtype.py`'s contract is "generic + numpy information only". Every consumer — `nbytes_of`, staging math, TMA
  descriptors, structural stamps, render — assumes element type ⇒ fixed byte width. A composite dtype leaks block
  metadata into every dtype comparison, Tensor hash, and wire format.
- The block shape is a property of the *pairing of two tensors*, not of an *element*. Putting it on the element
  type is a category error — the same instinct the one-kind Fold IR rejects (structure lives in the term, readings
  are derived).

### 2. Graph layer: the scale is a first-class tensor; dequant is existing algebra

A quantized weight enters the graph as TWO constants — `W: f8e4m3 (N, K)` and `S: f32 (N/bn, K/bk)` — and dequant
is spelled with ops the tensor IR already has (the elementwise layer is generic named numpy-backed ops with
broadcast semantics):

```
dequant(W, S) = reshape( cast_f16(W).view(N/bn, bn, K/bk, bk) * S.view(N/bn, 1, K/bk, 1), (N, K) )
```

Every granularity is this ONE form with a different block shape — **granularity is derived, not declared**:
`block[i] = W.shape[i] // S.shape[i]`, recoverable from the two shapes at any point, no metadata field needed.

| scheme | scale shape | derived block |
| --- | --- | --- |
| per-tensor (`quant_method: fp8`) | `(1, 1)` | `(N, K)` |
| per-out-channel (llm-compressor) | `(N, 1)` | `(1, K)` |
| 2-D block (DeepSeek-style) | `(N/128, K/128)` | `(128, 128)` |

No new op kinds, no shape specializations — reshape + broadcast-multiply + cast is the whole vocabulary, and the
numpy interpreter evaluates it for free (reference backend, bind-time folding, accuracy A/B).

### 3. Metadata home: `ConstantOp.quant`

The one place that must know about the pairing *before* the graph is algebra is the load path — the scale has to
be found in the checkpoint. One new field on `ConstantOp`, stamped at the constant's birth site in the trace's
graph builder (see Phase 0 item 3) and consumed by the loader at bind time:

```
quant: QuantSpec | None   # (scale_path, scale_shape, scale_dtype, inverse: bool)
```

(`inverse` covers `weight_scale_inv` checkpoints.) `source_parts` (the sibling-linear concat) composes: each part
brings its own scale, concatenated on the same axis. Two consumers of the same spec:

- **M1** — never expands into the graph: the dequant chain is appended to `load_ops` and the numpy interpreter
  runs it at bind time (the chain already executes arbitrary frontend ops). The constant materializes at the
  compute dtype; kernels see nothing new.
- **M2** — a frontend normalization pass (`expand_quantized_constants`, beside the existing 00x passes) rewrites
  the constant into the algebraic cone above. `W` stays fp8 in memory; the cast + multiply ride the B-operand cone
  into the kernel — the computed-A precedent (a prologue on an operand cone, realized in-kernel instead of
  materialized).

### When the spelling happens — trace is quantization-blind

Quantization is spelled in three stages, none of them at torch-IR time:

1. **Trace (torch IR → frontend graph): never sees quantization.** The trace runs `torch.export` over a live
   module, and for a quantized checkpoint that module is the bf16 architecture twin built from config — the fp8
   tensors cannot even pass through torch without transformers' own fp8 kernels. Tracing the *quantized* module
   instead would capture whatever ops the HF quantizer implementation emits (custom triton ops, tensor
   subclasses) — hostage to their implementation details and version drift. Deeper reason: quantization is a
   property of the **checkpoint**, not the **architecture** — one traced graph serves the bf16, fp8-per-tensor,
   and fp8-block releases of the same model. The trace artifact stays shared; weights are plain `ConstantOp`s at
   the traced compute dtype.
2. **Graph construction: metadata attaches, structure unchanged.** The trace's graph builder reads
   `quantization_config`, pairs each weight with its scale, and stamps `ConstantOp.quant` when the constant is
   born (ops are immutable on the graph — see Phase 0 item 3). The graph is still structurally the unquantized
   graph; the loader consumes the spec at bind time.
3. **Frontend normalization: the algebra is spelled** by `expand_quantized_constants`, running LATE in the
   decomposition band — after `035_merge_sibling_linears` and the `050`/`060` constant folds, so the sibling
   concat (`source_parts`) and the layout chain (`load_ops`) have settled onto the plain `ConstantOp` those
   passes pattern-match; before the optimization band (whose cast passes, `005`/`007`, then get to massage the
   dequant cast like any other) and long before the `992` stamp, `ShapeKey`, and scheduling — so search identity
   sees the cone.

The invariant that makes the deferral safe: **the cone's output tensor has exactly the dtype/shape the trace
promised**, so every pass between trace and expansion is unaffected, and expansion is semantics-preserving up to
rounding.

One subtlety at stage 3: the constant's folded `load_ops` layout chain applies to the fp8 *bits*, and the scale
must receive the block-image of the same chain (a transpose of `W` transposes the scale grid). A layout op
commutes with dequant iff it maps whole blocks to whole blocks — true for the transposes `040_linear` produces;
a chain that crosses block boundaries (some reshapes under 2-D block scales) fails the check and that constant
falls back to bind-time dequant (the M1 path) instead of expanding. Graceful degradation per constant, never a
compile error.

### What the kernel does with the algebra — the commute split

The multiply's fate depends on whether the scale varies along the reduce axis, and this is where block-based
quantization differs *materially* from per-tensor/per-channel — the type system above is identical, the schedule
is not:

- **Scale constant along K** (per-tensor; per-out-channel): the multiply commutes out of the contraction —
  `Σ_k a·(s·w) = s·Σ_k a·w` — a mul-hoist rewrite over the fold, same reassociation category as split-K. It
  lands as an epilogue multiply on the f32 accumulator. Loop structure unchanged; the plain contraction reading
  and its whole existing schedule space apply untouched.
- **Scale varying along K** (2-D block): the multiply does NOT commute past the fold. The fold factors into two
  levels — `Σ_kb s[kb] · (Σ_{k∈kb} a·w)` — which the tile IR expresses natively as a nested fold. At the mma
  tier this is structurally the `FragmentPromote` slot the f16-accumulate atom already owns (periodically drain
  chunk partials into a shadow f32 accumulator): a K-blocked scale is the same periodic drain with a multiply.
  One legality predicate (in the `_legality.py` style): the K chunking must divide the quant block (bk | 128) so
  drains align with scale boundaries — constraining the knob space, not specializing codegen.

Accuracy note: M1 and M2 are NOT bit-identical — M1 rounds `s·w` per element into the compute dtype; M2's epilogue
form applies `s` once on the f32 partial (strictly *less* rounding). Same reassociation category as split-K, so it
rides the normal accuracy gate, not FAST_MATH.

### Identity / search fallout

- The dequant cone stamps `S_dtype_f8*`; a 2-D-block weight additionally stamps a second reduce level — the same
  ≥2-reduce-axis signature logic the `fused` kind uses. Per-channel W8A16 keeps the plain contraction reading (an
  epilogue mul doesn't change loop structure) and needs only the `ShapeKey` dtype-class field; whether K-blocked
  forms need their own golden `kind` is decided at M2 by whether their schedule space actually diverges.

### Why `QuantSpec` existed (deleted 2026-08-06 — the dissolve-early migration)

Originally kept because the decomposition band's constant-shaped assumptions (`035` merging, the `050`/`060` layout
folds) pattern-match a plain `ConstantOp`, and `load_ops` was a single-source chain while the cone has two sources.
The deletion path was then executed: (1) `load_ops` generalized past single-source via `ConstantOp.source_graph` —
an N-source bind record (the constant-only mini-graph itself, leaf `ConstantOp`s naming source paths) evaluated
through the numpy backend at bind time, with the trailing `load_ops` chain still composable AFTER it; (2) the
constant-matching passes needed no cone-transparency at all, because the generic `032_fold_constant_subgraphs` rule
collapses the cone BEFORE they run (and a folded record naturally fails `035`'s bare-source check). Stage 2 of the
spelling story is superseded accordingly: there is no metadata stage — the birth-time speller emits the stage-3
algebra directly, and the layout-commute machinery `180` carried died with it (at birth there is no chain to
commute past).

### Future-proofing: sub-byte (NVFP4 / int4)

The same three layers extend without new machinery:

- Packed sub-byte carriers enter as `StructuredType` (the `F16x2` precedent): e.g. `f4x2` — one byte holding two
  e2m1 values — keeping `nbytes` integral at the storage layer.
- NVFP4's two-level scaling (e4m3 block-16 scales + one f32 tensor scale) is two block-multiplies composed in the
  same algebra — composition, not new representation.
- Deliberately NOT built: a general quantized-tensor type carrying scale/zero-point/group metadata at the dtype
  level (the GGUF direction). An asymmetric zero-point, if ever needed, is one extra subtract in the cone.

### Concrete type-system deltas

- `dtype.py`: `F8E4M3`, `F8E5M2` + torch-name aliases (~10 lines).
- `backend/cuda/dtype.py`: CUDA names (`__nv_fp8_e4m3`), `<cuda_fp8.h>`, byte table entries.
- `ir/base.py`: `ConstantOp.quant: QuantSpec | None`.
- `loader/safetensors.py`: `quantization_config` → `QuantSpec`; fp8 shards read as uint8 bits + a numpy e4m3/e5m2
  decode helper (safetensors will not hand fp8 to numpy).
- Cast availability in the tensor IR's conversion op for `load_ops` / the cone (verify the exact spelling the 005
  cast handling uses when implementing).
- M2 only: `expand_quantized_constants` pass; f8→f16 `cvt` helpers in the CUDA render target; the `bk | block`
  legality predicate.

## Milestones

### Phase 0 — decomposition-band pre-work (audited 2026-08-06)

Audit result: the band is in good shape — small single-purpose rules, shared bodies already factored
(`_fold_constant.py`, `_helpers.py`). Three items are genuine pre-work; each is independently landable and
emission-neutral (verify with the kernel-source digest A/B, `scripts/digest_kernels.py`):

1. **`fold_into_constant` rebuilds the constant field-by-field** (`_fold_constant.py`): `ConstantOp(name=...,
   load_ops=..., source_path=..., source_parts=..., source_shape=..., source_dtype=...)`. Any new field —
   `quant` — is silently dropped by this copy. Switch to `dataclasses.replace(inp_x.op, load_ops=new_load_ops)`
   so new fields propagate by construction. This is the ONLY field-list reconstruction of a checkpoint constant
   in the compiler (verified repo-wide; `035`'s constructions build genuinely new merged constants, and the
   trace / flash / sdpa sites construct synthetic value constants). Fix the stale "004a/004b" numbering in the
   same file's docstring while there.
2. **Homogeneous-dtype stamping erases the weight's dtype identity.** `matmul_decompose` stamps ONE dtype
   (`a.output.dtype` or the caller's) on BOTH operand chains — unsqueeze, broadcast, multiply, reduce — and
   `040_linear` stamps the transposed weight with `out.dtype` (the activation-side dtype), not the weight's own.
   Benign today because every matmul's operands share a dtype; wrong the moment weight dtype ≠ activation dtype —
   exactly the M2 fp8-B situation, where the fabricated stamp would spell an fp8 tensor as bf16 and hide the cast
   from every downstream dtype gate. Refactor: layout/broadcast nodes inherit their INPUT's dtype (branch-local
   propagation); only genuinely computing nodes (multiply, reduce) take the result dtype. For today's
   uniform-dtype graphs branch-local ≡ blanket, so the digest gate must come back byte-identical.
3. **`QuantSpec` attaches at the constant's birth site, not post-hoc.** Ops are immutable once on the graph, so
   "the loader sets `ConstantOp.quant`" (stage 2 above) would mean node replacement after the fact. The trace's
   graph construction (`trace/torch.py`, the one non-synthetic `ConstantOp` birth site) already knows the model
   id and reads the checkpoint config — stamp `quant` there, from `quantization_config`, when the constant is
   created. Stage 2 of the spelling story is amended accordingly; stages 1 and 3 are unchanged (the torch trace
   still never sees quantization ops; the algebra still spells in the frontend band).

Recorded constraints (not refactors — facts the FP8 work must respect):

- **The `050`/`060` fold is sm_90-gated** (cp.async-path behavior preservation; matvec exception below sm_90).
  So a weight reaches its matmul in TWO forms depending on target: folded `load_ops` (sm_90+) or an in-graph
  `TransposeOp` → IndexMap (sm_89). `expand_quantized_constants` must rewrite at the constant only and rely on
  cast-commutes-with-layout — never assume the folded form. (The plan's stage-3 ordering rationale still holds
  on sm_90+; on sm_89 it is simply vacuous.)
- **Mixed-dtype operand demotion is the load-bearing M2 risk.** The dequant cast spells as an elementwise
  `copy` with a dtype boundary, and `007_sink_narrowing_cast`'s findings document what happens to a matmul
  whose operand cone carries one: no plain mma tier — the copy transports move raw bytes and cannot convert —
  so the op demotes to the `sync` compute-fill (measured on gemma-4: 1.12 vs 1.61 TB/s on the identical weight
  footprint). `007`'s producer-retype rescue cannot apply to a checkpoint constant (its bits are the checkpoint's).
  Consequence for M2 sequencing: the fragment-path convert (M2 step 3) is what makes fp8-B reach the warp tier
  AT ALL — land it before, not after, the staging/byte-width work is judged, or every fp8 A/B will measure the
  demoted tier and undersell the format.

### M1 — FP8 ingestion, dequant-on-load (correctness lane; no kernel changes)

1. `dtype.py`: add `F8E4M3` / `F8E5M2` + aliases; `backend/cuda/dtype.py` traits.
2. Loader: read `quantization_config`, pair each `weight` with its `weight_scale` (and honor
   `modules_to_not_convert`), and attach a dequant step to the constant's `load_ops` chain so the bound array
   comes out in the compute dtype. safetensors cannot hand fp8 to numpy — read the raw bits (uint8 view) and
   decode; add the e4m3/e5m2 decode as a numpy helper next to the bf16-bits precedent.
3. Trace: quantized checkpoints can't go through the live-module trace as-is (transformers would need its own fp8
   kernels). Trace the *architecture* at bf16/f16 from config, bind real weights via the safetensors path — the
   loader already bypasses the nn.Module round-trip. The eager reference for accuracy A/B loads the dequantized
   state dict into the traced module.
4. Verify: synthetic fp8 checkpoint fixture (tiny linear, all three scale granularities) → `compile`/`run`
   accuracy vs the dequantized torch reference; then one real layer of `Qwen/Qwen3.6-27B-FP8`.
   Digest gate (`scripts/digest_kernels.py`): kernels for non-fp8 models byte-identical before/after.

Deliverable: any FP8 checkpoint compiles and runs correctly everywhere emmy runs today (weights upcast in memory).

### M2 — FP8 storage through the kernel (W8A16: fp8 B operand, f16 mma)

1. Graph/IR: let a constant keep dtype `f8e4m3` end-to-end; buffers/uploads carry uint8 bits (bf16 precedent).
2. Staging: 1-byte elements through the B slab — cp.async vector widths, swizzle stride math, TMA descriptor
   (`CUtensorMap` U8 dtype) all currently assume 2/4-byte elements; audit `nbytes_of` call sites.
3. Fragment path: convert staged fp8 → f16 fragments before the existing m16n8k16 atoms
   (`cvt.rn.f16x2.e4m3x2` pairs), folding the per-tensor scale into the convert or the per-channel scale into the
   f32 epilogue on the C fragment (exact either way, see design note).
4. Search: `S_dtype_f8*` stamp in `992_stamp_structural_features`; `ShapeKey` dtype-class field; golden spellings
   `_DTYPES += {"fp8": "f8e4m3", ...}`; snippet/golden forms accept the dtype; new forks named per the
   precision-knob conventions (precise names; these are NOT accuracy knobs — no FAST_MATH gate).
5. Verify: per-kernel accuracy vs the dequant reference through the existing accuracy gate (NOT bit-parity with
   M1 — the epilogue-scale form rounds less, see the accuracy note above); `run --bench` on the golden matmul
   shapes at fp8 vs bf16 on 4090 (sm_89) and 5090 (sm_120); goldens seeded for the new keys via the manual pinned
   `--ab` method.

Deliverable: fp8-stored linears run on the warp tier with f16 compute — the decode-side memory win.

### M3 — Native fp8 mma (W8A8) — optional, research-class

- Atoms: `mma_m16n8k32_e4m3_f32` (+ e5m2), sm_89+ PTX; sm_120 forms verified separately.
- Dynamic per-token activation quant = an amax statistic prologue feeding the contraction — structurally the
  computed-A shape the fused golden kind already models; that machinery is the natural home.
- Epilogue: `act_scale ⊗ weight_scale` on the f32 accumulator.
- Accuracy-changing → FAST_MATH-umbrella knob, off by default; combinatorial tests per the precision conventions.
- Only worth starting once M2 A/Bs show the remaining gap to stock vLLM fp8 is mma-rate-bound, not bytes-bound.

### M4 — Serving + model-level integration

- `gen_runner` / plan constants: fp8 bits carrier on upload (uint8), pack-key includes the dtype class.
- `emmy serve --generate` a real FP8 checkpoint; A/B vs stock vLLM fp8 on the same card.
- Recipes: nothing needed — recipes already point at FP8 repos; this makes the emmy-compiled path consume them.

## Open questions

- Whether Qwen3.8-27B keeps `model_type: qwen3_5` (evidence says likely; 3.5→3.6 changed zero architecture
  fields). Does not affect this plan — only the model-level follow-up.
- Block-scale (128×128 DeepSeek-style) in-kernel: scale changes along K → per-K-chunk multiply inside the fold;
  deferred past M2 unless a target checkpoint needs it.
- `lm_head` at vocab 248320 in fp8: largest GEMM, biggest storage win, but also the accuracy-sensitive one —
  check whether reference releases keep it unquantized (`modules_to_not_convert`).
