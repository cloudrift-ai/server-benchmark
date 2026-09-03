# Rules Test Architecture

## Purpose

Every graph transformation rule **must** be tested for numerical correctness,
not just structural properties. The pattern:

1. Build a graph containing the target op with concrete shapes.
2. Run the graph through `NumpyBackend` → **before** values.
3. Apply the decomposition rule via `Pass.apply()`.
4. Run the rewritten graph through `NumpyBackend` → **after** values.
5. Assert `before ≈ after` with `np.testing.assert_allclose`.

This catches semantic bugs that structural tests (checking which ops are
present) cannot: wrong axis in a reduction, swapped operands, missing
scale constant, incorrect coordinate mapping, etc.

## File Layout

```
tests/compiler/passes/
├── conftest.py                     # RecordingDump fixture for rule-fired assertions
├── test_decompose_rules.py         # decomposition rules (structural + correctness)
├── test_optimization_rules.py      # optimization rules (structural + correctness)
├── test_fusion_rules.py            # maximal/multi-output fusion structure and Loop-runner correctness
├── test_matcher.py                 # Pattern matcher unit tests
├── test_maximal_fusion.py          # one-pass maximal fusion, including nested reductions
├── test_twisted_rewrite.py         # general exp-family Tile rewrite: softmax and masked/unmasked SDPA
├── test_matmul_rules.py            # matmul-specific rewrite rules
├── test_reduction_rules.py         # reduction-pattern rewrite rules
├── test_register_tile_rules.py     # register-tile lowering rules
├── test_partition_planner_rules.py # partition-planner pass
├── test_partition_planner_forks.py # partition-planner fork generation
├── test_launch_geometry_rules.py   # launch-geometry pass
├── test_move_catalog.py           # schedule catalogs, site trees, and independent-root compatibility
├── test_cut_forks.py              # fused/cut Fold-edge offers and pinned CUDA lowering
├── test_placement_routing.py       # frontend placement pins, routing rows, and MIMO preservation
├── test_split_fresh_kernels.py    # generic cross-CTA Fold splitting and fresh-piece invariants
├── test_masked_tile.py             # masked-tile pass (dynamic-shape boundary guard)
├── test_stage_inputs_classify.py   # Stage-input classifier
├── test_lowering_accuracy.py       # 040 / 060 / 070 + TMA end-to-end CUDA accuracy
├── test_knob_pinning.py            # EMMY_KNOBS-pinned regression configs (article-reproduction tile/transport sweep)
├── test_warp_specialize_deadlock.py # WS=1 stranded-TMA deadlock (Qwen3 k_linear_mean_reduce) regressions
├── test_tile_naming.py             # provenance-driven k_<op>_<suffix> kernel naming
├── test_shared_constant_cone.py    # one broadcast constant, two sibling cones — one declaration per scope
└── test_pipeline_semantics.py      # full pass chain (decompose → opt → fuse) vs numpy
```

`tests/compiler/helpers.py` exposes `matmul_graph(m, k, n)` — the shared (m,k)@(k,n)→(m,n) graph builder used by the
lowering / backend / e2e tests — plus the `requires_cuda` skip marker. `tests/compiler/conftest.py` owns the
`run_graph` parametrized fixture.

## Covered Rules

### Decomposition (`passes/frontend/decomposition/`)

| Rule file          | Op                      | Structural | Correctness       |
|--------------------|-------------------------|------------|-------------------|
| `010_sdpa.py`      | `SdpaOp`                | ✓          | ✓                 |
| `020_silu.py`      | `ElementwiseOp("silu")` | ✓ (f16/bf16 opmath; f32/f64 controls) | ✓                 |
| `030_pow.py`       | `ElementwiseOp("pow")`  | ✓          | ✓                 |
| `040_linear.py`    | `LinearOp`              | ✓          | ✓ (± bias)        |
| `070_matmul.py`    | `MatmulOp`              | ✓          | ✓ (± bias)        |
| `090_mean.py`      | `MeanOp`                | ✓          | ✓                 |
| `110_unsqueeze.py` | `UnsqueezeOp`           | —          | ✓ (dim=0, dim=-1) |
| `120_transpose.py` | `TransposeOp`           | —          | ✓                 |
| `130_reshape.py`   | `ReshapeOp`             | —          | ✓                 |
| `140_slice.py`     | `SliceOp`               | —          | ✓                 |
| `150_cat.py`       | `CatOp`                 | —          | ✓                 |

### Optimization (`passes/frontend/optimization/`)

| Rule file                          | Op                          | Structural | Correctness                       |
|------------------------------------|-----------------------------|------------|-----------------------------------|
| `002_insert_broadcast_indexmap.py` | `ElementwiseOp` (broadcast) | ✓          | ✓ (1D, scalar, 3D, RMSNorm chain) |

### Fusion (`passes/loop/lifting/` + `passes/loop/fusion/`)

Lifting wraps each surviving tensor primitive (elementwise / reduce / scan / indexmap / gather) in a trivial
single-op `LoopOp`. Fusion takes a maximal downstream region; separate terminal branches become output ports, and
one splicer worklist inlines common producers once across all roots. `test_fusion_rules.py` runs lifting followed by
fusion as a single pass; `tests/compiler/ir/loop/test_splicer.py` covers the multi-root worklist and scope rules
and output equivalence clusters directly, while the pass tests exercise the resulting graph through Loop and CUDA
lowering.

| Rule file                              | Op                         | Tested via                                                                         |
|----------------------------------------|----------------------------|------------------------------------------------------------------------------------|
| `loop/lifting/010_lift_elementwise.py` | `ElementwiseOp` → `LoopOp` | `test_fusion_rules.py` (pass fixpoint)                                             |
| `loop/lifting/020_lift_reduce.py`      | `ReduceOp` → `LoopOp`      | `test_fusion_rules.py::test_contraction_*`                                         |
| `loop/lifting/025_lift_scan.py`        | `ScanOp` → `LoopOp`        | `test_pipeline_semantics.py::test_scan_*`                                          |
| `loop/lifting/030_lift_indexmap.py`    | `IndexMapOp` → `LoopOp`    | `test_optimization_rules.py::test_matmul_with_transpose_fuses_to_one_kernel` (e2e) |
| `loop/lifting/040_lift_gather.py`      | `GatherOp` → `LoopOp`      | `test_torch_ops.py::test_gather`                                                   |
| `loop/lifting/090_spell_store_rounding.py` | public narrowing boundary → typed `copy` | `test_spell_store_rounding.py`                             |
| `loop/fusion/010_merge_loop_ops.py`    | `LoopOp → LoopOp` (splice) | `test_fusion_rules.py` (fixpoint)                                                  |

Numerical correctness for lifted + merged kernels runs through the
numpy backends in three places:

- `test_fusion_rules.py::test_*_correctness` — runs the pre- and
  post-fusion graph through `NumpyBackend` (which uses `LoopOp.forward`
  post-fusion) and asserts outputs match.
- `tests/compiler/e2e/test_accuracy.py` — full-pipeline coverage on
  toy shapes (pointwise, reduce, matmul, RMSNorm, softmax) via the
  `run_graph` fixture parameterized over `numpy` / `loop` / `cuda`.
- `tests/compiler/e2e/test_block.py` — real transformer block (TinyLlama
  layer 0 with random weights, `seq_len=8` for the CPU lane, `seq_len=32`
  for the CUDA lane) compiled end-to-end and compared against PyTorch
  eager. The `_cpu` variant runs `LoopBackend` + CPU eager (always
  on, ~3s); the `_cuda` variants are gated by `@requires_cuda`.

### Tile lowering (`passes/lowering/tile/`)

`test_twisted_rewrite.py` traces softmax, SDPA, and causal SDPA through total lift and the same `020_twisted` rule,
then checks the resulting carrier arity, the derived contraction sites, and that plain and causal SDPA reach both MMA
sites through the CUDA pipeline. The direct projection boundary exhaustively compares its production rows with the
literal compatible subset of the independent Cartesian product before checking materialization. The plain-reduction
boundary also compares the production set with that literal reference: its independent product contains mismatched
node and kernel worker choices that only the compatibility relation may reject. `test_schedule_walk.py` pins the
target enumeration contracts — independent node and edge domains, traversal-order-invariant compatible membership,
computed and derived folds keyed as schedule sites, and exact pins — without flattening a live space into test memory.
The scalar-contraction boundary likewise compares production with the literal reference, proves the independent
product is larger than the accepted set, and checks that a selected output tile alone produces placed materialization
geometry.
The gmem-direct tensor-core boundary repeats that proof over a typed f16 contraction and asserts that the independent
kernel domain contains both warp inventories and raster choices. It also proves that compatibility rejects grouped
rasterization beside every untiled node choice.
The staged-edge boundary limits the transport catalog to one copy choice, projects that choice independently onto both
operand edges, and compares production against the literal product. Only equal edge choices survive compatibility,
and expanding a selected staged row derives one `ResolvedStage` per edge without changing the typed schedule.
The compute-fill boundary makes the same bounded comparison with direct, `d1/smem`, and `d2/smem` edge factors. A
warp node that needs the fill accepts only equal smem edges, while scalar support keeps direct transport in each
independent public factor.
The schedule-restriction boundary proves that exact `WORK` and addressed `TILE` parameters leave every independent
factor unchanged, then compares production with Algorithm 1(c, p, t) under the same immutable `c`. The production
driver carries that context intact. The context may prune a prefix when the restriction and compatibility state prove
that it has no accepted completion; an opaque predicate exposes no such proof and is observed only at complete
assignments. The test deliberately bounds the factor catalogs so the literal oracle remains fast. A complete `c` also
proves its singleton without changing a factor. Composed GPU cases cover nested and sibling fragment agreements; no
composed-only enumerator or post-product membership rule exists.
The structural-split boundary proves that the outer pass consumes a pinned GRID stage before constructing `c` for a
fresh piece. The piece's schedule restriction therefore compares only the remaining schedule stages. Sampled walks
and composed cross-CTA split pieces exercise the same enumeration, and the duration baseline records their bounded
test cost.
The producer-band boundary projects uniform, `+p1`, and `+p2` kernel choices before reading parameters, proves an exact
`WORK` parameter leaves that domain unchanged, and checks that only compatible TMA edge assignments survive.
The shared-constant cone fixture also pins a multi-channel contraction to the scalar tier: every channel remains in one
serial Fold body, so independently spliced operand cones share the one legal broadcast binding.
`test_move_catalog.py` also pins the pure-register, one-dimensional thread, parallel-register, and cooperative-width ×
ILP catalog products, then verifies that a matching `WORK` pin selects only existing rows and an unmatched pin returns
no row, so restriction cannot manufacture a worker inventory. Precision gates restrict atom choices that remain in the
fixed domain; exact raster parameters likewise cannot add a value outside the static raster catalog.
`test_schedule_pool_cache.py` pins the session memo:
sharing, hit equality, read-only payloads, and the keying that holds pin states, dtypes, extents, stores and the
sample apart. `test_move_catalog.py` checks that independent
roots with reversed M/N readings combine only when their tile
widths and unit counts match on the physical output axes, and that f32 computed-A contractions retain scalar
output-tile rows when no MMA atom applies. `test_cut_forks.py` proves that `040_schedule` does not call the schedule
enumerator while an undecided cuttable seam remains, and calls it once placement is decided. It also checks fused and
closed Fold-edge choices for SDPA score
production, causal SDPA, and multi-output roots, then pins each representative cut through CUDA lowering, and proves
child-identity schedule receipts round-trip: under a pinned cut each child's stored identity decodes only its own
kernel's schedule rows and keys its evidence row by that identity, including when target-boundary drift makes the
regenerated
Loop target contain several kernels and the stored identity must select one. Direct
contraction-operand cuts remain strict xfails until Tile IR represents their materialized workspace dtype.
The recipe program's monoid laws are covered
independently by `tests/compiler/ir/pure/test_twist.py`; end-to-end softmax and attention accuracy remain covered by
the e2e suites.

## Adding a New Rule Test

When adding a new rewrite rule, add both test types in `test_decompose_rules.py`:

```python
def test_<op>_decomposes():
    """Structural: verify the original op is gone and expected ops are present."""
    ...

def test_<op>_correctness():
    """Numerical: verify before == after through the numpy backend."""
    g = _make_<op>_graph()
    inputs = {"x": rng.standard_normal(...).astype(np.float32)}
    before = _run(g, inputs)
    after = _run(_apply(g, _load("<rule_file>.py")), inputs)
    _assert_close(before, after)
```

Use small concrete shapes (avoid symbolic dims) so the numpy backend
can execute the graph. `IndexMapOp.forward` iterates in Python, so keep
tensor sizes under ~1000 elements for fast tests.
