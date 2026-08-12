# Backend Architecture

Three backends with a shared interpreter. Every backend exposes the
same two-step API (`backend/base.py`):

```python
compiled = backend.compile(graph)
result   = backend.run(compiled, input_data={"x": np.ndarray(...)})
# RunResult(outputs: dict[name, ndarray], time_ms: float | None)
bench    = await backend.benchmark_async(compiled, warmup=, num_iters=)  # async-only; callers asyncio.run at the CLI boundary
# BenchmarkResult(time_ms, min_ms, max_ms, num_launches, per_launch)
```

What differs is `compile()` — how far the backend lowers the graph:

| Backend        | `compile(graph)` does                              | `run()` goes through |
|----------------|----------------------------------------------------|----------------------|
| `NumpyBackend` | returns the graph as-is (no-op)                    | default `Backend.run`|
| `LoopBackend`  | runs decomposition → optimization → fusion         | default `Backend.run`|
| `CudaBackend`  | fusion + `lowering/kernel` + `lowering/cuda`       | cupy/NVRTC dispatch  |

`numpy` and `loop` backends share the same runtime path — the only
distinction is whether the graph has been fused yet. See
`backend/cuda/ARCHITECTURE.md` for the CUDA-specific dispatch.

## Torch reference (`torch_ref.py`)

Not a `Backend` — a small Graph→torch evaluator that runs a frontend-dialect graph through **real PyTorch**, the eager /
`torch.compile` baseline for `emmy run --ir`. Each frontend / tensor op is mapped to its torch twin
(`RmsNormOp`→`F.rms_norm`, `LayerNormOp`→`F.layer_norm`, `SdpaOp`→`F.scaled_dot_product_attention`,
`LinearOp`→`F.linear`, `ElementwiseOp`/`ReduceOp`→the torch elementwise/reduce, layout ops→view/transpose/cat).
FP8 tensors remain exact `uint8` bit carriers; `to_f8*` casts to torch float8 and reinterprets its storage, while
`from_f8*` performs the inverse reinterpretation before widening. `is_runnable(graph)` is `True` only when every
compute op and every elementwise operation name has a mapping. Data-dependent `GatherOp` / `ScatterOp` remain
unsupported, so `run --ir` falls back to emmy-only benchmarking for those graphs. `build_callable(graph,
input_tensors)` returns a pure `fn(*tensors)` (scalar constants read inline) so `torch.compile` can trace it. Symbolic
graphs work too: `build_callable` binds every symbolic axis name to its concrete extent read off the supplied tensors
(the CUDA launch convention) and bakes the env into the per-node callables — shape-resolving sites (`ReshapeOp` target
shape, `IndexMapOp` out-shape and coord/select exprs) eval through it, so a dynamic frontend provenance slice
gets the same vs-torch comparison as a static one (benched at the `Dim` hint by
`commands/run.py::bench_lowered_vs_torch`, which sizes its random inputs by hint-resolving symbolic dims). Used to
benchmark decoded golden programs and in-memory provenance slices against torch.

The strict run path evaluates Emmy and eager on the same inputs and returns a direct `rtol=atol=1e-3` proof with
error statistics. The isolated CUDA worker transports that proof and the eager reference outputs back to the parent,
so exact-pinned rows can be checked against the same reference rather than against another Emmy realization.

## Backend ABC (`base.py`)

`Backend` is an abstract base class with `compile`, `run`, `benchmark`.

`Backend.run` provides a default implementation: walks the compiled
graph in topological order and calls `node.op.forward(*args)` at each
compute node. `InputOp` and `ConstantOp` boundaries are seeded from
`input_data`; values are stored **per buffer** — a multi-output node's
`forward` returns a tuple matched positionally to `node.outputs`, and
each result is reshaped to its own output tensor's shape when statically
known.

Every op implements `forward(*inputs: np.ndarray) → np.ndarray`,
including `LoopOp`: its `forward` delegates to `ir/loop/runner.py`
(`execute_loop_op_cpp`), which renders the body to C++ and JIT-compiles
it in-process via cppyy / Cling. That's why the same default `run` works
for pre-fusion graphs (LoopOp absent) and post-fusion graphs (LoopOp is
just another `Op` subclass).
`NumpyBackend` and `LoopBackend` inherit it verbatim; `CudaBackend`
overrides with cupy dispatch.

The default `benchmark` does wall-time iterations around `run`; the
CUDA backend overrides it to populate per-launch CUDA-event timings.

Result dataclasses:

- `RunResult(outputs, time_ms)`
- `BenchmarkResult(time_ms, min_ms?, max_ms?, num_launches, per_launch?, captured, e2e_ms?, e2e_min_ms?)` —
  `time_ms`/`min_ms` sum per-launch solo windows; `e2e_ms`/`e2e_min_ms` (automatic for multi-launch programs
  under capture) time the whole program as replays of one all-launches CUDA graph — the only
  end-to-end-comparable number for multi-kernel programs.
- `LaunchTime(idx, kernel_name, time_ms)` — one per kernel per bench run.

## Numpy backend (`numpy/`)

Thinnest backend. `compile` returns the graph; `run` is inherited from
`Backend`. Used for correctness testing (no GPU required) and as the
ground truth the loop and CUDA backends are triangulated against.

## Loop backend (`loop/`)

Runs the fusion pipeline to turn the graph into `Graph[LoopOp]`, then
executes via the inherited `Backend.run`. Since `LoopOp.forward` works,
this is the numpy backend with fusion in front.

Used as the second axis of triangulation: **loop vs numpy disagreement
implicates fusion; loop vs CUDA disagreement implicates codegen.**

## CUDA backend (`cuda/`)

See `cuda/ARCHITECTURE.md`. Runs the full lowering chain and dispatches
kernels via cupy `RawKernel` (NVRTC-compiled).

## Execution plan + pack (`plan.py`, `pack.py`)

`plan.py` defines the **execution plan** — the serializable runtime projection of a lowered `Graph[CudaOp]`:
buffer specs (one `BufferSpec` per BUFFER — a multi-output node mints one per output slot, each with its own
role via `graph.buffer_role`), scalar/runtime constants, the launch list (`LaunchSpec.writes` names every
buffer a launch produces; the slab planner's first-write test reads it, falling back to `node_id` for plans
stored before the field existed), symbolic-axis plumbing, kernel refs (source and/or a content-addressed
cubin-cache key), and per-weight checkpoint bindings (`source_path` + a pack-own load-op vocabulary applied
with pure numpy). `plan_from_graph` is the seam the whole runtime builds from: after it runs,
nothing reads the graph again — `CompiledProgram.build(graph)` is exactly `build_from_plan(plan_from_graph(g))`,
so a plan loaded from disk and a freshly compiled one share one launch path. The JSON form (`plan_to_dict` /
`plan_from_dict`) carries symbolic shapes and ceil-div grid factors through a tiny self-contained expression
grammar (`int` literal, `"name"` var, `[op, lhs, rhs]`), deliberately not the compiler's `Expr` classes — the
on-disk format survives compiler changes; only runtime-contract changes bump `PLAN_FORMAT_VERSION`.
CUDA-specific launch fields (TMA descriptors) nest under a `"cuda"` key so another backend can add its own
namespace and its own `build_from_plan` equivalent.

Static by-value kernel parameters use `LaunchSpec.scalar_args`, with the scalar name retained at its ABI position in
`arg_names`. Only CUDA's primitive integer and floating-point types are accepted. Plans carrying these arguments use
format 4, and the source, typed values, launch geometry, and buffer shapes all remain part of the normal plan/pack
identity. Symbolic `runtime_args` remain a separate tail ABI resolved from the request shape.

`plan_cache.py` is the process-local reuse seam for repeated compiled structure within one immutable compile session.
It keys the exact graph wire form after loader spelling and ABI hints, erasing only external tensor addresses while
preserving their alias pattern and `source_parts` order. The stored `ExecutionPlan` template carries binding slots;
every lookup returns a fresh plan whose `WeightSpec`s contain that graph instance's real paths. Unknown compiler-minted
paths or unresolved slots fail closed, and only these instantiated plans may reach source loading or pack serialization.
This is deliberately not a persistent cache: `pack.py` owns cross-process environment/model validity.

**Indirect operands** (`LaunchSpec.indirect_args`, `(arg, table_arg, sel_arg, slot)` per marked input): the
kernel takes `const T* const* <arg>__table, const int* <arg>__sel, int <arg>__slot` in place of the plain
pointer and resolves `table[sel[slot]]` in a body preamble — the serving MoE fixed-slot dispatch, where the
weight base pointer comes from a device table indexed by the router's indices tensor. The change is ABI-level
only: it enters as the `cuda.indirect_inputs` graph hint (set by the caller before compile, read by the final
kernel lowering), so schedules, goldens, and the tile search are untouched, and non-indirect kernel sources stay
byte-identical (`scripts/digest_kernels.py` is the gate). `arg_order` keeps the plain operand name; `_launch`
expands it in place to `arrays[table], arrays[sel], slot`. An indirect operand staged through a TMA descriptor
fails the lowering loudly (descriptors bake the base address at encode). A plan carrying the field serializes as
`PLAN_FORMAT_INDIRECT` (2) — a runtime that ignored it would pass the wrong arg pack, so old readers reject such
a plan and fall back to the full compile; plans without the field keep format 1 byte-compatibly.

`pack.py` bundles plans on disk: one directory per model × GPU × serving shape holding `manifest.json` (validity
key + environment tags + provenance + program index) and `plan/<program>.json`. The validity key is composed by
the *runner*, and "model" there must cover everything the compiled programs read off the CHECKPOINT — not just its
architecture config. A compressed checkpoint is the case that makes the difference: two rungs of one conversion
share an architecture config and differ only in the per-tensor rates, which set the coded extents, so the runner
adds the loader's checkpoint digest (`loader.quant.checkpoint_quant_digest`) to the key.
Cubins are **not** copied — plans
reference the shared `EMMY_CUBIN_CACHE` by content-addressed key, so packs dedupe kernels against each other and
the docker bake ships pack + cubin cache + model snapshot together. `load_pack` returns `None` on *any*
disqualifier (format/environment/key mismatch, unparsable plan, evicted cubin) and the caller falls back to the
full compile — a stale pack costs a recompile, never a wrong result. The serving boot integration
(`EMMY_PACK_DIR`) lives in `emmy/serving/runner.py`; see `emmy/serving/ARCHITECTURE.md`.

## Invariants

- The default `Backend.run` must work on any graph where every op has
  `forward`. It doesn't know about dialects — it just dispatches
  through `Op.forward`.
- A new backend (ROCm, SYCL, Metal) reuses `ir/` and
  `pipeline/passes/lowering/` wholesale; only its own dispatch layer
  (equivalent of `cuda/program.py`) needs to be written.
