# Execution-plan pack: a serializable compiled-program format for fast serving boot

## Problem

On a fully-warm boot (cubins cached, goldens/prior/DB seeded), serving still redoes the entire compiler frontend
every time: HF model load, torch.export trace, the full CUDA pass pipeline, greedy fork resolution, and CUDA C
codegen for every kernel — codegen is unavoidable today because the source string *is* the cubin-cache key. The
only skipped stage is the nvcc/cicc/ptxas subprocess (`nvcc.py` cache hit). Measured costs: ~1–2 min for the 0.6B
embedding model warm; the gemma-4 generative path compiles ~96 programs (pre/post × 48 layers + decode/prefill
twins) through that chain per boot; a bloated online file once stalled a boot 15 min in pure resolution. The
a1145d00 serve-boot memos (prior parse, DB perf index) amortize only *within* one process — nothing persists.

## Key insight: serialize the runtime projection, not the Graph

`_Compiled` (`backend/cuda/program.py`) is the compiler↔runtime seam. After `_compile(graph)` runs, nothing
downstream ever touches the Graph again: slab liveness is computed from the launch list
(`_plan_slab` → `compute_live_intervals(sizes, compiled.launches)`), allocation from the buffer specs, launch
resolution from the symbolic bindings. `_Compiled` is pure data except the loaded kernel modules:

- `bufs`: name, shape (`Dim` — static int / symbolic var / composite expr), dtype, role
- `constants: {name: float}` (scalars are NOT inlined into kernel source — value-independent codegen),
  `runtime_constants: {name: Expr}`
- `launches`: kernel name, arg names, grid/block (`GridDimSpec` — factors are int | sym-name | `Expr` ceil-div),
  smem, zero_outputs, `TmaDescMeta` tuple, runtime_args
- `symbolic_bindings` / `symbolic_hints` / `symbolic_caps`
- `kernels: {name: RawKernel}` — the only live objects; rebuilt at load

Weights are referenced by `ConstantOp.source_path` (matches `state_dict` keys verbatim) and bound late — never
baked into codegen. So a serialized plan is re-bindable from any same-config checkpoint (fine-tunes share packs).

## Decisions (from the design discussion)

1. **Reuse the existing cubin cache; reference binaries by content-addressed key** (the sha1 from
   `nvcc._cache_key`), never by filesystem path. Multiple packs (and the ~2 layer types × 48 layers within one
   pack) dedupe to the same cubin files automatically.
2. **Missing/invalidated cubin → recompile.** The pack loader falls back to the full frontend compile path when a
   referenced cubin is absent or the manifest key mismatches. No embedded kernel sources in the pack.
3. **Stack-independent format.** The plan schema carries only tensors/launches/symbols/binary-refs plus a tiny
   self-contained expression grammar — no compiler IR classes, no op types. Compiler changes (new passes,
   re-tunes, refactors) do NOT invalidate a pack; it keeps serving its frozen snapshot. What versions the format
   is the *runtime contract* (arg-passing convention, `runtime_args` by value, `zero_outputs` semantics,
   bf16-bits-as-uint16 encoding, TMA descriptor construction) — changes there bump the format version.
4. **Validity key** = format version × model id + `config.json` hash × serving shape (max_model_len, batched-token
   capacity, decode/prefill bucket, dtype) × {backend, arch `sm_XX(+a)`, toolkit tag, nvcc flags}. Compiler git
   rev, goldens/prior state, tune date are recorded as *provenance metadata only* — informational, not validity.
5. **Backend-extensible (ROCm later).** Neutral core (buffers, symbols, constants, launches, grid/block/smem);
   CUDA-specific fields (TMA descriptors, `sm_XXa` tag) live in a namespaced `backend.cuda` section. Loaders are
   polymorphic: `from_plan(plan, weights, arena) → program` dispatched on `manifest.backend`. A ROCm pack changes
   the manifest backend/arch, the binary keys (hsaco cache analog), and supplies its own loader. No ROCm code now
   — just the namespacing.
6. **No code duplication: the main launch path goes through the plan.** `CompiledProgram.build(graph)` becomes
   `build_from_plan(plan_from_graph(graph))`; the pack loader calls the same `build_from_plan`. One runtime path
   whether the plan came from a fresh compile or from disk.
7. **Frozen picks are a feature.** A pack sidesteps the known cross-process fork-pick nondeterminism (the bimodal
   picks that force `warm.sh`'s fixpoint loop) — picks are frozen in the artifact. Staleness after an
   `emmy tune` is accepted; regeneration is the answer (slots into the warm/bake image flow).

## Pack layout

```
pack/
  manifest.json          # validity key + provenance + program index
  plan/<program>.json    # serialized plan per program (embed, L00.pre.sym, L00.pre.decode, …)
```

Cubins stay in the shared `EMMY_CUBIN_CACHE` (content-addressed); weights stay external (the HF checkpoint),
bound by `source_path` at load.

**Docker image = model + cubins + pack.** The `docker/vllm-emmy-gemma4` build already bakes the HF model
snapshot (`warm/hf`) and the cubin cache (`warm/cubin`) into the image; the pack joins them as a third baked
artifact (`warm/pack`), so a container cold-start ships everything: weights (no download), cubins (no nvcc), and
the pack (no trace/pipeline/resolution/codegen). `warm.sh` emits the pack during its serving run; the Dockerfile
`COPY`s it next to the cubin cache; `verify.sh`'s zero-recompile check tightens to zero-frontend.

### Plan JSON schema (per program)

- `buffers`: `[{name, shape: [dimexpr…], dtype, role}]`
- `constants`: `{name: float}`; `runtime_constants`: `{name: dimexpr}`
- `launches`: `[{kernel, args: [name…], grid: [[factor…]×3], block: [[factor…]×3], smem, zero_outputs,
  runtime_args, cuda: {tma: [{name, src_buf, box_extents, swizzle}]}}]`
- `symbols`: `{bindings: {sym: [input_buf, dim_idx]}, hints: {sym: int}, caps: {sym: int}}`
- `kernels`: `{name: {binary_key, uses_tma}}`

**Expression grammar** (dims, grid factors, runtime constants): JSON prefix form — `int` literal, `"name"` var,
`[op, lhs, rhs]` with `op ∈ {"+", "-", "*", "//"}`. This is the only non-plain data in the format; defining it in
the format (not reusing compiler `Expr`/`Dim`) is what buys stack independence. The loader maps it back to the
runtime's expression objects. (Bonus: sidesteps the known Graph-JSON bug where `Expr`-valued grid factors don't
round-trip — the pack never serializes the Graph.)

## Implementation steps

1. **`ExecutionPlan` dataclasses + `plan_from_graph`** (`emmy/compiler/backend/plan.py`): extract the pure-data
   half of `_compile` (buffers, constants, launches, symbols, kernel specs `{source, uses_tma}`); `_Compiled`
   becomes plan + loaded kernels. → verify: existing CPU tests green; `_compile` behavior unchanged.
2. **Plan serialization** (`plan_to_dict` / `plan_from_dict` + the expression grammar): round-trip every field
   including `Expr` grid factors, composite `Dim` shapes, `TmaDescMeta`. → verify: new round-trip tests, no GPU
   needed (property: `plan_from_dict(plan_to_dict(p)) == p` over representative graphs, incl. symbolic SDPA).
3. **Wire the runtime through the plan**: `CompiledProgram.build(graph, …)` = `plan_from_graph` +
   `build_from_plan`; kernel loading accepts `binary_key` (direct cubin load — factor a load-by-path helper out of
   `nvcc.load_function`) with source-compile as the from-graph path. → verify: full GPU suite green (`make test`),
   no behavior change.
4. **Pack save/load** (`emmy/compiler/backend/pack.py`): manifest write (compose the key from `nvcc._cache_key`
   components + model config hash + serving-shape params), save resolves each kernel's cubin key after compile;
   load validates the key, builds each program via `build_from_plan`, and **falls back to the frontend compile on
   any missing cubin / key mismatch** (decision 2). → verify: save→load→run matches compile→run on a small model;
   corrupt/missing-cubin test exercises the fallback.
5. **Serving integration**: runners (`EmmyForwardRunner.create`, `EmmyGenRunner.from_model`) check a pack path
   (env var, e.g. `EMMY_PACK_DIR` via `config.py`) before the trace-and-compile path; emit-on-boot so the warm
   flow produces the pack (upgrade `docker/vllm-emmy-gemma4` bake from cubins-only to pack; `verify.sh` becomes a
   zero-frontend check). Loading weights directly from safetensors (skip `AutoModel` instantiation) is the
   follow-on win once the pack removes the trace. → verify: A/B a warm serve boot with/without pack on a GPU box;
   embeddings/generations byte-match; boot time drops to ~weight-load + module-load + alloc.
6. **Docs**: `backend/cuda/ARCHITECTURE.md` (plan/pack format, loader), `serving/ARCHITECTURE.md` (boot path),
   `CLAUDE.md` env-var note via `config.py`.

Steps 1–3 are pure refactor + serialization, testable without a GPU. Step 4–5 need a CUDA box for the A/B.

## Remaining per-boot costs after the pack (accepted)

`cuModuleLoad` (~25 ms × unique kernels), weight load + upload, buffer/arena allocation, TMA descriptor rebuild
(bakes device pointers), CUDA graph capture on first request per seq_len. All inherent to a fresh process/GPU.
