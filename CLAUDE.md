# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Emmy is a Python tool for deploying and benchmarking LLM inference on GPU servers. It supports vLLM and SGLang engines, provides a CLI for local and remote (SSH) deployment of models via Docker Compose, plus automated benchmarking across multiple servers.

The `README.md` is intentionally short — example-driven, no narrative. For details, consult the ARCHITECTURE.md files:

- **CLI usage** (deploy local/ssh/cloud, bench, teardown, vm, hardware-aware deploy, fixed-host mode, experiments, CI workflow) → [`emmy/commands/ARCHITECTURE.md`](emmy/commands/ARCHITECTURE.md)
- **Serving** (vLLM out-of-tree embedding plugin — emmy-compiled kernels behind vLLM's `/v1/embeddings`; `serving` extra) → [`emmy/serving/ARCHITECTURE.md`](emmy/serving/ARCHITECTURE.md)
- **Recipes vs experiments** (`recipes/` = the one recommended serving config per model, what `emmy deploy` runs;
  `experiments/` = benchmark grids, what `emmy bench` runs) → [`recipes/ARCHITECTURE.md`](recipes/ARCHITECTURE.md)
- **Prebuilt serving images** (per-model warm/bake/verify/push release pipeline — `make serve-* MODEL=<hf-id>`, the
  model-slug naming schema, the cubin cache-key parity contract, the golden coverage gate) →
  [`docker/vllm-emmy-serve/ARCHITECTURE.md`](docker/vllm-emmy-serve/ARCHITECTURE.md)
- **Recipe format** (matrices/cross/zip combinators, variant filtering, deep merge, named fields, extra_args validation, command recipes, aggregate, docker_options, driver/cuda pinning, SGLang) → [`emmy/recipe/ARCHITECTURE.md`](emmy/recipe/ARCHITECTURE.md)
- **Compiler** (Graph IR dialects, passes, backends) → [`emmy/compiler/ARCHITECTURE.md`](emmy/compiler/ARCHITECTURE.md) and child docs
- **Pipeline / autotune** (pass framework, knob/fork system, online/offline-prior search, two-level tune) →
  [`emmy/compiler/pipeline/ARCHITECTURE.md`](emmy/compiler/pipeline/ARCHITECTURE.md)
- **Tile lowering** (LoopOp → TileOp; **purely algebraic moveset — no shape specializations**. The stored tile IR
  has exactly THREE node kinds (all in `ir/tile/ir.py`): the general **`Fold`** — `reduce(⊕) ∘ map(f)`, stored in
  the λ-foldMap spelling (1m–1p): an iteration `axis`, a pure **`lift`** `Lambda` (`λ(k, v₁…vₙ) → S` — the
  element's SINGLETON state; ι spelled in the lift, softmax's is `(x, 1)`), the TRUE monoid's flat **`(init,
  combine)`** fields (ONE program, its results the fold's real accumulator names; the free helpers in
  `ir/stmt/algebra`), and a symmetric tuple of
  **`operands`** (the CLOSED inputs, each an edge, bound POSITIONALLY to the lift params); the bilinear
  **`Contraction`** node (1s) — every recognized contraction stores as this kind: its own reduce `k_axis`, the
  shared `a` operand edge and the product `Channel`s `(b_i, acc_i)` (arity N = the fused gate⊗up edge; sharing is
  the node's arity), with the placement/schedule fields (`axes`, `lead_axes`, `tile`, `stage`) UNSET in the stored
  term — caller facts, stamped onto a `replace()` copy at the point of use (`Contraction.placed`, a pure field
  stamp; `as_fold()` survives only as the node's DERIVED λ reading, consumed by the cross-partition `Reduction`
  machinery and the PLANAR demotion, its loop body byte-identical to the node's own) — and the lift/projection
  wrapper `Map` (`fn: Lambda` + `sources`, bound positionally; `fn.results` replaced the `out` last-def convention).
  The serial step and the `Accum` forms are DERIVED (combine at the singleton; the twist
  family selected structurally, never stored), and loops carry NO algebra — `Loop`/`StridedLoop` hold only their
  `AxisRole`; the retired `Algebra` bundle's lowering-side reads live in `passes/lowering/_reduction.Reduction`
  (the materializer's + `030_split_reduce`'s view). `Fold.from_loop` reconstructs the algebra from the loop BODY
  alone (degenerate facts off its `Accum`s; a twisted merge regenerated-and-byte-compared, or extracted against a
  `like` fold for a split partial), returns `None` for a non-λ-representable loop (the
  callers keep the raw-loop-IR `Map` escape) and its byte-identity gate compares the derived body/axis only — the
  matvec demotion is a formation fact. There is NO stored `step` SEQUENCE (deleted at step 7): the composed
  evaluations DERIVE — flash's kv stream λ-spells with its QK score a HOISTED operand edge and its PV
  synthesized+memoized inside the derived blocked evaluation (`Fold.step_stmts()` the one consumer read), and
  split-K's outer reduce is the identity-lift composition (`ir.composed_contraction` the one read). A bare sum,
  RMSNorm's statistic and flash's stream are `Fold` at different monoid arities; a matmul, the fused gate⊗up edge,
  flash's QK score and the derived PV are the `Contraction` node kind (the kind IS the `CONTRACTION` role — no
  bilinear parse; a `Fold`'s role stays DERIVED (`Fold.role`), never stored: TWISTED off the derived twist family,
  CONTRACTION off the composed split-K operand, PLANAR otherwise — so an unbindable matvec-shaped
  contraction, whose loads recognition keeps inline in the lift instead of building the node, derives PLANAR and
  takes the reduce tiers at schedule dispatch with no recognition-time demotion rewrite). **Sharing is arity**: the
  gate⊗up node reads one cone edge across two channels — no privileged operand slot, no let table, no reference arm.
  `Fold.loop` splices each operand's body before the first read of its bound param and flattens
  nested nodes in place — the derived loop depends only on the stored params; kernel identity is the α-INVARIANT
  TERM HASH (`ops.term_key`: canonical renumbering + hash-time ANF body-order canonicalization, consumed by
  `op_cache_key`'s TileOp arm and `Graph.structural_key`'s op field — never the lowered nest). An operand edge has
  two inhabitants
  — MATERIALIZED (a gmem `Load`) or COMPUTED (the node itself, stored INLINE; the cone via `_atomize.make_cone`).
  **Edge iff closed** holds BY CONSTRUCTION (positional operand binding; `ir.captured_values` demoted to the
  validation reading) and decides cut legality: closed subtrees may hoist to edges; combine's derived material —
  flash's PV, whose `P` reads the running state — sits BELOW the seam lattice, a derived schedule site excluded
  from PLACE (`Site.derived`), while flash's QK operand edge IS a PLACE
  site. **The `Contraction`'s placement/schedule fields are STAMPED, never stored**: the stored node is pure
  algebra (`k_axis` + the `a` edge + `Channel`s), and the placed reading the tensor-core/staged tiers require —
  the `(m, n)` `Side` geometry + `tile`/`stage` — is stamped onto a `replace()` copy by `Contraction.placed(m,
  n, lead, tile, stage)` from the CALLER's placement axes (trailing grid for a root kernel; `place.free` threads
  to the materializer via `Ctx.free` for flash) and the `TileOp.schedule` slices (`ir.shared_operand` is the
  placement-free cone read). The A/B asymmetry that is real — A M-resident/compute-fillable, B streamed — is a
  SCHEDULE fact read off the node's roles (`isinstance(c.b, Load)` eligibility gates), not a storage fact. A cone's
  SOURCE is the row-invariant prologue (the per-row statistic) and its `body` the per-cell normalize, so the K seam
  is the node boundary (`ops.cone_seam`). A projection has ONE home, the wrapping `Map.fn` — never a node field —
  and since 1q the fn of every recognized term is a STRICT pure `Lambda`: the root-store `Write`s (and the
  rms/softmax output-sweep `Loop` around them) ride `TileOp.stores` — `Store` decorations at the kernel boundary,
  reconstituted on demand by `effect_tail` (the scheduler's tail gates, the materializer's peel and
  `030_split_reduce` all read through it, so the lowered kernels are byte-identical to the stored-`Write` era; the
  raw-loop-IR kernels that are not recognized algebra — the un-recognized escape cell, `030`'s finalize, the coop
  fused-tail sibling — keep an impure fn through the one `_loop_ir_fn` arm).
  A bare reduce is a root `Fold`; softmax/RMSNorm is `Map(fn=per-cell normalize, sources=(Fold,))` + a sweep
  `Store`; the fused norm→linear /
  gate⊗up composition is `Map(fn=combine, sources=(Contraction,))` over the product node (a fork sibling of its
  coop-reduce form — option-0 stays coop; warp mma rows ride the sync compute-fill); a pure pointwise cell is a
  `Map(sources=())` + its root `Store`s. Every schedule slice (`TilePlan` / `ReducePlan` / `Stage`) lives in `TileOp.schedule` — a dict keyed by the
  tree-path codec's canonical key (`ir/tile/path.py`: ONE walker + resolver, short-path-canonical — bare for the
  primary node, `TILE@dd`/`TILE@pj` on flash; read/written through `ops.Sched`), the term staying pure and
  IMMUTABLE across the schedule search; the `TileOp` keeps `op + place + work + workers + knobs + schedule` (`work`
  is the ONE worker inventory, derived loudly from the TILE slices), and a sliced axis's window is the one
  `Axis.window`; the root stores are `TileOp.stores`. The stampers spell knob keys via the same resolver and
  VALUES site-locally (step 7): the worker inventory once in `WORK` (`w<M>x<N>[+p<np>]`/`t<N>[x<M>]`, sealed by
  `seal_workers`; the retired `WSPEC` row family's producer band rides `+p`), `TILE`/`REDUCE` values shed their
  worker tokens — so the stamped row IS the stored/golden spelling (legacy spellings are loudly-validated pin
  aliases; the golden corpus re-spelled mechanically). Dispatch reads the
  role/algebra off the node (`ops.axis_role`/`reduce_loop` recurse through `Map.sources`), and `ops.lower` flattens
  any node back to the same loop nest — no stored `Monoid`/`Semiring` kind. Flash is the `TWISTED` fold on the
  streaming schedule, its QK a hoisted operand-edge `Contraction` and its PV the derived evaluation's
  synthesized contraction node — a twisted monoid is a monoid,
  selected structurally not as a distinct kind) →
  [`emmy/compiler/pipeline/passes/ARCHITECTURE.md`](emmy/compiler/pipeline/passes/ARCHITECTURE.md)

When the user asks about a CLI flag, recipe field, or matrix combinator, read the relevant ARCHITECTURE.md before
answering — they hold the detail that is no longer in this file or the README.

## Prerequisites

- Python 3.12+ with `venv`
- `make setup` to create the virtual environment and install dependencies
- Docker and Docker Compose for local deployments
- `HF_TOKEN` environment variable for HuggingFace model downloads
- `EMMY_DUMP_DIR` environment variable (optional) — when set, all compiler stages dump intermediate artifacts (graphs, CUDA kernels, execution plans) to this directory for debugging. Per kernel, the dump also writes a `<kname>.torch.json` reproducer — the original PyTorch ops that kernel implements (sliced by op provenance), with an `i/N` coverage header (full vs partial) — runnable via `emmy run --ir <kname>.torch.json --bench` to reproduce accuracy / latency vs torch for that op. Kernels are named after the ops they realize (`k_rms_norm`, `k_sdpa_reduce`)
- `EMMY_TUNE_DB` environment variable (optional) — overrides the default tuning SQLite cache path
  (`~/.cache/emmy/autotune.db`). `emmy tune` reads from / writes to this path. NOTE: greedy `compile` / `run`
  resolve forks through the deploy evidence hierarchy — the live card's recorded goldens first (the repo-shipped
  verified tier; consulted, never trained on), then measured reservoir/DB evidence, then the global `Prior` (the
  online prior with its offline cold-start fallback; the old `_best_fork` DB→fork replay was removed). The online prior
  is a separate JSON checkpoint (`EMMY_ONLINE_FILE` → `~/.cache/emmy/online.json`; legacy `EMMY_PRIOR_FILE` still
  accepted) that `tune` writes and
  `compile` / `run` read. See [`emmy/compiler/pipeline/ARCHITECTURE.md`](emmy/compiler/pipeline/ARCHITECTURE.md)
  for the prior / two-level autotune story.

All `EMMY_*` config env vars are read and written through one module — `emmy/config.py`, the sole owner of
`os.environ` for these vars (the `EMMY_<KNOB>` namespace is the one exception, owned by
`compiler/pipeline/knob.py`; provider/secret vars stay with `emmy/redact.py`). CLI `--flag` overrides (e.g.
`--nvcc-flags`) resolve through `config.py` inside the library, not the command layer, so programmatic callers and tests
get the same precedence. `config.py` is the source of truth for the full var list — do not maintain a copy here.

## Running Tests

```bash
make test
```

`make test` compiles CUDA kernels at **`-Xcicc -O1`** (the suite is `nvcc`/`cicc`-compile-bound, not GPU-bound — `-O1`
dodges the cicc/LLVM unroll blowup on big register-tile kernels, ~3× faster wall time). This is the **correctness lane**:
`-O1` changes runtime perf, not numerics, and the deployable perf tests (`tests/perf`, `-m perf`) are skipped here — they
run at `-O3` via `make bench-kernels`. To re-run the suite at deployable `-O3`, prefix `EMMY_NVCC_FLAGS=` (empty) or
run `pytest` directly.

Or for a specific test file:

```bash
./venv/bin/pytest tests/test_recipe.py -v
```

When running a large subset (e.g. `tests/compiler/`), pass the same `-n auto --dist=loadgroup` flags `make test` uses to
parallelize (add `-p no:randomly` for a stable order):

```bash
./venv/bin/pytest tests/compiler/ -p no:randomly -n auto --dist=loadgroup
```

`-n auto` spawns one worker per core; `--dist=loadgroup` keeps tests sharing an `xdist_group` (e.g. CUDA context) on the
same worker.

## CLI Commands

The full CLI reference — every command, subcommand, flag, and example — lives in
[`emmy/commands/ARCHITECTURE.md`](emmy/commands/ARCHITECTURE.md). Do **not** duplicate that reference here; read
it before answering any CLI-flag question. Quickstart for the common paths:

| Command | Purpose |
| --- | --- |
| `emmy deploy {local,ssh,cloud} <model> …` | deploy via docker compose locally, over SSH, or on a freshly provisioned cloud VM |
| `emmy bench recipes/* [--filter KEY=PATTERN] [--no-teardown]` | deploy + benchmark + teardown across cloud VMs; `teardown <run_dir>` cleans up afterwards |
| `emmy vm create gpu --gpu NAME --gpu-count N` | provision a GPU VM by name (also `vm create/delete {gcp,cloudrift}`) |
| `emmy serve <model> [--generate] [--bench] [vllm flags…]` | serve an embedding (or `--generate` chat) model via vLLM with the emmy plugin |
| `emmy compile <model_or_ir> [--layer N] [--ir STAGE] [--dynamic …] [--target sm_NN]` | trace + run the compiler; print or save any IR stage |
| `emmy run <model_or_ir_or_--code> [--bench]` | compile + execute on the CUDA backend, check accuracy, optionally bench vs eager / `torch.compile` |
| `emmy tune <target> [--bench] [--gpus N]` | two-level autotune; writes the online prior + tune DB |
| `emmy eval {knobs,online,offline,golden,variants,failures} [--dataset {golden,db,nodes}]` | inspect the priors / tune DB |
| `emmy {pull,trace,generate,inspect,compare} …` | model download, IR tracing, the naive generation oracle, IR inspection, dump diffing |

Quick test models / scripts (for local iteration):

- Ungated Llama-arch smoke model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`; GPU embedding model (0.6B): `Qwen/Qwen3-Embedding-0.6B`
- Benchmark/profiling helpers live under `scripts/` (`bench_block.py`, `bench_model_kernels.py`, `bench_golden_set.py`,
  `bench_gen_*.py`, `profile_gen_decode.py`, `capture_gen_twins.py`, `new_models.py`, `merge_node_db.py`,
  `remote_node_collect.py`, `golden_neighbor_bench.py`, `digest_kernels.py` — the kernel-source byte-identity
  gate for tile-IR storage migrations) — run with `--help` for usage;
  the skills that drive them document the flows.

## Key Make Targets

- `make setup` — create venv and install dependencies (includes ruff)
- `make test` — run `pytest` using the venv (skips `perf`-marked tests; see `tests/perf/ARCHITECTURE.md`). Compiles
  kernels at `-Xcicc -O1` for ~3× faster nvcc (correctness lane; perf tests use `-O3` via `make bench-kernels`)
- `make lint` — run `ruff check` and `ruff format --check`
- `make format` — auto-format code and fix lint violations
- `make bench` — run benchmarks (`emmy bench recipes/*`)
- `make bench-kernels` — run per-kernel perf comparison vs PyTorch (`tests/perf/`, requires CUDA)
- `make clean` — remove venv and generated files

## Documentation Conventions

These are invariants — they hold for every doc change, no exceptions:

- **Plans are ephemeral. Never reference `plans/*.md` from durable docs (CLAUDE.md, README.md, any `ARCHITECTURE.md`) or
  from code (comments/docstrings).** A plan is a transient working note; anything worth keeping gets written into the
  durable doc itself, and the plan pointer is dropped. (`grep -rn "plans/" --include='*.py' emmy/` and over the
  durable docs must stay empty.) Plan *lifecycle* is governed by the Contribution Instructions below.
- **`ARCHITECTURE.md` files describe concepts, invariants, and the few key entry-point modules — not every file.** Do
  NOT add exhaustive per-file "module tree" tables or `file.py:123` line-number citations; they churn on every refactor
  and rot immediately. Name a module/symbol only when it is a genuine entry point, and refer to it by name, not line.
- **CLAUDE.md routes; it does not duplicate.** Each subsystem's detail lives in its nearest `ARCHITECTURE.md`; CLAUDE.md
  points there. Do not re-enumerate the CLI, env vars, or any reference list that already has a canonical home.

**Wrap every `.md` file in the repo to ~120 characters.** This includes `README.md`, every `ARCHITECTURE.md`, every file
under `docs/`, and any other markdown anywhere in the tree. Do NOT wrap at 70–80 characters — that is the default
markdown habit, and it is wrong for this repo. Aim for lines in the 90–120 range.

Table rows, ASCII diagrams, and long URLs may overflow past 120 if wrapping would hurt readability — that's the only
acceptable reason to go wider. Python code stays under 140 chars (Ruff-enforced).

## Contribution Instructions

IMPORTANT: You MUST follow ALL of these steps for EVERY code change. Do NOT skip any step.

### Writing code

1. Create a feature branch from `main` (e.g. `feature/my-new-feature`) — NEVER commit directly to `main`
2. Write code following guidelines in `STYLE.md`, `README.md` and `ARCHITECTURE.md` files in respective folders
3. Add tests if reasonable (in `tests/` following `tests/ARCHITECTURE.md` guidelines)

### Before committing (MANDATORY — do NOT skip these)

You MUST complete ALL of the following checks before every commit. These are not optional:

4. **Update `STYLE.md`** if any style changes were introduced — READ the current `STYLE.md` and compare
5. **Update `README.md`** if project setup, structure, or usage patterns changed — READ the current `README.md` and compare
6. **Update `CLAUDE.md`** if general instructions are no longer accurate — READ this file and compare
7. **Update `ARCHITECTURE.md`** files in every directory that was modified — READ each relevant `ARCHITECTURE.md` and compare
8. **Prune `plans/`**: if the change executed/landed a plan, **delete that plan file**. Then enforce the cap — if
   `plans/` holds more than 10 files, remove the executed/obsolete ones; if all remaining plans are still incomplete,
   remove the oldest. Never add a `plans/*.md` reference to durable docs or code (see Documentation Conventions).
9. **Run tests**: `make test` — fix any failures before proceeding
10. **Run linter**: `make lint` — if it fails, run `make format` and re-check

### Submitting

11. Push and open a PR

# Behavioral Guidelines:

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
