# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Emmy is a Python tool for deploying and benchmarking LLM inference on GPU servers. It supports vLLM and SGLang engines, provides a CLI for local and remote (SSH) deployment of models via Docker Compose, plus automated benchmarking across multiple servers.

The `README.md` is intentionally short — example-driven, no narrative. For details, consult the ARCHITECTURE.md files:

- **CLI usage** (deploy local/ssh/cloud, bench, teardown, vm, hardware-aware deploy, fixed-host mode, experiments, CI workflow) → [`emmy/commands/ARCHITECTURE.md`](emmy/commands/ARCHITECTURE.md)
- **Serving** (vLLM out-of-tree embedding plugin — emmy-compiled kernels behind vLLM's `/v1/embeddings`; `serving` extra) → [`emmy/serving/ARCHITECTURE.md`](emmy/serving/ARCHITECTURE.md)
- **Recipe format** (matrices/cross/zip combinators, variant filtering, deep merge, named fields, extra_args validation, command recipes, aggregate, docker_options, driver/cuda pinning, SGLang) → [`emmy/recipe/ARCHITECTURE.md`](emmy/recipe/ARCHITECTURE.md)
- **Compiler** (Graph IR dialects, passes, backends) → [`emmy/compiler/ARCHITECTURE.md`](emmy/compiler/ARCHITECTURE.md) and child docs
- **Pipeline / autotune** (pass framework, knob/fork system, online/offline-prior search, two-level tune) →
  [`emmy/compiler/pipeline/ARCHITECTURE.md`](emmy/compiler/pipeline/ARCHITECTURE.md)
- **Tile lowering** (LoopOp → TileOp; **purely algebraic moveset — no shape specializations**. The stored tile IR is
  a tree of **structural nodes** (all in `ir/tile/ir.py`): a `PLANAR`/`TWISTED` reduce lifts to a typed `Reduction`
  (its `Carrier` + reduce `axis` + `partial` split out, the fold `Loop` synthesized on demand, holding no
  projection — a reduce that COMPOSES another node, split-K's sliced contraction or flash's score, puts it in
  `partial`, the one composition rule); EVERY recognized contraction — per-cell scalar included — is a `Contraction`
  node (nodified at recognize time with a deferred `TilePlan()`; an unbindable one demotes to `PLANAR`) holding ONE
  shared `a` **operand edge** plus product **`channels`** `(b_i, acc_i)`. An operand edge has two inhabitants — the
  two things an input can be: MATERIALIZED (a gmem `Load`) or COMPUTED (the node itself, stored INLINE on the edge —
  the cone via the one builder `_atomize.make_cone`, flash's `P`). Tree ownership gives an inline node exactly one
  consumer, so **sharing is arity, not naming**: "these two matmuls read the same A" is one arity-2 product
  contraction (the fused gate/up MLP edge — a product semiring outputting N matrices), scheduled and lowered as ONE
  unit through `Contraction.loop`'s derived product fold (N-component identity-family carrier); there is no let
  table, no reference arm, and no resolve step. Closure (`ir.captured_values`: no value name captured from the
  enclosing body, iteration vars excluded) is the placement-cut precondition — a capturing inline operand like
  flash's `P` is legal, just uncuttable. The A/B asymmetry that is real — A is M-resident and
  compute-fillable, B is the K×N operand the loop streams — is a SCHEDULE fact: every staged/mma tier states
  `isinstance(c.b, Load)` as an eligibility precondition and declines a computed B to the gmem-direct tier, which
  lowers it through the same `contraction_loop` builder. A cone's own SOURCE is the
  row-invariant prologue (the per-row statistic) and its `body` is the per-cell normalize, so the K seam is the node
  boundary (`ops.cone_seam`); the lift / projection wrapper
  is a `Map` (`body` + `sources` — `project ∘ reduce`, `source` the len-≤1 compat read). A
  projection has ONE home, the wrapping `Map.body` — never a node field. A bare reduce is the root `Reduction`;
  softmax/RMSNorm is a `Map(body=sweep, sources=(Reduction,))`; the fused norm→linear / gate-up composition is a
  `Map(body=combine, sources=(Contraction,))` over the product node (a fork sibling of its coop-reduce form —
  option-0 stays coop; the warp mma rows ride the sync compute-fill); a pure pointwise cell is a `Map(sources=())`;
  the only annotated `Loop`s still riding a flat `Map.body` are `030_split_reduce`'s sliced partials. Every schedule
  slice rides the node it decorates (a `Contraction`'s `tile` / `stage`, a `Reduction`'s `reduce` / `stage`) — the
  `TileOp` keeps only `op + place + workers + knobs` — and a sliced axis's window (its split parentage plus
  a cross-CTA slice's absolute base/bound) is the one `Axis.window`.
  Dispatch reads the role/carrier off the node (`ops.axis_role`/`reduce_loop` recurse through `Map.sources`), and
  `ops.lower` flattens any node back to the same loop nest — there is no stored `Monoid`/`Semiring` node kind (those
  wrappers were retired). Flash attention is the `TWISTED` reduce on the streaming schedule, a twisted monoid is a
  monoid, selected structurally not as a distinct kind) →
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
  `remote_node_collect.py`, `golden_neighbor_bench.py`) — run with `--help` for usage;
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
