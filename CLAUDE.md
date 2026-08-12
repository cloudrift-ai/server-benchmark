# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Emmy is a Python tool for deploying and benchmarking LLM inference on GPU servers. It supports vLLM and SGLang engines, provides a CLI for local and remote (SSH) deployment of models via Docker Compose, plus automated benchmarking across multiple servers.

`README.md` is the canonical project overview and architecture index. Read it first, then use its links to locate the
relevant subsystem documentation. Do not duplicate the architecture index in this file.

When the user asks about a CLI flag, recipe field, or matrix combinator, use the README index to find and read the
relevant `ARCHITECTURE.md` before answering.

## Prerequisites

- Python 3.12+ with `venv`
- `make setup` to create the virtual environment and install dependencies
- Docker and Docker Compose for local deployments
- `HF_TOKEN` environment variable for HuggingFace model downloads
- `EMMY_DUMP_DIR` environment variable (optional) — when set, compiler stages dump intermediate debug artifacts
  (graphs, CUDA kernels, execution plans) to this directory. Frontend provenance slices used by `tune --bench` stay
  in memory; stable Torch IR is persisted only inside golden YAML. Kernels are named after the operations they realize
  (`k_rms_norm`, `k_sdpa_reduce`).
- `EMMY_TUNE_DB` environment variable (optional) — overrides the default tuning SQLite cache path
  (`~/.cache/emmy/autotune.db`). `emmy tune` reads from / writes to this path. NOTE: greedy `compile` / `run`
  resolve forks through the deploy evidence hierarchy — the live card's recorded goldens first (the repo-shipped
  verified tier; consulted, never trained on), then measured reservoir/DB evidence, then the global `Prior` (the
  online prior with its offline cold-start fallback; the old `_best_fork` DB→fork replay was removed). The online prior
  is a separate JSON checkpoint (`EMMY_ONLINE_FILE` → `~/.cache/emmy/online.json`; legacy `EMMY_PRIOR_FILE` still
  accepted) that `tune` writes and `compile` / `run` read. Use the README architecture index for the prior and
  two-level autotune design.

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

The full CLI reference is linked from the README architecture index. Do **not** duplicate that reference here; read
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

- Ungated generative smoke model: `Qwen/Qwen3-0.6B` (Qwen3 arch — same family as the embedding smoke model;
  serving-validated on a 4080, tuned TPOT 1.28x stock; defaults to thinking mode — pass `enable_thinking: false`
  in chat probes for terse outputs). `TinyLlama/TinyLlama-1.1B-Chat-v1.0` stays as the ungated **Llama-arch**
  smoke model. GPU embedding model (0.6B): `Qwen/Qwen3-Embedding-0.6B`
- Benchmark/profiling helpers live under `scripts/` (`bench_block.py`, `bench_golden_set.py`, `profile_gen_decode.py`,
  `capture_gen_twins.py`, `new_models.py`, `merge_node_db.py`, `digest_kernels.py` — the kernel-source byte-identity
  gate for tile-IR storage migrations, each case also asserting its pins reached a kernel) — run with `--help` for
  usage;
  the skills that drive them document the flows.

## Key Make Targets

- `make setup` — create venv and install dependencies (includes ruff)
- `make test` — run `pytest` using the venv (skips `perf`-marked tests; see the `tests/perf` architecture). Compiles
  kernels at `-Xcicc -O1` for ~3× faster nvcc (correctness lane; perf tests use `-O3` via `make bench-kernels`)
- `make test-durations` — re-measure `tests/durations.json`, the checked-in per-test timings the suite balances its
  xdist workers on; commit the result when the balance has drifted
- `make lint` — run `ruff check` and `ruff format --check`
- `make format` — auto-format code and fix lint violations
- `make bench` — run benchmarks (`emmy bench recipes/*`)
- `make bench-kernels` — run per-kernel perf comparison vs PyTorch (`tests/perf/`, requires CUDA)
- `make wheel` — build the wheel into `dist/` (stages the bundled recipes first; see the Release section of
  `README.md`)
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
- **README.md routes; CLAUDE.md does not duplicate.** README is the canonical architecture index. Each subsystem's
  detail lives in its nearest `ARCHITECTURE.md`; do not repeat links, CLI details, environment variables, or reference
  lists here.
- **Only use established terminology — [`GLOSSARY.md`](GLOSSARY.md) is the stable vocabulary.** In code comments,
  documentation, reports, commit messages, PR bodies, and when communicating with the user, use glossary terms, other
  established repo/field terms, or plain language. Never coin new labels; replace any invented term with the correct
  established term or a plain-word explanation.

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

**Keep PRs minimal.** Retain only durable implementation, tests, documentation, recipes, and publication evidence.
Delete exploratory scripts, intermediate experiments, run snapshots, and executed plans once their conclusions are
encoded in a durable artifact.

**Do not script open-ended reasoning.** Code should implement stable, reusable mechanics with a clear contract.
Model- or experiment-dependent judgment—such as interpreting heterogeneous benchmark evidence or deciding how to
assemble every possible serving report—belongs in skill instructions and agent reasoning, not in the benchmark
harness, result validators, or a growing family of one-off scripts. Simple, readable mechanical post-processing may
be embedded directly in a recipe; if the logic needs a large decision tree or model-specific policy, keep it out of
code.

### Before committing (MANDATORY — do NOT skip these)

You MUST complete ALL of the following checks before every commit. These are not optional:

4. **Update `STYLE.md`** if any style changes were introduced — READ the current `STYLE.md` and compare
5. **Update `README.md`** if project setup, structure, or usage patterns changed — READ the current `README.md` and compare
6. **Update `CLAUDE.md`** if general instructions are no longer accurate — READ this file and compare
7. **Update `ARCHITECTURE.md`** files in every directory that was modified — READ each relevant `ARCHITECTURE.md` and compare
8. **Prune `plans/`**: if the change executed/landed a plan, **delete that plan file**. Then enforce the cap — if
   `plans/` holds more than 10 files, remove the executed/obsolete ones; if all remaining plans are still incomplete,
   remove the oldest. Never add a `plans/*.md` reference to durable docs or code (see Documentation Conventions).
9. **Check terminology**: review every text this change adds or edits — code comments, docstrings, docs, report
   text, the commit message and PR body — against [`GLOSSARY.md`](GLOSSARY.md). Remove any invented terms and
   replace them with the correct established term or a plain-word explanation (see Documentation Conventions).
10. **Run tests**: `make test` — fix any failures before proceeding
11. **Run linter**: `make lint` — if it fails, run `make format` and re-check

### Before submitting (MANDATORY — do NOT skip these)

You MUST audit the complete diff after the before-committing checks and before requesting review:

12. **Remove unnecessary functionality**: Which new functionality can be removed?
13. **Reuse existing mechanisms**: Which existing CLI, library, recipe, or skill can be reused instead?
14. **Rethink touched functionality**: Can existing functionality be rearchitected around the PR's needs so one
    simpler shared design replaces parallel or specialized paths?
15. **Remove obsolete code**: Delete existing code that the PR makes unnecessary. Apply the boy-scout rule within the
    PR's scope and leave touched code cleaner, without expanding into an unrelated refactor.
16. **Keep reasoning out of code**: Which logic should not be scripted and should become concise agent instructions?
17. **Minimize the diff**: Can the same outcome be achieved with fewer changed lines, files, flags, and abstractions?
18. **Apply the audit findings**: perform the removals and consolidation before requesting review. Tests must protect
    the smaller contract, not preserve unnecessary machinery.

### Submitting

19. Push and open a PR

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
- Apply the boy-scout rule within the PR's scope: simplify adjacent functionality exposed by the change when that
  cleanup reduces the total design and remains covered by tests.
- Do not turn scoped cleanup into an unrelated refactor.
- Match existing style, even if you'd do it differently.
- Remove existing dead code or duplication in the touched path when it is confidently obsolete; otherwise mention it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Remove pre-existing code that the new design makes unnecessary.

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
