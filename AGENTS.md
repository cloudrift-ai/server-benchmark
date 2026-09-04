# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

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
- `EMMY_FREEZE_DIR` environment variable (optional) — overrides the measurement freeze the prior is evaluated
  against (`emmy eval prior --dataset nodes`, and `emmy fit`'s measured cells). Defaults to the repo-checked
  `emmy/compiler/pipeline/search/freezes/` — a digest-pinned, version-stamped snapshot that is identical on every
  machine, which is what makes a reported prior number reproducible. The tune DB and the online reservoir are
  machine-local and mutable; reach them with `--db` when you want one machine's data, not as the default.
  The freeze's payload YAML is tracked in **git LFS**; its manifest is plain git so provenance stays diffable.
  Re-freeze with `scripts/freeze_node_store.py`.
- `EMMY_TUNE_DB` environment variable (optional) — overrides the default tuning SQLite cache path
  (`~/.cache/emmy/autotune.db`). `emmy tune` reads from / writes to this path. NOTE: greedy `compile` / `run` /
  `serve` resolve forks through ONE measured-evidence pick — the reservoir, this DB's `perf` rows and the golden rows
  in scope (the live card's repository goldens, or the file `--golden PATH` names) rank fastest-first, a row spelling
  a placement or a cross-CTA split prices that kernel-set decision, and the global `Prior` (the online prior with its
  offline cold-start fallback) decides only where nothing was measured; `--strict-evidence` turns that fall-through
  into an error. `run --golden PATH --bench` writes what it measures back into this DB, which is how a golden row
  becomes what the next compile picks. The online prior
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

`make test` compiles CUDA kernels at **`-Xcicc -O1`** — the **correctness lane**: `-O1` changes runtime perf, not
numerics, and the deployable perf tests (`tests/perf`, `-m perf`) are skipped here, running at `-O3` via
`make bench-kernels`. It also sets `EMMY_GOLDEN_FILE=` (set, empty): no repository golden is evidence in this lane,
because the lane never asks how fast a pick is and importing a card's goldens is work every worker process would
repeat; a test that needs golden evidence scopes it itself (`--golden PATH`, `golden.records_override`). To re-run the suite at deployable `-O3`, prefix `EMMY_NVCC_FLAGS=` (empty) or run `pytest`
directly.

The lane saves far less than this file used to claim. Measured on an RTX 5090 (CUDA 13.0, 16 cores, one repo, only the
opt level varying): cold cubin cache **923 s at `-O1` vs 1031 s at `-O3`** (1.12×); warm **718 s vs 760 s** (1.06×);
identical results every run. The retired "~3× faster" was never re-measured after the WMMA→`mma.sync` migration
removed the cicc unroll blowup it rested on. The cold/warm gap also puts kernel compilation at roughly a fifth of the
suite's wall time, so it is not the dominant cost either. Keeping `-O1` here buys ~12% cold; dropping it would leave
one compile regime everywhere in the repo.

Checked-in model goldens are not exercised by the per-commit test suite. The nightly `onboard-model` workflow owns
their repository validation, strict decode, and exact-GPU replay so model qualification stays with its GPU evidence.
The decode half alone is also reachable off the default lane, on any machine: `make test-goldens` strictly decodes
every checked-in golden file (one case per file) against the current compiler, which is how you see a card's rows go
green again after a tuning round re-records them.

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

### The realization corpus

`tests/compiler/realization/` replays pinned schedules from checked-in case files. A case's expectation is its
filename: no suffix means every stage must pass, `_xfail_<stage>` means it is a known gap expected to fail at
`offered`, `realized`, `built` or `correct`.

- A case **without** a suffix that fails is a regression. Fix the compiler. **Never add an `_xfail_` suffix to make a
  red test green** — that converts a regression into a recorded gap and the ratchet stops meaning anything.
- A case **with** a suffix that passes means the gap closed. `git mv` the file to drop the suffix; do not delete the
  case.
- A **stale case** failure means a kernel identity or a schedule codec changed and the stored derived data no longer
  matches. `make test` detects this on its own, on any machine; `make test-corpus-regen` is the fix. It refuses to
  write when a case's verdict also changed; that refusal is the signal, not an obstacle to work around.
- **The corpus never asks for something this machine cannot do.** With no GPU, the only obligation is the stale case
  above, and it is always fixable where you are: `offered` and `realized` run at the case's declared capability, while
  `built` and `correct` run only on a card whose capability equals it.
- **Latency is measured in `tests/perf/`, whose case list IS the corpus.** `make test` compiles at `-O1` and never
  measures; `make bench-kernels` benches every closed case the card can run, prints the comparison against eager and
  `torch.compile`, and reports a case slower than its stored number. A regression there is a finding, not a failure.
- **Never write a benchmark script.** `emmy run --golden FILE --bench --record` benches a golden and writes its
  timings back. If it cannot express what you need, that is a missing flag to add, not a script to write.

Before adding a case, read `tests/compiler/realization/ARCHITECTURE.md` — it owns what earns a case, the knob spelling
rules, and what deliberately stays in Python.

### macOS: the suite exits 139 (SIGSEGV)

The loop backend JIT-compiles kernels in-process through cppyy / Cling (`emmy/compiler/ir/loop/runner.py`), and
cppyy-cling 6.32.8 bundles LLVM 16. That compiler cannot parse the libc++ headers in the Xcode 26 SDK, whose
`is_convertible.h` uses the `__is_nothrow_convertible` builtin unconditionally. Cling faults while building its
precompiled header, so every test that imports cppyy dies on a native crash: `pytest tests/compiler/ir/ --collect-only`
exits 139 during *collection*, and `make test` loses its xdist workers to "node down: Not properly terminated".

Rebuild the precompiled header once against an SDK whose libc++ Cling can still parse — any installed 15.x will do:

```bash
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk CLING_REBUILD_PCH=1 \
  ./venv/bin/python -c 'import cppyy; cppyy.cppdef("int probe() { return 42; }"); assert cppyy.gbl.probe() == 42'
```

This writes `venv/lib/python3.12/site-packages/cppyy_backend/etc/allDict.cxx.pch.20.6.32.8`, after which cppyy imports
cleanly with no `SDKROOT` set. Repeat it whenever `venv/` is recreated or cppyy is reinstalled. Once cppyy releases a
Cling built on a newer LLVM, upgrading it replaces this workaround.

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
| `emmy eval {knobs,prior,golden,variants,failures} [--dataset {golden,db,nodes}]` | inspect the priors / tune DB |
| `emmy {pull,trace,generate,inspect,compare} …` | model download, IR tracing, the naive generation oracle, IR inspection, dump diffing |

Quick test models / scripts (for local iteration):

- Ungated generative smoke model: `Qwen/Qwen3-0.6B` (Qwen3 arch — same family as the embedding smoke model;
  serving-validated on a 4080, tuned TPOT 1.28x stock; defaults to thinking mode — pass `enable_thinking: false`
  in chat probes for terse outputs). `TinyLlama/TinyLlama-1.1B-Chat-v1.0` stays as the ungated **Llama-arch**
  smoke model. GPU embedding model (0.6B): `Qwen/Qwen3-Embedding-0.6B`
- Benchmark/profiling helpers live under `scripts/` (`bench_block.py`, `profile_gen_decode.py`,
  `capture_gen_twins.py`, `new_models.py`, `merge_node_db.py`, `digest_kernels.py` — the kernel-source byte-identity
  gate for tile-IR storage migrations, each case also asserting its pins reached a kernel) — run with `--help` for
  usage;
  the skills that drive them document the flows.
- **Never write a benchmark script.** `emmy run --bench --json PATH` is the machine-readable record every consumer
  reads; `--golden FILE` benches every realization in a golden and `--realization NAME` selects one. If the CLI
  cannot express what you need, that is a missing flag to add, not a script to write — the two scripts that
  re-implemented it (one parsing stdout with a regex, one diffing perf snapshots) had silently stopped working before
  anyone noticed.

## Key Make Targets

- `make setup` — create venv and install dependencies (includes ruff)
- `make test` — run `pytest` using the venv (skips the off-lane `perf` / `goldens` tests). Compiles
  kernels at `-Xcicc -O1` (correctness lane, ~12% faster than `-O3` on a cold cache; perf tests use `-O3` via
  `make bench-kernels`)
- `make test-corpus-regen` — restamp the realization corpus's derived half after a kernel-identity or schedule-codec
  change (`make test` detects the staleness on any machine; this applies the fix)
- `make test-goldens` — strict-decode every checked-in golden against the current compiler (off the default lane,
  no GPU needed; a stale row is detectable anywhere, re-recording it is what needs the card)
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

- **Plans are ephemeral. Never reference `plans/*.md` from durable docs (AGENTS.md, README.md, any `ARCHITECTURE.md`) or
  from code (comments/docstrings).** A plan is a transient working note; anything worth keeping gets written into the
  durable doc itself, and the plan pointer is dropped. (`grep -rn "plans/" --include='*.py' emmy/` and over the
  durable docs must stay empty.) Plan *lifecycle* is governed by the Contribution Instructions below.
- **`ARCHITECTURE.md` files describe concepts, invariants, and the few key entry-point modules — not every file.** Do
  NOT add exhaustive per-file "module tree" tables or `file.py:123` line-number citations; they churn on every refactor
  and rot immediately. Name a module/symbol only when it is a genuine entry point, and refer to it by name, not line.
- **README.md routes; AGENTS.md does not duplicate.** README is the canonical architecture index. Each subsystem's
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
code. It is fine to write code that processes structural data by selecting fields, reshaping rows, sorting, joining,
or producing a CSV, TSV, or JSON table. Do not write scripts that interpret results or assemble human-readable
reports; agents perform that reasoning and write the report.

### Before committing (MANDATORY — do NOT skip these)

You MUST complete ALL of the following checks before every commit. These are not optional:

4. **Update `STYLE.md`** if any style changes were introduced — READ the current `STYLE.md` and compare
5. **Update `README.md`** if project setup, structure, or usage patterns changed — READ the current `README.md` and compare
6. **Update `AGENTS.md`** if general instructions are no longer accurate — READ this file and compare
7. **Update `ARCHITECTURE.md`** files in every directory that was modified — READ each relevant `ARCHITECTURE.md` and compare
8. **Prune `plans/`**: if the change executed/landed a plan, **delete that plan file**. Then enforce the cap — if
   `plans/` holds more than 10 files, remove the executed/obsolete ones; if all remaining plans are still incomplete,
   remove the oldest. Never add a `plans/*.md` reference to durable docs or code (see Documentation Conventions).
9. **Check terminology**: review every text this change adds or edits — code comments, docstrings, docs, report
   text, the commit message and PR body — against [`GLOSSARY.md`](GLOSSARY.md). Remove any invented terms and
   replace them with the correct established term or a plain-word explanation (see Documentation Conventions).
10. **Run tests**: `make test` — fix any failures before proceeding
11. **Run linter**: `make lint` — if it fails, run `make format` and re-check
12. **Decode the goldens** whenever the change touches the compiler (anything under `emmy/compiler/`, and always
    for a schedule-codec, enumeration or knob-spelling change): `make test-goldens`. Off the default lane and needs no
    GPU. `make test` guards the realization corpus against exactly this class of change and nothing guarded the
    checked-in model goldens, so a rebuild of the schedule space can invalidate every recorded row in silence. If rows
    go red, name the change that did it in the PR body — do **not** re-record them to make it green, which enshrines
    the regression as the new reference.

### Before submitting (MANDATORY — do NOT skip these)

You MUST audit the complete diff after the before-committing checks and before requesting review:

13. **Remove unnecessary functionality**: Which new functionality can be removed?
14. **Reuse existing mechanisms**: Which existing CLI, library, recipe, or skill can be reused instead?
15. **Rethink touched functionality**: Can existing functionality be rearchitected around the PR's needs so one
    simpler shared design replaces parallel or specialized paths?
16. **Remove obsolete code**: Delete existing code that the PR makes unnecessary. Apply the boy-scout rule within the
    PR's scope and leave touched code cleaner, without expanding into an unrelated refactor.
17. **Keep reasoning and reports out of code**: Which logic should become concise agent instructions? Code may
    transform structural data, but scripts must not interpret results or assemble human-readable reports.
18. **Minimize the diff**: Can the same outcome be achieved with fewer changed lines, files, flags, and abstractions?
19. **Check the core line balance**: run `git diff --stat main -- emmy/`. A PR that introduces no new functionality
    (a refactor, a cleanup, a fix) must NOT increase the line count under `emmy/` — net growth there without new
    capability is the typical sign of architectural creep: another special case, helper, or early return layered onto
    a design that no longer fits. If the balance is positive, do not shave lines cosmetically to pass the check —
    restructure so the layering disappears, or say explicitly in the PR body why the growth is justified.
20. **Apply the audit findings**: perform the removals and consolidation before requesting review. Tests must protect
    the smaller contract, not preserve unnecessary machinery.

### Submitting

21. Push and open a PR

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
