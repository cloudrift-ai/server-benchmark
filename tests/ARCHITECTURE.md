# Test Architecture

## Overview

All tests use **pytest** with **pytest-asyncio** (`asyncio_mode = "auto"` in `pyproject.toml`) and live in the `tests/` directory, organized into subdirectories that mirror the `emmy/` source tree. Tests are designed to run without GPU hardware, Docker, or network access — every external interaction is avoided via dry-run mode or by testing pure functions directly.

## Directory Structure

`tests/` mirrors the `emmy/` source tree: a test directory exists because a source package does, and a test
module is named for the source module it covers. To find the tests for `emmy/<a>/<b>.py`, look in
`tests/<a>/test_<b>.py`. Three files sit at the root: `conftest.py` — shared fixtures plus the CUDA / LPT xdist routing
hook (see **Running**) — `test_emmy.py`, which mirrors `emmy/emmy.py`, the CLI entrypoint that belongs to no package,
and `test_logging_setup.py`, which covers the process-wide CLI and serving-plugin logging boundary.

Mirroring is the rule, not a coincidence — when a source package grows subpackages, the test directory follows.
`tests/compiler/pipeline/search/` is the worked example: its `data/`, `policy/`, and `prior/` subdirectories exist
because `emmy/compiler/pipeline/search/` has them, so a test for `policy/greedy.py` lives in `policy/`, one for
`prior/offline.py` in `prior/`, and only tests of the package's own top-level modules (`db.py`, `features.py`,
`slice.py`, …) stay flat. Tests that span several modules of a package — a cross-cutting property like
deploy-pick order invariance, or a process-wide cache over two subsystems — sit at the level that owns all of
them, not inside one arbitrary child.

Eight directories break the `emmy/` mirror deliberately, because their organizing axis is the *kind* of test or their
source lives outside the package:

| Directory | Axis |
|---|---|
| `compiler/e2e/` | end-to-end coverage matrices — the whole pipeline per regime (matmul / reduce / attention / fused), not per pass |
| `compiler/realization/` | checked-in reproducers of pinned schedules, replayed as data (see its `ARCHITECTURE.md`) |
| `compiler/cli/` | `emmy <command>` as a subprocess, via the `run_cli` fixture |
| `compiler/fixtures/` | checked-in traces and model configs, not tests |
| `perf/` | GPU perf comparison vs PyTorch, gated by the `perf` marker (see `tests/perf/ARCHITECTURE.md`) |
| `github/` | unit tests for repository automation helpers under `.github/scripts/` and `.github/workflows/scripts/` |
| `scripts/` | tests for executable helpers under the repository's `scripts/` directory |
| `architecture/` | repository-wide dependency and layering invariants |

The GitHub automation tests also pin workflow-level safety contracts that cannot be expressed inside a helper, such
as loading discovery and onboarding control code from the exact workflow commit while editing the rolling branch.

Three small organizing directories are also intentional:

| Directory | Purpose |
|---|---|
| `benchmark/models/` | named-model configuration contracts spanning recipes, experiments, and runtime images |
| `serving/generation/` | the generation runner, loop, capture, and vLLM generation adapter as one serving workflow |
| `support/` | suite-wide data shared across source-subsystem boundaries; never tests or pytest hooks |

Do not add a directory merely to shorten a file listing. These directories exist because their tests share one workflow
or span several source trees. `compiler/passes/` and `perf/` carry their own `ARCHITECTURE.md`; read those before adding
to them.

`serving/` compiles under its **own golden**, not a cold search. These tests ask whether the runner stitches,
dispatches and allocates correctly — vLLM integration and materialization — and cold deploy is the prior's contract,
tested elsewhere. So `serving/helpers.py` spells every shape the lane compiles in one place (`RUNNERS` for the runner
builds, `WRAPPERS` for the attention-split carves), `serving/regen.py` writes the golden covering exactly those, and the
`built` / `build_runner` fixtures build inside `helpers.evidence_scope()`. Two properties follow, and both are the
point: a replay costs a rebind instead of ~1M `schedule()` calls per model, and the compile no longer depends on the
machine-local tune DB and online prior a cold pick resolves through. Strict evidence keeps the golden honest — a fork no
row decides raises `EvidenceError` naming the kernel, so a stale or partial golden fails loudly instead of quietly
restoring the search. Regenerate with `python -m tests.serving.regen` when a shape changes or a new one joins a table.

## Test Layers

The suite runs in four layers, distinguished by what they touch rather than by where they live:

- **Unit** — pure functions and dataclasses with synthetic inputs. No I/O. Compiler IR units also pin construction
  idempotence and output-specification round trips, including sibling output sweeps. The bulk of the suite.
- **CLI dry-run** — the full argument-parsing → config-loading → orchestration path invoked as a subprocess with
  `--dry-run`, stopping just before any real side effect (SSH, Docker, file writes). Covers `deploy ssh/local/cloud`,
  `bench`, `teardown`, and `vm create/delete/audit`. These use real recipes from `recipes/` so config drift fails a
  test.
- **GPU** — guarded by `requires_cuda` / `requires_sm90` / `importorskip` so they skip cleanly off-GPU, and routed
  onto a serial worker chain by the root conftest (see **Running**).
- **End-to-end** — a traced model or snippet through the whole compiler, compared against PyTorch eager or numpy.

A test belongs to the lowest layer that can prove the property. Reach for a subprocess or a GPU only when the
behavior genuinely lives there — each costs roughly an order of magnitude more wall time than the layer below it.

## Fixtures and Helpers

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `project_root` | session | Absolute path to repo root |
| `recipes_dir` | session | Absolute path to `recipes/` |
| `run_cli` | session | Callable that invokes `python -m emmy.emmy` as a subprocess |
| `make_bench_config` | function | Factory that writes a temp `config.yaml` for bench tests (benchmark section only) |
| `tmp_recipe_dir` | function | Temp directory with a sample `recipe.yaml` for unit tests |
| `sample_config` | function | Single-instance vLLM config dict for compose tests |
| `sample_config_sglang` | function | Single-instance SGLang config dict for compose tests |
| `sample_config_multi` | function | Multi-instance config dict for compose tests |

`conftest.py` files expose only pytest-discovered fixtures and hooks; private hook implementation stays beside its hook.
Reusable callables live in an explicit helper module at the nearest directory shared by their callers: compiler-wide
helpers and CUDA markers in `tests/compiler/helpers.py`, the hand-spelled tile-term builders (`slab`, `projection`,
`reduction`, `contraction` — the vocabulary the total lift forms, for fixtures the lift cannot yet form from Loop IR)
in `tests/compiler/terms.py`, search-only node builders in `tests/compiler/pipeline/search/helpers.py`, and the
cross-subsystem synthetic checkpoint builder in `tests/support/checkpoints.py`. Test modules import those dependencies
directly; they never import from another test module or from `conftest.py`. A test module never re-declares a builder
the shared module already provides.

## Conventions

- **Prefer combinatorial coverage matrices over per-capability test files.** The compiler e2e suite covers each
  regime with one parameterized matrix (`test_matmul_coverage.py`'s tile × stage × reduce × static/dynamic grid,
  `test_reduce_coverage.py`, `test_attention_coverage.py`, `test_fused_edge.py`'s producer × tier product) rather
  than a file per capability. When a legacy one-off test's behavior is subsumed by such a matrix (or by the matmul
  coverage matrix specifically), DELETE the one-off — do not maintain both. A new capability extends the nearest
  matrix with a parameter/case before it earns its own file.
- **Keep model/card qualification out of the default suite.** `make test` does not retrace and compile a complete
  serving-twin matrix for a named checkpoint and GPU. The serving-image release workflow owns exact
  model/revision/card qualification. Retain a small model fixture only when it proves reusable behavior that a
  synthetic input cannot.
- **Do not load checked-in golden YAML in the default suite — except the realization corpus.** Unit tests use
  synthetic records and working files, and the nightly `onboard-model` workflow owns repository schema validation,
  strict decode, and exact-GPU replay for `recipes/*/golden/*.yaml`. `tests/compiler/realization/cases/` differs on
  both counts that motivated the rule: its files are hand-minimized reproducers rather than model qualification
  evidence, and they carry no measurement claim, so nothing about them is card-specific. They also target a
  **declared** capability rather than the live card — `Context.from_target(compute_cap)` — so the enumeration and
  lowering stages are machine-independent and an sm_70 lockout is exercised on a box that has no sm_70. Only the
  build and accuracy stages consult the live card, and they gate on `device_compute_capability() == compute_cap`,
  beside `requires_sm90` in spirit but keyed on equality rather than a floor. The one golden lane inside pytest sits
  off the default one: `make test-goldens` (the `goldens` marker) strictly decodes every checked-in file, one case
  per file. It is deselected for COST, not for card-dependence — decoding replays each record at its declared
  capability, so a stale row is detectable on any machine; re-recording one is what needs the card.
- **Keep one subprocess smoke per report path.** Filtering, join, and presentation variants use small synthetic
  records at the owning unit layer instead of launching the CLI repeatedly over the full repository corpus.
- **Async tests** — tests for async functions are plain `async def` (no decorator needed; `asyncio_mode = "auto"` handles it). Mock async callables with `AsyncMock`.
- **No mocking** — dry-run mode is the primary strategy for testing command orchestration without side effects.
- **Real recipes** — CLI dry-run tests use recipes from the `recipes/` directory to catch config drift.
- **Recipe lifecycle** — named-model deployment assertions skip when their recipe is disabled; maintained and
  best-effort recipes keep the full serving contract coverage, while obsolete recipes remain covered by lifecycle
  validation. Catalog and command tests cover the versioned JSON inventory, effective deployment metadata,
  query-filtered inventory, the minimal installation-selected CLI, editable-versus-wheel catalog selection, recipe-name
  materialization, and validated shell creation. Repository-automation tests validate required lifecycle rationales
  and heat scores, the unbounded set of onboarding shells, each shell's one-to-three-entry deployment matrix, and the
  bounded read-only source-agent configuration. Notification tests cover modified-model lifecycle groups with heat and
  validated deployment/performance summaries. Query tests cover constrained expression parsing, implicit deployment
  expansion, external candidates, heat ordering, lifecycle ordering, and the versioned row result.
  Discovery-filter tests pin deterministic recipe batching, exact score coverage, mechanically preserved onboarding
  shells, filtering of repeated existing candidates, derived best-effort decisions, and the bounded tool-free scoring
  subagent. Qualification-manifest tests also pin the requested operation mode, exact model ID, target, preserved
  lifecycle tag
  and heat, compact notification evidence, current-platform records inside the named archive, bounded compatibility
  fixes, regression notices, and isolation from other platform results before artifacts may be staged.
- **Temp recipes** — unit tests and multi-instance edge cases create throwaway recipes via `tmp_path`.
- **Plain functions** — no test classes; tests are grouped by file and separated with comment headers.
- **Assertions on stdout** — dry-run tests verify that the correct commands and messages appear in the expected order.
- **Mirror source layout, subpackages included** — test directories match `emmy/` subdirectories (e.g.
  `tests/deploy/` ↔ `emmy/deploy/`), and that holds all the way down: when a source package gains a subpackage,
  its tests move into a matching test subpackage rather than staying flat beside their new siblings. One test
  module per source module; a file covering several modules of a package sits at the level that owns them all.
  The exceptions are the kind- and workflow-organized directories listed under **Directory Structure**.
- **One file per subject, not per bug.** A behavior discovered later belongs in the file that already owns its
  subject, as a new section with a comment header — not in a new file named after the incident. Several small
  files re-declaring the same fixtures is the signal to merge them; a file whose sections share no scaffolding
  and no subject is the signal to leave them apart.
- **Known failures are marked inline** with `@pytest.mark.xfail`, carrying a reason that says what was
  removed or broken and when it should come back. For a deliberate whole-subsystem removal whose casualties span
  dozens of files, prefer one registry module of exact node ids applied as a **strict** xfail from the root
  `conftest.py` — exact ids, never path globs, so each id is an acceptance obligation and the list shrinking to
  empty is the completion gate. Arbitrary assertion failures stay visible, and strict XPASS closes recovered
  obligations.
- **Card-conditional expectations stay inline**, non-strict, at their own test — a flaky or SKU-specific failure
  needs a reason that names the condition.

## Running

While developing, run only the tests that cover the change, under a two-minute budget. The whole suite takes many
minutes and belongs to the finalization stage of a PR — see the Contribution Instructions in `AGENTS.md`.

```bash
pytest tests/deploy/test_recipe.py -v   # single file — the development lane
pytest tests/deploy/ -k recipe -v      # a few tests — the development lane
pytest tests/ -v                       # all tests, finalization only (skips off-lane `perf` / `goldens` tests)
pytest tests/perf/ -m perf -v          # GPU perf suite (see tests/perf/ARCHITECTURE.md)
pytest tests/compiler/pipeline/search/ -m goldens -v   # strict-decode the checked-in goldens (make test-goldens)
```

Under `make test` (`-n auto --dist=loadgroup`) the root `conftest.py` routes every CUDA-touching test onto two
serial chains via dynamic `xdist_group` markers — `cuda` for in-process device work (one shared context, keeps
the attention-chain accuracy thresholds deterministic) and `cuda-cli` for `run_cli` subprocess tests (each owns a
fresh CUDA context; bounding their concurrency prevents GPU OOM from ~30 simultaneous subprocesses). CUDA items
are detected via the `requires_cuda` skipif reason, a `[cuda...]` callspec id, or an explicit
`xdist_group("cuda")` pytestmark (the `tests/serving/**/*_gpu.py` convention — honoring it matters because the LPT
bucketing would otherwise add a function-level group that shadows the module-level mark). The hook is
`tryfirst` because xdist's worker-side hook bakes group names into nodeids before plain conftest hooks run —
without it the markers land too late and CUDA tests silently scatter across workers. Non-CUDA tests are
LPT-bucketed across the remaining workers by cost, so the makespan approaches the load of the heaviest bucket
rather than whatever an arbitrary split produced.

Routing every CUDA test onto one chain is what makes a faulting kernel expensive: an illegal or misaligned access
leaves the context in a STICKY error state that no in-process caller can clear, so every later CUDA call in that
worker returns the same status. One fault therefore takes the whole chain down, and the reports name the innocent
tests that followed rather than the one that faulted — the run that motivated the guard showed 1 failure and 51
errors. A fault does not even have to fail anything: an `xfail(strict=False)` swallows it and the context stays
poisoned regardless. So the root `conftest.py` probes the context after every test (the `deviceSynchronize` probe the
bench worker already relies on in `_bench_worker._context_dirty`) and stops the process at the NEXT test's setup —
deferred deliberately, because dying mid-test leaves the controller re-queueing the poisoning test onto a fresh worker
and poisoning that one too. Under xdist the controller respawns the worker on a clean context and reschedules only
what it had not reached, so the run still finishes and reports exactly one failure, named.

Those costs come from `tests/durations.json` — a checked-in nodeid → seconds map — with the box's own pytest cache
overlaid on top. The committed file exists because CI starts every job with an empty cache: without a baseline the
bucketing never fired there and the long poles landed wherever chance put them. It records only entries at or above
0.05 s (a few hundred lines rather than the full ~2600, and 99% of the suite's wall time); anything unlisted is
assumed to cost 0.05 s. Regenerate it with `make test-durations` — which REPLACES the file with that run's timings, so
renamed and deleted tests drop out instead of lingering as ghost slots the bucketer plans around. The refresh runs on
one xdist loadgroup worker: execution stays serial, while CUDA node IDs keep the canonical ``@cuda`` / ``@cuda-cli``
suffixes the parallel suite uses for lookup. Point it at the whole suite, never a subset.

Two things keep it honest. `make test` passes `--durations=25`, so every run (CI included) prints its slowest tests and
a new long pole shows up in the log immediately. And the session-end gate in `conftest.py` fails any run where a test
took **5 s or more without being in the baseline**, naming the offenders and asking for `make test-durations`. The bar
sits far above the 0.05 s recording threshold on purpose — CI runners are several times slower than a dev box, and the
gap guarantees nothing near the threshold can drift across it. It is a session hook rather than a test case because
only the controller, and only after the last report, has every test's duration; an xdist worker sees just its own slice.

The `perf` marker gates **suite-wide**, not just `tests/perf/`: the root `tests/conftest.py` hook skips every
perf-marked item unless `-m perf` was passed, and since the root conftest loads for any `tests/` collection the gate
also covers subset runs like `pytest tests/serving/`. Reserve `perf` for two things — the
perf-comparison tests `make bench-kernels` runs, and tests that genuinely cannot ride the parallel suite (today the
two in-process vLLM engine tests, `test_vllm_plugin_gpu.py` / `generation/test_vllm_plugin_gen_gpu.py`: the engine demands a
large fraction of the card FREE at startup, plus checkpoint downloads and minutes of whole-model compile — since
`make bench-kernels` only runs `tests/perf/`, these two run nowhere by default; exercise them explicitly with
`pytest tests/serving/ -m perf` on a machine with the card mostly free). A perf
mark on anything else silently drops it from `make test` even on GPU machines (this hid the serving runner's GPU
correctness pins for a while). GPU correctness tests guard themselves with `requires_cuda` / `importorskip` instead.

`goldens` is the second off-lane marker, gated by the same hook. `tests/compiler/pipeline/search/test_golden.py`
strictly decodes every checked-in golden file — one case per file, so a card's rows go green as a whole when a tuning
round re-records them — and `make test-goldens` runs it. A full pass re-derives every recorded row's enumeration,
minutes per file, which is why it is off the default lane; the derivation memo
(`~/.cache/emmy/golden_identity.json`, keyed by compiler fingerprint and record content) makes a re-run cost only the
rows that actually changed.

Repository golden *qualification* is intentionally outside pytest. Model goldens are GPU-specific qualification
evidence, so the nightly `onboard-model` workflow validates the selected recipe-local file, strictly decodes every
row, and replays it on the named GPU. This keeps expensive model/card qualification out of the default suite. Only
the decode half is also reachable without the card, off the default lane (`make test-goldens` above) — it targets each
record's declared capability; the measured replay is what the nightly's GPU is for.

Optional adapter tests use `pytest.importorskip` for their own dependency extras. The network-free tiny Diffusers DiT
trace runs when the `image` extra is installed; the real checkpoint/CUDA comparison is additionally `perf`-marked and
requires `EMMY_RUN_DIT_PRETRAINED=1`, so normal CI never downloads the multi-gigabyte checkpoint.

`tests/compiler/helpers.py` exposes `device_compute_capability()` and the `requires_sm90` skip marker. The
mma.sync warp tier (swizzled `ldmatrix` + `mma.sync`, TMA transport) auto-enumerates and is validated on **sm_90+**;
on sm_80-89 it is pin-only and currently non-functional for two independent reasons — the `sm_NNa` arch-accelerated
target the TMA path emits is rejected by nvcc (`Unsupported gpu architecture 'sm_89a'`), and `ldmatrix` itself faults
at runtime on at least Ada (sm_89). Tests that **force** the warp tier via a warp `TILE` codec (`<atom>/…`) + `STAGE`
carry `requires_sm90` so they skip below sm_90 instead of faulting (a single warp-tier fault corrupts the shared `cuda`
context and cascades `cudaErrorIllegalAddress` into every later test on the worker, CUDA or not). The warp-tier matmul
coverage all lives in `test_matmul_coverage.py` — the scalar vs warp `TILE` accuracy/structure matrix, the
masked-symbolic sweep (symbolic M/N/K at off-hint sizes), the static-vs-dynamic parity across the `STAGE=d2/smem-async` and
`d2/smem-tma` transports, and the operand-pipelining transforms — the gmem→smem ring (`d<depth>/smem-async`) and the smem→register
double-buffer (`/p<n>`), each asserted **bit-identical** to the single-buffer / gmem-direct baseline (a pure perf
transform) — gating its GPU cases on `requires_sm90` / `_supports_tma()` (≥ sm_90); its GPU-less render / structure cases
run anywhere. The TMA accuracy path additionally exercises the host descriptor encoder (`backend/cuda/_tma.py`). The same
gate applies to TMA-transport `STAGE` pins (`…/tma…`) anywhere: below sm_90 the pin refuses rather than selecting a
different transport, so `test_attention_coverage.py`'s TMA-staged flash cases carry `requires_sm90` (their `cp`
siblings run on sm_80+). Golden-scoped CLI tests are the other environment trap: `--realization` without `--golden
PATH` and
`eval --dataset golden` resolve against the
**live card's** recordings, so tests asserting specific golden names (or monkeypatching `GOLDEN_RECORDS` with card-less
fakes) must pin themselves off-GPU (`torch.cuda.is_available → False` in-process, `CUDA_VISIBLE_DEVICES=""` for
`run_cli` subprocesses) to take the multi-card-union path — otherwise they pass or fail depending on which shapes the
local card happens to have recorded.
