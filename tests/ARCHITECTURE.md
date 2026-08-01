# Test Architecture

## Overview

All tests use **pytest** with **pytest-asyncio** (`asyncio_mode = "auto"` in `pyproject.toml`) and live in the `tests/` directory, organized into subdirectories that mirror the `emmy/` source tree. Tests are designed to run without GPU hardware, Docker, or network access — every external interaction is avoided via dry-run mode or by testing pure functions directly.

## Directory Structure

`tests/` mirrors the `emmy/` source tree: a test directory exists because a source package does, and a test
module is named for the source module it covers. To find the tests for `emmy/<a>/<b>.py`, look in
`tests/<a>/test_<b>.py`. The one file at the root that belongs to no package is `conftest.py` — shared fixtures
plus the CUDA / LPT xdist routing hook (see **Running**).

Mirroring is the rule, not a coincidence — when a source package grows subpackages, the test directory follows.
`tests/compiler/pipeline/search/` is the worked example: its `data/`, `policy/`, and `prior/` subdirectories exist
because `emmy/compiler/pipeline/search/` has them, so a test for `policy/greedy.py` lives in `policy/`, one for
`prior/offline.py` in `prior/`, and only tests of the package's own top-level modules (`db.py`, `features.py`,
`slice.py`, …) stay flat. Tests that span several modules of a package — a cross-cutting property like
deploy-pick order invariance, or a process-wide cache over two subsystems — sit at the level that owns all of
them, not inside one arbitrary child.

Four directories break the mirror deliberately, because their organizing axis is the *kind* of test, not the
source module:

| Directory | Axis |
|---|---|
| `compiler/e2e/` | end-to-end coverage matrices — the whole pipeline per regime (matmul / reduce / attention / fused), not per pass |
| `compiler/cli/` | `emmy <command>` as a subprocess, via the `run_cli` fixture |
| `compiler/fixtures/` | checked-in traces and model configs, not tests |
| `perf/` | GPU perf comparison vs PyTorch, gated by the `perf` marker (see `tests/perf/ARCHITECTURE.md`) |

`compiler/passes/` and `compiler/perf/` carry their own `ARCHITECTURE.md`; read those before adding to them.

## Test Layers

The suite runs in four layers, distinguished by what they touch rather than by where they live:

- **Unit** — pure functions and dataclasses with synthetic inputs. No I/O. The bulk of the suite.
- **CLI dry-run** — the full argument-parsing → config-loading → orchestration path invoked as a subprocess with
  `--dry-run`, stopping just before any real side effect (SSH, Docker, file writes). Covers `deploy ssh/local/cloud`,
  `bench`, `teardown`, and `vm create/delete`. These use real recipes from `recipes/` so config drift fails a test.
- **GPU** — guarded by `requires_cuda` / `requires_sm90` / `importorskip` so they skip cleanly off-GPU, and routed
  onto a serial worker chain by the root conftest (see **Running**).
- **End-to-end** — a traced model or snippet through the whole compiler, compared against PyTorch eager or numpy.

A test belongs to the lowest layer that can prove the property. Reach for a subprocess or a GPU only when the
behavior genuinely lives there — each costs roughly an order of magnitude more wall time than the layer below it.

## Shared Fixtures (`conftest.py`)

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

## Conventions

- **Prefer combinatorial coverage matrices over per-capability test files.** The compiler e2e suite covers each
  regime with one parameterized matrix (`test_matmul_coverage.py`'s tile × stage × reduce × static/dynamic grid,
  `test_reduce_coverage.py`, `test_attention_coverage.py`, `test_fused_edge.py`'s producer × tier product) rather
  than a file per capability. When a legacy one-off test's behavior is subsumed by such a matrix (or by the matmul
  coverage matrix specifically), DELETE the one-off — do not maintain both. A new capability extends the nearest
  matrix with a parameter/case before it earns its own file.
- **Async tests** — tests for async functions are plain `async def` (no decorator needed; `asyncio_mode = "auto"` handles it). Mock async callables with `AsyncMock`.
- **No mocking** — dry-run mode is the primary strategy for testing command orchestration without side effects.
- **Real recipes** — CLI dry-run tests use recipes from the `recipes/` directory to catch config drift.
- **Temp recipes** — unit tests and multi-instance edge cases create throwaway recipes via `tmp_path`.
- **Plain functions** — no test classes; tests are grouped by file and separated with comment headers.
- **Assertions on stdout** — dry-run tests verify that the correct commands and messages appear in the expected order.
- **Mirror source layout, subpackages included** — test directories match `emmy/` subdirectories (e.g.
  `tests/deploy/` ↔ `emmy/deploy/`), and that holds all the way down: when a source package gains a subpackage,
  its tests move into a matching test subpackage rather than staying flat beside their new siblings. One test
  module per source module; a file covering several modules of a package sits at the level that owns them all.
  The exceptions are the four kind-organized directories listed under **Directory Structure**.
- **One file per subject, not per bug.** A behavior discovered later belongs in the file that already owns its
  subject, as a new section with a comment header — not in a new file named after the incident. Several small
  files re-declaring the same fixtures is the signal to merge them; a file whose sections share no scaffolding
  and no subject is the signal to leave them apart.
- **Known failures are marked inline** with `@pytest.mark.xfail`, carrying a reason that says what was
  removed or broken and when it should come back.
- **One exception: the xfail registry** (`tests/xfail_registry.py`). When a whole subsystem is removed on purpose,
  the casualties span dozens of files and inline marks would bury the intent in unrelated modules; the registry
  lists the exact node ids in one place with one reason, and the root `conftest.py` applies a non-strict
  `xfail` mark at collection. Exact ids, never path globs — each id is an acceptance obligation for the
  replacement, and the file shrinking to empty is the completion gate. Today it holds the tests the removed tile
  scheduler took down. Reach for it only for a deliberate removal of that size; an ordinary known failure still
  gets an inline mark.

## Running

```bash
pytest tests/ -v                       # all tests (skips `perf`-marked tests)
pytest tests/deploy/test_recipe.py -v  # single file
pytest tests/planner/ -v               # single directory
pytest tests/perf/ -m perf -v          # GPU perf suite (see tests/perf/ARCHITECTURE.md)
```

Under `make test` (`-n auto --dist=loadgroup`) the root `conftest.py` routes every CUDA-touching test onto two
serial chains via dynamic `xdist_group` markers — `cuda` for in-process device work (one shared context, keeps
the attention-chain accuracy thresholds deterministic) and `cuda-cli` for `run_cli` subprocess tests (each owns a
fresh CUDA context; bounding their concurrency prevents GPU OOM from ~30 simultaneous subprocesses). CUDA items
are detected via the `requires_cuda` skipif reason, a `[cuda...]` callspec id, or an explicit
`xdist_group("cuda")` pytestmark (the `tests/serving/*_gpu.py` convention — honoring it matters because the LPT
bucketing would otherwise add a function-level group that shadows the module-level mark). The hook is
`tryfirst` because xdist's worker-side hook bakes group names into nodeids before plain conftest hooks run —
without it the markers land too late and CUDA tests silently scatter across workers. Non-CUDA tests are
LPT-bucketed across the remaining workers using the cached duration table.

The `perf` marker gates **suite-wide**, not just `tests/perf/`: collecting `tests/` loads `tests/perf/conftest.py`,
whose hook skips every perf-marked item unless `-m perf` was passed. Reserve `perf` for two things — the
perf-comparison tests `make bench-kernels` runs, and tests that genuinely cannot ride the parallel suite (today the
two in-process vLLM engine tests, `test_vllm_plugin_gpu.py` / `test_vllm_plugin_gen_gpu.py`: the engine demands a
large fraction of the card FREE at startup, plus checkpoint downloads and minutes of whole-model compile). A perf
mark on anything else silently drops it from `make test` even on GPU machines (this hid the serving runner's GPU
correctness pins for a while). GPU correctness tests guard themselves with `requires_cuda` / `importorskip` instead.

Optional adapter tests use `pytest.importorskip` for their own dependency extras. The network-free tiny Diffusers DiT
trace runs when the `image` extra is installed; the real checkpoint/CUDA comparison is additionally `perf`-marked and
requires `EMMY_RUN_DIT_PRETRAINED=1`, so normal CI never downloads the multi-gigabyte checkpoint.

`tests/compiler/conftest.py` also exposes `device_compute_capability()` and the `requires_sm90` skip marker. The
mma.sync warp tier (swizzled `ldmatrix` + `mma.sync`, TMA transport) auto-enumerates and is validated on **sm_90+**;
on sm_80-89 it is pin-only and currently non-functional for two independent reasons — the `sm_NNa` arch-accelerated
target the TMA path emits is rejected by nvcc (`Unsupported gpu architecture 'sm_89a'`), and `ldmatrix` itself faults
at runtime on at least Ada (sm_89). Tests that **force** the warp tier via a warp `TILE` codec (`a:<atom>/…`) + `STAGE`
carry `requires_sm90` so they skip below sm_90 instead of faulting (a single warp-tier fault corrupts the shared `cuda`
context and cascades `cudaErrorIllegalAddress` into every later test on the worker, CUDA or not). The warp-tier matmul
coverage all lives in `test_matmul_coverage.py` — the scalar vs warp `TILE` accuracy/structure matrix, the
masked-symbolic sweep (symbolic M/N/K at off-hint sizes), the static-vs-dynamic parity across the `STAGE=d2/cp` and
`d2/tma` transports, and the operand-pipelining transforms — the gmem→smem ring (`d<depth>/cp`) and the smem→register
double-buffer (`/p<n>`), each asserted **bit-identical** to the single-buffer / gmem-direct baseline (a pure perf
transform) — gating its GPU cases on `requires_sm90` / `_supports_tma()` (≥ sm_90); its GPU-less render / structure cases
run anywhere. The TMA accuracy path additionally exercises the host descriptor encoder (`backend/cuda/_tma.py`). The same
gate applies to TMA-transport `STAGE` pins (`…/tma…`) anywhere: below sm_90 the pin declines and the kernel stays
gmem-direct, so `test_attention_coverage.py`'s TMA-staged flash cases carry `requires_sm90` (their `cp` siblings run on
sm_80+). Golden-scoped CLI tests are the other environment trap: `--golden` / `--dataset golden` resolve against the
**live card's** recordings, so tests asserting specific golden names (or monkeypatching `GOLDEN_CONFIGS` with card-less
fakes) must pin themselves off-GPU (`torch.cuda.is_available → False` in-process, `CUDA_VISIBLE_DEVICES=""` for
`run_cli` subprocesses) to take the multi-card-union path — otherwise they pass or fail depending on which shapes the
local card happens to have recorded.
