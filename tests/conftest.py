"""Shared pytest fixtures for all test modules."""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

# Cross-process GPU lock for CUDA tests. Set on conftest import so every
# xdist worker (and any subprocess it spawns) coordinates on the same
# path. With this set, ``CudaBackend.run`` (via
# :func:`emmy.compiler.backend.cuda.program.run_program`) holds the
# lock end-to-end across compile + allocate + ``pre_run`` callback +
# kernel launches + ``.get()``. Tests that compare emmy against
# torch eager pass ``pre_run=<eager closure>`` so the eager forward
# and the emmy launches share one uninterrupted GPU window —
# without this serialization, peer-worker CUDA activity interleaves
# with our kernels and the per-position fp32 rounding drift breaks
# the accuracy comparison.
os.environ.setdefault("EMMY_GPU_LOCK", "/tmp/emmy-gpu.lock")


@pytest.fixture(autouse=True)
def _isolate_prior_file(tmp_path, monkeypatch):
    """Point the online-prior checkpoint at a per-test temp path so the
    greedy compile driver (which loads the global prior) never picks up a
    dev machine's ``~/.cache/emmy/online.json`` — tests stay deterministic
    (empty prior → option-0), and a test that tunes writes only its own file."""
    monkeypatch.setenv("EMMY_ONLINE_FILE", str(tmp_path / "prior.json"))


@pytest.fixture(autouse=True)
def _isolate_offline_file(monkeypatch):
    """Drop any dev-machine ``EMMY_OFFLINE_FILE`` override so tests always score
    through the repo-checked ``offline_weights.json``. Unlike the prior file, the
    default here must NOT be a tmp path — a missing offline artifact is a hard
    error by design (no silent fallback), and the shipped one is what tests
    exercise."""
    monkeypatch.delenv("EMMY_OFFLINE_FILE", raising=False)


@pytest.fixture(autouse=True)
def _seed_rng():
    """Pin RNGs for every test so numerical-tolerance assertions
    (e.g. ``test_torch_ops.test_unary``) don't flake on inputs that
    happen to land in tight regions. Determinism > tolerance — a real
    precision regression should still trip these tests.

    Also reseeds module-level ``rng = np.random.default_rng(...)``
    Generators in test modules. They're instantiated once at import,
    so successive ``rng.uniform`` calls inside parametrized tests
    drift across the session and produce order-dependent flakes
    (sigmoid/tanh/rsqrt at near-zero inputs etc.). Re-binding ``rng``
    to a fresh ``default_rng`` with the original seed restores
    intra-test determinism without changing any test's input
    distribution."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    for mod in list(sys.modules.values()):
        if mod is None or not getattr(mod, "__name__", "").startswith("tests."):
            continue
        rng = getattr(mod, "rng", None)
        if isinstance(rng, np.random.Generator):
            seed = getattr(mod, "_RNG_SEED", 0)
            mod.rng = np.random.default_rng(seed)


PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RECIPES_DIR = os.path.join(PROJECT_ROOT, "recipes")

# ── LPT static bucketing for pytest-xdist ───────────────────────────
# Record per-test call durations in the pytest cache; next run partitions
# items across N worker buckets via LPT (longest-processing-time-first)
# greedy — each item goes to the currently-lightest bucket. Buckets are
# tagged via @pytest.mark.xdist_group so `--dist=loadgroup` routes every
# item in a bucket to the same worker. Theoretical makespan is the load
# of the heaviest bucket (lower bound = longest single test).
#
# The pytest cache only helps a box that has already run the suite once —
# CI starts every job with an empty one, so the balancing never fired there
# and the long poles landed wherever chance put them. ``durations.json`` is
# the checked-in baseline that makes the FIRST run balanced: a nodeid → seconds
# map, regenerated with ``make test-durations`` (``pytest --write-durations``).
# It holds only the entries worth scheduling around (see ``_MIN_RECORDED``);
# anything absent is assumed cheap (``_UNKNOWN_COST``). A stale or partial
# baseline costs balance, never correctness — the cache overlays it, so a local
# run's own measurements always win over the committed numbers.

_DURATIONS_KEY = "test_durations/call"
_DURATIONS_FILE = os.path.join(os.path.dirname(__file__), "durations.json")
_CALL_DURATIONS: dict[str, float] = {}

#: Below this the entry is not worth a line in the baseline — a test this cheap
#: cannot move the makespan, and listing all of them would churn the file on
#: every rename.
_MIN_RECORDED = 0.05
#: What an unlisted test is assumed to cost when bucketing (see ``_MIN_RECORDED``).
_UNKNOWN_COST = 0.05
#: A test this slow MUST be in the baseline or the bucketing plans around a hole — the
#: staleness gate below fails the run until it is recorded. Deliberately far above
#: ``_MIN_RECORDED``: the bar has to survive the machine-speed spread (a CI runner is
#: several times slower than a dev box), and only a pole-sized test can actually distort
#: the plan. Nothing near the recording threshold can drift into this range.
_GATE_SECONDS = 5.0


def pytest_addoption(parser):
    parser.addoption(
        "--write-durations",
        action="store_true",
        help="Rewrite tests/durations.json (the checked-in LPT bucketing baseline) from this run's timings.",
    )


def pytest_runtest_logreport(report):
    if report.when == "call":
        _CALL_DURATIONS[report.nodeid] = report.duration


def _load_baseline() -> dict[str, float]:
    try:
        with open(_DURATIONS_FILE) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def pytest_sessionfinish(session):
    # ``workerinput`` marks an xdist WORKER — only the controller sees every
    # test's report, and letting each worker write would race on the file.
    is_controller = not hasattr(session.config, "workerinput")
    if session.config.getoption("--write-durations") and _CALL_DURATIONS and is_controller:
        # This run's measurements REPLACE the file rather than merging into it, so a
        # renamed or deleted test drops out on its own. Merging kept ghosts alive, and a
        # ghost is worse than a missing entry: the bucketer plans a slot for a test that
        # will never run (the two whole-card gate entries, 340 s and 150 s, outlived the
        # split into shards). Regenerate with `make test-durations`, which runs the WHOLE
        # suite — pointing this at a subset writes a baseline covering only that subset.
        fresh = {k: round(v, 2) for k, v in _CALL_DURATIONS.items() if v >= _MIN_RECORDED}
        with open(_DURATIONS_FILE, "w") as fh:
            json.dump(dict(sorted(fresh.items())), fh, indent=1)
            fh.write("\n")

    # Staleness gate: a test heavy enough to shape the schedule must be IN the baseline,
    # or CI plans its buckets around a hole and the new long pole lands wherever chance
    # puts it — the exact failure this whole mechanism exists to prevent. Checked here
    # rather than as a test case: only the controller, and only after the last report,
    # holds every test's duration (an xdist worker sees just its own slice).
    if is_controller and _CALL_DURATIONS:
        baseline = _load_baseline()
        missing = sorted(((d, n) for n, d in _CALL_DURATIONS.items() if d >= _GATE_SECONDS and n not in baseline), reverse=True)
        if missing:
            session.exitstatus = 1
            print(f"\nERROR: {len(missing)} test(s) at or over {_GATE_SECONDS:g}s are missing from {_DURATIONS_FILE}:")
            for d, n in missing:
                print(f"  {d:7.1f}s  {n}")
            print("Run `make test-durations` and commit the result, so the worker bucketing plans around them.")

    cache = getattr(session.config, "cache", None)
    if cache is None or not _CALL_DURATIONS:
        return
    existing = cache.get(_DURATIONS_KEY, {}) or {}
    existing.update(_CALL_DURATIONS)
    cache.set(_DURATIONS_KEY, existing)


def _num_workers(config) -> int | None:
    """Mirror xdist's -n resolution: int, 'auto', 'logical', or None."""
    try:
        n = config.getoption("numprocesses", None)
    except ValueError:
        return None
    if n in (None, 0):
        return None
    if isinstance(n, int):
        return n if n >= 1 else None
    if n in ("auto", "logical"):
        return os.cpu_count() or 1
    try:
        return int(n)
    except (TypeError, ValueError):
        return None


def _is_cuda_item(item) -> bool:
    """True iff this test item issues CUDA work.

    Detected via (a) a ``skipif`` marker whose reason starts with
    ``"CUDA not available"`` (the ``requires_cuda`` decorator used across
    ``tests/compiler/``), (b) a ``[cuda...]`` callspec id (the
    ``run_graph`` fixture's third variant + every ``test_e2e_accuracy``
    parametrization), or (c) an explicit ``xdist_group("cuda")`` marker
    (the ``tests/serving/*_gpu.py`` pytestmark convention). The explicit
    marker MUST be honored here: otherwise the LPT bucketing below adds a
    function-level ``w<N>`` group that shadows the module-level ``cuda``
    mark (``get_closest_marker`` prefers function-level), scattering the
    test off the serialized CUDA worker. One of those signals is true for
    every test that actually touches the device today; new CUDA-using
    tests inherit routing for free as long as they reuse the conventions."""
    for mark in item.iter_markers(name="skipif"):
        reason = mark.kwargs.get("reason", "")
        if isinstance(reason, str) and reason.startswith("CUDA not available"):
            return True
    for mark in item.iter_markers(name="xdist_group"):
        if mark.args and mark.args[0] == _CUDA_GROUP:
            return True
    nid = item.nodeid
    return "[cuda" in nid or "-cuda-" in nid or nid.endswith("-cuda]")


# xdist_group for every IN-PROCESS CUDA-touching test. The host only has
# one GPU; running CUDA tests across multiple xdist workers concurrently
# would mean two processes pushing kernels onto the same device. Even
# with ``EMMY_GPU_LOCK`` serializing the kernel-launch window and
# ``backend.run(pre_run=...)`` pulling the torch eager forward into the
# same lock, multi-kernel attention schedules still occasionally drift
# enough across worker contexts (per-context SM scheduling differs, and
# fp32 atomic-add commit order with it) to break the
# ``test_attention_chains`` / ``test_block_accuracy`` 1e-4 thresholds.
# Pinning all in-process CUDA tests to one group makes them run
# sequentially on one worker; non-CUDA tests still parallelize via the
# LPT buckets below.
_CUDA_GROUP = "cuda"

# Separate group for CUDA tests that drive the CLI through the ``run_cli``
# fixture: each spawns a FRESH subprocess with its own CUDA context, so
# they don't share the in-process worker's context and don't need to ride
# the (long) ``cuda`` chain. They still need bounded concurrency — left
# ungrouped, ~30 workers can each hold a live CUDA subprocess (~1 GB a
# piece) and OOM the card — so they serialize among themselves on a
# second worker, in parallel with the in-process chain. (Sharding this
# chain 3-way was tried and bought only ~5 s — the in-process ``cuda``
# chain is the critical path — so one shard keeps it simple.)
_CUDA_CLI_GROUP = "cuda-cli"


# ``tryfirst``: xdist's worker-side ``WorkerInteractor.pytest_collection_modifyitems``
# bakes each item's ``xdist_group`` into the nodeid it reports to the
# controller's loadgroup scheduler — and pluggy calls it BEFORE a plain
# conftest hook (the interactor registers after conftests, so LIFO order
# puts it first). Without ``tryfirst`` every marker added here lands too
# late: the routing silently degrades to plain ``load`` and CUDA tests
# scatter across workers (concurrent CUDA contexts → flaky GPU OOM in
# the ``run_cli`` subprocess tests, accuracy drift in attention chains).
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    import heapq

    # Deselect ``perf`` unless explicitly requested. Lives here — not in
    # ``tests/perf/conftest.py`` — so the gate holds for ANY ``tests/``
    # collection (e.g. ``pytest tests/serving/``), not only runs that happen
    # to collect ``tests/perf/`` and load its conftest.
    selected = config.getoption("-m") or ""
    if "perf" not in selected:
        skip_perf = pytest.mark.skip(reason="perf marker not selected; run with `pytest -m perf`")
        for item in items:
            if "perf" in item.keywords:
                item.add_marker(skip_perf)

    # Step 1: pin every CUDA-touching item to an xdist_group so each
    # chain lands on one worker and runs sequentially — ``cuda`` for
    # in-process device work, ``cuda-cli`` for ``run_cli`` subprocess
    # tests (own CUDA context per subprocess; see the group comments
    # above). Skip the LPT bucketing for those items entirely — they're
    # already grouped.
    cuda_items: list = []
    other_items: list = []
    for it in items:
        if _is_cuda_item(it):
            group = _CUDA_CLI_GROUP if "run_cli" in getattr(it, "fixturenames", ()) else _CUDA_GROUP
            it.add_marker(pytest.mark.xdist_group(group))
            cuda_items.append(it)
        else:
            other_items.append(it)

    # The committed baseline first, this box's own cache over it — a local run's
    # measurements beat the checked-in numbers on the machine that took them,
    # while CI (empty cache) still gets a balanced first run off the baseline.
    durations = _load_baseline()
    cache = getattr(config, "cache", None)
    if cache is not None:
        durations.update(cache.get(_DURATIONS_KEY, {}) or {})
    nworkers = _num_workers(config)
    if not durations or nworkers is None or nworkers < 2:
        items[:] = cuda_items + other_items
        return

    def dur(item) -> float:
        return durations.get(item.nodeid, _UNKNOWN_COST)

    sorted_others = sorted(other_items, key=dur, reverse=True)

    # Reserve one worker per CUDA group (``cuda`` + ``cuda-cli``);
    # LPT-bucket the rest across the remaining workers. With small
    # nworkers we fall back to a single bucket (no-op grouping). Sum
    # CUDA-item durations into one CUDA load so it competes for ordering
    # with the other heavy buckets.
    #
    # Off-GPU there is nothing to reserve FOR: every CUDA item skips in
    # microseconds, so the two chains cost nothing and holding workers back
    # for them just shrinks the pool. That was the CI shape — a 4-core runner
    # squeezed the whole suite onto 2 workers to reserve 2 for chains of
    # pure skips.
    cuda_load = sum(dur(it) for it in cuda_items)
    other_workers = max(1, nworkers - (2 if torch.cuda.is_available() else 0))

    # LPT: pop the lightest bucket, add this item, push back.
    buckets: list[tuple[float, int, list]] = [(0.0, w, []) for w in range(other_workers)]
    heapq.heapify(buckets)
    for it in sorted_others:
        load, wid, bucket = heapq.heappop(buckets)
        bucket.append(it)
        heapq.heappush(buckets, (load + dur(it), wid, bucket))

    # Tag non-CUDA items with their bucket's xdist_group so loadgroup
    # routes them together. CUDA items keep their pre-applied ``cuda`` /
    # ``cuda-cli`` group from step 1.
    buckets_sorted = sorted(buckets, key=lambda b: -b[0])
    reordered: list = []
    for _load, wid, bucket in buckets_sorted:
        group = f"w{wid}"
        for it in bucket:
            it.add_marker(pytest.mark.xdist_group(group))
            reordered.append(it)
    # Put CUDA bucket first when it dominates load, otherwise interleave
    # with the largest non-CUDA bucket. Heaviest-first dispatch lets xdist
    # start the longest serial chain immediately.
    if cuda_load >= buckets_sorted[0][0]:
        items[:] = cuda_items + reordered
    else:
        items[:] = reordered + cuda_items


@pytest.fixture(scope="session")
def project_root():
    """Absolute path to the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def recipes_dir():
    """Absolute path to the recipes/ directory."""
    return RECIPES_DIR


@pytest.fixture(scope="session")
def run_cli(project_root):
    """Return a callable that invokes the emmy CLI as a subprocess."""

    def _run(*args):
        result = subprocess.run(
            [sys.executable, "-m", "emmy.emmy", *args],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        return result.returncode, result.stdout, result.stderr

    return _run


@pytest.fixture
def make_bench_config(recipes_dir):
    """Return a factory that writes a temporary bench config.yaml."""

    def _make(tmp_dir):
        config = {
            "benchmark": {
                "local_results_dir": os.path.join(str(tmp_dir), "results"),
                "model_dir": "/hf_models",
            },
        }
        config_path = os.path.join(str(tmp_dir), "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path

    return _make


# ── Compiler dump fixture ──────────────────────────────────────────


@pytest.fixture
def dump_dir(request):
    """Dump compilation artifacts to _test_data/<test_name>/ for manual inspection."""
    safe_name = request.node.name.replace("[", "_").replace("]", "_").replace("/", "_")
    dump_path = Path(PROJECT_ROOT) / "_test_data" / safe_name

    from emmy.compiler.pipeline.dump import CompilerDump

    return CompilerDump(dir=dump_path)


# ── Unit-test fixtures ──────────────────────────────────────────────


@pytest.fixture
def tmp_recipe_dir(tmp_path):
    """Create a temp directory with a sample recipe.yaml using matrices format."""
    recipe = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "context_length": 8192,
                "vllm": {
                    "image": "vllm/vllm-openai:v0.17.0",
                },
            }
        },
        "benchmark": {
            "max_concurrency": 128,
            "num_prompts": 256,
            "random_input_len": 4000,
            "random_output_len": 4000,
        },
        "matrices": [
            {
                "deploy.gpu": "NVIDIA GeForce RTX 5090",
                "deploy.gpu_count": 1,
            },
            {
                "deploy.gpu": "NVIDIA H200 141GB",
                "deploy.gpu_count": 8,
                "engine.llm.tensor_parallel_size": 8,
                "engine.llm.context_length": 16384,
                "engine.llm.vllm.extra_args": "--kv-cache-dtype fp8",
                "benchmark.random_input_len": 8000,
                "benchmark.random_output_len": 8000,
            },
            {
                "deploy.gpu": "NVIDIA H100 80GB",
                "deploy.gpu_count": 4,
                "engine.llm.tensor_parallel_size": 4,
                "engine.llm.vllm.extra_args": "--kv-cache-dtype fp8",
            },
        ],
    }

    recipe_path = tmp_path / "recipe.yaml"
    with open(recipe_path, "w") as f:
        yaml.dump(recipe, f)

    return str(tmp_path)


@pytest.fixture
def sample_config():
    """Return a resolved config dict for testing compose generation."""
    return {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "context_length": 8192,
                "vllm": {
                    "image": "vllm/vllm-openai:v0.17.0",
                },
            }
        },
        "benchmark": {
            "max_concurrency": 128,
            "num_prompts": 256,
            "random_input_len": 4000,
            "random_output_len": 4000,
        },
    }


@pytest.fixture
def sample_config_sglang():
    """Return a resolved config dict for SGLang compose generation."""
    return {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "context_length": 8192,
                "sglang": {
                    "image": "lmsysorg/sglang:v0.5.9",
                },
            }
        },
        "benchmark": {
            "max_concurrency": 128,
            "num_prompts": 256,
            "random_input_len": 4000,
            "random_output_len": 4000,
        },
    }


@pytest.fixture
def sample_config_multi():
    """Return a resolved config dict for multi-instance testing."""
    return {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 4,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "context_length": 16384,
                "vllm": {
                    "image": "vllm/vllm-openai:v0.17.0",
                },
            }
        },
        "_num_instances": 2,
    }
