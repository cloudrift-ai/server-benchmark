"""Reusable support for compiler tests.

Pytest fixtures stay in ``conftest.py``. Ordinary callables and shared skip markers live here so their dependencies
are explicit at each call site and test modules never import implementation from another test module.
"""

from __future__ import annotations

import asyncio
import functools
from itertools import product

import numpy as np
import pytest


def classic_cartesian_assignments(context):
    """Enumerate a classic context's literal kernel × node × edge test oracle."""
    from emmy.compiler.ir.schedule import Schedule, ScheduleRefused

    nodes = tuple(context.node_choices(site) for site in context.tile_op.node_sites)
    edges = tuple(context.edge_choices(site) for site in context.tile_op.edge_sites)
    for kernel, node_values, edge_values in product(context.kernels, product(*nodes), product(*edges)):
        assignment = Schedule(
            kernel,
            dict(zip(context.tile_op.node_sites, node_values, strict=True)),
            dict(zip(context.tile_op.edge_sites, edge_values, strict=True)),
        )
        try:
            context.extend(assignment)
        except (ScheduleRefused, TypeError):
            accepted = False
        else:
            accepted = True
        yield assignment, accepted


def enumerate_classic_reference(context):
    """Yield the accepted subset of the literal classic assignment product."""
    for assignment, accepted in classic_cartesian_assignments(context):
        if accepted:
            yield assignment


def has_cuda_gpu() -> bool:
    """Check if cupy is importable and sees at least one CUDA device."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def has_cuda_toolchain() -> bool:
    """Check that Emmy can both see a GPU and compile CUDA kernels."""
    if not has_cuda_gpu():
        return False
    from emmy.compiler.backend.cuda.nvcc import nvcc_path

    return nvcc_path() is not None


requires_cuda = pytest.mark.skipif(
    not has_cuda_toolchain(),
    reason="CUDA not available (need cupy + GPU + nvcc)",
)


@functools.cache
def device_compute_capability() -> tuple[int, int] | None:
    """Return the live CUDA device compute capability, or ``None`` when no GPU is visible."""
    if not has_cuda_gpu():
        return None
    import cupy as cp

    cap = str(cp.cuda.Device().compute_capability)
    return (int(cap[:-1]), int(cap[-1]))


# Skip the mma.sync warp-tier tests below sm_90. On sm_80-89 the pin-only path is currently non-functional because
# nvcc rejects the ``sm_NNa`` target and ``ldmatrix`` faults at runtime on at least Ada (sm_89).
requires_sm90 = pytest.mark.skipif(
    (device_compute_capability() or (0, 0)) < (9, 0),
    reason="mma.sync warp tier needs sm_90+ (pin-only / non-functional on sm_80-89: ldmatrix fault + sm_NNa TMA compile)",
)


def direct_classic_leaf(fork_point):
    """Select the direct classic schedule without flattening the schedule tree."""
    from emmy.compiler.pipeline.fork import Fork
    from emmy.compiler.pipeline.knob import SCHEDULE_FAMILIES, family_of

    def allowed(option) -> bool:
        knobs = option.knobs if isinstance(option, Fork) else getattr(option, "knobs", {})
        return all(family_of(key) not in SCHEDULE_FAMILIES or str(value) == "" for key, value in knobs.items())

    def descend(options):
        for option in options:
            if not allowed(option):
                continue
            if not isinstance(option, Fork) or option.is_leaf:
                return option
            try:
                return descend(option.expand())
            except ValueError:
                continue
        raise ValueError("fork has no direct classic schedule")

    return descend(fork_point.options)


def dtype_tol(dtype) -> dict[str, float]:
    """Return default ``np.testing.assert_allclose`` tolerances for a compiler dtype."""
    return {"rtol": 5e-3, "atol": 5e-3} if dtype.name == "f16" else {"rtol": 1e-4, "atol": 1e-5}


def dtype_input_scale(dtype) -> float:
    """Scale random inputs so large fp16 reductions do not overflow."""
    return 0.05 if dtype.name == "f16" else 1.0


def skip_if_no_cuda() -> None:
    """Skip the current test when Emmy cannot compile CUDA kernels."""
    if not has_cuda_toolchain():
        pytest.skip("CUDA not available (need cupy + GPU + nvcc)")


def matmul_graph(m: int, k: int, n: int):
    """Build a plain ``(m, k) @ (k, n) -> (m, n)`` graph for compiler tests."""
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.frontend.ir import MatmulOp

    graph = Graph()
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k)), node_id="a")
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n)), node_id="b")
    graph.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m, n)), node_id="o")
    graph.inputs = ["a", "b"]
    graph.outputs = ["o"]
    return graph


def inject_constants(input_data: dict[str, np.ndarray], graph) -> dict[str, np.ndarray]:
    """Supply scalar and rebound tensor constants omitted from backend input data."""
    from emmy.compiler.ir.base import ConstantOp

    for node_id, node in graph.nodes.items():
        if not isinstance(node.op, ConstantOp) or node_id in input_data:
            continue
        if node.op.value is not None:
            input_data[node_id] = np.array([node.op.value], dtype=np.float32)
            continue
        source = input_data.get(node.op.name)
        if source is not None:
            from emmy.compiler.loader.binder import apply_load_ops

            array = np.asarray(source, dtype=np.float32)
            if node.op.load_ops and node.op.source_shape is not None:
                array = array.reshape(node.op.source_shape)
            input_data[node_id] = apply_load_ops(array, node.op.load_ops)
    return input_data


def drain_tune(pipeline, graph, *, on=None, **kwargs):
    """Synchronously collect :meth:`Pipeline.tune_async` terminals for tests."""

    async def _collect():
        out = []
        try:
            async for candidate in pipeline.tune_async(graph, **kwargs):
                out.append(candidate)
                if on is not None and on(candidate):
                    break
        finally:
            backend = kwargs.get("backend")
            if backend is not None and hasattr(backend, "aclose_async_worker"):
                await backend.aclose_async_worker()
        return out

    return asyncio.run(_collect())


def run_inner_reward(
    fused_graph,
    *,
    ctx,
    db,
    backend=None,
    backends=None,
    patience,
    ucb_c=None,
    explore_eps=0.0,
    seed=0,
    progress=None,
    prior=None,
    run_id="",
):
    """Synchronously run the two-level strategy's separable terminal scoring for tests."""
    from emmy.compiler.pipeline import TuningSearch
    from emmy.compiler.pipeline.search.strategy import TwoLevelStrategy

    if ucb_c is None:
        ucb_c = TuningSearch.DEFAULT_UCB_C
    strategy = TwoLevelStrategy(
        db=db,
        patience=patience,
        backend=backend,
        backends=backends,
        ucb_c=ucb_c,
        explore_eps=explore_eps,
        progress=progress,
        prior_seed=seed,
        run_id=run_id,
        prior=prior,
    )
    return asyncio.run(strategy._evaluate_terminal(fused_graph, ctx))


def run_two_level(graph, *, ctx, **kwargs):
    """Synchronously run :class:`two_level.TwoLevelStrategy` for tests."""
    from emmy.compiler.pipeline.search.strategy import TwoLevelStrategy

    return asyncio.run(TwoLevelStrategy(**kwargs).run(graph, ctx))


def dyn_M(mode: str, M: int):
    """Return ``Dim('seq_len')`` for dynamic mode and the integer ``M`` otherwise."""
    if mode == "dynamic":
        from emmy.compiler.dim import Dim

        return Dim("seq_len")
    return M


def from_pretrained_or_skip(loader, *args, **kwargs):
    """Run a Hugging Face download, skipping the test when the Hub is unavailable."""
    try:
        return loader(*args, **kwargs)
    except OSError as exc:
        model = args[0] if args else kwargs.get("pretrained_model_name_or_path", "?")
        pytest.skip(f"HuggingFace Hub unavailable for {model} (likely rate-limited): {exc}")
