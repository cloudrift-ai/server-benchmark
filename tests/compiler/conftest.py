"""Conftest for ``tests/compiler/``.

Defines the ``run_graph`` parametrized fixture that runs an accuracy test
through each backend (numpy / loop / cuda). A test that takes ``run_graph``
automatically executes three times under different param IDs — any
disagreement between backends makes bug attribution mechanical.

Reusable functions and skip markers live in ``helpers.py``; conftest contains only fixtures.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from tests.compiler.helpers import inject_constants, skip_if_no_cuda


@pytest.fixture(params=["f32", "f16"], ids=["f32", "f16"])
def dtype(request):
    """Parametrize a test over the float dtypes emmy supports.

    Yields a :class:`emmy.compiler.dtype.DataType`. Tests that take
    this fixture run once per dtype; combined with ``run_graph``, the
    full matrix is (backend × dtype) minus the loop/fp16 cell (skipped).
    """
    from emmy.compiler import dtype as _dt  # noqa: PLC0415

    return _dt.get(request.param)


@pytest.fixture(params=["numpy", "loop", "cuda"])
def run_graph(request) -> Callable:
    """Return a callable ``run(graph, input_data) -> dict[name, ndarray]``.

    Each parametrized variant routes through a different backend; the
    callable hides the compile/run split. ``input_data`` values are
    numpy arrays with declared shapes; outputs are ndarrays reshaped to
    match the graph's declared output shapes.

    When tests also take the ``dtype`` fixture (parametrized over
    ``[F32, F16]``), this fixture will skip the ``loop`` backend for the
    fp16 row — the cppyy-driven loop runner is hardcoded to ``float`` in
    its generated kernels and has no fp16 path today.
    """
    kind = request.param

    if kind == "cuda":
        skip_if_no_cuda()
    if kind == "loop":
        # If the test also takes a ``dtype`` fixture and it's fp16, skip
        # the loop backend (cppyy runner is f32-only — see
        # ``ir/loop/runner.py``). The fixture is opt-in, so tests that
        # don't request ``dtype`` aren't affected.
        dtype_node = request.node.callspec.params.get("dtype") if hasattr(request.node, "callspec") else None
        if dtype_node is not None and getattr(dtype_node, "name", None) == "f16":
            pytest.skip("loop backend (cppyy) has no fp16 path; covered by numpy + cuda")

    def _run(graph, input_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if kind == "numpy":
            from emmy.compiler.backend.numpy import NumpyBackend

            be = NumpyBackend()
            return be.run(be.compile(graph), input_data=input_data)[0].outputs
        if kind == "loop":
            from emmy.compiler.backend.loop import LoopBackend

            be = LoopBackend()
            compiled = be.compile(graph)
            augmented = inject_constants(dict(input_data), compiled)
            return be.run(compiled, input_data=augmented)[0].outputs
        # cuda
        from emmy.compiler.backend.cuda.backend import CudaBackend

        be = CudaBackend()
        compiled = be.compile(graph)
        augmented = inject_constants(dict(input_data), compiled)
        return be.run(compiled, input_data=augmented)[0].outputs

    return _run


@pytest.fixture(params=["static", "dynamic"])
def shape_mode(request) -> str:
    """Parametrize a test over a static vs dynamic (symbolic-M) shape. Any test
    that names this fixture runs once per mode — the static/dynamic parity
    automation. Pair with :func:`dyn_M` to flip a graph's leading/M axis:

        def test_x(shape_mode):
            g = build_graph(M=dyn_M(shape_mode, 256), ...)   # int 256 or Dim('seq_len')

    The dynamic mode compiles ONE symbolic kernel (``Dim('seq_len')``, runtime
    ``int seq_len`` arg) tiled at the 512 hint and run at the concrete M fed in
    the input arrays; the static mode bakes M. Use tile-divisor M for strict
    static-vs-dynamic parity (off-hint masked sizes live in the masked-symbolic
    section of ``test_matmul_coverage.py``)."""
    return request.param
