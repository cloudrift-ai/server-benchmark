"""Tests for ``lowering/cuda/005_delegate_zero_init``: an atomic accumulator's per-launch
zero-init rides a dataflow-predecessor kernel as a ``ZeroPrologue`` (WS2 of the decode
kernel-count plan) — and the capture's FIRST atomic kernel, with no kernel predecessor, keeps
its runtime memset. GPU-less: compiles a chained-matmul snippet against a mocked 5090 target
with the atomic split pinned via ``EMMY_KNOBS``."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from emmy.compiler.context import Context
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline


@pytest.fixture(scope="module")
def chained_atomic():
    """``(x @ w1) @ w2`` fp16 with ``REDUCE=g4a`` pinned on every matmul: two atomic kernels,
    the first with no kernel predecessor (keeps its memset), the second delegating to the
    first."""
    mp = pytest.MonkeyPatch()
    # Per-knob vars, NOT EMMY_KNOBS: the aggregate splat runs at knob-module IMPORT time, long
    # before this fixture — and calling apply_knobs_env here would write env keys monkeypatch
    # never cleans up.
    mp.setenv("EMMY_TILE", "a:mma_m16n8k16_f16_f32/w1x8/f2x2/k4")
    mp.setenv("EMMY_REDUCE", "g4a")
    mp.setenv("EMMY_RASTER", "")
    mp.setenv("EMMY_WSPEC", "")
    try:
        from emmy.commands.trace import graph_from_code

        code = (
            "x = torch.randn(32,3840,dtype=torch.float16)\n"
            "w1 = torch.randn(3840,3840,dtype=torch.float16)\n"
            "w2 = torch.randn(3840,3840,dtype=torch.float16)\n"
            "(x @ w1) @ w2"
        )
        graph, _slug, _bundle = graph_from_code(code)
        ctx = Context.from_target((12, 0), gpu_name="NVIDIA GeForce RTX 5090")
        yield Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx)
    finally:
        mp.undo()


def _cuda_ops(g):
    return {nid: n.op for nid, n in g.nodes.items() if isinstance(n.op, CudaOp)}


def test_second_atomic_delegates_to_first(chained_atomic):
    ops = _cuda_ops(chained_atomic)
    delegating = [(nid, op) for nid, op in ops.items() if op.zero_prologues]
    assert len(delegating) == 1, f"exactly one kernel should carry the delegated zero: {[(n, o.zero_prologues) for n, o in delegating]}"
    (pid, pop) = delegating[0]
    (target,) = pop.zero_prologues
    # the delegating kernel's launch passes the target buffer as an arg, and its source zeroes it
    assert target in pop.arg_order
    assert "__zp" in pop.kernel_name  # re-suffixed: launches resolve kernels by name
    assert "blockIdx.x == 0" in pop.kernel_source
    # the delegated buffer no longer memsets on its own kernel
    top = ops[target]
    assert target not in top.zero_outputs


def test_first_atomic_keeps_its_memset(chained_atomic):
    ops = _cuda_ops(chained_atomic)
    # the first matmul has no KernelOp predecessor — its own output stays in zero_outputs
    first = [op for nid, op in ops.items() if op.zero_prologues]
    assert first, "delegating kernel exists"
    delegator = first[0]
    assert delegator.kernel_name != ""
    assert any(op.zero_outputs for op in ops.values()), "the capture's first atomic kernel keeps its runtime memset"
