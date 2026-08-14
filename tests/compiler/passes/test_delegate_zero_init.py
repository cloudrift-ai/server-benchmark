"""Tests for ``lowering/cuda/005_delegate_zero_init``: an atomic accumulator's per-launch
zero-init rides a dataflow-predecessor kernel as a ``ZeroPrologue`` (WS2 of the decode
kernel-count plan) — and the capture's FIRST atomic kernel, with no kernel predecessor, keeps
its runtime memset. GPU-less: compiles a chained F32-matmul graph against a mocked 5090 target
with the atomic split pinned via ``EMMY_KNOBS``. F32 is intentional: an atomic split may apply
only a distributive projection per partition; an F16 tensor-result rounding boundary must happen
once after the complete reduction and therefore uses the deferred-kernel finalize."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from emmy.compiler.context import Context
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline


def _compile_chain(n: int):
    """``(x @ w1) @ w2`` F32 with the matmul seam cut and ``REDUCE=g4a`` pinned on each
    piece — two atomic kernels whose outputs are ``32×n`` (the delegation-cap lever:
    512 → 64 KB per accumulator, well under ``_MAX_DELEGATED_WORDS``; 3840 → 480 KB,
    past it)."""
    mp = pytest.MonkeyPatch()
    # Per-knob vars, NOT EMMY_KNOBS: the aggregate splat runs at knob-module IMPORT time, long
    # before this fixture — and calling apply_knobs_env here would write env keys monkeypatch
    # never cleans up.
    mp.setenv("EMMY_TILE", "f2x2")
    mp.setenv("EMMY_WORK", "t16x16")
    mp.setenv("EMMY_REDUCE", "g4a")
    # Gate-free loop fusion composes the two matmuls. This test is about CUDA zero-init
    # delegation between kernels, so route the recognized contraction seam back to two pieces.
    mp.setenv("EMMY_PLACE", "cut")
    mp.setenv("EMMY_RASTER", "")
    try:
        graph = Graph()
        graph.add_node(InputOp(), [], Tensor("x", (32, 3840), dtype=F32), node_id="x")
        graph.add_node(InputOp(), [], Tensor("w1", (3840, n), dtype=F32), node_id="w1")
        graph.add_node(InputOp(), [], Tensor("w2", (n, n), dtype=F32), node_id="w2")
        graph.add_node(MatmulOp(), ["x", "w1"], Tensor("m1", (32, n), dtype=F32), node_id="m1")
        graph.add_node(MatmulOp(), ["m1", "w2"], Tensor("m2", (32, n), dtype=F32), node_id="m2")
        graph.inputs, graph.outputs = ["x", "w1", "w2"], ["m2"]
        ctx = Context.from_target((12, 0), gpu_name="NVIDIA GeForce RTX 5090")
        return Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx)
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def chained_atomic():
    """The small chain: the first matmul has no kernel predecessor (keeps its memset), the
    second delegates its 64 KB accumulator to the first."""
    return _compile_chain(512)


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


def test_oversized_accumulator_keeps_its_memset():
    """A 480 KB accumulator (32×3840 F32) sits past ``_MAX_DELEGATED_WORDS`` — one CTA zeroing
    it serially costs ~10× the MEMSET node it would replace (the m64 gate_up workspace burned
    14 µs/launch at 983 KB), so delegation is refused and every atomic output keeps its runtime
    memset."""
    ops = _cuda_ops(_compile_chain(3840))
    assert not any(op.zero_prologues for op in ops.values()), "no kernel may carry a delegated zero past the cap"
    atomics = [op for op in ops.values() if op.zero_outputs]
    assert len(atomics) >= 2, "both atomic matmuls keep their runtime memsets"
