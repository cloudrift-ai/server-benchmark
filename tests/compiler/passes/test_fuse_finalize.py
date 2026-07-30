"""Tests for ``lowering/tile/032_fuse_finalize``: a deferred split-reduce finalize inlines into
its consumers' read sites under ``PLACE@fin=fuse`` (the decode-parity split-chain closer) — and
stays a separate kernel by default. GPU-less structural checks compile a norm-after-matvec
snippet against a mocked 5090 target with the deferred split pinned; the CUDA cell re-runs the
same composition on the live device and matches numpy."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from emmy.compiler.context import Context
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from tests.compiler.conftest import requires_cuda

_CODE = (
    "x = torch.randn(1,512,dtype=torch.float16)\n"
    "w = torch.randn(512,256,dtype=torch.float16)\n"
    "nw = torch.randn(256,dtype=torch.float16)\n"
    "F.rms_norm(torch.matmul(x, w), (256,), nw)"
)


def _compiled(fin_pin: str | None):
    """The norm-after-matvec graph lowered with ``REDUCE=g4k/b128t`` pinned on every reduce
    (matvec AND norm split to partial+finalize pairs) and ``PLACE@fin`` optionally pinned."""
    mp = pytest.MonkeyPatch()
    mp.setenv("EMMY_TILE", "")
    mp.setenv("EMMY_REDUCE", "g4k/b128t")
    mp.setenv("EMMY_RASTER", "")
    mp.setenv("EMMY_WSPEC", "")
    if fin_pin is not None:
        mp.setenv("EMMY_PLACE@FIN", fin_pin)
    try:
        from emmy.commands.trace import graph_from_code

        graph, _slug, _bundle = graph_from_code(_CODE)
        ctx = Context.from_target((12, 0), gpu_name="NVIDIA GeForce RTX 5090")
        return Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx)
    finally:
        mp.undo()


def _cuda_ops(g) -> dict[str, CudaOp]:
    return {nid: n.op for nid, n in g.nodes.items() if isinstance(n.op, CudaOp)}


def test_default_keeps_finalize_kernel():
    """Without a pin the built-in ``PLACE@fin`` default is ``cut`` — the matvec's deferred
    finalize stays its own kernel (partial + finalize pairs for both reduces: 4 kernels)."""
    ops = _cuda_ops(_compiled(None))
    assert len(ops) == 4, f"expected 4 kernels (2 partial+finalize pairs), got {sorted(ops)}"
    assert any(nid == "matmul_reduce" for nid in ops), f"matvec finalize node missing: {sorted(ops)}"
    assert not any("__fin" in op.kernel_name for op in ops.values())


def test_fuse_inlines_finalize_into_consumers():
    """``PLACE@fin=fuse`` dissolves the matvec finalize: its fold opens the norm partial AND the
    norm finalize (the body-level sweep reader), both renamed ``__fin``, both reading the
    ``__partial`` workspace; the dead output buffer is gone from every kernel signature."""
    g = _compiled("fuse")
    ops = _cuda_ops(g)
    assert len(ops) == 3, f"expected 3 kernels after the finalize inline, got {sorted(ops)}"
    assert "matmul_reduce" not in ops, "the matvec finalize node must dissolve"
    fused = [op for op in ops.values() if "__fin" in op.kernel_name]
    assert len(fused) == 2, f"norm partial + norm finalize must both carry __fin: {[op.kernel_name for op in ops.values()]}"
    for nid in ops:
        edges = g.nodes[nid].inputs
        assert "matmul_reduce" not in edges, f"{nid} still reads the dead finalize output"
    readers = [nid for nid in ops if "matmul_reduce__partial" in g.nodes[nid].inputs]
    assert readers, f"a fused consumer must gain the workspace edge: {[(nid, g.nodes[nid].inputs) for nid in ops]}"


@requires_cuda
def test_fuse_finalize_matches_numpy():
    """The inlined fold computes the same values: seed → in-order partition fold → projection,
    per read site. Full e2e on the live device vs numpy."""
    mp = pytest.MonkeyPatch()
    mp.setenv("EMMY_TILE", "")
    mp.setenv("EMMY_REDUCE", "g4k/b128t")
    mp.setenv("EMMY_RASTER", "")
    mp.setenv("EMMY_WSPEC", "")
    mp.setenv("EMMY_PLACE@FIN", "fuse")
    try:
        from emmy.commands.trace import graph_from_code
        from emmy.compiler.backend.cuda.backend import CudaBackend

        graph, _slug, bundle = graph_from_code(_CODE)
        module, args, kwargs = bundle
        be = CudaBackend()
        compiled = be.compile(graph)
        ins = {nid: t.detach().numpy() for nid, t in zip(graph.inputs, args, strict=False)}
        got = list(be.run(compiled, input_data=ins)[0].outputs.values())[0]
        ref = module(*args, **kwargs).detach().numpy().astype(np.float32)
        np.testing.assert_allclose(np.asarray(got).astype(np.float32).reshape(ref.shape), ref, atol=0.5, rtol=0.1)
    finally:
        mp.undo()
