"""Knob pins whose contract is an emitted-source substring or a refusal.

Pinning rides on the ``EMMY_KNOBS="K1=V1,..."`` env-var mechanism (see
``emmy/compiler/pipeline/knob.py`` — ``apply_knobs_env`` splats the aggregate into per-knob
``EMMY_<K>=V`` vars at import time, and ``Knob.narrow`` overrides the schedule's candidate codecs
with the pinned value in ``lowering/tile/040_schedule`` so only the matching variant is built).

What this file kept is the half a corpus row cannot state: a substring the lowered kernel must (or
must not) carry, and a pin the scheduler must REFUSE — the pin names a kernel, so quietly lowering
a different schedule would deploy something the caller did not ask for. The accuracy half — "this
program under this pinned schedule computes the right answer" — moved to
``tests/compiler/realization/``, whose cases replay one program and one authored schedule through
offered → realized → built → correct.

Most tests here are compile-only; the two that reach nvcc are skipped without CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.compiler.helpers import pin_classic, requires_cuda


def _run_with_knobs(graph, inputs: dict[str, np.ndarray], out_name: str, knobs: dict, monkeypatch) -> np.ndarray:
    """Set the per-knob ``EMMY_<K>`` env vars (the same pinning
    mechanism ``EMMY_KNOBS=...`` uses after ``apply_knobs_env``
    splats it) so the partition planner filters its variant enumeration
    down to the single ``TileParams`` we want to verify, then compile
    + run via the CUDA backend."""
    from emmy.compiler.backend.cuda.backend import CudaBackend

    pin_classic(monkeypatch, knobs)

    be = CudaBackend()
    compiled = be.compile(graph)
    return be.run(compiled, input_data=inputs)[0].outputs[out_name]


# The eighth-golden-sweep TMA box regression: a warp register tile with tile_m > 256 (16 mma rows
# × w4 × f8 = 512) paired with a TMA stage encoded an A box of (512, bk) and crashed at
# ``cuTensorMapEncodeTiled`` ("TMA box dim 0 extent 512 outside the hardware range 1..256"). The
# warp stage resolver gates the box extent, and a pinned stage the resolver declines REFUSES —
# the pin names a kernel, so silently deploying its gmem-direct sibling would deploy something the
# user did not ask for (the same contract ``test_scalar_cpasync_pin_refuses_odd_stride`` states).
_OVERSIZED_BOX_KNOBS = {"TILE": "mma_m16n8k16_f16_f32/f8x2/k2", "WORK": "w4x2", "STAGE": "d2/smem-tma"}


def test_warp_tma_pin_refuses_oversized_box(monkeypatch):
    """fp16 warp matmul pinned to a 512-row register tile + TMA stage — the box-extent gate must
    refuse the pin instead of encoding an illegal (512, bk) descriptor box or silently selecting
    gmem-direct. Compile-only (the refusal is scheduler-side). No CUDA needed."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline

    g = _build_f16_matmul_graph(512, 512, 512)
    pin_classic(monkeypatch, _OVERSIZED_BOX_KNOBS)
    with pytest.raises(ValueError, match="does not resolve"):
        Pipeline.build(TILE_PASSES).run(g, ctx=Context.from_target((9, 0)))


# The 512³ fp16 shape the two pins here need. On cc>=9.0 the F16 atom is eligible whenever the
# K-loads are F16, so at >=512³ the greedy compile *prefers* the tensor-core variant — which is
# what makes an over-ceiling warp pin or a declined stage load-bearing here (at <=256³ greedy would
# reach the same tier on its own and the pin would prove nothing).
def _build_f16_matmul_graph(M: int, N: int, K: int):
    from emmy.compiler.dtype import F16
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.frontend.ir import MatmulOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("c", (M, N), dtype=F16), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g


def test_sgemm_inner_reduce_is_unrolled(monkeypatch):
    """``assembly/030_mark_unroll`` flags the small FMA inner reduce (the ``BK=32`` K
    loop, ≤ 64 trips) for ``#pragma unroll``, giving ptxas the register-resident
    operand reuse + ILP the hand-tuned SGEMM relies on (the ``TM=26`` hero tile runs
    at 255 regs / ~293 µs with the unroll, ~126 regs / ~384 µs without — the lever for
    the article's ~96 %-of-cuBLAS number). The K-outer pipeline loop (> 64 unrolled
    trips) stays rolled. Compile-only (inspects the kernel source). No CUDA needed."""
    from emmy.compiler.context import Context
    from emmy.compiler.ir.cuda.ir import CudaOp
    from emmy.compiler.pipeline import KERNEL_PASSES, Pipeline

    g = _build_2d_matmul_graph(_ARTICLE_DIMS)
    pin_classic(monkeypatch, {"TILE": "f4x26", "WORK": "t32x8", "STAGE": "d2/smem-tma"})
    res = Pipeline.build([*KERNEL_PASSES, "lowering/cuda"]).run(g, ctx=Context.from_target((12, 0)))
    src = "\n".join(n.op.kernel_source for n in res.nodes.values() if isinstance(n.op, CudaOp))
    assert "#pragma unroll" in src, "the small FMA inner reduce must be marked for #pragma unroll"


def test_flat_output_sweep_lowers_with_its_axis_bound(monkeypatch):
    """A fused matmul/stack projection can reach materialization as a zero-axis fold with no
    operand edge. Its boundary output sweep still wraps the projection body, and each scalar-tile
    cell must reference the sweep's bound coordinate rather than a suffixed undefined name."""
    import torch
    import torch.nn as nn

    pin_classic(
        monkeypatch,
        {
            "WORK": "t16x8",
            "TILE": "f2x2",
            "STAGE": "",
            # Serial K: the subject is the sweep's bound coordinate in ONE kernel, and an unpinned
            # REDUCE leaves the cross-CTA split fair game — a split's pieces are new kernels with
            # their own sweeps.
            "REDUCE": "",
            "LOOPIFY": "0",
            "INTERLEAVE_LOADS": "1",
            "VECTORIZE_LOADS": "1",
        },
    )

    from emmy.compiler.context import Context
    from emmy.compiler.ir.cuda.ir import CudaOp
    from emmy.compiler.pipeline import KERNEL_PASSES, Pipeline
    from emmy.compiler.trace.torch import trace_module

    class StackMatmul(nn.Module):
        def forward(self, x, a, b):
            return torch.stack((-x, torch.matmul(a, b)[..., :2]), dim=-1)

    graph = trace_module(
        StackMatmul(),
        (
            torch.randn(1, 4, 8, 2),
            torch.randn(1, 4, 8, 8),
            torch.randn(1, 4, 8, 8),
        ),
    )
    result = Pipeline.build([*KERNEL_PASSES, "lowering/cuda"]).run(graph, ctx=Context.from_target((7, 0)))
    source = "\n".join(node.op.kernel_source for node in result.nodes.values() if isinstance(node.op, CudaOp))
    assert "for (int a4 = 0; a4 < 2; a4++)" in source
    assert "a4__c" not in source


def test_output_sweep_declines_the_warp_tier(monkeypatch):
    """A matmul whose boundary store adds an output sweep cannot use the straight-line MMA
    fragment epilogue. Unpinned scheduling must keep the bound scalar fallback."""
    import torch
    import torch.nn as nn

    monkeypatch.setenv("EMMY_LOOPIFY", "0")

    from emmy.compiler.context import Context
    from emmy.compiler.ir.cuda.ir import CudaOp
    from emmy.compiler.pipeline import KERNEL_PASSES, Pipeline
    from emmy.compiler.trace.torch import trace_module

    class StackMatmul(nn.Module):
        def forward(self, x, a, b):
            return torch.stack((-x, torch.matmul(a, b)[..., :2]), dim=-1)

    graph = trace_module(
        StackMatmul(),
        (
            torch.randn(1, 4, 8, 2),
            torch.randn(1, 4, 8, 8),
            torch.randn(1, 4, 8, 8),
        ),
    )
    result = Pipeline.build([*KERNEL_PASSES, "lowering/cuda"]).run(graph, ctx=Context.from_target((7, 0)))
    source = "\n".join(node.op.kernel_source for node in result.nodes.values() if isinstance(node.op, CudaOp))
    # The output-sweep coordinate must be bound by the scalar kernel itself (loop or decode) —
    # the exact loop spelling is fusion-order-dependent and not the contract.
    assert "int a4" in source
    assert "mma.sync" not in source


@pytest.mark.parametrize("a_dtype", ["f8", "f32"])
def test_unrealizable_warp_pin_falls_back_to_a_bound_scalar_grid(a_dtype, monkeypatch):
    """A graph-wide warp pin can name a tier a sibling's operand dtypes do not select. The
    scheduler then leaves that term unmapped; scalar materialization must restore its free-axis
    grid rather than emit loads and stores that reference coordinates no thread binds.

    Two dtype-choice drops, one per arm: an ``f8`` ``a`` edge over 16-bit channels is a mixed-width
    byte gather no fp8 atom reads (and the 16-bit converting smem compute fill cannot carry a
    1-byte A), and an all-``f32`` contraction selects no tensor-core atom on any target. An
    f16-channel ``f32`` ``a`` would ride the converting fill instead — realizable, so it would not
    exercise the fallback."""
    from emmy.compiler.context import Context
    from emmy.compiler.dtype import F8E4M3, F16, F32
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.cuda.ir import CudaOp
    from emmy.compiler.ir.frontend.ir import LinearOp
    from emmy.compiler.pipeline import KERNEL_PASSES, Pipeline

    pin_classic(
        monkeypatch,
        {
            "WORK": "w1x1",
            "TILE": "mma_m8n8k4_f16_f32/f4x4/k8",
            "REDUCE": "",
            "STAGE": "",
            "LOOPIFY": "0",
            "RASTER": "",
        },
    )

    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (1, 8, 32), {"f8": F8E4M3, "f32": F32}[a_dtype]), node_id="x")
    graph.add_node(InputOp(), [], Tensor("w", (4, 32), F16 if a_dtype == "f8" else F32), node_id="w")
    graph.add_node(LinearOp(), ["x", "w"], Tensor("o", (1, 8, 4), F32), node_id="o")
    graph.inputs, graph.outputs = ["x", "w"], ["o"]

    result = Pipeline.build([*KERNEL_PASSES, "lowering/cuda"]).run(graph, ctx=Context.from_target((7, 0)))
    [source] = [node.op.kernel_source for node in result.nodes.values() if isinstance(node.op, CudaOp)]
    assert "int a0 =" in source and "int a1 =" in source
    assert "if (_gid < 32)" in source
    assert "mma.sync" not in source


@requires_cuda
def test_unstaged_atom_lowers_gmem_direct(monkeypatch):
    """When the greedy compile picks the tensor-core atom variant but its operands
    aren't staged for ``ldmatrix`` (``TMA=0`` + a deliberately-large warp register
    tile whose slabs don't fit the smem budget), ``005_lower_atom_tile`` lowers them
    to a **gmem-direct fragment load** (``emmy_mma_load_{a,b}_gmem``) instead of
    raising — ldmatrix is smem-only, so the gmem path lets an unstageable MMA tile
    compile rather than crash. Compile-only (inspects the kernel source).

    Two facts make this fire: (1) the over-ceiling ``FM=26`` warp register pin is
    authoritative (``warp_reg_offers`` bypasses the ``_MAX_WARP_CELLS`` search
    ceiling for a full pin), so the warp build proceeds; (2) with **no** ``STAGE``
    pin the budget-aware ``120_stage`` filter prunes every over-budget staging subset
    to the empty one (``FM=26`` slabs blow the smem cap), so greedy's option-0 stages
    nothing and the operands lower gmem-direct."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.ir.cuda.ir import CudaOp

    g = _build_f16_matmul_graph(512, 512, 512)
    # Pin the WARP-tier geometry via a warp ``TILE`` codec (the over-ceiling ``f26x4`` register
    # tile + atom-K chunk) and leave STAGE unpinned — an explicit STAGE pin is authoritative (no
    # budget filter), but here we want the budget-aware filter to decline the over-budget staging so
    # the operands fall to the gmem-direct path. The bare atom token forces the warp (mma) tier.
    pin_classic(monkeypatch, {"TILE": "mma_m16n8k16_f16_f32/f26x4/k2"})
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    compiled = CudaBackend().compile(g)  # no longer raises
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if isinstance(n.op, CudaOp))
    assert "emmy_mma_load_a_gmem" in src and "emmy_mma_load_b_gmem" in src, "unstaged operands not loaded gmem-direct"
    assert "mma.sync.aligned.m16n8k16" in src, "tensor-core path not taken (scalar fallback?)"


# 2048×2048 fp32 matmul — the hero shape of the matmul-optimization blog posts ("Modern GPU Matmul
# Optimization" and "Surfacing a 60% performance bug in cuBLAS"), whose ``TM=26`` scalar register
# tile reaches ~106 % of cuBLAS. The tile geometries themselves are corpus cases now; the shape
# stays because the unroll contract below is stated at it.
_ARTICLE_DIMS = {"M": 2048, "K": 2048, "N": 2048}


def _build_2d_matmul_graph(dims: dict):
    """2D matmul ``a (M, K) @ b (K, N)`` — the canonical SGEMM shape
    the article kernel targets, no leading batch dim."""
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.frontend.ir import MatmulOp

    M, K, N = dims["M"], dims["K"], dims["N"]
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (M, K)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (M, N)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g


# The ninth-golden-sweep cp.async alignment regression: a (M,K)@(K,N) matmul whose B rows are NOT
# 16 B-aligned (N=3 fp32 → 12 B stride) pinned to a cp.async staging ring issued vectorized
# ``cp.async`` copies at misaligned global addresses — ``CUDA_ERROR_MISALIGNED_ADDRESS`` + a 1 s
# watchdog hang at runtime. The scalar stage resolver's 16 B inner-stride gate (previously
# TMA-only) now covers cp.async too: an invalid stage pin refuses instead of silently lowering a
# different gmem-direct schedule.
_ODD_STRIDE_CPASYNC_KNOBS = {"TILE": "f2x4", "WORK": "t16x8", "STAGE": "d2/smem-async"}


@requires_cuda
def test_scalar_cpasync_pin_refuses_odd_stride(monkeypatch):
    """fp32 matmul with a 12 B B-row stride pinned to a cp.async ring — the alignment gate must
    refuse instead of issuing misaligned ``cp.async`` copies or selecting gmem-direct."""
    from emmy.compiler.dtype import F32
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.frontend.ir import MatmulOp

    M, K, N = 4, 8, 3
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (M, K), dtype=F32), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N), dtype=F32), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (M, N), dtype=F32), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    rng = np.random.default_rng(0)
    inputs = {"a": rng.standard_normal((M, K), dtype=np.float32), "b": rng.standard_normal((K, N), dtype=np.float32)}
    with pytest.raises(ValueError, match="does not resolve"):
        _run_with_knobs(g, inputs, "c", _ODD_STRIDE_CPASYNC_KNOBS, monkeypatch)
