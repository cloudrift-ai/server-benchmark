"""The fused-edge e2e coverage — a producer fused into its matmul consumer, ONE kernel.

Black-box (the rebuild's recovery contract): build the graph, compile through ``CudaBackend``,
assert the kernel count + accuracy vs a numpy reference. One combinatorial test covers the MAP
producer × tier matrix (``f(x, …) @ w`` — the demoted-cone shapes a real model emits before a
linear); the MONOID producer (RMSNorm → Linear) is the one special case (its staged-shared-row
structural pin lives with ``test_fused_prologue_compiles_in_budget``).

The **warp tier** engages under a warp ``TILE`` pin: the demoted cone nodifies to a computed-A
``Contraction`` (the demoted warp option) and the producer COMPUTE-FILLS the A slab the
``ldmatrix`` drain reads (the mma tier's ``sync`` transport). The MONOID (rmsnorm) cone is
recognize-nodified (``010_recognize``'s ``bind_prologue_contraction`` merge): its statistic reduce
rides the A cone as a per-row prologue (``sync_stat_fill``), the warp rows are real fork siblings
of the coop ``Map`` form (exercised unpinned below), and a masked / symbolic M clamp-reads.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler import dtype as _dt
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp, MatmulOp, ReshapeOp, RmsNormOp, SdpaOp, TransposeOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from tests.compiler.conftest import requires_cuda, requires_sm90

F16 = _dt.get("f16")
_M, _K, _N = 32, 64, 32  # M != K so the row / col broadcasts are unambiguous

_WARP_TILE = "a:mma_m16n8k16_f16_f32/w1x1/f2x2/k2"  # tile 32x16, bk 32 — exact cover of the 32x64x32 shape


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _producer_graph(kind: str) -> tuple[Graph, tuple[str, ...]]:
    """The producer subgraph writing the ``xn`` intermediate — its op(s) + extra inputs."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (_M, _K), F16), node_id="x")
    if kind == "relu":
        g.add_node(ElementwiseOp("relu"), ["x"], Tensor("xn", (_M, _K), F16), node_id="xn")
        return g, ("x",)
    if kind == "sigmoid":
        g.add_node(ElementwiseOp("sigmoid"), ["x"], Tensor("xn", (_M, _K), F16), node_id="xn")
        return g, ("x",)
    if kind == "multiply":
        g.add_node(InputOp(), [], Tensor("y", (_M, _K), F16), node_id="y")
        g.add_node(ElementwiseOp("multiply"), ["x", "y"], Tensor("xn", (_M, _K), F16), node_id="xn")
        return g, ("x", "y")
    if kind == "broadcast":
        # ``xn[m,k] = x[m,k] · rs[m] · cs[k]`` — a row and a col broadcast, the rmsnorm
        # scale-application shape.
        g.add_node(InputOp(), [], Tensor("rs", (_M, 1), F16), node_id="rs")
        g.add_node(InputOp(), [], Tensor("cs", (1, _K), F16), node_id="cs")
        g.add_node(ElementwiseOp("multiply"), ["x", "rs"], Tensor("t", (_M, _K), F16), node_id="t")
        g.add_node(ElementwiseOp("multiply"), ["t", "cs"], Tensor("xn", (_M, _K), F16), node_id="xn")
        return g, ("x", "rs", "cs")
    raise ValueError(kind)


_PRODUCER_REFS = {
    "relu": lambda i: np.maximum(i["x"], 0),
    "sigmoid": lambda i: _sigmoid(i["x"]),
    "multiply": lambda i: i["x"] * i["y"],
    "broadcast": lambda i: i["x"] * i["rs"] * i["cs"],
}


def _compile_run(g: Graph, ins: dict) -> tuple[np.ndarray, list[str]]:
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    be = CudaBackend()
    compiled = be.compile(g)
    srcs = [n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None)]
    got = list(be.run(compiled, input_data=ins)[0].outputs.values())[0]
    return np.asarray(got), srcs


@requires_cuda
@pytest.mark.parametrize("producer", list(_PRODUCER_REFS))
@pytest.mark.parametrize("tier", ["scalar", "warp"])
def test_fused_map_matmul(tier, producer, monkeypatch):
    """``f(x, …) @ w`` computes in **one** kernel matching numpy — the MAP producer fused into the
    matmul, no gmem round-trip for the ``xn`` intermediate. Covers unary (relu / sigmoid),
    multi-input (multiply), and broadcast-operand (``x·rs[m]·cs[k]``) producers on the scalar tier;
    the ``warp`` cells additionally demand the ``mma.sync`` tier (the compute-filled A slab;
    the broadcast cell is xfailed — its producer recognizes as a flat un-annotated ``Map``)."""
    if tier == "warp":
        monkeypatch.setenv("EMMY_TILE", _WARP_TILE)
    g, extra = _producer_graph(producer)
    g.add_node(InputOp(), [], Tensor("w", (_K, _N), F16), node_id="w")
    g.add_node(MatmulOp(), ["xn", "w"], Tensor("o", (_M, _N), F16), node_id="o")
    g.inputs, g.outputs = [*extra, "w"], ["o"]

    rng = np.random.default_rng(0)
    ins = {nid: (rng.standard_normal(tuple(d.as_static() for d in g.nodes[nid].output.shape)) * 0.3).astype(np.float16) for nid in g.inputs}
    got, srcs = _compile_run(g, ins)
    # The invariant is that the producer cone is FUSED — no kernel signature carries a
    # materialized intermediate: neither the graph's own ``xn`` buffer nor a cut workspace
    # (``020_cut_edge``'s ``<out>__cone`` / ``<out>__stat`` — a cut realization materializes the
    # cone under those names, so a name-anchored ``xn`` check alone would miss it). Kernel COUNT
    # is not the invariant: a stat-free cone binds as a computed-A ``Contraction``, which
    # legitimately offers split-K REDUCE rows, and a chosen one emits a ``__partial`` + finalize
    # pair that still carries the cone inline.
    sigs = [s.split("{")[0] for s in srcs]
    leaked = [m for m in ("xn", "__cone", "__stat") if any(m in sig for sig in sigs)]
    assert not leaked, f"{producer}/{tier}: the cone round-tripped through gmem ({leaked}): {sigs}"
    if tier == "warp":
        assert any("emmy_mma" in s for s in srcs), f"{producer}/warp: the pinned warp tier must engage on the fused matmul"
    f32 = {k: v.astype(np.float32) for k, v in ins.items()}
    ref = _PRODUCER_REFS[producer](f32) @ f32["w"]
    np.testing.assert_allclose(got.reshape(_M, _N).astype(np.float32), ref, atol=0.1, rtol=2e-2)


@requires_cuda
@pytest.mark.parametrize("tier", ["scalar", "warp"])
def test_fused_rmsnorm_linear(tier, monkeypatch):
    """The **MONOID** producer fuses: ``rmsnorm(x)·nw @ wg`` computes in one kernel matching a
    numpy reference (whether the linear's N axis rides the grid or a tail sweep is the schedule's
    choice — the staged-shared-row structural pin lives with `test_fused_prologue_compiles_in_budget`,
    whose shape keeps the sweep in-tail). The ``warp`` cell demands the mma tier on the fused
    matmul — the recognize-nodified computed-A ``Contraction`` (the statistic prologue rides the
    A cone, run once per tile row by the sync stat fill)."""
    if tier == "warp":
        monkeypatch.setenv("EMMY_TILE", _WARP_TILE)
    # Pin the serial fold in BOTH cells: the computed-A form offers redundant-statistic split-K
    # siblings on this decode-M shape, and this test pins the ONE-kernel fused form. The scalar
    # cell is an unpinned greedy compile, so without the pin the device's evidence may steer it
    # to a (legal) 2-kernel split row (seen on sm_89). The split form has its own test
    # (test_fused_cone_splitk_matches_reference); the free pick, test_fused_rmsnorm_linear_unpinned.
    monkeypatch.setenv("EMMY_REDUCE", "")
    S, H, inter = 32, 1024, 3072
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, S, H), F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (H,), F16), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (inter, H), F16), node_id="wg")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, S, H), F16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("o", (1, S, inter), F16), node_id="o")
    g.inputs, g.outputs = ["x", "nw", "wg"], ["o"]

    rng = np.random.default_rng(0)
    ins = {
        "x": (rng.standard_normal((1, S, H)) * 0.3).astype(np.float16),
        "nw": (rng.standard_normal((H,)) * 0.3).astype(np.float16),
        "wg": (rng.standard_normal((inter, H)) * 0.1).astype(np.float16),
    }
    got, srcs = _compile_run(g, ins)
    assert len(srcs) == 1, f"the fused norm→linear must be ONE kernel, got {len(srcs)}"
    if tier == "warp":
        assert "emmy_mma" in srcs[0], "warp: the pinned mma tier must engage on the fused matmul"
    x, nw, wg = (ins[k].astype(np.float32) for k in ("x", "nw", "wg"))
    rms = x[0] * (1.0 / np.sqrt((x[0] ** 2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
    # The fp16 fused-prologue path carries a few % relative error on large elements (cooperative
    # rms-scale reduce + fp16 matmul accumulate at K=1024) — rtol=0.1 catches a regression without
    # flaking; atol absorbs near-zero elements where relative error is meaningless.
    np.testing.assert_allclose(got.reshape(S, inter).astype(np.float32), rms @ wg.T, atol=0.5, rtol=0.1)


def _rmsnorm_linear_graph(S, H: int, inter: int) -> Graph:
    """``rmsnorm(x)·nw @ wg`` — ``S`` is an int (static) or a ``Dim`` (symbolic seq)."""
    Sd = S if isinstance(S, Dim) else S
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, Sd, H), F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (H,), F16), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (inter, H), F16), node_id="wg")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, Sd, H), F16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("o", (1, Sd, inter), F16), node_id="o")
    g.inputs, g.outputs = ["x", "nw", "wg"], ["o"]
    return g


def _rmsnorm_linear_check(g: Graph, S: int, H: int, inter: int, *, want_mma: bool, kernels: tuple = (1,)) -> None:
    rng = np.random.default_rng(0)
    ins = {
        "x": (rng.standard_normal((1, S, H)) * 0.3).astype(np.float16),
        "nw": (rng.standard_normal((H,)) * 0.3).astype(np.float16),
        "wg": (rng.standard_normal((inter, H)) * 0.1).astype(np.float16),
    }
    got, srcs = _compile_run(g, ins)
    assert len(srcs) in kernels, f"the fused norm→linear must be {kernels} kernel(s), got {len(srcs)}"
    if want_mma:
        assert any("emmy_mma" in s for s in srcs), "the pinned mma tier must engage on the fused matmul"
    x, nw, wg = (ins[k].astype(np.float32) for k in ("x", "nw", "wg"))
    rms = x[0] * (1.0 / np.sqrt((x[0] ** 2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
    np.testing.assert_allclose(got.reshape(S, inter).astype(np.float32), rms @ wg.T, atol=0.5, rtol=0.1)


@requires_cuda
@pytest.mark.parametrize(
    "tile",
    [
        "a:mma_m16n8k16_f16_f32/w2x4/f2x4/k4",  # bk 64 elems + tile_n 128 → both slabs B128
        "a:mma_m16n8k16_f16_f32/w2x2/f2x2/k2",  # bk 32 elems + tile_n 32 → both slabs B64
    ],
)
def test_fused_sync_fill_slab_swizzle(tile, monkeypatch):
    """The sync compute-fill's slab swizzle round-trips — fill-side ``Write`` XOR = drain-side
    ``ldmatrix`` XOR, at both derived modes (B64 / B128). Regression for the silent fill-side
    drop (a pass reconstructing ``Write`` without ``swizzle`` left the fill row-major under a
    swizzled drain — a scrambled A tile whose matmul output still PASSED the runner's loose
    accuracy tolerance, caught only by this rtol). The structural assert pins that the fill
    actually swizzles — a future silent drop fails loudly here instead of shipping garbage.
    The B128 cell also pins ``RASTER`` on the fused-cone grid (a computed-A kernel's output is
    the same static 2-D block-tile grid, and the grouped launch order is its B-re-streaming
    fix): the ``_rsub`` grouped decode must be emitted, not silently degraded to flat."""
    monkeypatch.setenv("EMMY_TILE", tile)
    monkeypatch.setenv("EMMY_REDUCE", "")  # serial fold — the swizzle inspection needs the ONE fused kernel
    if tile.endswith("k4"):
        monkeypatch.setenv("EMMY_RASTER", "gn8")
    S, H, inter = 64, 1024, 3072
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, S, H), F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (H,), F16), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (inter, H), F16), node_id="wg")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, S, H), F16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("o", (1, S, inter), F16), node_id="o")
    g.inputs, g.outputs = ["x", "nw", "wg"], ["o"]
    rng = np.random.default_rng(0)
    ins = {
        "x": (rng.standard_normal((1, S, H)) * 0.3).astype(np.float16),
        "nw": (rng.standard_normal((H,)) * 0.3).astype(np.float16),
        "wg": (rng.standard_normal((inter, H)) * 0.1).astype(np.float16),
    }
    got, srcs = _compile_run(g, ins)
    assert len(srcs) == 1 and "emmy_mma" in srcs[0]
    assert "emmy_swizzle" in srcs[0], "the sync compute-fill must swizzle its slabs (fill-side Write XOR)"
    if tile.endswith("k4"):
        assert "_rsub" in srcs[0], "RASTER must resolve on the fused-cone grid (grouped CTA decode emitted)"
    x, nw, wg = (ins[k].astype(np.float32) for k in ("x", "nw", "wg"))
    rms = x[0] * (1.0 / np.sqrt((x[0] ** 2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
    np.testing.assert_allclose(got.reshape(S, inter).astype(np.float32), rms @ wg.T, atol=0.5, rtol=0.1)


@requires_cuda
def test_place_cone_cut_splits_the_kernels(monkeypatch):
    """``PLACE@cone=cut`` (here via the bare ``EMMY_PLACE`` env pin) keeps the norm producer OUT
    of the matmul kernel — the ``xn`` intermediate stays a MEMORY edge and the program lowers to
    MULTIPLE kernels (the fusion-pass guard; the tile recognizer's gate alone could only demote
    an already-fused kernel to its coop form, never split it — the pin used to be a silent
    no-op). The matmul side then owns a plain gmem A, so the staged cp.async/TMA tiers apply —
    measured 496-503 µs vs the exhaustively-optimized fused kernel's ~660 on the gemma gate_up
    shape (5090). Numerics must match the same numpy reference as the fused form."""
    monkeypatch.setenv("EMMY_PLACE", "cut")
    S, H, inter = 32, 1024, 3072
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, S, H), F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (H,), F16), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (inter, H), F16), node_id="wg")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, S, H), F16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("o", (1, S, inter), F16), node_id="o")
    g.inputs, g.outputs = ["x", "nw", "wg"], ["o"]
    rng = np.random.default_rng(0)
    ins = {
        "x": (rng.standard_normal((1, S, H)) * 0.3).astype(np.float16),
        "nw": (rng.standard_normal((H,)) * 0.3).astype(np.float16),
        "wg": (rng.standard_normal((inter, H)) * 0.1).astype(np.float16),
    }
    got, srcs = _compile_run(g, ins)
    assert len(srcs) >= 2, f"PLACE@cone=cut must split the norm from the matmul, got {len(srcs)} kernel(s)"
    x, nw, wg = (ins[k].astype(np.float32) for k in ("x", "nw", "wg"))
    rms = x[0] * (1.0 / np.sqrt((x[0] ** 2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
    np.testing.assert_allclose(got.reshape(S, inter).astype(np.float32), rms @ wg.T, atol=0.5, rtol=0.1)


@requires_cuda
def test_place_cone_cut_degenerate_m1(monkeypatch):
    """``PLACE@cone=cut`` on the DEGENERATE M=1 composition — per-token decode's norm→matvec.
    At S=1 the reshape-folded lift leaves the contraction with ZERO free axes; the recognizer
    synthesizes a unit ``_um`` axis so the composition still binds (instead of a grid-1 fused
    schedule ~300× off the memory floor) and the cut guard accepts ``free ≤ 1``. The cut's
    consumer half then re-lowers through ``rename_ssa_sequential`` — the shape that miscompiled
    (``acc1``-undefined) when ``Loop.rewrite`` carried its ``Carrier`` verbatim past the body's
    Accum renumber. Compiling, splitting, and matching numpy is the whole regression."""
    monkeypatch.setenv("EMMY_PLACE", "cut")
    S, H, inter = 1, 1024, 3072
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, S, H), F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (H,), F16), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (inter, H), F16), node_id="wg")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, S, H), F16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("o", (1, S, inter), F16), node_id="o")
    g.inputs, g.outputs = ["x", "nw", "wg"], ["o"]
    rng = np.random.default_rng(0)
    ins = {
        "x": (rng.standard_normal((1, S, H)) * 0.3).astype(np.float16),
        "nw": (rng.standard_normal((H,)) * 0.3).astype(np.float16),
        "wg": (rng.standard_normal((inter, H)) * 0.1).astype(np.float16),
    }
    got, srcs = _compile_run(g, ins)
    assert len(srcs) >= 2, f"PLACE@cone=cut must split the M=1 norm from the matvec, got {len(srcs)} kernel(s)"
    x, nw, wg = (ins[k].astype(np.float32) for k in ("x", "nw", "wg"))
    rms = x[0] * (1.0 / np.sqrt((x[0] ** 2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
    np.testing.assert_allclose(got.reshape(S, inter).astype(np.float32), rms @ wg.T, atol=0.5, rtol=0.1)


@requires_cuda
@requires_sm90
@pytest.mark.parametrize("runtime_s", [31, 130])
def test_fused_rmsnorm_linear_symbolic_m(runtime_s, monkeypatch):
    """The MONOID warp fused edge over a **symbolic seq axis** — ONE masked-M mma kernel deployed
    at the hint and run at off-hint sizes straddling the 64-row tile (31 under, 130 over + tail):
    the sync compute-fill / stat-prologue σ clamp the overhanging rows and the ``RegStore`` guard
    discards their store."""
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f32/f2x2/k2")  # tile 64×32 — every runtime S is masked
    monkeypatch.setenv("EMMY_WORK", "w2x2")
    monkeypatch.setenv("EMMY_REDUCE", "")  # serial fold — the ONE-masked-kernel contract is what's under test
    H, inter = 256, 512
    g = _rmsnorm_linear_graph(Dim("seq_len", hint=64), H, inter)
    _rmsnorm_linear_check(g, runtime_s, H, inter, want_mma=True)


@requires_cuda
def test_fused_rmsnorm_linear_unpinned():
    """UNPINNED greedy on the fused norm→linear: the merged fork (coop ``Map`` rows + the
    computed-A Contraction's warp rows) lowers and matches numpy whichever row the prior picks —
    the fork-integrity e2e (runs on any CUDA device; option-0 stays the coop row). The pick may
    be the 1-kernel fused row or a 2-kernel redundant-statistic split row; both are legal fork
    members on this decode-M shape."""
    S, H, inter = 32, 256, 512
    _rmsnorm_linear_check(_rmsnorm_linear_graph(S, H, inter), S, H, inter, want_mma=False, kernels=(1, 2))


@requires_cuda
@requires_sm90
@pytest.mark.parametrize("runtime_s", [31, 130])
def test_fused_gate_up_swiglu_symbolic_m(runtime_s, monkeypatch):
    """The MULTI-FOLD (product-monoid) fused edge — ``swiglu(rmsnorm(x)·nw @ Wg, rmsnorm(x)·nw @
    Wu)`` in ONE mma kernel: the two folds share the compute-filled A slab (one ldmatrix'd A
    fragment feeding per-fold B slabs / C fragments) and the SwiGLU combine rides the store's
    fragment epilogue; the seq axis is symbolic with a masked M tail (31 under / 130 over the
    64-row tile)."""
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f32/f2x2/k2")
    monkeypatch.setenv("EMMY_WORK", "w2x2")
    monkeypatch.setenv("EMMY_REDUCE", "")  # serial fold — ONE masked kernel is the contract; the split form has its own test
    S, H, inter = runtime_s, 256, 512
    Sd = Dim("seq_len", hint=64)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, Sd, H), F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (H,), F16), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (inter, H), F16), node_id="wg")
    g.add_node(InputOp(), [], Tensor("wu", (inter, H), F16), node_id="wu")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, Sd, H), F16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("gate", (1, Sd, inter), F16), node_id="gate")
    g.add_node(LinearOp(), ["xn", "wu"], Tensor("up", (1, Sd, inter), F16), node_id="up")
    g.add_node(ElementwiseOp("silu"), ["gate"], Tensor("sg", (1, Sd, inter), F16), node_id="sg")
    g.add_node(ElementwiseOp("multiply"), ["sg", "up"], Tensor("o", (1, Sd, inter), F16), node_id="o")
    g.inputs, g.outputs = ["x", "nw", "wg", "wu"], ["o"]

    rng = np.random.default_rng(0)
    ins = {
        "x": (rng.standard_normal((1, S, H)) * 0.3).astype(np.float16),
        "nw": (rng.standard_normal((H,)) * 0.3).astype(np.float16),
        "wg": (rng.standard_normal((inter, H)) * 0.1).astype(np.float16),
        "wu": (rng.standard_normal((inter, H)) * 0.1).astype(np.float16),
    }
    got, srcs = _compile_run(g, ins)
    assert len(srcs) == 1, f"the fused gate/up edge must be ONE kernel, got {len(srcs)}"
    assert "emmy_mma" in srcs[0], "the pinned mma tier must engage on the multi-fold contraction"
    x, nw, wg, wu = (ins[k].astype(np.float32) for k in ("x", "nw", "wg", "wu"))
    rms = x[0] * (1.0 / np.sqrt((x[0] ** 2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
    gate, up = rms @ wg.T, rms @ wu.T
    ref = gate * _sigmoid(gate) * up
    np.testing.assert_allclose(got.reshape(S, inter).astype(np.float32), ref, atol=0.5, rtol=0.1)


@requires_cuda
@pytest.mark.parametrize("tier", ["scalar", "warp"])
def test_mixed_dtype_matmul_demotes_a_to_mma(tier, monkeypatch):
    """An **f32-A × f16-B** matmul — the erased-downcast signature (torch cannot execute a mixed
    matmul, so the model itself rounded A; the tracer maps ``to``/``type_as`` to pass-throughs,
    e.g. Gemma's ``self._norm(x.float()).type_as(x)`` feeding every f16-weight projection) —
    reaches the mma tier through the demoting cone wrap: the sync
    compute-fill converts the f32 value on the slab store, since the copy transports move raw
    bytes and cannot. The scalar cell pins nothing and just checks the mixed kernel still
    compiles and matches; both compare against the f16-rounded-A reference."""
    if tier == "warp":
        monkeypatch.setenv("EMMY_TILE", _WARP_TILE)
    F32 = _dt.get("f32")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (_M, _K), F32), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (_K, _N), F16), node_id="w")
    g.add_node(MatmulOp(), ["x", "w"], Tensor("o", (_M, _N), F16), node_id="o")
    g.inputs, g.outputs = ["x", "w"], ["o"]

    rng = np.random.default_rng(0)
    ins = {
        "x": (rng.standard_normal((_M, _K)) * 0.3).astype(np.float32),
        "w": (rng.standard_normal((_K, _N)) * 0.3).astype(np.float16),
    }
    got, srcs = _compile_run(g, ins)
    if tier == "warp":
        assert any("emmy_mma" in s for s in srcs), "the mixed-dtype matmul must engage the mma tier via the demoting sync fill"
    ref = ins["x"].astype(np.float16).astype(np.float32) @ ins["w"].astype(np.float32)
    np.testing.assert_allclose(got.reshape(_M, _N).astype(np.float32), ref, atol=0.1, rtol=2e-2)


@requires_cuda
def test_sdpa_consumer_projection_reaches_mma(monkeypatch):
    """The attention output projection — ``linear(reshape(transpose(sdpa(q, k, v))))`` over a
    **symbolic seq axis, causal**: gemma's o_proj composition — must reach the mma tier under a
    warp ``TILE`` pin. The flash rewrite splices a REBUILT projection op into the graph mid-batch;
    a later match applying against the swapped op used to read ``_seed_io_placeholders``'
    ``(f32, ())`` stubs (``Match.is_alive`` snapshots node identity and cannot see an op swap), so
    the warp atom gate refused the all-f16 contraction and the deploy fell to a scalar tile — 16x its
    own measured mma rows on the gemma-4-12B layer. ``Candidate.try_rewrite``'s apply-time
    ``populate_io`` refresh restores the graph-true dtypes; this pins it."""
    monkeypatch.setenv("EMMY_TILE", _WARP_TILE)
    B, H, S, D = 1, 4, 64, 64
    NO = 96  # projection out-features
    Sd = Dim("seq_len", hint=S)
    g = Graph()
    for name in ("q", "k", "v"):
        g.add_node(InputOp(), [], Tensor(name, (B, H, Sd, D), F16), node_id=name)
    g.add_node(InputOp(), [], Tensor("wo", (NO, H * D), F16), node_id="wo")
    g.add_node(SdpaOp(is_causal=True), ["q", "k", "v"], Tensor("att", (B, H, Sd, D), F16), node_id="att")
    g.add_node(TransposeOp(axes=(1, 2)), ["att"], Tensor("attt", (B, Sd, H, D), F16), node_id="attt")
    g.add_node(ReshapeOp(shape=(B, Sd, H * D)), ["attt"], Tensor("attr", (B, Sd, H * D), F16), node_id="attr")
    g.add_node(LinearOp(has_bias=False), ["attr", "wo"], Tensor("o", (B, Sd, NO), F16), node_id="o")
    g.inputs, g.outputs = ["q", "k", "v", "wo"], ["o"]

    rng = np.random.default_rng(0)
    ins = {n: (rng.standard_normal((B, H, S, D)) * 0.3).astype(np.float16) for n in ("q", "k", "v")}
    ins["wo"] = (rng.standard_normal((NO, H * D)) * 0.1).astype(np.float16)
    got, srcs = _compile_run(g, ins)
    proj = [s for s in srcs if "wo" in s]
    assert proj, "no kernel references the projection weight"
    assert any("emmy_mma" in s for s in proj), "the sdpa-consumer projection must honor the warp TILE pin (the placeholder-I/O offer gap)"
    q, k, v, wo = (ins[n].astype(np.float32) for n in ("q", "k", "v", "wo"))
    scores = (q[0] @ np.swapaxes(k[0], -1, -2)) / np.sqrt(D)
    scores += np.triu(np.full((S, S), -np.inf, dtype=np.float32), k=1)  # causal mask
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    att = p @ v[0]  # (H, S, D)
    ref = np.transpose(att, (1, 0, 2)).reshape(S, H * D) @ wo.T
    np.testing.assert_allclose(got.reshape(S, NO).astype(np.float32), ref, atol=0.5, rtol=0.1)


@requires_cuda
@pytest.mark.parametrize("stage", ["d1/sync", "d2/sync"])
def test_fused_cone_splitk_matches_reference(stage, monkeypatch):
    """Redundant-statistic split-K on a computed-A (norm→linear) cone: the contraction K is
    sliced across CTAs (``REDUCE=g4k``) while the k-invariant stat prologue stays FULL-ROW in
    every partition (each recomputes it), and the deferred finalize sums the partials before
    the projection — the output must match the fp32 reference. Parametrized over both sync
    depths (``d1`` + the asymmetric B-only prefetch ring ``d2``). The decode-M shape class
    (M=32) is where this split pays: the un-split cone grid starves the SMs."""
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f32/f2x2/k2")
    monkeypatch.setenv("EMMY_WORK", "w1x4")
    monkeypatch.setenv("EMMY_REDUCE", "g4k")
    monkeypatch.setenv("EMMY_STAGE", stage)
    S, H, inter = 32, 1024, 3072
    # CANONICAL-B matmul (w shaped (H, inter)) rather than the transposed-B linear: a
    # transposed B clamps the sync ring back to d1 (nothing async to overlap), so the d2
    # cell would silently re-test d1. The real decode twins bake weights as constants
    # (the transpose folds), so canonical-B is the shape class the ring actually serves.
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, S, H), _dt.F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (H,), _dt.F16), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (H, inter), _dt.F16), node_id="wg")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, S, H), _dt.F16), node_id="xn")
    g.add_node(MatmulOp(), ["xn", "wg"], Tensor("o", (1, S, inter), _dt.F16), node_id="o")
    g.inputs, g.outputs = ["x", "nw", "wg"], ["o"]
    rng = np.random.default_rng(0)
    ins = {
        "x": (rng.standard_normal((1, S, H)) * 0.3).astype(np.float16),
        "nw": (rng.standard_normal((H,)) * 0.3).astype(np.float16),
        "wg": (rng.standard_normal((H, inter)) * 0.1).astype(np.float16),
    }
    got, srcs = _compile_run(g, ins)
    assert len(srcs) == 2, f"split-K must produce partial + finalize, got {len(srcs)} kernel(s)"
    assert any("emmy_mma" in s for s in srcs), "the mma tier must engage on the split partial"
    if stage == "d2/sync":
        assert any("cp.async" in s for s in srcs), "the d2 B-only prefetch ring must issue cp.async on the canonical-B slab"
    x, nw, wg = (ins[k].astype(np.float32) for k in ("x", "nw", "wg"))
    rms = x[0] * (1.0 / np.sqrt((x[0] ** 2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
    np.testing.assert_allclose(got.reshape(S, inter).astype(np.float32), rms @ wg, atol=0.5, rtol=0.1)


@requires_cuda
def test_fused_gate_up_splitk_matches_reference(monkeypatch):
    """Redundant-statistic split-K on the MULTI-CHANNEL (gate/up) fused cone: both channels'
    additive states slice over K across CTAs (the stat prologue full-row per partition), the
    deferred finalize folds the 2-component carrier and applies the SwiGLU ⊗-combine ONCE after
    the cross-partition sums — the output must match the fp32 reference. The decode-M shape
    class (M=32) is where the multi-channel split pays (the gemma gate⊗up twin)."""
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f32/f2x2/k2")
    monkeypatch.setenv("EMMY_WORK", "w1x4")
    monkeypatch.setenv("EMMY_REDUCE", "g4k")
    S, H, inter = 32, 256, 512
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, S, H), F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (H,), F16), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (inter, H), F16), node_id="wg")
    g.add_node(InputOp(), [], Tensor("wu", (inter, H), F16), node_id="wu")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, S, H), F16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("gate", (1, S, inter), F16), node_id="gate")
    g.add_node(LinearOp(), ["xn", "wu"], Tensor("up", (1, S, inter), F16), node_id="up")
    g.add_node(ElementwiseOp("silu"), ["gate"], Tensor("sg", (1, S, inter), F16), node_id="sg")
    g.add_node(ElementwiseOp("multiply"), ["sg", "up"], Tensor("o", (1, S, inter), F16), node_id="o")
    g.inputs, g.outputs = ["x", "nw", "wg", "wu"], ["o"]

    rng = np.random.default_rng(0)
    ins = {
        "x": (rng.standard_normal((1, S, H)) * 0.3).astype(np.float16),
        "nw": (rng.standard_normal((H,)) * 0.3).astype(np.float16),
        "wg": (rng.standard_normal((inter, H)) * 0.1).astype(np.float16),
        "wu": (rng.standard_normal((inter, H)) * 0.1).astype(np.float16),
    }
    got, srcs = _compile_run(g, ins)
    assert len(srcs) == 2, f"split-K must produce partial + finalize, got {len(srcs)} kernel(s)"
    assert any("emmy_mma" in s for s in srcs), "the mma tier must engage on the split partial"
    x, nw, wg, wu = (ins[k].astype(np.float32) for k in ("x", "nw", "wg", "wu"))
    rms = x[0] * (1.0 / np.sqrt((x[0] ** 2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
    gate, up = rms @ wg.T, rms @ wu.T
    ref = gate / (1.0 + np.exp(-gate)) * up
    np.testing.assert_allclose(got.reshape(S, inter).astype(np.float32), ref, atol=0.5, rtol=0.1)
