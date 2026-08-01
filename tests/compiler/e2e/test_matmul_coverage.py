"""Matmul (SEMIRING) coverage — the contraction across every tier, one file.

The scalar contraction's free-axis output tile (the ``TILE`` codec) and the tensor-core warp
fragment (the ``WARP`` codec) are the two materializers for the same SEMIRING ``project ∘ contract``
algebra; this file pins each and checks accuracy vs numpy/torch AND the matching lowering
structure, over static AND symbolic (dynamic-grid) shapes. Sections:

- **scalar TILE tier** — register-tile variants, fused epilogues, operand staging (the ``STAGE``
  codec), and the FN>1 blocked-GEMM / fused-prologue regression reproducers.
- **warp MMA tier** — ``mma.sync`` plain / transposed-B / epilogue coverage, static-vs-dynamic
  parity across cp.async + TMA transports, the staging invariants (bit-identical, bf16), and
  atomic-free split-K.
- **masked symbolic warp tier** — off-hint straddling sizes for symbolic M / N / K (the
  boundary-guard + clamp + zero-fill interplay the tile-divisor sweeps can't reach), the demoted
  B-cone, the batched / softmax-P@V split-consumers, cp.async AND TMA.

Pure GPU accuracy (no ``-O1`` numerics change), so it runs in the correctness lane. The CPU-render
structure tests (forced sm_120) need no GPU; warp-tier accuracy needs sm_90+.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import BF16, F16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import CUDA_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.pipeline.knob import family_value
from emmy.compiler.pipeline.search.features import mma_atom

from ..conftest import dyn_M, requires_cuda, requires_sm90


def _has_cuda() -> bool:
    try:
        import cupy as cp  # noqa: PLC0415

        return cp.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _supports_tma() -> bool:
    """TMA (``cp.async.bulk.tensor``) needs sm_90+ (Hopper / Blackwell)."""
    if not _has_cuda():
        return False
    import cupy as cp  # noqa: PLC0415

    return int(cp.cuda.Device().compute_capability) >= 90


def _dtype(name: str):
    return {"f16": F16, "f32": F32}[name]


# =========================================================================== #
# Scalar TILE tier — register-tile variants, epilogues, staging, regressions.
# =========================================================================== #

# Square base shape; divisible by every variant's parallel·register product so the static
# column is exact-cover and the dynamic column runs at an off-divisor length (masked tail).
_M = _K = _N = 64
_DYN_M = 70  # off the 64 base → a partial last register-row when M is register-tiled
_SHAPES = ("static", "dynamic")


def _matmul_graph(mode: str):
    """``(1, M, K) @ (K, N)``; ``mode='dynamic'`` makes the M (row) axis symbolic."""
    Mg = dyn_M(mode, _M)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (1, Mg, _K)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (_K, _N)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (1, Mg, _N)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g


def _pin_tile(monkeypatch, pin: tuple[str, str]) -> None:
    """Pin an output tile: the site-local ``TILE`` value plus the ``WORK`` inventory its unit
    widths live in (the two halves of one tile — a fixture spells them as one pair)."""
    value, work = pin
    monkeypatch.setenv("EMMY_TILE", value)
    monkeypatch.setenv("EMMY_WORK", work)


def _run(mode: str, pin: tuple[str, str], monkeypatch) -> tuple[np.ndarray, np.ndarray, str]:
    """Compile the matmul under the pinned ``EMMY_TILE`` codec, run on seeded inputs at the
    mode's runtime M, and return ``(output, reference, kernel_source)``."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    _pin_tile(monkeypatch, pin)
    m = _DYN_M if mode == "dynamic" else _M
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1, m, _K), dtype=np.float32)
    b = rng.standard_normal((_K, _N), dtype=np.float32)
    be = CudaBackend()
    compiled = be.compile(_matmul_graph(mode))
    got = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["c"])
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))
    return got, (a @ b), src


# (label, TILE codec, expects-register-replication, expected __launch_bounds__ or None).
#   none        ("")            — one thread per cell, no register replication / unroll
#   reg_inner   (f4)            — 4 register cells along N, B-load shared across them
#   reg_2d      (f2x2)         — full 2×2 register block, both operands reused
#   single_cta  (f2x4 + t32x16) — par·reg == 64×64 ⇒ one 512-thread CTA (static)
#   reg_f3      (f3)            — a stride-3 register row: the store vectorizer must refuse to pack
#                                  across the misaligned cell stride (the old FN=3 hang/fault shape)
_VARIANTS = {
    "none": (("", ""), False, None),
    "reg_inner": (("f4", ""), True, None),
    "reg_f3": (("f3", ""), True, None),
    "reg_2d": (("f2x2", ""), True, None),
    "single_cta": (("f2x4", "t32x16"), True, 512),
}


@pytest.mark.parametrize("variant", list(_VARIANTS))
@pytest.mark.parametrize("mode", _SHAPES)
@requires_cuda
def test_matmul_tile_coverage(variant, mode, monkeypatch):
    pin, has_reg, launch_bounds = _VARIANTS[variant]
    got, ref, src = _run(mode, pin, monkeypatch)

    diff = float(np.abs(got - ref.reshape(got.shape)).max())
    assert diff < 1e-3, f"{variant}/{mode}: matmul mismatch (max abs err {diff})"

    has_copy = "__c0_1" in src or "__c1_0" in src  # a replicated register-cell binding
    if has_reg:
        assert has_copy, f"{variant}/{mode}: expected replicated register cells (__c*)"
        assert "#pragma unroll" in src, f"{variant}/{mode}: the small inner reduce must be unrolled"
    else:
        assert not has_copy, f"{variant}/{mode}: per-cell tier must not replicate register cells"
    if launch_bounds is not None:
        assert f"__launch_bounds__({launch_bounds})" in src, f"{variant}/{mode}: expected a {launch_bounds}-thread CTA"

    if mode == "dynamic":
        # The dynamic-grid tier: the launch sizes from the runtime extent (the symbolic ``Dim``
        # threaded as an ``int`` arg), and a register-tiled symbolic axis guards its tail store.
        assert "int seq_len" in src, f"{variant}/dynamic: symbolic grid must carry the runtime extent arg"


# The contraction's reduce-axis partition (the ``REDUCE`` codec) composes onto the SEMIRING exactly
# like a plain monoid reduce — a contraction is a monoid with a ⊗ lift (``Semiring.as_monoid``), so
# its K axis folds cooperatively (``b`` BLOCK, warp-shuffle butterfly), across ILP register chains
# (``r`` REG), or both, through the same ``_reduce`` materializer. A non-``g`` ``REDUCE`` pin was
# silently ignored on a contraction before. (Output is per-cell here — composing the partition WITH
# an output TILE is the remaining step.)
_REDUCE_VARIANTS = {
    "coop": ("b4", "__shfl"),  # 4 threads cooperatively fold K
    "ilp": ("r4", None),  # 4 ILP register-accumulator chains, one thread
    "coop_ilp": ("r2/b4", "__shfl"),  # composed: 2 ILP chains × 4 coop threads
}


@pytest.mark.parametrize("variant", list(_REDUCE_VARIANTS))
@requires_cuda
def test_matmul_reduce_partition(variant, monkeypatch):
    """A non-output-tiled contraction honours a cooperative / ILP ``REDUCE`` pin (silently ignored
    before): the K axis partitions across threads / register chains and still matches numpy."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    spec, marker = _REDUCE_VARIANTS[variant]
    monkeypatch.delenv("EMMY_TILE", raising=False)
    monkeypatch.setenv("EMMY_REDUCE", spec)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1, _M, _K), dtype=np.float32)
    b = rng.standard_normal((_K, _N), dtype=np.float32)
    be = CudaBackend()
    compiled = be.compile(_matmul_graph("static"))
    got = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["c"])
    np.testing.assert_allclose(got.reshape(_M, _N), (a @ b)[0], atol=1e-3, rtol=1e-3)
    if marker is not None:
        src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))
        assert marker in src, f"{variant}: expected the cooperative combine ({marker}) in the lowered kernel"


# Fused epilogues — a projection ``Map`` over the ``Semiring`` (``project ∘ contract``): the
# pointwise op folds into the contraction kernel's tail, replicated per register cell. Each is a
# distinct tail shape: a broadcast scalar, a per-``n`` bias (shared across the ``m`` cells), a
# pure activation, and a full ``(m, n)`` residual (no sharing). Pinned to a 2×2 register tile.
_EPILOGUE_TILE = ("f2x2", "t16x16")
_EPILOGUES = ("scale", "bias", "relu", "residual")


def _epilogue_graph(mode: str, epilogue: str):
    """``(1, M, K) @ (K, N)`` with a fused pointwise ``epilogue`` on the contraction output."""
    Mg = dyn_M(mode, _M)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (1, Mg, _K)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (_K, _N)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("ab", (1, Mg, _N)), node_id="ab")
    inputs = ["a", "b"]
    if epilogue == "scale":
        g.add_node(InputOp(), [], Tensor("s", (1,)), node_id="s")
        g.add_node(ElementwiseOp("multiply"), ["ab", "s"], Tensor("o", (1, Mg, _N)), node_id="o")
        inputs.append("s")
    elif epilogue == "bias":
        g.add_node(InputOp(), [], Tensor("bias", (_N,)), node_id="bias")
        g.add_node(ElementwiseOp("add"), ["ab", "bias"], Tensor("o", (1, Mg, _N)), node_id="o")
        inputs.append("bias")
    elif epilogue == "relu":
        g.add_node(ElementwiseOp("relu"), ["ab"], Tensor("o", (1, Mg, _N)), node_id="o")
    else:  # residual — a full (1, M, N) add (depends on both cell axes, no load sharing)
        g.add_node(InputOp(), [], Tensor("r", (1, Mg, _N)), node_id="r")
        g.add_node(ElementwiseOp("add"), ["ab", "r"], Tensor("o", (1, Mg, _N)), node_id="o")
        inputs.append("r")
    g.inputs, g.outputs = inputs, ["o"]
    return g


def _epilogue_ref(epilogue: str, feed: dict) -> np.ndarray:
    base = feed["a"] @ feed["b"]
    if epilogue == "scale":
        return base * feed["s"]
    if epilogue == "bias":
        return base + feed["bias"]
    if epilogue == "relu":
        return np.maximum(base, 0.0)
    return base + feed["r"]


@pytest.mark.parametrize("epilogue", _EPILOGUES)
@pytest.mark.parametrize("mode", _SHAPES)
@requires_cuda
def test_matmul_reg_tile_epilogue(epilogue, mode, monkeypatch):
    """A register-tiled contraction with a fused pointwise epilogue stays accurate AND folds the
    epilogue into the ONE contraction kernel (no separate elementwise launch), over static and
    symbolic M."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    _pin_tile(monkeypatch, _EPILOGUE_TILE)
    monkeypatch.setenv("EMMY_REDUCE", "")  # serial K: the subject is the fused epilogue, not the restored split-K fork
    m = _DYN_M if mode == "dynamic" else _M
    rng = np.random.default_rng(0)
    feed = {"a": rng.standard_normal((1, m, _K), dtype=np.float32), "b": rng.standard_normal((_K, _N), dtype=np.float32)}
    if epilogue == "scale":
        feed["s"] = np.array([1.5], dtype=np.float32)
    elif epilogue == "bias":
        feed["bias"] = rng.standard_normal((_N,), dtype=np.float32)
    elif epilogue == "residual":
        feed["r"] = rng.standard_normal((1, m, _N), dtype=np.float32)

    be = CudaBackend()
    compiled = be.compile(_epilogue_graph(mode, epilogue))
    got = np.asarray(be.run(compiled, input_data=feed)[0].outputs["o"])
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))

    ref = _epilogue_ref(epilogue, feed)
    diff = float(np.abs(got - ref.reshape(got.shape)).max())
    assert diff < 1e-3, f"{epilogue}/{mode}: fused-epilogue mismatch (max abs err {diff})"
    assert src.count("__global__") == 1, f"{epilogue}/{mode}: epilogue must fuse into the one contraction kernel"
    assert "__c0_1" in src or "__c1_0" in src, f"{epilogue}/{mode}: expected the register-tiled tail (__c*)"


# --- scalar operand staging (the orthogonal STAGE codec) --------------------
# The ``STAGE`` codec (``d<depth>/sync|cp|tma``) annotates the typed ``Stage`` schedule struct on
# a Semiring contraction; the materializer assembles the smem slab + cooperative producer from it.


def _scalar_stage_graph(M: int = 64, N: int = 64, K: int = 64) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (M, N)), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _node_stage(tile_op):
    """The resolved operand pipeline — a schedule slice in ``TileOp.schedule`` keyed on the node
    it decorates (1r); there is no node ``stage`` field and no ``TileOp.stage``."""
    from emmy.compiler.ir.tile.ops import sched_of  # noqa: PLC0415

    op = tile_op.op
    node = op.sources[0] if getattr(op, "sources", ()) else op
    return sched_of(tile_op).stage_of(node)


def test_scalar_matmul_stages_through_pipeline(monkeypatch) -> None:
    """The ``TILE_PASSES`` chain RESOLVES the ``STAGE`` codec against the scheduled contraction and
    stamps the resolved ``Stage`` (eligibility + sizing run once, scheduler-side): a ``tma`` pin on a
    register-tiled scalar matmul resolves with the depth-aware fit-to-smem ``bk_elems`` derived (the
    scalar gmem→smem ring — ``depth`` is honored, the K-chunk sized so ``depth`` slots fit 48 KiB);
    a ``sync`` pin — no contraction transport — resolves to ``None`` (gmem-direct). The stamped
    ``knobs`` codec is the RESOLVED spelling (honest variant identity), and the explicit OFF ``""``
    when resolution declines — the row must describe the pipeline the kernel actually has."""
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415

    monkeypatch.setenv("EMMY_TILE", "f2x2")
    monkeypatch.setenv("EMMY_WORK", "t16x16")
    monkeypatch.setenv("EMMY_REDUCE", "")  # serial K: the subject is stage resolution, not the split fork
    monkeypatch.setenv("EMMY_STAGE", "d2/tma")
    out = Pipeline.build(TILE_PASSES).run(_scalar_stage_graph(), ctx=Context.from_target((9, 0)))
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    # ``STAGE`` is keyed ``@<k_axis>`` (axis-named schedule knob); ``family_value`` reads it.
    assert family_value(tile_op.knobs, "STAGE") == "d2/tma", tile_op.knobs  # resolved at the pinned depth
    stage = _node_stage(tile_op)
    assert stage is not None and stage.transport == "tma", stage
    assert stage.depth == 2, stage  # the scalar ring honors the pinned depth (slots fit 48 KiB)
    assert stage.bk_elems == 64, stage  # derived depth-aware fit-to-smem K-chunk (K=64 divides)

    monkeypatch.setenv("EMMY_STAGE", "d1/sync")  # sync is not a contraction transport
    out = Pipeline.build(TILE_PASSES).run(_scalar_stage_graph(), ctx=Context.from_target((9, 0)))
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    assert family_value(tile_op.knobs, "STAGE") == "", tile_op.knobs  # declined pin stamps the OFF value
    assert _node_stage(tile_op) is None, _node_stage(tile_op)  # resolved: ineligible pin ⇒ gmem-direct


def test_scalar_masked_n_stage_declines(monkeypatch) -> None:
    """A masked-N (overhanging inner dim) SCALAR-tier contraction must DECLINE cp.async / TMA
    staging: the B-slab fill would clamp a chunk-start column into a row-crossing gmem address and
    hang the kernel on the misaligned 16 B copy (the warp tier refuses masked-N the same way via
    ``_can_stage_warp``; a transposed B also stays gmem-direct on the scalar tier — its N-major
    slab staging is warp-only, ``test_matmul_mma_trans_b_staged``). The pin resolves to
    gmem-direct — ``stage=None``, OFF-stamped."""
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415

    monkeypatch.setenv("EMMY_TILE", "f2x2")  # tile_n=32 overhangs N=48 ⇒ masked-N
    monkeypatch.setenv("EMMY_WORK", "t16x16")
    monkeypatch.setenv("EMMY_REDUCE", "")  # serial K: isolate the stage resolution
    for stage in ("d1/cp", "d2/cp/ring", "d2/tma"):
        monkeypatch.setenv("EMMY_STAGE", stage)
        out = Pipeline.build(TILE_PASSES).run(_scalar_stage_graph(M=64, N=48, K=64), ctx=Context.from_target((9, 0)))
        tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
        assert _node_stage(tile_op) is None, (stage, _node_stage(tile_op))  # masked-N declines staging (no row-crossing fill)
        assert family_value(tile_op.knobs, "STAGE") == "", (stage, tile_op.knobs)  # OFF-stamped (gmem-direct)


def test_tma_stage_declines_below_sm90(monkeypatch) -> None:
    """A ``d*/tma*`` STAGE pin below sm_90 DECLINES to gmem-direct rather than being resolved into a
    kernel the backend cannot compile (``nvcc fatal: Unsupported gpu architecture 'sm_89a'``). The
    same pin at sm_90 resolves (:func:`test_scalar_matmul_stages_through_pipeline`), and cp.async
    staging still rings below sm_90 — the gate is TMA-specific, not a staging disable."""
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415

    monkeypatch.setenv("EMMY_TILE", "f2x2")
    monkeypatch.setenv("EMMY_WORK", "t16x16")
    monkeypatch.setenv("EMMY_REDUCE", "")
    monkeypatch.setenv("EMMY_STAGE", "d2/tma")
    # sm_89 (Ada) is below the TMA floor: the pin declines, stamping the OFF ``""`` (gmem-direct).
    out = Pipeline.build(TILE_PASSES).run(_scalar_stage_graph(), ctx=Context.from_target((8, 9)))
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    assert family_value(tile_op.knobs, "STAGE") == "", tile_op.knobs  # tma declined below sm_90
    assert _node_stage(tile_op) is None, _node_stage(tile_op)

    # Control: cp.async is unaffected by the gate — a d2/cp/ring pin still rings at sm_89.
    monkeypatch.setenv("EMMY_STAGE", "d2/cp/ring")
    out = Pipeline.build(TILE_PASSES).run(_scalar_stage_graph(), ctx=Context.from_target((8, 9)))
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    stage = _node_stage(tile_op)
    assert stage is not None and stage.transport == "cp.async", stage


@requires_cuda
@pytest.mark.parametrize("stage", ["d2/cp/ring", "d3/cp/ring"])
def test_scalar_ring_matches_gmem_direct_bit_for_bit(monkeypatch, stage):
    """The SCALAR gmem→smem prefetch ring (``STAGE=d<depth>/cp``, depth ≥ 2) runs the same
    ``staged_kloop`` phases as the warp ring — the atom contributes only the slab drain — and is a
    PURE perf transform: bit-identical to the gmem-direct scalar register tile, and the kernel
    actually rings (cp.async + a multi-slot slab)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    M, N, K = 64, 64, 512
    rng = np.random.default_rng(1)
    a = rng.standard_normal((M, K), dtype=np.float32)
    b = rng.standard_normal((K, N), dtype=np.float32)

    def _go(st: str | None) -> tuple[np.ndarray, str]:
        monkeypatch.setenv("EMMY_TILE", "f2x2")
        monkeypatch.setenv("EMMY_WORK", "t16x16")
        monkeypatch.setenv("EMMY_REDUCE", "")
        # Pin STAGE="" for the baseline — unpinned, the offline prior may legitimately stage.
        monkeypatch.setenv("EMMY_STAGE", st if st else "")
        be = CudaBackend()
        compiled = be.compile(_scalar_stage_graph(M, N, K))
        src = compiled.nodes["o"].op.kernel_source
        got = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"])
        return got, src

    staged, staged_src = _go(stage)
    gmem, gmem_src = _go(None)
    assert "cp.async" in staged_src and "__shared__" in staged_src, "the scalar ring must stage via cp.async"
    assert "cp.async" not in gmem_src, "the gmem-direct baseline must not stage"
    np.testing.assert_array_equal(staged, gmem)  # bit-identical: staging perturbs nothing
    np.testing.assert_allclose(staged.reshape(M, N), a @ b, atol=1e-3, rtol=1e-3)


def test_warp_matmul_stamps_wspec(monkeypatch) -> None:
    """A legal ``WSPEC`` split — a warp tile over a resolved TMA stage — STAMPS: ``workers``
    resolves on the ``TileOp`` and the producer band rides the ONE ``WORK`` entry as its ``+p``
    suffix (F1 — the rows carry no WSPEC key). The honest-stamping rule holds because the
    staged K-loop materializes the split (the producer/compute band split in
    ``_stage._wspec_kloop``)."""
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415

    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16/f2x2/k2")  # warp (mma) tier
    monkeypatch.setenv("EMMY_WORK", "w2x2")
    monkeypatch.setenv("EMMY_REDUCE", "")  # serial K: the subject is the WSPEC split, not the split fork
    monkeypatch.setenv("EMMY_STAGE", "d2/tma/ring")
    monkeypatch.setenv("EMMY_WSPEC", "p1")
    out = Pipeline.build(TILE_PASSES).run(_scalar_stage_graph(), ctx=Context.from_target((9, 0)))
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    assert "WSPEC" not in tile_op.knobs, tile_op.knobs  # the band spells on WORK, never a row key
    assert tile_op.knobs.get("WORK") == "w2x2+p1", tile_op.knobs  # stamped: the split materializes
    assert tile_op.workers is not None and tile_op.workers.aux_warps == 1, tile_op.workers


_WSPEC_WARP = ("mma_m16n8k16_f16/f2x2/k2", "w2x2")


@pytest.mark.parametrize(
    ("tile", "stage", "wspec"),
    [
        (_WSPEC_WARP, None, "p2"),  # warp tier, no STAGE → producer has nothing to drive
        (_WSPEC_WARP, "d2/cp", "p2"),  # cp.async: wait-group is issuing-thread-scoped — not driveable
        (_WSPEC_WARP, "d2/tma/ring", "p1:q8"),  # the producer q window is reserved (inert)
        (("f4x4", "t32x8"), "d2/cp", "p2"),  # scalar tier — WSPEC is only read on the warp path
    ],
)
def test_wspec_degrades_to_uniform(monkeypatch, tile: tuple[str, str], stage: str | None, wspec: str) -> None:
    """The ``WSPEC`` pin degrades to uniform (``workers=None``) when the split can't materialize —
    no ``STAGE`` to drive, a non-TMA transport, a reserved per-role param (``q``), or the scalar
    tier (no mma consumer). The same pin-validity rule the other codecs follow; the knob is then
    not explicitly stamped (only the ``off=""`` default)."""
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415

    _pin_tile(monkeypatch, tile)
    if stage is not None:
        monkeypatch.setenv("EMMY_STAGE", stage)
    monkeypatch.setenv("EMMY_WSPEC", wspec)
    out = Pipeline.build(TILE_PASSES).run(_scalar_stage_graph(), ctx=Context.from_target((9, 0)))
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    assert tile_op.workers is None
    assert tile_op.knobs.get("WSPEC", "") == ""  # off-default only, no explicit pin stamped


@requires_cuda
@requires_sm90
@pytest.mark.parametrize(
    ("wspec", "stage", "M"),
    [
        ("p1", "d2/tma/ring", 128),  # the ring split (the golden-config shape family)
        ("p2", "d2/tma/ring", 128),  # two producer warps
        ("p1", "d1/tma", 128),  # single-buffer split (no prefetch, handshake only)
        ("p1", "d2/tma/ring", 100),  # masked M — the producer's box zero-fills, the store guards
    ],
)
def test_warp_specialized_matmul_matches_reference(monkeypatch, wspec: str, stage: str, M: int) -> None:
    """The warp-SPECIALIZED staged mma matmul (``WSPEC=p<n>`` over a TMA stage) matches numpy: the
    producer band rides past ``block_threads`` (the launch widens, the grid decode wraps), the TMA
    fill moves to the producer's elected thread, and the compute band drains through the data /
    empty mbarrier handshake. Accuracy-gated (bit-identity vs the uniform pipeline is NOT expected —
    the split changes scheduling, not the mma math)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    N, K = 128, 256
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (M, K), dtype=F16), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N), dtype=F16), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (M, N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16/f2x2/k2")
    monkeypatch.setenv("EMMY_WORK", "w2x2")
    monkeypatch.setenv("EMMY_STAGE", stage)
    monkeypatch.setenv("EMMY_WSPEC", wspec)
    monkeypatch.setenv("EMMY_REDUCE", "")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)
    be = CudaBackend()
    compiled = be.compile(g)
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))
    aux = 32 * int(wspec[1:])
    assert f"__launch_bounds__({128 + aux})" in src, "the producer band must widen the launch"
    assert "threadIdx.x % 128" in src, "the grid decode must wrap the aux band onto the CTA's cells"
    assert "mbarrier_arrive(" in src, "the compute band must release ring slots on the empty mbarrier"
    got = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"])
    ref = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(got.astype(np.float32), ref, atol=0.25, rtol=1e-2)


@requires_cuda
@pytest.mark.parametrize("stage_mask", ["11", "10", "01", "00"])
@pytest.mark.parametrize("shape", [(64, 64, 64), (64, 47, 64), (128, 128, 128)])
def test_staged_scalar_matmul_matches_reference(monkeypatch, stage_mask, shape) -> None:
    """Every stage subset (both / A-only / B-only / none) lowers to a kernel that
    matches a numpy matmul, including a masked (non-divisor) output axis."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    monkeypatch.setenv("EMMY_STAGE", stage_mask)
    # Pin a small in-budget scalar tile so the deep-BK emission default can't overflow the staged
    # smem slab on the larger shape. Legacy env pins route through the ingest mapper.
    for k, v in (("BN", "16"), ("BM", "16"), ("FN", "2"), ("FM", "2"), ("BK", "16")):
        monkeypatch.setenv(f"EMMY_{k}", v)
    M, N, K = shape
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K), dtype=np.float32)
    b = rng.standard_normal((K, N), dtype=np.float32)
    be = CudaBackend()
    out = be.run(be.compile(_scalar_stage_graph(M, N, K)), input_data={"a": a, "b": b})[0].outputs
    got = list(out.values())[0].reshape(M, N)
    np.testing.assert_allclose(got, a @ b, atol=1e-3, rtol=1e-3)


# --- FN>1 blocked-GEMM / fused-prologue regressions -------------------------
# The per-cell + replicator + ``dedup_replicated`` pipeline reproduces register-blocked GEMM at
# the autotune knob bundles that used to fault: smem-vectorize misalignment hangs (FN=3), and the
# fused-prologue duplication that blew the nvcc compile budget (FN=32 / FN=64).


def _random(shape: tuple[int, ...], *, seed: int = 0, scale: float = 1.0, dtype=np.float32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(dtype)


def _numpy_reference(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from emmy.compiler.backend.numpy import NumpyBackend  # noqa: PLC0415

    be = NumpyBackend()
    return be.run(be.compile(graph), input_data=inputs)[0].outputs


def _assert_close(out: np.ndarray, ref: np.ndarray, *, atol_rel: float = 0.05, atol_min: float = 1e-3) -> None:
    """Tolerance scales with the reference peak — matmul reductions over K elements drift by
    ~K·eps on f32, the blocked-vs-per-cell difference is well below that floor."""
    assert out.shape == ref.shape, f"shape mismatch {out.shape} vs {ref.shape}"
    assert np.all(np.isfinite(out)), "output has non-finite values"
    peak = float(np.max(np.abs(ref)))
    atol = max(atol_min, atol_rel * peak)
    np.testing.assert_allclose(out, ref, atol=atol, rtol=atol_rel)


# (label, knobs, N, line-budget, assert-single-x-smem). The fused RMSNorm+Linear prologue must
# fold to ONE body-level chain (+ short per-cell guarded multiplies) rather than duplicate per
# register cell — else the rendered kernel blows the ~2 s nvcc budget.
_FUSED_PROLOGUE = {
    "rmsnorm_linear_n4096": {"N": 4096, "lines": 360, "one_smem": True},
    "qwen_lmhead_n4099": {"N": 4099, "lines": 850, "one_smem": False},
}


@requires_cuda
@pytest.mark.parametrize("case", list(_FUSED_PROLOGUE))
def test_fused_prologue_compiles_in_budget(case):
    """A fused RMSNorm→Linear at lm_head-style shapes keeps the N-invariant prologue chain (mean
    reduce + rsqrt + ``norm_weight·v``) as ONE body-level copy, so the rendered kernel stays under
    the nvcc budget (line count below threshold) and matches the numpy reference. The
    duplicated-prologue regression inflates the body well past these thresholds (and, for the
    divisible-N case, opens a SECOND ``x_smem`` slab)."""
    import re as _re  # noqa: PLC0415

    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    spec = _FUSED_PROLOGUE[case]
    M, K, N = 2, 1024, spec["N"]
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (M, K)), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("wn", (K,)), node_id="wn")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("wl", (N, K)), node_id="wl")
    g.add_node(op=RmsNormOp(eps=1e-6), inputs=["x", "wn"], output=Tensor("xn", (M, K)), node_id="xn")
    g.add_node(op=LinearOp(), inputs=["xn", "wl"], output=Tensor("o", (M, N)), node_id="o")
    g.inputs, g.outputs = ["x", "wn", "wl"], ["o"]

    inputs = {
        "x": _random((M, K), seed=5),
        "wn": _random((K,), seed=6, scale=0.1),
        "wl": _random((N, K), seed=7, scale=0.02),  # scaled so output stays bounded
    }
    # Reference BEFORE backend.compile (which mutates ops in place).
    ref = _numpy_reference(g, inputs)["o"]

    backend = CudaBackend()
    compiled = backend.compile(g)
    cuda_ops = [n.op for n in compiled.nodes.values() if isinstance(n.op, CudaOp)]
    assert cuda_ops, "expected a CudaOp in the lowered graph"
    cuda_src = "\n".join(op.kernel_source for op in cuda_ops)
    n_lines = cuda_src.count("\n")
    assert n_lines < spec["lines"], (
        f"{case}: rendered kernel is {n_lines} lines (budget {spec['lines']}) — a regression that "
        f"fails to dedup the N-invariant prologue chain inflates it."
    )
    if spec["one_smem"]:
        # ONE smem allocation for the RMSNorm input shared by the mean reduce + matmul body; a
        # regression opening its own RegisterTile staging context forces a second ``x_smem``.
        x_smem_decls = len(_re.findall(r"__shared__\s+(?:__align__\([0-9]+\)\s+)?float\s+x_smem\b", cuda_src))
        assert x_smem_decls == 1, f"{case}: expected 1 ``__shared__ float x_smem`` decl (per-cell shares staging); got {x_smem_decls}"

    out = backend.run(compiled, input_data=inputs)[0].outputs["o"]
    _assert_close(out, ref, atol_min=1e-3)


# =========================================================================== #
# Warp MMA tier (tensor-core mma.sync) — the WARP codec materializer.
# =========================================================================== #
# The contraction tiles onto ``WM·WN`` warps of ``mma_m16n8k16`` atom cells: f16/bf16 operands,
# f32 accumulate, f16|f32 store. Requires sm_90+ (the warp tier is pin-only / non-functional
# below: ldmatrix host fault + ``sm_NNa`` TMA compile).
_WARP_PIN = ("mma_m16n8k16_f16/f4x8/k2", "w2x2")  # WM·FM·atom_m = WN·FN·atom_n = 128 tile, 128 threads
_F16 = "f16"


def _mma_matmul_graph(mode: str, M: int, N: int, K: int, out: str, trans: bool):
    """A hand-built ``C[i,j] = Σ_k A[i,k]·B[…]`` over f16 operands; ``trans`` makes B ``[j,k]``
    (Q@Kᵀ, K last). ``mode='dynamic'`` makes the M (row) axis symbolic (the dynamic-grid tier)."""
    from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415
    from emmy.compiler.ir.expr import Var  # noqa: PLC0415
    from emmy.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write  # noqa: PLC0415
    from emmy.compiler.ir.stmt import Accum, Assign  # noqa: PLC0415

    Mg = dyn_M(mode, M)
    i, j, k = Axis("i", Mg), Axis("j", N), Axis("k", K)
    b_index = (Var("j"), Var("k")) if trans else (Var("k"), Var("j"))
    op = LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Loop(
                                axis=k,
                                body=(
                                    Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
                                    Load(name="b_v", input="b", index=b_index),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                    Accum(name="acc", value="p"),
                                ),
                            ),
                            Write(output="c", index=(Var("i"), Var("j")), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
    )
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Mg, K), dtype=F16), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (N, K) if trans else (K, N), dtype=F16), node_id="b")
    g.add_node(op, ["a", "b"], Tensor("c", (Mg, N), dtype=_dtype(out)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g


def _mma_symbolic_k_graph(M: int, N: int, *, trans: bool):
    """``C[i,j] = Σ_k A[i,k]·B[…]`` with the contraction (K) axis SYMBOLIC (``seq_len``) — ``trans``
    makes B ``[j,k]`` (transposed-B, K contiguous), so the warp mma zero-fills the masked-K tail
    through the (n,k)-swapped trans helper. M / N are static tile divisors (no M/N mask)."""
    from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415
    from emmy.compiler.ir.expr import Var  # noqa: PLC0415
    from emmy.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write  # noqa: PLC0415
    from emmy.compiler.ir.stmt import Accum, Assign  # noqa: PLC0415

    Kg = Dim("seq_len")
    i, j, k = Axis("i", M), Axis("j", N), Axis("k", Kg)
    b_index = (Var("j"), Var("k")) if trans else (Var("k"), Var("j"))
    op = LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Loop(
                                axis=k,
                                body=(
                                    Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
                                    Load(name="b_v", input="b", index=b_index),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                    Accum(name="acc", value="p"),
                                ),
                            ),
                            Write(output="c", index=(Var("i"), Var("j")), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
    )
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (M, Kg), dtype=F16), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (N, Kg) if trans else (Kg, N), dtype=F16), node_id="b")
    g.add_node(op, ["a", "b"], Tensor("c", (M, N), dtype=F16), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g


def _compile_run_mma(graph, feed: dict) -> tuple[np.ndarray, str]:
    """Compile the (already WARP-pinned) graph and run it on the seeded f16 operands in ``feed``;
    return ``(output, kernel_source)``."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    be = CudaBackend()
    compiled = be.compile(graph)
    got = np.asarray(be.run(compiled, input_data=feed)[0].outputs["c"])
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))
    return got, src


# (M, N, K, out_dtype, transposed-B). 128 / 256 are tile (128) multiples (exact-cover static);
# the dynamic column runs at M+2 to straddle the tile and exercise the masked store / clamp.
_MMA_CASES = [
    (128, 128, 128, "f32", False),
    (256, 256, 128, "f32", False),
    (128, 128, 128, "f16", False),
    (128, 256, 128, "f16", False),
    (128, 128, 128, "f32", True),  # transposed-B (Q@Kᵀ)
    (128, 128, 128, "f16", True),
]


@pytest.mark.parametrize(("M", "N", "K", "out", "trans"), _MMA_CASES)
@pytest.mark.parametrize("mode", _SHAPES)
@requires_sm90
@requires_cuda
def test_matmul_mma_coverage(M, N, K, out, trans, mode, monkeypatch):
    """An f16×f16 matmul under the pinned ``EMMY_TILE`` codec lowers to ``mma.sync`` and
    agrees with the f32 reference (within fp16 tolerance) — for canonical AND transposed-B
    operands, f16 AND f32 output, over a STATIC M (tile divisor) and a SYMBOLIC M run at a
    straddling length (the dynamic-grid tier + the masked-tile store)."""
    _pin_tile(monkeypatch, _WARP_PIN)
    # Pin the REDUCE codec serial: the split-K f32 workspace legalized ``g<w>k`` for f16 output,
    # and an unpinned greedy pick can split this shape (partials + tail kernel), moving the
    # ``__floats2half2_rn`` downconvert out of the fused epilogue these source assertions check.
    monkeypatch.setenv("EMMY_REDUCE", "")
    run_m = M if mode == "static" else M + 2
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((run_m, K)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((N, K) if trans else (K, N)) * 0.1).astype(np.float16)
    got, src = _compile_run_mma(_mma_matmul_graph(mode, M, N, K, out, trans), {"a": a, "b": b})

    ref = a.astype(np.float32) @ (b.T if trans else b).astype(np.float32)
    diff = float(np.abs(got.reshape(run_m, N) - ref).max())
    tol = 5e-2 if out == _F16 else 1e-2
    assert diff < tol, f"{M}x{N}x{K} out={out} trans={trans}/{mode}: mma mismatch (max abs err {diff})"

    assert "mma.sync.aligned.m16n8k16" in src, "the s16816 mma.sync instruction must be emitted"
    assert "emmy_mma_load_a_gmem" in src, "operands must load via the gmem-direct fragment helper"
    assert "wmma::" not in src, "the mma.sync path must not mix in legacy wmma intrinsics"
    assert ("__floats2half2_rn" in src) == (out == _F16), "f16 output needs the fp32→fp16 __half2 downconvert"
    if trans:
        assert "emmy_mma_load_b_gmem_trans" in src, "transposed-B must use the gmem-direct trans helper"
    if mode == "dynamic":
        assert "int seq_len" in src, "the symbolic-M grid must carry the runtime extent arg"


# =========================================================================== #
# staged transposed-B (the serving F.linear layout) — the N-major B slab.
# =========================================================================== #

_TRANS_STAGES = {"cp": "d2/cp/ring", "tma": "d2/tma/ring"}


@pytest.mark.parametrize("transport", ["cp", "tma"])
@pytest.mark.parametrize("mode", _SHAPES)
@requires_sm90
@requires_cuda
def test_matmul_mma_trans_b_staged(transport, mode, monkeypatch):
    """A transposed-B (the serving ``F.linear`` layout) warp matmul STAGES under cp.async and TMA:
    the B slab keeps the operand's own N-MAJOR orientation (``tile_n`` rows × ``bk`` K cols — K
    stride-1 in gmem and smem alike, so the fill's chunks stay contiguous) and drains via the
    plain (no ``.trans``) ldmatrix — no gmem-direct trans helper in the kernel. Bit-identical to
    the gmem-direct sibling (same mma order — staging perturbs nothing), and correct at a
    symbolic/masked M (the A-fill clamp; the B slab has no M dimension). This is the serving
    ``.lin`` fork's transport — the gap class the gmem-direct-only enumeration left 1.3–2.75×
    behind cuBLAS."""
    M = N = K = 128
    _pin_tile(monkeypatch, _WARP_PIN)
    monkeypatch.setenv("EMMY_REDUCE", "")
    run_m = M if mode == "static" else M + 2
    rng = np.random.default_rng(2)
    a = (rng.standard_normal((run_m, K)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((N, K)) * 0.1).astype(np.float16)

    monkeypatch.setenv("EMMY_STAGE", "")
    gmem, gmem_src = _compile_run_mma(_mma_matmul_graph(mode, M, N, K, _F16, True), {"a": a, "b": b})
    monkeypatch.setenv("EMMY_STAGE", _TRANS_STAGES[transport])
    staged, src = _compile_run_mma(_mma_matmul_graph(mode, M, N, K, _F16, True), {"a": a, "b": b})

    # Call-site patterns (`helper(_b…`) — the preamble defines every helper unconditionally,
    # so bare-name membership can't tell a call from a definition.
    assert "_b_smem" in src, "the transposed B must stage into an smem slab"
    assert "emmy_mma_load_b_gmem(_b" not in src and "emmy_mma_load_b_gmem_trans(_b" not in src, (
        "a staged B must not fall back to the gmem-direct helpers"
    )
    assert "emmy_ldmatrix_x2_trans(_b" not in src, "the N-major slab drains via the PLAIN x2 (no .trans)"
    assert "emmy_ldmatrix_x2(_b" in src or "emmy_ldmatrix_x4_pair(_b" in src, "the b_trans staged drain must ldmatrix the slab"
    assert "emmy_mma_load_b_gmem_trans(_b" in gmem_src, "the gmem-direct sibling still uses the trans helper"
    np.testing.assert_array_equal(staged, gmem)  # bit-identical: staging perturbs nothing
    ref = a.astype(np.float32) @ b.T.astype(np.float32)
    diff = float(np.abs(staged.reshape(run_m, N).astype(np.float32) - ref).max())
    assert diff < 5e-2, f"staged trans-B ({transport}/{mode}): mismatch vs f32 reference (max abs err {diff})"


def test_trans_b_offers_staged_rows(monkeypatch):
    """The transposed-B fork OFFERS the staged transports unpinned (no GPU — enumeration only).
    The serving ``F.linear`` forks used to enumerate gmem-direct rows ONLY (the ``.lin`` golden
    drift-warning class); the N-major B slab lifts that: d*/cp* and d*/tma* spellings ride the
    fork at sm_90+, cp.async alone below the TMA floor (sm_89)."""
    from emmy.compiler.pipeline.search.golden_eval import enumerate_graph  # noqa: PLC0415

    for var in ("EMMY_TILE", "EMMY_STAGE", "EMMY_REDUCE", "EMMY_WSPEC"):
        monkeypatch.delenv(var, raising=False)

    def stages(cc) -> set[str]:
        rows = enumerate_graph(_mma_matmul_graph("static", 128, 128, 128, _F16, True), Context.from_target(cc))
        return {str(v) for r in rows for kk, v in r.items() if kk.startswith("STAGE")}

    offered = stages((12, 0))
    assert any("/cp" in s for s in offered), offered
    assert any("/tma" in s for s in offered), offered
    offered_ada = stages((8, 9))
    assert any("/cp" in s for s in offered_ada), offered_ada
    assert not any("/tma" in s for s in offered_ada), offered_ada  # TMA stays sm_90-gated


# =========================================================================== #
# f16-accumulate warp tier (the ``_f16acc`` atom — chunked f16→f32 register promote).
# =========================================================================== #
# The mma chain accumulates into packed f16 fragments (``_ch*``) at the full HMMA rate and a
# periodic ``FragmentPromote`` folds them into the f32 shadows the store reads (``_c*``) —
# staged rows promote per bk chunk, the gmem-direct K-loop every ``_F16ACC_STEPS`` plus a final
# fold. Pin-driven here (the enumeration gate is F16_MMA_F32_ACC / FAST_MATH — covered by
# ``test_f16acc_enumeration_gate``); accuracy is checked against BOTH the f32 reference and the
# f32-accumulate SIBLING (same schedule, atom swapped) — the sibling bound is what the loose fp16
# eager tolerance can't see.
_F16ACC_PIN = ("mma_m16n8k16_f16_f16/f4x8/k2", "w2x2")
_F16ACC_SIBLING_PIN = _WARP_PIN  # the same w2x2 / f4x8 / k2 schedule on the f32-accumulate atom
_F16ACC_STAGES = {"gmem": "", "cp": "d2/cp/ring", "tma": "d4/tma/ring"}


@pytest.mark.parametrize("stage", list(_F16ACC_STAGES))
@pytest.mark.parametrize("mode", _SHAPES)
@requires_sm90
@requires_cuda
def test_matmul_mma_f16acc_coverage(stage, mode, monkeypatch):
    """The f16-accumulate atom over every transport (gmem-direct / cp.async ring / TMA) and a
    static AND symbolic-M shape: the kernel carries the packed f16 mma targets + the chunk
    promote, and agrees with the f32 reference and (tightly) with its f32-accumulate sibling."""
    if stage == "tma" and not _supports_tma():
        pytest.skip("TMA needs sm_90+")
    M = N = K = 128
    monkeypatch.setenv("EMMY_STAGE", _F16ACC_STAGES[stage])
    run_m = M if mode == "static" else M + 2
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((run_m, K)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    feed = {"a": a, "b": b}
    _pin_tile(monkeypatch, _F16ACC_PIN)
    got, src = _compile_run_mma(_mma_matmul_graph(mode, M, N, K, _F16, False), feed)
    _pin_tile(monkeypatch, _F16ACC_SIBLING_PIN)
    sib, sib_src = _compile_run_mma(_mma_matmul_graph(mode, M, N, K, _F16, False), feed)

    ref = a.astype(np.float32) @ b.astype(np.float32)
    diff = float(np.abs(got.reshape(run_m, N).astype(np.float32) - ref).max())
    assert diff < 5e-2, f"f16acc {stage}/{mode}: mismatch vs f32 reference (max abs err {diff})"
    sib_diff = float(np.abs(got.astype(np.float32) - sib.astype(np.float32)).max())
    assert sib_diff < 5e-3, f"f16acc {stage}/{mode}: drifted from the f32-accumulate sibling (max abs err {sib_diff})"

    assert "emmy_mma_m16n8k16_f16_f16(_ch0_0" in src, "the mma chain must target the packed f16 fragments"
    assert "emmy_mma_promote_f16acc(_c0_0, _ch0_0);" in src, "the chunk promote into the f32 shadow must be emitted"
    assert "_ch0_0" not in sib_src, "the f32-accumulate sibling must not declare f16 mma fragments"


@requires_sm90
@requires_cuda
def test_matmul_mma_f16acc_symbolic_k(monkeypatch):
    """A SYMBOLIC-K f16acc matmul run at a non-multiple length: the masked-K tail zero-fills the
    operand fragments and the unconditional final promote (after the K-loop) folds the partial
    last chunk — the gmem-direct cadence path's tail case."""
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f16/f1x1")
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    monkeypatch.delenv("EMMY_STAGE", raising=False)
    M = N = 128
    run_k = 70  # off the 16-step grid AND off the _F16ACC_STEPS·16 promote period
    rng = np.random.default_rng(1)
    a = (rng.standard_normal((M, run_k)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((run_k, N)) * 0.1).astype(np.float16)
    got, src = _compile_run_mma(_mma_symbolic_k_graph(M, N, trans=False), {"a": a, "b": b})
    ref = a.astype(np.float32) @ b.astype(np.float32)
    diff = float(np.abs(got.reshape(M, N).astype(np.float32) - ref).max())
    assert diff < 5e-2, f"f16acc symbolic-K: mismatch vs f32 reference (max abs err {diff})"
    assert "emmy_mma_promote_f16acc(_c0_0, _ch0_0);" in src


def test_f16acc_enumeration_gate(monkeypatch):
    """The precision-gated enumeration (no GPU): unset, no f16acc forks anywhere; ``FAST_MATH``
    offers them on the consumer dies (sm_120) but NOT the datacenter parts (sm_90, full-rate
    f32-accumulate); the precise ``F16_MMA_F32_ACC`` pin is authoritative both ways."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden_eval import enumerate_graph  # noqa: PLC0415

    def offers(cc, **env) -> bool:
        for var in ("EMMY_FAST_MATH", "EMMY_F16_MMA_F32_ACC"):
            monkeypatch.delenv(var, raising=False)
        for var, val in env.items():
            monkeypatch.setenv(var, val)
        # A fresh graph per gate state — enumeration caches rows per graph object.
        rows = enumerate_graph(_mma_matmul_graph("static", 128, 128, 128, _F16, False), Context.from_target(cc))
        return any("mma_m16n8k16_f16_f16/" in str(v) for r in rows for kk, v in r.items() if kk.startswith("TILE"))

    assert not offers((12, 0)), "gate unset: no f16acc forks"
    assert offers((12, 0), EMMY_FAST_MATH="1"), "FAST_MATH on a consumer die must offer the forks"
    assert not offers((9, 0), EMMY_FAST_MATH="1"), "FAST_MATH must not offer them on a full-rate f32-acc target"
    assert offers((9, 0), EMMY_F16_MMA_F32_ACC="1"), "the precise pin offers everywhere"
    assert not offers((12, 0), EMMY_FAST_MATH="1", EMMY_F16_MMA_F32_ACC="0"), "the precise pin wins over the umbrella"


def _mma_qk_graph(B: int, H: int, M: int, N: int, D: int):
    """A hand-built **batched, transposed-B** ``S[b,h,m,kv] = Σ_dd Q[b,h,m,dd]·K[b,h,kv,dd]`` over
    f16 operands — exactly the score contraction flash's warp materializer constructs (leading
    ``(b, h)`` batch axes carried verbatim, ``dd`` last in K's index so it's a ``b_trans`` Q@Kᵀ)."""
    from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415
    from emmy.compiler.ir.expr import Var  # noqa: PLC0415
    from emmy.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write  # noqa: PLC0415
    from emmy.compiler.ir.stmt import Accum, Assign  # noqa: PLC0415

    b, h, m, kv, dd = Axis("b", B), Axis("h", H), Axis("m", M), Axis("kv", N), Axis("dd", D)
    dd_loop = Loop(
        axis=dd,
        body=(
            Load(name="q_v", input="q", index=(Var("b"), Var("h"), Var("m"), Var("dd"))),
            Load(name="k_v", input="k", index=(Var("b"), Var("h"), Var("kv"), Var("dd"))),
            Assign(name="p", op=ElementwiseImpl("multiply"), args=("q_v", "k_v")),
            Accum(name="acc", value="p"),
        ),
    )
    # Write the score cell after the dd reduction (inside the kv loop), then nest the grid axes.
    kv_loop = Loop(axis=kv, body=(dd_loop, Write(output="c", index=(Var("b"), Var("h"), Var("m"), Var("kv")), value="acc")))
    nest = kv_loop
    for ax in (m, h, b):
        nest = Loop(axis=ax, body=(nest,))
    op = LoopOp(body=(nest,))
    g = Graph()
    g.add_node(InputOp(), [], Tensor("q", (B, H, M, D), dtype=F16), node_id="q")
    g.add_node(InputOp(), [], Tensor("k", (B, H, N, D), dtype=F16), node_id="k")
    g.add_node(op, ["q", "k"], Tensor("c", (B, H, M, N), dtype=F32), node_id="c")
    g.inputs, g.outputs = ["q", "k"], ["c"]
    return g


@pytest.mark.parametrize(("B", "H", "M", "N", "D"), [(1, 4, 128, 128, 64), (2, 3, 128, 128, 32)])
@requires_sm90
@requires_cuda
def test_mma_batched_qk_matches_torch(B, H, M, N, D, monkeypatch):
    """The shared contraction codegen tiles a **batched transposed-B** Q@Kᵀ (flash's score shape:
    leading ``(b, h)`` axes + ``b_trans``) onto ``mma.sync`` and agrees with torch — the reuse seam
    the warp-flash materializer rides (a ``Contraction`` placed under leading batch grid axes)."""
    import torch  # noqa: PLC0415

    _pin_tile(monkeypatch, _WARP_PIN)
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((B, H, M, D)) * 0.1).astype(np.float16)
    k = (rng.standard_normal((B, H, N, D)) * 0.1).astype(np.float16)
    got, src = _compile_run_mma(_mma_qk_graph(B, H, M, N, D), {"q": q, "k": k})

    ref = torch.matmul(torch.tensor(q).float(), torch.tensor(k).float().transpose(-1, -2)).numpy()
    diff = float(np.abs(got.reshape(B, H, M, N) - ref).max())
    assert diff < 1e-2, f"batched QKᵀ {B}x{H}x{M}x{N}x{D}: mma mismatch (max abs err {diff})"
    assert "mma.sync.aligned.m16n8k16" in src, "the batched Q@Kᵀ must emit the s16816 mma.sync"
    assert "emmy_mma_load_b_gmem_trans" in src, "transposed-B (K last) must use the gmem-direct trans helper"


def _mma_pv_graph(B: int, H: int, M: int, KV: int, D: int):
    """A hand-built **batched, canonical-B** ``O[b,h,m,d] = Σ_kv P[b,h,m,kv]·V[b,h,kv,d]`` over f16
    operands — the flash P@V contraction (the second mma): leading ``(b, h)`` batch axes, reduce over
    ``kv``, and V indexed ``[kv, d]`` (``kv`` first, the canonical non-transposed B)."""
    from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415
    from emmy.compiler.ir.expr import Var  # noqa: PLC0415
    from emmy.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write  # noqa: PLC0415
    from emmy.compiler.ir.stmt import Accum, Assign  # noqa: PLC0415

    b, h, m, d, kv = Axis("b", B), Axis("h", H), Axis("m", M), Axis("d", D), Axis("kv", KV)
    kv_loop = Loop(
        axis=kv,
        body=(
            Load(name="p_v", input="p", index=(Var("b"), Var("h"), Var("m"), Var("kv"))),
            Load(name="v_v", input="v", index=(Var("b"), Var("h"), Var("kv"), Var("d"))),
            Assign(name="pr", op=ElementwiseImpl("multiply"), args=("p_v", "v_v")),
            Accum(name="acc", value="pr"),
        ),
    )
    # Write the output cell after the kv reduction (inside the d loop), then nest the grid axes.
    d_loop = Loop(axis=d, body=(kv_loop, Write(output="c", index=(Var("b"), Var("h"), Var("m"), Var("d")), value="acc")))
    nest = d_loop
    for ax in (m, h, b):
        nest = Loop(axis=ax, body=(nest,))
    op = LoopOp(body=(nest,))
    g = Graph()
    g.add_node(InputOp(), [], Tensor("p", (B, H, M, KV), dtype=F16), node_id="p")
    g.add_node(InputOp(), [], Tensor("v", (B, H, KV, D), dtype=F16), node_id="v")
    g.add_node(op, ["p", "v"], Tensor("c", (B, H, M, D), dtype=F32), node_id="c")
    g.inputs, g.outputs = ["p", "v"], ["c"]
    return g


@pytest.mark.parametrize(("B", "H", "M", "KV", "D"), [(1, 4, 128, 128, 128), (2, 3, 128, 256, 128)])
@requires_sm90
@requires_cuda
def test_mma_batched_pv_matches_torch(B, H, M, KV, D, monkeypatch):
    """The shared contraction codegen tiles a **batched canonical-B** P@V (flash's second mma:
    leading ``(b, h)`` axes, reduce over ``kv``, V stored ``[kv, d]``) onto ``mma.sync`` and agrees
    with torch — the other half of the contraction reuse the warp-flash materializer rides."""
    import torch  # noqa: PLC0415

    _pin_tile(monkeypatch, _WARP_PIN)
    rng = np.random.default_rng(0)
    p = (rng.standard_normal((B, H, M, KV)) * 0.1).astype(np.float16)
    v = (rng.standard_normal((B, H, KV, D)) * 0.1).astype(np.float16)
    got, src = _compile_run_mma(_mma_pv_graph(B, H, M, KV, D), {"p": p, "v": v})

    ref = torch.matmul(torch.tensor(p).float(), torch.tensor(v).float()).numpy()
    diff = float(np.abs(got.reshape(B, H, M, D) - ref).max())
    assert diff < 1e-2, f"batched P@V {B}x{H}x{M}x{KV}x{D}: mma mismatch (max abs err {diff})"
    assert "mma.sync.aligned.m16n8k16" in src, "the batched P@V must emit the s16816 mma.sync"
    assert "emmy_mma_load_b_gmem(" in src, "canonical B (kv first) must use the gmem-direct non-trans helper"


# Fused epilogues over the warp tier — a projection ``Map`` (or a causal ``Select``) folds into
# the ``RegStore`` per fragment element.
_MMA_EPILOGUES = ("bias", "relu", "residual", "causal")


def _mma_epilogue_graph(mode: str, epilogue: str):
    """An f16 ``128³`` matmul with a fused ``epilogue`` (causal uses a hand-built coord ``Select``;
    the rest fold a frontend ``ElementwiseOp``)."""
    M = N = K = 128
    Mg = dyn_M(mode, M)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Mg, K), dtype=F16), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N), dtype=F16), node_id="b")
    if epilogue == "causal":
        from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var  # noqa: PLC0415
        from emmy.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write  # noqa: PLC0415
        from emmy.compiler.ir.stmt import Accum, Assign, Select, SelectBranch  # noqa: PLC0415

        i, j, k = Axis("i", Mg), Axis("j", N), Axis("k", K)
        op = LoopOp(
            body=(
                Loop(
                    axis=i,
                    body=(
                        Loop(
                            axis=j,
                            body=(
                                Load(name="mz", input="mz", index=(Literal(0, "int"),)),
                                Load(name="mf", input="mf", index=(Literal(0, "int"),)),
                                Loop(
                                    axis=k,
                                    body=(
                                        Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
                                        Load(name="b_v", input="b", index=(Var("k"), Var("j"))),
                                        Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                        Accum(name="acc", value="p"),
                                    ),
                                ),
                                Select(
                                    name="out",
                                    branches=(
                                        SelectBranch(value="acc", select=BinaryExpr("<=", Var("j"), Var("i"))),
                                        SelectBranch(value="mf", select=BinaryExpr(">", Var("j"), Var("i"))),
                                    ),
                                ),
                                Write(output="c", index=(Var("i"), Var("j")), value="out"),
                            ),
                        ),
                    ),
                ),
            ),
        )
        g.add_node(InputOp(), [], Tensor("mz", (1,), dtype=F32), node_id="mz")
        g.add_node(InputOp(), [], Tensor("mf", (1,), dtype=F32), node_id="mf")
        g.add_node(op, ["a", "b", "mz", "mf"], Tensor("c", (Mg, N), dtype=F32), node_id="c")
        g.inputs, g.outputs = ["a", "b", "mz", "mf"], ["c"]
        return g

    g.add_node(MatmulOp(), ["a", "b"], Tensor("ab", (Mg, N), dtype=F32), node_id="ab")
    inputs = ["a", "b"]
    if epilogue == "bias":
        g.add_node(InputOp(), [], Tensor("bias", (N,), dtype=F32), node_id="bias")
        g.add_node(ElementwiseOp("add"), ["ab", "bias"], Tensor("c", (Mg, N), dtype=F32), node_id="c")
        inputs.append("bias")
    elif epilogue == "relu":
        g.add_node(ElementwiseOp("relu"), ["ab"], Tensor("c", (Mg, N), dtype=F32), node_id="c")
    else:  # residual
        g.add_node(InputOp(), [], Tensor("r", (Mg, N), dtype=F32), node_id="r")
        g.add_node(ElementwiseOp("add"), ["ab", "r"], Tensor("c", (Mg, N), dtype=F32), node_id="c")
        inputs.append("r")
    g.inputs, g.outputs = inputs, ["c"]
    return g


@pytest.mark.parametrize("epilogue", _MMA_EPILOGUES)
@pytest.mark.parametrize("mode", _SHAPES)
@requires_sm90
@requires_cuda
def test_matmul_mma_epilogue_coverage(epilogue, mode, monkeypatch):
    """A warp-tier matmul with a fused pointwise / causal epilogue stays accurate AND folds the
    epilogue into the ONE mma.sync kernel (the per-element ``RegStore`` chain), over static and
    symbolic M."""
    _pin_tile(monkeypatch, _WARP_PIN)
    monkeypatch.setenv("EMMY_REDUCE", "")  # serial K: the subject is the fused epilogue, not the restored split-K fork
    M = N = K = 128
    run_m = M if mode == "static" else M + 2
    rng = np.random.default_rng(1)
    a = (rng.standard_normal((run_m, K)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    base = a.astype(np.float32) @ b.astype(np.float32)
    feed = {"a": a, "b": b}
    if epilogue == "bias":
        bias = rng.standard_normal((N,)).astype(np.float32)
        feed["bias"] = bias
        ref = base + bias
    elif epilogue == "relu":
        ref = np.maximum(base, 0.0)
    elif epilogue == "residual":
        r = rng.standard_normal((run_m, N)).astype(np.float32)
        feed["r"] = r
        ref = base + r
    else:  # causal
        feed["mz"] = np.array([0.0], np.float32)
        feed["mf"] = np.array([-1e30], np.float32)
        keep = np.arange(N)[None, :] <= np.arange(run_m)[:, None]
        ref = np.where(keep, base, -1e30)

    got, src = _compile_run_mma(_mma_epilogue_graph(mode, epilogue), feed)
    diff = float(np.abs(got.reshape(run_m, N) - ref).max())
    assert diff < 1e-2, f"{epilogue}/{mode}: fused-epilogue mma mismatch (max abs err {diff})"
    assert "mma.sync.aligned.m16n8k16" in src, f"{epilogue}/{mode}: must reach the warp tier"
    assert src.count("__global__") == 1, f"{epilogue}/{mode}: epilogue must fuse into the one mma kernel"
    if epilogue == "causal":
        assert "?" in src, "the causal mask must render as a per-element ternary"


# --- static-vs-dynamic parity across cp.async + TMA transports --------------
# One matmul op compiled BOTH shape-specialised (static M) and dynamic (Dim('seq_len')), run at
# the SAME runtime M, across cp.async AND TMA (pinned). K static so the source innermost dim stays
# static — TMA-eligible. The ``shape_mode`` × ``transport`` fixtures fan one body over the matrix.
_PN, _PK = 1024, 512
_WARP_CODEC = ("mma_m16n8k16_f16/f2x2/k2", "w2x2")  # WM=WN=FM=FN=2, BK=2 — the 64-row tile


def _parity_mma_graph(mode: str, *, M: int):
    """``a @ b`` with the M axis static (``mode='static'``) or symbolic (``Dim('seq_len')``)."""
    m_dim = dyn_M(mode, M)
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m_dim, _PK), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (_PK, _PN), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m_dim, _PN), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


@pytest.fixture(params=["cp.async", "tma"])
def transport(request, monkeypatch) -> str:
    """Pin the warp tile (``WARP`` codec) + the operand-staging transport (``STAGE`` codec —
    ``d2/cp`` = cp.async, ``d2/tma`` = cp.async.bulk.tensor). The "pinned knobs" fixture."""
    _pin_tile(monkeypatch, _WARP_CODEC)
    monkeypatch.setenv("EMMY_STAGE", "d2/tma" if request.param == "tma" else "d2/cp")
    # REDUCE pinned serial: without it the pick may ride a g<w>k split sibling (a partial +
    # finalize PAIR — these tests assert on the single ``o`` kernel), depending on how the
    # live prior ranks the split rows. The pinned path must not hang on prior rank order.
    monkeypatch.setenv("EMMY_REDUCE", "")
    return request.param


def test_pinned_transport_and_shape_fire(shape_mode, transport):
    """CPU render (forced sm_120): the pinned knobs select the intended transport and the symbolic
    M threads a runtime ``seq_len`` arg — so the accuracy test below exercises the path it claims."""
    lowered = Pipeline.build(CUDA_PASSES).run(_parity_mma_graph(shape_mode, M=512), ctx=Context(compute_capability=(12, 0)))
    src = lowered.nodes["o"].op.kernel_source
    assert "mma.sync.aligned.m16n8k16" in src and "ldmatrix" in src, "must be on the s16816 tensor-core tier"
    if transport == "tma":
        assert "cp.async.bulk.tensor" in src, f"{shape_mode}/tma: TMA must fire"
        assert "CUtensorMap" in src, "TMA kernel must take the descriptor param"
    else:
        assert "cp.async.bulk.tensor" not in src, f"{shape_mode}/cp.async: TMA must NOT fire"
        assert "cp.async" in src, f"{shape_mode}/cp.async: operands must stage via cp.async"
        assert "__shared__" in src, f"{shape_mode}/cp.async: staged operands need an smem slab"
    if shape_mode == "dynamic":
        assert "int seq_len" in src, "dynamic kernel must carry the runtime extent arg"
    else:
        assert "int seq_len" not in src, "static kernel bakes M — no runtime extent arg"


@requires_sm90
@requires_cuda
@pytest.mark.parametrize("M", [256, 512])
def test_static_dynamic_mma_parity(shape_mode, transport, M):
    """The SAME matmul, compiled static and dynamic, on cp.async and TMA, is accurate vs torch —
    so the four paths agree. ``M`` is a multiple of the WM·FM·16 = 64-row tile, where ALL four
    combos fire. Dynamic robustness at off-hint sizes is covered by the masked sweep below."""
    if transport == "tma" and not _supports_tma():
        pytest.skip("TMA needs sm_90+ (Hopper / Blackwell)")
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    be = CudaBackend()
    compiled = be.compile(_parity_mma_graph(shape_mode, M=M))
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((M, _PK)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((_PK, _PN)) * 0.1).astype(np.float16)
    result, _ = be.run(compiled, input_data={"a": a, "b": b})
    got = result.outputs["o"].astype(np.float32)
    want = a.astype(np.float32) @ b.astype(np.float32)
    assert got.shape == (M, _PN)
    diff = np.abs(got - want).max()
    assert diff < 5e-2, f"{shape_mode}/{transport} M={M}: max abs err {diff}"


@requires_sm90
@requires_cuda
@pytest.mark.parametrize("M", [128, 256])
def test_staged_matches_gmem_direct_bit_for_bit(monkeypatch, M):
    """cp.async operand staging is a PURE perf transform: the staged kernel (``STAGE=d2/cp``) must
    produce **bit-identical** output to the gmem-direct baseline (same ``WARP`` tile, no ``STAGE``)
    on the same inputs — and actually stage (cp.async + smem slab) where the baseline does not."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    rng = np.random.default_rng(0)
    a = (rng.standard_normal((M, _PK)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((_PK, _PN)) * 0.1).astype(np.float16)

    def _go(stage: str | None) -> tuple[np.ndarray, str]:
        _pin_tile(monkeypatch, _WARP_CODEC)
        monkeypatch.setenv("EMMY_REDUCE", "")  # serial K: the baseline must not reroute through the restored split-K fork
        # Pin STAGE="" for the baseline — unpinned, the offline prior may legitimately stage.
        monkeypatch.setenv("EMMY_STAGE", stage if stage else "")
        be = CudaBackend()
        compiled = be.compile(_parity_mma_graph("static", M=M))
        src = compiled.nodes["o"].op.kernel_source
        got = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"])
        return got, src

    staged, staged_src = _go("d2/cp")
    gmem, gmem_src = _go(None)
    assert "cp.async" in staged_src and "__shared__" in staged_src, "STAGE=d2/cp must stage via a cp.async smem slab"
    assert "cp.async" not in gmem_src, "the gmem-direct baseline must not stage"
    np.testing.assert_array_equal(staged, gmem)  # bit-identical: staging perturbs nothing
    want = a.astype(np.float32) @ b.astype(np.float32)
    assert np.abs(staged.astype(np.float32).reshape(M, _PN) - want).max() < 5e-2


@requires_sm90
@requires_cuda
@pytest.mark.parametrize("tr", ["cp", "tma"])
@pytest.mark.parametrize("M", [128, 256])
def test_register_double_buffer_matches_single_buffer_bit_for_bit(monkeypatch, tr, M):
    """The smem→register double-buffer (``STAGE=d2/<tr>/p2``) is a PURE perf transform over the
    single-buffer staged kernel (``d2/<tr>``): same loads, same mmas, only ldmatrix-prefetched
    onto alternate fragment slots — so the output is **bit-identical**, and the source actually
    ping-pongs (``_a0_s0``/``_a0_s1`` fragments the single-buffer kernel does not emit)."""
    if tr == "tma" and not _supports_tma():
        pytest.skip("TMA needs sm_90+ (Hopper / Blackwell)")
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    rng = np.random.default_rng(0)
    a = (rng.standard_normal((M, _PK)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((_PK, _PN)) * 0.1).astype(np.float16)

    def _go(stage: str) -> tuple[np.ndarray, str]:
        _pin_tile(monkeypatch, _WARP_CODEC)
        monkeypatch.setenv("EMMY_REDUCE", "")  # serial K: the subject is the p2 double-buffer, not the split fork
        monkeypatch.setenv("EMMY_STAGE", stage)
        be = CudaBackend()
        compiled = be.compile(_parity_mma_graph("static", M=M))
        src = compiled.nodes["o"].op.kernel_source
        got = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"])
        return got, src

    single, single_src = _go(f"d2/{tr}")
    double, double_src = _go(f"d2/{tr}/p2")
    assert "_a0_s0" in double_src and "_a0_s1" in double_src, "p2 must declare per-slot ping-pong fragments"
    assert "_s0" not in single_src, "the single-buffer kernel must not slot its fragments"
    np.testing.assert_array_equal(double, single)  # bit-identical: prefetch reordering perturbs nothing


@requires_sm90
@requires_cuda
@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("M", [128, 256])
def test_cp_async_deep_ring_matches_gmem_direct_bit_for_bit(monkeypatch, depth, M):
    """The gmem→smem ring (``STAGE=d<depth>/cp``, depth≥2) prefetches ``depth-1`` K-chunks ahead so
    the cp.async copy overlaps the mma. It is a PURE perf transform: bit-identical to the gmem-direct
    baseline, and the kernel allocates ``depth`` ring slots (``depth`` cp.async ``commit_group``\\ s —
    the prologue primes ``depth-1``, the steady loop commits one prefetch per chunk)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    rng = np.random.default_rng(1)
    a = (rng.standard_normal((M, _PK)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((_PK, _PN)) * 0.1).astype(np.float16)

    def _go(stage: str | None) -> tuple[np.ndarray, str]:
        _pin_tile(monkeypatch, _WARP_CODEC)
        monkeypatch.setenv("EMMY_REDUCE", "")  # serial K: the baseline must not reroute through the restored split-K fork
        # Pin STAGE="" for the baseline — unpinned, the offline prior may legitimately stage.
        monkeypatch.setenv("EMMY_STAGE", stage if stage else "")
        be = CudaBackend()
        compiled = be.compile(_parity_mma_graph("static", M=M))
        src = compiled.nodes["o"].op.kernel_source
        got = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"])
        return got, src

    ring, ring_src = _go(f"d{depth}/cp")
    gmem, _ = _go(None)
    np.testing.assert_array_equal(ring, gmem)  # bit-identical: prefetch perturbs nothing
    # ``depth`` ``emmy_cp_async_commit();`` CALLS in the body (the trailing ``;`` excludes the one
    # ``emmy_cp_async_commit() {`` helper definition in the cp.async prelude).
    assert ring_src.count("emmy_cp_async_commit();") == depth, f"a depth-{depth} ring must issue {depth} cp.async commit groups"


@requires_sm90
@requires_cuda
@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("M", [128, 256])
def test_tma_deep_ring_matches_gmem_direct_bit_for_bit(monkeypatch, depth, M):
    """The TMA gmem→smem ring (``STAGE=d<depth>/tma``, depth≥2) is the same one :func:`staged_kloop`
    the cp.async ring runs — ``depth`` becomes the sole buffering knob across transports. TMA rides a
    **per-slot mbarrier array** (``_mbar[depth]``, each slot's parity toggled per generation
    ``chunk // ring``) instead of a cp.async commit group. A PURE perf transform: bit-identical to the
    gmem-direct baseline, allocating ``depth`` ring slots."""
    if not _supports_tma():
        pytest.skip("TMA needs sm_90+ (Hopper / Blackwell)")
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    rng = np.random.default_rng(2)
    a = (rng.standard_normal((M, _PK)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((_PK, _PN)) * 0.1).astype(np.float16)

    def _go(stage: str | None) -> tuple[np.ndarray, str]:
        _pin_tile(monkeypatch, _WARP_CODEC)
        monkeypatch.setenv("EMMY_REDUCE", "")  # serial K: the baseline must not reroute through the restored split-K fork
        # Pin STAGE="" for the baseline — unpinned, the offline prior may legitimately stage.
        monkeypatch.setenv("EMMY_STAGE", stage if stage else "")
        be = CudaBackend()
        compiled = be.compile(_parity_mma_graph("static", M=M))
        src = compiled.nodes["o"].op.kernel_source
        got = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"])
        return got, src

    ring, ring_src = _go(f"d{depth}/tma")
    gmem, _ = _go(None)
    np.testing.assert_array_equal(ring, gmem)  # bit-identical: the mbarrier-phased prefetch perturbs nothing
    assert f"_mbar[{depth}]" in ring_src, f"a depth-{depth} TMA ring must declare a {depth}-slot mbarrier array"
    assert ring_src.count("mbarrier_init(&_mbar[") == depth, f"each of the {depth} ring slots' mbarriers must be initialized"


def test_tma_staged_slab_is_swizzled(monkeypatch):
    """A TMA-staged mma kernel swizzles its operand slabs (CPU render, forced sm_120): the
    descriptors carry a non-NONE mode picked from each slab's inner span (``_WARP_CODEC``'s
    ``tile_n`` = 2·2·8 = 32 fp16 elems = 64 B rows → B64, same for A's 32-elem K chunk) and every
    staged ``ldmatrix`` read XORs its address to undo the hardware chunk permutation. Swizzle
    relocates smem bytes only — the bit-identity tests above pin the numerics; this pins the mode
    ON (the rebuilt transport shipped NONE-swizzle slabs and the conflict-bound drain cost the
    fp16 squares ~1.3-1.5× vs the pre-rebuild swizzled bar)."""
    _pin_tile(monkeypatch, _WARP_CODEC)
    monkeypatch.setenv("EMMY_STAGE", "d2/tma")
    monkeypatch.setenv("EMMY_REDUCE", "")
    lowered = Pipeline.build(CUDA_PASSES).run(_parity_mma_graph("static", M=256), ctx=Context(compute_capability=(12, 0)))
    op = lowered.nodes["o"].op
    modes = {d.name: d.swizzle for d in op.tma_descriptors}
    assert modes and all(m != "NONE" for m in modes.values()), f"TMA slabs must swizzle: {modes}"
    assert modes["_desc_b"] == "B64", f"B slab (tile_n=32 fp16 = 64 B rows) must pick B64: {modes}"
    # Every staged ldmatrix applies the XOR via the preamble helper (`emmy_swizzle_<mode>(e)` —
    # `e ^ (((e >> 6) & mask) << 3)`), and the helper for the picked mode is defined.
    assert "emmy_swizzle_b64(" in op.kernel_source, "the staged ldmatrix drain must XOR its slab address via the helper"
    assert "int emmy_swizzle_b64(int e)" in op.kernel_source, "the swizzle helper must be emitted in the preamble"


def test_cp_staged_slab_is_swizzled(monkeypatch):
    """A cp.async-staged mma kernel swizzles its operand slabs in SOFTWARE (CPU render, forced
    sm_89): the same modes the TMA path derives (``_WARP_CODEC``: A's 32-elem fp16 K chunk and
    B's 32-elem tile_n rows are both 64 B → B64) are applied as an address XOR on each
    ``cp.async`` fill DESTINATION and undone by the same XOR in the ldmatrix drain — fill and
    drain agree by construction, so the staged-vs-gmem bit-identity tests above pin the
    numerics; this pins the mode ON. The unswizzled cp slab left the drain 4-way (64 B rows) /
    8-way (128 B rows) bank-conflicted — the measured sm_89 residual to cuBLAS (conflict
    replays were 81% of the gate_up fm golden's shared-mem wavefronts; the XOR won −12–17%)."""
    _pin_tile(monkeypatch, _WARP_CODEC)
    monkeypatch.setenv("EMMY_STAGE", "d2/cp")
    monkeypatch.setenv("EMMY_REDUCE", "")
    lowered = Pipeline.build(CUDA_PASSES).run(_parity_mma_graph("static", M=256), ctx=Context(compute_capability=(8, 9)))
    op = lowered.nodes["o"].op
    src = op.kernel_source
    assert not op.tma_descriptors, "sm_89 has no TMA — the fills must be cp.async"
    assert "int emmy_swizzle_b64(int e)" in src, "the swizzle helper must be emitted in the preamble"
    # Producer side: every cp.async fill writes through the XOR'd destination index...
    assert "emmy_cp_async_cg(&_a_smem[emmy_swizzle_b64(" in src, "the A slab fill must XOR its smem destination"
    assert "emmy_cp_async_cg(&_b_smem[emmy_swizzle_b64(" in src, "the B slab fill must XOR its smem destination"
    # ...and the drain reads back through the identical XOR (fill/drain symmetry).
    assert "emmy_ldmatrix" in src and src.count("emmy_swizzle_b64(") >= 4, "the ldmatrix drains must apply the matching XOR"


@requires_sm90
@requires_cuda
def test_bf16_operands_stage_via_cp_async(monkeypatch):
    """The bf16 MMA atom (``mma_m16n8k16_bf16``) stages through cp.async and stays accurate vs
    torch — the cp.async byte-width fill must handle the 2-byte bf16 operand. (No native numpy
    bf16: feed the bits as uint16 and reinterpret the uint16 output.)"""
    import torch  # noqa: PLC0415

    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_bf16/f2x2/k2")
    monkeypatch.setenv("EMMY_WORK", "w2x2")
    monkeypatch.setenv("EMMY_STAGE", "d2/cp")
    M = 256
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, _PK), dtype=BF16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (_PK, _PN), dtype=BF16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (M, _PN), dtype=BF16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    be = CudaBackend()
    compiled = be.compile(g)
    src = compiled.nodes["o"].op.kernel_source
    assert "cp.async" in src and "mma.sync.aligned.m16n8k16" in src, "bf16 operands must stage on the mma tier"
    assert "cp.async.bulk.tensor" not in src, "cp.async transport must not emit TMA"
    torch.manual_seed(0)
    qa = (torch.randn(M, _PK) * 0.1).to(torch.bfloat16)
    qb = (torch.randn(_PK, _PN) * 0.1).to(torch.bfloat16)
    data = {"a": qa.view(torch.uint16).numpy(), "b": qb.view(torch.uint16).numpy()}
    got_bits = np.asarray(be.run(compiled, input_data=data)[0].outputs["o"]).astype(np.uint16)
    got = torch.from_numpy(got_bits).view(torch.bfloat16).float().numpy().reshape(M, _PN)
    want = (qa.float() @ qb.float()).numpy()
    diff = float(np.abs(got - want).max())
    assert diff < 1e-1, f"bf16 staged mma mismatch (max abs err {diff})"


# --- split-K finalizes on the warp tier --------------------------------------
# MMA split-K rides the structural ``Fold(axis=ksplit, step=[Contraction(k_axis=kslice)])`` fork
# (``_schedule._splitk_option``): the inner ``Contraction`` factorizes to mma exactly like a non-split
# matmul. Deferred (``g2k``): ``030_split_reduce`` retargets each partition's C-fragment into a
# ``ws[ksplit, M, N]`` workspace summed by a sibling additive finalize kernel — NO codegen
# ``atomicAdd``. Atomic (``g2a``): ONE kernel — each partition's C-fragment ``atomicAdd``\\ s into the
# per-launch zero-init'd output (``RegStore.atomic`` → the packed ``__half2`` red, ``zero_outputs``);
# scalar-tier atomic split-K is covered by ``test_reduce_coverage``'s cross-CTA matrix.
def _splitk_mma_graph(m: int, k: int, n: int) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m, n), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


@requires_sm90
@requires_cuda
@pytest.mark.parametrize("finalize", ["deferred", "atomic"])
def test_mma_splitk_finalize(monkeypatch, finalize):
    """fp16 MMA split-K through the structural fork (warp ``TILE`` atom + ``REDUCE=g2k``/``g2a``).
    Deferred (``g2k``) sums each partition's C-fragment through a workspace + additive finalize kernel
    on the tensor-core tier — mma present, NO ``atomicAdd``. Atomic (``g2a``) is ONE kernel: each
    partition's C-fragment ``atomicAdd``\\ s into the zero-init'd output via ``RegStore.atomic`` (the
    packed ``__half2`` red — no ``__partial`` workspace, no finalize node). Both are accurate (the
    atomic arm adds one f16 rounding per partition — inside the shared 2e-2 tolerance)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16/f2x2/k2")
    monkeypatch.setenv("EMMY_WORK", "w2x2")
    monkeypatch.setenv("EMMY_REDUCE", "g2k" if finalize == "deferred" else "g2a")
    m, k, n = 128, 512, 128
    be = CudaBackend()
    rng = np.random.default_rng(4)
    a = (rng.standard_normal((m, k)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.1).astype(np.float16)
    compiled = be.compile(_splitk_mma_graph(m, k, n))
    src = "\n".join(node.op.kernel_source for node in compiled.nodes.values() if getattr(node.op, "kernel_source", None))
    assert "mma.sync.aligned.m16n8k16" in src, "must be on the tensor-core tier"
    if finalize == "deferred":
        assert "atomicAdd" not in src, "atomic-free split-K must not emit atomicAdd"
        assert "__partial" in src, "the deferred finalize writes partials to a workspace"
    else:
        assert "atomicAdd(reinterpret_cast<__half2*>" in src, "the atomic finalize rides the packed-pair red"
        assert "__partial" not in src, "the atomic finalize is single-kernel (no workspace)"
        zeroed = [n.op.zero_outputs for n in compiled.nodes.values() if getattr(n.op, "zero_outputs", ())]
        assert zeroed == [("o",)], f"the atomic output must be zero-init'd per launch, got {zeroed}"
    out = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"]).reshape(m, n)
    ref = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=2e-2, atol=2e-2)


@requires_sm90
@requires_cuda
@pytest.mark.parametrize("transport", ["cp", "tma"])
def test_staged_splitk_matches_gmem_direct_bit_for_bit(monkeypatch, transport):
    """Operand staging composes with split-K: the ``STAGE`` resolved against the SLICED inner node
    threads onto the split partial (``030_split_reduce``), whose K-loop stages its slice through the smem
    pipeline. A pure perf transform — the staged split is **bit-identical** to the gmem-direct
    split (same partials, same finalize), and the partial kernel actually stages."""
    if transport == "tma" and not _supports_tma():
        pytest.skip("TMA needs sm_90+ (Hopper / Blackwell)")
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    m, k, n = 128, 512, 128
    rng = np.random.default_rng(4)
    a = (rng.standard_normal((m, k)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.1).astype(np.float16)

    def _go(stage: str | None) -> tuple[np.ndarray, str]:
        monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16/f2x2/k2")
        monkeypatch.setenv("EMMY_WORK", "w2x2")
        monkeypatch.setenv("EMMY_REDUCE", "g2k")
        # The baseline pins STAGE="" (gmem-direct) explicitly — unpinned, the offline prior may
        # legitimately pick a staged row (D_stage_* terms), which is not the baseline this test wants.
        monkeypatch.setenv("EMMY_STAGE", stage if stage else "")
        be = CudaBackend()
        compiled = be.compile(_splitk_mma_graph(m, k, n))
        partial_src = compiled.nodes["o__partial"].op.kernel_source  # the deferred-finalize partial kernel
        got = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"])
        return got, partial_src

    marker = "cp.async" if transport == "cp" else "cp.async.bulk.tensor"
    staged, staged_src = _go(f"d2/{transport}")
    gmem, gmem_src = _go(None)
    assert marker in staged_src and "__shared__" in staged_src, f"the split partial must stage via {transport}"
    assert "cp.async" not in gmem_src, "the gmem-direct split partial must not stage"
    np.testing.assert_array_equal(staged, gmem)  # bit-identical: staging perturbs nothing
    ref = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(staged.astype(np.float32).reshape(m, n), ref, rtol=2e-2, atol=2e-2)


# =========================================================================== #
# Compile-time schedule guards — pins that would silently lower to a wrong / un-launchable
# kernel. Run the TILE pass only (no GPU): the schedule rejects the pin with a clear
# ``ValueError`` instead of corrupting numerics (warp static-K tail) or failing the launch
# (oversized TILE parallel block).
# =========================================================================== #


def _guard_mm_graph(M, N, K, *, dtype=F16) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K), dtype=dtype), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=dtype), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (M, N), dtype=dtype), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _run_tile_pass(graph: Graph):
    return Pipeline.build(TILE_PASSES).run(graph, ctx=Context.from_target((12, 0)))


def test_warp_static_k_indivisible_rejected(monkeypatch) -> None:
    """A WARP pin whose static K is not a multiple of ``atom_k·bk`` is rejected — the warp
    K-loop has no static-K tail masking, so lowering it would silently corrupt the result (the
    error the accuracy gate's mean-error escape clause misses)."""
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16/f1x1")  # K-step 16
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    with pytest.raises(ValueError, match="does not divide the static contraction K=100"):
        _run_tile_pass(_guard_mm_graph(128, 128, 100))


def test_warp_static_k_divisible_ok(monkeypatch) -> None:
    """The same pin on a K that IS a multiple of the K-step lowers without the guard firing."""
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16/f1x1")
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    _run_tile_pass(_guard_mm_graph(128, 128, 128))  # 128 % 16 == 0 — no raise


def test_warp_symbolic_k_not_guarded(monkeypatch) -> None:
    """A symbolic K reaches the masked tier (ceil-div grid + zero-filled partial slab), so the
    static-K guard does not fire even when the hint is not a K-step multiple."""
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16/f1x1/k2")  # K-step 32
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    _run_tile_pass(_guard_mm_graph(64, 128, Dim("seq_len")))  # symbolic K — no raise


def test_tile_block_over_thread_limit_rejected(monkeypatch) -> None:
    """A TILE parallel tile over the 1024-thread/CTA limit is rejected at compile time instead
    of failing the launch with an opaque ``CUDA_ERROR_INVALID_VALUE``."""
    monkeypatch.setenv("EMMY_TILE", "")  # 16384 threads
    monkeypatch.setenv("EMMY_WORK", "t128x128")
    with pytest.raises(ValueError, match="exceeds the 1024-thread/CTA limit"):
        _run_tile_pass(_guard_mm_graph(256, 256, 256, dtype=F32))


def test_tile_block_within_limit_ok(monkeypatch) -> None:
    """A TILE parallel tile within the thread limit lowers without the guard firing."""
    monkeypatch.setenv("EMMY_TILE", "")  # 64 threads
    monkeypatch.setenv("EMMY_WORK", "t8x8")
    _run_tile_pass(_guard_mm_graph(256, 256, 256, dtype=F32))


# =========================================================================== #
# Masked symbolic warp tier — off-hint straddling sizes, every symbolic axis.
# =========================================================================== #
# A matmul whose M (and/or N, K) axis is symbolic reaches the mma.sync warp tier as a MASKED tile:
# the planner ceil-divs the grid, hoists the K-pipeline above a boundary ``Cond`` (clamped slab
# fill), stamps per-element row/col guards onto the ``RegStore``, and zero-fills the partial final
# K slab past a symbolic reduce extent. One cached kernel serves every runtime size. The point is
# off-hint / straddling sizes (1, 31, 130, 700 — NOT tile-divisor multiples), which exercise the
# boundary-guard + clamp + zero-fill interplay the tile-divisor parity sweep cannot reach.
_MASK_WARP = ("mma_m16n8k16_f16/f2x2/k2", "w2x2")
# REDUCE pinned serial for the same reason as the ``transport`` fixture: a g<w>k split
# sibling deploys a partial+finalize pair, and these tests assert on the one ``o`` kernel.
_CP_KNOBS = {"TILE": _MASK_WARP[0], "WORK": _MASK_WARP[1], "STAGE": "d2/cp", "REDUCE": ""}
_TMA_KNOBS = {"TILE": _MASK_WARP[0], "WORK": _MASK_WARP[1], "STAGE": "d2/tma", "REDUCE": ""}


def _symbolic_m_graph(*, K: int = 512, N: int = 1024) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (Dim("seq_len"), K), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (Dim("seq_len"), N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _symbolic_k_graph(*, M: int = 64, N: int = 128) -> Graph:
    """A @ B with the REDUCE axis symbolic — the SDPA P@V shape after the demoted-matmul split."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, Dim("seq_len")), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (Dim("seq_len"), N), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (M, N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _batched_symbolic_mk_graph(*, H: int = 16, N: int = 128) -> Graph:
    """The SDPA P@V split-consumer: a BATCHED matmul (``H`` heads) whose M (query) AND K (key)
    axes are both symbolic ``seq_len`` — ``xna[H, seq, seq] @ xnb[H, seq, N]``."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("xna", (H, Dim("seq_len"), Dim("seq_len")), dtype=F16), node_id="xna")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("xnb", (H, Dim("seq_len"), N), dtype=F16), node_id="xnb")
    g.add_node(op=MatmulOp(), inputs=["xna", "xnb"], output=Tensor("o", (H, Dim("seq_len"), N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["xna", "xnb"], ["o"]
    return g


def _demoted_symbolic_n_graph(K: int = 128) -> Graph:
    """Computed-B-cone matmul (the rotary QK^T shape): an elementwise scale on BOTH operands feeds
    a transposed-``[N, K]`` Linear, so fusion demotes the matmul and ``010_split_demoted``
    materializes the canonical ``xnb[K, N]`` producer. M = N = ``Dim('seq_len')``."""
    s = Dim("seq_len")
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (s, K), dtype=F16), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("sx", (s, K), dtype=F16), node_id="sx")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("w", (s, K), dtype=F16), node_id="w")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("sw", (s, K), dtype=F16), node_id="sw")
    g.add_node(op=ElementwiseOp("multiply"), inputs=["x", "sx"], output=Tensor("xs", (s, K), dtype=F16), node_id="xs")
    g.add_node(op=ElementwiseOp("multiply"), inputs=["w", "sw"], output=Tensor("ws", (s, K), dtype=F16), node_id="ws")
    g.add_node(op=LinearOp(), inputs=["xs", "ws"], output=Tensor("o", (s, s), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["x", "sx", "w", "sw"], ["o"]
    return g


def _pv_softmax_graph(H: int = 16, N: int = 128) -> Graph:
    """Softmax(scores) @ V with the reduce K = ``seq_len`` symbolic (the SDPA P@V shape). Fusion
    demotes the matmul; ``010_split_demoted`` materializes the softmax-prob A cone + symbolic-K gemm."""
    from emmy.compiler.ir.frontend.ir import SoftmaxOp  # noqa: PLC0415

    s = Dim("seq_len")
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("scores", (H, s, s), dtype=F16), node_id="scores")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("v", (H, s, N), dtype=F16), node_id="v")
    g.add_node(op=SoftmaxOp(axis=-1), inputs=["scores"], output=Tensor("probs", (H, s, s), dtype=F16), node_id="probs")
    g.add_node(op=MatmulOp(), inputs=["probs", "v"], output=Tensor("o", (H, s, N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["scores", "v"], ["o"]
    return g


@pytest.mark.parametrize("transport", ["cp", "tma"])
def test_masked_symbolic_m_structure(transport, monkeypatch):
    """End-to-end render (CPU, forced sm_120): the symbolic-M masked warp kernel carries the
    runtime ``seq_len`` arg + the mma.sync pipeline. cp.async stages a clamped A-slab fill with
    per-element row guards on the fragment store; TMA takes a ``CUtensorMap`` descriptor param and
    stages the A operand with ``cp.async.bulk.tensor`` (TMA zero-fills the masked overhang)."""
    knobs = _CP_KNOBS if transport == "cp" else _TMA_KNOBS
    for k, v in knobs.items():
        monkeypatch.setenv(f"EMMY_{k}", v)
    lowered = Pipeline.build(CUDA_PASSES).run(_symbolic_m_graph(), ctx=Context(compute_capability=(12, 0)))
    kop = lowered.nodes["o"].op
    assert mma_atom(kop.knobs) == "mma_m16n8k16_f16_f32"
    src = kop.kernel_source
    assert "int seq_len" in src, "runtime extent must be a kernel arg"
    assert "mma.sync.aligned.m16n8k16" in src
    if transport == "cp":
        assert kop.knobs.get("S_ext_n_symbolic_axis"), "symbolic-M warp row must carry a symbolic axis"
        assert "ldmatrix" in src
        # Clamp on the hoisted cooperative A fill: bound by the runtime extent, fall back to last row.
        assert "< seq_len) ?" in src and "seq_len - 1" in src, "A-slab fill must clamp to the runtime extent"
        # Per-element row guards from the RegStore (both fragment row blocks).
        assert "+ _g < (seq_len)" in src and "+ _g + 8 < (seq_len)" in src, "fragment store must row-guard against seq_len"
    else:
        # STAGE is stamped per-node (axis-keyed ``STAGE@<k_axis>``), not bare — find this node's.
        stage = next((v for k, v in kop.knobs.items() if k.split("@")[0] == "STAGE"), "")
        assert stage.endswith("/tma"), f"symbolic-M with static innermost dim must stage via TMA: {stage!r}"
        assert "cp.async.bulk.tensor" in src, "A operand must stage via TMA"
        assert "CUtensorMap" in src, "kernel must take the TMA descriptor param"


def test_batched_symbolic_mk_reaches_warp(monkeypatch):
    """The batched masked-M + masked-K P@V consumer must reach the mma.sync tier (the
    ``classify_matmul_operands`` batch-aware B test), not stay a LoopOp."""
    for k, v in _CP_KNOBS.items():
        monkeypatch.setenv(f"EMMY_{k}", v)
    lowered = Pipeline.build(CUDA_PASSES).run(_batched_symbolic_mk_graph(), ctx=Context(compute_capability=(12, 0)))
    kop = lowered.nodes["o"].op
    assert mma_atom(kop.knobs) == "mma_m16n8k16_f16_f32", "batched symbolic M+K matmul must reach the warp tier"
    src = kop.kernel_source
    assert "mma.sync.aligned.m16n8k16" in src and "ldmatrix" in src
    assert "int seq_len" in src, "runtime extent must be a kernel arg"


def test_transposed_b_symbolic_k_zero_fills(monkeypatch):
    """A warp-tier A @ Bᵀ with symbolic K (transposed-B, K contiguous) emits the (n,k)-swapped
    K-zero-fill helper — K is summed by the mma, so the straddling final K tile must read +0.0
    past ``seq_len``, not a duplicate. (Accuracy at straddling sizes: ``symbolic_k_trans`` below.)"""
    _pin_tile(monkeypatch, _MASK_WARP)
    lowered = Pipeline.build(CUDA_PASSES).run(_mma_symbolic_k_graph(64, 32, trans=True), ctx=Context(compute_capability=(12, 0)))
    kop = lowered.nodes["c"].op
    src = kop.kernel_source
    assert "mma.sync.aligned.m16n8k16" in src, "transposed-B symbolic-K must reach the warp tier"
    assert "emmy_mma_load_b_gmem_trans_kzero" in src, "transposed-B symbolic-K must zero-fill via the (n,k)-swapped trans helper"


# (label, env, seqs, make). ``make(seq)`` builds (graph, feed, want) for one off-hint runtime
# size; the driver compiles once per case and runs at each straddling size. ``env`` is the full
# ``EMMY_*`` pin set (some cases route to the scalar tier with no WARP pin).


def _make_symbolic_m(seq):
    g = _symbolic_m_graph()
    rng = np.random.default_rng(0)
    b = (rng.standard_normal((512, 1024)) * 0.1).astype(np.float16)
    a = (rng.standard_normal((seq, 512)) * 0.1).astype(np.float16)
    return g, {"a": a, "b": b}, a.astype(np.float32) @ b.astype(np.float32)


def _make_symbolic_mn(seq):
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("q", (Dim("seq_len"), 128), dtype=F16), node_id="q")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("kT", (128, Dim("seq_len")), dtype=F16), node_id="kT")
    g.add_node(op=MatmulOp(), inputs=["q", "kT"], output=Tensor("o", (Dim("seq_len"), Dim("seq_len")), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["q", "kT"], ["o"]
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    kt = (rng.standard_normal((128, seq)) * 0.1).astype(np.float16)
    return g, {"q": q, "kT": kt}, q.astype(np.float32) @ kt.astype(np.float32)


def _make_symbolic_m_residual(seq):
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (Dim("seq_len"), 512), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (512, 1024), dtype=F16), node_id="b")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("r", (Dim("seq_len"), 1024), dtype=F16), node_id="r")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("mm", (Dim("seq_len"), 1024), dtype=F16), node_id="mm")
    g.add_node(op=ElementwiseOp("add"), inputs=["mm", "r"], output=Tensor("o", (Dim("seq_len"), 1024), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b", "r"], ["o"]
    rng = np.random.default_rng(1)
    a = (rng.standard_normal((seq, 512)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((512, 1024)) * 0.1).astype(np.float16)
    r = (rng.standard_normal((seq, 1024)) * 0.1).astype(np.float16)
    want = a.astype(np.float32) @ b.astype(np.float32) + r.astype(np.float32)
    return g, {"a": a, "b": b, "r": r}, want


def _make_symbolic_k(seq):
    g = _symbolic_k_graph()
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((64, seq)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    return g, {"a": a, "b": b}, a.astype(np.float32) @ b.astype(np.float32)


def _make_transposed_symbolic_k(seq):
    g = _mma_symbolic_k_graph(64, 32, trans=True)
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((64, seq)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((32, seq)) * 0.1).astype(np.float16)
    return g, {"a": a, "b": b}, a.astype(np.float32) @ b.astype(np.float32).T


def _make_demoted_n(seq):
    g = _demoted_symbolic_n_graph()
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    sx = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    w = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    sw = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    want = (x * sx).astype(np.float32) @ (w * sw).astype(np.float32).T
    return g, {"x": x, "sx": sx, "w": w, "sw": sw}, want


def _make_batched_mk(seq):
    g = _batched_symbolic_mk_graph()
    rng = np.random.default_rng(0)
    xna = (rng.standard_normal((16, seq, seq)) * 0.1).astype(np.float16)
    xnb = (rng.standard_normal((16, seq, 128)) * 0.1).astype(np.float16)
    return g, {"xna": xna, "xnb": xnb}, np.matmul(xna.astype(np.float32), xnb.astype(np.float32))


def _make_pv_softmax(seq):
    g = _pv_softmax_graph()
    rng = np.random.default_rng(seq)
    scores = (rng.standard_normal((16, seq, seq)) * 2).astype(np.float16)
    v = (rng.standard_normal((16, seq, 128)) * 0.1).astype(np.float16)
    sc = scores.astype(np.float32)
    e = np.exp(sc - sc.max(-1, keepdims=True))
    probs = e / e.sum(-1, keepdims=True)
    want = np.matmul(probs, v.astype(np.float32))
    return g, {"scores": scores, "v": v}, want


# Each case: (env, delenv, seqs, make). The demoted / batched cases route to the scalar tier or
# greedy lower (no WARP pin); the rest pin the cp.async (or TMA) warp staging.
_MASKED_CASES = {
    "symbolic_m_cp": (_CP_KNOBS, (), [1, 31, 512, 700], _make_symbolic_m),
    "symbolic_m_tma": (_TMA_KNOBS, (), [1, 31, 512, 700], _make_symbolic_m),
    "symbolic_mn_cp": (_CP_KNOBS, (), [31, 512, 700], _make_symbolic_mn),
    "residual_cp": (_CP_KNOBS, (), [100], _make_symbolic_m_residual),
    "symbolic_k_cp": (_CP_KNOBS, (), [16, 31, 130, 512, 700], _make_symbolic_k),
    # Transposed-B (A @ Bᵀ, K contiguous) symbolic-K: the mma zero-fills the masked-K tail through
    # the (n,k)-swapped trans helper. Gmem-direct (no STAGE), M/N are tile divisors so only K masks.
    "symbolic_k_trans": ({"TILE": _MASK_WARP}, (), [16, 31, 130, 512, 700], _make_transposed_symbolic_k),
    # Routed through the SCALAR tier (no WARP pin): the batched-warp masked-M+K fragment codegen
    # at runtime is a separate gap, so accuracy rides the scalar tier (the structure render
    # reaches the warp tier — see ``test_batched_symbolic_mk_reaches_warp``).
    "batched_mk": ({}, (), [16, 31, 130, 512, 700], _make_batched_mk),
    # The demoted B-cone / softmax-P@V splits run under GREEDY (the multi-kernel producer rejects
    # a global warp pin under ``validate_pins``); ``SPLIT_CONE`` forces the demotion split.
    "demoted_n": ({"SPLIT_CONE": "1"}, (), [31, 130, 512, 700], _make_demoted_n),
    "demoted_pv_tma": ({**_CP_KNOBS, "SPLIT_CONE": "1"}, ("TMA",), [16, 31, 130, 512, 700], _make_pv_softmax),
}
_MASKED_PARAMS = [(label, seq) for label, (_e, _d, seqs, _m) in _MASKED_CASES.items() for seq in seqs]


@requires_sm90
@requires_cuda
@pytest.mark.parametrize("label,seq", _MASKED_PARAMS)
def test_masked_symbolic_accuracy(label, seq, monkeypatch):
    """One compiled symbolic kernel is accurate at runtime sizes below / at / above the 512 hint —
    including the straddling cases (1, 31, 130, 700 are not tile-divisor multiples), which exercise
    the masked-M row guard, the masked-N column store, the zero-filled partial-K slab, the demoted
    B-cone overhang, and the TMA-staged P@V — each fed as a synthetic standalone graph."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    env, delenv, _seqs, make = _MASKED_CASES[label]
    for k, v in env.items():
        monkeypatch.setenv(f"EMMY_{k}", v)
    for k in delenv:
        monkeypatch.delenv(f"EMMY_{k}", raising=False)

    graph, feed, want = make(seq)
    be = CudaBackend()
    compiled = be.compile(graph)
    got = np.asarray(be.run(compiled, input_data=feed)[0].outputs[graph.outputs[0]]).astype(np.float32)
    assert got.shape == want.shape, f"{label}/seq={seq}: shape {got.shape} vs {want.shape}"
    diff = float(np.abs(got - want).max())
    assert diff < 5e-2, f"{label}/seq={seq}: masked symbolic mma mismatch (max abs err {diff})"


# --------------------------------------------------------------------------- #
# RASTER — the CTA launch-order codec (kernel-scoped like WSPEC; grouped stripes for L2 reuse
# of the streamed operand).
# --------------------------------------------------------------------------- #

_RASTER_TILE = ("mma_m16n8k16_f16_f32/f2x4/k2", "w2x2")  # M-tile 64, N-tile 64 — Em = M/64, En = N/64


def _raster_kop(monkeypatch, M: int, raster: str | None = None):
    _pin_tile(monkeypatch, _RASTER_TILE)
    monkeypatch.setenv("EMMY_STAGE", "d2/cp/ring")
    if raster is not None:
        monkeypatch.setenv("EMMY_RASTER", raster)
    g = _mma_matmul_graph("static", M, 2048, 1024, "f16", False)
    return Pipeline.build(CUDA_PASSES).run(g, ctx=Context(compute_capability=(12, 0))).nodes["c"].op


def test_raster_gm_pin_groups_the_launch_order(monkeypatch):
    """``EMMY_RASTER=gm8``: the 2-D contraction grid renders the grouped CTA decode — 8 M
    block-tiles iterate fastest within each launch stripe, so consecutive CTAs share the
    streamed B slab (the 2026-07-12 4090 NCU finding: the flat N-fastest order streamed B from
    DRAM once per M-row, ``A + C + B×2`` measured exactly). ``Em = 1280/64 = 20`` is ragged
    against the group of 8, so the decode carries the ``min``-clamped tail-group size; the
    resolved codec value is stamped on the kernel knobs (honest stamping)."""
    kop = _raster_kop(monkeypatch, 1280, raster="gm8")
    assert kop.knobs.get("RASTER") == "gm8"
    src = kop.kernel_source
    assert "_rsub" in src, "the grouped (m, n) sub-id decode must be emitted"
    assert "_rsz = min(8," in src, "a ragged Em (20 % 8) needs the tail-group clamp"


def test_raster_gn_pin_groups_the_transpose(monkeypatch):
    """``gn<G>`` groups N block-tiles fastest (the A-streamed regime); ``En = 2048/64 = 32``
    divides 4, so the decode takes the clamp-free form."""
    kop = _raster_kop(monkeypatch, 1280, raster="gn4")
    assert kop.knobs.get("RASTER") == "gn4"
    src = kop.kernel_source
    assert "_rsub" in src and "_rsz" not in src


def test_raster_default_is_the_flat_order(monkeypatch):
    """Unpinned, a cold greedy pick takes the conservative option-0 (``""`` — the flat
    N-fastest order, byte-identical to the historical codegen) and stamps the decided-empty
    value; the ``gm8`` sibling is a fork row for the search/goldens to arbitrate, never a
    silent default."""
    kop = _raster_kop(monkeypatch, 1280)
    assert kop.knobs.get("RASTER") == ""
    assert "_rsub" not in kop.kernel_source


def test_raster_fork_offers_both_orders(monkeypatch):
    """The enumeration carries the ``RASTER`` family on every contraction row — the flat ``""``
    and the ``gm8`` sibling — so the search can price them per shape (live-fork capture, no
    GPU). Non-contraction kernels never spell the key."""
    from emmy.compiler.pipeline.search.golden_eval import enumerate_graph  # noqa: PLC0415

    g = _mma_matmul_graph("static", 1280, 2048, 1024, "f16", False)
    rows = enumerate_graph(g, Context.from_target((12, 0)))
    vals = {r.get("RASTER") for r in rows if any(k.split("@")[0] == "TILE" for k in r)}
    assert vals == {"", "gm8"}, f"every contraction row must spell RASTER, flat first: {vals}"


@requires_sm90
def test_raster_ragged_group_matches_torch(monkeypatch):
    """Accuracy through a ragged tail group (``Em = 20``, groups of 8 → 8/8/4): every output
    tile must be produced exactly once under the remapped CTA order."""
    _pin_tile(monkeypatch, _RASTER_TILE)
    monkeypatch.setenv("EMMY_STAGE", "d2/cp/ring")
    monkeypatch.setenv("EMMY_RASTER", "gm8")
    rng = np.random.default_rng(7)
    a = rng.standard_normal((1280, 256), dtype=np.float32).astype(np.float16)
    b = rng.standard_normal((256, 512), dtype=np.float32).astype(np.float16)
    g = _mma_matmul_graph("static", 1280, 512, 256, "f16", False)
    got, src = _compile_run_mma(g, {"a": a, "b": b})
    assert "_rsub" in src, "the grouped decode must be live in the benched kernel"
    want = a.astype(np.float32) @ b.astype(np.float32)
    diff = float(np.abs(got.astype(np.float32) - want).max())
    assert diff < 5e-2, f"grouped-raster matmul mismatch (max abs err {diff})"


@requires_sm90
def test_raster_gn_matches_torch(monkeypatch):
    """The transposed grouping (``gn4``) — same exactly-once contract."""
    _pin_tile(monkeypatch, _RASTER_TILE)
    monkeypatch.setenv("EMMY_STAGE", "d2/cp/ring")
    monkeypatch.setenv("EMMY_RASTER", "gn4")
    rng = np.random.default_rng(11)
    a = rng.standard_normal((1280, 256), dtype=np.float32).astype(np.float16)
    b = rng.standard_normal((256, 512), dtype=np.float32).astype(np.float16)
    g = _mma_matmul_graph("static", 1280, 512, 256, "f16", False)
    got, src = _compile_run_mma(g, {"a": a, "b": b})
    assert "_rsub" in src
    want = a.astype(np.float32) @ b.astype(np.float32)
    diff = float(np.abs(got.astype(np.float32) - want).max())
    assert diff < 5e-2, f"gn-raster matmul mismatch (max abs err {diff})"


def test_raster_symbolic_grid_stays_flat(monkeypatch):
    """A symbolic-M (masked-tile) grid renders through the dynamic decode path, which does not
    carry the swizzle — the enumeration must decide the flat ``""`` only there (offering ``gm8``
    would stamp a launch order the kernel doesn't realize: the silent-degrade family)."""
    from emmy.compiler.pipeline.search.golden_eval import enumerate_graph  # noqa: PLC0415

    g = _mma_matmul_graph("dynamic", 1280, 2048, 1024, "f16", False)
    rows = enumerate_graph(g, Context.from_target((12, 0)))
    vals = {r.get("RASTER") for r in rows if any(k.split("@")[0] == "TILE" for k in r)}
    assert vals == {""}, f"symbolic grids must stay flat: {vals}"


def test_regstore_rewrite_preserves_atomic():
    """``RegStore``'s registered ``_rewrite`` reconstructs the stmt field-by-field — dropping
    ``atomic`` there silently degraded a rewritten (f16acc array-fragment rolled) atomic split-K
    store to racing plain assigns: the partitions clobber instead of accumulate, numerically wrong
    with no loud failure (found on the gemma-4 mlp_down.m32.lin g4a pinned compile)."""
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.kernel.ir import RegStore
    from emmy.compiler.ir.sigma import Sigma

    s = RegStore(dst_buffer="o", dst_index=(Var("m"), Var("n")), frag="_c0_0", shape=(16, 8, 16), atomic=True)
    out = s.rewrite(lambda n: n, Sigma({}))
    assert isinstance(out, RegStore) and out.atomic, "atomic must survive the σ-rewrite reconstruction"


@requires_sm90
@requires_cuda
def test_mma_splitk_atomic_f16acc_staged_rolled(monkeypatch):
    """The f16acc atom's array-fragment (rolled-store) form threads ``RegStore.atomic`` through the
    stmt rewrite: the staged ``d2/tma/ring`` + ``g2a`` pin on a deep-K matmul must emit ``atomicAdd``
    (the pre-fix render emitted racing plain stores here — the accuracy assert below fails ~4x-low
    values loudly if the flag is ever dropped again)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f16/f2x2/k4")
    monkeypatch.setenv("EMMY_WORK", "w1x8")
    monkeypatch.setenv("EMMY_REDUCE", "g2a")
    monkeypatch.setenv("EMMY_STAGE", "d2/tma/ring")
    m, k, n = 32, 1024, 256
    be = CudaBackend()
    rng = np.random.default_rng(7)
    a = (rng.standard_normal((m, k)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.1).astype(np.float16)
    compiled = be.compile(_splitk_mma_graph(m, k, n))
    src = "\n".join(node.op.kernel_source for node in compiled.nodes.values() if getattr(node.op, "kernel_source", None))
    assert "atomicAdd" in src, "the rolled f16acc atomic split-K must emit atomicAdd stores"
    out = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"]).reshape(m, n)
    ref = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=3e-2, atol=3e-2)
