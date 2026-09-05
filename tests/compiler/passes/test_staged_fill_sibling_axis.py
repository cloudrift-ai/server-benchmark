"""The staged fill's gmem σ binds a residual reference to the operand's SIBLING output axis.

A staged operand slab is CTA-shared across the other output axis (B across the m rows, A across the n
columns), so its gmem address must be sibling-invariant in VALUE — but the sibling var can still appear
SYNTACTICALLY, through a flat-index reshape residue: a merged / reshaped projection weight's row index
arrives as ``((m·1024 + n) / 128 % 8) · 128 + (m·1024 + n) % 128``, where the ``m`` contribution is a
multiple of every modulus and folds away. After the tile split the kernel decodes only the ``m_b`` /
``m_u`` split vars, never the bare axis name, so a fill σ that substitutes only its OWN tile axis + K
emitted the unsplit name — nvcc: ``identifier "a0" is undefined`` (the qwen3-8b v_proj ``d1/smem``
greedy pick on sm_80). The fill and TMA box-origin σ now bind the sibling to its block base
(``m_b · tile_m`` — always in-bounds), under which a value-dead residue evaluates unchanged.

A packed weight's block-scale slab carries the same obligation through a second route. Its cells are
COMPUTED — the fill evaluates the decode cone that combines the e4m3 block codes with the weight's
per-tensor scale — and a placement cut can materialize that scale into its own workspace, indexed by
the consuming kernel's outer free axes. The cone then reads ``ws[m]``, and at M=1 the m side is the
elided unit row the matvec recovers, so the leak spelled ``identifier "_um" is undefined`` (nvfp4
Qwen3-8B's M=1 v_proj on sm_120, which degraded every single-token decode to the 32-row bucket)."""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, F32, U8
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Expr, Literal, Var
from emmy.compiler.ir.schedule import Tile, Work
from emmy.compiler.ir.schedule.packing import PackedKBlockB
from emmy.compiler.ir.stmt import Load
from emmy.compiler.ir.stmt.leaves import Assign
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _packed_operands, _slab_operands, _sync_operands, _tile_base
from emmy.compiler.pipeline.passes.lowering.kernel._stage import CtaTile
from tests.compiler.terms import contraction, projection

K16 = "mma_m16n8k16_f16_f32"


def _lit(n: int) -> Literal:
    return Literal(n, "int")


def _row_residue(sibling: Expr, own: Expr) -> Expr:
    """The flat-index reshape residue: ``((s·1024 + o) / 128 % 8) · 128 + (s·1024 + o) % 128`` —
    value-equal to ``o`` for ``o < 1024`` (the ``s`` term is a multiple of both moduli)."""
    flat = sibling * _lit(1024) + own
    return (flat / _lit(128)) % _lit(8) * _lit(128) + flat % _lit(128)


def _mn(m: int = 32, n: int = 1024, m_name: str = "m"):
    tile = Tile.parse(f"{K16}/f1x4/k8", Work.parse("w1x1"))
    return tile.at(Axis(m_name, Dim(m)), Axis("n", Dim(n))).mn


def _free(exprs) -> set[str]:
    return set().union(*(set(e.free_vars()) for e in exprs))


def _assert_sibling_bound(op, sibling_name: str, block_name: str) -> None:
    idx = op.index(_lit(0))(Var("_row"), Var("_col"))
    assert sibling_name not in _free(idx), f"{op.tag} fill leaks the unsplit sibling var {sibling_name!r}"
    assert block_name in _free(idx), f"{op.tag} fill does not bind the sibling block var {block_name!r}"
    coords = op.coords(_lit(0))
    assert sibling_name not in _free(coords), f"{op.tag} TMA box origin leaks the unsplit sibling var {sibling_name!r}"


def test_slab_operands_bind_the_sibling_axis_to_its_block_base():
    """Both copy-transport operands: a dead sibling residue in the gmem index substitutes to the
    sibling's ``_b`` block base instead of leaking the unsplit axis name."""
    mn = _mn()
    ka = Axis("k", Dim(2048))
    a_index = (Var("m"), _row_residue(Var("n"), Var("k")) * _lit(0) + Var("k"))  # dead n residue on A
    b_index = (_row_residue(Var("m"), Var("n")), Var("k"))  # the merged-weight row residue on B
    a_op, b_op = _slab_operands(
        index_srcs=(a_index, b_index),
        bufs=("x", "w"),
        mn=mn,
        k_axis=ka,
        bk_elems=128,
        base=_tile_base(mn),
    )
    _assert_sibling_bound(b_op, "m", "m_b")
    _assert_sibling_bound(a_op, "n", "n_b")


def test_sync_transport_async_b_binds_the_sibling_axis():
    """The reproducer's exact path — the ``d1/smem`` transport's async (cp.async) B fill over a
    weight whose row index carries the m residue."""
    mn = _mn()
    ka = Axis("k", Dim(2048))
    a = Load(name="in1", input="x", index=(Var("m"), Var("k")), dtype=F16)
    b = Load(name="in0", input="w", index=(_row_residue(Var("m"), Var("n")), Var("k")), dtype=F16)
    c = contraction(ka, a, (b, "acc"))
    cta = CtaTile(linear_tid=Var("_t"), n_threads=32)
    _, _, async_ops, *_ = _sync_operands(c, 128, mn, cta, k_axis=ka)
    b_op = next(op for op in async_ops if op.tag == "b")
    _assert_sibling_bound(b_op, "m", "m_b")


def _packed_matvec():
    """A packed-pair B at the ``_um`` unit row, whose per-tensor scale arrives from a cut workspace.

    ``ws`` stands for the placement cut's workspace: the scale is one value, and the cut indexes
    the buffer by the consuming kernel's outer free axis, so the decode cone reads ``ws[_um]``."""
    ka = Axis("k", Dim(2048))
    a = Load(name="in1", input="x", index=(Var("_um"), Var("k")), dtype=F16)
    scale = Load(name="in0", input="ws", index=(Var("_um"),), dtype=F32)
    codes = Load(name="in3", input="wsb", index=(Var("n"), Var("k") / _lit(16)), dtype=F32)
    factor = Assign(name="fac", op=ElementwiseImpl("multiply"), args=("in0", "in3"))
    bits = Load(name="in2", input="wbits", index=(Var("n"), Var("k") / _lit(2)), dtype=U8)
    pairs = Load(name="in6", input="pairs", index=(Var("in2") * _lit(2) + Var("k") % _lit(2),), dtype=F16)
    value = Assign(name="bv", op=ElementwiseImpl("multiply"), args=("in6", "fac"))
    b = projection(body=(scale, codes, factor, bits, pairs, value), results=("bv",))
    packed = PackedKBlockB(bits=bits, table=pairs, factor="fac", block=16)
    return contraction(ka, a, (b, "acc")), packed, ka


def test_packed_block_scale_fill_binds_the_sibling_axis():
    """The packed weight's COMPUTE-filled block-scale slab: a decode cone reading the per-tensor
    scale out of a cut workspace binds the m coordinate to its block base, not the bare axis."""
    mn = _mn(m=1, m_name="_um")
    c, packed, ka = _packed_matvec()
    cta = CtaTile(linear_tid=Var("_t"), n_threads=128)
    _, filled, *_ = _packed_operands(c, packed, 128, mn, "b128", U8, pad=0, cta=cta, k_axis=ka)
    scale_op = next(op for op in filled if op.tag == "bs")
    stmts, _ = scale_op.value(_lit(0), Var("_row"), Var("_col"))
    reads = [stmt for stmt in stmts if isinstance(stmt, Load) and stmt.input == "ws"]
    assert reads, "the block-scale fill must evaluate the cut workspace read"
    free = _free(reads[0].index)
    assert "_um" not in free, "the block-scale fill leaks the unsplit sibling var '_um'"
    assert "_um_b" in free, "the block-scale fill does not bind the sibling block var '_um_b'"
