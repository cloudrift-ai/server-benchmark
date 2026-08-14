"""The staged fill's gmem σ binds a residual reference to the operand's SIBLING output axis.

A staged operand slab is CTA-shared across the other output axis (B across the m rows, A across the n
columns), so its gmem address must be sibling-invariant in VALUE — but the sibling var can still appear
SYNTACTICALLY, through a flat-index reshape residue: a merged / reshaped projection weight's row index
arrives as ``((m·1024 + n) / 128 % 8) · 128 + (m·1024 + n) % 128``, where the ``m`` contribution is a
multiple of every modulus and folds away. After the tile split the kernel decodes only the ``m_b`` /
``m_u`` split vars, never the bare axis name, so a fill σ that substitutes only its OWN tile axis + K
emitted the unsplit name — nvcc: ``identifier "a0" is undefined`` (the qwen3-8b v_proj ``d1/sync``
greedy pick on sm_80). The fill and TMA box-origin σ now bind the sibling to its block base
(``m_b · tile_m`` — always in-bounds), under which a value-dead residue evaluates unchanged."""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Expr, Literal, Var
from emmy.compiler.ir.schedule import TilePlan, Workers
from emmy.compiler.ir.stmt import Load
from emmy.compiler.ir.tile import Channel, Fold
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _slab_operands, _sync_operands, _tile_base
from emmy.compiler.pipeline.passes.lowering.kernel._stage import CtaTile

K16 = "mma_m16n8k16_f16_f32"


def _lit(n: int) -> Literal:
    return Literal(n, "int")


def _row_residue(sibling: Expr, own: Expr) -> Expr:
    """The flat-index reshape residue: ``((s·1024 + o) / 128 % 8) · 128 + (s·1024 + o) % 128`` —
    value-equal to ``o`` for ``o < 1024`` (the ``s`` term is a multiple of both moduli)."""
    flat = sibling * _lit(1024) + own
    return (flat / _lit(128)) % _lit(8) * _lit(128) + flat % _lit(128)


def _mn(m: int = 32, n: int = 1024):
    tile = TilePlan.parse(f"{K16}/f1x4/k8", Workers.parse("w1x1"))
    return tile.at(Axis("m", Dim(m)), Axis("n", Dim(n))).mn


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
    """The reproducer's exact path — the ``d1/sync`` transport's async (cp.async) B fill over a
    weight whose row index carries the m residue."""
    mn = _mn()
    ka = Axis("k", Dim(2048))
    a = Load(name="in1", input="x", index=(Var("m"), Var("k")), dtype=F16)
    b = Load(name="in0", input="w", index=(_row_residue(Var("m"), Var("n")), Var("k")), dtype=F16)
    c = Fold.contraction(k_axis=ka, a=a, channels=(Channel(b=b, acc="acc"),))
    cta = CtaTile(linear_tid=Var("_t"), n_threads=32)
    _, _, async_ops, _ = _sync_operands(c, 128, mn, cta)
    b_op = next(op for op in async_ops if op.tag == "b")
    _assert_sibling_bound(b_op, "m", "m_b")
