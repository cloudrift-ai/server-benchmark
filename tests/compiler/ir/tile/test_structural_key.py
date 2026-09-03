"""Kernel identity — ``TileOp.identity_key()``.

A term has no key of its own: the kernel keys on the Loop IR its term lowers to (``TileOp.loop_body``),
so every invariant below is stated on a ``TileOp`` over the term. What these pin: (a) α-INVARIANCE —
SSA renames, consistent buffer renames and independent-stmt interleavings never move the key;
(b) DISCRIMINATION — structure (ops, extents, arity) and the cross-scope buffer-ALIASING pattern (which
scopes read the same buffer) always move it; (c) the lowered program is memoized on the kernel;
(d) identity is the term alone (name / knobs / schedule excluded), while the deploy identity
(``with_io`` + ``with_knobs``) folds the knob row back in so same-term / different-knobs fork variants
never collide."""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.ir.tile import Placement
from emmy.compiler.ir.tile.ir import TileOp
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
from tests.compiler.terms import contraction, projection, slab

K = 512


def _bare(*, buf: str = "x", acc: str = "acc0", load: str = "in0", v: str = "v1", extent: int = K, op: str = "multiply") -> Fold:
    """A lifted squared-sum reduce — a slab operand under a planar fold."""
    body = Body(
        (
            Load(name=load, input=buf, index=(Var("m"), Var("k"))),
            Assign(name=v, op=op, args=(load, load)),
            Accum(name=acc, value=v, op="add", axes=("k",)),
        )
    )
    return fold_from_loop(Loop(axis=Axis("k", extent), body=body))


def _contraction(*, a_buf: str = "A", b_bufs: tuple[str, ...] = ("W0",), names: str = "") -> Fold:
    channels = tuple((slab(f"b{names}{i}_e", w, "k", "n"), f"acc{names}{i}") for i, w in enumerate(b_bufs))
    return contraction(Axis("k", 256), slab(f"a{names}_e", a_buf, "m", "k"), *channels)


def _wrapped(inner: Fold, *, buf: str, load: str = "in3", v: str = "v5") -> Fold:
    """A projection (zero-axis) fold over ``inner`` whose lift also reads ``buf`` — RMSNorm's
    twice-read-edge shape when ``buf`` is the inner fold's own input."""
    (bound,) = inner.combine.results
    body = (Load(name=load, input=buf, index=(Var("m"), Var("a2"))), Assign(name=v, op="multiply", args=(load, bound)))
    return projection((inner,), body, (v,))


def _tile(term: Fold, *, k: int = K, **fields) -> TileOp:
    """The kernel over ``term``: every free coordinate on the grid at one extent, the reduce axes in
    the table — the extents are the kernel's, the term names them."""
    free = tuple(Axis(name, 8) for name in sorted(term.free_axes))
    return TileOp(op=term, place=Placement(free=free), axes=(*free, Axis("k", k)), **fields)


def _key(term: Fold, *, k: int = K, **flags) -> str:
    """The EXACT kernel's key: ``structural=True`` is the schedule-equivalent reading, which collapses
    what only changes latency (an extent among them)."""
    return _tile(term, k=k).identity_key(structural=False, **flags)


# ---- α-invariance ------------------------------------------------------------------------------ #


def test_ssa_renames_never_move_the_key() -> None:
    assert _key(_bare()) == _key(_bare(acc="zz", load="q9", v="t3"))


def test_buffer_renames_never_move_the_key() -> None:
    # Positional canonicalization: WHICH buffer is read is a graph fact, not kernel identity —
    # what the key holds is the aliasing PATTERN (below), exactly as before.
    assert _key(_bare()) == _key(_bare(buf="y"))
    a = _wrapped(_bare(), buf="x")
    b = _wrapped(_bare(buf="y"), buf="y")  # x→y renamed CONSISTENTLY across both scopes
    assert _key(a) == _key(b)


def test_independent_stmt_interleaving_never_moves_the_key() -> None:
    load = Load(name="in0", input="x", index=(Var("a0"), Var("a1")))
    add = Assign(name="v1", op="add", args=("in0", "in0"))
    mul = Assign(name="v2", op="multiply", args=("in0", "in0"))
    one = projection(body=(load, add, mul), results=("v1", "v2"))
    two = projection(body=(load, mul, add), results=("v1", "v2"))
    assert _key(one) == _key(two)


# ---- discrimination ---------------------------------------------------------------------------- #


def test_structure_always_moves_the_key() -> None:
    base = _key(_bare())
    assert base != _key(_bare(extent=1024), k=1024)  # axis extent
    assert base != _key(_bare(op="add"))  # lift op
    assert base != _key(_contraction(), k=256)  # arity / role
    assert _key(_contraction(), k=256) != _key(_contraction(b_bufs=("W0", "W1")), k=256)  # channels
    inner = _bare()
    assert _key(inner) != _key(_wrapped(inner, buf="z"))  # wrapper depth


def test_buffer_aliasing_always_moves_the_key() -> None:
    # Within one scope: A reading the channel's own buffer (x·x-style sharing) is a different
    # kernel than A reading a second buffer, even though both spell positionally.
    assert _key(_contraction(a_buf="B"), k=256) != _key(_contraction(a_buf="W0"), k=256)
    # Across scopes — the twice-read edge: the wrapper re-reading the inner fold's input is a
    # different kernel than the wrapper reading a third buffer.
    assert _key(_wrapped(_bare(), buf="x")) != _key(_wrapped(_bare(), buf="z"))


# ---- memoization ------------------------------------------------------------------------------- #


def test_the_lowered_program_is_memoized_on_the_kernel() -> None:
    tile = _tile(_contraction(), k=256)
    first = tile.loop_body
    assert tile.loop_body is first, "the closed program must memoize on the immutable kernel"
    assert tile.identity_key() == tile.identity_key()


# ---- the op views ------------------------------------------------------------------------------ #


def test_tileop_content_identity_is_the_term_alone() -> None:
    term = _contraction()
    a = _tile(term, k=256, name="k_a")
    b = _tile(term, k=256, name="k_b", knobs={"TILE": "f4"})
    assert a.identity_key(structural=False) == b.identity_key(structural=False)
    assert TileOp().identity_key() is None  # placeholder
    assert not hasattr(term, "structural_key"), "a term carries no key of its own — the kernel keys on its lowered body"


def test_deploy_identity_folds_the_knobs_back_in() -> None:
    term = _contraction()
    plain, knobbed = _tile(term, k=256, name="k"), _tile(term, k=256, name="k", knobs={"TILE": "f4"})
    assert plain.identity_key(with_io=True, with_knobs=True) is not None
    assert plain.identity_key(with_io=True, with_knobs=True) != knobbed.identity_key(
        with_io=True, with_knobs=True
    )  # fork variants must not collide
    assert plain.identity_key(with_io=True, with_knobs=True) == _tile(_contraction(names="r"), k=256, name="renamed").identity_key(
        with_io=True, with_knobs=True
    )  # α-equal terms share
