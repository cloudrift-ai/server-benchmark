"""The bottom-up term key — ``Fold.structural_key()`` / ``TileOp.structural_key()``.

What these pin: (a) α-INVARIANCE — SSA renames, consistent buffer renames and independent-stmt
interleavings never move the key; (b) DISCRIMINATION — structure (ops, extents, arity) and the
cross-scope buffer-ALIASING pattern (which scopes read the same buffer) always move it;
(c) BOTTOM-UP reuse — a child's key is computed once and a parent built over the same subtree
object answers from the child's memo; (d) the ``TileOp`` view — identity is the term alone
(name / knobs / schedule excluded), while ``Op.cache_key`` folds the knob dict back in so
same-term / different-knobs fork variants never collide."""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.ir.tile.ir import ProjectionRegion, TileOp
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
from emmy.compiler.structural import Structural


def _bare(*, buf: str = "x", acc: str = "acc0", load: str = "in0", v: str = "v1", extent: int = 512, op: str = "multiply") -> Fold:
    """A lifted squared-sum reduce — loads inline, PLANAR (the demoted shape)."""
    body = Body(
        (
            Load(name=load, input=buf, index=(Var("m"), Var("k"))),
            Assign(name=v, op=op, args=(load, load)),
            Accum(name=acc, value=v, op="add", axes=("k",)),
        )
    )
    fold = fold_from_loop(Loop(axis=Axis("k", extent), body=body))
    assert fold is not None
    return fold


def _contraction(*, a_buf: str = "A", b_bufs: tuple[str, ...] = ("W0",), names: str = "") -> Fold:
    chans = tuple(
        Channel(b=Load(name=f"b{names}{i}_e", input=w, index=(Var("k"), Var("n"))), acc=f"acc{names}{i}") for i, w in enumerate(b_bufs)
    )
    return Fold.contraction(
        k_axis=Axis("k", 256),
        a=Load(name=f"a{names}_e", input=a_buf, index=(Var("m"), Var("k"))),
        channels=chans,
    )


def _wrapped(inner: Fold, *, buf: str, load: str = "in3", v: str = "v5") -> Fold:
    """A projection (zero-axis) fold over ``inner`` whose lift also reads ``buf`` — RMSNorm's
    twice-read-edge shape when ``buf`` is the inner fold's own input."""
    (bound,) = inner.combine.results
    body = Body(
        (
            Load(name=load, input=buf, index=(Var("m"), Var("a2"))),
            Assign(name=v, op="multiply", args=(load, bound)),
        )
    )
    return Fold(axis=None, operands=(inner,), lift=Lambda.closing((bound,), body, (v,)))


# ---- α-invariance ------------------------------------------------------------------------------ #


def test_ssa_renames_never_move_the_key() -> None:
    assert _bare().structural_key() == _bare(acc="zz", load="q9", v="t3").structural_key()


def test_buffer_renames_never_move_the_key() -> None:
    # Positional canonicalization: WHICH buffer is read is a graph fact, not term identity —
    # what the key holds is the aliasing PATTERN (below), exactly as before.
    assert _bare().structural_key() == _bare(buf="y").structural_key()
    a = _wrapped(_bare(), buf="x")
    b = _wrapped(_bare(buf="y"), buf="y")  # x→y renamed CONSISTENTLY across both scopes
    assert a.structural_key() == b.structural_key()


def test_independent_stmt_interleaving_never_moves_the_key() -> None:
    load = Load(name="in0", input="x", index=(Var("a0"), Var("a1")))
    add = Assign(name="v1", op="add", args=("in0", "in0"))
    mul = Assign(name="v2", op="multiply", args=("in0", "in0"))
    one = Fold(axis=None, lift=Lambda.closing((), Body((load, add, mul)), ("v1", "v2")))
    two = Fold(axis=None, lift=Lambda.closing((), Body((load, mul, add)), ("v1", "v2")))
    assert one.structural_key() == two.structural_key()


def test_projection_region_keys_ignore_ssa_and_buffer_spelling() -> None:
    def term(*, value: str, buffer: str, extent: int = 4) -> Fold:
        region = ProjectionRegion(
            axis=Axis("q", extent),
            lift=Lambda.closing(("q",), Body((Load(name=value, input=buffer, index=(Var("m"), Var("q"))),)), (value,)),
        )
        return Fold.projection(body=Body((region,)), results=())

    base = term(value="v0", buffer="x")
    assert base.structural_key() == term(value="renamed", buffer="other").structural_key()
    assert base.structural_key() != term(value="v0", buffer="x", extent=8).structural_key()


def test_twisted_state_renames_never_move_the_key() -> None:
    # The generated exp-family combine's internal temps are namespaced on the state names; the
    # key renames through ``rename_combine``'s regeneration lockstep, so state spelling is free.
    from emmy.compiler.ir.pure.carrier import exp_combine_states

    def softmax(names: tuple[str, str]) -> Fold:
        other = tuple(f"{name}__o" for name in names)
        return Fold(
            axis=Axis("k", 2048),
            lift=Lambda.closing(("k",), Body((Load(name="x0", input="x", index=(Var("m"), Var("k"))),)), ("x0", 1.0)),
            init=(-1e30, 0.0),
            combine=Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names),
        )

    assert softmax(("m_i", "l_i")).structural_key() == softmax(("p", "q")).structural_key()


# ---- discrimination ---------------------------------------------------------------------------- #


def test_structure_always_moves_the_key() -> None:
    base = _bare().structural_key()
    assert base != _bare(extent=1024).structural_key()  # axis extent
    assert base != _bare(op="add").structural_key()  # lift op
    assert base != _contraction().structural_key()  # arity / role
    assert _contraction().structural_key() != _contraction(b_bufs=("W0", "W1")).structural_key()  # channels
    inner = _bare()
    assert inner.structural_key() != _wrapped(inner, buf="z").structural_key()  # wrapper depth


def test_buffer_aliasing_always_moves_the_key() -> None:
    # Within one scope: A reading the channel's own buffer (x·x-style sharing) is a different
    # kernel than A reading a second buffer, even though both spell positionally.
    assert _contraction(a_buf="B").structural_key() != _contraction(a_buf="W0").structural_key()
    # Across scopes — the twice-read edge: the wrapper re-reading the inner fold's input is a
    # different kernel than the wrapper reading a third buffer.
    assert _wrapped(_bare(), buf="x").structural_key() != _wrapped(_bare(), buf="z").structural_key()


# ---- bottom-up reuse --------------------------------------------------------------------------- #


def test_the_key_is_cached_on_the_term() -> None:
    child = _contraction()
    first = child.structural_key()
    assert "_lowered_key" in child.__dict__, "the key must memoize on the immutable term"
    assert child.structural_key() is first


# ---- the op views ------------------------------------------------------------------------------ #


def test_tileop_content_identity_is_the_term_alone() -> None:
    term = _contraction()
    a = TileOp(op=term, name="k_a")
    b = TileOp(op=term, name="k_b", knobs={"TILE": "f4"})
    assert a.identity_key(structural=False) == b.identity_key(structural=False)
    assert TileOp().identity_key() is None  # placeholder
    assert isinstance(term, Structural)


def test_cache_key_folds_the_knobs_back_in() -> None:
    term = _contraction()
    plain, knobbed = TileOp(op=term, name="k"), TileOp(op=term, name="k", knobs={"TILE": "f4"})
    assert plain.identity_key(with_io=True, with_knobs=True) is not None
    assert plain.identity_key(with_io=True, with_knobs=True) != knobbed.identity_key(
        with_io=True, with_knobs=True
    )  # fork variants must not collide
    assert plain.identity_key(with_io=True, with_knobs=True) == TileOp(op=_contraction(names="r"), name="renamed").identity_key(
        with_io=True, with_knobs=True
    )  # α-equal terms share
