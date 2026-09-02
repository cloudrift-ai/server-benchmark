"""The λ-spelled :class:`Fold` (1o) — degenerate folds store ``lift: Lambda`` + the flat
``(init, combine)`` pair (the ``Monoid`` wrapper dissolved at 1r) and
DERIVE the serial step, the ``Accum`` forms, and the ``carrier`` annotation.

The contract these pin: mechanical Loop-to-Fold lifting, direct twisted Fold construction, and
canonical rewriting of lift and monoid state in lockstep."""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Lambda, component_ops, degenerate, merge_stmts
from emmy.compiler.ir.pure.carrier import exp_combine_states
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Const, Load, Loop
from emmy.compiler.ir.stmt.passes import rewrite
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop


def _dissolved_loop(*, axes_stamped: bool = True) -> Loop:
    """The canonical lifted shape — ``acc += x²`` with the lifting pass's ``axes`` stamp."""
    acc = Accum(name="acc0", value="v1", op="add", axes=("k",) if axes_stamped else ())
    body = Body(
        (
            Load(name="in0", input="x", index=(Var("m"), Var("k"))),
            Assign(name="v1", op="multiply", args=("in0", "in0")),
            acc,
        )
    )
    return Loop(axis=Axis("k", 512), body=body)


def test_from_loop_stores_the_canonical_shape_lambda_spelled() -> None:
    loop = _dissolved_loop()
    fold = fold_from_loop(loop)
    assert fold is not None and fold.lift is not None
    # The iteration var, then the residual the body reads: loads stay inline in the lift, so the
    # enclosing row coordinate they index by binds as a trailing param — a term has no free names.
    assert fold.lift.params == ("k", "m")
    assert fold.lift.results == ("v1",)
    assert degenerate(fold.combine) and fold.combine.results == ("acc0",)
    # The gate's promise: the derived loop IS the captured loop — carrier annotation included.
    assert fold.loop == loop
    assert fold.out == "acc0"
    assert fold.axis is not None  # loads inline, no operand edges — the demoted shape


def test_from_loop_stamps_an_unstamped_accumulator() -> None:
    loop = _dissolved_loop(axes_stamped=False)
    fold = fold_from_loop(loop)
    assert fold.loop == _dissolved_loop(axes_stamped=True)


def _view(arity: int = 2) -> Fold:
    chans = tuple(Channel(b=Load(name=f"b{i}_e", input=f"W{i}", index=(Var("k"), Var("n"))), acc=f"acc{i}") for i in range(arity))
    return Fold.contraction(
        k_axis=Axis("k", 256),
        a=Load(name="a_e", input="A", index=(Var("m"), Var("k"))),
        channels=chans,
    )


# --- the twisted derivation (1p) — online softmax stores lift (x, 1) + Monoid(init, combine) ---- #


def _softmax_loop() -> Loop:
    """The canonical online-softmax shape — ``[Load x, *dissolved merge]`` over the (m, l)
    exp-family state, exactly as the Tile twisted rewrite builds it."""
    from emmy.compiler.ir.pure.carrier import exp_merge

    body = Body((Load(name="x0", input="x", index=(Var("m"), Var("k"))), *exp_merge(("m_i", "l_i"), ("x0", 1.0), key="m_i")))
    return Loop(axis=Axis("k", 2048), body=body)


def _softmax_fold() -> Fold:
    names = ("m_i", "l_i")
    other = tuple(f"{name}__o" for name in names)
    return Fold(
        axis=Axis("k", 2048),
        lift=Lambda.closing(
            ("k",), Body((Load(name="x0", input="x", index=(Var("m"), Var("k"))), Const(name="one", value=1.0))), ("x0", "one")
        ),
        init=(ElementwiseImpl("maximum").identity, 0.0),
        combine=Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names),
    )


def test_twisted_fold_stores_the_true_monoid() -> None:
    loop = _softmax_loop()
    fold = _softmax_fold()
    assert fold.lift.results == ("x0", "one")  # ι spelled in the lift — the singleton state, its 1 a def
    # The pivot seeds the max op's finite IDENTITY (−1e30), never −inf: an all-masked carrier
    # slice (a coop strided lane, a split-KV chunk) would rescale ``subtract(−inf, −inf)`` — NaN.
    assert fold.init == (ElementwiseImpl("maximum").identity, 0.0)
    assert not degenerate(fold.combine)
    assert fold.combine.results == ("m_i", "l_i")  # recognition's names thread through
    # The derived serial step (combine at the singleton) reproduces the dissolved merge exactly.
    assert fold.loop == loop
    assert fold.axis is not None and component_ops(fold.combine) is None
    assert component_ops(fold.combine) is None  # twisted — derived structurally, never stored


def test_twisted_identity_lift_merges_complete_states() -> None:
    """An identity lift receives monoid elements, so it uses the stored combine directly."""
    algebra = _softmax_fold()
    loads = (
        Load(name="m_p", input="partial", index=(Var("part"),)),
        Load(name="l_p", input="partial", index=(Var("part"),)),
    )
    fold = Fold(
        axis=Axis("part", 2),
        operands=loads,
        lift=Lambda(params=("part", "m_p", "l_p"), body=Body(), results=("m_p", "l_p")),
        init=algebra.init,
        combine=algebra.combine,
    )
    step = fold.step_stmts()
    expected = merge_stmts(algebra.combine, ("m_p", "l_p"), dtype=None)
    assert [(stmt.name, stmt.value) for stmt in step if isinstance(stmt, Accum)] == [
        (stmt.name, stmt.value) for stmt in expected if isinstance(stmt, Accum)
    ]


def test_twisted_rewrite_regenerates_the_combine_over_renamed_state() -> None:
    fold = _softmax_fold()
    ren = {"m_i": "m2", "l_i": "l2", "x0": "s0"}
    out = rewrite(fold, lambda n: ren.get(n, n), Sigma.IDENTITY, lambda a: a)
    assert out.combine.results == ("m2", "l2")
    assert out.lift.results == ("s0", "one")
    # The regenerated combine still passes the formation verification (the fold constructed).
    assert component_ops(out.combine) is None


def test_rewrite_renames_lift_monoid_and_carrier_in_lockstep() -> None:
    fold = _view(2)
    ren = {"acc0": "r0", "acc1": "r1", "a_e": "av", "b0_e": "b0v", "b1_e": "b1v", "acc0__v": "r0__v", "acc1__v": "r1__v"}
    out = rewrite(fold, lambda n: ren.get(n, n), Sigma.IDENTITY, lambda a: a)
    assert out.combine.results == ("r0", "r1")
    assert out.lift.params == ("k", "b0v", "av", "b1v")
    assert out.lift.results == ("r0__v", "r1__v")
    assert [s.name for s in out.loop.body if isinstance(s, Accum)] == ["r0", "r1"]


def _demoted_edge_loop() -> Loop:
    """The DEMOTED fused edge — ``acc += f(x) * w[k, n]`` over ``k``: a computed pure-MAP cone A
    (the producer) times a gmem ``Load`` B, its loads kept INLINE in the lift (recognition's
    unbindable-contraction route), so the fold derives ``PLANAR``."""
    body = Body(
        (
            Load(name="x_e", input="x", index=(Var("m"), Var("k"))),
            Assign(name="a_e", op="multiply", args=("x_e", "scale")),
            Load(name="b_e", input="w", index=(Var("k"), Var("n"))),
            Assign(name="p", op="multiply", args=("a_e", "b_e")),
            Accum(name="acc", value="p", op="add", axes=("k",)),
        )
    )
    return Loop(axis=Axis("k", 1024), body=body)


def test_demoted_edge_algebra_reads_off_the_stored_params() -> None:
    """The demoted warp option reads the fused edge's ⊕ / ⊗ off ``combine`` and ``lift`` — never
    off a derived step. This pins that the two agree, since that tier is pin-driven and neither the
    off-GPU suite nor ``digest_kernels.py`` reaches it.

    The equivalence is structural: the degenerate arm of the derived step interleaves exactly one
    ``Accum(name=combine.results[i], value=lift.results[i], op=⊕ᵢ)`` per component into the lift
    body, so every field the tier reads has a stored home."""
    fold = fold_from_loop(_demoted_edge_loop())
    assert fold is not None and fold.axis is not None

    derived = list(fold.step_stmts())
    accums = [s for s in derived if isinstance(s, Accum)]
    assert len(accums) == 1

    ops = component_ops(fold.combine)
    assert ops is not None and len(ops) == 1
    assert ops[0].name == accums[0].op.name == "add"  # the ⊕ — off ``combine``, not the Accum
    assert str(fold.lift.results[0]) == accums[0].value  # the folded value — off the lift's result
    assert fold.combine.results[0] == accums[0].name  # the accumulator name — off ``combine``
    # The ⊗ lift stmt and every operand Load have their home in the lift body: dropping the derived
    # step loses nothing the tier parses.
    assert [s for s in derived if isinstance(s, Assign)] == [s for s in fold.lift.body if isinstance(s, Assign)]
    assert [s for s in derived if isinstance(s, Load)] == [s for s in fold.lift.body if isinstance(s, Load)]


def test_demoted_edge_composes_test_reads_stored_structure() -> None:
    """``_composes`` (the "bare statistic reduce" negation) scans the two places a node can be
    STORED — an operand edge, or inline in the lift body — instead of the derived step."""
    fold = fold_from_loop(_demoted_edge_loop())
    assert fold is not None
    stored = any(isinstance(s, Fold) for s in fold.lift.body) or any(isinstance(e, Fold) for e in fold.operands)
    step = any(isinstance(s, Fold) for s in fold.step_stmts()) or any(isinstance(e, Fold) for e in fold.operands)
    assert stored is step is False
