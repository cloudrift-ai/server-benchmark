"""The λ-spelled :class:`Fold` (1o) — degenerate folds store ``lift: Lambda`` + the flat
``(init, combine)`` pair (the ``Monoid`` wrapper dissolved at 1r) and
DERIVE the serial step, the ``Accum`` forms, and the ``carrier`` annotation.

The contract these pin: (a) :meth:`Fold.from_loop` keeps the λ spelling ONLY when the derived
loop reproduces the captured one byte-identically (the construction-time gate) — recognition's
canonical dissolved shapes migrate, a non-reproducible shape returns ``None`` (the raw-loop-IR
escape; the retired ``step`` fallback is gone);
(b) :meth:`Contraction.as_fold` stores λ-spelled and round-trips through
``as_fold`` with a byte-identical derived loop; (c) the rewrite canonicalizer renames
lift / monoid / derived carrier in lockstep."""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, component_ops, degenerate
from emmy.compiler.ir.stmt.passes import rewrite
from emmy.compiler.ir.tile import Channel, Contraction, Fold


def _dissolved_loop(*, axes_stamped: bool = True) -> Loop:
    """The canonical recognized shape — ``acc += x²`` with the lifting pass's ``axes`` stamp."""
    acc = Accum(name="acc0", value="v1", op="add", axes=("k",) if axes_stamped else ())
    body = Body(
        (
            Load(name="in0", input="x", index=(Var("m"), Var("k"))),
            Assign(name="v1", op="multiply", args=("in0", "in0")),
            acc,
        )
    )
    return Loop(axis=Axis("k", 512), body=body, role=AxisRole.PLANAR)


def test_from_loop_stores_the_canonical_shape_lambda_spelled() -> None:
    loop = _dissolved_loop()
    fold = Fold.from_loop(loop)
    assert fold is not None and fold.lift is not None
    assert fold.lift.params == ("k",)  # the iteration var; loads stay inline in the lift
    assert fold.lift.results == ("v1",)
    assert degenerate(fold.combine) and fold.combine.results == ("acc0",)
    # The gate's promise: the derived loop IS the captured loop — carrier annotation included.
    assert fold.loop == loop
    assert fold.out == "acc0"
    assert fold.role is AxisRole.PLANAR  # loads inline, no operand edges — the demoted shape


def test_from_loop_declines_a_non_reproducible_shape() -> None:
    # An un-stamped Accum (axes=()) is not the canonical dissolved shape — the derived Accum
    # carries axes=(axis,), so the byte-identity gate declines and ``from_loop`` returns ``None``:
    # the raw-loop-IR escape replaced the retired step fallback.
    loop = _dissolved_loop(axes_stamped=False)
    assert Fold.from_loop(loop) is None


def _view(arity: int = 2) -> Contraction:
    chans = tuple(Channel(b=Load(name=f"b{i}_e", input=f"W{i}", index=(Var("k"), Var("n"))), acc=f"acc{i}") for i in range(arity))
    return Contraction(
        k_axis=Axis("k", 256),
        a=Load(name="a_e", input="A", index=(Var("m"), Var("k"))),
        channels=chans,
    )


def test_as_fold_is_lambda_spelled_and_loop_byte_identical() -> None:
    """``as_fold`` is the node's DERIVED λ reading — the algebra spelling ``Reduction`` (the
    cross-partition programs) and ``Fold.demoted`` consume. Its derived loop must stay
    byte-identical to the node's own synthesized loop at every arity."""
    for arity in (1, 2):
        view = _view(arity)
        fold = view.as_fold()
        assert fold.lift is not None
        # The derived serial step + operand splice reproduce the node's synthesized product loop
        # BODY exactly (the same triple ``from_loop``'s byte-identity gate compares; the λ reading
        # derives PLANAR by design — the demotion's spelling — so the role annotation differs).
        assert (fold.loop.body, fold.loop.axis, fold.loop.unroll) == (view.loop.body, view.loop.axis, view.loop.unroll)


# --- the twisted derivation (1p) — online softmax stores lift (x, 1) + Monoid(init, combine) ---- #


def _softmax_loop() -> Loop:
    """The recognized online-softmax shape — ``[Load x, *dissolved merge]`` over the (m, l)
    exp-family state, exactly as ``_softmax.try_online_softmax`` builds it."""
    from emmy.compiler.ir.stmt.carrier import exp_merge

    body = Body((Load(name="x0", input="x", index=(Var("m"), Var("k"))), *exp_merge(("m_i", "l_i"), ("x0", 1.0), key="m_i")))
    return Loop(axis=Axis("k", 2048), body=body, role=AxisRole.TWISTED)


def test_twisted_from_loop_stores_the_true_monoid() -> None:
    loop = _softmax_loop()
    fold = Fold.from_loop(loop)
    assert fold is not None and fold.lift is not None
    assert fold.lift.results == ("x0", 1.0)  # ι spelled in the lift — the singleton state
    assert fold.init == (float("-inf"), 0.0)
    assert not degenerate(fold.combine)
    assert fold.combine.results == ("m_i", "l_i")  # recognition's names thread through
    # The derived serial step (combine at the singleton) reproduces the dissolved merge exactly.
    assert fold.loop == loop
    assert fold.role is AxisRole.TWISTED
    assert component_ops(fold.combine) is None  # twisted — derived structurally, never stored


def test_twisted_composed_fold_is_lambda_spelled_with_derived_evaluation() -> None:
    """Flash's kv fold stores the λ spelling (step 7 — the composed step dissolved): the QK
    score is a hoisted operand edge, the PV contraction is SYNTHESIZED into the derived blocked
    evaluation (memoized — one identity per stored fold), and the derived step reproduces the
    retired step material exactly."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.pipeline.passes.lowering.tile._flash import _flash_op

    op, _stores = _flash_op("Q", "K", "V", [1, 2], Dim(16), Dim(16), 8, 8)
    (red,) = op.sources
    assert red.lift is not None
    assert red.role is AxisRole.TWISTED
    assert red.lift.results[1:] == (1.0, "v_e")  # ι: (score, 1, v) — the singleton state
    assert isinstance(red.operands[0], Contraction) and red.operands[0].out == "sacc"  # the hoisted QK edge
    stmts = red.step_stmts()
    assert stmts[0] is red.operands[0]  # the derived head position — ahead of the lift body
    pvs = [s for s in stmts[1:] if isinstance(s, Contraction)]
    assert len(pvs) == 1 and pvs[0].out == "O_i__pv"  # the synthesized PV contraction
    assert pvs[0].channels[0].b is red.operands[1]  # B is the fold's own value operand edge
    assert red.step_stmts()[1:] == stmts[1:] and red.step_stmts()[0] is stmts[0]  # memoized — one identity
    assert red.loop == red.loop  # the derived loop is deterministic from the stored params


def test_twisted_rewrite_regenerates_the_combine_over_renamed_state() -> None:
    fold = Fold.from_loop(_softmax_loop())
    ren = {"m_i": "m2", "l_i": "l2", "x0": "s0"}
    out = rewrite(fold, lambda n: ren.get(n, n), Sigma.IDENTITY, lambda a: a)
    assert out.combine.results == ("m2", "l2")
    assert out.lift.results == ("s0", 1.0)
    # The regenerated combine still passes the formation verification (the fold constructed).
    assert component_ops(out.combine) is None


def test_rewrite_renames_lift_monoid_and_carrier_in_lockstep() -> None:
    fold = _view(2).as_fold()
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
    return Loop(axis=Axis("k", 1024), body=body, role=AxisRole.PLANAR)


def test_demoted_edge_algebra_reads_off_the_stored_params() -> None:
    """The demoted warp option reads the fused edge's ⊕ / ⊗ off ``combine`` and ``lift`` — never
    off a derived step. This pins that the two agree, since that tier is pin-driven and neither the
    off-GPU suite nor ``digest_kernels.py`` reaches it.

    The equivalence is structural: the degenerate arm of the derived step interleaves exactly one
    ``Accum(name=combine.results[i], value=lift.results[i], op=⊕ᵢ)`` per component into the lift
    body, so every field the tier reads has a stored home."""
    fold = Fold.from_loop(_demoted_edge_loop())
    assert fold is not None and fold.role is AxisRole.PLANAR

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
    fold = Fold.from_loop(_demoted_edge_loop())
    assert fold is not None
    stored = any(isinstance(s, (Contraction, Fold)) for s in fold.lift.body) or any(
        isinstance(e, (Contraction, Fold)) for e in fold.operands
    )
    step = any(isinstance(s, (Contraction, Fold)) for s in fold.step_stmts()) or any(
        isinstance(e, (Contraction, Fold)) for e in fold.operands
    )
    assert stored is step is False
