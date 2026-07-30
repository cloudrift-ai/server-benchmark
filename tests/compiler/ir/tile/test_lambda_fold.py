"""The λ-spelled :class:`Fold` (1o) — degenerate folds store ``lift: Lambda`` + ``Monoid`` and
DERIVE the serial step, the ``Accum`` forms, and the ``carrier`` annotation.

The contract these pin: (a) :meth:`Fold.from_loop` keeps the λ spelling ONLY when the derived
loop reproduces the captured one byte-identically (the construction-time gate) — recognition's
canonical dissolved shapes migrate, twisted / composed / non-canonical shapes keep ``step``;
(b) :meth:`ContractionView.as_fold` stores λ-spelled and round-trips through
``contraction_view`` with a byte-identical derived loop; (c) the rewrite canonicalizer renames
lift / monoid / derived carrier in lockstep."""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import TilePlan
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.ir.stmt.passes import rewrite
from emmy.compiler.ir.tile import Channel, ContractionView, Fold, contraction_view


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
    return Loop(axis=Axis("k", 512), body=body, role=AxisRole.PLANAR, carrier=acc.as_carrier())


def test_from_loop_stores_the_canonical_shape_lambda_spelled() -> None:
    loop = _dissolved_loop()
    fold = Fold.from_loop(loop)
    assert fold.lift is not None and len(fold.step) == 0
    assert fold.lift.params == ("k",)  # the iteration var; loads stay inline in the lift
    assert fold.lift.results == ("v1",)
    assert fold.monoid.degenerate and fold.monoid.combine.results == ("acc0",)
    # The gate's promise: the derived loop IS the captured loop — carrier annotation included.
    assert fold.loop == loop
    assert fold.out == "acc0"
    assert fold.role is AxisRole.PLANAR  # loads inline, no operand edges — the demoted shape


def test_from_loop_keeps_step_spelling_when_the_derivation_diverges() -> None:
    # An un-stamped Accum (axes=()) is not the canonical dissolved shape — the derived Accum
    # carries axes=(axis,), so the byte-identity gate declines and the captured step stands.
    loop = _dissolved_loop(axes_stamped=False)
    fold = Fold.from_loop(loop)
    assert fold.lift is None and len(fold.step) == 3
    assert fold.loop == loop  # reconstruction stays exact either way


def _view(arity: int = 2) -> ContractionView:
    chans = tuple(Channel(b=Load(name=f"b{i}_e", input=f"W{i}", index=(Var("k"), Var("n"))), acc=f"acc{i}") for i in range(arity))
    return ContractionView(
        axes=(Axis("m", 64), Axis("n", 64)),
        k_axis=Axis("k", 256),
        a=Load(name="a_e", input="A", index=(Var("m"), Var("k"))),
        channels=chans,
        tile=TilePlan(),
    )


def test_as_fold_is_lambda_spelled_and_loop_byte_identical() -> None:
    for arity in (1, 2):
        view = _view(arity)
        fold = view.as_fold()
        assert fold.lift is not None and len(fold.step) == 0
        assert fold.role is AxisRole.CONTRACTION
        # The derived serial step + operand splice reproduce the view's synthesized product loop.
        assert fold.loop == view.loop
        # And the bilinear parse reads the lift back — the round-trip that keeps identity fixed.
        rt = contraction_view(fold, view.m_axis, view.n_axis)
        assert rt == view


def test_rewrite_renames_lift_monoid_and_carrier_in_lockstep() -> None:
    fold = _view(2).as_fold()
    ren = {"acc0": "r0", "acc1": "r1", "a_e": "av", "b0_e": "b0v", "b1_e": "b1v", "acc0__v": "r0__v", "acc1__v": "r1__v"}
    out = rewrite(fold, lambda n: ren.get(n, n), Sigma.IDENTITY, lambda a: a)
    assert out.monoid.combine.results == ("r0", "r1")
    assert out.carrier.state.names == ("r0", "r1")  # the derived annotation tracks
    assert out.lift.params == ("k", "b0v", "av", "b1v")
    assert out.lift.results == ("r0__v", "r1__v")
    assert [s.name for s in out.loop.body if isinstance(s, Accum)] == ["r0", "r1"]
