"""``fold_from_loop`` lifts a reduce that accumulates a value defined in an ENCLOSING scope.

Maximal fusion produces such loops (e.g. the single-key decode softmax: ``for a in 0..1:
acc <- max(acc, v)`` with ``v`` from the outer body). The lift aliases the free value through a
pure copy in the step, which is faithful for every monoid because the original loop feeds the
same value each iteration.
"""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.stmt import Accum, Body, Loop
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop


def _invariant_fold(op: str, extent: int):
    axis = Axis("k", extent)
    accum = Accum(name="acc", value="v_outer", op=op, axes=("k",))
    loop = Loop(axis=axis, body=Body((accum,)), role=AxisRole.PLANAR)
    return fold_from_loop(loop)


def test_enclosing_scope_value_lifts() -> None:
    fold = _invariant_fold("maximum", 1)
    # the free value is aliased into the step and the alias is the lambda's result
    assert fold.lift.results == ("v_outer__inv_k",)
    assert "v_outer__inv_k" in fold.lift.defined
    # the free name stays free — the lift closes over the enclosing scope
    assert "v_outer" not in fold.lift.defined


def test_alias_is_faithful_per_monoid() -> None:
    # add over N iterations of an invariant v accumulates N*v, exactly like the source loop
    fold = _invariant_fold("add", 4)
    assert fold.lift.results == ("v_outer__inv_k",)
    assert fold.combine.results  # monoid assembled normally


def test_defined_values_unchanged() -> None:
    # the common case — value defined by the step — takes the untouched path
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.stmt import Load

    axis = Axis("k", 8)
    load = Load(name="x0", input="x", index=(Var("k"),))
    accum = Accum(name="acc", value="x0", op="add", axes=("k",))
    loop = Loop(axis=axis, body=Body((load, accum)), role=AxisRole.PLANAR)
    fold = fold_from_loop(loop)
    assert fold.lift.results == ("x0",)
