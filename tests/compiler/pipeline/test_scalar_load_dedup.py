"""Scope hygiene for the register-tile scalar ``Load`` dedup.

``_dedup_loads`` collapses identical scalar ``Load``\\ s in one statement list and rewires the
dropped names to the survivor. The rewrite has to reach *into* nested scopes — an epilogue cell's
output sweep consumes the loads hoisted above it — but ``Stmt.rewrite`` renames a stmt's own
bindings as well as its reads and descends unconditionally, while SSA names bound inside a ``Loop``
body are scoped to that body.

So an inner scope that merely re-uses the *spelling* of a dropped outer name used to be captured:
its binding was renamed to the survivor, redeclaring it inside the loop and rewiring the inner
arithmetic to the outer buffer's value. The two production call sites (the scalar drain, the
register-tile epilogue cell) pass flat-ish bodies, but the epilogue cell can carry nested output
sweeps or conditions, so the capture was reachable in principle.
"""

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _dedup_loads

ZERO = (Literal(0, "int"),)


def test_dedup_loads_does_not_capture_a_rebinding_inner_scope() -> None:
    """The loop re-binds both ``in0`` and ``in1`` from different buffers — its own bindings and its
    arithmetic must survive the outer ``in1`` -> ``in0`` alias untouched."""
    inner = Body(
        (
            Load(name="in0", input="x", index=ZERO),
            Load(name="in1", input="y", index=ZERO),
            Assign(name="v", op="add", args=("in0", "in1")),
        )
    )
    body = [
        Load(name="in0", input="const", index=ZERO),
        Load(name="in1", input="const", index=ZERO),  # duplicate -> dropped, alias in1 -> in0
        Loop(axis=Axis("a", 4), body=inner),
    ]

    kept = _dedup_loads(body)

    assert kept[0] == Load(name="in0", input="const", index=ZERO)
    assert kept[1].body == inner


def test_dedup_loads_still_rewires_an_inner_use_of_the_dropped_name() -> None:
    """The mirror case: the loop only *reads* the dropped name, so the rewrite must descend."""
    body = [
        Load(name="in0", input="const", index=ZERO),
        Load(name="in1", input="const", index=ZERO),
        Loop(axis=Axis("a", 4), body=Body((Assign(name="v", op="add", args=("in0", "in1")),))),
    ]

    kept = _dedup_loads(body)

    assert [s.name for s in kept if isinstance(s, Load)] == ["in0"]
    assert kept[-1].body == Body((Assign(name="v", op="add", args=("in0", "in0")),))
