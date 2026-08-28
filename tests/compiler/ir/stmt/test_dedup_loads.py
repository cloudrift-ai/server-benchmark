"""Tests for Load deduplication during body normalization."""

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Assign, Load, Write
from emmy.compiler.ir.stmt.normalize import dedup_loads, normalize_body


def test_normalize_body_dedups_loads_and_rewires_gather_indices() -> None:
    body = Body(
        (
            Loop(
                axis=Axis("a", 4),
                body=(
                    Load(name="idx0", input="indices", index=(Var("a"),)),
                    Load(name="idx1", input="indices", index=(Var("a"),)),
                    Load(name="x0", input="values", index=(Var("idx0"),)),
                    Load(name="x1", input="values", index=(Var("idx1"),)),
                    Assign(name="sum", op="add", args=("x0", "x1")),
                    Write(output="out", index=(Var("a"),), value="sum"),
                ),
            ),
        )
    )

    (loop,) = normalize_body(body, hoist=False)

    assert [stmt for stmt in loop.body if isinstance(stmt, Load)] == [
        Load(name="in0", input="indices", index=(Var("a0"),)),
        Load(name="in1", input="values", index=(Var("in0"),)),
    ]
    assert loop.body[-2] == Assign(name="v0", op="add", args=("in1", "in1"))


ZERO = (Literal(0, "int"),)


def test_dedup_loads_does_not_capture_a_rebinding_inner_scope() -> None:
    """A nested scope re-binding a deduped name binds a DIFFERENT variable — the outer alias must
    stop there, or the loop is handed a redeclaration of the survivor and the wrong arithmetic."""
    inner = Body(
        (
            Load(name="in0", input="x", index=ZERO),
            Load(name="in1", input="y", index=ZERO),
            Assign(name="v", op="add", args=("in0", "in1")),
        )
    )
    body = Body(
        (
            Load(name="in0", input="const", index=ZERO),
            Load(name="in1", input="const", index=ZERO),  # duplicate -> dropped, alias in1 -> in0
            Loop(axis=Axis("a", 4), body=inner),
        )
    )

    out = dedup_loads(body)

    assert out[0] == Load(name="in0", input="const", index=ZERO)
    assert out[1].body == inner


def test_dedup_loads_still_rewires_an_inner_use_of_the_dropped_name() -> None:
    """The mirror case: the loop only *reads* the dropped name, so the alias must reach inside."""
    body = Body(
        (
            Load(name="in0", input="const", index=ZERO),
            Load(name="in1", input="const", index=ZERO),
            Loop(axis=Axis("a", 4), body=Body((Assign(name="v", op="add", args=("in0", "in1")),))),
        )
    )

    out = dedup_loads(body)

    assert [s.name for s in out if isinstance(s, Load)] == ["in0"]
    assert out[-1].body == Body((Assign(name="v", op="add", args=("in0", "in0")),))
