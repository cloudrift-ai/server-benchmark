"""Tests for Load deduplication during body normalization."""

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Assign, Load, Write
from emmy.compiler.ir.stmt.normalize import normalize_body


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
