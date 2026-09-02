"""Stored Fold-tree transformations."""

from emmy.compiler.dtype import F32
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure.fold import Fold, result_slice
from emmy.compiler.ir.stmt import Assign, Body, Load


def test_result_slice_preserves_a_materialized_operand() -> None:
    source = Load(name="loaded", input="x", index=(Var("k"),))
    projection = Fold.projection(
        operands=(source,),
        body=Body(
            (
                Assign(name="kept", op="copy", args=("loaded",), dtype=F32),
                Assign(name="dropped", op="copy", args=("loaded",), dtype=F32),
            )
        ),
        results=("kept", "dropped"),
    )

    sliced = result_slice(projection, {"kept"})

    assert sliced.operands == (source,)
    assert sliced.body == Body((projection.body[0],))
    assert sliced.defines() == ("kept",)
