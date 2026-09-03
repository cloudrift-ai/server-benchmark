"""The hand-spelled term builders spell what the total lift forms from Loop IR — so a fixture built
through ``tests.compiler.terms`` exercises the same term the compiler would schedule."""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
from tests.compiler.terms import contraction, projection, slab


def test_a_contraction_is_the_lifted_matmul_loop() -> None:
    loop = Loop(
        axis=Axis("k", 16),
        body=Body(
            (
                Load(name="a", input="x", index=(Var("m"), Var("k"))),
                Load(name="b", input="w", index=(Var("k"), Var("n"))),
                Assign(name="acc__v", op=ElementwiseImpl("multiply"), args=("a", "b")),
                Accum(name="acc", value="acc__v", op=ElementwiseImpl("add"), axes=("k",)),
            )
        ),
    )
    built = contraction(
        "k", Load(name="a", input="x", index=(Var("m"), Var("k"))), (Load(name="b", input="w", index=(Var("k"), Var("n"))), "acc")
    )
    assert built.canonical() == fold_from_loop(loop).canonical()
    view = built.as_contraction()
    assert (view.axis, view.left, view.right, view.product.name, view.plus.name) == ("k", "m", "n", "multiply", "add")


def test_two_channels_over_one_a_are_one_contraction() -> None:
    a = slab("a", "x", "m", "k")
    fused = contraction(Axis("k", 16), a, (slab("g", "Wg", "k", "n"), "acc_g"), (slab("u", "Wu", "k", "n"), "acc_u"))
    assert fused.operands[0] is a and len(fused.operands) == 3
    assert fused.as_contraction() is not None and fused.combine.results == ("acc_g", "acc_u")
    assert fused.as_contraction().plus.name == "add" and fused.init == (0.0, 0.0)


def test_a_projection_exposes_its_last_definition_or_passes_its_operand_through() -> None:
    stat = contraction("k", slab("a", "x", "m", "k"), (slab("b", "w", "k", "n"), "acc"))
    cell = projection((stat,), (Assign(name="r", op="rsqrt", args=("acc",)), Assign(name="o", op="multiply", args=("r", "r"))))
    assert cell.axis is None and cell.exposes == ("o",) and cell.free_axes == {"m", "n"}
    assert projection((stat,)).exposes == ("acc",)
