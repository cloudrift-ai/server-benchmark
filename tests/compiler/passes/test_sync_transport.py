"""Generic sync compute-fill lowering invariants."""

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.pipeline.passes.lowering.kernel._stage import CtaTile, SyncOperand, SyncTransport


def test_compute_fill_suffixes_nested_ssa_for_every_vector_cell() -> None:
    """Replicated computed operands must rename definitions inside nested statistic folds."""

    def value(_k0, _row, col):
        reduce_axis = Axis("r", 4)
        loop = Loop(
            axis=reduce_axis,
            role=AxisRole.PLANAR,
            body=Body(
                (
                    Load(name="value", input="x", index=(col, Var(reduce_axis.name))),
                    Accum(name="acc", value="value", op=ElementwiseImpl("add"), axes=(reduce_axis.name,)),
                )
            ),
        )
        close = Assign(name="out", op=ElementwiseImpl("add"), args=("acc", "acc"))
        return [loop, close], "out"

    operand = SyncOperand(tag="b", shape=(1, 8), value=value)
    transport = SyncTransport(operands=(operand,), slab_dtype="half", cta=CtaTile(Literal(0, "int"), 1), elem_bytes=2)
    fill = Body(tuple(transport.fill(k0=Literal(0, "int"), slot=Literal(0, "int"), k0_cur=Literal(0, "int"))))

    assert {acc.name for acc in fill.accums} == {f"acc__c{i}" for i in range(8)}
    assert {load.names[0] for load in fill.loads if load.input == "x"} == {f"value__c{i}" for i in range(8)}
    assert {assign.name for assign in fill.iter_of_type(Assign)} == {f"out__c{i}" for i in range(8)}
