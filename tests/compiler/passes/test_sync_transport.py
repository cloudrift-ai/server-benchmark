"""Generic sync compute-fill lowering invariants."""

from emmy.compiler.dtype import F8E4M3
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
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


def test_compute_fill_vectorizes_a_permuted_flat_physical_run() -> None:
    """A quotient/remainder coordinate pair may still describe one vector load."""

    def value(k0, row, col):
        k = BinaryExpr("+", k0, col)
        lane = BinaryExpr("%", k, Literal(8, "int"))
        permuted = (
            (lane / Literal(4, "int")) * Literal(4, "int")
            + (lane % Literal(2, "int")) * Literal(2, "int")
            + (lane / Literal(2, "int")) % Literal(2, "int")
        )
        flat = row * Literal(256, "int") + permuted
        raw = Load(
            name="raw",
            input="weight",
            index=(flat / Literal(128, "int"), flat % Literal(128, "int")),
            dtype=F8E4M3,
        )
        scale = Load(name="scale", input="scale", index=(k / Literal(128, "int"), row))
        decoded = Assign(name="decoded", op=ElementwiseImpl("from_f8e4m3"), args=("raw",))
        out = Assign(name="out", op=ElementwiseImpl("multiply"), args=("decoded", "scale"))
        return [raw, scale, decoded, out], "out"

    operand = SyncOperand(tag="b", shape=(1, 8), value=value)
    transport = SyncTransport(operands=(operand,), slab_dtype="half", cta=CtaTile(Literal(0, "int"), 1), elem_bytes=2)
    fill = Body(tuple(transport.fill(k0=Literal(0, "int"), slot=Literal(0, "int"), k0_cur=Literal(0, "int"))))

    weight = [load for load in fill.loads if load.input == "weight"]
    scale = [load for load in fill.loads if load.input == "scale"]
    assert len(weight) == 1
    assert weight[0].names == tuple(f"raw__c{j}" for j in (0, 2, 1, 3, 4, 6, 5, 7))
    assert len(scale) == 1 and scale[0].is_scalar


def test_compute_fill_keeps_a_noncontiguous_physical_run_scalar() -> None:
    def value(_k0, row, col):
        load = Load(name="raw", input="weight", index=(row, col * Literal(2, "int")))
        return [load], "raw"

    operand = SyncOperand(tag="b", shape=(1, 8), value=value)
    transport = SyncTransport(operands=(operand,), slab_dtype="half", cta=CtaTile(Literal(0, "int"), 1), elem_bytes=2)
    fill = Body(tuple(transport.fill(k0=Literal(0, "int"), slot=Literal(0, "int"), k0_cur=Literal(0, "int"))))

    assert len([load for load in fill.loads if load.input == "weight"]) == 8


def test_compute_fill_keeps_collapsed_run_scalar_when_leading_stride_can_be_misaligned() -> None:
    def value(_k0, row, col):
        width = Literal(130, "int")
        load = Load(name="raw", input="weight", index=(row, col / width, col % width))
        return [load], "raw"

    operand = SyncOperand(tag="b", shape=(1, 8), value=value)
    transport = SyncTransport(operands=(operand,), slab_dtype="half", cta=CtaTile(Literal(0, "int"), 1), elem_bytes=2)
    fill = Body(tuple(transport.fill(k0=Literal(0, "int"), slot=Literal(0, "int"), k0_cur=Literal(0, "int"))))

    assert len([load for load in fill.loads if load.input == "weight"]) == 8
