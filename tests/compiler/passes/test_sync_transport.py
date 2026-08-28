"""Generic sync compute-fill lowering invariants."""

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import swizzle_fn, swizzle_xor
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.pipeline.passes.lowering._reduction import Reduction
from emmy.compiler.pipeline.passes.lowering.kernel._stage import CtaTile, SyncOperand, SyncTransport, software_swizzle


def _sum_fold(axis: str, acc: str) -> Fold:
    value = f"{acc}_value"
    return Fold(
        axis=Axis(axis, 4),
        lift=Lambda(
            params=(axis,),
            body=Body((Load(name=value, input="x", index=(Var(axis),)),)),
            results=(value,),
        ),
        init=(0.0,),
        combine=Lambda(
            params=(acc, f"{acc}__o"),
            body=Body((Assign(name=acc, op="add", args=(acc, f"{acc}__o")),)),
            results=(acc,),
        ),
    )


def test_cone_stat_follows_the_first_top_level_reduce_in_lowering_order() -> None:
    """The stat algebra must belong to the first reduce ``Loop`` that the cone prologue lowers.

    Attention can place its softmax statistic directly in a zero-axis projection body, while the
    norm→linear form reaches its statistic through a projection operand. A reduction nested inside
    the first top-level fold must not win merely because a recursive tree walk encounters it.
    """
    nested = _sum_fold("nested", "nested_acc")
    first = Fold(
        axis=Axis("first", 4),
        lift=Lambda(params=("first",), body=Body((nested,)), results=(nested.out,)),
        init=(0.0,),
        combine=Lambda(
            params=("first_acc", "first_acc__o"),
            body=Body((Assign(name="first_acc", op="add", args=("first_acc", "first_acc__o")),)),
            results=("first_acc",),
        ),
    )
    body_prologue = Fold.projection(body=Body((first,)))
    body_cone = Fold.projection(body=Body((Assign(name="cell", op="copy", args=(first.out,)),)), operands=(body_prologue,))
    assert Reduction.of_cone_stat(body_cone).fold is first

    operand_stat = _sum_fold("operand", "operand_acc")
    operand_prologue = Fold.projection(body=Body((Assign(name="scale", op="copy", args=(operand_stat.out,)),)), operands=(operand_stat,))
    operand_cone = Fold.projection(body=Body((Assign(name="cell", op="copy", args=("scale",)),)), operands=(operand_prologue,))
    assert Reduction.of_cone_stat(operand_cone).fold is operand_stat


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


def test_software_swizzle_shifts_by_the_slab_row_not_the_atom() -> None:
    """A software-filled slab's XOR reads its ROW index — which needs the slab's OWN stride once a
    row is wider than one swizzle atom.

    The mode's default element shift is ``log2(atom elems)``; that IS the row only while a row is
    exactly one atom. A 128-elem fp16 row (the flash slabs' head dim) is TWO 128 B atoms, so the
    atom bit stays inside the shifted field, consecutive rows collapse onto a quarter of the chunk
    positions, and the ``ldmatrix`` drain over 16 rows goes multi-way bank-conflicted (measured
    7.8-way at ``D = 128`` on an A100, none at ``D = 64``). A one-atom row — and a non-power-of-two
    one, where no row bit is extractable — keeps the PLAIN spelling, so every slab that already
    drained conflict-free renders byte-identically.
    """
    assert software_swizzle(64, 2) == "B128"  # one 128 B atom: the default shift already is the row
    assert software_swizzle(32, 2) == "B64"  # one 64 B atom
    assert software_swizzle(128, 2) == "B128@7"  # two atoms: shift by the row (2**7 elems), not the atom
    assert software_swizzle(256, 2) == "B128@8"
    assert software_swizzle(96, 2) == "B64"  # not a power of two — no row bit to shift by
    assert software_swizzle(4, 2) == "NONE"  # no atom fits

    assert swizzle_xor("B128") == (6, 0x7)
    assert swizzle_xor("B128@7") == (7, 0x7)  # the override moves the shift, never the row mask
    assert swizzle_xor("NONE") is None
    # One emitted helper per distinct (mask, shift): the plain mode keeps its name, so an
    # unchanged slab's kernel source is unchanged.
    assert swizzle_fn("B128") == "emmy_swizzle_b128"
    assert swizzle_fn("B128@7") == "emmy_swizzle_b128_s7"
