"""``loop/lifting/090_spell_store_rounding`` — a narrowing store's rounding becomes a statement.

The rule that used to live in the fusion splicer, where it was reconstructed from ``Op.source``
provenance and therefore disagreed between an in-process compile and one resumed from serialized
IR. It is a value fact now, so fusion preserves it without knowing it exists.
"""

from __future__ import annotations

import pytest

from emmy.compiler.dtype import F16, F32, DataType
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import Accum, Assign, Axis, Load, Loop, LoopOp, Write
from emmy.compiler.pipeline import Pipeline

_RULE = "090_spell_store_rounding"
A0 = Axis("a0", 4)
K = Axis("k", 16)


def _reduce_kernel() -> LoopOp:
    """``out[a0] = sum_k x[a0, k]`` — the accumulator is unstamped, so it accumulates in f32."""
    return LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Loop(
                        axis=K,
                        body=(
                            Load(name="value", input="x", index=(Var("a0"), Var("k"))),
                            Accum(name="acc", value="value", op="add"),
                        ),
                    ),
                    Write(output="out", index=(Var("a0"),), value="acc"),
                ),
            ),
        ),
    )


def _pointwise_kernel() -> LoopOp:
    """``out[a0] = exp(x[a0])`` — an Assign already carries its input's width."""
    return LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="value", input="x", index=(Var("a0"),)),
                    Assign(name="activated", op="exp", args=("value",)),
                    Write(output="out", index=(Var("a0"),), value="activated"),
                ),
            ),
        ),
    )


def _run(kernel: LoopOp, out: Tensor, *, input_shape: tuple = (4, 16)) -> LoopOp:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", input_shape, F16), node_id="x")
    graph.add_node(kernel, ["x"], out, node_id="out")
    graph.outputs = ["out"]
    return Pipeline.build(["loop/lifting"], select=[_RULE]).run(graph).nodes["out"].op


def _conversions(op: LoopOp) -> list[DataType]:
    return [s.dtype for s in op.body.iter() if isinstance(s, Assign) and s.op.name == "copy" and s.dtype is not None]


def test_narrowing_store_to_a_real_buffer_spells_its_rounding() -> None:
    """f32 accumulator, f16 destination: the store rounds, so the rounding becomes a stmt."""
    op = _run(_reduce_kernel(), Tensor("out", (4,), F16))
    assert _conversions(op) == [F16]
    write = next(s for s in op.body.iter() if isinstance(s, Write))
    converted = next(s for s in op.body.iter() if isinstance(s, Assign) and s.op.name == "copy")
    accum = next(s for s in op.body.iter() if isinstance(s, Accum))
    assert write.value == converted.name  # the store reads the rounded value...
    assert converted.args == (accum.name,)  # ...which converts the accumulator


def test_transient_buffer_spells_nothing() -> None:
    """A buffer the source program never materialized has no rounding to preserve."""
    op = _run(_reduce_kernel(), Tensor("out", (4,), F16, transient=True))
    assert _conversions(op) == []


def test_matching_dtype_spells_nothing() -> None:
    """An f32 destination is already the accumulation width."""
    op = _run(_reduce_kernel(), Tensor("out", (4,), F32))
    assert _conversions(op) == []


def test_non_accumulator_value_spells_nothing() -> None:
    """An ``Assign`` chain carries its inputs' width — narrowing it would INVENT a rounding."""
    op = _run(_pointwise_kernel(), Tensor("out", (4,), F16), input_shape=(4,))
    assert _conversions(op) == []


@pytest.mark.parametrize("dtype", [F16, F32])
def test_rule_is_idempotent(dtype: DataType) -> None:
    """The rewritten store reads an Assign, so a second scan finds nothing to do."""
    once = _run(_reduce_kernel(), Tensor("out", (4,), dtype))
    twice = _run(once, Tensor("out", (4,), dtype))
    assert _conversions(twice) == _conversions(once)
