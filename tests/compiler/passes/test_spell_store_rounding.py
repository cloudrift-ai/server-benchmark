"""``loop/lifting/090_spell_store_rounding`` — a narrowing store's rounding becomes a statement.

The rule that used to live in the fusion splicer, where it was reconstructed from ``Op.source``
provenance and therefore disagreed between an in-process compile and one resumed from serialized
IR. It is a value fact now, so fusion preserves it without knowing it exists.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dtype import F16, F32, DataType
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import Accum, Assign, Axis, Load, Loop, LoopOp, Write
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs

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


def _copy_kernel(source: str, output: str) -> LoopOp:
    return LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="value", input=source, index=(Var("a0"),)),
                    Write(output=output, index=(Var("a0"),), value="value"),
                ),
            ),
        ),
    )


def _add_kernel(left: str, right: str, output: str) -> LoopOp:
    return LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="left", input=left, index=(Var("a0"),)),
                    Load(name="right", input=right, index=(Var("a0"),)),
                    Assign(name="sum", op="add", args=("left", "right")),
                    Write(output=output, index=(Var("a0"),), value="sum"),
                ),
            ),
        ),
    )


def _private_reduction_graph(*, public_copy: bool) -> Graph:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (4, 16), F16), node_id="x")
    graph.add_node(InputOp(), [], Tensor("residual", (4,), F16), node_id="residual")
    graph.add_node(
        _reduce_kernel().rename_buffers({"out": "stage"}),
        ["x"],
        Tensor("stage", (4,), F16, transient=True),
        node_id="stage",
    )
    if public_copy:
        graph.add_node(_copy_kernel("stage", "rounded"), ["stage"], Tensor("rounded", (4,), F16), node_id="rounded")
        graph.add_node(
            _add_kernel("rounded", "residual", "out"),
            ["rounded", "residual"],
            Tensor("out", (4,), F16),
            node_id="out",
        )
    else:
        graph.add_node(
            _add_kernel("stage", "residual", "rounded"),
            ["stage", "residual"],
            Tensor("rounded", (4,), F16),
            node_id="rounded",
        )
        graph.add_node(ElementwiseOp("negative"), ["rounded"], Tensor("out", (4,), F16), node_id="out")
    graph.inputs = ["x", "residual"]
    graph.outputs = ["out"]
    return graph


def _run(kernel: LoopOp, out: Tensor, *, input_shape: tuple = (4, 16), consumed: bool = True) -> LoopOp:
    """Compile ``kernel`` writing ``out``. ``consumed`` puts a reader downstream — only then can
    fusion delete the store, which is the whole reason to spell its rounding."""
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", input_shape, F16), node_id="x")
    graph.add_node(kernel, ["x"], out, node_id="out")
    if consumed:
        graph.add_node(ElementwiseOp("negative"), ["out"], replace(out, name="reader"), node_id="reader")
    graph.outputs = ["reader" if consumed else "out"]
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


def test_unconsumed_store_spells_nothing() -> None:
    """Nothing reads it, so no fusion can delete the Write that already rounds.

    Spelling it there would give a bare reduce a one-statement projection it did not have — a
    cuttable PLACE seam, which a cold fork took, splitting an f16 ``sum`` into the reduce writing
    f32 to a workspace plus a second kernel doing only the convert.
    """
    op = _run(_reduce_kernel(), Tensor("out", (4,), F16), consumed=False)
    assert _conversions(op) == []


def test_public_copy_of_private_reduction_keeps_its_rounding_through_placement() -> None:
    """A decomposition's private accumulator reaches its public f16 boundary through a copy."""
    graph = _private_reduction_graph(public_copy=True)

    fused = Pipeline.build(["loop/lifting", "loop/fusion"]).run(graph)
    kernel = next(node.op for node in fused.nodes.values() if isinstance(node.op, LoopOp))
    assert _conversions(kernel) == [F16]

    with pinned_knobs({"PLACE": "cut"}):
        placed = Pipeline.build(TILE_PASSES).run(graph, ctx=Context.from_target((7, 0)))
    tiles = [node.op for node in placed.nodes.values() if isinstance(node.op, TileOp)]
    consumer = next(tile for tile in tiles if "out" in tile.outputs)
    copies = [stmt.dtype for stmt in Body(consumer.op.lower()).iter() if isinstance(stmt, Assign) and stmt.op.name == "copy"]
    assert len(tiles) == 2
    assert copies == [F16]


def test_public_computation_from_private_reduction_stays_full_width() -> None:
    """A private reduction used by public computation is not itself a rounding boundary."""
    fused = Pipeline.build(["loop/lifting", "loop/fusion"]).run(_private_reduction_graph(public_copy=False))
    kernel = next(node.op for node in fused.nodes.values() if isinstance(node.op, LoopOp))
    assert _conversions(kernel) == []
