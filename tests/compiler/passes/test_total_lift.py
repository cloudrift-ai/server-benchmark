"""Canonical Loop IR lifts completely into a Fold tree without recognition.

The prologue-placement tests at the end were deleted with the recognition-era suite and are
RESTORED: one of them guards a miscompile that actually shipped (the decode-attention ``v1``/``v3``
defect, where a pure prologue value sank past the statement that reads it and the lowered kernel
referenced an undefined name). Deleting a regression test for a fixed miscompile removes the only
thing that would notice it coming back — and it comes back silently, as a compile or a wrong
number, never as a slow kernel.
"""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure.fold import Channel, Fold, deep_defines
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tensor.ir import ReduceOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline


def _tile(body: Body):
    graph = Graph()
    graph.add_node(LoopOp(body=body), [], Tensor("out", (1,)), node_id="out")
    graph.outputs = ["out"]
    return Pipeline.build(["lowering/tile"], select=["lift"]).run(graph).nodes["out"].op


def _matmul_body(epilogue=(), k_extent: int = 128) -> Body:
    m, n, k = Axis("m", Dim(32)), Axis("n", Dim(64)), Axis("k", Dim(k_extent))
    inner = Body(
        (
            Load(name="xv", input="x", index=(Var("m"), Var("k"))),
            Load(name="wv", input="w", index=(Var("n"), Var("k"))),
            Assign(name="prod", op=ElementwiseImpl("multiply"), args=("xv", "wv")),
            Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
        )
    )
    cell = (Loop(axis=k, body=inner), *epilogue, Write(output="out", index=(Var("m"), Var("n")), value="acc"))
    return Body((Loop(axis=m, body=Body((Loop(axis=n, body=Body(cell)),))),))


def test_matmul_cell_lifts_and_canonicalizes_as_contraction() -> None:
    tile = _tile(_matmul_body())
    assert [axis.extent for axis in tile.place.free] == [Dim(32), Dim(64)]
    node = tile.op
    assert isinstance(node, Fold) and node.axis is not None
    assert node.role is AxisRole.CONTRACTION
    assert isinstance(node.a, Load) and node.a.input == "x"
    assert isinstance(node.b, Load) and node.b.input == "w"
    assert len(tile.stores) == 1 and tile.stores[0].sweep is None


def test_epilogue_stays_in_the_projection_body() -> None:
    epilogue = (
        Load(name="bias", input="b", index=(Var("n"),)),
        Assign(name="outv", op=ElementwiseImpl("add"), args=("acc", "bias")),
    )
    m, n, k = Axis("m", Dim(32)), Axis("n", Dim(64)), Axis("k", Dim(128))
    inner = Body(
        (
            Load(name="xv", input="x", index=(Var("m"), Var("k"))),
            Load(name="wv", input="w", index=(Var("n"), Var("k"))),
            Assign(name="prod", op=ElementwiseImpl("multiply"), args=("xv", "wv")),
            Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
        )
    )
    cell = (Loop(axis=k, body=inner), *epilogue, Write(output="out", index=(Var("m"), Var("n")), value="outv"))
    tile = _tile(Body((Loop(axis=m, body=Body((Loop(axis=n, body=Body(cell)),))),)))
    node = tile.op
    assert isinstance(node, Fold) and node.axis is None and len(node.operands) == 1
    assert node.operands[0].axis is not None
    assert {stmt.input for stmt in node.body if isinstance(stmt, Load)} == {"b"}


def test_non_distributing_lift_stays_planar() -> None:
    m, n, k = Axis("m", Dim(32)), Axis("n", Dim(64)), Axis("k", Dim(128))
    inner = Body(
        (
            Load(name="xv", input="x", index=(Var("m"), Var("k"))),
            Load(name="wv", input="w", index=(Var("n"), Var("k"))),
            Assign(name="prod", op=ElementwiseImpl("maximum"), args=("xv", "wv")),
            Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
        )
    )
    cell = (Loop(axis=k, body=inner), Write(output="out", index=(Var("m"), Var("n")), value="acc"))
    tile = _tile(Body((Loop(axis=m, body=Body((Loop(axis=n, body=Body(cell)),))),)))
    node = tile.op
    assert isinstance(node, Fold) and node.axis is not None and node.role is AxisRole.PLANAR
    assert node.semiring is None


def test_contraction_formation_gates_on_the_semiring() -> None:
    import pytest

    k = Axis("k", Dim(16))
    a = Load(name="av", input="a", index=(Var("m"), Var("k")))
    b = Load(name="bv", input="b", index=(Var("n"), Var("k")))
    with pytest.raises(ValueError, match="semiring"):
        Fold.contraction(k_axis=k, a=a, channels=(Channel(b=b, acc="acc"),), product="maximum")
    node = Fold.contraction(k_axis=k, a=a, channels=(Channel(b=b, acc="acc"),))
    assert tuple(op.name for op in node.semiring) == ("multiply", "add")


def test_total_lift_fires_through_the_pipeline() -> None:
    graph = Graph()
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    graph.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    graph.inputs, graph.outputs = ["x"], ["o"]
    lowered = Pipeline.build(TILE_PASSES).run(graph)

    from emmy.compiler.ir.tile import TileOp

    assert any(isinstance(node.op, TileOp) for node in lowered.nodes.values())


# ===================================================================
# Prologue placement (restored)
# ===================================================================


def test_reduce_feeding_prologue_reaches_the_fold_step() -> None:
    """A pure prologue value the reduce body reads must reach the fold's step; an epilogue-only
    value must not.

    The canonical form binds the prologue ONCE in the projection and lets the fold's lambda read it
    as an enclosing-scope name, rather than sinking the load into the loop — same reachability, one
    load instead of one per step. What matters, and what the miscompile below is about, is only
    that the step can see it and that the epilogue-only value stays out of the step."""
    m, k = Axis("m", Dim(8)), Axis("k", Dim(16))
    cell = (
        Load(name="scale", input="s", index=(Literal(0),)),
        Load(name="bias", input="b", index=(Literal(0),)),
        Loop(
            axis=k,
            body=Body(
                (
                    Load(name="xv", input="x", index=(Var("m"), Var("k"))),
                    Assign(name="sx", op=ElementwiseImpl("multiply"), args=("xv", "scale")),
                    Accum(name="acc", value="sx", op=ElementwiseImpl("add"), axes=("k",)),
                )
            ),
        ),
        Assign(name="outv", op=ElementwiseImpl("add"), args=("acc", "bias")),
        Write(output="out", index=(Var("m"),), value="outv"),
    )
    node = _tile(Body((Loop(axis=m, body=Body(cell)),))).op
    assert isinstance(node, Fold) and node.axis is None
    folds = [child for child in (*node.operands, *node.lift.body) if isinstance(child, Fold)]
    assert len(folds) == 1, "the reduce did not lift to exactly one fold"
    step = folds[0]
    scale = next(stmt.name for stmt in node.body if isinstance(stmt, Load) and stmt.input == "s")
    bias = next(stmt.name for stmt in node.body if isinstance(stmt, Load) and stmt.input == "b")
    reads = {name for stmt in step.lift.body for name in stmt.deps()}
    reads.update(name for stmt in step.lift.body if isinstance(stmt, Load) for name in (stmt.input,))
    assert "x" in reads and scale in reads, "the fold's step cannot see the prologue value it multiplies by"
    assert bias not in reads, "an epilogue-only value must not reach the fold's step"


def test_multi_pass_cell_defines_every_name_before_it_is_read() -> None:
    """The decode-attention ``v1``/``v3`` miscompile: a pure prologue value read by an EARLIER
    reduce pass must stay ahead of it. Sinking it into a later pass's lambda leaves the earlier
    pass reading an undefined name — a wrong kernel, not a slow one, and no numerics assert
    downstream attributes the wrong answer to placement.

    The subject is SDPA, the shape the defect came from: three passes over one cell, where the
    score feeds the maximum, the denominator AND the value product. The assertion is the general
    well-formedness invariant, so it survives however those passes are represented — every name a
    statement reads is defined by an earlier statement at the same or an enclosing level."""
    from emmy.commands.trace import graph_from_code  # noqa: PLC0415
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415
    from emmy.compiler.pipeline import LOOP_PASSES  # noqa: PLC0415

    graph, _, _ = graph_from_code(
        "F.scaled_dot_product_attention("
        "torch.randn(1, 1, 8, 4, dtype=torch.float16), "
        "torch.randn(1, 1, 8, 4, dtype=torch.float16), "
        "torch.randn(1, 1, 8, 4, dtype=torch.float16))"
    )
    graph = Pipeline.build(LOOP_PASSES).run(graph)
    graph = Pipeline.build(["lowering/tile"], select=["lift"]).run(graph)
    tiles = [node.op for node in graph.nodes.values() if isinstance(node.op, TileOp)]
    assert tiles, "SDPA produced no Tile IR"

    def check(body, defined):
        for stmt in body:
            if isinstance(stmt, Loop):
                check(list(stmt.body), set(defined) | {stmt.axis.name})
                defined |= set(deep_defines(stmt))
                continue
            reads = set() if isinstance(stmt, Fold) else set(stmt.deps())
            assert reads <= defined, f"{stmt} reads {sorted(reads - defined)} before definition"
            defined |= set(deep_defines(stmt)) if isinstance(stmt, Fold) else set(stmt.defines())

    for tile in tiles:
        check(list(tile.op.lower()), {axis.name for axis in tile.place.free})
