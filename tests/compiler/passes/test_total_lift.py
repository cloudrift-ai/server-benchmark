"""Canonical Loop IR lifts completely into a Fold tree without recognition.

The prologue-placement tests at the end were deleted with the recognition-era suite and are
RESTORED: one of them guards a miscompile that actually shipped (the decode-attention ``v1``/``v3``
defect, where a pure prologue value sank past the statement that reads it and the lowered kernel
referenced an undefined name). Deleting a regression test for a fixed miscompile removes the only
thing that would notice it coming back — and it comes back silently, as a compile or a wrong
number, never as a slow kernel.
"""

from __future__ import annotations

import numpy as np

from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tensor.ir import ReduceOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams
from tests.compiler.terms import contraction


def deep_defines(node) -> tuple[str, ...]:
    """Every name a term tree or statement defines — lift bodies and exposed names, operands included."""
    if isinstance(node, Fold):
        inner = (name for stmt in node.lift.body for name in deep_defines(stmt))
        return (*inner, *node.exposes, *(name for edge in node.operands for name in deep_defines(edge)))
    return (*node.defines(), *(name for body in node.nested() for stmt in body for name in deep_defines(stmt)))


def _grid(tile) -> frozenset[str]:
    """The kernel-scope binding: the grid binds the free axes, the term opens its output sweeps."""
    return frozenset(axis.name for axis in tile.place.free)


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
    assert node.as_contraction() is not None
    assert [edge.as_slab().load.input for edge in node.operands] == ["x", "w"]
    assert len(tile.output_specs) == 1 and tile.output_specs[0].sweep is None


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
    assert {stmt.input for stmt in node.lift.body if isinstance(stmt, Load)} == {"b"}


def test_a_product_argument_computed_in_the_step_factors_into_a_cone_operand() -> None:
    """The norm→linear step ``acc += (x[m,k] * r[m]) * w[n,k]`` is a semiring step whose A is
    COMPUTED: formation hoists the ``x * r`` chain into a zero-axis operand so the lift is the
    product alone and the term reads as a contraction, A first, over the cone and the weight slab."""
    m, n, k = Axis("m", Dim(32)), Axis("n", Dim(64)), Axis("k", Dim(128))
    inner = Body(
        (
            Load(name="xv", input="x", index=(Var("m"), Var("k"))),
            Load(name="rv", input="r", index=(Var("m"),)),
            Assign(name="xn", op=ElementwiseImpl("multiply"), args=("xv", "rv")),
            Load(name="wv", input="w", index=(Var("n"), Var("k"))),
            Assign(name="prod", op=ElementwiseImpl("multiply"), args=("xn", "wv")),
            Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
        )
    )
    cell = (Loop(axis=k, body=inner), Write(output="out", index=(Var("m"), Var("n")), value="acc"))
    tile = _tile(Body((Loop(axis=m, body=Body((Loop(axis=n, body=Body(cell)),))),)))
    node = tile.op
    view = node.as_contraction()
    assert view is not None and (tile.axis_of(view.left).extent, tile.axis_of(view.right).extent) == (Dim(32), Dim(64))
    cone, weight = node.operands
    assert cone.axis is None and len(cone.exposes) == 1 and weight.as_slab().load.input == "w"
    assert {edge.as_slab().load.input for edge in cone.operands} == {"x", "r"}
    assert [stmt.op.name for stmt in node.lift.body] == ["multiply"]


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
    assert isinstance(node, Fold) and node.axis is not None and node.axis is not None
    assert node.as_contraction() is None


def test_the_contraction_reading_gates_on_the_semiring() -> None:
    """Formation stores any product; the BILINEAR reading is what a ``(⊗, ⊕)`` pair that is not a
    registered semiring never gets."""
    a = Load(name="av", input="a", index=(Var("m"), Var("k")))
    b = Load(name="bv", input="b", index=(Var("n"), Var("k")))
    assert contraction("k", a, (b, "acc"), product="maximum").as_contraction() is None
    view = contraction("k", a, (b, "acc")).as_contraction()
    assert (view.product.name, view.plus.name) == ("multiply", "add")


def test_total_lift_fires_through_the_pipeline() -> None:
    graph = Graph()
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    graph.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    graph.inputs, graph.outputs = ["x"], ["o"]
    lowered = Pipeline.build(TILE_PASSES).run(graph)

    from emmy.compiler.ir.tile import TileOp

    assert any(isinstance(node.op, TileOp) for node in lowered.nodes.values())


def test_singleton_reduce_over_an_enclosing_value_lifts_as_a_projection() -> None:
    """Decode softmax can hoist its sole value ahead of the extent-one max/sum loop.  The
    canonical Loop IR must collapse that identity reduction before total lift, rather than ask a
    fold lambda to return a name defined only by its enclosing projection."""
    m, k = Axis("m", Dim(4)), Axis("k", Dim(1))
    body = Body(
        (
            Loop(
                axis=m,
                body=Body(
                    (
                        Load(name="xv", input="x", index=(Var("m"),)),
                        Loop(
                            axis=k,
                            body=Body((Accum(name="acc", value="xv", op="maximum", axes=("k",)),)),
                        ),
                        Write(output="out", index=(Var("m"),), value="acc"),
                    )
                ),
            ),
        )
    )

    tile = _tile(body)

    graph = Graph()
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    graph.add_node(op=LoopOp(body=body), inputs=["x"], output=Tensor("out", (4,)), node_id="out")
    graph.inputs, graph.outputs = ["x"], ["out"]
    values = np.array([3.0, -2.0, 7.5, 0.25], dtype=np.float32)
    result = NumpyBackend().run(NumpyBackend().compile(graph), input_data={"x": values})[0].outputs["out"]

    assert tuple(axis.extent for axis in tile.place.free) == (Dim(4),)
    assert isinstance(tile.op, Fold) and tile.op.axis is None
    assert not any(edge.axis is not None for edge in tile.op.operands)
    assert tile.op.lower(_grid(tile), tile.output_specs, tile.axes)[-1].value in deep_defines(tile.op)
    np.testing.assert_array_equal(result, values)


def test_sibling_q_and_kv_regions_total_lift_with_separate_outputs() -> None:
    """A shared row feeding different Q and K/V extents remains one pure TileOp, on ONE output axis."""
    m, q, kv, k = Axis("m", 3), Axis("q", 4), Axis("kv", 2), Axis("k", 5)

    def contraction(axis: Axis, weight: str, acc: str) -> Loop:
        return Loop(
            axis=k,
            body=Body(
                (
                    Load(name=f"x_{acc}", input="x", index=(Var("m"), Var("k"))),
                    Load(name=f"w_{acc}", input=weight, index=(Var(axis.name), Var("k"))),
                    Assign(name=f"p_{acc}", op="multiply", args=(f"x_{acc}", f"w_{acc}")),
                    Accum(name=acc, value=f"p_{acc}", op="add", axes=("k",)),
                )
            ),
        )

    q_region = Loop(
        axis=q,
        body=Body(
            (
                contraction(q, "wq", "q_acc"),
                Write(output="q_out", index=(Var("m"), Var("q")), value="q_acc"),
            )
        ),
    )
    kv_region = Loop(
        axis=kv,
        body=Body(
            (
                contraction(kv, "wk", "k_acc"),
                contraction(kv, "wv", "v_acc"),
                Write(output="k_out", index=(Var("m"), Var("kv")), value="k_acc"),
                Write(output="v_out", index=(Var("m"), Var("kv")), value="v_acc"),
            )
        ),
    )
    body = Body((Loop(axis=m, body=Body((q_region, kv_region))),))

    tile = _tile(body)

    # The root projection has nothing of its own: its operands are the q contraction and the k/v
    # pair normalization merged over their shared row; the writes are boundary specs. Each output
    # loop's axis is a contraction output, so post-init promotes it onto the grid like any
    # contraction sweep, and the kernel-scope program is flat.
    assert tile.op.axis is None and not tile.op.lift.body
    assert [(edge.as_contraction() is not None, len(edge.exposes)) for edge in tile.op.operands] == [(True, 1), (True, 2)]
    # ONE output axis, the widest. A second free axis for the narrower region would make the grid
    # the product of the two output widths and re-enumerate each region over the other's cells.
    assert [axis.extent for axis in tile.place.free] == [Dim(3), Dim(4)]
    assert all(spec.sweep is None for spec in tile.output_specs) and len(tile.output_specs) == 3
    # The k/v region rides the shared axis's first two cells, so its stores carry that bound; q
    # spans the axis and carries none.
    assert {spec.write.output: spec.guard for spec in tile.output_specs} == {"q_out": None, "k_out": ("a1", 2), "v_out": ("a1", 2)}
    # The closed program is ONE nest over the shared axis: both regions are evaluated over it, the
    # narrow one clamp-reading its weights (``% 2``) on the cells past its own extent.
    (n_loop,) = tile.loop_body
    (m_loop,) = n_loop.body
    assert (n_loop.axis.extent, m_loop.axis.extent) == (Dim(4), Dim(3))
    assert {write.output for write in m_loop.body.writes} == {"q_out", "k_out", "v_out"}
    assert any("% 2" in load.index[0].pretty() for load in m_loop.body.iter_of_type(Load))
    seam_scopes = {frozenset(axis.extent for axis in seam.axes) for seam in cuttable_seams(tile)}
    assert seam_scopes == {frozenset((Dim(3), Dim(4)))}


def test_an_output_sweeps_epilogue_lifts_to_a_term_declaring_the_sweep_axis() -> None:
    """The per-cell projection under an output loop is a zero-axis term of the level, evaluated
    over the sweep coordinate it declares and reading the reduce as an operand; the root wrapper
    over it dissolves, the sweep axis (a contraction output) promotes onto the grid, and the
    store follows the term's exposed value."""
    m, n, k = Axis("m", 3), Axis("n", 4), Axis("k", 5)
    contraction = Loop(
        axis=k,
        body=Body(
            (
                Load(name="xv", input="x", index=(Var("m"), Var("k"))),
                Load(name="wv", input="w", index=(Var("n"), Var("k"))),
                Assign(name="p", op="multiply", args=("xv", "wv")),
                Accum(name="acc", value="p", op="add", axes=("k",)),
            )
        ),
    )
    sweep = Loop(
        axis=n,
        body=Body(
            (
                contraction,
                Load(name="b", input="bias", index=(Var("n"),)),
                Assign(name="y", op="add", args=("acc", "b")),
                Write(output="out", index=(Var("m"), Var("n")), value="y"),
            )
        ),
    )
    tile = _tile(Body((Loop(axis=m, body=Body((sweep,))),)))

    epilogue = tile.op
    assert epilogue.axis is None and len(epilogue.exposes) == 1
    assert [edge.as_contraction() is not None for edge in epilogue.operands] == [True]
    assert Dim(4) in {axis.extent for axis in tile.axes if axis.name in epilogue.free_axes}
    assert [axis.extent for axis in tile.place.free] == [Dim(3), Dim(4)]
    (spec,) = tile.output_specs
    assert spec.write.values == epilogue.exposes and spec.sweep is None
    assert [type(stmt).__name__ for stmt in tile.op.lower(_grid(tile), tile.output_specs, tile.axes)] == ["Loop", "Load", "Assign", "Write"]


# ===================================================================
# Prologue placement (restored)
# ===================================================================


def test_reduce_feeding_prologue_reaches_the_fold_step() -> None:
    """A pure prologue value the reduce body reads must reach the fold's step; an epilogue-only
    value must not.

    The lift closes the fold over the prologue value it reads: the load arrives as the fold's own
    slab operand, hoisted once ahead of the loop by ``Fold.lower``, rather than sunk into the step.
    What matters, and what the miscompile below is about, is only that the step can see it and
    that the epilogue-only value stays out of the fold."""
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
    folds = [edge for edge in node.operands if edge.axis is not None]
    assert len(folds) == 1, "the reduce did not lift to exactly one fold"
    step = folds[0]
    buffers = {edge.as_slab().load.input for edge in step.operands if edge.as_slab() is not None}
    assert buffers == {"x", "s"}, "the fold's step cannot see the prologue value it multiplies by"
    assert {stmt.input for stmt in node.lift.body if isinstance(stmt, Load)} == {"b"}, "an epilogue-only value must not reach the fold"


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
            reads = set(stmt.deps())
            assert reads <= defined, f"{stmt} reads {sorted(reads - defined)} before definition"
            defined |= set(stmt.defines())

    for tile in tiles:
        check(list(tile.op.lower(_grid(tile), tile.output_specs, tile.axes)), {axis.name for axis in tile.place.free})
