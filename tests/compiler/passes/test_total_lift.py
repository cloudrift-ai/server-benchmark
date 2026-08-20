"""The total lift (``lowering/tile/_lift``): any loop nest → its Fold tree, no algebra dispatch.

Covers the lift's own contract — totality (the worst case is the identity transform), the
in-place typed-fold replacement, prologue routing, root formation's scope rule, and the boundary
store split. What algebra a lifted fold IS (contraction, online softmax, the monoid-producer
composition) is classification, tested with its passes when they land
(``tests/_total_lift_rebuild.py``)."""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.loop.ir import LoopOp
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tensor.ir import ReduceOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.tile._lift import recognized_tile


def _tile(body: Body):
    return recognized_tile(LoopOp(body=body), "out")


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


def test_matmul_cell_lifts_and_classifies_as_contraction():
    tile = _tile(_matmul_body())
    # ``LoopOp`` normalization renames axes; the output-ordered free pair is (m, n) by extent.
    assert [a.extent for a in tile.place.free] == [Dim(32), Dim(64)]
    node = tile.op
    assert isinstance(node, Fold) and node.axis is not None, "the bare reduce IS the root node"
    assert node.role is AxisRole.CONTRACTION, "the tree-level binder rebinds the lifted fold"
    assert isinstance(node.a, Load) and node.a.input == "x"
    assert isinstance(node.b, Load) and node.b.input == "w"
    assert len(tile.stores) == 1 and tile.stores[0].sweep is None


def test_epilogue_stays_in_the_projection_body():
    epi = (
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
    cell = (Loop(axis=k, body=inner), *epi, Write(output="out", index=(Var("m"), Var("n")), value="outv"))
    tile = _tile(Body((Loop(axis=m, body=Body((Loop(axis=n, body=Body(cell)),))),)))
    node = tile.op
    assert isinstance(node, Fold) and node.axis is None and len(node.operands) == 1
    assert node.operands[0].axis is not None
    assert {getattr(s, "input", None) for s in node.body if isinstance(s, Load)} == {"b"}


def test_two_pass_softmax_lifts_both_folds_with_cross_operand_read():
    """The second pass's λ reads the first pass's carried state as a free name — the shape the
    online-softmax pairing will classify."""
    m, k1, k2 = Axis("m", Dim(8)), Axis("k", Dim(16)), Axis("k2", Dim(16))
    pass1 = Loop(
        axis=k1,
        body=Body(
            (
                Load(name="xv", input="x", index=(Var("m"), Var("k"))),
                Accum(name="mx", value="xv", op=ElementwiseImpl("maximum"), axes=("k",)),
            )
        ),
    )
    pass2 = Loop(
        axis=k2,
        body=Body(
            (
                Load(name="xw", input="x", index=(Var("m"), Var("k2"))),
                Assign(name="sh", op=ElementwiseImpl("subtract"), args=("xw", "mx")),
                Assign(name="ex", op=ElementwiseImpl("exp"), args=("sh",)),
                Accum(name="den", value="ex", op=ElementwiseImpl("add"), axes=("k2",)),
            )
        ),
    )
    cell = (pass1, pass2, Write(output="out", index=(Var("m"),), value="den"))
    tile = _tile(Body((Loop(axis=m, body=Body(cell)),)))
    node = tile.op
    assert isinstance(node, Fold) and node.axis is None
    # The pairing consumes the sibling pair into ONE TWISTED fold — the exp/LSE carrier; the
    # normalize projection reads its two carried states.
    assert len(node.operands) == 1
    tw = node.operands[0]
    assert tw.role is AxisRole.TWISTED and len(tw.combine.results) == 2
    assert not any(isinstance(s, Loop) for s in node.body)


def test_paired_twisted_fold_round_trips_the_loop_dialect():
    """The closure property: cut fragments and split partials re-enter recognition as lowered
    loop IR, so whatever spelling ``Fold.loop`` emits must re-lift to the same node."""
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop

    tile = _tile(_softmax_body())
    tw = tile.op.operands[0]
    assert tw.role is AxisRole.TWISTED
    assert fold_from_loop(tw.loop) == tw


def test_expectation_channel_joins_the_pair_as_a_third_component():
    """A sibling additive fold whose value factors as the pair's exp weight × a value cone joins
    the same TWISTED fold as an EXPECTATION component (flash's P·V shape)."""
    m = Axis("m", Dim(8))
    k1, k2, k3 = (Axis(nm, Dim(16)) for nm in ("k", "k2", "k3"))
    pass1 = Loop(
        axis=k1,
        body=Body(
            (
                Load(name="xv", input="x", index=(Var("m"), Var("k"))),
                Accum(name="mx", value="xv", op=ElementwiseImpl("maximum"), axes=("k",)),
            )
        ),
    )
    pass2 = Loop(
        axis=k2,
        body=Body(
            (
                Load(name="xw", input="x", index=(Var("m"), Var("k2"))),
                Assign(name="sh", op=ElementwiseImpl("subtract"), args=("xw", "mx")),
                Assign(name="ex", op=ElementwiseImpl("exp"), args=("sh",)),
                Accum(name="den", value="ex", op=ElementwiseImpl("add"), axes=("k2",)),
            )
        ),
    )
    chan = Loop(
        axis=k3,
        body=Body(
            (
                Load(name="xu", input="x", index=(Var("m"), Var("k3"))),
                Assign(name="sh2", op=ElementwiseImpl("subtract"), args=("xu", "mx")),
                Assign(name="ex2", op=ElementwiseImpl("exp"), args=("sh2",)),
                Load(name="vv", input="v", index=(Var("m"), Var("k3"))),
                Assign(name="pv", op=ElementwiseImpl("multiply"), args=("ex2", "vv")),
                Accum(name="o", value="pv", op=ElementwiseImpl("add"), axes=("k3",)),
            )
        ),
    )
    cell = (
        pass1,
        pass2,
        chan,
        Assign(name="outv", op=ElementwiseImpl("divide"), args=("o", "den")),
        Write(output="out", index=(Var("m"),), value="outv"),
    )
    tile = _tile(Body((Loop(axis=m, body=Body(cell)),)))
    node = tile.op
    assert len(node.operands) == 1
    tw = node.operands[0]
    assert tw.role is AxisRole.TWISTED and len(tw.combine.results) == 3
    assert {s.input for s in tw.lift.body if isinstance(s, Load)} == {"x", "v"}


def _softmax_body() -> Body:
    m, k1, k2 = Axis("m", Dim(8)), Axis("k", Dim(16)), Axis("k2", Dim(16))
    pass1 = Loop(
        axis=k1,
        body=Body(
            (
                Load(name="xv", input="x", index=(Var("m"), Var("k"))),
                Accum(name="mx", value="xv", op=ElementwiseImpl("maximum"), axes=("k",)),
            )
        ),
    )
    pass2 = Loop(
        axis=k2,
        body=Body(
            (
                Load(name="xw", input="x", index=(Var("m"), Var("k2"))),
                Assign(name="sh", op=ElementwiseImpl("subtract"), args=("xw", "mx")),
                Assign(name="ex", op=ElementwiseImpl("exp"), args=("sh",)),
                Accum(name="den", value="ex", op=ElementwiseImpl("add"), axes=("k2",)),
            )
        ),
    )
    cell = (pass1, pass2, Write(output="out", index=(Var("m"),), value="den"))
    return Body((Loop(axis=m, body=Body(cell)),))


def test_unliftable_reduce_stays_a_verbatim_raw_loop():
    """A typed accumulator is precision the λ spelling does not carry — the loop keeps its raw
    spelling IN PLACE, and the projection wraps it as the escape subtree."""
    from emmy.compiler.dtype import F32

    m, k = Axis("m", Dim(8)), Axis("k", Dim(16))
    reduce = Loop(
        axis=k,
        body=Body(
            (
                Load(name="xv", input="x", index=(Var("m"), Var("k"))),
                Accum(name="acc", value="xv", op=ElementwiseImpl("add"), axes=("k",), dtype=F32),
            )
        ),
    )
    cell = (reduce, Write(output="out", index=(Var("m"),), value="acc"))
    tile = _tile(Body((Loop(axis=m, body=Body(cell)),)))
    node = tile.op
    assert isinstance(node, Fold) and node.axis is None and not node.operands
    assert any(isinstance(s, Loop) and s.is_reduce for s in node.body)


def test_reduce_feeding_prologue_sinks_into_the_fold():
    """A pure prologue value the reduce body reads moves inside the loop before extraction, so
    the λ defines what it reads per step; an epilogue-only value stays in the projection."""
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
    tile = _tile(Body((Loop(axis=m, body=Body(cell)),)))
    node = tile.op
    assert isinstance(node, Fold) and node.axis is None and len(node.operands) == 1
    fold = node.operands[0]
    assert {s.input for s in fold.lift.body if isinstance(s, Load)} == {"s", "x"}
    assert {getattr(s, "input", None) for s in node.body if isinstance(s, Load)} == {"b"}


def test_fold_reading_a_body_defined_name_is_restored_verbatim():
    """Operands lower before the projection body, so a fold whose λ reads a name only a RAW body
    member defines cannot hoist — root formation restores it to its byte-exact loop."""
    m, k, k2 = Axis("m", Dim(8)), Axis("k", Dim(16)), Axis("k2", Dim(16))
    from emmy.compiler.dtype import F32

    raw = Loop(  # unliftable (typed accumulator) producer the second reduce reads
        axis=k,
        body=Body(
            (
                Load(name="xv", input="x", index=(Var("m"), Var("k"))),
                Accum(name="pre", value="xv", op=ElementwiseImpl("add"), axes=("k",), dtype=F32),
            )
        ),
    )
    consumer = Loop(
        axis=k2,
        body=Body(
            (
                Load(name="yv", input="y", index=(Var("m"), Var("k2"))),
                Assign(name="sy", op=ElementwiseImpl("multiply"), args=("yv", "pre")),
                Accum(name="acc", value="sy", op=ElementwiseImpl("add"), axes=("k2",)),
            )
        ),
    )
    cell = (raw, consumer, Write(output="out", index=(Var("m"),), value="acc"))
    tile = _tile(Body((Loop(axis=m, body=Body(cell)),)))
    node = tile.op
    assert isinstance(node, Fold) and node.axis is None and not node.operands
    raw_loops = [s for s in node.body if isinstance(s, Loop) and s.is_reduce]
    assert len(raw_loops) == 2, "the consumer restored beside the escape, order preserved"


def test_non_distributing_lift_declines_to_planar():
    """The contraction reading is stated on the semiring traits: ``maximum`` is not a registered
    ⊗ over ``add`` (``distributes_over`` says no), so the fold keeps its PLANAR reduce reading
    with loads inline — no op-name list anywhere in the decision."""
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


def test_contraction_formation_gates_on_the_semiring():
    """``Fold.contraction`` asserts the laws its consumers rely on — an unregistered (⊗, ⊕) pair
    is a formation error, and a built node exposes its instance via ``semiring``."""
    import pytest

    from emmy.compiler.ir.pure.fold import Channel

    k = Axis("k", Dim(16))
    a = Load(name="av", input="a", index=(Var("m"), Var("k")))
    b = Load(name="bv", input="b", index=(Var("n"), Var("k")))
    with pytest.raises(ValueError, match="semiring"):
        Fold.contraction(k_axis=k, a=a, channels=(Channel(b=b, acc="acc"),), product="maximum")
    node = Fold.contraction(k_axis=k, a=a, channels=(Channel(b=b, acc="acc"),))
    assert tuple(o.name for o in node.semiring) == ("multiply", "add")


def test_recognize_fires_through_the_pipeline():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]
    lowered = Pipeline.build(TILE_PASSES).run(g)
    from emmy.compiler.ir.tile import TileOp

    tiles = [n.op for n in lowered.nodes.values() if isinstance(n.op, TileOp)]
    assert tiles, "the lift fired and nothing downstream traffics in LoopOp"
