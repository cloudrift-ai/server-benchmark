from __future__ import annotations

from pathlib import Path

import pytest

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import Channel, Fold, Lambda, M, is_contraction
from emmy.compiler.ir.pure.closure import Closure
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.ir.tile.normalize import _share_common_cones, normalize_fold_tree
from emmy.compiler.ir.tile.path import family_sites, sites
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams
from emmy.compiler.pipeline.search.golden import _lifted_target, load_golden_file, load_golden_records


def _lift(body: Body) -> TileOp:
    graph = Graph()
    graph.add_node(LoopOp(body=body), [], Tensor("out", (1,)), node_id="out")
    graph.outputs = ["out"]
    return Pipeline.build(["lowering/tile"], select=["lift"]).run(graph).nodes["out"].op


def _planar_matmul() -> Fold:
    axis = Axis("k", Dim(32))
    body = Body(
        (
            Load(name="left", input="x", index=(Var("m"), Var("k"))),
            Load(name="right", input="w", index=(Var("n"), Var("k"))),
            Assign(name="product", op="multiply", args=("left", "right")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), names=("acc",))
    return Fold(axis=axis, lift=Lambda.closing(("k",), body, ("product",)), init=init, combine=combine)


def test_tile_post_init_canonicalizes_contraction() -> None:
    tile = TileOp(
        op=Fold.projection(operands=(_planar_matmul(),)),
        place=Placement(free=(Axis("m", 8), Axis("n", 16))),
    )

    assert isinstance(tile.op, Fold) and tile.op.role is AxisRole.CONTRACTION
    assert tile.op.a.input == "x"
    assert tile.op.b.input == "w"
    assert TileOp(op=tile.op, place=tile.place).op == tile.op


def test_tile_post_init_canonicalizes_broadcast_batched_contraction() -> None:
    axis = Axis("k", Dim(32))
    body = Body(
        (
            Load(name="left", input="x", index=(Var("k"), Var("batch") * 8 + Var("m"))),
            Load(name="right", input="w", index=(Var("n"), Var("k"))),
            Assign(name="product", op="multiply", args=("left", "right")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), names=("acc",))
    planar = Fold(axis=axis, lift=Lambda.closing(("k",), body, ("product",)), init=init, combine=combine)

    tile = TileOp(
        op=planar,
        place=Placement(free=(Axis("batch", 4), Axis("m", 8), Axis("n", 16))),
    )

    assert tile.op.role is AxisRole.CONTRACTION
    assert tile.op.a.input == "w"
    assert tile.op.b.input == "x"


def test_tile_post_init_recovers_an_elided_unit_contraction_row() -> None:
    axis = Axis("k", Dim(16))
    body = Body(
        (
            Load(name="left", input="x", index=(Literal(0, "int"), Var("k"))),
            Load(name="right", input="w", index=(Var("n"), Var("k"))),
            Assign(name="product", op="multiply", args=("left", "right")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), names=("acc",))
    planar = Fold(axis=axis, lift=Lambda.closing(("k",), body, ("product",)), init=init, combine=combine)

    tile = TileOp(
        op=planar,
        place=Placement(free=(Axis("n", 16),)),
        output_specs=(OutputSpec(Write(output="out", index=(Literal(0, "int"), Var("n")), value="acc")),),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("_um", "n")
    assert tile.op.role is AxisRole.CONTRACTION


@pytest.mark.parametrize(
    "index",
    (
        (Var("n"), Literal(0, "int")),
        (Literal(0, "int"), Var("n") * 2),
    ),
    ids=("varying-coordinate-before-zero", "strided-column"),
)
def test_tile_post_init_does_not_infer_a_unit_row_from_a_non_dense_boundary(index) -> None:
    axis = Axis("k", Dim(16))
    body = Body(
        (
            Load(name="left", input="x", index=(Literal(0, "int"), Var("k"))),
            Load(name="right", input="w", index=(Var("n"), Var("k"))),
            Assign(name="product", op="multiply", args=("left", "right")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), names=("acc",))
    planar = Fold(axis=axis, lift=Lambda.closing(("k",), body, ("product",)), init=init, combine=combine)

    tile = TileOp(
        op=planar,
        place=Placement(free=(Axis("n", 16),)),
        output_specs=(OutputSpec(Write(output="out", index=index, value="acc")),),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("n",)


def test_contraction_promotes_a_shared_store_sweep_once() -> None:
    n = Axis("n", 16)
    stores = tuple(OutputSpec(write=Write(output="out", index=(Var("m"), Var("n")), value="acc"), sweep=n) for _ in range(2))

    tile = TileOp(op=_planar_matmul(), place=Placement(free=(Axis("m", 8),)), output_specs=stores)

    assert tuple(axis.name for axis in tile.place.free) == ("m", "n")
    assert all(store.sweep is None for store in tile.output_specs)


def test_contraction_promotes_a_shared_store_sweep_after_grid_mapping() -> None:
    """A mapped tile keeps promotion as a construction invariant."""
    m, n = Axis("m", 8), Axis("n", 16)
    normalized = TileOp(op=_planar_matmul(), place=Placement(free=(m, n))).op
    tile = TileOp(
        op=normalized,
        place=Placement(free=(m,), grid=(m,), mapped=True),
        output_specs=(OutputSpec(write=Write(output="out", index=(Var("m"), Var("n")), value="acc"), sweep=n),),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("m", "n")
    assert tuple(axis.name for axis in tile.place.grid) == ("m", "n")
    assert tile.place.is_mapped
    assert tile.output_specs[0].sweep is None


def test_nested_contraction_promotes_a_shared_store_sweep() -> None:
    """A sibling reduction can be the root-most node while a later contraction reads the sweep axis."""
    m, n = Axis("m", 8), Axis("n", 16)
    stat_axis = Axis("r", 4)
    init, combine = M(ElementwiseImpl("add"), names=("stat",))
    stat = Fold(
        axis=stat_axis,
        lift=Lambda.closing(("r",), Body((Load(name="sample", input="s", index=(Var("m"), Var("r"))),)), ("sample",)),
        init=init,
        combine=combine,
    )
    root = Fold.projection(
        operands=(stat, _planar_matmul()),
        body=Body((Assign(name="result", op="add", args=("stat", "acc")),)),
        results=("result",),
    )
    tile = TileOp(
        op=root,
        place=Placement(free=(m,)),
        output_specs=(OutputSpec(write=Write(output="out", index=(Var("m"), Var("n")), value="result"), sweep=n),),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("m", "n")
    assert tile.output_specs[0].sweep is None


def test_nested_contraction_promotes_a_swept_column_beside_an_implicit_unit_row() -> None:
    """A nested linear site turns the swept column into grid placement before scheduling."""
    n, k, r = Axis("n", 16), Axis("k", 32), Axis("r", 4)
    stat_init, stat_combine = M(ElementwiseImpl("add"), names=("stat",))
    stat = Fold(
        axis=r,
        lift=Lambda(
            params=("r",),
            body=Body((Load(name="sample", input="s", index=(Var("r"),)),)),
            results=("sample",),
        ),
        init=stat_init,
        combine=stat_combine,
    )
    linear_init, linear_combine = M(ElementwiseImpl("add"), names=("acc",))
    linear = Fold(
        axis=k,
        lift=Lambda.closing(
            ("k",),
            Body(
                (
                    Load(name="left", input="x", index=(Literal(0, "int"), Var("k"))),
                    Load(name="right", input="w", index=(Var("k"), Var("n"))),
                    Assign(name="product", op="multiply", args=("left", "right")),
                )
            ),
            ("product",),
        ),
        init=linear_init,
        combine=linear_combine,
    )
    root = Fold.projection(
        operands=(stat, linear),
        body=Body((Assign(name="result", op="add", args=("stat", "acc")),)),
        results=("result",),
    )
    tile = TileOp(
        op=root,
        output_specs=(
            OutputSpec(
                write=Write(
                    output="out",
                    index=(Literal(0, "int"), Literal(0, "int"), Var("n")),
                    value="result",
                ),
                sweep=n,
            ),
        ),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("_um", "n")
    assert tile.output_specs[0].sweep is None
    assert any(site.node.role is AxisRole.CONTRACTION for site in sites(tile.op))
    assert TileOp(op=tile.op, place=tile.place, output_specs=tile.output_specs).op is tile.op


def test_matvec_recovers_an_implicit_unit_row_through_an_output_reshape() -> None:
    """A split head/value boundary is still one varying matrix column coordinate."""
    n, k = Axis("n", 2048), Axis("k", 1024)
    init, combine = M(ElementwiseImpl("add"), names=("acc",))
    product = Fold(
        axis=k,
        lift=Lambda.closing(
            ("k",),
            Body(
                (
                    Load(name="left", input="x", index=(Var("k"),)),
                    Load(name="right", input="w", index=(Var("k"), Var("n"))),
                    Assign(name="product", op="multiply", args=("left", "right")),
                )
            ),
            ("product",),
        ),
        init=init,
        combine=combine,
    )
    tile = TileOp(
        op=product,
        place=Placement(free=(n,)),
        output_specs=(
            OutputSpec(
                write=Write(
                    output="out",
                    index=(Literal(0, "int"), Literal(0, "int"), Var("n") / 128, Var("n") % 128),
                    value="acc",
                )
            ),
        ),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("_um", "n")
    assert isinstance(tile.op, Fold) and tile.op.role is AxisRole.CONTRACTION


def test_promoted_attention_output_sweep_closes_the_a100_b_seam_idempotently() -> None:
    """The reduced Qwen3 target needs its promoted value-width axis to close computed B."""
    case = Path(__file__).parents[2] / "realization/cases/attention/rmsnorm-gqa-b-cut.yaml"
    (record,) = load_golden_records(load_golden_file(case))

    tile = _lifted_target(record)
    reconstructed = TileOp(op=tile.op, name=tile.name, place=tile.place, output_specs=tile.output_specs)

    assert tuple(axis.name for axis in tile.place.free) == ("a0", "a1", "a6")
    assert all(spec.sweep is None for spec in tile.output_specs)
    assert reconstructed.op is tile.op
    # The authored seam carries the lexical provider closure needed to cut the input projection
    # without duplicating the statistic.
    seams = {seam.spelling: seam for seam in cuttable_seams(tile)}
    seam = seams["PLACE@map.fold.a.fold.b1"]
    assert seam.requires


def _key_swept_score(shared_reader: bool = False) -> TileOp:
    """A reduced attention shape: a key sweep whose body computes the per-key statistic and rsqrt
    feeding the score contraction's computed B operand cone. With ``shared_reader`` the sweep's
    own result also reads the rsqrt, so the chain does not die into the edge."""
    d, t = Axis("d", Dim(16)), Axis("t", Dim(8))
    stat_init, stat_combine = M(ElementwiseImpl("add"), names=("ss",))
    stat = Fold(
        axis=d,
        lift=Lambda.closing(
            ("d",),
            Body(
                (
                    Load(name="ksq", input="k", index=(Var("t"), Var("d"))),
                    Assign(name="sq", op="multiply", args=("ksq", "ksq")),
                )
            ),
            ("sq",),
        ),
        init=stat_init,
        combine=stat_combine,
    )
    b_cone = Fold.projection(
        body=Body(
            (
                Load(name="kv", input="k", index=(Var("t"), Var("d"))),
                Assign(name="kn", op="multiply", args=("kv", "inv")),
            )
        ),
        results=("kn",),
    )
    a_edge = Load(name="qv", input="q", index=(Var("m"), Var("d")))
    score = Fold.contraction(k_axis=Axis("d", Dim(16)), a=a_edge, channels=(Channel(b=b_cone, acc="acc"),))
    result = Assign(name="mixed", op="multiply", args=("acc", "inv"))
    sweep_init, sweep_combine = M(ElementwiseImpl("maximum"), names=("mx",))
    sweep = Fold(
        axis=t,
        operands=(stat, score),
        lift=Lambda.closing(
            ("t", "ss", "acc"),
            Body((Assign(name="inv", op="rsqrt", args=("ss",)), *((result,) if shared_reader else ()))),
            ("mixed",) if shared_reader else ("acc",),
        ),
        init=sweep_init,
        combine=sweep_combine,
    )
    return TileOp(op=Fold.projection(operands=(sweep,)), place=Placement(free=(Axis("m", 4),)))


def test_key_swept_statistic_closes_the_computed_b_operand() -> None:
    """The chain feeding only the score's B cone moves onto the edge, closing it at its axes."""
    tile = _key_swept_score()

    sweep = tile.op if tile.op.axis is not None else tile.op.operands[0]
    (score,) = (edge for edge in sweep.operands if is_contraction(edge))
    assert score.role is AxisRole.CONTRACTION
    b = score.b
    assert isinstance(b, Fold) and frozenset(b.environment) <= {"m", "t", "d"}
    assert any(site.node.axis is not None for site in sites(b) if isinstance(site.node, Fold))
    assert TileOp(op=tile.op, place=tile.place).op is tile.op
    # A synthetic TileOp carries no graph output tensors, so the workspace dtype rule cannot
    # resolve a cuttable seam here; the reduced-target test below asserts the offered seam.
    assert id(b) in {id(site.node) for site in family_sites("PLACE", sites(tile.op))}


def test_key_swept_statistic_stays_when_a_sibling_reads_it() -> None:
    """A chain the sweep's own result also reads is not moved — the step's work is never duplicated."""
    tile = _key_swept_score(shared_reader=True)

    sweep = tile.op if tile.op.axis is not None else tile.op.operands[0]
    assert any(isinstance(stmt, Assign) and stmt.name == "inv" for stmt in sweep.lift.body)
    (score,) = (edge for edge in sweep.operands if is_contraction(edge))
    assert "inv" in score.b.environment
    assert id(score.b) not in {id(seam.node) for seam in cuttable_seams(tile)}


def test_normalization_shares_structurally_identical_cones() -> None:
    """The tree-wide invariant: after normalization, no two DISTINCT Fold objects in the tree are
    structurally equal with the same captures — copies fusion inlined into several consumption
    sites (attention's softmax statistics, once in the weight cone and once in the epilogue) are
    one object, so placement sees one value and a composed cut materializes it once. Severed
    sharing is the recompute class PR #679 measured at three orders of magnitude."""
    case = Path(__file__).parents[2] / "realization/cases/attention/rmsnorm-qk-sdpa-composed-cut.yaml"
    (record,) = load_golden_records(load_golden_file(case))
    tile = _lifted_target(record)

    def folds(node, out):
        if isinstance(node, Fold):
            out.setdefault(id(node), node)
            for edge in node.operands:
                folds(edge, out)
            for stmt in node.lift.body:
                folds(stmt, out)
        return out

    def unify_key(node: Fold):
        return Closure(Lambda(params=(), body=Body((node,)), results=node.defines()), ()).canonical()

    by_identity = folds(tile.op, {})
    by_value: dict[tuple, list[Fold]] = {}
    for node in by_identity.values():
        by_value.setdefault((node.structural_key(), node.deps(), node.defines()), []).append(node)
    for twins in by_value.values():
        distinct_equals = [(a, b) for index, a in enumerate(twins) for b in twins[index + 1 :] if unify_key(a) == unify_key(b)]
        assert not distinct_equals, "normalization left same-value cones as distinct objects"


def test_reduced_qk_attention_offers_the_statistic_arm_b_seams_idempotently() -> None:
    """The reduced Qwen3 q/k-norm SDPA target offers the score contractions' K operand cones."""
    case = Path(__file__).parents[2] / "realization/cases/attention/rmsnorm-qk-sdpa-stat-b-cut.yaml"
    (record,) = load_golden_records(load_golden_file(case))

    tile = _lifted_target(record)
    seams = {seam.spelling: seam for seam in cuttable_seams(tile)}
    seam = seams["PLACE@map.fold.a.map.fold.fold.b1"]
    # The score contractions' K cones are one VALUE. Retaining the shared statistic at its
    # defining scope lets normalization collapse one more duplicate, leaving one sibling whose
    # differently named key axis is recorded by the capture correspondence.
    assert len(seam.siblings) == 1
    assert all(dict(pairs).keys() == dict(seam.siblings[0][1]).keys() for _, pairs in seam.siblings)
    assert "PLACE@map.fold.a.map.fold.fold.b2" not in seams
    assert TileOp(op=tile.op, name=tile.name, place=tile.place, output_specs=tile.output_specs).op is tile.op


def _statistic_under_two_binders() -> TileOp:
    """A row statistic feeding a contraction nested under a separate fold binder."""
    r, k, t = Axis("r", Dim(16)), Axis("k", Dim(16)), Axis("t", Dim(4))
    stat_init, stat_combine = M(ElementwiseImpl("add"), names=("ss",))
    stat = Fold(
        axis=r,
        lift=Lambda.closing(
            ("r",),
            Body(
                (
                    Load(name="xr", input="x", index=(Var("m"), Var("r"))),
                    Assign(name="sq", op="multiply", args=("xr", "xr")),
                )
            ),
            ("sq",),
        ),
        init=stat_init,
        combine=stat_combine,
    )
    inv = Assign(name="inv", op="rsqrt", args=("ss",))
    b_cone = Fold.projection(
        body=Body(
            (
                Load(name="wv", input="w", index=(Var("t"), Var("k"))),
                Assign(name="wn", op="multiply", args=("wv", "inv")),
            )
        ),
        results=("wn",),
    )
    score = Fold.contraction(
        k_axis=k,
        a=Load(name="qv", input="q", index=(Var("m"), Var("k"))),
        channels=(Channel(b=b_cone, acc="acc"),),
    )
    sweep_init, sweep_combine = M(ElementwiseImpl("maximum"), names=("mx",))
    sweep = Fold(
        axis=t,
        operands=(score,),
        lift=Lambda.closing(("t", "acc"), Body(), ("acc",)),
        init=sweep_init,
        combine=sweep_combine,
    )
    # The statistic and its rsqrt are the sweep's SOURCE — a projection evaluates its operands
    # before its body, so the chain the sweep's cone captures rides its own edge rather than a
    # sibling body position.
    source = Fold.projection(operands=(stat,), body=Body((inv,)), results=("inv",))
    return TileOp(op=Fold.projection(operands=(source, sweep), results=("mx",)), place=Placement(free=(Axis("m", 4),)))


def _loop_scopes(stmts, enclosing: tuple[str, ...] = ()) -> list[tuple[str, tuple[str, ...]]]:
    """Each synthesized loop axis paired with its enclosing loop axes."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for stmt in stmts:
        axis = getattr(stmt, "axis", None)
        inner = (*enclosing, axis.name) if axis is not None else enclosing
        if axis is not None:
            out.append((axis.name, enclosing))
        for body in stmt.nested():
            out.extend(_loop_scopes(list(body), inner))
    return out


def test_a_statistic_is_not_sunk_beneath_new_binders() -> None:
    """Projection closure must preserve the statistic's evaluation multiplicity."""
    tile = _statistic_under_two_binders()

    scopes = [enclosing for name, enclosing in _loop_scopes(tile.op.lower()) if name == "r"]
    assert scopes == [()], scopes


def test_a_fold_reading_a_sweep_axis_declares_it_on_the_operand_edge() -> None:
    """A per-column reduce whose streamed load reads the boundary store's sweep axis is an operand
    EDGE like any other term, and it DECLARES that axis.

    Body position used to carry this: a body member is re-wrapped inside the sweep ``Loop`` by
    reconstitution while an edge lowers at kernel scope, so a sweep-reading fold had to stay in the
    body or its column rendered as an undefined identifier (found live: DeepSeek-V4 post16's
    ``k_div_36``). A term cannot be a body member — a Fold tree composes through ``operands`` — so
    the fact is carried explicitly instead: the sweep axis is a lift param, which is what lets the
    evaluation domain be recovered from the term rather than inferred from where it sits."""
    k = Axis("k", Dim(4))
    j = Axis("j", Dim(4))
    body = Body((Load(name="xin", input="x", index=(Var("m"), Var("k"), Var("j"))),))
    init, combine = M(ElementwiseImpl("add"), names=("acc",))
    fold = Fold(axis=k, lift=Lambda.closing(("k",), body, ("xin",)), init=init, combine=combine)
    epilogue = Assign(name="v", op="relu", args=("acc",))

    tile = TileOp(
        op=Fold.projection(operands=(fold,), body=Body((epilogue,)), results=("v",)),
        place=Placement(free=(Axis("m", 8),)),
        output_specs=(OutputSpec(write=Write(output="out", index=(Var("m"), Var("j")), value="v"), sweep=j),),
    )

    assert not any(isinstance(stmt, Fold) for stmt in tile.op.body)
    (edge,) = tile.op.operands
    assert j.name in edge.lift.params, edge.lift.params
    assert j.name in edge.deps(), edge.deps()


def test_contraction_clusters_alpha_equivalent_shared_operands() -> None:
    axis = Axis("k", Dim(32))
    body = Body(
        (
            Load(name="left0", input="x", index=(Var("m"), Var("k"))),
            Load(name="right0", input="w0", index=(Var("n"), Var("k"))),
            Assign(name="product0", op="multiply", args=("left0", "right0")),
            Load(name="left1", input="x", index=(Var("m"), Var("k"))),
            Load(name="right1", input="w1", index=(Var("n"), Var("k"))),
            Assign(name="product1", op="multiply", args=("left1", "right1")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), ElementwiseImpl("add"), names=("acc0", "acc1"))
    planar = Fold(
        axis=axis,
        lift=Lambda.closing(("k",), body, ("product0", "product1")),
        init=init,
        combine=combine,
    )

    tile = TileOp(
        op=Fold.projection(operands=(planar,)),
        place=Placement(free=(Axis("m", 8), Axis("n", 16))),
    )

    contraction = tile.op.operands[0]
    assert contraction.role is AxisRole.CONTRACTION
    assert len(contraction.channels) == 2
    assert contraction.a.input == "x"


def test_contraction_coalesces_overlapping_equivalent_shared_operands() -> None:
    axis = Axis("k", Dim(32))
    body = Body(
        (
            Load(name="left", input="x", index=(Var("m"), Var("k"))),
            Load(name="scale", input="s", index=(Var("k"),)),
            Assign(name="scaled0", op="multiply", args=("left", "scale")),
            Assign(name="scaled1", op="multiply", args=("left", "scale")),
            Load(name="right0", input="w0", index=(Var("n"), Var("k"))),
            Load(name="right1", input="w1", index=(Var("n"), Var("k"))),
            Assign(name="product0", op="multiply", args=("scaled0", "right0")),
            Assign(name="product1", op="multiply", args=("scaled1", "right1")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), ElementwiseImpl("add"), names=("acc0", "acc1"))
    planar = Fold(
        axis=axis,
        lift=Lambda.closing(("k",), body, ("product0", "product1")),
        init=init,
        combine=combine,
    )

    tile = TileOp(op=planar, place=Placement(free=(Axis("m", 8), Axis("n", 16))))

    assert tile.op.role is AxisRole.CONTRACTION
    assert len(tile.op.channels) == 2
    assert tile.op.a.out == "scaled0"
    assert sum(isinstance(stmt, Assign) and stmt.name.startswith("scaled") for stmt in tile.op.a.body) == 1


def test_contraction_orients_a_shared_commutative_argument_first() -> None:
    axis = Axis("k", Dim(32))
    body = Body(
        (
            Load(name="left0", input="x0", index=(Var("m"), Var("k"))),
            Load(name="left1", input="x1", index=(Var("m"), Var("k"))),
            Load(name="right", input="w", index=(Var("k"), Var("n"))),
            Assign(name="product0", op="multiply", args=("left0", "right")),
            Assign(name="product1", op="multiply", args=("left1", "right")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), ElementwiseImpl("add"), names=("acc0", "acc1"))
    planar = Fold(
        axis=axis,
        lift=Lambda.closing(("k",), body, ("product0", "product1")),
        init=init,
        combine=combine,
    )

    tile = TileOp(op=planar, place=Placement(free=(Axis("m", 8), Axis("n", 16))))

    assert tile.op.role is AxisRole.CONTRACTION
    view = tile.op.as_contraction()
    assert (view.product.name, view.plus.name) == ("multiply", "add")
    assert tile.op.a.input == "w"
    assert [channel.b.input for channel in tile.op.channels] == ["x0", "x1"]
    assert TileOp(op=tile.op, place=tile.place).op is tile.op


def test_contraction_computes_an_equivalent_channel_once() -> None:
    axis = Axis("k", Dim(32))
    body = Body(
        (
            Load(name="left", input="x", index=(Var("m"), Var("k"))),
            Load(name="right", input="w", index=(Var("n"), Var("k"))),
            Assign(name="product0", op="multiply", args=("left", "right")),
            Assign(name="product1", op="multiply", args=("left", "right")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), ElementwiseImpl("add"), names=("acc0", "acc1"))
    planar = Fold(
        axis=axis,
        lift=Lambda.closing(("k",), body, ("product0", "product1")),
        init=init,
        combine=combine,
    )

    tile = TileOp(op=planar, place=Placement(free=(Axis("m", 8), Axis("n", 16))))

    assert tile.op.role is AxisRole.CONTRACTION and len(tile.op.channels) == 2
    assert len(tile.op.operands) == 2 and tile.op.channels[0].b is tile.op.channels[1].b
    assert sum(isinstance(stmt, Load) and stmt.input == "w" for stmt in tile.op.loop.body) == 1


def test_projection_keeps_only_the_maximal_shared_operand() -> None:
    small = Fold.projection(body=Body((Load(name="a", input="x", index=(Var("m"),)),)), results=("a",))
    large = Fold.projection(
        operands=(small,),
        body=Body((Assign(name="b", op="copy", args=("a",)),)),
        results=("a", "b"),
    )

    projection = Fold.projection(operands=(small, large), body=Body((Assign(name="o", op="copy", args=("b",)),)))

    assert projection.operands == (large,)
    assert projection.lift.params == ("a", "b")
    assert sum(isinstance(stmt, Load) and stmt.input == "x" for stmt in projection.lower()) == 1


def test_semiring_merges_overlapping_operand_cones_into_one_multi_result_edge() -> None:
    axis = Axis("k", Dim(32))
    body = Body(
        (
            Load(name="left", input="x", index=(Var("m"), Var("k"))),
            Load(name="scale", input="s", index=(Var("k"),)),
            Assign(name="scaled", op="multiply", args=("left", "scale")),
            Load(name="right0", input="w0", index=(Var("k"), Var("n"))),
            Load(name="right1", input="w1", index=(Var("k"), Var("n"))),
            Assign(name="product0", op="multiply", args=("left", "right0")),
            Assign(name="product1", op="multiply", args=("scaled", "right1")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), ElementwiseImpl("add"), names=("acc0", "acc1"))
    planar = Fold(
        axis=axis,
        lift=Lambda.closing(("k",), body, ("product0", "product1")),
        init=init,
        combine=combine,
    )

    tile = TileOp(op=planar, place=Placement(free=(Axis("m", 8), Axis("n", 16))))

    shared = next(edge for edge in tile.op.operands if isinstance(edge, Fold) and edge.lift.results == ("left", "scaled"))
    assert tuple(shared.lift.results) == ("left", "scaled")
    # The operand correspondence is the param PREFIX; the tail is the declared environment (the
    # enclosing axes the binder supplied), which is why this checks a prefix rather than equality.
    bound = ("k", "right0", "left", "scaled", "right1")
    assert tuple(tile.op.lift.params[: len(bound)]) == bound
    assert set(tile.op.lift.params[len(bound) :]) <= {"m", "n"}, tile.op.lift.params
    assert sum(isinstance(stmt, Load) and stmt.input == "x" for stmt in tile.op.loop.body) == 1


def _computed_matmul(*, computed_a: bool, computed_b: bool) -> TileOp:
    axis = Axis("k", Dim(32))
    body = []
    body.append(Load(name="left", input="x", index=(Var("m"), Var("k"))))
    left = "left"
    if computed_a:
        body.extend(
            (
                Load(name="left_scale", input="xs", index=(Var("k"),)),
                Assign(name="computed_left", op="multiply", args=(left, "left_scale")),
            )
        )
        left = "computed_left"
    body.append(Load(name="right", input="w", index=(Var("k"), Var("n"))))
    right = "right"
    if computed_b:
        body.extend(
            (
                Load(name="right_scale", input="ws", index=(Var("k"),)),
                Assign(name="computed_right", op="multiply", args=(right, "right_scale")),
            )
        )
        right = "computed_right"
    body.append(Assign(name="product", op="multiply", args=(left, right)))
    init, combine = M(ElementwiseImpl("add"), names=("acc",))
    planar = Fold(axis=axis, lift=Lambda.closing(("k",), Body(body), ("product",)), init=init, combine=combine)
    return TileOp(
        op=Fold.projection(operands=(planar,)),
        place=Placement(free=(Axis("m", 8), Axis("n", 16))),
    )


def test_contraction_factors_a_computed_operand_cone() -> None:
    tile = _computed_matmul(computed_a=True, computed_b=False)

    assert tile.op.role is AxisRole.CONTRACTION
    assert isinstance(tile.op.a, Fold) and tile.op.a.axis is None and tile.op.a.out == "computed_left"
    assert isinstance(tile.op.b, Load) and tile.op.b.input == "w"


def test_contraction_factors_b_computed_operand_cone() -> None:
    tile = _computed_matmul(computed_a=False, computed_b=True)

    assert tile.op.role is AxisRole.CONTRACTION
    assert isinstance(tile.op.a, Load) and tile.op.a.input == "x"
    assert isinstance(tile.op.b, Fold) and tile.op.b.axis is None and tile.op.b.out == "computed_right"


def test_contraction_factors_both_computed_operand_cones_idempotently() -> None:
    tile = _computed_matmul(computed_a=True, computed_b=True)

    assert tile.op.role is AxisRole.CONTRACTION
    assert isinstance(tile.op.a, Fold) and isinstance(tile.op.b, Fold)
    assert TileOp(op=tile.op, place=tile.place).op is tile.op


def test_contraction_preserves_computed_operand_statement_order() -> None:
    j, k = Axis("j", Dim(32)), Axis("k", Dim(64))
    max_init, max_combine = M(ElementwiseImpl("maximum"), names=("row_max",))
    row_max = Fold(
        axis=j,
        lift=Lambda.closing(("j",), Body((Load(name="score", input="s", index=(Var("m"), Var("j"))),)), ("score",)),
        init=max_init,
        combine=max_combine,
    )
    sum_init, sum_combine = M(ElementwiseImpl("add"), names=("value",))
    value = Fold(
        axis=j,
        lift=Lambda.closing(
            ("j",),
            Body(
                (
                    Load(name="v", input="v", index=(Var("j"), Var("k"))),
                    Assign(name="weighted", op="multiply", args=("inv", "v")),
                )
            ),
            ("weighted",),
        ),
        init=sum_init,
        combine=sum_combine,
    )
    outer_init, outer_combine = M(ElementwiseImpl("add"), names=("out",))
    outer = Fold(
        axis=k,
        operands=(row_max, value),
        lift=Lambda.closing(
            ("k", "row_max", "value"),
            Body(
                (
                    Assign(name="inv", op="reciprocal", args=("row_max",)),
                    Load(name="weight", input="w", index=(Var("k"), Var("n"))),
                    Assign(name="product", op="multiply", args=("value", "weight")),
                )
            ),
            ("product",),
        ),
        init=outer_init,
        combine=outer_combine,
    )

    tile = TileOp(op=outer, place=Placement(free=(Axis("m", 8), Axis("n", 16))))

    assert tile.op.role is AxisRole.CONTRACTION
    computed = tile.op.a
    lowered = computed.lower()
    assert isinstance(lowered[0], Loop) and lowered[0].axis.name == "j"
    assert isinstance(lowered[1], Assign) and lowered[1].name == "inv"
    assert isinstance(lowered[2], Loop) and lowered[2].axis.name == "j"
    assert TileOp(op=tile.op, place=tile.place).op is tile.op


def test_share_common_cones_unifies_internally_renamed_copies() -> None:
    """Alpha-equal cones with identical captures and interface names collapse to one object even
    when their internal binder spelling differs — plain structural equality would keep them
    distinct and silently sever the sharing."""

    def cone(load: str, product: str, row: str = "m") -> Fold:
        body = Body(
            (
                Load(name=load, input="x", index=(Var(row), Var("k"))),
                Assign(name=product, op="multiply", args=(load, load)),
            )
        )
        init, combine = M(ElementwiseImpl("add"), names=("acc",))
        return Fold(axis=Axis("k", Dim(8)), lift=Lambda.closing(("k",), body, (product,)), init=init, combine=combine)

    def consumer(inner: Fold, out: str) -> Fold:
        return Fold.projection(operands=(inner,), body=Body((Assign(name=out, op="exp", args=("acc",)),)), results=(out,))

    def rooted(second: Fold) -> Fold:
        return _share_common_cones(
            Fold.projection(
                operands=(consumer(cone("l1", "p1"), "u"), consumer(second, "v")),
                body=Body((Assign(name="w", op="add", args=("u", "v")),)),
                results=("w",),
            )
        )

    unified = rooted(cone("l2", "p2"))
    assert unified.operands[0].operands[0] is unified.operands[1].operands[0]

    distinct = rooted(cone("l2", "p2", row="q"))  # same form, different capture — a different value
    assert distinct.operands[0].operands[0] is not distinct.operands[1].operands[0]


def test_closure_equality_includes_captured_axes() -> None:
    # A lambda binds every name it reads, so the enclosing row coordinate is a trailing param;
    # ``axes`` then names WHICH of those params are the environment, not a permission to capture.
    first = Lambda.closing(("k",), Body((Load(name="x", input="q", index=(Var("row"), Var("k"))),)), ("x",))
    second = Lambda.closing(("depth",), Body((Load(name="value", input="q", index=(Var("query"), Var("depth"))),)), ("value",))
    assert first.params == ("k", "row") and second.params == ("depth", "query")

    assert Closure(first, ("row", "k")) == Closure(second, ("query", "depth"))  # equality is alpha-invariant
    assert Closure(first, ("row", "k")) != Closure(first, ("k", "row"))  # capture order is the positional identity
    # An axis the lambda does not bind is not an unused environment slot any more — it is a name
    # nothing supplies, and the closure refuses it rather than carrying it into the canonical form.
    with pytest.raises(ValueError, match="not params of the lambda"):
        Closure(second, ("unused", "query", "depth"))


def test_total_lift_produces_canonical_contraction() -> None:
    m, n, k = Axis("m", Dim(8)), Axis("n", Dim(16)), Axis("k", Dim(32))
    inner = Body(
        (
            Load(name="left", input="x", index=(Var("m"), Var("k"))),
            Load(name="right", input="w", index=(Var("n"), Var("k"))),
            Assign(name="product", op="multiply", args=("left", "right")),
            Accum(name="acc", value="product", op="add", axes=("k",)),
        )
    )
    body = Body(
        (
            Loop(
                axis=m,
                body=Body(
                    (Loop(axis=n, body=Body((Loop(axis=k, body=inner), Write(output="out", index=(Var("m"), Var("n")), value="acc")))),)
                ),
            ),
        )
    )

    tile = _lift(body)

    assert tile.op.role is AxisRole.CONTRACTION
    assert tile.op.loop.role is AxisRole.CONTRACTION


def _fed_chain_root(feed: bool) -> Fold:
    """A zero-axis root whose body holds a scalar chain and a reduce; ``feed`` wires the reduce's
    nested cone to CAPTURE the chain's result (the composed placement cut's consumer shape)."""
    scale_arg = "v25" if feed else "left"
    inner = Fold.projection(
        operands=(Load(name="left", input="x", index=(Var("k"),)),),
        body=Body((Assign(name="scaled", op="multiply", args=("left", scale_arg)),)),
        results=("scaled",),
    )
    init, combine = M(ElementwiseImpl("add"), names=("acc",))
    fold = Fold(
        axis=Axis("k", Dim(8)),
        operands=(inner,),
        init=init,
        lift=Lambda(params=("k", "scaled"), body=Body((Assign(name="acc__v", op="copy", args=("scaled",)),)), results=("acc__v",)),
        combine=combine,
    )
    chain = (Load(name="ws", input="cutbuf", index=()), Assign(name="v25", op="rsqrt", args=("ws",)))
    # A mixed stmt/term sequence, handed to the former unwrapped: a ``Body`` may not hold a term,
    # and separating this is precisely what ``Fold.projection`` is for.
    return Fold.projection(body=(*chain, fold), results=("acc",))


def test_a_body_fed_fold_takes_its_provider_as_a_sibling_operand() -> None:
    """A fold whose subtree reads a name the scalar chain defines becomes an operand edge, and the
    chain becomes a PROVIDER operand ordered ahead of it.

    Body membership used to carry this: a projection evaluates its operands before its scalar body,
    so hoisting a fed fold emitted its capture as an undefined identifier (DeepSeek-V4 post4096's
    two-cut consumer piece). A term cannot be a body member, so the ordering is carried structurally
    instead — ``_ordered_projection`` splits at the scalar and makes the prefix a source projection,
    and ``splice_operands`` places a provider before the edge that reads it."""
    out = normalize_fold_tree(_fed_chain_root(feed=True))

    assert not any(isinstance(stmt, Fold) for stmt in out.lift.body), "a term is never a body member"
    lowered = [type(stmt).__name__ for stmt in out.lower()]
    assert "Loop" in lowered, lowered
    rendered = "\n".join(line for stmt in out.lower() for line in stmt.pretty())
    assert rendered.index("v25") < rendered.index("acc"), "the provider is emitted before its consumer"


def test_an_unfed_fold_is_an_operand_edge_too() -> None:
    """The same shape with the cone reading only its own operand: also an edge. Position no longer
    varies with the capture, because position no longer encodes anything."""
    out = normalize_fold_tree(_fed_chain_root(feed=False))

    assert any(isinstance(edge, Fold) and edge.axis is not None for edge in out.operands)
    assert not any(isinstance(stmt, Fold) for stmt in out.lift.body)
