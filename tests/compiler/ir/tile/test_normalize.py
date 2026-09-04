"""The canonical Tile IR tree — what formation states about a term and what only the whole tree
can: ``TileOp.__post_init__`` (an elided unit row recovered from the stores, a shared store sweep
promoted to a free axis, the identity projection dissolved), the bilinear form the lift builds
(cone operands, orientation, one shared A), and ``_share_common_cones`` restoring one object per
value."""

from __future__ import annotations

import pytest

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import Fold, Lambda
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.ir.tile.normalize import _share_common_cones
from emmy.compiler.ir.tile.path import family_sites, sites
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
from tests.compiler.helpers import case_target_tile
from tests.compiler.terms import contraction, projection, reduction, slab

M8, N16, K32 = Axis("m", 8), Axis("n", 16), Axis("k", Dim(32))


def _lift(body: Body) -> TileOp:
    graph = Graph()
    graph.add_node(LoopOp(body=body), [], Tensor("out", (1,)), node_id="out")
    graph.outputs = ["out"]
    return Pipeline.build(["lowering/tile"], select=["lift"]).run(graph).nodes["out"].op


def _reduce_loop(*stmts, axis: Axis = K32) -> Fold:
    """A reduce loop over ``axis`` lifted to its term: the ONE former, so a fixture reads exactly like
    a lifted kernel — every load a slab, every computed product argument a cone operand."""
    return fold_from_loop(Loop(axis=axis, body=Body(stmts)))


def _matmul(a_index=None, b_index=None, *, extent: int = 32) -> Fold:
    return _reduce_loop(
        Load(name="left", input="x", index=a_index or (Var("m"), Var("k"))),
        Load(name="right", input="w", index=b_index or (Var("n"), Var("k"))),
        Assign(name="product", op="multiply", args=("left", "right")),
        Accum(name="acc", value="product", op="add", axes=("k",)),
        axis=Axis("k", Dim(extent)),
    )


def _input(edge: Fold) -> str | None:
    """The buffer a slab operand reads, ``None`` for a computed edge."""
    view = edge.as_slab()
    return None if view is None else view.load.input


def _tile(op, *axes: Axis, **fields) -> TileOp:
    free = fields.pop("free", ())
    return TileOp(op=op, place=Placement(free=free), axes=(*free, *axes), **fields)


# ---- construction canonical forms -------------------------------------------------------------- #


def test_tile_post_init_canonicalizes_contraction() -> None:
    tile = _tile(projection((_matmul(),)), K32, free=(M8, N16))

    assert isinstance(tile.op, Fold) and tile.op.as_contraction() is not None  # the identity projection dissolves
    assert _input(tile.op.operands[0]) == "x"
    assert _input(tile.op.operands[1]) == "w"
    assert TileOp(op=tile.op, place=tile.place, axes=tile.axes).op == tile.op


def test_tile_post_init_canonicalizes_broadcast_batched_contraction() -> None:
    planar = _matmul(a_index=(Var("k"), Var("batch") * 8 + Var("m")))

    tile = _tile(planar, K32, free=(Axis("batch", 4), M8, N16))

    assert tile.op.as_contraction() is not None
    assert _input(tile.op.operands[0]) == "w"
    assert _input(tile.op.operands[1]) == "x"


def test_tile_post_init_recovers_an_elided_unit_contraction_row() -> None:
    planar = _matmul(a_index=(Literal(0, "int"), Var("k")), extent=16)

    tile = _tile(
        planar,
        Axis("k", Dim(16)),
        free=(N16,),
        output_specs=(OutputSpec(Write(output="out", index=(Literal(0, "int"), Var("n")), value="acc")),),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("_um", "n")
    assert tile.op.as_contraction() is not None


@pytest.mark.parametrize(
    "index",
    (
        (Var("n"), Literal(0, "int")),
        (Literal(0, "int"), Var("n") * 2),
    ),
    ids=("varying-coordinate-before-zero", "strided-column"),
)
def test_tile_post_init_does_not_infer_a_unit_row_from_a_non_dense_boundary(index) -> None:
    planar = _matmul(a_index=(Literal(0, "int"), Var("k")), extent=16)

    tile = _tile(planar, Axis("k", Dim(16)), free=(N16,), output_specs=(OutputSpec(Write(output="out", index=index, value="acc")),))

    assert tuple(axis.name for axis in tile.place.free) == ("n",)


def test_contraction_promotes_a_shared_store_sweep_once() -> None:
    stores = tuple(OutputSpec(write=Write(output="out", index=(Var("m"), Var("n")), value="acc"), sweep=(N16,)) for _ in range(2))

    tile = _tile(_matmul(), N16, K32, free=(M8,), output_specs=stores)

    assert tuple(axis.name for axis in tile.place.free) == ("m", "n")
    assert all(store.sweep == () for store in tile.output_specs)


def test_contraction_promotes_a_shared_store_sweep_after_grid_mapping() -> None:
    """A mapped tile keeps promotion as a construction invariant."""
    normalized = _tile(_matmul(), K32, free=(M8, N16)).op
    tile = TileOp(
        op=normalized,
        place=Placement(free=(M8,), grid=(M8,), mapped=True),
        axes=(M8, N16, K32),
        output_specs=(OutputSpec(write=Write(output="out", index=(Var("m"), Var("n")), value="acc"), sweep=(N16,)),),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("m", "n")
    assert tuple(axis.name for axis in tile.place.grid) == ("m", "n")
    assert tile.place.is_mapped
    assert tile.output_specs[0].sweep == ()


def test_nested_contraction_promotes_a_shared_store_sweep() -> None:
    """A sibling reduction can be the root-most node while a later contraction reads the sweep axis."""
    r = Axis("r", 4)
    stat = reduction(r, (slab("sample", "s", "m", "r"),), (Assign(name="stat__v", op="copy", args=("sample",)),), ("stat",))
    root = projection((stat, _matmul()), (Assign(name="result", op="add", args=("stat", "acc")),), ("result",))
    tile = _tile(
        root,
        N16,
        r,
        K32,
        free=(M8,),
        output_specs=(OutputSpec(write=Write(output="out", index=(Var("m"), Var("n")), value="result"), sweep=(N16,)),),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("m", "n")
    assert tile.output_specs[0].sweep == ()


def test_nested_contraction_promotes_a_swept_column_beside_an_implicit_unit_row() -> None:
    """A nested linear site turns the swept column into grid placement before scheduling."""
    r = Axis("r", 4)
    stat = reduction(r, (slab("sample", "s", "r"),), (Assign(name="stat__v", op="copy", args=("sample",)),), ("stat",))
    linear = _matmul(a_index=(Literal(0, "int"), Var("k")), b_index=(Var("k"), Var("n")))
    root = projection((stat, linear), (Assign(name="result", op="add", args=("stat", "acc")),), ("result",))
    tile = _tile(
        root,
        N16,
        r,
        K32,
        output_specs=(
            OutputSpec(write=Write(output="out", index=(Literal(0, "int"), Literal(0, "int"), Var("n")), value="result"), sweep=(N16,)),
        ),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("_um", "n")
    assert tile.output_specs[0].sweep == ()
    assert any(site.node.as_contraction() is not None for site in sites(tile.op))
    assert TileOp(op=tile.op, place=tile.place, axes=tile.axes, output_specs=tile.output_specs).op is tile.op


def test_matvec_recovers_an_implicit_unit_row_through_an_output_reshape() -> None:
    """A split head/value boundary is still one varying matrix column coordinate."""
    n, k = Axis("n", 2048), Axis("k", Dim(1024))
    product = _matmul(a_index=(Var("k"),), b_index=(Var("k"), Var("n")), extent=1024)
    tile = _tile(
        product,
        k,
        free=(n,),
        output_specs=(
            OutputSpec(
                write=Write(output="out", index=(Literal(0, "int"), Literal(0, "int"), Var("n") / 128, Var("n") % 128), value="acc")
            ),
        ),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("_um", "n")
    assert isinstance(tile.op, Fold) and tile.op.as_contraction() is not None


def test_promoted_attention_output_sweep_closes_the_a100_b_seam_idempotently() -> None:
    """The reduced Qwen3 target needs its promoted value-width axis to close computed B."""
    tile = case_target_tile("attention/rmsnorm-gqa-b-cut.yaml")
    reconstructed = TileOp(op=tile.op, name=tile.name, place=tile.place, axes=tile.axes, output_specs=tile.output_specs)

    assert tuple(axis.name for axis in tile.place.free) == ("a0", "a1", "a6")
    assert all(spec.sweep == () for spec in tile.output_specs)
    assert reconstructed.op is tile.op
    # The authored seam — the score's K cone — is offered on the promoted tree.
    assert "PLACE@map.1/twist.1/inner.2/map" in {seam.spelling for seam in cuttable_seams(tile)}


# ---- closure at formation ---------------------------------------------------------------------- #


def _key_swept_score(shared_reader: bool = False) -> TileOp:
    """A reduced attention shape: a key sweep whose body computes the per-key statistic and rsqrt
    feeding the score contraction's computed B operand cone. With ``shared_reader`` the sweep's
    own result also reads the rsqrt, so the chain does not die into the edge."""
    d, t = Axis("d", Dim(16)), Axis("t", Dim(8))
    stat = Loop(
        axis=d,
        body=Body(
            (
                Load(name="ksq", input="k", index=(Var("t"), Var("d"))),
                Assign(name="sq", op="multiply", args=("ksq", "ksq")),
                Accum(name="ss", value="sq", op="add", axes=("d",)),
            )
        ),
    )
    score = Loop(
        axis=d,
        body=Body(
            (
                Load(name="qv", input="q", index=(Var("m"), Var("d"))),
                Load(name="kv", input="k", index=(Var("t"), Var("d"))),
                Assign(name="kn", op="multiply", args=("kv", "inv")),
                Assign(name="prod", op="multiply", args=("qv", "kn")),
                Accum(name="acc", value="prod", op="add", axes=("d",)),
            )
        ),
    )
    mixed = (Assign(name="mixed", op="multiply", args=("acc", "inv")),) if shared_reader else ()
    sweep = Loop(
        axis=t,
        body=Body(
            (
                stat,
                Assign(name="inv", op="rsqrt", args=("ss",)),
                score,
                *mixed,
                Accum(name="mx", value="mixed" if shared_reader else "acc", op="maximum", axes=("t",)),
            )
        ),
    )
    return _lift(Body((Loop(axis=Axis("m", 4), body=Body((sweep, Write(output="out", index=(Var("m"),), value="mx")))),)))


def _score_and_b(tile: TileOp) -> tuple[Fold, Fold, Fold]:
    sweep = tile.op if tile.op.axis is not None else tile.op.operands[0]
    (score,) = (edge for edge in sweep.operands if edge.as_contraction() is not None)
    return sweep, score, score.operands[1]


def test_key_swept_statistic_closes_the_computed_b_operand() -> None:
    """The chain feeding only the score's B cone rides the edge, closed at its axes: the rsqrt is an
    operand cone of the B cone, which reads the statistic through it."""
    tile = _key_swept_score()

    _, score, b = _score_and_b(tile)
    assert isinstance(b, Fold) and b.axis is None
    assert {tile.axis_of(name).extent for name in b.free_axes} <= {Dim(4), Dim(8), Dim(16)}, b.free_axes  # m, t, d
    assert any(site.node.axis is not None for site in sites(b))
    assert TileOp(op=tile.op, place=tile.place, axes=tile.axes, output_specs=tile.output_specs).op is tile.op
    # A synthetic TileOp carries no graph output tensors, so the workspace dtype rule cannot
    # resolve a cuttable seam here; the reduced-target test below asserts the offered seam.
    assert id(b) in {id(site.node) for site in family_sites("PLACE", sites(tile.op))}


def test_key_swept_statistic_stays_when_a_sibling_reads_it() -> None:
    """A chain the sweep's own result also reads keeps its place in the sweep's step, and the B
    cone reads it through an operand cone of its own over the SAME statistic fold: a term is
    closed, so the cheap rsqrt is spelled where each reader stands, while the reduce it derives
    from is one shared object and never repeats."""
    tile = _key_swept_score(shared_reader=True)

    sweep, score, b = _score_and_b(tile)
    assert any(isinstance(stmt, Assign) and stmt.op == ElementwiseImpl("rsqrt") for stmt in sweep.lift.body)
    (inv,) = [edge for edge in b.operands if edge.as_slab() is None]
    assert any(isinstance(stmt, Assign) and stmt.op == ElementwiseImpl("rsqrt") for stmt in inv.lift.body)
    (stat,) = inv.operands
    assert stat.axis is not None and any(stat is edge for edge in sweep.operands), "the statistic fold is shared, not copied"


def test_normalization_shares_structurally_identical_cones() -> None:
    """The tree-wide invariant: after normalization, no two DISTINCT Fold objects in the tree are
    the same value with the same interface names — copies fusion inlined into several consumption
    sites (attention's softmax statistics, once in the weight cone and once in the epilogue) are
    one object, so placement sees one value and a composed cut materializes it once. Severed
    sharing is the recompute class PR #679 measured at three orders of magnitude."""
    tile = case_target_tile("attention/rmsnorm-qk-sdpa-composed-cut_xfail_realized.yaml")

    by_identity = {id(site.node): site.node for site in sites(tile.op)}
    by_value: dict[tuple, list[Fold]] = {}
    for node in by_identity.values():
        by_value.setdefault((node.canonical(), node.exposes), []).append(node)
    twins = [nodes for nodes in by_value.values() if len(nodes) > 1]
    assert not twins, "normalization left same-value cones as distinct objects"


def _statistic_under_two_binders() -> TileOp:
    """A row statistic feeding a contraction nested under a separate fold binder."""
    r, k, t = Axis("r", Dim(16)), Axis("k", Dim(16)), Axis("t", Dim(4))
    stat = reduction(r, (slab("xr", "x", "m", "r"),), (Assign(name="ss__v", op="multiply", args=("xr", "xr")),), ("ss",))
    # The statistic and its rsqrt are the B cone's SOURCE — an operand a projection evaluates
    # before its body, so the chain rides its own edge rather than a sibling body position.
    source = projection((stat,), (Assign(name="inv", op="rsqrt", args=("ss",)),), ("inv",))
    b_cone = projection((slab("wv", "w", "t", "k"), source), (Assign(name="wn", op="multiply", args=("wv", "inv")),), ("wn",))
    score = contraction(k, slab("qv", "q", "m", "k"), (b_cone, "acc"))
    sweep = reduction(t, (score,), (Assign(name="mx__v", op="copy", args=("acc",)),), ("mx",), "maximum")
    return _tile(projection((sweep,), results=("mx",)), r, k, t, free=(Axis("m", 4),))


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
    """Placement must preserve the statistic's evaluation multiplicity: a row statistic is evaluated
    once per row, ahead of every binder that reads it."""
    tile = _statistic_under_two_binders()

    scopes = [enclosing for name, enclosing in _loop_scopes(tile.op.lower(axes=tile.axes)) if name == "r"]
    assert scopes == [()], scopes


def test_a_fold_reading_a_sweep_axis_declares_it_on_the_operand_edge() -> None:
    """A per-column reduce whose streamed load reads the boundary store's sweep axis is an operand
    EDGE like any other term, and it DECLARES that axis.

    Body position used to carry this: a body member is re-wrapped inside the sweep ``Loop`` by
    reconstitution while an edge lowers at kernel scope, so a sweep-reading fold had to stay in the
    body or its column rendered as an undefined identifier (found live: DeepSeek-V4 post16's
    ``k_div_36``). A term cannot be a body member — a Fold tree composes through ``operands`` — so
    the fact is carried explicitly instead: the sweep axis is a free coordinate of the edge (its
    slab declares it), which is what lets the evaluation domain be recovered from the term rather
    than inferred from where it sits."""
    k, j = Axis("k", Dim(4)), Axis("j", Dim(4))
    fold = _reduce_loop(
        Load(name="xin", input="x", index=(Var("m"), Var("k"), Var("j"))),
        Accum(name="acc", value="xin", op="add", axes=("k",)),
        axis=k,
    )
    root = projection((fold,), (Assign(name="v", op="relu", args=("acc",)),), ("v",))

    tile = _tile(
        root,
        j,
        k,
        free=(M8,),
        output_specs=(OutputSpec(write=Write(output="out", index=(Var("m"), Var("j")), value="v"), sweep=(j,)),),
    )

    assert not any(isinstance(stmt, Fold) for stmt in tile.op.lift.body)
    (edge,) = tile.op.operands
    assert "j" in edge.free_axes, edge.free_axes


# ---- the bilinear form the lift builds ---------------------------------------------------------- #


def _channels(fold: Fold) -> tuple[Fold, ...]:
    """A contraction's streamed operands, one per channel in order — the edges after A."""
    return fold.operands[1:]


def test_contraction_clusters_alpha_equivalent_shared_operands() -> None:
    planar = _reduce_loop(
        Load(name="left0", input="x", index=(Var("m"), Var("k"))),
        Load(name="right0", input="w0", index=(Var("n"), Var("k"))),
        Assign(name="product0", op="multiply", args=("left0", "right0")),
        Load(name="left1", input="x", index=(Var("m"), Var("k"))),
        Load(name="right1", input="w1", index=(Var("n"), Var("k"))),
        Assign(name="product1", op="multiply", args=("left1", "right1")),
        Accum(name="acc0", value="product0", op="add", axes=("k",)),
        Accum(name="acc1", value="product1", op="add", axes=("k",)),
    )

    tile = _tile(planar, K32, free=(M8, N16))

    contraction = tile.op
    assert contraction.as_contraction() is not None
    assert len(contraction.combine.results) == 2
    assert _input(contraction.operands[0]) == "x"


def test_contraction_coalesces_overlapping_equivalent_shared_operands() -> None:
    planar = _reduce_loop(
        Load(name="left", input="x", index=(Var("m"), Var("k"))),
        Load(name="scale", input="s", index=(Var("k"),)),
        Assign(name="scaled0", op="multiply", args=("left", "scale")),
        Assign(name="scaled1", op="multiply", args=("left", "scale")),
        Load(name="right0", input="w0", index=(Var("n"), Var("k"))),
        Load(name="right1", input="w1", index=(Var("n"), Var("k"))),
        Assign(name="product0", op="multiply", args=("scaled0", "right0")),
        Assign(name="product1", op="multiply", args=("scaled1", "right1")),
        Accum(name="acc0", value="product0", op="add", axes=("k",)),
        Accum(name="acc1", value="product1", op="add", axes=("k",)),
    )

    tile = _tile(planar, K32, free=(M8, N16))

    assert tile.op.as_contraction() is not None
    assert len(tile.op.combine.results) == 2
    a = tile.op.operands[0]
    assert a.axis is None and a.exposes == ("scaled0",)
    assert sum(isinstance(stmt, Assign) and stmt.name.startswith("scaled") for stmt in a.lift.body) == 1


def test_contraction_orients_a_shared_commutative_argument_first() -> None:
    planar = _reduce_loop(
        Load(name="left0", input="x0", index=(Var("m"), Var("k"))),
        Load(name="left1", input="x1", index=(Var("m"), Var("k"))),
        Load(name="right", input="w", index=(Var("k"), Var("n"))),
        Assign(name="product0", op="multiply", args=("left0", "right")),
        Assign(name="product1", op="multiply", args=("left1", "right")),
        Accum(name="acc0", value="product0", op="add", axes=("k",)),
        Accum(name="acc1", value="product1", op="add", axes=("k",)),
    )

    tile = _tile(planar, K32, free=(M8, N16))

    view = tile.op.as_contraction()
    assert view is not None and (view.product.name, view.plus.name) == ("multiply", "add")
    assert _input(tile.op.operands[0]) == "w"
    assert [_input(edge) for edge in _channels(tile.op)] == ["x0", "x1"]
    assert TileOp(op=tile.op, place=tile.place, axes=tile.axes).op is tile.op


def test_contraction_computes_an_equivalent_channel_once() -> None:
    planar = _reduce_loop(
        Load(name="left", input="x", index=(Var("m"), Var("k"))),
        Load(name="right", input="w", index=(Var("n"), Var("k"))),
        Assign(name="product0", op="multiply", args=("left", "right")),
        Assign(name="product1", op="multiply", args=("left", "right")),
        Accum(name="acc0", value="product0", op="add", axes=("k",)),
        Accum(name="acc1", value="product1", op="add", axes=("k",)),
    )

    tile = _tile(planar, K32, free=(M8, N16))

    assert tile.op.as_contraction() is not None and len(tile.op.combine.results) == 2
    assert len(tile.op.operands) == 2, "two channels over one B occupy one operand slot"
    assert sum(isinstance(stmt, Load) and stmt.input == "w" for stmt in tile.op.lower(axes=tile.axes).iter()) == 1


def test_projection_keeps_only_the_maximal_shared_operand() -> None:
    small = projection(body=(Load(name="a", input="x", index=(Var("m"),)),), results=("a",))
    large = projection((small,), (Assign(name="b", op="copy", args=("a",)),), ("a", "b"))

    root = projection((small, large), (Assign(name="o", op="copy", args=("b",)),))

    assert root.operands == (large,)
    assert root.lift.params == ("a", "b")
    assert sum(isinstance(stmt, Load) and stmt.input == "x" for stmt in root.lower(axes=())) == 1


def test_semiring_merges_overlapping_operand_cones_into_one_multi_result_edge() -> None:
    planar = _reduce_loop(
        Load(name="left", input="x", index=(Var("m"), Var("k"))),
        Load(name="scale", input="s", index=(Var("k"),)),
        Assign(name="scaled", op="multiply", args=("left", "scale")),
        Load(name="right0", input="w0", index=(Var("k"), Var("n"))),
        Load(name="right1", input="w1", index=(Var("k"), Var("n"))),
        Assign(name="product0", op="multiply", args=("left", "right0")),
        Assign(name="product1", op="multiply", args=("scaled", "right1")),
        Accum(name="acc0", value="product0", op="add", axes=("k",)),
        Accum(name="acc1", value="product1", op="add", axes=("k",)),
    )

    tile = _tile(planar, K32, free=(M8, N16))

    assert tile.op.as_contraction() is not None
    shared = tile.op.operands[0]
    assert tuple(shared.exposes) == ("left", "scaled"), "the overlapping arguments are ONE multi-result A edge"
    assert sum(isinstance(stmt, Load) and stmt.input == "x" for stmt in tile.op.lower(axes=tile.axes).iter()) == 1


def _computed_matmul(*, computed_a: bool, computed_b: bool) -> TileOp:
    body: list = [Load(name="left", input="x", index=(Var("m"), Var("k")))]
    left = "left"
    if computed_a:
        body += [
            Load(name="left_scale", input="xs", index=(Var("k"),)),
            Assign(name="computed_left", op="multiply", args=(left, "left_scale")),
        ]
        left = "computed_left"
    body.append(Load(name="right", input="w", index=(Var("k"), Var("n"))))
    right = "right"
    if computed_b:
        body += [
            Load(name="right_scale", input="ws", index=(Var("k"),)),
            Assign(name="computed_right", op="multiply", args=(right, "right_scale")),
        ]
        right = "computed_right"
    body += [Assign(name="product", op="multiply", args=(left, right)), Accum(name="acc", value="product", op="add", axes=("k",))]
    return _tile(projection((_reduce_loop(*body),)), K32, free=(M8, N16))


def test_contraction_factors_a_computed_operand_cone() -> None:
    tile = _computed_matmul(computed_a=True, computed_b=False)

    assert tile.op.as_contraction() is not None
    a, b = tile.op.operands
    assert a.axis is None and a.as_slab() is None and a.exposes == ("computed_left",)
    assert _input(b) == "w"


def test_contraction_factors_b_computed_operand_cone() -> None:
    tile = _computed_matmul(computed_a=False, computed_b=True)

    assert tile.op.as_contraction() is not None
    a, b = tile.op.operands
    assert _input(a) == "x"
    assert b.axis is None and b.as_slab() is None and b.exposes == ("computed_right",)


def test_contraction_factors_both_computed_operand_cones_idempotently() -> None:
    tile = _computed_matmul(computed_a=True, computed_b=True)

    assert tile.op.as_contraction() is not None
    assert all(edge.as_slab() is None and edge.axis is None for edge in tile.op.operands)
    assert TileOp(op=tile.op, place=tile.place, axes=tile.axes).op is tile.op


def test_contraction_preserves_computed_operand_statement_order() -> None:
    """A computed A that reads a statistic through a chain lowers in dependency order: the row-max
    loop, the reciprocal, the value loop that reads it."""
    j, k = Axis("j", Dim(32)), Axis("k", Dim(64))
    row_max = Loop(
        axis=j,
        body=Body(
            (Load(name="score", input="s", index=(Var("m"), Var("j"))), Accum(name="row_max", value="score", op="maximum", axes=("j",)))
        ),
    )
    value = Loop(
        axis=j,
        body=Body(
            (
                Load(name="v", input="v", index=(Var("j"), Var("k"))),
                Assign(name="weighted", op="multiply", args=("inv", "v")),
                Accum(name="value", value="weighted", op="add", axes=("j",)),
            )
        ),
    )
    outer = Loop(
        axis=k,
        body=Body(
            (
                row_max,
                Assign(name="inv", op="reciprocal", args=("row_max",)),
                value,
                Load(name="weight", input="w", index=(Var("k"), Var("n"))),
                Assign(name="product", op="multiply", args=("value", "weight")),
                Accum(name="out", value="product", op="add", axes=("k",)),
            )
        ),
    )
    cell = Body((outer, Write(output="out", index=(Var("m"), Var("n")), value="out")))
    tile = _lift(Body((Loop(axis=M8, body=Body((Loop(axis=N16, body=cell),))),)))

    assert tile.op.as_contraction() is not None
    computed = tile.op.operands[0]
    lowered = computed.lower(axes=tile.axes)
    assert isinstance(lowered[0], Loop) and isinstance(lowered[-1], Loop), [type(stmt).__name__ for stmt in lowered]
    assert any(isinstance(stmt, Assign) and stmt.op.name == "reciprocal" for stmt in lowered[1:-1])
    assert TileOp(op=tile.op, place=tile.place, axes=tile.axes, output_specs=tile.output_specs).op is tile.op


def test_share_common_cones_unifies_internally_renamed_copies() -> None:
    """Alpha-equal cones with identical captures and interface names collapse to one object even
    when their internal binder spelling differs — plain structural equality would keep them
    distinct and silently sever the sharing."""

    def cone(load: str, product: str, row: str = "m") -> Fold:
        body = Body((Load(name=load, input="x", index=(Var(row), Var("k"))), Assign(name=product, op="multiply", args=(load, load))))
        return Fold(lift=Lambda.closing(("k",), body, (product,)), init=(0.0,), combine=Lambda.componentwise(("add",), ("acc",)))

    def consumer(inner: Fold, out: str) -> Fold:
        return projection((inner,), (Assign(name=out, op="exp", args=("acc",)),), (out,))

    def rooted(second: Fold) -> Fold:
        root = projection((consumer(cone("l1", "p1"), "u"), consumer(second, "v")), (Assign(name="w", op="add", args=("u", "v")),), ("w",))
        return _share_common_cones(root)

    unified = rooted(cone("l2", "p2"))
    assert unified.operands[0].operands[0] is unified.operands[1].operands[0]

    distinct = rooted(cone("l2", "p2", row="q"))  # same form, different capture — a different value
    assert distinct.operands[0].operands[0] is not distinct.operands[1].operands[0]


def test_total_lift_produces_canonical_contraction() -> None:
    inner = Body(
        (
            Load(name="left", input="x", index=(Var("m"), Var("k"))),
            Load(name="right", input="w", index=(Var("n"), Var("k"))),
            Assign(name="product", op="multiply", args=("left", "right")),
            Accum(name="acc", value="product", op="add", axes=("k",)),
        )
    )
    cell = Body((Loop(axis=K32, body=inner), Write(output="out", index=(Var("m"), Var("n")), value="acc")))
    tile = _lift(Body((Loop(axis=M8, body=Body((Loop(axis=N16, body=cell),))),)))

    assert tile.op.as_contraction() is not None
    assert tile.op.axis is not None


def _fed_chain_root(feed: bool) -> Fold:
    """A scalar chain beside a reduce whose step reads the chain's result when ``feed`` (the composed
    placement cut's consumer shape), lifted: the chain becomes the reduce's PROVIDER operand."""
    scale_arg = "v25" if feed else "left"
    loop = Loop(
        axis=Axis("k", Dim(8)),
        body=Body(
            (
                Load(name="left", input="x", index=(Var("k"),)),
                Assign(name="scaled", op="multiply", args=("left", scale_arg)),
                Accum(name="acc", value="scaled", op="add", axes=("k",)),
            )
        ),
    )
    chain = (Load(name="ws", input="cutbuf", index=(Literal(0, "int"),)), Assign(name="v25", op="rsqrt", args=("ws",)))
    return _lift(Body((*chain, loop, Write(output="out", index=(Literal(0, "int"),), value="acc"))))


def test_a_body_fed_fold_takes_its_provider_as_a_sibling_operand() -> None:
    """A fold whose subtree reads a name the scalar chain defines becomes an operand edge, and the
    chain becomes a PROVIDER operand ordered ahead of it.

    Body membership used to carry this: a projection evaluates its operands before its scalar body,
    so hoisting a fed fold emitted its capture as an undefined identifier (DeepSeek-V4 post4096's
    two-cut consumer piece). A term cannot be a body member, so the ordering is carried structurally
    instead — the lift closes the reduce over the chain as an operand cone, and ``Fold.lower``
    places every operand ahead of the term that reads it."""
    tile = _fed_chain_root(feed=True)
    out = tile.op

    assert not any(isinstance(stmt, Fold) for stmt in out.lift.body), "a term is never a body member"
    lowered = out.lower(axes=tile.axes)
    assert any(isinstance(stmt, Loop) for stmt in lowered), [type(stmt).__name__ for stmt in lowered]
    kinds = ["provider" if isinstance(stmt, Assign) and stmt.op.name == "rsqrt" else type(stmt).__name__ for stmt in lowered]
    assert kinds.index("provider") < kinds.index("Loop"), "the provider is emitted before its consumer"


def test_an_unfed_fold_is_an_operand_edge_too() -> None:
    """The same shape with the reduce reading only its own operand: also an edge. Position no longer
    varies with the capture, because position no longer encodes anything."""
    out = _fed_chain_root(feed=False).op

    assert out.axis is not None or any(edge.axis is not None for edge in out.operands)
    assert not any(isinstance(stmt, Fold) for stmt in out.lift.body)
