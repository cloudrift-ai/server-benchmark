"""The Loop-IR → Tile-IR boundary fires for every kernel kind.

``lowering/tile/010_recognize`` is the sole recognizer that lifts a
``LoopOp`` into the tile IR (a ``Map`` / ``Fold`` / bilinear ``Fold``
node). These assert it fires on the two simplest kinds — pointwise and
reduce — transitively proving the axes got lifted and the kernel entered
the tile dialect (no planner / launch-geometry fallback needed), and that
the MONOID-producer composition (the fused norm→linear edge) nodifies to
a computed-A bilinear ``Fold`` fork sibling of the ``Map`` form.
"""

from __future__ import annotations

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.frontend.ir import LinearOp, RmsNormOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure.fold import Fold, deep_defines, deep_reads, stmt_axis_names
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import LOOP_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import flatten_leaves
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import _same_program, fold_from_loop
from emmy.compiler.pipeline.pipeline import Run


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


def _lift_tree(body: Body):
    """The lift + classify read over a hand-built body, axis names preserved (``recognized_tile``
    goes through ``LoopOp`` normalization, which renames) — the same sequence the pass runs."""
    from emmy.compiler.ir.tile import split_effects
    from emmy.compiler.pipeline.passes.lowering.tile import _lift as L
    from emmy.compiler.pipeline.passes.lowering.tile._classify import classify

    free, cell = L._peel(Body(tuple(body)))
    cell = L._lift_cell(list(cell))
    split = split_effects(tuple(cell))
    cell, stores = (list(split[0]), split[1]) if split is not None else (cell, ())
    node = L._form_root(cell)
    free = L._order_free_by_output(node, free, stores)
    return classify(node, free), free, stores


def _fused(node, free):
    """The fused computed-A view over a hand-built tree — the derivation the schedule and the
    golden decode run (``fused_view`` reads a ``TileOp``)."""
    from emmy.compiler.ir.tile import Placement, TileOp
    from emmy.compiler.pipeline.passes.lowering.tile._classify import fused_view

    return fused_view(TileOp(op=node, place=Placement(free=tuple(free)), stores=()))


def test_recognize_fires_on_pointwise(recording_dump):
    g = Graph()
    _input(g, "x", (4, 8))
    g.add_node(op=ElementwiseOp("relu"), inputs=["x"], output=Tensor("o", (4, 8)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES).run(g, dump=recording_dump)
    assert "recognize" in recording_dump.fired_rules("lowering/tile")


def test_recognize_fires_on_reduction(recording_dump):
    g = Graph()
    _input(g, "x", (4, 8))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES).run(g, dump=recording_dump)
    assert "recognize" in recording_dump.fired_rules("lowering/tile")


def test_lift_partitions_independent_reduce_and_epilogue_preamble():
    """Independent loop-invariant values may feed opposite sides of a contraction.

    This is the shape of DiT's final MLP projection: GELU constants feed computed A
    inside K, while the linear bias feeds only the accumulator epilogue. Grouping the
    whole preamble together demotes the contraction to a scalar ``Map``.
    """
    m, n, k = (Axis(name, Dim(extent)) for name, extent in (("m", 32), ("n", 64), ("k", 128)))
    body = Body(
        (
            Load(name="one", input="one_buf", index=(Literal(0),)),
            Loop(
                axis=m,
                body=Body(
                    (
                        Loop(
                            axis=n,
                            body=Body(
                                (
                                    Load(name="bias", input="bias_buf", index=(Var("n"),)),
                                    Loop(
                                        axis=k,
                                        body=Body(
                                            (
                                                Load(name="xv", input="x", index=(Var("m"), Var("k"))),
                                                Assign(name="av", op=ElementwiseImpl("add"), args=("xv", "one")),
                                                Load(name="wv", input="w", index=(Var("n"), Var("k"))),
                                                Assign(name="prod", op=ElementwiseImpl("multiply"), args=("av", "wv")),
                                                Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
                                            )
                                        ),
                                    ),
                                    Assign(name="outv", op=ElementwiseImpl("add"), args=("acc", "bias")),
                                    Write(output="out", index=(Var("m"), Var("n")), value="outv"),
                                )
                            ),
                        ),
                    )
                ),
            ),
        )
    )

    node, free, stores = _lift_tree(body)

    assert [axis.name for axis in free] == ["m", "n"]
    # A projecting fold over the stored contraction whose shared A is the computed cone (the
    # GELU-constant preamble folded INSIDE K), the bias load riding the projection body — the
    # prologue routing kept the two independent feeds apart.
    from emmy.compiler.ir.pure.fold import operand_body

    assert isinstance(node, Fold) and node.axis is None and len(node.operands) == 1
    fold = node.operands[0]
    a_edge = fold.a
    assert a_edge is not None and not isinstance(a_edge, Load), "A must be the computed cone"
    a_loads = {st.input for st in operand_body(a_edge) if isinstance(st, Load)}
    assert a_loads == {"one_buf", "x"}
    assert isinstance(node.body[0], Load) and node.body[0].input == "bias_buf"


def test_lift_recognizes_contraction_between_views_of_same_packed_buffer():
    """Q and K can occupy different affine regions of one load-time-packed QKV buffer."""
    m, n, k = (Axis(name, Dim(extent)) for name, extent in (("m", 32), ("n", 32), ("k", 64)))
    body = Body(
        (
            Loop(
                axis=m,
                body=Body(
                    (
                        Loop(
                            axis=n,
                            body=Body(
                                (
                                    Loop(
                                        axis=k,
                                        body=Body(
                                            (
                                                Load(name="q", input="packed_qkv", index=(Var("m"), Var("k"))),
                                                Load(
                                                    name="key",
                                                    input="packed_qkv",
                                                    index=(Var("n"), Var("k") + Literal(64)),
                                                ),
                                                Assign(name="prod", op=ElementwiseImpl("multiply"), args=("q", "key")),
                                                Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
                                            )
                                        ),
                                    ),
                                    Write(output="score", index=(Var("m"), Var("n")), value="acc"),
                                )
                            ),
                        ),
                    )
                ),
            ),
        )
    )

    node, free, stores = _lift_tree(body)

    assert [axis.name for axis in free] == ["m", "n"]
    # Both views of the packed buffer hoist as materialized operand edges of the stored node.
    from emmy.compiler.ir.pure.fold import is_contraction

    con = node if is_contraction(node) else node.operands[0]
    assert is_contraction(con)
    a_edge = con.a
    assert isinstance(a_edge, Load)
    assert {e.input for e in (con.a, *(ch.b for ch in con.channels)) if isinstance(e, Load)} == {"packed_qkv"}


# --------------------------------------------------------------------------- #
# The MONOID-producer composition — ``rmsnorm(x)·nw @ w`` nodifies to a computed-A
# bilinear ``Fold`` fork sibling of the ``Fold.projection(source=Fold)`` form (``010_recognize``'s
# ``bind_prologue_contraction`` merge). Pipeline-only (no CUDA): resolve the tile passes with a
# capturing ``decide`` and assert the fork rows / the picked node's structure.
# --------------------------------------------------------------------------- #


def _norm_linear_graph(dt=F16, S: int | Dim = 32, H: int = 1024, inter: int = 3072) -> Graph:
    g = Graph()
    Sd = S if isinstance(S, Dim) else Dim(S)
    g.add_node(InputOp(), [], Tensor("x", (1, Sd, Dim(H)), dtype=dt), node_id="x")
    g.add_node(InputOp(), [], Tensor("wn", (Dim(H),), dtype=dt), node_id="wn")
    g.add_node(InputOp(), [], Tensor("w", (Dim(inter), Dim(H)), dtype=dt), node_id="w")
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (1, Sd, Dim(H)), dtype=dt), node_id="xn")
    g.add_node(LinearOp(), ["xn", "w"], Tensor("y", (1, Sd, Dim(inter)), dtype=dt), node_id="y")
    g.inputs, g.outputs = ["x", "wn", "w"], ["y"]
    return g


def _m1_linear_graph() -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, Dim(1), Dim(1152)), dtype=F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (Dim(1152), Dim(1152)), dtype=F16), node_id="w")
    g.add_node(LinearOp(), ["x", "w"], Tensor("y", (1, Dim(1), Dim(1152)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "w"], ["y"]
    return g


def _m1_contraction_tile(*, row: int | Var = 0, context: str | None = None) -> TileOp:
    """A direct unit-row contraction, optionally nested under an ordinary output axis."""
    n, k = Axis("n", Dim(16)), Axis("k", Dim(16))
    row_index = Literal(row) if isinstance(row, int) else row
    load_prefix = (Var(context),) if context is not None else ()
    store_prefix = (Var(context),) if context is not None else ()
    cell = Body(
        (
            Loop(
                axis=n,
                body=Body(
                    (
                        Loop(
                            axis=k,
                            body=Body(
                                (
                                    Load(name="a", input="x", index=(*load_prefix, Literal(0), Var("k"))),
                                    Load(name="b", input="w", index=(Var("n"), Var("k"))),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a", "b")),
                                    Accum(name="acc", value="p", op=ElementwiseImpl("add"), axes=("k",)),
                                )
                            ),
                        ),
                        Write(output="o", index=(*store_prefix, row_index, Var("n")), value="acc"),
                    )
                ),
            ),
        )
    )
    body = cell if context is None else Body((Loop(axis=Axis(context, Dim(8)), body=cell),))
    node, free, stores = _lift_tree(body)
    from emmy.compiler.ir.tile import Placement

    return TileOp(op=node, place=Placement(free=tuple(free)), stores=stores)


def test_unit_contraction_view_restores_only_the_literal_zero_output_row():
    from emmy.compiler.pipeline.passes.lowering.tile._classify import unit_contraction_view

    view = unit_contraction_view(_m1_contraction_tile())
    assert view is not None and [axis.name for axis in view[1]] == ["_um", "n"]
    assert unit_contraction_view(_m1_contraction_tile(row=1)) is None
    assert unit_contraction_view(_m1_contraction_tile(row=Var("m"))) is None


@pytest.mark.parametrize("context", ["batch", "head"])
def test_unit_contraction_view_does_not_reclassify_output_context(context):
    """A second output axis is not M unless a realized split receipt proves otherwise."""
    from emmy.compiler.pipeline.passes.lowering.tile._classify import unit_contraction_view

    tile = _m1_contraction_tile(context=context)
    assert [axis.name for axis in tile.place.free] == [context, "n"]
    assert unit_contraction_view(tile) is None


def _resolve(g: Graph, pick=None, ctx: Context | None = None) -> tuple[list[dict], TileOp]:
    """Run the tile passes, capturing every fork leaf's knob row; ``pick`` selects the applied
    leaf (default: option-0, the emission-order head). Returns ``(rows, the one TileOp)``."""
    rows: list[dict] = []

    def decide(fp):
        from emmy.compiler.pipeline.pipeline import _is_structural_option

        leaves = flatten_leaves(fp.options)
        # The placement fork (fused option-0 + cut fragments) precedes the schedule fork;
        # these tests assert on the SCHEDULE rows, so take the fused side and harvest nothing.
        if any(_is_structural_option(leaf) for leaf in leaves):
            return next(leaf for leaf in leaves if not _is_structural_option(leaf))
        rows.extend(dict(getattr(leaf, "knobs", {}) or {}) for leaf in leaves)
        if pick is not None:
            for leaf in leaves:
                if pick(dict(getattr(leaf, "knobs", {}) or {})):
                    return leaf
        return leaves[0]

    rg, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx or Context.from_target((12, 0))).resolve(g, decide)
    tiles = [n.op for n in rg.nodes.values() if isinstance(n.op, TileOp)]
    assert len(tiles) == 1, f"expected the fused norm→linear to be ONE kernel, got {len(tiles)}"
    return rows, tiles[0]


def _is_warp_row(row: dict) -> bool:
    # F1 site grammar: the tier discriminator is the row's ONE WORK entry, not an ``a:`` token.
    return str(row.get("WORK", "")).startswith("w")


def test_wide_m1_flinear_offers_the_warp_k_fold_and_a_pin_realizes_it(monkeypatch):
    """A coalesced wide-K M=1 F.linear OFFERS the 32-wide cooperative K fold beside the serial
    one, and a pin naming it realizes exactly that kernel.

    This used to assert that an unpinned deploy PICKED the coop fold — B's stored layout
    classified the shape and promoted the band ahead of the serial row. That ordering is gone:
    which of the two wins is measured evidence's answer, so an unpinned compile with none may take
    either, and the enumeration's obligation is only that both are reachable."""
    from emmy.compiler.pipeline.knob import family_of, family_value

    monkeypatch.delenv("EMMY_WORK", raising=False)
    monkeypatch.delenv("EMMY_REDUCE", raising=False)
    ctx = Context.from_target((8, 9), gpu_name="NVIDIA GeForce RTX 4080")
    rows, _ = _resolve(_m1_linear_graph(), ctx=ctx)
    offered = {v for row in rows for k, v in row.items() if family_of(k) == "REDUCE"}
    assert "" in offered and "coop" in offered, offered
    assert any(row.get("WORK") == "t32" and family_value(row, "REDUCE") == "coop" for row in rows), rows

    monkeypatch.setenv("EMMY_WORK", "t32")
    monkeypatch.setenv("EMMY_REDUCE", "coop")
    _, tile = _resolve(_m1_linear_graph(), ctx=ctx)
    assert [v for k, v in tile.knobs.items() if family_of(k) == "REDUCE"] == ["coop"]
    assert tile.knobs.get("WORK") == "t32"


def test_direct_m1_contraction_offers_scalar_and_volta_mma_rows():
    """Restoring the unit M axis adds the Volta tier without losing scalar schedules."""
    ctx = Context.from_target((7, 0), gpu_name="NVIDIA Tesla V100 SXM3 32GB")
    rows, _ = _resolve(_m1_linear_graph(), ctx=ctx)

    assert any(str(row.get("WORK", "")).startswith("t") for row in rows)
    assert any(str(row.get("WORK", "")).startswith("w") and "mma_m8n8k4_f16_f32" in str(row.get("TILE", "")) for row in rows)


def test_norm_linear_offers_both_the_map_rows_and_the_warp_contraction_rows():
    """The merged fork carries BOTH readings: the ``Map``-form reduce rows (lowerable everywhere)
    and the computed-A bilinear fold form's warp rows — every one riding a
    resolved ``sync`` compute-fill stage, at BOTH depths (``d1`` + the asymmetric B-ring ``d2``
    as fork siblings), with the K partition either decided-empty or a redundant-statistic
    split — deferred ``g<w>k`` or, this fixture's plain-store tail being distributive, the
    single-kernel atomic ``g<w>a`` (the single-channel computed-A split-K family).

    Membership, not position: this used to require the Map form's cooperative row to LEAD, because
    that was the row a prior-free compile deployed. Nothing leads any more, so the assertion is
    that each reading contributed rows and that each spells its own families correctly.

    Both readings spell the SAME key set — the contraction tree's, since that is the union's one
    namespace: bare ``TILE`` / ``STAGE`` / ``REDUCE`` for the product fold and ``@<stat axis>`` for
    the cone's statistic, each stamped as a DECIDED EMPTY where its reading has no such site."""
    rows, _ = _resolve(_norm_linear_graph())
    assert rows, "no fork was offered for the fused norm→linear"
    # The Map reading: the statistic fold cooperates (its width in WORK, F1 site grammar) and the
    # contraction's own families are stamped as decided empties.
    coop_map = [
        r
        for r in rows
        if not _is_warp_row(r)
        and any(isinstance(v, str) and v.startswith("coop") for v in r.values())
        and str(r.get("WORK", "")).startswith("t")
        and r["TILE"] == ""
        and r["STAGE"] == ""
        and r["REDUCE"] == ""
    ]
    assert coop_map, f"the Map reading contributed no cooperative stat-reduce row: {rows[:4]}"
    warp = [r for r in rows if _is_warp_row(r)]
    assert warp, "the bilinear fold form contributed no warp rows"
    stages_seen = set()
    reds_seen = set()
    for r in warp:
        stat = [(k, v) for k, v in r.items() if "@" in k]
        assert stat and all(v == "" for _, v in stat), f"the compute fill realizes the statistic itself — its site stays empty: {r}"
        assert r["STAGE"] in ("d1/smem", "d2/smem"), f"warp rows must ride the resolved reg compute-fill: {r}"
        assert r["REDUCE"] == "" or (r["REDUCE"].startswith("g") and r["REDUCE"].endswith(("k", "a"))), (
            f"the computed-A form allows only the empty or split (g<w>k / g<w>a) K partition: {r}"
        )
        stages_seen.add(r["STAGE"])
        reds_seen.add(r["REDUCE"])
    # This fixture's weight is a graph INPUT, so the linear's B is transposed and the d2
    # B-only ring clamps back to d1 (nothing async to overlap) — d1 alone is correct here.
    # The canonical-B (constant-weight) shape class exercises d2 in
    # test_fused_cone_splitk_matches_reference.
    assert "d1/smem" in stages_seen, f"the resolved reg compute-fill must be offered: {stages_seen}"
    assert "" in reds_seen and any(v for v in reds_seen), f"both the serial and split K partitions must be offered: {reds_seen}"


def test_norm_linear_cone_is_an_inline_node_tree():
    """The computed-A cone lives ONCE, inline on an operand edge of the stored fold, as a real node
    tree: its SOURCE is
    the row-invariant prologue — the per-row statistic (a projected reduce over the stat
    :class:`Fold`) plus any k-invariant cone prefix — and its ``body`` is the per-cell
    normalize. The K seam is therefore the NODE BOUNDARY: ``ops.cone_seam`` reads it instead of
    re-scanning stmts for "the maximal leading run that never indexes K", and the statistic is
    addressable (and later cuttable) in its own right. Lowering flattens the whole thing back to the
    identical ``[stat loop, …, cone]`` stmt run. The stored form is the role=CONTRACTION fold; the
    bilinear ``Fold`` reading is the PLACED stamp (``the placed reading``)."""
    from emmy.compiler.ir.pure.fold import operand_body, operand_name, refs_axis
    from emmy.compiler.ir.tile.ops import cone_seam

    _, tile = _resolve(_norm_linear_graph(), pick=_is_warp_row)
    # The single-channel form's projection was ONLY the root ``Write`` — moved to ``TileOp.stores``
    # (1q), so the row stores the BARE product fold (the ``Map`` wrapper drops with its last stmt).
    fold = tile.op
    assert fold.role is AxisRole.CONTRACTION, "an empty identity projection must not wrap the product fold"
    assert len(tile.stores) == 1 and tile.stores[0].write.output == "y"
    c = fold
    assert not isinstance(c.a, Load)
    cone = c.a
    assert (isinstance(cone, Fold) and cone.axis is None) and cone.out == operand_name(c.a)
    assert cone.operands[0].operands[0].role is AxisRole.PLANAR, "the statistic reduce is the prologue's source"
    # The seam IS the boundary: prologue row-invariant, body k-varying, stats the bridged values.
    pro, cell, stats = cone_seam(cone, c.axis.name)
    assert pro == tuple(cone.operands[0].lower()) and cell == tuple(cone.body)
    assert not any(refs_axis(s, c.axis.name) for s in pro), "the prologue never indexes K — it runs once per row"
    assert any(refs_axis(s, c.axis.name) for s in cell), "the per-cell body is the k-varying remainder"
    assert stats, "the statistic bridges through the stat smem rows"
    # The operand body is the flattened cone verbatim: the stat loop, its sweep, then the cone.
    assert operand_body(c.a) == tuple(cone.lower()) == (*pro, *cell)


def _attention_cone_term() -> tuple[Fold, Fold]:
    """The attention shape of the computed-A cone, built directly: the ``softmax(Q·Kᵀ)·V``
    contraction over the KV axis whose A cone is ``exp(s − m)·(1/d)`` over a COMPUTED score — one
    edge for the row statistic (the twisted ``(m, d)`` pair) and one for the per-cell score
    contraction ``s = Σ_d Q·K``. Returns ``(root, cone)``."""
    from emmy.compiler.ir.pure import Lambda
    from emmy.compiler.ir.pure.carrier import exp_combine_states
    from emmy.compiler.ir.pure.fold import Channel

    def score(kv_name: str, dd: Axis, acc: str) -> Fold:
        q = Load(names=(f"{acc}__q",), input="q", index=(Var("m"), Var(dd.name)))
        k = Load(names=(f"{acc}__k",), input="k", index=(Var(kv_name), Var(dd.name)))
        return Fold.contraction(k_axis=dd, a=q, channels=(Channel(b=k, acc=acc),))

    names, other = ("mx", "dn"), ("mx__o", "dn__o")
    stat = Fold(
        axis=Axis("kv", Dim(32)),
        operands=(score("kv", Axis("dd", Dim(16)), "s1"),),
        lift=Lambda(params=("kv", "s1"), body=Body(()), results=("s1", 1.0)),
        init=(float("-inf"), 0.0),
        combine=Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names),
    )
    # The prologue passes ``mx`` through beside its own ``rd``: the cell binds both positionally
    # (``make_cone``'s closure rule), so no λ in the tree captures.
    prologue = Fold.projection(body=Body((Assign(name="rd", op="reciprocal", args=("dn",)),)), operands=(stat,), results=("rd", "mx"))
    cone = Fold.projection(
        body=Body(
            (
                Assign(name="ex0", op="subtract", args=("s2", "mx")),
                Assign(name="ex1", op="exp", args=("ex0",)),
                Assign(name="pw", op="multiply", args=("rd", "ex1")),
            )
        ),
        operands=(prologue, score("kvb", Axis("ddb", Dim(16)), "s2")),
    )
    v = Load(names=("vl",), input="v", index=(Var("kvb"), Var("n")))
    pv = Fold.contraction(k_axis=Axis("kvb", Dim(32)), a=cone, channels=(Channel(b=v, acc="o"),))
    return Fold.projection(body=Body(()), operands=(pv,)), cone


def test_cone_per_cell_edge_is_evaluated_inline_and_carries_no_slice():
    """A cone may carry more than the one row-invariant statistic edge: attention's score
    contraction is a PER-CELL producer its ``exp(s − m)`` reads. The K seam splits the EDGES the
    same way it splits the stmts — a k-invariant edge is the prologue (run once per tile row), a
    k-varying one is per-cell and splices into the cell ahead of its first use — so the fill
    computes the score instead of reading a name nothing defines.

    Such an edge is evaluated INLINE from lowered loop IR, so no ``TILE`` / ``REDUCE`` / ``STAGE``
    slice can address it and none is offered (a value there would be unrealizable, and the rows
    carrying it would emit the identical kernel). It stays a ``PLACE`` seam: cutting it is exactly
    how the score becomes a kernel of its own — the two-kernel form evidence prices against this
    one."""
    from emmy.compiler.ir.pure.fold import refs_axis
    from emmy.compiler.ir.tile.ops import cone_seam
    from emmy.compiler.ir.tile.path import family_sites, sites

    root, cone = _attention_cone_term()
    pv = root.operands[0]
    pro, cell, stats = cone_seam(cone, pv.axis.name)
    assert pro == tuple(cone.operands[0].lower()), "the k-invariant statistic edge is the prologue"
    assert not any(refs_axis(s, pv.axis.name) for s in pro)
    assert cell[0] == cone.operands[1].lower()[0], "the score edge leads the per-cell cell"
    assert stats == ("mx", "rd"), f"the statistic bridges through the stat smem rows: {stats}"
    # Every name the cell reads is defined by the cell, the bridged stats, or an axis.
    defined = {nm for s in cell for nm in deep_defines(s)} | set(stats) | stmt_axis_names(cell) | {"m", "n", pv.axis.name}
    assert deep_reads(list(cell)) <= defined, f"the fill reads an undefined name: {deep_reads(list(cell)) - defined}"

    all_sites = sites(root)
    (inline,) = [s for s in all_sites if s.inline]
    assert inline.node is cone.operands[1] and inline.axis == "ddb"
    for family in ("TILE", "REDUCE", "STAGE"):
        assert not [s for s in family_sites(family, all_sites) if s.node is inline.node], f"{family} must not address an inline node"
    assert [s for s in family_sites("PLACE", all_sites) if s.node is inline.node], "the score edge stays a cuttable seam"
    # …and the statistic edge keeps its own reduce site — it is realized per tile ROW, not per cell.
    stat = cone.operands[0].operands[0]
    assert any(s.node is stat for s in family_sites("REDUCE", all_sites))


@pytest.mark.parametrize("shape", ["sdpa", "norm_linear"])
def test_cone_cell_lambda_is_closed(shape):
    """Every λ in a stored term is CLOSED: the per-cell normalize reads the statistic it
    normalizes against (softmax's ``m``, RMSNorm's ``rsqrt``) through the prologue's RESULTS,
    bound positionally, never as a captured name. The seam between statistic and normalize is
    then a positional edge like every other; what bridges through the stat smem rows is exactly
    that edge's results."""
    from emmy.compiler.ir.tile.ops import cone_seam
    from emmy.compiler.pipeline.passes.lowering.tile._cut import _captured_values, cuttable_seams

    _, tile = _resolve_sdpa(is_causal=True) if shape == "sdpa" else _resolve(_norm_linear_graph(), pick=_is_warp_row)
    fold = tile.op.operands[0] if (isinstance(tile.op, Fold) and tile.op.axis is None) else tile.op
    cone = fold.a
    prologue = cone.operands[0]
    assert "captures" not in tile.pretty_body()
    if shape == "sdpa":  # ``exp(s − m)`` reads the carrier's ``m`` itself; RMSNorm's cell reads only the projected rsqrt
        from emmy.compiler.ir.pure.fold import is_contraction, operand_name

        score_edges = cone.operands[1:]
        assert len(score_edges) == 1 and is_contraction(score_edges[0]), "the cell has one computed score operand"
        score = score_edges[0]
        assert cone.lift.params == (*prologue.lift.results, operand_name(score)), (
            "bridged statistics and the computed score bind positionally"
        )
        groups = [cut for cut in cuttable_seams(tile.op, tile.stores, tile.place.free) if len(cut.members) == 2]
        assert len(groups) == 1 and any(member.node is score for member in groups[0].members), (
            "the closed computed score must be one use of the grouped placement inverse"
        )
        assert set(prologue.operands[0].combine.results) & set(prologue.lift.results), "the statistic's own state passes through"
    else:
        assert cone.lift.params == prologue.lift.results, "the cell binds every prologue result positionally"
    _, _, stats = cone_seam(cone, fold.axis.name)
    assert set(stats) == set(prologue.lift.results), f"the bridge is the prologue's results: {stats} vs {prologue.lift.results}"
    axes = stmt_axis_names(cone.lower()) | {a.name for a in (*tile.place.free, *tile.place.grid)} | {fold.axis.name}
    assert not _captured_values(cone, axes) and not _captured_values(prologue, axes)


def test_cone_per_cell_edge_reaches_the_per_cell_emitter():
    """The same edge on the untiled tier: the emitter's node-walk lowers EVERY operand edge of a
    zero-axis node, so the score's own reduce loop is emitted ahead of the cell that reads it.
    Walking only the first edge left the cell reading a name nothing defined — nvcc's
    ``identifier "s2" is undefined``."""
    from emmy.compiler.pipeline.passes.lowering.kernel._factor import Ctx, _emit

    _root, cone = _attention_cone_term()
    body = Body(tuple(_emit(cone, Ctx(grid=())).body))
    defined = {nm for s in body for nm in deep_defines(s)} | stmt_axis_names(body) | {"m", "n", "kvb"}
    assert "s2" in defined, "the cone's score edge never reached the emitted body"
    assert deep_reads(list(body)) <= defined, f"the emitted body reads an undefined name: {deep_reads(list(body)) - defined}"


# --------------------------------------------------------------------------------------------- #
# The COMPOSED STEP — a reduce whose per-element step reads a producer it computes itself
# (attention's per-key score contraction ``Σ_d Q·K`` inside the streaming softmax statistic).
# --------------------------------------------------------------------------------------------- #


def _score_loop(kv: str, dd: str, acc: str, tag: str) -> Loop:
    """``Σ_dd Q[m, dd]·K[kv, dd]`` as a raw reduce ``Loop`` — the per-key score, spelled the way
    loop fusion leaves it when it splices the QK producer into its consumer's step."""
    return Loop(
        axis=Axis(dd, Dim(16)),
        body=Body(
            (
                Load(names=(f"kl{tag}",), input="k", index=(Var(kv), Var(dd))),
                Load(names=(f"ql{tag}",), input="q", index=(Var("m"), Var(dd))),
                Assign(name=f"pr{tag}", op="multiply", args=(f"kl{tag}", f"ql{tag}")),
                Accum(name=acc, value=f"pr{tag}", op="add", axes=(dd,)),
            )
        ),
    )


def _composed_rowmax() -> Loop:
    """The row-max pass over a COMPUTED score: ``max_kv (Σ_d Q·K)·scale``."""
    return Loop(
        axis=Axis("kv", Dim(32)),
        body=Body(
            (
                _score_loop("kv", "d0", "s0", "0"),
                Assign(name="sc0", op="multiply", args=("s0", 0.25)),
                Accum(name="mx", value="sc0", op="maximum", axes=("kv",)),
            )
        ),
    )


def _composed_sumexp() -> Loop:
    """The ``Σ exp(score − mx)`` pass over its own copy of the same computed score."""
    return Loop(
        axis=Axis("kv", Dim(32)),
        body=Body(
            (
                _score_loop("kv", "d1", "s1", "1"),
                Assign(name="sc1", op="multiply", args=("s1", 0.25)),
                Assign(name="df", op="subtract", args=("sc1", "mx")),
                Assign(name="ex", op="exp", args=("df",)),
                Accum(name="dn", value="ex", op="add", axes=("kv",)),
            )
        ),
    )


def test_composed_step_reads_its_producer_as_an_operand_edge():
    """A reduce whose step computes what it folds reads as a fold with the producer on an OPERAND
    EDGE — not as the raw-loop escape. The lift binds the producer's carried state positionally,
    and ``fold_from_loop``'s byte-identity gate proves the reading: the derived step re-places the
    edge ahead of its first use and flattens it back to the identical nest."""
    loop = _composed_rowmax()
    fold = fold_from_loop(loop)
    assert fold is not None, "the composed step fell to the raw-loop escape"
    (edge,) = fold.operands
    assert isinstance(edge, Fold) and edge.axis.name == "d0", "the score producer is an operand edge"
    assert fold.lift.params == ("kv", "s0"), f"the lift binds the producer's state positionally: {fold.lift.params}"
    assert [type(s).__name__ for s in fold.lift.body] == ["Assign"], "only the scale survives in the lift body"
    # Role-blind: the ``AxisRole`` is a DERIVED read, so the re-derived producer carries one where
    # the raw pre-annotation loop does not. The program is what has to match.
    assert _same_program(fold.loop.body, loop.body), "the derived loop is not the captured program"


def test_composed_step_keeps_a_row_invariant_prologue_ahead_of_its_producer():
    """An edge is PLACED at its first use, not prepended: a loop-invariant scalar ahead of the
    producer keeps its position, so the byte-identity gate still accepts. Prepending unconditionally
    read this step as a different program and cost it every schedule tier."""
    loop = _composed_rowmax()
    body = Body((Load(names=("sv",), input="scale", index=(Literal(0, "int"),)), *loop.body))
    fold = fold_from_loop(Loop(axis=loop.axis, body=body))
    assert fold is not None and _same_program(fold.loop.body, body)


def test_online_softmax_pairs_two_composed_passes():
    """The pairing compares the two passes' score cones by CONTENT, so two separately-traced copies
    of one score — different bound axis, different temps — pair and fuse into ONE twisted stream.
    Without that the fused attention cell keeps three passes over the score instead of two."""
    from emmy.compiler.pipeline.passes.lowering.tile._classify import pair_softmax

    fmax, fsum = fold_from_loop(_composed_rowmax()), fold_from_loop(_composed_sumexp())
    assert fmax is not None and fsum is not None
    node = pair_softmax(Fold.projection(body=Body(()), operands=(fmax, fsum)))
    assert len(node.operands) == 1, "the pair must collapse to one TWISTED stream"
    tw = node.operands[0]
    assert tw.role is AxisRole.TWISTED
    assert len(tw.operands) == 1 and tw.operands[0].axis is not None, "the fused stream keeps ONE score producer"


def test_split_k_reindexes_the_cones_producer_edge():
    """A split partition σ-reindexes the cone's K-VARYING producer edge, not only its body stmts:
    the slice's own k coordinate reaches gmem THROUGH that node. Leaving it alone makes every
    partition recompute partition 0's scores (a silently wrong result, not a slow one)."""
    from emmy.compiler.ir.expr import BinaryExpr
    from emmy.compiler.ir.sigma import Sigma
    from emmy.compiler.pipeline.passes.lowering.tile._schedule import _sliced_edge

    _root, cone = _attention_cone_term()
    sigma = Sigma({"kvb": BinaryExpr("+", Var("_ks"), Var("kvb"))})
    sliced = _sliced_edge(cone, sigma, "kvb")
    score = sliced.operands[1]
    assert [e.pretty() for ld in score.lower()[0].body if isinstance(ld, Load) for e in ld.index if "_ks" in e.pretty()], (
        "the producer edge kept partition 0's k coordinate"
    )
    assert sliced.operands[0] == cone.operands[0], "the row-invariant statistic stays FULL-ROW in every partition"


def test_norm_linear_fp32_keeps_map_rows_only():
    """No 16-bit mma atom ⇒ the bilinear fold form contributes ZERO rows (never a raising row) and
    the fork is exactly the Map-form reduce rows — the graceful fallback."""
    rows, tile = _resolve(_norm_linear_graph(dt=F32))
    assert rows and not any(_is_warp_row(r) for r in rows)
    assert isinstance(tile.op, Fold) and tile.op.axis is None


def test_norm_linear_symbolic_m_offers_warp_rows():
    """A symbolic seq axis (masked M) stays eligible — the sync fill clamps the row coordinate, so
    only N/K demand exact cover."""
    rows, _ = _resolve(_norm_linear_graph(S=Dim("seq_len")))
    assert any(_is_warp_row(r) for r in rows)


def _mlp_gate_up_graph(S: int = 32) -> Graph:
    H, inter = 1024, 3072
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, Dim(S), Dim(H)), dtype=F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("wn", (Dim(H),), dtype=F16), node_id="wn")
    g.add_node(InputOp(), [], Tensor("wg", (Dim(inter), Dim(H)), dtype=F16), node_id="wg")
    g.add_node(InputOp(), [], Tensor("wu", (Dim(inter), Dim(H)), dtype=F16), node_id="wu")
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (1, Dim(S), Dim(H)), dtype=F16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("gate", (1, Dim(S), Dim(inter)), dtype=F16), node_id="gate")
    g.add_node(LinearOp(), ["xn", "wu"], Tensor("up", (1, Dim(S), Dim(inter)), dtype=F16), node_id="up")
    g.add_node(ElementwiseOp("silu"), ["gate"], Tensor("sg", (1, Dim(S), Dim(inter)), dtype=F16), node_id="sg")
    g.add_node(ElementwiseOp("multiply"), ["sg", "up"], Tensor("o", (1, Dim(S), Dim(inter)), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["x", "wn", "wg", "wu"], ["o"]
    return g


def _materialized_gate_up_graph(S: int = 32, *, up_dtype=F16) -> Graph:
    """Two projections over one materialized activation, followed by SwiGLU."""
    H, inter = 256, 512
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, Dim(S), Dim(H)), dtype=F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("wg", (Dim(inter), Dim(H)), dtype=F16), node_id="wg")
    g.add_node(InputOp(), [], Tensor("wu", (Dim(inter), Dim(H)), dtype=up_dtype), node_id="wu")
    g.add_node(LinearOp(), ["x", "wg"], Tensor("gate", (1, Dim(S), Dim(inter)), dtype=F16), node_id="gate")
    g.add_node(LinearOp(), ["x", "wu"], Tensor("up", (1, Dim(S), Dim(inter)), dtype=F16), node_id="up")
    g.add_node(ElementwiseOp("silu"), ["gate"], Tensor("sg", (1, Dim(S), Dim(inter)), dtype=F16), node_id="sg")
    g.add_node(ElementwiseOp("multiply"), ["sg", "up"], Tensor("o", (1, Dim(S), Dim(inter)), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["x", "wg", "wu"], ["o"]
    return g


def test_materialized_gate_up_offers_one_shared_a_mma_fill_and_scalar_fallback():
    """One materialized A may feed compatible gate/up MMA channels through one sync fill."""
    from emmy.compiler.ir.pure.fold import is_contraction
    from emmy.compiler.pipeline.knob import family_value

    rows, tile = _resolve(_materialized_gate_up_graph(), pick=_is_warp_row, ctx=Context.from_target((7, 0)))
    node = tile.op.operands[0] if isinstance(tile.op, Fold) and tile.op.axis is None else tile.op
    assert is_contraction(node)
    assert isinstance(node.a, Load) and len(node.channels) == 2
    assert {ch.b.input for ch in node.channels} == {"wg", "wu"}
    warp = [row for row in rows if _is_warp_row(row)]
    assert warp and any(not _is_warp_row(row) for row in rows), "the original planar fallback remains a sibling"
    assert {family_value(row, "STAGE") for row in warp} == {"d1/smem", "d2/smem"}
    assert all(family_value(row, "STAGE") for row in warp), "multi-channel MMA cannot use the single-fold gmem-direct path"
    assert all(str(family_value(row, "TILE")).startswith("mma_m8n8k4_f16_f32/") for row in warp)


def test_materialized_gate_up_refuses_a_byte_transport_pin(monkeypatch):
    """A `STAGE` pin naming a copy transport refuses on a product with several B channels.

    The compute fill is mandatory here and the byte-transport emitters carry one channel, so there
    is no row for the pin to name. The packed-pair (NVFP4) weight cone is the one node shape that
    does offer those rows beside the fill, and it has a single channel by construction — which is
    why the resolver's exception reads the node rather than the pin's transport."""
    monkeypatch.setenv("EMMY_STAGE", "d2/smem-async")
    with pytest.raises(ValueError, match="no smem-async sibling"):
        _resolve(_materialized_gate_up_graph(), ctx=Context.from_target((12, 0)))


def test_materialized_gate_up_mixed_b_dtypes_keeps_scalar_fallback_only():
    """A byte-copied B channel whose dtype differs from the atom must decline the MMA fill."""
    rows, _tile = _resolve(_materialized_gate_up_graph(up_dtype=F32), ctx=Context.from_target((7, 0)))
    assert rows and not any(_is_warp_row(row) for row in rows)


def _bind_materialized_channels(*, b_layouts=("kn", "kn"), mix_a_axis=False):
    """Bind the minimal two-channel product fold, or return ``None`` when it is ineligible."""
    from emmy.compiler.pipeline.passes.lowering.tile._classify import bind_bilinear
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import _stamp_axes

    body = [
        Load(
            name="av",
            input="x",
            index=(Var("m") + Var("n"), Var("k")) if mix_a_axis else (Var("m"), Var("k")),
        )
    ]
    for i, layout in enumerate(b_layouts):
        index = (Var("k"), Var("n")) if layout == "kn" else (Var("n"), Var("k"))
        body.extend(
            (
                Load(name=f"b{i}", input=f"w{i}", index=index),
                Assign(name=f"p{i}", op="multiply", args=("av", f"b{i}")),
                Accum(name=f"acc{i}", value=f"p{i}", op="add", axes=("k",)),
            )
        )
    fold = fold_from_loop(_stamp_axes(Loop(axis=Axis("k", Dim(32)), body=Body(tuple(body)))))
    assert fold is not None
    return bind_bilinear(fold, "m", "n", frozenset({"m", "n"}))


def test_materialized_channels_with_disagreeing_b_layouts_decline():
    """One shared A fragment cannot feed B slabs with different K orientations."""
    assert _bind_materialized_channels(b_layouts=("kn", "nk")) is None


def test_materialized_channels_with_mixed_role_axis_decline():
    """An A index mixing both output axes has no addressable MMA slab role."""
    assert _bind_materialized_channels(mix_a_axis=True) is None


def test_mlp_gate_up_nodifies_as_two_channel_product_contraction():
    """The fused gate/up MLP edge — TWO ⊗-folds sharing one normalized-row A value (fusion
    duplicates the cone SSA per fold; the matcher dedupes by value-tree equality) with the SwiGLU
    combine as projection — nodifies to ``Fold.projection(body=combine, operands=(bilinear fold,))``: ONE
    product-carrier contraction, two ``(b, acc)`` channels over its single inline A cone (sharing
    is arity), the root ``Write`` a boundary ``Store`` (1q), and offers warp sync rows."""

    rows, tile = _resolve(_mlp_gate_up_graph(), pick=_is_warp_row)
    assert (isinstance(tile.op, Fold) and tile.op.axis is None) and len(tile.op.operands) == 1
    node = tile.op.operands[0]
    assert len(node.channels) == 2 and (not isinstance(node.a, Load))
    assert {ch.b.input for ch in node.channels} == {"wg", "wu"}
    # The projection body is PURE (the SwiGLU combine); the root store rides ``TileOp.stores``.
    assert all(s.pure for s in tile.op.body) and not isinstance(tile.op.body[-1], Write)
    assert len(tile.stores) == 1 and tile.stores[0].write.output == "o"
    assert any(_is_warp_row(r) for r in rows)
    assert any(not _is_warp_row(r) for r in rows), "the per-cell reading must stay reachable beside the warp rows"


def _normed_sdpa_graph():
    """Gemma-shaped attention, torch-traced: RMSNorm'd Q/K/V (computed operand cones), GQA
    (16 q / 8 kv heads), causal, head_dim 256 — the fusion-boundary case where the flash unit
    used to fragment (the V-norm fused into the P@V product first, the halves merge then
    tripped the work-blowup guard, and the P@V materialized its full weight×V outer product)."""
    from emmy.commands.trace import graph_from_code

    norm = "(lambda t: t*torch.rsqrt((t.float()*t.float()).mean(-1,keepdim=True)+1e-6).to(t.dtype))"
    code = (
        f"torch.nn.functional.scaled_dot_product_attention({norm}(torch.randn(1,16,128,256,dtype=torch.float16)), "
        f"{norm}(torch.randn(1,8,128,256,dtype=torch.float16)), {norm}(torch.randn(1,8,128,256,dtype=torch.float16)), "
        "is_causal=True, enable_gqa=True)"
    )
    graph, _, _ = graph_from_code(code)
    return graph


def test_bind_contraction_declined_cone_raises_not_positional():
    """When the ⊗ lift names a COMPUTED A whose cone declines the bind (here: an n-indexed
    load riding the cone), the binder DECLINES — the fold keeps its PLANAR reading, which
    computes the full body. Falling through to the positional first-(m,k)-load rule instead
    binds a cone-INTERNAL load as A and silently drops the rest of the cone (the gemma GeGLU
    wrong-kernel class the lift binding fixed)."""
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.elementwise import ElementwiseImpl
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.stmt import Accum, Assign, Body, Load
    from emmy.compiler.pipeline.passes.lowering.tile._classify import bind_bilinear
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import _stamp_axes

    m, n, k = Var("m"), Var("n"), Var("k")
    body = Body(
        (
            Load(name="bv", input="B", index=(n, k)),
            Load(name="av", input="A", index=(m, k)),
            Assign(name="cv", op=ElementwiseImpl("exp"), args=("av",)),
            Load(name="nv", input="Z", index=(n, k)),
            Assign(name="cw", op=ElementwiseImpl("multiply"), args=("cv", "nv")),
            Assign(name="pv", op=ElementwiseImpl("multiply"), args=("cw", "bv")),
            Accum(name="acc", op=ElementwiseImpl("add"), value="pv"),
        )
    )
    loop = Loop(axis=Axis(name="k", extent=Dim(64)), body=body)
    fold = fold_from_loop(_stamp_axes(loop))
    assert fold is not None
    assert bind_bilinear(fold, "m", "n", frozenset({"m", "n"})) is None
    assert fold.role is AxisRole.PLANAR


# --------------------------------------------------------------------------- #
# Group formation — the ``b_trans`` gate. Channels whose B operands are stored the
# other way round were never legally fusable (one shared A fragment, one slab
# orientation), so they simply never group: recognition declines the composition and
# the reduce ``Map`` form stands alone. Driven directly on ``bind_prologue_contraction``
# (the frontend always emits canonical B, so the disagreeing shape has no graph).
# --------------------------------------------------------------------------- #


def _prologue_shape(*, b_layouts, cone_per_channel=False):
    """The recognized MONOID-producer shape: a per-row statistic reduce, its scalar sweep, and a
    column loop folding one ⊗-channel per entry of ``b_layouts`` over the shared normalized row.
    ``cone_per_channel`` spells the fusion-duplicated form: each channel carries its OWN copy of
    the normalize cone (fresh SSA names, and odd copies commute the multiply's args)."""
    from emmy.compiler.ir.axis import Axis, AxisRole
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.pure.fold import Fold
    from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write

    m, n, k, r = Axis("m", 8), Axis("n", 16), Axis("k", 32), Axis("r", 32)
    fold = Accum(name="sacc", value="sq", op="add", axes=("r",))
    stat = fold_from_loop(
        Loop(
            axis=r,
            body=Body(
                (Load(name="x_e", input="x", index=(Var("m"), Var("r"))), Assign(name="sq", op="multiply", args=("x_e", "x_e")), fold)
            ),
            role=AxisRole.PLANAR,
        )
    )
    assert stat is not None
    kbody = (
        []
        if cone_per_channel
        else [Load(name="x_k", input="x", index=(Var("m"), Var("k"))), Assign(name="xh", op="multiply", args=("x_k", "rs"))]
    )
    for i, trans in enumerate(b_layouts):
        idx = (Var("n"), Var("k")) if trans else (Var("k"), Var("n"))
        xh = "xh"
        if cone_per_channel:
            xh = f"xh{i}"
            args = (f"x_k{i}", "rs") if i % 2 == 0 else ("rs", f"x_k{i}")
            kbody += [
                Load(name=f"x_k{i}", input="x", index=(Var("m"), Var("k"))),
                Assign(name=xh, op="multiply", args=args),
            ]
        kbody += [
            Load(name=f"b{i}", input=f"w{i}", index=idx),
            Assign(name=f"v{i}", op="multiply", args=(xh, f"b{i}")),
            Accum(name=f"acc{i}", value=f"v{i}", op="add"),
        ]
    accs = tuple(f"acc{i}" for i in range(len(b_layouts)))
    tail = (Assign(name="y", op="multiply", args=accs),) if len(accs) > 1 else ()
    nloop = Loop(
        axis=n,
        body=Body(
            (Loop(axis=k, body=Body(tuple(kbody))), *tail, Write(output="o", index=(Var("m"), Var("n")), value="y" if tail else accs[0]))
        ),
    )
    return Fold.projection(body=Body((Assign(name="rs", op="rsqrt", args=("sacc",)), nloop)), operands=(stat,)), (m,)


def test_channels_with_agreeing_b_layouts_form_one_product_node():
    from emmy.compiler.ir.tile.ops import cone_seam

    node, free = _prologue_shape(b_layouts=(False, False))
    bound = _fused(node, free)
    assert bound is not None
    c_map, _, _stores = bound
    (product,) = c_map.operands  # the stored contraction node
    assert product.role is AxisRole.CONTRACTION
    assert len(product.channels) == 2, "two channels over ONE shared edge — sharing is the node's arity"
    assert product.a is not None
    prologue, _cell, stats = cone_seam(product.a, product.axis.name)
    assert stats == ("rs",)
    assert sum(isinstance(s, Load) and s.input == "x" for s in Body(prologue).iter()) == 1


def test_channels_with_disagreeing_b_layouts_never_group():
    """A group-formation GATE, not a node assert: the composition declines and the caller keeps the
    reduce ``Map`` form, rather than building a node whose channels cannot share a slab."""
    node, free = _prologue_shape(b_layouts=(False, True))
    assert _fused(node, free) is None


def test_duplicated_cone_with_commuted_args_still_shares_one_a():
    """Gate-free loop fusion inlines the producer cone once PER channel — fresh SSA names, and one
    copy may spell a commutative op's args the other way round (``x̂·s`` vs ``s·x̂``). Value-tree
    equality must key both copies equal, or the composition declines and the fused kernel demotes
    to the scalar tier (found live: the gemma-4 geglu edge's recorded goldens drifted in-model)."""
    node, free = _prologue_shape(b_layouts=(False, False), cone_per_channel=True)
    bound = _fused(node, free)
    assert bound is not None, "the per-channel cone copies key equal — ONE shared A operand"
    c_map, _, _stores = bound
    (product,) = c_map.operands
    assert len(product.channels) == 2, "both ⊗-folds grouped over the one shared cone"


# --------------------------------------------------------------------------- #
# The MASKED score cone — SDPA's ``softmax(Q·Kᵀ + mask)·V``. A coordinate-predicated
# ``Select`` in the score is an ordinary pure stmt of the cone, so the masked region must
# reach the SAME computed-A contraction the unmasked one reaches.
# --------------------------------------------------------------------------- #


def _sdpa_graph(*, is_causal: bool, B: int = 1, H: int = 2, S: int = 64, D: int = 32) -> Graph:
    from emmy.compiler.ir.frontend.ir import SdpaOp

    g = Graph()
    for name in ("q", "k", "v"):
        g.add_node(InputOp(), [], Tensor(name, (Dim(B), Dim(H), Dim(S), Dim(D)), dtype=F16), node_id=name)
    g.add_node(SdpaOp(is_causal=is_causal), ["q", "k", "v"], Tensor("o", (Dim(B), Dim(H), Dim(S), Dim(D)), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["q", "k", "v"], ["o"]
    return g


def _resolve_sdpa(is_causal: bool) -> tuple[list[dict], TileOp]:
    """Run the tile passes over an SDPA graph, picking the warp row wherever one is offered.
    Returns ``(the softmax·V node's fork rows — empty when the cell took the flat zero-axis
    escape and no schedule fork was offered at all, its TileOp)``."""
    rows: dict[str, list[dict]] = {}

    def decide(fp):
        from emmy.compiler.pipeline.pipeline import _is_structural_option

        leaves = flatten_leaves(fp.options)
        if any(_is_structural_option(leaf) for leaf in leaves):
            return next(leaf for leaf in leaves if not _is_structural_option(leaf))
        harvest = [dict(getattr(leaf, "knobs", {}) or {}) for leaf in leaves]
        rows.setdefault(fp.node_id, []).extend(harvest)
        return next((leaf for leaf, row in zip(leaves, harvest, strict=True) if _is_warp_row(row)), leaves[0])

    rg, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(_sdpa_graph(is_causal=is_causal), decide)
    return rows.get("o", []), rg.nodes["o"].op


@pytest.mark.parametrize("is_causal", [False, True])
def test_masked_sdpa_reaches_the_computed_a_contraction(is_causal):
    """SDPA's ``softmax·V`` half binds as ONE computed-A contraction over the ``(m, d)`` statistic
    — masked as well as unmasked. The mask is a ``Select`` on the score, which the twisted λ read
    and the MAP cone both carry, so the fork offers the mma tier either way. When they did not, the
    causal region kept the flat zero-axis ``Fold`` escape: a knob-less serial kernel, two orders of
    magnitude off the split it replaced, on the shape every decoder-only model uses."""
    rows, tile = _resolve_sdpa(is_causal)
    assert any(_is_warp_row(r) for r in rows), "the masked softmax·V must offer the warp/mma tier, not just the scalar escape"
    assert any("mma" in str(r.get("TILE", "")) for r in rows), "no mma TILE was offered for the softmax·V contraction"
    assert tile.knobs.get("TILE", "").startswith("mma"), f"the picked row must realize the contraction: {tile.knobs.get('TILE')!r}"


def test_masked_score_cone_keeps_its_predicate_per_cell():
    """The mask ``Select`` reads the contraction axis through its PREDICATE, not an index, so the
    K seam must place it in the per-cell body. Hoisted into the row-invariant prologue it would be
    evaluated once per row, at a coordinate that is not even bound there."""
    from emmy.compiler.ir.stmt import Select
    from emmy.compiler.ir.tile.ops import cone_seam

    _, tile = _resolve_sdpa(is_causal=True)
    fold = tile.op.operands[0] if (isinstance(tile.op, Fold) and tile.op.axis is None) else tile.op
    assert fold.role is AxisRole.CONTRACTION and not isinstance(fold.a, Load)
    pro, cell, _stats = cone_seam(fold.a, fold.axis.name)
    assert any(isinstance(s, Select) for s in cell), "the mask predicates the per-cell weight"
    assert not any(isinstance(s, Select) for s in pro), "the mask is k-varying — it never joins the row prologue"


def test_normed_q_k_scores_bind_both_computed_cones_with_a_cuttable_b_seam():
    """Gemma-shaped attention scores: RMSNorm'd Q against RMSNorm'd K. Root formation chains the
    score fold over both statistics' projected scales; the binder reads it as ONE contraction
    whose A and B are both computed cones, each sourcing its own statistic. The ``b`` seam —
    the normalized keys — is then a legal cut: the form that turns the per-query replay of the
    key statistic into a materialized operand the mma tier streams."""
    from emmy.compiler.ir.pure.fold import is_contraction
    from emmy.compiler.ir.tile.path import spell
    from emmy.compiler.pipeline.passes.lowering.tile._classify import fused_view
    from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams
    from emmy.compiler.pipeline.passes.lowering.tile._lift import recognized_tile

    g = Pipeline.build(LOOP_PASSES).run(_normed_sdpa_graph())
    scores = next(n for n in g.nodes.values() if isinstance(n.op, LoopOp) and n.id.endswith("_scaled"))
    tile = recognized_tile(scores.op, name=scores.id)
    pro = fused_view(tile)
    assert pro is not None, "the chained score column binds through the fused view"
    node = pro[0].operands[0]
    assert is_contraction(node) and isinstance(node.a, Fold) and isinstance(node.channels[0].b, Fold), "both operands computed"
    seams = [spell(pro[0], "PLACE", s.node) for s in cuttable_seams(pro[0], pro[2], (*tile.place.free, *pro[1]))]
    assert "PLACE@b" in seams, seams


def _m1_norm_gate_up_body() -> Body:
    """One row of norm → gate/up → SiLU·up as loop fusion leaves it: the row statistic, its
    epilogue, the SiLU's fill constant (``one``) loaded ahead of the column sweep, and the
    combine tail reading it after the two-channel contraction."""
    lit0 = Literal(0, "int")
    stat = Loop(
        axis=Axis("k0", Dim(64)),
        body=Body(
            (
                Load(names=("xs",), input="x", index=(lit0, lit0, Var("k0"))),
                Assign(name="xf", op="copy", args=("xs",), dtype=F32),
                Assign(name="sq", op="multiply", args=("xf", "xf")),
                Accum(name="ss", value="sq", op="add", axes=("k0",)),
            )
        ),
    )
    col = Loop(
        axis=Axis("k", Dim(64)),
        body=Body(
            (
                Load(names=("xc",), input="x", index=(lit0, lit0, Var("k"))),
                Load(names=("wn",), input="w", index=(Var("k"),)),
                Assign(name="xcf", op="copy", args=("xc",), dtype=F32),
                Assign(name="wnf", op="copy", args=("wn",), dtype=F32),
                Assign(name="sc", op="multiply", args=("rs", "xcf")),
                Assign(name="nm", op="multiply", args=("sc", "wnf")),
                Assign(name="nh", op="copy", args=("nm",), dtype=F16),
                Load(names=("g",), input="wg", index=(Var("k"), Var("n"))),
                Assign(name="pg", op="multiply", args=("g", "nh")),
                Accum(name="ag", value="pg", op="add", axes=("k",)),
                Load(names=("u",), input="wu", index=(Var("k"), Var("n"))),
                Assign(name="pu", op="multiply", args=("nh", "u")),
                Accum(name="au", value="pu", op="add", axes=("k",)),
            )
        ),
    )
    sweep = Loop(
        axis=Axis("n", Dim(128)),
        body=Body(
            (
                col,
                Assign(name="ng", op="negative", args=("ag",)),
                Assign(name="eg", op="exp", args=("ng",)),
                Assign(name="dn", op="add", args=("one", "eg")),
                Assign(name="sg", op="reciprocal", args=("dn",)),
                Assign(name="si", op="multiply", args=("ag", "sg")),
                Assign(name="o", op="multiply", args=("au", "si")),
                Write(output="out", index=(lit0, lit0, Var("n")), value="o"),
            )
        ),
    )
    return Body(
        (
            stat,
            Load(names=("cnt",), input="count", index=(lit0,)),
            Assign(name="mean", op="divide", args=("ss", "cnt")),
            Load(names=("eps",), input="eps", index=(lit0,)),
            Assign(name="ve", op="add", args=("eps", "mean")),
            Assign(name="rs", op="rsqrt", args=("ve",)),
            Load(names=("one",), input="silu_one", index=(lit0,)),
            sweep,
        )
    )


def test_fused_view_tail_prefix_is_alpha_renamed_from_the_statistic_prologue():
    """At one row the fused gate/up reading prepends the tail's stat-free cone stmts (the
    SiLU's fill constant) so the store side re-evaluates them — a SECOND spelling of names the
    statistic prologue also defines. A PLACE cut flattens both into one raw loop body (the
    parent piece), so the copies must carry their own names or the piece defines a name twice
    and the cut is refused (Qwen3.8 ``PLACE@a0=cut`` on norm→gate/up M=1)."""
    from emmy.compiler.ir.tile import Placement, TileOp
    from emmy.compiler.ir.tile.ir import effect_tail
    from emmy.compiler.pipeline.passes.lowering.tile._classify import fused_view
    from emmy.compiler.pipeline.passes.lowering.tile._cut import _nest
    from emmy.compiler.pipeline.passes.lowering.tile._lift import recognized_tile

    tile = recognized_tile(LoopOp(body=_m1_norm_gate_up_body()))
    pro = fused_view(TileOp(op=tile.op, place=Placement(free=tuple(tile.place.free)), stores=tuple(tile.stores)))
    assert pro is not None, "the one-row norm→gate/up must bind the fused (computed-A) reading"
    tree, added_axes, stores = pro
    assert any(s.name.endswith("__p") for s in tree.body if isinstance(s, Load)), [s for s in tree.body]
    # The parent piece of any cut is this flattening; it must be a valid single-scope program.
    LoopOp(body=Body(tuple(_nest(effect_tail(tree.lower(), stores), [*tile.place.free, *added_axes]))))
