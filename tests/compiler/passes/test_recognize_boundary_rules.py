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
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from emmy.compiler.ir.tile import Fold, TileOp
from emmy.compiler.ir.tile.ir import deep_defines, deep_reads, stmt_axis_names
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import flatten_leaves
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
from emmy.compiler.pipeline.pipeline import Run


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


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
    from emmy.compiler.pipeline.passes.lowering.tile import _lift as lift_mod

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

    node, free, stores = lift_mod._lift(list(body), "out")

    assert [axis.name for axis in free] == ["m", "n"]
    # The λ-era spelling: a projecting Map over the stored role=CONTRACTION fold whose shared A
    # is the computed cone (the GELU-constant preamble folded INSIDE K), the bias load riding the
    # projection body — the preamble split kept the two independent feeds apart.
    from emmy.compiler.ir.tile.ir import operand_body

    assert isinstance(node, Fold) and node.axis is None and len(node.operands) == 1
    fold = node.operands[0]
    a_edge = fold.a
    assert a_edge is not None and not isinstance(a_edge, Load), "A must be the computed cone"
    a_loads = {st.input for st in operand_body(a_edge) if isinstance(st, Load)}
    assert a_loads == {"one_buf", "x"}
    assert isinstance(node.body[0], Load) and node.body[0].input == "bias_buf"


def test_lift_recognizes_contraction_between_views_of_same_packed_buffer():
    """Q and K can occupy different affine regions of one load-time-packed QKV buffer."""
    from emmy.compiler.pipeline.passes.lowering.tile import _lift as lift_mod

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

    node, free, stores = lift_mod._lift(list(body), "score")

    assert [axis.name for axis in free] == ["m", "n"]
    # Both views of the packed buffer hoist as materialized operand edges of the stored node.
    from emmy.compiler.ir.tile.ir import is_contraction

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


def test_wide_m1_flinear_uses_single_warp_k_fold():
    """A coalesced wide-K M=1 F.linear must not expose the dominated serial
    reduction to deploy selection, while tune retains the complete fork."""
    from dataclasses import replace

    from emmy.compiler.pipeline.knob import family_of

    ctx = Context.from_target((8, 9), gpu_name="NVIDIA GeForce RTX 4080")
    _, tile = _resolve(_m1_linear_graph(), ctx=ctx)
    # F1 site grammar: the coop WIDTH rides the ONE WORK entry; the REDUCE value is site-local.
    reduce_specs = [v for k, v in tile.knobs.items() if family_of(k) == "REDUCE"]
    assert reduce_specs == ["coop"]
    assert tile.knobs.get("WORK") == "t32"
    tune_rows, _ = _resolve(_m1_linear_graph(), ctx=replace(ctx, validate_pins=False))
    offered = {v for row in tune_rows for k, v in row.items() if family_of(k) == "REDUCE"}
    assert "" in offered and "coop" in offered
    assert any(row.get("WORK") == "t32" for row in tune_rows)


def test_norm_linear_offers_map_rows_then_warp_contraction_rows():
    """The merged fork: the ``Map``-form reduce rows lead (option-0 = the conservative coop pick,
    lowerable everywhere), then the computed-A bilinear fold form's warp rows — every one riding a
    resolved ``sync`` compute-fill stage, at BOTH depths (``d1`` + the asymmetric B-ring ``d2``
    as fork siblings), with the K partition either decided-empty or a redundant-statistic
    split — deferred ``g<w>k`` or, this fixture's plain-store tail being distributive, the
    single-kernel atomic ``g<w>a`` (the single-channel computed-A split-K family).

    Both readings spell the SAME key set — the contraction tree's, since that is the union's one
    namespace: bare ``TILE`` / ``STAGE`` / ``REDUCE`` for the product fold and ``@<stat axis>`` for
    the cone's statistic, each stamped as a DECIDED EMPTY where its reading has no such site."""
    rows, _ = _resolve(_norm_linear_graph())
    assert rows, "no fork was offered for the fused norm→linear"
    assert not _is_warp_row(rows[0]), "option-0 must be the Map-form coop row, not a warp row"
    # F1: a coop partition spells the site value ``coop`` with its width in the WORK entry.
    assert any(isinstance(v, str) and v.startswith("coop") for v in rows[0].values()), "option-0 must cooperate on the stat reduce"
    assert str(rows[0].get("WORK", "")).startswith("t"), "the coop width rides the WORK inventory"
    assert rows[0]["TILE"] == "" and rows[0]["STAGE"] == "" and rows[0]["REDUCE"] == "", (
        f"the Map reading must stamp the contraction's families as decided empties: {rows[0]}"
    )
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
    from emmy.compiler.ir.tile.ir import operand_body, operand_name, refs_axis
    from emmy.compiler.ir.tile.ops import cone_seam

    _, tile = _resolve(_norm_linear_graph(), pick=_is_warp_row)
    # The single-channel form's projection was ONLY the root ``Write`` — moved to ``TileOp.stores``
    # (1q), so the row stores the BARE product fold (the ``Map`` wrapper dropped with its last stmt).
    fold = tile.op.operands[0] if (isinstance(tile.op, Fold) and tile.op.axis is None) else tile.op
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
    from emmy.compiler.ir.stmt import Lambda
    from emmy.compiler.ir.stmt.carrier import exp_combine_states
    from emmy.compiler.ir.tile import Channel

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
    prologue = Fold.projection(body=Body((Assign(name="rd", op="reciprocal", args=("dn",)),)), operands=(stat,))
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
    from emmy.compiler.ir.tile.ir import refs_axis
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


def _mlp_gate_up_graph() -> Graph:
    S, H, inter = 32, 1024, 3072
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
    assert not _is_warp_row(rows[0]), "option-0 stays the coop reduce row"


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
    load riding the cone), ``bind_contraction`` must raise ``LoweringError`` — the recognizer
    then demotes the cell to PLANAR, which computes the full body. Falling through to the
    positional first-(m,k)-load rule instead binds a cone-INTERNAL load as A and silently
    drops the rest of the cone (the gemma GeGLU wrong-kernel class the lift binding fixed)."""
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.elementwise import ElementwiseImpl
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.stmt import Accum, Assign, Body, Load
    from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_contraction
    from emmy.compiler.pipeline.pipeline import LoweringError

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
    loop = Loop(axis=Axis(name="k", extent=Dim(64)), body=body, role=AxisRole.CONTRACTION)
    with pytest.raises(LoweringError, match="computed cone"):
        bind_contraction(loop, "m", "n", Body(()))


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
    from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
    from emmy.compiler.ir.tile import Fold

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
    from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_prologue_contraction

    node, free = _prologue_shape(b_layouts=(False, False))
    bound = bind_prologue_contraction(node, free)
    assert bound is not None
    c_map, _, _stores = bound
    (product,) = c_map.operands  # the stored contraction node
    assert product.role is AxisRole.CONTRACTION
    assert len(product.channels) == 2, "two channels over ONE shared edge — sharing is the node's arity"
    assert product.a is not None


def test_channels_with_disagreeing_b_layouts_never_group():
    """A group-formation GATE, not a node assert: the composition declines and the caller keeps the
    reduce ``Map`` form, rather than building a node whose channels cannot share a slab."""
    from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_prologue_contraction

    node, free = _prologue_shape(b_layouts=(False, True))
    assert bind_prologue_contraction(node, free) is None


def test_duplicated_cone_with_commuted_args_still_shares_one_a():
    """Gate-free loop fusion inlines the producer cone once PER channel — fresh SSA names, and one
    copy may spell a commutative op's args the other way round (``x̂·s`` vs ``s·x̂``). Value-tree
    equality must key both copies equal, or the composition declines and the fused kernel demotes
    to the scalar tier (found live: the gemma-4 geglu edge's recorded goldens drifted in-model)."""
    from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_prologue_contraction

    node, free = _prologue_shape(b_layouts=(False, False), cone_per_channel=True)
    bound = bind_prologue_contraction(node, free)
    assert bound is not None, "the per-channel cone copies key equal — ONE shared A operand"
    c_map, _, _stores = bound
    (product,) = c_map.operands
    assert len(product.channels) == 2, "both ⊗-folds grouped over the one shared cone"
