"""The Loop-IR → Tile-IR boundary fires for every kernel kind.

``lowering/tile/010_recognize`` is the sole recognizer that lifts a
``LoopOp`` into the tile IR (a ``Map`` / ``Fold`` / ``ContractionView``
node). These assert it fires on the two simplest kinds — pointwise and
reduce — transitively proving the axes got lifted and the kernel entered
the tile dialect (no planner / launch-geometry fallback needed), and that
the MONOID-producer composition (the fused norm→linear edge) nodifies to
a computed-A ``ContractionView`` fork sibling of the ``Map`` form.
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
from emmy.compiler.ir.tile import ContractionView, Map, TileOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import flatten_leaves
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
    from importlib import import_module

    recognize = import_module("emmy.compiler.pipeline.passes.lowering.tile.010_recognize")

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

    node, free = recognize._lift(list(body), "out")

    assert [axis.name for axis in free] == ["m", "n"]
    assert isinstance(node, Contraction) and node.a_computed
    assert {stmt.input for stmt in node.a_body if isinstance(stmt, Load)} == {"one_buf", "x"}
    assert isinstance(node.epilogue[0], Load) and node.epilogue[0].input == "bias_buf"


def test_lift_recognizes_contraction_between_views_of_same_packed_buffer():
    """Q and K can occupy different affine regions of one load-time-packed QKV buffer."""
    from importlib import import_module

    recognize = import_module("emmy.compiler.pipeline.passes.lowering.tile.010_recognize")

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

    node, free = recognize._lift(list(body), "score")

    assert [axis.name for axis in free] == ["m", "n"]
    assert isinstance(node, Contraction)
    assert isinstance(node.a_operand, Load)
    assert node.a_operand.input == node.b_load.input == "packed_qkv"


# --------------------------------------------------------------------------- #
# The MONOID-producer composition — ``rmsnorm(x)·nw @ w`` nodifies to a computed-A
# ``ContractionView`` fork sibling of the ``Map(source=Fold)`` form (``010_recognize``'s
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
        leaves = flatten_leaves(fp.options)
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
    return any("a:" in str(v) for v in row.values())


def test_wide_m1_flinear_uses_single_warp_k_fold():
    """A coalesced wide-K M=1 F.linear must not expose the dominated serial
    reduction to deploy selection, while tune retains the complete fork."""
    from dataclasses import replace

    ctx = Context.from_target((8, 9), gpu_name="NVIDIA GeForce RTX 4080")
    _, tile = _resolve(_m1_linear_graph(), ctx=ctx)
    reduce_specs = [v for k, v in tile.knobs.items() if k.startswith("REDUCE@")]
    assert reduce_specs == ["b32"]
    tune_rows, _ = _resolve(_m1_linear_graph(), ctx=replace(ctx, validate_pins=False))
    offered = {v for row in tune_rows for k, v in row.items() if k.startswith("REDUCE@")}
    assert "" in offered and "b32" in offered


def test_rtx4080_dit_qkv_narrows_to_measured_deploy_schedule():
    """The SKU-exact deploy keeps one stable computed-A QKV row; tune mode keeps
    the full legal search space."""
    from dataclasses import replace

    ctx = Context.from_target((8, 9), gpu_name="NVIDIA GeForce RTX 4080")
    rows, _ = _resolve(_norm_linear_graph(S=256, H=1152, inter=3456), pick=_is_warp_row, ctx=ctx)
    warp = [r for r in rows if _is_warp_row(r)]
    assert len(warp) == 1
    assert any(v == "a:mma_m16n8k16_f16_f32/w4x1/f2x4/k4" for k, v in warp[0].items() if k.startswith("TILE@"))
    assert any(v == "d1/sync" for k, v in warp[0].items() if k.startswith("STAGE@"))

    tune_rows, _ = _resolve(
        _norm_linear_graph(S=256, H=1152, inter=3456),
        pick=_is_warp_row,
        ctx=replace(ctx, validate_pins=False),
    )
    assert len([r for r in tune_rows if _is_warp_row(r)]) > 1


def test_norm_linear_offers_map_rows_then_warp_contraction_rows():
    """The merged fork: the ``Map``-form reduce rows lead (option-0 = the conservative coop pick,
    lowerable everywhere), then the computed-A ContractionView form's warp rows — every one riding a
    resolved ``sync`` compute-fill stage, at BOTH depths (``d1`` + the asymmetric B-ring ``d2``
    as fork siblings), with the K partition either decided-empty or a redundant-statistic
    split — deferred ``g<w>k`` or, this fixture's plain-store tail being distributive, the
    single-kernel atomic ``g<w>a`` (the single-channel computed-A split-K family)."""
    rows, _ = _resolve(_norm_linear_graph())
    assert rows, "no fork was offered for the fused norm→linear"
    assert not _is_warp_row(rows[0]), "option-0 must be the Map-form coop row, not a warp row"
    assert any(v.startswith("b") for v in rows[0].values() if isinstance(v, str)), "option-0 must cooperate on the stat reduce"
    warp = [r for r in rows if _is_warp_row(r)]
    assert warp, "the ContractionView form contributed no warp rows"
    stages_seen = set()
    reds_seen = set()
    for r in warp:
        stage = [v for k, v in r.items() if k.startswith("STAGE")]
        red = [v for k, v in r.items() if k == "REDUCE"]
        assert stage and all(v in ("d1/sync", "d2/sync") for v in stage), f"warp rows must ride the resolved sync compute-fill: {r}"
        assert all(v == "" or (v.startswith("g") and v.endswith(("k", "a"))) for v in red), (
            f"the computed-A form allows only the empty or split (g<w>k / g<w>a) K partition: {r}"
        )
        stages_seen.update(stage)
        reds_seen.update(red)
    # This fixture's weight is a graph INPUT, so the linear's B is transposed and the d2
    # B-only ring clamps back to d1 (nothing async to overlap) — d1 alone is correct here.
    # The canonical-B (constant-weight) shape class exercises d2 in
    # test_fused_cone_splitk_matches_reference.
    assert "d1/sync" in stages_seen, f"the resolved sync compute-fill must be offered: {stages_seen}"
    assert "" in reds_seen and any(v for v in reds_seen), f"both the serial and split K partitions must be offered: {reds_seen}"


def test_norm_linear_warp_pick_is_computed_a_contraction():
    """Picking a warp row materializes the recognize-built ``Map(body=projection, source=node)``
    tree — the same ``project ∘ contract`` spelling the Fold tiers use: the source is a
    computed-A :class:`ContractionView` holding its A cone INLINE, a real node tree
    (``Map(body=per-cell normalize, sources=(Fold(stat),))``), one channel, its (m, n)
    output on the grid (the column axis joined); the ``Map`` body carries the ``Write``; and the knob
    stamps the DB rows key on (``PLACE@cone`` + the decided-empty stat ``REDUCE``). Xfail-parked on
    the PLACE wipe — the ``PLACE@cone`` stamp returns with the phase-4 realizer."""
    _, tile = _resolve(_norm_linear_graph(), pick=_is_warp_row)
    assert isinstance(tile.op, Map)
    c = tile.op.sources[0]
    assert isinstance(c, ContractionView) and c.a_computed and len(tile.op.sources) == 1
    stat_loop = c.a.sources[0].sources[0].loop
    assert isinstance(stat_loop, Loop) and stat_loop.is_reduce and stat_loop.role is AxisRole.PLANAR
    assert [a.extent.as_static() for a in c.axes] == [32, 3072]
    assert c.axes[0].name == tile.place.grid[-2].name and c.axes[1].name == tile.place.grid[-1].name
    assert len(c.channels) == 1 and [type(s) for s in tile.op.body] == [Write]
    assert {"x", "wn", "w"} <= set(c.external_reads())
    assert c.stage is not None and c.stage.transport == "sync" and c.stage.bk_elems > 0
    assert tile.knobs.get("PLACE@cone") == "fuse"
    # Phase 3: the stamped keys are the CANONICAL codec spellings — bare for the primary product
    # fold, the explicit axis form for the cone's stat.
    assert tile.knobs.get(f"REDUCE@{stat_loop.axis.name}") == ""
    assert tile.knobs.get("TILE", "").startswith("a:")
    assert tile.knobs.get("STAGE") == "d1/sync"


def test_norm_linear_cone_is_an_inline_node_tree():
    """The computed-A cone lives ONCE, inline on an operand edge of the stored fold, as a real node
    tree: its SOURCE is
    the row-invariant prologue — the per-row statistic (a projected reduce over the stat
    :class:`Fold`) plus any k-invariant cone prefix — and its ``body`` is the per-cell
    normalize. The K seam is therefore the NODE BOUNDARY: ``ops.cone_seam`` reads it instead of
    re-scanning stmts for "the maximal leading run that never indexes K", and the statistic is
    addressable (and later cuttable) in its own right. Lowering flattens the whole thing back to the
    identical ``[stat loop, …, cone]`` stmt run. The stored form is the role=CONTRACTION fold; the
    ``ContractionView`` reading is the DERIVED view (``contraction_view``)."""
    from emmy.compiler.ir.tile import contraction_view
    from emmy.compiler.ir.tile.ir import _refs_axis
    from emmy.compiler.ir.tile.ops import cone_seam, lower

    _, tile = _resolve(_norm_linear_graph(), pick=_is_warp_row)
    grid = tile.place.grid
    # The single-channel form's projection was ONLY the root ``Write`` — moved to ``TileOp.stores``
    # (1q), so the row stores the BARE product fold (the ``Map`` wrapper dropped with its last stmt).
    fold = tile.op.sources[0] if isinstance(tile.op, Map) else tile.op
    assert len(tile.stores) == 1 and tile.stores[0].write.output == "y"
    c = contraction_view(fold, grid[-2], grid[-1], tuple(grid[:-2]))
    assert c is not None and c.a_computed
    cone = c.a
    assert isinstance(cone, Map) and cone.out == c.a_name
    assert cone.sources[0].sources[0].role is AxisRole.PLANAR, "the statistic reduce is the prologue's source"
    # The seam IS the boundary: prologue row-invariant, body k-varying, stats the bridged values.
    pro, cell, stats = cone_seam(cone)
    assert pro == tuple(lower(cone.sources[0])) and cell == tuple(cone.body)
    assert not any(_refs_axis(s, c.k_axis.name) for s in pro), "the prologue never indexes K — it runs once per row"
    assert any(_refs_axis(s, c.k_axis.name) for s in cell), "the per-cell body is the k-varying remainder"
    assert stats, "the statistic bridges through the stat smem rows"
    # The operand body is the flattened cone verbatim: the stat loop, its sweep, then the cone.
    assert c.a_body == tuple(lower(cone)) == (*pro, *cell)


def test_norm_linear_fp32_keeps_map_rows_only():
    """No 16-bit mma atom ⇒ the ContractionView form contributes ZERO rows (never a raising row) and
    the fork is exactly the Map-form reduce rows — the graceful fallback."""
    rows, tile = _resolve(_norm_linear_graph(dt=F32))
    assert rows and not any(_is_warp_row(r) for r in rows)
    assert isinstance(tile.op, Map)


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
    combine as projection — nodifies to ``Map(body=combine, sources=(ContractionView,))``: ONE
    product-carrier contraction, two ``(b, acc)`` channels over its single inline A cone (sharing
    is arity), the root ``Write`` a boundary ``Store`` (1q), and offers warp sync rows."""
    from emmy.compiler.ir.tile import contraction_view

    rows, tile = _resolve(_mlp_gate_up_graph(), pick=_is_warp_row)
    assert isinstance(tile.op, Map) and len(tile.op.sources) == 1
    grid = tile.place.grid
    node = contraction_view(tile.op.sources[0], grid[-2], grid[-1], tuple(grid[:-2]))
    assert node is not None and len(node.channels) == 2 and node.a_computed
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


def test_normed_gqa_sdpa_certifies_flash():
    """The fused flash form must certify when Q/K/V ride computed (RMSNorm) cones — the
    fusion-boundary guards keep the cones materialized (flash streams plain buffers) instead
    of letting them de-certify the unit: the P@V product merges with its sum-reduce FIRST
    (`_pending_contraction_half`), nothing compute-bearing fuses into the (future) offer site
    (`_sum_contracts_exp_producer` / `is_fold_offer_site`), and the score producer keeps
    plain-Load Q/K (`_feeds_softmax`). Pre-guards this graph lowered to a fragmented sdpa
    whose P@V wrote its full [b,h,m,n,d] outer product (Gemma finding 2)."""
    pytest.importorskip("torch")

    def decide(fp):
        return flatten_leaves(fp.options)[0]

    terminal, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(_normed_sdpa_graph(), decide)
    flash = [
        n.op
        for n in terminal.nodes.values()
        if type(n.op).__name__ == "TileOp"
        and isinstance(n.op.op, Map)
        and n.op.op.sources
        and getattr(n.op.op.sources[0], "role", None) is AxisRole.TWISTED
    ]
    assert flash, "no TWISTED flash kernel certified for the normed GQA sdpa"
    src = flash[0].op.sources[0]
    from emmy.compiler.ir.tile import is_contraction_fold

    assert is_contraction_fold(src.step_stmts()[0]), "flash did not absorb the score contraction (fold stayed cut)"


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


def _prologue_shape(*, b_layouts):
    """The recognized MONOID-producer shape: a per-row statistic reduce, its scalar sweep, and a
    column loop folding one ⊗-channel per entry of ``b_layouts`` over the shared normalized row."""
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
    from emmy.compiler.ir.tile import Fold

    m, n, k, r = Axis("m", 8), Axis("n", 16), Axis("k", 32), Axis("r", 32)
    fold = Accum(name="sacc", value="sq", op="add")
    stat = Fold(
        carrier=fold.as_carrier(),
        axis=r,
        step=Body((Load(name="x_e", input="x", index=(Var("m"), Var("r"))), Assign(name="sq", op="multiply", args=("x_e", "x_e")), fold)),
    )
    kbody = [Load(name="x_k", input="x", index=(Var("m"), Var("k"))), Assign(name="xh", op="multiply", args=("x_k", "rs"))]
    for i, trans in enumerate(b_layouts):
        idx = (Var("n"), Var("k")) if trans else (Var("k"), Var("n"))
        kbody += [
            Load(name=f"b{i}", input=f"w{i}", index=idx),
            Assign(name=f"v{i}", op="multiply", args=("xh", f"b{i}")),
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
    return Map(body=Body((Assign(name="rs", op="rsqrt", args=("sacc",)), nloop)), sources=(stat,)), (m,)


def test_channels_with_agreeing_b_layouts_form_one_product_node():
    from emmy.compiler.ir.tile import shared_operand
    from emmy.compiler.ir.tile.ir import _parse_bilinear
    from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_prologue_contraction

    node, free = _prologue_shape(b_layouts=(False, False))
    bound = bind_prologue_contraction(node, free)
    assert bound is not None
    c_map, _, _stores = bound
    (product,) = c_map.sources  # the stored role=CONTRACTION fold
    parsed = _parse_bilinear(product)
    assert parsed is not None and len(parsed[1]) == 2, "two components over ONE shared edge — sharing is edge reuse"
    assert not isinstance(shared_operand(product), type(None)) and shared_operand(product) is parsed[0]


def test_channels_with_disagreeing_b_layouts_never_group():
    """A group-formation GATE, not a node assert: the composition declines and the caller keeps the
    reduce ``Map`` form, rather than building a node whose channels cannot share a slab."""
    from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_prologue_contraction

    node, free = _prologue_shape(b_layouts=(False, True))
    assert bind_prologue_contraction(node, free) is None
