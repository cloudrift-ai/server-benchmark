"""The Loop-IR → Tile-IR boundary fires for every kernel kind.

``lowering/tile/010_recognize`` is the sole recognizer that lifts a
``LoopOp`` into the tile IR (a ``Map`` / ``Reduction`` / ``Contraction``
node). These assert it fires on the two simplest kinds — pointwise and
reduce — transitively proving the axes got lifted and the kernel entered
the tile dialect (no planner / launch-geometry fallback needed), and that
the MONOID-producer composition (the fused norm→linear edge) nodifies to
a computed-A ``Contraction`` fork sibling of the ``Map`` form.
"""

from __future__ import annotations

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp, RmsNormOp
from emmy.compiler.ir.stmt import Loop, Write
from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from emmy.compiler.ir.tile import Contraction, Map, TileOp
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


# --------------------------------------------------------------------------- #
# The MONOID-producer composition — ``rmsnorm(x)·nw @ w`` nodifies to a computed-A
# ``Contraction`` fork sibling of the ``Map(source=Reduction)`` form (``010_recognize``'s
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


def _resolve(g: Graph, pick=None) -> tuple[list[dict], TileOp]:
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

    rg, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(g, decide)
    tiles = [n.op for n in rg.nodes.values() if isinstance(n.op, TileOp)]
    assert len(tiles) == 1, f"expected the fused norm→linear to be ONE kernel, got {len(tiles)}"
    return rows, tiles[0]


def _is_warp_row(row: dict) -> bool:
    return any("a:" in str(v) for v in row.values())


def test_norm_linear_offers_map_rows_then_warp_contraction_rows():
    """The merged fork: the ``Map``-form reduce rows lead (option-0 = the conservative coop pick,
    lowerable everywhere), then the computed-A Contraction form's warp rows, each riding the
    mandatory resolved ``sync`` compute-fill stage with the contraction K partition decided-empty."""
    rows, _ = _resolve(_norm_linear_graph())
    assert rows, "no fork was offered for the fused norm→linear"
    assert not _is_warp_row(rows[0]), "option-0 must be the Map-form coop row, not a warp row"
    assert any(v.startswith("b") for v in rows[0].values() if isinstance(v, str)), "option-0 must cooperate on the stat reduce"
    warp = [r for r in rows if _is_warp_row(r)]
    assert warp, "the Contraction form contributed no warp rows"
    for r in warp:
        stage = [v for k, v in r.items() if k.startswith("STAGE@")]
        red = [v for k, v in r.items() if k.startswith("REDUCE@")]
        assert stage == ["d1/sync"], f"warp rows must ride the resolved sync compute-fill: {r}"
        assert all(v == "" for v in red), f"the computed-A form has no reduce partition: {r}"


def test_norm_linear_warp_pick_is_computed_a_contraction():
    """Picking a warp row materializes the recognize-built ``Map(body=projection, source=node)``
    tree — the same ``project ∘ contract`` spelling the Reduction tiers use: the source is a
    computed-A :class:`Contraction` whose A cone is headed by the annotated PLANAR statistic reduce
    ``Loop``, one fold channel, its (m, n) output on the grid (the column axis joined); the ``Map``
    body carries the ``Write``; and the knob stamps the DB rows key on (``PLACE@cone`` + the
    decided-empty stat ``REDUCE``)."""
    _, tile = _resolve(_norm_linear_graph(), pick=_is_warp_row)
    assert isinstance(tile.op, Map)
    c = tile.op.source
    assert isinstance(c, Contraction) and c.a_computed and len(c.folds) == 1
    stat_loop = c.a_operand[0]
    assert isinstance(stat_loop, Loop) and stat_loop.is_reduce and stat_loop.role is AxisRole.PLANAR
    assert [a.extent.as_static() for a in c.axes] == [32, 3072]
    assert c.axes[0].name == tile.place.grid[-2].name and c.axes[1].name == tile.place.grid[-1].name
    assert len(c.epilogue) == 0 and [type(s) for s in tile.op.body] == [Write]
    assert {"x", "wn", "w"} <= set(c.external_reads())
    assert tile.stage is not None and tile.stage.transport == "sync" and tile.stage.bk_elems > 0
    assert tile.knobs.get("PLACE@cone") == "fuse"
    assert tile.knobs.get(f"REDUCE@{stat_loop.axis.name}") == ""
    assert tile.knobs.get(f"TILE@{c.k_axis.name}", "").startswith("a:")
    assert tile.knobs.get(f"STAGE@{c.k_axis.name}") == "d1/sync"


def test_norm_linear_fp32_keeps_map_rows_only():
    """No 16-bit mma atom ⇒ the Contraction form contributes ZERO rows (never a raising row) and
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


def test_mlp_gate_up_nodifies_as_two_fold_contraction():
    """The fused gate/up MLP edge — TWO ⊗-folds sharing one normalized-row A value (fusion
    duplicates the cone SSA per fold; the matcher dedupes by value-tree equality) with the SwiGLU
    combine as projection — nodifies to ``Map(body=combine…Write, source=Contraction(folds=2))``,
    the product-monoid fold, and offers warp sync rows."""
    rows, tile = _resolve(_mlp_gate_up_graph(), pick=_is_warp_row)
    assert isinstance(tile.op, Map) and isinstance(tile.op.source, Contraction)
    c = tile.op.source
    assert len(c.folds) == 2 and c.a_computed
    assert {bl.input for bl, _ in c.folds} == {"wg", "wu"}
    assert isinstance(tile.op.body[-1], Write)
    assert any(_is_warp_row(r) for r in rows)
    assert not _is_warp_row(rows[0]), "option-0 stays the coop reduce row"
