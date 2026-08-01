"""The stored `Contraction` node vs its placed `TilePlan` slice — the boundary the tile IR draws.

A contraction node is pure ALGEBRA (`k_axis` + the `a` edge + its `Channel`s). Placement and
schedule ride the slice: a `TilePlan` bound to its `(m, n)` output axes by `.at()`, from which the
`Side` geometry derives. The two travel as a pair; there is no fused view type.

Helpers that EDIT the algebra take and return the NODE, so a caller's placed slice is untouched.
Both pinned here sit on GPU-gated paths (the mixed-dtype warp demotion, the warp-flash PV stream),
so the ordinary CPU suite never executes them; calling them directly is what makes the regression
visible off-GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

from emmy.compiler.dtype import F16, F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import TilePlan
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Load
from emmy.compiler.ir.tile import Channel, Contraction
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _scalar_protected, copy_cell
from emmy.compiler.pipeline.passes.lowering.kernel._twist import _pv_streamed
from emmy.compiler.pipeline.passes.lowering.tile._schedule import _demote_mixed_a

_M, _N, _K = Axis("m", 128), Axis("n", 128), Axis("k", 64)


def _node() -> Contraction:
    return Contraction(
        k_axis=_K,
        a=Load(name="a", input="A", index=(Var("m"), Var("k"))),
        channels=(Channel(b=Load(name="b", input="B", index=(Var("k"), Var("n"))), acc="acc"),),
    )


def _slice(tile: TilePlan | None = None) -> TilePlan:
    """A tile PLACED on the (m, n) output axes — what the schedule stores and the tiers read."""
    return (tile or TilePlan(units=(2, 1), regs=(2, 1))).at(_M, _N)


def test_the_node_carries_no_placement_or_schedule() -> None:
    """The invariant the split exists to keep: a contraction's identity is its algebra, so the
    same node under two tiles is ONE term, and all the geometry hangs off the slice."""
    node = _node()
    t1, t2 = _slice(TilePlan(units=(2, 1), regs=(2, 1))), _slice(TilePlan(units=(4, 1), regs=(4, 1)))
    assert not hasattr(node, "tile") and not hasattr(node, "axes")
    assert t1.launch_threads != t2.launch_threads  # the geometry is the SLICE's


def test_placement_is_not_part_of_a_tile_s_identity() -> None:
    """``TilePlan.axes`` is ``compare=False``: placement is not a search dimension. Two plans
    differing only in placement are the SAME tile, so enumeration dedup, the stamped knob row and
    the golden / prior keys never see an axis."""
    bare = TilePlan(units=(2, 1), regs=(2, 1))
    assert bare == bare.at(_M, _N) and hash(bare) == hash(bare.at(_M, _N))
    assert bare.spell() == bare.at(_M, _N).spell()
    assert bare.at(_M, _N).m.name == "m" and bare.at(_M, _N).n.name == "n"


def test_pv_streamed_swaps_the_stream_axis_on_the_stored_node() -> None:
    """The warp-flash PV contracts the whole key block, so its intra-block axis is swapped for the
    stream axis — a pure ALGEBRA edit, so it takes and returns the stored node and the caller's
    placed ``TilePlan`` is untouched."""
    node, tile = _node(), _slice()
    got = _pv_streamed(node, Axis("kv", 256))
    assert isinstance(got, Contraction) and got.k_axis.name == "kv"
    assert tile.axes == (_M, _N)  # the slice still carries the placement the swap never saw


def test_demote_mixed_a_rewrites_the_a_edge_on_the_stored_node() -> None:
    """A mixed-dtype (f32-A × 16-bit-B) contraction re-expresses its A ``Load`` as a computed cone
    so it can ride the demoting sync compute-fill. The rewrite is pure ALGEBRA, so it takes and
    returns the stored :class:`Contraction`; the caller re-binds it to its own placement."""
    kernel = SimpleNamespace(inputs={"A": SimpleNamespace(dtype=F32, shape=()), "B": SimpleNamespace(dtype=F16, shape=())})
    node, tile = _node(), _slice()
    out = _demote_mixed_a(kernel, node)
    assert isinstance(out, Contraction) and out.a_computed  # the A edge became an inline cone
    assert tile.axes == (_M, _N)  # the caller's placed slice is untouched by an algebra edit


def test_the_lead_grid_axes_survive_the_per_cell_rename() -> None:
    """Why the leading grid axes reach ``_atom`` at all. The scalar tier replicates the operand
    reads and the projection tail once per register cell, suffixing every SSA name that is not
    shared (:func:`copy_cell`). A leading (batch) grid axis IS shared — the whole cell block sits at
    one batch coordinate — so it must be protected; renaming it would emit a reference to a
    variable no enclosing loop defines. They ride as ``_atom``'s own ``lead`` (the grid's fact,
    passed by ``_factor``), never on the ``TilePlan`` slice, whose reading is the tiled cell.
    The batched scalar-tile shape that reaches this is absent from the corpus (a batched matmul
    takes the warp tier, and a bare matmul's epilogue is empty), so the mechanism is pinned here."""
    node, tile = _node(), _slice()
    prot = _scalar_protected(node, tile, (Axis("bt", 8),))
    assert "bt" in prot and {"m_b", "m_u", "n_b", "n_u", "k"} <= prot

    body = (Load(name="a", input="A", index=(Var("bt"), Var("m"), Var("k"))),)
    [copied] = copy_cell(body, Sigma({}), "__ar0", prot)
    assert copied.name == "a__ar0"  # the per-cell value is the cell's own
    assert copied.index[0] == Var("bt")  # ... the batch coordinate is not

    # Unthreaded (the schedule-side probes, which have no grid and no per-cell emission): the
    # shared coordinate is captured by the rename — the failure the threading prevents.
    [captured] = copy_cell(body, Sigma({}), "__ar0", _scalar_protected(node, tile))
    assert captured.index[0] == Var("bt__ar0")


def test_demote_mixed_a_passes_through_a_uniform_dtype_contraction() -> None:
    kernel = SimpleNamespace(inputs={"A": SimpleNamespace(dtype=F16, shape=()), "B": SimpleNamespace(dtype=F16, shape=())})
    node = _node()
    assert _demote_mixed_a(kernel, node) is node  # identity ⇒ the caller keeps its binding
