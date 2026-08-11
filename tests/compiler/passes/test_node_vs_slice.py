"""The stored `bilinear fold` node vs its placed `TilePlan` slice — the boundary the tile IR draws.

A contraction node is pure ALGEBRA (`k_axis` + the `a` edge + its `Channel`s). Placement and
schedule ride the slice: a `TilePlan` bound to its `(m, n)` output axes by `.at()`, from which the
`Side` geometry derives. The two travel as a pair; there is no fused view type.

Helpers that EDIT the algebra take and return the NODE, so a caller's placed slice is untouched.
The one pinned here sits on a GPU-gated path (the warp-flash PV stream), so the ordinary CPU suite
never executes it; calling it directly is what makes the regression visible off-GPU.
"""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.schedule import TilePlan
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, Load, Loop, Select, SelectBranch, Write
from emmy.compiler.ir.tile import Channel, Fold
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _scalar_protected, copy_cell
from emmy.compiler.pipeline.passes.lowering.kernel._twist import _pv_streamed

_M, _N, _K = Axis("m", 128), Axis("n", 128), Axis("k", 64)


def _node() -> Fold:
    return Fold.contraction(
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
    assert not hasattr(node, "tile") and not hasattr(node, "axes")


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
    assert got.role is AxisRole.CONTRACTION and got.axis.name == "kv"
    assert tile.axes == (_M, _N)  # the slice still carries the placement the swap never saw


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


def test_the_output_sweep_axis_survives_the_per_cell_rename() -> None:
    """A scalar register tile replicates its projection per output cell, including an output sweep.
    The sweep coordinate is bound by the copied loop and must remain unsuffixed inside its select and
    write; only the per-cell SSA values receive the cell suffix."""
    node, tile = _node(), _slice()
    sweep = Axis("sweep", 2)
    body = Body(
        (
            Loop(
                axis=sweep,
                body=Body(
                    (
                        Select(
                            name="value",
                            branches=(
                                SelectBranch(value="input", select=BinaryExpr("==", Var("sweep"), Literal(0, "int"))),
                                SelectBranch(value="acc", select=Literal(1, "int")),
                            ),
                        ),
                        Write(output="out", index=(Var("m"), Var("n"), Var("sweep")), value="value"),
                    )
                ),
            ),
        )
    )

    [copied] = copy_cell(body, Sigma({}), "__c0_0", _scalar_protected(node, tile, body=body))
    select, write = copied.body
    assert copied.axis.name == "sweep"
    assert select.name == "value__c0_0" and write.values == ("value__c0_0",)
    assert select.branches[0].select == BinaryExpr("==", Var("sweep"), Literal(0, "int"))
    assert write.index[-1] == Var("sweep")


# --- the (m, n) binding lives on the SCHEDULING STRUCTURE, not at each reader ------------------- #


def test_sched_places_the_root_contraction_on_the_grids_trailing_pair() -> None:
    """``Sched.tile_of`` hands back a slice that is ALREADY placed, so a reader never states a
    placement rule of its own. The root contraction tiles the kernel grid's trailing ``(m, n)``."""
    from emmy.compiler.ir.schedule import Placement
    from emmy.compiler.ir.tile import TileOp
    from emmy.compiler.ir.tile.ops import Sched, sched_of

    node = _node()
    tile = TileOp(op=node, place=Placement(free=(_M, _N), grid=(_M, _N), mapped=True))
    Sched(node, tile.schedule).put("TILE", node, TilePlan(units=(2, 1), regs=(2, 1)))

    placed = sched_of(tile).tile_of(node)
    assert placed.axes == (_M, _N)
    assert (placed.m.axis, placed.n.axis) == (_M, _N)  # the Side geometry derives from the binding
    # The stored slice is untouched — placement is bound on READ, and `axes` is compare=False, so
    # the row's identity (and every golden / prior key) never sees it.
    assert tile.schedule["TILE"].axes is None
    assert tile.schedule["TILE"] == placed


def test_an_unmapped_placement_leaves_the_slice_unplaced() -> None:
    """A rank-<2 or undecided grid supplies no pair; the slice comes back unplaced and the caller
    takes its untiled path rather than getting a wrong binding."""
    from emmy.compiler.ir.schedule import Placement
    from emmy.compiler.ir.tile import TileOp
    from emmy.compiler.ir.tile.ops import Sched, sched_of

    node = _node()
    tile = TileOp(op=node, place=Placement(free=(_M,), grid=(_M,)))
    Sched(node, tile.schedule).put("TILE", node, TilePlan(units=(2, 1), regs=(2, 1)))
    assert sched_of(tile).tile_of(node).axes is None


def _flash_tile():
    """The flash shape as a scheduled ``TileOp`` — the three-depth tree whose two contraction sites
    take DIFFERENT placement rules (the hoisted QK edge vs the derived PV)."""
    import importlib.util
    import pathlib

    from emmy.compiler.ir.schedule import Placement
    from emmy.compiler.ir.tile import TileOp
    from emmy.compiler.ir.tile.ops import Sched

    spec = importlib.util.spec_from_file_location("_codec_fx", pathlib.Path(__file__).parents[1] / "ir" / "tile" / "test_path_codec.py")
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    root, stream, qk, pv = fx._flash_tree()
    b0, b1, m, d = Axis("b0", 2), Axis("b1", 8), Axis("m", 512), Axis("d", 64)
    tile = TileOp(op=root, place=Placement(free=(b0, b1, m, d), grid=(b0, b1, m), mapped=True))
    sched = Sched(root, tile.schedule)
    sched.put("TILE", qk, TilePlan(units=(4, 1), regs=(1, 2)))
    sched.put("TILE", pv, TilePlan(units=(4, 1), regs=(1, 16)))
    return tile, stream, qk, pv


def test_the_two_flash_contraction_sites_take_different_placement_rules() -> None:
    """The rule is a function of the SITE, which is what lets ONE binding on the scheduling
    structure replace the three hand-written ``.at(...)`` calls the readers used to carry:

    - the hoisted QK edge tiles ``(free m, its PARENT fold's stream axis)`` — the score block;
    - the DERIVED PV tiles ``(free m, free d)`` — it sits below the seam lattice, so the grid says
      nothing about it.
    """
    from emmy.compiler.ir.tile.ops import sched_of

    tile, stream, qk, pv = _flash_tile()
    sched = sched_of(tile)
    assert tuple(a.name for a in sched.tile_of(qk).axes) == ("m", stream.axis.name)
    assert tuple(a.name for a in sched.tile_of(pv).axes) == ("m", "d")
    # ...and neither is the ROOT rule (the grid's trailing pair would be (b1, m)).
    assert sched.tile_of(qk).axes[1].name != "m"
