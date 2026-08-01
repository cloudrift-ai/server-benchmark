"""The axis-realization layer (`kernel/_tiling.py`) — the four levels a schedule's plan becomes
bound `Axis` objects through, exercised without a node, a `Ctx` or any algebra.

`020_schedule` decides the plan; nothing it produces is an axis a kernel loops over. This layer is
the other half, and it is algebra-free by construction: it takes a `Side` pair, integer counts and
three callables. These tests hold that boundary — they never build a `Contraction` — and pin the
per-cell coordinate arithmetic `AxisOffset.base` accumulates across the levels, which is what the
emitted σ substitutions index with.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.schedule import Side
from emmy.compiler.ir.stmt import Body, Write
from emmy.compiler.pipeline.passes.lowering.kernel._tiling import atomize, grid_tile, register_tile, shrink_axis, unit_tile


def _side(name: str, extent: int, *, tile: int, units: int, reg: int) -> Side:
    return Side(axis=Axis(name, extent), tile=tile, units=units, reg=reg, block=name + "_b", unit=name + "_u")


def _mn() -> tuple[Side, Side]:
    # An 8×8 output cell: 4 units × 2 registers on m, 8 units × 1 on n (scalar atoms, so a unit is
    # one thread and the CTA is 4·8 = 32 threads).
    return _side("m", 512, tile=8, units=4, reg=2), _side("n", 256, tile=8, units=8, reg=1)


def _empty_callables():
    return dict(state_decls=lambda _cells: [], reduce_region=lambda _c, _o, _mn: ([], []), store=lambda _i, _j, _o, _mn: [])


def test_the_levels_accumulate_the_per_cell_coordinate() -> None:
    """`base(r)` is `block·(units·reg·atom) + unit·(reg·atom) + r·atom` once the UNIT level is
    present — the coordinate every emitted σ indexes the cell with. Each level contributes exactly
    one term, so the assembled expression is the contract between them."""
    mn = _mn()
    t = unit_tile(register_tile(atomize((1, 1)), mn), mn)
    # The block var is bound by grid_tile; bind it here the same way so `base` is readable without
    # reaching into the sealed Tile.
    off = tuple(replace(o, block_var=s.block) for o, s in zip(t.offset, mn, strict=True))

    m_cell1 = off[0].base(1)  # m: block·(4·2·1) + unit·(2·1) + 1·1
    assert m_cell1 == Var("m_b") * Literal(8, "int") + Var("m_u") * Literal(2, "int") + Literal(1, "int")
    n_cell0 = off[1].base(0)  # n: block·(8·1·1) + unit·(1·1) + 0
    assert n_cell0 == Var("n_b") * Literal(8, "int") + Var("n_u") * Literal(1, "int") + Literal(0, "int")

    # Without the UNIT level the offset is the bare `block·reg + r` — the degenerate arm.
    bare = replace(register_tile(atomize((1, 1)), mn).offset[0], block_var="m_b")
    assert bare.base(1) == Var("m_b") * Literal(2, "int") + Literal(1, "int")


def test_the_bound_axes_are_grid_then_unit_with_no_lane_for_a_scalar_atom() -> None:
    """The `Tile`'s axis order IS the loop nest: leading (untiled) grid axes, the shrunk block
    axes, then the unit axes. A scalar atom (`lanes == 1`) emits no `_lane` axis."""
    mn = _mn()
    bt = Axis("bt", 8)
    t = grid_tile(unit_tile(register_tile(atomize((1, 1)), mn), mn), mn=mn, lead_axes=(bt,), block_threads=32, **_empty_callables())
    assert [a.name for a in t.axes] == ["bt", "m_b", "n_b", "m_u", "n_u"]
    assert t.block_threads == 32
    assert t.raster_axes == ("m_b", "n_b")  # a 2-D-tiled output is rasterization-eligible


def test_a_warp_cooperative_atom_appends_the_lane_axis() -> None:
    mn = _mn()
    t = grid_tile(unit_tile(register_tile(atomize((16, 8)), mn), mn), mn=mn, block_threads=128, lanes=32, **_empty_callables())
    assert [a.name for a in t.axes] == ["m_b", "n_b", "m_u", "n_u", "_lane"]


def test_an_untiled_output_binds_no_block_axis_and_is_not_rasterizable() -> None:
    """The reduce tier / degenerate fold: `mn == (None, None)`, so the whole grid rides
    `lead_axes` and there are no (m, n) block axes to rasterize."""
    grid = (Axis("b", 4), Axis("s", 128))
    t = grid_tile(atomize((1, 1)), mn=(None, None), lead_axes=grid, block_threads=None, **_empty_callables())
    assert [a.name for a in t.axes] == ["b", "s"]
    assert t.raster_axes is None and t.block_threads is None


def test_the_cells_the_callables_receive_are_the_register_grid() -> None:
    """One store per register cell — `reg_m × reg_n` of them, and the same `(i, j)` set the state
    decls saw. This is the whole contract the tiers plug into."""
    mn = _mn()
    seen: dict[str, object] = {}
    stores = []

    def state_decls(cells):
        seen["cells"] = list(cells)
        return []

    def store(i, j, _off, _mn):
        stores.append((i, j))
        return [Write(output="out", index=(Var("m"),), value="v")]

    t = grid_tile(
        unit_tile(register_tile(atomize((1, 1)), mn), mn),
        mn=mn,
        block_threads=32,
        state_decls=state_decls,
        reduce_region=lambda _c, _o, _mn: ([], []),
        store=store,
    )
    assert seen["cells"] == [(0, 0), (1, 0)]  # reg_m=2 × reg_n=1
    assert stores == [(0, 0), (1, 0)]
    assert len(t.body) == 2  # one Write spliced per cell


def test_shrink_axis_keeps_a_symbolic_extent_symbolic() -> None:
    """The grid axis for a register-tiled free axis is `ceil(E / reg)`, so a dynamic extent sizes
    the launch from the runtime value rather than collapsing to a static count."""
    assert shrink_axis(Axis("m", 128), 1) == Axis("m", 128)  # reg 1 is the identity
    shrunk = shrink_axis(Axis("m", 128), 4)
    assert shrunk.extent.as_static() == 32
    assert shrunk.window is not None and shrunk.window.parent.name == "m"


def test_the_layer_needs_no_node_ctx_or_algebra() -> None:
    """The extraction's point, held as a test: every case above built a real `Tile` from a `Side`
    pair, integer counts and three callables. Guard the module namespace so an algebra / emission
    dependency cannot creep back in — that is what makes the layer separately testable at all."""
    from emmy.compiler.pipeline.passes.lowering.kernel import _tiling

    names = {n for n in dir(_tiling) if not n.startswith("__")}
    for banned in ("Contraction", "Fold", "Map", "Placed", "Ctx", "reduce_codegen", "store_sink", "copy_cell"):
        assert banned not in names, f"{banned} leaked into the tiling layer"


def test_the_tile_body_is_state_then_reduce_then_stores() -> None:
    """The splice order the seal guarantees: accumulator state, the reduce region's top decls +
    K-loop, then the per-cell stores."""
    mn = _mn()
    mk = lambda tag: Write(output=tag, index=(Var("m"),), value="v")  # noqa: E731
    t = grid_tile(
        unit_tile(register_tile(atomize((1, 1)), mn), mn),
        mn=mn,
        block_threads=32,
        state_decls=lambda _cells: [mk("state")],
        reduce_region=lambda _c, _o, _mn: ([mk("top")], [mk("kloop")]),
        store=lambda i, _j, _o, _mn: [mk(f"store{i}")],
    )
    assert [s.output for s in t.body] == ["state", "top", "kloop", "store0", "store1"]
    assert isinstance(t.body, Body)
