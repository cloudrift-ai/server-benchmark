"""The algebra-free seal around already-blockified output axes."""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.schedule import Side
from emmy.compiler.ir.stmt import Body, Write
from emmy.compiler.pipeline.passes.lowering.kernel._tiling import grid_tile


def _side(name: str, extent: int, *, tile: int, units: int, reg: int) -> Side:
    return Side(axis=Axis(name, extent), tile=tile, units=units, reg=reg, atom=1, block=name + "_b", unit=name + "_u")


def _mn() -> tuple[Side, Side]:
    # An 8×8 output cell: 4 units × 2 registers on m, 8 units × 1 on n (scalar atoms, so a unit is
    # one thread and the CTA is 4·8 = 32 threads).
    return _side("m", 512, tile=8, units=4, reg=2), _side("n", 256, tile=8, units=8, reg=1)


def _empty_callables():
    return dict(state_decls=lambda _cells: [], reduce_region=lambda _c, _o, _mn: ([], []), store=lambda _i, _j, _o, _mn: [])


def test_the_bound_block_owns_the_per_cell_coordinate() -> None:
    """A resolved side supplies the complete block, unit, register, and atom coordinate."""
    mn = _mn()
    m_cell1 = mn[0].base(1)
    assert m_cell1 == Var("m_b") * Literal(8, "int") + Var("m_u") * Literal(2, "int") + Literal(1, "int")
    n_cell0 = mn[1].base(0)
    assert n_cell0 == Var("n_b") * Literal(8, "int") + Var("n_u") * Literal(1, "int") + Literal(0, "int")


def test_the_bound_axes_are_grid_then_unit_with_no_lane_for_a_scalar_atom() -> None:
    """The `Tile`'s axis order IS the loop nest: leading (untiled) grid axes, the shrunk block
    axes, then the unit axes. A scalar atom (`lanes == 1`) emits no `_lane` axis."""
    mn = _mn()
    bt = Axis("bt", 8)
    t = grid_tile(mn=mn, lead_axes=(bt,), block_threads=32, **_empty_callables())
    assert [a.name for a in t.axes] == ["bt", "m_b", "n_b", "m_u", "n_u"]
    assert t.block_threads == 32
    assert t.raster_axes == ("m_b", "n_b")  # a 2-D-tiled output is rasterization-eligible


def test_a_warp_cooperative_atom_appends_the_lane_axis() -> None:
    mn = _mn()
    t = grid_tile(mn=mn, block_threads=128, lanes=32, **_empty_callables())
    assert [a.name for a in t.axes] == ["m_b", "n_b", "m_u", "n_u", "_lane"]


def test_an_untiled_output_binds_no_block_axis_and_is_not_rasterizable() -> None:
    """The reduce tier / degenerate fold: `mn == (None, None)`, so the whole grid rides
    `lead_axes` and there are no (m, n) block axes to rasterize."""
    grid = (Axis("b", 4), Axis("s", 128))
    t = grid_tile(mn=(None, None), lead_axes=grid, block_threads=None, **_empty_callables())
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
        mn=mn,
        block_threads=32,
        state_decls=state_decls,
        reduce_region=lambda _c, _o, _mn: ([], []),
        store=store,
    )
    assert seen["cells"] == [(0, 0), (1, 0)]  # reg_m=2 × reg_n=1
    assert stores == [(0, 0), (1, 0)]
    assert len(t.body) == 2  # one Write spliced per cell


def test_the_bound_grid_axis_keeps_a_symbolic_extent_symbolic() -> None:
    """A block grid is a ceiling division of the logical extent by the bound tile width."""
    shrunk = _side("m", 128, tile=4, units=1, reg=4).axes[0]
    assert shrunk.extent.as_static() == 32
    assert shrunk.window is not None and shrunk.window.parent.name == "m"


def test_the_layer_needs_no_node_ctx_or_algebra() -> None:
    """The extraction's point, held as a test: every case above built a real `Tile` from a `Side`
    pair, integer counts and three callables. Guard the module namespace so an algebra / emission
    dependency cannot creep back in — that is what makes the layer separately testable at all.

    Every banned name must NAME SOMETHING, checked here against the modules it would leak from. A
    guard listing a symbol that no longer exists cannot fire and reads as coverage it does not
    provide: three of this list's entries had rotted that way — `Map` and `Placed` named classes
    the one-kind collapse deleted, and a glossary rename left the literal `"bilinear fold"`, which
    has a space in it and so can never be an identifier at all."""
    from emmy.compiler.ir.pure import fold as pure_fold
    from emmy.compiler.ir.tile import ir as tile_ir
    from emmy.compiler.pipeline.passes.lowering.kernel import _atom, _factor, _tiling

    banned = ("Fold", "TileOp", "Ctx", "reduce_codegen", "store_sink", "copy_cell")
    live = {n for mod in (pure_fold, tile_ir, _factor, _atom) for n in dir(mod)}
    assert set(banned) <= live, f"the guard names nothing: {sorted(set(banned) - live)}"

    names = {n for n in dir(_tiling) if not n.startswith("__")}
    for name in banned:
        assert name not in names, f"{name} leaked into the tiling layer"


def test_the_tile_body_is_state_then_reduce_then_stores() -> None:
    """The splice order the seal guarantees: accumulator state, the reduce region's top decls +
    K-loop, then the per-cell stores."""
    mn = _mn()
    mk = lambda tag: Write(output=tag, index=(Var("m"),), value="v")  # noqa: E731
    t = grid_tile(
        mn=mn,
        block_threads=32,
        state_decls=lambda _cells: [mk("state")],
        reduce_region=lambda _c, _o, _mn: ([mk("top")], [mk("kloop")]),
        store=lambda i, _j, _o, _mn: [mk(f"store{i}")],
    )
    assert [s.output for s in t.body] == ["state", "top", "kloop", "store0", "store1"]
    assert isinstance(t.body, Body)
