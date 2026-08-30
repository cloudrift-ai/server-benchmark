"""The scalar register tile reads a CELL-varying operand once per cell, not once per row / column.

The tile's reuse — one A read per register row, one B read per column — holds for a gmem ``Load``
edge, which indexes its own axes only. A COMPUTED edge may read the OTHER output axis: the qwen3-0.6b
layer-0 o_proj arrives as ``out[m, n] = Σ_k B[n, k] · A[m, k, n]``, its A cone broadcast over n. Read
once per row, that A is both the wrong value for every column past the first and a reference to a
coordinate nothing binds — the kernel decodes only the ``_b`` / ``_u`` split vars, so the per-copy
rename suffixed the surviving axis name and nvcc rejected the kernel (``identifier "a1__ar9" is
undefined``, 10 errors, one per register row, sm_89).

The emission is exercised the way ``_factor._bind``'s output-tiled arm makes it — the atom's
``reduce_codegen`` + ``store_sink`` sealed through ``grid_tile`` — so the assertions read the same
``Tile`` the CUDA backend renders, without needing a GPU.
"""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure.fold import Channel, Fold
from emmy.compiler.ir.schedule import Tile
from emmy.compiler.ir.stmt import Assign, Body, Load, Stmt, Write
from emmy.compiler.pipeline.passes.lowering.kernel._atom import reduce_codegen, store_sink
from emmy.compiler.pipeline.passes.lowering.kernel._tiling import atomize, grid_tile, register_tile, unit_tile

_M, _N, _K = Axis("m", 8), Axis("n", 8), Axis("k", 4)
_PLAN = Tile(units=(2, 2), regs=(2, 2))  # a scalar 2×2 thread tile of 2×2 register cells


def _a_load() -> Load:
    """The ordinary A edge — indexed by its own ``(m, k)``, so it is shared down the register row."""
    return Load(name="a", input="A", index=(Var("m"), Var("k")))


def _b_load() -> Load:
    return Load(name="b", input="B", index=(Var("n"), Var("k")))


def _cone(name: str, buf: str, index: tuple) -> Fold:
    """A computed operand edge (an inline producer cone) whose load carries ``index`` — the o_proj
    shape's broadcast A when ``index`` mentions n."""
    body = Body((Load(name=f"{name}_l", input=buf, index=index), Assign(name=name, op="multiply", args=(f"{name}_l", f"{name}_l"))))
    return Fold.projection(body=body)


def _tile(a, b):
    """The scalar contraction ``a ⊗ b`` bound to the grid — the ``Tile`` and its ``(m, n)`` sides."""
    c = Fold.contraction(k_axis=_K, a=a, channels=(Channel(b=b, acc="acc"),))
    plan = _PLAN.at(_M, _N)
    mn = plan.mn
    state, reduce_region = reduce_codegen(c, plan)
    epilogue = Body((Write(output="out", index=(Var("m"), Var("n")), value=c.acc),))
    return grid_tile(
        unit_tile(register_tile(atomize(plan.atom.shape[:2]), mn), mn),
        mn=mn,
        block_threads=plan.launch_threads,
        lanes=plan.atom.lanes,
        state_decls=state,
        reduce_region=reduce_region,
        store=store_sink(c, plan, epilogue),
    ), mn


def _unbound(tile) -> set[str]:
    """The names the tile body reads without defining them or binding them as an axis. Anything left
    is an identifier the rendered kernel would not define."""
    body = Body(tile.body)
    reads = {v for s in body.iter() for v in (*s.deps(), *(fv for e in s.exprs() for fv in e.free_vars()))}
    defined = {n for s in body.iter() for n in s.defines()}
    return reads - defined - set(body.axis_names) - {a.name for a in tile.axes}


def _loads_of(tile, buf: str) -> tuple[Stmt, ...]:
    return tuple(s for s in Body(tile.body).loads if s.input == buf)


def _muls(tile) -> list[Stmt]:
    return [s for s in Body(tile.body).iter() if isinstance(s, Assign) and s.name.startswith("acc__v")]


def _cells(mn) -> list[tuple[int, int]]:
    return [(i, j) for i in range(mn[0].reg) for j in range(mn[1].reg)]


def test_a_operand_varying_along_n_is_read_per_cell() -> None:
    """The reproducer. A's cone reads n, so each register cell gets its own copy, σ-bound to BOTH
    coordinates — no axis name survives free, and each cell's multiply reads its own value."""
    tile, mn = _tile(_cone("a", "A", (Var("m"), Var("k"), Var("n"))), _b_load())
    assert _unbound(tile) == set()
    assert len(_loads_of(tile, "A")) == len(_cells(mn))
    assert {s.args[1] for s in _muls(tile)} == {f"a__ar{i}_{j}" for i, j in _cells(mn)}


def test_b_operand_varying_along_m_is_read_per_cell() -> None:
    """The mirror hazard on the B side — a computed B whose cone reads m cannot be shared across the
    column either."""
    tile, mn = _tile(_a_load(), _cone("b", "B", (Var("n"), Var("k"), Var("m"))))
    assert _unbound(tile) == set()
    assert len(_loads_of(tile, "B")) == len(_cells(mn))
    assert {s.args[0] for s in _muls(tile)} == {f"b__bc{i}_{j}" for i, j in _cells(mn)}


def test_computed_a_shared_along_n_is_read_once_per_register_row() -> None:
    """The f32 fallback's useful case: A is computed but independent of n, so the scalar tile
    evaluates it once per register row and every column contracts with that same value."""
    tile, mn = _tile(_cone("a", "A", (Var("m"), Var("k"))), _b_load())
    assert _unbound(tile) == set()
    assert len(_loads_of(tile, "A")) == mn[0].reg
    assert {s.args[1] for s in _muls(tile)} == {f"a__ar{i}" for i in range(mn[0].reg)}


def test_materialized_operands_keep_the_row_and_column_reuse() -> None:
    """The common matmul emission is untouched: one A read per register row, one B read per column —
    the arithmetic-intensity reuse the register tile exists for."""
    tile, mn = _tile(_a_load(), _b_load())
    assert _unbound(tile) == set()
    assert len(_loads_of(tile, "A")) == mn[0].reg
    assert len(_loads_of(tile, "B")) == mn[1].reg
    assert {s.args for s in _muls(tile)} == {(f"b__bc{j}", f"a__ar{i}") for i, j in _cells(mn)}
