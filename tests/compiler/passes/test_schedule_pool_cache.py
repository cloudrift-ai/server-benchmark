"""The schedule pool cache — ``ctx.session_cache`` memoizing ``_enumerate`` per term.

What these pin: (a) SHARING — a second resolve against the same ``Context`` answers the
enumeration from the pool, and two same-shape kernels in one graph share one enumeration;
(b) EQUALITY — a cache hit serves byte-identical rows, so the fork a policy walks is the same
either way; (c) ISOLATION — pool rows are read-only, and a live schedule pin changes the pool
key rather than narrowing a shared pool in place."""

from __future__ import annotations

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import flatten_leaves
from emmy.compiler.pipeline.knob import canonical_row_key, family_of
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.space import TILE


def _matmul_graph(n: int = 1, dtype: str = "f32") -> Graph:
    """``n`` independent same-shape matmuls (distinct inputs, so fusion cannot merge them)."""
    g = Graph()
    outs = []
    for i in range(n):
        g.add_node(InputOp(), [], Tensor(f"a{i}", (1, Dim(64), Dim(64)), dtype=dtype), node_id=f"a{i}")
        g.add_node(InputOp(), [], Tensor(f"b{i}", (Dim(64), Dim(64)), dtype=dtype), node_id=f"b{i}")
        g.add_node(MatmulOp(), [f"a{i}", f"b{i}"], Tensor(f"c{i}", (1, Dim(64), Dim(64)), dtype=dtype), node_id=f"c{i}")
        outs.append(f"c{i}")
    g.inputs, g.outputs = [n_ for i in range(n) for n_ in (f"a{i}", f"b{i}")], outs
    return g


def _resolve(ctx: Context, graph: Graph) -> list[tuple]:
    """Resolve ``graph``'s forks on option-0, returning every TILE-fork leaf's row identity."""
    idents: list[tuple] = []

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        for leaf in leaves:
            row = dict(getattr(leaf, "knobs", {}) or {})
            if any("TILE" in family_of(k) for k in row):
                idents.append(canonical_row_key(row))
        return leaves[0]

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(graph, decide)
    return idents


def test_second_resolve_hits_the_pool_and_serves_identical_rows() -> None:
    ctx = Context.from_target((12, 0))
    first = _resolve(ctx, _matmul_graph())
    assert first and ctx.session_cache.misses >= 1
    misses = ctx.session_cache.misses
    second = _resolve(ctx, _matmul_graph())
    assert ctx.session_cache.hits >= 1 and ctx.session_cache.misses == misses, "the second resolve must not re-enumerate"
    assert first == second, "a cache hit must serve the same rows in the same order"


def test_two_same_shape_kernels_share_one_enumeration() -> None:
    ctx = Context.from_target((12, 0))
    idents = _resolve(ctx, _matmul_graph(n=2))
    assert ctx.session_cache.hits >= 1, "the second kernel must answer from the first's pool"
    half = len(idents) // 2
    assert len(idents) == 2 * half and idents[:half] == idents[half:], "both kernels must see the same fork"


def test_fresh_context_starts_cold() -> None:
    ctx1, ctx2 = Context.from_target((12, 0)), Context.from_target((12, 0))
    _resolve(ctx1, _matmul_graph())
    _resolve(ctx2, _matmul_graph())
    assert ctx2.session_cache.hits == 0, "a fresh Context is a fresh session — no cross-session sharing"


def test_pool_rows_are_read_only() -> None:
    ctx = Context.from_target((12, 0))
    _resolve(ctx, _matmul_graph())
    (pool,) = ctx.session_cache._store.values()
    assert pool.rows
    with pytest.raises(TypeError):
        pool.rows[0]["TILE"] = "corrupted"


def test_a_dtype_change_keys_a_different_pool() -> None:
    """The dtype is atom-eligibility input the TERM deliberately omits (buffers are positional,
    Loads carry no dtype) — an f16 trace of the same shape must never serve the f32 pool. The
    original incident: one shared ctx, an f16 then an f32 SDPA trace, and the f32 fork came back
    with 24 warp geometries."""
    ctx = Context.from_target((12, 0))
    f32 = _resolve(ctx, _matmul_graph())
    f16 = _resolve(ctx, _matmul_graph(dtype="f16"))
    assert ctx.session_cache.misses >= 2, "an f16 twin must enumerate its own pool"
    assert set(f16) != set(f32), "the f16 pool must differ (the warp tier is dtype-gated)"


def test_a_live_pin_keys_a_different_pool() -> None:
    ctx = Context.from_target((12, 0))
    unpinned = _resolve(ctx, _matmul_graph())
    with TILE.pinned("f2x8"):
        pinned = _resolve(ctx, _matmul_graph())
    assert ctx.session_cache.misses >= 2, "a pin state must never share the unpinned pool"
    assert 0 < len(pinned) < len(unpinned), "the pinned fork must be a narrowing of the unpinned one"
    # And back: the unpinned pool is still served intact after the pin lifts.
    assert _resolve(ctx, _matmul_graph()) == unpinned
