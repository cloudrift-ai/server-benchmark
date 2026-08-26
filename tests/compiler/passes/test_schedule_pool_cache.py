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
from emmy.compiler.pipeline.fork import iter_leaves
from emmy.compiler.pipeline.knob import canonical_row_key, family_of
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.space import REDUCE, TILE


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
    """Resolve ``graph``'s forks on the first emitted leaf, returning its row per TILE fork."""
    idents: list[tuple] = []

    def decide(fp):
        leaf = next(iter_leaves(fp.options))
        row = dict(getattr(leaf, "knobs", {}) or {})
        if any("TILE" in family_of(k) for k in row):
            idents.append(canonical_row_key(row))
        return leaf

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
    _resolve(ctx, _matmul_graph())
    before = set(ctx.session_cache._store)
    _resolve(ctx, _matmul_graph(dtype="f16"))
    assert ctx.session_cache.misses >= 2, "an f16 twin must enumerate its own pool"
    new = set(ctx.session_cache._store) - before
    assert len(before) == len(new) == 1
    f32 = ctx.session_cache._store[next(iter(before))]
    f16 = ctx.session_cache._store[next(iter(new))]

    def sample(pool):
        return tuple(canonical_row_key(dict(pool.rows[i])) for i in {0, len(pool.rows) // 2, len(pool.rows) - 1})

    assert (f16.total, sample(f16)) != (f32.total, sample(f32)), "the f16 pool must differ (the warp tier is dtype-gated)"


def test_a_precision_gate_pin_keys_a_different_pool() -> None:
    """The ``F16_MMA_F32_ACC`` gate is not a schedule family but it changes which rows are
    OFFERED (the f16-accumulate atom siblings), so it must ride the pin fingerprint — the fit's
    fm-pinned reconstruction shares one Context per card and would otherwise collide with the
    unpinned pools."""
    from emmy.compiler.pipeline.knob import schedule_pin_fingerprint
    from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC

    clean = schedule_pin_fingerprint()
    with F16_MMA_F32_ACC.pinned("1"):
        assert schedule_pin_fingerprint() != clean, "the precision gate must change the pool key"
    assert schedule_pin_fingerprint() == clean


def test_a_live_pin_keys_a_different_pool() -> None:
    ctx = Context.from_target((12, 0))
    unpinned = _resolve(ctx, _matmul_graph())
    (unpinned_pool,) = ctx.session_cache._store.values()
    before = set(ctx.session_cache._store)
    with TILE.pinned("f2x8"):
        _resolve(ctx, _matmul_graph())
    (pinned_key,) = set(ctx.session_cache._store) - before
    pinned_pool = ctx.session_cache._store[pinned_key]
    assert ctx.session_cache.misses >= 2, "a pin state must never share the unpinned pool"
    assert 0 < len(pinned_pool.rows) < len(unpinned_pool.rows), "the pinned fork must be a narrowing of the unpinned one"
    # And back: the unpinned pool is still served intact after the pin lifts.
    assert _resolve(ctx, _matmul_graph()) == unpinned


def test_a_live_context_never_samples() -> None:
    """``dataclasses.replace`` SHARES the session cache, so a sampled Context and the live one it
    was derived from sit on ONE memo. That is why the sample is part of the pool's cache KEY rather
    than merely of the Context: without it the first compile to run would decide what every later
    one sees, and a live deploy could be served a pool with most of its candidates missing."""
    from dataclasses import replace

    from emmy.compiler.pipeline.search.pool import PoolSample

    ctx = Context.from_target((12, 0))
    sampled = replace(ctx, pool_sample=PoolSample(rows=8))
    assert sampled.session_cache is ctx.session_cache, "the shared memo is the hazard this test exists for"

    with REDUCE.pinned(""):
        drawn = _resolve(sampled, _matmul_graph())
        (drawn_pool,) = ctx.session_cache._store.values()
        before = set(ctx.session_cache._store)
        full = _resolve(ctx, _matmul_graph())
        (full_key,) = set(ctx.session_cache._store) - before
        full_pool = ctx.session_cache._store[full_key]
        assert 0 < len(drawn_pool.rows) < len(full_pool.rows), "the sampled Context sees a draw, the live one the whole pool"
        assert drawn_pool.total == full_pool.total
        assert set(map(canonical_row_key, drawn_pool.rows)) < set(map(canonical_row_key, full_pool.rows))
        assert _resolve(sampled, _matmul_graph()) == drawn, "each keeps its own memo entry"
        assert _resolve(ctx, _matmul_graph()) == full
