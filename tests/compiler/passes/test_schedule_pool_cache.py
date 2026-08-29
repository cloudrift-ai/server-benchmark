"""The schedule pool cache — ``ctx.session_cache`` memoizing the walk's prescan per term.

What these pin: (a) SHARING — a second resolve against the same ``Context`` answers the
enumeration from the memoized per-node option lists, and two same-shape kernels in one graph
share one prescan; (b) EQUALITY — a cache hit replays the walk to byte-identical rows, so the
fork a policy walks is the same either way; (c) ISOLATION — the memoized options are read-only,
and a live schedule pin (or any other enumeration input: dtypes, extents, stores, the sample)
changes the pool key rather than narrowing a shared pool in place."""

from __future__ import annotations

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import LOOP_PASSES, TILE_PASSES, Pipeline
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


def test_pool_options_are_read_only() -> None:
    """The memo shares OPTION LISTS, not rows — each leaf row is a fresh dict merged per walk, so
    a consumer mutating a row cannot corrupt the pool. The shared objects themselves must refuse
    writes: frozen options over read-only knob mappings."""
    ctx = Context.from_target((12, 0))
    _resolve(ctx, _matmul_graph())
    (pool,) = ctx.session_cache._store.values()
    assert pool.options
    option = next(opt for opts in pool.options for opt in opts if opt.knobs)
    with pytest.raises(TypeError):
        option.knobs["TILE"] = "corrupted"


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

    def spelled(pool):
        return {tuple(sorted(opt.knobs.items())) for opts in pool.options for opt in opts}

    assert spelled(f16) != spelled(f32), "the f16 pool must differ (the warp tier is dtype-gated)"


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

    def tiles(pool):
        return {v for opts in pool.options for opt in opts for k, v in opt.knobs.items() if family_of(k) == "TILE"}

    assert 0 < len(tiles(pinned_pool)) < len(tiles(unpinned_pool)), "the pinned offer must be a narrowing of the unpinned one"
    # And back: the unpinned pool is still served intact after the pin lifts.
    assert _resolve(ctx, _matmul_graph()) == unpinned


def test_a_live_context_never_samples() -> None:
    """``dataclasses.replace`` SHARES the session cache, so a sampled Context and the live one it
    was derived from sit on ONE memo. That is why the sample is part of the pool's cache KEY rather
    than merely of the Context: without it the first compile to run would decide what every later
    one sees, and a live deploy could be served a pool with most of its candidates missing."""
    from dataclasses import replace

    from emmy.compiler.pipeline.search.golden_eval import enumerate_graph
    from emmy.compiler.pipeline.search.pool import PoolSample

    ctx = Context.from_target((12, 0))
    sampled = replace(ctx, pool_sample=PoolSample(rows=8))
    assert sampled.session_cache is ctx.session_cache, "the shared memo is the hazard this test exists for"

    with REDUCE.pinned(""):
        drawn = _resolve(sampled, _matmul_graph())
        (drawn_pool,) = ctx.session_cache._store.values()
        assert 0 < len(drawn_pool.rows) <= 8 < drawn_pool.total, "the sampled Context sees a draw, never the pool"
        before = set(ctx.session_cache._store)
        full = _resolve(ctx, _matmul_graph())
        (full_key,) = set(ctx.session_cache._store) - before
        full_pool = ctx.session_cache._store[full_key]
        assert not hasattr(full_pool, "rows"), "the live memo is the prescan — no row is ever retained for a live compile"
        live_rows = enumerate_graph(_matmul_graph(), ctx).rows
        assert drawn_pool.total == len(live_rows), "the draw reports the exact size of the pool the live walk yields"
        drawn_keys = {canonical_row_key(dict(row)) for row in drawn_pool.rows}
        assert drawn_keys < {canonical_row_key(row) for row in live_rows}
        assert _resolve(sampled, _matmul_graph()) == drawn, "each keeps its own memo entry"
        assert _resolve(ctx, _matmul_graph()) == full


def _unmapped_tile(m: int, n: int, k: int = 64, dtype: str = "f16"):
    """The lifted, unscheduled ``TileOp`` for one ``m x k @ k x n`` matmul, knobs and all.

    Lifting by hand rather than through ``lowering/tile`` is what lets the assertion below name
    the pool key directly instead of inferring a collision from downstream symptoms.
    """
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import lift_loop_op

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Dim(m), Dim(k)), dtype=dtype), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (Dim(k), Dim(n)), dtype=dtype), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (Dim(m), Dim(n)), dtype=dtype), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    node = Pipeline.build(LOOP_PASSES).run(g).nodes["o"]
    tile = lift_loop_op(node.op, name=node.op.name)
    tile.knobs, tile.inputs, tile.outputs = dict(node.op.knobs), dict(node.op.inputs), dict(node.op.outputs)
    return tile


def test_transposed_free_extents_do_not_share_a_pool() -> None:
    """Per-axis extents are an ENUMERATION input, so they must be a key part.

    The term's algebra digest canonicalizes sizes away and the knobs carry only the lossy
    ``S_ext_*`` summary (count / product / max), which ``8x512`` and ``512x8`` agree on. They
    therefore reach identical ``structural_key`` AND ``cache_key`` while their spaces differ by
    7x — the enumeration sizes the coop band against ``_inner_free`` and the fragment store
    against the free axes — so sharing a pool let whichever compiled first decide the other.
    """
    from emmy.compiler.pipeline.passes.lowering.tile._schedule import schedule

    ctx = Context.from_target((12, 0))
    wide, tall = _unmapped_tile(8, 512), _unmapped_tile(512, 8)

    # The consequence of transposing: the spaces differ.
    def total(tile) -> int:
        return sum(1 for _ in iter_leaves(schedule(tile, "t", tile.knobs, ctx)))

    assert total(wide) != total(tall), "transposed M/N must not enumerate the same space"
    assert schedule(wide, "t", wide.knobs, ctx)[0].pool_id != schedule(tall, "t", tall.knobs, ctx)[0].pool_id, (
        "so they must not share a pool entry"
    )


def test_split_dim_store_does_not_share_an_identity() -> None:
    """A buffer's SHAPE and the store's index are enumeration inputs, so both identities carry them.

    The same iteration space can reach its output flat (``128x128``) or through a re-fused split
    axis spelled as a dim pair (``4x32x128``, index ``a0/32, a0%32, a1``). The fragment store can
    address the pair only under a divisibility rule, so the split form loses the warp tier — 50538
    candidates against 10284. The term carries neither fact: ``TileOp.structural_key`` excludes the
    stores by design and the algebra digest canonicalizes sizes away, so both reached one
    deploy identity as well as one pool digest. The deploy collision is the worse half: a
    golden measured on the flat kernel would be handed to a kernel that cannot realize its row.
    """
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import lift_loop_op
    from emmy.compiler.pipeline.passes.lowering.tile._schedule import schedule

    matmul = "(torch.randn(128,64,dtype=torch.float16) @ torch.randn(64,128,dtype=torch.float16))"
    ctx = Context.from_target((12, 0))

    def lifted(code: str):
        graph, _, _ = graph_from_code(code)
        out = Pipeline.build(LOOP_PASSES).run(graph)
        node = [n for n in out.nodes.values() if isinstance(n.op, LoopOp)][-1]
        tile = lift_loop_op(node.op, name=node.op.name)
        tile.knobs, tile.inputs, tile.outputs = dict(node.op.knobs), dict(node.op.inputs), dict(node.op.outputs)
        return tile

    flat, split = lifted(matmul), lifted(f"{matmul}.reshape(4,32,128)")

    # The premise: the iteration space agrees — only the store boundary differs.
    assert [a.extent.as_static() for a in flat.place.free] == [a.extent.as_static() for a in split.place.free]

    def total(tile) -> int:
        return sum(1 for _ in iter_leaves(schedule(tile, "t", tile.knobs, ctx)))

    assert total(flat) != total(split), "a split-pair store must not offer the same tiers"
    assert flat.deploy_identity() != split.deploy_identity(), "so a golden must not join across them"
    assert schedule(flat, "t", flat.knobs, ctx)[0].pool_id != schedule(split, "t", split.knobs, ctx)[0].pool_id, (
        "and they must not share a pool"
    )


def test_a_split_receipt_never_shares_a_pool_with_its_receipt_free_twin(monkeypatch) -> None:
    """A kernel that realized a cross-CTA split carries the sliced axis's partition ``Window``
    receipt, and the walk consumes a live ``REDUCE`` pin's ``g``-half against it — so under
    ``EMMY_REDUCE=g2k`` the partial piece memoizes its STRIPPED options while a receipt-free twin
    of the same algebra must RAISE, never hit that memo. The receipt separates the two keys twice
    over (it is an explicit key term, and ``form``'s field walk happens to serialize the
    ``compare=False`` ``Axis.window`` into the term digest); this holds whichever layer does it."""
    from dataclasses import replace

    from emmy.compiler.ir.tensor.ir import ReduceOp
    from emmy.compiler.ir.tile.ir import TileOp
    from emmy.compiler.ir.tile.ops import carries_partition
    from emmy.compiler.pipeline.passes.lowering.tile._schedule import schedule

    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (Dim(4), Dim(512)), dtype="f32"), node_id="x")
    g.add_node(ReduceOp(axis=1), ["x"], Tensor("s", (Dim(4), Dim(1)), dtype="f32"), node_id="s")
    g.inputs, g.outputs = ["x"], ["s"]

    ctx = Context.from_target((12, 0))
    captured: list[TileOp] = []

    def decide(fp):
        op = fp.root_op
        if isinstance(op, TileOp) and op.op is not None and not op.place.is_mapped and carries_partition(op.op):
            if not any(op is seen for seen in captured):
                captured.append(op)
        return next(iter_leaves(fp.options))

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(g, decide)
    assert captured, "the pinned split must have offered its partial piece to the schedule walk"
    partial = captured[0]

    def strip(ax):
        return replace(ax, window=None) if ax.window is not None and ax.window.partition else ax

    twin = TileOp(op=partial.op.rewrite(lambda n: n, None, strip), place=partial.place, output_specs=partial.output_specs)
    twin.knobs, twin.inputs, twin.outputs = dict(partial.knobs), dict(partial.inputs), dict(partial.outputs)
    assert not carries_partition(twin.op), "the twin differs from the partial in the receipt alone"
    with pytest.raises(ValueError, match="names a cross-CTA split"):
        schedule(twin, "t", twin.knobs, ctx)
    hits = ctx.session_cache.hits
    assert schedule(partial, "t", partial.knobs, ctx), "the receipt-carrying partial still replays"
    assert ctx.session_cache.hits > hits, "…and it replays from its own memo entry"


def test_an_axis_renamed_twin_enumerates_its_own_spellings() -> None:
    """The spelled key vocabulary is a pool-key term: tree-path keys spell axis names, so a twin
    differing only in an axis name must enumerate rows under ITS spellings — replaying the other
    kernel's memo would hand materialization keys its own tree cannot decode. (Axis names are
    recognition-canonical identity, so the term digest also separates the twins today; the
    explicit key term and this test hold if that ever changes.)"""
    from dataclasses import replace

    from emmy.commands.trace import graph_from_code
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.sigma import Sigma
    from emmy.compiler.ir.tile.ir import TileOp
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import lift_loop_op
    from emmy.compiler.pipeline.passes.lowering.tile._schedule import schedule

    norm = "(lambda t: t*torch.rsqrt((t.float()*t.float()).mean(-1,keepdim=True)+1e-6).to(t.dtype))"
    code = f"torch.nn.functional.linear({norm}(torch.randn(128, 256, dtype=torch.float16)), torch.randn(256, 256, dtype=torch.float16))"
    out = Pipeline.build(LOOP_PASSES).run(graph_from_code(code)[0])
    node = [n for n in out.nodes.values() if isinstance(n.op, LoopOp)][-1]
    tile = lift_loop_op(node.op, name=node.op.name)
    tile.knobs, tile.inputs, tile.outputs = dict(node.op.knobs), dict(node.op.inputs), dict(node.op.outputs)

    ctx = Context.from_target((12, 0))
    row = dict(next(iter_leaves(schedule(tile, "t", tile.knobs, ctx))).knobs)
    spelled = next(k for k in row if "@" in k)
    old = spelled.split("@", 1)[1]
    new = old + "x"

    def ren(ax):
        return replace(ax, name=new) if ax.name == old else ax

    twin_op = tile.op.rewrite(lambda n: n, Sigma({old: Var(new)}), ren)
    twin = TileOp(op=twin_op, place=tile.place, output_specs=tile.output_specs)
    twin.knobs, twin.inputs, twin.outputs = dict(tile.knobs), dict(tile.inputs), dict(tile.outputs)
    twin_row = dict(next(iter_leaves(schedule(twin, "t", twin.knobs, ctx))).knobs)
    family = spelled.split("@", 1)[0]
    assert f"{family}@{new}" in twin_row, "the twin must enumerate under its own vocabulary"
    assert spelled not in twin_row, "…and never replay the other kernel's spellings"
