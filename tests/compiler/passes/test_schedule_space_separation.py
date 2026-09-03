"""Schedule-space separation — enumeration inputs that must move the offered space / the pool
stamp, spelled as direct assertions on `schedule()`. Every call enumerates its own prescan, and
`Fork.pool_id` is the stable identity of that space.

What these pin: (a) a live precision-gate pin rides the pin fingerprint; (b) transposed free
extents and a split-dim store change both the enumerated space and the stamp; (c) an α-renamed
twin enumerates rows under ITS OWN spelled vocabulary."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.fork import iter_leaves

classic_forks = import_module("emmy.compiler.pipeline.passes.lowering.tile.040_schedule").classic_forks


def _unmapped_tile(m: int, n: int, k: int = 64, dtype: str = "f16"):
    """The lifted, unscheduled ``TileOp`` for one ``m x k @ k x n`` matmul, knobs and all.

    Lifting by hand rather than through ``lowering/tile`` lets the assertion below inspect the
    schedule-space stamp directly instead of inferring a collision from downstream symptoms.
    """
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import lift_loop_op

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Dim(m), Dim(k)), dtype=dtype), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (Dim(k), Dim(n)), dtype=dtype), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (Dim(m), Dim(n)), dtype=dtype), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    node = Pipeline.build(LOOP_PASSES).run(g).nodes["o"]
    tile = lift_loop_op(node.op, name=node.op.name)
    return replace(tile, knobs=dict(node.op.knobs), inputs=dict(node.op.inputs), outputs=dict(node.op.outputs))


def test_a_precision_gate_pin_stamps_a_different_space() -> None:
    """The ``F16_MMA_F32_ACC`` gate is not a schedule family but it changes which rows are
    OFFERED (the f16-accumulate atom siblings), so it must ride the pin fingerprint — the fit's
    fm-pinned reconstruction shares one Context per card and would otherwise collide with the
    unpinned pools."""
    from emmy.compiler.pipeline.knob import schedule_pin_fingerprint
    from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC

    clean = schedule_pin_fingerprint()
    with F16_MMA_F32_ACC.pinned("1"):
        assert schedule_pin_fingerprint() != clean, "the precision gate must change the schedule-space stamp"
    assert schedule_pin_fingerprint() == clean


def test_transposed_free_extents_stamp_different_spaces() -> None:
    """Per-axis extents are an ENUMERATION input, so they must be a key part.

    The term's algebra digest canonicalizes sizes away and the knobs carry only the lossy
    ``S_ext_*`` summary (count / product / max), which ``8x512`` and ``512x8`` agree on. They
    therefore have the same algebra digest and structural feature summary while their spaces
    differ by 7x — the enumeration sizes the coop band against ``_inner_free`` and the fragment
    store against the free axes. Their space stamps must preserve that distinction.
    """
    ctx = Context.from_target((12, 0))
    wide, tall = _unmapped_tile(8, 512), _unmapped_tile(512, 8)

    # The consequence of transposing: the spaces differ.
    def total(tile) -> int:
        return sum(1 for _ in iter_leaves(classic_forks(tile, "t", tile.knobs, ctx)))

    assert total(wide) != total(tall), "transposed M/N must not enumerate the same space"
    assert classic_forks(wide, "t", wide.knobs, ctx)[0].pool_id != classic_forks(tall, "t", tall.knobs, ctx)[0].pool_id, (
        "so they must not share a schedule-space stamp"
    )


def test_split_dim_store_does_not_share_an_identity() -> None:
    """A buffer's SHAPE and the store's index are enumeration inputs, so both identities carry them.

    The same iteration space can reach its output flat (``128x128``) or through a re-fused split
    axis spelled as a dim pair (``4x32x128``, index ``a0/32, a0%32, a1``). The fragment store can
    address the pair only under a divisibility rule, so the split form loses the warp tier — 50538
    candidates against 10284. The term carries neither fact: ``TileOp.structural_key`` excludes the
    stores by design and the algebra digest canonicalizes sizes away. The complete Loop-body
    identity and the schedule-space stamp must both carry the store boundary so a golden measured
    on the flat kernel is never handed to a kernel that cannot realize its row.
    """
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import lift_loop_op

    matmul = "(torch.randn(128,64,dtype=torch.float16) @ torch.randn(64,128,dtype=torch.float16))"
    ctx = Context.from_target((12, 0))

    def lifted(code: str):
        graph, _, _ = graph_from_code(code)
        out = Pipeline.build(LOOP_PASSES).run(graph)
        node = [n for n in out.nodes.values() if isinstance(n.op, LoopOp)][-1]
        tile = lift_loop_op(node.op, name=node.op.name)
        return replace(tile, knobs=dict(node.op.knobs), inputs=dict(node.op.inputs), outputs=dict(node.op.outputs))

    flat, split = lifted(matmul), lifted(f"{matmul}.reshape(4,32,128)")

    # The premise: the iteration space agrees — only the store boundary differs.
    assert [a.extent.as_static() for a in flat.place.free] == [a.extent.as_static() for a in split.place.free]

    def total(tile) -> int:
        return sum(1 for _ in iter_leaves(classic_forks(tile, "t", tile.knobs, ctx)))

    assert total(flat) != total(split), "a split-pair store must not offer the same tiers"
    assert flat.identity_key(with_io=True) != split.identity_key(with_io=True), "so a golden must not join across them"
    assert classic_forks(flat, "t", flat.knobs, ctx)[0].pool_id != classic_forks(split, "t", split.knobs, ctx)[0].pool_id, (
        "and they must not share a schedule-space stamp"
    )


def test_an_axis_renamed_twin_preserves_the_node_id_vocabulary() -> None:
    """Classic site keys are stable ordinals, not structural paths or axis names. An alpha-renamed
    twin therefore enumerates the same key vocabulary while resolving each ``NodeId`` against its
    own problem-local index."""
    from dataclasses import replace

    from emmy.commands.trace import graph_from_code
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.sigma import Sigma
    from emmy.compiler.ir.stmt.passes import rewrite
    from emmy.compiler.ir.tile.ir import TileOp
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import lift_loop_op

    norm = "(lambda t: t*torch.rsqrt((t.float()*t.float()).mean(-1,keepdim=True)+1e-6).to(t.dtype))"
    code = f"torch.nn.functional.linear({norm}(torch.randn(128, 256, dtype=torch.float16)), torch.randn(256, 256, dtype=torch.float16))"
    out = Pipeline.build(LOOP_PASSES).run(graph_from_code(code)[0])
    node = [n for n in out.nodes.values() if isinstance(n.op, LoopOp)][-1]
    tile = lift_loop_op(node.op, name=node.op.name)
    tile = replace(tile, knobs=dict(node.op.knobs), inputs=dict(node.op.inputs), outputs=dict(node.op.outputs))

    ctx = Context.from_target((12, 0))
    row = dict(next(iter_leaves(classic_forks(tile, "t", tile.knobs, ctx))).knobs)
    spelled = next(k for k in row if "@" in k)
    old = spelled.split("@", 1)[1]
    new = old + "x"

    def ren(ax):
        return replace(ax, name=new) if ax.name == old else ax

    twin_op = rewrite(tile.op, lambda n: n, Sigma({old: Var(new)}), ren)
    twin = TileOp(op=twin_op, place=tile.place, output_specs=tile.output_specs, axes=tuple(ren(axis) for axis in tile.axes))
    twin = replace(twin, knobs=dict(tile.knobs), inputs=dict(tile.inputs), outputs=dict(tile.outputs))
    twin_row = dict(next(iter_leaves(classic_forks(twin, "t", twin.knobs, ctx))).knobs)
    assert spelled in twin_row
    assert {key for key in row if "@" in key} == {key for key in twin_row if "@" in key}
