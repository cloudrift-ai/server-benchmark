"""The `Placed` view vs the stored `Contraction` node — the boundary the tile IR draws.

A contraction node is pure algebra (`k_axis` + the `a` edge + its `Channel`s); placement and
schedule ride the lowering-side `Placed` view. Helpers that EDIT the algebra of a placed
contraction must go through `Placed.replace_node`, not `dataclasses.replace` — the latter raises
`TypeError` because the algebra names are not fields of the view.

Both helpers pinned here sit on GPU-gated paths (the mixed-dtype warp demotion, the warp-flash PV
stream), so the ordinary CPU suite never executes them; calling them directly is what makes the
regression visible off-GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

from emmy.compiler.dtype import F16, F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import TilePlan
from emmy.compiler.ir.stmt import Load
from emmy.compiler.ir.tile import Channel, Contraction
from emmy.compiler.pipeline.passes.lowering._placed import Placed, place
from emmy.compiler.pipeline.passes.lowering.kernel._twist import _pv_streamed
from emmy.compiler.pipeline.passes.lowering.tile._schedule import _demote_mixed_a

_M, _N, _K = Axis("m", 128), Axis("n", 128), Axis("k", 64)


def _placed(tile: TilePlan | None = None) -> Placed:
    node = Contraction(
        k_axis=_K,
        a=Load(name="a", input="A", index=(Var("m"), Var("k"))),
        channels=(Channel(b=Load(name="b", input="B", index=(Var("k"), Var("n"))), acc="acc"),),
    )
    return place(node, _M, _N, (), tile=tile or TilePlan(units=(2, 1), regs=(2, 1)))


def test_the_node_carries_no_placement_or_schedule() -> None:
    """The invariant the view exists to keep: a contraction's identity is its algebra, so the
    same node under two tiles is ONE term."""
    p1, p2 = _placed(TilePlan(units=(2, 1), regs=(2, 1))), _placed(TilePlan(units=(4, 1), regs=(4, 1)))
    assert p1.node == p2.node and hash(p1.node) == hash(p2.node)
    assert not hasattr(p1.node, "tile") and not hasattr(p1.node, "axes")
    assert p1.block_threads != p2.block_threads  # the geometry is the VIEW's


def test_replace_node_edits_the_algebra_and_keeps_the_binding() -> None:
    p = _placed()
    swapped = p.replace_node(k_axis=Axis("kv", 256))
    assert isinstance(swapped, Placed)
    assert swapped.k_axis.name == "kv"  # the algebra changed
    assert swapped.axes == p.axes and swapped.tile == p.tile  # the binding rode along


def test_pv_streamed_swaps_the_stream_axis_on_a_placed_pv() -> None:
    """The warp-flash PV contracts the whole key block, so its intra-block axis is swapped for the
    stream axis — an ALGEBRA edit on a placed node."""
    p = _placed()
    got = _pv_streamed(p, Axis("kv", 256))
    assert isinstance(got, Placed) and got.k_axis.name == "kv"
    assert got.tile == p.tile and got.axes == p.axes


def test_demote_mixed_a_rewrites_the_a_edge_on_a_placed_node() -> None:
    """A mixed-dtype (f32-A × 16-bit-B) contraction re-expresses its A ``Load`` as a computed cone
    so it can ride the demoting sync compute-fill — again an algebra edit through the view."""
    kernel = SimpleNamespace(inputs={"A": SimpleNamespace(dtype=F32, shape=()), "B": SimpleNamespace(dtype=F16, shape=())})
    p = _placed()
    out = _demote_mixed_a(kernel, p)
    assert isinstance(out, Placed) and out.a_computed  # the A edge became an inline cone
    assert out.tile == p.tile and out.axes == p.axes


def test_demote_mixed_a_passes_through_a_uniform_dtype_contraction() -> None:
    kernel = SimpleNamespace(inputs={"A": SimpleNamespace(dtype=F16, shape=()), "B": SimpleNamespace(dtype=F16, shape=())})
    p = _placed()
    assert _demote_mixed_a(kernel, p) is p
