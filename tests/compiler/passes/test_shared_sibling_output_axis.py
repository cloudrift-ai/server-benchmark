"""Sibling output sweeps share ONE grid axis; the narrow ones guard their stores.

A kernel that writes several outputs of DIFFERENT extents carries one output sweep per extent
(``_fromloop.extract_output_specs``), and ``TileOp.__post_init__`` promotes a sweep a contraction is
evaluated over onto the placement, so the contraction under it gets a grid to parallelise on.

Promoting each such sweep as its OWN free axis makes the grid their CARTESIAN PRODUCT, and a kernel
whose grid is a product enumerates every region once per cell of every other region. The stores are
idempotent, so the kernel stays correct while doing the work of the product of its output axes
instead of their sum: the fused q/k/v projection below recomputed and rewrote q once per column of
the k/v axis — a 32x multiplier on a serving kernel — and the NVFP4 post-attention re-encode, whose
three outputs are 2048, 256 and 4096 wide, asked for a 2^31-point launch.

The placement instead maps every promoted sibling onto ONE axis, the widest, and guards each
narrower region's store to its own extent. One enumeration, the tensor-core tier preserved (the
tiled root still finds an ``(m, n)`` pair on the grid), and the redundant recompute gone.

The oracle is the placement's own arithmetic plus the guards the narrow regions carry.
"""

from __future__ import annotations

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp
from emmy.compiler.ir.stmt.leaves import Write
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import KERNEL_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.target import set_target

_CAP = (12, 0)
_TOKENS = Dim("num_tokens")
_K, _N_Q, _N_KV = 64, 64, 32  # q's N differs from k/v's — two output extents on one kernel

#: Why these are open: the promotion still appends one free axis per sibling sweep. Closing it
#: needs a per-region store guard, which the masked-N machinery spells only for the TILED
#: root's own store (``_atom._guard_writes`` off a ``Side``), not for a sibling riding the grid.
_GAP = "sibling output sweeps still promote as a product grid (stage-2b bug 6)"


def _qkv_graph() -> Graph:
    """The fused projection: one shared computed operand feeding three linears, q wider than k/v.
    The same shape ``test_untiled_grid_axis`` builds — there for the axis BINDING, here for the
    placement's shape."""
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (_TOKENS, _K), dtype=F32), node_id="x")
    for weight, n in (("wq", _N_Q), ("wk", _N_KV), ("wv", _N_KV)):
        graph.add_node(InputOp(), [], Tensor(weight, (n, _K), dtype=F32), node_id=weight)
    graph.add_node(ElementwiseOp("multiply"), ["x", "x"], Tensor("xn", (_TOKENS, _K), dtype=F32), node_id="xn")
    for out, weight, n in (("q", "wq", _N_Q), ("k", "wk", _N_KV), ("v", "wv", _N_KV)):
        graph.add_node(LinearOp(), ["xn", weight], Tensor(out, (_TOKENS, n), dtype=F32), node_id=out)
    graph.inputs, graph.outputs = ["x", "wq", "wk", "wv"], ["q", "k", "v"]
    return graph


def _run(passes) -> Graph:
    set_target(_CAP)
    try:
        return Pipeline.build(passes).run(_qkv_graph(), ctx=Context(compute_capability=_CAP))
    finally:
        set_target(None)


def _the_tile() -> TileOp:
    tiles = [node.op for node in _run(TILE_PASSES).nodes.values() if isinstance(node.op, TileOp)]
    assert len(tiles) == 1, "the three projections must fuse into one kernel for this shape to arise"
    return tiles[0]


def _static_extents(tile: TileOp) -> list[int]:
    """The free axes' static extents — the symbolic token axis dropped, so the assertion is about
    the OUTPUT axes the promotion decides."""
    return [axis.extent.as_static() for axis in tile.place.free if axis.extent.is_static]


@pytest.mark.xfail(strict=True, reason=_GAP)
def test_sibling_output_sweeps_promote_onto_one_axis() -> None:
    """Two output extents (64, 32) must leave ONE static free axis, the wider — not both."""
    extents = _static_extents(_the_tile())
    assert extents == [_N_Q], f"expected one shared output axis of {_N_Q}, got free extents {extents}"


@pytest.mark.xfail(strict=True, reason=_GAP)
def test_the_grid_enumerates_its_output_cells_once() -> None:
    """The product of the free extents is the SUM shape, not the product of the output widths: the
    32x recompute the per-sibling promotion caused."""
    tile = _the_tile()
    product = 1
    for extent in _static_extents(tile):
        product *= extent
    assert product == _N_Q, f"grid enumerates {product} output columns; one enumeration is {_N_Q}"


@pytest.mark.xfail(strict=True, reason=_GAP)
def test_the_narrow_outputs_guard_their_stores_and_the_widest_does_not() -> None:
    """k and v live on the first 32 columns of a 64-wide axis, so their stores are guarded by that
    extent; q spans the axis and takes no such guard."""
    kernels = [node.op for node in _run(KERNEL_PASSES).nodes.values() if getattr(node.op, "body", None) is not None]
    assert len(kernels) == 1
    guarded: dict[str, bool] = {}

    def walk(stmts, conditions: tuple[str, ...]) -> None:
        for stmt in stmts:
            nested = stmt.nested()
            cond = getattr(stmt, "cond", None)
            inner = (*conditions, cond.pretty()) if cond is not None else conditions
            for name in (getattr(stmt, "output", None),) if isinstance(stmt, Write) else ():
                if name is not None:
                    guarded[name] = guarded.get(name, False) or any(str(_N_KV) in text for text in inner)
            for body in nested:
                walk(list(body), inner)

    walk(list(kernels[0].body), ())
    assert guarded.get("k") and guarded.get("v"), f"k/v stores must be guarded to their {_N_KV} columns; got {guarded}"
    assert not guarded.get("q"), f"q spans the shared axis and needs no {_N_KV} guard; got {guarded}"
