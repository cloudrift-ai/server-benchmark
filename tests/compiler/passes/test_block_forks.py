"""Blocking a twisted carrier — the form, and what the row does with it.

Blocking is a NORMALIZATION, not a decision: it re-associates a fold over ``k_o × k_i`` without
changing what the kernel computes, and the width is the FORM's — read off the stream's own extent,
so nothing downstream sizes itself against a symbol. It therefore adds no schedule family and no
option, and the row that schedules a blocked site chunks the block with its own ``bk`` exactly as
it chunks any other K.

It runs as a pass rather than from ``TileOp.__post_init__`` because the kernel-set cut mints
carriers of its own: a piece cut away from the value channel has nothing a block would give it.
"""

from __future__ import annotations

import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.ir.schedule.classic_projection import project_classic
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.block import MAX_BLOCK, block, block_width, is_blocked
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline

_CTX = Context.from_target((12, 0))

_MATMUL = """
x = torch.randn(256, 1024, dtype=torch.float16)
torch.nn.Linear(1024, 512, bias=False, dtype=torch.float16)(x)
"""
_REDUCE = """
x = torch.randn(64, 4096, dtype=torch.float32)
x.sum(-1)
"""
_SOFTMAX = """
x = torch.randn(64, 512, dtype=torch.float32)
torch.softmax(x, -1)
"""
_SDPA = """
import torch.nn.functional as F
q = torch.randn(1, 4, 128, 32, dtype=torch.float16)
F.scaled_dot_product_attention(q, q.clone(), q.clone())
"""


def _tile(code: str, blocked: bool = True) -> TileOp:
    """The one unmapped ``TileOp`` of ``code``, through the blocking pass (or short of it)."""
    graph, _, _ = graph_from_code(code)
    graph = Pipeline.build(LOOP_PASSES).run(graph)
    select = ["lift", "twisted", *(["block"] if blocked else [])]
    graph = Pipeline.build(["lowering/tile"], select=select).run(graph)
    tiles = [node.op for node in graph.nodes.values() if isinstance(node.op, TileOp) and node.op.op is not None]
    assert len(tiles) == 1, [tile.name for tile in tiles]
    return tiles[0]


def _site(tile: TileOp, path: str) -> int:
    return next(index for index in range(len(tile.sites)) if tile.sites[index].path == path)


# ---- which carriers are blocked ---------------------------------------------------------------- #


def test_a_twisted_carrier_with_a_value_channel_is_blocked() -> None:
    """Attention's ``P·V`` is a coefficient of a twisted ⊕ until the block separates the two
    monoids, and then it is a contraction the tensor-core tier can read."""
    tile = _tile(_SDPA)
    outer = tile.sites[_site(tile, "map.1/twist")].node
    assert outer.as_reduction().ops is None  # still the twisted carrier
    inner = [edge for edge in outer.operands if edge.axis is not None]
    assert len(inner) == 3  # the block pivot, the expectation, the denominator
    assert [edge.as_contraction() is not None for edge in inner] == [False, True, False]


@pytest.mark.parametrize("code", [_MATMUL, _REDUCE, _SOFTMAX])
def test_nothing_a_block_gives_nothing_to_is_blocked(code: str) -> None:
    """A contraction's block is already spelled by ``bk`` and a plain reduction's partition by
    ``REDUCE``. A plain online SOFTMAX is twisted and still declines: its channels are sums of the
    weight itself, so nothing in it comes out bilinear and the block would buy a second pass over
    the stream for nothing."""
    tile = _tile(code)
    assert not is_blocked(tile.axes)
    for site in tile.sites:
        if site.node.axis is not None:
            assert block(site.node, tile.axis_of(site.node.axis)) is None


def test_the_channels_share_one_binder_and_one_weight() -> None:
    """Two passes over a block is the floor — the weights read a pivot that has to be finished —
    and the channels are the second of them, together: one loop, one weight instance."""
    tile = _tile(_SDPA)
    outer = tile.sites[_site(tile, "map.1/twist")].node
    pivot, expectation, denominator = (edge for edge in outer.operands if edge.axis is not None)
    assert pivot.axis != expectation.axis
    assert expectation.axis == denominator.axis
    assert expectation.operands[0] is denominator.operands[0]


# ---- the width is the form's -------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("extent", "width"),
    [(128, 64), (512, 64), (64, 32), (96, 48), (127, None), (2, 1)],
)
def test_the_width_is_a_whole_fraction_of_the_stream(extent: int, width: int | None) -> None:
    """The largest power-of-two fraction within :data:`MAX_BLOCK`. A whole number of blocks always:
    an odd extent is not cut at all, because a masked tail would have to be threaded through the
    pivot and every channel separately."""
    assert block_width(extent) == width
    assert width is None or (extent % width == 0 and width <= MAX_BLOCK)


def test_the_block_lives_on_the_axes_and_only_there() -> None:
    """The outer axis walks the stream's own extent in strides and the binders' extent IS the
    block, so the σ that reads the absolute coordinate is plain ``k_o + k_i`` and no lambda binds
    a width."""

    def terms(term):
        yield term
        for edge in term.operands:
            yield from terms(edge)

    tile = _tile(_SDPA)
    outer = tile.axis_of("a2_o")
    assert outer.extent.as_static() == 128
    assert outer.step.value == 64
    assert outer.trips == 2
    binders = [axis for axis in tile.axes if axis.window is not None and axis.window.block and axis.step is None]
    assert binders and all(axis.extent.as_static() == 64 for axis in binders)
    assert all(axis.extent.is_static for axis in tile.axes)
    assert tile.loop_body is not None


def test_blocking_is_idempotent() -> None:
    """Every installed axis carries the receipt, so the pass fires once."""
    tile = _tile(_SDPA)
    from emmy.compiler.ir.tile.block import block_tree  # noqa: PLC0415

    assert block_tree(tile.op, tile.axes) is None


# ---- what the scheduler does with it ------------------------------------------------------------- #


def test_the_created_site_carries_the_tensor_core_tier() -> None:
    """Blocking exists to make this domain non-empty: no site inside a twisted carrier is bilinear
    until the two monoids are separated."""
    tile = _tile(_SDPA)
    choices = project_classic(tile, _CTX).nodes[_site(tile, "map.1/twist.2/inner")]
    assert choices and all(choice.tile.is_warp for choice in choices)


def test_a_site_inside_a_block_consumes_exactly_the_block() -> None:
    """The channel's K-step IS the block (one trip) and the score's fragment grid covers it, so a
    row that half-covers either is not offered — it would emit a different blocking from the one
    the term spells."""
    tile = _tile(_SDPA)
    domains = project_classic(tile, _CTX)
    channel = domains.nodes[_site(tile, "map.1/twist.2/inner")]
    assert {choice.tile.atom.atom_k * choice.tile.bk for choice in channel} == {64}
    score = domains.nodes[_site(tile, "map.1/twist.2/inner.1/map.1/inner")]
    assert {choice.tile.regs[1] * choice.tile.atom.atom_n for choice in score} == {64}


def test_no_row_key_spells_the_block() -> None:
    """Blocking adds no codec family: the block is the form's, and the row chunks it with ``bk``."""
    from emmy.compiler.ir.schedule.classic import ClassicScheduleCodec, ClassicScheduleContext  # noqa: PLC0415

    tile = _tile(_SDPA)
    context = ClassicScheduleContext(tile, _CTX, project_classic(tile, _CTX))
    assert not any(key.startswith("BLOCK") for key in ClassicScheduleCodec(context).keys())
