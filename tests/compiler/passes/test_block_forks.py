"""Blocking a twisted carrier — the canonical form, and the width the row binds.

Blocking is a NORMALIZATION, not a decision: it re-associates a fold over ``k_o × k_i`` without
changing what the kernel computes, and the width appears nowhere in the term — the outer axis
strides and each binder's extent is a symbol. So it runs from ``TileOp.__post_init__``, adds no
schedule family and no option, and every block form is the same kernel. The width is bound at
materialization, from the ``TILE`` at the site blocking created, whose mma K-step it is.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.ir.axis import block_width_var
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import Reduce, Schedule, Tile
from emmy.compiler.ir.schedule.classic import ProjectionSchedule, ReductionSchedule
from emmy.compiler.ir.schedule.classic_projection import _block_widths, project_classic
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.block import bind_widths, block, is_blocked
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
_SDPA = """
import torch.nn.functional as F
q = torch.randn(1, 4, 128, 32, dtype=torch.float16)
F.scaled_dot_product_attention(q, q.clone(), q.clone())
"""


def _tile(code: str) -> TileOp:
    """The one unmapped ``TileOp`` of ``code`` — already normalized, so already blocked."""
    graph, _, _ = graph_from_code(code)
    graph = Pipeline.build(LOOP_PASSES).run(graph)
    graph = Pipeline.build(["lowering/tile"], select=["lift", "twisted"]).run(graph)
    tiles = [node.op for node in graph.nodes.values() if isinstance(node.op, TileOp) and node.op.op is not None]
    assert len(tiles) == 1, [tile.name for tile in tiles]
    return tiles[0]


def _site(tile: TileOp, path: str) -> int:
    return next(index for index in range(len(tile.sites)) if tile.sites[index].path == path)


# ---- which carriers are blocked ---------------------------------------------------------------- #


def test_a_twisted_carrier_is_blocked_by_normalization() -> None:
    """Attention's ``P·V`` is a coefficient of a twisted ⊕ until the block splits the two monoids,
    and then it is a contraction the tensor-core tier can read."""
    tile = _tile(_SDPA)
    outer = tile.sites[_site(tile, "map.1/twist")].node
    assert outer.as_reduction().ops is None  # still the twisted carrier
    inner = [edge for edge in outer.operands if edge.axis is not None]
    assert len(inner) == 3  # the block pivot, the expectation, the denominator
    assert [edge.as_contraction() is not None for edge in inner] == [False, True, False]


@pytest.mark.parametrize("code", [_MATMUL, _REDUCE])
def test_nothing_else_is_blocked(code: str) -> None:
    """A contraction's block is already spelled by ``bk`` and a plain reduction's partition by
    ``REDUCE``, so splitting either term would restate another family's decision as a shape."""
    tile = _tile(code)
    assert not is_blocked(tile.axes)
    node = tile.sites[0].node
    assert block(node, tile.axis_of(node.axis)) is None


# ---- the width is nowhere in the term ------------------------------------------------------------ #


def test_the_width_lives_on_the_axes_and_only_there() -> None:
    """The outer axis walks the stream's own extent in strides and each binder's extent is the
    symbol, so the σ that reads the absolute coordinate is plain ``k_o + k_i``."""
    tile = _tile(_SDPA)
    variable = block_width_var("a2")
    outer = tile.axis_of("a2_o")
    assert outer.extent.as_static() == tile.axis_of("a2").extent.as_static()
    assert outer.step == Var(variable)
    assert outer.block_width == variable
    binders = [axis for axis in tile.axes if axis.window is not None and axis.window.block == variable and not axis.window.trip]
    assert binders and all(axis.extent.expr == Var(variable) for axis in binders)


def test_the_term_names_no_width() -> None:
    """Parameter-free is what makes it a normalization: no lambda binds the symbol, and the whole
    kernel still lowers with the width unbound."""

    def terms(term):
        yield term
        for edge in term.operands:
            yield from terms(edge)

    tile = _tile(_SDPA)
    variable = block_width_var("a2")
    assert not any(variable in term.lift.params for term in terms(tile.op))
    assert tile.loop_body is not None  # lowers with the width still a symbol


def test_blocking_is_idempotent() -> None:
    """Every installed axis carries the receipt, so a reconstructed TileOp blocks nothing again."""
    tile = _tile(_SDPA)
    again = replace(tile)
    assert again.op == tile.op
    assert [axis.name for axis in again.axes] == [axis.name for axis in tile.axes]


# ---- the width the row binds --------------------------------------------------------------------- #


def _assignment(tile: TileOp, tiles: dict[int, Tile]) -> Schedule:
    nodes = {}
    for site in tile.node_sites:
        chosen = tiles.get(site, Tile())
        nodes[site] = ReductionSchedule(chosen, Reduce()) if tile.views[site].axis is not None else ProjectionSchedule(chosen)
    return Schedule(None, nodes, {})


def test_the_tile_at_the_created_site_binds_the_width() -> None:
    """A blocked site's inner axis IS its K, so the block is exactly that tile's mma K-step."""
    tile = _tile(_SDPA)
    site = _site(tile, "map.1/twist.2/inner")
    atom = next(choice.tile.atom for choice in project_classic(tile, _CTX).nodes[site] if choice.tile.is_warp)
    widths = _block_widths(tile, _assignment(tile, {site: Tile(atom=atom, bk=4)}))
    assert widths == {block_width_var("a2"): atom.atom_k * 4}


def test_a_scalar_row_leaves_the_stream_in_one_trip() -> None:
    """Nothing spells a K-step, so the width falls back to the whole extent — which is the
    unblocked kernel, reached without a second form of the term."""
    tile = _tile(_SDPA)
    assert _block_widths(tile, _assignment(tile, {})) == {block_width_var("a2"): 128}


def test_binding_moves_only_the_axis_table() -> None:
    """The term is untouched by the binding, which is why every block form is the same kernel."""
    tile = _tile(_SDPA)
    bound = bind_widths(tile.axes, {block_width_var("a2"): 64})
    assert all(axis.extent.is_static for axis in bound)
    assert next(axis for axis in bound if axis.name == "a2_p").extent.as_static() == 64
    assert next(axis for axis in bound if axis.name == "a2_o").step.value == 64


# ---- what the scheduler does with it ------------------------------------------------------------- #


def test_the_created_site_carries_the_tensor_core_tier() -> None:
    """Blocking exists to make this domain non-empty: no site inside a twisted carrier is bilinear
    until the two monoids are separated."""
    tile = _tile(_SDPA)
    choices = project_classic(tile, _CTX).nodes[_site(tile, "map.1/twist.2/inner")]
    assert any(choice.tile.is_warp for choice in choices)
    # ``bk`` ranges freely, because it is what DEFINES the block rather than something the block
    # constrains — the two are one quantity, spelled once.
    assert len({choice.tile.atom.atom_k * choice.tile.bk for choice in choices if choice.tile.is_warp}) > 1


def test_the_block_axis_sizes_the_fragment_it_tiles() -> None:
    """A nested contraction takes its parent fold's axis as the tile's n. For a BLOCK window that
    is the block's own axis — the enclosing fold walks one block at a time — where a cross-CTA
    slice reads its pre-split parent instead."""
    from emmy.compiler.ir.tile.ops import Sched  # noqa: PLC0415

    tile = _tile(_SDPA)
    sched = Sched(tile, place=tile.place.on_grid())
    placed = sched.placed(tile.sites[_site(tile, "map.1/twist.1/reduce.1/inner")].node, Tile())
    assert [axis.name for axis in placed.axes] == ["a1", "a2_p"]


def test_no_row_key_spells_the_block() -> None:
    """Blocking adds no codec family: the width rides the ``TILE`` whose K-step it is."""
    from emmy.compiler.ir.schedule.classic import ClassicScheduleCodec, ClassicScheduleContext  # noqa: PLC0415

    tile = _tile(_SDPA)
    context = ClassicScheduleContext(tile, _CTX, project_classic(tile, _CTX))
    assert not any(key.startswith("BLOCK") for key in ClassicScheduleCodec(context).keys())
