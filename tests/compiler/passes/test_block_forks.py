"""Blocking a reduce stream into the form the schedule enumerates over.

Blocking is not a decision of its own: it re-associates a fold over ``k_o × k_i`` without changing
what the kernel computes, so it happens inside the schedule walk, the kernel keeps ONE identity
across every block form, and the width needs no codec family — a blocked contraction's inner axis
IS its K, so the row's ``TILE`` at that site already spells it. A plain reduction is not blocked at
all: ``REDUCE`` and the cross-CTA split already partition its axis.
"""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.ir.schedule.classic_projection import block_widths, project_classic
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.tile._block import block_tree, blockable_streams, is_blocked

_SCHEDULE = import_module("emmy.compiler.pipeline.passes.lowering.tile.040_schedule")
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
    """The one unmapped ``TileOp`` of ``code``, lifted and twisted but not yet blocked."""
    graph, _, _ = graph_from_code(code)
    graph = Pipeline.build(LOOP_PASSES).run(graph)
    graph = Pipeline.build(["lowering/tile"], select=["lift", "twisted"]).run(graph)
    tiles = [node.op for node in graph.nodes.values() if isinstance(node.op, TileOp) and node.op.op is not None]
    assert len(tiles) == 1, [tile.name for tile in tiles]
    return tiles[0]


def _stream(code: str):
    """``(tile, the one blockable fold, its axis)``."""
    tile = _tile(code)
    streams = blockable_streams(tile.op, tile.axes)
    assert len(streams) == 1, streams
    node, axis = streams[0]
    return tile, node, axis


def _blocked(code: str, width: int) -> TileOp:
    tile, node, _ = _stream(code)
    op, axes = block_tree(tile.op, tile.axes, {id(node): width})
    return replace(tile, op=op, axes=axes)


def _outer(tile: TileOp):
    return tile.op if tile.op.axis is not None else next(edge for edge in tile.op.operands if edge.axis is not None)


# ---- what each level runs ---------------------------------------------------------------------- #


def test_a_blocked_contraction_keeps_its_bilinear_inner_level() -> None:
    """A matmul blocks into a plain add over a contraction of exactly one block."""
    tile, node, _ = _stream(_MATMUL)
    outer = _outer(_blocked(_MATMUL, 64))
    (inner,) = [edge for edge in outer.operands if edge.axis is not None]
    assert inner.as_contraction() is not None
    assert outer.as_contraction() is None  # one operand: the outer level is a plain reduce
    assert outer.combine == node.combine and outer.init == node.init
    assert [op.name for op in outer.as_reduction().ops] == [op.name for op in inner.as_reduction().ops] == ["add"]


def test_a_blocked_twisted_carrier_exposes_a_bilinear_channel() -> None:
    """The whole point: attention's ``P·V`` is a coefficient of a twisted ⊕ until the block splits
    the two monoids, and then it is a contraction the tensor-core tier can read."""
    _, node, _ = _stream(_SDPA)
    outer = _outer(_blocked(_SDPA, 64))
    assert node.as_reduction().ops is None  # twisted: no componentwise monoid
    assert outer.combine == node.combine  # the carrier is unchanged; only its operands moved
    inner = [edge for edge in outer.operands if edge.axis is not None]
    assert len(inner) == 3  # the block pivot, the expectation, the denominator
    assert [edge.as_contraction() is not None for edge in inner] == [False, True, False]


def test_a_plain_reduction_is_left_to_reduce() -> None:
    """``REDUCE`` already spells its partition and the cross-CTA split already factors its axis."""
    tile = _tile(_REDUCE)
    assert blockable_streams(tile.op, tile.axes) == ()
    assert _SCHEDULE.block_problems(tile, _CTX) == [tile]


def test_every_installed_axis_carries_the_block_receipt() -> None:
    """``Window(block=True)`` is what stops a second block; ``trip`` marks the outer of the pair."""
    blocked = _blocked(_MATMUL, 64)
    assert is_blocked(blocked.axes)
    installed = [axis for axis in blocked.axes if axis.window is not None and axis.window.block]
    assert len(installed) == 2
    assert [axis.extent.as_static() for axis in installed] == [16, 64]
    assert [axis.window.trip for axis in installed] == [True, False]  # the outer of the pair
    assert not any(axis.window.partition for axis in installed)


# ---- the width domain -------------------------------------------------------------------------- #


def test_the_widths_are_the_atom_k_steps() -> None:
    """The block IS the fragment depth, so a twisted carrier offers ``atom_k × bk``."""
    tile, node, axis = _stream(_SDPA)
    assert block_widths(tile, _CTX, node, axis) == (64, 32, 16)


def test_a_contraction_is_offered_no_block() -> None:
    """``bk`` already says how many atom K-steps one inner step consumes and the materializer
    chunks K by it, so re-associating the term would restate that field as a shape."""
    tile, node, axis = _stream(_MATMUL)
    assert node.as_contraction() is not None
    assert block_widths(tile, _CTX, node, axis) == ()
    assert _SCHEDULE.block_problems(tile, _CTX) == [tile]


def test_no_width_divides_into_a_masked_tail() -> None:
    """The ceil form is built and correct, but a masked tail is a realization we do not have."""
    tile, node, axis = _stream(_SDPA)
    extent = axis.extent.as_static()
    assert all(extent % width == 0 and width < extent for width in block_widths(tile, _CTX, node, axis))


# ---- the problems the walk enumerates over ------------------------------------------------------ #


def test_the_unblocked_form_leads() -> None:
    """A cold walk descends what the kernel already was."""
    tile = _tile(_SDPA)
    problems = _SCHEDULE.block_problems(tile, _CTX)
    assert problems[0] is tile
    assert len(problems) == 1 + len(block_widths(tile, _CTX, *blockable_streams(tile.op, tile.axes)[0]))
    assert all(is_blocked(problem.axes) for problem in problems[1:])


def test_every_block_form_is_the_same_kernel() -> None:
    """Blocking changes no kernel identity: it re-associates a fold, it does not change the compute.
    Only the row tells the forms apart, which is why they share one pool and one recorded identity."""
    tile = _tile(_SDPA)
    problems = _SCHEDULE.block_problems(tile, _CTX)
    assert len(problems) > 1
    assert all(problem.knobs == tile.knobs for problem in problems)
    assert _SCHEDULE.rewrite.__module__  # the pass identifies on the unblocked tile, not the problem


def test_a_decided_kernel_is_never_blocked_again() -> None:
    blocked = _blocked(_SDPA, 64)
    assert _SCHEDULE.block_problems(blocked, _CTX) == [blocked]


# ---- what the scheduler does with it ------------------------------------------------------------ #


def _warp_choices(tile: TileOp, path: str) -> list:
    site = next(index for index in range(len(tile.sites)) if tile.sites[index].path == path)
    return [choice for choice in project_classic(tile, _CTX).nodes[site] if choice.tile.is_warp]


def test_the_block_decides_the_tile_k_step() -> None:
    """The block width and the ``k<bk>`` half of a ``TILE`` value are one quantity, so a blocked
    site offers only the tiles that agree with the block it sits in — no second spelling."""
    choices = _warp_choices(_blocked(_SDPA, 64), "map.1/twist.2/inner")
    assert choices
    assert {choice.tile.atom.atom_k * choice.tile.bk for choice in choices} == {64}


def test_an_unblocked_contraction_still_enumerates_every_k_step() -> None:
    choices = _warp_choices(_tile(_MATMUL), "inner")
    assert len({choice.tile.atom.atom_k * choice.tile.bk for choice in choices}) > 1


def test_the_block_axis_sizes_the_fragment_it_tiles() -> None:
    """A nested contraction takes its parent fold's axis as the tile's n. For a BLOCK window that
    is the block's own extent — the enclosing fold walks one block at a time — where a cross-CTA
    slice reads its pre-split parent instead."""
    from emmy.compiler.ir.schedule import Tile as TileChoice  # noqa: PLC0415
    from emmy.compiler.ir.tile.ops import Sched  # noqa: PLC0415

    blocked = _blocked(_SDPA, 64)
    sched = Sched(blocked, place=blocked.place.on_grid())
    site = next(s for s in blocked.sites if s.path == "map.1/twist.1/reduce.1/inner")
    placed = sched.placed(site.node, TileChoice())
    assert [axis.name for axis in placed.axes] == ["a1", "a2_p"]
    assert placed.axes[1].extent.as_static() == 64
