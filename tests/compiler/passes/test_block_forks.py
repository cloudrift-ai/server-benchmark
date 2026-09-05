"""Blocking a reduce stream, and the width the scheduler binds.

The three carriers block through one split and differ only in which monoid the inner level runs, so
the tests are written per carrier and per stage: the symbolic tree, the width domain read off it,
the substitution, and the offer that carries the decision. What blocking is FOR is the last group —
a bilinear inner level, and a ``TILE`` domain the block width has already decided the K-step of.
"""

from __future__ import annotations

from importlib import import_module

import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.ir.schedule.classic_projection import project_classic
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.tile._block import (
    bind_widths,
    block_tree,
    block_widths,
    is_blocked,
    width_var,
    width_vars,
)

_BLOCK = import_module("emmy.compiler.pipeline.passes.lowering.tile.025_block")
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
    """``(tile, pre-block fold, width variable, blocked outer fold, blocked axes)`` for one stream."""
    tile = _tile(code)
    blocked, axes, streams = block_tree(tile.op, tile.axes)
    assert len(streams) == 1, streams
    node, var, outer = streams[0]
    return tile, node, var, outer, blocked, axes


# ---- the symbolic tree ------------------------------------------------------------------------ #


@pytest.mark.parametrize("code", [_MATMUL, _REDUCE, _SDPA])
def test_block_tree_mints_one_symbolic_width_per_stream(code: str) -> None:
    """The width is a variable, in both extents: the outer's ceil trip count and the inner's own."""
    tile, node, var, outer, _, axes = _stream(code)
    assert var == width_var(node.axis)
    assert width_vars(axes) == (var,)
    table = {axis.name: axis for axis in axes}
    assert var in table[outer.axis].extent.expr.free_vars()
    inner = [edge for edge in outer.operands if edge.axis is not None]
    assert inner and all(table[edge.axis].extent.expr.free_vars() == {var} for edge in inner)


@pytest.mark.parametrize("code", [_MATMUL, _REDUCE, _SDPA])
def test_bind_widths_leaves_no_symbol_and_no_free_coordinate(code: str) -> None:
    """Substituting the width ends it: no axis reads it, and no lambda still binds it."""
    _, _, var, _, blocked, axes = _stream(code)
    op, table = bind_widths(blocked, axes, {var: 32})
    assert width_vars(table) == ()
    assert all(axis.extent.is_static for axis in table)
    assert not any(var in term.lift.params for term in _terms(op))


def _terms(term):
    yield term
    for edge in term.operands:
        yield from _terms(edge)


@pytest.mark.parametrize("code", [_MATMUL, _REDUCE, _SDPA])
def test_every_installed_axis_carries_the_block_receipt(code: str) -> None:
    """``Window(block=True)`` is what stops a second decision, and it survives the substitution."""
    _, _, var, _, blocked, axes = _stream(code)
    _, table = bind_widths(blocked, axes, {var: 32})
    assert is_blocked(table)
    installed = [axis for axis in table if axis.window is not None and axis.window.block]
    assert len(installed) >= 2  # the outer trip axis and at least one block binder
    assert not any(axis.window.partition for axis in installed)


# ---- what each level runs ---------------------------------------------------------------------- #


def test_a_blocked_contraction_keeps_its_bilinear_inner_level() -> None:
    """A matmul blocks into a plain add over a contraction of exactly one block."""
    tile, node, _, outer, _, axes = _stream(_MATMUL)
    (inner,) = [edge for edge in outer.operands if edge.axis is not None]
    assert inner.as_contraction() is not None
    assert outer.as_contraction() is None  # one operand: the outer level is a plain reduce
    assert outer.combine == node.combine and outer.init == node.init
    assert [op.name for op in outer.as_reduction().ops] == [op.name for op in inner.as_reduction().ops] == ["add"]


def test_a_blocked_reduction_runs_the_same_monoid_at_both_levels() -> None:
    tile, node, _, outer, _, _ = _stream(_REDUCE)
    (inner,) = [edge for edge in outer.operands if edge.axis is not None]
    assert inner.as_contraction() is None
    assert [op.name for op in outer.as_reduction().ops] == [op.name for op in inner.as_reduction().ops]
    assert outer.combine == node.combine


def test_a_blocked_twisted_carrier_exposes_a_bilinear_channel() -> None:
    """The whole point: attention's ``P·V`` is a coefficient of a twisted ⊕ until the block splits
    the two monoids, and then it is a contraction the tensor-core tier can read."""
    _, node, _, outer, _, _ = _stream(_SDPA)
    assert node.as_reduction().ops is None  # twisted: no componentwise monoid
    assert outer.combine == node.combine  # the carrier is unchanged; only its operands moved
    inner = [edge for edge in outer.operands if edge.axis is not None]
    assert len(inner) == 3  # the block pivot, the expectation, the denominator
    assert [edge.as_contraction() is not None for edge in inner] == [False, True, False]


# ---- the width domain -------------------------------------------------------------------------- #


def test_a_bilinear_inner_level_admits_exactly_the_atom_k_steps() -> None:
    """The block width IS the fragment depth, so a blocked matmul offers ``atom_k × bk``."""
    tile, node, _, outer, _, _ = _stream(_MATMUL)
    assert block_widths(tile, outer, tile.axis_of(node.axis), _CTX) == (128, 64, 32, 16)


def test_a_planar_inner_level_admits_the_cross_cta_trip_ladder() -> None:
    """A blocked reduction's trip count is what a cross-CTA split partitions, so the widths are
    the extent over the split widths."""
    tile, node, _, outer, _, _ = _stream(_REDUCE)
    assert block_widths(tile, outer, tile.axis_of(node.axis), _CTX) == (2048, 1024, 512, 256, 128)


def test_no_width_divides_into_a_masked_tail() -> None:
    """The ceil form is built and correct, but a masked tail is a realization we do not have."""
    tile, node, _, outer, _, _ = _stream(_MATMUL)
    extent = tile.axis_of(node.axis).extent.as_static()
    assert all(extent % width == 0 and width < extent for width in block_widths(tile, outer, tile.axis_of(node.axis), _CTX))


# ---- the offer ---------------------------------------------------------------------------------- #


def _arms(code: str, tile: TileOp | None = None):
    tile = tile if tile is not None else _tile(code)
    node = type("N", (), {"op": tile})()
    out = _BLOCK.rewrite(None, node, _CTX)
    return out if isinstance(out, list) else [out]


def test_a_planar_stream_offers_the_declined_value_first() -> None:
    """Unblocked is a complete schedule space already, so nothing regresses without evidence."""
    arms = _arms(_MATMUL)
    assert [arm.knobs["BLOCK"] for arm in arms] == ["", "b128", "b64", "b32", "b16"]


def test_a_twisted_stream_offers_its_widths_first() -> None:
    """A twisted carrier has no bilinear site until it is blocked, so declining is a different
    algorithm rather than a schedule alternative."""
    arms = _arms(_SDPA)
    assert [arm.knobs["BLOCK"] for arm in arms][-1] == ""
    assert all(value.startswith("b") for value in [arm.knobs["BLOCK"] for arm in arms][:-1])


def test_the_declined_arm_returns_the_kernel_it_was_offered() -> None:
    """A decision to leave a stream alone is the absence of a rewrite, not the inverse of one."""
    tile = _tile(_MATMUL)
    declined = next(arm for arm in _arms(_MATMUL, tile) if arm.knobs["BLOCK"] == "").materialize()
    assert declined.op is tile.op
    assert declined.axes == tile.axes
    assert declined.knobs["BLOCK"] == ""


def test_a_blocked_arm_binds_the_width_it_spells() -> None:
    tile = _tile(_MATMUL)
    blocked = next(arm for arm in _arms(_MATMUL, tile) if arm.knobs["BLOCK"] == "b64").materialize()
    assert width_vars(blocked.axes) == ()
    assert blocked.axis_of("a2_i").extent.as_static() == 64
    assert is_blocked(blocked.axes)


def test_a_decided_kernel_is_never_offered_again() -> None:
    """Both receipts stand alone: the row a declined stream leaves, and the IR a blocked one does."""
    tile = _tile(_MATMUL)
    for value in ("", "b64"):
        decided = next(arm for arm in _arms(_MATMUL, tile) if arm.knobs["BLOCK"] == value).materialize()
        with pytest.raises(Exception, match="already carry their block decision"):
            _arms(_MATMUL, decided)


def test_a_block_pin_is_authoritative(monkeypatch) -> None:
    monkeypatch.setenv("EMMY_BLOCK", "b32")
    assert [arm.knobs["BLOCK"] for arm in _arms(_MATMUL)] == ["b32"]
    monkeypatch.setenv("EMMY_BLOCK", "")
    assert [arm.knobs["BLOCK"] for arm in _arms(_MATMUL)] == [""]


# ---- what the scheduler does with it ------------------------------------------------------------ #


def _warp_choices(tile: TileOp, path: str) -> list:
    site = next(index for index in range(len(tile.sites)) if tile.sites[index].path == path)
    return [choice for choice in project_classic(tile, _CTX).nodes[site] if choice.tile.is_warp]


def test_the_block_width_decides_the_tile_k_step() -> None:
    """The ``k<bk>`` half of a ``TILE`` value and the block width are one quantity: once the block
    is decided, every warp tile at that site runs exactly one block per step."""
    tile = _tile(_MATMUL)
    blocked = next(arm for arm in _arms(_MATMUL, tile) if arm.knobs["BLOCK"] == "b64").materialize()
    choices = _warp_choices(blocked, "reduce.1/inner")
    assert choices
    assert {choice.tile.atom.atom_k * choice.tile.bk for choice in choices} == {64}


def test_an_unblocked_contraction_still_enumerates_every_k_step() -> None:
    tile = _tile(_MATMUL)
    choices = _warp_choices(tile, "inner")
    assert len({choice.tile.atom.atom_k * choice.tile.bk for choice in choices}) > 1


def test_the_block_axis_sizes_the_fragment_it_tiles() -> None:
    """A nested contraction takes its parent fold's axis as the tile's n. For a BLOCK window that
    is the block's own extent — the enclosing fold walks one block at a time — where a cross-CTA
    slice reads its pre-split parent instead."""
    from emmy.compiler.ir.schedule import Tile as TileChoice  # noqa: PLC0415
    from emmy.compiler.ir.tile.ops import Sched  # noqa: PLC0415

    tile = _tile(_SDPA)
    blocked = next(arm for arm in _arms(_SDPA, tile) if arm.knobs["BLOCK"] == "b64").materialize()
    sched = Sched(blocked, place=blocked.place.on_grid())
    site = next(s for s in blocked.sites if s.path == "map.1/twist.1/reduce.1/inner")
    placed = sched.placed(site.node, TileChoice())
    assert [axis.name for axis in placed.axes] == ["a1", "a2_p"]
    assert placed.axes[1].extent.as_static() == 64
