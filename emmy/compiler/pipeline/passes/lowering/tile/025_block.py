"""Offer the block width of every blockable reduce stream, before scheduling sees the tree.

Blocking is a STRUCTURAL decision, like a placement cut: it changes the term the classic problem is
built over, so it is consumed here, ahead of ``030_cut`` and ``040_schedule``, and its value is
spelled on the same tree-path routes ``PLACE`` uses. What the fork decides is the WIDTH — the tree
is blocked once, symbolically (``_block.block_tree``), the domain is read off that tree
(``_block.block_widths``), and each arm re-blocks the streams it decided a width for and
substitutes it.

The order of the arms is the cold fallback, and it differs by carrier for one reason. A PLANAR fold
already has its whole schedule space unblocked — the block is an addition, so its declined value
leads and nothing regresses without evidence. A TWISTED carrier has no bilinear site at all until
it is blocked: declining is not a schedule alternative there but a different algorithm, so its
widths lead.

A ``BLOCK`` pin is authoritative, exactly as a ``PLACE`` pin is: it narrows the stream's domain to
the one value, ``""`` included, and the offer never returns a sibling the pin excluded.

The receipt against deciding twice is doubled, because either half can arrive alone: the decided
kernel carries its ``BLOCK`` value in ``knobs`` (a declined stream leaves no trace in the IR), and
every axis a block installs carries ``Window(block=True)`` (a graph splice mints kernels whose row
was consumed).
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import Sched
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.knob import axis_of, family_of
from emmy.compiler.pipeline.passes.lowering.tile._block import bind_widths, block_tree, block_widths, is_blocked
from emmy.compiler.pipeline.search.space import BLOCK

PATTERN = [Pattern("root", TileOp)]


def _decided(tile: TileOp) -> bool:
    """Whether this kernel's block decision has already been consumed."""
    return is_blocked(tile.axes) or any(family_of(key) == "BLOCK" for key in tile.knobs)


def _keys(tile: TileOp, streams: tuple) -> tuple[str, ...]:
    """One ``BLOCK`` key per decided stream — bare for a kernel with one, else its route."""
    view = Sched(tile)
    return tuple("BLOCK" if len(streams) == 1 else f"BLOCK@{view.site_of(node).path}" for node, _, _ in streams)


def _domain(tile: TileOp, key: str, stream: tuple, ctx) -> tuple[str, ...]:
    """One stream's offered values, cold-fallback order first, narrowed by a live pin."""
    node, _, outer = stream
    widths = tuple(f"b{width}" for width in block_widths(tile, outer, tile.axis_of(node.axis), ctx))
    values = (*widths, "") if node.as_reduction().ops is None else ("", *widths)
    element = axis_of(key)
    pin = BLOCK.narrow_at(element) if element else BLOCK.raw()
    return values if pin is None else tuple(value for value in values if value == pin)


def rewrite(match: Match, root: Node, ctx=None):
    del match
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped or tile.schedule is not None:
        raise RuleSkipped("TileOp already scheduled / nothing to rewrite")
    if _decided(tile):
        raise RuleSkipped("this kernel's reduce streams already carry their block decision")
    got = block_tree(tile.op, tile.axes)
    if got is None:
        raise RuleSkipped("no blockable reduce stream")
    _, _, streams = got
    keys = _keys(tile, streams)
    domains = tuple(_domain(tile, key, stream, ctx) for key, stream in zip(keys, streams, strict=True))
    if any(not values for values in domains):
        raise ValueError(f"{tile.name!r}: no BLOCK value survives the pin on this kernel's reduce streams")

    options = [DeferredFork(lambda r=row: _arm(tile, keys, streams, r), row) for row in _rows(keys, domains)]
    return options if len(options) > 1 else options[0]


def _rows(keys: tuple[str, ...], domains: tuple[tuple[str, ...], ...]) -> list[dict[str, str]]:
    """The cartesian of the per-stream domains, each stream's leading value first."""
    rows: list[dict[str, str]] = [{}]
    for key, values in zip(keys, domains, strict=True):
        rows = [{**row, key: value} for row in rows for value in values]
    return rows


def _arm(tile: TileOp, keys: tuple[str, ...], streams: tuple, row: dict[str, str]) -> TileOp:
    """One decided kernel: the streams this row gave a width to, blocked at it.

    Re-blocked from the ORIGINAL tree rather than un-blocked from the enumerated one — a decision
    to leave a stream alone is the absence of a rewrite, not the inverse of one.
    """
    picked = {id(node): row[key] for key, (node, _, _) in zip(keys, streams, strict=True) if row[key]}
    if not picked:
        return replace(tile, knobs={**tile.knobs, **row})
    blocked, axes, chosen = block_tree(tile.op, tile.axes, only=frozenset(picked))
    widths = {var: int(picked[id(node)][1:]) for node, var, _ in chosen}
    op, table = bind_widths(blocked, axes, widths)
    return replace(tile, op=op, axes=table, knobs={**tile.knobs, **row})
