"""Offer fused and legal stored-Fold-edge kernel placements."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.path import resolve, sites
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.knob import family_of, family_pins
from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams, output_map, realize

PATTERN = [Pattern("root", TileOp)]


def _pin(tile: TileOp, seams) -> tuple[str, str] | None:
    pins = [(name, value) for name, value in family_pins("PLACE") if family_of(name) == "PLACE"]
    if not pins:
        return None
    all_sites = sites(tile.op)
    by_node = {id(seam.node): seam for seam in seams}
    for name, value in pins:
        if value not in {"fuse", "cut"}:
            raise ValueError(f"bad PLACE value {value!r}; expected 'fuse' or 'cut'")
        if value == "fuse" and name == "PLACE":
            return name, value
        site = resolve(tile.op, name, all_sites=all_sites)
        if site is None or id(site.node) not in by_node:
            # KNOWN GAP: a bare ``PLACE`` resolves through the path codec's primary rule over ALL
            # PLACE sites, which on a fused norm→linear lands on the contraction's A-cone edge —
            # a site ``cuttable_seams`` excludes (contraction operands cut at their inner map /
            # statistic folds, e.g. ``PLACE@map`` / ``PLACE@a1``), so the bare pin raises here
            # even though the kernel HAS cuttable seams. Red since the maximal-fusion tree shape
            # (#648, pre-dating the schedule-walk rebuild); the fix is to resolve a bare pin among
            # the cuttable seams (e.g. the root-most seam), not against every PLACE site.
            raise ValueError(f"PLACE pin {name!r} does not address a cuttable Fold edge in this kernel")
        return by_node[id(site.node)].spelling, value
    return None


def rewrite(match: Match, root: Node, ctx=None):
    del ctx
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped or tile.schedule or tile.placement_decided:
        raise RuleSkipped("TileOp already placed / scheduled")
    seams = cuttable_seams(tile)
    if not seams:
        raise RuleSkipped("no closed stored Fold edge")

    renamed = output_map(root)
    match.output = renamed
    pinned = _pin(tile, seams)
    if pinned is not None:
        spelling, value = pinned
        if value == "fuse":
            return DeferredFork(lambda: replace(tile, placement_decided=True), {spelling: "fuse"})
        seam = next(seam for seam in seams if seam.spelling == spelling)
        return DeferredFork(lambda: realize(match, root, seam, renamed), {spelling: "cut"}, structural=True)

    options = [DeferredFork(lambda: replace(tile, placement_decided=True), {"PLACE": "fuse"})]
    options.extend(
        DeferredFork(lambda seam=seam: realize(match, root, seam, renamed), {seam.spelling: "cut"}, structural=True) for seam in seams
    )
    return options
