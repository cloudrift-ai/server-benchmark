"""Offer fused and legal stored-Fold-edge kernel placements."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.path import MissingSiteError, resolve, sites
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.knob import family_of, family_pins
from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams, output_map, realize

PATTERN = [Pattern("root", TileOp)]


def _pin(tile: TileOp, seams) -> tuple[tuple, str, bool] | None:
    """The authoritative placement the live PLACE pins spell for THIS kernel, or ``None``.

    Returns ``(seams, value, scoped)``: every scoped ``PLACE@site=cut`` pin that resolves on this
    kernel joins ONE composed decision — the seams all live on this kernel's tree, so one
    realization cuts them together and the pieces stay decided, which is what lets a pinned
    compile spell a multi-workspace route without recursive re-placement. A scoped pin whose site
    path does not exist on this kernel addresses another kernel of the graph; a kernel none of
    the pins address decides FUSE, so the unpinned fork never returns under a pin-driven compile.
    A pin that resolves to a site no cut realizes is an addressing error and raises. A scoped
    ``PLACE@site=fuse`` excludes that seam from the composed cut (alone, it decides fuse under
    that spelling). Bare ``PLACE`` pins apply only when no scoped pin addressed this kernel."""
    pins = [(name, value) for name, value in family_pins("PLACE") if family_of(name) == "PLACE"]
    if not pins:
        return None
    for _, value in pins:
        if value not in {"fuse", "cut"}:
            raise ValueError(f"bad PLACE value {value!r}; expected 'fuse' or 'cut'")
    all_sites = sites(tile.op)
    # A duplicate cone folded into a cluster seam stays addressable: pinning any
    # member's site names the one shared decision.
    by_node = {id(node): seam for seam in seams for node in (seam.node, *(sibling for sibling, _ in seam.siblings))}
    cut: list = []
    fused: list[str] = []
    missing = False
    for name, value in pins:
        if name == "PLACE":
            continue
        try:
            site = resolve(tile.op, name, all_sites=all_sites)
        except MissingSiteError:
            missing = True  # the key addresses a seam of another kernel in the graph
            continue
        if id(site.node) not in by_node:
            raise ValueError(f"PLACE pin {name!r} does not address a cuttable Fold edge in this kernel")
        seam = by_node[id(site.node)]
        if value == "fuse":
            fused.append(seam.spelling)
        elif not any(chosen is seam for chosen in cut):
            cut.append(seam)
    cut = [seam for seam in cut if seam.spelling not in fused]
    if cut:
        return tuple(cut), "cut", True
    if fused:
        return (fused[0],), "fuse", True
    for name, value in pins:
        if name != "PLACE":
            continue
        if value == "fuse":
            return (name,), value, False
        # A bare ``PLACE=cut`` names the placement DECISION, not a site: the codec's primary
        # rule ranges over ALL PLACE sites and can land on an edge no cut realizes (an unclosed
        # cone, a seam whose workspace dtypes stay undetermined), so a bare pin resolves among
        # the CUTTABLE seams instead: the root-most one.
        depth = {id(site.node): site.depth for site in all_sites}
        seam = min(seams, key=lambda s: depth[id(s.node)])
        return (seam,), value, False
    if missing:
        # A pin-driven compile whose scoped pins all address other kernels decides FUSE here —
        # deterministic, and the unpinned placement fork never returns under a pin.
        return ("PLACE",), "fuse", False
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
        chosen, value, scoped = pinned
        if value == "fuse":
            (spelling,) = chosen
            return DeferredFork(lambda: replace(tile, placement_decided=True), {spelling: "fuse"})
        return DeferredFork(
            lambda: realize(match, root, chosen, renamed, placement_decided=scoped),
            {seam.spelling: "cut" for seam in chosen},
            structural=True,
        )

    options = [DeferredFork(lambda: replace(tile, placement_decided=True), {"PLACE": "fuse"})]
    options.extend(
        DeferredFork(lambda seam=seam: realize(match, root, (seam,), renamed), {seam.spelling: "cut"}, structural=True) for seam in seams
    )
    return options
