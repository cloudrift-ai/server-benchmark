"""Enumerate every kernel-set cut before schedule assignments.

Stored-Fold-edge placement is the first domain and cross-CTA reduction splitting is the second.
The rule runs to a fixpoint, so each successful choice and every fresh piece re-enters these ordered
domains before scheduling.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.path import MissingSiteError, resolve, sites
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.knob import family_of, family_pins
from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams, output_map, realize
from emmy.compiler.pipeline.passes.lowering.tile._split import split_forks

PATTERN = [Pattern("root", TileOp)]
FIXPOINT = True


def _seam_index(seams) -> dict[int, object]:
    """Map every seam node and clustered sibling to its shared decision."""
    return {id(node): seam for seam in seams for node in (seam.node, *(sibling for sibling, _ in seam.siblings))}


def _with_required(chosen, by_node: dict[int, object], refuse: frozenset = frozenset()) -> tuple:
    """Expand chosen seams with every transitively required producer seam."""
    out = list(chosen)
    queue = list(out)
    while queue:
        for _, producer in queue.pop().requires:
            required = by_node[id(producer)]
            if required.spelling in refuse:
                raise ValueError(f"PLACE pins fuse {required.spelling!r}, the producer a pinned dependent cut requires")
            if not any(member is required for member in out):
                out.append(required)
                queue.append(required)
    return tuple(out)


def _placement_restriction(tile: TileOp, seams) -> tuple[tuple, str] | None:
    """The authoritative placement spelled by live PLACE pins, or ``None``.

    This restriction is consumed entirely by the cut pass before classic schedule enumeration.
    Every scoped ``PLACE@site=cut`` pin that resolves on this kernel joins ONE composed decision —
    the seams all live on this kernel's tree, so one realization cuts them together and the pieces
    stay decided. A bare ``PLACE=cut`` consumes its root-most cut the same way. A scoped pin whose
    site path does not exist on this kernel addresses another kernel of the graph; a kernel none of
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
    by_node = _seam_index(seams)
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
    cut = list(_with_required(cut, by_node, refuse=frozenset(fused)))
    if cut:
        return tuple(cut), "cut"
    if fused:
        return (fused[0],), "fuse"
    for name, value in pins:
        if name != "PLACE":
            continue
        if value == "fuse":
            return (name,), value
        # A bare ``PLACE=cut`` names the placement DECISION, not a site: the codec's primary
        # rule ranges over ALL PLACE sites and can land on an edge no cut realizes (an unclosed
        # cone, a seam whose workspace dtypes stay undetermined), so a bare pin resolves among
        # the CUTTABLE seams instead: the root-most one.
        # Provider-closed and dependent seams are scoped-pin-only. A bare pin selects this one
        # root-most plain seam and is consumed on the fresh pieces.
        plain = [seam for seam in seams if not (seam.providers or seam.requires)]
        if not plain:
            return ("PLACE",), "fuse"
        depth = {id(site.node): site.depth for site in all_sites}
        seam = min(plain, key=lambda candidate: depth[id(candidate.node)])
        return (seam,), value
    if missing:
        # A pin-driven compile whose scoped pins all address other kernels decides FUSE here —
        # deterministic, and the unpinned placement fork never returns under a pin.
        return ("PLACE",), "fuse"
    return None


def _placement_forks(match: Match, root: Node, tile: TileOp):
    """Return the next stored-edge cut fork, or ``None`` when that domain is consumed."""
    seams = cuttable_seams(tile)
    if not seams:
        return None

    renamed = output_map(root)
    match.output = renamed
    pinned = _placement_restriction(tile, seams)
    if pinned is not None:
        chosen, value = pinned
        if value == "fuse":
            (spelling,) = chosen
            return DeferredFork(lambda: replace(tile, placement_decided=True), {spelling: "fuse"})
        return DeferredFork(
            lambda: realize(match, root, chosen, renamed, placement_decided=True),
            {seam.spelling: "cut" for seam in chosen},
            structural=True,
        )

    options = [DeferredFork(lambda: replace(tile, placement_decided=True), {"PLACE": "fuse"})]
    by_node = _seam_index(seams)
    closures: dict[frozenset[str], tuple] = {}
    for seam in seams:
        closure = _with_required((seam,), by_node)
        closures.setdefault(frozenset(member.spelling for member in closure), closure)
    options.extend(
        DeferredFork(
            lambda closure=closure: realize(match, root, closure, renamed),
            {member.spelling: "cut" for member in closure},
            structural=True,
        )
        for closure in closures.values()
    )
    return options


def rewrite(match: Match, root: Node, ctx=None):
    del ctx
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped or tile.schedule is not None:
        raise RuleSkipped("TileOp already scheduled")
    choices = None if tile.placement_decided else _placement_forks(match, root, tile)
    if choices is None:
        choices = split_forks(match, root)
    if choices is None:
        raise RuleSkipped("no pending kernel-set cut")
    choices = choices if isinstance(choices, list) else [choices]
    return choices if len(choices) > 1 else choices[0]
