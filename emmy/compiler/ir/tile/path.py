"""The tree-path knob codec (phase 2) — ONE walker + resolver over the recognized tile tree.

Grammar: ``FAMILY@<node-path>[.<axis>][<n>] = value``. A schedule key addresses the node (or edge)
it decorates by POSITION in the recognized term tree — no parallel namespace of hand-invented site
names. The walker (:func:`sites`) enumerates every ``(path, node, schedule-bearing axis)`` triple
once; the resolver (:func:`resolve`) and the canonical speller (:func:`spell`) are total over the
sugar forms and shared by the pin layer, the stampers (phase 3) and the seam enumerator (phase 4).

Sugar levels, shortest first — **short paths are canonical** (stampers and ALL stored evidence use
the shortest spelling unique for the kernel's tree, which is why every live golden spelling is
already canonical and the corpus needs zero migration):

- bare ``FAMILY`` — resolves to the PRIMARY (root-most schedule-bearing) node for that family; with
  no eligible node the family "doesn't apply" (``None`` — the decided-empty stamp); with two nodes
  at the same minimal depth it raises naming the candidates (the flash ``TILE@dd`` / ``TILE@pj``
  pair). The bare-resolution guard keeps the ~46 stored bare keys meaning what they always meant:
  bare ``REDUCE`` on norm_linear/geglu is the contraction's K fold, never the cone's stat.
- ``FAMILY@<axis>`` — the axis is the real discriminator (the leaf key for TILE/REDUCE/STAGE);
  unique axis name → that node (``TILE@dd``).
- ``FAMILY@<path>.<axis>`` — path segments are lowercase node-kind tokens from the root (``map`` /
  ``fold``) or operand-edge labels (``a`` / ``b`` — VIEW-ROLE sugar read off the bilinear parse,
  the shared A edge vs a channel's B). Any unique ANCHORED SUBSEQUENCE of the full segment path is
  accepted at pin time (the last path component must be the node's own segment); the canonical
  spelling is the shortest unique one, deepest anchors first (``REDUCE@a.fold.k`` for the cone's
  stat when the axis name collides with the contraction's).
- the ordinal ``<n>`` (1-based, canonicalized traversal order) exists ONLY for true same-path
  collisions — two sites with identical full path AND axis; current kernels never need it.

RESERVED (reject cleanly, never reuse): the graph-level placement grammar — the ``in.<operand>``
path prefix and the leading-``=`` value-name pin form. Whoever restores graph-level placement owns
that namespace; this parser only refuses to let a tile key squat on it.

Ambiguity failures are LOUD by design: a stored short key broken by a future structural change must
fail and be re-spelled by hand (``test_golden_spelling_canonical`` is the tripwire) — never
silently re-keyed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.tile.ir import Fold, Map, _parse_bilinear

#: The knob families whose keys address a tree site (``WSPEC`` / ``RASTER`` / ``LOOPIFY`` stay
#: root-global and bare). ``PLACE`` (phase 4) is the per-seam edge property: its sites are every
#: NON-ROOT node — each in-tree child names its parent↔child seam — and its values are
#: ``cut | fuse``, resolved from ROUTING golden entries / pins, never a schedule slice.
PATH_FAMILIES = ("TILE", "REDUCE", "STAGE", "PLACE")

#: The path-segment vocabulary: node kinds + the contraction operand-edge role labels.
_SEGMENT_TOKENS = frozenset({"map", "fold", "a", "b"})

_ORDINAL_RE = re.compile(r"^([a-z]+)(\d+)$")


@dataclass(frozen=True)
class Site:
    """One schedule-bearing tree position: the node, its schedule-bearing axis (``None`` for a
    pointwise ``Map``), the FULL segment path from the root (this node's own segment last), and the
    1-based ``ordinal`` among sites sharing the identical ``(segments, axis)`` (1 when unique —
    the no-collision common case, where the ordinal is never spelled). ``derived`` marks a site
    living in a λ-spelled fold's DERIVED evaluation (flash's synthesized PV contraction) — a real
    schedule site (``TILE@pj``) but combine material BELOW the seam lattice, never a ``PLACE``
    target."""

    node: object
    axis: str | None
    segments: tuple[str, ...]
    ordinal: int = 1
    derived: bool = False

    @property
    def depth(self) -> int:
        return len(self.segments)


def _seg(node) -> str:
    return "map" if isinstance(node, Map) else "fold"


def _edge_labels(fold: Fold) -> dict[int, str]:
    """The view-role labels of a contraction fold's INLINE operand edges, keyed by operand index:
    the shared-A edge is ``a``, a channel's streamed edge ``b`` (read off the bilinear parse — the
    same read :func:`~emmy.compiler.ir.tile.ir.contraction_view` makes). A fold that doesn't parse
    bilinearly labels nothing — its inline operands keep their node-kind segments."""
    parsed = _parse_bilinear(fold)
    if parsed is None:
        return {}
    a_edge, chans = parsed
    labels: dict[int, str] = {}
    for i, edge in enumerate(fold.operands):
        if edge is a_edge:
            labels[i] = "a"
        elif any(edge is b for b, _ in chans):
            labels[i] = "b"
    return labels


def _stmt_children(stmt):
    """Structural nodes embedded in a plain stmt's nested bodies (a composed step reached through
    a ``Loop`` — ``030``'s sliced partials)."""
    for body in stmt.nested():
        for child in body:
            if isinstance(child, (Map, Fold)):
                yield child
            else:
                yield from _stmt_children(child)


def _walk(node, prefix: tuple[str, ...], out: list[tuple[object, tuple[str, ...], bool]], derived: bool = False) -> None:
    out.append((node, prefix, derived))
    if isinstance(node, Map):
        for src in node.sources:
            if isinstance(src, (Map, Fold)):
                _walk(src, (*prefix, _seg(src)), out, derived)
        for s in node.body:
            for child in _stmt_children(s) if not isinstance(s, (Map, Fold)) else (s,):
                _walk(child, (*prefix, _seg(child)), out, derived)
    elif isinstance(node, Fold):
        labels = _edge_labels(node)
        for i, edge in enumerate(node.operands):
            if isinstance(edge, (Map, Fold)):
                _walk(edge, (*prefix, labels.get(i, _seg(edge))), out, derived)
        # The DERIVED evaluation's children — synthesized nodes (flash's PV, memoized on the
        # fold) are real schedule sites, marked ``derived`` (combine material below the seam
        # lattice; a lift-body inline node — the demoted cone — likewise: un-realizable as a
        # seam). Operand edges embedded at their derived head position were walked above.
        edge_ids = {id(e) for e in node.operands}
        step_derived = True
        for s in node.step_stmts():
            if id(s) in edge_ids:
                continue
            for child in _stmt_children(s) if not isinstance(s, (Map, Fold)) else (s,):
                _walk(child, (*prefix, _seg(child)), out, step_derived)


def sites(root) -> tuple[Site, ...]:
    """Every structural node in ``root``'s tree as a :class:`Site`, root first (the ONE walk the
    resolver, the stampers and the seam enumerator share). Ordinals are assigned in traversal order
    among sites with identical ``(segments, axis)``."""
    if root is None:
        return ()
    nodes: list[tuple[object, tuple[str, ...], bool]] = []
    _walk(root, (_seg(root),), nodes)
    counts: dict[tuple, int] = {}
    result: list[Site] = []
    for node, segments, derived in nodes:
        axis = node.axis.name if isinstance(node, Fold) else None
        key = (segments, axis)
        counts[key] = counts.get(key, 0) + 1
        result.append(Site(node=node, axis=axis, segments=segments, ordinal=counts[key], derived=derived))
    return tuple(result)


def family_sites(family: str, all_sites: tuple[Site, ...]) -> tuple[Site, ...]:
    """The sites ``family`` may decorate: every fold for ``REDUCE`` / ``STAGE``; ``TILE`` takes the
    ``role=CONTRACTION`` folds plus the pure pointwise ROOT ``Map`` (the register-strip tier — a
    non-root sourceless ``Map``, e.g. a one-load demoted cone, is not a strip target); ``PLACE``
    every NON-ROOT node (the child names its parent↔child seam — cut legality is structural,
    edge-iff-closed by construction)."""
    if family not in PATH_FAMILIES:
        raise ValueError(f"{family!r} is not a tree-path knob family (have {PATH_FAMILIES})")
    if family == "PLACE":
        return tuple(s for s in all_sites if s.depth > 1 and not s.derived)
    if family in ("REDUCE", "STAGE"):
        return tuple(s for s in all_sites if isinstance(s.node, Fold))
    out = []
    for s in all_sites:
        if isinstance(s.node, Fold) and s.node.role is AxisRole.CONTRACTION:
            out.append(s)
        elif isinstance(s.node, Map) and not s.node.sources and s.depth == 1:
            out.append(s)
    return tuple(out)


def primary(family: str, fam_sites: tuple[Site, ...]) -> Site | None:
    """The PRIMARY (root-most) site of ``family`` — the unique minimum-depth one, or ``None`` when
    the two shallowest tie (bare is then ambiguous: flash's two ``TILE`` contractions)."""
    if not fam_sites:
        return None
    depth = min(s.depth for s in fam_sites)
    heads = [s for s in fam_sites if s.depth == depth]
    return heads[0] if len(heads) == 1 else None


@dataclass(frozen=True)
class _Key:
    family: str
    segments: tuple[str, ...]
    axis: str | None
    ordinal: int | None

    @property
    def bare(self) -> bool:
        return not self.segments and self.axis is None and self.ordinal is None


def parse_key(key: str) -> _Key:
    """Split a knob key into ``(family, path segments, axis, ordinal)``, rejecting the RESERVED
    graph-level placement forms. Non-final components must be exact segment tokens; the final
    component is read as an axis name unless it is a segment token (an ordinal on the final
    component is split off at match time — see :func:`resolve` — so an axis literally named
    ``k2`` keeps winning over an ordinal reading)."""
    family, at, suffix = key.partition("@")
    if not at:
        return _Key(family=family, segments=(), axis=None, ordinal=None)
    if suffix.startswith("=") or suffix == "in" or suffix.startswith("in."):
        raise ValueError(f"knob key {key!r} is reserved for graph-level placement")
    if not suffix:
        raise ValueError(f"knob key {key!r} has an empty @-suffix")
    comps = suffix.split(".")
    segments: list[str] = []
    axis: str | None = None
    ordinal: int | None = None
    for i, comp in enumerate(comps):
        last = i == len(comps) - 1
        if not comp:
            raise ValueError(f"knob key {key!r} has an empty path component")
        if comp in _SEGMENT_TOKENS:
            if axis is not None:
                raise ValueError(f"knob key {key!r}: path segment {comp!r} after the axis")
            segments.append(comp)
        elif last:
            m = _ORDINAL_RE.match(comp)
            if m and m.group(1) in _SEGMENT_TOKENS:
                segments.append(m.group(1))
                ordinal = int(m.group(2))
            else:
                axis = comp
        else:
            raise ValueError(f"knob key {key!r}: unknown path segment {comp!r} (expect {sorted(_SEGMENT_TOKENS)} or a final axis)")
    return _Key(family=family, segments=tuple(segments), axis=axis, ordinal=ordinal)


def _admits(site: Site, segments: tuple[str, ...]) -> bool:
    """Whether ``segments`` is an ANCHORED subsequence of the site's full path — the last key
    segment must be the node's own segment, earlier ones may skip ancestors (``a.fold`` names the
    stat under the cone edge without spelling the wrapper maps)."""
    if not segments:
        return True
    if segments[-1] != site.segments[-1]:
        return False
    pos = 0
    for want in segments[:-1]:
        try:
            pos = site.segments.index(want, pos, len(site.segments) - 1) + 1
        except ValueError:
            return False
    return True


def _match(key: _Key, fam_sites: tuple[Site, ...]) -> list[Site]:
    out = [s for s in fam_sites if _admits(s, key.segments)]
    if key.axis is not None:
        out = [s for s in out if s.axis == key.axis]
    if key.ordinal is not None:
        out = [s for s in out if s.ordinal == key.ordinal]
    return out


def _spellings(family: str, site: Site, fam_sites: tuple[Site, ...]) -> str:
    """The canonical (shortest unique) spelling of ``site`` under ``family`` — see :func:`spell`."""
    if primary(family, fam_sites) is site:
        return family
    axis_part = f".{site.axis}" if site.axis is not None else ""
    if site.axis is not None and sum(1 for s in fam_sites if s.axis == site.axis) == 1:
        return f"{family}@{site.axis}"
    # Path forms: shortest anchored subsequence unique among the family's sites; among equal-length
    # candidates prefer EDGE LABELS, then the deepest anchors (``a.fold`` over ``fold.fold`` /
    # ``map.fold`` for the cone stat — the label names the seam a reader recognizes).
    own = site.segments[-1]
    ancestors = site.segments[:-1]
    for length in range(1, len(site.segments) + 1):
        best: tuple[tuple[int, tuple[int, ...]], tuple[str, ...]] | None = None
        for positions in _subsequences(len(ancestors), length - 1):
            segs = (*(ancestors[p] for p in positions), own)
            matches = [s for s in fam_sites if _admits(s, segs) and (site.axis is None or s.axis == site.axis)]
            if len(matches) == 1:
                rank = (sum(1 for t in segs if t in ("a", "b")), positions)
                if best is None or rank > best[0]:
                    best = (rank, segs)
        if best is not None:
            return f"{family}@{'.'.join(best[1])}{axis_part}"
    # True same-path collision: the full path + axis still names several sites — spell the ordinal.
    return f"{family}@{'.'.join(site.segments)}{axis_part}{site.ordinal}"


def _subsequences(n: int, k: int):
    """All strictly-increasing index tuples of length ``k`` into ``range(n)`` (tiny trees — brute
    force is fine)."""
    from itertools import combinations  # noqa: PLC0415

    yield from combinations(range(n), k)


def spell(root, family: str, node, *, all_sites: tuple[Site, ...] | None = None) -> str:
    """The CANONICAL key addressing ``node`` under ``family`` on ``root``'s tree — the shortest
    spelling unique for the tree: bare for the primary, ``FAMILY@<axis>`` when the axis alone
    discriminates, else the shortest anchored path subsequence (deepest anchors preferred), with
    the 1-based ordinal only on a true same-path collision. Stampers and stored evidence use this
    spelling and nothing else."""
    all_sites = sites(root) if all_sites is None else all_sites
    fam_sites = family_sites(family, all_sites)
    for s in fam_sites:
        if s.node is node:
            return _spellings(family, s, fam_sites)
    raise ValueError(f"node {type(node).__name__} is not a {family} site of this tree")


def resolve(root, key: str, *, all_sites: tuple[Site, ...] | None = None) -> Site | None:
    """Resolve a knob ``key`` (any sugar level) to the :class:`Site` it addresses on ``root``'s
    tree. Total over the sugar forms and idempotent (``resolve(spell(...))`` is the same site):

    - bare, no eligible site → ``None`` (the family doesn't apply — drop / decided-empty);
    - bare, several sites → the PRIMARY, or ``ValueError`` naming the canonical candidates;
    - suffixed, no match → ``ValueError`` (a stored short key broken by a structural change must
      fail loudly — never silently re-key);
    - suffixed, several matches → ``ValueError`` naming the candidates.

    A final component with trailing digits is first read as a literal axis name; only when no site
    carries that axis is it retried as ``<axis><ordinal>`` (so an axis named ``k2`` never loses to
    an ordinal reading)."""
    parsed = parse_key(key)
    all_sites = sites(root) if all_sites is None else all_sites
    fam_sites = family_sites(parsed.family, all_sites)
    if parsed.bare:
        if not fam_sites:
            return None
        if len(fam_sites) == 1:
            return fam_sites[0]
        head = primary(parsed.family, fam_sites)
        if head is not None:
            return head
        cands = " or ".join(sorted(_spellings(parsed.family, s, fam_sites) for s in fam_sites))
        raise ValueError(f"{parsed.family} is ambiguous: use {cands}")
    matches = _match(parsed, fam_sites)
    if not matches and parsed.axis is not None and parsed.ordinal is None:
        m = re.match(r"^(.*?)(\d+)$", parsed.axis)
        if m and m.group(1):
            retry = _Key(family=parsed.family, segments=parsed.segments, axis=m.group(1), ordinal=int(m.group(2)))
            matches = _match(retry, fam_sites)
    if not matches:
        raise ValueError(f"knob key {key!r} names no site on this tree (a structural change broke a stored key?)")
    if len(matches) > 1:
        cands = " or ".join(sorted(_spellings(parsed.family, s, fam_sites) for s in matches))
        raise ValueError(f"knob key {key!r} is ambiguous: use {cands}")
    return matches[0]


def canonical(root, key: str, *, all_sites: tuple[Site, ...] | None = None) -> str | None:
    """The canonical spelling of ``key`` on ``root``'s tree — ``resolve`` then ``spell``; ``None``
    when the family doesn't apply (a bare decided-empty stamp)."""
    all_sites = sites(root) if all_sites is None else all_sites
    site = resolve(root, key, all_sites=all_sites)
    if site is None:
        return None
    return spell(root, parse_key(key).family, site.node, all_sites=all_sites)


__all__ = ["PATH_FAMILIES", "Site", "canonical", "family_sites", "parse_key", "primary", "resolve", "sites", "spell"]
