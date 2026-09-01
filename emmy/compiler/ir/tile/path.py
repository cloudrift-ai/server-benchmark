"""The structural placement path codec over the stored Fold tree.

Grammar: ``PLACE@<node-path>[.<axis>][<n>] = value``. A selected result appends
``.result.<position>``. Placement addresses a stored Fold edge by position because it is a
structural decision made before schedule sites exist; ``PLACE@root`` addresses the root
output-region boundary without admitting that boundary to bare ``PLACE`` resolution. A shortest
unique path spelling is canonical; ambiguity and stale paths fail loudly. The retired
``in.<operand>`` prefix and leading-``=`` value-name form remain reserved.

Classic choices never use this codec. A classic problem constructs sites only after every structural choice is
consumed, and its strict codec addresses integer node ids and ``(consumer, operand)`` edge tuples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations

from emmy.compiler.ir.pure.fold import Fold, _operand_result_names
from emmy.compiler.ir.pure.tree import walk
from emmy.compiler.structural import instance_memo

#: The only family a tree path may address. Schedule identities belong to ``schedule.classic``.
PATH_FAMILIES = ("PLACE",)

#: The path-segment vocabulary: node kinds + the contraction operand-edge role labels.
_SEGMENT_TOKENS = frozenset({"root", "map", "fold", "a", "b"})

#: Split a final component into a literal prefix plus trailing digits. :func:`resolve` tries the
#: unsplit literal first, then moves the split left one digit at a time. Thus ``a22`` can resolve as
#: axis ``a2`` ordinal 2 without stealing an axis literally named ``a22``.
_AXIS_ORDINAL_RE = re.compile(r"^(.*?)(\d+)$")


class MissingSiteError(ValueError):
    """A suffixed knob key that names NO site on this tree. A ``ValueError`` so every existing
    handler still reads it as a broken stored key failing loudly; its own type so a caller
    holding GRAPH-scoped pins (the placement rule) can tell "this key addresses another kernel"
    apart from an ambiguity, which stays a plain ``ValueError``."""


class UnknownSiteError(Exception):
    """A node outside the stored tree was used for a placement or lowering lookup."""


@dataclass(frozen=True)
class Site:
    """One structural tree position: the node, its axis (``None`` for a pointwise zero-axis fold),
    the full segment path from the root (this node's own segment last), and the
    1-based ``ordinal`` among sites sharing the identical ``(segments, axis)`` (1 when unique —
    the no-collision common case, where the ordinal is never spelled). ``derived`` marks a site
    living in a λ-spelled fold's derived evaluation (flash's synthesized PV contraction). Every
    Fold remains a site; residence choices on an enclosing edge never hide its algebra."""

    node: object
    axis: str | None
    segments: tuple[str, ...]
    ordinal: int = 1
    derived: bool = False
    #: One-based selected result position. Ordinary stored node sites carry ``None``; result
    #: sites are resolved structural addresses and never join :func:`sites`.
    result: int | None = None

    @property
    def depth(self) -> int:
        return len(self.segments)


def sites(root) -> tuple[Site, ...]:
    """Every structural node in ``root``'s tree as a :class:`Site`, root first — the ONE node walk
    in the layer — a reading of the ONE walk (:func:`~emmy.compiler.ir.pure.tree.walk`), which owns
    the traversal rules and the segment vocabulary. This adds only what the CODEC needs: the
    per-site ordinal among sites with identical ``(segments, axis)``, assigned in traversal order.
    An operand subtree has exactly one home (its edge), so the tree stays a tree and no visited set
    is needed."""
    if root is None:
        return ()
    counts: dict[tuple, int] = {}
    result: list[Site] = []
    for visit in walk(root):
        node = visit.node
        axis = node.axis.name if isinstance(node, Fold) and node.axis is not None else None
        key = (visit.segments, axis)
        counts[key] = counts.get(key, 0) + 1
        result.append(Site(node=node, axis=axis, segments=visit.segments, ordinal=counts[key], derived=visit.derived))
    return tuple(result)


def family_sites(family: str, all_sites: tuple[Site, ...]) -> tuple[Site, ...]:
    """Stored non-root Fold edges eligible for structural placement."""
    if family not in PATH_FAMILIES:
        raise ValueError(f"{family!r} is not a structural path family (have {PATH_FAMILIES})")
    return tuple(s for s in all_sites if s.depth > 1 and not s.derived)


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
    if len(comps) > 1 and comps[-1].isdigit():
        # The EXPLICIT ordinal component — the collision-proof spelling :func:`_spellings` mints
        # when the concatenated ``<axis><n>`` form would be captured by a literal axis name
        # (``a13`` + 1 → ``a131`` beside a real ``a131`` axis). Unambiguous by construction:
        # segment tokens and axis names are letter-led, so an all-digit component can only be an
        # ordinal.
        ordinal = int(comps[-1])
        comps = comps[:-1]
    for i, comp in enumerate(comps):
        last = i == len(comps) - 1
        if not comp:
            raise ValueError(f"knob key {key!r} has an empty path component")
        if comp in _SEGMENT_TOKENS:
            if axis is not None:
                raise ValueError(f"knob key {key!r}: path segment {comp!r} after the axis")
            segments.append(comp)
        elif last:
            # Keep the final component literal. ``resolve`` only reinterprets a trailing digit run
            # after the literal axis reading fails, which is what lets an axis named ``k2`` win
            # over the old ``k`` + ordinal-2 spelling.
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


def _spellings(family: str, site: Site, fam_sites: tuple[Site, ...], head: Site | None = None) -> str:
    """The canonical (shortest unique) spelling of ``site`` under ``family`` — see :func:`spell`.
    ``head`` is the family's primary when the caller already resolved it (the bulk table builder
    resolves it once for every site)."""
    if (primary(family, fam_sites) if head is None else head) is site:
        return family
    axis_part = f".{site.axis}" if site.axis is not None else ""
    if site.axis is not None and sum(1 for s in fam_sites if s.axis == site.axis) == 1:
        return f"{family}@{site.axis}"
    identical = [s for s in fam_sites if s.segments == site.segments and s.axis == site.axis]
    if len(identical) > 1:
        # No subsequence can distinguish identical full paths.  Skip the exponential search and
        # use the ordinal arm directly; this is precisely the arm ordinals exist for.
        return _ordinal_spelling(family, site, fam_sites)
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
    return _ordinal_spelling(family, site, fam_sites)


def _ordinal_spelling(family: str, site: Site, fam_sites: tuple[Site, ...]) -> str:
    """The ordinal arm's spelling, honoring the round-trip law: the concatenated ``<axis><n>``
    form is canonical, but ``resolve`` reads a final component as a LITERAL axis first (so an axis
    named ``k2`` never loses to ``k`` + ordinal 2) — so when any sibling site's axis equals the
    concatenation, the concat form would be captured (mis-resolving silently with one such site,
    ambiguous with several) and the explicit dotted ordinal is minted instead."""
    axis_part = f".{site.axis}" if site.axis is not None else ""
    captured = f"{site.axis if site.axis is not None else site.segments[-1]}{site.ordinal}"
    if any(s.axis == captured for s in fam_sites):
        return f"{family}@{'.'.join(site.segments)}{axis_part}.{site.ordinal}"
    return f"{family}@{'.'.join(site.segments)}{axis_part}{site.ordinal}"


def _subsequences(n: int, k: int):
    """All strictly-increasing index tuples of length ``k`` into ``range(n)`` (tiny trees — brute
    force is fine)."""
    yield from combinations(range(n), k)


def spell(root, family: str, node, *, all_sites: tuple[Site, ...] | None = None) -> str:
    """The CANONICAL key addressing ``node`` under ``family`` on ``root``'s tree — the shortest
    spelling unique for the tree: bare for the primary, ``FAMILY@<axis>`` when the axis alone
    discriminates, else the shortest anchored path subsequence (deepest anchors preferred), with
    the 1-based ordinal only on a true same-path collision. Stampers and stored evidence use this
    spelling and nothing else."""
    if node is root:
        if family not in PATH_FAMILIES:
            family_sites(family, ())  # raise the structural-family error
        return f"{family}@root"
    tables = instance_memo(root, "_memo_spellings")
    table = tables.get(family)
    if table is None:
        # The whole family spells as ONE derived table (an ``instance_memo`` on the immutable
        # root): the schedule pricing loops re-spell the same sites once per candidate row, and
        # per-call spelling repeats the primary resolution and the uniqueness scans per site.
        all_sites = sites(root) if all_sites is None else all_sites
        fam_sites = family_sites(family, all_sites)
        head = primary(family, fam_sites)
        table = {}
        for s in fam_sites:
            if id(s.node) not in table:  # a shared subtree keeps its FIRST site's spelling
                table[id(s.node)] = _spellings(family, s, fam_sites, head=head)
        tables[family] = table
    spelled = table.get(id(node))
    if spelled is not None:
        return spelled
    all_sites = sites(root) if all_sites is None else all_sites
    if not any(s.node is node for s in all_sites):
        raise UnknownSiteError(
            f"{type(node).__name__} is not a site of this tree — the caller holds a copied or "
            f"rebuilt node, not the stored object the site walk enumerated"
        )
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
    result_key = _result_key(key)
    if result_key is not None:
        base, position = result_key
        site = resolve(root, base, all_sites=all_sites)
        if site is None or not isinstance(site.node, Fold) or site.node.axis is not None:
            raise MissingSiteError(f"knob key {key!r} names no zero-axis result on this tree")
        results = _operand_result_names(site.node)
        if len(results) < 2 or position > len(results):
            raise MissingSiteError(f"knob key {key!r} names no result on this tree")
        return Site(site.node, site.axis, site.segments, site.ordinal, site.derived, position)

    parsed = parse_key(key)
    matched_key = parsed
    all_sites = sites(root) if all_sites is None else all_sites
    if parsed.segments == ("root",) and parsed.axis is None and parsed.ordinal is None:
        family_sites(parsed.family, ())  # validate the family without admitting root to bare PLACE
        return all_sites[0] if all_sites else None
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
    if not matches and parsed.axis is not None:
        m = _AXIS_ORDINAL_RE.match(parsed.axis)
        if m and m.group(1):
            digit_start = len(m.group(1))
            # Preserve as many trailing digits as possible in the axis first: ``a22`` reads
            # ``a2`` + ordinal 2 before ``a`` + ordinal 22.
            for split in range(len(parsed.axis) - 1, digit_start - 1, -1):
                prefix, suffix = parsed.axis[:split], parsed.axis[split:]
                readings = [_Key(parsed.family, parsed.segments, prefix, int(suffix))]
                if prefix in _SEGMENT_TOKENS:
                    readings.append(_Key(parsed.family, (*parsed.segments, prefix), None, int(suffix)))
                for retry in readings:
                    matches = _match(retry, fam_sites)
                    if matches:
                        matched_key = retry
                        break
                if matches:
                    break
    if not matches:
        raise MissingSiteError(f"knob key {key!r} names no site on this tree (a structural change broke a stored key?)")
    if len({id(s.node) for s in matches}) > 1:
        # An EXACT full-path match outranks subsequence admissions: a shallow site's full path is
        # an anchored subsequence of every deeper same-axis path, so without this preference the
        # canonical full-path spelling (the ordinal arm's fallback) could never name the shallow
        # site at all. Only consulted at the ambiguity point — sugar that was unique stays unique.
        exact = [s for s in matches if s.segments == matched_key.segments]
        if len({id(s.node) for s in exact}) == 1:
            return exact[0]
        cands = " or ".join(sorted(_spellings(parsed.family, s, fam_sites) for s in matches))
        raise ValueError(f"knob key {key!r} is ambiguous: use {cands}")
    # Several matches that are ONE node are not ambiguous: a shared subtree is a site at each path
    # it appears under (MoE experts under one ``Map``, a repeated fold step), and one node carries
    # one schedule, so the key names one decision however many paths reach it. :func:`spell` keys
    # by node identity and already gives them a single spelling; refusing it here would make that
    # spelling unresolvable.
    return matches[0]


def canonical(root, key: str, *, all_sites: tuple[Site, ...] | None = None) -> str | None:
    """The canonical spelling of ``key`` on ``root``'s tree — ``resolve`` then ``spell``; ``None``
    when the family doesn't apply (a bare decided-empty stamp)."""
    all_sites = sites(root) if all_sites is None else all_sites
    site = resolve(root, key, all_sites=all_sites)
    if site is None:
        return None
    if site.result is not None:
        return spell_result(root, site.node, site.result, all_sites=all_sites)
    return spell(root, parse_key(key).family, site.node, all_sites=all_sites)


def _result_key(key: str) -> tuple[str, int] | None:
    """Return the base node key and one-based position from a selected-result spelling."""
    family, at, suffix = key.partition("@")
    if not at:
        return None
    components = suffix.split(".")
    if len(components) < 2 or components[-2] != "result" or not components[-1].isdigit():
        return None
    position = int(components[-1])
    if position < 1:
        raise ValueError(f"knob key {key!r} has a non-positive result position")
    base_suffix = ".".join(components[:-2])
    return (family if not base_suffix else f"{family}@{base_suffix}"), position


def spell_result(root, node: Fold, position: int, *, all_sites: tuple[Site, ...] | None = None) -> str:
    """Canonical structural spelling for one result of a multi-result zero-axis Fold."""
    results = _operand_result_names(node)
    if node.axis is not None or len(results) < 2 or not 1 <= position <= len(results):
        raise ValueError("a result address requires a valid position on a multi-result zero-axis Fold")
    base = spell(root, "PLACE", node, all_sites=all_sites)
    return f"PLACE@result.{position}" if base == "PLACE" else f"{base}.result.{position}"


__all__ = [
    "PATH_FAMILIES",
    "MissingSiteError",
    "Site",
    "UnknownSiteError",
    "canonical",
    "family_sites",
    "parse_key",
    "primary",
    "resolve",
    "sites",
    "spell",
    "spell_result",
]
