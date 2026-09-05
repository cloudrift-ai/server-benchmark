"""The structural placement path codec over the stored Fold tree.

Grammar: ``PLACE@<kind>.<i>/…/<kind>`` — a ROUTE from the root. Every segment but the last names a
node stood on and the 1-based operand taken from it (``map.1``: at the root map, take operand 1);
the last names the kind of the node arrived at, the stored Fold edge the placement decision is
about. So ``PLACE@map.1/twist.1/inner.2/map`` reads "root map, first operand; the twist there,
first operand; the contraction there, second operand: a map — cut it". The kinds are the term's
derived readings (``map`` a zero-axis term, ``reduce`` a planar fold, ``inner`` a bilinear one,
``twist`` a rescaling carrier, ``scan`` an observed fold), never a stored tag. Placement addresses
a stored edge by position because it is a structural decision made before a classic problem
exists.

A spelling is unique by construction — the tree is established by the algebra, and a position in
it moves only when the computation does — so there are no ordinals, no shortest-unique search and
no axis names, the three ingredients that used to move a recorded key under a tree that had not
changed. The kinds are redundant with the positions; they make a key readable and a stale one fail
loudly, at the first segment whose kind is not what stands there. A bare ``PLACE`` is input sugar
for the family's one site, or its primary (the unique shallowest) among several; the retired
``in.<operand>`` prefix and leading-``=`` form stay reserved.

Classic choices never use this codec. A classic problem constructs sites only after every
structural choice is consumed, and its strict codec addresses integer node ids and
``(consumer, operand)`` edge tuples.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.structural import instance_memo

#: The families a tree path may address — the STRUCTURAL decisions, made before a classic problem
#: exists. Schedule identities belong to ``schedule.classic``.
PATH_FAMILIES = ("PLACE", "BLOCK")

#: The node kinds a route may name — each a derived reading of the term.
KINDS = ("map", "reduce", "inner", "twist", "scan")


class MissingSiteError(ValueError):
    """A suffixed knob key that names NO site on this tree. A ``ValueError`` so every existing
    handler still reads it as a broken stored key failing loudly; its own type so a caller
    holding GRAPH-scoped pins (the placement rule) can tell "this key addresses another kernel"
    apart from an ambiguity, which stays a plain ``ValueError``."""


class UnknownSiteError(Exception):
    """A node outside the stored tree was used for a placement or lowering lookup."""


def kind(node) -> str:
    """The node's kind as a route names it — its derived reading, in the order the readings nest:
    an observed fold is a ``scan`` whatever it folds, a rescaling carrier is a ``twist`` whatever
    it carries, a bilinear planar fold is an ``inner``."""
    if node.axis is None:
        return "map"
    if node.observe is not None:
        return "scan"
    view = node.as_reduction()
    if view is not None and view.twisted:
        return "twist"
    return "inner" if node.as_contraction() is not None else "reduce"


@dataclass(frozen=True)
class Site:
    """One structural tree position: the node, its axis (``None`` for a zero-axis term), the
    ``hops`` that reach it — ``(kind of the node departed, 1-based operand taken)`` per step from
    the root, the root's own tuple empty — and ``scope``, the axes (by name) the route binds above
    the node. Every Fold but a slab is a site; residence choices on an enclosing edge never hide its
    algebra."""

    node: object
    axis: str | None
    hops: tuple[tuple[str, int], ...]
    scope: tuple[str, ...] = ()

    @property
    def depth(self) -> int:
        return len(self.hops)

    @property
    def path(self) -> str:
        """The route spelled: each departure ``kind.index``, the arrival's kind last."""
        return "/".join((*(f"{label}.{index}" for label, index in self.hops), kind(self.node)))

    def under(self, other: Site) -> bool:
        """Whether this site lies below ``other`` — ``other``'s route is a proper prefix of this one's."""
        return len(self.hops) > len(other.hops) and self.hops[: len(other.hops)] == other.hops


def sites(root) -> tuple[Site, ...]:
    """Every node of ``root``'s tree that carries a schedule decision, as a :class:`Site`, preorder,
    root first — the ONE node walk in the layer. A slab is not a site: a gmem read has no residence,
    partition or tile of its own — the atom reads it through its parent — so it takes no hop. A
    subterm reached down two paths (the epilogue's ``1/l`` cone reads the fold the root also holds)
    is ONE site at the first position that reached it: sharing is edge reuse, and a shared value has
    one schedule — the same rule ``TileOp.sites`` keeps."""
    if root is None:
        return ()
    result: list[Site] = []
    seen: set[int] = set()

    def visit(node, scope: tuple[str, ...], hops: tuple[tuple[str, int], ...]) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        result.append(Site(node=node, axis=node.axis, hops=hops, scope=scope))
        inner = scope if node.axis is None else (*scope, node.axis)
        label = kind(node)
        for position, edge in enumerate(node.operands, start=1):
            if edge.as_slab() is None:
                visit(edge, inner, (*hops, (label, position)))

    visit(root, (), ())
    return tuple(result)


def family_sites(family: str, all_sites: tuple[Site, ...]) -> tuple[Site, ...]:
    """Stored non-root Fold edges eligible for structural placement."""
    if family not in PATH_FAMILIES:
        raise ValueError(f"{family!r} is not a structural path family (have {PATH_FAMILIES})")
    return tuple(s for s in all_sites if s.depth > 0)


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
    hops: tuple[tuple[str, int], ...]
    target: str | None  # the arrival's kind; ``None`` for a bare key

    @property
    def bare(self) -> bool:
        return self.target is None


def parse_key(key: str) -> _Key:
    """Split a knob key into its family, its departures and the arrival's kind, rejecting the
    RESERVED graph-level placement forms and any segment off the grammar."""
    family, at, suffix = key.partition("@")
    if not at:
        return _Key(family=family, hops=(), target=None)
    if suffix.startswith("=") or suffix == "in" or suffix.startswith("in."):
        raise ValueError(f"knob key {key!r} is reserved for graph-level placement")
    if not suffix:
        raise ValueError(f"knob key {key!r} has an empty @-suffix")
    *departures, arrival = suffix.split("/")
    if arrival not in KINDS:
        raise ValueError(f"knob key {key!r}: the last segment names the node arrived at by kind, one of {KINDS}, not {arrival!r}")
    hops: list[tuple[str, int]] = []
    for comp in departures:
        label, dot, index = comp.partition(".")
        if label not in KINDS:
            raise ValueError(f"knob key {key!r}: unknown path segment {label!r} (kinds are {KINDS})")
        if not dot or not index.isdigit() or int(index) < 1 or str(int(index)) != index:
            raise ValueError(f"knob key {key!r}: departure {comp!r} needs a 1-based operand index, <kind>.<index>")
        hops.append((label, int(index)))
    return _Key(family=family, hops=tuple(hops), target=arrival)


def spell(root, family: str, node, *, all_sites: tuple[Site, ...] | None = None) -> str:
    """The CANONICAL key addressing ``node`` under ``family`` on ``root``'s tree: bare when the
    family has that one site, else ``FAMILY@<route>``. Stampers and stored evidence use this
    spelling and nothing else."""
    tables = instance_memo(root, "_memo_spellings")
    table = tables.get(family)
    if table is None:
        # The whole family spells as ONE derived table (an ``instance_memo`` on the immutable
        # root): the schedule pricing loops re-spell the same sites once per candidate row.
        all_sites = sites(root) if all_sites is None else all_sites
        fam_sites = family_sites(family, all_sites)
        table = {id(s.node): family if len(fam_sites) == 1 else f"{family}@{s.path}" for s in fam_sites}
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
    """Resolve a knob ``key`` to the :class:`Site` it addresses on ``root``'s tree. Total over both
    forms and idempotent (``resolve(spell(...))`` is the same site):

    - bare, no eligible site → ``None`` (the family doesn't apply — drop / decided-empty);
    - bare, several sites → the PRIMARY, or ``ValueError`` naming the canonical candidates;
    - a route is walked segment by segment, and one whose kind is not what stands there, whose
      index is past the node's operands, or that arrives at a slab is a :class:`MissingSiteError`
      — a stored key broken by a structural change fails loudly, never silently re-keys.
    """
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
        shallowest = min(s.depth for s in fam_sites)
        cands = " or ".join(sorted(f"{parsed.family}@{s.path}" for s in fam_sites if s.depth == shallowest))
        raise ValueError(f"{parsed.family} is ambiguous: use {cands}")

    def missing(why: str) -> MissingSiteError:
        routes = ", ".join(s.path for s in fam_sites[:12]) + (", …" if len(fam_sites) > 12 else "")
        return MissingSiteError(f"knob key {key!r} names no site on this tree: {why} (the tree's sites: {routes})")

    node = root
    for position, (label, index) in enumerate(parsed.hops, start=1):
        stood = kind(node)
        if stood != label:
            raise missing(f"segment {position} stands on a {stood}, not a {label}")
        if index > len(node.operands):
            raise missing(f"segment {position} takes operand {index} of {len(node.operands)}")
        node = node.operands[index - 1]
    if kind(node) != parsed.target:
        raise missing(f"it arrives at a {kind(node)}, not a {parsed.target}")
    site = next((s for s in fam_sites if s.node is node), None)
    if site is None:
        raise missing("it arrives at a slab, which carries no placement of its own")
    return site


def canonical(root, key: str, *, all_sites: tuple[Site, ...] | None = None) -> str | None:
    """The canonical spelling of ``key`` on ``root``'s tree — ``resolve`` then ``spell``; ``None``
    when the family doesn't apply (a bare decided-empty stamp)."""
    all_sites = sites(root) if all_sites is None else all_sites
    site = resolve(root, key, all_sites=all_sites)
    if site is None:
        return None
    return spell(root, parse_key(key).family, site.node, all_sites=all_sites)


__all__ = [
    "KINDS",
    "PATH_FAMILIES",
    "MissingSiteError",
    "Site",
    "UnknownSiteError",
    "canonical",
    "family_sites",
    "kind",
    "parse_key",
    "primary",
    "resolve",
    "sites",
    "spell",
]
