"""Fork interface + implementations: the deferred fork options the search
engine ranks and resolves, and the hierarchical Fork-tree builder shared by
pipeline rules that enumerate a knob cartesian.

:class:`Fork` is the interface — ``knobs``, ``is_leaf``, ``expand()``.
Implementations hold their producer's state as data:
:class:`OptionFork` (a concrete ``Op``/``Graph`` leaf) and the tree node
classes :class:`_Branch` / :class:`_Leaf` built by :func:`build_fork_tree`.

The tree builder reads an addressable sequence of variant knob rows through
the root :class:`_Branch`: each ``Level`` groups
siblings by a (sub)tuple of knob values and collapses levels whose key has
a single distinct value across the group (rows with an empty key skip the
level). Below the last level every row becomes one :class:`_Leaf` carrying
its COMPLETE row as ``knobs`` — the row IS the variant identity (the
``S_*`` structural-feature knobs ride the merged dict), so the perf DB and
the online prior key leaves and branches by knobs alone, no structural
probing. ``expand()`` yields ``materialize(row)`` once the search engine
resolves a leaf.
Everything is lazy: construction reads no row, no Fork below the root exists until search expands
it, and branches retain indices into the shared sequence rather than row copies. Siblings are
emitted in grouping order — RANKING IS SEARCH POLICY: the
policies rank the frontier with the online prior (Forks carry no score).

The engine in ``pipeline.py`` consumes ``fork.knobs`` flat (it doesn't walk
ancestors): branch Forks pin their level's slice of the row, leaves carry
the whole row.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph
    from emmy.compiler.ir.base import Op

from emmy.compiler.ir.schedule import Schedule, ScheduleContext, schedule
from emmy.compiler.pipeline.knob import evidence_row_vouches, values_equal


class Fork(ABC):
    """Interface for a deferred fork option in the search tree.

    Two flavors share the interface:

    - **Branch Fork** (``is_leaf=False``) — produced explicitly by a rule's
      ``rewrite()`` to spawn a hierarchical fork point. ``expand()`` returns
      the next level of options (more Forks, concrete leaves, or a mix);
      the search loop drives this via :meth:`LazyCandidate.expand`.
    - **Leaf Fork** (``is_leaf=True``) — wraps one concrete ``Op`` /
      ``Graph`` rewrite. ``expand()`` returns ``[option]`` (one element);
      :meth:`LazyCandidate.resolve` invokes it once at resolve time to
      retrieve the leaf and apply it.

    Sharing one interface lets ``LazyCandidate.pending`` carry just
    ``Fork`` (no tagged union) — the search loop branches on
    ``Fork.is_leaf`` to decide expand-vs-resolve.

    ``knobs`` is the knob-delta this Fork pins (the variant identity the
    perf DB and the online prior key on, read without expanding). Ranking
    is SEARCH policy: the engine hands unranked siblings to ``Search.push``
    and the policy ranks them with the
    :class:`~emmy.compiler.pipeline.search.prior.Prior` (greedy
    ``mean_score`` argmin; MCTS PUCT). Forks carry no score of their own —
    the hand-coded per-fork scorer was removed when the online prior replaced
    it; siblings are emitted in grouping order and the cold/no-prior fallback
    is that emission order."""

    knobs: dict
    is_leaf: bool = False
    structural: bool = False
    #: The enumeration's minted pool identity, carried by every node of a schedule tree
    #: (the schedule adapter mints it from the variant key + hints + pins +
    #: split receipt + spelled key vocabulary). ``None`` for forks outside a schedule enumeration.
    #: Consumers key memoized decisions on THIS, never on a re-derived identity.
    pool_id: str | None = None
    #: Upper bound on the enumeration's leaf count (Π of the per-node option tuples × the RASTER
    #: fan-out — legality only shrinks it), carried the same way. ``None`` outside a schedule
    #: enumeration. The greedy cold-pool budget triggers on this without walking anything.
    pool_bound: int | None = None
    #: Upper bound on the option checks one random descent performs (the sum of per-node option
    #: tuple lengths). ``None`` outside a schedule enumeration. The cold-pool sampler uses this
    #: to bound work as well as the number of complete rows it draws.
    pool_descent_bound: int | None = None

    @abstractmethod
    def expand(self) -> list[Op | Graph | Fork]: ...

    def leaves(self) -> Iterator[Op | Graph | Fork]:
        """Stream complete descendants without retaining the expanded tree."""
        if self.is_leaf:
            yield self
        else:
            yield from iter_leaves(self.expand())

    def admits(self, row: Mapping) -> bool:
        """Whether a knob ``row`` — complete, or partial with the undecided knobs absent — can lie
        below this branch: every knob the branch has decided agrees with the row. The one descent
        rule for a row that names a leaf (the decision memo's replay, the evidence pick's direct
        descent to a measured row). The base reading is value equality; a tree whose branches
        carry a prefix of the leaf's spelling or a level's projection of it refines it."""
        return all(
            name not in row or values_equal(name, row[name], value)
            for name, value in self.knobs.items()
            if not name.startswith(("S_", "H_"))
        )


@dataclass(frozen=True)
class OptionFork(Fork):
    """Leaf Fork around an already-concrete rewrite option. Built by
    :meth:`LazyCandidate.from_option` so every ``LazyCandidate.pending``
    carries a uniform Fork shape."""

    option: Op | Graph
    knobs: dict = field(default_factory=dict)
    is_leaf = True

    @property
    def structural(self) -> bool:
        from emmy.compiler.graph import Graph  # noqa: PLC0415

        return isinstance(self.option, Graph)

    def expand(self) -> list[Op | Graph | Fork]:
        return [self.option]


@dataclass(frozen=True)
class DeferredFork(Fork):
    """A lazy concrete leaf whose selected ``Op`` or ``Graph`` is built on expansion."""

    materialize: Callable[[], Op | Graph]
    knobs: dict = field(default_factory=dict)
    structural: bool = False
    is_leaf = True

    def expand(self) -> list[Op | Graph | Fork]:
        return [self.materialize()]


@dataclass(frozen=True)
class _ScheduleTree:
    """Shared callbacks and identity for one generic lazy schedule tree."""

    branch_knobs: Mapping
    row_delta: Callable[[ScheduleContext, ScheduleContext], Mapping]
    leaf: Callable[[Schedule], Fork]
    pool_id: str
    pool_bound: int
    pool_descent_bound: int

    def step(self, context: ScheduleContext, row: Mapping) -> list[Fork]:
        forks = []
        for child in schedule(context, recursive=False):
            if isinstance(child, Schedule):
                forks.append(self.leaf(child))
            else:
                forks.append(_ScheduleFork(self, child, {**row, **self.row_delta(context, child)}))
        return forks


@dataclass(frozen=True)
class _ScheduleFork(Fork):
    """One immutable prefix in a generic lazy schedule tree."""

    tree: _ScheduleTree
    context: ScheduleContext
    row: Mapping

    @property
    def knobs(self) -> dict:
        return {**self.tree.branch_knobs, **self.row}

    @property
    def pool_id(self) -> str:
        return self.tree.pool_id

    @property
    def pool_bound(self) -> int:
        return self.tree.pool_bound

    @property
    def pool_descent_bound(self) -> int:
        return self.tree.pool_descent_bound

    def expand(self) -> list[Fork]:
        return self.tree.step(self.context, self.row)

    def admits(self, row: Mapping) -> bool:
        """A schedule branch spells each decided knob as the PREFIX of what its leaves will spell
        (``w2x2`` before ``w2x2+p1``, ``…/f2x2`` before ``…/f2x2/k2``; an OFF default ``''`` before
        anything), so the row's value must extend the branch's value at a segment boundary."""
        for name, value in self.knobs.items():
            if name.startswith(("S_", "H_")) or name not in row:
                continue
            want, have = str(row[name]), str(value)
            if have and want != have and not want.startswith((have + "/", have + "+")):
                return False
        return True


def schedule_forks(
    context: ScheduleContext,
    *,
    branch_knobs: Mapping,
    row_delta: Callable[[ScheduleContext, ScheduleContext], Mapping],
    leaf: Callable[[Schedule], Fork],
    pool_id: str,
    pool_bound: int,
    pool_descent_bound: int,
) -> list[Fork]:
    """Represent any schedule context as a lazy pipeline Fork tree."""
    tree = _ScheduleTree(dict(branch_knobs), row_delta, leaf, pool_id, pool_bound, pool_descent_bound)
    return tree.step(context, {})


def iter_leaves(options: Iterable[Op | Graph | Fork]) -> Iterator[Op | Graph | Fork]:
    """Yield complete leaves depth-first without retaining the expanded tree."""
    for option in options:
        if isinstance(option, Fork):
            yield from option.leaves()
        else:
            yield option


def flatten_leaves(options: Sequence[Op | Graph | Fork]) -> list[Op | Graph | Fork]:
    """Expand every option down to its leaf options, **depth-first in emission
    order** — each option's leaves precede the next's, so a tie in a prior's
    scores still falls to enumeration order (option-0 first). Branch Forks
    expand recursively — cheap, building only the next levels' knob dicts;
    leaf Forks and concrete ``Op`` / ``Graph`` options terminate, their
    materialization deferred to whoever applies the one chosen leaf. Used for
    small non-schedule forks whose alternatives must be compared together;
    schedule spaces instead retain this hierarchy during greedy descent."""
    return list(iter_leaves(options))


def fork_signature(root_op: Op, options: Sequence[Op | Graph | Fork], ctx) -> frozenset:
    """The ``S_*`` signature every candidate at one fork shares — the key a measured row of this
    kernel is filed under. The offer op's structural stamp under the run's context features, plus
    the stamps the enumeration itself minted on its options (``S_warp_eligible``: a property of
    the offered space, carried on the pool's top level and inherited by every leaf). Read here by
    the deploy's evidence pick and by the golden replay that keys a record's rows, so the two
    agree by construction."""
    base = {**ctx.features(), **dict(getattr(root_op, "knobs", None) or {})}
    for option in options:
        base.update((key, value) for key, value in (getattr(option, "knobs", None) or {}).items() if key.startswith("S_"))
    return frozenset((key, str(value)) for key, value in base.items() if key.startswith("S_"))


def leaf_for(options: Sequence[Op | Graph | Fork], row: Mapping, *, skip: Callable[[dict], bool] | None = None):
    """The first leaf a (possibly partial) knob ``row`` vouches for, as ``(leaf, its knobs)``, or
    ``None`` — descending only the branches that admit the row (:meth:`Fork.admits`), so the walk
    instantiates O(path × siblings) Forks whatever the pool size. ``skip`` drops a leaf by its knobs
    (a blocklisted tile). The one descent the evidence pick and the golden replay share."""
    for option in options:
        if isinstance(option, Fork) and not option.is_leaf:
            if option.admits(row):
                found = leaf_for(option.expand(), row, skip=skip)
                if found is not None:
                    return found
            continue
        knobs = leaf_knobs(option)
        if skip is not None and skip(knobs):
            continue
        tunable = {key: str(value) for key, value in knobs.items() if not key.startswith(("S_", "H_"))}
        if evidence_row_vouches(tunable, row):
            return option, knobs
    return None


def leaf_knobs(leaf: Op | Graph | Fork) -> dict:
    """A leaf's complete knob row: a leaf ``Fork`` carries it as ``knobs``; a concrete ``Op``
    carries its own; a ``Graph`` splice has no single row (scored structurally, never by knobs) —
    empty, matching how ``LazyCandidate.from_option`` treats it during the tuning search."""
    from emmy.compiler.graph import Graph  # noqa: PLC0415

    if isinstance(leaf, Fork):
        return dict(leaf.knobs)
    return dict(getattr(leaf, "knobs", None) or {}) if not isinstance(leaf, Graph) else {}


# ---------------------------------------------------------------------------
# Hierarchical Fork-tree builder (``Level`` + ``build_fork_tree``).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Level:
    """One grouping level in the Fork tree.

    ``knob_names`` and ``key`` must agree in arity: ``key(row)`` returns a
    tuple of the same length as ``knob_names``, in matching order — or an
    EMPTY tuple when the level doesn't apply to ``row``. Rows with an
    empty key skip the level: their next-level subtree splices up as
    siblings of the level's keyed branches, so e.g. scalar tile variants
    carry no ``MMA`` branch while warp variants of the same kernel do.
    Across all Levels the ``knob_names`` should partition the knob set the
    caller wants BRANCHES to pin (no duplicates along a path); levels need
    not cover every row knob — leaves carry the complete row regardless.
    """

    knob_names: tuple[str, ...]
    key: Callable[[dict], tuple]
    partition_key: str | None = None


@dataclass(frozen=True)
class _Tree[P]:
    """One tree's shared builder state — every :class:`_Branch` /
    :class:`_Leaf` node holds a reference back here instead of capturing
    closures."""

    levels: tuple[Level, ...]
    materialize: Callable[[dict], Op | Graph]

    def build_level(self, group: _Rows, depth: int) -> list[Fork]:
        """Build the sibling Forks one level down from a branch at
        ``depth`` (in grouping order — ranking is the search's job)."""
        if depth == len(self.levels):
            # One leaf per row, carrying the COMPLETE row as its knobs —
            # the DB-matchable variant identity (levels may not cover
            # every knob, e.g. FK).
            return [_Leaf(tree=self, knobs=dict(row)) for row in group]
        level = self.levels[depth]
        # Rows whose key is empty skip the level (it doesn't apply to them) — their next-level
        # subtree splices up as siblings of the keyed branches below. Addressable products supply
        # this partition structurally; ordinary sequences use the indexed fallback.
        keyed, skipped = group.partition(level)
        if not keyed:
            # Level applies to nothing in this group — skip it wholesale.
            return self.build_level(group, depth + 1)
        # Single-value collapse: the level adds no choice, so skip the
        # 1-child Fork wrapper and recurse straight into the next level.
        if skipped is None and len(keyed) == 1:
            return self.build_level(keyed[0][1], depth + 1)
        siblings: list[Fork] = [
            _Branch(tree=self, group=sub, next_depth=depth + 1, knobs=dict(zip(level.knob_names, key, strict=True))) for key, sub in keyed
        ]
        if skipped is not None:
            siblings.extend(self.build_level(skipped, depth + 1))
        return siblings


@dataclass(frozen=True)
class _Rows:
    """An index view over one shared addressable row space.

    Branches retain integer indices, never copies of the row dictionaries. The root uses a
    ``range`` and therefore does not read or allocate anything proportional to the schedule space
    until that branch is expanded.
    """

    source: Sequence[dict]
    indices: Sequence[int]

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[dict]:
        return (self.source[index] for index in self.indices)

    def indexed(self) -> Iterator[tuple[int, dict]]:
        return ((index, self.source[index]) for index in self.indices)

    def subset(self, indices: Sequence[int]) -> _Rows:
        return _Rows(self.source, tuple(indices))

    def partition(self, level: Level) -> tuple[list[tuple[tuple, _Rows]], _Rows | None]:
        """Partition this view at ``level``, delegating to a structural row space when available."""
        full = isinstance(self.indices, range) and self.indices == range(len(self.source))
        structural = getattr(self.source, "partition", None)
        if full and level.partition_key is not None and structural is not None:
            keyed = []
            skipped = None
            for value, source in structural(level.partition_key):
                view = _Rows(source, range(len(source)))
                if value == "":
                    skipped = view
                else:
                    keyed.append(((value,), view))
            return keyed, skipped

        keyed: dict[tuple, list[int]] = {}
        skipped: list[int] = []
        for index, row in self.indexed():
            key = level.key(row)
            (keyed.setdefault(key, []) if key else skipped).append(index)
        groups = [(key, self.subset(indices)) for key, indices in keyed.items()]
        return groups, self.subset(skipped) if skipped else None


@dataclass(frozen=True)
class _Branch(Fork):
    """Branch node: a subgroup of knob rows pinned to ``knobs`` by its
    level key. The subtree below doesn't exist until the engine pops the
    branch and ``expand()`` builds the next level."""

    tree: _Tree
    group: _Rows
    next_depth: int
    knobs: dict

    def expand(self) -> list[Op | Graph | Fork]:
        return self.tree.build_level(self.group, self.next_depth)

    def admits(self, row: Mapping) -> bool:
        """The level that keyed this branch projects the row to this branch's key; a row lacking
        one of the level's knobs is undecided there and admitted, a row the level does not apply
        to belongs to the skipped siblings."""
        if self.next_depth == 0:
            return True  # the root: no level keyed it
        level = self.tree.levels[self.next_depth - 1]
        if any(name not in row for name in level.knob_names):
            return True
        try:
            key = tuple(str(part) for part in level.key(dict(row)))
        except KeyError:
            return True
        return key == tuple(str(self.knobs[name]) for name in level.knob_names)

    def leaves(self) -> Iterator[Fork]:
        """Stream the subgroup's complete rows directly when a policy needs every leaf.

        Branch construction exists for recursive search.  An exhaustive policy already needs the
        full rows, so replaying every grouping level would only rescan and regroup the same index
        set.
        """
        for row in self.group:
            yield _Leaf(tree=self.tree, knobs=dict(row))


@dataclass(frozen=True)
class _Leaf(Fork):
    """Leaf node: one knob row (= ``knobs``, the complete variant
    identity); ``expand()`` materializes its Op/Graph."""

    tree: _Tree
    knobs: dict
    is_leaf = True

    def expand(self) -> list[Op | Graph | Fork]:
        return [self.tree.materialize(self.knobs)]


def build_fork_tree(
    *,
    params: Sequence[dict],
    levels: Sequence[Level],
    materialize: Callable[[dict], Op | Graph],
) -> Fork:
    """Return the ROOT branch ``Fork`` of a lazy tree grouping the knob
    rows ``params`` per ``levels`` (outermost first); below the last
    level each row becomes one leaf carrying its complete row as
    ``knobs``. ``params`` must be non-empty (a rule with nothing to
    enumerate has no fork point — skip the rule instead) and ``levels``
    non-empty; both raise ``ValueError``.

    Nothing is built at call time — the root is a :class:`_Branch` over a range into the shared row
    sequence; each branch's ``expand()`` builds the next level
    on demand, so greedy descent instantiates O(path) Forks instead of
    one per row (~42k for a matmul-class kernel) and MCTS pays one level
    per pop. Siblings are emitted in grouping order; ranking is the
    search policy's job (the online prior), not the tree's.
    """
    if not params:
        raise ValueError("build_fork_tree: params must be non-empty")
    if not levels:
        raise ValueError("build_fork_tree: at least one Level required")
    tree = _Tree(levels=tuple(levels), materialize=materialize)
    return _Branch(tree=tree, group=_Rows(params, range(len(params))), next_depth=0, knobs={})
