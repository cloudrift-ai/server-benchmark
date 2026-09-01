"""The semantic model for the classic grid/CTA/warp/thread/register schedule.

The model contains choices only.  A :class:`ClassicProblem` supplies the immutable Fold tree and
target; :class:`ClassicScheduleContext` derives identities and classification from that problem and
is the only compatibility authority for a complete assignment. Codecs, enumeration order, search
state, and materialization data do not belong here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product

from frozendict import frozendict

from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.structural import instance_memo

from .base import Schedule, ScheduleContext, ScheduleRefused
from .choices import (
    PlacedTile,
    Raster,
    Reduce,
    ResolvedStage,
    Stage,
    Tile,
    Work,
    derive_inventory,
    resolve_site_tile,
)
from .views import EdgeSite, NodeId, NodeView, Projection, Reduction, node_view, schedule_edges, schedule_nodes

CLASSIC_NODE_FAMILIES = ("TILE", "REDUCE")
CLASSIC_EDGE_FAMILIES = ("STAGE",)
CLASSIC_FAMILIES = (*CLASSIC_NODE_FAMILIES, *CLASSIC_EDGE_FAMILIES)


def node_id_spelling(node_id: NodeId) -> str:
    """Return one node identity's canonical wire spelling."""
    if type(node_id) is not int or node_id < 0:
        raise ValueError(f"node id must be a non-negative integer, got {node_id!r}")
    return f"n{node_id}"


def parse_node_id(value: str) -> NodeId:
    """Parse one canonical node identity."""
    if not value.startswith("n") or not value[1:].isdigit() or str(int(value[1:])) != value[1:]:
        raise ValueError(f"node id must be n<ordinal>, got {value!r}")
    return int(value[1:])


def edge_site_spelling(edge: EdgeSite) -> str:
    """Return one consumer operand position's canonical wire spelling."""
    if not _is_edge_site(edge):
        raise ValueError(f"edge site must be a (node id, operand) pair, got {edge!r}")
    return f"{node_id_spelling(edge[0])}.e{edge[1]}"


def parse_edge_site(value: str) -> EdgeSite:
    """Parse one canonical consumer operand position."""
    node, separator, operand = value.partition(".e")
    if separator != ".e" or not operand.isdigit() or str(int(operand)) != operand:
        raise ValueError(f"edge site must be n<ordinal>.e<operand>, got {value!r}")
    return parse_node_id(node), int(operand)


def _is_node_id(node_id: object) -> bool:
    return type(node_id) is int and node_id >= 0


def _is_edge_site(edge: object) -> bool:
    return isinstance(edge, tuple) and len(edge) == 2 and _is_node_id(edge[0]) and type(edge[1]) is int and edge[1] >= 0


@dataclass(frozen=True)
class KernelSchedule:
    """Kernel-scoped choices."""

    work: Work
    raster: Raster

    def __post_init__(self) -> None:
        if not isinstance(self.work, Work) or not isinstance(self.raster, Raster):
            raise TypeError("classic kernel choices must be Work and Raster values")


@dataclass(frozen=True)
class ProjectionSchedule:
    """The choices of a projection node."""

    tile: Tile

    def __post_init__(self) -> None:
        if not isinstance(self.tile, Tile):
            raise TypeError("classic projection TILE must be an unplaced Tile choice")


@dataclass(frozen=True)
class ReductionSchedule:
    """The choices of a reduction node, including a contraction-capable reduction."""

    tile: Tile
    reduce: Reduce

    def __post_init__(self) -> None:
        if not isinstance(self.tile, Tile):
            raise TypeError("classic reduction TILE must be an unplaced Tile choice")
        if not isinstance(self.reduce, Reduce):
            raise TypeError("classic reduction REDUCE must be a Reduce choice")


NodeSchedule = ProjectionSchedule | ReductionSchedule


@dataclass(frozen=True)
class EdgeSchedule:
    """The transport choice of one operand use."""

    stage: Stage

    def __post_init__(self) -> None:
        if not isinstance(self.stage, Stage):
            raise TypeError("classic edge STAGE must be a Stage choice")


type ClassicAssignment = Schedule[KernelSchedule, NodeSchedule, EdgeSchedule]


def _allow_schedule(_schedule: ClassicAssignment) -> bool:
    return True


@dataclass(frozen=True)
class AxisAgreement:
    """One physical-axis geometry claim carried by a local schedule offer."""

    name: str
    tile: int
    units: int


@dataclass(frozen=True)
class FragmentAgreement:
    """One producer or consumer claim at a fragment seam."""

    role: str
    edge: str
    value: tuple

    def __post_init__(self) -> None:
        if self.role not in ("need", "offer"):
            raise ValueError(f"fragment agreement role must be need or offer, got {self.role!r}")


@dataclass(frozen=True)
class LocalSupport:
    """Static support for one node choice and its incident edge choices.

    The public domains are projections of these records.  A support record is not a schedule:
    placed geometry and fragment facts remain derived compatibility evidence and never enter a
    :class:`Schedule` value.
    """

    node: NodeSchedule
    edges: Mapping[EdgeSite, EdgeSchedule]
    work: Work | None = None
    axes: tuple[AxisAgreement, ...] = ()
    fragments: tuple[FragmentAgreement, ...] = ()
    raster_eligible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.node, (ProjectionSchedule, ReductionSchedule)):
            raise TypeError("classic local support requires a node schedule")
        if not isinstance(self.edges, Mapping) or any(
            not _is_edge_site(edge) or not isinstance(choice, EdgeSchedule) for edge, choice in self.edges.items()
        ):
            raise TypeError("classic local support edges must map consumer operand pairs to EdgeSchedule")
        if self.work is not None and not isinstance(self.work, Work):
            raise TypeError("classic local support work must be Work or None")
        object.__setattr__(self, "edges", frozendict(self.edges))


@dataclass(frozen=True)
class ClassicDomains:
    """Independent finite domains plus one lazy local-support projection.

    ``kernel``, ``nodes`` and ``edges`` are the literal Cartesian factors. ``_support`` derives
    the static evidence for one already-selected node and its incident edges. It is deliberately
    lazy: the context applies ``c`` before asking for expensive placed and staged facts.
    """

    kernel: tuple[KernelSchedule, ...]
    nodes: Mapping[NodeId, tuple[NodeSchedule, ...]]
    edges: Mapping[EdgeSite, tuple[EdgeSchedule, ...]]
    _support: Callable[[NodeId, NodeSchedule, Mapping[EdgeSite, EdgeSchedule]], LocalSupport | None] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.kernel or any(not isinstance(choice, KernelSchedule) for choice in self.kernel):
            raise TypeError("classic kernel domain must contain KernelSchedule choices")
        for name, values, site_test, choice_type in (
            ("node", self.nodes, _is_node_id, (ProjectionSchedule, ReductionSchedule)),
            ("edge", self.edges, _is_edge_site, EdgeSchedule),
        ):
            if not isinstance(values, Mapping) or any(not site_test(site) for site in values):
                raise TypeError(f"classic {name} domains have invalid site keys")
            if any(not choices or any(not isinstance(choice, choice_type) for choice in choices) for choices in values.values()):
                raise TypeError(f"classic {name} domains have invalid choices")
        if not callable(self._support):
            raise TypeError("classic domains require a local-support projection")
        object.__setattr__(self, "nodes", frozendict({site: tuple(choices) for site, choices in self.nodes.items()}))
        object.__setattr__(self, "edges", frozendict({edge: tuple(choices) for edge, choices in self.edges.items()}))

    def local_support(
        self,
        site: NodeId,
        node: NodeSchedule,
        edges: Mapping[EdgeSite, EdgeSchedule],
    ) -> LocalSupport | None:
        """Derive static compatibility evidence for one local product member."""
        return self._support(site, node, edges)

    def local_frontier(
        self,
        site: NodeId,
        nodes: tuple[NodeSchedule, ...],
        edge_domains: tuple[tuple[EdgeSite, tuple[EdgeSchedule, ...]], ...],
    ) -> tuple[LocalSupport, ...]:
        """Return the locally supported node-plus-incident-edge frontier.

        The caller has already applied its immutable restriction to each factor. Classic transport
        currently requires one edge value per consumer, so mixed incident-edge products have no
        completion and are never materialized. The result is shared by every prefix carrying the
        same restriction-local factors.
        """
        cache = instance_memo(self, "_memo_local_frontier")
        key = (site, nodes, edge_domains)
        if cached := cache.get(key):
            return cached
        if key in cache:
            return ()
        if edge_domains:
            common = set(edge_domains[0][1])
            for _edge, choices in edge_domains[1:]:
                common.intersection_update(choices)
            edge_picks = tuple(
                frozendict({edge: choice for edge, _choices in edge_domains}) for choice in edge_domains[0][1] if choice in common
            )
        else:
            edge_picks = (frozendict(),)
        result = tuple(support for node in nodes for edges in edge_picks if (support := self.local_support(site, node, edges)) is not None)
        cache[key] = result
        return result

    def compatible_frontier(
        self,
        site: NodeId,
        nodes: tuple[NodeSchedule, ...],
        edge_domains: tuple[tuple[EdgeSite, tuple[EdgeSchedule, ...]], ...],
        work: Work | None,
        previous_nodes: tuple[NodeSchedule, ...],
        axes: tuple[tuple[str, tuple[int, int]], ...],
        fragments: tuple[tuple[tuple[str, str], tuple], ...],
        allowed_works: frozenset[tuple[str, tuple[int, ...]]] | None,
    ) -> tuple[LocalSupport, ...]:
        """Index the local frontier by the prefix facts that can reject it."""
        cache = instance_memo(self, "_memo_compatible_frontier")
        key = (site, nodes, edge_domains, work, previous_nodes, axes, fragments, allowed_works)
        if cached := cache.get(key):
            return cached
        if key in cache:
            return ()
        result = tuple(
            support
            for support in self.local_frontier(site, nodes, edge_domains)
            if ClassicScheduleContext._prefix_relation_refusal(
                support,
                work=work,
                previous_nodes=previous_nodes,
                axes=axes,
                fragments=fragments,
                allowed_works=allowed_works,
            )
            is None
        )
        cache[key] = result
        return result

    def __getstate__(self):
        """Pickle stored domains only; derived frontiers recompute after transport."""
        return {name: self.__dict__[name] for name in self.__dataclass_fields__ if name in self.__dict__}

    @property
    def product_size(self) -> int:
        """Number of assignments in the unfiltered Cartesian product."""
        size = len(self.kernel)
        for choices in (*self.nodes.values(), *self.edges.values()):
            size *= len(choices)
        return size


@dataclass(frozen=True)
class ClassicMaterialization:
    """Placed geometry and resolved transport facts derived from an accepted schedule."""

    tiles: Mapping[NodeId, PlacedTile]
    stages: Mapping[EdgeSite, ResolvedStage]

    def __post_init__(self) -> None:
        if not isinstance(self.tiles, Mapping) or not isinstance(self.stages, Mapping):
            raise TypeError("classic materialization tiles and stages must be mappings")
        if any(not _is_node_id(site) or not isinstance(tile, PlacedTile) for site, tile in self.tiles.items()):
            raise TypeError("classic materialization tiles must map node ids to PlacedTile")
        if any(not _is_edge_site(edge) or not isinstance(stage, ResolvedStage) for edge, stage in self.stages.items()):
            raise TypeError("classic materialization stages must map consumer operand pairs to ResolvedStage")
        object.__setattr__(self, "tiles", frozendict(self.tiles))
        object.__setattr__(self, "stages", frozendict(self.stages))

    def validate(self, schedule: ClassicAssignment, root: object, *, place: object, workers: object) -> None:
        """Validate classic lowering facts against their semantic assignment."""
        if not isinstance(schedule, Schedule):
            raise TypeError("classic materialization requires a Schedule")
        if not isinstance(root, Fold):
            raise TypeError("classic materialization requires a Fold root")
        from emmy.compiler.ir.tile.ops import Sched  # noqa: PLC0415

        context = ClassicScheduleContext(ClassicProblem(root, target=None))
        try:
            context.extend(schedule)
        except ScheduleRefused as error:
            raise ValueError(f"TileOp carries a refused classic schedule: {error}") from error
        expected_tiles = {
            site
            for site, assignment in schedule.nodes.items()
            if assignment.tile.is_tiled and isinstance(context.views[site], Reduction) and context.views[site].contraction is not None
        }
        if set(self.tiles) != expected_tiles:
            raise ValueError("classic materialization must contain exactly the tiled node sites")
        expected_stages = {edge for edge, assignment in schedule.edges.items() if not assignment.stage.is_direct}
        if set(self.stages) != expected_stages:
            raise ValueError("classic materialization must contain exactly the staged edge sites")
        placement = Sched(root, place=place)
        for site, placed in self.tiles.items():
            choice = schedule.nodes[site].tile
            expected = placement.placed(context.node(site), choice)
            if placed.choice != choice or placed != expected:
                raise ValueError(f"materialized tile at {node_id_spelling(site)} does not derive from its classic choice")
        for edge, resolved in self.stages.items():
            if edge not in schedule.edges or resolved.choice != schedule.edges[edge].stage:
                raise ValueError(f"materialized stage at {edge_site_spelling(edge)} does not derive from its classic choice")
        producer = workers.producer_warps if workers is not None else 0
        if schedule.kernel.work.producer != producer:
            raise ValueError(f"classic producer band {schedule.kernel.work.producer} disagrees with WarpSpec producer band {producer}")


@dataclass(frozen=True)
class ClassicProblem:
    """The unscheduled Fold tree and compilation target."""

    root: Fold
    target: object

    def __post_init__(self) -> None:
        if not isinstance(self.root, Fold):
            raise TypeError("classic problem root must be a Fold")


@dataclass(frozen=True)
class _ClassicSpace:
    """Immutable indexes shared by every prefix of one ``p``/``t`` product."""

    folds: tuple[Fold, ...]
    node_sites: tuple[NodeId, ...]
    edge_sites: tuple[EdgeSite, ...]
    site_by_identity: frozendict
    views: frozendict
    tile_sites: tuple[NodeId, ...]
    reduction_sites: tuple[NodeId, ...]
    stage_edges: tuple[EdgeSite, ...]
    incident_edges: frozendict
    kernel_set: frozenset
    node_sets: frozendict
    edge_sets: frozendict
    values: frozendict


@dataclass(frozen=True)
class ClassicScheduleContext(ScheduleContext[KernelSchedule, NodeSchedule, EdgeSchedule]):
    """Immutable classic ``c + p + t`` compatibility-composition state."""

    problem: ClassicProblem
    domains: ClassicDomains | None = None
    order: tuple[NodeId, ...] | None = None
    position: int = 0
    _assignment: ClassicAssignment = field(default_factory=lambda: Schedule(None, {}, {}), repr=False)
    _work: Work | None = field(default=None, repr=False)
    _axes: Mapping[str, tuple[int, int]] = field(default_factory=frozendict, repr=False)
    _fragments: Mapping[tuple[str, str], tuple] = field(default_factory=frozendict, repr=False)
    _raster_eligible: bool = field(default=False, repr=False)
    _predicate: Callable[[ClassicAssignment], bool] = field(default=_allow_schedule, repr=False, compare=False)
    _pins: Mapping[str, tuple[tuple[str, str], ...]] | None = field(default=None, repr=False, compare=False)
    _allow_f16_accumulate: bool = field(default=True, repr=False, compare=False)
    _allow_fp8: bool = field(default=True, repr=False, compare=False)
    _ignore_unsupported_global: bool = field(default=False, repr=False, compare=False)
    _allowed_works: frozenset[tuple[str, tuple[int, ...]]] | None = field(default=None, repr=False, compare=False)
    _unsupported_global: bool | None = field(default=None, repr=False, compare=False)
    _restricted_kernels: tuple[KernelSchedule, ...] | None = field(default=None, repr=False, compare=False)
    _restricted_kernel_set: frozenset[KernelSchedule] | None = field(default=None, repr=False, compare=False)
    _restricted_nodes: Mapping[NodeId, tuple[NodeSchedule, ...]] | None = field(default=None, repr=False, compare=False)
    _restricted_edges: Mapping[EdgeSite, tuple[EdgeSchedule, ...]] | None = field(default=None, repr=False, compare=False)
    _space: _ClassicSpace | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._space is None:
            object.__setattr__(self, "_space", self._build_space())
        assert self._space is not None
        order = self._space.node_sites if self.order is None else tuple(self.order)
        if len(order) != len(self._space.node_sites) or set(order) != set(self._space.node_sites):
            raise ValueError("classic composition order must contain every node site exactly once")
        if not 0 <= self.position <= len(order):
            raise ValueError("classic composition position is outside its node order")
        if not callable(self._predicate):
            raise TypeError("classic schedule restriction must be callable")
        object.__setattr__(self, "order", order)
        if not isinstance(self._assignment, Schedule):
            raise TypeError("classic context assignment must be a Schedule")
        object.__setattr__(self, "_axes", frozendict(self._axes))
        object.__setattr__(self, "_fragments", frozendict(self._fragments))
        if self._pins is not None:
            object.__setattr__(self, "_pins", frozendict({family: tuple(values) for family, values in self._pins.items()}))
            if self._unsupported_global is None:
                object.__setattr__(
                    self,
                    "_unsupported_global",
                    not self._ignore_unsupported_global
                    and any(
                        key == family and value and not self._supports_global(family, value)
                        for family, pins in self._pins.items()
                        for key, value in pins
                    ),
                )
            if self._restricted_kernels is None:
                kernels = tuple(kernel for kernel in self.kernels if self._kernel_restriction_allows(kernel))
                nodes = frozendict(
                    {
                        site: tuple(choice for choice in self.node_choices(site) if self._node_restriction_allows(site, choice))
                        for site in self.node_sites
                    }
                )
                edges = frozendict(
                    {
                        edge: tuple(choice for choice in self.edge_choices(edge) if self._edge_restriction_allows(edge, choice))
                        for edge in self.edge_sites
                    }
                )
                object.__setattr__(self, "_restricted_kernels", kernels)
                object.__setattr__(self, "_restricted_kernel_set", frozenset(kernels))
                object.__setattr__(self, "_restricted_nodes", nodes)
                object.__setattr__(self, "_restricted_edges", edges)
            if self._allowed_works is None:
                assert self._restricted_kernels is not None
                object.__setattr__(
                    self,
                    "_allowed_works",
                    frozenset((kernel.work.kind, kernel.work.units) for kernel in self._restricted_kernels),
                )
            if self.position == 0 and not self.assignment.nodes:
                self._validate_local_pin_support()

    def _build_space(self) -> _ClassicSpace:
        folds = schedule_nodes(self.problem.root)
        node_sites = tuple(range(len(folds)))
        edge_sites = schedule_edges(folds)
        views = frozendict({site: node_view(folds[site]) for site in node_sites})
        tile_sites = tuple(
            site
            for site in node_sites
            if (isinstance(views[site], Reduction) and views[site].contraction is not None)
            or (isinstance(views[site], Projection) and site == node_sites[0] and not folds[site].operands)
        )
        reduction_sites = tuple(site for site in node_sites if isinstance(views[site], Reduction))
        stage_edges = tuple(edge for edge in edge_sites if isinstance(views[edge[0]], Reduction) and views[edge[0]].contraction is not None)
        incident = frozendict({site: tuple(edge for edge in edge_sites if edge[0] == site) for site in node_sites})
        kernel_set = frozenset()
        node_sets = frozendict()
        edge_sets = frozendict()
        values = {}
        if self.domains is not None:
            expected_nodes = set(node_sites)
            expected_edges = set(edge_sites)
            if set(self.domains.nodes) != expected_nodes:
                raise ValueError("classic domains must cover every node site exactly once")
            if set(self.domains.edges) != expected_edges:
                raise ValueError("classic domains must cover every edge site exactly once")
            kernel_set = frozenset(self.domains.kernel)
            node_sets = frozendict({site: frozenset(values) for site, values in self.domains.nodes.items()})
            edge_sets = frozendict({edge: frozenset(values) for edge, values in self.domains.edges.items()})

            def node_key(family, site, sites):
                return family if len(sites) == 1 else f"{family}@{node_id_spelling(site)}"

            values["WORK"] = tuple(dict.fromkeys(choice.work.spell() for choice in self.domains.kernel))
            values["RASTER"] = tuple(dict.fromkeys(choice.raster.spell() for choice in self.domains.kernel))
            values.update(
                {
                    node_key("TILE", site, tile_sites): tuple(dict.fromkeys(choice.tile.spell() for choice in self.domains.nodes[site]))
                    for site in tile_sites
                }
            )
            values.update(
                {
                    node_key("REDUCE", site, reduction_sites): tuple(
                        dict.fromkeys(choice.reduce.spell() for choice in self.domains.nodes[site] if isinstance(choice, ReductionSchedule))
                    )
                    for site in reduction_sites
                }
            )
            stage_consumers = tuple(dict.fromkeys(edge[0] for edge in stage_edges))
            for site in stage_consumers:
                edges = tuple(edge for edge in stage_edges if edge[0] == site)
                common = set.intersection(*({choice.stage.spell() for choice in self.domains.edges[edge]} for edge in edges))
                values[node_key("STAGE", site, stage_consumers)] = tuple(
                    dict.fromkeys(choice.stage.spell() for choice in self.domains.edges[edges[0]] if choice.stage.spell() in common)
                )
        return _ClassicSpace(
            folds,
            node_sites,
            edge_sites,
            frozendict({id(node): site for site, node in enumerate(folds)}),
            views,
            tile_sites,
            reduction_sites,
            stage_edges,
            incident,
            kernel_set,
            node_sets,
            edge_sets,
            frozendict(values),
        )

    @property
    def node_sites(self) -> tuple[NodeId, ...]:
        assert self._space is not None
        return self._space.node_sites

    @property
    def edge_sites(self) -> tuple[EdgeSite, ...]:
        assert self._space is not None
        return self._space.edge_sites

    @property
    def views(self) -> Mapping[NodeId, NodeView]:
        assert self._space is not None
        return self._space.views

    @property
    def tile_sites(self) -> tuple[NodeId, ...]:
        assert self._space is not None
        return self._space.tile_sites

    @property
    def reduction_sites(self) -> tuple[NodeId, ...]:
        assert self._space is not None
        return self._space.reduction_sites

    @property
    def stage_edges(self) -> tuple[EdgeSite, ...]:
        assert self._space is not None
        return self._space.stage_edges

    @property
    def kernels(self) -> tuple[KernelSchedule, ...]:
        return self.domains.kernel if self.domains is not None else (KernelSchedule(Work(), Raster()),)

    def node_choices(self, site: NodeId) -> tuple[NodeSchedule, ...]:
        view = self.views[site]
        if self.domains is not None:
            try:
                return self.domains.nodes[site]
            except KeyError:
                raise ValueError(f"classic domains do not contain {node_id_spelling(site)}") from None
        return (ProjectionSchedule(Tile()),) if isinstance(view, Projection) else (ReductionSchedule(Tile(), Reduce()),)

    def edge_choices(self, edge: EdgeSite) -> tuple[EdgeSchedule, ...]:
        self.operand(edge)
        if self.domains is not None:
            try:
                return self.domains.edges[edge]
            except KeyError:
                raise ValueError(f"classic domains do not contain {edge_site_spelling(edge)}") from None
        return (EdgeSchedule(Stage.direct()),)

    def node(self, site: NodeId) -> Fold:
        assert self._space is not None
        if type(site) is not int or not 0 <= site < len(self._space.folds):
            raise KeyError(f"unknown node site {site!r}")
        return self._space.folds[site]

    def site(self, node: Fold) -> NodeId:
        assert self._space is not None
        try:
            return self._space.site_by_identity[id(node)]
        except KeyError:
            raise KeyError("Fold is not a node of this classic problem") from None

    def operand(self, edge: EdgeSite):
        if not _is_edge_site(edge):
            raise KeyError(f"invalid edge site {edge!r}")
        consumer, operand = edge
        try:
            return self.node(consumer).operands[operand]
        except IndexError:
            raise KeyError(f"unknown operand {operand} at {node_id_spelling(consumer)}") from None

    def producer(self, edge: EdgeSite) -> NodeId | None:
        value = self.operand(edge)
        return self.site(value) if isinstance(value, Fold) else None

    def incident_edges(self, site: NodeId) -> tuple[EdgeSite, ...]:
        assert self._space is not None
        return self._space.incident_edges[site]

    def node_key(self, family: str, site: NodeId) -> str:
        sites = {"TILE": self.tile_sites, "REDUCE": self.reduction_sites}.get(family)
        if sites is None:
            raise ValueError(f"{family} is not a classic node family")
        if site not in sites:
            raise ValueError(f"{node_id_spelling(site)} is not a {family} site")
        return family if len(sites) == 1 else f"{family}@{node_id_spelling(site)}"

    def stage_key(self, edge: EdgeSite) -> str:
        if edge not in self.stage_edges:
            raise ValueError(f"{edge_site_spelling(edge)} is not a STAGE edge")
        consumers = tuple(dict.fromkeys(candidate[0] for candidate in self.stage_edges))
        return "STAGE" if len(consumers) == 1 else f"STAGE@{node_id_spelling(edge[0])}"

    def keys(self) -> tuple[str, ...]:
        stage_consumers = tuple(dict.fromkeys(edge[0] for edge in self.stage_edges))
        return (
            "WORK",
            "RASTER",
            *(self.node_key("TILE", site) for site in self.tile_sites),
            *(self.node_key("REDUCE", site) for site in self.reduction_sites),
            *(self.stage_key(next(edge for edge in self.stage_edges if edge[0] == site)) for site in stage_consumers),
        )

    def values(self, key: str) -> tuple[str, ...]:
        if self.domains is None or key not in self.keys():
            raise ValueError(f"classic key {key!r} has no projected domain")
        assert self._space is not None
        return self._space.values[key]

    def restrict(
        self,
        pins: Mapping[str, Sequence[tuple[str, str]]],
        *,
        split_consumed: bool = False,
        allow_f16_accumulate: bool = True,
        allow_fp8: bool = True,
        validate_pins: bool = True,
    ) -> ClassicScheduleContext:
        """Return ``c + p + t`` with raw schedule parameters normalized exactly once."""
        values = {family: tuple(pins.get(family, ())) for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")}
        if split_consumed:
            values["REDUCE"] = tuple(
                (key, "/".join(part for part in value.split("/") if not part.startswith("g"))) for key, value in values["REDUCE"]
            )
        return replace(
            self,
            _pins=frozendict(values),
            _allow_f16_accumulate=allow_f16_accumulate,
            _allow_fp8=allow_fp8,
            _ignore_unsupported_global=not validate_pins,
            _allowed_works=None,
            _unsupported_global=None,
            _restricted_kernels=None,
            _restricted_kernel_set=None,
            _restricted_nodes=None,
            _restricted_edges=None,
        )

    def with_restriction(self, predicate: Callable[[ClassicAssignment], bool]) -> ClassicScheduleContext:
        """Return a context with one opaque complete-assignment restriction ``c``."""
        if not callable(predicate):
            raise TypeError("classic schedule restriction must be callable")
        return replace(self, _predicate=predicate)

    @property
    def nodes_complete(self) -> bool:
        assert self.order is not None
        return self.position == len(self.order)

    @property
    def next_site(self) -> NodeId | None:
        assert self.order is not None
        return None if self.nodes_complete else self.order[self.position]

    @property
    def assignment(self) -> ClassicAssignment:
        return self._assignment

    def extensions(self) -> Iterator[ClassicAssignment]:
        """Yield a lazy compatible frontier of one node and its incident edges."""
        if self.assignment.kernel is not None:
            return
        if self.domains is None:
            raise ValueError("classic compatibility composition requires projected domains")
        if self.nodes_complete:
            kernels = self._restricted_kernels if self._restricted_kernels is not None else self.kernels
            for kernel in kernels:
                if self._kernel_composes(kernel):
                    yield Schedule(kernel, {}, {})
            return
        assert self.next_site is not None
        site = self.next_site
        incident = self.incident_edges(site)
        nodes = self._restricted_nodes[site] if self._restricted_nodes is not None else self.node_choices(site)
        edges = tuple(
            (
                edge,
                self._restricted_edges[edge] if self._restricted_edges is not None else self.edge_choices(edge),
            )
            for edge in incident
        )
        previous_nodes = tuple(self.assignment.nodes.values()) if self._work is None else ()
        frontier = self.domains.compatible_frontier(
            site,
            nodes,
            edges,
            self._work,
            previous_nodes,
            tuple(self._axes.items()),
            tuple(self._fragments.items()),
            self._allowed_works,
        )
        for support in frontier:
            yield Schedule(None, {site: support.node}, support.edges)

    def _validate_local_pin_support(self) -> None:
        """Reject a non-direct STAGE parameter that no local node/edge support can realize."""
        assert self.domains is not None and self._restricted_nodes is not None and self._restricted_edges is not None
        for site in self.node_sites:
            stage_keys = tuple(dict.fromkeys(self.stage_key(edge) for edge in self.incident_edges(site) if edge in self.stage_edges))
            if not any(pin for key in stage_keys for pin in self._applicable_pins("STAGE", key)):
                continue
            edge_domains = tuple((edge, self._restricted_edges[edge]) for edge in self.incident_edges(site))
            if not self.domains.compatible_frontier(
                site,
                self._restricted_nodes[site],
                edge_domains,
                None,
                (),
                (),
                (),
                self._allowed_works,
            ):
                self._raise_empty_pinned_frontier(("STAGE",), site)

    def _raise_empty_pinned_frontier(self, families: tuple[str, ...], site: NodeId | None = None) -> None:
        """Keep an authoritative parameter refusal loud when no compatible pick realizes it."""
        if self._pins is None:
            return
        keys = {
            "WORK": ("WORK",),
            "RASTER": ("RASTER",),
            "TILE": (self.node_key("TILE", site),) if site in self.tile_sites else (),
            "REDUCE": (self.node_key("REDUCE", site),) if site in self.reduction_sites else (),
            "STAGE": tuple(dict.fromkeys(self.stage_key(edge) for edge in self.incident_edges(site) if edge in self.stage_edges))
            if site is not None
            else (),
        }
        for family in families:
            for key in keys[family]:
                if pins := self._applicable_pins(family, key):
                    raise ValueError(f"{family} pin {pins[-1]!r} at {key} does not resolve to a compatible schedule")

    def extend(self, pick: ClassicAssignment) -> ClassicScheduleContext:
        """Compose a frontier pick or validate and accept one complete assignment."""
        if not isinstance(pick, Schedule) or self.assignment.kernel is not None:
            self._refuse("classic extension requires an incomplete context and a Schedule pick")
        if pick.kernel is not None and (pick.nodes or pick.edges):
            return self._extend_complete(pick)
        if self.nodes_complete:
            return self._finish(pick)

        return self._extend_local(pick)

    def _extend_complete(self, pick: ClassicAssignment) -> ClassicScheduleContext:
        if any(pick.nodes.get(site) != choice for site, choice in self.assignment.nodes.items()) or any(
            pick.edges.get(edge) != choice for edge, choice in self.assignment.edges.items()
        ):
            self._refuse("complete assignment disagrees with the existing classic prefix")
        self._require_complete(pick)
        assert self.order is not None and pick.kernel is not None
        work = Work(pick.kernel.work.kind, pick.kernel.work.units)
        return replace(self, position=len(self.order), _assignment=pick, _work=work)

    def _extend_local(self, pick: ClassicAssignment) -> ClassicScheduleContext:
        site = self.next_site
        assert self._space is not None
        incident = self.incident_edges(site)
        if (
            site is None
            or self.domains is None
            or pick.kernel is not None
            or set(pick.nodes) != {site}
            or set(pick.edges) != set(incident)
            or pick.nodes[site] not in self._space.node_sets[site]
            or any(pick.edges[edge] not in self._space.edge_sets[edge] for edge in incident)
        ):
            self._refuse("pick is outside the next independent classic position", site)
        node = pick.nodes[site]
        if self._restricted_nodes is not None:
            restricted = node not in self._restricted_nodes[site] or any(
                choice not in self._restricted_edges[edge] for edge, choice in pick.edges.items()
            )
        else:
            restricted = not self._node_restriction_allows(site, node) or any(
                not self._edge_restriction_allows(edge, choice) for edge, choice in pick.edges.items()
            )
        if restricted:
            self._refuse("pick is outside the schedule restriction", site)
        support = self.domains.local_support(site, node, pick.edges)
        if support is None:
            self._refuse("pick has no local classic support", site)
        if why := self._support_refusal(site, support):
            self._refuse(why, site)
        work = support.work or self._work
        nodes = {**self.assignment.nodes, site: support.node}
        axes = {**self._axes, **{claim.name: (claim.tile, claim.units) for claim in support.axes}}
        fragments = {**self._fragments, **{(claim.role, claim.edge): claim.value for claim in support.fragments}}
        return replace(
            self,
            position=self.position + 1,
            _assignment=Schedule(None, nodes, {**self.assignment.edges, **support.edges}),
            _work=work,
            _axes=axes,
            _fragments=fragments,
            _raster_eligible=self._raster_eligible or support.raster_eligible,
        )

    def _support_refusal(self, site: NodeId, support: LocalSupport) -> str | None:
        """Return why one locally supported pick cannot extend this prefix."""
        if not self._local_restriction_allows(site, support):
            return "pick is outside the schedule restriction"
        why = self._prefix_relation_refusal(
            support,
            work=self._work,
            previous_nodes=tuple(self.assignment.nodes.values()) if self._work is None else (),
            axes=tuple(self._axes.items()),
            fragments=tuple(self._fragments.items()),
            allowed_works=self._allowed_works,
        )
        if why is not None:
            return why
        if self._allowed_works is None and not self._work_restriction_allows(support.work or self._work):
            return "pick cannot reach a kernel allowed by the schedule restriction"
        return None

    @staticmethod
    def _prefix_relation_refusal(
        support: LocalSupport,
        *,
        work: Work | None,
        previous_nodes: tuple[NodeSchedule, ...],
        axes: tuple[tuple[str, tuple[int, int]], ...],
        fragments: tuple[tuple[tuple[str, str], tuple], ...],
        allowed_works: frozenset[tuple[str, tuple[int, ...]]] | None,
    ) -> str | None:
        """Return the one work/axis/fragment refusal shared by frontier indexing and extension."""
        if work is not None and support.work is not None and support.work != work:
            return "pick requires a different worker inventory"
        resolved_work = support.work or work
        if allowed_works is not None and resolved_work is not None and (resolved_work.kind, resolved_work.units) not in allowed_works:
            return "pick cannot reach a kernel allowed by the schedule restriction"
        if work is None and resolved_work is not None:
            if not all(choice.tile.is_canonical_for(resolved_work) for choice in (*previous_nodes, support.node)):
                return "pick is not canonical for the worker inventory"
        elif not support.node.tile.is_canonical_for(resolved_work):
            return "pick is not canonical for the worker inventory"
        axis_values = dict(axes)
        for claim in support.axes:
            value = (claim.tile, claim.units)
            if axis_values.get(claim.name, value) != value:
                return "pick disagrees on physical-axis geometry"
        fragment_values = dict(fragments)
        for claim in support.fragments:
            key = (claim.role, claim.edge)
            if fragment_values.setdefault(key, claim.value) != claim.value:
                return "pick repeats a fragment endpoint inconsistently"
            other_role = "need" if claim.role == "offer" else "offer"
            other = fragment_values.get((other_role, claim.edge))
            if other is None:
                continue
            need, offer = (claim.value, other) if claim.role == "need" else (other, claim.value)
            if offer[0] == "free":
                compatible = need[0] != "step"
            else:
                compatible = (
                    need[0] in ("warp", "step")
                    and offer[0] == "warp"
                    and need[1] == offer[1]
                    and need[2] == offer[2]
                    and offer[3] == 1
                    and offer[4] == need[3]
                )
            if not compatible:
                return "pick is incompatible at a fragment seam"
        return None

    def _finish(self, pick: ClassicAssignment) -> ClassicScheduleContext:
        assert self._space is not None
        if (
            not self.nodes_complete
            or self.assignment.kernel is not None
            or not isinstance(pick.kernel, KernelSchedule)
            or pick.nodes
            or pick.edges
            or pick.kernel not in self._space.kernel_set
            or (self._restricted_kernel_set is not None and pick.kernel not in self._restricted_kernel_set)
            or not self._kernel_composes(pick.kernel)
        ):
            self._refuse("pick is incompatible with the classic kernel position")
        assignment = Schedule(pick.kernel, self.assignment.nodes, self.assignment.edges)
        self._require_kernel_prefix(assignment)
        if self._unsupported_global:
            self._refuse("global schedule pin is unsupported by every applicable site")
        if not self._predicate(assignment):
            self._refuse("schedule restriction rejects assignment")
        return replace(self, _assignment=assignment)

    def _require_kernel_prefix(self, schedule: ClassicAssignment) -> None:
        """Validate the kernel facts not already proved by local prefix composition."""
        kernel_work = Work(schedule.kernel.work.kind, schedule.kernel.work.units)
        warp_size = getattr(self.problem.target, "warp_size", 32)
        compute_threads = kernel_work.count * (warp_size if kernel_work.kind == "warp" else 1)
        producer_threads = schedule.kernel.work.producer * warp_size
        if producer_threads > compute_threads:
            self._refuse("producer band cannot outnumber the compute band")
        if compute_threads + producer_threads > getattr(self.problem.target, "max_threads_per_cta", 1024):
            self._refuse("worker inventory exceeds the target thread limit")
        if not schedule.kernel.work.producer:
            return
        for site, assignment in schedule.nodes.items():
            if not assignment.tile.is_tiled:
                continue
            if isinstance(assignment, ReductionSchedule) and assignment.reduce.needs_split:
                self._refuse("a producer band cannot accompany a cross-CTA reduction", site)
            edges = tuple(edge for edge in self.incident_edges(site) if edge in self.stage_edges)
            if not edges or any(schedule.edges[edge].stage.transport != "smem-tma" for edge in edges):
                self._refuse("a producer band requires TMA transport at every tiled consumer", site)

    def _require_complete(self, schedule: ClassicAssignment) -> None:
        """Raise :class:`ScheduleRefused` with the first complete-assignment refusal."""
        self._require_intrinsic(schedule)
        self._require_domain_membership(schedule)
        if self.domains is not None and not self._support_relation_accepts(schedule):
            raise ScheduleRefused("complete choices violate the classic compatibility relation")
        self._require_restriction(schedule)

    def _refuse(self, reason: str, site: NodeId | EdgeSite | None = None) -> None:
        if site is None:
            raise ScheduleRefused(reason)
        where = node_id_spelling(site) if type(site) is int else edge_site_spelling(site)
        raise ScheduleRefused(f"{where}: {reason}")

    def _require_intrinsic(self, schedule: ClassicAssignment) -> None:
        if not isinstance(schedule, Schedule) or not isinstance(schedule.kernel, KernelSchedule):
            self._refuse("assignment must contain a classic kernel schedule")
        if any(not isinstance(value, (ProjectionSchedule, ReductionSchedule)) for value in schedule.nodes.values()):
            self._refuse("classic node assignments must contain projection or reduction schedules")
        if any(not isinstance(value, EdgeSchedule) for value in schedule.edges.values()):
            self._refuse("classic edge assignments must contain edge schedules")
        expected_nodes = set(self.node_sites)
        if missing := expected_nodes - schedule.nodes.keys():
            self._refuse("missing node assignment", min(missing))
        if extra := schedule.nodes.keys() - expected_nodes:
            self._refuse("node assignment is outside this problem", min(extra))
        expected_edges = set(self.edge_sites)
        if missing := expected_edges - schedule.edges.keys():
            self._refuse("missing edge assignment", min(missing))
        if extra := schedule.edges.keys() - expected_edges:
            self._refuse("edge assignment is outside this problem", min(extra))

        claimed_work = None
        for site in self.node_sites:
            view = self.views[site]
            assignment = schedule.nodes[site]
            if isinstance(view, Projection) and not isinstance(assignment, ProjectionSchedule):
                self._refuse("projection site requires a projection schedule", site)
            if isinstance(view, Reduction) and not isinstance(assignment, ReductionSchedule):
                self._refuse("reduction site requires a reduction schedule", site)
            if isinstance(assignment.tile, PlacedTile):
                self._refuse("node choices cannot contain placed tile geometry", site)
            if site not in self.tile_sites and assignment.tile != Tile():
                self._refuse("this node has no tile choice", site)
            if assignment.tile.is_warp and hasattr(self.problem.target, assignment.tile.atom.target_feature):
                if not assignment.tile.atom.available_on(self.problem.target):
                    self._refuse("tile atom is unavailable on the target", site)
            coop = assignment.reduce.coop if isinstance(assignment, ReductionSchedule) else 1
            try:
                work = derive_inventory((assignment.tile,), coop=coop)
            except ValueError as error:
                self._refuse(str(error), site)
            if work is not None:
                if claimed_work is not None and claimed_work != work:
                    self._refuse("node choices require different worker inventories", site)
                claimed_work = work
            if not assignment.tile.is_canonical_for(schedule.kernel.work):
                self._refuse("node TILE is not canonical under the kernel WORK", site)
        kernel_work = Work(schedule.kernel.work.kind, schedule.kernel.work.units)
        if (claimed_work or Work()) != kernel_work:
            self._refuse("kernel WORK does not realize the node choices")
        warp_size = getattr(self.problem.target, "warp_size", 32)
        compute_threads = kernel_work.count * (warp_size if kernel_work.kind == "warp" else 1)
        producer_threads = schedule.kernel.work.producer * warp_size
        if producer_threads > compute_threads:
            self._refuse("producer band cannot outnumber the compute band")
        if compute_threads + producer_threads > getattr(self.problem.target, "max_threads_per_cta", 1024):
            self._refuse("worker inventory exceeds the target thread limit")
        if not schedule.kernel.raster.is_direct and not any(
            isinstance(self.views[site], Reduction) and self.views[site].contraction is not None and schedule.nodes[site].tile.is_tiled
            for site in self.node_sites
        ):
            self._refuse("RASTER requires a tiled contraction site")
        for edge in self.edge_sites:
            stage = schedule.edges[edge].stage
            if edge not in self.stage_edges and not stage.is_direct:
                self._refuse("this edge has no staged transport choice", edge)
            if not stage.is_direct and not schedule.nodes[edge[0]].tile.is_tiled:
                self._refuse("staged transport requires a tiled consumer", edge)
            if not stage.is_direct and hasattr(self.problem.target, "has_cp_async") and not stage.available_on(self.problem.target):
                self._refuse("transport is unavailable on the target", edge)
        for site in self.node_sites:
            if len({schedule.edges[edge].stage for edge in self.incident_edges(site)}) > 1:
                self._refuse("one contraction currently requires one transport choice across its operands", site)
        if schedule.kernel.work.producer:
            for site in self.node_sites:
                assignment = schedule.nodes[site]
                if not assignment.tile.is_tiled:
                    continue
                if isinstance(assignment, ReductionSchedule) and assignment.reduce.needs_split:
                    self._refuse("a producer band cannot accompany a cross-CTA reduction", site)
                edges = tuple(edge for edge in self.incident_edges(site) if edge in self.stage_edges)
                if not edges or any(schedule.edges[edge].stage.transport != "smem-tma" for edge in edges):
                    self._refuse("a producer band requires TMA transport at every tiled consumer", site)

    def _require_domain_membership(self, schedule: ClassicAssignment) -> None:
        if self.domains is None:
            return
        assert self._space is not None
        if schedule.kernel not in self._space.kernel_set:
            self._refuse("kernel choice is outside its independent domain")
        for site in self.node_sites:
            if schedule.nodes[site] not in self._space.node_sets[site]:
                self._refuse("node choice is outside its independent domain", site)
        for edge in self.edge_sites:
            if schedule.edges[edge] not in self._space.edge_sets[edge]:
                self._refuse("edge choice is outside its independent domain", edge)

    def _support_relation_accepts(self, schedule: ClassicAssignment) -> bool:
        start = replace(
            self,
            position=0,
            _assignment=Schedule(None, {}, {}),
            _work=None,
            _axes=frozendict(),
            _fragments=frozendict(),
            _raster_eligible=False,
            _predicate=_allow_schedule,
            _pins=None,
            _allow_f16_accumulate=True,
            _allow_fp8=True,
            _allowed_works=None,
            _unsupported_global=None,
            _restricted_kernels=None,
            _restricted_kernel_set=None,
            _restricted_nodes=None,
            _restricted_edges=None,
        )

        context = start
        try:
            while not context.nodes_complete:
                assert context.next_site is not None
                site = context.next_site
                edges = {edge: schedule.edges[edge] for edge in context.incident_edges(site)}
                context = context.extend(Schedule(None, {site: schedule.nodes[site]}, edges))
        except ScheduleRefused:
            return False
        return context._kernel_composes(schedule.kernel)

    def _kernel_composes(self, kernel: KernelSchedule) -> bool:
        work = self._work or Work()
        return kernel.work.kind == work.kind and kernel.work.units == work.units and (kernel.raster.is_direct or self._raster_eligible)

    def _supports_global(self, family: str, value: str) -> bool:
        return any(key.partition("@")[0] == family and value in self.values(key) for key in self.keys())

    def _allows_value(self, family: str, key: str, value: str) -> bool:
        return all(pin == value for pin in self._applicable_pins(family, key))

    def _applicable_pins(self, family: str, key: str) -> tuple[str, ...]:
        """Return exact pins, or global pins whose value belongs to this projected site."""
        assert self._pins is not None
        exact = tuple(pin for pin_key, pin in self._pins[family] if pin_key == key and key != family)
        if exact:
            return exact
        return tuple(pin for pin_key, pin in self._pins[family] if pin_key == family and pin in self.values(key))

    def _kernel_restriction_allows(self, kernel: KernelSchedule) -> bool:
        """Whether the kernel support can still satisfy the immutable restriction ``c``."""
        if self._pins is None:
            return True
        if not self._allows_value("WORK", "WORK", kernel.work.spell()):
            return False
        if not self._allows_value("RASTER", "RASTER", kernel.raster.spell()):
            return False
        allow_transposed = any(key == "RASTER" and value.startswith("gn") for key, value in self._pins["RASTER"])
        return kernel.raster.orient != "n" or allow_transposed

    def _work_restriction_allows(self, work: Work | None) -> bool:
        """Whether a claimed prefix inventory can still reach a kernel allowed by ``c``."""
        if work is None:
            return True
        if self._allowed_works is not None:
            return (work.kind, work.units) in self._allowed_works
        return any(
            kernel.work.kind == work.kind and kernel.work.units == work.units and self._kernel_restriction_allows(kernel)
            for kernel in self.kernels
        )

    def _local_restriction_allows(self, site: NodeId, support: LocalSupport) -> bool:
        """Whether one local support can still satisfy the immutable restriction ``c``."""
        return self._node_restriction_allows(site, support.node) and all(
            self._edge_restriction_allows(edge, choice) for edge, choice in support.edges.items()
        )

    def _node_restriction_allows(self, site: NodeId, choice: NodeSchedule) -> bool:
        """Whether one independent node value can still satisfy ``c``."""
        if self._pins is not None:
            if site in self.tile_sites and not self._allows_value("TILE", self.node_key("TILE", site), choice.tile.spell()):
                return False
            if isinstance(choice, ReductionSchedule) and not self._allows_value(
                "REDUCE", self.node_key("REDUCE", site), choice.reduce.spell()
            ):
                return False
        if choice.tile.is_warp:
            atom = choice.tile.atom
            pinned = self._tile_is_pinned(site, choice.tile)
            if atom.operand_dtype("a").nbytes == 1 and not self._allow_fp8 and not pinned:
                return False
            if atom.operand_dtype("c").nbytes == 2 and not self._allow_f16_accumulate and not pinned:
                return False
        return True

    def _tile_is_pinned(self, site: NodeId, tile: Tile) -> bool:
        """Whether ``c`` explicitly selects this TILE value at ``site``.

        Precision controls restrict unpinned enumeration policy; an authored TILE still selects a
        legal independent-domain value. Keeping that exception here makes pinned full schedules
        and lazy extension use the same restriction relation.
        """
        if self._pins is None or site not in self.tile_sites:
            return False
        key = self.node_key("TILE", site)
        return bool(self._applicable_pins("TILE", key)) and self._allows_value("TILE", key, tile.spell())

    def _edge_restriction_allows(self, edge: EdgeSite, choice: EdgeSchedule) -> bool:
        """Whether one independent edge value can still satisfy ``c``."""
        return self._pins is None or edge not in self.stage_edges or self._allows_value("STAGE", self.stage_key(edge), choice.stage.spell())

    def _require_restriction(self, schedule: ClassicAssignment) -> None:
        if not self._predicate(schedule):
            self._refuse("schedule restriction rejects assignment")
        if self._pins is None:
            return
        if self._unsupported_global:
            self._refuse("global schedule pin is unsupported by every applicable site")
        if not self._allows_value("WORK", "WORK", schedule.kernel.work.spell()):
            self._refuse("WORK is outside the schedule restriction")
        if not self._allows_value("RASTER", "RASTER", schedule.kernel.raster.spell()):
            self._refuse("RASTER is outside the schedule restriction")
        allow_transposed = any(key == "RASTER" and value.startswith("gn") for key, value in self._pins["RASTER"])
        if schedule.kernel.raster.orient == "n" and not allow_transposed:
            self._refuse("transposed RASTER is outside the schedule restriction")
        for site, choice in schedule.nodes.items():
            if site in self.tile_sites and not self._allows_value("TILE", self.node_key("TILE", site), choice.tile.spell()):
                self._refuse("TILE is outside the schedule restriction", site)
            if isinstance(choice, ReductionSchedule) and not self._allows_value(
                "REDUCE", self.node_key("REDUCE", site), choice.reduce.spell()
            ):
                self._refuse("REDUCE is outside the schedule restriction", site)
            if choice.tile.is_warp:
                atom = choice.tile.atom
                pinned = self._tile_is_pinned(site, choice.tile)
                if atom.operand_dtype("a").nbytes == 1 and not self._allow_fp8 and not pinned:
                    self._refuse("FP8 TILE is outside the precision restriction", site)
                if atom.operand_dtype("c").nbytes == 2 and not self._allow_f16_accumulate and not pinned:
                    self._refuse("f16-accumulate TILE is outside the precision restriction", site)
        for edge, choice in schedule.edges.items():
            if edge in self.stage_edges and not self._allows_value("STAGE", self.stage_key(edge), choice.stage.spell()):
                self._refuse("STAGE is outside the schedule restriction", edge)

    def node_assignment(self, site: NodeId) -> NodeSchedule:
        return self.assignment.nodes[site]

    def edge_assignment(self, edge: EdgeSite) -> EdgeSchedule:
        return self.assignment.edges[edge]

    @property
    def work(self) -> Work | None:
        return self._work


def classic_cartesian_assignments(
    context: ClassicScheduleContext,
) -> Iterator[tuple[ClassicAssignment, bool]]:
    """Enumerate the literal classic kernel × node × edge product and its acceptance bit."""
    node_domains = tuple(context.node_choices(site) for site in context.node_sites)
    edge_domains = tuple(context.edge_choices(site) for site in context.edge_sites)
    for kernel, node_values, edge_values in product(context.kernels, product(*node_domains), product(*edge_domains)):
        assignment = Schedule(
            kernel,
            dict(zip(context.node_sites, node_values, strict=True)),
            dict(zip(context.edge_sites, edge_values, strict=True)),
        )
        try:
            context.extend(assignment)
        except (ScheduleRefused, TypeError):
            accepted = False
        else:
            accepted = True
        yield assignment, accepted


def enumerate_classic_reference(context: ClassicScheduleContext) -> Iterator[ClassicAssignment]:
    """Literal Algorithm 1 oracle for a classic context's independent domains."""
    for assignment, accepted in classic_cartesian_assignments(context):
        if accepted:
            yield assignment


class ClassicScheduleCodec:
    """Strict wire boundary for complete classic schedules.

    Kernel families are bare. A node family is bare when it has one applicable site and carries a
    :class:`NodeId` suffix only when the family is ambiguous. STAGE is one transport decision per
    consumer node and follows the same rule. Decoding accepts no aliases, missing direct values,
    or unknown fields.
    """

    def __init__(self, context: ClassicScheduleContext) -> None:
        if not isinstance(context, ClassicScheduleContext):
            raise TypeError("classic codec requires a ClassicScheduleContext")
        self.context = context
        self._key_order = context.keys()
        self._keys = frozenset(self._key_order)

    def encode(self, schedule: ClassicAssignment) -> dict[str, str]:
        """Encode one accepted typed schedule in canonical scope order."""
        accepted = self.context.extend(schedule).assignment
        return self._encode_accepted(accepted)

    def _encode_accepted(self, schedule: ClassicAssignment) -> dict[str, str]:
        """Encode a schedule whose compatibility was already checked at this boundary."""
        row = {
            "WORK": schedule.kernel.work.spell(),
            "RASTER": schedule.kernel.raster.spell(),
        }
        for site in self.context.tile_sites:
            row[self.context.node_key("TILE", site)] = schedule.nodes[site].tile.spell()
        for site in self.context.reduction_sites:
            assignment = schedule.nodes[site]
            assert isinstance(assignment, ReductionSchedule)
            row[self.context.node_key("REDUCE", site)] = assignment.reduce.spell()
        stage_consumers = tuple(dict.fromkeys(edge[0] for edge in self.context.stage_edges))
        for site in stage_consumers:
            edges = tuple(edge for edge in self.context.stage_edges if edge[0] == site)
            stages = {schedule.edges[edge].stage for edge in edges}
            if len(stages) != 1:
                raise ValueError(f"{node_id_spelling(site)}: one STAGE value must cover every operand edge")
            row[self.context.stage_key(edges[0])] = stages.pop().spell()
        return row

    def decode(self, row: Mapping[str, str]) -> ClassicAssignment:
        """Decode one complete canonical row and reject every other key set or assignment."""
        self._check_keys(row)

        work = Work.parse(row["WORK"])
        nodes: dict[NodeId, NodeSchedule] = {}
        for site in self.context.node_sites:
            view = self.context.views[site]
            reduce = None
            if isinstance(view, Reduction):
                reduce = Reduce.parse(row[self.context.node_key("REDUCE", site)], work)
            tile = (
                resolve_site_tile(
                    row[self.context.node_key("TILE", site)],
                    work,
                    reduce.coop if reduce is not None else 1,
                )
                if site in self.context.tile_sites
                else Tile()
            )
            nodes[site] = ProjectionSchedule(tile) if reduce is None else ReductionSchedule(tile, reduce)
        schedule = Schedule(
            KernelSchedule(work, Raster.parse(row["RASTER"])),
            nodes,
            {
                edge: EdgeSchedule(Stage.parse(row[self.context.stage_key(edge)]))
                if edge in self.context.stage_edges
                else EdgeSchedule(Stage.direct())
                for edge in self.context.edge_sites
            },
        )
        return self._validate_row(schedule, row)

    def _validate_row(self, schedule: ClassicAssignment, row: Mapping[str, str]) -> ClassicAssignment:
        """Validate a parsed assignment and its claimed canonical row exactly once."""
        accepted = self.context.extend(schedule).assignment
        canonical = self._encode_accepted(accepted)
        if dict(row) != canonical:
            raise ValueError("classic schedule row is not its typed schedule's canonical encoding")
        return accepted

    def _check_keys(self, row: Mapping[str, str]) -> None:
        """Require the codec's exact key set before parsing or validating values."""
        expected = self._keys
        actual = set(row)
        if missing := expected - actual:
            raise ValueError(f"classic schedule row is missing {', '.join(sorted(missing))}")
        if extra := actual - expected:
            raise ValueError(f"classic schedule row has unknown keys {', '.join(sorted(extra))}")

    def keys(self) -> tuple[str, ...]:
        """Return accepted keys in canonical encoding order."""
        return self._key_order

    def values(self, key: str) -> tuple[str, ...]:
        """Return one canonical key's values in independent-domain order."""
        return self.context.values(key)

    def node_key(self, family: str, site: NodeId) -> str:
        """Return the canonical key for one TILE or REDUCE site."""
        return self.context.node_key(family, site)

    def stage_key(self, edge: EdgeSite) -> str:
        """Return the canonical STAGE key for one transport edge."""
        return self.context.stage_key(edge)


__all__ = [
    "AxisAgreement",
    "CLASSIC_EDGE_FAMILIES",
    "CLASSIC_FAMILIES",
    "CLASSIC_NODE_FAMILIES",
    "ClassicAssignment",
    "ClassicProblem",
    "ClassicMaterialization",
    "ClassicDomains",
    "ClassicScheduleCodec",
    "ClassicScheduleContext",
    "EdgeSchedule",
    "classic_cartesian_assignments",
    "edge_site_spelling",
    "enumerate_classic_reference",
    "KernelSchedule",
    "FragmentAgreement",
    "LocalSupport",
    "NodeSchedule",
    "ProjectionSchedule",
    "ReductionSchedule",
    "node_id_spelling",
    "parse_edge_site",
    "parse_node_id",
]
