"""The semantic model for the classic grid/CTA/warp/thread/register schedule.

``ClassicDomains`` defines the independent product and ``ClassicScheduleContext`` is its
compatibility authority over one unscheduled ``TileOp`` and target. ``ClassicScheduleCodec`` and
``ClassicMaterialization`` are the wire and lowering boundaries for accepted assignments. Search
state and pipeline Forks do not belong here.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace

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
from .views import ContractionFacts, EdgeSite, NodeId

CLASSIC_FAMILIES = ("TILE", "REDUCE", "STAGE")


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


def classic_node_key(sites, family: str, site: NodeId) -> str:
    """Return the canonical key for one node-scoped classic family: bare when the family has one
    applicable site on this kernel, else ``FAMILY@<route>`` — the site's route from the root in the
    tree-path grammar (``TILE@map.1/twist.1/inner``), the spelling placement keys use too."""
    family_sites = sites.family_sites.get(family)
    if family_sites is None:
        raise ValueError(f"{family} is not a classic node family")
    if site not in family_sites:
        raise ValueError(f"{sites.sites[site].path} is not a {family} site")
    return family if len(family_sites) == 1 else f"{family}@{sites.sites[site].path}"


def classic_stage_key(sites, edge: EdgeSite) -> str:
    """Return the canonical key for one staged consumer: bare for one consumer, else its route."""
    if edge not in sites.stage_edges:
        raise ValueError(f"{edge_site_spelling(edge)} is not a STAGE edge")
    consumers = tuple(dict.fromkeys(candidate[0] for candidate in sites.stage_edges))
    return "STAGE" if len(consumers) == 1 else f"STAGE@{sites.sites[edge[0]].path}"


@dataclass(frozen=True)
class _AxisAgreement:
    """One physical-axis geometry claim carried by a local schedule offer."""

    name: str
    tile: int
    units: int


@dataclass(frozen=True)
class _FragmentAgreement:
    """One producer or consumer claim at a fragment seam."""

    role: str
    edge: str
    value: tuple

    def __post_init__(self) -> None:
        if self.role not in ("need", "offer"):
            raise ValueError(f"fragment agreement role must be need or offer, got {self.role!r}")


@dataclass(frozen=True)
class _LocalSupport:
    """Static support for one node choice and its incident edge choices.

    The public domains are projections of these records.  A support record is not a schedule:
    placed geometry and fragment facts remain derived compatibility evidence and never enter a
    :class:`Schedule` value.
    """

    node: NodeSchedule
    edges: Mapping[EdgeSite, EdgeSchedule]
    work: Work | None = None
    axes: tuple[_AxisAgreement, ...] = ()
    fragments: tuple[_FragmentAgreement, ...] = ()
    raster_eligible: bool = False
    producer_eligible: bool = True

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


def _target_memo(tile_op, target, slot: str) -> dict:
    """The named memo of one kernel's ``p + t`` derivations, riding the tile it derives from.

    Every candidate schedule over one tile shares these tables, and the composition context is
    replaced at each step, so the tile owns them. The target is retained beside its table so the
    id keying it cannot be recycled while the table lives.
    """
    table = instance_memo(tile_op, slot)
    if id(target) not in table:
        table[id(target)] = (target, {})
    return table[id(target)][1]


def _computed_edge(node: Fold) -> bool:
    """Whether any operand is a computed cone rather than a gmem read.

    ``as_slab``, not ``axis is None``: a slab ITERATES (its coordinates are its own axes) and
    simply does not reduce, so the old test — written when a materialized edge was a bare ``Load``
    and only a cone was a Fold — now calls every operand computed.
    """
    return any(edge.as_slab() is None for edge in node.operands)


def _needs_fill(tile_op, node: Fold, plan: Tile) -> bool:
    from . import staging  # noqa: PLC0415

    return plan.is_warp and (_computed_edge(node) or (len(node.operands) - 1) > 1 or staging.converting_a(node, plan.atom, tile_op.inputs))


def _kstep_refusal(k_axis, plan: Tile) -> str | None:
    if not (plan.is_warp and plan.atom.operand_dtype("a").nbytes == 1):
        return None
    if not k_axis.extent.is_static:
        return f"atom {plan.atom.name}: fp8 fragment loads require a static K"
    step = plan.atom.atom_k * plan.bk
    extent = k_axis.extent.as_static()
    return None if extent % step == 0 else f"warp TILE K-step {step} does not divide the static contraction K={extent}"


def _plan_node_refusal(tile_op, node: Fold, plan: Tile, placed: PlacedTile, facts: ContractionFacts) -> str | None:
    from . import staging  # noqa: PLC0415

    refusal = _kstep_refusal(facts.k_axis, plan)
    if refusal is not None or not _needs_fill(tile_op, node, plan):
        return refusal
    converting = staging.converting_a(node, plan.atom, tile_op.inputs)
    return staging.computed_operand_cover(node, placed, converting=converting, k_axis=facts.k_axis) or staging.computed_operand_copy_dtype(
        node,
        placed,
        tile_op.inputs,
        converting=converting,
    )


def _resolve_stage(
    tile_op,
    target,
    node: Fold,
    plan: Tile,
    placed: PlacedTile,
    choice: Stage,
    facts: ContractionFacts,
) -> ResolvedStage | None:
    from . import staging  # noqa: PLC0415

    packed = tile_op.packed_reading(node)
    packed_copy = packed[0] is not None and choice.transport in ("smem-async", "smem-tma")
    if _needs_fill(tile_op, node, plan) and not packed_copy:
        return staging.resolve_fill_stage(
            node,
            placed,
            target.max_dynamic_smem,
            choice.depth,
            inputs=tile_op.inputs,
            seam=facts.seam,
            k_axis=facts.k_axis,
            producer=facts.producer,
            producer_k=tile_op.axis_of(facts.producer.axis) if facts.producer is not None else None,
            axes=tile_op.axes,
        )
    if plan.is_warp:
        return staging.resolve_warp_stage(
            node,
            placed,
            choice,
            target.max_dynamic_smem,
            tile_op.inputs,
            readings=packed,
            k_axis=facts.k_axis,
        )
    return staging.resolve_scalar_stage(node, placed, choice, tile_op.inputs, target.max_dynamic_smem, facts.k_axis)


def _fragment_agreements(
    site: NodeId,
    plan: Tile,
    placed: PlacedTile,
    stage: ResolvedStage | None,
    facts: ContractionFacts,
    producer_sites: frozenset[NodeId],
) -> tuple[_FragmentAgreement, ...]:
    out = []
    if site in producer_sites:
        if not plan.is_tiled:
            offer = ("free",)
        elif plan.is_warp:
            offer = ("warp", plan.atom.shape, plan.atom.fragment_layout, placed.n.units, placed.n.tile)
        else:
            offer = ("scalar",)
        out.append(_FragmentAgreement("offer", node_id_spelling(site), offer))
    if facts.need is not None:
        if plan.is_warp and stage is not None and stage.transport == "smem":
            need = ("step" if facts.need_step else "warp", plan.atom.shape, plan.atom.fragment_layout, stage.bk_elems)
        else:
            need = ("free",)
        out.append(_FragmentAgreement("need", node_id_spelling(facts.need), need))
    return tuple(out)


def _fragment_registers(atom, role: str) -> int:
    explicit = atom.fragment_nregs(role)
    if explicit is not None:
        return explicit
    m, n, k = atom.ptx_shape
    dtype = atom.operand_dtype(role)
    if role == "a":
        return m * k * dtype.nbytes // 128
    if role == "b":
        return n * k * dtype.nbytes // 128
    return m * n // (64 if dtype.nbytes == 2 else 32)


def _paired_budget_refusal(node: Fold, producer: Fold | None, placed: PlacedTile, stage: ResolvedStage | None) -> str | None:
    if not (placed.is_warp and stage is not None and producer is not None):
        return None
    from .catalog import MAX_REGISTERS_PER_CTA, MAX_REGISTERS_PER_THREAD  # noqa: PLC0415

    atom = placed.atom
    if stage.bk_elems % atom.atom_n:
        return None
    a_regs = _fragment_registers(atom, "a")
    b_regs = _fragment_registers(atom, "b")
    c_regs = _fragment_registers(atom, "c")
    if atom.operand_dtype("c").nbytes == 2:
        c_regs += atom.atom_m * atom.atom_n // 32
    depth = max(1, stage.reg_depth)
    channels = len(node.operands) - 1
    consumer_c = channels * placed.reg_m * placed.reg_n * c_regs
    consumer = placed.reg_m * depth * a_regs + channels * (placed.reg_n * depth * b_regs + placed.reg_m * placed.reg_n * c_regs)
    producer_n = stage.bk_elems // atom.atom_n
    producer_regs = placed.reg_m * a_regs + (len(producer.operands) - 1) * (producer_n * b_regs + placed.reg_m * producer_n * c_regs)
    required = max(consumer, consumer_c + producer_regs)
    available = min(MAX_REGISTERS_PER_THREAD, MAX_REGISTERS_PER_CTA // placed.block_threads)
    if required <= available:
        return None
    return (
        f"paired contractions require at least {required} live fragment registers/thread, over the "
        f"{available}-register envelope at {placed.block_threads} threads/CTA"
    )


@dataclass(frozen=True)
class ClassicDomains:
    """The literal independent kernel, node, and edge factors of Algorithm 1."""

    kernel: tuple[KernelSchedule, ...]
    nodes: Mapping[NodeId, tuple[NodeSchedule, ...]]
    edges: Mapping[EdgeSite, tuple[EdgeSchedule, ...]]

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
        object.__setattr__(self, "nodes", frozendict({site: tuple(choices) for site, choices in self.nodes.items()}))
        object.__setattr__(self, "edges", frozendict({edge: tuple(choices) for edge, choices in self.edges.items()}))

    def __getstate__(self):
        """Pickle declared domains, never derived membership indexes."""
        return {name: self.__dict__[name] for name in self.__dataclass_fields__ if name in self.__dict__}

    @property
    def product_size(self) -> int:
        """Number of assignments in the unfiltered Cartesian product."""
        size = len(self.kernel)
        for choices in (*self.nodes.values(), *self.edges.values()):
            size *= len(choices)
        return size

    @property
    def kernel_set(self) -> frozenset[KernelSchedule]:
        """Indexed kernel membership for compatibility checks."""
        memo = instance_memo(self, "_memo_membership")
        if "kernel" not in memo:
            memo["kernel"] = frozenset(self.kernel)
        return memo["kernel"]

    def node_set(self, site: NodeId) -> frozenset[NodeSchedule]:
        """Indexed node-domain membership for one site."""
        memo = instance_memo(self, "_memo_membership")
        nodes = memo.setdefault("nodes", {})
        if site not in nodes:
            nodes[site] = frozenset(self.nodes[site])
        return nodes[site]

    def edge_set(self, edge: EdgeSite) -> frozenset[EdgeSchedule]:
        """Indexed edge-domain membership for one site."""
        memo = instance_memo(self, "_memo_membership")
        edges = memo.setdefault("edges", {})
        if edge not in edges:
            edges[edge] = frozenset(self.edges[edge])
        return edges[edge]


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

    def validate(self, schedule: ClassicAssignment, source: object, *, place: object, workers: object) -> None:
        """Validate classic lowering facts against their semantic assignment."""
        if not isinstance(schedule, Schedule):
            raise TypeError("classic materialization requires a Schedule")
        from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415

        if not isinstance(source, TileOp):
            raise TypeError("classic materialization requires a TileOp")
        from emmy.compiler.ir.tile.ops import Sched  # noqa: PLC0415

        context = ClassicScheduleContext(source)
        try:
            context.extend(schedule)
        except ScheduleRefused as error:
            raise ValueError(f"TileOp carries a refused classic schedule: {error}") from error
        source_tile = context.tile_op
        expected_tiles = {
            site
            for site, assignment in schedule.nodes.items()
            if assignment.tile.is_tiled and source_tile.views[site].as_contraction() is not None
        }
        if set(self.tiles) != expected_tiles:
            raise ValueError("classic materialization must contain exactly the tiled node sites")
        expected_stages = {edge for edge, assignment in schedule.edges.items() if not assignment.stage.is_direct}
        if set(self.stages) != expected_stages:
            raise ValueError("classic materialization must contain exactly the staged edge sites")
        placement = Sched(source, place=place)
        for site, placed in self.tiles.items():
            choice = schedule.nodes[site].tile
            expected = placement.placed(source_tile.sites[site].node, choice)
            if placed.choice != choice or placed != expected:
                raise ValueError(f"materialized tile at {node_id_spelling(site)} does not derive from its classic choice")
        for edge, resolved in self.stages.items():
            if edge not in schedule.edges or resolved.choice != schedule.edges[edge].stage:
                raise ValueError(f"materialized stage at {edge_site_spelling(edge)} does not derive from its classic choice")
        producer = workers.producer_warps if workers is not None else 0
        if schedule.kernel.work.producer != producer:
            raise ValueError(f"classic producer band {schedule.kernel.work.producer} disagrees with WarpSpec producer band {producer}")


@dataclass(frozen=True)
class ClassicScheduleContext(ScheduleContext[KernelSchedule, NodeSchedule, EdgeSchedule]):
    """Immutable classic ``c + p + t`` compatibility-composition state.

    The problem ``p`` is the unscheduled ``tile_op`` — its Fold root indexes every site through
    its own site index, and its typed inputs answer every operand-shape question — composed
    against the target ``t``. Derivations shared by every candidate ride memo tables on the tile.
    """

    tile_op: object
    target: object = None
    domains: ClassicDomains | None = None
    order: tuple[NodeId, ...] | None = None
    position: int = 0
    _assignment: ClassicAssignment = field(default_factory=lambda: Schedule(None, {}, {}), repr=False)
    _work: Work | None = field(default=None, repr=False)
    _axes: Mapping[str, tuple[int, int]] = field(default_factory=frozendict, repr=False)
    _fragments: Mapping[tuple[str, str], tuple] = field(default_factory=frozendict, repr=False)
    _raster_eligible: bool = field(default=False, repr=False)
    _producer_eligible: bool = field(default=True, repr=False)
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
    #: Membership indexes over the two tuples above. The tuples keep enumeration ORDER, which
    #: decides every tie; these answer ``in``, which a composition step asks once per site and
    #: once per incident edge. Scanning the tuples instead spent 29% of one SDPA_L schedule walk
    #: inside the choices' generated ``__eq__``. ``_restricted_kernel_set`` indexes the kernels
    #: the same way.
    _restricted_node_sets: Mapping[NodeId, frozenset[NodeSchedule]] | None = field(default=None, repr=False, compare=False)
    _restricted_edge_sets: Mapping[EdgeSite, frozenset[EdgeSchedule]] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(getattr(self.tile_op, "op", None), Fold):
            raise TypeError("classic composition requires a TileOp owning a Fold root")
        order = self.tile_op.node_sites if self.order is None else tuple(self.order)
        if len(order) != len(self.tile_op.node_sites) or set(order) != set(self.tile_op.node_sites):
            raise ValueError("classic composition order must contain every node site exactly once")
        if self.domains is not None:
            if set(self.domains.nodes) != set(self.tile_op.node_sites):
                raise ValueError("classic domains must cover every node site exactly once")
            if set(self.domains.edges) != set(self.tile_op.edge_sites):
                raise ValueError("classic domains must cover every edge site exactly once")
        if not 0 <= self.position <= len(order):
            raise ValueError("classic composition position is outside its node order")
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
                        for site in self.tile_op.node_sites
                    }
                )
                edges = frozendict(
                    {
                        edge: tuple(choice for choice in self.edge_choices(edge) if self._edge_restriction_allows(edge, choice))
                        for edge in self.tile_op.edge_sites
                    }
                )
                object.__setattr__(self, "_restricted_kernels", kernels)
                object.__setattr__(self, "_restricted_kernel_set", frozenset(kernels))
                object.__setattr__(self, "_restricted_nodes", nodes)
                object.__setattr__(self, "_restricted_edges", edges)
                object.__setattr__(self, "_restricted_node_sets", frozendict({site: frozenset(c) for site, c in nodes.items()}))
                object.__setattr__(self, "_restricted_edge_sets", frozendict({edge: frozenset(c) for edge, c in edges.items()}))
            if self._allowed_works is None:
                assert self._restricted_kernels is not None
                object.__setattr__(
                    self,
                    "_allowed_works",
                    frozenset((kernel.work.kind, kernel.work.units) for kernel in self._restricted_kernels),
                )
            if self.position == 0 and not self.assignment.nodes:
                self._validate_stage_restriction()

    @property
    def kernels(self) -> tuple[KernelSchedule, ...]:
        return self.domains.kernel if self.domains is not None else (KernelSchedule(Work(), Raster()),)

    def node_choices(self, site: NodeId) -> tuple[NodeSchedule, ...]:
        view = self.tile_op.views[site]
        if self.domains is not None:
            try:
                return self.domains.nodes[site]
            except KeyError:
                raise ValueError(f"classic domains do not contain {node_id_spelling(site)}") from None
        return (ProjectionSchedule(Tile()),) if view.axis is None else (ReductionSchedule(Tile(), Reduce()),)

    def edge_choices(self, edge: EdgeSite) -> tuple[EdgeSchedule, ...]:
        self.operand(edge)
        if self.domains is not None:
            try:
                return self.domains.edges[edge]
            except KeyError:
                raise ValueError(f"classic domains do not contain {edge_site_spelling(edge)}") from None
        return (EdgeSchedule(Stage.direct()),)

    def node(self, site: NodeId) -> Fold:
        if type(site) is not int or not 0 <= site < len(self.tile_op.sites):
            raise KeyError(f"unknown node site {site!r}")
        return self.tile_op.sites[site].node

    def site(self, node: Fold) -> NodeId:
        return self.tile_op.node_id(node)

    def operand(self, edge: EdgeSite):
        if not _is_edge_site(edge):
            raise KeyError(f"invalid edge site {edge!r}")
        consumer, operand = edge
        try:
            return self.node(consumer).operands[operand]
        except (TypeError, IndexError):
            raise KeyError(f"unknown classic edge site {edge!r}") from None

    def producer(self, edge: EdgeSite) -> NodeId | None:
        value = self.operand(edge)
        return self.tile_op.node_id(value) if isinstance(value, Fold) else None

    def incident_edges(self, site: NodeId) -> tuple[EdgeSite, ...]:
        try:
            return self.tile_op.incident_edges[site]
        except KeyError:
            raise KeyError(f"unknown node site {site!r}") from None

    def node_key(self, family: str, site: NodeId) -> str:
        return classic_node_key(self.tile_op, family, site)

    def stage_key(self, edge: EdgeSite) -> str:
        return classic_stage_key(self.tile_op, edge)

    def keys(self) -> tuple[str, ...]:
        stage_consumers = tuple(dict.fromkeys(edge[0] for edge in self.tile_op.stage_edges))
        return (
            "WORK",
            "RASTER",
            *(self.node_key("TILE", site) for site in self.tile_op.family_sites["TILE"]),
            *(self.node_key("REDUCE", site) for site in self.tile_op.family_sites["REDUCE"]),
            *(self.stage_key(next(edge for edge in self.tile_op.stage_edges if edge[0] == site)) for site in stage_consumers),
        )

    def values(self, key: str) -> tuple[str, ...]:
        if self.domains is None or key not in self.keys():
            raise ValueError(f"classic key {key!r} has no projected domain")
        if key == "WORK":
            return tuple(dict.fromkeys(choice.work.spell() for choice in self.domains.kernel))
        if key == "RASTER":
            return tuple(dict.fromkeys(choice.raster.spell() for choice in self.domains.kernel))
        for site in self.tile_op.family_sites["TILE"]:
            if key == self.node_key("TILE", site):
                return tuple(dict.fromkeys(choice.tile.spell() for choice in self.domains.nodes[site]))
        for site in self.tile_op.family_sites["REDUCE"]:
            if key == self.node_key("REDUCE", site):
                return tuple(
                    dict.fromkeys(choice.reduce.spell() for choice in self.domains.nodes[site] if isinstance(choice, ReductionSchedule))
                )
        edges = tuple(edge for edge in self.tile_op.stage_edges if key == self.stage_key(edge))
        common = set.intersection(*({choice.stage.spell() for choice in self.domains.edges[edge]} for edge in edges))
        return tuple(dict.fromkeys(choice.stage.spell() for choice in self.domains.edges[edges[0]] if choice.stage.spell() in common))

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
            _restricted_node_sets=None,
            _restricted_edge_sets=None,
        )

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
        frontier = self._compatible_frontier(site, nodes, edges)
        for support in frontier:
            yield Schedule(None, {site: support.node}, support.edges)

    def _local_frontier(
        self,
        site: NodeId,
        nodes: tuple[NodeSchedule, ...],
        edge_domains: tuple[tuple[EdgeSite, tuple[EdgeSchedule, ...]], ...],
    ) -> tuple[_LocalSupport, ...]:
        """Derive one granular node-plus-incident-edge frontier lazily."""
        cache = _target_memo(self.tile_op, self.target, "_memo_local_frontier")
        key = (site, id(nodes), tuple((edge, id(choices)) for edge, choices in edge_domains))
        if key in cache:
            return cache[key]
        if edge_domains:
            common = set(edge_domains[0][1])
            for _edge, choices in edge_domains[1:]:
                common.intersection_update(choices)
            edge_picks = tuple(
                frozendict({edge: choice for edge, _choices in edge_domains}) for choice in edge_domains[0][1] if choice in common
            )
        else:
            edge_picks = (frozendict(),)
        result = tuple(support for node in nodes for edges in edge_picks if (support := self._local_support(site, node, edges)) is not None)
        cache[key] = result
        return result

    def _compatible_frontier(
        self,
        site: NodeId,
        nodes: tuple[NodeSchedule, ...],
        edge_domains: tuple[tuple[EdgeSite, tuple[EdgeSchedule, ...]], ...],
    ) -> tuple[_LocalSupport, ...]:
        """Filter one local frontier through this exact immutable prefix."""
        frontier = self._local_frontier(site, nodes, edge_domains)
        if self._work is not None:
            indexes = _target_memo(self.tile_op, self.target, "_memo_frontier_by_work")
            key = (site, id(nodes), tuple((edge, id(choices)) for edge, choices in edge_domains))
            if key not in indexes:
                by_work = {}
                for support in frontier:
                    by_work.setdefault(support.work, []).append(support)
                indexes[key] = {work: tuple(supports) for work, supports in by_work.items()}
            frontier = (*indexes[key].get(None, ()), *indexes[key].get(self._work, ()))
        return tuple(support for support in frontier if self._support_refusal(site, support) is None)

    def _validate_stage_restriction(self) -> None:
        """Keep an addressed non-direct STAGE restriction authoritative and diagnostic."""
        assert self._pins is not None and self._restricted_nodes is not None and self._restricted_edges is not None
        if not self.tile_op.stage_edges:
            return
        from .staging import stage_target  # noqa: PLC0415

        for key, spelling in self._pins["STAGE"]:
            if not spelling or (key != "STAGE" and key not in self.keys()):
                continue
            choice = Stage.parse(spelling)
            if why := stage_target(choice, self.target):
                raise ValueError(why)
            if key == "STAGE" and not self._supports_global("STAGE", spelling):
                raise ValueError(f"STAGE pin {spelling!r} does not resolve for this contraction")
        for site in self.tile_op.node_sites:
            keys = tuple(dict.fromkeys(self.stage_key(edge) for edge in self.incident_edges(site) if edge in self.tile_op.stage_edges))
            pins = tuple(pin for key in keys for pin in self._applicable_pins("STAGE", key) if pin)
            if not pins:
                continue
            edge_domains = tuple((edge, self._restricted_edges[edge]) for edge in self.incident_edges(site))
            if not self._compatible_frontier(site, self._restricted_nodes[site], edge_domains):
                raise ValueError(f"STAGE pin {pins[-1]!r} does not resolve for this contraction")

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
        self._require_complete_shape(pick)
        context = self._restart()
        assert context.order is not None and pick.kernel is not None
        for site in context.order:
            context = context._extend_local(
                Schedule(None, {site: pick.nodes[site]}, {edge: pick.edges[edge] for edge in context.incident_edges(site)})
            )
        return context._finish(Schedule(pick.kernel, {}, {}))

    def _restart(self) -> ClassicScheduleContext:
        """Return the empty prefix carrying this context's unchanged ``c + p + t``."""
        return replace(
            self,
            position=0,
            _assignment=Schedule(None, {}, {}),
            _work=None,
            _axes=frozendict(),
            _fragments=frozendict(),
            _raster_eligible=False,
            _producer_eligible=True,
        )

    def _extend_local(self, pick: ClassicAssignment) -> ClassicScheduleContext:
        site = self.next_site
        incident = self.incident_edges(site)
        if site is None or pick.kernel is not None or set(pick.nodes) != {site} or set(pick.edges) != set(incident):
            self._refuse("pick is outside the next independent classic position", site)
        node = pick.nodes[site]
        if not isinstance(node, (ProjectionSchedule, ReductionSchedule)) or any(
            not isinstance(choice, EdgeSchedule) for choice in pick.edges.values()
        ):
            self._refuse("pick contains a value from another schedule family", site)
        view = self.tile_op.views[site]
        if view.axis is None and not isinstance(node, ProjectionSchedule):
            self._refuse("projection site requires a projection schedule", site)
        if view.axis is not None and not isinstance(node, ReductionSchedule):
            self._refuse("reduction site requires a reduction schedule", site)
        if isinstance(node.tile, PlacedTile):
            self._refuse("node choices cannot contain placed tile geometry", site)
        if self.domains is not None and (
            node not in self.domains.node_set(site) or any(pick.edges[edge] not in self.domains.edge_set(edge) for edge in incident)
        ):
            self._refuse("pick is outside the next independent classic position", site)
        if (
            self._pins is not None
            and self._restricted_nodes is not None
            and (
                node not in self._restricted_node_sets[site]
                or any(choice not in self._restricted_edge_sets[edge] for edge, choice in pick.edges.items())
            )
        ):
            self._refuse(self._node_restriction_refusal(site, node) or "pick is outside the schedule restriction", site)
        if self._pins is not None and self._restricted_nodes is None:
            if why := self._node_restriction_refusal(site, node):
                self._refuse(why, site)
            if any(not self._edge_restriction_allows(edge, choice) for edge, choice in pick.edges.items()):
                self._refuse("pick is outside the schedule restriction", site)
        support = self._local_support(site, node, pick.edges)
        if support is None:
            self._refuse("pick has no local classic support", site)
        if why := self._support_refusal(site, support):
            self._refuse(why, site)
        work = support.work or self._work
        nodes = {**self.assignment.nodes, site: support.node}
        axes = {**self._axes, **{claim.name: (claim.tile, claim.units) for claim in support.axes}}
        fragments = {**self._fragments, **{(claim.role, claim.edge): claim.value for claim in support.fragments}}
        return self._advance(
            position=self.position + 1,
            _assignment=Schedule(None, nodes, {**self.assignment.edges, **support.edges}),
            _work=work,
            _axes=frozendict(axes),
            _fragments=frozendict(fragments),
            _raster_eligible=self._raster_eligible or support.raster_eligible,
            _producer_eligible=self._producer_eligible and support.producer_eligible,
        )

    def _advance(self, **changed) -> ClassicScheduleContext:
        """This context with the fields ONE composition step changes, skipping ``__post_init__``.

        Everything that ``__post_init__`` derives or proves belongs to ``tile_op``, ``domains`` or
        ``_pins`` — the node order covering every site exactly once, the domains covering it, the
        restriction tables and the works they allow — and a step touches none of the three, so a
        step re-derives only conclusions it already carries. Its own remaining checks do not reach a
        step either: the position bound holds because ``_extend_local`` advances only off a
        ``next_site``, and the stage restriction is validated at position 0. The caller passes
        ``_axes`` / ``_fragments`` already frozen, which is the one normalization lost with the
        skipped ``__post_init__``. ``replace`` re-ran all of it once per composition step —
        43.5k times for one SDPA_L schedule walk.

        The public ``extend`` keeps the validating path: a pick decoded from a golden row or handed
        in by a caller has proved none of this."""
        advanced = object.__new__(type(self))
        advanced.__dict__.update(self.__dict__)
        advanced.__dict__.update(changed)
        return advanced

    def _local_support(
        self,
        site: NodeId,
        node: NodeSchedule,
        edges: Mapping[EdgeSite, EdgeSchedule],
    ) -> _LocalSupport | None:
        """Derive the local ``p + t`` facts and decide their compatibility in one place."""
        facts = self.tile_op.contractions.get(site)
        if facts is None:
            return self._intrinsic_support(site, node, edges)
        materialization = getattr(self.tile_op, "materialization", None)
        if self.target is None and (
            materialization is None
            or (node.tile.is_tiled and site not in materialization.tiles)
            or any(not choice.stage.is_direct and edge not in materialization.stages for edge, choice in edges.items())
        ):
            return None
        cache = _target_memo(self.tile_op, self.target, "_memo_classic_local_support")
        key = (site, node, tuple(edges.items()))
        if key in cache:
            return cache[key]
        tile_op = self.tile_op
        fold = self.node(site)
        view = self.tile_op.views[site]
        incident = self.incident_edges(site)
        if set(edges) != set(incident):
            cache[key] = None
            return None
        if len(set(edges.values())) > 1:
            self._refuse("one contraction currently requires one transport choice across its operands", site)
        from emmy.compiler.ir.tile.ops import Sched  # noqa: PLC0415

        geometry = Sched(tile_op, place=tile_op.place.on_grid()).placed(fold, node.tile)
        if node.tile.is_tiled and not isinstance(geometry, PlacedTile):
            cache[key] = None
            return None
        if isinstance(geometry, PlacedTile):
            if _plan_node_refusal(tile_op, fold, node.tile, geometry, facts) is not None:
                cache[key] = None
                return None
        stage = next(iter(edges.values())).stage if edges else Stage.direct()
        resolved_stage = None
        if view.as_contraction() is None or not node.tile.is_tiled:
            if not stage.is_direct:
                cache[key] = None
                return None
        elif self.target is None:
            materialization = getattr(tile_op, "materialization", None)
            resolved = {materialization.stages.get(edge) for edge in edges} if materialization is not None else set()
            resolved.discard(None)
            resolved_stage = next(iter(resolved)) if len(resolved) == 1 else None
        elif _needs_fill(tile_op, fold, node.tile):
            packed_copy = tile_op.packed_reading(fold)[0] is not None and stage.transport in ("smem-async", "smem-tma")
            if not packed_copy and stage not in (Stage(depth=1), Stage(depth=2)):
                cache[key] = None
                return None
            resolved_stage = _resolve_stage(tile_op, self.target, fold, node.tile, geometry, stage, facts)
        elif not stage.is_direct:
            resolved_stage = _resolve_stage(tile_op, self.target, fold, node.tile, geometry, stage, facts)
        if not stage.is_direct and (resolved_stage is None or resolved_stage.choice != stage):
            cache[key] = None
            return None
        if isinstance(geometry, PlacedTile):
            if _paired_budget_refusal(fold, facts.producer, geometry, resolved_stage) is not None:
                cache[key] = None
                return None
        support = _LocalSupport(
            node,
            edges,
            work=derive_inventory((node.tile,), coop=node.reduce.coop if isinstance(node, ReductionSchedule) else 1),
            axes=(
                tuple(_AxisAgreement(side.axis.name, side.tile, side.units) for side in geometry.mn)
                if node.tile.is_tiled and isinstance(geometry, PlacedTile)
                else ()
            ),
            fragments=(
                _fragment_agreements(
                    site,
                    node.tile,
                    geometry,
                    resolved_stage,
                    facts,
                    frozenset(candidate.need for candidate in self.tile_op.contractions.values() if candidate.need is not None),
                )
                if isinstance(geometry, PlacedTile)
                else ()
            ),
            raster_eligible=node.tile.is_tiled and view.as_contraction() is not None,
            producer_eligible=not (tile_op.packed_reading(fold)[0] is not None and stage.transport == "smem-tma"),
        )
        cache[key] = support
        return support

    def _intrinsic_support(
        self,
        site: NodeId,
        node: NodeSchedule,
        edges: Mapping[EdgeSite, EdgeSchedule],
    ) -> _LocalSupport | None:
        """Derive the target-independent local relation when no finite domains are attached."""
        if site not in self.tile_op.family_sites["TILE"] and node.tile != Tile():
            return None
        if node.tile.is_warp and hasattr(self.target, node.tile.atom.target_feature):
            if not node.tile.atom.available_on(self.target):
                return None
        stages = {choice.stage for choice in edges.values()}
        if len(stages) > 1:
            self._refuse("one contraction currently requires one transport choice across its operands", site)
        if any(edge not in self.tile_op.stage_edges and not choice.stage.is_direct for edge, choice in edges.items()):
            return None
        if any(not choice.stage.is_direct and not node.tile.is_tiled for choice in edges.values()):
            return None
        if any(
            not choice.stage.is_direct and hasattr(self.target, "has_cp_async") and not choice.stage.available_on(self.target)
            for choice in edges.values()
        ):
            return None
        coop = node.reduce.coop if isinstance(node, ReductionSchedule) else 1
        try:
            work = derive_inventory((node.tile,), coop=coop)
        except ValueError:
            return None
        view = self.tile_op.views[site]
        return _LocalSupport(
            node,
            edges,
            work=work,
            raster_eligible=node.tile.is_tiled and view.as_contraction() is not None,
        )

    def _support_refusal(self, site: NodeId, support: _LocalSupport) -> str | None:
        """Return why one locally supported pick cannot extend this prefix."""
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
        support: _LocalSupport,
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
        if (
            not self.nodes_complete
            or self.assignment.kernel is not None
            or not isinstance(pick.kernel, KernelSchedule)
            or pick.nodes
            or pick.edges
            or (self.domains is not None and pick.kernel not in self.domains.kernel_set)
            or (self._restricted_kernel_set is not None and pick.kernel not in self._restricted_kernel_set)
        ):
            self._refuse("pick is incompatible with the classic kernel position")
        work = self._work or Work()
        if pick.kernel.work.kind != work.kind or pick.kernel.work.units != work.units:
            self._refuse("kernel WORK does not realize the node choices")
        if not pick.kernel.raster.is_direct and not self._raster_eligible:
            self._refuse("RASTER requires a tiled contraction site")
        if pick.kernel.work.producer and not self._producer_eligible:
            self._refuse("producer band is incompatible with the selected transport")
        assignment = Schedule(pick.kernel, self.assignment.nodes, self.assignment.edges)
        self._require_kernel_prefix(assignment)
        if self._unsupported_global:
            self._refuse("global schedule pin is unsupported by every applicable site")
        return replace(self, _assignment=assignment)

    def _require_kernel_prefix(self, schedule: ClassicAssignment) -> None:
        """Validate the kernel facts not already proved by local prefix composition."""
        kernel_work = Work(schedule.kernel.work.kind, schedule.kernel.work.units)
        warp_size = getattr(self.target, "warp_size", 32)
        compute_threads = kernel_work.count * (warp_size if kernel_work.kind == "warp" else 1)
        producer_threads = schedule.kernel.work.producer * warp_size
        if producer_threads > compute_threads:
            self._refuse("producer band cannot outnumber the compute band")
        if compute_threads + producer_threads > getattr(self.target, "max_threads_per_cta", 1024):
            self._refuse("worker inventory exceeds the target thread limit")
        if not schedule.kernel.work.producer:
            return
        for site, assignment in schedule.nodes.items():
            if not assignment.tile.is_tiled:
                continue
            if isinstance(assignment, ReductionSchedule) and assignment.reduce.needs_split:
                self._refuse("a producer band cannot accompany a cross-CTA reduction", site)
            edges = tuple(edge for edge in self.incident_edges(site) if edge in self.tile_op.stage_edges)
            if not edges or any(schedule.edges[edge].stage.transport != "smem-tma" for edge in edges):
                self._refuse("a producer band requires TMA transport at every tiled consumer", site)

    def _refuse(self, reason: str, site: NodeId | EdgeSite | None = None) -> None:
        if site is None:
            raise ScheduleRefused(reason)
        where = node_id_spelling(site) if type(site) is int else edge_site_spelling(site)
        raise ScheduleRefused(f"{where}: {reason}")

    def _require_complete_shape(self, schedule: ClassicAssignment) -> None:
        """Validate only complete-assignment structure before replaying normal transitions."""
        if not isinstance(schedule, Schedule) or not isinstance(schedule.kernel, KernelSchedule):
            self._refuse("assignment must contain a classic kernel schedule")
        if any(not isinstance(value, (ProjectionSchedule, ReductionSchedule)) for value in schedule.nodes.values()):
            self._refuse("classic node assignments must contain projection or reduction schedules")
        if any(not isinstance(value, EdgeSchedule) for value in schedule.edges.values()):
            self._refuse("classic edge assignments must contain edge schedules")
        expected_nodes = set(self.tile_op.node_sites)
        if missing := expected_nodes - schedule.nodes.keys():
            self._refuse("missing node assignment", min(missing))
        if extra := schedule.nodes.keys() - expected_nodes:
            self._refuse("node assignment is outside this problem", min(extra))
        expected_edges = set(self.tile_op.edge_sites)
        if missing := expected_edges - schedule.edges.keys():
            self._refuse("missing edge assignment", min(missing))
        if extra := schedule.edges.keys() - expected_edges:
            self._refuse("edge assignment is outside this problem", min(extra))

        for site in self.tile_op.node_sites:
            view = self.tile_op.views[site]
            assignment = schedule.nodes[site]
            if view.axis is None and not isinstance(assignment, ProjectionSchedule):
                self._refuse("projection site requires a projection schedule", site)
            if view.axis is not None and not isinstance(assignment, ReductionSchedule):
                self._refuse("reduction site requires a reduction schedule", site)
            if isinstance(assignment.tile, PlacedTile):
                self._refuse("node choices cannot contain placed tile geometry", site)

    def _kernel_composes(self, kernel: KernelSchedule) -> bool:
        work = self._work or Work()
        return (
            kernel.work.kind == work.kind
            and kernel.work.units == work.units
            and (not kernel.work.producer or self._producer_eligible)
            and (kernel.raster.is_direct or self._raster_eligible)
        )

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
        if self.domains is None and self._pins is None:
            return True
        if self._allowed_works is not None:
            return (work.kind, work.units) in self._allowed_works
        return any(
            kernel.work.kind == work.kind and kernel.work.units == work.units and self._kernel_restriction_allows(kernel)
            for kernel in self.kernels
        )

    def _node_restriction_allows(self, site: NodeId, choice: NodeSchedule) -> bool:
        """Whether one independent node value can still satisfy ``c``."""
        return self._node_restriction_refusal(site, choice) is None

    def _node_restriction_refusal(self, site: NodeId, choice: NodeSchedule) -> str | None:
        """Return why one independent node value cannot satisfy ``c``."""
        if self._pins is not None:
            tiled = site in self.tile_op.family_sites["TILE"]
            if tiled and not self._allows_value("TILE", self.node_key("TILE", site), choice.tile.spell()):
                return "TILE is outside the schedule restriction"
            if isinstance(choice, ReductionSchedule) and not self._allows_value(
                "REDUCE", self.node_key("REDUCE", site), choice.reduce.spell()
            ):
                return "REDUCE is outside the schedule restriction"
        if choice.tile.is_warp:
            atom = choice.tile.atom
            pinned = self._tile_is_pinned(site, choice.tile)
            if atom.operand_dtype("a").logical_elems == 1 and atom.operand_dtype("a").nbytes == 1 and not self._allow_fp8 and not pinned:
                return "FP8 TILE is outside the precision restriction"
            if atom.operand_dtype("c").nbytes == 2 and not self._allow_f16_accumulate and not pinned:
                return "f16-accumulate TILE is outside the precision restriction"
        return None

    def _tile_is_pinned(self, site: NodeId, tile: Tile) -> bool:
        """Whether ``c`` explicitly selects this TILE value at ``site``.

        Precision controls restrict unpinned enumeration policy; an authored TILE still selects a
        legal independent-domain value. Keeping that exception here makes pinned full schedules
        and lazy extension use the same restriction relation.
        """
        if self._pins is None or site not in self.tile_op.family_sites["TILE"]:
            return False
        key = self.node_key("TILE", site)
        return bool(self._applicable_pins("TILE", key)) and self._allows_value("TILE", key, tile.spell())

    def _edge_restriction_allows(self, edge: EdgeSite, choice: EdgeSchedule) -> bool:
        """Whether one independent edge value can still satisfy ``c``."""
        if self._pins is None or edge not in self.tile_op.stage_edges:
            return True
        return self._allows_value("STAGE", self.stage_key(edge), choice.stage.spell())

    def node_assignment(self, site: NodeId) -> NodeSchedule:
        return self.assignment.nodes[site]

    def edge_assignment(self, edge: EdgeSite) -> EdgeSchedule:
        return self.assignment.edges[edge]

    @property
    def work(self) -> Work | None:
        return self._work


class ClassicScheduleCodec:
    """Strict wire boundary for complete classic schedules.

    Kernel families are bare. A node family is bare when it has one applicable site and carries
    its site's route (``@map.1/twist.1/inner``) only when the family is ambiguous. STAGE is one
    transport decision per consumer node and follows the same rule. Decoding accepts no aliases, missing direct values,
    or unknown fields.
    """

    def __init__(self, context: ClassicScheduleContext) -> None:
        if not isinstance(context, ClassicScheduleContext):
            raise TypeError("classic codec requires a ClassicScheduleContext")
        self.context = context
        self.tile_op = context.tile_op
        stage_consumers = tuple(dict.fromkeys(edge[0] for edge in self.tile_op.stage_edges))
        self._key_order = (
            "WORK",
            "RASTER",
            *(classic_node_key(self.tile_op, "TILE", site) for site in self.tile_op.family_sites["TILE"]),
            *(classic_node_key(self.tile_op, "REDUCE", site) for site in self.tile_op.family_sites["REDUCE"]),
            *(
                classic_stage_key(self.tile_op, next(edge for edge in self.tile_op.stage_edges if edge[0] == site))
                for site in stage_consumers
            ),
        )
        self._keys = frozenset(self._key_order)

    def encode(self, schedule: ClassicAssignment) -> dict[str, str]:
        """Encode one accepted typed schedule in canonical scope order."""
        accepted = self.context.extend(schedule).assignment
        return self._encode(accepted)

    def _encode(self, schedule: ClassicAssignment) -> dict[str, str]:
        """Encode a schedule already accepted by this codec's context traversal."""
        row = {
            "WORK": schedule.kernel.work.spell(),
            "RASTER": schedule.kernel.raster.spell(),
        }
        for site in self.tile_op.family_sites["TILE"]:
            row[classic_node_key(self.tile_op, "TILE", site)] = schedule.nodes[site].tile.spell()
        for site in self.tile_op.family_sites["REDUCE"]:
            assignment = schedule.nodes[site]
            assert isinstance(assignment, ReductionSchedule)
            row[classic_node_key(self.tile_op, "REDUCE", site)] = assignment.reduce.spell()
        stage_consumers = tuple(dict.fromkeys(edge[0] for edge in self.tile_op.stage_edges))
        for site in stage_consumers:
            edges = tuple(edge for edge in self.tile_op.stage_edges if edge[0] == site)
            stages = {schedule.edges[edge].stage for edge in edges}
            if len(stages) != 1:
                raise ValueError(f"{node_id_spelling(site)}: one STAGE value must cover every operand edge")
            row[classic_stage_key(self.tile_op, edges[0])] = stages.pop().spell()
        return row

    def delta(self, before: ClassicScheduleContext, after: ClassicScheduleContext) -> dict[str, str]:
        """Encode the canonical row fields introduced by one compatibility step."""
        if any(context.tile_op != self.context.tile_op for context in (before, after)):
            raise ValueError("classic codec delta requires contexts for its problem")
        row = {}
        assert before.order is not None and after.order is not None
        for site in after.order[before.position : after.position]:
            node = after.node_assignment(site)
            if site in after.tile_op.family_sites["TILE"]:
                row[after.node_key("TILE", site)] = node.tile.spell()
            if site in after.tile_op.family_sites["REDUCE"]:
                assert isinstance(node, ReductionSchedule)
                row[after.node_key("REDUCE", site)] = node.reduce.spell()
            staged = tuple(edge for edge in after.incident_edges(site) if edge in after.tile_op.stage_edges)
            if staged:
                choices = {after.edge_assignment(edge) for edge in staged}
                if len(choices) == 1:
                    row[after.stage_key(staged[0])] = choices.pop().stage.spell()
        if after.work is not None:
            row["WORK"] = after.work.spell()
        return row

    def decode(self, row: Mapping[str, str]) -> ClassicAssignment:
        """Decode one complete canonical row and reject every other key set or assignment."""
        schedule = self._parse(row)
        return self._validate_row(schedule, row)

    def _parse(self, row: Mapping[str, str]) -> ClassicAssignment:
        """Parse typed values before a reconstructed TileOp supplies materialization for validation."""
        self._check_keys(row)

        work = Work.parse(row["WORK"])
        nodes: dict[NodeId, NodeSchedule] = {}
        for site in self.tile_op.node_sites:
            view = self.tile_op.views[site]
            reduce = None
            if view.axis is not None:
                reduce = Reduce.parse(row[classic_node_key(self.tile_op, "REDUCE", site)], work)
            tile = (
                resolve_site_tile(
                    row[classic_node_key(self.tile_op, "TILE", site)],
                    work,
                    reduce.coop if reduce is not None else 1,
                )
                if site in self.tile_op.family_sites["TILE"]
                else Tile()
            )
            nodes[site] = ProjectionSchedule(tile) if reduce is None else ReductionSchedule(tile, reduce)
        return Schedule(
            KernelSchedule(work, Raster.parse(row["RASTER"])),
            nodes,
            {
                edge: EdgeSchedule(Stage.parse(row[classic_stage_key(self.tile_op, edge)]))
                if edge in self.tile_op.stage_edges
                else EdgeSchedule(Stage.direct())
                for edge in self.tile_op.edge_sites
            },
        )

    def _validate_row(self, schedule: ClassicAssignment, row: Mapping[str, str]) -> ClassicAssignment:
        """Validate a parsed assignment and its claimed canonical row exactly once."""
        accepted = self.context.extend(schedule).assignment
        canonical = self._encode(accepted)
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


__all__ = [
    "CLASSIC_FAMILIES",
    "ClassicAssignment",
    "ClassicMaterialization",
    "ClassicDomains",
    "ClassicScheduleCodec",
    "ClassicScheduleContext",
    "EdgeSchedule",
    "edge_site_spelling",
    "KernelSchedule",
    "NodeSchedule",
    "ProjectionSchedule",
    "ReductionSchedule",
    "node_id_spelling",
    "parse_edge_site",
    "parse_node_id",
]
