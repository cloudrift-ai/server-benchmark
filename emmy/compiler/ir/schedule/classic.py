"""The semantic model for the classic grid/CTA/warp/thread/register schedule.

The model contains choices only.  A :class:`ClassicProblem` supplies the immutable Fold tree and
target; :class:`ClassicScheduleContext` derives identities and classification from that problem and
is the only compatibility authority for a complete assignment. Codecs, enumeration order, search
state, and materialization data do not belong here.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace

from frozendict import frozendict

from emmy.compiler.ir.address import gmem_axis_step, split_addressable
from emmy.compiler.ir.atom import ATOM_REGISTRY, atoms_for
from emmy.compiler.ir.fold_tree import children, walk
from emmy.compiler.ir.packed import match_packed_b_node, match_packed_pair_node
from emmy.compiler.ir.pure.fold import Fold, deep_reads, edge_refs_axis, is_contraction
from emmy.compiler.ir.stmt import Accum, Body, Load, Loop, Write
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
from .views import EdgeSite, NodeId, NodeView, Projection, Reduction, ScheduleInventory

CLASSIC_FAMILIES = ("TILE", "REDUCE", "STAGE")

_MAX_REGISTERS_PER_THREAD = 255
_MAX_REGISTERS_PER_CTA = 64 * 1024


def _pickle_fields(value) -> dict:
    """Return declared state only, excluding memo tables attached to frozen dataclasses."""
    return {name: value.__dict__[name] for name in value.__dataclass_fields__ if name in value.__dict__}


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


@dataclass(frozen=True)
class _ContractionFacts:
    """Problem and target facts needed to compose one contraction schedule."""

    k_axis: object
    seam: tuple | None = None
    producer: Fold | None = None
    need: str | None = None
    packed: tuple = (None, None)
    need_step: bool = False
    warp_refusal: str | None = None


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
    axes: tuple[tuple[str, int, int], ...] = ()
    fragments: tuple[tuple[str, str, tuple], ...] = ()
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


def _computed_edge(node: Fold) -> bool:
    return any(isinstance(edge, Fold) and edge.axis is None for edge in (node.a, *(channel.b for channel in node.channels)))


def _needs_fill(tile_op, node: Fold, plan: Tile) -> bool:
    from . import staging  # noqa: PLC0415

    return plan.is_warp and (_computed_edge(node) or len(node.channels) > 1 or staging.converting_a(node, plan.atom, tile_op.inputs))


def _kstep_refusal(k_axis, plan: Tile) -> str | None:
    if not (plan.is_warp and plan.atom.operand_dtype("a").nbytes == 1):
        return None
    if not k_axis.extent.is_static:
        return f"atom {plan.atom.name}: fp8 fragment loads require a static K"
    step = plan.atom.atom_k * plan.bk
    extent = k_axis.extent.as_static()
    return None if extent % step == 0 else f"warp TILE K-step {step} does not divide the static contraction K={extent}"


def _plan_node_refusal(tile_op, node: Fold, plan: Tile, placed: PlacedTile, facts: _ContractionFacts) -> str | None:
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
    facts: _ContractionFacts,
) -> ResolvedStage | None:
    from . import staging  # noqa: PLC0415

    packed_copy = facts.packed[0] is not None and choice.transport in ("smem-async", "smem-tma")
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
        )
    if plan.is_warp:
        return staging.resolve_warp_stage(
            node,
            placed,
            choice,
            target.max_dynamic_smem,
            tile_op.inputs,
            readings=facts.packed,
        )
    return staging.resolve_scalar_stage(node, placed, choice, tile_op.inputs, target.max_dynamic_smem)


def _fragment_agreements(
    site: NodeId,
    plan: Tile,
    placed: PlacedTile,
    stage: ResolvedStage | None,
    facts: _ContractionFacts,
    producer_sites: frozenset[str],
) -> tuple[tuple[str, str, tuple], ...]:
    out = []
    key = node_id_spelling(site)
    if key in producer_sites:
        if not plan.is_tiled:
            offer = ("free",)
        elif plan.is_warp:
            offer = ("warp", plan.atom.shape, plan.atom.fragment_layout, placed.n.units, placed.n.tile)
        else:
            offer = ("scalar",)
        out.append(("offer", key, offer))
    if facts.need is not None:
        if plan.is_warp and stage is not None and stage.transport == "smem":
            need = ("step" if facts.need_step else "warp", plan.atom.shape, plan.atom.fragment_layout, stage.bk_elems)
        else:
            need = ("free",)
        out.append(("need", facts.need, need))
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

    atom = placed.atom
    if stage.bk_elems % atom.atom_n:
        return None
    a_regs = _fragment_registers(atom, "a")
    b_regs = _fragment_registers(atom, "b")
    c_regs = _fragment_registers(atom, "c")
    if atom.operand_dtype("c").nbytes == 2:
        c_regs += atom.atom_m * atom.atom_n // 32
    depth = max(1, stage.reg_depth)
    channels = len(node.channels)
    consumer_c = channels * placed.reg_m * placed.reg_n * c_regs
    consumer = placed.reg_m * depth * a_regs + channels * (placed.reg_n * depth * b_regs + placed.reg_m * placed.reg_n * c_regs)
    producer_n = stage.bk_elems // atom.atom_n
    producer_regs = placed.reg_m * a_regs + len(producer.channels) * (producer_n * b_regs + placed.reg_m * producer_n * c_regs)
    required = max(consumer, consumer_c + producer_regs)
    available = min(_MAX_REGISTERS_PER_THREAD, _MAX_REGISTERS_PER_CTA // placed.block_threads)
    if required <= available:
        return None
    return (
        f"paired contractions require at least {required} live fragment registers/thread, over the "
        f"{available}-register envelope at {placed.block_threads} threads/CTA"
    )


def _atom_refusal(atom, a_dtype, a_step, a_is_load: bool, tail: list, free: tuple, shapes: dict) -> str | None:
    """Return why one otherwise available atom cannot bind this problem."""
    converting = a_is_load and a_dtype is not None and a_dtype.nbytes >= 2 and a_dtype != atom.operand_dtype("a")
    if a_is_load and not converting and (a_step is None or a_step[0] != 1 or (a_step[1] and a_step[1] % atom.atom_k)):
        motion = "unknown" if a_step is None else f"{a_step[0]} elements per column"
        return (
            f"warp TILE: A fragment loaders read {atom.atom_k} contraction columns CONTIGUOUSLY, "
            f"but this operand's gmem index moves {motion}"
        )
    roles = [(free[-1].name, atom.shape[1], "n", True)]
    if len(free) >= 2:
        roles.append((free[-2].name, atom.shape[0], "m", False))
    for stmt in tail:
        if not isinstance(stmt, (Load, Write)):
            continue
        buffer = stmt.input if isinstance(stmt, Load) else stmt.output
        shape = getattr(shapes.get(buffer), "shape", None)
        for name, extent, role, trailing in roles:
            if not split_addressable(stmt.index, shape, name, extent, trailing):
                return f"warp TILE: the {role} axis reaches {buffer} through an unsupported split dimension"
    return None


def _fold_states(op: Fold) -> frozenset[str]:
    """Return the Fold state names visible to the projection tail."""
    if op.axis is not None:
        return frozenset(op.defines())
    return frozenset(name for edge in (*op.operands, *op.body) if isinstance(edge, Fold) for name in edge.defines())


def _fragment_epilogue_ok(tail: list, states: frozenset[str]) -> bool:
    """Whether every output is a straight-line projection of a Fold state."""
    definitions: set[str] = set()
    for stmt in tail:
        if isinstance(stmt, Loop):
            return False
        if isinstance(stmt, Load) and {name for index in stmt.index for name in index.free_vars()} & definitions:
            return False
        definitions.update(stmt.defines())
    body = Body(tail)
    return all(body.backward_cone(stmt.values).external_reads & states for stmt in tail if isinstance(stmt, Write))


def _channel_dtype(tile_op, node: Fold, target):
    """Return the one tensor-core dtype shared by a contraction's B channels."""
    from emmy.compiler.ir.tile.ops import edge_dtypes  # noqa: PLC0415

    dtypes = {edge_dtypes(channel.b, tile_op.inputs)[0] for channel in node.channels}
    if len(dtypes) == 1:
        return next(iter(dtypes))
    eligible = {dtype for dtype in dtypes if dtype is not None and atoms_for(dtype, ctx=target)}
    return next(iter(eligible)) if len(eligible) == 1 else None


def _node_refusal(tile_op, target, node: Fold, fragment_epilogue: bool, packed: tuple) -> str | None:
    """Return why problem and target facts rule out every tensor-core atom."""
    from emmy.compiler.ir.tile.ops import edge_dtypes  # noqa: PLC0415

    ring = node.semiring
    if ring is None or tuple(operator.name for operator in ring) != ("multiply", "add"):
        return "the mma atom realizes only the (multiply, add) semiring instance"
    if not tile_op.inputs:
        return "no typed inputs expose operand dtypes"
    if len(tile_op.place.free) < 2:
        return "the grid supplies no output-axis pair for a fragment"
    if not fragment_epilogue:
        return "the projection epilogue is not a per-fragment straight-line program"
    if isinstance(node.a, Fold) and node.a.axis is not None:
        return "a nested scheduling site inhabits the A edge"
    if len(node.channels) == 1 and isinstance(node.channels[0].b, Fold) and node.channels[0].b.axis is not None:
        return "a nested scheduling site inhabits the B edge"

    dtype = edge_dtypes(node.a, tile_op.inputs)[0]
    if packed[1] is not None:
        return None
    if dtype is not None and dtype.logical_elems != 1:
        return f"a packed {dtype} A pairs with no packed peer; no atom multiplies packed codes against decoded ones"
    if dtype is not None and dtype.nbytes == 1:
        if not isinstance(node.a, Load):
            return "fp8 fragment loads require a materialized A edge"
        if _channel_dtype(tile_op, node, target) != dtype:
            return "fp8 fragment loads require one matching operand dtype"
        if not atoms_for(dtype, ctx=target):
            return f"no tensor-core atom takes a {dtype} multiplicand on this target"
        return None

    atom_dtype = dtype if atoms_for(dtype, ctx=target) else _channel_dtype(tile_op, node, target)
    if atom_dtype is None:
        return "no operand dtype selects a tensor-core atom family"
    if atom_dtype.nbytes == 1 and atom_dtype != dtype:
        return "a demoting compute fill cannot produce an fp8 fragment"
    if not (atoms_for(atom_dtype, ctx=target) or atoms_for(atom_dtype, acc=atom_dtype, ctx=target)):
        return f"no tensor-core atom takes a {atom_dtype} multiplicand on this target"
    return None


def _sibling_fragment_edges(root: Fold, inventory: ScheduleInventory) -> dict[int, str]:
    """Map each sibling-step consumer to the one contraction producing its computed edge."""
    out = {}
    for node, _axes in walk(root):
        if not (node.axis is not None and not is_contraction(node) and node.combine is not None):
            continue
        steps = node.step_stmts()
        states = set(node.combine.results)
        for position, consumer in ((i, stmt) for i, stmt in enumerate(steps) if is_contraction(stmt)):
            accumulated = any(
                isinstance(stmt, Accum) and stmt.name in states and stmt.value in consumer.defines() for stmt in steps[position + 1 :]
            )
            reads = {name for edge in consumer.operands if isinstance(edge, Fold) for name in deep_reads(edge.lower())}
            if not accumulated or not reads:
                continue
            cone = Body(tuple(steps[:position])).backward_cone(reads)
            producers = tuple(stmt for stmt in cone.members if is_contraction(stmt))
            if len(producers) == 1:
                out[id(consumer)] = node_id_spelling(inventory.site(producers[0]))
    return out


def _contraction_facts(tile_op, target, inventory: ScheduleInventory) -> dict[NodeId, _ContractionFacts]:
    """Derive the complete immutable contraction facts of one classic problem."""
    from emmy.compiler.ir.tile.ops import cone_seam, projection_tail  # noqa: PLC0415
    from emmy.compiler.ir.tile.path import sites  # noqa: PLC0415

    root = tile_op.op
    parents: dict[int, Fold] = {}
    for node, axes in walk(root):
        for child, _child_axes in children(node, axes):
            parents.setdefault(id(child), node)
    derived = {id(site.node) for site in sites(root) if site.derived}
    sibling = _sibling_fragment_edges(root, inventory)
    tail = projection_tail(tile_op)
    fragment_epilogue = _fragment_epilogue_ok(tail, _fold_states(root))
    facts = {}
    for node, _axes in walk(root):
        if not (node.axis is not None and is_contraction(node)):
            continue
        site = inventory.site(node)
        if site in facts:
            continue
        packed = (match_packed_b_node(node, tile_op.inputs), match_packed_pair_node(node, tile_op.inputs))
        warp_refusal = _node_refusal(tile_op, target, node, fragment_epilogue, packed)
        parent = parents.get(id(node))
        if (
            id(node) in derived
            and node.axis.extent.is_static
            and node.axis.extent.as_static() == 1
            and isinstance(parent, Fold)
            and parent.axis is not None
        ):
            assert parent.combine is not None and node.combine is not None
            seam = ((), (), tuple(parent.combine.results[: -len(node.combine.results)]))
            k_axis = parent.axis
        else:
            seam = cone_seam(node.a, node.axis.name) if isinstance(node.a, Fold) and warp_refusal is None else None
            k_axis = node.axis
        producer = None
        if isinstance(node.a, Fold):
            nested = tuple(site.node for site in sites(node.a) if is_contraction(site.node) and edge_refs_axis(site.node, k_axis.name))
            producer = nested[0] if len(nested) == 1 else None
        need = sibling.get(id(node))
        need_step = need is not None
        if need is None and producer is not None:
            need = node_id_spelling(inventory.site(producer))
        facts[site] = _ContractionFacts(
            k_axis=k_axis,
            seam=seam,
            producer=producer,
            need=need,
            packed=packed,
            need_step=need_step,
            warp_refusal=warp_refusal,
        )
    return facts


def _warp_atoms(tile_op, target, node: Fold, facts: _ContractionFacts) -> tuple[str, ...]:
    """Project tensor-core atoms from one contraction's problem and target facts."""
    from emmy.compiler.ir.tile.ops import edge_dtypes, projection_tail  # noqa: PLC0415

    if facts.warp_refusal is not None:
        return ()
    dtype = edge_dtypes(node.a, tile_op.inputs)[0]
    a_is_load = isinstance(node.a, Load)
    a_step = gmem_axis_step(node.a, node.axis.name, tile_op.inputs) if a_is_load else None
    tail = projection_tail(tile_op)
    shapes = {**tile_op.inputs, **tile_op.outputs}

    def bindable(names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            name for name in names if _atom_refusal(ATOM_REGISTRY[name], dtype, a_step, a_is_load, tail, tile_op.place.free, shapes) is None
        )

    if (pair := facts.packed[1]) is not None:
        if any(operand.bits is None for operand in pair.b):
            return ()
        weights = {tile_op.inputs[operand.bits.input].dtype for operand in pair.b}
        return atoms_for(next(iter(weights)), ctx=target) if len(weights) == 1 else ()
    if dtype is not None and dtype.nbytes == 1:
        return bindable(atoms_for(dtype, ctx=target))
    atom_dtype = dtype if atoms_for(dtype, ctx=target) else _channel_dtype(tile_op, node, target)
    base = bindable(atoms_for(atom_dtype, ctx=target))
    reduced_acc = bindable(atoms_for(atom_dtype, acc=atom_dtype, ctx=target))
    return tuple(dict.fromkeys((*base, *reduced_acc)))


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

    def validate(self, schedule: ClassicAssignment, source: object, *, place: object, workers: object) -> None:
        """Validate classic lowering facts against their semantic assignment."""
        if not isinstance(schedule, Schedule):
            raise TypeError("classic materialization requires a Schedule")
        from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415

        if not isinstance(source, TileOp):
            raise TypeError("classic materialization requires a TileOp")
        from emmy.compiler.ir.tile.ops import Sched  # noqa: PLC0415

        context = ClassicScheduleContext(ClassicProblem.from_tile(source, target=None))
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
        placement = Sched(source.op, place=place)
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
    """The complete immutable problem ``p`` and target ``t`` for classic scheduling."""

    root: Fold
    target: object
    tile_op: object = field(repr=False, compare=False)
    contractions: Mapping[NodeId, _ContractionFacts] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.root, Fold):
            raise TypeError("classic problem root must be a Fold")
        if getattr(self.tile_op, "op", None) is not self.root:
            raise ValueError("classic problem source must own its Fold root")
        if not isinstance(self.contractions, Mapping) or any(
            not _is_node_id(site) or not isinstance(facts, _ContractionFacts) for site, facts in self.contractions.items()
        ):
            raise TypeError("classic contraction facts must be keyed by node id")
        object.__setattr__(self, "contractions", frozendict(self.contractions))

    def __getstate__(self):
        """Pickle immutable problem fields, never derived support caches."""
        return _pickle_fields(self)

    @classmethod
    def from_tile(cls, tile_op, target) -> ClassicProblem:
        """Capture all immutable source facts used by compatibility composition."""
        inventory = ScheduleInventory.from_root(tile_op.op, nodes=tile_op.nodes, edges=tile_op.node_edges)
        return cls(tile_op.op, target, tile_op, _contraction_facts(tile_op, target, inventory))


@dataclass(frozen=True)
class _ClassicSpace:
    """Immutable indexes shared by every prefix of one ``p``/``t`` product."""

    inventory: ScheduleInventory
    kernel_set: frozenset
    node_sets: frozendict
    edge_sets: frozendict
    values: frozendict

    def __getstate__(self):
        """Pickle immutable indexes, never derived frontier caches."""
        return _pickle_fields(self)


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
    _space: _ClassicSpace | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._space is None:
            object.__setattr__(self, "_space", self._build_space())
        assert self._space is not None
        sites = self._space.inventory.node_sites
        order = sites if self.order is None else tuple(self.order)
        if len(order) != len(sites) or set(order) != set(sites):
            raise ValueError("classic composition order must contain every node site exactly once")
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
                self._validate_tile_restriction()
                self._validate_stage_restriction()

    def _build_space(self) -> _ClassicSpace:
        inventory = ScheduleInventory.from_root(
            self.problem.root,
            nodes=self.problem.tile_op.nodes,
            edges=self.problem.tile_op.node_edges,
        )
        node_sites = inventory.node_sites
        edge_sites = inventory.edges
        tile_sites = inventory.tile_sites
        reduction_sites = inventory.reduction_sites
        stage_edges = inventory.stage_edges
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
            inventory,
            kernel_set,
            node_sets,
            edge_sets,
            frozendict(values),
        )

    @property
    def node_sites(self) -> tuple[NodeId, ...]:
        assert self._space is not None
        return self._space.inventory.node_sites

    @property
    def edge_sites(self) -> tuple[EdgeSite, ...]:
        assert self._space is not None
        return self._space.inventory.edges

    @property
    def views(self) -> Mapping[NodeId, NodeView]:
        assert self._space is not None
        return self._space.inventory.views

    @property
    def tile_sites(self) -> tuple[NodeId, ...]:
        assert self._space is not None
        return self._space.inventory.tile_sites

    @property
    def reduction_sites(self) -> tuple[NodeId, ...]:
        assert self._space is not None
        return self._space.inventory.reduction_sites

    @property
    def stage_edges(self) -> tuple[EdgeSite, ...]:
        assert self._space is not None
        return self._space.inventory.stage_edges

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
        return self._space.inventory.node(site)

    def site(self, node: Fold) -> NodeId:
        assert self._space is not None
        try:
            return self._space.inventory.site(node)
        except KeyError:
            raise KeyError("Fold is not a node of this classic problem") from None

    def operand(self, edge: EdgeSite):
        if not _is_edge_site(edge):
            raise KeyError(f"invalid edge site {edge!r}")
        assert self._space is not None
        return self._space.inventory.operand(edge)

    def incident_edges(self, site: NodeId) -> tuple[EdgeSite, ...]:
        assert self._space is not None
        return self._space.inventory.incident_edges(site)

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
                if self._kernel_refusal(kernel) is None:
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
        assert self._space is not None
        cache = instance_memo(self._space, "_memo_local_frontier")
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
            assert self._space is not None
            indexes = instance_memo(self._space, "_memo_frontier_by_work")
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
        if not self.stage_edges:
            return
        from .staging import stage_target  # noqa: PLC0415

        for key, spelling in self._pins["STAGE"]:
            if not spelling or (key != "STAGE" and key not in self.keys()):
                continue
            choice = Stage.parse(spelling)
            if why := stage_target(choice, self.problem.target):
                raise ValueError(why)
            if key == "STAGE" and not self._supports_global("STAGE", spelling):
                raise ValueError(f"STAGE pin {spelling!r} does not resolve for this contraction")
        for site in self.node_sites:
            keys = tuple(dict.fromkeys(self.stage_key(edge) for edge in self.incident_edges(site) if edge in self.stage_edges))
            pins = tuple(pin for key in keys for pin in self._applicable_pins("STAGE", key) if pin)
            if not pins:
                continue
            edge_domains = tuple((edge, self._restricted_edges[edge]) for edge in self.incident_edges(site))
            if not self._compatible_frontier(site, self._restricted_nodes[site], edge_domains):
                raise ValueError(f"STAGE pin {pins[-1]!r} does not resolve for this contraction")

    def _validate_tile_restriction(self) -> None:
        """Report target-unavailable atoms; structural incompatibility simply empties the slice."""
        assert self._pins is not None
        for key, spelling in self._pins["TILE"]:
            name = spelling.partition("/")[0]
            if not spelling or not name.startswith("mma_") or (key != "TILE" and key not in self.keys()):
                continue
            atom = ATOM_REGISTRY.get(name)
            if atom is None or self.problem.target is None or atom.available_on(self.problem.target):
                continue
            cc = self.problem.target.compute_capability
            raise ValueError(f"atom {name} requires target feature {atom.target_feature}, which is unavailable on sm_{cc[0]}{cc[1]}")

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
        assert self._space is not None
        incident = self.incident_edges(site)
        if site is None or pick.kernel is not None or set(pick.nodes) != {site} or set(pick.edges) != set(incident):
            self._refuse("pick is outside the next independent classic position", site)
        node = pick.nodes[site]
        if not isinstance(node, (ProjectionSchedule, ReductionSchedule)) or any(
            not isinstance(choice, EdgeSchedule) for choice in pick.edges.values()
        ):
            self._refuse("pick contains a value from another schedule family", site)
        view = self.views[site]
        if isinstance(view, Projection) and not isinstance(node, ProjectionSchedule):
            self._refuse("projection site requires a projection schedule", site)
        if isinstance(view, Reduction) and not isinstance(node, ReductionSchedule):
            self._refuse("reduction site requires a reduction schedule", site)
        if isinstance(node.tile, PlacedTile):
            self._refuse("node choices cannot contain placed tile geometry", site)
        if self.domains is not None and (
            node not in self._space.node_sets[site] or any(pick.edges[edge] not in self._space.edge_sets[edge] for edge in incident)
        ):
            self._refuse("pick is outside the next independent classic position", site)
        if (
            self._pins is not None
            and self._restricted_nodes is not None
            and (
                node not in self._restricted_nodes[site]
                or any(choice not in self._restricted_edges[edge] for edge, choice in pick.edges.items())
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
        axes = {**self._axes, **{name: (tile, units) for name, tile, units in support.axes}}
        fragments = {**self._fragments, **{(role, edge): value for role, edge, value in support.fragments}}
        return replace(
            self,
            position=self.position + 1,
            _assignment=Schedule(None, nodes, {**self.assignment.edges, **support.edges}),
            _work=work,
            _axes=axes,
            _fragments=fragments,
            _raster_eligible=self._raster_eligible or support.raster_eligible,
            _producer_eligible=self._producer_eligible and support.producer_eligible,
        )

    def _local_support(
        self,
        site: NodeId,
        node: NodeSchedule,
        edges: Mapping[EdgeSite, EdgeSchedule],
    ) -> _LocalSupport | None:
        """Derive the local ``p + t`` facts and decide their compatibility in one place."""
        cache = instance_memo(self.problem, "_memo_classic_local_support")
        key = (site, node, tuple(edges.items()))
        if key in cache:
            return cache[key]
        tile_op = self.problem.tile_op
        fold = self.node(site)
        view = self.views[site]
        incident = self.incident_edges(site)
        if set(edges) != set(incident):
            cache[key] = None
            return None
        if len(set(edges.values())) > 1:
            self._refuse("one contraction currently requires one transport choice across its operands", site)
        from emmy.compiler.ir.tile.ops import Sched  # noqa: PLC0415

        problem_memo = instance_memo(self.problem, "_memo_classic_problem")
        sched = problem_memo.get("sched")
        if sched is None:
            sched = Sched(tile_op.op, place=tile_op.place.on_grid())
            problem_memo["sched"] = sched
        geometry = sched.placed(fold, node.tile)
        facts = self.problem.contractions.get(site)
        needs_geometry = isinstance(view, Reduction) and node.tile.is_tiled
        if needs_geometry and not isinstance(geometry, PlacedTile):
            cache[key] = None
            return None
        if isinstance(geometry, PlacedTile) and facts is not None:
            if _plan_node_refusal(tile_op, fold, node.tile, geometry, facts) is not None:
                cache[key] = None
                return None
        stage = next(iter(edges.values())).stage if edges else Stage.direct()
        resolved_stage = None
        if not isinstance(view, Reduction) or view.contraction is None or not node.tile.is_tiled:
            if not stage.is_direct:
                cache[key] = None
                return None
        elif self.problem.target is None:
            materialization = getattr(tile_op, "materialization", None)
            resolved = {materialization.stages.get(edge) for edge in edges} if materialization is not None else set()
            resolved.discard(None)
            resolved_stage = next(iter(resolved)) if len(resolved) == 1 else None
        elif _needs_fill(tile_op, fold, node.tile):
            packed_copy = facts.packed[0] is not None and stage.transport in ("smem-async", "smem-tma")
            if not packed_copy and stage not in (Stage(depth=1), Stage(depth=2)):
                cache[key] = None
                return None
            resolved_stage = _resolve_stage(tile_op, self.problem.target, fold, node.tile, geometry, stage, facts)
        elif not stage.is_direct:
            resolved_stage = _resolve_stage(tile_op, self.problem.target, fold, node.tile, geometry, stage, facts)
        if not stage.is_direct and (resolved_stage is None or resolved_stage.choice != stage):
            cache[key] = None
            return None
        if facts is not None and isinstance(geometry, PlacedTile):
            if _paired_budget_refusal(fold, facts.producer, geometry, resolved_stage) is not None:
                cache[key] = None
                return None
        support = _LocalSupport(
            node,
            edges,
            work=derive_inventory((node.tile,), coop=node.reduce.coop if isinstance(node, ReductionSchedule) else 1),
            axes=(
                tuple((side.axis.name, side.tile, side.units) for side in geometry.mn)
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
                    frozenset(candidate.need for candidate in self.problem.contractions.values() if candidate.need is not None),
                )
                if facts is not None and isinstance(geometry, PlacedTile)
                else ()
            ),
            raster_eligible=isinstance(view, Reduction) and view.contraction is not None and node.tile.is_tiled,
            producer_eligible=not (facts is not None and facts.packed[0] is not None and stage.transport == "smem-tma"),
        )
        cache[key] = support
        return support

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
        for name, tile, units in support.axes:
            value = (tile, units)
            if axis_values.get(name, value) != value:
                return "pick disagrees on physical-axis geometry"
        fragment_values = dict(fragments)
        for role, edge, value in support.fragments:
            key = (role, edge)
            if fragment_values.setdefault(key, value) != value:
                return "pick repeats a fragment endpoint inconsistently"
            other_role = "need" if role == "offer" else "offer"
            other = fragment_values.get((other_role, edge))
            if other is None:
                continue
            need, offer = (value, other) if role == "need" else (other, value)
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
            or (self.domains is not None and pick.kernel not in self._space.kernel_set)
            or (self._restricted_kernel_set is not None and pick.kernel not in self._restricted_kernel_set)
        ):
            self._refuse("pick is incompatible with the classic kernel position")
        if refusal := self._kernel_refusal(pick.kernel):
            self._refuse(*refusal)
        if self._unsupported_global:
            self._refuse("global schedule pin is unsupported by every applicable site")
        return replace(self, _assignment=Schedule(pick.kernel, self.assignment.nodes, self.assignment.edges))

    def _kernel_refusal(self, kernel: KernelSchedule) -> tuple[str, NodeId | None] | None:
        """Return why the final kernel value cannot compose with this completed local prefix."""
        work = self._work or Work()
        if kernel.work.kind != work.kind or kernel.work.units != work.units:
            return "kernel WORK does not realize the node choices", None
        if not kernel.raster.is_direct and not self._raster_eligible:
            return "RASTER requires a tiled contraction site", None
        if kernel.work.producer and not self._producer_eligible:
            return "producer band is incompatible with the selected transport", None
        warp_size = getattr(self.problem.target, "warp_size", 32)
        compute_threads = work.count * (warp_size if work.kind == "warp" else 1)
        producer_threads = kernel.work.producer * warp_size
        if producer_threads > compute_threads:
            return "producer band cannot outnumber the compute band", None
        if compute_threads + producer_threads > getattr(self.problem.target, "max_threads_per_cta", 1024):
            return "worker inventory exceeds the target thread limit", None
        if not kernel.work.producer:
            return None
        for site, assignment in self.assignment.nodes.items():
            if not assignment.tile.is_tiled:
                continue
            if isinstance(assignment, ReductionSchedule) and assignment.reduce.needs_split:
                return "a producer band cannot accompany a cross-CTA reduction", site
            edges = tuple(edge for edge in self.incident_edges(site) if edge in self.stage_edges)
            if not edges or any(self.assignment.edges[edge].stage.transport != "smem-tma" for edge in edges):
                return "a producer band requires TMA transport at every tiled consumer", site
        return None

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

        for site in self.node_sites:
            view = self.views[site]
            assignment = schedule.nodes[site]
            if isinstance(view, Projection) and not isinstance(assignment, ProjectionSchedule):
                self._refuse("projection site requires a projection schedule", site)
            if isinstance(view, Reduction) and not isinstance(assignment, ReductionSchedule):
                self._refuse("reduction site requires a reduction schedule", site)
            if isinstance(assignment.tile, PlacedTile):
                self._refuse("node choices cannot contain placed tile geometry", site)

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
            if site in self.tile_sites and not self._allows_value("TILE", self.node_key("TILE", site), choice.tile.spell()):
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
        if self._pins is None or site not in self.tile_sites:
            return False
        key = self.node_key("TILE", site)
        return bool(self._applicable_pins("TILE", key)) and self._allows_value("TILE", key, tile.spell())

    def _edge_restriction_allows(self, edge: EdgeSite, choice: EdgeSchedule) -> bool:
        """Whether one independent edge value can still satisfy ``c``."""
        return self._pins is None or edge not in self.stage_edges or self._allows_value("STAGE", self.stage_key(edge), choice.stage.spell())

    @property
    def work(self) -> Work | None:
        return self._work


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
        return self._encode(accepted)

    def _encode(self, schedule: ClassicAssignment) -> dict[str, str]:
        """Encode a schedule already accepted by this codec's context traversal."""
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
        schedule = self._parse(row)
        return self._validate_row(schedule, row)

    def _parse(self, row: Mapping[str, str]) -> ClassicAssignment:
        """Parse typed values before a reconstructed TileOp supplies materialization for validation."""
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
        return Schedule(
            KernelSchedule(work, Raster.parse(row["RASTER"])),
            nodes,
            {
                edge: EdgeSchedule(Stage.parse(row[self.context.stage_key(edge)]))
                if edge in self.context.stage_edges
                else EdgeSchedule(Stage.direct())
                for edge in self.context.edge_sites
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
    "ClassicProblem",
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
