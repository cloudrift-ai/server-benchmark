"""Symbolic loop blocking for Tile IR.

A :class:`BlockAxis` says which schedule parameters decompose one logical axis. It carries no
chosen width. ``025_block`` installs one :class:`SiteBlocks` value per Fold site before any
schedule is selected; the classic schedule only binds those parameters.

The table is also the one owner of placement. A contraction's output axes are discovered here,
once, instead of being rediscovered while each schedule candidate is lowered.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.sigma import Sigma


@dataclass(frozen=True)
class BlockAxis:
    """One logical axis and the symbolic schedule parameters that block it.

    ``parameter`` identifies one width shared by every use of the same block. Most axes have a
    site-local parameter. Axes introduced while reassociating a twisted carrier share the
    parameter of their common source axis, which lets the existing TILE choices bind one width
    consistently across the score and value contractions without a BLOCK codec family.
    """

    axis: Axis
    parameter: str
    levels: tuple[str, ...]
    limit: int | None = None

    def __post_init__(self) -> None:
        if not self.parameter or not self.levels:
            raise ValueError("a symbolic block axis requires a parameter and at least one level")

    def partition(self, parts: int, *, outer: str, inner: str | None = None) -> tuple[Axis, Axis, Sigma]:
        """Bind the GRID level to ``parts`` contiguous slices."""
        if type(parts) is not int or parts < 1:
            raise ValueError(f"block partition count must be a positive integer, got {parts!r}")
        extent = self.axis.extent.as_static()
        if extent % parts:
            raise ValueError(f"block partition count {parts} does not divide {self.axis.name} extent {extent}")
        width = extent // parts
        outer_axis = Axis(outer, Dim(parts))
        inner_axis = replace(
            self.axis,
            name=inner or self.axis.name,
            extent=Dim(width),
            window=Window(parent=self.axis.source_axis or self.axis, partition=True),
        )
        index = BinaryExpr("+", BinaryExpr("*", Var(outer_axis.name), Literal(width, "int")), Var(inner_axis.name))
        return outer_axis, inner_axis, Sigma({self.axis.name: index})


@dataclass(frozen=True)
class SiteBlocks:
    """The symbolic output and reduction blocks owned by one Fold schedule site."""

    output: tuple[BlockAxis, BlockAxis] | tuple[()] = ()
    reduce: BlockAxis | None = None

    def __post_init__(self) -> None:
        if self.output and len(self.output) != 2:
            raise ValueError("a contraction block requires exactly two output axes")

    def claims(self, choice, geometry=None, stage=None) -> tuple[BlockClaim, ...]:
        """Bind this site's parameters from one existing schedule choice.

        No value is selected here. TILE and REDUCE already selected the unit, register, atom and
        grid widths, while STAGE already resolved a scalar copy chunk. This method only projects
        those values onto the symbolic block parameters declared by the pass.
        """
        out = []
        tile = choice.tile
        if self.output:
            for index, part in enumerate(self.output):
                if geometry is None:
                    if part.limit is not None:
                        out.append(BlockClaim(part.parameter, 1, part.limit))
                    continue
                side = geometry.mn[index]
                width = side.reg * side.atom if part.limit is not None else side.tile
                out.append(BlockClaim(part.parameter, width, part.limit))
        if self.reduce is not None:
            reduction = getattr(choice, "reduce", None)
            axis = self.reduce.axis
            structural = axis.window is not None and axis.window.block
            if structural and axis.step is not None:
                if not isinstance(axis.step, Literal):
                    raise ValueError("a structural block step must be static before scheduling")
                width = axis.step.value
            elif structural and not self.output:
                width = axis.extent.as_static()
            elif tile.is_warp:
                width = tile.atom.atom_k * tile.bk
            elif stage is not None and stage.bk_elems:
                width = stage.bk_elems
            elif reduction is not None:
                width = reduction.parallel
            else:
                width = 1
            out.append(BlockClaim(self.reduce.parameter, width, self.reduce.limit))
        return tuple(out)


@dataclass(frozen=True)
class BlockClaim:
    """One concrete width supplied to a symbolic block parameter by a schedule choice."""

    parameter: str
    width: int
    limit: int | None = None

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 1:
            raise ValueError(f"block width must be a positive integer, got {self.width!r}")

    @property
    def accepted(self) -> bool:
        """Whether the chosen width respects the form's fixed compatibility bound."""
        return self.limit is None or self.width == self.limit


@dataclass(frozen=True)
class BoundBlockAxis:
    """One symbolic axis after the schedule has supplied its block width."""

    symbolic: BlockAxis
    width: int
    name: str = "_ks"
    factors: tuple[tuple[str, int], ...] = ()
    transposed: bool = False

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 1:
            raise ValueError(f"block width must be a positive integer, got {self.width!r}")
        if self.symbolic.limit is not None and self.width != self.symbolic.limit:
            raise ValueError(f"block {self.symbolic.parameter!r} requires width {self.symbolic.limit}, got {self.width}")
        if any(level not in self.symbolic.levels or type(width) is not int or width < 1 for level, width in self.factors):
            raise ValueError("bound block factors must be positive widths of declared levels")
        if len({level for level, _ in self.factors}) != len(self.factors):
            raise ValueError("a bound block level may be supplied only once")
        product = 1
        for _, factor in self.factors:
            product *= factor
        if self.factors and product != self.width:
            raise ValueError(f"bound block factors cover {product} elements, expected {self.width}")

    @property
    def axis(self) -> Axis:
        """The outer axis walking the logical source one block at a time."""
        source = self.symbolic.axis
        return Axis(
            self.name,
            source.extent,
            window=Window(parent=source.source_axis or source, block=True),
            step=Literal(self.width, "int"),
        )

    @property
    def inner(self) -> Axis:
        """The local coordinate within one block."""
        source = self.symbolic.axis
        return Axis(f"{self.name}_i", self.width, window=Window(parent=source.source_axis or source, block=True))

    @property
    def extent(self) -> int | Dim:
        """The source extent in the form consumed by kernel lowering."""
        extent = self.symbolic.axis.extent
        return extent.as_static() if extent.is_static else extent

    @property
    def chunks(self) -> int | Dim:
        """The number of blocks covering the source axis."""
        extent = self.symbolic.axis.extent
        return -(-extent.as_static() // self.width) if extent.is_static else extent.ceil_div(self.width)

    def factor(self, level: str) -> int:
        """The bound width at one declared level, or one when the level is inactive."""
        if level not in self.symbolic.levels:
            raise ValueError(f"block {self.symbolic.parameter!r} has no {level!r} level")
        return dict(self.factors).get(level, 1)

    def on(self, symbolic: BlockAxis) -> BoundBlockAxis:
        """Use this shared parameter binding on another declaration of the same block."""
        if symbolic.parameter != self.symbolic.parameter:
            raise ValueError(
                f"cannot move block parameter {self.symbolic.parameter!r} onto {symbolic.parameter!r}"
            )
        return replace(self, symbolic=replace(self.symbolic, axis=symbolic.axis, limit=symbolic.limit))

    @property
    def lane(self) -> Axis | None:
        """The cooperative BLOCK axis of a scalar reduction, when active."""
        width = self.factor("block")
        return Axis(f"{self.symbolic.axis.name}_co", width) if width > 1 else None

    def loop(self, axis: Axis, lane: Axis | None = None) -> BoundBlockLoop:
        """Bind a serial reduce loop to its BLOCK and REGISTER decomposition."""
        coop = self.factor("block")
        return self._loop(axis, coop, lane)

    def _loop(self, axis: Axis, coop: int, lane: Axis | None) -> BoundBlockLoop:
        registers = self.factor("register")
        if (lane is None) != (coop == 1) or (lane is not None and lane.extent != coop):
            raise ValueError("reduce loop lane does not match its bound BLOCK level")
        source_step = axis.step.value if isinstance(axis.step, Literal) else 1
        parallel = coop * registers
        trips = axis.trips
        masked = registers > 1 and not (trips is not None and trips % parallel == 0)
        start = (
            Literal(0, "int")
            if lane is None
            else BinaryExpr("*", Var(lane.name), Literal(source_step, "int"))
            if source_step > 1
            else Var(lane.name)
        )
        return BoundBlockLoop(
            axis=axis,
            lane=lane,
            registers=registers,
            register_span=coop * source_step,
            stride=parallel * source_step,
            masked=masked,
            start=start,
        )

    def transposed_loop(self, axis: Axis, output: Axis, *, warp_size: int = 32) -> BoundTransposedLoop:
        """Bind the transposed cooperative reduction's K and output lane axes."""
        if not self.transposed:
            raise ValueError("a non-transposed block has no transposed loop")
        coop = self.factor("block")
        if coop % warp_size:
            raise ValueError(f"transposed BLOCK width {coop} must be a multiple of warp size {warp_size}")
        k_ways = coop // warp_size
        output_lane = Axis(f"{output.name}_ln", warp_size)
        k_lane = Axis(f"{axis.name}_co", k_ways) if k_ways > 1 else None
        output_block = Axis(f"{output.name}_blk", output.extent.ceil_div(warp_size), window=Window(parent=output))
        cell = BinaryExpr(
            "+",
            BinaryExpr("*", Var(output_block.name), Literal(warp_size, "int")),
            Var(output_lane.name),
        )
        overhang = not (output.extent.is_static and output.extent.as_static() % warp_size == 0)
        return BoundTransposedLoop(
            loop=self._loop(axis, k_ways, k_lane),
            output_block=output_block,
            output_lane=output_lane,
            cell=cell,
            overhang=overhang,
            threads=coop,
        )


@dataclass(frozen=True)
class BoundBlockLoop:
    """The concrete serial loop left after a bound reduction block is distributed."""

    axis: Axis
    lane: Axis | None
    registers: int
    register_span: int
    stride: int
    masked: bool
    start: object


@dataclass(frozen=True)
class BoundTransposedLoop:
    """A transposed BLOCK decomposition across K groups and output lanes."""

    loop: BoundBlockLoop
    output_block: Axis
    output_lane: Axis
    cell: object
    overhang: bool
    threads: int


@dataclass(frozen=True)
class BoundSiteBlocks:
    """One site's block axes after materialization."""

    output: tuple = ()
    reduce: BoundBlockAxis | None = None


def bind_site(blocks: SiteBlocks, choice, geometry=None, stage=None, *, name: str = "_ks") -> BoundSiteBlocks:
    """Bind a site's declared block parameters from its existing schedule choice."""
    widths = {claim.parameter: claim.width for claim in blocks.claims(choice, geometry, stage)}
    reduce = None
    if blocks.reduce is not None:
        try:
            width = widths[blocks.reduce.parameter]
        except KeyError:
            raise ValueError(f"schedule did not bind block parameter {blocks.reduce.parameter!r}") from None
        tile = choice.tile
        reduction = getattr(choice, "reduce", None)
        if tile.is_warp:
            factors = (("tile", tile.bk), ("atom", tile.atom.atom_k))
            transposed = False
        elif stage is not None and stage.bk_elems:
            factors = (("tile", stage.bk_elems), ("atom", 1))
            transposed = False
        elif reduction is not None:
            factors = (("grid", reduction.cta), ("block", reduction.coop), ("register", reduction.reg))
            transposed = reduction.coop_transposed
        else:
            factors = ()
            transposed = False
        reduce = BoundBlockAxis(blocks.reduce, width, name, factors, transposed)
    output = geometry.mn if blocks.output and geometry is not None else ()
    return BoundSiteBlocks(output, reduce)


def _parameter(axis: Axis, site: int, role: str) -> tuple[str, int | None]:
    window = axis.window
    if window is not None and window.block and window.parent is not None:
        width = axis.step.value if isinstance(axis.step, Literal) else axis.extent.as_static()
        return f"{window.parent.name}.block", width
    return f"b{site}.{role}", None


def _output_axes(tile, node) -> tuple[Axis, Axis] | None:
    """Derive the contraction's ``(m, n)`` axes once, during blockification."""
    place = tile.place.on_grid()
    free = tuple(place.free)
    site = tile.sites[tile.node_id(node)]
    ancestors = tuple(candidate for candidate in tile.sites if site.under(candidate))

    def orient(mn):
        view = node.as_contraction()
        if mn is None or view is None:
            return mn
        order = {axis.name: (position, axis) for position, axis in enumerate((*place.free, *place.grid))}
        left = max((order[name] for name in view.left_axes if name in order), default=None)
        right = max((order[name] for name in view.right_axes if name in order), default=None)
        if left is not None and right is not None:
            return left[1], right[1]
        first, second = mn
        return (second, first) if second.name == view.left and first.name != view.left else mn

    if all(candidate.node.axis is None for candidate in ancestors):
        return orient(place.root_mn)
    if len(free) < 2:
        return None
    parent = next(
        (
            found
            for depth in range(len(site.hops) - 1, -1, -1)
            if (found := next((candidate for candidate in tile.sites if candidate.hops == site.hops[:depth]), None)) is not None
            and found.node.axis is not None
        ),
        None,
    )
    axis = tile.axis_of(parent.node.axis) if parent is not None else None
    if axis is None:
        return None
    return orient((free[-2], axis.window.parent if axis.window is not None and not axis.window.block else axis))


def blockify(tile) -> tuple[SiteBlocks, ...]:
    """Declare every Fold site's symbolic axis decomposition."""
    out = []
    for site, record in enumerate(tile.sites):
        node = record.node
        output = ()
        if node.as_contraction() is not None and (mn := _output_axes(tile, node)) is not None:
            blocks = []
            for role, axis in zip(("m", "n"), mn, strict=True):
                parameter, limit = _parameter(axis, site, role)
                blocks.append(BlockAxis(axis, parameter, ("grid", "unit", "register", "atom"), limit))
            output = tuple(blocks)
        reduce = None
        if node.axis is not None:
            axis = tile.axis_of(node.axis)
            parameter, limit = _parameter(axis, site, "k")
            levels = (
                ("grid", "block", "register", "tile", "atom", "serial")
                if node.as_contraction() is not None
                else ("grid", "block", "register", "serial")
            )
            reduce = BlockAxis(axis, parameter, levels, limit)
        out.append(SiteBlocks(output=output, reduce=reduce))
    return tuple(out)


__all__ = [
    "BlockAxis",
    "BlockClaim",
    "BoundBlockAxis",
    "BoundBlockLoop",
    "BoundSiteBlocks",
    "BoundTransposedLoop",
    "SiteBlocks",
    "bind_site",
    "blockify",
]
