"""Tile schedule type system — how a kernel's axes bind to the hardware, and the codecs that ser/des
the schedule knobs.

This root ``ir`` module is the merge of the former ``ir/tile/{codec,role,schedule}.py``. The schedule
value types are used by both the tile IR and the kernel materializer, so they live at the ir root
beside :mod:`~emmy.compiler.ir.atom`, not under ``ir/tile``.

**The schedule is separate from the combine.** The combine (the ⊕) lives in the op tree
(:mod:`emmy.compiler.ir.pure.algebra` + :mod:`~emmy.compiler.ir.tile.ir`). This module owns the
leaf choice types (:class:`Reduce` / :class:`Tile` / :class:`Stage` / :class:`WarpSpec` plus
:class:`Placement`); :mod:`emmy.compiler.ir.classic_schedule` composes them into the accepted
kernel, node, and edge assignment stored on ``TileOp.classic``. The Fold term itself carries no
schedule field.

A reduction's only freedom is **how the reduce axis is partitioned across hardware levels**
(:class:`Reduce`); the combine *mechanism* at each level is **derived** from the level
(:meth:`ReduceStage.combine`), and the combine *algebra* rides the carrier (the ``Twist``). So the
same op + the same materializer extend across kernel kinds — only the carrier and the partition
change.

The schedule is **flat and kind-free**: a kernel's structure is read from its node's derived
:class:`~emmy.compiler.ir.axis.AxisRole` (``Fold.role``), not a Python type, so a pointwise
cell, a ``PLANAR`` / ``TWISTED`` reduce, and a ``CONTRACTION`` contraction all schedule through the
same value types and use the subset their axes admit. Warp specialization is **orthogonal** — an
optional ``workers: WarpSpec | None`` root field (``None`` = uniform SIMT), a producer band split
*over* the fixed pipeline.

**The codec values are SITE-LOCAL**: a kernel's worker inventory is spelled exactly once, in the
``WORK`` family (:class:`Work`), so a ``TILE`` value carries no unit widths
(``<atom>/f<FM>x<FN>[/k<bk>]`` warp | ``f<fn>[x<fm>]`` scalar) and a ``REDUCE`` value no coop width
(``[g<n>[a|k]][/coop[-t]][/r<n>]``). Both therefore ``parse`` **against** a :class:`Work` and are
hand-written here. The empty ``TILE`` is only the per-cell choice; a parallel unit-register thread
tile spells ``f1``, so the site value and ``WORK`` decode without cross-family inference. There is
no second, self-contained reading — the retired embedded-worker grammar
(``a:<atom>/w../f..``, ``n../f..``, ``b<n>``) raises.

``STAGE`` — the one codec that needs no inventory — parses and spells by hand for the same reason,
so **every** codec here is now read the same way: one canonical ``/``-separated token string, with
one token order and each field binding at most once. There is no schema engine any more; it was
built to serve four codecs, and the site-local rewrite left it serving one.

The tunable knob *declarations* that spell the live codecs (``WORK`` / ``TILE`` / ``STAGE`` /
``REDUCE``) live one layer up in :mod:`emmy.compiler.pipeline.search.space` — they are ``Knob``
instances, a search concern; the retired ``WSPEC`` knob is gone from that list, its producer band
absorbed into ``WORK``'s ``+p<n>`` and its one surviving field into :class:`WarpSpec`.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from dataclasses import replace as dc_replace

from emmy.compiler.ir.atom import SCALAR_ATOM, Atom, AtomKind, atom_for
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Expr, Literal

# ===========================================================================================
# The one rule every codec below shares. Each value type parses and spells its own grammar by
# hand — the site-local TILE / REDUCE against a Work inventory, STAGE on its own.
# ===========================================================================================


def _codec_width(num: str, *, tok: str, codec: str) -> int:
    """Parse a codec field's positive-integer width, rejecting empty / non-numeric / ``< 1``
    values with a clear message. A semantic width of ``1`` is the legal identity but is omitted
    from the canonical wire spelling. The ``codec`` name rides the message so a bad pin names the
    knob it came from (``bad REDUCE token ...`` / ``bad TILE token ...``)."""
    if not num.isdigit() or int(num) < 1:
        raise ValueError(f"bad {codec} token {tok!r}: expected a positive integer width, got {num!r}")
    return int(num)


def _canonical_choice(codec: str, spec: str | None, choice):
    """Return ``choice`` only when ``spec`` is its one wire spelling."""
    if spec is not None and spec != choice.spell():
        raise ValueError(f"{codec} spelling {spec!r} is not canonical; expected {choice.spell()!r}")
    return choice


# ===========================================================================================
# Schedule value types — the codec value objects that ride the structural nodes + TileOp.
# ===========================================================================================


class Level(enum.Enum):
    """One hardware level the reduce axis can be partitioned across, coarse→fine."""

    GRID = "grid"  # across CTAs (split-K) — realized by the structural 035_split_reduce fork, never the in-kernel walk
    BLOCK = "block"  # cooperative threads within a CTA (warp shuffle / smem tree)
    REG = "reg"  # ILP register-fold accumulators
    SERIAL = "serial"  # the per-thread serial remainder (never spelled — derived)


class FoldMove(enum.Enum):
    """The per-level combine *mechanism* — the placement-keyed fold MOVE, derived from the
    :class:`Level` (where the reduced axis sits), never tuned and never re-decided at a consumer.
    :meth:`ReduceStage.combine` is the ONE selector; every fold emitter consumes its output —
    ``_factor.emit_combine`` (SHFL butterfly / SMEM tree at scalar residence) and
    ``035_split_reduce`` (the cross-CTA ATOMIC / KERNEL finalize as a graph rewrite)."""

    SERIAL = "serial"  # no cross-unit combine (the serial / reg remainder)
    REG = "reg"  # register tree (ILP) — TODO(reg)
    SHFL = "shfl"  # lane-level ``__shfl_xor_sync`` butterfly (within-warp)
    SMEM = "smem"  # cross-warp / block-wide smem tree-halve (within-block)
    ATOMIC = "atomic"  # cross-CTA ``atomicAdd`` finalize (035_split_reduce's one-kernel arm)
    KERNEL = "kernel"  # cross-CTA workspace + deferred sibling combine kernel (035_split_reduce)


@dataclass(frozen=True)
class ReduceStage:
    """One level's **tuned** partition: a ``width`` of partials at a hardware ``level``.

    The combine *mechanism* is **derived** (:meth:`combine`), not stored — the level
    implies the fold, and a BLOCK width derives warp-shuffle vs hierarchical-smem from the
    warp size. ``width`` is power-of-two for BLOCK (the butterfly / tree reorder).

    ``finalize`` is meaningful only at ``GRID``: how ``035_split_reduce`` realizes the cross-CTA
    combine — ``"atomic"`` (the partial kernel ``atomicAdd``\\ s into the output, one kernel,
    additive carriers only) or ``"kernel"`` (a deferred sibling combine kernel over a
    workspace, the only legal arm for a twisted carrier). The ``g<n>[a|k]`` codec letter."""

    level: Level
    width: int = 1
    finalize: str = "kernel"  # GRID only: "atomic" | "kernel" (the g<n> finalize letter)
    # BLOCK only (the ``coop-t`` codec token): swap the cooperative thread mapping for a
    # k-major B operand — warp lanes sweep the OUTPUT axis (contiguous B loads at every k
    # step) and the k partition rides the upper thread bits; the combine becomes the
    # lane-indexed smem tree across k-slices (no shuffle stage — each lane holds a
    # different output). The interleaved default keeps lanes on the reduce axis.
    transposed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.level, Level):
            raise TypeError("ReduceStage level must be a Level")
        if type(self.width) is not int or self.width < 1:
            raise ValueError(f"ReduceStage width must be a positive integer, got {self.width!r}")
        if self.finalize not in ("atomic", "kernel"):
            raise ValueError(f"ReduceStage finalize must be atomic or kernel, got {self.finalize!r}")
        if self.level is not Level.GRID and self.finalize != "kernel":
            raise ValueError("only a GRID ReduceStage can select atomic finalization")
        if type(self.transposed) is not bool:
            raise TypeError("ReduceStage transposed must be a bool")
        if self.level is not Level.BLOCK and self.transposed:
            raise ValueError("only a BLOCK ReduceStage can transpose its cooperative mapping")

    def combine(self, *, warp_size: int, segmented: bool = False) -> tuple[FoldMove, ...]:
        """The derived per-level combine fold(s), fine→coarse within this stage — the ONE
        placement-keyed move selector every fold emitter consumes (see :class:`FoldMove`).

        - ``SERIAL`` / ``REG`` → ``()`` (no cross-unit combine; REG-fold is TODO(reg)).
        - ``GRID`` → ``(ATOMIC,)`` or ``(KERNEL,)`` per the ``finalize`` letter (the split-K
          cross-CTA move — realized by the structural ``035_split_reduce`` fork; the carrier /
          projection refusals live beside that offer, which alone holds the context).
        - ``BLOCK`` → the intra-CTA hierarchy: a lone ``SHFL`` when ``segmented`` (the
          per-row segmented butterfly for strided-cooperative rows) or ``width ≤ warp``
          (one warp); ``(SHFL, SMEM)`` when ``width`` is a clean warp multiple (lanes then
          the cross-warp tree); else a standalone ``(SMEM,)`` block tree. Power-of-two
          ``width`` required."""
        if self.level in (Level.SERIAL, Level.REG):
            return ()
        if self.level is Level.GRID:
            return (FoldMove.ATOMIC,) if self.finalize == "atomic" else (FoldMove.KERNEL,)
        # BLOCK.
        w = self.width
        if w & (w - 1):
            raise ValueError(f"BLOCK reduce width must be a power of two, got {w}")
        if segmented or w <= warp_size:
            return (FoldMove.SHFL,)
        if w % warp_size == 0:
            return (FoldMove.SHFL, FoldMove.SMEM)
        return (FoldMove.SMEM,)


@dataclass(frozen=True)
class Reduce:
    """The kernel's single reduce partition — the **tuned widths only**, coarse→fine.

    There is one reduce fold per kernel (1:1 and singular — the fold owns the axis),
    so the plan holds no axis; the per-thread ``serial`` remainder is derived by the
    materializer as ``ceil(extent / parallel)``. ``stages=()`` is the scalar serial fold
    (today's one-thread-per-cell tier)."""

    stages: tuple[ReduceStage, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.stages, tuple) or any(not isinstance(stage, ReduceStage) for stage in self.stages):
            raise TypeError("Reduce stages must be a tuple of ReduceStage values")
        levels = tuple(stage.level for stage in self.stages)
        canonical = tuple(level for level in (Level.GRID, Level.BLOCK, Level.REG) if level in levels)
        if levels != canonical or any(stage.width == 1 for stage in self.stages):
            raise ValueError("Reduce stages must be unique, active, and ordered GRID -> BLOCK -> REG")

    @classmethod
    def of(cls, *, cta: int = 1, coop: int = 1, reg: int = 1, finalize: str = "kernel", coop_transposed: bool = False) -> Reduce:
        """Build a plan from per-level widths (1 = absent). Order is coarse→fine:
        GRID (cta) → BLOCK (coop) → REG (reg). ``finalize`` rides the GRID stage;
        ``coop_transposed`` rides the BLOCK stage (the ``coop-t`` k-major lane swap)."""
        if any(type(width) is not int or width < 1 for width in (cta, coop, reg)):
            raise ValueError(f"Reduce widths must be positive integers, got cta={cta!r}, coop={coop!r}, reg={reg!r}")
        if finalize not in ("atomic", "kernel"):
            raise ValueError(f"Reduce finalize must be atomic or kernel, got {finalize!r}")
        if type(coop_transposed) is not bool:
            raise TypeError("Reduce coop_transposed must be a bool")
        if cta == 1 and finalize != "kernel":
            raise ValueError("Reduce finalization is meaningful only when cta > 1")
        if coop == 1 and coop_transposed:
            raise ValueError("Reduce cooperative transposition is meaningful only when coop > 1")
        stages: list[ReduceStage] = []
        if cta > 1:
            stages.append(ReduceStage(Level.GRID, cta, finalize=finalize))
        if coop > 1:
            stages.append(ReduceStage(Level.BLOCK, coop, transposed=coop_transposed))
        if reg > 1:
            stages.append(ReduceStage(Level.REG, reg))
        return cls(tuple(stages))

    def spell(self) -> str:
        """The ``REDUCE`` codec value for this plan — the pipeline coarse→fine, SITE-LOCAL: the
        coop WIDTH lives in the kernel's ``WORK`` inventory, never here, so the value is
        ``[g<n>[a|k]][/coop[-t]][/r<n>]``. ``""`` for the scalar serial fold (the per-thread
        serial remainder is never spelled — it derives as ``ceil(extent / parallel)``). The GRID
        finalize letter IS kept: ``a``/``k`` is the atomic-vs-deferred finalize MODE, a site-local
        fact — ``g4a`` and ``g2k`` are semantically different rows, both live in the golden
        corpus."""
        parts: list[str] = []
        if self.cta > 1:
            parts.append(f"g{self.cta}{'a' if self.finalize == 'atomic' else 'k'}")
        if self.coop > 1:
            parts.append("coop-t" if self.coop_transposed else "coop")
        if self.reg > 1:
            parts.append(f"r{self.reg}")
        return "/".join(parts)

    @classmethod
    def parse(cls, spec: str | None, work: Work | None) -> Reduce:
        """Decode a ``REDUCE`` value against the kernel's ``WORK`` inventory (inverse of
        :meth:`spell` — the coop width is ``work.count``, never the string). Empty / ``None`` =
        the scalar serial fold. An unknown token raises, so a width-carrying spelling (the retired
        ``b<n>`` embedded-worker grammar) is a loud error, not a silent second reading."""
        s = (spec or "").strip()
        cta, coop, reg, finalize, transposed = 1, 1, 1, "kernel", False
        for t in s.split("/") if s else ():
            if t.startswith("g"):
                body = t[1:]
                if body.endswith(("a", "k")):
                    finalize = "atomic" if body[-1] == "a" else "kernel"
                    body = body[:-1]
                cta = _codec_width(body, tok=t, codec="REDUCE")
            elif t in ("coop", "coop-t"):
                if work is None or work.kind != "thread":
                    raise ValueError(f"REDUCE {spec!r}: 'coop' requires a thread WORK inventory (t<N>)")
                coop, transposed = work.count, t.endswith("-t")
            elif t.startswith("r") and t[1:].isdigit():
                reg = _codec_width(t[1:], tok=t, codec="REDUCE")
            else:
                raise ValueError(f"REDUCE {spec!r}: unknown token {t!r} (expect g<n>[a|k] / coop[-t] / r<n>)")
        return _canonical_choice(
            "REDUCE",
            spec,
            cls.of(cta=cta, coop=coop, reg=reg, finalize=finalize, coop_transposed=transposed),
        )

    @property
    def parallel(self) -> int:
        """The total parallel degree = ∏ stage widths (the lane/CTA fan-out the serial
        remainder divides into)."""
        p = 1
        for s in self.stages:
            p *= s.width
        return p

    @property
    def needs_split(self) -> bool:
        """True iff any stage is a cross-CTA GRID split (``035_split_reduce`` territory)."""
        return any(s.level is Level.GRID for s in self.stages)

    def _width(self, level: Level) -> int:
        for s in self.stages:
            if s.level is level:
                return s.width
        return 1

    @property
    def coop(self) -> int:
        """The BLOCK (cooperative-thread) width, or 1 if no BLOCK stage."""
        return self._width(Level.BLOCK)

    @property
    def cta(self) -> int:
        """The GRID (cross-CTA split) width, or 1 if no GRID stage."""
        return self._width(Level.GRID)

    @property
    def finalize(self) -> str:
        """The GRID stage's cross-CTA finalize — ``"atomic"`` | ``"kernel"`` (``"kernel"``
        if no GRID stage; the value is only meaningful when :attr:`needs_split`)."""
        for s in self.stages:
            if s.level is Level.GRID:
                return s.finalize
        return "kernel"

    @property
    def reg(self) -> int:
        """The REG (ILP fold) width, or 1 if no REG stage."""
        return self._width(Level.REG)

    @property
    def coop_transposed(self) -> bool:
        """True iff the BLOCK stage carries the ``coop-t`` k-major lane swap."""
        return any(s.transposed for s in self.stages if s.level is Level.BLOCK)


@dataclass(frozen=True)
class Tile:
    """The contraction's output tile — **one descriptor for both tiers**, discriminated by
    :attr:`atom`: a tensor-core :class:`AtomKind` (the warp mma tile) or the scalar
    :class:`~emmy.compiler.ir.atom.ScalarAtom` (the register sub-tile, the ``Scalar``
    fragment). Each tiled output axis splits into a **unit** width (warps for mma / parallel threads
    for scalar — :attr:`units`) and a **register** width (atom sub-cells / register cells per unit —
    :attr:`regs`); ``bk`` chunks the K (contraction) axis (mma only). The all-``1`` scalar tile (a
    scalar ``atom``, ``units``/``regs`` ``(1, 1)``) is the per-cell tier — one thread per output cell.

    ``units`` / ``regs`` are stored ``(m, n)`` — OUTER axis first — on both tiers, and the tier's
    own codec order is applied in :meth:`spell` / :meth:`parse`, which is where the wire format
    belongs (the warp value spells m-then-n, the scalar value n-then-m). They were stored in each
    tier's native order instead, with four accessors existing only to undo the flip and every raw
    ``regs[0]`` three files away needing a comment to say which axis it meant. Spelled by the
    unified ``TILE`` knob — the warp form ``<atom>/f<FM>x<FN>[/k<bk>]`` or the scalar
    ``f<fn>[x<fm>]``; the unit widths ride the kernel's ``WORK`` inventory, so :meth:`parse` needs
    it and :attr:`is_warp` discriminates the tier on the object. Decided by the tile schedule."""

    atom: Atom = SCALAR_ATOM
    units: tuple[int, int] = (1, 1)  # (m, n): warp (WM, WN) / scalar (par_m, par_n)
    regs: tuple[int, int] = (1, 1)  # (m, n): warp (FM, FN) / scalar (reg_m, reg_n)
    bk: int = 1  # K-chunk per inner mma step, in atom_k units (mma only; 1 for scalar)

    def __post_init__(self) -> None:
        if not isinstance(self.atom, (AtomKind, type(SCALAR_ATOM))):
            raise TypeError("Tile atom must be an AtomKind or the scalar atom")
        if not isinstance(self.atom, AtomKind) and self.atom != SCALAR_ATOM:
            raise ValueError("Tile scalar atom must use the canonical scalar value")
        if isinstance(self.atom, AtomKind) and atom_for(self.atom.name) != self.atom:
            raise ValueError(f"Tile atom {self.atom.name!r} is not the registered canonical value")
        for field, widths in (("units", self.units), ("regs", self.regs)):
            if not isinstance(widths, tuple) or len(widths) != 2 or any(type(width) is not int or width < 1 for width in widths):
                raise ValueError(f"Tile {field} must be two positive integer widths, got {widths!r}")
        if type(self.bk) is not int or self.bk < 1:
            raise ValueError(f"Tile bk must be a positive integer, got {self.bk!r}")
        if not self.is_warp and self.bk != 1:
            raise ValueError("a scalar Tile cannot carry an mma K chunk")

    def spell(self) -> str:
        """The ``TILE`` codec value for this tile — SITE-LOCAL: the worker tokens live in the
        kernel-global ``WORK`` family, so the warp form is ``<atom>/f<FM>x<FN>[/k<bk>]`` (the atom
        name bare — the tier discriminator IS the worker kind, never a per-``TILE`` spelling) and
        the scalar form ``f<fn>[x<fm>]``. The per-cell tier alone spells ``""``; a parallel
        unit-register thread tile spells ``f1`` so the wire representation is unambiguous."""
        if self.is_warp:
            k = f"/k{self.bk}" if self.bk > 1 else ""
            return f"{self.atom.name}/f{self.reg_m}x{self.reg_n}{k}"
        if not any(v > 1 for v in self.regs):
            return "f1" if any(v > 1 for v in self.units) else ""
        # The scalar value is n-then-m — the codec's order, applied here rather than stored.
        return f"f{self.reg_n}" if self.reg_m == 1 else f"f{self.reg_n}x{self.reg_m}"

    @classmethod
    def parse(cls, spec: str | None, work: Work | None) -> Tile:
        """Decode a ``TILE`` value against the kernel's ``WORK`` inventory (inverse of
        :meth:`spell` — the unit widths come from ``work``, never the string): the warp form
        ``<atom>/f<FM>x<FN>[/k<bk>]`` or the scalar ``f<fn>[x<fm>]``. Empty / ``None`` is the
        per-cell tier; ``f1`` is the unit-register thread tile when ``WORK`` supplies a parallel
        thread inventory. An unknown token raises, so retired
        aliases and embedded-worker spellings (``a:<atom>/w../f..``, ``n../f..``) are loud errors,
        not silent second readings."""
        s = spec or ""
        if not s:
            return _canonical_choice("TILE", spec, cls())
        tokens = s.split("/")
        # ``Work.units`` is the WORK codec's own order (warp m-then-n, thread n-then-m); a plan
        # stores (m, n), so the thread inventory transposes on the way in.
        units = (1, 1) if work is None else (tuple(work.units) if work.kind == "warp" else (work.units[1], work.units[0]))
        atom = None
        if not tokens[0].startswith(("f", "k")):
            try:  # a non-atom leading token is a retired ``n<N>x<M>`` / ``a:<atom>`` spelling
                atom = atom_for(tokens[0])
            except ValueError as e:
                raise ValueError(f"TILE {spec!r}: {e}") from None
            if work is None or work.kind != "warp":
                raise ValueError(f"TILE {spec!r}: a warp atom requires a warp WORK inventory (w<M>x<N>)")
            tokens = tokens[1:]
        regs, bk = (1, 1), 1
        for t in tokens:
            if t.startswith("f") and t[1:2].isdigit():
                d = t[1:].split("x")
                if len(d) > 2:
                    raise ValueError(f"bad TILE token {t!r}: expected ≤2 dims, got {len(d)}")
                widths = [_codec_width(x, tok=t, codec="TILE") for x in d]
                fst, snd = widths[0], (widths[1] if len(widths) == 2 else 1)
                regs = (fst, snd) if atom is not None else (snd, fst)  # warp f<FM>x<FN> / scalar f<fn>[x<fm>]
            elif atom is not None and t.startswith("k") and t[1:].isdigit():
                bk = _codec_width(t[1:], tok=t, codec="TILE")
            else:
                raise ValueError(f"TILE {spec!r}: unknown token {t!r} (expect [<atom>/]f<fn>[x<fm>][/k<bk>])")
        choice = cls(units=units, regs=regs) if atom is None else cls(atom=atom, units=units, regs=regs, bk=bk)
        return _canonical_choice("TILE", spec, choice)

    @property
    def is_warp(self) -> bool:
        """True iff this is the tensor-core (mma) tile — :attr:`atom` is a real :class:`AtomKind`."""
        return isinstance(self.atom, AtomKind)

    @property
    def is_tiled(self) -> bool:
        """True iff this materializes a tile: a warp tile always does; a scalar tile only when some
        unit / register width > 1 (else it is the per-cell tier — one thread per output cell)."""
        return self.is_warp or any(v > 1 for v in (*self.units, *self.regs))

    @property
    def units_m(self) -> int:
        """Units on the outer (m) output axis — ``WM`` (warp) / ``par_m`` (scalar)."""
        return self.units[0]

    @property
    def units_n(self) -> int:
        """Units on the inner (n) output axis — ``WN`` (warp) / ``par_n`` (scalar)."""
        return self.units[1]

    @property
    def reg_m(self) -> int:
        """Register sub-cells on the outer (m) axis — ``FM`` (warp) / ``reg_m`` (scalar)."""
        return self.regs[0]

    @property
    def reg_n(self) -> int:
        """Register sub-cells on the inner (n) axis — ``FN`` (warp) / ``reg_n`` (scalar)."""
        return self.regs[1]

    @property
    def block_threads(self) -> int:
        """The per-CTA thread count = ``units_m · units_n · atom.lanes`` (mma: ``WM·WN·32``;
        scalar: ``par_n·par_m·1``)."""
        return self.units[0] * self.units[1] * self.atom.lanes

    @property
    def tile_m(self) -> int:
        """The per-CTA output rows = ``units_m · reg_m · atom_m``."""
        return self.units_m * self.reg_m * self.atom.atom_m

    @property
    def tile_n(self) -> int:
        """The per-CTA output cols = ``units_n · reg_n · atom_n``."""
        return self.units_n * self.reg_n * self.atom.atom_n

    @property
    def launch_threads(self) -> int | None:
        """:attr:`block_threads`, or ``None`` for the scalar default block size — the reading the
        launch / WSPEC gates want (``1`` thread per CTA is "unset", not a real block)."""
        bt = self.block_threads
        return bt if bt > 1 else None

    def at(self, m_axis: Axis, n_axis: Axis) -> PlacedTile:
        """Derive placed geometry without adding axes to this choice value."""
        return PlacedTile(self, (m_axis, n_axis))

    def placed_on(self, place: Placement) -> PlacedTile | Tile:
        """This tile bound to the ROOT contraction's ``(m, n)`` — :attr:`Placement.root_mn` — and
        left UNPLACED when the grid cannot supply the pair. The scheduler binds through here at
        option assembly and :meth:`Sched.tile_of` re-derives the same pair for a reader that takes
        the slice off the tree, so the depth-1 rule and its degradation are stated ONCE."""
        axes = place.root_mn
        return self.at(*axes) if axes is not None else self


@dataclass(frozen=True)
class PlacedTile:
    """Materialization geometry derived from an axis-free :class:`Tile` choice and two axes."""

    choice: Tile
    axes: tuple[Axis, Axis]

    def __post_init__(self) -> None:
        if not isinstance(self.choice, Tile):
            raise TypeError("PlacedTile choice must be a Tile")
        if not isinstance(self.axes, tuple) or len(self.axes) != 2 or any(not isinstance(axis, Axis) for axis in self.axes):
            raise TypeError("PlacedTile axes must be two Axis values")

    def __getattr__(self, name: str):
        """Expose choice geometry to lowering without duplicating its fields."""
        return getattr(object.__getattribute__(self, "choice"), name)

    # ---- the (m, n) output sides: each axis paired with its derived per-CTA tile geometry (width /
    # unit / register counts) + the bound block/unit var names (the original m/n names live in the
    # operand indices, so the bound axes take a fresh ``_b`` / ``_u`` suffix). --------------------- #
    @property
    def mn(self) -> tuple[Side, Side]:
        """The ``(m, n)`` output :class:`Side` pair — each placed axis with its derived geometry.
        Requires :meth:`at` to have bound the placement; :attr:`m` / :attr:`n` index it."""
        dims = ((self.tile_m, self.units_m, self.reg_m), (self.tile_n, self.units_n, self.reg_n))
        return tuple(
            Side(axis=ax, tile=tile, units=units, reg=reg, block=ax.name + "_b", unit=ax.name + "_u")
            for ax, (tile, units, reg) in zip(self.axes, dims, strict=True)
        )

    @property
    def m(self) -> Side:
        return self.mn[0]

    @property
    def n(self) -> Side:
        return self.mn[1]


@dataclass(frozen=True)
class Work:
    """The kernel's ONE worker inventory (1r in-memory; the step-7 ``WORK`` wire family) — the
    per-site ``w``/``n`` worker tokens factored out of the ``TILE`` values into a single
    kernel-global slot (``w4x1`` warps for the mma tier / ``t16x8`` threads for the scalar tier /
    ``t512`` the 1-D cooperative width), plus the ``+p<n>`` PRODUCER band the retired ``WSPEC``
    family spelled (aux warps are inventory too). Derived at option assembly from the TILE
    node choices (:func:`derive_workers`), FAILING LOUDLY when two sites disagree. The explicit
    slot makes an inconsistent pair of exact node-site choices unrepresentable. Kernel-global-ness
    encodes today's TRUE invariant:
    fragment-resident dataflow between composed folds shares the warp map."""

    kind: str = "direct"  # direct | warp (mma tier) | thread (scalar / coop tiers)
    units: tuple[int, int] = (1, 1)  # the native codec order of the TILE value the slot came from
    producer: int = 0  # dedicated producer warps (the WSPEC absorb — ``+p<n>``; warp kind only)

    def __post_init__(self) -> None:
        if self.kind not in ("direct", "warp", "thread"):
            raise ValueError(f"bad Work kind {self.kind!r} (expect direct | warp | thread)")
        if not isinstance(self.units, tuple) or len(self.units) != 2 or any(type(width) is not int or width < 1 for width in self.units):
            raise ValueError(f"Work units must be two positive widths, got {self.units!r}")
        if type(self.producer) is not int or self.producer < 0:
            raise ValueError(f"Work producer band must be a non-negative integer, got {self.producer!r}")
        if self.kind == "direct" and (self.units != (1, 1) or self.producer):
            raise ValueError("direct Work cannot carry worker geometry or a producer band")
        if self.kind != "warp" and self.producer:
            raise ValueError("a producer band requires warp Work")

    @property
    def is_direct(self) -> bool:
        """Whether this choice leaves launch geometry to the direct per-cell path."""
        return self.kind == "direct"

    @property
    def count(self) -> int:
        """Work in the inventory (warps or threads, by ``kind``) — the COMPUTE band; the
        producer band is on top."""
        return self.units[0] * self.units[1]

    def spell(self) -> str:
        """The ``WORK`` codec string: ``w<M>x<N>[+p<np>]`` (warps; both dims always) /
        ``t<N>x<M>`` (the scalar thread tile, native n-then-m order) / ``t<N>`` (the 1-D
        cooperative width — a trailing ``x1`` is suppressed for threads only)."""
        if self.is_direct:
            return ""
        if self.kind == "warp":
            base = f"w{self.units[0]}x{self.units[1]}"
            return f"{base}+p{self.producer}" if self.producer else base
        if self.units[1] == 1:
            return f"t{self.units[0]}"
        return f"t{self.units[0]}x{self.units[1]}"

    @classmethod
    def parse(cls, spec: str | None) -> Work:
        """Decode a ``WORK`` codec string (inverse of :meth:`spell`); ``""`` / ``None`` = no
        inventory (the per-cell / pure-reduce forms keep their derived launch geometry)."""
        if not spec:
            return _canonical_choice("WORK", spec, cls())
        s = spec
        producer = 0
        if "+" in s:
            s, _, band = s.partition("+")
            if not band.startswith("p") or not band[1:].isdigit():
                raise ValueError(f"WORK {spec!r}: expected a +p<n> producer band, got {band!r}")
            producer = int(band[1:])
        kind = {"w": "warp", "t": "thread"}.get(s[:1])
        if kind is None:
            raise ValueError(f"WORK {spec!r}: expected w<M>x<N>[+p<n>] or t<N>[x<M>]")
        dims = s[1:].split("x")
        if not all(d.isdigit() for d in dims) or len(dims) > 2:
            raise ValueError(f"WORK {spec!r}: bad worker dims {s[1:]!r}")
        if kind == "warp" and (len(dims) != 2 or producer < 0):
            raise ValueError(f"WORK {spec!r}: the warp inventory spells both dims (w<M>x<N>)")
        if kind == "thread" and producer:
            raise ValueError(f"WORK {spec!r}: a producer band needs the warp inventory")
        units = (int(dims[0]), int(dims[1]) if len(dims) == 2 else 1)
        return _canonical_choice("WORK", spec, cls(kind=kind, units=units, producer=producer))


def resolve_site_tile(spec: str | None, work: Work | None, coop: int = 1) -> Tile:
    """Decode one exact site-local :class:`Tile` spelling against the kernel inventory.

    ``coop`` remains in the signature for row consumers that resolve ``TILE`` and ``REDUCE``
    together, but no longer disambiguates the tile: ``""`` is always per-cell and ``f1`` is the
    parallel unit-register thread tile. The distinction therefore survives encode/decode without
    consulting another node's claim on the shared ``WORK`` inventory."""
    del coop
    return Tile.parse(spec, work)


def plan_workers(plan: Tile | None) -> Work | None:
    """The worker inventory ONE tile plan implies — ``None`` for an untiled plan or a 1-thread
    thread inventory (a bare register strip: the per-cell forms keep their derived launch
    geometry). A plan stores ``(m, n)``; ``Work.units`` is the ``WORK`` codec's own order, so
    the thread inventory transposes on the way out — the inverse of :meth:`Tile.parse`."""
    if plan is None or not plan.is_tiled:
        return None
    units = (plan.units_m, plan.units_n) if plan.is_warp else (plan.units_n, plan.units_m)
    w = Work(kind="warp" if plan.is_warp else "thread", units=units)
    return None if (w.kind == "thread" and w.count == 1) else w


def derive_workers(tiles) -> Work | None:
    """Fold the worker inventory out of a kernel's typed ``Tile`` choices (any iterable) —
    ``None`` when no choice tiles (the per-cell / pure-reduce forms keep their
    derived launch geometry). Raises on DISAGREEMENT between sites: one kernel has ONE worker
    inventory (the flash QK/PV pair shares the warp map by construction; a pin pair that
    disagrees must fail loudly, never be half-ignored)."""
    work: Work | None = None
    for plan in tiles:
        w = plan_workers(plan)
        if w is None:
            continue
        if work is None:
            work = w
        elif work != w:
            raise ValueError(f"disagreeing worker geometry across TILE sites: {work.spell()} vs {w.spell()} — one kernel, one inventory")
    return work


def derive_inventory(tiles, *, coop: int = 1, producer: int = 0) -> Work | None:
    """The kernel's ONE worker inventory, folded out of its typed ``Tile`` choices and then
    reconciled with a cooperative ``REDUCE`` width and a ``+p`` producer band. ``None`` when
    nothing tiles and no band cooperates (the per-cell / pure-reduce forms keep their derived
    launch geometry).

    Raises on DISAGREEMENT, and the rule is EXACT rather than a headcount: a cooperative reduce is
    the 1-D thread inventory ``t<coop>``, so it is co-representable with a tiled site only when
    that site's own inventory IS ``(coop, 1)`` threads. Comparing worker COUNTS instead would call
    a ``t8x4`` scalar tile — or worse, a ``w4x8`` warp tile — compatible with a 32-wide coop that
    neither realizes.

    ONE home for the rule: the private enumerator uses it to prune incompatible prefixes and
    ``ClassicScheduleContext.accepts`` uses it again at each complete typed leaf."""
    work = derive_workers(tiles)
    if coop > 1:
        band = Work(kind="thread", units=(coop, 1))
        if work is not None and work != band:
            raise ValueError(f"disagreeing worker geometry: TILE workers {work.spell()} vs coop width {coop} — one kernel, one inventory")
        work = band
    if producer:
        if work is None or work.kind != "warp":
            raise ValueError(f"a +p{producer} producer band needs a warp inventory; got {work.spell() if work else 'none'}")
        work = dc_replace(work, producer=producer)
    return work


@dataclass(frozen=True)
class Placement:
    """Kind-neutral free-axis → grid binding (the parallel output axes and their grid
    mapping). ``010_lift`` builds an UNMAPPED placement (just ``free``); the schedule
    maps every free axis onto ``grid`` (the per-cell tier)."""

    free: tuple[Axis, ...] = ()
    grid: tuple[Axis, ...] = ()
    #: Set by the scheduling transition (:meth:`on_grid`) — the EXPLICIT "the grid has been
    #: decided" bit. A non-empty ``grid`` already says so, but a **free-less** kernel (a decode
    #: row whose every axis folded into the tile) maps onto an EMPTY grid, so its scheduled
    #: placement is value-identical to its unmapped one. the schedule is the step that has to
    #: tell them apart — reading "no free axes" as mapped silently skipped every such kernel and
    #: leaked it unscheduled (10 of the gemma-4 decode twins' kernels).
    mapped: bool = False

    @property
    def is_mapped(self) -> bool:
        """True once the grid has been decided — the explicit :attr:`mapped` bit, or a bound
        ``grid`` (so a scheduled placement built directly with its grid, as the flash / strip /
        split options do, still reads mapped)."""
        return self.mapped or bool(self.grid)

    @property
    def root_mn(self) -> tuple[Axis, Axis] | None:
        """The ``(m, n)`` output axes a ROOT (depth-1) contraction tiles — the grid's TRAILING
        pair — or ``None`` when the grid cannot supply them (unmapped, or rank < 2: the caller's
        untiled path). The one home of the depth-1 rule; :meth:`Tile.placed_on` and
        :meth:`Sched._mn_for` both read it."""
        grid = tuple(self.grid)
        return (grid[-2], grid[-1]) if len(grid) >= 2 else None

    def on_grid(self) -> Placement:
        """The scalar-tier mapping: bind every free axis onto the thread grid."""
        return Placement(free=self.free, grid=self.free, mapped=True)


# --------------------------------------------------------------------------- #
# The operand-transport + warp-split descriptors. ``Stage`` (the operand's intermediate storage
# and fill mechanism) is the operand
# pipeline; ``WarpSpec`` (the WSPEC worker split) partitions the CTA's warps into producer /
# compute bands over that fixed pipeline — both resolved scheduler-side and applied verbatim by
# the materializer (the liveness-scheduled ``lowering/kernel/_stage.pipelined_kloop``, via its
# whole-body ``staged_kloop`` entry).
# --------------------------------------------------------------------------- #


#: The transport tokens: the shared-memory intermediate named by its fill mechanism — the
#: synchronous thread fill (``smem``: a byte copy of materialized edges, or the compute fill
#: evaluating a computed edge into its slab), ``cp.async`` (``smem-async``) and TMA
#: (``smem-tma``). An EMPTY ``STAGE`` is no intermediate at all: gmem→register on a
#: materialized operand, register-to-register on a computed one.
_TRANSPORTS = ("direct", "smem", "smem-async", "smem-tma")

#: The ``STAGE`` grammar, rendered into every parse error so a bad pin names what it could have said.
_STAGE_EXPECT = "expect d<n> / smem|smem-async|smem-tma / p<n>"


@dataclass(frozen=True)
class Stage:
    """One operand-transport pipeline over the serial reduce loop — one ``Stage`` per reduce
    loop (a reduce ``Loop`` ⇒ one reduce axis ⇒ one pipeline). The schedule's
    operand-staging knob, decided by the tile schedule and materialized in
    ``010_materialize``.

    ``Stage.direct()`` is the register / gmem-direct baseline (no slab); every other ``Stage``
    means staging is **on** (the reused gmem operands ride a shared-memory slab). Spelled by the
    ``STAGE`` codec ``d<depth>/smem|smem-async|smem-tma[/p<reg_depth>]``
    (decided by the tile schedule). Eligibility and sizing produce a separate
    :class:`ResolvedStage`; this choice never stores shared-memory names or a derived K chunk.

    The pipeline has two buffering levels down the memory hierarchy, each with its own depth:
    ``depth`` is the **gmem→smem** ring (a synchronous slot fill or the cp.async / TMA prefetch
    over the serial reduce loop), ``reg_depth`` is the **smem→register** double-buffer (the
    fragment-load ping-pong over the inner atom-K steps, breaking the WAR hazard on the operand fragments). They are
    orthogonal — ``d3/smem-async/p2`` is a 3-deep gmem ring feeding a 2-deep register ping-pong.
    ``reg_depth = 1`` (the default) is the "optional register" OFF point (no inner prefetch).
    The slab K-*granularity* (how much K is resident) is ``Tile.bk``, NOT a third depth
    here — granularity and buffer depth are kept distinct.

    Rotation and refill discipline are NOT fields: the staged K-loop derives the slot count from
    ``depth`` alone (``lowering/kernel/_stage``), which is why the retired ``ring`` flag compiled
    byte-identically with and without it."""

    depth: int = 1  # gmem→smem ring depth over the reduce loop (1 = single buffer, no prefetch)
    transport: str = "smem"  # smem | smem-async | smem-tma (the intermediate and its fill mechanism)
    reg_depth: int = 1  # smem→register double-buffer depth (1 = no inner ldmatrix prefetch)

    def __post_init__(self) -> None:
        if self.transport not in _TRANSPORTS:
            raise ValueError(f"bad Stage transport {self.transport!r} (expect smem | smem-async | smem-tma)")
        if type(self.depth) is not int or self.depth < 1:
            raise ValueError(f"Stage depth must be a positive integer, got {self.depth!r}")
        if type(self.reg_depth) is not int or self.reg_depth < 1:
            raise ValueError(f"Stage reg_depth must be a positive integer, got {self.reg_depth!r}")
        if self.transport == "direct" and (self.depth != 1 or self.reg_depth != 1):
            raise ValueError("direct Stage cannot carry pipeline depths")

    @classmethod
    def direct(cls) -> Stage:
        """The explicit no-intermediate transport choice."""
        return cls(transport="direct")

    @property
    def is_direct(self) -> bool:
        """Whether the operand moves directly to registers."""
        return self.transport == "direct"

    @classmethod
    def parse(cls, spec: str | None) -> Stage:
        """Decode the ``STAGE`` knob codec into a stage: ``/``-separated tokens —
        ``d<depth>`` (gmem→smem ring depth), ``smem`` | ``smem-async`` | ``smem-tma`` (the
        intermediate and its fill mechanism: a synchronous thread fill — byte-copying a
        materialized edge, evaluating a computed one, converting when the dtypes differ — the
        cp.async ring, or TMA; the ABSENCE of a stage means no intermediate at all — gmem→register
        for a materialized operand, register→register evaluation for a computed one), and an
        optional ``p<reg_depth>`` (smem→register double-buffer depth). Empty / ``None`` is the
        explicit gmem-direct choice. Shared-memory names and K extent are resolved later.

        Tokens have one canonical order, and each field binds at most once: reordered or repeated
        fields raise instead of quietly deploying a kernel the pin did not name. Raises
        ``ValueError`` and only ``ValueError`` on any malformed input (the featurizers degrade on
        it), naming the codec so a bad pin names its knob."""
        s = spec or ""
        if not s:
            return _canonical_choice("STAGE", spec, cls.direct())
        seen: set[str] = set()
        depth, transport, reg_depth = 1, "smem", 1

        def once(field: str, tok: str) -> None:
            if field in seen:
                raise ValueError(f"bad STAGE codec {spec!r}: {field!r} is spelled twice (at {tok!r})")
            seen.add(field)

        for t in s.split("/") if s else ():
            if t in _TRANSPORTS:
                once("transport", t)
                transport = t
            elif t.startswith("d"):
                once("d", t)
                depth = _codec_width(t[1:], tok=t, codec="STAGE")
            elif t.startswith("p"):
                once("p", t)
                reg_depth = _codec_width(t[1:], tok=t, codec="STAGE")
            else:
                raise ValueError(f"bad STAGE token {t!r} ({_STAGE_EXPECT})")
        return _canonical_choice("STAGE", spec, cls(depth=depth, transport=transport, reg_depth=reg_depth))

    def spell(self) -> str:
        """The ``STAGE`` codec string for this stage (inverse of :meth:`parse`). Depth and
        transport are explicit; ``reg_depth`` is spelled only when ≥ 2 (the ``p1`` default is
        omitted)."""
        if self.is_direct:
            return ""
        toks = [f"d{self.depth}", self.transport]
        if self.reg_depth > 1:
            toks.append(f"p{self.reg_depth}")
        return "/".join(toks)

    @property
    def is_async(self) -> bool:
        """True for the asynchronous-copy transports (``cp.async`` / ``tma``) — the ones
        that issue a commit/wait or mbarrier handshake rather than a plain ``__syncthreads``."""
        return self.transport in ("smem-async", "smem-tma")

    def available_on(self, ctx) -> bool:
        """Whether this stage's copy instruction family exists on ``ctx`` — the ONE statement of
        the transport/target rule (``cp.async`` is sm_80+, TMA sm_90+; the synchronous ``smem``
        fill is universal). ``stage_moves`` filters the catalog through it, exactly as the atom
        registry filters through :meth:`AtomKind.available_on`; a PIN's refusal message reads it
        through the scheduler's ``stage_target``."""
        if self.transport == "smem-async":
            return ctx.has_cp_async
        if self.transport == "smem-tma":
            return ctx.has_tma
        return True


@dataclass(frozen=True)
class ResolvedStage:
    """Materialization facts derived from a :class:`Stage` choice at one problem edge."""

    choice: Stage
    smem: tuple[str, ...] = ()
    bk_elems: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.choice, Stage):
            raise TypeError("ResolvedStage choice must be a Stage")
        if not isinstance(self.smem, tuple) or any(not isinstance(name, str) for name in self.smem):
            raise TypeError("ResolvedStage smem names must be a tuple of strings")
        if self.choice.is_direct:
            raise ValueError("a direct Stage has no resolved materialization")
        if type(self.bk_elems) is not int or self.bk_elems < 0:
            raise ValueError(f"resolved stage bk_elems must be a non-negative integer, got {self.bk_elems!r}")

    def __getattr__(self, name: str):
        """Expose the originating choice to lowering without copying its fields."""
        return getattr(object.__getattribute__(self, "choice"), name)


@dataclass(frozen=True)
class WarpSpec:
    """The producer band — the dedicated warps split off the uniform pipeline to drive the
    ``Stage`` gmem→smem load half, ORTHOGONAL to the pipeline (``reduce`` / ``tile`` / ``stage``):
    it adds no pipeline parameter, only the warp split. The COMPUTE (mma-consumer) warps are
    implicit, sized by ``Tile.units``, so the CTA launches ``Tile.block_threads +
    32·producer_warps`` threads. ``workers is None`` on the schedule is uniform SIMT (every warp
    does every role's work, software-pipelined in-warp); a constructed ``WarpSpec`` means
    specialization is on.

    The band is DECIDED as inventory — it rides ``WORK``'s ``+p<n>`` (``Work.producer``), which
    is where the retired ``WSPEC`` knob's one live decision went. ``ClassicScheduleContext`` and
    the private enumerator require a resolved TMA stage on an unsplit kernel. The box copy is issued by one elected
    thread onto a slot mbarrier any thread can parity-wait, so the fill moves warps freely, while
    ``cp.async``'s wait-group is issuing-thread-scoped and an ``smem`` compute fill has no async load
    half.

    Materialized by the staged K-loop (``lowering/kernel/_stage.staged_kloop``): the producer band
    rides at the TAIL of the thread block (``threadIdx.x >= block_threads``), so the compute warps'
    grid decode is untouched; the producer drives the TMA fill (one elected thread arms the slot
    mbarrier), the compute warps parity-wait, drain and release the slot on an "empty" mbarrier
    ring, with ``SetMaxNReg`` register redistribution between the branches."""

    producer_warps: int

    def __post_init__(self) -> None:
        if type(self.producer_warps) is not int or self.producer_warps < 1:
            raise ValueError(f"producer_warps must be a positive integer, got {self.producer_warps!r}")

    def spell(self) -> str:
        """The band's ``p<n>`` spelling — the dump's ``band`` line; ``""`` for no band."""
        return f"p{self.producer_warps}" if self.producer_warps else ""


@dataclass(frozen=True)
class Raster:
    """The CTA rasterization order — the bare kernel-scoped ``RASTER`` choice (one launch order
    per grid). It changes NO per-CTA work, layout, or schedule — only
    which flat CTA id maps to which ``(m, n)`` output block tile: ``gm<G>`` iterates ``G`` M
    block-tiles fastest within each launch stripe (consecutive CTAs then share the streamed B
    slab, so its repeat reads hit L2 instead of DRAM — the 2026-07-12 4090 NCU finding measured
    the flat order streaming B once per M-row, ``A + C + B×2`` exactly); ``gn<G>`` is the
    transpose (A the streamed operand). The explicit ``Raster()`` choice is the flat N-fastest row-major order.
    Eligible only on a 2-D-tiled contraction grid (the kernel IR tile's ``raster_axes``, stamped
    by the materializer's ``grid_tile`` seal)."""

    orient: str = "direct"  # direct | m (group M block-tiles) | n (group N block-tiles)
    group: int = 1  # stripe group size, ≥ 2 when grouped

    def __post_init__(self) -> None:
        if type(self.group) is not int:
            raise ValueError(f"Raster group must be an integer, got {self.group!r}")
        if self.orient == "direct":
            if self.group != 1:
                raise ValueError("direct Raster must have group 1")
            return
        if self.orient not in ("m", "n") or self.group < 2:
            raise ValueError(f"grouped Raster requires m/n orientation and group >= 2, got {self.orient!r}/{self.group}")

    @property
    def is_direct(self) -> bool:
        """Whether CTA ids use the ordinary flat row-major order."""
        return self.orient == "direct"

    def spell(self) -> str:
        """Return the canonical ``RASTER`` value."""
        return "" if self.is_direct else f"g{self.orient}{self.group}"

    @classmethod
    def parse(cls, spec: str | None) -> Raster:
        """Decode ``gm<G>`` / ``gn<G>`` (``""`` / ``None`` → explicit flat order). Raises
        ``ValueError`` on any other spelling — the loud pin contract."""
        if not spec:
            return _canonical_choice("RASTER", spec, cls())
        m = _RASTER_RE.fullmatch(spec)
        if m is None or int(m.group(2)) < 2:
            raise ValueError(f"RASTER: expected gm<G> / gn<G> with G >= 2 (or empty for the flat order), got {spec!r}")
        return _canonical_choice("RASTER", spec, cls(orient=m.group(1), group=int(m.group(2))))


_RASTER_RE = re.compile(r"g([mn])(\d+)")


def _ext_expr(axis: Axis) -> Expr:
    """The axis extent as an ``Expr`` — a literal int (static) or the symbolic ``Dim`` expr."""
    return Literal(axis.extent.as_static(), "int") if axis.extent.is_static else axis.extent.expr


def _overhangs(axis: Axis, tile: int) -> bool:
    """True iff a ``tile``-wide CTA block overhangs ``axis`` (symbolic or non-divisible extent) —
    so its tail must be masked."""
    if tile <= 1:
        return False
    return not (axis.extent.is_static and axis.extent.as_static() % tile == 0)


@dataclass(frozen=True)
class Side:
    """One tiled output axis of a contraction — the outer ``m`` or inner ``n`` — paired with its
    derived per-CTA tile geometry. The two ride as a ``(m, n)`` pair (:attr:`Fold.mn`)
    mirroring :class:`Tile`'s own ``units`` / ``regs`` tuples, so the factorizer
    threads one object per axis instead of a dozen loose ``*_m`` / ``*_n`` args. The tile width,
    unit / register counts, and bound block/unit var names are derived from an axis + a
    :class:`Tile`; the ``mask`` / ``ext`` follow from the axis + width.

    Lives here, beside ``Tile``, because it is SCHEDULE geometry, not algebra: ``ir/tile/ir.py``
    already imports this module, so a tile-side home would make the schedule slice unable to derive
    its own ``(m, n)`` reading."""

    axis: Axis  # the output axis (a param)
    tile: int  # the per-CTA width = units · reg · atom_dim
    units: int  # parallel units on this axis (warps for mma / threads for scalar)
    reg: int  # register sub-cells per unit on this axis
    block: str  # the bound grid-block var (the axis name + ``_b``)
    unit: str  # the bound unit var (the axis name + ``_u``)

    @property
    def name(self) -> str:
        return self.axis.name

    @property
    def mask(self) -> bool:
        """True iff a ``tile``-wide CTA block overhangs the axis (its tail must be masked)."""
        return _overhangs(self.axis, self.tile)

    @property
    def ext(self) -> Expr:
        """The axis extent as an ``Expr`` — a static literal or the symbolic ``Dim`` expr."""
        return _ext_expr(self.axis)


__all__ = [
    "AtomKind",
    "FoldMove",
    "Level",
    "Placement",
    "PlacedTile",
    "Side",
    "Reduce",
    "ReduceStage",
    "Raster",
    "ResolvedStage",
    "Stage",
    "Tile",
    "WarpSpec",
    "Work",
    "derive_workers",
]
