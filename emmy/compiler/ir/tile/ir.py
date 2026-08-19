"""Tile IR — a map/reduce kernel with its *schedule* made explicit.

One :class:`TileOp` is the article's reduction skeleton — ``project ∘ reduce(⊕, e) ∘ map(f)`` —
scheduled but not yet bound to hardware threads. It sits between Loop IR (pure iteration) and
Kernel IR (threads / smem):

    Loop IR ──lowering/tile──▶ Tile IR ──lowering/kernel──▶ Kernel IR

The whole point of the layer is the article's thesis: **the schedule is separate from the
combine.** The combine is not defined here — it is the :class:`~emmy.compiler.ir.pure.fold.Fold`
term (``ir/pure/fold.py``), which a ``TileOp`` holds whole in ``op``. What this module owns is
everything the term deliberately does not carry:

- the root-global schedule fields — the free-axis → grid :class:`~.schedule.Placement` (``place``),
  the ONE worker inventory (``work``) and the warp-spec split (``workers``);
- the per-node schedule SLICES in ``TileOp.schedule`` (``{codec key → resolved TilePlan /
  ReducePlan / Stage}``, keyed by the tree-path codec and read through ``ops.Sched``);
- the kernel's EFFECTS — the root-store :class:`Store` decorations and the ``effect_tail`` /
  ``split_effects`` pair that reconstitutes the effectful stmt stream from them.

That split is the layer's invariant, not a convenience. The stored term is pure algebra, IMMUTABLE
across the whole schedule search — a fork is a different slice map, never a rebuilt tree — which is
what makes kernel identity (``TileOp.structural_key``) the algebra alone, with placement, slices,
workers and stores all excluded. Tile IR stores only pure terms; statements appear when the term is
lowered, never inside it (``ir/ARCHITECTURE.md``, "Pure terms vs statements").

There is no per-kind kernel/schedule type: dispatch reads the role structurally off the node (a
fold's role derives), so a projection, a reduction and a contraction all ride the same ``TileOp``.
The kernel materializer reads the schedule off the slice beside the node — it never re-recognizes
structure the tile IR already holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import Op
from emmy.compiler.ir.schedule import Placement, WarpSpec
from emmy.compiler.ir.stmt import Body, Loop, Stmt, Write
from emmy.compiler.ir.stmt.body import _member_reads
from emmy.compiler.structural import digest


@dataclass(frozen=True)
class Store:
    """One ROOT-STORE decoration at the kernel boundary — the effect the stored term no
    longer carries. ``write`` is the store verbatim (target buffer, index template, stored value
    names, the atomic flag — holding the ``Write`` whole keeps every field lossless), and it is
    NOT part of the term: ``TileOp.stores`` owns the tuple, and consumers reconstitute the
    effectful stmt stream via :func:`effect_tail`. A ``sweep`` store's ``Write`` rides a per-cell output ``Loop`` over
    that axis (rms/softmax's normalize sweep, ``unroll`` preserved); the swept members are the
    trailing projection stmts reading the axis (:func:`_sweep_start`). Conversion sites go
    through :func:`split_effects`, whose reconstitution round-trip gate is what keeps kernel
    sources byte-identical to the stored-``Write`` era."""

    write: Write
    sweep: Axis | None = None
    unroll: bool = False


def _sweep_start(stmts, axis_name: str) -> int:
    """The first index of the trailing projection run a ``sweep`` store's output ``Loop``
    wraps — the earliest stmt reading the sweep axis (SSA deps + Expr free vars, deep). The
    trailing-RUN rule (everything from that stmt on is swept) is deliberately simple; the
    :func:`split_effects` round-trip gate is what proves it reproduces the captured loop."""
    for i, s in enumerate(stmts):
        if axis_name in _member_reads(s):
            return i
    return len(stmts)


def effect_tail(stmts, stores) -> list[Stmt]:
    """Reassemble the EFFECTFUL projection stmt stream from a pure projection body + the
    kernel-boundary ``stores`` — the ONE reconstitution rule the scheduler's tail gates, the
    materializer's zero-axis ``Fold`` peel and ``030_split_reduce`` share, so the lowered kernels stay
    byte-identical to the stored-``Write`` era. A plain store appends its ``Write``; a
    ``sweep`` store wraps the trailing run of stmts reading its axis (:func:`_sweep_start`)
    into the per-cell output ``Loop``, the ``Write`` last."""
    out = list(stmts)
    for st in stores:
        if st.sweep is None:
            out.append(st.write)
        else:
            i = _sweep_start(out, st.sweep.name)
            out = [*out[:i], Loop(axis=st.sweep, body=Body((*out[i:], st.write)), unroll=st.unroll)]
    return out


def split_effects(stmts) -> tuple[tuple[Stmt, ...], tuple[Store, ...]] | None:
    """Split an effectful projection stmt stream into ``(pure stmts, Store decorations)`` — the
    conversion-side inverse of :func:`effect_tail`, valid ONLY when the reconstitution
    round-trips byte-identically (checked here; ``None`` otherwise — the caller keeps the
    raw-loop-IR spelling, the 1o construction-gate pattern). Recognized shapes: a trailing run
    of top-level root ``Write``\\ s, or ONE trailing non-reduce output sweep ``Loop`` of pure
    stmts whose last stmt is the ``Write``. An already-pure stream returns ``(stmts, ())``."""
    original = list(stmts)
    rest = list(stmts)
    stores: list[Store] = []
    while rest and isinstance(rest[-1], Write):
        stores.insert(0, Store(write=rest.pop()))
    if not stores and rest and isinstance(rest[-1], Loop) and not rest[-1].is_reduce:
        loop = rest[-1]
        inner = list(loop.body)
        if inner and isinstance(inner[-1], Write) and all(s.pure for s in inner[:-1]):
            stores.insert(0, Store(write=inner[-1], sweep=loop.axis, unroll=loop.unroll))
            rest = [*rest[:-1], *inner[:-1]]
    if not all(s.pure for s in rest):
        return None
    if effect_tail(rest, stores) != original:
        return None
    return tuple(rest), tuple(stores)


@dataclass
class TileOp(Op):
    """One scheduled map/reduce kernel (see module docstring).

    Holds the structural-IR root ``op`` (a :class:`Fold`, at any role, or ``None`` for a
    placeholder node) plus the schedule fields — not a pre-lowered body. The per-cell loop-IR
    body is generated at materialize time by ``op.lower()``, and a bare reduction / contraction's
    output ``Write`` is glue generated there too (from ``place.grid`` + the graph node's output
    buffer; see ``lowering/kernel/010_materialize``). ``inputs`` / ``outputs`` come from the base
    :meth:`Op.populate_io` (graph edges) — no body walk.

    Schedule fields (all defaulted, so a fresh / placeholder node is well-formed):

    - ``place`` — the free-axis → grid binding (:class:`~.schedule.Placement`); root-global.
    - ``workers`` — the warp-specialization split (:class:`~.schedule.WarpSpec`); root-global, ``None`` =
      uniform SIMT.

    There is **no** let table: a computed operand is stored inline on its edge, and sharing is the
    product contraction's arity (see the module docstring), so stored trees are already
    resolved and every walk is a plain tree walk. The per-node schedule SLICES live in
    ``schedule``: ``{codec key → resolved TilePlan / ReducePlan / Stage}``, keyed by the
    tree-path codec's canonical key (:mod:`~emmy.compiler.ir.tile.path` — a fold may carry all
    three families at once, so the path alone cannot key the map; the family selects the slice
    kind, so key and value agree by construction). The ``op`` term is pure algebra, IMMUTABLE
    across the whole schedule search — a fork is a different map, never a rebuilt tree. Read /
    write through :class:`~emmy.compiler.ir.tile.ops.Sched` (``ops.reduce_plan`` is the plan
    accessor); ``lower`` never sees the slices, so kernel identity (``Op.cache_key``) is
    untouched. The contraction operand→role binding is not a
    ``TileOp`` field either — a tiled contraction carries its A operand / channels on
    its stored fold (``op``), the single source of truth, resolved recognize-side
    (``_lift._nodify_contraction``); the placed reading only PLACES that node."""

    op: object = None
    name: str = ""
    place: Placement = field(default_factory=Placement)
    workers: WarpSpec | None = None
    schedule: dict = field(default_factory=dict)
    # The kernel's ROOT-STORE decorations (``Store``): the output ``Write``\\ s (and the
    # rms/softmax output-sweep spelling) — a kernel-boundary fact beside ``place``. Empty for a
    # bare reduction / contraction — its grid-cell store
    # stays the materializer's default glue (``_factor.with_store``). Consumers reconstitute
    # the effectful stmt stream via ``effect_tail`` — never read a ``Write`` out of the term.
    stores: tuple = ()
    # The ONE worker inventory (``ir.schedule.Workers``): the ``w``/``n`` worker
    # tokens factored out of the per-site TILE values, derived at option assembly
    # (``ops.Sched.seal_workers`` — loud on cross-site disagreement). ``None`` = the per-cell /
    # pure-reduce forms (derived launch geometry). The wire format spells the inventory ONCE, in
    # ``WORK``; the site values carry no worker tokens and the retired embedded spellings raise.
    work: object = None

    def pretty_body(self) -> str:
        """The structural dump — delegated to :mod:`~emmy.compiler.ir.tile._dump`, which owns
        every presentation concern in the layer."""
        from emmy.compiler.ir.tile._dump import tile_body  # noqa: PLC0415 — presentation, loaded on demand

        return tile_body(self)

    def structural_key(self) -> str:
        """Kernel identity — the stored term's α-invariant digest (``""`` for a placeholder).
        Placement, schedule slices, workers and stores are deliberately EXCLUDED: identity is
        the algebra alone (the NO-schedule-fields rule above), so every fork sibling of one term
        shares the key and no emission path can leak a schedule into it."""
        return self.op.structural_key() if self.op is not None else ""

    def cache_key(self) -> str | None:
        return digest(type(self).__name__, self.structural_key(), self._knob_key())


__all__ = [
    "Store",
    "TileOp",
    "effect_tail",
    "split_effects",
]
