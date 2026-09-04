"""``Body`` — an immutable sequence of body Stmts with built-in
def-use / iteration / transform queries.

Implemented as a ``tuple`` subclass so it interoperates transparently
everywhere a ``tuple[Stmt, ...]`` was previously accepted: iteration,
indexing, length, slicing, equality, hashing, and ``isinstance(body,
tuple)`` all work without thinking. The methods on Body are the
recommended way to phrase common analyses (def-use, iteration,
type-filtered lookups) so they can be added incrementally without
rippling through call sites.

Phase 1 surface (this file): the protocol that lets every
``tuple[Stmt, ...]`` site accept Body, plus :meth:`iter` / :meth:`map`
as method-shaped wrappers around the existing free functions.

Phase 2 surface: def-use queries (``definitions``, ``axis_dependencies``,
``deps_closure``, ``depends_on`` / ``independent``, ``deps_of``), type-filtered lookups
(``loads``, ``writes``, ``accums``, …), and dependence cones
(:class:`Cone`, :meth:`Body.backward_cone`
/ :meth:`Body.defs_die_at`) — the shared substrate behind the rules
that slice computed-operand cones. Region transforms (``replace_at``,
``partition_at``) remain follow-ups; add as needed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property, lru_cache

from emmy.compiler.ir.expr import Expr, Var
from emmy.compiler.ir.stmt.base import Stmt


@dataclass(frozen=True)
class Cone:
    """A dependence cone over ONE scope level: the subset of a Body's
    immediate stmts closed under SSA dependence (in body order), plus every
    name the cone reads from outside itself — sibling/enclosing scopes and
    axis vars alike. Built by :meth:`Body.backward_cone`.

    Construction never fails and applies no eligibility judgment: an
    unresolved name is data (``external_reads``), not an error. Which
    external reads are acceptable, which member kinds are cuttable, and
    whether the cone's values escape (:meth:`Body.defs_die_at`) are the
    calling rule's conditions — bail decisions stay in rules, the dataflow
    walk lives here (``passes/ARCHITECTURE.md``: phrase conditions over cone
    properties instead of re-walking shapes).

    A member is a whole top-level stmt: a wrapper (Loop / Cond / Tile) joins
    as a unit, exposing names per :func:`_exposed_defines` and reading per
    :func:`free_names` (subtree rolled up, internally-bound axes
    excluded). Axis vars from enclosing scopes survive into
    ``external_reads`` — intersect with an axis-name set to get the cone's
    axis usage, subtract it to get the SSA names that must resolve
    elsewhere."""

    members: tuple[Stmt, ...]
    external_reads: frozenset[str]

    @property
    def loads(self) -> tuple[Stmt, ...]:
        """Every ``Load`` in the members, nested included, body order —
        the cone's leaf operands (dtype checks, graph resolution)."""
        return Body(self.members).loads


@dataclass(frozen=True)
class Factored:
    """One product split by axis variance — what :meth:`Body.factor` reports.

    ``invariant`` and ``varying`` are ATOM NAMES with multiplicity: a square is listed twice, so a
    consumer sees ``x·x`` for what it is rather than as one factor — the step that is not a semiring
    step, and so carries no contraction reading. Each invariant carries whether it divides, since
    ``Σ x/c`` equals ``(Σ x)/c`` only for an invariant ``c``. ``spine`` is the statements the reading
    saw through, which is exactly what a consumer removes when it rebuilds a lift around the split.
    """

    invariant: tuple[tuple[str, bool], ...]
    varying: tuple[str, ...]
    spine: tuple[Stmt, ...]


#: What :meth:`Body.linearize` reports — one ``(monomial, coefficient)`` pair per term, where a
#: monomial is its streamed atoms with their exponents and the empty one is the constant term.
type Monomials = tuple[tuple[tuple[tuple[str, int], ...], object], ...]


def _exposed_defines(s: Stmt) -> set[str]:
    """SSA names ``s`` makes visible at its own scope level — own defines
    plus every nested define. A deliberate over-approximation: nested non-Accum
    names don't truly cross a Loop boundary, but well-formed SSA never reads them
    from outside, so resolving through them is harmless and cheap."""
    out = set(s.defines())
    for body in s.nested():
        for c in body.iter():
            out.update(c.defines())
    return out


def free_names(s: Stmt) -> frozenset[str]:
    """Every name ``s`` (whole subtree) reads from its enclosing scope — SSA reads AND index
    coordinates, less what the subtree defines or binds.

    The WIDE reading, for callers that must resolve a statement against everything around it: a
    dependence cone needs the axis vars as much as the value names, since both have to be
    available where the cone lands. Callers asking a narrower question — is this VALUE read? —
    want :attr:`Body.ssa_uses`, which never reports a coordinate. Mixing the two is what let an
    index ``Var`` be mistaken for a value read; keeping both spellings is what lets each caller
    say which it meant.
    """
    reads: set[str] = set()
    defs: set[str] = set()

    def walk(st: Stmt, bound: frozenset[str]) -> None:
        reads.update(set(st.deps()) - bound)
        for e in st.exprs():
            reads.update(e.free_vars() - bound)
        defs.update(st.defines())
        if st.deps_deep:
            return  # ``deps()`` already rolls up the subtree scope-correctly; the flat re-walk
            # cannot see the lift's params, so a factored operand cone's result read inside the
            # lift would leak out as a phantom capture
        inner_bound = bound | st.binds_axes()
        for body in st.nested():
            for c in body:
                walk(c, inner_bound)

    walk(s, frozenset())
    return frozenset(reads - defs)


class Body(tuple[Stmt, ...]):
    """Immutable Stmt sequence. Tuple-subclass so existing tuple-shaped
    APIs accept Body for free; preserves its own type through
    :meth:`__getitem__` slicing and :meth:`__add__` concatenation so
    callers don't keep falling back to plain tuples.

    Constructed from any iterable: ``Body(some_tuple)``,
    ``Body([s1, s2])``, ``Body(s for s in ... if ...)``.

    No ``__slots__`` — instances retain ``__dict__`` so
    ``functools.cached_property`` works for the analysis methods we
    add incrementally (``def_table``, ``external_reads``, etc.). The
    per-instance dict adds a small memory overhead vs a bare tuple,
    but Body counts are bounded by the number of kernel bodies in a
    pipeline run (tens to hundreds), so it's not a concern.
    """

    def __new__(cls, stmts: Iterable[Stmt] = ()) -> Body:
        members = tuple(stmts)
        # A body holds STATEMENTS. A pure term is not one — ``Fold`` duck-types the statement
        # protocol so the shared walks can reach it, but it is not a ``Stmt`` and does not belong
        # in a statement sequence: a Fold tree composes through ``operands``, and a term sitting in
        # a body is a second, competing composition mechanism. Checked by TYPE rather than by
        # naming the kinds that are excluded, so the rule holds for anything else that duck-types
        # its way in later.
        stray = [type(member).__name__ for member in members if not isinstance(member, Stmt)]
        if stray:
            raise TypeError(f"Body holds non-statement member(s) {stray}; a term composes through operand edges")
        return super().__new__(cls, members)

    def __getitem__(self, key):
        r = super().__getitem__(key)
        return Body(r) if isinstance(key, slice) else r

    def __add__(self, other: Iterable[Stmt]) -> Body:
        if isinstance(other, tuple):
            return Body(tuple.__add__(self, other))
        return Body(tuple.__add__(self, tuple(other)))

    def __radd__(self, other: Iterable[Stmt]) -> Body:
        if isinstance(other, tuple):
            return Body(tuple.__add__(other, self))
        return Body(tuple.__add__(tuple(other), self))

    # No custom ``__repr__`` — inherit ``tuple.__repr__`` so a Body
    # round-trips through ``repr(...)`` / ``eval(...)`` as a tuple.
    # Loop / Tile / Cond / StridedLoop / LoopOp / TileOp /
    # KernelOp ``__post_init__`` coerce on construction so the ingest
    # path ends up with Body either way.

    @staticmethod
    def coerce(value: Body | Iterable[Stmt]) -> Body:
        """Wrap if not already a Body. Used by ``LoopOp`` /
        ``TileOp`` ``__post_init__`` so the legacy
        ``Op(body=tuple_value)`` construction shape keeps working."""
        return value if isinstance(value, Body) else Body(value)

    # -- iteration -------------------------------------------------------

    def iter(self) -> Iterator[Stmt]:
        """Pre-order iteration over this body and every nested body
        (``Loop`` / ``Tile`` / ``Cond`` / ``StridedLoop`` recurse via
        ``Stmt.nested()``)."""
        for s in self:
            yield s
            for child_body in s.nested():
                yield from child_body.iter()

    # -- transformation --------------------------------------------------

    def rename_buffers(self, rename) -> Body:  # noqa: ANN001 — any str->str mapping
        """This body with every external-buffer reference renamed through ``rename`` — the
        body-level face of :meth:`Stmt.rename_buffers` (the recursive :meth:`map` reaches every
        nested leaf, so wrapper stmts need no handling)."""
        return self.map(lambda s: s.rename_buffers(rename))

    def map(self, fn: Callable[[Stmt], Stmt | None | Iterable[Stmt]]) -> Body:
        """Recursive 1:N body transformer. Post-order: each block stmt's
        nested body is mapped first, then ``fn`` is applied to the
        children-rewritten wrapper. Returns a new Body with each stmt
        replaced by ``fn(stmt)``:

        - a single ``Stmt`` (kept in place of the input),
        - ``None`` (drop the input), or
        - an iterable of ``Stmt`` (inline all of them).

        ``fn`` is called on *every* stmt including ``Loop`` / ``Tile`` /
        ``Cond`` / ``StridedLoop`` wrappers — but with their bodies already
        recursively mapped, so ``fn`` only needs to handle the leaf cases
        it cares about (callers no longer need a self-recursive
        ``Loop(..., body=s.body.map(fn))`` branch). Mirrors :meth:`iter`'s
        full-tree traversal.

        Iterable returns *replace* the wrapper: the returned stmts are
        spliced into the body as-is (their interiors have already been
        recursed when the wrapper was visited), so a caller that
        ``return tuple(c for c in s.body)`` to inline a Loop's body sees
        already-rewritten children.
        """

        def descend(s: Stmt) -> Stmt:
            nested = s.nested()
            if not nested:
                return s
            return s.with_bodies(tuple(b.map(fn) for b in nested))

        out: list[Stmt] = []
        for s in self:
            r = fn(descend(s))
            if r is None:
                continue
            if isinstance(r, Stmt):
                out.append(r)
            else:
                out.extend(r)
        return Body(out)

    # -- generic backward dataflow --------------------------------------

    def fold[T](
        self,
        fn: Callable[[Stmt, tuple[T | None, ...], frozenset[str]], T],
    ) -> dict[int, T]:
        """Generic backward dataflow over this body's def-use DAG.

        Walks every stmt in source order (= SSA topo order). At each stmt,
        calls ``fn(stmt, child_T, bound)``:

        - ``child_T`` — one entry per name in ``stmt.deps()``, pulled from
          the running memo via ``self.definitions``. ``None`` when the dep
          is read but not defined locally (Tile-input buffer reference,
          constant, or an SSA from an enclosing scope — i.e. an external
          read). Position-preserving: same order as ``stmt.deps()``.
        - ``bound`` — set of axis names introduced by enclosing
          ``Loop`` / ``StridedLoop`` / ``Tile`` wrappers via
          ``Stmt.binds_axes()``. ``Cond`` doesn't bind axes. Callbacks
          that don't care about scope can ignore this.

        Returns the per-stmt memo keyed by ``id(stmt)`` — ``Tile`` is a
        non-frozen dataclass and not hashable, so id-keying is the
        lowest-friction choice. Callers that want a name-keyed view do
        ``{n: memo[id(s)] for s in body.iter() for n in s.defines()}``.

        Recursion order: nested bodies are processed *before* the wrapper
        stmt. So when ``fn`` is called on a wrapper that doesn't define a
        name itself (Loop / Tile / StridedLoop), the memo entries for any
        Accums inside its body already exist — downstream consumers at the
        wrapper's scope can read them through ``deps()``.

        Caveat: when multiple stmts define the same SSA name (matmul-shape
        bodies with several ``Accum`` stmts sharing one accumulator),
        ``self.definitions`` resolves to the last definer. ``child_T`` will
        carry that last definer's ``T`` only. Callers needing a multi-defs
        union (e.g. unioning axes across all Accums for ``acc``) iterate
        ``body.accums`` themselves at the call site.
        """
        memo: dict[int, T] = {}
        defs = self.definitions

        def walk(body: Body, bound: frozenset[str]) -> None:
            for s in body:
                child_bound = bound | s.binds_axes()
                for child_body in s.nested():
                    walk(child_body, child_bound)
                child_T: tuple[T | None, ...] = tuple(memo.get(id(defs[d])) if d in defs else None for d in s.deps())
                memo[id(s)] = fn(s, child_T, bound)

        walk(self, frozenset())
        return memo

    # -- def-use analysis ------------------------------------------------

    @cached_property
    def definitions(self) -> dict[str, Stmt]:
        """Map every SSA name produced anywhere inside this body
        (recursive) to its defining ``Stmt``.

        Built once per Body via :meth:`Stmt.defines` over :meth:`iter`;
        cached on the instance, so repeated queries (``def_of`` from
        many call sites in a single rule) are O(1) after the first
        access. Body is immutable, so the cache stays valid for its
        lifetime.

        Names not present in the dict are either Tile-input buffer
        references, constants, or SSA names defined in an enclosing
        scope outside this body — i.e. external reads.
        """
        return {n: s for s in self.iter() for n in s.defines()}

    def as_expr(self, name: str, through) -> Expr:  # noqa: ANN001 — any op predicate
        """The value cone of ``name`` read as one :class:`Expr` — :meth:`Stmt.as_expr` driven over
        THIS scope's own definitions.

        Two scope rules, both load-bearing. Only a statement this body binds DIRECTLY resolves, so a
        value bound inside a nested body (a loop's per-iteration temp) stays an atom — it is not one
        value at this level, and reading through it would state an equality that does not hold. And
        resolution is POSITIONAL: an argument binds the nearest PRECEDING definition of its name, or
        stays an atom when none precedes it. A monoid's combine rebinds its accumulators by design
        (``acc = add(sa, sb)`` whose ``sa`` reads the incoming ``acc``), so a name-keyed reading
        would either loop or answer with a value from the future.

        A name this scope does not bind is an atom too — a param, a coordinate, an enclosing scope's
        def. Every atom reaches the caller through ``free_vars()``, and the reading evaluates
        through ``Expr.eval`` with no extra code.
        """
        bindings: dict[str, list[tuple[int, Stmt]]] = {}
        read_at: dict[str, int] = {}
        for index, stmt in enumerate(self):
            for used in stmt.deps():
                read_at.setdefault(used, index)
            for bound in stmt.defines():
                bindings.setdefault(bound, []).append((index, stmt))
        memo: dict[tuple[str, int], Expr] = {}

        def read(current: str, before: int) -> Expr:
            preceding = [pair for pair in bindings.get(current, ()) if pair[0] < before]
            if not preceding:
                return Var(current)
            index, stmt = preceding[-1]
            if (current, index) not in memo:
                # A SHADOWING binding's opaque value carries its binding position. A name read
                # before it is bound already means something else in this scope — the incoming
                # accumulator a combine rebinds — so handing back the bare name would answer ``m``
                # for what is really ``maximum(m, m__o)``, silently and plausibly. A name only ever
                # bound before it is read (an ordinary temp) keeps its own spelling, which is what
                # lets a consumer act on the atom names the split reports.
                atom = f"{current}@{index}" if read_at.get(current, index) < index else current
                memo[current, index] = stmt.as_expr(atom, lambda arg: read(arg, index), through)
            return memo[current, index]

        return read(name, len(self))

    def _spine(self, name: str, expr: Expr) -> tuple[Stmt, ...]:
        """The statements a reading of ``name`` consumed — this scope's cone of it minus the CONES
        of the atoms that survived into ``expr``. Subtracting the atoms themselves is not enough: an
        atom is opaque, so everything that built it (``exp``'s ``s − m``) belongs to the atom."""
        atoms = expr.free_vars()
        kept = {id(stmt) for atom in atoms for stmt in self.backward_cone((atom,)).members}
        return tuple(stmt for stmt in self.backward_cone((name,)).members if id(stmt) not in kept)

    def factor(self, name: str, varies, through) -> Factored | None:  # noqa: ANN001 — any predicates
        """The value cone of ``name`` split into invariant and varying factors by ``varies``, or
        ``None`` when it is not one product.

        Refusals here are STRUCTURAL — the reading is a sum, or a denominator streams, so there is
        no product to split and ``Σ`` cannot commute past it. Whether a legal split is USEFUL (does
        it have both an invariant and a varying side?) is the calling rule's condition, not this one's.
        """
        import sympy  # noqa: PLC0415

        expr = self.as_expr(name, through)
        invariant: list[tuple[str, bool]] = []
        varying: list[str] = []
        for base, power in sympy.expand(expr.symbolic()).as_powers_dict().items():
            if base.is_number:
                continue
            if not isinstance(base, sympy.Symbol) or not power.is_Integer:
                return None  # a compound base — a sum, a call over several atoms — is not a factor
            count, divides = abs(int(power)), int(power) < 0
            if varies(base.name):
                if divides:
                    return None  # nothing licenses moving a fold into a varying denominator
                varying.extend([base.name] * count)
            else:
                invariant.extend([(base.name, divides)] * count)
        return Factored(tuple(invariant), tuple(varying), self._spine(name, expr))

    def linearize(self, name: str, varies, through) -> Monomials | None:  # noqa: ANN001 — any predicates
        """The value cone of ``name`` as monomials in the STREAMED atoms, each with its coefficient.

        ``((("s", 2),), 1)`` reads "the ``s²`` term has coefficient 1"; the empty monomial is the
        term free of streamed atoms. Every coefficient is an expression over the invariants by
        construction, which is what makes this the reading for "is this fold's summand linear in what
        it streams?" — and what lets a consumer ask whether a carrier's component merges as
        ``α·s + β·s__o`` with the coefficients drawn from somewhere it allows.

        ``None`` when the cone is not polynomial in the streamed atoms (a stream in a denominator or
        under an uninterpreted call) or when nothing streams at all.

        Coefficients come back as sympy expressions: a consumer DECIDES with them — is this the bare
        atom ``alpha``? is it free of the other states? — and then acts on names it already holds.
        Building IR from one is out of bounds (see :meth:`Expr.symbolic`).
        """
        import sympy  # noqa: PLC0415

        expr = self.as_expr(name, through)
        streamed = sorted(atom for atom in expr.free_vars() if varies(atom))
        if not streamed:
            return None
        try:
            poly = sympy.Poly(sympy.expand(expr.symbolic()), *(sympy.Symbol(atom) for atom in streamed))
        except sympy.PolynomialError:
            return None
        return tuple(
            (tuple((atom, power) for atom, power in zip(streamed, monomial, strict=True) if power), coeff)
            for monomial, coeff in poly.terms()
        )

    @cached_property
    def axis_names(self) -> frozenset[str]:
        """Every axis name bound by any wrapper anywhere in this body
        (``Loop`` / ``StridedLoop`` / ``Tile.axes``). Axes from
        enclosing scopes above this body are not included."""
        return frozenset(ax for s in self.iter() for ax in s.binds_axes())

    @cached_property
    def _exported_accums(self) -> frozenset[str]:
        """Accumulator names exposed by this immutable subtree."""
        from emmy.compiler.ir.stmt.leaves import Accum  # noqa: PLC0415

        out: set[str] = set()
        for stmt in self:
            if isinstance(stmt, Accum):
                out.add(stmt.name)
            for child in stmt.nested():
                out.update(child._exported_accums)
        return frozenset(out)

    @cached_property
    def ssa_defs(self) -> frozenset[str]:
        """Every SSA definition in this immutable subtree."""
        out: set[str] = set()
        for stmt in self:
            out.update(stmt.defines())
            for child in stmt.nested():
                out.update(child.ssa_defs)
        return frozenset(out)

    @cached_property
    def ssa_uses(self) -> frozenset[str]:
        """Every name a statement of this immutable subtree reads — a ``Load`` index's ``Var`` names
        among them, since a coordinate is the same ``Var`` a gathered value read would be. The
        immediate reads only (:attr:`deps_closure` reports the transitive ones); what
        :meth:`Lambda.closing` binds as params, coordinates included.
        """
        out: set[str] = set()
        for stmt in self:
            out.update(stmt.deps())
            for child in stmt.nested():
                out.update(child.ssa_uses)
        return frozenset(out)

    @cached_property
    def axis_dependencies(self) -> dict[str, frozenset[str]]:
        """Map each SSA definition to the axes that its value depends on.

        Unlike :attr:`deps_closure`, this summary never retains transitive
        SSA names. Its total size is bounded by definitions × axes, which is
        the representation normalization needs for invariant motion.

        An Accum exported from a Loop loses that Loop's reduction axis. A
        StridedLoop keeps its axis because the partial value still varies by
        partition. These are the same outside-the-loop semantics as
        :attr:`deps_closure`.
        """
        from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop  # noqa: PLC0415
        from emmy.compiler.ir.stmt.leaves import Accum  # noqa: PLC0415

        dependencies: dict[str, frozenset[str]] = {}
        axis_names = self.axis_names

        def _immediate_axes(stmt: Stmt) -> frozenset[str]:
            reads: set[str] = set(stmt.deps())
            for expr in stmt.exprs():
                reads.update(expr.free_vars())
            axes = reads & axis_names
            for name in reads:
                axes.update(dependencies.get(name, frozenset()))
            return frozenset(axes)

        def walk(body: Body) -> None:
            for stmt in body:
                for child in stmt.nested():
                    walk(child)
                if isinstance(stmt, Loop):
                    for child in stmt.body:
                        if isinstance(child, Accum):
                            dependencies[child.name] = dependencies.get(child.value, frozenset()) - {stmt.axis.name}
                    continue
                if isinstance(stmt, StridedLoop):
                    for child in stmt.body:
                        if isinstance(child, Accum):
                            dependencies[child.name] = dependencies.get(child.value, frozenset())
                    continue
                axes = _immediate_axes(stmt)
                for name in stmt.defines():
                    dependencies[name] = axes

        walk(self)
        return dependencies

    @cached_property
    def deps_closure(self) -> dict[str, frozenset[str]]:
        """For every SSA name defined in this body (recursive), the
        set of names it transitively reads. Values include both SSA
        names (defined elsewhere in the body or externally) and axis
        names (free vars from Load/Write indices, Select predicates,
        Cond conditions, etc.).

        ``Accum`` is recorded with the *outside-the-loop* form: the
        immediately-enclosing reduce-Loop's axis is subtracted from
        the value's closure, because the reduced result no longer
        varies with that axis. Reads of an Accum's *running* value
        from inside its own loop body get the wrong answer here —
        passes that gate on those (the in-loop online-softmax merge
        pattern) keep the explicit ``deps_of(c)`` check that returns
        the Accum's defining stmt directly.

        This is the substrate behind :meth:`depends_on` and
        :meth:`independent`. Most call sites prefer those phrased
        helpers over poking the closure directly.
        """
        closure: dict[str, frozenset[str]] = {}

        def _immediate(s: Stmt) -> set[str]:
            reads: set[str] = set(s.deps())
            for e in s.exprs():
                reads.update(e.free_vars())
            return reads

        def _transitive(reads: set[str]) -> frozenset[str]:
            out: set[str] = set(reads)
            for r in reads:
                out |= closure.get(r, frozenset())
            return frozenset(out)

        def walk(body: Body) -> None:
            for s in body:
                # Recurse first (post-order) so inner Accums / leaves are
                # in ``closure`` before we record this stmt or close the
                # wrapper.
                for child in s.nested():
                    walk(child)
                # After a Loop / StridedLoop closes, its body's Accums
                # become visible at the outer scope with the loop axis
                # subtracted (Loop) or kept (StridedLoop — partial value
                # carries the strided axis). Mirrors hoist_loop_invariants.
                from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop  # noqa: PLC0415
                from emmy.compiler.ir.stmt.leaves import Accum  # noqa: PLC0415

                if isinstance(s, Loop):
                    for c in s.body:
                        if isinstance(c, Accum):
                            closure[c.name] = closure.get(c.value, frozenset()) - {s.axis.name}
                    continue
                if isinstance(s, StridedLoop):
                    for c in s.body:
                        if isinstance(c, Accum):
                            closure[c.name] = closure.get(c.value, frozenset())
                    continue
                # Leaves and non-Loop wrappers (Tile, Cond): record
                # closure for each name this stmt defines.
                for name in s.defines():
                    closure[name] = _transitive(_immediate(s))

        walk(self)
        return closure

    def _stmt_reads(self, s: Stmt) -> frozenset[str]:
        """Names ``s`` transitively reads (axes + SSA), with axes bound
        by ``s`` subtracted. For leaf stmts this matches
        ``deps_closure[s.defines()[0]]``; for compound stmts (Loop /
        StridedLoop / Tile / Cond) it rolls up every nested stmt's
        reads and removes the wrapper's own bound axes — so e.g.
        ``Loop(b, ...)._stmt_reads()`` does not contain ``b``."""
        closure = self.deps_closure
        seeds: set[str] = set(s.deps())
        for e in s.exprs():
            seeds.update(e.free_vars())
        for sub in s.nested():
            for c in sub.iter():
                seeds.update(c.defines())
                seeds.update(c.deps())
                for e in c.exprs():
                    seeds.update(e.free_vars())
        out: set[str] = set(seeds)
        for n in seeds:
            out |= closure.get(n, frozenset())
        return frozenset(out) - s.binds_axes()

    def depends_on(self, a: Stmt | str | Iterable[Stmt | str], b: str | Iterable[str]) -> bool:
        """True iff anything in ``a`` transitively reads any name in
        ``b``. Directional — does not check whether ``b`` reads ``a``;
        callers can swap arg order to flip direction.

        ``a`` may be a name, a ``Stmt``, or an iterable of either.
        Passing a ``Stmt`` expands it to its read set: for a leaf this
        is the closure of its defined name; for a compound stmt
        (``Loop`` / ``StridedLoop`` / ``Tile`` / ``Cond``) it's the
        rolled-up reads of every nested stmt with the wrapper's own
        bound axes subtracted. ``b`` is always names — SSA or axis.
        Names not in :attr:`deps_closure` (external references — Tile-
        input buffers, ConstantOps, names from enclosing scopes) are
        treated as having empty closure.
        """
        b_set = {b} if isinstance(b, str) else set(b)
        if not b_set:
            return False
        closure = self.deps_closure
        a_iter: Iterable[Stmt | str] = [a] if isinstance(a, (str, Stmt)) else a
        for x in a_iter:
            if isinstance(x, Stmt):
                if not self._stmt_reads(x).isdisjoint(b_set):
                    return True
            else:
                if x in b_set or not closure.get(x, frozenset()).isdisjoint(b_set):
                    return True
        return False

    def independent(self, a: Stmt | str | Iterable[Stmt | str], b: Stmt | str | Iterable[Stmt | str]) -> bool:
        """True iff ``a`` and ``b`` share no dataflow path — neither
        ``a`` transitively reads any name in ``b`` nor vice versa.
        Symmetric counterpart to :meth:`depends_on`. Use this when
        asking "are these two things related at all?" (fusion safety,
        motion legality); use :meth:`depends_on` when direction
        matters (invariance, hoist gates).

        For symmetric usage, ``Stmt`` arguments on either side expand
        to their *defined* names (what the stmt produces) when used as
        a read target — same swap-and-call behavior as
        :meth:`depends_on`."""

        def _as_target(x: Stmt | str | Iterable[Stmt | str]) -> set[str]:
            items = [x] if isinstance(x, (str, Stmt)) else list(x)
            out: set[str] = set()
            for it in items:
                if isinstance(it, Stmt):
                    for sub in it.nested():
                        for c in sub.iter():
                            out.update(c.defines())
                    out.update(it.defines())
                else:
                    out.add(it)
            return out

        return not (self.depends_on(a, _as_target(b)) or self.depends_on(b, _as_target(a)))

    def deps_of(self, stmt: Stmt) -> tuple[Stmt | None, ...]:
        """Defining stmts inside this body for each of ``stmt``'s SSA
        reads, in the same order as ``stmt.deps()``. Position-preserving:
        each entry is the ``Stmt`` that produces the corresponding dep,
        or ``None`` if the dep is read but not defined locally (Tile-
        input buffer reference, constant, or an SSA from an enclosing
        scope — i.e. an external read).

        Replaces the ``[body.def_of(d) for d in stmt.deps()]`` pattern
        rules used to write inline. Use ``isinstance(s, T)`` predicates
        to filter results — ``None`` won't match any concrete stmt
        type, so external reads drop out automatically; check
        ``s is None`` explicitly when the gate cares about externals."""
        defs = self.definitions
        return tuple(defs.get(d) for d in stmt.deps())

    # -- dependence cones --------------------------------------------------

    def backward_cone(self, roots: Iterable[str]) -> Cone:
        """The backward dependence :class:`Cone` of ``roots`` over THIS
        body's immediate stmts: every top-level member whose exposed names
        transitively feed a root, in body order. Names resolving to no
        member here (axis vars, enclosing/sibling scopes, buffer constants)
        surface in ``external_reads`` — chain another scope level by seeding
        its ``backward_cone`` with them. A root not defined at this level is
        itself an external read (members come out empty)."""
        by_name: dict[str, Stmt] = {}
        for s in self:
            for n in _exposed_defines(s):
                by_name[n] = s
        member_ids: set[int] = set()
        external: set[str] = set()
        pending = list(roots)
        seen: set[str] = set()
        while pending:
            n = pending.pop()
            if n in seen:
                continue
            seen.add(n)
            s = by_name.get(n)
            if s is None:
                external.add(n)
                continue
            if id(s) in member_ids:
                continue
            member_ids.add(id(s))
            pending.extend(free_names(s))
        return Cone(members=tuple(s for s in self if id(s) in member_ids), external_reads=frozenset(external))

    def defs_die_at(self, members: Iterable[Stmt], *, roots: Iterable[str], allowed: Iterable[Stmt]) -> bool:
        """True iff no stmt in this body outside ``members`` reads a name
        they expose — except the ``allowed`` stmts, which may read names in
        ``roots`` (and nothing else from the cone). The escape check for
        cutting a cone out: its values must die at the designated consumers
        (e.g. the matmul multiplies) or the cut would break a reader left
        behind. ``members`` may span several scope levels of this body
        (a cell cone plus its prologue deps); each member's whole subtree
        counts as inside."""
        members = tuple(members)  # may arrive as a generator; iterated twice
        member_ids = {id(s) for s in members}
        moved_defs: set[str] = set()
        for s in members:
            moved_defs |= _exposed_defines(s)
        root_set = frozenset(roots)
        allowed_ids = {id(s) for s in allowed}

        def walk(stmts: Iterable[Stmt]) -> bool:
            for s in stmts:
                if id(s) in member_ids:
                    continue  # the whole subtree moves; internal uses are fine
                reads = set(s.deps()) & moved_defs
                if reads and not (id(s) in allowed_ids and reads <= root_set):
                    return False
                for sub in s.nested():
                    if not walk(sub):
                        return False
            return True

        return walk(self)

    # -- type-filtered lookups -------------------------------------------

    def of_type(self, *types: type) -> tuple[Stmt, ...]:
        """Top-level stmts in this body matching any of the given
        types. The named helpers below (:meth:`loads`, :meth:`writes`,
        :meth:`accums`, :meth:`loops`, :meth:`stages`) walk the entire
        body recursively; use ``of_type`` when you need only the
        top-level slice (e.g. "Loops directly in this Tile body, not
        inside nested wrappers")."""
        return tuple(s for s in self if isinstance(s, types))

    def iter_of_type(self, *types: type) -> tuple[Stmt, ...]:
        """All stmts (recursive — via :meth:`iter`) matching any of the
        given types. The base primitive the named helpers
        (:meth:`loads`, :meth:`writes`, ...) wrap."""
        return tuple(s for s in self.iter() if isinstance(s, types))

    @cached_property
    def loads(self) -> tuple[Stmt, ...]:
        """All ``Load`` stmts in the body (recursive). Replaces the
        per-Op ``loads`` properties on ``LoopOp`` / ``TileOp`` /
        ``KernelOp``. Cached on the instance — Body is immutable."""
        from emmy.compiler.ir.stmt.leaves import Load  # noqa: PLC0415

        return self.iter_of_type(Load)

    @cached_property
    def writes(self) -> tuple[Stmt, ...]:
        """All ``Write`` stmts in the body (recursive)."""
        from emmy.compiler.ir.stmt.leaves import Write  # noqa: PLC0415

        return self.iter_of_type(Write)

    @cached_property
    def accums(self) -> tuple[Stmt, ...]:
        """All ``Accum`` stmts in the body (recursive). May contain
        multiple Accums sharing a single accumulator name (matmul-shape
        K-inner reduces, 008's per-cell replicated accumulator chains).
        Validation enforces op-consistency across same-name Accums in
        ``LoopOp.__post_init__``; callers that want a one-per-name view
        can dedup at the call site (``{a.name: a for a in body.accums}``)."""
        from emmy.compiler.ir.stmt.leaves import Accum  # noqa: PLC0415

        return self.iter_of_type(Accum)

    @cached_property
    def loops(self) -> tuple[Stmt, ...]:
        """All ``Loop`` stmts in the body (recursive)."""
        from emmy.compiler.ir.stmt.blocks import Loop  # noqa: PLC0415

        return self.iter_of_type(Loop)

    # -- structural identity --------------------------------------------

    def structural_key(self, *, structural: bool = True) -> str:
        """Implements :class:`emmy.compiler.structural.Structural`.

        Canonical digest used for structural-equivalence queries. Two
        bodies that differ only by SSA / axis names, commutative-arg
        order, or external-buffer names produce the same key; with the
        default ``structural=True`` the specific op within a
        compute-unit cluster (``add`` vs ``sub``, ``div`` vs ``mod``)
        also normalizes away — the SCHEDULE-EQUIVALENT reading, since
        cluster siblings offer the same schedule space and want the same
        schedule. ``structural=False`` keeps the exact ops: the reading
        for consumers to whom ``relu`` and ``gelu`` are different
        kernels (their latency differs even under one schedule).

        Built by re-running :func:`normalize_body` with ``hoist=False``
        (safe for both Loop-IR and Tile-IR bodies — hoisting can move
        Loads above Stage decls in Tile bodies) and
        ``canonical_buffers=True`` (renames ``Load.input`` /
        ``Write.output`` to ``b0, b1, ...``). Cached on the
        instance — Body is immutable."""
        return self._structural_key_clustered if structural else self._structural_key_exact

    @cached_property
    def _structural_key_clustered(self) -> str:
        # Both flavors delegate to a module-level lru_cache keyed by Body
        # content (Body is ``tuple[Stmt, ...]`` and every Stmt subclass is
        # a frozen dataclass, so the cache key is structural). Two
        # different Body instances with identical stmts share the one
        # ``normalize_body`` call — matters in tune mode where
        # ``_record_op_inventory`` walks the source chain of every
        # CudaOp in every terminal and hammers ``identity_key(with_io=True, with_knobs=True)`` ->
        # ``Body.structural_key()`` on bodies that frequently recur
        # structurally across variants.
        return _shared_structural_key(self, True)

    @cached_property
    def _structural_key_exact(self) -> str:
        return _shared_structural_key(self, False)


@lru_cache(maxsize=4096)
def _shared_structural_key(body: Body, cluster: bool) -> str:
    """Module-level memoization for :meth:`Body.structural_key`.

    The formula is fixed per flavor: ``normalize_body(body, hoist=False,
    canonical_buffers=True, cluster_ops=cluster)`` rendered through
    :func:`~emmy.compiler.structural.form`. Structural, not the
    pretty text it used to join: ``pretty()`` is the human rendering, and
    a cosmetic change to how a statement prints must not re-key every
    kernel that contains it. With every concrete ``Stmt`` subclass a frozen
    dataclass and ``Body`` a ``tuple[Stmt, ...]`` subclass, equal-content
    bodies hash equal — so two structurally identical Body instances
    share one normalize+pretty walk through this cache. Tune mode hits
    this hard from ``_record_op_inventory`` (one ``identity_key(with_io=True, with_knobs=True)`` call
    per ancestor in every CudaOp's source chain, per terminal candidate).

    Generic :func:`normalize_body` callers with other flags don't share
    this cache — ``cluster_ops=True`` collapses semantically distinct ops
    to a single cluster representative (``add``↔``sub``, ``div``↔``mod``,
    …), which is the right canonicalization for structural-equivalence
    queries but would be a *correctness bug* for any callsite running
    the normalized body.
    """
    from emmy.compiler.ir.stmt.normalize import normalize_body  # noqa: PLC0415
    from emmy.compiler.structural import digest, form  # noqa: PLC0415

    normalized = normalize_body(body, hoist=False, canonical_buffers=True, cluster_ops=cluster)
    return digest(form(normalized))


def refs_axis(s: Stmt, name: str) -> bool:
    """``s`` references axis ``name`` in any carried expr (deep) — ``Stmt.exprs``: a ``Load`` /
    ``Write`` index, a ``Select``'s branch predicates. Both spellings are coordinate reads, so both
    make the stmt vary with the axis; a mask ``Select`` read as invariant would be hoisted out of
    the per-cell body it predicates."""
    if any(name in e.free_vars() for e in s.exprs()):
        return True
    return any(refs_axis(child, name) for b in s.nested() for child in b)


def stmt_axis_names(stmts) -> set[str]:
    """Every loop induction variable bound anywhere in ``stmts`` (deep). A composed structural node
    sitting in the body needs no special case — it is a ``Stmt``, so its children are reached through
    the same ``nested()`` walk as any block stmt's."""
    out: set[str] = set()
    for s in stmts:
        ax = getattr(s, "axis", None)
        if ax is not None and hasattr(ax, "name"):
            out.add(ax.name)
        for b in s.nested():
            out |= stmt_axis_names(b)
    return out
