"""``Lambda`` — the ONE binder kind, shared by every IR level.

A lambda is a PURE term: explicit binders over the reused stmt vocabulary, an A-normal-form
let-chain in ``body``, the returned defs named by ``results``. It binds; it never executes.
Nothing here is a :class:`~emmy.compiler.ir.stmt.base.Stmt` — see ``ir/ARCHITECTURE.md`` for the
invariant and for how a pure term reaches a statement position (it is RENDERED to stmts, never
spliced in as one).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.stmt.base import pretty_body
from emmy.compiler.ir.stmt.body import Body, _exposed_defines
from emmy.compiler.ir.stmt.leaves import Assign, Load


def _canonical_body_order(body: Body) -> Body:
    """Return a deterministic dependency-respecting order for a pure ANF body."""
    stmts = tuple(body)
    if len(stmts) <= 1:
        return body

    def token(stmt) -> tuple:
        op = getattr(stmt, "op", None)
        return (
            type(stmt).__name__,
            getattr(op, "name", "") if op is not None else "",
            stmt.input if isinstance(stmt, Load) else "",
            len(getattr(stmt, "args", ()) or ()),
        )

    def reads(stmt) -> set[str]:
        out = set(stmt.deps())
        for nested in stmt.nested():
            for child in nested:
                out |= reads(child)
        return out

    definitions = [
        set(stmt.defines()) | {name for nested in stmt.nested() for child in nested for name in child.defines()} for stmt in stmts
    ]
    dependencies = [reads(stmt) for stmt in stmts]
    placed = []
    remaining = list(range(len(stmts)))
    while remaining:
        remaining_definitions = {name for index in remaining for name in definitions[index]}
        ready = [index for index in remaining if not dependencies[index] & (remaining_definitions - definitions[index])]
        if not ready:
            return body
        selected = min(ready, key=lambda index: (token(stmts[index]), index))
        placed.append(stmts[selected])
        remaining.remove(selected)
    return Body(placed)


def _normalize_body(body: Body) -> Body:
    """The construction canonicalization of a pure body — an idempotent transform, applied by
    :meth:`Lambda.__post_init__`: a dependency-safe statement order and sorted commutative arguments,
    the context-independent storage invariants a term's structural identity reads directly."""
    ordered = _canonical_body_order(body)
    return Body(
        replace(stmt, args=tuple(sorted(stmt.args))) if isinstance(stmt, Assign) and stmt.op.commutative and len(stmt.args) > 1 else stmt
        for stmt in ordered
    )


@dataclass(frozen=True)
class Lambda:
    """Explicit binders over the REUSED stmt vocabulary — the ONE binder kind, common to every IR
    level. Not a second expression language: ``body`` is a :class:`Body` of PURE stmts only
    (A-normal form ≙ a let-chain), ``params`` the binders, ``results`` the returned defs
    (replacing every ``out`` / last-def convention).

    ``__post_init__`` validates the LOCAL formation invariant: every body stmt passes the
    :attr:`Stmt.pure` trait (declared on the interface, conservative ``False`` default —
    ``Load`` / ``Assign`` and the structural nodes opt in; ``Accum`` / ``Write`` / ``Init`` /
    ``Loop`` never do; no isinstance whitelist, so a new stmt kind is excluded until it declares
    itself) and every result is defined. The CONTEXTUAL half — free names ⊆ params ∪ enclosing
    iteration vars — is the consuming Fold's check, since a bare Lambda
    cannot know its scope.

    α-invariance is CANONICAL RENUMBERING (the existing rename machinery), not de Bruijn:
    :meth:`canonical` renumbers params (``_p0…``) and internal defs (``_v0…``) in walk order,
    leaving free names untouched; :meth:`alpha_eq` compares canonical forms."""

    params: tuple[str, ...]
    body: Body
    results: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.params, tuple):
            object.__setattr__(self, "params", tuple(self.params))
        body = _normalize_body(Body.coerce(self.body))
        if not isinstance(self.body, Body) or body != self.body:
            object.__setattr__(self, "body", body)
        if not isinstance(self.results, tuple):
            object.__setattr__(self, "results", tuple(self.results))
        impure = [type(s).__name__ for s in self.body if not s.pure]
        if impure:
            raise ValueError(f"Lambda body must be pure — impure stmt kind(s): {impure}")
        defined = set(self.params)
        for s in self.body:
            defined |= _exposed_defines(s)
        missing = [r for r in self.results if r not in defined]
        if missing:
            raise ValueError(f"Lambda results {missing} are not defined by the body or params")
        # CLOSED IN VALUES. A lambda is a function, so every VALUE its body reads is a param or one
        # of its own defs. Asked through ``Body.ssa_uses`` — SSA reads only — so an index
        # COORDINATE is not mistaken for a value: an axis is the space the body is evaluated over,
        # supplied by the enclosing binder, and a ``Load``'s index ``Var`` is the same ``Var`` a
        # value read would be. Deciding it by kind is what this reading does and what asking for
        # every free name could not: binding coordinates as params removed the distinction rather
        # than making it, and every downstream reader then had to resolve those unapplied trailing
        # params itself. :meth:`closing` FORMS a closed lambda; this only refuses.
        free = self.body.ssa_uses - defined
        if free:
            raise ValueError(
                f"Lambda body reads {sorted(free)} it does not bind. Pass them as params — "
                f"Lambda.closing(params, body, results) appends whatever the body still reads."
            )

    @classmethod
    def closing(cls, params: tuple[str, ...], body, results: tuple) -> Lambda:
        """Build a CLOSED lambda from its parts — the one former.

        A term is a function, so it has no free VALUES. The caller supplies the params it knows —
        a fold's iteration var, the names its operand edges bind — and every remaining value the
        body reads is appended as a TRAILING param. An index coordinate is not one: it rides the
        operand edges' index expressions and is supplied by the enclosing binder. Trailing, never interleaved: the
        operand correspondence is the param PREFIX, so appending leaves every positional read of
        it intact.

        Built in ONE step: the body is normalized here, so the residual is computed against the
        same body the lambda stores rather than against a throwaway built open and rebuilt.

        Callers form; :meth:`__post_init__` refuses. They stay separate because a constructor that
        repaired its own input would enforce nothing."""
        body = _normalize_body(Body.coerce(body))
        bound = set(params)
        for stmt in body:
            bound |= _exposed_defines(stmt)
        # VALUE reads the body does not define, plus any RESULT it does not define either: a write
        # may pass an enclosing value straight through (``o[j] = acc`` over an already-reduced
        # accumulator), and that result has no def to name, so it binds as a param like any read.
        # Coordinates are deliberately absent — see the closedness gate in ``__post_init__``.
        residual = set(body.ssa_uses)
        residual |= set(results)
        return cls(params=(*params, *sorted(residual - bound)), body=body, results=tuple(results))

    @classmethod
    def componentwise(cls, ops, names: tuple[str, ...]) -> Lambda:
        """The componentwise binary program — each result its own ⊕ over its two operands,
        ``nᵢ = ⊕ᵢ(nᵢ, nᵢ__o)``: ``S × S → S`` with the second operand spelled ``<n>__o``, a plain
        fold's combine, the shape :meth:`components` reads back."""
        other = tuple(f"{name}__o" for name in names)
        body = Body(tuple(Assign(name=name, op=op, args=(name, second)) for name, op, second in zip(names, ops, other, strict=True)))
        return cls(params=(*names, *other), body=body, results=tuple(names))

    def components(self) -> tuple[ElementwiseImpl, ...] | None:
        """The per-result ops when this program is componentwise — every result ``rᵢ = ⊕ᵢ(pᵢ, pₙ₊ᵢ)``
        on its own, in either argument order for a commutative ⊕ — else ``None``: the planar-vs-
        twisted reading of a fold's combine (a cross-component read or a rescale temp fails it)."""
        n = len(self.results)
        if len(self.body) != n or len(self.params) != 2 * n:
            return None
        definitions = self.body.definitions
        ops = []
        for index, result in enumerate(self.results):
            stmt, pair = definitions.get(result), (self.params[index], self.params[n + index])
            if not isinstance(stmt, Assign) or (stmt.args != pair and not (stmt.op.commutative and stmt.args == pair[::-1])):
                return None
            ops.append(stmt.op)
        return tuple(ops)

    @property
    def defined(self) -> frozenset[str]:
        """Every name this lambda binds — params plus every def its body exposes (deep)."""
        out = set(self.params)
        for s in self.body:
            out |= _exposed_defines(s)
        return frozenset(out)

    def __getstate__(self):
        """Pickle the stored fields only — memoized reads recompute after transport."""
        return {name: self.__dict__[name] for name in self.__dataclass_fields__ if name in self.__dict__}

    def cone(self, name: str) -> Lambda:
        """The closed cone of one definition — the statements of the body ``name`` depends on, as a
        lambda over what they read (a param names itself: an empty body returning it). What a
        reader asks of one result without the rest of the body: the score a fold's lift computes,
        the value one component of a projection denotes. Params in :meth:`closing`'s order."""
        members = () if name in self.params else tuple(self.body.backward_cone((name,)).members)
        return Lambda.closing((), Body(members), (name,))

    def rename(self, names) -> Lambda:
        """α-rename params, body and results through one map — a mapping or a callable — the
        lockstep every consumer of a stored lambda applies: the fold's rewrite handler, the
        combine's state rename, a recipe instantiated onto a term's names."""
        lookup = names.get if hasattr(names, "get") else None

        def rn(name: str) -> str:
            return lookup(name, name) if lookup is not None else names(name)

        return Lambda(
            params=tuple(rn(p) for p in self.params),
            body=Body(tuple(s.rewrite(rn) for s in self.body)),
            results=tuple(rn(r) for r in self.results),
        )

    def canonical(self) -> Lambda:
        """The α-canonical form: params renumber to ``_p0…`` and internal defs to ``_v0…`` in
        walk order; free names (and float results) pass through unchanged. Deterministic, so two
        lambdas equal up to bound-name choice have EQUAL canonical forms — the α-invariant
        equality/hash substrate (:meth:`alpha_eq`)."""
        mapping = {p: f"_p{i}" for i, p in enumerate(self.params)}
        n = 0
        for s in self.body.iter():
            for d in s.defines():
                if d not in mapping:
                    mapping[d] = f"_v{n}"
                    n += 1
        return self.rename(mapping)

    def alpha_eq(self, other: Lambda) -> bool:
        """α-invariant equality — canonical forms compared structurally."""
        return isinstance(other, Lambda) and self.canonical() == other.canonical()

    def pretty(self, indent: str = "") -> list[str]:
        rs = ", ".join(self.results)
        head = f"{indent}λ({', '.join(self.params)}) -> ({rs})"
        return [head, *pretty_body(self.body, indent + "    ")]
