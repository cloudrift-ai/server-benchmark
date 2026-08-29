"""``Lambda`` — the ONE binder kind, shared by every IR level.

A lambda is a PURE term: explicit binders over the reused stmt vocabulary, an A-normal-form
let-chain in ``body``, the returned defs named by ``results``. It binds; it never executes.
Nothing here is a :class:`~emmy.compiler.ir.stmt.base.Stmt` — see ``ir/ARCHITECTURE.md`` for the
invariant and for how a pure term reaches a statement position (it is RENDERED to stmts, never
spliced in as one).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from emmy.compiler.ir.pure.normalize import normalize_lambda_body
from emmy.compiler.ir.stmt.base import pretty_body
from emmy.compiler.ir.stmt.body import Body, _exposed_defines, _member_reads


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

    A result may also be a bare ``float`` literal — the injection ι is spelled in the lift
    (softmax's singleton is ``(x, 1)``, flash's ``(s, 1, v)``), and a literal component has no
    def to name (mirrors the ``Channel.term: str | float`` convenience it replaces).

    α-invariance is CANONICAL RENUMBERING (the existing rename machinery), not de Bruijn:
    :meth:`canonical` renumbers params (``_p0…``) and internal defs (``_v0…``) in walk order,
    leaving free names untouched; :meth:`alpha_eq` compares canonical forms."""

    params: tuple[str, ...]
    body: Body
    results: tuple[str | float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.params, tuple):
            object.__setattr__(self, "params", tuple(self.params))
        body = normalize_lambda_body(Body.coerce(self.body))
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
        missing = [r for r in self.results if isinstance(r, str) and r not in defined]
        if missing:
            raise ValueError(f"Lambda results {missing} are not defined by the body or params")

    @property
    def defined(self) -> frozenset[str]:
        """Every name this lambda binds — params plus every def its body exposes (deep)."""
        out = set(self.params)
        for s in self.body:
            out |= _exposed_defines(s)
        return frozenset(out)

    def free_names(self) -> frozenset[str]:
        """Names the body reads that this lambda does not bind — the contextual-invariant read
        the consuming Fold checks against its iteration vars."""
        return self._free_names

    @cached_property
    def _free_names(self) -> frozenset[str]:
        # Memoized: the lambda is immutable, and every scope walk that reaches a nested Fold asks
        # its lift's free names again — uncached, a deep fused tree pays the full body walk once
        # per enclosing level.
        reads: set[str] = set()
        for s in self.body:
            reads |= _member_reads(s)
        return frozenset(reads) - self.defined

    def __getstate__(self):
        """Pickle the stored fields only — memoized reads recompute after transport."""
        return {name: self.__dict__[name] for name in self.__dataclass_fields__ if name in self.__dict__}

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

        def rn(name: str) -> str:
            return mapping.get(name, name)

        return Lambda(
            params=tuple(mapping[p] for p in self.params),
            body=Body(tuple(s.rewrite(rn) for s in self.body)),
            results=tuple(rn(r) if isinstance(r, str) else r for r in self.results),
        )

    def alpha_eq(self, other: Lambda) -> bool:
        """α-invariant equality — canonical forms compared structurally."""
        return isinstance(other, Lambda) and self.canonical() == other.canonical()

    def pretty(self, indent: str = "") -> list[str]:
        rs = ", ".join(r if isinstance(r, str) else repr(r) for r in self.results)
        head = f"{indent}λ({', '.join(self.params)}) -> ({rs})"
        return [head, *pretty_body(self.body, indent + "    ")]
