"""Shared base class for body-carrying IR ops (``LoopOp`` / ``TileOp`` /
``KernelOp``).

Lives inside the ``ir.stmt`` package — and imports its dependencies
(``Body``, ``Load``, ``Write``, ``Stmt``, ``pretty_body``) from their leaf
modules directly rather than via the package ``__init__`` — so that
``ir.stmt.normalize`` (which loads ``Stage`` from ``ir.tile.ir``) doesn't
re-enter ``ir.stmt.__init__`` through the back door when ``tile.ir``
imports ``BodyOp``. The leaf-module imports load cleanly because none of
them touches ``ir.tile``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from emmy.compiler.dtype import F32
from emmy.compiler.ir.base import ConstantOp, Op
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.base import pretty_body as _pretty_body_stmts
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Load, Write
from emmy.compiler.structural import digest
from emmy.compiler.tensor import Tensor


@dataclass
class BodyOp(Op):
    """Shared base for IR ops that carry a structured :class:`Body` of
    stmts plus a kernel name — ``LoopOp`` (loop IR), ``TileOp`` (tile IR),
    ``KernelOp`` (kernel IR). Provides:

    - ``body`` / ``name`` fields, plus a ``__post_init__`` that coerces
      tuple-bodies to :class:`Body`,
    - ``__iter__`` over the body's deep stmt iterator,
    - ``loads`` / ``writes`` typed iters,
    - a unified ``pretty_body`` returning the indented body listing
      (the surrounding dump emits the kernel name / I/O label).

    ``__post_init__`` pre-populates ``inputs`` / ``outputs`` (inherited
    from :class:`Op`) with body-derived names keyed to placeholder
    ``Tensor(name, (), F32)`` values, so pre-match callers (lifting /
    fusion / tests) already see the right keys + ordering without a
    matcher run. The body walk is generic — each :class:`Stmt`
    subclass declares its :meth:`Stmt.external_reads` /
    :meth:`Stmt.external_writes` / :meth:`Stmt.local_decls`, and
    ``BodyOp`` aggregates them (filtering reads / writes that name a
    locally-declared buffer like an ``Smem`` or ``Stage``). The
    matcher's :meth:`populate_io` hook then overrides each entry with
    the real graph-sourced ``Tensor`` and sanity-checks that the body
    has no Load / Write naming a buffer the dict doesn't cover."""

    body: Body = field(default_factory=Body)
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            self.body = Body.coerce(self.body)
        self._seed_io_placeholders()

    def _seed_io_placeholders(self) -> None:
        """Populate empty ``inputs`` / ``outputs`` with body-derived names
        keyed to placeholder ``Tensor(name, (), F32)`` values. Skips
        either dict if the caller already supplied entries."""
        in_names, out_names = self._derive_io_names()
        if not self.inputs:
            self.inputs = {n: Tensor(n, (), F32) for n in in_names}
        if not self.outputs:
            self.outputs = {n: Tensor(n, (), F32) for n in out_names}

    def _derive_io_names(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Walk the body once; aggregate per-stmt
        :meth:`Stmt.external_reads` / :meth:`Stmt.external_writes` and
        filter out any name covered by some stmt's
        :meth:`Stmt.local_decls` (smem / staged buffers). Returns
        ``(input_names, output_names)`` in body first-use order."""
        decls: set[str] = set()
        for s in self:
            decls.update(s.local_decls())
        ins: dict[str, None] = {}
        outs: dict[str, None] = {}
        for s in self:
            for r in s.external_reads():
                if r not in decls:
                    ins.setdefault(r, None)
            for w in s.external_writes():
                if w not in decls:
                    outs.setdefault(w, None)
        return tuple(ins), tuple(outs)

    def __iter__(self) -> Iterator[Stmt]:
        return self.body.iter()

    @property
    def loads(self) -> tuple[Load, ...]:
        return self.body.iter_of_type(Load)

    @property
    def writes(self) -> tuple[Write, ...]:
        return self.body.iter_of_type(Write)

    def populate_io(self, graph, node) -> None:  # noqa: ANN001, ARG002 — matches Op.populate_io signature
        """Override :meth:`Op.populate_io`. Sanity-checks that every body
        external read / write names a buffer already covered by
        ``inputs`` / ``outputs`` (caught early — a rule that mutated the
        body without re-seeding I/O surfaces here, not on a later mystery
        KeyError), then replaces each entry's placeholder Tensor with the
        real graph-sourced one when the buffer is a graph node."""
        from emmy.compiler.ir.stmt.leaves import ZeroPrologue  # noqa: PLC0415 — leaf import; ir.stmt.__init__ would cycle

        body_in, body_out = self._derive_io_names()
        stray_in = [n for n in body_in if n not in self.inputs]
        if stray_in:
            raise ValueError(f"{type(self).__name__}: body has Load buffers {stray_in} not covered by inputs={list(self.inputs)}")
        stray_out = [n for n in body_out if n not in self.outputs]
        if stray_out:
            raise ValueError(f"{type(self).__name__}: body has Write buffers {stray_out} not covered by outputs={list(self.outputs)}")
        # Single-source-of-truth: a body-written buffer that IS a graph buffer
        # must be produced by THIS node (an output slot), never by another node.
        # The one sanctioned exception is a delegated zero-init (``ZeroPrologue``
        # writes a downstream accumulator on purpose — ``005_delegate_zero_init``).
        delegated = {s.dst for s in self.body.iter() if isinstance(s, ZeroPrologue)}
        own = set(node.buffer_names())
        foreign = [n for n in body_out if n not in own and n not in delegated and graph.producer(n) is not None]
        if foreign:
            raise ValueError(f"{type(self).__name__}: body writes buffer(s) {foreign} produced by another node — declare them as outputs")
        for name in self.inputs:
            t = _tensor_for_buffer(graph, name)
            if t is not None:
                self.inputs[name] = t
        for name in self.outputs:
            t = _tensor_for_buffer(graph, name)
            if t is not None:
                self.outputs[name] = t

    def pretty_body(self) -> str:
        """Indented body listing — one stmt per line via per-stmt
        ``pretty``. Callers (``format_kernels`` /
        ``Candidate._format_nodes``) emit the surrounding name / I/O
        label themselves; duplicating it here would just rot."""
        return "\n".join(_pretty_body_stmts(self.body, "    "))

    def cache_key(self) -> str | None:
        """Override :meth:`Op.cache_key`: digest of the dialect tag plus the canonical body
        (:meth:`Body.structural_key` — SSA / axis / commutative-arg / external-buffer names
        normalized away) plus the knob dict."""
        return digest(type(self).__name__, self.body.structural_key(), self._knob_key())


def _tensor_for_buffer(graph, name: str) -> Tensor | None:  # noqa: ANN001 — Graph lives in compiler.graph; would cycle to import.
    """Return the graph tensor traveling under buffer ``name`` (any output
    slot of its producer), but for ``ConstantOp`` producers stamp
    ``constant=True`` (and ``value`` for scalar literals) onto a fresh
    Tensor so downstream consumers can recognize literal-scalar buffers
    without re-querying the graph for ConstantOp predecessors. ``None``
    when the buffer isn't a graph buffer (matcher-known external)."""
    t = graph.buffer(name)
    if t is None:
        return None
    node = graph.producer(name)
    if isinstance(node.op, ConstantOp):
        return Tensor(t.name, t.shape, t.dtype, constant=True, value=node.op.value)
    return t
