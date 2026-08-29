"""Base Op class and boundary sentinels shared across every IR level.

``Op`` is the root of all tensor operations. ``InputOp`` and ``ConstantOp``
are boundary sentinels that carry tensors into the graph without computing
anything — they appear unchanged from the frontend stage all the way through
to fusion, and the numpy backend supplies their values at runtime.

The shared base for body-carrying ops (``LoopOp`` / ``KernelOp``) is
:class:`emmy.compiler.ir.stmt.ir.BodyOp`; ``TileOp`` subclasses ``Op``
directly (it stores a pure term, not a body).

``_keepdim_axis`` is a small shape helper used by both ``ReduceOp`` (minimal IR,
``tensor.py``) and ``MeanOp`` (frontend IR, ``frontend.py``). Keeping it here
avoids a frontend→tensor dependency for a single function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from frozendict import frozendict

from emmy.compiler.structural import digest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from emmy.compiler.graph import Graph, Node
    from emmy.compiler.ir.expr import Expr
    from emmy.compiler.tensor import Tensor


class _ClassProperty:
    """A read-only property answered by the CLASS (``@classmethod @property`` chaining is gone
    since Python 3.13) — reachable from both ``LoopOp.dialect`` and ``op.dialect``."""

    def __init__(self, fget) -> None:
        self.fget = fget

    def __get__(self, obj, owner):
        return self.fget(owner)


def _io_fingerprint(op: Op) -> tuple:
    """The dtypes and hint-free shapes of an op's buffers — operands, then outputs, in program
    order: the two boundary facts the body digest canonicalizes away (buffer names normalize to
    ``b0, b1, …``, so the types ride beside the key). An f16 and an f32 trace of one body key
    apart (different atom eligibility), and so do a ``(128, 128)`` and a ``(4, 32, 128)`` output
    over one iteration space — the buffer's dim spelling decides store addressability
    (``lowering/_addr.split_addressable``), and a golden measured on the flat kernel must not
    join a kernel that cannot realize its row."""

    def sig(t) -> tuple:
        return (str(t.dtype), tuple(str(d.as_static()) if d.is_static else "sym" for d in t.shape))

    return (*(sig(t) for t in op.inputs.values()), "->", *(sig(t) for t in op.outputs.values()))


@dataclass
class Op:
    """Base class for all operations.

    ``source`` is the predecessor op in any rewrite chain — the engine
    stamps it automatically on every 1:1 in-place rebind
    (``_apply_one`` Op branch), so a fully-lowered ``CudaOp`` keeps the
    full path back through ``KernelOp → TileOp → LoopOp`` (or any other
    chain a future pipeline introduces) via repeated ``.source``
    traversals. Keyword-only so positional construction of subclass
    fields keeps working; excluded from :meth:`Graph.structural_key`
    via ``_STRUCTURAL_SKIP_FIELDS`` — pure attribution metadata, not
    part of dataflow identity.

    ``inputs`` / ``outputs`` are per-op-instance buffer maps populated by
    the matcher (``_match_at``) just after a match is built — the matcher
    calls :meth:`populate_io` on every matched node, snapping in real
    :class:`Tensor` values from the surrounding graph so rule rewrites can
    read shapes / dtypes off ``op.inputs[name]`` without re-querying.
    Default: predecessor outputs by predecessor id, this node's output by
    its own id. :class:`BodyOp` overrides to walk body-derived buf names
    instead. Excluded from equality / repr — pure derived view of the
    surrounding graph at match time.
    """

    source: Op | None = field(default=None, kw_only=True, repr=False, compare=False)
    # Free-form metadata dict for rules to stamp the knobs they used
    # (e.g. ``{"BN": 64, "BM": 64}`` from ``005_blockify_launch``). The
    # engine's ``_apply_one`` merges the predecessor's knobs forward on
    # every 1:1 rebind, so a fully-lowered ``CudaOp`` carries every
    # autotune knob picked along the chain. Excluded from structural
    # identity and equality — pure attribution metadata.
    knobs: dict = field(default_factory=dict, kw_only=True, repr=False, compare=False)
    inputs: frozendict[str, Tensor] = field(default_factory=frozendict, kw_only=True, repr=False, compare=False)
    outputs: frozendict[str, Tensor] = field(default_factory=frozendict, kw_only=True, repr=False, compare=False)

    def __setattr__(self, name: str, value) -> None:
        """The identity-safety hook: ``inputs`` / ``outputs`` land as ``frozendict`` (immutable
        in place; Python 3.15's builtin, backfilled by the ``frozendict`` package on our 3.12+
        floor), and their reassignment — the ONE way the io view changes (``populate_io``, the
        ``BodyOp`` seed, test setups) — drops the cached io digest. Invalidation lives here,
        not at call sites, so it cannot be forgotten."""
        if name in ("inputs", "outputs"):
            self.__dict__.pop("_io_key", None)
            if not isinstance(value, frozendict):
                value = frozendict(value)
        super().__setattr__(name, value)

    def populate_io(self, graph: Graph, node: Node) -> None:
        """Refresh ``inputs`` / ``outputs`` from the surrounding graph.
        Called by the matcher after a match is built. Default mapping
        works for non-body ops (one Tensor per predecessor / this node).
        :class:`BodyOp` overrides to walk body-derived buf names."""
        self.inputs = frozendict({pid: t for pid in node.inputs if (t := graph.buffer(pid)) is not None})
        self.outputs = frozendict(zip(node.buffer_names(), node.outputs, strict=True))

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        """Derive the output shape from input shapes. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}.infer_output_shape not implemented")

    def forward(self, *inputs):
        """Compute the operation using numpy arrays. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}.forward not implemented")

    def pretty_body(self) -> str | None:
        """Render this op's body for kernel dumps. Default: None (skip)."""
        return None

    def validate(self, ctx) -> bool:  # noqa: ARG002 — ``ctx`` consumed by subclass overrides
        """Sanity-check this op against the compilation context. Return
        ``False`` to signal that the engine should *drop* this op (e.g.
        skip a fork variant that would produce an unrunnable kernel).
        Default: always valid. Override in subclasses to enforce per-op
        invariants — ``TileOp.validate`` for instance checks that the
        post-register-tile thread count fits the hardware launch budget.
        """
        return True

    def _knob_key(self) -> tuple:
        """The knob half of the variant key (``identity_key(with_knobs=True)``) — the dict as a sorted item tuple."""
        return tuple(sorted(self.knobs.items())) if self.knobs else ()

    # ---- the identity interface — THE home, and the WHOLE public surface is one function:
    # :meth:`identity_key`, a lattice over the ONE base fact (the canonical Loop-IR body,
    # :meth:`_body_identity`, the single override point) with one flag per additional fact.
    # The named points live at call sites, not as methods: ``identity_key(with_io=True)`` is
    # the durable deploy join key (golden records, corpus stamps, child receipts);
    # ``identity_key(with_io=True, with_knobs=True)`` is the variant key (the search tree,
    # measurement stores — knobs included because same-body / different-knobs variants must not
    # collide with their parent, and a measurement belongs to (kernel, knob row)). There is
    # deliberately NO schedule-space key here: the enumeration stamp is scheduler plumbing,
    # minted at its one site (``lowering/tile/_schedule``).
    # A fact a schedule reads that neither the body nor the io carries is a modeling gap to fix
    # there, never a side-channel fingerprint. ---- #

    def _body_identity(self, *, structural: bool = True) -> str | None:
        """Does this op COMPUTE the same thing, spelling aside? — the canonical digest of the
        op's complete Loop-IR body (``Body.structural_key``: SSA / axis / buffer names and
        commutative-arg order normalized away; the default ``structural=True`` also collapses
        compute-unit op clusters — the schedule-equivalent reading), or ``None`` for an op kind
        that carries no Loop-IR body. The ONE identity override point: ``BodyOp`` answers with
        its stored body, ``TileOp`` with the schedule-free :attr:`TileOp.loop_body` it derives
        from its term. Cached on the body, which is immutable. Private — consumers go through
        :meth:`identity_key` (the bare lattice point equals this)."""
        return None

    @cached_property
    def _io_key(self) -> tuple:
        """The cached io digest input (:func:`_io_fingerprint`) — the ONE Op-level identity
        cache, with the ONE dependency ``__setattr__`` invalidates. The body key caches on the
        immutable ``Body`` / term, and the knob key is always recomputed (:meth:`_knob_key`),
        so knob and body mutations need no invalidation at all."""
        return _io_fingerprint(self)

    def identity_key(self, *, structural: bool = True, with_io: bool = False, with_knobs: bool = False) -> str | None:
        """THE identity — one function, one lattice. The base fact is the canonical Loop-IR
        body digest; each flag folds in one more fact:

        - ``structural`` — collapse compute-unit op clusters (the schedule-equivalent reading,
          the default: cluster siblings share a schedule space and want the same schedule) vs
          the exact kernel (``structural=False`` — their latency differs);
        - ``with_io`` — the io dtype/shape fingerprint: what a deployed kernel is bound to;
        - ``with_knobs`` — the knob row: which variant of the kernel this op is.

        ``None`` for an op kind that carries no Loop-IR body. ``CudaOp`` overrides the whole
        function with its rendered-source digest: past rendering, body / io / knobs are all
        realized into the artifact and the flags have nothing left to fold. The same kernel
        reached via different rewrite paths produces the same key — ``source`` is never part
        of it, and neither is the dialect tag: every stage of one rewrite chain keys off the
        same Loop-IR content."""
        key = self._body_identity(structural=structural)
        if key is None:
            return None
        parts: list = [key]
        if with_io:
            parts.append(self._io_key)
        if with_knobs:
            parts.append(self._knob_key())
        return key if len(parts) == 1 else digest(*parts)

    @_ClassProperty
    def dialect(cls) -> str | None:  # noqa: N805 — a class property; ``cls`` receives the owner
        """The lowering-stage tag for a kernel-bearing op (``"loop"`` / ``"tile"`` /
        ``"kernel"`` / ``"cuda"``); ``None`` for everything that is not one kernel of work
        (boundary sentinels, pre-fusion ops). DERIVED, never declared: kernel-bearing is
        exactly "overrides :meth:`_body_identity` or :meth:`identity_key`", and the tag is the
        class name minus its ``Op`` suffix, lowercased. A future subclass of a dialect op that wants its PARENT's tag must
        override this — the name rule tags it by its own name."""
        if cls.identity_key is Op.identity_key and cls._body_identity is Op._body_identity:
            return None
        return cls.__name__.removesuffix("Op").lower()

    def source_chain(self) -> Iterator[Op]:
        """Yield this op and every predecessor along the rewrite chain (:attr:`source`)."""
        cur: Op | None = self
        while cur is not None:
            yield cur
            cur = cur.source


@dataclass
class InputOp(Op):
    """Sentinel for graph input tensors (no computation)."""

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        raise NotImplementedError("InputOp has no inputs; use node.output.shape directly")

    def forward(self, *inputs):
        raise NotImplementedError("InputOp is a sentinel; value is supplied by the executor")


@dataclass
class ConstantOp(Op):
    """Fixed tensor: weights, RoPE tables, scalars. Not an activation.

    ``load_ops`` is an ordered tuple of frontend ``Op`` instances applied
    to the source tensor at bind time, in order. Const-folding passes
    absorb a foldable op (``TransposeOp``, ``ReshapeOp``, ...) into this
    chain, so a chain of layout ops over a constant collapses to a single
    ``ConstantOp`` whose loader executes the chain via the reference
    NumPy backend. ``output.shape`` already reflects the post-chain shape.

    ``source_path`` / ``source_shape`` / ``source_dtype`` are the source
    tensor's address (HF parameter/buffer attribute path) and pre-chain
    layout — what the loader needs to read from safetensors before
    running ``load_ops``. Empty for scalar constants and for synthetic
    constants emitted by passes (which never reach the loader).

    ``source_parts`` is the MULTI-source alternative to ``source_path``: an ordered tuple of
    ``(path, pre-chain shape)`` pairs the loader reads individually and concatenates along
    **axis 0** before running ``load_ops`` (the N-concat of sibling
    projection weights — axis 0 is the ``(N, K)`` out-features axis). Exactly one of
    ``source_path`` / ``source_parts`` / ``source_graph`` is set on a loadable constant;
    ``source_shape`` / ``source_dtype`` describe the post-concat (or post-evaluation) source.

    **Value binding** — a scalar constant binds its value EXACTLY ONE of three ways, never
    ambiguously (enforced in :meth:`__post_init__`):

    - ``value`` — a **static** scalar fixed at trace/compile time (eps, scale, a folded dim).
    - ``context_value`` — a **runtime** scalar bound at launch from the context (the
      ``sym_values`` symbolic-dim env) via an :class:`~emmy.compiler.ir.expr.Expr` over
      symbolic-dim names; e.g. a dynamic mean's divisor = the runtime reduce-axis size
      (``Var("seq_len")``). The backend fills the buffer with ``float(expr.eval(sym_values))``
      per run.
    - **neither** — an **executor-supplied** tensor (weights / RoPE tables loaded from
      safetensors, or fed as ``input_data``).

    Static-value readers gate on ``value is not None``; a ``context_value`` constant has
    ``value is None`` so they correctly treat it as non-static (the backend resolves it).
    """

    name: str
    value: float | int | None = None  # a STATIC scalar (None ⇒ not a static constant)
    context_value: Expr | None = None  # a RUNTIME scalar bound from context (sym_values); see class doc
    load_ops: tuple[Op, ...] = ()
    source_path: str | None = None
    source_parts: tuple[tuple[str, tuple[int, ...]], ...] = ()  # axis-0 concat of (path, shape) parts; see class doc
    source_shape: tuple[int, ...] | None = None
    source_dtype: str | None = None
    # N-source bind record: a constant-only frontend mini-graph whose leaf
    # ``ConstantOp``s name checkpoint source paths. The loader binds each leaf
    # source (its own dtype rules apply — an f8-dtype leaf binds raw bits),
    # evaluates the graph through the reference NumPy backend, and only THEN
    # runs this constant's ``load_ops`` chain — so later layout folds
    # (``050``/``060``) compose onto a folded constant exactly as onto a plain
    # one. Produced by the generic constant folder when it collapses a static
    # computation cone; ``None`` for every ordinary source constant.
    # ``source_shape`` / ``source_dtype`` describe the EVALUATED (pre-chain)
    # result. ``fold_into_constant`` rebuilds constants via
    # ``dataclasses.replace``, so the field propagates through the layout-fold
    # band by construction.
    source_graph: Graph | None = None

    def __post_init__(self) -> None:
        if self.value is not None and self.context_value is not None:
            raise ValueError(f"ConstantOp {self.name!r} binds EITHER a static value OR a context_value, not both")
        if self.source_path is not None and self.source_parts:
            raise ValueError(f"ConstantOp {self.name!r} binds EITHER a source_path OR source_parts, not both")
        if self.source_graph is not None and (self.source_path is not None or self.source_parts):
            raise ValueError(f"ConstantOp {self.name!r} binds EITHER a source_graph record OR checkpoint source paths, not both")
        if self.source_parts:  # normalize the JSON round-trip's nested lists back to hashable tuples
            self.source_parts = tuple((p, tuple(int(d) for d in s)) for p, s in self.source_parts)

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        raise NotImplementedError("ConstantOp has no inputs; use node.output.shape directly")

    def forward(self, *inputs):
        if self.value is not None:
            return np.array([self.value], dtype=np.float32)
        raise NotImplementedError("ConstantOp with value=None must be supplied by the executor")


def _keepdim_axis(shape: tuple, axis: int | str) -> tuple:
    """Return shape with the given axis set to 1 (keepdim=True semantics).

    This is the reduction's output shape: the reduced dim collapses to 1,
    preserving rank. Rank-preservation is a Tensor IR invariant — it lets
    elementwise ops consume reduction outputs without any implicit reshape
    (see ``pipeline/passes/frontend/decomposition/_broadcast.py``'s ``broadcast_to`` for the explicit
    broadcast wrapper that expands (…, 1, …) back up).
    """
    if not isinstance(axis, int):
        return tuple(shape)
    a = axis if axis >= 0 else len(shape) + axis
    if a < 0 or a >= len(shape):
        return tuple(shape)
    return tuple(shape[:a]) + (1,) + tuple(shape[a + 1 :])
