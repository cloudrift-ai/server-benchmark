"""CudaOp — graph-level wrapper around a rendered CUDA kernel.

Produced by ``passes/lowering/cuda`` by rendering each ``KernelOp`` body
to a ``__global__`` source string. The final graph before codegen is
``Graph[CudaOp + InputOp + ConstantOp]``; the CUDA backend walks it in
topological order, emits one ``kernel_name<<<grid, block>>>(args)``
launch per node, and wires buffer pointers by node id.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from emmy.compiler.ir.base import Op
from emmy.compiler.ir.expr import Expr


@dataclass(frozen=True)
class TmaDescMeta:
    """Metadata the CUDA backend needs to encode a TMA descriptor at launch.

    ``name`` matches the kernel signature parameter (added to
    ``arg_order`` after the buffer args). ``src_buf`` names the graph
    buffer whose device pointer + shape feed
    ``cuTensorMapEncodeTiled``. ``box_extents`` and ``swizzle`` are the
    descriptor's per-dim box and swizzle mode."""

    name: str
    src_buf: str
    box_extents: tuple[int, ...]
    swizzle: str = "NONE"


# int = static factor; str = symbolic axis name (resolved at launch from
# sym_values); Expr = a composite extent (e.g. ceil-div ``(seq_len+15)//16`` for
# a hint-driven masked block axis) evaluated against sym_values at launch.
_GridFactor = int | str | Expr
GridDimSpec = tuple[_GridFactor, ...]  # product of factors → one grid dim's extent


@dataclass(frozen=True)
class CudaOp(Op):
    """One CUDA kernel invocation as a graph-op.

    ``grid`` and ``block`` each carry three per-dim ``GridDimSpec`` tuples;
    every entry in a spec is multiplied together at launch time. Pure-int
    specs (e.g. ``((128,), (1,), (1,))``) describe static launch geometry;
    specs containing strings (e.g. ``(("seq_len",), (1,), (1,))``)
    reference symbolic dims that the launch resolver looks up in the
    runtime ``sym_values`` env. ``runtime_args`` lists those symbolic
    names in the order they appear in the kernel signature (one ``int``
    parameter per name, slotted after the buffer args and before any
    TMA descriptor params).
    """

    kernel_source: str = ""  # complete __global__ function
    kernel_name: str = ""
    arg_order: tuple[str, ...] = ()  # kernel-param names in positional order
    grid: tuple[GridDimSpec, GridDimSpec, GridDimSpec] = ((1,), (1,), (1,))
    block: tuple[GridDimSpec, GridDimSpec, GridDimSpec] = ((1,), (1,), (1,))
    smem_bytes: int = 0
    zero_outputs: tuple[str, ...] = ()
    # Buffers this kernel zero-writes as a DELEGATED prologue (``ZeroPrologue`` stmts injected
    # by ``005_delegate_zero_init``): downstream kernels' atomic accumulators. Not read at
    # launch (the kernel body does the zeroing) — the slab planner uses it to start those
    # buffers' live intervals at THIS launch instead of their own producer's.
    zero_prologues: tuple[str, ...] = ()
    comment: str = ""
    tma_descriptors: tuple[TmaDescMeta, ...] = field(default_factory=tuple)
    runtime_args: tuple[str, ...] = ()
    # Indirect operands: ``(arg_name, table_arg, sel_arg, slot)`` per marked input. The kernel
    # signature replaces ``const T* <arg>`` with ``const T* const* <table>, const int* <sel>,
    # int <slot>`` and resolves the base pointer in a body preamble; ``arg_order`` keeps the
    # plain operand name and the launcher expands it in place (``program._launch``). ``slot``
    # is 0 at compile — the serving runner stamps per-instance slot literals onto plan copies.
    indirect_args: tuple[tuple[str, str, str, int], ...] = ()

    def pretty_body(self) -> str:
        return self.kernel_source

    def identity(self, *, structural: bool = True):
        """Override :meth:`Op.identity`: a rendered kernel's identity is the identity of the FORK
        that offered it — the newest pre-schedule op on the rewrite chain (its own ``TileOp``,
        else the fused ``LoopOp``), body, io and stage tag alike. The rendered source is a
        realization of that fork under a decision, not a second notion of what the kernel is:
        keying a measurement on the source digest put the store's identity and the deploy join's
        identity in different alphabets, so a measured row could never answer the fork that
        produced it.

        Three nearer answers are wrong. The io must come from the fork too — the cuda passes
        rename and re-slab buffers, which would split the join at its io half. The intervening
        ``KernelOp`` must be skipped — its body is the schedule already materialized, so it keys
        apart from the tile term the fork decided. And NEWEST, not oldest: a split's pieces are
        forks of their own, and attributing a piece to the pre-split parent would price half the
        work as the whole.

        ``None`` for a synthetic ``CudaOp`` with no chain (hand-built kernels, imported sources):
        it computes something no Loop-IR op describes, so there is no identity to store a
        measurement under."""
        for op in self.source_chain():
            if type(op).dialect not in ("tile", "loop"):
                continue
            identity = op.identity(structural=structural)
            if identity is not None:
                return identity
        return None

    def _body_identity(self, *, structural: bool = True) -> str | None:
        """The chain identity's body half — see :meth:`identity`. Overridden (rather than left to
        the base, which would answer ``None``) so ``dialect`` still reads ``"cuda"`` and a graph
        digest walking a nested op reaches the same Loop-IR content."""
        identity = self.identity(structural=structural)
        return identity.body if identity is not None else None


def resolve_dim(spec, sym_values: dict[str, int]) -> int:
    """Multiply a ``GridDimSpec``'s factors, resolving ``str`` factors
    from ``sym_values`` and ``Expr`` factors via ``Expr.eval`` (e.g. a
    ceil-div block extent ``(seq_len+15)//16``). Accepts a bare ``int``
    (legacy static grid) as shorthand for a single-int spec — keeps
    pre-symbolic CudaOps working until every producer has been migrated.
    Raises ``KeyError`` on an unknown symbolic name."""
    if isinstance(spec, int):
        return spec
    total = 1
    for factor in spec:
        if isinstance(factor, int):
            total *= factor
        elif isinstance(factor, str):
            total *= sym_values[factor]
        else:
            total *= factor.eval(sym_values)
    return total
