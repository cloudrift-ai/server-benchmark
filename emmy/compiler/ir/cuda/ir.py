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
from emmy.compiler.structural import digest


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


@dataclass
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
    # Native sources may declare ``extern __shared__`` even below the 48 KiB
    # opt-in threshold. Emmy-rendered kernels use static storage in that regime.
    dynamic_smem: bool = False
    zero_outputs: tuple[str, ...] = ()
    # Buffers this kernel zero-writes as a DELEGATED prologue (``ZeroPrologue`` stmts injected
    # by ``005_delegate_zero_init``): downstream kernels' atomic accumulators. Not read at
    # launch (the kernel body does the zeroing) — the slab planner uses it to start those
    # buffers' live intervals at THIS launch instead of their own producer's.
    zero_prologues: tuple[str, ...] = ()
    comment: str = ""
    tma_descriptors: tuple[TmaDescMeta, ...] = field(default_factory=tuple)
    runtime_args: tuple[str, ...] = ()
    # By-value scalar parameters: ``(arg_name, dtype, value)``. The name stays in
    # ``arg_order`` at the kernel-signature position; the runtime substitutes the typed
    # scalar instead of looking up a device buffer. Kept distinct from ``runtime_args``:
    # those resolve symbolic extents per launch, while these are immutable properties of
    # one compiled invocation.
    scalar_args: tuple[tuple[str, str, int | float], ...] = ()
    # Indirect operands: ``(arg_name, table_arg, sel_arg, slot)`` per marked input. The kernel
    # signature replaces ``const T* <arg>`` with ``const T* const* <table>, const int* <sel>,
    # int <slot>`` and resolves the base pointer in a body preamble; ``arg_order`` keeps the
    # plain operand name and the launcher expands it in place (``program._launch``). ``slot``
    # is 0 at compile — the serving runner stamps per-instance slot literals onto plan copies.
    indirect_args: tuple[tuple[str, str, str, int], ...] = ()

    def pretty_body(self) -> str:
        return self.kernel_source

    def cache_key(self) -> str | None:
        """Override :meth:`Op.cache_key`: digest of the rendered source + launch params (the
        bits that determine runtime behavior). Name-invariant: the kernel function name is
        rendered into the source (``void <name>(...)``) but doesn't change runtime behavior,
        so it normalizes out — renaming a kernel (e.g. via op provenance) neither busts the
        perf cache nor blocks an isolated-kernel tune from transferring to a whole-model
        compile."""
        src = self.kernel_source.replace(self.kernel_name, "_K_") if self.kernel_name else self.kernel_source
        return digest(
            type(self).__name__,
            src,
            self.arg_order,
            self.grid,
            self.block,
            self.smem_bytes,
            self.dynamic_smem,
            self.scalar_args,
        )


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
