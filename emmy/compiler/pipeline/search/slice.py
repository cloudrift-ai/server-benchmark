"""Single-kernel structural slice: isolate one finalized kernel node into a
standalone graph.

Used in two places:

- the **dump** sink writes one ``<kname>.json`` reproducer per kernel
  (``CompilerDump._dump_kernel_subgraphs``), and
- the **two-level tuner** (`search.two_level`) slices each post-fusion kernel
  into its own graph so the inner per-op search explores only that op's forks.

The slice keeps the root kernel node plus its transitive ``ConstantOp`` /
``InputOp`` producers (so scalar-constant inlining and load-op replay behave
identically) and replaces every *compute* ancestor — another kernel feeding
this one — with a synthetic ``InputOp`` boundary, so the result is standalone.
The root op is shared **by reference**: its body (and therefore
:meth:`~emmy.compiler.ir.base.Op.cache_key`) is byte-for-byte the full-graph op's, which is what lets
inner-tuned ``perf`` / ``lowering`` rows transfer back to the assembled graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph


def _kernel_compute_types() -> tuple[type, ...]:
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415

    return (LoopOp, KernelOp, CudaOp)


def collect_kernel_ancestors(
    graph: Graph, root_id: str, compute_types: tuple[type, ...], absorb: frozenset[str] = frozenset()
) -> tuple[set[str], set[str]]:
    """Collect ``root_id`` + its transitive ``ConstantOp`` / ``InputOp``
    ancestors. Compute-op ancestors (another kernel feeding this one) are
    returned in the ``synthetic`` set — they become synthetic ``InputOp``
    boundaries in the slice and their own producers are NOT walked — EXCEPT
    the ``absorb`` set: a producer a fusion of the root would consume (the
    flash score matmul) is kept as a real node and walked through, so the
    fusion can fire inside the slice."""
    from emmy.compiler.ir.base import ConstantOp, InputOp  # noqa: PLC0415

    keep: set[str] = {root_id}
    synthetic: set[str] = set()
    stack = list(graph.nodes[root_id].inputs)
    while stack:
        cur = stack.pop()
        # ``cur`` is a buffer name — canonicalize to its producing node id
        # (identity for primary outputs) so ``keep`` stays node-granular.
        node = graph.producer(cur)
        cur = node.id if node is not None else cur
        if cur in keep:
            continue
        keep.add(cur)
        if node is None:
            continue
        if isinstance(node.op, compute_types):
            if cur in absorb:
                stack.extend(node.inputs)
            else:
                synthetic.add(cur)
            continue
        if isinstance(node.op, (ConstantOp, InputOp)):
            stack.extend(node.inputs)
    return keep, synthetic


def topo_order(graph: Graph, keep: set[str]) -> list[str]:
    """Topo-sorted node ids restricted to ``keep`` (producers first)."""
    return [nid for nid in graph.topological_order() if nid in keep]


def single_node_graph(graph: Graph, node_id: str, absorb: frozenset[str] = frozenset()) -> Graph:
    """Slice ``graph`` to the single kernel node ``node_id`` plus its
    leaf-op closure, with every compute-op input turned into a synthetic
    ``InputOp``. Returns a standalone :class:`Graph` whose sole output is
    ``node_id`` and whose ``inputs`` list its synthetic boundaries + real
    graph inputs — sized identically to the full graph (partition
    enumeration depends on the producers' extents). ``absorb`` names
    producer kernels kept as REAL nodes (a fusion of the root consumes
    them in-slice — the flash score matmul); the slice is then that
    fusion's whole offer, one or two kernels depending on the trajectory."""
    from emmy.compiler.graph import Graph as _Graph  # noqa: PLC0415
    from emmy.compiler.ir.base import InputOp  # noqa: PLC0415

    keep, synthetic = collect_kernel_ancestors(graph, node_id, _kernel_compute_types(), absorb)
    sub = _Graph()
    for kid in topo_order(graph, keep):
        src = graph.nodes[kid]
        if kid in synthetic:
            # A synthetic boundary mirrors ALL of the producer's buffers so a
            # consumer edge naming a non-primary buffer still resolves in-slice,
            # and every buffer becomes a slice input (fed bench data by name).
            sub.add_node(InputOp(), [], outputs=src.outputs, node_id=src.id)
            sub.inputs.extend(src.buffer_names())
        else:
            sub.add_node(src.op, list(src.inputs), outputs=src.outputs, node_id=src.id)
            if isinstance(src.op, InputOp) and kid in graph.inputs:
                sub.inputs.append(kid)
    # Every buffer of the sliced node is a slice output (a non-primary buffer
    # with no in-slice consumer must plan as an output, not dead scratch).
    sub.outputs.extend(graph.nodes[node_id].buffer_names())
    return sub
