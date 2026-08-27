"""Shared loop-dialect kernel NAMING — ``010_stamp_loop_names`` is a thin wrapper over
:func:`name_for_loop`, factored out so callsites that build LoopOps outside the pass pipeline
(e.g. fragment builders) can name their kernels with a plain import instead of reaching into a
leading-digit pass module via ``importlib`` (a pass file's ``NNN_…`` stem isn't a legal import
name).

Structural identity (the ``S_*`` features that used to be stamped by a twin pair of rules here
and in ``lowering/tile``) is owned by the ``IdentityStrategy``
(``pipeline/passes/identity.py``) — computed there and materialized into knobs at the engine's
events, not by rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emmy.compiler import provenance

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph, Node
    from emmy.compiler.ir.loop import LoopOp


def name_for_loop(op: LoopOp, node: Node, graph: Graph) -> str:
    """The provenance-derived ``k_…`` kernel label ``010_stamp_loop_names``
    stamps onto ``op``, factored out so fragment builders name their LoopOps
    the same way. Threads the node id + per-node provenance + graph-wide
    coverage totals into :func:`provenance.name_for`, plus the io dtype
    signature — the body carries no buffer decls, so without it two kernels
    identical except for an operand/output dtype hash to the same name and the
    plan's per-name kernel dedup hands one of them the other's code (an
    f32-writing twin storing f16 bytes)."""

    def _dt(buf: str) -> str:
        t = graph.buffer(buf)
        return str(t.dtype) if t is not None else "?"

    dtype_sig = ",".join(_dt(ld.input) for ld in op.body.loads) + "->" + ",".join(_dt(w.output) for w in op.body.writes)
    return provenance.name_for(op, node.id, provenance.get(node), provenance.totals(graph), dtype_sig=dtype_sig)
