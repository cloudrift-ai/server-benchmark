"""IdentityStrategy — a kernel's structural identity, owned end to end.

The ``S_*`` row (an extent-aware histogram of the kernel body — the structural identity that
keys the tune DB's evidence, featurizes into the online prior, and folds into ``identity_key(with_io=True, with_knobs=True)``'s
knob half) is computed here and MATERIALIZED into ``op.knobs`` at exactly two moments, once per
kernel, at birth:

- **fusion settled** — the end of the pipeline's last non-lowering pass (``on_pass_end`` at the
  computed stamp boundary; run start for a pipeline entering at lowering): the fused body is
  final, so the identity reflects the final form; earlier would give the same logical kernel two
  identities (pre- and post-stamp) and split the tune DB's keyings.
- **minted during lowering** — a cross-CTA split's pieces (``on_splice`` of a lowering pass):
  fresh knob-less TileOps stamped before the fragment enters the graph, so no rule can
  observe an unstamped kernel.

Materializing into knobs (rather than compute-on-read everywhere) is deliberate: the stamped row
rides the engine's rebind knob-merge into every later dialect, which is what keeps a terminal
CudaOp's cache key, its DB rows, and the prior's feature columns carrying the loop-birth
identity its own body could not reproduce. The read API below is knobs-first for the same
reason — compute is the fallback for an op nothing stamped yet.

Direct writes to shared ops are safe here: sibling candidates share op objects only before their
trajectories diverge, and the stamp is a deterministic function of the body — any "leak" writes
the values the sibling would have written. Writes are copy-on-write (a fresh dict), never a
mutation of a possibly-shared knob dict.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from sys import float_info
from typing import TYPE_CHECKING

from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.leaves import Assign, Mma
from emmy.compiler.ir.tile import TileOp, lower_with_output_specs
from emmy.compiler.pipeline.knob import STRUCT_PREFIX
from emmy.compiler.pipeline.strategy import PassEndEvent, PipelineStrategy, RunStartEvent, SpliceEvent
from emmy.compiler.structural import digest

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph


class IdentityStrategy(PipelineStrategy):
    """Stamp every kernel's ``S_*`` structural identity at birth and serve the
    one spelling of identity to every reader (``signature`` / ``op_sig``)."""

    @staticmethod
    def _stamp_boundary(passes: tuple[str, ...]) -> str | None:
        """The pass whose END finalizes the fused kernel bodies for THIS pipeline: the last
        non-lowering pass (``loop/stamp`` in the full pipeline; ``loop/fusion`` in a shorthand
        pipeline that skips the naming pass). ``None`` for a pipeline that starts at lowering
        (a loop-stage IR resume, a slice tune) — its entry kernels are already final. Computed
        per event from the pass list, never stored: this instance is shared across runs."""
        pre = [name for name in passes if name and not name.startswith("lowering/")]
        return pre[-1] if pre else None

    def on_run_start(self, e: RunStartEvent) -> None:
        # A pipeline entering AFTER the fusion boundary never fires a boundary pass end — but
        # its entry graph's kernels are already final, so they stamp at the door. A
        # pipeline that still runs fusion defers to the pass-end stamp: a premature stamp would
        # ride the rebind knob-merge onto fused bodies it no longer describes.
        if self._stamp_boundary(e.passes) is None:
            for node in e.graph.nodes.values():
                self._stamp(node, e.graph)

    def on_pass_end(self, e: PassEndEvent) -> None:
        if e.pass_name != self._stamp_boundary(e.passes):
            return
        for node in e.graph.nodes.values():
            self._stamp(node, e.graph)

    def on_splice(self, e: SpliceEvent) -> None:
        # Kernels minted inside lowering. Fusion-era splices are skipped: their kernels are
        # intermediate bodies whose identity is not final until the stamp boundary.
        if not e.pass_name.startswith("lowering/"):
            return
        for node in e.fragment.nodes.values():
            op = node.op
            if not isinstance(op, (LoopOp, TileOp)):
                continue
            if op.source is None and e.root_op.dialect == "loop":
                node.op = op = replace(op, source=e.root_op)
            # Fragment buffers carry the operand Tensors (the pieces' builders add them), so the
            # dtype features read the same values the assembled graph would give.
            self._stamp(node, e.fragment)

    def _stamp(self, node, graph: Graph) -> None:
        op = node.op
        if not isinstance(op, (LoopOp, TileOp)) or any(k.startswith(STRUCT_PREFIX) for k in op.knobs):
            return
        body = _identity_body(op)
        node.op = replace(op, knobs={**op.knobs, **structure_features(body, graph)})

    # --- the read API: the one spelling of identity ------------------------------------------

    def signature(self, op, graph: Graph | None = None) -> tuple:
        """The sorted ``S_*`` row — golden-record identity. Knobs-first (the stamped row IS the
        identity every key already embeds); computed from the body only for an op nothing
        stamped yet (pass ``graph`` for the dtype features then)."""
        stamped = tuple(sorted((k, float(v)) for k, v in (getattr(op, "knobs", None) or {}).items() if k.startswith(STRUCT_PREFIX)))
        if stamped:
            return stamped
        body = _identity_body(op)
        if body is None:
            return ()
        return tuple(sorted(structure_features(body, graph).items()))

    def op_sig(self, op, graph: Graph | None = None) -> str:
        """Digest of :meth:`signature` — the tune DB node-table key and the kernel-inventory
        dedup key."""
        return digest(*self.signature(op, graph))


# ---------------------------------------------------------------------------
# The feature function — the identity's content
# ---------------------------------------------------------------------------


def _identity_body(op) -> Body | None:
    """Return the loop-shaped body used only to compute a kernel's structural features."""
    if isinstance(op, LoopOp):
        return op.body
    if not isinstance(op, TileOp):
        return getattr(op, "body", None)
    body = Body(lower_with_output_specs(op.op, op.output_specs))
    for axis in reversed(op.place.free):
        body = Body((Loop(axis=axis, body=body),))
    return body


def structure_features(body: Body, graph: Graph | None = None) -> dict[str, float]:
    """Flat ``S_``-prefixed structural feature dict for a LoopOp ``body``:
    the extent-free skeleton merged with the ``S_ext_*`` loop extents.

    ``graph`` supplies operand dtypes for the ``S_dtype_*`` multiset; omit it
    (e.g. ad-hoc callers without a surrounding graph) to skip dtype features.
    Values are floats so the dict drops straight into the numeric knob row."""
    return {**_skeleton(body, graph), **_extents(body)}


def _skeleton(body: Body, graph: Graph | None) -> dict[str, float]:
    """Extent-free histogram: stmt-type counts + pointwise/reduce op multisets
    + loop-nest roles/depth + operand dtype multiset."""
    feats: Counter[str] = Counter()
    loads = body.loads
    feats["S_n_load"] = len(loads)
    feats["S_n_distinct_input"] = len({ld.input for ld in loads})
    feats["S_n_write"] = len(body.writes)
    feats["S_n_accum"] = len(body.accums)
    feats["S_n_mma"] = len(body.iter_of_type(Mma))
    feats["S_n_cond"] = len(body.iter_of_type(Cond))
    assigns = body.iter_of_type(Assign)
    feats["S_n_assign"] = len(assigns)
    for s in assigns:
        feats[f"S_pw_{s.op.name}"] += 1
    for s in body.accums:
        feats[f"S_reduce_{s.op.name}"] += 1
    loops = body.loops
    feats["S_n_loop"] = len(loops)
    feats["S_n_reduce_loop"] = sum(1 for loop in loops if loop.is_reduce)
    feats["S_n_free_loop"] = sum(1 for loop in loops if not loop.is_reduce)
    feats["S_loop_depth"] = _loop_depth(body)
    if graph is not None:
        for ld in loads:
            t = graph.buffer(ld.input)
            dt = str(t.dtype) if t is not None else "?"
            feats[f"S_dtype_{dt}"] += 1
    return {k: float(v) for k, v in feats.items()}


def _loop_depth(body: Body) -> int:
    """Max ``Loop`` nesting depth along any path (non-Loop wrappers like
    ``Cond`` recurse without incrementing)."""
    best = 0
    for s in body:
        if isinstance(s, Loop):
            best = max(best, 1 + _loop_depth(s.body))
        else:
            for nested in s.nested():
                best = max(best, _loop_depth(nested))
    return best


def _serial_cell_work(body) -> float:
    """Worst per-cell serial trip count: the max over loop-nest paths of the product of the
    static reduce-loop extents along the path. Nest-aware where ``S_ext_reduce_prod`` is flat —
    sibling reduces take the max, nested reduces multiply — so a subtree re-evaluated under an
    enclosing reduce is priced by the trips a thread actually serializes (DeepSeek-V4
    ``post4096``'s elected consumer piece recomputed a 16384-step statistics contraction inside a
    4096-step reduce: flat product 2^36-blind, nest product the honest 2^30). Free and sweep
    loops are excluded (grid-distributed / conservative) and a symbolic extent contributes no
    factor, so the value is a lower bound; saturates at the largest finite float."""
    best = 1.0
    for s in body:
        if isinstance(s, Loop):
            inner = _serial_cell_work(s.body)
            ext = s.axis.extent
            if s.is_reduce and ext.is_static:
                extent = float(ext.as_static())
                inner = float_info.max if extent and inner > float_info.max / extent else inner * extent
            best = max(best, inner)
        else:
            for nested in s.nested():
                best = max(best, _serial_cell_work(nested))
    return best


def _extents(body: Body) -> dict[str, float]:
    """Continuous ``S_ext_*`` loop extents, split by free vs reduce axis
    (``Loop.is_reduce``). Symbolic axes (non-static extent) are excluded from
    the products and counted in ``S_ext_n_symbolic_axis``."""
    free: list[int] = []
    reduce_: list[int] = []
    n_symbolic = 0
    for loop in body.loops:
        ext = loop.axis.extent
        if not ext.is_static:
            n_symbolic += 1
            continue
        (reduce_ if loop.is_reduce else free).append(ext.as_static())

    def bounded_product(values: list[int]) -> float:
        """Multiply extent features without constructing an unbounded Python integer."""
        value = 1.0
        for extent in values:
            if extent and value > float_info.max / extent:
                return float_info.max
            value *= extent
        return value

    return {
        "S_ext_n_free_axis": float(len(free)),
        "S_ext_free_prod": bounded_product(free),
        "S_ext_free_max": float(max(free)) if free else 0.0,
        "S_ext_n_reduce_axis": float(len(reduce_)),
        "S_ext_reduce_prod": bounded_product(reduce_),
        "S_ext_reduce_max": float(max(reduce_)) if reduce_ else 0.0,
        "S_ext_n_symbolic_axis": float(n_symbolic),
        "S_ext_serial_cell_work": _serial_cell_work(body),
    }
