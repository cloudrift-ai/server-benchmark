"""Worklist-driven splicer for a DAG of ``LoopOp``s.

Three public entry points wrap the same underlying ``_Splicer``:

- :func:`splice_loop_ops` — pairwise producer / consumer helper.
- :func:`splice_loops` — tag-generic N-way: caller supplies ``loops``
  (tag → ``LoopOp``), ``splice_edges`` ((origin_tag, src) →
  (target_tag, target_output)), and optional output roots.
- :func:`splice_graph` — consumes a ``Graph`` fragment directly;
  classifies each Load by its node.inputs edge (LoopOp → splice,
  otherwise → external slot in first-seen order).

Before seeding roots, ``splice_graph`` collapses each output equivalence
cluster: a single-owner chain of same-dtype copies proven to preserve every
flat address. The computed source's Write retargets to the live output shape,
so a terminal reshape does not force reduction reconstruction at its loads.

Algorithm. Seed: every selected root ``Write``. Each iteration pops
one pending dep and emits its def, queueing that def's own deps.
Resolution dispatches on stmt kind:

- **Load on a splice edge** — emit a copy alias at the demand scope;
  σ is solved by pairing target's ``Write.index`` against the reader's
  σ-substituted index, and the target's ``Write.value`` is queued under
  the solved σ. A narrowing reduction at a declared frontend output keeps its
  tensor dtype when the consumer has a distinct origin; a decomposition-private
  output may reconstruct within the frontend operation. The target's expression
  chain reconstructs piecemeal.
- **Accum** — freshen its reduce axis, place
  ``Loop(fresh_reduce_axis, Accum(...))`` at
  ``_scope_for_axes(ref_scope, required_c_axes)``, queue the Accum's
  ``value`` under σ extended with the fresh reduce binding.
- **Plain Assign / Select / Load** (non-splice source) — ``rewrite``
  the original stmt through ``(rename_ssa, sigma)`` and insert at the
  demand scope.

Unified dedup. A single table keyed on
``(origin, name, emit_scope, σ.restrict(live_axes))`` decides whether
to share an existing emission or allocate a fresh one. ``live_axes``
comes from ``LoopMeta`` and is the set of axes transitively reachable
through the stmt's Expr subtrees — σ bindings outside that set are
irrelevant and collapsed. Same key → share; different emit scope or
different live-σ → emit twice. This handles plain-stmt sharing, Accum
scope multiplicity (SDPA QK^T at softmax-max vs softmax-output), and
multi-output splice targets uniformly.

``LoopBuilder.insert`` is pure tree-splicing: descend the body along
the enclosure path, creating ``Loop`` nodes if missing, prepend at the
leaf. The worklist resolves deps in reverse-topological order so
consumers demand before producers — that *usually* yields defined-
before-use. The exception is the dedup case: when a stmt's operand
hits an existing binding emitted earlier in the worklist, the new stmt
still prepends above the existing one — landing above its own dep.
That sibling inversion is fixed up by the generic
``topo_sort_siblings`` pass in :mod:`emmy.compiler.ir.stmt.normalize`,
which runs inside ``LoopOp.__post_init__`` — so the splicer doesn't
need its own ordering pass; constructing the ``LoopOp`` is enough.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from emmy.compiler.dtype import F32, DataType
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, SimplifyCtx, Var, affine_form
from emmy.compiler.ir.loop.builder import LoopBuilder
from emmy.compiler.ir.loop.ir import (
    Accum,
    Assign,
    Axis,
    Load,
    Loop,
    LoopMeta,
    LoopOp,
    Scope,
    Select,
    Stmt,
    Write,
)
from emmy.compiler.ir.sigma import Sigma

logger = logging.getLogger(__name__)


class _NotSupported(Exception):
    """Pattern not handled — caller converts to ``None`` return.

    Always raised with a human-readable reason: ``splice_loops`` logs
    ``type(exc).__name__: <reason>`` at DEBUG, so ``compile -vv`` shows
    *which* unsupported pattern a given producer→consumer edge hit (σ-solve
    failure, missing Write, scope shortfall, …) without re-instrumenting the
    splicer."""


# Unified binding key: ``(origin, name, emit_scope, sigma.restrict(enclosing))``.
# ``emit_scope`` is where the stmt lands in the merged body; ``sigma`` is
# restricted to the stmt's own enclosing axis names — the only bindings that
# affect its rewrite (Load.index / Select.select) or its dep resolution.
_BindKey = tuple[str, str, Scope, Sigma]

# Exact flat-address comparison is finite. This covers the largest intended
# output reshape while bounding temporary numpy grids during graph analysis.
_OUTPUT_EQUIVALENCE_VERIFY_CAP = 1 << 25


@dataclass(frozen=True)
class _OutputEquivalenceCluster:
    """A single-owner chain of flat-address-identical output copies.

    ``buffers`` runs from the computed source to the live graph output;
    ``copy_nodes`` are the intervening LoopOp nodes in the same order.
    """

    buffers: tuple[str, ...]
    copy_nodes: tuple[str, ...]


@dataclass
class _Demand:
    """A pending dep in the worklist.

    ``bound_as`` is the fresh name the dep's def will bind in the merged
    body — allocated at queue time so callers can reference it without
    waiting for resolution.
    """

    name: str
    origin: str  # tag identifying which loop this def came from
    sigma: Sigma
    demand_scope: Scope
    bound_as: str


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def splice_loop_ops(producer: LoopOp, consumer: LoopOp, source: str) -> LoopOp | None:
    """Pairwise splicer: inline ``producer`` into every ``consumer`` Load
    whose ``source`` (buf name) matches the producer's output. Returns
    ``None`` when the pattern isn't supported.

    Thin wrapper over ``splice_loops``. The merged kernel's Loads keep
    their original buf names — no remap needed since names are stable
    across kernels. The producer's output buf name comes from its (sole)
    Write.
    """
    prod_writes = [s for s in producer if isinstance(s, Write)]
    if len(prod_writes) != 1:
        return None
    prod_buf = prod_writes[0].output
    return splice_loops(
        loops={"producer": producer, "consumer": consumer},
        splice_edges={("consumer", source): ("producer", prod_buf)},
    )


def splice_loops(
    loops: dict[str, LoopOp],
    splice_edges: dict[tuple[str, str], tuple[str, str]],
    *,
    splice_dtypes: dict[tuple[str, str], DataType] | None = None,
    roots: tuple[tuple[str, str], ...] | None = None,
) -> LoopOp | None:
    """Splice a DAG of ``LoopOp``s into one merged kernel.

    ``loops`` maps an opaque tag to each participating ``LoopOp``.
    ``splice_edges`` identifies which Loads are inlined from another
    registered loop: key ``(origin_tag, source_buf)`` → value
    ``(target_tag, target_output_buf)`` meaning "this loop's Load whose
    ``source`` is ``source_buf`` reads ``target_tag``'s Write whose
    ``output`` is ``target_output_buf`` and should be inlined."
    ``splice_dtypes`` optionally gives an edge's required dtype conversion.
    The emitted copy alias retains that dtype so reconstructing a reduction
    cannot bypass its tensor boundary.

    Non-splice Loads keep their original ``source`` buf names — buf
    identity is global, no remap needed. ``roots`` selects the observable
    Writes as ``(loop_tag, output_buffer)`` pairs. When omitted, every Write
    of the unique loop that never appears as a splice target is selected.
    Returns ``None`` if roots cannot be derived or any splice edge hits an
    unsupported pattern.
    """
    if roots is None:
        target_tags = {tag for tag, _out in splice_edges.values()}
        candidates = [tag for tag in loops if tag not in target_tags]
        if len(candidates) != 1:
            return None
        root_tag = candidates[0]
        roots = tuple((root_tag, write.output) for write in loops[root_tag].writes)
    if not roots:
        return None
    try:
        return _Splicer(
            loops={tag: op.analyze() for tag, op in loops.items()},
            splice_edges=splice_edges,
            splice_dtypes=splice_dtypes or {},
            roots=roots,
        ).run()
    except (_NotSupported, ValueError) as exc:
        # _NotSupported = splicer hit an unsupported pattern (σ-solve, scope).
        # ValueError = LoopOp construction validation rejected the emitted body.
        # Both surface to callers as None; debug log preserves which one for
        # future investigation without polluting normal output.
        logger.debug("splice_loops rejected pattern: %s: %s", type(exc).__name__, exc)
        return None


def splice_graph(graph) -> tuple[LoopOp, list[str]] | None:
    """Splice a subgraph of ``LoopOp`` nodes into one merged kernel.

    Each ``LoopOp`` node in ``graph`` becomes a registered loop tagged
    by its node id. Within each LoopOp node, a Load whose source points
    at another ``LoopOp`` node becomes a splice edge; a Load whose
    source points at a non-``LoopOp`` node (e.g. ``InputOp``) becomes
    an external read, assigned a slot in first-seen order.

    Every graph output is a root, so separate terminal branches become one
    multi-output LoopOp. A single-owner chain of flat-address-identical copies
    ending at a root is an output equivalence cluster: its source Write is
    retargeted to the root shape before ordinary dependency reconstruction.
    Returns ``(merged_op, external_buffer_ids)`` where the ids are the
    non-``LoopOp`` inputs in merged first-use order. Returns ``None`` if an
    output is not produced by a ``LoopOp`` or any splice edge hits an
    unsupported pattern.
    """
    if not graph.outputs:
        return None

    loop_nodes = {n.id: n for n in graph.nodes.values() if isinstance(n.op, LoopOp)}
    loops = {nid: node.op for nid, node in loop_nodes.items()}
    root_overrides: dict[str, tuple[str, str]] = {}
    collapsed: set[str] = set()
    for cluster in _output_equivalence_clusters(graph, loop_nodes):
        source, output = cluster.buffers[0], cluster.buffers[-1]
        source_node = graph.producer(source)
        if source_node is None:
            continue
        retargeted = _retarget_equivalent_output(graph, loops[source_node.id], source, output)
        if retargeted is None:
            continue
        loops[source_node.id] = retargeted
        collapsed.update(cluster.copy_nodes)
        root_overrides[output] = (source_node.id, output)
    for nid in collapsed:
        loops.pop(nid, None)

    roots: list[tuple[str, str]] = []
    for output in graph.outputs:
        if output in root_overrides:
            roots.append(root_overrides[output])
            continue
        root_node = graph.producer(output)
        if root_node is None or root_node.id not in loops:
            return None
        roots.append((root_node.id, output))
    splice_edges: dict[tuple[str, str], tuple[str, str]] = {}
    splice_dtypes: dict[tuple[str, str], DataType] = {}
    external_order: list[str] = []
    seen_external: set[str] = set()

    for node_id, op in loops.items():
        for ld in op.body.loads:
            inp = ld.input
            # A Load is a splice edge if its source buf names another LoopOp node;
            # otherwise it's an external input. We key edges off the buf name
            # (Load.source is the producing node's id), not a positional input
            # index — so a single edge entry covers every Load that reads the
            # same producer.
            input_producer = graph.producer(inp)
            if input_producer is not None and input_producer.id in loops:
                producer_id = input_producer.id
                splice_edges[(node_id, inp)] = (producer_id, inp)
                producer = graph.buffer(inp)
                producer_meta = loops[producer_id].analyze()
                producer_write = next((write for write, _ in producer_meta.writes if write.output == inp), None)
                producer_value = producer_meta.defs.get(producer_write.value) if producer_write is not None else None
                producer_origin = _ultimate_source(loops[producer_id])
                same_origin = producer_origin is _ultimate_source(op)
                origin_outputs = getattr(producer_origin, "outputs", {})
                private_output = producer_origin is not loops[producer_id] and bool(origin_outputs) and inp not in origin_outputs
                if (
                    not same_origin
                    and not private_output
                    and isinstance(producer_value, Accum)
                    and producer is not None
                    and producer.dtype != (producer_value.dtype or F32)
                ):
                    splice_dtypes[(node_id, inp)] = producer.dtype
            elif inp not in seen_external:
                seen_external.add(inp)
                external_order.append(inp)

    merged = splice_loops(
        loops=loops,
        splice_edges=splice_edges,
        splice_dtypes=splice_dtypes,
        roots=tuple(roots),
    )
    if merged is None:
        return None
    return merged, external_order


def _output_equivalence_clusters(graph, loop_nodes: dict[str, object]) -> tuple[_OutputEquivalenceCluster, ...]:
    """Find single-owner flat-address-copy chains ending at graph outputs."""
    clusters: list[_OutputEquivalenceCluster] = []
    for output in graph.outputs:
        if graph.buffer_users(output):
            continue
        buffers = [output]
        copy_nodes: list[str] = []
        current = output
        while True:
            copy = graph.producer(current)
            if copy is None or copy.id not in loop_nodes:
                break
            source = _flat_address_identity_source(graph, copy, current)
            if source is None or source in graph.outputs or graph.buffer_users(source) != {copy.id}:
                break
            source_node = graph.producer(source)
            if source_node is None or source_node.id not in loop_nodes:
                break
            buffers.append(source)
            copy_nodes.append(copy.id)
            current = source
        if copy_nodes:
            clusters.append(
                _OutputEquivalenceCluster(
                    buffers=tuple(reversed(buffers)),
                    copy_nodes=tuple(reversed(copy_nodes)),
                )
            )
    return tuple(clusters)


def _flat_address_identity_source(graph, node, output: str) -> str | None:
    """Return the source buffer when ``node`` is an exact flat-address copy to ``output``."""
    if not isinstance(node.op, LoopOp) or node.buffer_names() != (output,):
        return None
    if any(not isinstance(stmt, (Loop, Load, Write)) for stmt in node.op.body.iter()):
        return None
    loads = node.op.body.loads
    writes = node.op.body.writes
    if len(loads) != 1 or len(writes) != 1:
        return None
    load, write = loads[0], writes[0]
    if not load.is_scalar or not write.is_scalar or write.value != load.name or write.output != output:
        return None
    if write.atomic or write.swizzle != "NONE":
        return None

    source = graph.buffer(load.input)
    destination = graph.buffer(output)
    if source is None or destination is None or source.dtype != destination.dtype:
        return None
    source_strides = _static_strides(source.shape)
    destination_strides = _static_strides(destination.shape)
    if source_strides is None or destination_strides is None:
        return None
    source_numel = math.prod(dim.as_static() for dim in source.shape)
    destination_numel = math.prod(dim.as_static() for dim in destination.shape)
    if source_numel != destination_numel or destination_numel > _OUTPUT_EQUIVALENCE_VERIFY_CAP:
        return None
    extents = _loop_extents(node.op)
    if extents is None or math.prod(extents.values()) != destination_numel:
        return None
    if len(load.index) != len(source_strides) or len(write.index) != len(destination_strides):
        return None

    axes = tuple(extents)
    grids = np.meshgrid(*(np.arange(extents[name]) for name in axes), indexing="ij", sparse=True)
    env = dict(zip(axes, grids, strict=True))
    try:
        source_flat = sum(np.asarray(expr.eval(env)) * stride for expr, stride in zip(load.index, source_strides, strict=True))
        destination_flat = sum(np.asarray(expr.eval(env)) * stride for expr, stride in zip(write.index, destination_strides, strict=True))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return load.input if bool(np.all(source_flat == destination_flat)) else None


def _retarget_equivalent_output(graph, op: LoopOp, source: str, output: str) -> LoopOp | None:
    """Retarget ``source`` Writes to an equivalent output without rebuilding their computation."""
    source_tensor = graph.buffer(source)
    output_tensor = graph.buffer(output)
    if source_tensor is None or output_tensor is None:
        return None
    source_strides = _static_strides(source_tensor.shape)
    if source_strides is None or _static_strides(output_tensor.shape) is None:
        return None
    extents = _loop_extents(op)
    if extents is None or any(isinstance(stmt, Load) and stmt.input == source for stmt in op.body.iter()):
        return None
    source_writes = [write for write in op.body.writes if write.output == source]
    if not source_writes or any(not write.is_scalar or len(write.index) != len(source_strides) for write in source_writes):
        return None

    replacement: dict[int, Write] = {}
    for write in source_writes:
        terms: dict[str, int] = {}
        constant = 0
        for expr, stride in zip(write.index, source_strides, strict=True):
            affine = affine_form(expr, set(extents))
            if affine is None:
                return None
            anchor = affine[0].simplify(SimplifyCtx.empty())
            if not isinstance(anchor, Literal) or not isinstance(anchor.value, int):
                return None
            for name, coefficient in affine[1].items():
                terms[name] = terms.get(name, 0) + coefficient * stride
            constant += anchor.value * stride
        new_index = _decompose_flat(terms, constant, output_tensor.shape, extents)
        if new_index is None:
            return None
        replacement[id(write)] = replace(write, output=output, index=tuple(new_index))

    retargeted = LoopOp(
        body=op.body.map(lambda stmt: replacement.get(id(stmt), stmt)),
        name=op.name,
        source=op.source,
        knobs=dict(op.knobs),
    )
    return retargeted


def _static_strides(shape) -> list[int] | None:
    """Return row-major element strides, or ``None`` for a symbolic shape."""
    strides: list[int] = []
    step = 1
    for dim in reversed(tuple(shape)):
        if not dim.is_static:
            return None
        strides.append(step)
        step *= dim.as_static()
    return list(reversed(strides))


def _loop_extents(op: LoopOp) -> dict[str, int] | None:
    """Return each distinct static loop-axis extent, declining conflicting reuse."""
    extents: dict[str, int] = {}
    for stmt in op.body.iter():
        if not isinstance(stmt, Loop):
            continue
        if not stmt.axis.extent.is_static:
            return None
        extent = stmt.axis.extent.as_static()
        if stmt.axis.name in extents and extents[stmt.axis.name] != extent:
            return None
        extents[stmt.axis.name] = extent
    return extents


def _decompose_flat(terms: dict[str, int], constant: int, output_shape, extents: dict[str, int]) -> list[Expr] | None:
    """Partition an affine flat address into bounded row-major output indices."""
    strides = _static_strides(output_shape)
    if strides is None:
        return None
    dimensions: list[list[tuple[int, str | None]]] = [[] for _ in strides]
    for name, coefficient in terms.items():
        if coefficient < 0 or name not in extents:
            return None
        if coefficient == 0:
            continue
        for index, stride in enumerate(strides):
            if coefficient % stride == 0:
                dimensions[index].append((coefficient // stride, name))
                break
        else:
            return None
    if constant < 0:
        return None
    for index, stride in enumerate(strides):
        quotient = constant // stride
        if quotient:
            dimensions[index].append((quotient, None))
            constant -= quotient * stride
    if constant:
        return None

    result: list[Expr] = []
    for slot, dim in zip(dimensions, output_shape, strict=True):
        high = sum(coefficient * (extents[name] - 1 if name is not None else 1) for coefficient, name in slot)
        if high >= dim.as_static():
            return None
        expression: Expr = Literal(0, "int")
        for coefficient, name in slot:
            term: Expr = (
                Literal(coefficient, "int")
                if name is None
                else (Var(name) if coefficient == 1 else BinaryExpr("*", Var(name), Literal(coefficient, "int")))
            )
            expression = term if isinstance(expression, Literal) and expression.value == 0 else BinaryExpr("+", expression, term)
        result.append(expression)
    return result


def _ultimate_source(op):
    """The frontend-origin object at the end of an op rewrite chain."""
    root = op
    for source in op.source_chain():
        root = source
    return root


# ---------------------------------------------------------------------------
# _Splicer — all per-splice state + the worklist loop
# ---------------------------------------------------------------------------


class _Splicer(LoopBuilder):
    """Multi-loop splicer driven by an explicit splice-edge graph.

    Each registered loop has a tag (opaque string). ``splice_edges``
    identifies which Loads are inlined from another registered loop;
    all other Loads are re-indexed into the merged kernel's external
    input list. ``roots`` names the exact Writes that seed the traversal.

    Inherits body building (``insert`` / ``fresh`` / ``finish``) from
    ``LoopBuilder``; adds the worklist of pending demands and the dedup
    table that keeps the merged body minimal. Worklist dep-resolution
    is reverse-topological — producers demanded after consumers — so
    the builder's prepend-at-leaf behavior yields defined-before-use
    ordering naturally.
    """

    def __init__(
        self,
        *,
        loops: dict[str, LoopMeta],
        splice_edges: dict[tuple[str, str], tuple[str, str]],
        splice_dtypes: dict[tuple[str, str], DataType],
        roots: tuple[tuple[str, str], ...],
    ) -> None:
        used: set[str] = set()
        for meta in loops.values():
            used |= _collect_names(meta.op)
        super().__init__(used_names=used)
        self.loops = loops
        self.splice_edges = splice_edges
        self.splice_dtypes = splice_dtypes
        self.roots = roots
        self._pending: deque[_Demand] = deque()
        # Dedup: a stmt is uniquely identified by its (origin, name), the
        # emit scope it lands at in the merged body, and the σ restricted
        # to its own enclosing — the only bindings that affect its rewrite.
        # Same key → share a single emission.
        self._binding: dict[_BindKey, str] = {}
        # Sigma expressions stay live for one splice. Cache by object identity so repeated
        # dependency placement does not recursively walk the same large coordinate tree, while
        # avoiding structural hashing (which would perform another recursive tree walk).
        self._free_vars_by_expr_id: dict[int, tuple[Expr, frozenset[str]]] = {}

    def run(self) -> LoopOp:
        self._seed()
        while self._pending:
            self._resolve(self._pending.popleft())
        # Topological reordering of siblings runs inside LoopOp.__post_init__
        # (normalize_body → topo_sort_siblings), so the dedup case where a
        # consumer prepends above its already-emitted producer is fixed up
        # at construction time, not here.
        return LoopOp(body=self.finish())

    # -- Seed: every selected root Write, with its value queued -------------

    @staticmethod
    def _write_observes_running_accumulator(meta: LoopMeta, write: Write, scope: Scope) -> bool:
        """Whether ``write`` observes an accumulator before its reduce loop completes."""
        defining = meta.defs.get(write.value)
        reduce_axis = meta.reduce_axes.get(write.value)
        return isinstance(defining, Accum) and reduce_axis is not None and reduce_axis in scope.enclosing

    def _seed(self) -> None:
        for root_tag, output in self.roots:
            root = self.loops.get(root_tag)
            if root is None:
                raise _NotSupported(f"root names unknown loop {root_tag!r}")
            found = next(((write, scope) for write, scope in root.writes if write.output == output), None)
            if found is None:
                raise _NotSupported(f"root loop {root_tag!r} has no Write to {output!r}")
            w, scope = found
            if self._write_observes_running_accumulator(root, w, scope):
                raise _NotSupported(f"root Write to {w.output!r} observes running accumulator {w.value!r}; ordered loop cannot be spliced")
            v_bound = self._ensure_dep(w.value, root_tag, Sigma(), scope)
            self.insert(
                Write(
                    output=w.output,
                    index=w.index,
                    value=v_bound,
                    value_dtype=w.value_dtype,
                    atomic=w.atomic,
                    swizzle=w.swizzle,
                ),
                scope,
            )

    # -- Dep binding: look up or queue --------------------------------------

    def _ensure_dep(self, name: str, origin: str, sigma: Sigma, ref_scope: Scope) -> str:
        """Return the merged-body name for ``(origin, name)`` at the emit
        scope induced by ``ref_scope`` and σ. Queue a new demand the first
        time the key is seen.
        """
        meta = self.loops[origin]
        if name not in meta.defs:
            raise _NotSupported(f"_ensure_dep: {name!r} is not defined in loop {origin!r}")

        required_axes = tuple(
            mapped
            for axis in meta.scopes[name].enclosing
            for mapped in _remap_axis_names(axis, sigma, ref_scope, free_vars=self._expr_free_vars)
        )
        emit_scope = _scope_for_axes(ref_scope, required_axes)

        # σ restricted to axes transitively used in Expr subtrees reachable
        # from this stmt. Bindings outside this set don't affect any emitted
        # stmt, so keeping them in the key would cause spurious duplicate
        # emissions.
        restricted = sigma.restrict(meta.live_axes[name])
        key = (origin, name, emit_scope, restricted)
        existing = self._binding.get(key)
        if existing is not None:
            return existing
        bound = self.fresh(name)
        self._binding[key] = bound
        self._pending.append(_Demand(name=name, origin=origin, sigma=sigma, demand_scope=emit_scope, bound_as=bound))
        return bound

    def _expr_free_vars(self, expr: Expr) -> frozenset[str]:
        """Memoize one expression's variables for this splice by object identity."""
        key = id(expr)
        cached = self._free_vars_by_expr_id.get(key)
        if cached is not None and cached[0] is expr:
            return cached[1]
        variables = expr.free_vars()
        self._free_vars_by_expr_id[key] = (expr, variables)
        return variables

    # -- Resolution dispatch -------------------------------------------------

    def _resolve(self, d: _Demand) -> None:
        stmt = self.loops[d.origin].defs[d.name]

        if isinstance(stmt, Load):
            edge = self.splice_edges.get((d.origin, stmt.input))
            if edge is not None:
                target_tag, target_output_buf = edge
                self._resolve_splice_load(stmt, d, target_tag, target_output_buf)
            else:
                self._resolve_external_load(stmt, d)
        elif isinstance(stmt, Accum):
            self._resolve_accum(stmt, d)
        elif isinstance(stmt, (Assign, Select)):
            self._resolve_plain(stmt, d)
        else:
            raise _NotSupported(f"_resolve: unsupported stmt type {type(stmt).__name__} for {d.name!r} in loop {d.origin!r}")

    def _resolve_plain(self, stmt: Stmt, d: _Demand) -> None:
        """Generic Assign / Select emission — rewrite the stmt with fresh args
        and σ-substituted Exprs, insert at ``d.demand_scope``."""
        rename = {arg: self._ensure_dep(arg, d.origin, d.sigma, d.demand_scope) for arg in stmt.deps()}
        rename[stmt.name] = d.bound_as  # type: ignore[attr-defined]
        self.insert(stmt.rewrite(lambda n: rename.get(n, n), d.sigma), d.demand_scope)

    def _resolve_external_load(self, stmt: Load, d: _Demand) -> None:
        """A Load that isn't a splice edge — keep its buf name as-is (buf
        identity is global across kernels), σ-sub the index, emit.

        A data-dependent index (gather ``weight[(int)in0, a]``) reads SSA names
        via ``Load.deps()``. Resolve each one that names a source-loop def the
        same way ``_resolve_plain`` resolves an Assign's args, then ``rewrite``
        renames it inside the index. Axis-name deps aren't in ``meta.defs`` and
        are left to σ."""
        meta = self.loops[d.origin]
        rename = {v: self._ensure_dep(v, d.origin, d.sigma, d.demand_scope) for v in stmt.deps() if v in meta.defs}
        rename[stmt.name] = d.bound_as
        self.insert(stmt.rewrite(lambda n: rename.get(n, n), d.sigma), d.demand_scope)

    def _resolve_splice_load(self, stmt: Load, d: _Demand, target_tag: str, target_output_buf: str) -> None:
        """A Load that's a splice edge to another registered loop — emit a
        copy alias and queue the target loop's ``Write.value`` under the
        solved σ. The target's expression chain reconstructs piecemeal over
        subsequent iterations. ``target_output_buf`` selects which ``Write``
        of the target is the splice source when the target has multiple outputs."""
        target = self.loops[target_tag]
        found = next(((w, scope) for w, scope in target.writes if w.output == target_output_buf), None)
        if found is None:
            raise _NotSupported(
                f"splice edge into {target_tag!r}: no Write with output={target_output_buf!r} "
                f"(target writes {[w.output for w, _ in target.writes]}) — usually a buf-name != node-id mismatch on the producer"
            )
        target_write, target_scope = found
        if self._write_observes_running_accumulator(target, target_write, target_scope):
            raise _NotSupported(
                f"splice edge into {target_tag!r} observes running accumulator {target_write.value!r}; ordered loop cannot be spliced"
            )
        source_meta = self.loops[d.origin]
        index_rename = {
            name: Var(self._ensure_dep(name, d.origin, d.sigma, d.demand_scope)) for name in stmt.deps() if name in source_meta.defs
        }
        effective_index = tuple(d.sigma.apply(e).substitute(index_rename) for e in stmt.index)
        sigma = _solve_sigma(target_write.index, effective_index, {a.name for a in target.op.axes})
        if sigma is None:
            raise _NotSupported(f"σ-solve failed pairing target write index {target_write.index} against reader index {effective_index}")
        v_bound = self._ensure_dep(target_write.value, target_tag, sigma, d.demand_scope)
        dtype = self.splice_dtypes.get((d.origin, stmt.input))
        self.insert(Assign(name=d.bound_as, op="copy", args=(v_bound,), dtype=dtype), d.demand_scope)

    def _resolve_accum(self, stmt: Accum, d: _Demand) -> None:
        """Emit ``Loop(fresh_reduce_axis, [Accum(bound, value_bound, op)])`` at
        ``d.demand_scope``. The Accum's value is queued under σ extended with
        the fresh reduce-axis binding."""
        orig_axis = self.loops[d.origin].reduce_axes[stmt.name]
        fresh_name = self.fresh(orig_axis.name)
        reduce_axis = Axis(name=fresh_name, extent=orig_axis.extent)
        inner_sigma = d.sigma.extend(orig_axis.name, Var(fresh_name))
        inner_scope = Scope(enclosing=d.demand_scope.enclosing + (reduce_axis,))
        value_bound = self._ensure_dep(stmt.value, d.origin, inner_sigma, inner_scope)
        self.insert(Accum(name=d.bound_as, value=value_bound, op=stmt.op, axes=(fresh_name,)), inner_scope)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _scope_for_axes(ref_scope: Scope, required: tuple[str, ...]) -> Scope:
    """Shortest prefix of ``ref_scope`` whose axis set contains ``required``.

    Used two ways:
    - For Accums: places the reduce ``Loop`` at the innermost consumer
      scope where all σ-mapped producer enclosing axes are visible (today's
      behavior — further hoisting is left to later passes).
    - For plain producer stmts: picks the emit scope from the consumer's
      nest, tolerating producer's free-axis order differing from consumer's.
      A matmul producer ``(a0, a1, a2)`` σ-maps to consumer ``(a0, a2, a1)``
      (shuffled), but the consumer's scope ``(a0, a1, a2)`` covers the same
      axis set; emitting at the consumer's nest avoids a duplicate Loop tree.
    """
    names = tuple(a.name for a in ref_scope.enclosing)
    remaining = set(required)
    k = 0
    while remaining and k < len(names):
        remaining.discard(names[k])
        k += 1
    if remaining:
        raise _NotSupported(f"emit scope {names} is missing required axes {sorted(remaining)}")
    return Scope(enclosing=ref_scope.enclosing[:k])


def _remap_axis_names(
    axis: Axis,
    sigma: Sigma,
    ref_scope: Scope,
    *,
    free_vars: Callable[[Expr], frozenset[str]] | None = None,
) -> tuple[str, ...]:
    """Pick the merged-kernel axes that ``axis``'s σ target depends on.

    Every occurrence of the producer axis is substituted with the complete target expression by
    the caller's σ rewrite.  Placement therefore needs the shortest consumer scope containing
    *all* variables read by that expression: one for an offset/stride, several for a flatten /
    tile-coordinate map, and none when the reader fixes the producer axis to a constant.  The
    old single-variable restriction unnecessarily materialized a pure producer before layouts
    such as ``(tile_k, tile_n, lane) -> (k, n)`` even though substitution is exact.

    ``_scope_for_axes`` already accepts a set of required axes and chooses the common enclosing
    prefix, so multi-axis targets need no new loop representation.
    """
    target = sigma.get(axis.name)
    if target is None:
        return (axis.name,)
    variables = free_vars(target) if free_vars is not None else target.free_vars()
    scope_axes = tuple(a.name for a in ref_scope.enclosing)
    if any(name not in scope_axes for name in variables):
        # A non-axis variable is an SSA gather index. Keep the producer at the reader's current
        # scope, where the defining load/assign is available, instead of treating the SSA name as
        # a missing loop axis.
        return scope_axes
    return tuple(sorted(variables))


def _solve_sigma(
    writer: tuple[Expr, ...],
    reader: tuple[Expr, ...],
    producer_axes: set[str],
) -> Sigma | None:
    """Solve per-dim pairing ``writer[k] == reader[k]``. Supported writer
    forms: ``Var(a)`` (``a`` in ``producer_axes``) → bind ``a → reader[k]``;
    ``Literal(c)`` → no binding. Anything else → ``None``."""
    if len(writer) != len(reader):
        return None
    mapping: dict[str, Expr] = {}
    for w, r in zip(writer, reader, strict=True):
        if isinstance(w, Literal):
            continue
        if isinstance(w, Var) and w.name in producer_axes:
            existing = mapping.get(w.name)
            if existing is not None and existing != r:
                return None
            mapping[w.name] = r
            continue
        return None
    return Sigma(mapping)


def _collect_names(op: LoopOp) -> set[str]:
    """All SSA names plus all axis names used anywhere in ``op``."""
    names: set[str] = set()
    for s in op:
        if isinstance(s, Loop):
            names.add(s.axis.name)
        elif isinstance(s, (Load, Assign, Select, Accum)):
            names.add(s.name)
    return names
