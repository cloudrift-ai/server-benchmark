"""Fold a graph-output memcpy-identity reshape into its producer's ``Write``.

A traced flatten that survives to a GRAPH OUTPUT cannot fuse the normal way: ``010_merge_loop_ops``
inlines a producer's compute at the consumer's load sites, which would re-run a reduce-bearing
producer per output element (and the flatten's div/mod reader σ defeats the splicer anyway) — so
the copy stayed a real kernel (~1 µs/launch; three per gemma-4 decode pre twin, the q/k/v
head-layout flattens after the per-head qk norms).

This rule removes the copy without touching the producer's compute or loop structure: when the
consumer is a PURE single-``Load`` indexmap whose map is a flat-memory IDENTITY — same dtype, same
element count, and load flat address ≡ write flat address over the WHOLE (finite) output domain,
verified numerically and exactly via ``Expr.eval`` on the index arrays — the producer's ``Write``\\ s
retarget to the consumer's output buffer at the same flat address, re-decomposed onto the output
shape as clean per-dim AFFINE indices (no div/mod: the flat offset's affine terms partition onto
the output strides, each dim's slot range-checked against its extent). The producer keeps its
loops and its body verbatim; the copy kernel disappears and the producer's node takes the
consumer's place as the graph output.

Bails conservatively (``RuleSkipped``) on: a non-output consumer (ordinary fusion territory), a
producer with other consumers or that is itself a graph output, dtype / numel mismatch, a
non-identity map, symbolic extents, a domain too large to verify exactly, or a write offset whose
affine terms don't partition cleanly onto the output strides."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.loop import Loop, LoopOp
from emmy.compiler.ir.stmt import Body, Load, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._helpers import is_pure_indexmap as _is_pure_indexmap
from emmy.compiler.pipeline.passes.loop.fusion._helpers import wrap_merge_fragment as _wrap_merge_fragment

PATTERN = [
    Pattern("producer", LoopOp),
    Pattern("consumer", LoopOp),
]

# The identity check evaluates every index expr over the full output domain (exact, finite) —
# beyond this many elements the copy stays; a prefill-sized flatten (~8M cells) is well inside.
_VERIFY_CAP = 1 << 25


def _strides(shape) -> list[int] | None:
    """Row-major element strides for a static shape, or ``None`` on a symbolic extent."""
    out, acc = [], 1
    for d in reversed(list(shape)):
        if not getattr(d, "is_static", False):
            return None
        out.append(acc)
        acc *= d.as_static()
    return list(reversed(out))


def _loop_env(op: LoopOp) -> dict[str, int] | None:
    """Every loop var's static extent in ``op`` (symbolic → ``None``)."""
    env: dict[str, int] = {}
    for s in op.body.iter():
        if isinstance(s, Loop):
            if not s.axis.extent.is_static:
                return None
            env[s.axis.name] = s.axis.extent.as_static()
    return env


def _affine_terms(e: Expr) -> tuple[dict[str, int], int] | None:
    """``e`` as ``({var: coeff}, const)`` over integer +/* forms, or ``None`` if non-affine."""
    if isinstance(e, Literal):
        v = e.value
        return ({}, int(v)) if isinstance(v, (int, np.integer)) or float(v).is_integer() else None
    if isinstance(e, Var):
        return {e.name: 1}, 0
    if isinstance(e, BinaryExpr) and e.op in ("+", "-", "*"):
        lt, rt = _affine_terms(e.left), _affine_terms(e.right)
        if lt is None or rt is None:
            return None
        lv, lc = lt
        rv, rc = rt
        if e.op in ("+", "-"):
            sign = 1 if e.op == "+" else -1
            merged = dict(lv)
            for k, c in rv.items():
                merged[k] = merged.get(k, 0) + sign * c
            return merged, lc + sign * rc
        # "*": one side must be constant.
        if not lv:
            return {k: c * lc for k, c in rv.items()}, lc * rc
        if not rv:
            return {k: c * rc for k, c in lv.items()}, lc * rc
    return None


def _decompose_flat(terms: dict[str, int], const: int, out_shape, extents: dict[str, int]) -> list[Expr] | None:
    """Partition the affine flat offset ``Σ coeff·var + const`` onto ``out_shape``'s row-major
    strides as one affine index per dim, each dim's slot proven within ``[0, extent)`` from the
    vars' loop ranges. ``None`` when a term doesn't divide any stride, a slot overflows its
    extent, or a coefficient is negative (conservative)."""
    strides = _strides(out_shape)
    if strides is None:
        return None
    dims: list[list[tuple[int, str | None]]] = [[] for _ in strides]  # (local coeff, var|const)
    for var, coeff in terms.items():
        if coeff < 0 or var not in extents:
            return None
        if coeff == 0:
            continue
        for j, s in enumerate(strides):
            if coeff % s == 0:
                dims[j].append((coeff // s, var))
                break
        else:
            return None
    if const < 0:
        return None
    for j, s in enumerate(strides):  # distribute the constant, outermost first
        q = const // s
        if q:
            dims[j].append((q, None))
            const -= q * s
    if const:
        return None
    exprs: list[Expr] = []
    for slot, d in zip(dims, out_shape, strict=True):
        hi = sum(c * (extents[v] - 1 if v is not None else 1) for c, v in slot)
        if hi >= d.as_static():
            return None
        e: Expr = Literal(0, "int")
        for c, v in slot:
            t: Expr = Literal(c, "int") if v is None else (Var(v) if c == 1 else BinaryExpr("*", Var(v), Literal(c, "int")))
            e = t if (isinstance(e, Literal) and e.value == 0) else BinaryExpr("+", e, t)
        exprs.append(e)
    return exprs


def rewrite(match: Match, producer: Node, consumer: Node) -> Graph | None:
    graph = match.graph
    if not isinstance(producer.op, LoopOp) or not isinstance(consumer.op, LoopOp):
        raise RuleSkipped("producer or consumer is no longer a LoopOp")
    if producer.id not in consumer.inputs:
        raise RuleSkipped("producer is not an input of consumer")
    if consumer.id not in graph.outputs:
        raise RuleSkipped("consumer is not a graph output — ordinary fusion territory")
    if producer.id in graph.outputs:
        raise RuleSkipped("producer is itself a graph output — it must stay materialized")
    others = [n for n in graph.nodes.values() if n.id != consumer.id and producer.id in n.inputs]
    if others:
        raise RuleSkipped("producer has other consumers — its tensor must stay materialized")
    if not _is_pure_indexmap(consumer.op):
        raise RuleSkipped("consumer computes — not a pure indexmap copy")
    if producer.output.dtype != consumer.output.dtype:
        raise RuleSkipped("dtype-changing copy — not a flat-memory identity")

    loads = [s for s in consumer.op.body.iter() if isinstance(s, Load)]
    writes = [s for s in consumer.op.body.iter() if isinstance(s, Write)]
    if len(loads) != 1 or len(writes) != 1 or loads[0].input != producer.id:
        raise RuleSkipped("consumer is not a single-load single-store copy of the producer")
    p_strides, c_strides = _strides(producer.output.shape), _strides(consumer.output.shape)
    if p_strides is None or c_strides is None:
        raise RuleSkipped("symbolic shape — the finite identity check needs static extents")
    p_numel = math.prod(d.as_static() for d in producer.output.shape)
    c_numel = math.prod(d.as_static() for d in consumer.output.shape)
    if p_numel != c_numel or c_numel > _VERIFY_CAP:
        raise RuleSkipped(f"numel mismatch or domain too large to verify ({p_numel} vs {c_numel})")

    # --- exact identity check: load flat address == write flat address over the whole domain.
    c_env = _loop_env(consumer.op)
    if c_env is None:
        raise RuleSkipped("symbolic consumer loop — cannot enumerate the domain")
    axes = [s.axis for s in consumer.op.body.iter() if isinstance(s, Loop)]
    grids = np.meshgrid(*(np.arange(c_env[a.name]) for a in axes), indexing="ij", sparse=True)
    env = {a.name: g for a, g in zip(axes, grids, strict=True)}
    flat_load = sum(np.asarray(e.eval(env)) * s for e, s in zip(loads[0].index, p_strides, strict=True))
    flat_store = sum(np.asarray(e.eval(env)) * s for e, s in zip(writes[0].index, c_strides, strict=True))
    if not bool(np.all(flat_load == flat_store)):
        raise RuleSkipped("copy is not a flat-memory identity")

    # --- retarget every producer Write to the output buffer at the same flat address.
    p_env = _loop_env(producer.op)
    if p_env is None:
        raise RuleSkipped("symbolic producer loop — cannot range-check the re-decomposed index")
    old = producer.output.name

    def retarget(body: Body) -> Body | None:
        stmts: list = []
        for s in body:
            if isinstance(s, Loop):
                inner = retarget(s.body)
                if inner is None:
                    return None
                stmts.append(replace(s, body=inner))
            elif isinstance(s, Write) and s.output == old:
                terms: dict[str, int] = {}
                const = 0
                for e, stride in zip(s.index, p_strides, strict=True):
                    at = _affine_terms(e)
                    if at is None:
                        return None
                    for v, c in at[0].items():
                        terms[v] = terms.get(v, 0) + c * stride
                    const += at[1] * stride
                new_index = _decompose_flat(terms, const, consumer.output.shape, p_env)
                if new_index is None:
                    return None
                stmts.append(replace(s, output=consumer.output.name, index=tuple(new_index)))
            else:
                stmts.append(s)
        return Body(tuple(stmts))

    new_body = retarget(producer.op.body)
    if new_body is None:
        raise RuleSkipped("producer write offset resists the clean affine re-decomposition")
    merged = LoopOp(body=new_body)
    frag = _wrap_merge_fragment(graph, merged, consumer)
    match.output = consumer.id
    match.consumed = {producer.id, consumer.id}
    return frag
