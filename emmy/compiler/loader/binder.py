"""Apply ``ConstantOp.load_ops`` to a source ndarray via the reference
NumPy backend.

The loader produces raw source tensors (from safetensors or
``module.named_parameters()``); this binder is the small adapter that
runs each constant's ``load_ops`` chain on its source array. Reusing
the existing ``Backend.run`` interpreter (``backend/base.py``) means
every op already has a numpy ``forward()`` — no new code path.
"""

from __future__ import annotations

import numpy as np

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp


def apply_load_ops(source: np.ndarray, load_ops: tuple, dtype: str | None = None) -> np.ndarray:
    """Run ``load_ops`` over ``source`` using the NumPy backend.

    Builds a tiny single-input graph and dispatches it through the
    default ``Backend.run`` interpreter. Each load op must already have
    a working ``Op.forward`` (true for ``TransposeOp`` / ``ReshapeOp``
    by construction — those are the only ops the fold pass produces).

    ``dtype`` overrides the chain's tensor dtype token — needed for
    bits-carrier sources (an fp8 constant's raw uint8 bits), whose numpy
    dtype name (``uint8``) is not a registered ``DataType``.
    """
    if not load_ops:
        return np.ascontiguousarray(source)

    g = Graph()
    src_dtype = dtype or str(source.dtype)
    in_id = g.add_node(op=InputOp(), inputs=[], output=Tensor("src", tuple(source.shape), src_dtype))
    g.inputs.append(in_id)

    cur = in_id
    cur_shape = tuple(source.shape)
    for i, op in enumerate(load_ops):
        cur_shape = op.infer_output_shape([cur_shape])
        nid = g.add_node(op=op, inputs=[cur], output=Tensor(f"step_{i}", cur_shape, src_dtype))
        cur = nid
    g.outputs.append(cur)

    from emmy.compiler.backend.base import Backend

    class _NumpyInterp(Backend):
        def compile(self, graph):
            return graph

    result, _ = _NumpyInterp().run(g, input_data={in_id: np.ascontiguousarray(source)})
    return np.ascontiguousarray(result.outputs[cur])


def assemble_source(op, sources: dict[str, np.ndarray]) -> np.ndarray | None:
    """Assemble a constant's raw pre-chain source array from ``sources``: the single
    ``source_path`` lookup, the axis-0 concat of its ``source_parts`` (the
    ``merge_sibling_linears`` weight concat), or — on a plan's ``WeightSpec`` — the COMPUTED
    constant its ``source_op`` names (``("hadamard", (128,))``; no checkpoint key at all).
    ``None`` when any source is missing. Duck-typed over ``ConstantOp`` and
    ``backend.plan.WeightSpec`` alike — anything with ``source_path`` / ``source_parts``.
    ``source_graph`` bind records are the loaders' own business (:func:`evaluate_source_graph`)
    — this helper answers ``None`` for them (``sources`` holds per-path arrays, and a record
    has no single path); the plan projects the zero-leaf ones to ``source_op`` instead."""
    source_op = getattr(op, "source_op", None)
    if source_op is not None:
        from emmy.compiler.backend.plan import build_source_op  # noqa: PLC0415

        return build_source_op(source_op)
    if op.source_parts:
        parts = [sources.get(path) for path, _shape in op.source_parts]
        if any(p is None for p in parts):
            return None
        return np.concatenate([np.ascontiguousarray(p) for p in parts], axis=0)
    return sources.get(op.source_path) if op.source_path is not None else None


def evaluate_source_graph(record, sources: dict[str, np.ndarray]) -> np.ndarray | None:
    """Evaluate a ``ConstantOp.source_graph`` bind record against per-path ``sources``.

    Binds every leaf constant of the record through :func:`bind_constants` (each leaf's own
    dtype rules apply — an f8-dtype leaf binds raw uint8 bits, and leaf ``load_ops`` replay),
    then runs the record through the reference NumPy backend. ``None`` when any leaf source is
    missing from ``sources`` — the caller skips the constant, same as a missing plain source.
    """
    leaf_data = bind_constants(record, sources)
    if any(nid not in leaf_data for nid, _op in record.loadable_constants()):
        return None

    from emmy.compiler.backend.base import Backend  # noqa: PLC0415

    class _NumpyInterp(Backend):
        def compile(self, graph):
            return graph

    result, _ = _NumpyInterp().run(record, input_data=leaf_data)
    return np.ascontiguousarray(result.outputs[record.outputs[0]])


def bind_constants(graph: Graph, sources: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build the per-node ``input_data`` dict for every ``ConstantOp``.

    ``sources`` maps each ``ConstantOp.source_path`` (or ``source_parts`` part path) to its
    raw source ndarray (typically read from safetensors or pulled from a live
    ``nn.Module``). For each constant, this assembles the pre-chain source
    (:func:`assemble_source`), runs ``load_ops`` over it, and stores the result keyed by
    the constant's node id. Scalar constants (``value is not None``) are skipped — the
    backend materializes them on its own. Constants without a source are skipped too
    (synthetic constants emitted by passes carry their ``value`` directly).
    """
    out: dict[str, np.ndarray] = {}
    for nid, op in graph.loadable_constants():
        if op.source_graph is not None:
            # Folded constant subgraph (``032_fold_constant_subgraphs``): evaluate the record
            # over the same per-path sources, then run this constant's own trailing chain.
            val = evaluate_source_graph(op.source_graph, sources)
            if val is not None:
                out[nid] = apply_load_ops(val, op.load_ops)
            continue
        src = assemble_source(op, sources)
        if src is None:
            continue
        # An f8-dtype constant binds raw uint8 BITS (the in-graph decode-cone form — the
        # graph's own algebra owns the value semantics); its chain runs under the f8 token
        # because the carrier's numpy dtype (uint8) is not a registered DataType. Mirrors the
        # safetensors loader's bits-bind convention.
        dt = graph.nodes[nid].output.dtype
        tok = dt.name if getattr(dt, "name", None) in ("f8e4m3", "f8e5m2") and src.dtype == np.uint8 else None
        out[nid] = apply_load_ops(src, op.load_ops, dtype=tok)
    return out


def bind_constants_from_module(graph: Graph, module) -> dict[str, np.ndarray]:  # noqa: ANN001 — torch.nn.Module, duck-typed
    """Bind every parameter/buffer ``ConstantOp`` from a *live* ``nn.Module``.

    The whole-model trace wrapper carries computed buffers — the precomputed
    rotary ``cos``/``sin`` and the causal mask — that aren't in the checkpoint's
    safetensors, so :func:`load_constants_from_safetensors` can't supply them.
    Binding from the traced module's own ``state_dict`` covers those *and* the
    weights uniformly: each ``ConstantOp.source_path`` is the module-attribute
    path captured at trace time, so it matches a ``state_dict`` key verbatim.

    ``state_dict`` (not ``named_parameters``) is the right source because it
    lists *tied* weights under every name they're registered as — e.g. a model
    with ``tie_word_embeddings=True`` traces an ``lm_head.weight`` constant, but
    ``named_parameters`` dedups it down to the shared ``embed_tokens.weight``
    only, leaving the final projection unbound (→ zero logits). Tensors are cast
    to float32 numpy (the backend dtype; also sidesteps reading bf16 checkpoints
    through numpy). NOTE: a live module cannot supply an fp8 checkpoint's raw
    bits or scale tensors — constants whose sources only exist in the checkpoint
    (f8-dtype leaves, ``source_graph`` records over them) stay unbound here and
    must come from the safetensors loader."""
    sources: dict[str, np.ndarray] = {}
    for name, t in module.state_dict().items():
        sources[name] = t.detach().cpu().float().numpy()
    return bind_constants(graph, sources)
