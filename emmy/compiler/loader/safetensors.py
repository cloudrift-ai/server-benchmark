"""Read parameters/buffers straight from safetensors shards.

This bypasses the PyTorch ``nn.Module`` round-trip: given a model id (HF
repo) or a local directory of safetensors files, the loader resolves
each ``ConstantOp.source_path`` to a tensor in one of the shards, reads
it as a numpy array, and runs the recorded ``load_ops`` chain via the
NumPy backend.

The function keeps a small key-canonicalization table because HF
checkpoints sometimes carry a ``model.`` prefix that ``torch.export``
strips, and vice versa. We try the original name, then with ``model.``
added, then with ``model.`` removed — that covers every case we've
seen on Llama / Qwen / TinyLlama.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from emmy.compiler.graph import Graph
from emmy.compiler.loader.binder import apply_load_ops, evaluate_source_graph
from emmy.compiler.loader.quant import F8_SAFETENSORS_DTYPES, decode_f8

logger = logging.getLogger(__name__)


def _resolve_model_dir(model_id_or_path: str) -> Path:
    """Return a local directory containing the model's safetensors files.

    If the argument is an existing directory, use it as-is. Otherwise
    treat it as an HF repo id and snapshot-download it (cached).
    """
    p = Path(model_id_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id_or_path))


def _build_index(model_dir: Path) -> dict[str, Path]:
    """Map each tensor name in the model to the safetensors shard it lives in.

    Handles both the single-file (``model.safetensors``) and sharded
    (``model.safetensors.index.json``) layouts.
    """
    from safetensors import safe_open

    index_json = model_dir / "model.safetensors.index.json"
    if index_json.exists():
        weight_map = json.loads(index_json.read_text())["weight_map"]
        return {name: model_dir / shard for name, shard in weight_map.items()}

    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(single, framework="numpy") as f:
            return {name: single for name in f.keys()}

    raise FileNotFoundError(f"No safetensors files found under {model_dir}")


def _candidate_keys(source_path: str) -> list[str]:
    """Generate the names to try in the safetensors index for a constant.

    Tolerates a mismatch in the number of leading ``model.`` prefixes
    between the traced ``source_path`` and the checkpoint's key. The
    whole-model trace wrapper nests the CausalLM under ``self.model``, so
    a base-model parameter becomes ``model.model.<rest>`` — while some
    checkpoints (e.g. Qwen3-Embedding) store the *bare* ``<rest>`` key.
    Progressively strip each ``model.`` level so any of those forms
    resolves, and also offer one added prefix for the inverse case. Tried
    in order (as-is first); the first candidate present in the index wins,
    so a checkpoint that genuinely uses a ``model.`` key still matches it
    before a stripped form."""
    cands = [source_path]
    s = source_path
    while s.startswith("model."):
        s = s[len("model.") :]
        cands.append(s)
    cands.append("model." + source_path)
    seen: set[str] = set()
    return [c for c in cands if not (c in seen or seen.add(c))]


def _read_shard(
    shard_path: str, keys: list[str], sources: dict[str, np.ndarray], bits_keys: frozenset[str] = frozenset()
) -> dict[str, str]:
    """Read ``keys`` from one shard into ``sources``, decoding fp8 tensors to f32 values.

    safetensors' numpy framework has no fp8 carrier (numpy lacks a float8 dtype;
    ``get_tensor`` raises), so fp8 keys re-open the shard through the torch
    framework and a zero-copy ``uint8`` view exposes the bit pattern for the LUT
    decode. Chosen over parsing the shard's raw byte ranges ourselves (which
    would duplicate format knowledge here); torch is available in every flow
    that reaches a quantized checkpoint — the trace itself requires it. BF16
    tensors (the unquantized modules of a quantized checkpoint — embeddings,
    norms) take the same torch route, read as f32 values (the loader's value
    dtype for bf16). Everything else keeps the numpy path bit-identical.

    ``bits_keys`` names fp8-stored keys whose CONSUMING constant (an in-graph
    node or a ``source_graph`` record leaf) carries an f8 graph dtype — those
    bind the RAW uint8 bit pattern, no LUT decode, no scale; the graph's own
    decode cone owns the value semantics. Every other fp8 key decodes to f32
    values as before.

    Returns the fp8 keys read, ``{key: canonical fp8 token}`` — how callers
    tell a decoded-fp8 array from a natively-float one after the read.
    """
    from safetensors import safe_open

    fp8: dict[str, str] = {}  # key → canonical fp8 token
    bf16: list[str] = []
    with safe_open(shard_path, framework="numpy") as f:
        for k in set(keys):
            stored = f.get_slice(k).get_dtype()
            fmt = F8_SAFETENSORS_DTYPES.get(stored)
            if fmt is not None:
                fp8[k] = fmt
            elif stored == "BF16":
                bf16.append(k)
            else:
                sources[k] = f.get_tensor(k)
    if fp8 or bf16:
        import torch  # noqa: PLC0415

        with safe_open(shard_path, framework="pt") as ft:
            for k, fmt in fp8.items():
                bits = ft.get_tensor(k).view(torch.uint8).numpy()
                sources[k] = bits if k in bits_keys else decode_f8(bits, fmt)
            for k in bf16:
                sources[k] = ft.get_tensor(k).float().numpy()
    return fp8


def load_sources_by_path(model_id_or_path: str, paths) -> dict[str, np.ndarray]:
    """Read the checkpoint tensors ``paths`` names, keyed by the REQUESTED path.

    The path-keyed sibling of :func:`load_constants_from_safetensors`, for callers that hold a
    plan rather than a graph — the serving trunk binds an ``ExecutionPlan``'s ``WeightSpec``
    source paths (``serving/gen_runner.py``), and a plan carries no node dtypes to key the
    raw-bits rule on. Every key reads at its STORED value dtype (fp8 decodes to f32, BF16 reads
    as f32 — the loader convention; int carriers such as EXL3's ``.trellis`` codes keep their
    stored words). A path with no matching key is simply absent from the result, so the caller
    can fall back to a live module for it."""
    model_dir = _resolve_model_dir(model_id_or_path)
    index = _build_index(model_dir)
    by_shard: dict[str, list[str]] = {}
    resolved: dict[str, str] = {}
    for path in paths:
        key = next((c for c in _candidate_keys(path) if c in index), None)
        if key is None:
            continue
        resolved[path] = key
        by_shard.setdefault(str(index[key]), []).append(key)
    sources: dict[str, np.ndarray] = {}
    for shard_path, keys in by_shard.items():
        _read_shard(shard_path, keys, sources)
    return {path: sources[key] for path, key in resolved.items() if key in sources}


def _record_leaves(record: Graph):
    """Yield ``(source_path, dtype_name)`` for every leaf source a ``source_graph`` bind
    record needs, recursing into nested records."""
    for lid, lop in record.loadable_constants():
        if lop.source_graph is not None:
            yield from _record_leaves(lop.source_graph)
            continue
        dtype_name = record.nodes[lid].output.dtype.name
        for path in [p for p, _shape in lop.source_parts] if lop.source_parts else [lop.source_path]:
            yield path, dtype_name


def load_constants_from_safetensors(graph: Graph, model_id_or_path: str) -> dict[str, np.ndarray]:
    """Bind every parameter/buffer ``ConstantOp`` from the model's safetensors.

    Returns a dict keyed by node id, ready to feed into ``Backend.run``
    as ``input_data``. Scalar constants (``value is not None``) and
    constants without a source are skipped — the backend materializes
    them on its own.

    A constant whose graph dtype is an f8 dtype binds RAW BITS (uint8 carrier,
    no LUT decode, no scale): the graph's own decode cone owns the value
    semantics (the birth-time spelling — ``loader.quant``). Decode-to-values
    applies only when the graph wants a non-f8 dtype from fp8 storage.

    A ``source_graph`` constant (a cone folded by ``032_fold_constant_subgraphs``)
    binds each of the record's leaf sources under the same rules and evaluates
    the record through the NumPy backend; this constant's own trailing
    ``load_ops`` chain (e.g. a later-folded transpose) then runs on the result.
    """
    model_dir = _resolve_model_dir(model_id_or_path)
    index = _build_index(model_dir)

    f8_dtypes = set(F8_SAFETENSORS_DTYPES.values())
    needed: dict[str, list[str]] = {}  # shard path → list of keys
    resolved: dict[str, tuple[str, ...]] = {}  # node_id → safetensors key(s); >1 = source_parts concat
    record_keys: dict[str, dict[str, str]] = {}  # node_id → {leaf source_path: safetensors key}
    bits_nodes: set[str] = set()  # node_id → bind raw fp8 bits (see docstring)
    bits_keys: set[str] = set()  # the safetensors keys those nodes / record leaves resolve to

    def _resolve_key(nid: str, path: str) -> str | None:
        for cand in _candidate_keys(path):
            if cand in index:
                needed.setdefault(str(index[cand]), []).append(cand)
                return cand
        logger.warning("safetensors loader: no key matched for %s (source_path=%r)", nid, path)
        return None

    for nid, op in graph.loadable_constants():
        if op.source_graph is not None:
            leaves: dict[str, str] = {}
            leaf_bits: set[str] = set()
            for path, dt in _record_leaves(op.source_graph):
                key = leaves.get(path) or _resolve_key(nid, path)
                if key is None:
                    break
                leaves[path] = key
                if dt in f8_dtypes:
                    leaf_bits.add(key)
            else:  # all-or-nothing, as for concats
                record_keys[nid] = leaves
                bits_keys |= leaf_bits
            continue
        paths = [p for p, _shape in op.source_parts] if op.source_parts else [op.source_path]
        keys = [k for path in paths if (k := _resolve_key(nid, path)) is not None]
        if len(keys) == len(paths):  # all-or-nothing: a partial concat would bind garbage
            resolved[nid] = tuple(keys)
            if graph.nodes[nid].output.dtype.name in f8_dtypes:
                bits_nodes.add(nid)
                bits_keys.update(keys)

    sources: dict[str, np.ndarray] = {}
    for shard_path, keys in needed.items():
        _read_shard(shard_path, keys, sources, bits_keys=frozenset(bits_keys))

    out: dict[str, np.ndarray] = {}
    for nid, op in graph.loadable_constants():
        if op.source_graph is not None:
            leaves = record_keys.get(nid)
            if leaves is None:
                continue
            val = evaluate_source_graph(op.source_graph, {path: sources[key] for path, key in leaves.items()})
            if val is not None:
                out[nid] = apply_load_ops(val, op.load_ops)
            continue
        keys = resolved.get(nid)
        if keys is None:
            continue
        src = np.concatenate([sources[k] for k in keys], axis=0) if op.source_parts else sources[keys[0]]
        # A bits-bound constant's chain runs under its f8 dtype token — the raw
        # uint8 numpy dtype is not a registered DataType.
        dtype = graph.nodes[nid].output.dtype.name if nid in bits_nodes else None
        out[nid] = apply_load_ops(src, op.load_ops, dtype=dtype)
    return out
