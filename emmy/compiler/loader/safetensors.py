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

from emmy.compiler import dtype as _dtype
from emmy.compiler.graph import Graph
from emmy.compiler.loader.binder import apply_load_ops
from emmy.compiler.loader.quant import F8_SAFETENSORS_DTYPES, decode_f8, dequantize

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

    ``bits_keys`` names fp8-stored keys whose CONSUMING constant carries an f8
    graph dtype (M2's expanded form) — those bind the RAW uint8 bit pattern, no
    LUT decode, no scale; the graph's own dequant cone owns the value semantics.
    Every other fp8 key decodes to f32 values as before.

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


def _value_np_dtype(source_dtype: str | None) -> np.dtype:
    """Numpy VALUE dtype for a constant's traced dtype: the registry carrier when it
    holds real values (f32 / f16); f32 for bits-carrier dtypes (bf16 — uint16 holds
    the pattern, not values; f32 is the numpy-side value dtype, the same convention
    as ``bind_constants_from_module``)."""
    if source_dtype is None:
        return np.dtype(np.float32)
    np_dt = _dtype.get(source_dtype).np
    return np_dt if np.issubdtype(np_dt, np.floating) else np.dtype(np.float32)


def load_constants_from_safetensors(graph: Graph, model_id_or_path: str) -> dict[str, np.ndarray]:
    """Bind every parameter/buffer ``ConstantOp`` from the model's safetensors.

    Returns a dict keyed by node id, ready to feed into ``Backend.run``
    as ``input_data``. Scalar constants (``value is not None``) and
    constants without a ``source_path`` are skipped — the backend
    materializes them on its own.

    A constant carrying a :class:`~emmy.compiler.ir.base.QuantSpec` (``op.quant``,
    stamped by ``loader.quant.stamp_quant_specs``) dequantizes when its SOURCE is
    read — scale applied to the decoded values, result cast to the constant's
    traced compute dtype — BEFORE the ``load_ops`` chain runs, so the chain (and
    everything downstream) sees an ordinary tensor at the promised dtype/shape.

    A spec-LESS constant whose graph dtype is an f8 dtype binds RAW BITS (uint8
    carrier, no LUT decode, no scale): that is the M2 expanded form, where the
    graph's own dequant cone consumes the bits and the expansion already moved
    the spec's scale onto a separate constant. Decode-to-values applies only
    when the graph wants a non-f8 dtype from fp8 storage.
    """
    model_dir = _resolve_model_dir(model_id_or_path)
    index = _build_index(model_dir)

    f8_dtypes = set(F8_SAFETENSORS_DTYPES.values())
    needed: dict[str, list[str]] = {}  # shard path → list of keys
    resolved: dict[str, tuple[str, ...]] = {}  # node_id → safetensors key(s); >1 = source_parts concat
    bits_nodes: set[str] = set()  # node_id → bind raw fp8 bits (see docstring)
    bits_keys: set[str] = set()  # the safetensors keys those nodes resolve to
    for nid, op in graph.loadable_constants():
        paths = [p for p, _shape in op.source_parts] if op.source_parts else [op.source_path]
        keys: list[str] = []
        for path in paths:
            for cand in _candidate_keys(path):
                if cand in index:
                    keys.append(cand)
                    needed.setdefault(str(index[cand]), []).append(cand)
                    break
            else:
                logger.warning("safetensors loader: no key matched for %s (source_path=%r)", nid, path)
        if len(keys) == len(paths):  # all-or-nothing: a partial concat would bind garbage
            resolved[nid] = tuple(keys)
            if op.quant is None and graph.nodes[nid].output.dtype.name in f8_dtypes:
                bits_nodes.add(nid)
                bits_keys.update(keys)
        if op.quant is not None and op.quant.scale_path in index:
            needed.setdefault(str(index[op.quant.scale_path]), []).append(op.quant.scale_path)

    sources: dict[str, np.ndarray] = {}
    for shard_path, keys in needed.items():
        _read_shard(shard_path, keys, sources, bits_keys=frozenset(bits_keys))

    out: dict[str, np.ndarray] = {}
    for nid, op in graph.loadable_constants():
        keys = resolved.get(nid)
        if keys is None:
            continue
        src = np.concatenate([sources[k] for k in keys], axis=0) if op.source_parts else sources[keys[0]]
        if op.quant is not None:
            scale = sources.get(op.quant.scale_path)
            if scale is None:  # spec stamped from an index the checkpoint no longer matches
                logger.warning("safetensors loader: scale %r missing for %s; binding undequantized", op.quant.scale_path, nid)
            else:
                src = dequantize(src, scale, inverse=op.quant.inverse).astype(_value_np_dtype(op.source_dtype), copy=False)
        # A bits-bound constant's chain runs under its f8 dtype token — the raw
        # uint8 numpy dtype is not a registered DataType.
        dtype = graph.nodes[nid].output.dtype.name if nid in bits_nodes else None
        out[nid] = apply_load_ops(src, op.load_ops, dtype=dtype)
    return out
