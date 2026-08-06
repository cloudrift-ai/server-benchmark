"""FP8 checkpoint ingestion: quant-spec stamping and dequant-on-load math.

Two halves, both consumed by the safetensors loader (M1 of the FP8 plan —
weights dequantize to the compute dtype at bind time; no kernel sees fp8):

- Pure numpy fp8 math: :func:`decode_f8` (re-exported from
  :mod:`emmy.compiler.dtype`, the leaf module the LUT lives in so the
  ``from_f8*`` decode intrinsics in ``ir/elementwise.py`` share the one table)
  and :func:`dequantize` (scale application with the granularity DERIVED from
  the weight/scale shapes — per-tensor, per-out-channel and 2-D block are one
  broadcast form, see :class:`~emmy.compiler.ir.base.QuantSpec`).
- :func:`stamp_quant_specs`: pair each weight ``ConstantOp`` with its scale
  tensor per the checkpoint's ``config.json`` ``quantization_config`` and the
  safetensors index, stamping ``ConstantOp.quant``. Runs immediately after
  trace and before the pipeline — node replacement is safe there (nothing has
  consumed the graph yet), and the merge/fold passes then see the spec.
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np

from emmy.compiler.dtype import decode_f8  # noqa: F401 — re-exported; the LUT's home is the dtype layer
from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import QuantSpec

logger = logging.getLogger(__name__)

# safetensors dtype names → canonical dtype tokens for the fp8 storage formats.
F8_SAFETENSORS_DTYPES: dict[str, str] = {"F8_E4M3": "f8e4m3", "F8_E5M2": "f8e5m2"}


def dequantize(weight: np.ndarray, scale: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    """Apply a quantization scale to a decoded weight, deriving the block from the shapes.

    ``block[i] = weight.shape[i] // scale.shape[i]`` (exact divisibility required —
    ``ValueError`` otherwise). A one-element scale is the per-tensor case; otherwise the
    weight is viewed as interleaved ``(n_blocks_i, block_i)`` axis pairs and the scale as
    ``(n_blocks_i, 1)`` so a single broadcast multiply covers per-out-channel and 2-D
    block alike. ``inverse``: the stored scale is the reciprocal of the dequant
    multiplier (``weight_scale_inv`` checkpoints) — divide instead of multiply.
    """
    w = np.asarray(weight, dtype=np.float32)
    s = np.asarray(scale, dtype=np.float32)
    if s.size == 1:  # per-tensor: scalar scale in any stored rank ((), (1,), (1, 1))
        s0 = s.reshape(())
        return w / s0 if inverse else w * s0
    if s.ndim != w.ndim:
        raise ValueError(f"scale rank {s.ndim} != weight rank {w.ndim} (shapes {s.shape} vs {w.shape})")
    for wd, sd in zip(w.shape, s.shape, strict=True):
        if sd == 0 or wd % sd:
            raise ValueError(f"scale shape {s.shape} does not evenly divide weight shape {w.shape}")
    w_view = w.reshape(tuple(x for wd, sd in zip(w.shape, s.shape, strict=True) for x in (sd, wd // sd)))
    s_view = s.reshape(tuple(x for sd in s.shape for x in (sd, 1)))
    out = w_view / s_view if inverse else w_view * s_view
    return out.reshape(w.shape)


def _fp8_quant_config(model_dir: Path) -> dict | None:
    """The checkpoint's ``quantization_config`` when it declares an FP8 weight scheme.

    Two formats in the wild (see the FP8 plan): ``quant_method: "fp8"`` (official
    releases) and ``quant_method: "compressed-tensors"`` whose ``config_groups``
    quantize weights as 8-bit float (FP8 / FP8_DYNAMIC / FP8_BLOCK schemes). Any
    other scheme (int quants, no config) → ``None``, and stamping is a no-op.
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return None
    qc = json.loads(cfg_path.read_text()).get("quantization_config")
    if not isinstance(qc, dict):
        return None
    method = qc.get("quant_method")
    if method == "fp8":
        return qc
    if method == "compressed-tensors":
        for group in (qc.get("config_groups") or {}).values():
            weights = group.get("weights") if isinstance(group, dict) else None
            if isinstance(weights, dict) and weights.get("type") == "float" and int(weights.get("num_bits") or 0) == 8:
                return qc
    return None


def _is_skipped(weight_key: str, patterns: list[str]) -> bool:
    """Whether the weight's module is excluded from quantization.

    ``quant_method: fp8`` lists module names in ``modules_to_not_convert``;
    compressed-tensors lists them in ``ignore``, where a ``re:`` prefix marks a
    regex. Plain entries match the module path exactly or as a dotted prefix /
    suffix component (``"lm_head"`` matches both ``lm_head`` and
    ``model.lm_head``; a parent-module entry covers its children).
    """
    module = weight_key.rsplit(".", 1)[0]  # drop the ".weight" leaf
    for pat in patterns:
        if pat.startswith("re:"):
            if re.match(pat[3:], module):
                return True
        elif module == pat or module.endswith("." + pat) or module.startswith(pat + "."):
            return True
    return False


def load_dequantized_state_dict(model_dir: str | Path) -> dict[str, np.ndarray]:
    """Every checkpoint tensor as numpy VALUES, fp8 weights dequantized by their paired scale.

    Feeds the architecture twin of a quantized checkpoint (see
    ``trace.huggingface.load_quantized_twin``) so the eager / accuracy reference
    carries the real weights, not random init. bf16 / fp8 storage reads as f32
    values; each fp8 ``<prefix>.weight`` with a ``weight_scale`` /
    ``weight_scale_inv`` partner (and not excluded by ``modules_to_not_convert``
    / ``ignore``) is dequantized, and the consumed scale tensors are dropped
    from the result. An unquantized checkpoint passes through unchanged.
    """
    from emmy.compiler.loader.safetensors import _build_index, _read_shard  # noqa: PLC0415

    model_dir = Path(model_dir)
    index = _build_index(model_dir)
    qc = _fp8_quant_config(model_dir)
    patterns = list(qc.get("modules_to_not_convert") or []) + list(qc.get("ignore") or []) if qc else []

    by_shard: dict[str, list[str]] = {}
    for key, shard in index.items():
        by_shard.setdefault(str(shard), []).append(key)
    sources: dict[str, np.ndarray] = {}
    fp8_keys: dict[str, str] = {}
    for shard_path, keys in by_shard.items():
        fp8_keys.update(_read_shard(shard_path, keys, sources))

    out: dict[str, np.ndarray] = {}
    consumed: set[str] = set()
    for key in index:
        if qc is not None and key in fp8_keys and key.endswith(".weight") and not _is_skipped(key, patterns):
            prefix = key[: -len(".weight")]
            scale_key = next((k for k in (prefix + ".weight_scale", prefix + ".weight_scale_inv") if k in index), None)
            if scale_key is not None:
                out[key] = dequantize(sources[key], sources[scale_key], inverse=scale_key.endswith("_inv"))
                consumed.add(scale_key)
                continue
        out[key] = sources[key]
    for k in consumed:
        out.pop(k, None)
    return out


def stamp_quant_specs(graph: Graph, model_id_or_path: str) -> int:
    """Stamp ``ConstantOp.quant`` on every quantized weight of ``graph``.

    Source of truth is the CHECKPOINT, not the traced module (a quantized
    checkpoint is traced through its bf16 architecture twin, whose module
    carries no fp8 tensors): the ``config.json`` ``quantization_config``
    declares the scheme, and the safetensors index supplies the pairing — a
    weight stored as fp8 whose ``<prefix>.weight_scale`` (or
    ``.weight_scale_inv`` → ``inverse=True``) tensor is present in the index
    gets a :class:`QuantSpec`; ``modules_to_not_convert`` / ``ignore`` entries
    and weights with no scale tensor get NO spec and load as plain tensors.
    Unquantized checkpoints are a no-op — zero graph change. Returns the
    number of constants stamped.
    """
    # Function-level import: this module owns the pure math; the shard-index
    # helpers live with the loader that also consumes them.
    from safetensors import safe_open  # noqa: PLC0415

    from emmy.compiler.loader.safetensors import _build_index, _candidate_keys, _resolve_model_dir  # noqa: PLC0415

    model_dir = _resolve_model_dir(model_id_or_path)
    qc = _fp8_quant_config(model_dir)
    if qc is None:
        return 0
    patterns = list(qc.get("modules_to_not_convert") or []) + list(qc.get("ignore") or [])
    index = _build_index(model_dir)

    stamped = 0
    with ExitStack() as stack:
        handles: dict[str, object] = {}  # shard path → open safe_open handle (metadata reads only)

        def _slice(key: str):
            path = str(index[key])
            handle = handles.get(path)
            if handle is None:
                handle = handles[path] = stack.enter_context(safe_open(path, framework="numpy"))
            return handle.get_slice(key)

        for nid, op in graph.loadable_constants():
            if op.quant is not None or op.source_path is None:
                continue
            key = next((c for c in _candidate_keys(op.source_path) if c in index), None)
            if key is None or not key.endswith(".weight") or _is_skipped(key, patterns):
                continue
            stored = _slice(key).get_dtype()
            if stored not in F8_SAFETENSORS_DTYPES:
                continue  # e.g. a modules-kept-in-bf16 weight, or a non-fp8 group member
            prefix = key[: -len(".weight")]
            scale_key = next((k for k in (prefix + ".weight_scale", prefix + ".weight_scale_inv") if k in index), None)
            if scale_key is None:
                continue  # no paired scale in the index → loads as a plain tensor
            scale_slice = _slice(scale_key)
            spec = QuantSpec(
                scale_path=scale_key,
                scale_shape=tuple(int(d) for d in scale_slice.get_shape()),
                scale_dtype=scale_slice.get_dtype().lower(),
                inverse=scale_key.endswith("_inv"),
                fmt=F8_SAFETENSORS_DTYPES[stored],
            )
            graph.nodes[nid].op = replace(op, quant=spec)
            stamped += 1
    if stamped:
        logger.info("stamped %d quantized weight constant(s) from %s", stamped, model_dir)
    return stamped
