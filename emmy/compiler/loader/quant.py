"""Quantized-checkpoint ingestion: birth-time spelling of the decode algebra, plus the numpy math.

This module is the ONE place quantization-as-a-concept exists (together with the
safetensors loader that reads the checkpoint and ``trace/huggingface.py``'s
architecture-twin construction). Three checkpoint families share the design — FP8
(scale-paired bits, ``quant_method: "fp8"`` / compressed-tensors), AWQ GEMM
(packed int4 ``qweight`` / ``qzeros`` plus group scales), and EXL3 (trellis-coded
sibling tensors, ``quant_method: "exl3"``; decode math in ``loader/exl3.py``).
Per family:

- Pure numpy fp8 math: :func:`decode_f8` (re-exported from
  :mod:`emmy.compiler.dtype`, the leaf module the LUT lives in so the
  ``from_f8*`` decode intrinsics in ``ir/elementwise.py`` share the one table)
  and :func:`dequantize` (scale application with the granularity DERIVED from
  the weight/scale shapes — per-tensor, per-out-channel and 2-D block are one
  broadcast form).
- :func:`spell_quantized_constants`: immediately after trace, rewrite each
  fp8-stored weight ``ConstantOp`` into in-graph algebra — a bits constant
  (f8 dtype) + a scale constant + the decode-cast / broadcast-multiply cone.
  From that point the graph carries NO quantization metadata; a quantized
  weight is just constants + algebra, and the cone stays in-graph so the
  compressed device storage reaches the ordinary lowering and scheduling rules
  (``032_fold_constant_subgraphs`` declines every storage-decode cone).
- :func:`spell_dynamic_fp8_activations`: when the same checkpoint explicitly
  declares dynamic activation scaling, wrap each eligible linear input in the
  per-row amax / encode / decode algebra. The graph then carries the checkpoint's
  W8A8 computation directly; later passes still see only dtypes and tensor algebra.
- :func:`spell_quantized_inputs`: the input-sourced twin of the constant
  speller for graphs whose weights are forward-argument ``InputOp``s (the MoE
  serving seam's expert programs). Each named input becomes an fp8 bits input
  (uint8 carrier at the feed) plus a new scale input, with the same dequant
  cone spelled in-graph.
- :func:`load_dequantized_state_dict`: the eager / accuracy twin's state dict,
  every fp8 weight dequantized. Reads config + index directly (loader-band —
  it feeds the architecture twin before any graph exists, and its whole-dict
  semantics, consumed scales dropped, have no graph counterpart).
- :func:`spell_trellis_constants` and :func:`spell_trellis_inputs`: EXL3
  checkpoint siblings and their sole linear consumer are rewritten together at
  graph birth in loader/trellis.py. The serving trunk, experts, and coded output
  head all use the same format-neutral factorized algebra. No materialized
  decoded-weight fallback enters the compiler.
  :func:`load_dequantized_state_dict` uses the same decode for the eager twin.
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from emmy.compiler.dtype import decode_f8  # noqa: F401 — re-exported; the LUT's home is the dtype layer
from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.loader.exl3 import HAD_BLOCK, decode_trellis, fold_hadamard

logger = logging.getLogger(__name__)

# safetensors dtype names → canonical dtype tokens for the fp8 storage formats.
F8_SAFETENSORS_DTYPES: dict[str, str] = {"F8_E4M3": "f8e4m3", "F8_E5M2": "f8e5m2"}

# The sibling-tensor leaves one EXL3-quantized linear may ship beside its packed codes
# (``<module>.trellis``): the fp16 channel vectors, the codebook marker scalars (PRESENCE
# selects codebook 1 / 2), and the legacy packed sign words of old checkpoints. ``bias``
# is NOT here — it is a plain tensor the twin/loader reads by its own key.
_EXL3_SIBLING_LEAVES = ("suh", "svh", "mcg", "mul1", "su", "sv")

# AutoAWQ GEMM stores eight output-channel nibbles in one i32 word using
# ``[0, 2, 4, 6, 1, 3, 5, 7]`` as the pack order. Reading the shifts in this
# inverse order emits logical output channels directly, without a gather.
_AWQ4_LOGICAL_SHIFTS = (0, 16, 4, 20, 8, 24, 12, 28)


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


def _exl3_quant_config(model_dir: Path) -> dict | None:
    """The checkpoint's ``quantization_config`` when it declares the EXL3 trellis scheme.

    One format in the wild: ``config.json`` carries ``quant_method: "exl3"`` (plus
    ``bits`` / ``head_bits`` / ``calibration`` — informational only; the decoder needs
    nothing beyond each linear's sibling tensors, K coming from the trellis shape).
    ``quantization_config.json`` at the checkpoint root duplicates the per-module
    ``tensor_storage`` listing and is not read — the safetensors index is the pairing
    source, same as the fp8 speller. Anything else → ``None``.
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return None
    qc = json.loads(cfg_path.read_text()).get("quantization_config")
    if isinstance(qc, dict) and qc.get("quant_method") == "exl3":
        return qc
    return None


def _awq_quant_config(model_dir: Path) -> dict | None:
    """The checkpoint's AWQ declaration, when it is the GEMM int4 layout.

    Emmy's spelling below implements the canonical AutoAWQ/vLLM GEMM layout:
    eight output-channel nibbles per i32 word, per-input-channel groups, and
    explicit zero points. Other AWQ layouts must not be mistaken for it.
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return None
    qc = json.loads(cfg_path.read_text()).get("quantization_config")
    if not isinstance(qc, dict) or qc.get("quant_method") != "awq":
        return None
    bits = int(qc.get("bits", qc.get("w_bit", 0)) or 0)
    version = str(qc.get("version", "gemm")).lower()
    if bits != 4 or version != "gemm" or qc.get("zero_point", True) is not True:
        raise ValueError(
            "unsupported AWQ checkpoint: Emmy requires GEMM int4 with explicit zero points "
            f"(bits={bits}, version={version!r}, zero_point={qc.get('zero_point')!r})"
        )
    return qc


def is_awq_checkpoint(model_dir) -> bool:
    """Whether the checkpoint declares the supported packed AWQ GEMM scheme."""
    return _awq_quant_config(Path(model_dir)) is not None


def unpack_awq4(packed: np.ndarray) -> np.ndarray:
    """Unpack AutoAWQ GEMM i32 words to logical int4 output channels.

    The returned shape is ``(*packed.shape[:-1], packed.shape[-1] * 8)``.
    Unsigned views avoid implementation-defined signed shifts; masking then
    returns values in ``[0, 15]`` exactly.
    """
    words = np.asarray(packed)
    if words.ndim != 2 or words.dtype not in (np.dtype(np.int32), np.dtype(np.uint32)):
        raise ValueError(f"AWQ packed tensor must be rank-2 i32/u32, got shape={words.shape}, dtype={words.dtype}")
    shifts = np.asarray(_AWQ4_LOGICAL_SHIFTS, dtype=np.uint32)
    unpacked = (words.astype(np.uint32, copy=False)[..., None] >> shifts) & np.uint32(0xF)
    return unpacked.reshape(words.shape[0], words.shape[1] * 8).astype(np.int8)


def dequantize_awq4(qweight: np.ndarray, qzeros: np.ndarray, scales: np.ndarray, group_size: int) -> np.ndarray:
    """Decode one AutoAWQ GEMM weight to its ``(in, out)`` value matrix."""
    qweight = np.asarray(qweight)
    qzeros = np.asarray(qzeros)
    scales = np.asarray(scales)
    if qweight.ndim != 2 or qzeros.ndim != 2 or scales.ndim != 2:
        raise ValueError(
            f"AWQ qweight/qzeros/scales must be rank-2, got {qweight.shape}, {qzeros.shape}, {scales.shape}"
        )
    k, packed_n = qweight.shape
    groups, zero_packed_n = qzeros.shape
    n = packed_n * 8
    effective_group = k if int(group_size) == -1 else int(group_size)
    if effective_group <= 0 or groups * effective_group != k:
        raise ValueError(f"AWQ group geometry {groups} x {effective_group} does not cover input size {k}")
    if zero_packed_n != packed_n or scales.shape != (groups, n):
        raise ValueError(
            f"AWQ sibling geometry mismatch: qweight={qweight.shape}, qzeros={qzeros.shape}, scales={scales.shape}"
        )
    integers = unpack_awq4(qweight).astype(np.float32)
    zeros = np.repeat(unpack_awq4(qzeros), effective_group, axis=0).astype(np.float32)
    scale_values = np.repeat(scales.astype(np.float32), effective_group, axis=0)
    return (integers - zeros) * scale_values


def is_exl3_checkpoint(model_dir) -> bool:
    """Whether the checkpoint declares the EXL3 scheme.

    Keep detection in the loader band: serving needs only the narrow policy decision
    of whether the dense trunk may remain coded, not a second public format inventory.
    """
    return _exl3_quant_config(Path(model_dir)) is not None


def checkpoint_quant_digest(model_dir) -> str | None:
    """A short hash of the checkpoint's quantization declaration, or ``None`` when it declares
    none. Identifies the CODE RATES a cache key would otherwise miss.

    The trace builds its twin from the config with ``quantization_config`` STRIPPED (transformers
    would otherwise engage its own quantizer machinery), so a key hashed off the twin's config —
    the serving pack key — cannot see the scheme at all. That is invisible for one checkpoint and
    wrong across two: a repo publishing one EXL3 rung per branch has the same architecture config
    on every branch and differs only in the allocation, which sets the coded tensors' shapes and
    therefore the compiled programs. The rungs would share a pack.

    Preference order is most specific first: the EXL3 allocation sidecar (``quantization_config.json``
    — the per-module rate listing), else the ``quantization_config`` block of ``config.json`` (fp8,
    and an EXL3 checkpoint shipped without the sidecar).
    """
    import hashlib  # noqa: PLC0415

    from emmy.compiler.loader.exl3 import _SIDECAR  # noqa: PLC0415

    model_dir = Path(model_dir)
    sidecar = model_dir / _SIDECAR
    if sidecar.exists():
        return hashlib.sha1(sidecar.read_bytes()).hexdigest()[:16]
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return None
    qc = json.loads(cfg_path.read_text()).get("quantization_config")
    if not isinstance(qc, dict):
        return None
    return hashlib.sha1(json.dumps(qc, sort_keys=True).encode()).hexdigest()[:16]


def checkpoint_quant_summary(model_dir) -> str:
    """One line naming the checkpoint's quantization scheme and rate, for a boot log.

    Keeps format knowledge in this band while letting the serving runner report WHICH checkpoint
    it opened — the rate is the observable that distinguishes two rungs of one repo, so a boot
    that prints it is a boot whose revision pinning can be checked from the log alone.
    ``"unquantized"`` when the checkpoint declares no scheme.
    """
    model_dir = Path(model_dir)
    if (qc := _exl3_quant_config(model_dir)) is not None:
        return f"exl3 {qc.get('bits')} bpw (head {qc.get('head_bits')})"
    if (qc := _fp8_quant_config(model_dir)) is not None:
        return f"fp8 {qc.get('fmt') or qc.get('quant_method')}"
    if (qc := _awq_quant_config(model_dir)) is not None:
        return f"awq int{qc.get('bits')} g{qc.get('group_size', qc.get('q_group_size'))} {qc.get('version', 'gemm')}"
    return "unquantized"


def engine_config_overrides(hf_config) -> dict:
    """HF-config overrides a serving engine needs so it does not try to own weights emmy's
    loader already owns. ``{}`` for an ordinary checkpoint (and for ``None``, the caller's
    "config unreadable").

    The trellis-coded (EXL3) scheme is the case today: vLLM carries no method for it and refuses
    the boot outright, while nothing in the engine needs one — emmy's runner owns every coded
    weight, and the single engine-owned parameter (``lm_head``) decodes to fp16 at load
    (``serving/vllm_model_gen.py``). Presented as unquantized, the model is exactly what the
    engine then treats it as. This lives in the loader band rather than at the ``emmy serve``
    call site because naming a checkpoint scheme is frontend-band knowledge."""
    scheme = getattr(hf_config, "quantization_config", None)
    method = scheme.get("quant_method") if isinstance(scheme, dict) else getattr(scheme, "quant_method", None)
    return {"quantization_config": None} if method in {"exl3", "awq"} else {}


def strip_engine_quant_config(hf_config) -> None:
    """Remove checkpoint quantizer ownership from a shape-only engine config.

    Emmy owns coded-weight spelling for serving twins.  Keeping this mutation in the
    loader band prevents checkpoint-format metadata from leaking into serving graph
    construction while still letting callers build a weight-free architecture twin.
    """
    if getattr(hf_config, "quantization_config", None) is not None:
        delattr(hf_config, "quantization_config")


def _exl3_codebook(index, base: str) -> int:
    """The codebook id of the EXL3 linear at ``base``: marker-sibling PRESENCE in the
    index selects it (``mcg`` → 1, ``mul1`` → 2, neither → 0; stored values never read)."""
    return 1 if base + ".mcg" in index else 2 if base + ".mul1" in index else 0


def _is_skipped(weight_key: str, patterns: list[str]) -> bool:
    """Whether the weight's module is excluded from quantization.

    ``quant_method: fp8`` lists module names in ``modules_to_not_convert``;
    compressed-tensors lists them in ``ignore``, where a ``re:`` prefix marks a
    regex. Plain entries match the module path exactly or as a dotted prefix /
    suffix component (``"lm_head"`` matches both ``lm_head`` and
    ``model.lm_head``; a parent-module entry covers its children).
    """
    module = weight_key.rsplit(".", 1)[0]  # drop the parameter leaf (".weight", ".gate_up_proj", …)
    for pat in patterns:
        if pat.startswith("re:"):
            if re.match(pat[3:], module):
                return True
        elif module == pat or module.endswith("." + pat) or module.startswith(pat + "."):
            return True
    return False


def load_dequantized_state_dict(model_dir: str | Path) -> dict[str, np.ndarray]:
    """Every checkpoint tensor as numpy VALUES, including supported coded weights.

    Feeds the architecture twin of a quantized checkpoint (see
    ``trace.huggingface.load_quantized_twin``) so the eager / accuracy reference
    carries the real weights, not random init. bf16 / fp8 storage reads as f32
    values; each fp8 ``<key>`` with a ``<key>_scale`` / ``<key>_scale_inv``
    partner (and not excluded by ``modules_to_not_convert`` / ``ignore``) is
    dequantized, and the consumed scale tensors are dropped from the result.
    The general pairing subsumes the ``.weight`` → ``.weight_scale`` convention
    and covers non-``.weight`` leaves (gpt-oss 3-D expert params:
    ``…experts.gate_up_proj`` + ``…experts.gate_up_proj_scale``). An
    unquantized checkpoint passes through unchanged.

    EXL3 checkpoints: each linear's sibling tensors (``<module>.trellis`` +
    ``suh``/``svh`` + markers) decode to a ``<module>.weight`` value in the
    HF ``(out, in)`` orientation, fp16 (the decode's canonical precision);
    the consumed siblings are dropped. NOTE: whole-dict semantics — the full
    decoded footprint materializes in host memory, so this is for models (or
    config-truncated checkpoints) whose expanded weights fit in RAM.

    AWQ GEMM checkpoints: each ``<module>.qweight`` / ``qzeros`` / ``scales``
    triplet decodes to ``<module>.weight`` in HF ``(out, in)`` orientation.
    """
    from emmy.compiler.loader.safetensors import _build_index, _read_shard  # noqa: PLC0415

    model_dir = Path(model_dir)
    index = _build_index(model_dir)
    qc = _fp8_quant_config(model_dir)
    exl3 = _exl3_quant_config(model_dir) is not None
    awq = _awq_quant_config(model_dir)
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
        if awq is not None and key.endswith(".qweight"):
            base = key[: -len(".qweight")]
            qzeros_key, scales_key = base + ".qzeros", base + ".scales"
            if qzeros_key not in index or scales_key not in index:
                raise ValueError(f"AWQ linear {base!r} is missing qzeros or scales")
            out[base + ".weight"] = dequantize_awq4(
                sources[key], sources[qzeros_key], sources[scales_key], int(awq.get("group_size", awq.get("q_group_size", -1)))
            ).T
            consumed |= {key, qzeros_key, scales_key}
            continue
        if exl3 and key.endswith(".trellis"):
            base = key[: -len(".trellis")]
            sibs = {leaf for leaf in _EXL3_SIBLING_LEAVES if base + "." + leaf in index}
            consumed |= {base + "." + leaf for leaf in sibs} | {key}
            if not {"suh", "svh"} <= sibs:
                logger.warning("EXL3 linear %s: no suh/svh channel vectors (legacy packed-sign checkpoint?); left undecoded", base)
                continue
            cb = _exl3_codebook(index, base)
            out[base + ".weight"] = fold_hadamard(decode_trellis(sources[key], cb), sources[base + ".suh"], sources[base + ".svh"]).T
            continue
        if key in consumed:
            continue
        if awq is not None and key.endswith((".qzeros", ".scales")):
            base = key.rsplit(".", 1)[0]
            if base + ".qweight" in index:
                consumed.add(key)
                continue
        if qc is not None and key in fp8_keys and not _is_skipped(key, patterns):
            scale_key = next((k for k in (key + "_scale", key + "_scale_inv") if k in index), None)
            if scale_key is not None:
                out[key] = dequantize(sources[key], sources[scale_key], inverse=scale_key.endswith("_inv"))
                consumed.add(scale_key)
                continue
        out[key] = sources[key]
    for k in consumed:
        out.pop(k, None)
    return out


def _scale_layout(
    shape: tuple[int, ...], scale_shape: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], bool] | None:
    """Derive the broadcast layout pairing a scale onto a weight ``shape``.

    Returns ``(scale_decl, grid, block, degenerate)`` — the scale's declared graph shape, the
    block grid, the derived block sizes, and whether every axis is a whole-axis or per-element
    block (per-tensor / per-out-channel: the scale broadcasts straight onto the weight shape,
    no reshape pair) — or ``None`` when the scale does not evenly tile the weight."""
    per_tensor = int(np.prod(scale_shape)) == 1 if scale_shape else True
    if per_tensor:
        grid: tuple[int, ...] = (1,) * len(shape)
    else:
        if len(scale_shape) != len(shape) or any(sd == 0 or wd % sd for wd, sd in zip(shape, scale_shape, strict=True)):
            return None
        grid = tuple(scale_shape)
    block = tuple(s // g for g, s in zip(grid, shape, strict=True))
    degenerate = all(g in (1, s) for g, s in zip(grid, shape, strict=True))
    scale_decl = grid if degenerate else tuple(x for g in grid for x in (g, 1))
    return scale_decl, grid, block, degenerate


def _dequant_cone(
    g: Graph,
    w_id: str,
    scale_id: str,
    *,
    fmt: str,
    shape: tuple[int, ...],
    out_dtype,
    out_name: str,
    inverse: bool,
    grid: tuple[int, ...],
    block: tuple[int, ...],
    degenerate: bool,
) -> str:
    """Add the decode-cast / broadcast-multiply cone over the existing nodes ``w_id`` (bits)
    and ``scale_id`` (scale) to ``g``; returns the final node id. One form for every
    granularity (the block is DERIVED from the two shapes, :func:`_scale_layout`):

        bits → decode cast (``from_f8e4m3`` / ``from_f8e5m2`` — the LUT decode IS the cast's
        semantics; a plain ``copy`` would move uint8 bits, not values)
          → [reshape to the interleaved block grid — genuine 2-D block scales only]
          → multiply (divide on ``inverse``) by the broadcast scale
          → [reshape back]
    """
    # Function-level imports: the broadcast helper lives with the decomposition
    # rules that share it; importing lazily keeps the loader package light.
    from emmy.compiler.ir.frontend.ir import ReshapeOp  # noqa: PLC0415
    from emmy.compiler.ir.tensor.ir import ElementwiseOp  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    cast = g.add_node(
        op=ElementwiseOp(op=f"from_{fmt}"),
        inputs=[w_id],
        output=Tensor(f"{out_name}_dq", shape, out_dtype),
    )
    mul = ElementwiseOp(op="divide" if inverse else "multiply")
    if degenerate:
        s_bc = broadcast_to(g, scale_id, shape)
        return g.add_node(op=mul, inputs=[cast, s_bc], output=Tensor(out_name, shape, out_dtype))
    interleaved = tuple(x for gg, b in zip(grid, block, strict=True) for x in (gg, b))
    blk = g.add_node(op=ReshapeOp(shape=interleaved), inputs=[cast], output=Tensor(f"{out_name}_blk", interleaved, out_dtype))
    s_bc = broadcast_to(g, scale_id, interleaved)
    scaled = g.add_node(op=mul, inputs=[blk, s_bc], output=Tensor(f"{out_name}_sblk", interleaved, out_dtype))
    return g.add_node(op=ReshapeOp(shape=shape), inputs=[scaled], output=Tensor(out_name, shape, out_dtype))


def _spell_one(graph: Graph, nid: str, *, fmt: str, scale_key: str, scale_shape: tuple[int, ...], scale_dtype: str) -> bool:
    """Rewrite the weight constant ``nid`` into its dequant cone (:func:`_dequant_cone` over a
    bits ConstantOp reading the weight's ``source_path`` and a scale ConstantOp reading
    ``scale_key``). Returns ``True`` on success.

    The cone's OUTPUT tensor keeps exactly the dtype/shape the trace promised, so every
    later pass is unaffected. A scale that does not evenly divide the weight leaves the
    constant alone (``False``) — never a compile error.
    """
    from emmy.compiler.ir.frontend.ir import ReshapeOp  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    node = graph.nodes[nid]
    op, out = node.op, node.output
    if op.load_ops or op.source_parts or op.value is not None:
        return False  # only a pristine single-source trace constant is spelled (birth time — nothing has run yet)
    if any(not d.is_static for d in out.shape):
        return False
    shape = tuple(d.as_static() for d in out.shape)

    layout = _scale_layout(shape, tuple(scale_shape))
    if layout is None:
        logger.warning("quantized weight %s: scale shape %s does not tile %s; constant left alone", nid, scale_shape, shape)
        return False
    scale_decl, grid, block, degenerate = layout
    # Normalize the STORED scale layout (a ``()``/``(1,)`` per-tensor scale, or the
    # interleaved block form) to the declared shape at bind time.
    scale_ops: tuple = () if tuple(scale_shape) == scale_decl else (ReshapeOp(shape=scale_decl),)
    # A bf16-stored scale reads as f32 VALUES through the loader (the bits
    # carrier has no numpy value dtype), so the graph tensor says f32.
    graph_scale_dtype = "f32" if scale_dtype == "bf16" else scale_dtype

    frag = Graph()
    w = frag.add_node(
        op=ConstantOp(name=op.name, source_path=op.source_path, source_shape=shape, source_dtype=fmt),
        inputs=[],
        output=Tensor(f"{out.name}_bits", shape, fmt),
    )
    scale = frag.add_node(
        op=ConstantOp(
            name=f"{op.name}_scale",
            source_path=scale_key,
            source_shape=tuple(scale_shape),
            source_dtype=scale_dtype,
            load_ops=scale_ops,
        ),
        inputs=[],
        output=Tensor(f"{out.name}_scale", scale_decl, graph_scale_dtype),
    )
    final = _dequant_cone(
        frag,
        w,
        scale,
        fmt=fmt,
        shape=shape,
        out_dtype=out.dtype,
        out_name=out.name,
        inverse=scale_key.endswith("_inv"),
        grid=grid,
        block=block,
        degenerate=degenerate,
    )
    frag.outputs = [final]
    graph.splice(frag, consumed=[nid], output=nid)
    return True


def _spell_awq4_weight(
    graph: Graph,
    nid: str,
    *,
    base: str,
    qweight_shape: tuple[int, int],
    qzeros_shape: tuple[int, int],
    scales_shape: tuple[int, int],
    scale_dtype: str,
    group_size: int,
) -> None:
    """Replace one logical weight constant with packed AWQ decode algebra."""
    from emmy.compiler.ir.frontend.ir import ReshapeOp, TransposeOp  # noqa: PLC0415
    from emmy.compiler.ir.tensor.ir import ElementwiseOp, RangeOp  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    node = graph.nodes[nid]
    op, out = node.op, node.output
    if op.load_ops or op.source_parts or op.value is not None or any(not d.is_static for d in out.shape):
        raise ValueError(f"AWQ weight {nid!r} is not a pristine static checkpoint constant")
    if len(out.shape) != 2:
        raise ValueError(f"AWQ weight {nid!r} must be rank-2, got {tuple(out.shape)}")

    n, k = (d.as_static() for d in out.shape)
    qk, packed_n = qweight_shape
    groups, zero_packed_n = qzeros_shape
    effective_group = k if group_size == -1 else group_size
    if (
        qk != k
        or packed_n * 8 != n
        or zero_packed_n != packed_n
        or scales_shape != (groups, n)
        or effective_group <= 0
        or groups * effective_group != k
    ):
        raise ValueError(
            f"AWQ weight {nid!r} storage geometry qweight={qweight_shape}, qzeros={qzeros_shape}, "
            f"scales={scales_shape}, group={effective_group} does not reproduce logical {(n, k)}"
        )

    frag = Graph()
    qweight = frag.add_node(
        op=ConstantOp(name=f"{op.name}_qweight", source_path=base + ".qweight", source_shape=qweight_shape, source_dtype="i32"),
        inputs=[],
        output=Tensor(f"{out.name}_qweight", qweight_shape, "i32"),
    )
    qzeros = frag.add_node(
        op=ConstantOp(name=f"{op.name}_qzeros", source_path=base + ".qzeros", source_shape=qzeros_shape, source_dtype="i32"),
        inputs=[],
        output=Tensor(f"{out.name}_qzeros", qzeros_shape, "i32"),
    )
    graph_scale_dtype = "f32" if scale_dtype == "bf16" else scale_dtype
    scales = frag.add_node(
        op=ConstantOp(
            name=f"{op.name}_scales", source_path=base + ".scales", source_shape=scales_shape, source_dtype=scale_dtype
        ),
        inputs=[],
        output=Tensor(f"{out.name}_scales", scales_shape, graph_scale_dtype),
    )
    slots = frag.add_node(
        op=RangeOp(start=0, stop=8, step=1, dtype="i32"),
        inputs=[],
        output=Tensor(f"{out.name}_awq4_slots", (8,), "i32"),
    )
    slots = frag.add_node(
        op=ReshapeOp(shape=(1, 1, 8)),
        inputs=[slots],
        output=Tensor(f"{out.name}_awq4_slots_view", (1, 1, 8), "i32"),
    )
    two = const_bc(frag, name=f"{out.name}_awq4_two", value=2, target_shape=(1, 1, 8), dtype="i32")
    low_lane = frag.add_node(
        op=ElementwiseOp(op="remainder"),
        inputs=[slots, two],
        output=Tensor(f"{out.name}_awq4_low_lane", (1, 1, 8), "i32"),
    )
    high_lane = frag.add_node(
        op=ElementwiseOp(op="floor_divide"),
        inputs=[slots, two],
        output=Tensor(f"{out.name}_awq4_high_lane", (1, 1, 8), "i32"),
    )
    sixteen = const_bc(frag, name=f"{out.name}_awq4_sixteen", value=16, target_shape=(1, 1, 8), dtype="i32")
    four = const_bc(frag, name=f"{out.name}_awq4_four", value=4, target_shape=(1, 1, 8), dtype="i32")
    low_shift = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[low_lane, sixteen],
        output=Tensor(f"{out.name}_awq4_low_shift", (1, 1, 8), "i32"),
    )
    high_shift = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[high_lane, four],
        output=Tensor(f"{out.name}_awq4_high_shift", (1, 1, 8), "i32"),
    )
    shifts = frag.add_node(
        op=ElementwiseOp(op="add"),
        inputs=[low_shift, high_shift],
        output=Tensor(f"{out.name}_awq4_shifts", (1, 1, 8), "i32"),
    )

    def unpack(packed: str, shape: tuple[int, int], name: str) -> str:
        rows, words = shape
        expanded_shape = (rows, words, 8)
        view = frag.add_node(
            op=ReshapeOp(shape=(rows, words, 1)),
            inputs=[packed],
            output=Tensor(f"{name}_view", (rows, words, 1), "i32"),
        )
        words_bc = broadcast_to(frag, view, expanded_shape)
        shifts_bc = broadcast_to(frag, shifts, expanded_shape)
        shifted = frag.add_node(
            op=ElementwiseOp(op="right_shift"),
            inputs=[words_bc, shifts_bc],
            output=Tensor(f"{name}_shifted", expanded_shape, "i32"),
        )
        mask = const_bc(frag, name=f"{name}_mask", value=15, target_shape=expanded_shape, dtype="i32")
        nibble = frag.add_node(
            op=ElementwiseOp(op="bitwise_and"),
            inputs=[shifted, mask],
            output=Tensor(f"{name}_nibble", expanded_shape, "i32"),
        )
        return frag.add_node(
            op=ReshapeOp(shape=(rows, words * 8)),
            inputs=[nibble],
            output=Tensor(name, (rows, words * 8), "i32"),
        )

    integers = unpack(qweight, qweight_shape, f"{out.name}_integers")
    zeros_grouped = unpack(qzeros, qzeros_shape, f"{out.name}_zeros_grouped")

    zeros_view = frag.add_node(
        op=ReshapeOp(shape=(groups, 1, n)),
        inputs=[zeros_grouped],
        output=Tensor(f"{out.name}_zeros_view", (groups, 1, n), "i32"),
    )
    zeros_bc = broadcast_to(frag, zeros_view, (groups, effective_group, n))
    zeros = frag.add_node(
        op=ReshapeOp(shape=(k, n)),
        inputs=[zeros_bc],
        output=Tensor(f"{out.name}_zeros", (k, n), "i32"),
    )
    scale_view = frag.add_node(
        op=ReshapeOp(shape=(groups, 1, n)),
        inputs=[scales],
        output=Tensor(f"{out.name}_scales_view", (groups, 1, n), graph_scale_dtype),
    )
    scale_bc = broadcast_to(frag, scale_view, (groups, effective_group, n))
    scale_values = frag.add_node(
        op=ReshapeOp(shape=(k, n)),
        inputs=[scale_bc],
        output=Tensor(f"{out.name}_scale_values", (k, n), graph_scale_dtype),
    )
    centered = frag.add_node(
        op=ElementwiseOp(op="subtract"),
        inputs=[integers, zeros],
        output=Tensor(f"{out.name}_centered", (k, n), "i32"),
    )
    values = frag.add_node(
        op=ElementwiseOp(op="copy"),
        inputs=[centered],
        output=Tensor(f"{out.name}_values", (k, n), out.dtype),
    )
    if graph_scale_dtype != out.dtype.name:
        scale_values = frag.add_node(
            op=ElementwiseOp(op="copy"),
            inputs=[scale_values],
            output=Tensor(f"{out.name}_scale_values_cast", (k, n), out.dtype),
        )
    scaled = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[values, scale_values],
        output=Tensor(f"{out.name}_scaled", (k, n), out.dtype),
    )
    final = frag.add_node(
        op=TransposeOp(axes=(1, 0)),
        inputs=[scaled],
        output=Tensor(out.name, (n, k), out.dtype),
    )
    frag.outputs = [final]
    graph.splice(frag, consumed=[nid], output=nid)


def _spell_awq4_constants(graph: Graph, model_dir: Path, qc: dict) -> int:
    """Spell supported AWQ GEMM checkpoint constants as packed decode algebra."""
    from safetensors import safe_open  # noqa: PLC0415

    from emmy.compiler.loader.safetensors import _build_index, _candidate_keys  # noqa: PLC0415

    index = _build_index(model_dir)
    group_size = int(qc.get("group_size", qc.get("q_group_size", -1)))
    spelled = 0
    with ExitStack() as stack:
        handles: dict[str, object] = {}

        def _slice(key: str):
            path = str(index[key])
            handle = handles.get(path)
            if handle is None:
                handle = handles[path] = stack.enter_context(safe_open(path, framework="numpy"))
            return handle.get_slice(key)

        for nid, op in list(graph.loadable_constants()):
            if op.source_path is None or not op.source_path.endswith(".weight"):
                continue
            suffix = len(".weight")
            base = next(
                (
                    candidate[:-suffix]
                    for candidate in _candidate_keys(op.source_path)
                    if candidate.endswith(".weight") and candidate[:-suffix] + ".qweight" in index
                ),
                None,
            )
            if base is None:
                continue
            siblings = {leaf: base + "." + leaf for leaf in ("qweight", "qzeros", "scales")}
            missing = [key for key in siblings.values() if key not in index]
            if missing:
                raise ValueError(f"AWQ weight {nid!r} is missing checkpoint siblings {missing}")
            slices = {leaf: _slice(key) for leaf, key in siblings.items()}
            shapes = {leaf: tuple(int(d) for d in sl.get_shape()) for leaf, sl in slices.items()}
            _spell_awq4_weight(
                graph,
                nid,
                base=base,
                qweight_shape=shapes["qweight"],
                qzeros_shape=shapes["qzeros"],
                scales_shape=shapes["scales"],
                scale_dtype=slices["scales"].get_dtype().lower(),
                group_size=group_size,
            )
            spelled += 1
    if spelled:
        logger.info("spelled %d packed AWQ int4 weight constant(s) from %s", spelled, model_dir)
    return spelled


def spell_quantized_constants(graph: Graph, model_id_or_path: str) -> int:
    """Spell supported compressed weights as in-graph dequant algebra, at birth.

    Source of truth is the CHECKPOINT, not the traced module (a quantized
    checkpoint is traced through its bf16 architecture twin, whose module
    carries no fp8 tensors): the ``config.json`` ``quantization_config``
    declares the scheme, and the safetensors index supplies the pairing — a
    weight stored as fp8 whose ``<key>_scale`` (or ``<key>_scale_inv`` →
    divide) tensor is present in the index is rewritten into its dequant cone
    (:func:`_spell_one`); the general pairing subsumes the ``.weight`` →
    ``.weight_scale`` convention and covers non-``.weight`` leaves (gpt-oss
    3-D expert params). ``modules_to_not_convert`` / ``ignore`` entries and
    weights with no scale tensor are left alone and load as plain tensors. Runs immediately after trace and before the pipeline —
    node replacement is safe there (nothing has consumed the graph yet).
    Unquantized checkpoints are a no-op — zero graph change. Returns the
    number of constants spelled.
    """
    # Function-level import: this module owns the pure math + the spelling; the
    # shard-index helpers live with the loader that also consumes them.
    from safetensors import safe_open  # noqa: PLC0415

    from emmy.compiler.loader.safetensors import _build_index, _candidate_keys, _resolve_model_dir  # noqa: PLC0415

    model_dir = _resolve_model_dir(model_id_or_path)
    qc = _fp8_quant_config(model_dir)
    if (awq := _awq_quant_config(model_dir)) is not None:
        return _spell_awq4_constants(graph, model_dir, awq)
    if qc is None:
        return 0
    patterns = list(qc.get("modules_to_not_convert") or []) + list(qc.get("ignore") or [])
    index = _build_index(model_dir)

    spelled = 0
    with ExitStack() as stack:
        handles: dict[str, object] = {}  # shard path → open safe_open handle (metadata reads only)

        def _slice(key: str):
            path = str(index[key])
            handle = handles.get(path)
            if handle is None:
                handle = handles[path] = stack.enter_context(safe_open(path, framework="numpy"))
            return handle.get_slice(key)

        for nid, op in list(graph.loadable_constants()):
            if op.source_path is None or op.source_dtype in F8_SAFETENSORS_DTYPES.values():
                continue  # source-less, or an already-spelled bits constant (idempotency)
            key = next((c for c in _candidate_keys(op.source_path) if c in index), None)
            if key is None or _is_skipped(key, patterns):
                continue
            stored = _slice(key).get_dtype()
            if stored not in F8_SAFETENSORS_DTYPES:
                continue  # e.g. a modules-kept-in-bf16 weight, or a non-fp8 group member
            scale_key = next((k for k in (key + "_scale", key + "_scale_inv") if k in index), None)
            if scale_key is None:
                continue  # no paired scale in the index → loads as a plain tensor
            scale_slice = _slice(scale_key)
            if _spell_one(
                graph,
                nid,
                fmt=F8_SAFETENSORS_DTYPES[stored],
                scale_key=scale_key,
                scale_shape=tuple(int(d) for d in scale_slice.get_shape()),
                scale_dtype=scale_slice.get_dtype().lower(),
            ):
                spelled += 1
    if spelled:
        logger.info("spelled %d quantized weight constant(s) from %s", spelled, model_dir)
    return spelled


def _dynamic_activation_declaration(qc: dict | None) -> tuple[bool, str | None]:
    """Return ``(enabled, declared_fmt)`` for a supported dynamic FP8 activation scheme."""
    if not qc:
        return False, None
    if qc.get("quant_method") == "fp8" and qc.get("activation_scheme") == "dynamic":
        fmt = {"e4m3": "f8e4m3", "e5m2": "f8e5m2", "f8e4m3": "f8e4m3", "f8e5m2": "f8e5m2"}.get(qc.get("fmt"))
        return fmt is not None, fmt
    if qc.get("quant_method") == "compressed-tensors":
        groups = list((qc.get("config_groups") or {}).values())
        # Fail closed on mixed declarations: without resolving every group's target
        # selector, one dynamic group must not turn a weight-only group into W8A8.
        if groups and all(
            isinstance(group, dict)
            and isinstance(group.get("weights"), dict)
            and group["weights"].get("type") == "float"
            and int(group["weights"].get("num_bits") or 0) == 8
            and isinstance(group.get("input_activations"), dict)
            and group["input_activations"].get("type") == "float"
            and int(group["input_activations"].get("num_bits") or 0) == 8
            and group["input_activations"].get("dynamic") is True
            and group["input_activations"].get("strategy") == "token"
            for group in groups
        ):
            return True, None  # the paired weight's concrete f8 storage dtype selects the encode format
    return False, None


def _cone_storage_formats(graph: Graph, start: str) -> set[str]:
    """Return FP8 storage dtypes reachable upstream from one graph buffer."""
    pending = [start]
    seen: set[str] = set()
    formats: set[str] = set()
    while pending:
        buf = pending.pop()
        node = graph.producer(buf)
        if node is None or node.id in seen:
            continue
        seen.add(node.id)
        if isinstance(node.op, ConstantOp):
            dtype = node.output.dtype.name
            if dtype in F8_SAFETENSORS_DTYPES.values():
                formats.add(dtype)
        pending.extend(node.inputs)
    return formats


def _cone_has_fp8_encode(graph: Graph, start: str) -> bool:
    """Whether an activation buffer is already downstream of an FP8 encode."""
    from emmy.compiler.ir.tensor.ir import ElementwiseOp  # noqa: PLC0415

    pending = [start]
    seen: set[str] = set()
    while pending:
        buf = pending.pop()
        node = graph.producer(buf)
        if node is None or node.id in seen:
            continue
        seen.add(node.id)
        if isinstance(node.op, ElementwiseOp) and node.op.op.name.startswith("to_f8"):
            return True
        pending.extend(node.inputs)
    return False


def _fresh_buffer_name(graph: Graph, base: str) -> str:
    """Return a deterministic unused primary-buffer name."""
    name = base
    suffix = 2
    while name in graph.nodes or graph.buffer(name) is not None:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def _spell_dynamic_activation(graph: Graph, activation: str, fmt: str) -> str:
    """Spell one shared per-row dynamic FP8 activation value and return its decoded buffer."""
    from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    source = graph.buffer(activation)
    if source is None or not source.shape or source.dtype.name not in {"f16", "bf16", "f32"}:
        raise ValueError(f"dynamic FP8 activation {activation!r} must be a rank-1+ floating tensor")
    shape = tuple(source.shape)
    scale_shape = (*shape[:-1], 1)
    stem = _fresh_buffer_name(graph, f"{activation}_dynamic_fp8")

    absolute = graph.add_node(
        op=ElementwiseOp(op="abs"),
        inputs=[activation],
        output=Tensor(f"{stem}_abs", shape, "f32"),
    )
    amax = graph.add_node(
        op=ReduceOp(op="maximum", axis=-1),
        inputs=[absolute],
        output=Tensor(f"{stem}_amax", scale_shape, "f32"),
    )
    floor = const_bc(graph, name=f"{stem}_floor", value=1.0e-12, target_shape=scale_shape, dtype="f32")
    stable_amax = graph.add_node(
        op=ElementwiseOp(op="maximum"),
        inputs=[amax, floor],
        output=Tensor(f"{stem}_stable_amax", scale_shape, "f32"),
    )
    finite_max = 448.0 if fmt == "f8e4m3" else 57344.0
    denominator = const_bc(graph, name=f"{stem}_finite_max", value=finite_max, target_shape=scale_shape, dtype="f32")
    scale = graph.add_node(
        op=ElementwiseOp(op="divide"),
        inputs=[stable_amax, denominator],
        output=Tensor(f"{stem}_scale", scale_shape, "f32"),
    )
    scale_bc = broadcast_to(graph, scale, shape)
    normalized = graph.add_node(
        op=ElementwiseOp(op="divide"),
        inputs=[activation, scale_bc],
        output=Tensor(f"{stem}_normalized", shape, "f32"),
    )
    bits = graph.add_node(
        op=ElementwiseOp(op=f"to_{fmt}"),
        inputs=[normalized],
        output=Tensor(f"{stem}_bits", shape, fmt),
    )
    decoded = graph.add_node(
        op=ElementwiseOp(op=f"from_{fmt}"),
        inputs=[bits],
        output=Tensor(f"{stem}_decoded", shape, source.dtype),
    )
    restored = graph.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[decoded, scale_bc],
        output=Tensor(f"{stem}_value", shape, source.dtype),
    )

    # Trace inventories promote these intermediates to auxiliary outputs before
    # fusion. That preserves the genuine encode/scale boundary needed by a native
    # W8A8 contraction without changing ordinary model-call outputs.
    graph.nodes[bits].hints.set("trace.materialize", True)
    graph.nodes[scale].hints.set("trace.materialize", True)
    return restored


def spell_dynamic_fp8_activations(graph: Graph, model_id_or_path: str) -> int:
    """Spell checkpoint-declared dynamic FP8 activations in front of eligible linears.

    The official ``quant_method: fp8`` declaration supplies both the activation
    scheme and format. A linear is eligible only when its already-spelled weight
    cone contains that same FP8 storage dtype. Shared projection inputs reuse one
    quantized activation value. The zero-safe per-row scale is ``max(amax(abs(x)),
    1e-12) / finite_max``. Returns the number of rewired linears; unsupported or
    weight-only declarations are a no-op.
    """
    from emmy.compiler.ir.frontend.ir import LinearOp  # noqa: PLC0415
    from emmy.compiler.loader.safetensors import _resolve_model_dir  # noqa: PLC0415

    model_dir = _resolve_model_dir(model_id_or_path)
    enabled, declared_fmt = _dynamic_activation_declaration(_fp8_quant_config(model_dir))
    if not enabled:
        return 0

    eligible: list[tuple[str, str, str]] = []
    for node in list(graph.nodes.values()):
        if not isinstance(node.op, LinearOp) or len(node.inputs) < 2:
            continue
        activation, weight = node.inputs[:2]
        formats = _cone_storage_formats(graph, weight)
        if len(formats) != 1 or (declared_fmt is not None and formats != {declared_fmt}) or _cone_has_fp8_encode(graph, activation):
            continue
        eligible.append((node.id, activation, next(iter(formats))))

    rewritten: dict[tuple[str, str], str] = {}
    for linear_id, activation, fmt in eligible:
        key = (activation, fmt)
        restored = rewritten.get(key)
        if restored is None:
            restored = rewritten[key] = _spell_dynamic_activation(graph, activation, fmt)
        graph.replace_input(linear_id, activation, restored)
    if eligible:
        formats = ", ".join(sorted({fmt for _linear, _activation, fmt in eligible}))
        logger.info("spelled dynamic %s activation algebra for %d linear(s) from %s", formats, len(eligible), model_dir)
    return len(eligible)


def _trellis_dims(graph: Graph, nid: str, shapes: dict[str, tuple[int, ...]]) -> tuple[int, int, int, int] | None:
    """Validate a trellis weight constant against its sibling shapes → ``(n, k, n_pad, k_pad)``.

    ``(n, k)`` are the traced logical ``(out, in)`` dims; ``(n_pad, k_pad)`` the 128-padded
    extents the checkpoint actually stores. ``None`` (with a warning) when the siblings do not
    reproduce the traced weight shape, or when the constant is not a pristine single-source
    trace constant.
    """
    node = graph.nodes[nid]
    op, out = node.op, node.output
    if op.load_ops or op.source_parts or op.value is not None:
        return None  # only a pristine single-source trace constant is spelled (birth time — nothing has run yet)
    if any(not d.is_static for d in out.shape) or len(out.shape) != 2:
        return None
    n, k = (d.as_static() for d in out.shape)

    t_shape, suh_shape, svh_shape = shapes["trellis"], shapes["suh"], shapes["svh"]
    k_pad = suh_shape[0] if len(suh_shape) == 1 else -1
    n_pad = svh_shape[0] if len(svh_shape) == 1 else -1
    # EXL3 pads both dims to multiples of 128 at encode time; the decode cone slices the
    # padded weight back to the traced logical shape. The sibling extents must be exactly
    # the traced dims' roundups.
    pad_ok = (-(-k // 128) * 128, -(-n // 128) * 128) == (k_pad, n_pad) if k > 0 and n > 0 else False
    if len(t_shape) != 3 or not pad_ok or (t_shape[0] * 16, t_shape[1] * 16) != (k_pad, n_pad):
        return None
    return n, k, n_pad, k_pad


def _trellis_leaves(frag: Graph, node, base: str, shapes: dict[str, tuple[int, ...]]) -> tuple[str, str, str]:
    """Create the packed-code and channel-vector leaves used by generic reconstruction."""
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    op, out = node.op, node.output
    return tuple(
        frag.add_node(
            op=ConstantOp(name=f"{op.name}_{leaf}", source_path=f"{base}.{leaf}", source_shape=shapes[leaf], source_dtype=dt),
            inputs=[],
            output=Tensor(f"{out.name}_{leaf}", shapes[leaf], dt),
        )
        for leaf, dt in (("trellis", "i16"), ("suh", "f16"), ("svh", "f16"))
    )


def _linear_consumer(graph: Graph, nid: str):
    """Return the sole direct ``LinearOp`` consuming weight ``nid``, or fail closed."""
    from emmy.compiler.ir.frontend.ir import LinearOp  # noqa: PLC0415

    users = graph.users(nid)
    if len(users) != 1:
        raise ValueError(f"coded weight {nid!r} must have exactly one consumer, found {sorted(users)}")
    linear = graph.nodes[next(iter(users))]
    weight = graph.nodes[nid]
    if not isinstance(linear.op, LinearOp) or len(linear.inputs) < 2 or graph.producer(linear.inputs[1]) is not weight:
        raise ValueError(f"coded weight {nid!r} is not the direct B operand of one LinearOp")
    if linear.op.has_bias != (len(linear.inputs) == 3):
        raise ValueError(f"coded linear {linear.id!r} has inconsistent bias declaration/inputs")
    return linear


def _spell_trellis_linear(graph: Graph, nid: str, *, base: str, cb: int, shapes: dict[str, tuple[int, ...]]) -> None:
    """Rewrite one coded checkpoint weight and its linear consumer at graph birth."""
    from emmy.compiler.loader.trellis import spell_factored_linear  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment  # noqa: PLC0415

    dims = _trellis_dims(graph, nid, shapes)
    if dims is None:
        raise ValueError(f"coded weight {nid!r} does not match its checkpoint storage geometry")
    n, k, _n_pad, _k_pad = dims
    weight = graph.nodes[nid]
    linear = _linear_consumer(graph, nid)
    x = graph.producer(linear.inputs[0])
    bias = graph.producer(linear.inputs[2]) if linear.op.has_bias else None
    if x is None or (linear.op.has_bias and bias is None):
        raise ValueError(f"coded linear {linear.id!r} has an unsourced activation or bias")

    frag = open_fragment(graph, [x, *([bias] if bias is not None else [])])
    codes, suh, svh = _trellis_leaves(frag, weight, base, shapes)
    frag.outputs = [
        spell_factored_linear(
            frag,
            codes,
            suh,
            svh,
            cb=cb,
            weight_shape=(n, k),
            x=x.id,
            bias=bias.id if bias is not None else None,
            out=linear.output,
            weight_name=weight.output.name,
        )
    ]
    graph.splice(frag, consumed=[nid, linear.id], output=linear.id)


def spell_trellis_constants(graph: Graph, model_id_or_path: str) -> int:
    """Spell every EXL3 coded linear as generic factorized algebra at graph birth.

    Checkpoint-specific sibling discovery and codebook selection stop in this loader.
    The emitted graph contains only ordinary tensor/layout operations and never constructs
    the decoded dense weight. A coded weight outside the supported direct-linear contract is
    a compile error rather than an implicit materialization fallback.
    """
    from safetensors import safe_open  # noqa: PLC0415

    from emmy.compiler.loader.safetensors import _build_index, _candidate_keys, _resolve_model_dir  # noqa: PLC0415

    model_dir = _resolve_model_dir(model_id_or_path)
    if _exl3_quant_config(model_dir) is None:
        return 0
    index = _build_index(model_dir)

    spelled = 0
    with ExitStack() as stack:
        handles: dict[str, object] = {}

        def _shape(key: str) -> tuple[int, ...]:
            path = str(index[key])
            handle = handles.get(path)
            if handle is None:
                handle = handles[path] = stack.enter_context(safe_open(path, framework="numpy"))
            return tuple(int(d) for d in handle.get_slice(key).get_shape())

        for nid, op in list(graph.loadable_constants()):
            if op.source_path is None or not op.source_path.endswith(".weight"):
                continue
            suffix = len(".weight")
            base = next((c[:-suffix] for c in _candidate_keys(op.source_path) if c[:-suffix] + ".trellis" in index), None)
            if base is None:
                continue
            if base + ".suh" not in index or base + ".svh" not in index:
                raise ValueError(f"coded weight {nid!r}: checkpoint entry {base!r} has no suh/svh channel vectors")
            shapes = {leaf: _shape(base + "." + leaf) for leaf in ("trellis", "suh", "svh")}
            _spell_trellis_linear(graph, nid, base=base, cb=_exl3_codebook(index, base), shapes=shapes)
            spelled += 1
    if spelled:
        logger.info("spelled %d coded weight constant(s) from %s", spelled, model_dir)
    return spelled


def spell_quantized_inputs(
    graph: Graph,
    specs: dict[str, tuple[str, tuple[int, ...], str]],
    *,
    inverse: bool = False,
) -> dict[str, str]:
    """Spell named graph INPUTS as fp8 bits + scale inputs, the dequant cone in-graph.

    The input-sourced twin of :func:`spell_quantized_constants`, for graphs whose weights are
    forward-argument ``InputOp``s (the MoE serving seam's expert programs — one program per
    layer kind, per-expert 2-D weight slices fed per launch), where the constant speller can
    never fire. ``specs`` maps an input node id to ``(fmt, scale_shape, scale_dtype)``: the
    storage format token (``"f8e4m3"`` / ``"f8e5m2"``), the scale tensor's shape, and its
    STORED dtype (``"bf16"`` reads as f32 values, mirroring the loader convention).

    Each named input keeps its node id and its position in ``graph.inputs``, but its dtype
    becomes the f8 storage dtype — the feed binds the raw bit pattern on the uint8 carrier,
    the same rule as the constant side. A new ``<name>_scale`` input is appended to
    ``graph.inputs``, declared at the broadcast layout :func:`_scale_layout` derives
    (per-tensor / per-out-channel scales bind at their stored shape; a genuine 2-D block
    scale feeds reshaped to the interleaved ``(grid, 1)`` layout). The same decode-cast /
    broadcast-multiply cone as the constant speller (:func:`_dequant_cone`) re-creates the
    value tensor the trace promised — dtype, shape and consumers unchanged, so every later
    pass is unaffected. An input-rooted cone is not a constant subgraph, so the W8A16
    mul-hoist binding can absorb it exactly as on the constant path.

    ``inverse`` divides by the scale (the stored scale is the reciprocal multiplier).
    The caller names these inputs explicitly, so any mismatch raises ``ValueError`` instead
    of the constant speller's skip-and-continue. Returns ``{input id: scale input id}``.
    """
    from emmy.compiler.ir.base import InputOp  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    out_map: dict[str, str] = {}
    for name, (fmt, scale_shape, scale_dtype) in specs.items():
        if fmt not in F8_SAFETENSORS_DTYPES.values():
            raise ValueError(f"spell_quantized_inputs: unknown fp8 storage format {fmt!r}")
        node = graph.nodes.get(name)
        if node is None or not isinstance(node.op, InputOp) or name not in graph.inputs:
            raise ValueError(f"spell_quantized_inputs: {name!r} is not a graph input")
        out = node.output
        if out.dtype.name in F8_SAFETENSORS_DTYPES.values():
            raise ValueError(f"spell_quantized_inputs: input {name!r} already carries the f8 storage dtype")
        if any(not d.is_static for d in out.shape):
            raise ValueError(f"spell_quantized_inputs: input {name!r} has symbolic dims; only static weight inputs are spelled")
        shape = tuple(d.as_static() for d in out.shape)
        layout = _scale_layout(shape, tuple(scale_shape))
        if layout is None:
            raise ValueError(f"spell_quantized_inputs: scale shape {tuple(scale_shape)} does not tile input {name!r} shape {shape}")
        scale_decl, grid, block, degenerate = layout
        graph_scale_dtype = "f32" if scale_dtype == "bf16" else scale_dtype

        # Rewire order: park the traced input under a temporary id, re-mint ``name`` as the
        # bits input in the SAME ``graph.inputs`` slot, add the scale input, spell the cone,
        # then hand the parked node's consumers to the cone output and drop it — every step
        # through the index-consistent Graph mutators.
        tmp = f"{name}__dq_src"
        graph.rename_node(name, tmp)
        bits = graph.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape, fmt), node_id=name)
        graph.inputs = [bits if i == tmp else i for i in graph.inputs]
        sid = graph.add_node(
            op=InputOp(),
            inputs=[],
            output=Tensor(f"{name}_scale", scale_decl, graph_scale_dtype),
            node_id=f"{name}_scale",
        )
        graph.inputs.append(sid)
        final = _dequant_cone(
            graph,
            bits,
            sid,
            fmt=fmt,
            shape=shape,
            out_dtype=out.dtype,
            out_name=f"{out.name}_val",
            inverse=inverse,
            grid=grid,
            block=block,
            degenerate=degenerate,
        )
        graph.replace_node(tmp, final)  # the original consumers now read the cone's value
        graph.remove_node(tmp)
        out_map[name] = sid
    return out_map


def spell_trellis_inputs(graph: Graph, specs: dict[str, tuple[int, tuple[int, ...]]]) -> dict[str, tuple[str, str]]:
    """Spell coded weight inputs and their linear consumers as factorized algebra.

    Each logical weight input becomes packed int16 codes plus full channel-vector inputs.
    Its sole ``LinearOp`` consumer becomes the same generic factorized execution graph used
    for checkpoint constants. No decoded dense weight or checkpoint-shaped operation enters
    the compiler pipeline.
    """
    from emmy.compiler.ir.base import InputOp  # noqa: PLC0415
    from emmy.compiler.loader.trellis import spell_factored_linear  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    out_map: dict[str, tuple[str, str]] = {}
    for name, (cb, codes_shape) in specs.items():
        codes_shape = tuple(int(d) for d in codes_shape)
        node = graph.nodes.get(name)
        if node is None or not isinstance(node.op, InputOp) or name not in graph.inputs:
            raise ValueError(f"spell_trellis_inputs: {name!r} is not a graph input")
        out = node.output
        if any(not d.is_static for d in out.shape) or len(out.shape) != 2:
            raise ValueError(f"spell_trellis_inputs: input {name!r} must be a static 2-D (out, in) weight")
        n, k = (d.as_static() for d in out.shape)
        if len(codes_shape) != 3 or codes_shape[2] % 16:
            raise ValueError(f"spell_trellis_inputs: {name!r} codes shape {codes_shape} is not (k_pad/16, n_pad/16, 16*K)")
        k_pad, n_pad = codes_shape[0] * 16, codes_shape[1] * 16
        if (k_pad, n_pad) != (-(-k // HAD_BLOCK) * HAD_BLOCK, -(-n // HAD_BLOCK) * HAD_BLOCK):
            raise ValueError(f"spell_trellis_inputs: {name!r} codes shape {codes_shape} does not pad weight shape {(n, k)}")
        linear = _linear_consumer(graph, name)
        x = graph.producer(linear.inputs[0])
        bias = graph.producer(linear.inputs[2]) if linear.op.has_bias else None
        if x is None or (linear.op.has_bias and bias is None):
            raise ValueError(f"spell_trellis_inputs: coded linear {linear.id!r} has an unsourced activation or bias")

        tmp = f"{name}__coded_src"
        graph.rename_node(name, tmp)
        codes = graph.add_node(op=InputOp(), inputs=[], output=Tensor(name, codes_shape, "i16"), node_id=name)
        graph.inputs = [codes if item == tmp else item for item in graph.inputs]
        suh = graph.add_node(op=InputOp(), inputs=[], output=Tensor(f"{name}_suh", (k_pad,), "f16"), node_id=f"{name}_suh")
        svh = graph.add_node(op=InputOp(), inputs=[], output=Tensor(f"{name}_svh", (n_pad,), "f16"), node_id=f"{name}_svh")
        graph.inputs += [suh, svh]

        frag = open_fragment(graph, [x, codes, suh, svh, *([bias] if bias is not None else [])])
        frag.outputs = [
            spell_factored_linear(
                frag,
                codes,
                suh,
                svh,
                cb=cb,
                weight_shape=(n, k),
                x=x.id,
                bias=bias.id if bias is not None else None,
                out=linear.output,
                weight_name=f"{out.name}_decoded",
            )
        ]
        graph.splice(frag, consumed=[tmp, linear.id], output=linear.id)
        out_map[name] = (suh, svh)
    return out_map
