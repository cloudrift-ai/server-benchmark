"""Quantized-checkpoint ingestion: birth-time spelling of the decode algebra, plus the numpy math.

This module is the ONE place quantization-as-a-concept exists (together with the
safetensors loader that reads the checkpoint and ``trace/huggingface.py``'s
architecture-twin construction). Four checkpoint families share the design — FP8
(scale-paired bits, ``quant_method: "fp8"`` / compressed-tensors), AWQ GEMM
(packed int4 ``qweight`` / ``qzeros`` plus group scales), and EXL3 (trellis-coded
sibling tensors, ``quant_method: "exl3"``; decode math in ``loader/exl3.py``), plus
MXFP4 (two nibbles per byte with one E8M0 scale per 32 values). Per family:

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

from emmy.compiler.dtype import (  # noqa: F401 — re-exported; the LUTs' home is the dtype layer
    F4_VALUES,
    F8E4M3,
    F4E2M1x2,
    decode_f4x2,
    decode_f8,
    encode_f4x2,
    encode_f8,
)
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


def scale_is_reciprocal(scale_key: str) -> bool:  # noqa: ARG001 — the key is the one fact a caller has
    """Whether a checkpoint's stored scale is the RECIPROCAL of the dequant multiplier. It never
    is: DeepSeek's ``weight_scale_inv`` (Laguna, DeepSeek-V3/V4 lineage) names the inverse of the
    QUANTIZATION scale — ``q = w / s`` stored beside ``s`` — so the dequant is ``q * s`` exactly
    as for ``weight_scale`` (DeepSeek's own ``weight_dequant`` and vLLM's block-fp8 path
    multiply by it). Dividing by it, as the suffix once suggested here, scaled every
    ``_scale_inv`` weight by ``1/s²``. The ``inverse=`` plumbing stays for a checkpoint that
    declares a true reciprocal; the suffix alone never selects it."""
    return False


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


def fuse_nvfp4_scales(scale_bits: np.ndarray, scale_2: np.ndarray) -> np.ndarray:
    """The NVFP4 two-level scale collapsed to ONE f16 tensor ("fused scale"):
    ``e4m3-decode(scale_bits) * scale_2``.

    e4m3 values are exact in f16 (3 mantissa bits, range ±448), so the only rounding
    anywhere is the single f32→f16 round of the product. Kernels and the numpy oracle
    both consume this one tensor, sharing one rounding story.
    """
    assert scale_bits.dtype == np.uint8, f"fuse_nvfp4_scales expects the uint8 e4m3 bits carrier, got {scale_bits.dtype}"
    assert scale_2.dtype == np.float32 and scale_2.size == 1, (
        f"weight_scale_2 is one f32 per tensor, got {scale_2.dtype} shape {scale_2.shape}"
    )
    return (decode_f8(scale_bits, F8E4M3.name) * scale_2.reshape(())).astype(np.float16)


def dequantize_nvfp4(packed: np.ndarray, scale_bits: np.ndarray, scale_2: np.ndarray) -> np.ndarray:
    """f32 values of an NVFP4 weight: decode the packed e2m1 pairs (last axis doubles to K)
    and apply the fused f16 block scale, one per 16 elements along K.

    These are EXACTLY the numbers the kernel path computes, not an approximation: an
    e2m1 value carries ≤3 significand bits and the fused f16 scale ≤11, so every product
    fits f32's 24 — the oracle admits bitwise comparison, no tolerance windows.
    """
    assert packed.dtype == np.uint8, f"dequantize_nvfp4 expects the uint8 packed carrier, got {packed.dtype}"
    return dequantize(decode_f4x2(packed), fuse_nvfp4_scales(scale_bits, scale_2).astype(np.float32))


#: Elements per block scale along the last axis, and the two formats' largest finite values.
#: An NVFP4 checkpoint always uses 16, and the hardware's block-scaled mma reads 16 as well.
NVFP4_BLOCK = 16
_F4_MAX, _E4M3_MAX = 6.0, 448.0


def quantize_nvfp4(values: np.ndarray, *, block: int = NVFP4_BLOCK) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize f32 values TO the NVFP4 trio — the inverse of :func:`dequantize_nvfp4`, returning
    ``(packed, scale_bits, scale_2)`` in exactly the carriers a checkpoint stores: packed e2m1
    pairs, e4m3 block-scale BITS one per ``block`` elements along the last axis, and one f32 for
    the tensor.

    Two levels, because one is not enough: a block's scale must itself be stored in e4m3, whose
    range would clip on a tensor with a wide dynamic range. So ``scale_2`` carries the tensor's
    magnitude and each block's e4m3 scale is relative to it. Choosing ``scale_2 = amax /
    (6 * 448)`` puts the largest block scale near the top of e4m3 without exceeding it.

    The divisor applied per block is the FUSED scale, so quantize and dequantize meet on the one
    tensor :func:`fuse_nvfp4_scales` defines and share its single rounding story. A zero block
    keeps a zero scale, which dequantizes back to zero.

    The block scales come out non-negative, and an e4m3 bit pattern with a clear sign bit is
    also its unsigned ue4m3 reading — the form the hardware's block-scaled mma wants."""
    x = np.asarray(values, dtype=np.float32)
    assert x.shape[-1] % block == 0, f"quantize_nvfp4 needs the last axis divisible by {block}, got {x.shape[-1]}"
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // block, block)
    # Floored the same way the dynamic FP8 activation scale is, so an all-zero or vanishingly
    # small tensor cannot drive scale_2 to zero and make the block-scale division blow up.
    amax = max(float(np.abs(x).max()), 1e-12)
    # The fused scale is an f16 tensor, so the largest block scale has to fit one. That bounds the
    # tensor, and overshooting it would otherwise surface as a silent inf rather than an error.
    assert amax / _F4_MAX <= 65504.0, f"quantize_nvfp4 needs |values| <= {_F4_MAX * 65504.0:g} for an f16 fused scale, got {amax:g}"
    scale_2 = np.float32(amax / (_F4_MAX * _E4M3_MAX))
    block_scale = np.abs(blocks).max(axis=-1) / _F4_MAX
    scale_bits = encode_f8(block_scale / scale_2, F8E4M3.name)
    fused = fuse_nvfp4_scales(scale_bits, np.asarray(scale_2, dtype=np.float32).reshape(1)).astype(np.float32)
    packed = encode_f4x2(np.divide(blocks, fused[..., None], out=np.zeros_like(blocks), where=fused[..., None] != 0))
    return packed.reshape(*x.shape[:-1], x.shape[-1] // 2), scale_bits, np.asarray(scale_2, dtype=np.float32).reshape(1)


def _quantization_config(model_dir: str | Path) -> dict | None:
    """The checkpoint's declared ``quantization_config``, or ``None`` when it declares none.

    Takes the directory either way round — every recognizer reads the config through here, and a
    caller holding a plain string is the ordinary case."""
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        return None
    qc = json.loads(cfg_path.read_text()).get("quantization_config")
    return qc if isinstance(qc, dict) else None


def _quantizes_weights_at(qc: dict, num_bits: int, group_size: int | None = None) -> bool:
    """Whether ANY ``config_groups`` entry quantizes weights as float at this width.

    A checkpoint's groups are per-scheme, not per-scheme-per-checkpoint: one may hold the
    8-bit leaves and another the 4-bit ones. So this asks "is this scheme present", never
    "is this scheme the only one" — which leaf gets which is a per-tensor question that the
    stored sibling signatures answer, not a config one.
    """
    for group in (qc.get("config_groups") or {}).values():
        weights = group.get("weights") if isinstance(group, dict) else None
        if not isinstance(weights, dict) or weights.get("type") != "float":
            continue
        if int(weights.get("num_bits") or 0) != num_bits:
            continue
        if group_size is not None and int(weights.get("group_size") or 0) != group_size:
            continue
        return True
    return False


def _fp8_quant_config(model_dir: Path) -> dict | None:
    """The checkpoint's ``quantization_config`` when it quantizes ANY weights to FP8.

    Three conventions in the wild: ``quant_method: "fp8"`` (official releases);
    ``quant_method: "compressed-tensors"`` (llm-compressor) with an 8-bit float weight group;
    and ``quant_method: "modelopt"`` at ``quant_algo: "MIXED_PRECISION"`` with one — the
    TensorRT Model Optimizer form for a checkpoint whose schemes differ by leaf. Any other
    scheme (int quants, no config) → ``None``, and stamping is a no-op.
    """
    qc = _quantization_config(model_dir)
    if qc is None:
        return None
    method = qc.get("quant_method")
    if method == "fp8":
        return qc
    if method == "modelopt" and qc.get("quant_algo") != "MIXED_PRECISION":
        return None  # a pure modelopt scheme names itself in quant_algo; only the mixed one groups
    if method in ("compressed-tensors", "modelopt") and _quantizes_weights_at(qc, 8):
        return qc
    return None


def _declares_nvfp4_weights(qc: dict) -> bool:
    """Whether a ``quantization_config`` MAPPING quantizes ANY weights to NVFP4.

    Three conventions in the wild: ``quant_method: "modelopt"`` with ``quant_algo: "NVFP4"``
    (TensorRT Model Optimizer — the nvidia/* checkpoints); the same method at
    ``quant_algo: "MIXED_PRECISION"`` carrying a 4-bit float weight group; and
    ``quant_method: "compressed-tensors"`` (llm-compressor) whose ``config_groups`` quantize
    weights as 4-bit float over 16-element groups (the ``nvfp4-pack-quantized`` format). The
    group-size condition keeps 32-element-block 4-bit families (MXFP4) out — same e2m1
    values, different scale dtype and block.

    Split from :func:`_fp4_quant_config` for the callers holding the declaration already —
    a serving engine's HF config carries it, and never a checkpoint directory to read.
    """
    method = qc.get("quant_method")
    if method == "modelopt":
        algo = qc.get("quant_algo")
        return algo == "NVFP4" or (algo == "MIXED_PRECISION" and _quantizes_weights_at(qc, 4, 16))
    return method == "compressed-tensors" and _quantizes_weights_at(qc, 4, 16)


def _fp4_quant_config(model_dir: Path) -> dict | None:
    """The checkpoint's ``quantization_config`` when it quantizes ANY weights to NVFP4
    (:func:`_declares_nvfp4_weights` names the declarations that count), else ``None``.

    A MIXED_PRECISION checkpoint answers BOTH this and :func:`_fp8_quant_config`, by design:
    nvidia/Qwen3.6-27B-NVFP4 puts its attention and delta-net projections in fp8 and its MLP and
    lm_head in NVFP4, so both spellers must run over it and each takes the leaves whose stored
    siblings are its own. The two recognizers still decline each other's PURE checkpoints.
    """
    qc = _quantization_config(model_dir)
    return qc if qc is not None and _declares_nvfp4_weights(qc) else None


def _mxfp4_quant_config(model_dir: Path) -> dict | None:
    """The checkpoint's native MXFP4 declaration, or ``None``.

    OpenAI gpt-oss stores routed expert matrices as ``*_blocks`` uint8 tensors (two
    FP4 values per byte) beside ``*_scales`` uint8 E8M0 exponents. Emmy owns that
    storage directly; expanding it through the engine quantizer would defeat the
    expert-streaming memory contract.
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return None
    qc = json.loads(cfg_path.read_text()).get("quantization_config")
    return qc if isinstance(qc, dict) and qc.get("quant_method") == "mxfp4" else None


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


def decode_mxfp4(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Decode native MXFP4 storage to its logical ``(..., in, out)`` matrix.

    ``blocks`` is ``(..., out, groups, 16)``: each byte holds the even value in
    its low nibble and the odd value in its high nibble, so one group contains 32
    values. ``scales`` is ``(..., out, groups)`` and stores the base-2 exponent
    biased by 127. The result swaps the stored output/input axes to match the
    gpt-oss expert wrapper's ``x @ W`` contract.
    """
    blocks = np.asarray(blocks)
    scales = np.asarray(scales)
    if blocks.dtype != np.uint8 or scales.dtype != np.uint8:
        raise ValueError(f"MXFP4 blocks/scales must be uint8, got {blocks.dtype} and {scales.dtype}")
    if blocks.ndim < 3 or blocks.shape[-1] != 16 or scales.shape != blocks.shape[:-1]:
        raise ValueError(f"MXFP4 storage geometry mismatch: blocks={blocks.shape}, scales={scales.shape}")
    nibbles = np.stack((blocks & np.uint8(0xF), blocks >> np.uint8(4)), axis=-1).reshape(*blocks.shape[:-1], 32)
    # MXFP4 shares e2m1's value table with NVFP4; only the block size and the scale format differ.
    decoded = np.ldexp(np.asarray(F4_VALUES)[nibbles], scales.astype(np.int32)[..., None] - 127)
    dense = decoded.reshape(*blocks.shape[:-2], blocks.shape[-2] * 32)
    return np.swapaxes(dense, -2, -1).astype(np.float32)


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
        raise ValueError(f"AWQ qweight/qzeros/scales must be rank-2, got {qweight.shape}, {qzeros.shape}, {scales.shape}")
    k, packed_n = qweight.shape
    groups, zero_packed_n = qzeros.shape
    n = packed_n * 8
    effective_group = k if int(group_size) == -1 else int(group_size)
    if effective_group <= 0 or groups * effective_group != k:
        raise ValueError(f"AWQ group geometry {groups} x {effective_group} does not cover input size {k}")
    if zero_packed_n != packed_n or scales.shape != (groups, n):
        raise ValueError(f"AWQ sibling geometry mismatch: qweight={qweight.shape}, qzeros={qzeros.shape}, scales={scales.shape}")
    integers = unpack_awq4(qweight).astype(np.float32)
    zeros = np.repeat(unpack_awq4(qzeros), effective_group, axis=0).astype(np.float32)
    scale_values = np.repeat(scales.astype(np.float32), effective_group, axis=0)
    return (integers - zeros) * scale_values


def is_nvfp4_checkpoint(model_dir) -> bool:
    """Whether the checkpoint declares the supported NVFP4 packed-trio scheme.

    Same narrow purpose as :func:`is_awq_checkpoint` and :func:`is_exl3_checkpoint`: serving asks
    only whether the dense trunk may stay coded."""
    return _fp4_quant_config(Path(model_dir)) is not None


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
    fp4, fp8 = _fp4_quant_config(model_dir), _fp8_quant_config(model_dir)
    if fp4 is not None and fp8 is not None:
        # Both recognizers answering is the MIXED form, not a contradiction: the checkpoint holds
        # leaves of each scheme, and a boot log that named only one would misreport half the model.
        return f"mixed fp8+nvfp4 {fp4.get('quant_method')}"
    if fp4 is not None:
        return f"nvfp4 {fp4.get('quant_method')}"
    if fp8 is not None:
        return f"fp8 {fp8.get('fmt') or fp8.get('quant_method')}"
    if (qc := _awq_quant_config(model_dir)) is not None:
        return f"awq int{qc.get('bits')} g{qc.get('group_size', qc.get('q_group_size'))} {qc.get('version', 'gemm')}"
    if _mxfp4_quant_config(model_dir) is not None:
        return "mxfp4"
    return "unquantized"


def engine_config_overrides(hf_config) -> dict:
    """HF-config overrides a serving engine needs so it does not try to own weights emmy's
    loader already owns. ``{}`` for an ordinary checkpoint (and for ``None``, the caller's
    "config unreadable").

    EXL3, AWQ, MXFP4 and NVFP4 are owned by Emmy's loader and compiler. Presenting their
    shape-only architecture twin as unquantized prevents the engine from rejecting an otherwise
    supported device, standing up a second quantizer over weights the loader already reads, or
    trying to allocate a second decoded expert table. NVFP4 is recognized by the declaration
    itself (:func:`_declares_nvfp4_weights`), not by ``quant_method`` alone: modelopt spells
    fp8 checkpoints the same way, and those stay the engine's to own. This lives in the loader
    band because naming a checkpoint scheme is frontend-band knowledge."""
    scheme = getattr(hf_config, "quantization_config", None)
    method = scheme.get("quant_method") if isinstance(scheme, dict) else getattr(scheme, "quant_method", None)
    owned = method in {"exl3", "awq", "mxfp4"} or (isinstance(scheme, dict) and _declares_nvfp4_weights(scheme))
    return {"quantization_config": None} if owned else {}


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


def fp8_weight_profile(hf_config) -> tuple[str, tuple[int, int] | None, list[str]] | None:
    """``(fmt, weight block, skip patterns)`` when ``hf_config`` declares fp8 weights, else ``None``.

    Read off ``quantization_config`` BEFORE :func:`strip_engine_quant_config` drops it: the
    storage format token, the 2-D block one scale covers (``None`` = one per-tensor scale) and
    the module patterns the quantizer left unconverted (:func:`_skip_patterns`). A weight-free
    twin derives its scale shapes from this profile and the traced weight shapes alone."""
    qc = getattr(hf_config, "quantization_config", None)
    if qc is None:
        return None
    if not isinstance(qc, dict):
        qc = {key: getattr(qc, key, None) for key in ("quant_method", "fmt", "weight_block_size", *_SKIP_KEYS)}
    if qc.get("quant_method") != "fp8":
        return None
    block = qc.get("weight_block_size")
    fmt = "f8e5m2" if qc.get("fmt") == "e5m2" else "f8e4m3"
    return fmt, (None if block is None else tuple(int(b) for b in block)), _skip_patterns(qc)


def native_mxfp4_experts(hf_config) -> bool:
    """True when the ROUTED EXPERTS are stored as native MXFP4 under the checkpoint's own
    declaration (``expert_dtype: fp4``), whatever the trunk's ``quant_method`` says.

    DeepSeek V4 publishes an fp8 trunk beside fp4 experts, so the expert storage is named by
    this field and by nothing in ``quantization_config``. The serving loader and the twin
    capture both key on it, which is what keeps the recorded expert program the one serving
    binds."""
    return str(getattr(hf_config, "expert_dtype", "") or "") == "fp4"


def mxfp4_weight_profile(hf_config) -> list[str] | None:
    """MXFP4 skip patterns from a shape-only config, or ``None`` for another scheme.

    Two declarations select native MXFP4: ``quant_method: mxfp4`` for a wholly-MXFP4 checkpoint
    (gpt-oss), and :func:`native_mxfp4_experts` for one whose routed experts alone are MXFP4."""
    qc = getattr(hf_config, "quantization_config", None)
    if qc is None:
        return None
    if not isinstance(qc, dict):
        qc = {key: getattr(qc, key, None) for key in ("quant_method", *_SKIP_KEYS)}
    if qc.get("quant_method") == "mxfp4" or native_mxfp4_experts(hf_config):
        return _skip_patterns(qc)
    return None


_SKIP_KEYS = ("ignored_layers", "modules_to_not_convert", "ignore")


def _skip_patterns(qc: dict) -> list[str]:
    """The module patterns a ``quantization_config`` leaves unquantized, every spelling pooled."""
    return [pat for key in _SKIP_KEYS for pat in (qc.get(key) or [])]


def _is_skipped(weight_key: str, patterns: list[str]) -> bool:
    """Whether the weight's module is excluded from quantization.

    ``quant_method: fp8`` lists module names in ``modules_to_not_convert`` (the DeepSeek
    lineage writes ``ignored_layers``); compressed-tensors lists them in ``ignore``, where a
    ``re:`` prefix marks a regex. Plain entries match the module path exactly or as a dotted prefix /
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

    NVFP4 checkpoints: each packed ``<key>`` with the ``<key>_scale`` (e4m3 block
    scales, read as raw bits) + ``<key>_scale_2`` (f32) sibling pair dequantizes via
    :func:`dequantize_nvfp4`; both consumed scales are dropped. Activation-quant
    metadata (``input_scale``, kv-cache scales) passes through unconsumed — it
    belongs to the serving path, not the twin.

    Native MXFP4 checkpoints: each routed expert's ``<projection>_blocks`` /
    ``<projection>_scales`` pair decodes to the logical ``<projection>`` tensor in the
    architecture twin's ``(experts, in, out)`` orientation. The consumed storage tensors
    are dropped.

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
    qc4 = _fp4_quant_config(model_dir)
    mxfp4 = _mxfp4_quant_config(model_dir) is not None
    exl3 = _exl3_quant_config(model_dir) is not None
    awq = _awq_quant_config(model_dir)
    patterns = _skip_patterns(qc) if qc else []
    patterns4 = list(qc4.get("ignore") or []) if qc4 else []
    # NVFP4 trio signature: packed <key> + <key>_scale (e4m3) + <key>_scale_2 (f32).
    # The e4m3 block scales must arrive as raw bits — dequantize_nvfp4 owns the decode.
    fp4_scale_keys = (
        frozenset(k for k in index if k.endswith("_scale") and k + "_2" in index and k[: -len("_scale")] in index)
        if qc4 is not None
        else frozenset()
    )

    by_shard: dict[str, list[str]] = {}
    for key, shard in index.items():
        by_shard.setdefault(str(shard), []).append(key)
    sources: dict[str, np.ndarray] = {}
    fp8_keys: dict[str, str] = {}
    for shard_path, keys in by_shard.items():
        fp8_keys.update(_read_shard(shard_path, keys, sources, bits_keys=fp4_scale_keys))

    out: dict[str, np.ndarray] = {}
    consumed: set[str] = set()
    for key in index:
        if mxfp4 and key.endswith("_blocks"):
            base = key[: -len("_blocks")]
            scales_key = base + "_scales"
            if scales_key not in index:
                raise ValueError(f"MXFP4 tensor {key!r} is missing scales {scales_key!r}")
            out[base] = decode_mxfp4(sources[key], sources[scales_key])
            consumed |= {key, scales_key}
            continue
        if mxfp4 and key.endswith("_scales") and key[: -len("_scales")] + "_blocks" in index:
            consumed.add(key)
            continue
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
        if qc4 is not None and key + "_scale" in fp4_scale_keys and not _is_skipped(key, patterns4):
            if sources[key].dtype == np.uint8:
                out[key] = dequantize_nvfp4(sources[key], sources[key + "_scale"], sources[key + "_scale_2"])
                consumed |= {key + "_scale", key + "_scale_2"}
                continue
        if awq is not None and key.endswith((".qzeros", ".scales")):
            base = key.rsplit(".", 1)[0]
            if base + ".qweight" in index:
                consumed.add(key)
                continue
        if qc is not None and key in fp8_keys and not _is_skipped(key, patterns):
            scale_key = next((k for k in (key + "_scale", key + "_scale_inv") if k in index), None)
            if scale_key is not None:
                out[key] = dequantize(sources[key], sources[scale_key], inverse=scale_is_reciprocal(scale_key))
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
        inverse=scale_is_reciprocal(scale_key),
        grid=grid,
        block=block,
        degenerate=degenerate,
    )
    frag.outputs = [final]
    graph.splice(frag, consumed=[nid], output=nid)
    return True


def _f4_pair_table(graph: Graph, *, name: str, out_name: str, dtype) -> str:
    """Add the 256x2 e2m1 pair-value table to ``graph`` as a source-free COMPUTED constant
    (the trellis-codebook mechanism — ``ConstantOp.value`` carries scalars only) and return its
    node id: row ``b`` holds byte ``b``'s low and high e2m1 values, concatenated along a trailing
    axis of 2. Shared by the weight speller and the static activation speller, so both decode
    chains read one table shape."""
    from emmy.compiler.ir.expr import Literal, placeholder  # noqa: PLC0415
    from emmy.compiler.ir.tensor.ir import ElementwiseOp, IndexMapOp, IndexSource, RangeOp  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    tg = Graph()
    codes = tg.add_node(op=RangeOp(stop=256, dtype="i32"), inputs=[], output=Tensor("codes", (256,), "i32"), node_id="codes")
    mask = tg.add_node(op=ConstantOp(name="mask", value=15), inputs=[], output=Tensor("mask", (1,), "i32"))
    shift = tg.add_node(op=ConstantOp(name="shift", value=4), inputs=[], output=Tensor("shift", (1,), "i32"))
    low = tg.add_node(
        op=ElementwiseOp(op="bitwise_and"),
        inputs=[codes, broadcast_to(tg, mask, (256,))],
        output=Tensor("low", (256,), "i32"),
    )
    high = tg.add_node(
        op=ElementwiseOp(op="right_shift"),
        inputs=[codes, broadcast_to(tg, shift, (256,))],
        output=Tensor("high", (256,), "i32"),
    )
    vlow = tg.add_node(op=ElementwiseOp(op="from_f4e2m1"), inputs=[low], output=Tensor("vlow", (256,), "f32"))
    vhigh = tg.add_node(op=ElementwiseOp(op="from_f4e2m1"), inputs=[high], output=Tensor("vhigh", (256,), "f32"))
    pairs_t = tg.add_node(
        op=IndexMapOp(
            out_shape=(256, 2),
            sources=(
                IndexSource(input_idx=0, coord_map=(placeholder(0),), select=placeholder(1).lt(Literal(1, "int"))),
                IndexSource(input_idx=1, coord_map=(placeholder(0),)),
            ),
        ),
        inputs=[vlow, vhigh],
        output=Tensor("pairs", (256, 2), "f32"),
    )
    if dtype.name != "f32":
        pairs_t = tg.add_node(op=ElementwiseOp(op="copy"), inputs=[pairs_t], output=Tensor("pairs_cast", (256, 2), dtype))
    tg.outputs = [pairs_t]
    return graph.add_node(
        op=ConstantOp(name=name, source_graph=tg, source_shape=(256, 2), source_dtype=dtype),
        inputs=[],
        output=Tensor(out_name, (256, 2), dtype),
    )


def _spell_fp4_one(
    graph: Graph,
    nid: str,
    *,
    scale_key: str,
    packed_shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
    s2_shape: tuple[int, ...],
) -> bool:
    """Rewrite the NVFP4 weight constant ``nid`` into packed-bits + fused-scale decode algebra.

    The unpack is spelled as a pair-table gather — an embedding lookup: bits ``[…, K/2]``
    (``f4e2m1x2``) → i32 → gather into a 256×2 value-carried table (row ``b`` holds both
    e2m1 values of byte ``b``) → reshape to the promised ``[…, K]``. Chosen over in-graph
    bit ops (mask/shift on each byte plus an interleave of the two halves), which would
    say the same thing in ~3× the nodes; the graph term never dictates the emitted kernel
    code — the lowering recognizes the term — so the smaller term wins. The scale side
    decodes the e4m3 block scales, multiplies by ``scale_2``, and rounds ONCE to f16 (the
    same single rounding as :func:`fuse_nvfp4_scales`), then a per-16 block multiply
    applies it along the last axis. The cone's output keeps exactly the dtype/shape the
    trace promised, so every later pass is unaffected. Checkpoint shapes that do not
    match the packing leave the constant alone (``False``) — never a compile error.
    """
    from emmy.compiler.ir.frontend.ir import ReshapeOp  # noqa: PLC0415
    from emmy.compiler.ir.tensor.ir import ElementwiseOp, GatherOp  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    node = graph.nodes[nid]
    op, out = node.op, node.output
    if op.load_ops or op.source_parts or op.value is not None:
        return False  # only a pristine single-source trace constant is spelled (birth time — nothing has run yet)
    if any(not d.is_static for d in out.shape):
        return False
    shape = tuple(d.as_static() for d in out.shape)
    k = shape[-1]
    expected = (shape[:-1] + (k // 2,), shape[:-1] + (k // 16,))
    if k % 16 or (packed_shape, scale_shape) != expected or int(np.prod(s2_shape) if s2_shape else 1) != 1:
        logger.warning(
            "NVFP4 weight %s: stored shapes packed=%s scale=%s scale_2=%s do not match logical %s; constant left alone",
            nid,
            packed_shape,
            scale_shape,
            s2_shape,
            shape,
        )
        return False

    frag = Graph()
    bits = frag.add_node(
        op=ConstantOp(name=op.name, source_path=op.source_path, source_shape=packed_shape, source_dtype=F4E2M1x2.name),
        inputs=[],
        output=Tensor(f"{out.name}_bits", packed_shape, F4E2M1x2.name),
    )
    idx = frag.add_node(op=ElementwiseOp(op="copy"), inputs=[bits], output=Tensor(f"{out.name}_idx", packed_shape, "i32"))
    table = _f4_pair_table(frag, name=f"{op.name}_f4_pairs", out_name=f"{out.name}_f4_pairs", dtype=out.dtype)
    pairs = frag.add_node(op=GatherOp(axis=0), inputs=[table, idx], output=Tensor(f"{out.name}_pairs", packed_shape + (2,), out.dtype))
    vals = frag.add_node(op=ReshapeOp(shape=shape), inputs=[pairs], output=Tensor(f"{out.name}_vals", shape, out.dtype))

    s2_decl = (1,) * len(scale_shape)
    s_bits = frag.add_node(
        op=ConstantOp(name=f"{op.name}_scale", source_path=scale_key, source_shape=scale_shape, source_dtype=F8E4M3.name),
        inputs=[],
        output=Tensor(f"{out.name}_scale_bits", scale_shape, F8E4M3.name),
    )
    s_vals = frag.add_node(
        op=ElementwiseOp(op=f"from_{F8E4M3.name}"), inputs=[s_bits], output=Tensor(f"{out.name}_scale_vals", scale_shape, "f32")
    )
    s2 = frag.add_node(
        op=ConstantOp(
            name=f"{op.name}_scale_2",
            source_path=scale_key + "_2",
            source_shape=s2_shape,
            source_dtype="f32",
            load_ops=() if tuple(s2_shape) == s2_decl else (ReshapeOp(shape=s2_decl),),
        ),
        inputs=[],
        output=Tensor(f"{out.name}_scale_2", s2_decl, "f32"),
    )
    s2_bc = broadcast_to(frag, s2, scale_shape)
    fused32 = frag.add_node(
        op=ElementwiseOp(op="multiply"), inputs=[s_vals, s2_bc], output=Tensor(f"{out.name}_fused32", scale_shape, "f32")
    )
    # The format's single rounding point: the f32 product rounds once to f16 (fuse_nvfp4_scales parity).
    fused = frag.add_node(op=ElementwiseOp(op="copy"), inputs=[fused32], output=Tensor(f"{out.name}_fused", scale_shape, "f16"))
    if out.dtype.name != "f16":
        fused = frag.add_node(op=ElementwiseOp(op="copy"), inputs=[fused], output=Tensor(f"{out.name}_fused_cast", scale_shape, out.dtype))

    interleaved = shape[:-1] + (k // 16, 16)
    blk = frag.add_node(op=ReshapeOp(shape=interleaved), inputs=[vals], output=Tensor(f"{out.name}_blk", interleaved, out.dtype))
    fused_r = frag.add_node(
        op=ReshapeOp(shape=scale_shape + (1,)),
        inputs=[fused],
        output=Tensor(f"{out.name}_fused_r", scale_shape + (1,), out.dtype),
    )
    f_bc = broadcast_to(frag, fused_r, interleaved)
    scaled = frag.add_node(op=ElementwiseOp(op="multiply"), inputs=[blk, f_bc], output=Tensor(f"{out.name}_sblk", interleaved, out.dtype))
    final = frag.add_node(op=ReshapeOp(shape=shape), inputs=[scaled], output=Tensor(out.name, shape, out.dtype))
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
        op=ConstantOp(name=f"{op.name}_scales", source_path=base + ".scales", source_shape=scales_shape, source_dtype=scale_dtype),
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
    qc4 = _fp4_quant_config(model_dir)
    if qc is None and qc4 is None:
        return 0
    patterns = _skip_patterns(qc) if qc else []
    patterns4 = list(qc4.get("ignore") or []) if qc4 else []
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
            if op.source_path is None or op.source_dtype in F8_SAFETENSORS_DTYPES.values() or op.source_dtype == F4E2M1x2.name:
                continue  # source-less, or an already-spelled bits constant (idempotency)
            key = next((c for c in _candidate_keys(op.source_path) if c in index), None)
            if key is None:
                continue
            # NVFP4 trio signature (packed U8 weight + e4m3 block scales + f32 tensor scale) —
            # checked before the fp8 pairing, whose <key>_scale rule it would otherwise shadow.
            if qc4 is not None and key + "_scale" in index and key + "_scale_2" in index and not _is_skipped(key, patterns4):
                if _slice(key).get_dtype() == "U8" and _spell_fp4_one(
                    graph,
                    nid,
                    scale_key=key + "_scale",
                    packed_shape=tuple(int(d) for d in _slice(key).get_shape()),
                    scale_shape=tuple(int(d) for d in _slice(key + "_scale").get_shape()),
                    s2_shape=tuple(int(d) for d in _slice(key + "_scale_2").get_shape()),
                ):
                    spelled += 1
                continue
            if qc is None or _is_skipped(key, patterns):
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


def _cone_nodes(graph: Graph, start: str):
    """Every node upstream of buffer ``start``, each yielded once.

    The one walk the cone readings below share; each applies its own predicate per node, and the
    ones answering a yes/no question stop the walk early by breaking out of the generator."""
    pending, seen = [start], set()
    while pending:
        node = graph.producer(pending.pop())
        if node is None or node.id in seen:
            continue
        seen.add(node.id)
        yield node
        pending.extend(node.inputs)


def _cone_storage_formats(graph: Graph, start: str) -> set[str]:
    """Return FP8 storage dtypes reachable upstream from one graph buffer."""
    return {
        name
        for node in _cone_nodes(graph, start)
        if isinstance(node.op, ConstantOp) and (name := node.output.dtype.name) in F8_SAFETENSORS_DTYPES.values()
    }


def _cone_has_fp8_encode(graph: Graph, start: str) -> bool:
    """Whether an activation buffer is already downstream of an FP8 encode."""
    from emmy.compiler.ir.tensor.ir import ElementwiseOp  # noqa: PLC0415

    return any(isinstance(n.op, ElementwiseOp) and n.op.op.name.startswith("to_f8") for n in _cone_nodes(graph, start))


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


def _static_fp4_activation_declared(qc: dict) -> bool:
    """Whether an NVFP4 checkpoint declares STATIC 4-bit float input activations.

    modelopt 0.35+ and llm-compressor write a ``config_groups`` entry whose ``input_activations``
    is 4-bit static float over 16-element groups; older modelopt configs carry no
    ``config_groups`` at all, and there the per-linear ``input_scale`` tensors are the marker
    (checked per linear, not here). A dynamic declaration declines — the static per-tensor
    level is what this path's checkpoints calibrate."""
    groups = qc.get("config_groups")
    if not groups:
        return qc.get("quant_method") == "modelopt"
    for group in groups.values():
        acts = group.get("input_activations") if isinstance(group, dict) else None
        if (
            isinstance(acts, dict)
            and acts.get("type") == "float"
            and int(acts.get("num_bits") or 0) == 4
            and not acts.get("dynamic")
            and int(acts.get("group_size") or NVFP4_BLOCK) == NVFP4_BLOCK
        ):
            return True
    return False


def _cone_fp4_weight_base(graph: Graph, start: str) -> str | None:
    """The checkpoint module base (the ``.weight`` key minus its suffix) of the ONE packed NVFP4
    weight constant upstream of buffer ``start``, or ``None`` — no packed weight, several, or a
    source path outside the ``<module>.weight`` pairing that ``input_scale`` siblings follow."""
    paths = {
        node.op.source_path
        for node in _cone_nodes(graph, start)
        if isinstance(node.op, ConstantOp) and node.output.dtype.name == F4E2M1x2.name and node.op.source_path
    }
    if len(paths) != 1:
        return None
    path = next(iter(paths))
    return path.removesuffix(".weight") if path.endswith(".weight") else None


def _cone_has_fp4_encode(graph: Graph, start: str) -> bool:
    """Whether an activation buffer is already downstream of an e2m1 encode (idempotency)."""
    from emmy.compiler.ir.tensor.ir import ElementwiseOp  # noqa: PLC0415

    return any(isinstance(n.op, ElementwiseOp) and n.op.op.name == "to_f4e2m1" for n in _cone_nodes(graph, start))


def _shape_extents(shape) -> tuple[int | str, ...]:
    """A buffer's shape as the ``int | str`` extents an op's shape field takes — static dims as
    ints, symbolic ones by name.

    NVFP4 packs along the LAST axis only: two codes per byte over K, one block scale per 16
    elements of K. Both halves of the round trip below therefore do real int arithmetic on the
    last extent — the pair pack's ``k // 2``, the block split's ``k // 16``. That extent is always
    an int: K comes from the packed weight. Every LEADING axis just rides along through the
    elementwise / gather / reshape chain, so it may be the serving trace's symbolic ``num_tokens``
    and has to survive as such: resolving those too raises on the any-width program serving
    compiles beside its static token buckets."""
    return tuple(d.as_static() if d.is_static else str(d) for d in shape)


def _spell_static_fp4_quantize(
    graph: Graph, activation: str, scale_key: str, s2_shape: tuple[int, ...]
) -> tuple[str, str, str, str] | None:
    """Spell the QUANTIZE half of one static 4-bit activation round trip — the half every
    consumer of that activation shares. Returns ``(stem, packed codes, e4m3 block scales,
    ``input_scale``)``, or ``None`` when the activation's shape cannot carry it (symbolic or
    non-16-multiple K, a non-scalar ``input_scale``).

    The algebra is :func:`quantize_nvfp4` with the checkpoint's static per-linear ``input_scale``
    standing in for the tensor-derived ``scale_2``: per 16-element K block, the e4m3 block-scale
    round trip (``to_f8e4m3(amax / (6·s2))``), ONE f32→f16 rounding of the fused scale
    (:func:`fuse_nvfp4_scales` parity), the e2m1 encode of the block over the rounded fused
    scale, and the pair pack into an ``f4e2m1x2`` buffer. The quantize's divisor is floored at
    1e-12 so an all-zero block divides by the floor instead of by zero; its codes are zeros
    either way, and the decode multiplies by the unfloored scale.

    What the consumers share is the CODES and their raw block scales, never a reconstructed
    value: the reconstruction is spelled per consumer (:func:`_spell_static_fp4_decode`). Loop
    fusion materializes an activation's fan-out point, so this split is what decides that a
    shared activation reaches its matmuls as the packed pair beside e4m3 scales — the two leaves
    a packed weight constant already offers — instead of as a 16-bit dense buffer with the codes
    dissolved into the producer."""
    from emmy.compiler.ir.expr import Literal, placeholder  # noqa: PLC0415
    from emmy.compiler.ir.frontend.ir import ReshapeOp  # noqa: PLC0415
    from emmy.compiler.ir.tensor.ir import ElementwiseOp, IndexMapOp, IndexSource, ReduceOp  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    source = graph.buffer(activation)
    if source is None or not source.shape or source.dtype.name not in {"f16", "bf16", "f32"}:
        return None
    kd = source.shape[-1]
    if not kd.is_static or kd.as_static() % NVFP4_BLOCK or int(np.prod(s2_shape) if s2_shape else 1) != 1:
        return None
    k = kd.as_static()
    lead = _shape_extents(source.shape[:-1])
    blocked = (*lead, k // NVFP4_BLOCK, NVFP4_BLOCK)
    bshape = (*lead, k // NVFP4_BLOCK, 1)
    flat, half = (*lead, k), (*lead, k // 2)
    # Freshness must be probed on a DERIVED name: the stem itself never becomes a node, and
    # ``add_node`` silently falls back to an ``n<i>`` node id on a taken name while keeping the
    # duplicate TENSOR name — two buffers sharing one name then cross-read at the kernel level,
    # where references are by name. ``_bits`` exists in every quantize half, so it is the probe.
    stem, suffix = f"{activation}_static_fp4", 2
    while f"{stem}_bits" in graph.nodes or graph.buffer(f"{stem}_bits") is not None:
        stem = f"{activation}_static_fp4_{suffix}"
        suffix += 1
    dt = source.dtype

    blk = graph.add_node(op=ReshapeOp(shape=blocked), inputs=[activation], output=Tensor(f"{stem}_blk", blocked, dt))
    absolute = graph.add_node(op=ElementwiseOp(op="abs"), inputs=[blk], output=Tensor(f"{stem}_abs", blocked, "f32"))
    amax = graph.add_node(op=ReduceOp(op="maximum", axis=-1), inputs=[absolute], output=Tensor(f"{stem}_amax", bshape, "f32"))
    s2 = graph.add_node(
        op=ConstantOp(
            name=f"{stem}_scale_2",
            source_path=scale_key,
            source_shape=tuple(s2_shape),
            source_dtype="f32",
            load_ops=() if tuple(s2_shape) == (1,) * len(bshape) else (ReshapeOp(shape=(1,) * len(bshape)),),
        ),
        inputs=[],
        output=Tensor(f"{stem}_scale_2", (1,) * len(bshape), "f32"),
    )
    s2_bc = broadcast_to(graph, s2, bshape)
    fmax = const_bc(graph, name=f"{stem}_f4_max", value=_F4_MAX, target_shape=bshape, dtype="f32")
    denom = graph.add_node(op=ElementwiseOp(op="multiply"), inputs=[fmax, s2_bc], output=Tensor(f"{stem}_denom", bshape, "f32"))
    ratio = graph.add_node(op=ElementwiseOp(op="divide"), inputs=[amax, denom], output=Tensor(f"{stem}_ratio", bshape, "f32"))
    sbits = graph.add_node(op=ElementwiseOp(op="to_f8e4m3"), inputs=[ratio], output=Tensor(f"{stem}_scale_bits", bshape, F8E4M3.name))
    sdec = graph.add_node(op=ElementwiseOp(op=f"from_{F8E4M3.name}"), inputs=[sbits], output=Tensor(f"{stem}_scale_vals", bshape, "f32"))
    fused32 = graph.add_node(op=ElementwiseOp(op="multiply"), inputs=[sdec, s2_bc], output=Tensor(f"{stem}_fused32", bshape, "f32"))
    fused = graph.add_node(op=ElementwiseOp(op="copy"), inputs=[fused32], output=Tensor(f"{stem}_fused", bshape, "f16"))
    div32 = graph.add_node(op=ElementwiseOp(op="copy"), inputs=[fused], output=Tensor(f"{stem}_div32", bshape, "f32"))
    floor = const_bc(graph, name=f"{stem}_floor", value=1.0e-12, target_shape=bshape, dtype="f32")
    safe = graph.add_node(op=ElementwiseOp(op="maximum"), inputs=[div32, floor], output=Tensor(f"{stem}_safe", bshape, "f32"))
    safe_bc = broadcast_to(graph, safe, blocked)
    quot = graph.add_node(op=ElementwiseOp(op="divide"), inputs=[blk, safe_bc], output=Tensor(f"{stem}_norm", blocked, "f32"))
    codes = graph.add_node(op=ElementwiseOp(op="to_f4e2m1"), inputs=[quot], output=Tensor(f"{stem}_codes", blocked, "i32"))
    codes_f = graph.add_node(op=ReshapeOp(shape=flat), inputs=[codes], output=Tensor(f"{stem}_codes_flat", flat, "i32"))
    d = len(flat) - 1
    lead_ph = tuple(placeholder(i) for i in range(d))
    even = graph.add_node(
        op=IndexMapOp(out_shape=half, sources=(IndexSource(input_idx=0, coord_map=(*lead_ph, placeholder(d) * Literal(2, "int"))),)),
        inputs=[codes_f],
        output=Tensor(f"{stem}_even", half, "i32"),
    )
    odd = graph.add_node(
        op=IndexMapOp(
            out_shape=half,
            sources=(IndexSource(input_idx=0, coord_map=(*lead_ph, placeholder(d) * Literal(2, "int") + Literal(1, "int"))),),
        ),
        inputs=[codes_f],
        output=Tensor(f"{stem}_odd", half, "i32"),
    )
    four_bc = const_bc(graph, name=f"{stem}_shift", value=4, target_shape=half, dtype="i32")
    hi = graph.add_node(op=ElementwiseOp(op="left_shift"), inputs=[odd, four_bc], output=Tensor(f"{stem}_hi", half, "i32"))
    byte = graph.add_node(op=ElementwiseOp(op="bitwise_or"), inputs=[even, hi], output=Tensor(f"{stem}_byte", half, "i32"))
    bits = graph.add_node(op=ElementwiseOp(op="copy"), inputs=[byte], output=Tensor(f"{stem}_bits", half, F4E2M1x2.name))
    return stem, bits, sbits, s2


def _spell_static_fp4_decode(graph: Graph, quant: tuple[str, str, str, str], dtype) -> str:
    """Spell ONE consumer's reconstruction over a shared quantized activation and return its
    restored buffer.

    The shape is the packed weight constant's own decode chain (:func:`_spell_fp4_one`): a
    pair-table gather over the packed codes, times the block scale, with the two scale levels
    multiplied in f32 and rounded once to f16 (:func:`fuse_nvfp4_scales` parity). Spelled per
    consumer rather than shared, so each marked matmul carries both operands in that one
    decode-chain reading — the packed pair and the raw e4m3 scale each reached through a load of
    their own."""
    from emmy.compiler.ir.frontend.ir import ReshapeOp  # noqa: PLC0415
    from emmy.compiler.ir.tensor.ir import ElementwiseOp, GatherOp  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    quant_stem, bits, sbits, s2 = quant
    half = _shape_extents(graph.buffer(bits).shape)
    bshape = _shape_extents(graph.buffer(sbits).shape)
    flat, blocked = (*half[:-1], half[-1] * 2), (*bshape[:-1], NVFP4_BLOCK)
    # A reconstruction's own stem, never the quantize half's: the two spell some of the same
    # derived names (``_scale_vals``, ``_fused``), and ``add_node`` answers a taken name by
    # keeping the duplicate TENSOR name while renaming only the node — buffers that share a name
    # then cross-read at the kernel level, where references are by name.
    suffix = 1
    while f"{quant_stem}_r{suffix}_value" in graph.nodes or graph.buffer(f"{quant_stem}_r{suffix}_value") is not None:
        suffix += 1
    stem = f"{quant_stem}_r{suffix}"

    idx = graph.add_node(op=ElementwiseOp(op="copy"), inputs=[bits], output=Tensor(f"{stem}_idx", half, "i32"))
    table = _f4_pair_table(graph, name=f"{stem}_f4_pairs", out_name=f"{stem}_f4_pairs", dtype=dtype)
    pairs = graph.add_node(op=GatherOp(axis=0), inputs=[table, idx], output=Tensor(f"{stem}_pairs", (*half, 2), dtype))
    vblk = graph.add_node(op=ReshapeOp(shape=blocked), inputs=[pairs], output=Tensor(f"{stem}_vals", blocked, dtype))
    sdec = graph.add_node(op=ElementwiseOp(op=f"from_{F8E4M3.name}"), inputs=[sbits], output=Tensor(f"{stem}_scale_vals", bshape, "f32"))
    s2_bc = broadcast_to(graph, s2, bshape)
    fused32 = graph.add_node(op=ElementwiseOp(op="multiply"), inputs=[sdec, s2_bc], output=Tensor(f"{stem}_fused32", bshape, "f32"))
    # The format's single rounding point: the f32 product rounds once to f16 (fuse_nvfp4_scales parity).
    fused = graph.add_node(op=ElementwiseOp(op="copy"), inputs=[fused32], output=Tensor(f"{stem}_fused", bshape, "f16"))
    if dtype.name != "f16":
        fused = graph.add_node(op=ElementwiseOp(op="copy"), inputs=[fused], output=Tensor(f"{stem}_fused_cast", bshape, dtype))
    f_bc = broadcast_to(graph, fused, blocked)
    scaled = graph.add_node(op=ElementwiseOp(op="multiply"), inputs=[vblk, f_bc], output=Tensor(f"{stem}_sblk", blocked, dtype))
    return graph.add_node(op=ReshapeOp(shape=flat), inputs=[scaled], output=Tensor(f"{stem}_value", flat, dtype))


def spell_static_fp4_activations(graph: Graph, model_id_or_path: str) -> int:
    """Spell checkpoint-declared STATIC 4-bit activation quantization in front of NVFP4-marked
    linears — the activation half of the declared W4A4 program.

    A linear is marked when its already-spelled weight cone holds a packed NVFP4 constant AND the
    checkpoint stores that module's ``input_scale`` — modelopt's calibrated per-linear activation
    ``scale_2`` (one f32, ``calibration amax / (6 · 448)``). The graph's own meaning then becomes
    Σ x̂·ŵ program-wide, and the numpy backend stays the parity oracle. Unmarked linears keep
    their 16-bit activations untouched. The round trip is spelled in two halves: consumers
    reading one activation through equal-valued ``input_scale`` tensors share ONE quantize (the
    checkpoint calibrates a fused projection group — q/k/v, gate/up — to one scale, stored once
    per member; unequal values quantize per scale path), and each consumer then gets its own
    reconstruction over the shared codes. Returns the number of rewired linears; weight-only or
    non-NVFP4 checkpoints are a no-op. Runs after :func:`spell_quantized_constants`, whose
    spelled weight cones are the marker it reads."""
    from safetensors import safe_open  # noqa: PLC0415

    from emmy.compiler.ir.frontend.ir import LinearOp  # noqa: PLC0415
    from emmy.compiler.loader.safetensors import _build_index, _resolve_model_dir  # noqa: PLC0415

    model_dir = _resolve_model_dir(model_id_or_path)
    qc4 = _fp4_quant_config(model_dir)
    if qc4 is None or not _static_fp4_activation_declared(qc4):
        return 0
    index = _build_index(model_dir)

    eligible: list[tuple[str, str, str]] = []
    for node in list(graph.nodes.values()):
        if not isinstance(node.op, LinearOp) or len(node.inputs) < 2:
            continue
        activation, weight = node.inputs[:2]
        base = _cone_fp4_weight_base(graph, weight)
        if base is None or _cone_has_fp4_encode(graph, activation):
            continue
        scale_key = base + ".input_scale"
        if scale_key in index:
            eligible.append((node.id, activation, scale_key))
    if not eligible:
        return 0

    spelled = 0
    with ExitStack() as stack:
        handles: dict[str, object] = {}

        def _open(key: str):
            path = str(index[key])
            handle = handles.get(path)
            if handle is None:
                handle = handles[path] = stack.enter_context(safe_open(path, framework="numpy"))
            return handle

        quantized: dict[tuple[str, float], tuple[str, str, str, str] | None] = {}
        for linear_id, activation, scale_key in eligible:
            value = float(np.asarray(_open(scale_key).get_tensor(scale_key), dtype=np.float32).reshape(-1)[0])
            key = (activation, value)
            if key not in quantized:
                s2_shape = tuple(int(x) for x in _open(scale_key).get_slice(scale_key).get_shape())
                quantized[key] = _spell_static_fp4_quantize(graph, activation, scale_key, s2_shape)
                if quantized[key] is None:
                    logger.warning("static fp4 activation %s: shape cannot carry the block quantize; linear stays 16-bit", scale_key)
            quant = quantized[key]
            if quant is None:
                continue
            graph.replace_input(linear_id, activation, _spell_static_fp4_decode(graph, quant, graph.buffer(activation).dtype))
            spelled += 1
    if spelled:
        logger.info("spelled static 4-bit activation algebra for %d linear(s) from %s", spelled, model_dir)
    return spelled


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


def spell_mxfp4_inputs(
    graph: Graph,
    specs: dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    transposed: bool,
) -> dict[str, str]:
    """Spell logical expert inputs as native MXFP4 blocks plus E8M0 scales.

    ``specs[name]`` is ``(blocks_shape, scales_shape)`` for one expert slice. The re-minted
    ``name`` input binds ``(out, in/32, 16)`` uint8 blocks and the appended ``<name>_scale``
    input binds ``(out, in/32)`` uint8 scales; the nibbles always decode in that stored
    ``(out, in)`` orientation.

    ``transposed`` is the experts MODULE's weight layout
    (:func:`~emmy.compiler.trace.huggingface.moe_expert_layout`), one fact for the whole call
    rather than per input: ``True`` when the traced placeholder is the ``(in, out)`` matrix
    applied as ``x @ W``, so the decode ends in a transpose, and ``False`` for the
    ``F.linear`` ``(out, in)`` orientation the decode already lands in. It is declared rather
    than read off the shapes because a square expert matrix fits both readings, and the wrong
    one transposes the weights silently.

    Generic tensor algebra decodes the nibbles and scale exponents in-graph, so lowering can
    fuse those operations into the ordinary matrix multiplication instead of materializing a
    persistent dense expert table.
    """
    from emmy.compiler.ir.base import InputOp  # noqa: PLC0415
    from emmy.compiler.ir.frontend.ir import ReshapeOp, TransposeOp  # noqa: PLC0415
    from emmy.compiler.ir.tensor.ir import ElementwiseOp, RangeOp  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    out_map: dict[str, str] = {}
    for name, (blocks_shape, scales_shape) in specs.items():
        node = graph.nodes.get(name)
        if node is None or not isinstance(node.op, InputOp) or name not in graph.inputs:
            raise ValueError(f"spell_mxfp4_inputs: {name!r} is not a graph input")
        out = node.output
        if any(not d.is_static for d in out.shape):
            raise ValueError(f"spell_mxfp4_inputs: input {name!r} has symbolic dims; only static weights are supported")
        logical = tuple(d.as_static() for d in out.shape)
        blocks_shape, scales_shape = tuple(blocks_shape), tuple(scales_shape)
        if len(logical) != 2:
            raise ValueError(f"spell_mxfp4_inputs: input {name!r} must be rank-2, got {logical}")
        k, n = logical if transposed else logical[::-1]
        expected_blocks = (n, k // 32, 16) if k % 32 == 0 else None
        expected_scales = (n, k // 32) if k % 32 == 0 else None
        if blocks_shape != expected_blocks or scales_shape != expected_scales:
            raise ValueError(
                f"spell_mxfp4_inputs: {name!r} storage blocks={blocks_shape}, scales={scales_shape} do not reproduce logical {logical}"
            )

        tmp = f"{name}__dq_src"
        graph.rename_node(name, tmp)
        blocks = graph.add_node(op=InputOp(), inputs=[], output=Tensor(name, blocks_shape, "u8"), node_id=name)
        graph.inputs = [blocks if i == tmp else i for i in graph.inputs]
        scale_name = f"{name}_scale"
        scales = graph.add_node(
            op=InputOp(),
            inputs=[],
            output=Tensor(scale_name, scales_shape, "u8"),
            node_id=scale_name,
        )
        graph.inputs.append(scales)

        expanded = (n, k // 32, 16, 2)
        values_shape = (n, k // 32, 32)
        blocks_i32 = graph.add_node(
            op=ElementwiseOp(op="copy"),
            inputs=[blocks],
            output=Tensor(f"{name}_blocks_i32", blocks_shape, "i32"),
        )
        blocks_view = graph.add_node(
            op=ReshapeOp(shape=(n, k // 32, 16, 1)),
            inputs=[blocks_i32],
            output=Tensor(f"{name}_blocks_view", (n, k // 32, 16, 1), "i32"),
        )
        blocks_bc = broadcast_to(graph, blocks_view, expanded)
        lanes = graph.add_node(
            op=RangeOp(start=0, stop=2, step=1, dtype="i32"),
            inputs=[],
            output=Tensor(f"{name}_lanes", (2,), "i32"),
        )
        lanes = graph.add_node(
            op=ReshapeOp(shape=(1, 1, 1, 2)),
            inputs=[lanes],
            output=Tensor(f"{name}_lanes_view", (1, 1, 1, 2), "i32"),
        )
        lanes = broadcast_to(graph, lanes, expanded)
        four = const_bc(graph, name=f"{name}_four", value=4, target_shape=expanded, dtype="i32")
        shifts = graph.add_node(
            op=ElementwiseOp(op="multiply"),
            inputs=[lanes, four],
            output=Tensor(f"{name}_shifts", expanded, "i32"),
        )
        shifted = graph.add_node(
            op=ElementwiseOp(op="right_shift"),
            inputs=[blocks_bc, shifts],
            output=Tensor(f"{name}_shifted", expanded, "i32"),
        )
        mask15 = const_bc(graph, name=f"{name}_mask15", value=15, target_shape=expanded, dtype="i32")
        nibbles = graph.add_node(
            op=ElementwiseOp(op="bitwise_and"),
            inputs=[shifted, mask15],
            output=Tensor(f"{name}_nibbles4", expanded, "i32"),
        )
        nibbles = graph.add_node(
            op=ReshapeOp(shape=values_shape),
            inputs=[nibbles],
            output=Tensor(f"{name}_nibbles", values_shape, "i32"),
        )

        mask7 = const_bc(graph, name=f"{name}_mask7", value=7, target_shape=values_shape, dtype="i32")
        magnitude_i32 = graph.add_node(
            op=ElementwiseOp(op="bitwise_and"),
            inputs=[nibbles, mask7],
            output=Tensor(f"{name}_magnitude_i32", values_shape, "i32"),
        )
        magnitude = graph.add_node(
            op=ElementwiseOp(op="copy"),
            inputs=[magnitude_i32],
            output=Tensor(f"{name}_magnitude", values_shape, "f32"),
        )

        def constant(label: str, value: float, *, _name=name, _shape=values_shape) -> str:
            return const_bc(graph, name=f"{_name}_{label}", value=value, target_shape=_shape, dtype="f32")

        def binary(label: str, op: str, left: str, right: str, *, _name=name, _shape=values_shape) -> str:
            return graph.add_node(
                op=ElementwiseOp(op=op),
                inputs=[left, right],
                output=Tensor(f"{_name}_{label}", _shape, "f32"),
            )

        # FP4 magnitudes [0, .5, 1, 1.5, 2, 3, 4, 6] without a gather:
        # .5*m + .5*max(m-4, 0) + max(m-6, 0).
        zero = constant("zero", 0.0)
        half = constant("half", 0.5)
        mag4 = binary("mag4", "subtract", magnitude, constant("four_f32", 4.0))
        mag6 = binary("mag6", "subtract", magnitude, constant("six_f32", 6.0))
        base = binary("magnitude_half", "multiply", magnitude, half)
        extra4 = binary("extra4_pos", "maximum", mag4, zero)
        extra4 = binary("extra4", "multiply", extra4, half)
        extra6 = binary("extra6", "maximum", mag6, zero)
        magnitude_value = binary("magnitude_plus4", "add", base, extra4)
        magnitude_value = binary("magnitude_value", "add", magnitude_value, extra6)

        three = const_bc(graph, name=f"{name}_sign_shift", value=3, target_shape=values_shape, dtype="i32")
        sign_i32 = graph.add_node(
            op=ElementwiseOp(op="right_shift"),
            inputs=[nibbles, three],
            output=Tensor(f"{name}_sign_i32", values_shape, "i32"),
        )
        sign = graph.add_node(
            op=ElementwiseOp(op="copy"),
            inputs=[sign_i32],
            output=Tensor(f"{name}_sign", values_shape, "f32"),
        )
        sign = binary("sign_twice", "multiply", sign, constant("two_f32", 2.0))
        sign = binary("sign_value", "subtract", constant("one_f32", 1.0), sign)
        fp4 = binary("fp4", "multiply", magnitude_value, sign)

        scales_i32 = graph.add_node(
            op=ElementwiseOp(op="copy"),
            inputs=[scales],
            output=Tensor(f"{name}_scales_i32", scales_shape, "i32"),
        )
        scales_f32 = graph.add_node(
            op=ElementwiseOp(op="copy"),
            inputs=[scales_i32],
            output=Tensor(f"{name}_scales_f32", scales_shape, "f32"),
        )
        exponent = graph.add_node(
            op=ElementwiseOp(op="subtract"),
            inputs=[scales_f32, const_bc(graph, name=f"{name}_bias127", value=127.0, target_shape=scales_shape, dtype="f32")],
            output=Tensor(f"{name}_exponent", scales_shape, "f32"),
        )
        scale_values = graph.add_node(
            op=ElementwiseOp(op="pow"),
            inputs=[const_bc(graph, name=f"{name}_scale_base", value=2.0, target_shape=scales_shape, dtype="f32"), exponent],
            output=Tensor(f"{name}_scale_values", scales_shape, "f32"),
        )
        scale_values = graph.add_node(
            op=ReshapeOp(shape=(n, k // 32, 1)),
            inputs=[scale_values],
            output=Tensor(f"{name}_scale_view", (n, k // 32, 1), "f32"),
        )
        scale_values = broadcast_to(graph, scale_values, values_shape)
        decoded = binary("decoded_blocks", "multiply", fp4, scale_values)
        decoded = graph.add_node(
            op=ReshapeOp(shape=(n, k)),
            inputs=[decoded],
            output=Tensor(f"{name}_stored_orientation", (n, k), "f32"),
        )
        if transposed:
            decoded = graph.add_node(
                op=TransposeOp(axes=(1, 0)),
                inputs=[decoded],
                output=Tensor(f"{name}_logical_f32", logical, "f32"),
            )
        if out.dtype.name != "f32":
            decoded = graph.add_node(
                op=ElementwiseOp(op="copy"),
                inputs=[decoded],
                output=Tensor(f"{name}_logical", logical, out.dtype),
            )
        graph.replace_node(tmp, decoded)
        graph.remove_node(tmp)
        out_map[name] = scales
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
