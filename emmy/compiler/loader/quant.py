"""Quantized-checkpoint ingestion: birth-time spelling of the decode algebra, plus the numpy math.

This module is the ONE place quantization-as-a-concept exists (together with the
safetensors loader that reads the checkpoint and ``trace/huggingface.py``'s
architecture-twin construction). Two checkpoint families share the design — FP8
(scale-paired bits, ``quant_method: "fp8"`` / compressed-tensors) and EXL3
(trellis-coded sibling tensors, ``quant_method: "exl3"``; decode math in
``loader/exl3.py``). Per family:

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
  weight is just constants + algebra. By default the generic
  ``032_fold_constant_subgraphs`` pass then dissolves the cone back into one
  bind-time-evaluated constant; with ``EMMY_FP8_EXPAND`` the cone stays
  in-graph for the kernel path.
- :func:`spell_quantized_inputs`: the input-sourced twin of the constant
  speller for graphs whose weights are forward-argument ``InputOp``s (the MoE
  serving seam's expert programs). Each named input becomes an fp8 bits input
  (uint8 carrier at the feed) plus a new scale input, with the same dequant
  cone spelled in-graph; an input-rooted cone is not a constant subgraph, so
  it stays in-graph unconditionally — no ``EMMY_FP8_EXPAND`` analog.
- :func:`load_dequantized_state_dict`: the eager / accuracy twin's state dict,
  every fp8 weight dequantized. Reads config + index directly (loader-band —
  it feeds the architecture twin before any graph exists, and its whole-dict
  semantics, consumed scales dropped, have no graph counterpart).
- :func:`spell_trellis_constants`: the EXL3 sibling of the constant speller —
  each trellis-coded weight becomes three leaf constants (int16 codes + the
  f16 ``suh``/``svh`` channel vectors) joined by a ``TrellisDecodeOp``. By
  default that is the checkpoint-basis decode, which
  ``032_fold_constant_subgraphs`` collapses into a bind-time ``source_graph``
  record (the correctness lane); under ``EMMY_TRELLIS_EXPAND`` the consuming
  LINEAR is re-spelled instead, with the Hadamard / channel-vector basis
  restore moved onto the activations around a hat-basis decode — the form the
  warp tier decodes in-kernel. :func:`load_dequantized_state_dict` decodes the
  same siblings into ``.weight`` values for the eager twin.
- :func:`spell_trellis_inputs`: the input-sourced twin of that kernel-path
  spelling, for the MoE serving seam's expert programs. Each named weight input
  becomes the int16 codes input plus two appended channel-vector inputs, with
  the same activation-side chain; input-rooted, so it never folds and needs no
  ``EMMY_TRELLIS_EXPAND`` analog.
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
    return {"quantization_config": None} if method == "exl3" else {}


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
    """Every checkpoint tensor as numpy VALUES, fp8 weights dequantized by their paired scale.

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
    """
    from emmy.compiler.loader.safetensors import _build_index, _read_shard  # noqa: PLC0415

    model_dir = Path(model_dir)
    index = _build_index(model_dir)
    qc = _fp8_quant_config(model_dir)
    exl3 = _exl3_quant_config(model_dir) is not None
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


def spell_quantized_constants(graph: Graph, model_id_or_path: str) -> int:
    """Spell every fp8-stored weight of ``graph`` as in-graph dequant algebra, at birth.

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


def _trellis_dims(graph: Graph, nid: str, shapes: dict[str, tuple[int, ...]]) -> tuple[int, int, int, int] | None:
    """Validate a trellis weight constant against its sibling shapes → ``(n, k, n_pad, k_pad)``.

    ``(n, k)`` are the traced logical ``(out, in)`` dims; ``(n_pad, k_pad)`` the 128-padded
    extents the checkpoint actually stores. ``None`` (with a warning) when the siblings do not
    reproduce the traced weight shape, or when the constant is not a pristine single-source
    trace constant — never a compile error.
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
    # padded weight back to the traced logical shape (exactly the reference math — see
    # ``TrellisDecodeOp``). The sibling extents must be exactly the traced dims' roundups.
    pad_ok = (-(-k // 128) * 128, -(-n // 128) * 128) == (k_pad, n_pad) if k > 0 and n > 0 else False
    if len(t_shape) != 3 or not pad_ok or (t_shape[0] * 16, t_shape[1] * 16) != (k_pad, n_pad):
        logger.warning(
            "trellis weight %s: sibling shapes trellis=%s suh=%s svh=%s do not reproduce %s; constant left alone",
            nid,
            t_shape,
            suh_shape,
            svh_shape,
            (n, k),
        )
        return None
    return n, k, n_pad, k_pad


def _spell_trellis_one(graph: Graph, nid: str, *, base: str, cb: int, shapes: dict[str, tuple[int, ...]]) -> bool:
    """Rewrite the weight constant ``nid`` into its trellis decode cone: three leaf
    ConstantOps (``{base}.trellis`` int16 codes, ``{base}.suh`` / ``{base}.svh`` f16
    channel vectors) joined by a ``TrellisDecodeOp``. Returns ``True`` on success.

    The cone's OUTPUT tensor keeps exactly the dtype/shape the trace promised. Sibling
    shapes that do not reproduce the traced weight shape leave the constant alone
    (``False``) — never a compile error.
    """
    from emmy.compiler.ir.frontend.ir import TrellisDecodeOp  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    dims = _trellis_dims(graph, nid, shapes)
    if dims is None:
        return False
    n, k, _n_pad, _k_pad = dims
    node = graph.nodes[nid]
    op, out = node.op, node.output
    shape, t_shape = (n, k), shapes["trellis"]
    suh_shape, svh_shape = shapes["suh"], shapes["svh"]

    frag = Graph()
    codes = frag.add_node(
        op=ConstantOp(name=f"{op.name}_trellis", source_path=base + ".trellis", source_shape=t_shape, source_dtype="i16"),
        inputs=[],
        output=Tensor(f"{out.name}_trellis", t_shape, "i16"),
    )
    suh = frag.add_node(
        op=ConstantOp(name=f"{op.name}_suh", source_path=base + ".suh", source_shape=suh_shape, source_dtype="f16"),
        inputs=[],
        output=Tensor(f"{out.name}_suh", suh_shape, "f16"),
    )
    svh = frag.add_node(
        op=ConstantOp(name=f"{op.name}_svh", source_path=base + ".svh", source_shape=svh_shape, source_dtype="f16"),
        inputs=[],
        output=Tensor(f"{out.name}_svh", svh_shape, "f16"),
    )
    dec = frag.add_node(
        op=TrellisDecodeOp(cb=cb, out_features=n, in_features=k),
        inputs=[codes, suh, svh],
        output=Tensor(out.name, shape, out.dtype),
    )
    frag.outputs = [dec]
    graph.splice(frag, consumed=[nid], output=nid)
    return True


def _hadamard_constant(graph: Graph, dtype) -> str:
    """The shared ±1 128-block Hadamard matrix constant; created once per graph and dtype.

    It has no checkpoint source, so it rides a zero-leaf ``source_graph`` bind record over one
    :class:`~emmy.compiler.ir.frontend.ir.HadamardOp` — the loader evaluates the record through
    the reference NumPy backend exactly as it evaluates a folded cone, and every trellis linear
    in the graph shares the one node.
    """
    from emmy.compiler.ir.frontend.ir import HadamardOp  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    nid = f"_had{HAD_BLOCK}_{dtype.name}"
    if nid not in graph.nodes:
        record = Graph()
        record.outputs = [record.add_node(op=HadamardOp(size=HAD_BLOCK), inputs=[], output=Tensor(nid, (HAD_BLOCK, HAD_BLOCK), dtype))]
        graph.add_node(
            op=ConstantOp(name=nid, source_graph=record, source_shape=(HAD_BLOCK, HAD_BLOCK), source_dtype=dtype.name),
            inputs=[],
            output=Tensor(nid, (HAD_BLOCK, HAD_BLOCK), dtype),
            node_id=nid,
        )
    return nid


def _had_blocks(frag: Graph, src: str, had: str, *, shape: tuple, dtype, name: str) -> str:
    """The per-128-block Hadamard: contract a ``(…, w/128, 128)`` operand against the shared
    128x128 constant. ``LinearOp`` is ``x @ H.T`` and ``H`` is symmetric, so no transpose.

    Spelling the transform as plain matmul algebra — rather than a butterfly realization of its
    own — is what puts it on the existing tiers and lets the search schedule it; the operand
    arrives ALREADY 128-blocked (see :func:`_multiply`), so the matmul reads a plain buffer.
    """
    from emmy.compiler.ir.frontend.ir import LinearOp  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    return frag.add_node(op=LinearOp(), inputs=[src, had], output=Tensor(name, shape, dtype))


def _multiply(frag: Graph, src: str, factor: str, *, shape: tuple, dtype, name: str) -> str:
    """``src * factor`` with ``factor`` a scalar or trailing-axis vector, broadcast made explicit.

    ALSO the layout-absorbing step of the trellis chain: an index map (the encode-pad concat,
    the flat↔128-block reshapes, the output slice) that reaches a MATMUL's A operand is
    mis-lowered today — the fragment loaders take the operand's declared row stride, not the one
    the index implies — so every such map is spelled onto a pointwise instead, which composes
    index maps correctly. Keep that placement when editing this chain.
    """
    from emmy.compiler.ir.tensor.ir import ElementwiseOp  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    bc = broadcast_to(frag, factor, shape)
    return frag.add_node(op=ElementwiseOp(op="multiply"), inputs=[src, bc], output=Tensor(name, shape, dtype))


def _trellis_linear_chain(
    frag: Graph,
    x_id: str,
    *,
    had: str,
    codes: str,
    suh: str,
    svh: str,
    bias: str | None,
    cb: int,
    dims: tuple[int, int, int, int],
    lead: tuple[int, ...],
    dtype,
    pre: str,
    out_name: str,
) -> str:
    """Spell ``y = x @ W`` with the EXL3 basis restore moved onto the activations. Shared by
    both trellis spellers; returns the fragment's output node id.

    The checkpoint stores ``W = diag(suh) · H · W_hat · H · diag(svh)`` in ``(in, out)``
    orientation, with ``H`` the 128-block Sylvester Hadamard scaled ``1/sqrt(128)`` per side.
    Only ``W_hat`` — the raw per-tile decode — has an in-kernel realization, so the linear is
    re-spelled as:

        x → [pad to k_pad] → ·suh → H → ·1/16 → **@ W_hat** → ·1/8 → H → ·svh → [+bias]

    Everything but the ``@ W_hat`` step is plain graph algebra — broadcast multiplies and two
    128x128 matmuls — so it rides the existing tiers with no new kernel machinery. The middle
    step is the hat-basis ``TrellisDecodeOp`` at the FULL padded extent, which lifts to the
    per-element ``TrellisLoad`` cone and binds as computed B: the packed codes are the only
    weight bytes crossing DRAM.

    Two placement rules make the chain lower correctly, and both are load-bearing:

    - **The Hadamard operands arrive 128-BLOCKED and its scale is a separate multiply.** Each
      layout change (the encode-pad concat, the flat↔block reshapes, the output slice) is
      spelled onto a POINTWISE, never onto a matmul's A operand — see :func:`_multiply`.
    - **The 1/sqrt(128) per side is split as ``1/16`` before the weight and ``1/8`` after**, two
      exact powers of two whose product is exactly ``1/128``, so the shared ``H`` constant is
      plain ±1 (exact in f16/bf16) and both intermediates stay BELOW the balanced magnitude.

    Encode padding is exactly the reference math: the Hadamard mixes within each 128-block, so a
    K-side pad must be zeros on the activation (the weight-side fold's own row slice), and an
    N-side pad is contracted and then sliced off. Leaf contract: ``suh`` arrives already
    128-BLOCKED (``(k_pad/128, 128)``) and ``svh`` already sliced to the logical ``(n,)`` — each
    speller supplies them in the form its root admits (a load-op chain on a constant, the
    declared input shape on an input).
    """
    from emmy.compiler.ir.frontend.ir import CatOp, LinearOp, ReshapeOp, SliceOp, TrellisDecodeOp  # noqa: PLC0415
    from emmy.compiler.ir.tensor.ir import ElementwiseOp  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    n, k, n_pad, k_pad = dims
    pad_shape, out_shape, final_shape = (*lead, k_pad), (*lead, n_pad), (*lead, n)
    in_blk, out_blk = (*lead, k_pad // HAD_BLOCK, HAD_BLOCK), (*lead, n_pad // HAD_BLOCK, HAD_BLOCK)
    scale = {
        s: frag.add_node(op=ConstantOp(name=f"{pre}_r{s}", value=2.0**-s), inputs=[], output=Tensor(f"{pre}_r{s}", (1,), dtype))
        for s in (4, 3)
    }

    # --- input side: zero-pad to the encode extent, apply suh, Hadamard ---
    cur = x_id
    if k_pad != k:
        zero = frag.add_node(op=ConstantOp(name=f"{pre}_zero", value=0.0), inputs=[], output=Tensor(f"{pre}_zero", (1,), dtype))
        dim = frag.add_node(op=ConstantOp(name=f"{pre}_catdim", value=-1.0), inputs=[], output=Tensor(f"{pre}_catdim", (1,), "i32"))
        fill = broadcast_to(frag, zero, (*lead, k_pad - k))
        cur = frag.add_node(op=CatOp(), inputs=[cur, fill, dim], output=Tensor(f"{pre}_xpad", pad_shape, dtype))
    cur = frag.add_node(op=ReshapeOp(shape=in_blk), inputs=[cur], output=Tensor(f"{pre}_xblk", in_blk, dtype))
    cur = _multiply(frag, cur, suh, shape=in_blk, dtype=dtype, name=f"{pre}_xs")
    cur = _had_blocks(frag, cur, had, shape=in_blk, dtype=dtype, name=f"{pre}_xh")
    cur = frag.add_node(op=ReshapeOp(shape=pad_shape), inputs=[cur], output=Tensor(f"{pre}_xhf", pad_shape, dtype))
    cur = _multiply(frag, cur, scale[4], shape=pad_shape, dtype=dtype, name=f"{pre}_xr")

    # --- the weight: the hat-basis decode at the full padded extent (computed B) ---
    w_hat = frag.add_node(
        op=TrellisDecodeOp(cb=cb, out_features=n_pad, in_features=k_pad, hadamard=False),
        inputs=[codes],
        output=Tensor(f"{pre}_what", (n_pad, k_pad), dtype),
    )
    cur = frag.add_node(op=LinearOp(), inputs=[cur, w_hat], output=Tensor(f"{pre}_z", out_shape, dtype))

    # --- output side: Hadamard, apply svh, drop the encode padding, then bias ---
    # The LAST stage carries the traced output tensor's own name, so the spliced chain hands
    # the linear's consumers a buffer named exactly as before.
    cur = frag.add_node(op=ReshapeOp(shape=out_blk), inputs=[cur], output=Tensor(f"{pre}_zblk", out_blk, dtype))
    cur = _multiply(frag, cur, scale[3], shape=out_blk, dtype=dtype, name=f"{pre}_zr")
    cur = _had_blocks(frag, cur, had, shape=out_blk, dtype=dtype, name=f"{pre}_zh")
    cur = frag.add_node(op=ReshapeOp(shape=out_shape), inputs=[cur], output=Tensor(f"{pre}_zhf", out_shape, dtype))
    if n_pad != n:
        cur = frag.add_node(op=SliceOp(shape=final_shape, dim=-1, start=0), inputs=[cur], output=Tensor(f"{pre}_yc", final_shape, dtype))
    cur = _multiply(frag, cur, svh, shape=final_shape, dtype=dtype, name=f"{pre}_ys" if bias is not None else out_name)
    if bias is not None:
        bc = broadcast_to(frag, bias, final_shape)
        cur = frag.add_node(op=ElementwiseOp(op="add"), inputs=[cur, bc], output=Tensor(out_name, final_shape, dtype))
    return cur


def _lead_dims(x_t) -> tuple:
    """The activation's leading (non-contraction) extents, static ones as plain ints — a static
    ``Dim`` in a shape reaches numpy through the reference backend's reshape and is not an int."""
    return tuple(d.as_static() if d.is_static else d for d in x_t.shape[:-1])


def _linear_consumer(graph: Graph, nid: str):
    """The single ``LinearOp`` node consuming ``nid`` as its WEIGHT, else ``None``.

    Only the activation's CONTRACTION dim has to be static — the chain contracts and re-blocks
    that axis, while the leading (token) axes ride through as whatever ``Dim``s the trace gave
    them, so a symbolic-width program spells the same way a static one does."""
    from emmy.compiler.ir.frontend.ir import LinearOp  # noqa: PLC0415

    users = graph.consumers(nid)
    if len(users) != 1:
        return None
    lin = graph.nodes[users[0]]
    if not isinstance(lin.op, LinearOp) or len(lin.inputs) < 2 or lin.inputs[1] != nid or lin.inputs[0] == nid:
        return None
    x_t = graph.nodes[lin.inputs[0]].output
    if len(x_t.shape) < 2 or not x_t.shape[-1].is_static:
        return None
    return lin


def _spell_trellis_activation_one(graph: Graph, nid: str, *, base: str, cb: int, shapes: dict[str, tuple[int, ...]]) -> bool:
    """Rewrite the LINEAR consuming trellis weight CONSTANT ``nid`` into the activation-side
    basis form (:func:`_trellis_linear_chain`), with the codes / channel vectors as checkpoint
    constants.

    Returns ``False`` — never a compile error — when the weight is not consumed by exactly one
    ``LinearOp`` (or its contraction dim is symbolic); the caller then falls back to the folded
    checkpoint-basis cone, which is correct everywhere.
    """
    from emmy.compiler.ir.base import InputOp  # noqa: PLC0415
    from emmy.compiler.ir.frontend.ir import ReshapeOp, SliceOp  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    dims = _trellis_dims(graph, nid, shapes)
    if dims is None:
        return False
    n, k, n_pad, k_pad = dims
    op = graph.nodes[nid].op

    lin = _linear_consumer(graph, nid)
    if lin is None:
        return False
    x_t, y_t = graph.nodes[lin.inputs[0]].output, lin.output
    if x_t.shape[-1].as_static() != k:
        return False
    dtype, pre, lead = x_t.dtype, y_t.name, _lead_dims(x_t)
    in_blk = (k_pad // HAD_BLOCK, HAD_BLOCK)
    had = _hadamard_constant(graph, dtype)

    frag = Graph()
    for alias in (lin.inputs[0], had, *lin.inputs[2:]):
        t = graph.nodes[alias].output
        frag.add_node(op=InputOp(), inputs=[], output=Tensor(alias, t.shape, t.dtype), node_id=alias)
    codes = frag.add_node(
        op=ConstantOp(name=f"{op.name}_trellis", source_path=base + ".trellis", source_shape=shapes["trellis"], source_dtype="i16"),
        inputs=[],
        output=Tensor(f"{pre}_trellis", shapes["trellis"], "i16"),
    )
    # ``suh`` binds already 128-blocked (a reshape in the load chain, the pack vocabulary's own
    # form) so it broadcasts straight onto the blocked activation.
    suh = frag.add_node(
        op=ConstantOp(
            name=f"{op.name}_suh",
            source_path=base + ".suh",
            source_shape=(k_pad,),
            source_dtype="f16",
            load_ops=(ReshapeOp(shape=in_blk),),
        ),
        inputs=[],
        output=Tensor(f"{pre}_suh", in_blk, dtype),
    )
    svh = frag.add_node(
        op=ConstantOp(name=f"{op.name}_svh", source_path=base + ".svh", source_shape=(n_pad,), source_dtype="f16"),
        inputs=[],
        output=Tensor(f"{pre}_svh", (n_pad,), dtype),
    )
    if n_pad != n:  # the N-side encode pad is contracted and then sliced off, channel vector too
        svh = frag.add_node(op=SliceOp(shape=(n,), dim=0, start=0), inputs=[svh], output=Tensor(f"{pre}_svhc", (n,), dtype))
    out = _trellis_linear_chain(
        frag,
        lin.inputs[0],
        had=had,
        codes=codes,
        suh=suh,
        svh=svh,
        bias=lin.inputs[2] if lin.op.has_bias else None,
        cb=cb,
        dims=dims,
        lead=lead,
        dtype=dtype,
        pre=pre,
        out_name=y_t.name,
    )
    frag.outputs = [out]
    graph.splice(frag, consumed=[nid, lin.id], output=lin.id)
    return True


def spell_trellis_constants(graph: Graph, model_id_or_path: str, *, expand: bool | None = None) -> int:
    """Spell every EXL3 trellis-coded weight of ``graph`` as an in-graph decode cone, at birth.

    The EXL3 sibling of :func:`spell_quantized_constants`, called from the same post-trace
    site. Source of truth is the CHECKPOINT: ``config.json`` declares ``quant_method:
    "exl3"`` (:func:`_exl3_quant_config`), and the safetensors index supplies the pairing —
    a traced ``<module>.weight`` constant whose module ships ``<module>.trellis`` siblings
    is rewritten into the codes + ``suh``/``svh`` leaves and a ``TrellisDecodeOp`` (its ``cb``
    field recording the marker-sibling presence, so the markers themselves never enter the
    graph). From that point a quantized weight is just constants + algebra.

    Two spellings, chosen by ``EMMY_TRELLIS_EXPAND`` (``config.trellis_expand``) or, when the
    caller passes ``expand`` explicitly, by that — the serving trunk asks for the compressed
    lane per compile rather than per process. An explicit ``expand=True`` also stamps the
    ``trellis.expand`` graph hint, which is how ``032_fold_constant_subgraphs`` learns to leave
    the hat-basis cone in-graph without reading the env knob:

    - **default, the correctness lane** — :func:`_spell_trellis_one` spells the
      CHECKPOINT-BASIS decode (``hadamard=True``) as a constant-only cone, which
      ``032_fold_constant_subgraphs`` collapses into ONE bind-time
      ``ConstantOp(source_graph=record)`` evaluated through the reference NumPy backend: full
      value footprint in memory, no kernel change.
    - **the kernel path** — :func:`_spell_trellis_activation_one` rewrites the consuming
      LINEAR into the activation-side basis restore around a HAT-BASIS decode
      (``hadamard=False``), the form that lifts to the per-element ``TrellisLoad`` cone and
      binds as computed B, so only the packed codes cross DRAM. Any linear it declines (a
      weight consumed by something other than one ``LinearOp``, symbolic activation dims) falls
      back to the folded cone, which stays correct.

    Weights without a trellis sibling (embeddings, norms, routers, biases) load as plain
    tensors. Legacy checkpoints storing packed ``su``/``sv`` sign words instead of
    ``suh``/``svh`` are left alone with a warning. Idempotent: the spelled leaves' source
    paths do not end in ``.weight``, so a second run matches nothing. Unquantized
    checkpoints are a no-op — zero graph change. Returns the number of constants spelled.
    """
    from safetensors import safe_open  # noqa: PLC0415

    from emmy import config  # noqa: PLC0415
    from emmy.compiler.loader.safetensors import _build_index, _candidate_keys, _resolve_model_dir  # noqa: PLC0415

    model_dir = _resolve_model_dir(model_id_or_path)
    if _exl3_quant_config(model_dir) is None:
        return 0
    index = _build_index(model_dir)
    if expand is None:
        expand = config.trellis_expand()
    elif expand:
        graph.hints.set("trellis.expand", True)

    spelled = 0
    with ExitStack() as stack:
        handles: dict[str, object] = {}  # shard path → open safe_open handle (metadata reads only)

        def _shape(key: str) -> tuple[int, ...]:
            path = str(index[key])
            handle = handles.get(path)
            if handle is None:
                handle = handles[path] = stack.enter_context(safe_open(path, framework="numpy"))
            return tuple(int(d) for d in handle.get_slice(key).get_shape())

        for nid, op in list(graph.loadable_constants()):
            if op.source_path is None or not op.source_path.endswith(".weight"):
                continue  # EXL3 quantizes linear ``.weight``s only; spelled leaves also land here (idempotency)
            suffix = len(".weight")
            base = next((c[:-suffix] for c in _candidate_keys(op.source_path) if c[:-suffix] + ".trellis" in index), None)
            if base is None:
                continue
            if base + ".suh" not in index or base + ".svh" not in index:
                logger.warning("trellis weight %s: no suh/svh channel vectors (legacy packed-sign checkpoint?); constant left alone", nid)
                continue
            shapes = {leaf: _shape(base + "." + leaf) for leaf in ("trellis", "suh", "svh")}
            args = {"base": base, "cb": _exl3_codebook(index, base), "shapes": shapes}
            if expand and _spell_trellis_activation_one(graph, nid, **args):
                spelled += 1
            elif _spell_trellis_one(graph, nid, **args):
                spelled += 1
    if spelled:
        logger.info("spelled %d trellis-coded weight constant(s) from %s", spelled, model_dir)
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
    pass is unaffected. An input-rooted cone is not a constant subgraph, so
    ``032_fold_constant_subgraphs`` leaves it in-graph unconditionally and the W8A16
    mul-hoist binding can absorb it exactly as on the ``EMMY_FP8_EXPAND`` constant path.

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
    """Spell named graph INPUTS as EXL3 codes + channel-vector inputs, the basis restore in-graph.

    The input-sourced twin of :func:`spell_trellis_constants`' expand spelling, for graphs whose
    weights are forward-argument ``InputOp``s — the MoE serving seam's expert programs, where
    every routed expert feeds its own weight slices into ONE compiled program and the constant
    speller can never fire. ``specs`` maps a weight input's node id to ``(cb, codes_shape)``:
    the codebook id (marker-sibling presence in the checkpoint index) and the packed codes'
    stored shape ``(k_pad/16, n_pad/16, 16*K)``, which is where the encode-padded extents and K
    come from; the logical ``(n, k)`` come from the input's own declared shape.

    Each named input keeps its node id and its ``graph.inputs`` slot, but becomes the int16
    CODES input at the stored codes shape; two channel-vector inputs are APPENDED to
    ``graph.inputs`` per spec, in spec order and ``suh`` before ``svh``:

    - ``<name>_suh`` at the 128-BLOCKED shape ``(k_pad/128, 128)`` — the feed reshapes its
      stored ``(k_pad,)`` vector, which is free (a view), and the blocked declaration is what
      lets it broadcast straight onto the blocked activation with no in-graph layout op.
    - ``<name>_svh`` at the LOGICAL ``(n,)`` — the N-side encode pad is sliced off at the feed
      (the stored vector's leading ``n`` entries, contiguous from the same base pointer), so
      an indirect operand's table entry is unchanged and the graph carries no slice.

    The consuming ``LinearOp`` is rewritten into :func:`_trellis_linear_chain` exactly as on
    the constant path, sharing the graph-wide Hadamard constant. That constant is NOT
    per-expert: it stays a plain constant so the fixed-slot dispatch table-resolves only the
    per-expert operands. An input-rooted decode cone is not a constant subgraph, so
    ``032_fold_constant_subgraphs`` leaves it in-graph unconditionally — no
    ``EMMY_TRELLIS_EXPAND`` analog, the codes stay compressed by construction.

    The caller names these inputs explicitly, so any mismatch raises ``ValueError`` instead of
    the constant speller's skip-and-continue. Returns ``{input id: (suh id, svh id)}``.
    """
    from emmy.compiler.ir.base import InputOp  # noqa: PLC0415
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
        lin = _linear_consumer(graph, name)
        if lin is None:
            raise ValueError(f"spell_trellis_inputs: input {name!r} is not the weight of exactly one static LinearOp")
        x_t, y_t = graph.nodes[lin.inputs[0]].output, lin.output
        if x_t.shape[-1].as_static() != k:
            raise ValueError(f"spell_trellis_inputs: activation trailing dim {x_t.shape[-1]} != in_features {k} for {name!r}")
        dtype, pre, lead = x_t.dtype, y_t.name, _lead_dims(x_t)
        had = _hadamard_constant(graph, dtype)

        # Rewire order mirrors :func:`spell_quantized_inputs`: park the traced weight input under
        # a temporary id, re-mint ``name`` as the codes input in the SAME ``graph.inputs`` slot,
        # append the two channel-vector inputs, then splice the chain over the parked node and
        # its consuming linear — every step through the index-consistent Graph mutators.
        tmp = f"{name}__tr_src"
        graph.rename_node(name, tmp)
        codes = graph.add_node(op=InputOp(), inputs=[], output=Tensor(name, codes_shape, "i16"), node_id=name)
        graph.inputs = [codes if i == tmp else i for i in graph.inputs]
        suh = graph.add_node(
            op=InputOp(), inputs=[], output=Tensor(f"{name}_suh", (k_pad // HAD_BLOCK, HAD_BLOCK), dtype), node_id=f"{name}_suh"
        )
        svh = graph.add_node(op=InputOp(), inputs=[], output=Tensor(f"{name}_svh", (n,), dtype), node_id=f"{name}_svh")
        graph.inputs += [suh, svh]

        frag = Graph()
        for alias in (lin.inputs[0], had, codes, suh, svh, *lin.inputs[2:]):
            t = graph.nodes[alias].output
            frag.add_node(op=InputOp(), inputs=[], output=Tensor(alias, t.shape, t.dtype), node_id=alias)
        final = _trellis_linear_chain(
            frag,
            lin.inputs[0],
            had=had,
            codes=codes,
            suh=suh,
            svh=svh,
            bias=lin.inputs[2] if lin.op.has_bias else None,
            cb=cb,
            dims=(n, k, n_pad, k_pad),
            lead=lead,
            dtype=dtype,
            pre=pre,
            out_name=y_t.name,
        )
        frag.outputs = [final]
        graph.splice(frag, consumed=[tmp, lin.id], output=lin.id)
        out_map[name] = (suh, svh)
    return out_map
