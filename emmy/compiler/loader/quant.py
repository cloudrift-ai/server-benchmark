"""FP8 checkpoint ingestion: birth-time spelling of the dequant algebra, plus the numpy math.

This module is the ONE place quantization-as-a-concept exists (together with the
safetensors loader that reads the checkpoint and ``trace/huggingface.py``'s
architecture-twin construction). Three halves:

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
