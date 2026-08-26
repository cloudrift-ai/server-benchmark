"""Thin wrappers that make HuggingFace CausalLM models trace-friendly.

The goal is a module whose ``forward(input_ids)`` runs the full model and
returns logits, without HF's dynamic causal-mask construction polluting
the FX graph. The mask is precomputed and stapled on as a buffer; HF's
``_update_causal_mask`` hooks are neutralised before export.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


def build_full_model_wrapper(model, seq_len: int, dtype, *, dynamic: bool = False, slice_last_logits: bool = False) -> nn.Module:
    """Return an ``nn.Module`` with a trace-friendly forward.

    Static mode (``dynamic=False``, default): forward is
    ``forward(input_ids) -> logits`` and the module carries a precomputed
    ``(1, 1, seq_len, seq_len)`` causal mask + ``(1, seq_len)`` position_ids
    as buffers. HF's dynamic mask machinery is short-circuited so the
    traced graph is free of mask-construction ops (arange/cumsum/diff/
    eq/le/__and__/new_ones/index/ne).

    Dynamic mode (``dynamic=True``, plan M4): forward is
    ``forward(input_ids, attention_mask, position_ids) -> logits`` — the
    caller supplies the per-call mask + position_ids sized to the actual
    seq_len. The traced graph then has ``attention_mask`` and
    ``position_ids`` as inputs (shape ``(1, 1, seq_len, seq_len)`` and
    ``(1, seq_len)`` respectively); rewriting the seq_len dim to
    ``Dim('seq_len')`` post-trace yields a graph that compiles once and
    runs at any seq_len. ``seq_len`` is still used at construction time
    to seed the short-circuit hooks with a same-shape sentinel mask so
    HF's internal validation passes.

    Dynamic mode replaces the rotary embedding too — with cos/sin
    precomputed for positions ``0..DYNAMIC_DIM_MAX-1`` and sliced to the
    runtime seq_len in-graph (``_SlicedRotary``).

    ``slice_last_logits`` (dynamic only, the Phase-0 generation oracle): instead of
    returning the whole-prefix logits ``[1, S, vocab]``, run the trunk to hidden states,
    slice the **final** position (``hidden[:, -1:, :]``), and apply ``lm_head`` to that
    one row → ``[1, 1, vocab]``. The generate loop only needs the next-token logits, so
    this avoids the O(S·vocab) lm_head over every prefix position and the full-buffer host
    copy each step. The ``hidden[:, -1:, :]`` slice makes lm_head an **M=1 demoted
    matmul**: it lowers cold (the unbindable bilinear fold demotes to PLANAR) and
    ``140_slice`` normalizes the negative row index against the symbolic extent
    (``seq_len - 1``, a runtime kernel arg) — pinned by
    ``test_slice_last_logits_lowers_cold``. ``_CompiledLM`` uses this form. Keeping HF's in-graph
    rotary instead silently breaks: its ``inv_freq`` buffer is
    ``persistent=False`` and doesn't survive ``torch.export`` with its
    real value, so the traced cos/sin constant-fold to ``cos=1, sin=0``
    and RoPE degenerates to identity. (The bug was invisible to the
    zero-``input_ids`` accuracy check: identical value rows make the
    attention output independent of the attention weights.) The sliced
    buffer assumes positions are ``0..S-1`` — true for full-sequence
    prefill, which is the only dynamic-mode use.
    """
    import torch
    import torch.nn as nn

    if slice_last_logits and not dynamic:
        raise ValueError("slice_last_logits requires dynamic=True (the generation oracle is dynamic-seq_len)")

    class _PassThroughRotary(nn.Module):
        """Replaces the model's ``rotary_emb`` and returns precomputed real
        ``(cos, sin)`` — an ``nn.Module`` (not a bare callable) since the
        attribute it overrides is a registered submodule. Buffers live on this
        module (no wrapper backref) so it stays a clean leaf submodule."""

        def __init__(self, cos, sin) -> None:
            super().__init__()
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)

        def forward(self, *_, **__):
            return self.cos, self.sin

    class _SlicedRotary(nn.Module):
        """Dynamic-mode rotary replacement: cos/sin precomputed out to
        ``DYNAMIC_DIM_MAX`` positions, sliced to the runtime seq_len read off
        the ``position_ids`` argument — the slice end is a SymInt, so the
        traced graph keeps real rotary values at every seq_len."""

        def __init__(self, cos, sin) -> None:
            super().__init__()
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)

        def forward(self, x, position_ids, *_, **__):
            s = position_ids.shape[1]
            return self.cos[:, :s], self.sin[:, :s]

    class FullModelWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            mask = torch.zeros((seq_len, seq_len), dtype=dtype)
            mask.masked_fill_(torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1), float("-inf"))
            self.register_buffer("causal_mask", mask[None, None, :, :])
            # Precompute position_ids so HF doesn't call torch.arange at forward time.
            self.register_buffer("position_ids", torch.arange(seq_len, dtype=torch.long)[None, :])
            # Short-circuit HF's dynamic mask builder(s). Different transformers
            # releases use different names; patch whichever exists.
            inner = getattr(model, "model", model)
            for attr in ("_update_causal_mask", "_prepare_4d_causal_attention_mask"):
                if hasattr(inner, attr):
                    setattr(inner, attr, _PassThroughMask(self))
            # Short-circuit the rotary embedding in static mode. Its cos/sin are
            # ``inv_freq @ position_ids`` under ``@torch.no_grad`` / float32
            # autocast; ``torch.export`` folds that to ``cos=1, sin=0`` (the
            # ``inv_freq`` buffer, ``persistent=False``, doesn't survive tracing
            # with its real value), so RoPE degenerates to identity. Compute the
            # real per-position cos/sin eagerly and return them verbatim — the
            # trace then captures correct constant tensors. Dynamic mode hits
            # the same folding, so it precomputes out to DYNAMIC_DIM_MAX and
            # slices to the runtime seq_len in-graph (_SlicedRotary below).
            rotary = getattr(inner, "rotary_emb", None)
            if not dynamic and rotary is not None:
                with torch.no_grad():
                    sample = torch.zeros((1, seq_len, model.config.hidden_size), dtype=dtype)
                    cos, sin = rotary(sample, self.position_ids)
                inner.rotary_emb = _PassThroughRotary(cos, sin)
            elif rotary is not None:
                from emmy.compiler.trace.dynamic import DYNAMIC_DIM_MAX  # noqa: PLC0415

                # +1: torch.export guards a symbolic slice end STRICTLY below
                # the sliced extent (``cos[:, :S]`` with S == extent would
                # specialize), so the buffer carries one extra position.
                n_pos = DYNAMIC_DIM_MAX + 1
                with torch.no_grad():
                    sample = torch.zeros((1, n_pos, model.config.hidden_size), dtype=dtype)
                    full_pos = torch.arange(n_pos, dtype=torch.long)[None, :]
                    cos, sin = rotary(sample, full_pos)
                inner.rotary_emb = _SlicedRotary(cos, sin)

        if dynamic and slice_last_logits:

            def forward(self, input_ids, attention_mask, position_ids):
                # Generation oracle: run the trunk, then apply lm_head to ONLY the
                # final position so the output is [1, 1, vocab] (next-token logits),
                # not the whole-prefix [1, S, vocab].
                trunk = getattr(self.model, "model", self.model)
                out = trunk(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=False,
                )
                hidden = out[0]  # [1, S, H], final norm already applied by the trunk
                last = hidden[:, -1:, :]  # [1, 1, H] — the next-token position
                head = getattr(self.model, "lm_head", None)
                if head is None:
                    head = self.model.get_output_embeddings()
                return head(last)  # [1, 1, vocab]

        elif dynamic:

            def forward(self, input_ids, attention_mask, position_ids):
                # The caller controls mask + position_ids so the seq_len axis can
                # flow through to ``Dim('seq_len')`` after the rewrite step.
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=False,
                )
                return out[0]

        else:

            def forward(self, input_ids):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=self.causal_mask,
                    position_ids=self.position_ids,
                    use_cache=False,
                    return_dict=False,
                )
                return out[0]

    return FullModelWrapper()


def build_synthetic_ple(block, n_pos: int, dtype):
    """Synthetic ``per_layer_input`` for a single-decoder-layer trace, or ``None``
    for non-PLE architectures.

    Gemma-nano (E2B/E4B) Per-Layer Embeddings: each decoder layer computes
    ``hidden * per_layer_input`` (modeling_gemma4). The real tensor is a per-layer
    token embedding the whole-model path supplies; a single-layer trace has no
    tokens, so stand in a deterministic synthetic one sized ``[1, n_pos, ple_dim]``.
    Perf-representative — the PLE gate/mul/projection/norm kernels depend on shape,
    not values, and the accuracy check stays valid (emmy and torch see the same
    buffer). Non-uniform (randn, fixed seed) so the elementwise mul isn't
    algebraically folded to identity."""
    import torch

    ple_dim = int(getattr(block, "hidden_size_per_layer_input", 0) or 0)
    if not ple_dim:
        return None
    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        return torch.randn((1, n_pos, ple_dim), generator=g, dtype=torch.float32).to(dtype)


def build_layer_wrapper(block, rotary_emb, hidden_size: int, dtype, *, layer_type=None) -> nn.Module:
    """Trace-friendly single-decoder-layer wrapper for dynamic mode:
    ``forward(x)`` slices rotary cos/sin to the runtime seq_len and calls
    ``block(x, position_embeddings=(cos, sin))``.

    The static per-layer trace passes concrete ``(cos, sin)`` kwargs sized to
    the trace seq_len — exactly the specialisation dynamic mode must avoid. So
    this wrapper precomputes cos/sin out to ``DYNAMIC_DIM_MAX`` positions as
    buffers and slices them in-graph by ``x.shape[1]`` (a SymInt under
    ``torch.export``), the same trick as the whole-model ``_SlicedRotary``.
    The sliced buffer assumes positions ``0..S-1`` — full-sequence prefill,
    the only dynamic-mode use. The forward arg is named ``x``, so the CLI
    spec is ``--dynamic seq_len@x:1``.

    ``layer_type`` feeds rotary modules that key cos/sin on the layer's
    attention type (e.g. Gemma's sliding/global split); ``None`` for the
    common single-rope architectures."""
    import torch
    import torch.nn as nn

    from emmy.compiler.trace.dynamic import DYNAMIC_DIM_MAX

    # +1: torch.export guards a symbolic slice end STRICTLY below the sliced
    # extent (``cos[:, :s]`` with s == extent would specialize) — same as the
    # whole-model dynamic wrapper's rotary buffer.
    n_pos = DYNAMIC_DIM_MAX + 1
    with torch.no_grad():
        sample = torch.zeros((1, n_pos, hidden_size), dtype=dtype)
        full_pos = torch.arange(n_pos, dtype=torch.long)[None, :]
        try:
            cos, sin = rotary_emb(sample, full_pos, layer_type)
        except TypeError:
            cos, sin = rotary_emb(sample, full_pos)

    # Sliced in-graph like cos/sin; ``None`` for non-PLE architectures — they take
    # the unchanged path below.
    ple = build_synthetic_ple(block, n_pos, dtype)
    ple_dim = 0 if ple is None else ple.shape[-1]

    class LayerWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = block
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)
            if ple is not None:
                self.register_buffer("ple", ple)

        def forward(self, x):
            s = x.shape[1]
            pe = (self.cos[:, :s], self.sin[:, :s])
            if ple_dim:
                return self.block(x, per_layer_input=self.ple[:, :s], position_embeddings=pe)
            return self.block(x, position_embeddings=pe)

    return LayerWrapper()


def find_text_decoder(model):
    """Return the deepest module owning the text decoder layers and rotary embedding."""
    import torch.nn as nn  # noqa: PLC0415

    decoder = None
    for _name, module in model.named_modules():
        if isinstance(getattr(module, "layers", None), nn.ModuleList) and hasattr(module, "rotary_emb") and hasattr(module, "config"):
            decoder = module
    if decoder is None:
        raise ValueError(f"could not locate a text decoder in {type(model).__name__}")
    return decoder


def selected_layer_type(decoder, block, layer: int):
    """Return the config-authoritative attention type for one decoder layer."""
    configured = getattr(decoder.config, "layer_types", None)
    if configured is not None and 0 <= layer < len(configured):
        return configured[layer]
    block_type = getattr(block, "attention_type", None)
    if block_type is not None:
        return block_type
    return getattr(getattr(block, "self_attn", None), "layer_type", None)


def specialize_deepseek_full_coverage_compressor(block, seq_len: int) -> bool:
    """Replace a fixed-shape CSA block-bias tail with its exact all-entry form.

    DeepSeek V4's canonical 512-token audit has 128 CSA entries and ``index_topk=512``,
    so top-k selects every entry. The downstream compressor consumes only the selected
    index set to build ``block_bias``; ordering cannot affect that scatter. Retain the
    installed compressor's exact KV computation, discard its now-value-independent top-k
    bias, and rebuild the identical causal block bias without unsupported top-k/scatter.
    Return ``False`` for non-CSA layers and fail closed when top-k is genuinely selective.
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    attention = getattr(block, "self_attn", None)
    compressor = getattr(attention, "compressor", None)
    indexer = getattr(compressor, "indexer", None)
    if indexer is None:
        return False
    compress_rate = getattr(compressor, "compress_rate", None)
    index_compress_rate = getattr(indexer, "compress_rate", None)
    index_topk = getattr(indexer, "index_topk", None)
    if (
        not isinstance(compress_rate, int)
        or isinstance(compress_rate, bool)
        or compress_rate <= 0
        or not isinstance(index_topk, int)
        or isinstance(index_topk, bool)
        or index_topk <= 0
    ):
        raise ValueError(
            "DeepSeek V4 CSA audit requires positive integer compressor.compress_rate and indexer.index_topk, "
            f"got {compress_rate!r} and {index_topk!r}"
        )
    if index_compress_rate != compress_rate:
        raise ValueError(
            "DeepSeek V4 CSA audit requires the compressor and indexer to enumerate the same entries: "
            f"compressor.compress_rate={compress_rate}, indexer.compress_rate={index_compress_rate!r}"
        )
    compressed_len = seq_len // compress_rate
    if compressed_len <= 0 or index_topk < compressed_len:
        raise NotImplementedError(
            "DeepSeek V4 CSA audit cannot replace a selective top-k indexer: "
            f"seq_len={seq_len}, compress_rate={compress_rate}, compressed_len={compressed_len}, index_topk={index_topk}"
        )

    class FullCoverageCompressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = compressor
            self.compress_rate = compress_rate
            self.register_buffer("entry_indices", torch.arange(compressed_len, dtype=torch.long).view(1, 1, -1))

        def forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx):
            if past_key_values is not None:
                raise NotImplementedError("DeepSeek V4 full-coverage CSA audit specialization is stateless only")
            compressed_kv, _unused_bias = self.inner(hidden_states, q_residual, position_ids, past_key_values, layer_idx)
            causal_threshold = ((position_ids + 1) // self.compress_rate).unsqueeze(-1)
            visible = self.entry_indices < causal_threshold
            batch, _, actual_len, _ = compressed_kv.shape
            if actual_len != compressed_len:
                raise ValueError(
                    f"DeepSeek V4 CSA compressed length changed during static trace: expected {compressed_len}, got {actual_len}"
                )
            bias = compressed_kv.new_zeros((batch, 1, position_ids.shape[1], compressed_len))
            return compressed_kv, bias.masked_fill(~visible.unsqueeze(1), float("-inf"))

    attention.compressor = FullCoverageCompressor()
    return True


def trace_selected_layer(model, layer: int, seq_len: int, dtype, *, dynamic_shapes: dict | None = None):
    """Trace one already-loaded decoder layer through the canonical model-layer path.

    This is the library primitive shared by ``emmy trace --layer`` and config-only
    architecture inventory providers. It preserves hyper-connection lanes, model-specific
    rotary mappings, required attention kwargs, and the representative-MoE substitution a
    caller may have applied before materializing the layer.
    """
    import inspect  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from emmy.compiler.trace.torch import trace_module  # noqa: PLC0415

    decoder = find_text_decoder(model)
    if not 0 <= layer < len(decoder.layers):
        raise ValueError(f"layer {layer} not found (model has {len(decoder.layers)} layers)")
    block = decoder.layers[layer]
    hidden_size = decoder.config.hidden_size
    self_attn = getattr(block, "self_attn", None)
    is_deepseek_v4 = getattr(decoder.config, "model_type", None) == "deepseek_v4"
    deepseek_sliding_window = None
    if is_deepseek_v4:
        deepseek_sliding_window = getattr(self_attn, "sliding_window", None)
        if not isinstance(deepseek_sliding_window, int) or isinstance(deepseek_sliding_window, bool) or deepseek_sliding_window <= 0:
            raise ValueError(
                f"DeepSeek V4 selected-layer trace requires a positive self_attn.sliding_window, got {deepseek_sliding_window!r}"
            )
    hc_mult = int(getattr(getattr(block, "attn_hc", None), "hc_mult", 0) or 0)
    x_shape = (1, seq_len, hc_mult, hidden_size) if hc_mult else (1, seq_len, hidden_size)
    x = torch.randn(*x_shape, dtype=dtype)
    layer_type = selected_layer_type(decoder, block, layer)
    rotary_emb = decoder.rotary_emb
    if layer_type == "sliding_attention" and getattr(decoder, "swa_rotary_emb", None) is not None:
        rotary_emb = decoder.swa_rotary_emb

    if dynamic_shapes:
        wrapper = build_layer_wrapper(block, rotary_emb, hidden_size, dtype, layer_type=layer_type)
        graph = trace_module(wrapper, (x,), dynamic_shapes=dynamic_shapes)
        stamp_sliding_windows(
            graph,
            decoder.config,
            layer_type=layer_type,
            sliding_window=deepseek_sliding_window,
        )
        return graph, (wrapper, (x,), {})

    position_ids = torch.arange(seq_len).unsqueeze(0)
    if is_deepseek_v4 and layer_type == "compressed_sparse_attention":
        if not specialize_deepseek_full_coverage_compressor(block, seq_len):
            raise NotImplementedError("DeepSeek V4 CSA selected-layer trace requires the confirmed full-coverage compressor specialization")
    rotary_layer_types = tuple(getattr(rotary_emb, "layer_types", ()) or ())
    if rotary_layer_types and layer_type in rotary_layer_types:
        position_embeddings = rotary_emb(x, position_ids, layer_type)
    elif rotary_layer_types:
        position_embeddings = {rotary_type: rotary_emb(x, position_ids, rotary_type) for rotary_type in rotary_layer_types}
    else:
        try:
            position_embeddings = rotary_emb(x, position_ids, layer_type)
        except (KeyError, TypeError):
            position_embeddings = rotary_emb(x, position_ids)

    kwargs = {"position_embeddings": position_embeddings}
    block_parameters = inspect.signature(block.forward).parameters
    if "input_ids" in block_parameters:
        kwargs["input_ids"] = torch.zeros((1, seq_len), dtype=torch.long)
    if self_attn is not None:
        attention_parameters = inspect.signature(self_attn.forward).parameters
        if "position_ids" in attention_parameters:
            kwargs["position_ids"] = position_ids
        if "attention_mask" in attention_parameters:
            if is_deepseek_v4:
                from transformers.masking_utils import create_sliding_window_causal_mask  # noqa: PLC0415

                mask_inputs = x[:, :, 0, :] if x.ndim == 4 else x
                attention_mask = create_sliding_window_causal_mask(
                    config=decoder.config,
                    inputs_embeds=mask_inputs,
                    attention_mask=None,
                    past_key_values=None,
                    position_ids=position_ids,
                )
                if not isinstance(attention_mask, torch.Tensor):
                    raise ValueError(
                        "DeepSeek V4 selected-layer trace requires a materialized static sliding causal mask, "
                        f"got {type(attention_mask).__name__}"
                    )
                kwargs["attention_mask"] = attention_mask
            elif attention_parameters["attention_mask"].default is inspect.Parameter.empty:
                # A required declaration still needs an explicit ``None``.
                # Optional masks must be omitted: torch.export otherwise
                # retains a scalar placeholder for the non-tensor ``None``,
                # while the eager input flattener correctly drops it. The
                # resulting 4-vs-3 phantom input makes direct model benchmarks
                # impossible even though the default call is semantically
                # identical.
                kwargs["attention_mask"] = None
    ple = build_synthetic_ple(block, seq_len, dtype)
    if ple is not None:
        kwargs["per_layer_input"] = ple
    graph = trace_module(block, (x,), kwargs=kwargs, dynamic_shapes=dynamic_shapes)
    stamp_sliding_windows(
        graph,
        decoder.config,
        layer_type=layer_type,
        sliding_window=deepseek_sliding_window,
    )
    return graph, (block, (x,), kwargs)


def _build_pre_wrapper(block, *, float32_residual: bool = False):
    """The shared q/k/v carve of one HF decoder layer: ``pre(hidden[T, H]) -> (q, k, v)`` runs
    ``input_layernorm`` → separate projections → reshape-into-heads → q/k(/v) norm, returning
    **un-rotated** q,k,v in the 2-D seam ABI. Used by both the dense and the MoE split builders;
    carries the structural probes (PLE / clip_qkv rejects, norm placement, ``attention_k_eq_v``)."""
    import torch.nn as nn

    ple_dim = int(getattr(block, "hidden_size_per_layer_input", 0) or 0)
    if ple_dim:
        raise NotImplementedError(
            f"build_attention_split_wrapper: block carries Per-Layer Embeddings "
            f"(hidden_size_per_layer_input={ple_dim}, Gemma-nano E2B/E4B) — the attention-split carve "
            f"would silently drop the per_layer_input multiply; this model is not servable via the split path"
        )

    attn = block.self_attn
    clip_qkv = getattr(getattr(attn, "config", None), "clip_qkv", None)
    if clip_qkv is not None:
        raise NotImplementedError(
            f"build_attention_split_wrapper: clip_qkv={clip_qkv} (OLMo-style q/k/v clamping) has no seam in the "
            f"attention-split carve — the pre wrapper would silently skip the clamp, corrupting outputs"
        )
    head_dim = attn.head_dim
    num_heads = attn.q_proj.out_features // head_dim
    num_kv_heads = attn.k_proj.out_features // head_dim
    q_norm = getattr(attn, "q_norm", None)
    k_norm = getattr(attn, "k_norm", None)
    v_norm = getattr(attn, "v_norm", None)  # Gemma-4 RMSNorms V too; Qwen3 / Gemma-3 / Llama do not
    # Two q/k-norm placements exist: Qwen3 / Gemma-3/4 normalize PER HEAD over head_dim (after the
    # head reshape); OLMoE normalizes the FLAT projection (before the reshape) with a norm sized
    # over the whole projection width. Distinguish them by the norm's own width.
    flat_qk_norm = q_norm is not None and q_norm.weight.numel() != head_dim

    class Pre(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = block.input_layernorm
            self.q_proj, self.k_proj, self.v_proj = attn.q_proj, attn.k_proj, attn.v_proj
            self.q_norm, self.k_norm, self.v_norm = q_norm, k_norm, v_norm

        def forward(self, hidden):
            h = self.input_layernorm(hidden)  # [T, H]
            if float32_residual:
                h = h.to(self.q_proj.weight.dtype)
            t = h.shape[0]
            if flat_qk_norm:  # OLMoE: RMSNorm over the flat projection, before the head reshape
                q = self.q_norm(self.q_proj(h)).view(t, num_heads, head_dim)
                k = self.k_norm(self.k_proj(h)).view(t, num_kv_heads, head_dim)
                v = self.v_proj(h).view(t, num_kv_heads, head_dim)
                return q.reshape(t, num_heads * head_dim), k.reshape(t, num_kv_heads * head_dim), v.reshape(t, num_kv_heads * head_dim)
            q = self.q_proj(h).view(t, num_heads, head_dim)  # [T, Hq, D] — NO transpose
            kp = self.k_proj(h).view(t, num_kv_heads, head_dim)
            # Gemma-4's global layers set attention_k_eq_v (no v_proj): V reuses K's projection
            # (un-rotated; v_norm below still differs from k_norm). Otherwise V is its own projection.
            v = self.v_proj(h).view(t, num_kv_heads, head_dim) if self.v_proj is not None else kp
            k = kp
            if self.q_norm is not None:
                q = self.q_norm(q)  # per-head RMSNorm over D
                k = self.k_norm(kp)
            if self.v_norm is not None:
                v = self.v_norm(v)  # Gemma-4: per-head RMSNorm over D on V as well
            return q.reshape(t, num_heads * head_dim), k.reshape(t, num_kv_heads * head_dim), v.reshape(t, num_kv_heads * head_dim)

    return Pre()


def _attention_gate_layout(attn):
    """Return ``(projection, per_head, head_dim)`` for an optional attention gate.

    Transformers 5.12's built-in Laguna module omits ``gate_per_head`` even though
    its projection has one output per head.  Infer that older layout from the
    projection widths; newer modules keep their explicit declaration authoritative.
    """
    g_proj = getattr(attn, "g_proj", None)
    if g_proj is None:
        return None, False, 0
    head_dim = int(getattr(attn, "head_dim", 0) or 0)
    declared = getattr(attn, "gate_per_head", None)
    if declared is not None:
        return g_proj, bool(declared), head_dim
    gate_width = int(getattr(g_proj, "out_features", 0) or 0)
    q_width = int(getattr(getattr(attn, "q_proj", None), "out_features", 0) or 0)
    if gate_width <= 0 or q_width <= 0:
        raise ValueError("attention gate inference requires positive gate and query projection widths")
    if head_dim > 0 and gate_width * head_dim == q_width:
        return g_proj, True, head_dim
    if gate_width == q_width:
        return g_proj, False, head_dim
    raise ValueError(
        f"unsupported attention gate width {gate_width}: expected one channel per head or one per projected query channel ({q_width})"
    )


def build_attention_split_wrapper(block, *, float32_residual: bool = False):
    """Carve SDPA out of one HF decoder layer (Phase 1). Returns ``(pre, post)`` ``nn.Module``s over
    the flattened **``[num_tokens, H]``** per-token layout:

    - ``pre(hidden[T, H]) -> (q, k, v)`` runs ``input_layernorm`` → separate
      ``q_proj`` / ``k_proj`` / ``v_proj`` → reshape-into-heads → per-head ``q_norm`` /
      ``k_norm`` (Qwen3 only; OLMoE's FLAT pre-reshape placement is handled), and returns
      **un-rotated** q,k,v in the 2-D seam ABI ``q[T, Hq·D]``, ``k/v[T, Hkv·D]`` — exactly what
      vLLM's ``Attention.forward`` consumes. RoPE is applied downstream (by vLLM, or by the
      test/oracle reference).
    - ``post(attn_out[T, Hq·D], residual[T, H]) -> layer_out[T, H]`` runs ``o_proj`` →
      ``residual +`` → ``post_attention_layernorm`` → ``mlp`` → second residual. **Gemma-3/4** is
      instead a 4-norm layer: ``o_proj(attn)`` and ``mlp(...)`` each get wrapped in their OWN
      RMSNorm (``post_attention_layernorm`` / ``post_feedforward_layernorm``) BEFORE the residual
      add, and the MLP input passes through ``pre_feedforward_layernorm`` — selected when those
      feed-forward norms are present on the block.

    Reads the block's OWN submodules (not a ``self_attn`` substitution — HF
    ``self_attn.forward`` returns ``(attn_output, weights)``, and the block adds the
    residual after it, so swapping ``self_attn`` can't yield a clean pre-graph). NOTE: HF's
    ``.view(B,S,-1,D).transpose(1,2)`` assumes a ``[batch, seq, hidden]`` input; on the
    flattened ``[T, H]`` layout the reshape is ``.view(T, n_heads, D)`` with **no transpose**.
    Qwen3 / Gemma-3/4 (q/k norm) and Llama (no q/k norm) all share the ``pre``; the ``post``
    is the Llama/Qwen 2-norm form or the Gemma 4-norm form above.

    Rejects Gemma-nano PLE blocks (``hidden_size_per_layer_input``) and OLMo-style ``clip_qkv``
    (in :func:`_build_pre_wrapper`): the carve has no seam for either and would silently drop
    them, corrupting outputs."""
    import torch.nn as nn

    if hyper_connection_seam(block) is not None:
        raise NotImplementedError(
            "build_attention_split_wrapper: a hyper-connection block (DeepSeek V4) carries hc_mult residual streams and a "
            "self-contained attention sublayer; it takes the attention-sublayer seam of build_moe_split_wrapper"
        )
    pre = _build_pre_wrapper(block, float32_residual=float32_residual)  # carries PLE / clip_qkv rejects before attribute reads
    attn = block.self_attn

    class Post(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.o_proj = attn.o_proj
            # Laguna gates the flattened attention result before o_proj, using the SAME
            # normalized layer input that produced q/k/v.  The split ABI carries the original
            # residual into post, so recompute that inexpensive norm here rather than dropping
            # the gate or widening the vLLM-owned attention seam.
            self.g_proj, self.gate_per_head, self.gate_head_dim = _attention_gate_layout(attn)
            self.gate_input_layernorm = block.input_layernorm if self.g_proj is not None else None
            self.post_attention_layernorm = block.post_attention_layernorm
            self.mlp = block.mlp
            # Gemma-3/4 is a 4-norm layer: these two extra norms wrap the attention output and the
            # MLP, each applied BEFORE its residual add. Their presence selects the layout below.
            self.pre_feedforward_layernorm = getattr(block, "pre_feedforward_layernorm", None)
            self.post_feedforward_layernorm = getattr(block, "post_feedforward_layernorm", None)
            # Gemma-4 scales the whole layer output by a learned per-layer scalar. Checkpoints carry
            # values far from 1 (the 12B ranges 0.005–0.92), so dropping it inflates the residual
            # stream ~8× by mid-network and overflows fp16. Fresh (random-init) models hold 1.0,
            # which is why parity tests must set it explicitly to stay sensitive. register_buffer
            # (it is a buffer on the block) so the compile's constant binding picks it up.
            self.register_buffer("layer_scalar", getattr(block, "layer_scalar", None))
            self._emmy_laguna_exl3_post = "dense" if float32_residual else None

        def forward(self, attn_out, residual):
            if self.g_proj is not None:
                gate_input = self.gate_input_layernorm(residual)
                if float32_residual:
                    gate_input = gate_input.to(attn_out.dtype)
                gate = nn.functional.softplus(self.g_proj(gate_input).float()).to(attn_out.dtype)
                if self.gate_per_head:
                    attn_out = (attn_out.view(attn_out.shape[0], -1, self.gate_head_dim) * gate.unsqueeze(-1)).view(attn_out.shape[0], -1)
                else:
                    attn_out = attn_out * gate
            if self.pre_feedforward_layernorm is not None:  # Gemma 4-norm
                h = residual + self.post_attention_layernorm(self.o_proj(attn_out))
                h = h + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(h)))
                return h if self.layer_scalar is None else h * self.layer_scalar
            h = residual + self.o_proj(attn_out)  # Llama/Qwen: residual1 + o_proj(SDPA)
            xn = self.post_attention_layernorm(h)
            if float32_residual:
                xn = xn.to(attn_out.dtype)
            return h + self.mlp(xn)  # residual2 + MLP

    return pre, Post()


def moe_block_parts(mlp):
    """The ``(router, experts)`` pair of a token-choice MoE block, or ``None`` for a dense MLP.

    Matches the transformers-v5 experts interface: a router module (named ``gate`` in the
    OLMoE/Qwen lineage, ``router`` in gpt-oss) beside an ``experts`` module holding the
    per-expert weights as 3-D ``nn.Parameter``s (``gate_up_proj``, ``down_proj``). That storage
    layout is what lets serving pass per-expert dim-0 slices as program inputs."""
    import torch

    router = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
    experts = getattr(mlp, "experts", None)
    if router is None or experts is None:
        return None
    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    if not (isinstance(gate_up, torch.Tensor) and gate_up.dim() == 3 and isinstance(down, torch.Tensor) and down.dim() == 3):
        return None
    return router, experts


def moe_expert_layout(experts):
    """The experts module's weight-layout contract, read off the attributes transformers'
    ``@use_experts_implementation`` decorator stamps on every v5 experts class — NOT off the
    tensor shapes (gpt-oss ``down_proj`` is square ``(E, 2880, 2880)``, so shape sniffing
    cannot tell the orientations apart). Returns ``(transposed, interleaved, has_bias)``:

    - ``transposed`` — weights stored ``(E, in, out)`` and applied as ``x @ W`` (gpt-oss);
      ``False`` is the ``F.linear`` ``(E, out, in)`` orientation (OLMoE).
    - ``interleaved`` — gate/up live in the gate-up projection's OUT axis as even/odd
      columns (gpt-oss) rather than concatenated halves; the serving load de-interleaves
      once so the wrapper's ``chunk(2)`` split holds for both (see
      :func:`deinterleave_gate_up`).
    - ``has_bias`` — per-expert ``gate_up_proj_bias`` / ``down_proj_bias`` exist and become
      two more expert-program inputs."""
    return (
        bool(getattr(experts, "is_transposed", False)),
        not bool(getattr(experts, "is_concatenated", True)),
        bool(getattr(experts, "has_bias", False)),
    )


def replace_moe_with_traceable_expert(block) -> bool:
    """Replace token routing with one representative expert for trace inventory.

    Routing (top-k/sort/group/combine) is host orchestration in Emmy serving and
    is not a tuneable tensor kernel.  Inventory tracing still needs the routed
    expert algebra, so use expert zero with the same per-expert weights and keep
    any always-on shared expert.  The replacement is intentionally limited to
    the biasless, concatenated ``F.linear`` layout used by DeepSeek/OLMoE; other
    layouts retain their normal module and return ``False``.
    """
    import torch.nn as nn

    parts = moe_block_parts(getattr(block, "mlp", None))
    if parts is None:
        return False
    _router, experts = parts
    transposed, interleaved, has_bias = moe_expert_layout(experts)
    act_fn = getattr(experts, "act_fn", None)
    if transposed or interleaved or has_bias or act_fn is None:
        return False
    raw_limit = getattr(experts, "limit", None)
    limit = None if raw_limit is None else float(raw_limit)
    if limit is not None and limit <= 0:
        raise ValueError(f"representative MoE expert requires a positive clamp limit, got {raw_limit!r}")
    shared_experts = next(
        (module for attr in ("shared_experts", "shared_expert") if (module := getattr(block.mlp, attr, None)) is not None),
        None,
    )
    raw_routed_scale = getattr(block.mlp, "routed_scaling_factor", 1.0)
    routed_scaling_factor = float(1.0 if raw_routed_scale is None else raw_routed_scale)

    class RepresentativeExpert(nn.Module):
        def __init__(self):
            super().__init__()
            # Parameter views share the already-materialized layer storage; no
            # additional multi-gigabyte expert tensor is allocated.
            self.w_gate_up = nn.Parameter(experts.gate_up_proj[0], requires_grad=False)
            self.w_down = nn.Parameter(experts.down_proj[0], requires_grad=False)
            self.act_fn = act_fn
            self.limit = limit
            self.shared_experts = shared_experts
            self.routed_scaling_factor = routed_scaling_factor
            self._emmy_traceable_expert = True

        def forward(self, x, input_ids=None):  # noqa: ARG002 — block API compatibility
            gate, up = nn.functional.linear(x, self.w_gate_up).chunk(2, dim=-1)
            if self.limit is not None:
                # DeepSeek V4's routed experts clamp both SwiGLU branches before
                # activation. Keep the exact ``experts._apply_gate`` algebra in
                # the representative path; omitting it changes both graph shape
                # and eager values for the deployed clamp-10 experts.
                gate = gate.clamp(max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
            output = nn.functional.linear(self.act_fn(gate) * up, self.w_down)
            output = output * self.routed_scaling_factor
            if self.shared_experts is not None:
                output = output + self.shared_experts(x)
            return output

    block.mlp = RepresentativeExpert()
    return True


def deinterleave_gate_up(t):
    """De-interleave a gate-up tensor's LAST axis from even/odd gate/up columns into
    concatenated ``[gate | up]`` halves — a one-time exact column permutation applied at
    LOAD (weight bits, scale and bias alike), which restores the ``chunk(2, dim=-1)``
    algebra so ONE expert-wrapper spelling serves both the OLMoE and the gpt-oss layouts.
    The alternative — strided ``::2`` slices spelled in-graph — would make every gate/up
    read a stride-2 column access, which the staged tile transports do not lower well."""
    import torch

    return torch.cat([t[..., 0::2], t[..., 1::2]], dim=-1).contiguous()


def _interleave_gate_up(t):
    """Invert :func:`deinterleave_gate_up` for an eager gpt-oss architecture twin."""
    import torch  # noqa: PLC0415

    if t.shape[-1] % 2:
        raise ValueError(f"gate/up output dimension must be even, got {tuple(t.shape)}")
    half = t.shape[-1] // 2
    return torch.stack((t[..., :half], t[..., half:]), dim=-1).flatten(-2).contiguous()


def _materialize_mxfp4_expert_store(model, layers_store: dict[int, dict], dtype) -> None:
    """Decode a shard-streamed MXFP4 store into the selected eager-reference layers."""
    import torch  # noqa: PLC0415

    from emmy.compiler.loader.quant import decode_mxfp4  # noqa: PLC0415

    trunk = getattr(model, "model", model)
    trunk = getattr(trunk, "language_model", trunk)
    for layer, store in layers_store.items():
        experts = trunk.layers[layer].mlp.experts
        values = {
            "gate_up_proj": _interleave_gate_up(
                torch.from_numpy(decode_mxfp4(store["w_gate_up"].numpy(), store["w_gate_up_scale"].numpy())).to(dtype)
            ),
            "down_proj": torch.from_numpy(decode_mxfp4(store["w_down"].numpy(), store["w_down_scale"].numpy())).to(
                dtype=dtype, memory_format=torch.contiguous_format
            ),
        }
        if "b_gate_up" in store:
            values["gate_up_proj_bias"] = _interleave_gate_up(store["b_gate_up"]).to(dtype)
        if "b_down" in store:
            values["down_proj_bias"] = store["b_down"].to(dtype)
        experts.load_state_dict(values, strict=True, assign=True)


def retarget_constants_to_model(graph, wrapper, model) -> None:
    """Re-address wrapper-relative trace *parameters* to their full model paths.

    Selected-layer and serving-split wrappers expose paths such as ``self_attn.q_proj.weight``;
    quantized checkpoints store ``model.layers.4.self_attn.q_proj.trellis``.  Tensor identity is
    the stable bridge between them and avoids architecture-specific prefix guessing.  Retargeting
    must happen before the checkpoint spellers inspect ``source_path``.

    Buffers deliberately keep their wrapper paths.  Full-model tracing replaces non-persistent
    rotary state with initialized ``cos`` / ``sin`` buffers that exist only on the wrapper, not in
    safetensors.  Retargeting those buffers to the nested model path makes both checkpoint and
    live-module binding miss, and the CUDA runtime then materializes zero-filled RoPE tables.
    Quantization spellers target weight parameters, so buffers have no reason to cross this seam.
    """
    id_to_key = {id(tensor): path for path, tensor in model.named_parameters(remove_duplicate=False)}
    key_map = {}
    for path, tensor in wrapper.named_parameters(remove_duplicate=False):
        if (full := id_to_key.get(id(tensor))) is not None:
            key_map[path] = full
    for _node_id, op in graph.loadable_constants():
        if op.source_path in key_map:
            op.source_path = key_map[op.source_path]
        if op.source_parts:
            op.source_parts = tuple((key_map.get(path, path), shape) for path, shape in op.source_parts)


def hyper_connection_seam(block):
    """``(carrier_width, attn_out_width)`` of a hyper-connection decoder block, or ``None``.

    DeepSeek V4 keeps ``hc_mult`` parallel residual streams per token (mHC); the serving carrier is
    that stack flattened to ``[num_tokens, hc_mult * hidden]``, and the attention seam carries the whole
    attention sublayer's ``[num_tokens, hidden]`` output rather than q/k/v (the 1Cat fork's paged MLA
    attention owns the projections, compressors, indexer and output projection)."""
    attn_hc = getattr(block, "attn_hc", None)
    if attn_hc is None:
        return None
    hidden = int(block.input_layernorm.weight.shape[0])
    return int(attn_hc.hc_mult) * hidden, hidden


def place_routed_streams(mixed, routed, mix):
    """Close the hyper-connection MoE seam: ``mixed[T, hc·H] + mix[T, hc] ⊗ routed[T, H]``.

    The post program already placed the shared expert and mixed the streams (``post ⊗ shared +
    combᵀ · streams``); the routed combine runs in torch afterwards and lands on the streams through the
    same per-stream ``post`` weights, which the program returns as ``mix``."""
    t = routed.shape[0]
    return mixed + (mix.unsqueeze(-1) * routed.unsqueeze(-2)).reshape(t, -1)


def _build_hyper_connection_split(block):
    """``(pre, post)`` of one DeepSeek V4 layer over the flattened stream carrier.

    - ``pre(hidden[T, hc·H]) -> x[T, H]`` — the attention-site stream collapse and ``input_layernorm``:
      exactly what the fork's ``DeepseekV4Attention.forward(positions, x)`` consumes.
    - ``post(attn_out[T, H], residual[T, hc·H]) -> (mixed[T, hc·H], xn[T, H], mix[T, hc])`` —
      attention-site stream mixing (the collapse weights are recomputed from ``residual``, one small
      GEMM, so nothing but the carrier crosses the seam), feed-forward collapse + ``post_attention_layernorm``,
      the always-on shared expert placed on the streams, and the feed-forward ``post`` weights for the
      routed combine (:func:`place_routed_streams`).

    Both mirror ``DeepseekV4DecoderLayer.forward`` by calling the block's own hyper-connection modules on a
    ``[1, T, hc, H]`` view, so the Sinkhorn / sigmoid / float32 contract is the installed modeling code's."""
    import torch
    import torch.nn as nn

    hc = int(block.attn_hc.hc_mult)
    hidden_size = int(block.input_layernorm.weight.shape[0])
    shared_experts = block.mlp.shared_experts

    class Pre(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn_hc = block.attn_hc
            self.input_layernorm = block.input_layernorm

        def forward(self, hidden):  # ``hidden`` is the seam's arg name (the symbolic num_tokens spec keys on it)
            t = hidden.shape[0]
            _post, _comb, collapsed = self.attn_hc(hidden.view(1, t, hc, hidden_size))
            return self.input_layernorm(collapsed).view(t, hidden_size)

    class Post(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn_hc = block.attn_hc
            self.ffn_hc = block.ffn_hc
            self.post_attention_layernorm = block.post_attention_layernorm
            self.shared_experts = shared_experts

        def forward(self, attn_out, residual):
            t = attn_out.shape[0]
            dtype = residual.dtype
            streams = residual.view(1, t, hc, hidden_size)
            post, comb, _ = self.attn_hc(streams)
            streams = post.to(dtype).unsqueeze(-1) * attn_out.view(1, t, 1, hidden_size) + torch.matmul(
                comb.to(dtype).transpose(-1, -2), streams
            )
            post, comb, collapsed = self.ffn_hc(streams)
            xn = self.post_attention_layernorm(collapsed)
            mixed = post.to(dtype).unsqueeze(-1) * self.shared_experts(xn).unsqueeze(-2) + torch.matmul(
                comb.to(dtype).transpose(-1, -2), streams
            )
            return mixed.view(t, hc * hidden_size), xn.view(t, hidden_size), post.to(dtype).view(t, hc)

    return Pre(), Post()


def build_moe_split_wrapper(block, *, split_gate_up: bool = False, float32_residual: bool = False):
    """Carve one MoE decoder layer into the third-seam form. Returns ``(pre, post_attn, expert)``:

    - ``pre`` — the same q/k/v carve as :func:`build_attention_split_wrapper`.
    - ``post_attn(attn_out[T, Hq·D], residual[T, H]) -> (h[T, H], xn[T, H])`` — o_proj + residual,
      then the post-attention norm. BOTH come out: the router and the expert programs consume the
      normed ``xn``, and the layer output is ``h + combine(expert outputs)`` — the norm output must
      materialize at the seam because it is shared across the k routed experts. A DeepSeek/GLM-style
      always-on ``shared_experts`` module (a plain dense MLP over the same ``xn``) ordinarily
      folds INTO the returned ``h`` (``h = residual + o_proj(attn) + shared_experts(xn)``).
      A marked Laguna EXL3 block instead returns ``(h, xn, shared)`` with an fp32 residual and
      shared term plus an fp16 normalized activation. The runner adds the fp32 routed term without
      narrowing the architecture's residual stream. Laguna's softplus attention gate is applied
      before ``o_proj`` from a reconstructed normalized residual.
    - ``expert(x[T_e, H], w_gate_up, w_down [, b_gate_up, b_down]) -> y[T_e, H]`` — one expert's
      gated MLP with the weights as FORWARD ARGUMENTS (they trace as graph inputs, not constants),
      so ONE compiled program serves every expert of every same-shaped layer; the caller passes
      per-expert slices of the 3-D expert tensors at launch. Two layouts, selected by
      :func:`moe_expert_layout`: the OLMoE ``F.linear`` form (``(out, in)`` weights, ``act_fn``,
      no bias) and the gpt-oss ``x @ W`` form (``(in, out)`` weights, per-expert biases as two
      more forward args, and the clamped-SwiGLU activation — ``gate.clamp(max=limit)``,
      ``up.clamp(±limit)``, ``glu = gate·σ(α·gate)``, ``out = (up + 1)·glu`` with the module's
      ``alpha``/``limit``). Both spell the gate/up split as ``chunk(2, dim=-1)``: an interleaved
      checkpoint (gpt-oss) is de-interleaved once at load (:func:`deinterleave_gate_up`), never
      strided in-graph. ``split_gate_up`` takes gate and up as SEPARATE forward args instead
      (``expert(x, w_gate, w_up, w_down)``) — the EXL3 form, where each coded linear carries its
      own input-side channel vector, so the merged weight has no single basis to restore and the
      merged spelling would only add a concat the activation split undoes. Ordinarily a model's
      ``routed_scaling_factor`` multiplies each routed expert result here (and never its shared
      expert). A marked EXL3 architecture can instead fold it into router weights when fp16
      partials cannot safely carry the factor.

    The router (linear + softmax/sigmoid + topk — untraceable ops) and the weighted combine stay in
    torch, orchestrated by the serving runner. Token-choice top-k MoE with the 2-norm (Llama-style)
    block layout only; the Gemma 4-norm MoE form and Qwen-MoE's GATED shared expert
    (``shared_expert_gate``) are rejected loudly until a model needs them — a silent pass would
    drop (or mis-weight) the shared term from every layer's output."""
    import torch
    import torch.nn as nn

    parts = moe_block_parts(block.mlp)
    if parts is None:
        raise NotImplementedError("build_moe_split_wrapper: block.mlp does not expose the (router, experts) MoE interface")
    gate, experts = parts
    if getattr(block.mlp, "shared_expert_gate", None) is not None:
        raise NotImplementedError(
            "build_moe_split_wrapper: block.mlp carries a `shared_expert_gate` (Qwen-MoE gated shared expert) — "
            "the ungated shared-experts fold below would silently drop the gate, mis-weighting every layer's output"
        )
    shared_experts = next((m for a in ("shared_experts", "shared_expert") if (m := getattr(block.mlp, a, None)) is not None), None)
    raw_routed_scale = getattr(block.mlp, "routed_scaling_factor", 1.0)
    routed_scaling_factor = float(1.0 if raw_routed_scale is None else raw_routed_scale)
    routed_scale_folded = bool(getattr(gate, "_emmy_routed_base_scale_folded", False))
    expert_scaling_factor = 1.0 if routed_scale_folded else routed_scaling_factor
    if getattr(block, "pre_feedforward_layernorm", None) is not None:
        raise NotImplementedError("build_moe_split_wrapper: the Gemma 4-norm MoE block layout is not supported yet")
    hyper = _build_hyper_connection_split(block) if hyper_connection_seam(block) is not None else None
    if hyper is not None and (float32_residual or split_gate_up):
        raise NotImplementedError("build_moe_split_wrapper: the hyper-connection seam has no coded-trunk or float32-residual form")
    pre = hyper[0] if hyper is not None else _build_pre_wrapper(block, float32_residual=float32_residual)
    attn = block.self_attn

    class PostAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.o_proj = attn.o_proj
            self.g_proj, self.gate_per_head, self.gate_head_dim = _attention_gate_layout(attn)
            self.gate_input_layernorm = block.input_layernorm if self.g_proj is not None else None
            self.post_attention_layernorm = block.post_attention_layernorm
            # DeepSeek/GLM lineage: an always-on dense MLP beside the routed experts,
            # consuming the same normed xn. Folding it into h keeps the runner's
            # route-and-combine half seam-free: layer_out = h + Σ w_e · expert_e(xn).
            self.shared_experts = shared_experts
            self._emmy_shared_expert_float32 = shared_experts is not None and routed_scale_folded and float32_residual
            self._emmy_laguna_exl3_post = "sparse" if float32_residual else None

        def forward(self, attn_out, residual):
            if self.g_proj is not None:
                gate_input = self.gate_input_layernorm(residual)
                if float32_residual:
                    gate_input = gate_input.to(attn_out.dtype)
                gate = nn.functional.softplus(self.g_proj(gate_input).float()).to(attn_out.dtype)
                if self.gate_per_head:
                    attn_out = (attn_out.view(attn_out.shape[0], -1, self.gate_head_dim) * gate.unsqueeze(-1)).view(attn_out.shape[0], -1)
                else:
                    attn_out = attn_out * gate
            h = residual + self.o_proj(attn_out)
            xn = self.post_attention_layernorm(h)
            if float32_residual:
                xn = xn.to(attn_out.dtype)
            if self.shared_experts is not None:
                shared = self.shared_experts(xn)
                if self._emmy_shared_expert_float32:
                    return h, xn, shared
                h = h + shared
            return h, xn

    transposed, _interleaved, has_bias = moe_expert_layout(experts)
    act_fn = getattr(experts, "act_fn", None)
    clamped = act_fn is None and getattr(experts, "alpha", None) is not None and getattr(experts, "limit", None) is not None
    if act_fn is None and not clamped:
        raise NotImplementedError(
            "build_moe_split_wrapper: experts module has neither an act_fn nor the gpt-oss clamped-SwiGLU "
            "(alpha/limit) contract — the expert activation cannot be spelled"
        )

    if split_gate_up:
        if transposed or has_bias or act_fn is None:
            raise NotImplementedError(
                f"build_moe_split_wrapper: split gate/up experts need the F.linear, biasless, act_fn form "
                f"(got transposed={transposed}, has_bias={has_bias}, act_fn={act_fn})"
            )

        class ExpertFFN(nn.Module):
            # EXL3: gate/up are separate coded linears, each with its own channel vectors.
            def __init__(self) -> None:
                super().__init__()
                self.act_fn = act_fn
                self._emmy_output_float32 = routed_scale_folded

            def forward(self, x, w_gate, w_up, w_down):
                gate = nn.functional.linear(x, w_gate)
                up = nn.functional.linear(x, w_up)
                output = nn.functional.linear(self.act_fn(gate) * up, w_down)
                return output if expert_scaling_factor == 1.0 else output * expert_scaling_factor

    elif transposed and has_bias and clamped:
        alpha, limit = float(experts.alpha), float(experts.limit)

        class ExpertFFN(nn.Module):
            # gpt-oss: (in, out) weights applied as ``x @ W + b``; gate/up as chunk halves
            # (the load de-interleaved the checkpoint's even/odd columns); clamped-SwiGLU.
            def forward(self, x, w_gate_up, w_down, b_gate_up, b_down):
                gate, up = (x @ w_gate_up + b_gate_up).chunk(2, dim=-1)
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
                glu = gate * torch.sigmoid(gate * alpha)
                return (((up + 1) * glu) @ w_down + b_down) * expert_scaling_factor

    elif not transposed and not has_bias and act_fn is not None:
        # DeepSeek V4 clamps both SwiGLU branches before the activation (``swiglu_limit``); OLMoE has no limit.
        limit = getattr(experts, "limit", None)
        limit = None if limit is None else float(limit)
        if limit is not None and not limit > 0:
            raise ValueError(f"build_moe_split_wrapper: expert clamp limit must be positive, got {limit!r}")

        class ExpertFFN(nn.Module):
            # OLMoE / DeepSeek: (out, in) weights applied via F.linear; gated activation, optionally clamped.
            def __init__(self) -> None:
                super().__init__()
                self.act_fn = act_fn

            def forward(self, x, w_gate_up, w_down):
                gate, up = nn.functional.linear(x, w_gate_up).chunk(2, dim=-1)
                if limit is not None:
                    gate = gate.clamp(max=limit)
                    up = up.clamp(min=-limit, max=limit)
                return nn.functional.linear(self.act_fn(gate) * up, w_down) * expert_scaling_factor

    else:
        raise NotImplementedError(
            f"build_moe_split_wrapper: unsupported experts layout (transposed={transposed}, "
            f"has_bias={has_bias}, clamped={clamped}) — only the OLMoE (F.linear, biasless, act_fn) "
            f"and gpt-oss (x @ W, biased, clamped-SwiGLU) forms are spelled"
        )

    return pre, hyper[1] if hyper is not None else PostAttn(), ExpertFFN()


def promote_shared_expert_float32(graph) -> None:
    """Give one marked Laguna EXL3 shared expert the reference float32 compute cone.

    The checkpoint provenance is the guardrail: exactly one gate, up, and down coded linear
    must be present under ``shared_expert``. Gate/up outputs, the activation/product path into
    down, and down's result become float32; the surrounding marked residual stays float32 and the
    normalized activation stays float16. Trellis spelling preserves compressed operands while
    realizing the widened activation/result contract directly in the factorized contractions.
    """
    from emmy.compiler.dtype import F32  # noqa: PLC0415
    from emmy.compiler.ir.base import ConstantOp  # noqa: PLC0415
    from emmy.compiler.ir.frontend.ir import LinearOp  # noqa: PLC0415

    linears = {}
    suffixes = {f".{name}_proj.weight": name for name in ("gate", "up", "down")}
    for node in graph.nodes.values():
        if not isinstance(node.op, ConstantOp):
            continue
        source = node.op.source_path or ""
        if not any(part in f".{source}" for part in (".shared_expert.", ".shared_experts.")):
            continue
        kind = next((kind for suffix, kind in suffixes.items() if source.endswith(suffix)), None)
        if kind is None:
            continue
        users = [graph.nodes[uid] for uid in graph.consumers(node.id)]
        users = [user for user in users if isinstance(user.op, LinearOp) and len(user.inputs) >= 2 and user.inputs[1] == node.id]
        if len(users) != 1 or kind in linears:
            raise RuntimeError(f"marked shared expert needs one checkpoint-provenanced {kind}_proj linear, found {len(users)}")
        linears[kind] = users[0]
    if set(linears) != set(suffixes.values()):
        raise RuntimeError(f"marked shared expert needs gate/up/down checkpoint provenance, found {sorted(linears)}")

    down = linears["down"]
    reached = set()
    queue = [linears["gate"].id, linears["up"].id]
    while queue:
        nid = queue.pop()
        if nid in reached:
            continue
        reached.add(nid)
        graph.nodes[nid].output.dtype = F32
        for uid in graph.consumers(nid):
            if uid == down.id:
                reached.add(uid)
                continue
            queue.append(uid)
    if down.id not in reached:
        raise RuntimeError("marked shared-expert gate/up cone does not feed its checkpoint-provenanced down_proj")
    down.output.dtype = F32


def promote_expert_output_float32(graph) -> None:
    """Give one marked routed expert the reference float32 down reduction and output.

    The EXL3 expert graph receives ``w_down`` as a graph input. Exactly one linear must consume
    that input and directly produce the graph output; this provenance/cardinality check keeps
    the rewrite off gate/up and off ordinary expert programs. Trellis input spelling runs only
    after this rewrite, so the coded operand stays compressed while the contraction result uses
    float32.
    """
    from emmy.compiler.dtype import F32  # noqa: PLC0415
    from emmy.compiler.ir.base import InputOp  # noqa: PLC0415
    from emmy.compiler.ir.frontend.ir import LinearOp  # noqa: PLC0415

    weight = graph.nodes.get("w_down")
    if weight is None or not isinstance(weight.op, InputOp) or weight.id not in graph.inputs:
        raise RuntimeError("marked expert output needs a w_down graph input")
    users = [
        graph.nodes[uid]
        for uid in graph.consumers(weight.id)
        if isinstance(graph.nodes[uid].op, LinearOp) and len(graph.nodes[uid].inputs) >= 2 and graph.nodes[uid].inputs[1] == weight.id
    ]
    if len(users) != 1:
        raise RuntimeError(f"marked expert output needs one w_down-provenanced linear, found {len(users)}")
    down = users[0]
    if graph.outputs != [down.id]:
        raise RuntimeError(f"marked expert w_down linear must directly produce the graph output, found {graph.outputs}")
    down.output.dtype = F32


def promote_laguna_exl3_post_float32(graph, kind: str) -> None:
    """Preserve the reference Laguna EXL3 float32 block outputs.

    The split attention ABI remains float16. Exactly one checkpoint-provenanced ``o_proj``
    converts that attention result into the float32 residual stream. The dense first block also
    has exactly one non-shared ``mlp.down_proj`` whose reduction returns float32. Sparse routed
    and shared down projections are widened by their dedicated helpers.
    """
    from emmy.compiler.dtype import F32  # noqa: PLC0415
    from emmy.compiler.ir.base import ConstantOp  # noqa: PLC0415
    from emmy.compiler.ir.frontend.ir import LinearOp  # noqa: PLC0415

    if kind not in {"dense", "sparse"}:
        raise ValueError(f"unknown Laguna EXL3 post kind {kind!r}")

    def one_linear(suffix: str, *, exclude_shared: bool = False):
        matches = []
        for node in graph.nodes.values():
            if not isinstance(node.op, ConstantOp):
                continue
            source = node.op.source_path or ""
            if not source.endswith(suffix) or (exclude_shared and ".shared_expert" in source):
                continue
            users = [
                graph.nodes[uid]
                for uid in graph.consumers(node.id)
                if isinstance(graph.nodes[uid].op, LinearOp) and len(graph.nodes[uid].inputs) >= 2 and graph.nodes[uid].inputs[1] == node.id
            ]
            matches.extend(users)
        if len(matches) != 1:
            raise RuntimeError(f"marked Laguna EXL3 post needs one checkpoint-provenanced {suffix}, found {len(matches)}")
        return matches[0]

    one_linear(".self_attn.o_proj.weight").output.dtype = F32
    if kind == "dense":
        one_linear(".mlp.down_proj.weight", exclude_shared=True).output.dtype = F32


def stamp_sliding_windows(
    graph,
    config,
    *,
    layer_type: str | None = None,
    sliding_window: int | None = None,
) -> None:
    """Stamp per-layer sliding windows onto the traced ``SdpaOp`` nodes of ``graph``.

    The trace erases the window: a single-layer trace carries no mask at all (HF takes the
    ``is_causal`` path), and a whole-model trace materializes the banded mask as an opaque
    additive tensor. The wrapper builder KNOWS the semantics — ``config.sliding_window`` +
    ``layer_types`` — so it re-asserts them here as op metadata: a stamped SDPA's mask keeps at
    most the causal band ``kv ∈ [m − W + 1, m]`` (an explicit mask operand may keep less, e.g.
    padding), which is what lets the lowering skip key blocks wholly outside it.

    ``layer_type`` names a single-layer trace's attention type; ``None`` walks the whole model's
    SDPA nodes in execution order against ``config.layer_types`` (one SDPA per decoder layer —
    a count mismatch stamps nothing). Full-attention layers get the ``is_causal`` assertion
    alone — their stream end still derives through the whole-model trace's opaque bias operand."""
    from emmy.compiler.ir.frontend.ir import SdpaOp

    window = sliding_window if sliding_window is not None else getattr(config, "sliding_window", None)
    layer_types = getattr(config, "layer_types", None)
    if not window:
        return
    sdpa_nodes = [n for n in graph.nodes.values() if isinstance(n.op, SdpaOp)]
    if layer_type is not None:
        types = [layer_type] * len(sdpa_nodes)
    elif layer_types is not None and len(sdpa_nodes) == len(layer_types):
        types = list(layer_types)
    else:
        return
    deepseek_v4 = getattr(config, "model_type", None) == "deepseek_v4"
    deepseek_banded = {"heavily_compressed_attention", "compressed_sparse_attention"}
    if deepseek_v4 and len(sdpa_nodes) > 1 and layer_type is not None:
        raise NotImplementedError(
            f"DeepSeek V4 selected-layer sliding-window stamping expected at most one attention SDPA node, found {len(sdpa_nodes)}"
        )
    for node, lt in zip(sdpa_nodes, types, strict=True):
        if lt == "sliding_attention" or (deepseek_v4 and lt in deepseek_banded):
            node.op.sliding_window = window
        # Sliding AND full layers: the wrapper's mask is causal — asserting it structurally lets
        # the lowering derive the stream END (and, banded, the stream START) through the opaque
        # bias operand a whole-model trace carries.
        node.op.is_causal = True


def build_causal_mask(seq_len: int, dtype) -> torch.Tensor:  # noqa: F821
    """Return the ``(1, 1, seq_len, seq_len)`` causal mask the wrapper
    uses internally — exposed so callers in dynamic mode can construct a
    per-call mask sized to the actual seq_len."""
    import torch

    mask = torch.zeros((seq_len, seq_len), dtype=dtype)
    mask.masked_fill_(torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1), float("-inf"))
    return mask[None, None, :, :]


class _PassThroughMask:
    """Callable that returns the wrapper's precomputed causal mask verbatim."""

    def __init__(self, wrapper):
        self._wrapper = wrapper

    def __call__(self, *_, **__):
        return self._wrapper.causal_mask


def _is_quantized_dir(p) -> bool:
    """Whether the checkpoint at ``p`` declares a quantization scheme the loaders ingest
    (FP8 scale-paired bits, MXFP4 blocks, AWQ GEMM int4, or EXL3 siblings)."""
    from emmy.compiler.loader.quant import (  # noqa: PLC0415
        _awq_quant_config,
        _exl3_quant_config,
        _fp8_quant_config,
        _mxfp4_quant_config,
    )

    return any(config(p) is not None for config in (_fp8_quant_config, _mxfp4_quant_config, _awq_quant_config, _exl3_quant_config))


def quantized_checkpoint_dir(model_id_or_path: str, revision: str | None = None):
    """The local checkpoint dir when the model is a supported quantized checkpoint,
    else ``None``.

    Detection reads only ``config.json`` (for a repo id, a single cached hub
    download); the full snapshot is fetched only when the checkpoint IS
    quantized — the trace-and-bind path then needs the shards anyway (the
    dequant algebra is spelled from the safetensors index, weights read from
    the shards).

    The id may carry its revision as ``<repo>@<revision>``
    (:func:`~emmy.compiler.loader.safetensors.split_revision`); an explicit ``revision``
    argument wins. BOTH the detection read and the snapshot must use it — a repo that
    publishes one rung per branch has a different ``config.json`` (and a different
    per-tensor bit allocation) on each, so the default branch is a different model.
    """
    from pathlib import Path  # noqa: PLC0415

    p = Path(model_id_or_path)
    if p.is_dir():
        return p if _is_quantized_dir(p) else None
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    from emmy.compiler.loader.safetensors import _resolve_model_dir, split_revision  # noqa: PLC0415

    repo, tagged = split_revision(model_id_or_path)
    revision = revision or tagged
    try:
        cfg_path = hf_hub_download(repo, "config.json", revision=revision)
    except Exception as e:  # detection is best-effort: an unreadable config means
        # "not a quantized checkpoint here" and the pre-existing from_pretrained
        # path keeps ownership of error reporting (bad id, gated repo, offline).
        logger.debug("quantized-checkpoint detection skipped for %s: %s", model_id_or_path, e)
        return None
    if not _is_quantized_dir(Path(cfg_path).parent):
        return None
    return _resolve_model_dir(repo, revision)


# Expert-tensor leaves of a transformers-v5 experts module → the expert program's INPUT names
# (the contract between the streaming loader below and the serving runner's expert launches).
_EXPERT_LEAVES = {
    "gate_up_proj": "w_gate_up",
    "down_proj": "w_down",
    "gate_up_proj_blocks": "w_gate_up",
    "down_proj_blocks": "w_down",
    "gate_up_proj_scale": "w_gate_up_scale",
    "down_proj_scale": "w_down_scale",
    "gate_up_proj_scales": "w_gate_up_scale",
    "down_proj_scales": "w_down_scale",
    "gate_up_proj_bias": "b_gate_up",
    "down_proj_bias": "b_down",
}

# EXL3 lineage: each expert is its own MODULE (``…experts.E.{gate,up,down}_proj.<leaf>``), and
# gate/up stay SEPARATE program inputs — their ``suh`` channel vectors differ, so the merged
# gate_up weight has no single activation-side basis to restore (see ``spell_trellis_inputs``).
_EXL3_EXPERT_PROJ = {"gate_proj": "w_gate", "up_proj": "w_up", "down_proj": "w_down"}
_EXL3_EXPERT_LEAF = {"trellis": "", "suh": "_suh", "svh": "_svh"}
# The per-expert-module fp8 / dense lineage: one 2-D weight per expert projection, with its
# block scale (DeepSeek's ``weight_scale_inv`` is the dequant MULTIPLIER — see
# :func:`~emmy.compiler.loader.quant.scale_is_reciprocal`).
_PER_EXPERT_LEAF = {"weight": "", "weight_scale_inv": "_scale", "weight_scale": "_scale"}


def _auto_config_from_pretrained(model_dir, **kwargs):
    """Load an architecture config, trusting repository code only when required.

    Keep the quantized-twin path aligned with the ordinary Hugging Face trace path:
    built-in architectures stay on Transformers' implementation, while repositories
    whose config explicitly requires custom code get one guarded retry.

    A host process can re-register a model type onto its own minimal config class
    (vLLM's ``HFConfigParser`` does, via ``AutoConfig.register``), after which
    ``AutoConfig.from_pretrained`` returns that class instead of Transformers' own —
    dropping every derived field its ``__init__`` computes (DeepSeek V4 derives
    ``layer_types`` from ``compress_ratios``). When Transformers exports a native class
    of the same name, reload with it: the twin must see the architecture's own config.
    """
    import transformers  # noqa: PLC0415
    from transformers import AutoConfig  # noqa: PLC0415

    try:
        config = AutoConfig.from_pretrained(model_dir, **kwargs)
    except ValueError as e:
        if "trust_remote_code" not in str(e):
            raise
        return AutoConfig.from_pretrained(model_dir, trust_remote_code=True, **kwargs)
    native = getattr(transformers, type(config).__name__, None)
    if isinstance(native, type) and not isinstance(config, native):
        config = native.from_pretrained(model_dir, **kwargs)
    return config


def _auto_model_from_config(config, **kwargs):
    """Build an auto model with the same guarded custom-code retry as its config."""
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    try:
        return AutoModelForCausalLM.from_config(config, **kwargs)
    except ValueError as e:
        if "trust_remote_code" not in str(e):
            raise
        return AutoModelForCausalLM.from_config(config, trust_remote_code=True, **kwargs)


def _expert_slot(key: str):
    """``(layer_index, program_input_name, expert_index)`` when ``key`` is a per-expert tensor of
    a MoE layer's experts module, else ``None``. ``expert_index`` is ``None`` for the
    transformers-v5 E-STACKED 3-D params (already one tensor per layer) and an int for the
    per-expert-module lineage (EXL3 trellis siblings), which the loader stacks itself."""
    if ".experts." not in key or ".layers." not in key:
        return None
    layer = int(key.split(".layers.")[1].split(".")[0])
    tail = key.split(".experts.", 1)[1].split(".")
    if len(tail) == 1 and (name := _EXPERT_LEAVES.get(tail[0])) is not None:
        return layer, name, None
    if len(tail) == 3 and tail[0].isdigit() and tail[1] in _EXL3_EXPERT_PROJ and tail[2] in _EXL3_EXPERT_LEAF:
        return layer, _EXL3_EXPERT_PROJ[tail[1]] + _EXL3_EXPERT_LEAF[tail[2]], int(tail[0])
    if len(tail) == 3 and tail[0].isdigit() and tail[1] in _EXL3_EXPERT_PROJ and tail[2] in _PER_EXPERT_LEAF:
        # The per-expert-MODULE fp8 lineage (DeepSeek / Laguna): ``experts.<e>.<proj>.weight`` with
        # its block ``weight_scale_inv`` (or a plain bf16 ``weight`` on an ignored layer). The
        # loader stacks them into the E-leading ``w_gate_up`` / ``w_down`` program inputs.
        return layer, _EXL3_EXPERT_PROJ[tail[1]] + _PER_EXPERT_LEAF[tail[2]], int(tail[0])
    return None


def _expert_logical_dims(model, layer: int) -> dict[str, tuple[int, int]]:
    """``{input_name: (out_features, in_features)}`` for one MoE layer's EXL3 expert linears,
    read off the twin's E-stacked params — ``gate_up_proj`` ``(E, 2*I, H)`` and ``down_proj``
    ``(E, H, I)`` give the logical extents the checkpoint's 128-padded codes sit inside."""
    trunk = getattr(model, "model", model)
    trunk = getattr(trunk, "language_model", trunk)
    experts = trunk.layers[layer].mlp.experts
    inter, hidden = experts.gate_up_proj.shape[1] // 2, experts.gate_up_proj.shape[2]
    return {"w_gate": (inter, hidden), "w_up": (inter, hidden), "w_down": (hidden, inter)}


def _stack_exl3_experts(layer: int, by_name: dict, model) -> dict:
    """Stack one MoE layer's per-expert EXL3 tensors into the E-leading program inputs.

    Expert order is the checkpoint's own index and must be gapless (a hole is a broken
    checkpoint, not a layout to guess at). Channel vectors retain their full padded checkpoint
    extents, matching the generic reconstruction algebra's input contract."""
    import torch  # noqa: PLC0415

    from emmy.compiler.loader.exl3 import HAD_BLOCK  # noqa: PLC0415

    dims = _expert_logical_dims(model, layer)
    order = {name: sorted(d) for name, d in by_name.items()}
    n_e = len(next(iter(order.values())))
    out = {}
    for name, indices in order.items():
        if indices != list(range(n_e)):
            raise ValueError(f"expert store: layer {layer} input {name!r} covers experts {indices[:4]}... — expected 0..{n_e - 1}")
        stacked = torch.stack([by_name[name][e] for e in indices], dim=0)
        n, k = dims[name.removesuffix("_suh").removesuffix("_svh")]
        if name.endswith("_suh"):
            stacked = stacked[:, : -(-k // HAD_BLOCK) * HAD_BLOCK]
        elif name.endswith("_svh"):
            stacked = stacked[:, : -(-n // HAD_BLOCK) * HAD_BLOCK]
        elif (stacked.shape[1] * 16, stacked.shape[2] * 16) != (-(-k // HAD_BLOCK) * HAD_BLOCK, -(-n // HAD_BLOCK) * HAD_BLOCK):
            raise ValueError(f"expert store: layer {layer} {name!r} codes {tuple(stacked.shape[1:])} do not pad the twin's {(n, k)}")
        out[name] = stacked.contiguous()
    return out


def _stack_expert_modules(layer: int, by_name: dict, model) -> dict:  # noqa: ARG001 — same signature as the EXL3 stacker
    """Stack one MoE layer's per-expert-module fp8 (or dense) tensors into the E-leading program
    inputs the runner feeds: ``w_gate_up`` is the ``[gate | up]`` concatenation along the output
    axis (the de-interleaved convention the expert wrapper's ``chunk(2)`` reads), ``w_down`` as
    stored, and their block scales concatenated the same way. fp8 bits stay on the ``uint8``
    carrier; scales are f32 values. Expert order is the checkpoint's own index and must be
    gapless."""
    import torch  # noqa: PLC0415

    order = {name: sorted(d) for name, d in by_name.items()}
    n_e = len(next(iter(order.values())))
    stacked: dict = {}
    for name, indices in order.items():
        if indices != list(range(n_e)):
            raise ValueError(f"expert store: layer {layer} input {name!r} covers experts {indices[:4]}... — expected 0..{n_e - 1}")
        stacked[name] = torch.stack([by_name[name][e] for e in indices], dim=0)
    out: dict = {}
    for suffix in ("", "_scale"):
        gate, up = stacked.get("w_gate" + suffix), stacked.get("w_up" + suffix)
        if gate is not None and up is not None:
            out["w_gate_up" + suffix] = torch.cat([gate, up], dim=1).contiguous()
        elif gate is not None or up is not None:
            raise ValueError(f"expert store: layer {layer} has a gate without an up projection (or the reverse) at {suffix or 'weight'}")
        if (down := stacked.get("w_down" + suffix)) is not None:
            out["w_down" + suffix] = down.contiguous()
    return out


def _native_checkpoint_renamer(config, keys=()):
    """The checkpoint→module key translation for an architecture published in its own namespace.

    DeepSeek V4 ships the flat V3-style spelling no module ever sees: ``layers.N.attn.wq_a``,
    ``layers.N.attn_norm``, ``layers.N.ffn.experts.E.w1``, ``hc_attn_fn``, ``embed`` / ``head``, and
    ``.scale`` for every fp8 block-scale sibling. Transformers publishes that renaming itself
    (:func:`transformers.conversion_mapping.get_checkpoint_conversion_mapping`), so reuse it rather
    than keeping a second copy that can drift from the modeling code the twin is built from. Only its
    ``WeightRenaming`` entries apply here: the accompanying ``WeightConverter`` entries MERGE the
    routed experts into one dense stacked parameter, which is exactly what the serving load must not
    do — the experts stay compressed and per-expert until :func:`_stack_expert_modules`.

    Two normalizations finish the job, both keeping downstream name maps untouched: the routed
    ``w1``/``w3``/``w2`` projections take their ``gate``/``up``/``down`` module names (the same
    convention Transformers applies to this checkpoint's SHARED expert), and a quantization block
    scale becomes the ``weight_scale`` sibling spelling the fp8 pairing and :data:`_PER_EXPERT_LEAF`
    already read. A ``.scale`` leaf earns that second rule ONLY when the checkpoint also holds the
    module's ``.weight``: the hyper-connection blocks carry a LEARNED ``hc_attn_scale`` parameter
    whose name ends the same way, and renaming it would leave the twin's ``attn_hc.scale`` on meta.

    Returns identity for every other architecture: a checkpoint already in module naming must not be
    put through a rename chain.
    """
    if getattr(config, "model_type", None) != "deepseek_v4":
        return lambda key: key
    present = set(keys)  # hoisted: this checkpoint has 72k keys, and rebuilding the set per key is quadratic
    block_scales = {key for key in present if key.endswith(".scale") and key[: -len(".scale")] + ".weight" in present}
    from transformers.conversion_mapping import WeightRenaming, get_checkpoint_conversion_mapping  # noqa: PLC0415

    renames = [entry for entry in get_checkpoint_conversion_mapping("deepseek_v4") or [] if isinstance(entry, WeightRenaming)]
    if not renames:
        raise ValueError("transformers publishes no deepseek_v4 checkpoint renaming; cannot translate the native checkpoint")
    routed = re.compile(r"(\.experts\.\d+\.)w([123])\.")
    projections = {"1": "gate_proj.", "2": "down_proj.", "3": "up_proj."}

    def rename(key: str) -> str:
        quantization_scale = key in block_scales
        for entry in renames:
            renamed = entry.rename_source_key(key)
            key = renamed[0] if isinstance(renamed, tuple) else renamed
        key = routed.sub(lambda m: m.group(1) + projections[m.group(2)], key)
        if quantization_scale:
            key = key[: -len(".scale")] + ".weight_scale"
        # Transformers renames into the BARE base-model namespace and re-attaches the prefix at load
        # time; this loader assigns straight onto a head model, so carry it here. ``lm_head`` is the
        # one leaf that lives outside the trunk.
        return key if key.startswith("lm_head.") else "model." + key

    return rename


def _checkpoint_to_model_key(key: str) -> str:
    """Translate original Laguna checkpoint names to built-in Transformers names."""
    key = key.replace(".mlp.shared_expert.", ".mlp.shared_experts.")
    key = key.replace(".mlp.experts.e_score_correction_bias", ".mlp.gate.e_score_correction_bias")
    return key


def _apply_exl3_laguna_routed_scale(model, *, exl3: bool) -> None:
    """Fold Laguna EXL3's routed-up inverse scale into the router weights.

    Laguna's EXL3 architecture uses ``interm_div=128``: routed ``up_proj`` weights are stored
    after division by 128, and the reference runtime folds both the inverse factor and the
    model's base expert multiplier into selected routing weights. Wrap each sparse router with
    that complete factor after checkpoint state loading. Keeping it on routing weights avoids
    materializing scaled expert outputs in fp16. Dense blocks and always-on shared experts are
    intentionally unchanged. The checkpoint has no field for ``interm_div``; 128 is the
    official Laguna EXL3 architecture invariant.
    """
    import torch.nn as nn  # noqa: PLC0415

    config = getattr(model, "config", None)
    if not exl3 or getattr(config, "model_type", None) != "laguna":
        return

    class RoutedScoreScale(nn.Module):
        def __init__(self, router, scale) -> None:
            super().__init__()
            self.router = router
            self._emmy_exl3_laguna_scale = scale
            self._emmy_routed_base_scale_folded = True
            self._emmy_routed_accumulate_float32 = True

        def forward(self, hidden_states):
            routed = self.router(hidden_states)
            return (*routed[:-2], routed[-2] * self._emmy_exl3_laguna_scale, routed[-1])

    trunk = getattr(model, "model", model)
    trunk = getattr(trunk, "language_model", trunk)
    for block in trunk.layers:
        mlp = getattr(block, "mlp", None)
        parts = moe_block_parts(mlp) if mlp is not None else None
        if parts is None:
            continue
        router, _experts = parts
        raw_scale = getattr(mlp, "routed_scaling_factor", 1.0)
        total_scale = 128.0 * float(1.0 if raw_scale is None else raw_scale)
        applied = getattr(router, "_emmy_exl3_laguna_scale", None)
        if applied == total_scale:
            continue
        if applied is not None:
            raise RuntimeError(f"Laguna EXL3 routed scale is already {applied}, expected {total_scale}")
        name = "gate" if getattr(mlp, "gate", None) is router else "router"
        setattr(mlp, name, RoutedScoreScale(router, total_scale))


def _quantized_stage_owns(key: str, layer_range, *, include_embed: bool, include_norm: bool) -> bool:
    """Whether one PP stage needs a quantized checkpoint leaf.

    Layer leaves belong only to their absolute decoder interval. The embedding and final norm
    follow the first/last-stage ownership used by vLLM. The output head is deliberately outside
    this loader when PP is active: ``EmmyGenModel`` owns and decodes it on the last stage. Unknown
    model-level leaves stay replicated; they are small architecture state and some model families
    require them while tracing a local layer.

    A multi-token-prediction head is never owned. It is a separate speculative-decoding model that no
    twin instantiates, and on DeepSeek V4 it is not small: its 256 routed experts are 4,608 of the
    checkpoint's tensors, which this loader would otherwise read in full on every rank only to
    discard them as unexpected state.
    """
    if key.startswith("mtp.") or ".mtp." in key:
        return False
    if layer_range is None:
        return True
    if ".layers." in key:
        layer = int(key.split(".layers.", 1)[1].split(".", 1)[0])
        return layer_range[0] <= layer < layer_range[1]
    if key == "lm_head.weight" or key.startswith("lm_head.") or ".lm_head." in key:
        return False
    if ".embed_tokens." in key or key.startswith("embed_tokens."):
        return include_embed
    if key.endswith((".norm.weight", ".norm.bias")) or key.startswith("norm."):
        return include_norm
    return True


def load_quantized_split(
    model_dir,
    dtype,
    *,
    compress_trunk=False,
    layer_range=None,
    include_embed=True,
    include_norm=True,
    expert_range=None,
):
    """Architecture twin + expert store for a quantized (MoE) checkpoint — SHARD-STREAMED.

    The serving load path for a quantized checkpoint whose experts must stay compressed
    (native MXFP4 gpt-oss, or the FP8 DeepSeek / Laguna per-expert-module lineage):
    never materializes the whole dequantized dict (a 20B checkpoint dequantizes to ~42 GB of
    host values — the whole-dict ``load_dequantized_state_dict`` OOMs a 60 GB box). Instead:

    - The twin is built from config alone on the META device (weights never read at trace;
      the experts' would-be random init never materializes).
    - The DENSE trunk (attention projections + biases, norms, router, embeddings, lm_head)
      streams per shard: fp8 weights dequantize by their ``<key>_scale`` partner, everything
      casts to ``dtype``, and the tensors attach via ``load_state_dict(assign=True)``. The
      expert params are skipped — they stay meta on the twin.
    - The EXPERT tensors are collected into a per-layer store keyed by the expert program's
      INPUT names: FP8 weights as raw bits plus their f32 scale tensors, or MXFP4 blocks
      plus their uint8 E8M0 scales (the runner spells reconstruction in-graph and uploads
      one-byte storage instead of dense weights); biases remain ``dtype`` value tensors.
      Native MXFP4 serving requires every routed-expert layer to use that storage; a config
      that skips any routed experts is rejected instead of mixing formats in one store. Per-expert checkpoint modules
      stack into the same E-leading inputs, concatenating gate and up along the output axis.
      An interleaved gate/up layout de-interleaves here — bits, scale and bias alike
      (:func:`deinterleave_gate_up`).

    EXL3 (``fmt == "exl3"``) follows the same split with the trellis format's own shapes, while
    every routed expert keeps its PACKED CODES. Experts arrive as per-expert modules, so each
    ``(layer, projection, leaf)`` triple stacks into one E-leading tensor: ``w_gate`` /
    ``w_up`` / ``w_down`` (int16 codes) plus each one's ``_suh`` (128-blocked) and ``_svh``
    (sliced to the logical out extent) channel vectors — exactly the shapes
    ``spell_trellis_inputs`` declares. Gate and up stay SEPARATE (their ``suh`` differ).

    The dense TRUNK has two lanes, and ``compress_trunk`` picks between them:

    - **default (correctness)** — every coded trunk linear DECODES to values here, so the twin
      is a self-contained eager reference and any consumer that binds off the module gets real
      weights. Costs the full decoded footprint (GLM-4.5-Air: ~14 GiB).
    - **``compress_trunk=True`` (the serving lane)** — a coded trunk linear is left UNDECODED
      and its twin parameter is an uninitialized placeholder at the declared shape; the trunk
      keeps its codes and the caller re-sources those constants from the checkpoint itself
      (``serving/gen_runner.py`` retargets each traced constant to its checkpoint key and lets
      ``spell_trellis_constants`` fire). Nothing may read those parameters' VALUES — the store's
      ``"trunk"`` field says which lane produced the twin.

    ``layer_range=(start, end)`` restricts checkpoint reads to one pipeline stage's absolute
    decoder interval. ``include_embed`` and ``include_norm`` assign the two boundary tensors;
    all unowned parameters remain meta and must never be read by that stage.

    Returns ``(model, expert_store)`` with ``expert_store = {"fmt": "mxfp4" | "f8e4m3" | "exl3" | None,
    "layers": {layer_index: {input_name: tensor}}}`` (``fmt`` None = experts unquantized), plus
    ``"codebooks": {layer_index: {input_name: cb}}`` on the EXL3 path (the codebook id the
    speller stamps on each decode), ``"dir"`` (the resolved checkpoint directory) and
    ``"trunk"`` (``"values"`` or ``"codes"``).
    """
    from pathlib import Path  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    from emmy.compiler.loader.exl3 import decode_trellis, fold_hadamard  # noqa: PLC0415
    from emmy.compiler.loader.quant import (  # noqa: PLC0415
        _EXL3_SIBLING_LEAVES,
        _awq_quant_config,
        _exl3_codebook,
        _exl3_quant_config,
        _fp8_quant_config,
        _is_skipped,
        _mxfp4_quant_config,
        _skip_patterns,
        dequantize,
        dequantize_awq4,
        scale_is_reciprocal,
    )
    from emmy.compiler.loader.safetensors import _build_index  # noqa: PLC0415

    model_dir = Path(model_dir)
    config = _auto_config_from_pretrained(model_dir)
    if getattr(config, "quantization_config", None) is not None:
        delattr(config, "quantization_config")
    with torch.device("meta"):
        model = _auto_model_from_config(config)

    mxfp4_qc = _mxfp4_quant_config(model_dir)
    # DeepSeek V4 declares an fp8 TRUNK while storing its routed experts as native MXFP4: the expert
    # storage is named by ``expert_dtype``, not by ``quant_method``.
    native_mxfp4_experts = str(getattr(config, "expert_dtype", "") or "") == "fp4"
    qc = _fp8_quant_config(model_dir) or mxfp4_qc or {}
    awq = _awq_quant_config(model_dir)
    patterns = _skip_patterns(qc)
    exl3 = _exl3_quant_config(model_dir) is not None
    index = _build_index(model_dir)
    rename = _native_checkpoint_renamer(config, index)
    # The module-namespace view of the same index, so sibling lookups (block scales, coded leaves)
    # can be spelled the way the modules name them whatever the checkpoint calls them.
    renamed = {rename(key): key for key in index}
    by_shard: dict[str, list[str]] = {}
    for key, shard in index.items():
        by_shard.setdefault(str(shard), []).append(key)

    torch_f8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    state: dict = {}
    layers_store: dict[int, dict] = {}
    # EXL3: per-expert tensors arrive one module at a time, so collect them keyed by expert
    # index and stack once the whole checkpoint is read.
    per_expert: dict[int, dict[str, dict[int, object]]] = {}
    codebooks: dict[int, dict[str, int]] = {}
    coded_trunk: set[str] = set()  # compress_trunk: the coded trunk weights left undecoded
    fmt: str | None = None
    from contextlib import ExitStack  # noqa: PLC0415

    with ExitStack() as stack:
        handles: dict[str, object] = {}

        def _open(path: str):
            h = handles.get(path)
            if h is None:
                h = handles[path] = stack.enter_context(safe_open(path, framework="pt"))
            return h

        def _sibling(key: str):
            return _open(str(index[key])).get_tensor(key)

        for shard_path in sorted(by_shard):
            owned_keys = [
                key
                for key in sorted(by_shard[shard_path])
                if _quantized_stage_owns(
                    _checkpoint_to_model_key(rename(key)),
                    layer_range,
                    include_embed=include_embed,
                    include_norm=include_norm,
                )
            ]
            if not owned_keys:
                continue
            f = _open(shard_path)
            for k in owned_keys:
                if awq is not None and k.endswith(".qweight"):
                    base = k[: -len(".qweight")]
                    qzeros_key, scales_key = base + ".qzeros", base + ".scales"
                    if qzeros_key not in index or scales_key not in index:
                        raise ValueError(f"AWQ linear {base!r} is missing qzeros or scales")
                    model_key = _checkpoint_to_model_key(rename(base + ".weight"))
                    if compress_trunk:
                        coded_trunk.add(model_key)
                    else:
                        values = dequantize_awq4(
                            f.get_tensor(k).numpy(),
                            _sibling(qzeros_key).numpy(),
                            _sibling(scales_key).numpy(),
                            int(awq.get("group_size", awq.get("q_group_size", -1))),
                        ).T
                        state[model_key] = torch.from_numpy(values).to(dtype)
                    fmt = "awq4"
                    continue
                if awq is not None and k.endswith((".qzeros", ".scales")):
                    base = k.rsplit(".", 1)[0]
                    if base + ".qweight" in index:
                        continue
                slot = _expert_slot(rename(k))
                if slot is not None:
                    layer, name, expert = slot
                    if expert is not None and expert_range is not None:
                        if not expert_range[0] <= expert < expert_range[1]:
                            continue  # another tensor-parallel rank owns this expert; never read its bytes
                        # The store's expert axis is RANK-LOCAL: a shard stacks its own experts from
                        # index 0, and the router maps global selections onto that axis at dispatch.
                        expert -= expert_range[0]
                    t = f.get_tensor(k)
                    if native_mxfp4_experts:
                        # The published MXFP4 dialect: ``I8 [out, in/2]`` nibble pairs beside
                        # ``F8_E8M0 [out, in/32]`` exponents. Both are raw byte carriers, so VIEW
                        # (never cast) and give the blocks the ``(out, groups, 16)`` shape the
                        # decode contract declares.
                        fmt = "mxfp4"
                        t = t.view(torch.uint8)
                        if not name.endswith("_scale"):
                            t = t.reshape(t.shape[0], t.shape[1] // 16, 16)
                    elif mxfp4_qc is not None and name in {"w_gate_up", "w_down", "w_gate_up_scale", "w_down_scale"}:
                        if t.dtype != torch.uint8:
                            raise ValueError(
                                f"native MXFP4 serving does not support an unconverted routed expert: "
                                f"tensor {k!r} must be uint8 blocks/scales, got {t.dtype}"
                            )
                        fmt = "mxfp4"
                    elif t.dtype in torch_f8:
                        fmt = "f8e4m3" if t.dtype == torch.float8_e4m3fn else "f8e5m2"
                        t = t.view(torch.uint8)
                    elif name.endswith("_scale"):
                        t = t.float()  # bf16-stored scales read as f32 values (the loader convention)
                    elif t.dtype != torch.int16:  # trellis codes carry raw int16 words
                        t = t.to(dtype)
                    if expert is None:
                        layers_store.setdefault(layer, {})[name] = t
                    else:
                        if t.dtype == torch.int16 or name.endswith(("_suh", "_svh")):
                            fmt = "exl3"
                        per_expert.setdefault(layer, {}).setdefault(name, {})[expert] = t
                        if t.dtype == torch.int16:
                            codebooks.setdefault(layer, {})[name] = _exl3_codebook(index, k[: -len(".trellis")])
                    continue
                if exl3 and k.rsplit(".", 1)[-1] in _EXL3_SIBLING_LEAVES and k[: k.rfind(".")] + ".trellis" in index:
                    continue  # channel vectors + codebook markers: consumed by their module's trellis decode
                if exl3 and k.endswith(".trellis"):
                    base = k[: -len(".trellis")]
                    if base + ".suh" not in index or base + ".svh" not in index:
                        logger.warning("EXL3 linear %s: no suh/svh channel vectors; left undecoded", base)
                        continue
                    if compress_trunk:
                        coded_trunk.add(_checkpoint_to_model_key(rename(base + ".weight")))  # placeholder; real bytes stay coded
                        continue
                    w_hat = decode_trellis(f.get_tensor(k).numpy(), _exl3_codebook(index, base))
                    w = fold_hadamard(w_hat, _sibling(base + ".suh").numpy(), _sibling(base + ".svh").numpy()).T
                    state[_checkpoint_to_model_key(rename(base + ".weight"))] = torch.from_numpy(w).to(dtype)
                    continue
                # Sibling pairing runs in the MODULE namespace: a natively named checkpoint spells the
                # block scale ``.scale``, which only becomes the ``<weight>_scale`` sibling the fp8
                # dequant looks for after renaming. Matching raw keys here would leave every fp8 trunk
                # weight silently unscaled.
                mk = rename(k)
                if mk.endswith(("_scale", "_scale_inv")) and mk[: mk.rfind("_scale")] in renamed:
                    continue  # consumed by its base weight's dequant
                t = f.get_tensor(k)
                if t.dtype in torch_f8:
                    scale_key = next((c for c in (mk + "_scale", mk + "_scale_inv") if c in renamed), None)
                    if scale_key is not None and not _is_skipped(k, patterns):
                        scale_key = renamed[scale_key]
                        s = _open(str(index[scale_key])).get_tensor(scale_key)
                        vals = dequantize(t.float().numpy(), s.float().numpy(), inverse=scale_is_reciprocal(scale_key))
                        state[_checkpoint_to_model_key(rename(k))] = torch.from_numpy(vals).to(dtype)
                        continue
                    t = t.float()  # unpaired / skipped fp8: exact value decode, no scale
                state[_checkpoint_to_model_key(rename(k))] = t.to(dtype) if t.is_floating_point() else t

    # POP per layer: the stacked tensors are a full second copy of the expert bytes, so holding the
    # per-expert dict alive across the whole loop peaks at 2× (GLM-4.5-Air: 50 GiB, which no 60 GB
    # box survives). Dropping each layer's sources as it stacks keeps the peak at one layer over.
    for layer in sorted(per_expert):
        by_name = per_expert.pop(layer)
        stack = _stack_exl3_experts if any(t.dtype == torch.int16 for d in by_name.values() for t in d.values()) else _stack_expert_modules
        layers_store.setdefault(layer, {}).update(stack(layer, by_name, model))

    # De-interleave the gate/up family once, so the wrapper's chunk-half split holds.
    trunk = getattr(model, "model", model)
    trunk = getattr(trunk, "language_model", trunk)
    interleaved = False
    for block in trunk.layers:
        parts = moe_block_parts(block.mlp) if hasattr(block, "mlp") else None
        if parts is not None:
            _, interleaved, _ = moe_expert_layout(parts[1])
            break
    if interleaved:
        for store in layers_store.values():
            if fmt == "mxfp4":
                # Packed MXFP4 uses (E, out, groups[, 16]); its interleaved output
                # axis is dim 1, unlike the logical/FP8 tensors whose output is last.
                for name in ("w_gate_up", "w_gate_up_scale"):
                    if name in store:
                        t = store[name]
                        store[name] = torch.cat([t[:, 0::2], t[:, 1::2]], dim=1).contiguous()
                if "b_gate_up" in store:
                    store["b_gate_up"] = deinterleave_gate_up(store["b_gate_up"])
            else:
                for name in ("w_gate_up", "w_gate_up_scale", "b_gate_up"):
                    if name in store:
                        store[name] = deinterleave_gate_up(store[name])

    if coded_trunk:
        # UNINITIALIZED placeholders, deliberately: the twin needs a real tensor at the declared
        # shape for the trace, but reading one is a bug (the values live in the checkpoint's
        # codes). ``torch.empty`` keeps the pages untouched, so a 14 GiB decoded trunk costs no
        # resident host memory here.
        declared = model.state_dict()
        for key in coded_trunk:
            param = declared.get(key)
            if param is not None:
                state[key] = torch.empty(param.shape, dtype=dtype, device="cpu")

    _trim_padded_weights(model, state)  # EXL3 pads both dims of every coded linear to 128
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        logger.warning("quantized split load: %d unexpected checkpoint tensors (e.g. %s)", len(unexpected), unexpected[0])
    # DeepSeek/GLM/Laguna routers register this correction as a buffer.  Leaving it META when
    # the physical checkpoint is incomplete is not a safe zero default: it changes expert
    # selection and fails later with an opaque META-to-CUDA error.  The real Laguna EXL3
    # checkpoint stores the source-layout ``mlp.experts`` spelling, translated above; check the
    # actual load result rather than the logical quantization sidecar (which need not list it).
    owned_missing = [m for m in missing if _quantized_stage_owns(m, layer_range, include_embed=include_embed, include_norm=include_norm)]
    missing_routing_biases = [m for m in owned_missing if m.endswith(".mlp.gate.e_score_correction_bias")]
    if missing_routing_biases:
        sample = missing_routing_biases[0]
        raise ValueError(
            f"quantized split checkpoint is incomplete: missing routing correction bias {sample!r} "
            f"(expected checkpoint alias {sample.replace('.mlp.gate.', '.mlp.experts.')!r})"
        )
    missing_dense = [m for m in owned_missing if _expert_slot(m) is None]
    if missing_dense:
        logger.info("quantized split load: %d module tensors not in checkpoint (e.g. %s)", len(missing_dense), missing_dense[0])
    _apply_exl3_laguna_routed_scale(model, exl3=exl3)
    model.tie_weights()
    model.eval()
    return model, {
        "fmt": fmt,
        "layers": layers_store,
        "codebooks": codebooks,
        "dir": str(model_dir),
        "trunk": "codes" if coded_trunk else "values",
    }


def _trim_padded_weights(model, state: dict) -> None:
    """Slice encode-padded weight values back to the module's declared shape, in place.

    EXL3 pads both dims of every quantized linear to multiples of 128 at encode time
    (GLM-4.5-Air: ``intermediate_size`` 10944 → 11008), so a decoded state value can
    overhang the module's parameter. Slicing the leading extents is exactly the reference
    math (zero-padded activations in, sliced outputs). Guarded
    tightly: only when every overhanging dim is exactly the declared dim's roundup to 128 —
    anything else stays as-is for ``load_state_dict`` to report."""
    for key, param in model.state_dict().items():
        t = state.get(key)
        if t is None or t.shape == param.shape or t.dim() != param.dim():
            continue
        if all(td == -(-pd // 128) * 128 or td == pd for td, pd in zip(t.shape, param.shape, strict=True)):
            state[key] = t[tuple(slice(0, pd) for pd in param.shape)].contiguous()


def _pack_expert_state(model, state: dict) -> None:
    """Pack per-expert checkpoint weights into the transformers-v5 3-D expert params, in place.

    The DeepSeek/GLM checkpoint lineage stores each expert as its own module
    (``…experts.E.{gate,up,down}_proj.weight``), while the v5 experts module declares the
    E-stacked 3-D params (``…experts.gate_up_proj`` ``(E, 2*I, H)`` / ``…experts.down_proj``
    ``(E, H, I)``) — ``from_pretrained`` applies the hub conversion mapping, but a state dict
    built by the loaders must pack here. Per-expert ``nn.Linear`` weights are the ``(out,
    in)`` orientation by construction, so the packing is deterministic: stack, with gate/up
    concatenated along the out axis (the ``chunk(2)`` halves). Fires only when the model
    expects a packed param whose per-expert sources are all present in ``state``; a shape
    mismatch raises rather than loading a silently wrong twin."""
    import torch  # noqa: PLC0415

    for key, param in model.state_dict().items():
        m = re.match(r"(.*\.experts)\.(gate_up_proj|down_proj)$", key)
        if m is None or key in state:
            continue
        base, leaf = m.groups()
        n_experts = param.shape[0]
        parts = []
        consumed: list[str] = []
        for e in range(n_experts):
            names = [f"{base}.{e}.{p}.weight" for p in (("gate_proj", "up_proj") if leaf == "gate_up_proj" else ("down_proj",))]
            halves = [state.get(nm) for nm in names]
            consumed += names
            parts.append(None if any(h is None for h in halves) else halves[0] if len(halves) == 1 else torch.cat(halves, dim=0))
        if any(p is None for p in parts):
            continue  # not the per-expert layout (or an incomplete checkpoint) — leave for load_state_dict to report
        packed = torch.stack(parts, dim=0)
        if packed.shape != param.shape:
            raise ValueError(f"expert packing for {key}: packed shape {tuple(packed.shape)} != expected {tuple(param.shape)}")
        for nm in consumed:
            del state[nm]
        state[key] = packed


def _trim_padded_weights(model, state: dict) -> None:
    """Slice encode-padded weight values back to the module's declared shape, in place.

    EXL3 pads both dims of every quantized linear to multiples of 128 at encode time
    (GLM-4.5-Air: ``intermediate_size`` 10944 → 11008), so a decoded state value can
    overhang the module's parameter. Slicing the leading extents is exactly the reference
    math (zero-padded activations in, sliced outputs). Guarded
    tightly: only when every overhanging dim is exactly the declared dim's roundup to 128 —
    anything else stays as-is for ``load_state_dict`` to report."""
    for key, param in model.state_dict().items():
        t = state.get(key)
        if t is None or t.shape == param.shape or t.dim() != param.dim():
            continue
        if all(td == -(-pd // 128) * 128 or td == pd for td, pd in zip(t.shape, param.shape, strict=True)):
            state[key] = t[tuple(slice(0, pd) for pd in param.shape)].contiguous()


def _pack_expert_state(model, state: dict) -> None:
    """Pack per-expert checkpoint weights into the transformers-v5 3-D expert params, in place.

    The DeepSeek/GLM checkpoint lineage stores each expert as its own module
    (``…experts.E.{gate,up,down}_proj.weight``), while the v5 experts module declares the
    E-stacked 3-D params (``…experts.gate_up_proj`` ``(E, 2*I, H)`` / ``…experts.down_proj``
    ``(E, H, I)``) — ``from_pretrained`` applies the hub conversion mapping, but a state dict
    built by the loaders must pack here. Per-expert ``nn.Linear`` weights are the ``(out,
    in)`` orientation by construction, so the packing is deterministic: stack, with gate/up
    concatenated along the out axis (the ``chunk(2)`` halves). Fires only when the model
    expects a packed param whose per-expert sources are all present in ``state``; a shape
    mismatch raises rather than loading a silently wrong twin."""
    import torch  # noqa: PLC0415

    for key, param in model.state_dict().items():
        m = re.match(r"(.*\.experts)\.(gate_up_proj|down_proj)$", key)
        if m is None or key in state:
            continue
        base, leaf = m.groups()
        n_experts = param.shape[0]
        parts = []
        consumed: list[str] = []
        for e in range(n_experts):
            names = [f"{base}.{e}.{p}.weight" for p in (("gate_proj", "up_proj") if leaf == "gate_up_proj" else ("down_proj",))]
            halves = [state.get(nm) for nm in names]
            consumed += names
            parts.append(None if any(h is None for h in halves) else halves[0] if len(halves) == 1 else torch.cat(halves, dim=0))
        if any(p is None for p in parts):
            continue  # not the per-expert layout (or an incomplete checkpoint) — leave for load_state_dict to report
        packed = torch.stack(parts, dim=0)
        if packed.shape != param.shape:
            raise ValueError(f"expert packing for {key}: packed shape {tuple(packed.shape)} != expected {tuple(param.shape)}")
        for nm in consumed:
            del state[nm]
        state[key] = packed


def load_quantized_twin(model_dir, dtype):
    """Architecture twin of a quantized checkpoint, carrying the DEQUANTIZED real weights.

    A quantized checkpoint cannot go through ``from_pretrained`` as-is —
    transformers would engage its own quantizer machinery (or reject the
    scheme outright: EXL3), and the trace is quantization-blind (quantization
    is a property of the checkpoint, not the architecture; see the FP8 plan).
    So: build the plain architecture from config with ``quantization_config``
    stripped, then load the checkpoint's tensors with every quantized weight
    decoded (``loader.quant.load_dequantized_state_dict`` — FP8 scale pairs,
    native MXFP4 blocks/scales, and EXL3 trellis siblings alike) — the returned module is both the trace
    subject and the eager / accuracy reference. Per-expert checkpoint weights
    pack into the v5 3-D expert params (:func:`_pack_expert_state`).
    ``strict=False`` tolerates non-persistent buffers absent from the
    checkpoint (rotary ``inv_freq``); ``tie_weights()`` re-asserts tied
    embeddings after the load (see ``binder.py``'s
    state_dict-vs-named_parameters note).
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from emmy.compiler.loader.quant import load_dequantized_state_dict  # noqa: PLC0415

    config = _auto_config_from_pretrained(model_dir)
    if getattr(config, "quantization_config", None) is not None:
        delattr(config, "quantization_config")
    # Construct directly in the trace dtype.  Building in the framework default
    # (normally fp32) and converting afterwards briefly holds both copies; that is
    # enough to OOM even very large-memory hosts on checkpoints such as DeepSeek V4.
    model = _auto_model_from_config(config, dtype=dtype)
    state: dict = {}
    for k, v in load_dequantized_state_dict(model_dir).items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        state[k] = t.to(dtype) if t.is_floating_point() else t
    _trim_padded_weights(model, state)
    _pack_expert_state(model, state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        logger.warning("quantized twin: %d unexpected checkpoint tensors (e.g. %s)", len(unexpected), unexpected[0])
    if missing:
        logger.info("quantized twin: %d module tensors not in checkpoint (e.g. %s)", len(missing), missing[0])
    model.tie_weights()
    model.eval()
    return model


def load_architecture_trace_twin(model_id_or_path, dtype, layer: int, *, revision: str | None = None):
    """Config-only architecture twin for a selected-layer ``emmy trace``.

    Build the complete module hierarchy on ``meta`` so model weights are never
    downloaded or allocated.  For supported packed MoE layers, replace routing
    with one representative expert *before* ``to_empty`` materializes the selected
    block.  This ordering is important for very wide expert tables: Laguna has 256
    experts per sparse block, while inventory tracing needs only one expert's
    tensor algebra.

    ``revision`` is kept separate from ``model_id_or_path`` so Hub revision pins
    use the same config-only path as ordinary repository IDs.
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from transformers import AutoConfig, AutoModelForCausalLM  # noqa: PLC0415

    config_kwargs = {} if revision is None else {"revision": revision}
    try:
        config = AutoConfig.from_pretrained(model_id_or_path, **config_kwargs)
    except ValueError as exc:
        if "trust_remote_code" not in str(exc):
            raise
        config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True, **config_kwargs)
    if getattr(config, "quantization_config", None) is not None:
        delattr(config, "quantization_config")

    with torch.device("meta"):
        try:
            model = AutoModelForCausalLM.from_config(config, dtype=dtype)
        except ValueError as exc:
            if "trust_remote_code" not in str(exc):
                raise
            model = AutoModelForCausalLM.from_config(config, dtype=dtype, trust_remote_code=True)

    decoder = None
    for _name, module in model.named_modules():
        if isinstance(getattr(module, "layers", None), nn.ModuleList) and hasattr(module, "rotary_emb"):
            decoder = module
    if decoder is None:
        raise ValueError(f"could not locate a text decoder in {type(model).__name__}")
    if not 0 <= layer < len(decoder.layers):
        raise ValueError(f"layer {layer} not found (model has {len(decoder.layers)} layers)")

    block = decoder.layers[layer]
    # Drop the packed all-expert parameters while they are still meta tensors.
    # ``to_empty`` below then allocates only representative-expert + shared-expert
    # storage for supported MoE blocks, rather than every expert in the source.
    replaced = replace_moe_with_traceable_expert(block)
    if getattr(config, "model_type", None) == "deepseek_v4" and not replaced:
        raise NotImplementedError(
            f"DeepSeek V4 layer {layer} requires confirmed representative routed-expert replacement for architecture tracing"
        )
    block.to_empty(device="cpu")

    # Rotary buffers are non-persistent in transformers and the decoder-level
    # module is outside the selected block. Reconstruct it directly on CPU.
    rotary_type = type(decoder.rotary_emb)
    decoder.rotary_emb = rotary_type(decoder.config)
    # Older remote Laguna implementations use a separate sliding-window rotary
    # module. Rebuild it too when present so tracing never touches a meta buffer.
    if getattr(decoder, "swa_rotary_emb", None) is not None:
        decoder.swa_rotary_emb = type(decoder.swa_rotary_emb)(decoder.config)
    model.eval()
    return model


def load_quantized_trace_twin(model_dir, dtype, layer: int | None):
    """Shape-only architecture twin for ``emmy trace`` on a quantized checkpoint.

    A golden trace persists frontend programs and target shapes; it does not persist
    parameter values.  Materializing every decoded checkpoint tensor is therefore
    unnecessary for a single-layer inventory and can require multiple terabytes for
    models such as DeepSeek V4.  Build the complete architecture on ``meta`` so the
    requested layer keeps its original index/configuration, then materialize only that
    block.  The checkpoint-backed quantization speller still runs after export.

    Whole-model tracing retains :func:`load_quantized_twin`, because every layer is
    executed and an architecture-only materialization would no longer be bounded.
    """
    if layer is None:
        return load_quantized_twin(model_dir, dtype)
    return load_architecture_trace_twin(model_dir, dtype, layer)


def load_quantized_layer_twin(model_dir, dtype, layer: int):
    """Value-correct, shard-streamed twin for one quantized decoder layer.

    ``emmy run MODEL --layer N`` needs real eager-reference values, unlike the
    shape-only inventory path, but decoding the complete checkpoint before
    selecting one layer wastes almost all work and memory. Reuse the serving
    split loader with one stage interval, then reconstruct decoder-level
    non-persistent rotary modules on CPU exactly as the architecture twin does.
    """
    model, store = load_quantized_split(
        model_dir,
        dtype,
        layer_range=(layer, layer + 1),
        include_embed=False,
        include_norm=False,
    )
    if store.get("fmt") == "mxfp4":
        _materialize_mxfp4_expert_store(model, store["layers"], dtype)
    decoder = find_text_decoder(model)
    rotary_type = type(decoder.rotary_emb)
    decoder.rotary_emb = rotary_type(decoder.config)
    if getattr(decoder, "swa_rotary_emb", None) is not None:
        decoder.swa_rotary_emb = type(decoder.swa_rotary_emb)(decoder.config)
    model.eval()
    return model
