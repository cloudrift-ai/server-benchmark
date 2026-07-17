"""``EmmyGenModel`` — the vLLM out-of-tree **generative** model class (Phase 3).

Serve a decoder-only chat model (Qwen3 / Llama / Gemma-3/4) through emmy-compiled per-layer
kernels with vLLM owning the API / sampler / scheduler / paged KV-cache. Gemma's per-layer
sliding/global attention rides vLLM's `per_layer_sliding_window` + a per-layer-type RoPE:

    vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --runner generate \\
      --dtype float16 --hf-overrides '{"architectures":["EmmyGenModel"]}' \\
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16]}'

(the whole-step decode-capture default `emmy serve --generate` assembles; add `--enforce-eager`
instead to serve eager — the runner's `run_device` is capture-aware either way).

NOT ``IsAttentionFree``: it constructs real vLLM ``Attention`` layers (one per decoder
layer, unique ``prefix``) so vLLM allocates a KV-cache spec and runs paged attention. All
weight-bearing **trunk** compute (embed + per-layer pre/post + final norm) lives in the
emmy ``EmmyGenRunner``; vLLM owns only ``lm_head`` (loaded via ``load_weights``)
and applies RoPE through a ``get_rope`` module the model builds (a bare ``Attention`` does
none). ``forward`` brackets each vLLM attention call with two emmy replays (``pre`` /
``post``); RoPE is applied between ``pre`` and ``self.attn`` (A2).

Numpy host I/O at the runner boundary (the per-layer host-sync interleave — Top risk #1);
the device zero-copy path is the Phase-4 optimization.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from emmy import config as emmy_config  # aliased: `config` is the HF config in this module
from emmy.serving.gen_runner import EmmyGenRunner
from emmy.serving.vllm_model import _trunk_dtype_str

logger = logging.getLogger(__name__)


def _build_rotary(config, head_dim, max_position):
    """Construct the model's RoPE the way stock vLLM does — applying each architecture's
    default-theta mutation first (Qwen3 → 1e6, else a missing rope_theta silently falls
    back to 10000 → wrong logits)."""
    try:
        from vllm.transformers_utils.config import set_default_rope_theta

        if getattr(config, "model_type", None) == "qwen3":
            set_default_rope_theta(config, default_theta=1000000)
    except Exception:  # noqa: BLE001 — older vLLM without the helper; config carries theta already
        pass
    return get_rope(
        head_dim,
        max_position=max_position,
        rope_parameters=getattr(config, "rope_parameters", None),
        is_neox_style=True,
    )


def _build_rotaries(config, runner, n_layers, max_position):
    """One RoPE module per decoder layer, built at the LAYER's head_dim. Homogeneous models
    (Qwen3 / Llama) share a single rotary across all layers; Gemma-3/4 keys RoPE on the layer's
    attention type (local θ for ``sliding_attention``, global θ + partial/proportional for
    ``full_attention``) — and Gemma-4 also uses a larger head_dim on global layers — so one rotary
    is built per layer type at that type's head_dim. ``config.rope_parameters`` is then a dict keyed
    by those layer types, not a flat spec."""
    layer_types = getattr(config, "layer_types", None)
    rope_params = getattr(config, "rope_parameters", None)
    if layer_types and isinstance(rope_params, dict) and set(layer_types) <= set(rope_params):
        by_type: dict = {}
        for i in range(n_layers):
            lt = layer_types[i]
            if lt not in by_type:
                hd = runner.layer_meta(i)[0]
                by_type[lt] = get_rope(hd, max_position=max_position, is_neox_style=True, rope_parameters=rope_params[lt])
        return [by_type[layer_types[i]] for i in range(n_layers)]
    return [_build_rotary(config, runner.layer_meta(0)[0], max_position)] * n_layers


class EmmyGenModel(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        mc = vllm_config.model_config
        # Multimodal wrappers (gemma-4 "unified") nest the text attributes (layer_types,
        # rope_parameters, sliding_window, vocab/hidden size, final softcap) under ``text_config``.
        config = getattr(mc.hf_config, "text_config", mc.hf_config)
        self.config = config
        self.dtype = mc.dtype

        # Per-layer sliding/global attention (Gemma-3/4) IS supported: each vLLM Attention below
        # gets its layer's window and each layer its own-theta RoPE. A single uniform sliding
        # window (Qwen2-style ``use_sliding_window``) and dual-chunk are NOT wired — reject those.
        layer_types = getattr(config, "layer_types", None)
        if getattr(config, "use_sliding_window", False) and getattr(config, "sliding_window", None):
            raise NotImplementedError("EmmyGenModel: uniform sliding-window attention is not supported")
        if getattr(config, "dual_chunk_attention_config", None):
            raise NotImplementedError("EmmyGenModel: dual-chunk attention is not supported")

        # The flattened width T = num_tokens is the SUM of newly-scheduled tokens across all
        # requests per step (continuous batching), bounded by max_num_batched_tokens — NOT
        # max_model_len. It must stay within the compiler's dynamic-dim / RoPE-buffer cap.
        from emmy.compiler.trace.dynamic import DYNAMIC_DIM_MAX

        max_batched = vllm_config.scheduler_config.max_num_batched_tokens
        if max_batched and max_batched > DYNAMIC_DIM_MAX:
            raise ValueError(
                f"max_num_batched_tokens={max_batched} exceeds DYNAMIC_DIM_MAX ({DYNAMIC_DIM_MAX}); "
                f"serve with --max-num-batched-tokens {DYNAMIC_DIM_MAX} or lower"
            )

        self.runner = EmmyGenRunner.create(
            model_id=mc.model, dtype_str=_trunk_dtype_str(mc.dtype), decode_bucket=emmy_config.gen_decode_bucket()
        )
        n_layers = self.runner.num_layers

        sliding_window = getattr(config, "sliding_window", None)

        def _layer_window(i):
            # Gemma sliding layers get the config window; global layers (and homogeneous
            # full-causal models with no layer_types) get None → plain causal.
            if layer_types is not None and layer_types[i] == "sliding_attention":
                return sliding_window
            return None

        # One real vLLM Attention per layer — unique prefix (vLLM keys static_forward_context /
        # cache-spec discovery by it and rejects duplicates). No weights. per_layer_sliding_window
        # makes vLLM's paged attention window that layer (and size its KV-cache spec accordingly).
        # Dims come from the runner PER LAYER: Gemma-4's global layers use a larger head_dim than
        # its sliding ones, so a single (num_heads, head_dim) would misshape the global layers.
        attn = []
        for i in range(n_layers):
            hd, nh, nkv, scaling = self.runner.layer_meta(i)
            attn.append(
                Attention(
                    nh,
                    hd,
                    scaling,
                    num_kv_heads=nkv,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    per_layer_sliding_window=_layer_window(i),
                    prefix=f"{prefix}.layers.{i}.self_attn.attn",
                )
            )
        self.attn = nn.ModuleList(attn)
        # RoPE is NOT in Attention. One module per layer (applied between pre and attn): homogeneous
        # models share a single rotary; Gemma-3/4 keys theta AND head_dim on layer type (local/global).
        max_pos = getattr(config, "max_position_embeddings", 8192)
        self.rotary_emb = nn.ModuleList(_build_rotaries(config, self.runner, n_layers, max_pos))

        # vLLM owns ONLY lm_head; the runner owns embed + the trunk.
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=vllm_config.quant_config, prefix=f"{prefix}.lm_head"
        )
        # Gemma-4 softcaps final logits (``final_logit_softcapping``, e.g. 30.0); other models leave
        # it None (no-op). Passed to the LogitsProcessor exactly as stock vLLM's gemma3 does.
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            scale=getattr(config, "logit_scale", 1.0),
            soft_cap=getattr(config, "final_logit_softcapping", None),
        )

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        device = positions.device
        # clamp guards vLLM's _dummy_run garbage-id profiling batches (out-of-vocab → IndexError).
        ids = input_ids.clamp(0, self.config.vocab_size - 1)
        t = int(ids.shape[0])
        # Decode hot path (T <= bucket): device-resident, no host numpy round-trip (Phase A).
        # Prefill / larger T keeps the host path.
        if self.runner.has_device_decode and 0 < t <= self.runner.decode_bucket:
            return self._forward_device(ids, positions)
        ids_np = ids.cpu().numpy()
        hidden_np = self.runner.embed(ids_np)  # [T, H] numpy
        for layer in range(self.runner.num_layers):
            residual_np = hidden_np
            q_np, k_np, v_np = self.runner.forward_layer_pre(layer, hidden_np, positions)
            q = torch.from_numpy(np.ascontiguousarray(q_np)).to(device)
            k = torch.from_numpy(np.ascontiguousarray(k_np)).to(device)
            v = torch.from_numpy(np.ascontiguousarray(v_np)).to(device)
            q, k = self.rotary_emb[layer](positions, q, k)  # A2: per-layer RoPE (Gemma local/global theta)
            # Some rotary impls (e.g. 0.22's proportional Gemma4 rope) promote to fp32; flash-attn
            # rejects anything but fp16/bf16, so restore the trunk dtype (no-op when already right).
            q, k = q.to(self.dtype), k.to(self.dtype)
            attn_out = self.attn[layer](q, k, v)  # vLLM paged attention (pulls attn_metadata from forward context)
            hidden_np = self.runner.forward_layer_post(layer, attn_out.detach().cpu().numpy(), residual_np)
        hidden_np = self.runner.final_norm(hidden_np)
        return torch.from_numpy(np.ascontiguousarray(hidden_np)).to(device)

    def _forward_device(self, ids, positions):
        """Device-resident decode forward (T <= decode_bucket): q/k/v and attn_out stay CUDA
        tensors through RoPE + vLLM attention — no per-layer numpy↔torch host hop."""
        hidden = self.runner.embed_device(ids)  # [T, H] CUDA
        for layer in range(self.runner.num_layers):
            residual = hidden
            q, k, v = self.runner.forward_layer_pre_device(layer, hidden)
            q, k = self.rotary_emb[layer](positions, q, k)  # A2: per-layer RoPE (Gemma local/global theta)
            q, k = q.to(self.dtype), k.to(self.dtype)  # rotary may promote to fp32; flash-attn needs fp16/bf16
            attn_out = self.attn[layer](q, k, v)  # vLLM paged attention
            hidden = self.runner.forward_layer_post_device(layer, attn_out, residual)
        return self.runner.final_norm_device(hidden)

    def compute_logits(self, hidden_states, *args):
        return self.logits_processor(self.lm_head, hidden_states)

    def embed_input_ids(self, input_ids):
        # vLLM embedding hook → the runner owns embedding; on-device gather (no host hop).
        return self.runner.embed_device(input_ids.clamp(0, self.config.vocab_size - 1))

    def load_weights(self, weights):
        """vLLM owns ONLY ``lm_head`` (the runner already loaded embed + trunk). Load
        ``lm_head.weight`` from the checkpoint; when ``tie_word_embeddings`` the checkpoint
        may carry only the embedding, so accept an ``*embed_tokens.weight`` alias for the head
        (the multimodal 'unified' checkpoint nests it at ``model.language_model.embed_tokens.weight``)."""
        param = self.lm_head.weight
        loader = getattr(param, "weight_loader", default_weight_loader)
        tied = getattr(self.config, "tie_word_embeddings", False)
        loaded: set[str] = set()
        for name, w in weights:
            if name == "lm_head.weight":
                loader(param, w)
                loaded.add("lm_head.weight")
            elif tied and name.endswith("embed_tokens.weight") and "lm_head.weight" not in loaded:
                loader(param, w)
                loaded.add("lm_head.weight")
        return loaded
