"""``EmmyGenModel`` — the vLLM out-of-tree **generative** model class (Phase 3).

Serve a decoder-only chat model (Qwen3 / Llama / gemma-4 text) through emmy-compiled per-layer
kernels with vLLM owning the API / sampler / scheduler / paged KV-cache:

    vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --runner generate --enforce-eager \\
      --dtype float16 --hf-overrides '{"architectures":["EmmyGenModel"]}'

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

from emmy.serving.gen_runner import EmmyGenRunner
from emmy.serving.vllm_model import _trunk_dtype_str

logger = logging.getLogger(__name__)


def _rotate_half(x):
    """HF neox-style rotate-half (transformers' ``rotate_half``) on ``[..., D]``."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


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


class EmmyGenModel(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        mc = vllm_config.model_config
        config = mc.hf_config
        self.config = config
        self.dtype = mc.dtype
        # Multimodal wrappers (gemma-4 unified) nest the decoder config; all attention/geometry
        # fields read from the text config. Text-only models: text_cfg IS config.
        text_cfg = getattr(config, "text_config", None) or config
        self.text_config = text_cfg

        # Per-layer sliding attention is supported via vLLM's ``per_layer_sliding_window``
        # (keyed off ``layer_types``). The legacy flag without layer_types, dual-chunk, and
        # cross-layer KV sharing are not — reject rather than silently miscompute.
        layer_types = getattr(text_cfg, "layer_types", None)
        if getattr(text_cfg, "use_sliding_window", False) and getattr(text_cfg, "sliding_window", None) and not layer_types:
            raise NotImplementedError("EmmyGenModel: legacy use_sliding_window without layer_types is not supported")
        if getattr(text_cfg, "dual_chunk_attention_config", None):
            raise NotImplementedError("EmmyGenModel: dual-chunk attention is not supported")
        if getattr(text_cfg, "num_kv_shared_layers", 0):
            raise NotImplementedError("EmmyGenModel: cross-layer KV sharing (num_kv_shared_layers > 0) is not supported")

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

        self.runner = EmmyGenRunner.create(model_id=mc.model, dtype_str=_trunk_dtype_str(mc.dtype))
        n_layers = self.runner.num_layers
        head_dim = self.runner.head_dim
        self.layer_types = list(layer_types) if layer_types else ["full_attention"] * n_layers
        sliding_window = getattr(text_cfg, "sliding_window", None)

        # One real vLLM Attention per layer — unique prefix (vLLM keys static_forward_context
        # / cache-spec discovery by it and rejects duplicates). No weights. Geometry is
        # PER LAYER (gemma-4 global layers: wider head_dim, fewer KV heads), and sliding
        # layers carry their window so vLLM's paged attention masks them.
        self.attn = nn.ModuleList(
            [
                Attention(
                    nh,
                    d,
                    scaling,
                    num_kv_heads=nkv,
                    per_layer_sliding_window=(sliding_window if self.layer_types[i] == "sliding_attention" else None),
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.layers.{i}.self_attn.attn",
                )
                for i, (d, nh, nkv, scaling) in enumerate(self.runner.attn_metas)
            ]
        )
        # RoPE is NOT in Attention. Uniform models: one shared vLLM get_rope. Per-layer-type
        # rope (gemma-4): the trunk's OWN HF rotary module (handed through the runner) applied
        # per layer_type in forward — faithful to partial/proportional rope by construction.
        if self.runner.rotary_emb is not None:
            import copy

            self.hf_rotary = copy.deepcopy(self.runner.rotary_emb).float()
            self.rotary_emb = None
        else:
            self.hf_rotary = None
            self.rotary_emb = _build_rotary(config, head_dim, getattr(text_cfg, "max_position_embeddings", 8192))

        # vLLM owns ONLY lm_head; the runner owns embed + the trunk.
        self.lm_head = ParallelLMHead(
            text_cfg.vocab_size, text_cfg.hidden_size, quant_config=vllm_config.quant_config, prefix=f"{prefix}.lm_head"
        )
        self.logits_processor = LogitsProcessor(
            text_cfg.vocab_size,
            scale=getattr(text_cfg, "logit_scale", 1.0),
            soft_cap=getattr(text_cfg, "final_logit_softcapping", None),
        )

    def _apply_rope(self, layer, positions, q, k):
        """Rotate flat ``q[T, nh·D]`` / ``k[T, nkv·D]`` for ``layer``. Uniform models ride the
        shared vLLM ``get_rope``; per-layer-type models (gemma-4) apply the trunk's own HF
        rotary per layer_type (HF semantics: cos/sin computed fp32, cast to q dtype, full-width
        multiply with rotate-half)."""
        if self.hf_rotary is None:
            return self.rotary_emb(positions, q, k)
        d, nh, nkv, _ = self.runner.attn_metas[layer]
        t = q.shape[0]
        cos, sin = self.hf_rotary(q, positions.view(1, -1), self.layer_types[layer])  # [1, T, D] in q dtype
        cos = cos.squeeze(0).unsqueeze(1)  # [T, 1, D] — broadcast over heads
        sin = sin.squeeze(0).unsqueeze(1)
        q3 = q.view(t, nh, d)
        k3 = k.view(t, nkv, d)
        q3 = q3 * cos + _rotate_half(q3) * sin
        k3 = k3 * cos + _rotate_half(k3) * sin
        return q3.view(t, nh * d), k3.view(t, nkv * d)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        device = positions.device
        # clamp guards vLLM's _dummy_run garbage-id profiling batches (out-of-vocab → IndexError).
        ids = input_ids.clamp(0, self.text_config.vocab_size - 1)
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
            q, k = self._apply_rope(layer, positions, q, k)  # A2: RoPE applied at the seam
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
            q, k = self._apply_rope(layer, positions, q, k)  # A2: RoPE applied at the seam
            attn_out = self.attn[layer](q, k, v)  # vLLM paged attention
            hidden = self.runner.forward_layer_post_device(layer, attn_out, residual)
        return self.runner.final_norm_device(hidden)

    def compute_logits(self, hidden_states, *args):
        return self.logits_processor(self.lm_head, hidden_states)

    def embed_input_ids(self, input_ids):
        # vLLM embedding hook → the runner owns embedding; on-device gather (no host hop).
        return self.runner.embed_device(input_ids.clamp(0, self.text_config.vocab_size - 1))

    def load_weights(self, weights):
        """vLLM owns ONLY ``lm_head`` (the runner already loaded embed + trunk). Load
        ``lm_head.weight`` from the checkpoint; when ``tie_word_embeddings`` the checkpoint
        may carry only ``embed_tokens.weight``, so accept that alias for the head."""
        param = self.lm_head.weight
        loader = getattr(param, "weight_loader", default_weight_loader)
        tied = getattr(self.config, "tie_word_embeddings", False) or getattr(self.text_config, "tie_word_embeddings", False)
        embed_aliases = (
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            # multimodal wrappers (gemma-4 unified) nest the text embedding
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
        )
        loaded: set[str] = set()
        for name, w in weights:
            if name == "lm_head.weight":
                loader(param, w)
                loaded.add("lm_head.weight")
            elif tied and name in embed_aliases and "lm_head.weight" not in loaded:
                loader(param, w)
                loaded.add("lm_head.weight")
        return loaded
