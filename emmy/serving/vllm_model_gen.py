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

Speculative decoding (MTP drafter, e.g. gemma-4-assistant) works: vLLM's drafter reaches into
``target.model.embed_tokens`` to share the target's raw embedding with the draft head, so this
class exposes a thin ``.model`` shim over the tied embed/``lm_head`` weight the runner gathers from
(built only when a draft is configured). See ``_SharedRawEmbedding``.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import (
    _encode_layer_name,
    unified_attention_with_output,
    unified_kv_cache_update,
)
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


class _SharedRawEmbedding(nn.Module):
    """The token embedding vLLM's speculative-decoding drafter shares off the target model.

    vLLM's Gemma-4 MTP drafter reaches into ``target.model.embed_tokens``, adopts it as its OWN
    embedding, and then applies a ``* sqrt(hidden)`` normalizer itself (``Gemma4MultiTokenPredictor``)
    — the same ``embed_scale`` the emmy runner folds into its gather table — so this module must
    return the **raw, unscaled** table rows, exactly as stock vLLM's ``VocabParallelEmbedding`` does.

    It carries no weights of its own: gemma-4 ties the embedding to ``lm_head``, and on that tied path
    the emmy runner already gathers from the same ``lm_head.weight`` tensor (``adopt_embed_table``), so
    backing the drafter here shares ONE device table with the target rather than copying ~2 GiB. The
    forward guards that identity: if the runner is not gathering from this same raw tensor (an untied
    target, where ``lm_head`` is the distinct output head), it raises rather than silently feeding the
    draft head an embedding the target never uses.
    """

    def __init__(self, model: EmmyGenModel):
        super().__init__()
        # Not a registered submodule (``object.__setattr__``): the owner already owns this module,
        # so registering it back would make ``named_modules`` recurse through a cycle.
        object.__setattr__(self, "_model", model)

    @property
    def weight(self) -> torch.Tensor:
        return self._model.lm_head.weight

    def forward(self, input_ids):
        # Pure gather, no guards: vLLM torch.compiles the drafter's forward (this module included),
        # and dynamo cannot trace ``data_ptr()`` — the tie check runs once, eagerly, in
        # ``validate_tied`` at the end of ``EmmyGenModel.load_weights``.
        return F.embedding(input_ids.long(), self.weight)

    def validate_tied(self):
        """One-time eager check that the shared head weight IS the table the runner gathers from.

        Same storage, not object identity: ``Tensor.data`` yields a fresh view each access, and the
        runner captured its own view of this tensor at ``adopt_embed_table`` time. Called from
        ``load_weights`` (after the adopt hand-off), never from the compiled forward."""
        w = self.weight
        dev = getattr(self._model.runner, "_embed_weight_dev", None)
        if dev is None or dev.data_ptr() != w.data_ptr():
            raise RuntimeError(
                "EmmyGenModel MTP shim: the target's lm_head weight is not the tied embedding table "
                "the runner gathers from, so sharing it as the draft head's embedding would corrupt "
                "draft tokens. Speculative decoding needs a tied-embedding target (e.g. gemma-4)."
            )


class _EmmyTargetInner(nn.Module):
    """The ``.model`` attribute vLLM's spec-decode drafter expects on a target model.

    Stock vLLM model classes nest their trunk under ``.model`` and the drafter reads
    ``target.model.embed_tokens`` from it to share the embedding. EmmyGenModel keeps its trunk
    (embedding included) inside the emmy runner, not a vLLM-style ``.model`` submodule, so this thin
    container re-exposes just that one attribute — the shim's whole compatibility surface."""

    def __init__(self, embed_tokens: nn.Module):
        super().__init__()
        self.embed_tokens = embed_tokens


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
        decode_bucket = emmy_config.gen_decode_bucket()
        # RIDER HEADROOM: mnbt may sit up to decode_bucket ABOVE the dynamic-dim cap — the
        # chunk-twin + decode-twin row SPLIT covers T in (DYNAMIC_DIM_MAX, DYNAMIC_DIM_MAX
        # + bucket] (``EmmyGenRunner.rider_width``), so a full chunk step keeps carrying
        # its decode riders instead of freezing every decoding request for the chunk.
        rider = max(decode_bucket, 0)
        if max_batched and max_batched > DYNAMIC_DIM_MAX + rider:
            raise ValueError(
                f"max_num_batched_tokens={max_batched} exceeds DYNAMIC_DIM_MAX + decode_bucket "
                f"({DYNAMIC_DIM_MAX} + {rider}); serve with --max-num-batched-tokens {DYNAMIC_DIM_MAX + rider} or lower"
            )

        # ``max_tokens`` sizes the symbolic programs' device buffers at the widest step they
        # serve (the dynamic-dim cap; wider steps are the split's) — the device-resident
        # PREFILL path (``run_device_sym``) needs fixed capacity buffers; without it prefill
        # falls back to the per-layer host numpy hops. Capacity is pinned at the CAP, not at
        # mnbt: the pack key carries `max_tokens`/`prefill_bucket`, so tying them to the
        # scheduler knob would recompile the whole program set (and cold-build twins at
        # unseeded golden widths) every time mnbt is tuned per lane. The prefill-chunk twin
        # defaults to that same width (chunked prefill fills steps to mnbt whenever the
        # queue is deep); EMMY_GEN_PREFILL_BUCKET=0 disables, an explicit value overrides.
        capacity = DYNAMIC_DIM_MAX if max_batched else None
        prefill_bucket = emmy_config.gen_prefill_bucket()
        if prefill_bucket < 0:
            prefill_bucket = capacity or 0
        self.runner = EmmyGenRunner.create(
            model_id=mc.model,
            dtype_str=_trunk_dtype_str(mc.dtype),
            decode_bucket=decode_bucket,
            max_tokens=capacity,
            prefill_bucket=prefill_bucket,
        )
        if max_batched and max_batched > max(self.runner.prefill_capacity, self.runner.prefill_bucket + self.runner.rider_width):
            # The over-cap headroom was granted on the promise of the split; if the twin
            # families it needs failed to build, an over-wide step would silently take the
            # per-layer host numpy path — refuse the boot instead.
            raise ValueError(
                f"max_num_batched_tokens={max_batched} needs the chunk+decode twin split, which is "
                f"unavailable (prefill_bucket={self.runner.prefill_bucket}, rider_width={self.runner.rider_width}); "
                f"serve with --max-num-batched-tokens {self.runner.prefill_capacity} or lower"
            )
        n_layers = self.runner.num_layers

        # AUTHORITATIVE MoE capture guard (the serve command's config probe is best-effort UX
        # only): single-token decode is capture-legal through the runner's FIXED-SLOT tier
        # (``_moe_combine_slots`` — fixed launch set, no host sync), so an MoE boot may keep
        # whole-step decode capture at capture size 1. Every wider decode step still rides the
        # routed dispatch, which host-syncs and cannot be recorded — so capture sizes above 1,
        # or a boot where the fixed-slot tier failed to build (M=1 expert compile failure, or a
        # schedule that stages weights through TMA descriptors), are rejected LOUDLY here.
        # Failing with the real reason beats the cryptic CUDA 'operation not permitted during
        # stream capture' crash vLLM's capture pass would hit later.
        if self.runner._moe is not None and not mc.enforce_eager:
            cg_mode = getattr(vllm_config.compilation_config, "cudagraph_mode", None)
            if cg_mode is None or getattr(cg_mode, "name", str(cg_mode)) != "NONE":
                if not self.runner.has_moe_fixed_slot:
                    raise ValueError(
                        "MoE decode capture needs the fixed-slot expert tier, which is unavailable on this "
                        "boot (the M=1 expert program failed to build, or its schedule stages weights through "
                        "TMA descriptors — see the gen_runner warnings above); serve with --enforce-eager"
                    )
                sizes = getattr(vllm_config.compilation_config, "cudagraph_capture_sizes", None) or []
                over = sorted(s for s in sizes if s > 1)
                if over:
                    raise ValueError(
                        f"MoE decode capture is limited to capture size 1 (the fixed-slot tier covers "
                        f"single-token steps only; wider decode steps run the routed dispatch eager) — "
                        f"capture sizes {over} exceed that; use cudagraph_capture_sizes [1] (the "
                        f"emmy serve --generate default for MoE) or --enforce-eager"
                    )

        sliding_window = getattr(config, "sliding_window", None)

        def _layer_window(i):
            # Gemma sliding layers get the config window; global layers (and homogeneous
            # full-causal models with no layer_types) get None → plain causal.
            if layer_types is not None and layer_types[i] == "sliding_attention":
                return sliding_window
            return None

        # Attention sinks (gpt-oss): a per-layer ``sinks`` weight, shape [num_heads], that the
        # attention softmax treats as one extra always-present logit column. The config carries
        # no flag — sinks exist purely as checkpoint weights — so key on the architecture.
        # Passing ``sinks=`` into ``Attention`` makes vLLM's backend selection sinks-aware
        # (on sm_120 it lands on TRITON_ATTN; FA2 and FlashInfer both reject sinks there); the
        # params are the second vLLM-owned weight beside ``lm_head`` (``load_weights`` claims
        # ``*.self_attn.sinks``).
        self.sinks = None
        if getattr(config, "model_type", None) == "gpt_oss":
            self.sinks = nn.ParameterList(
                [nn.Parameter(torch.empty(self.runner.layer_meta(i)[1], dtype=mc.dtype), requires_grad=False) for i in range(n_layers)]
            )

        # One real vLLM Attention per layer — unique prefix (vLLM keys static_forward_context /
        # cache-spec discovery by it and rejects duplicates). No weights (except sinks, above).
        # per_layer_sliding_window makes vLLM's paged attention window that layer (and size its
        # KV-cache spec accordingly).
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
                    **({"sinks": self.sinks[i]} if self.sinks is not None else {}),
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
        # The checkpoint's generation config may list tokens to suppress outright (gemma-4 lists
        # the mm delimiters '<image|>'/'<audio|>'; a degenerate text prompt can genuinely rank
        # one top-1, and an emitted mm token decodes to empty text). HF generate honors the
        # list; stock vLLM's gemma4 models -inf those ids in compute_logits — mirror that.
        gen_cfg = vllm_config.model_config.try_get_generation_config()
        self._suppress_token_ids = gen_cfg.get("suppress_tokens") if gen_cfg else None
        # ``EMMY_GEN_ALIAS_ATTN`` — write attention output directly into the M=1 post twin's
        # ``attn_out`` input backing (the seam-copy closer; see ``config.gen_alias_attn``).
        self._alias_attn = bool(emmy_config.gen_alias_attn())

        # Speculative-decoding shim: vLLM's MTP drafter (vllm#41745, gemma-4) reaches into
        # ``target.model.embed_tokens`` to share the target's raw embedding with the draft head.
        # The runner keeps the trunk (embedding included) outside a vLLM-style ``.model`` submodule,
        # so expose that one attribute — backed by the tied embed/lm_head weight the runner gathers
        # from, no copy. Only built when a draft is configured; plain serving is untouched. The
        # drafter's ``* sqrt(hidden)`` normalizer only reproduces the target's embed scale when the
        # embedding is tied to ``lm_head``, so reject an untied target early and clearly.
        if getattr(vllm_config, "speculative_config", None) is not None:
            if not getattr(config, "tie_word_embeddings", False):
                raise NotImplementedError(
                    "EmmyGenModel: speculative decoding requires a tied-embedding target so the draft "
                    "head can share the raw embedding table (this checkpoint is untied). Serve without "
                    "--speculative-config, or use a tied target such as gemma-4."
                )
            self.model = _EmmyTargetInner(_SharedRawEmbedding(self))

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        device = positions.device
        # clamp guards vLLM's _dummy_run garbage-id profiling batches (out-of-vocab → IndexError).
        ids = input_ids.clamp(0, self.config.vocab_size - 1)
        t = int(ids.shape[0])
        # Device-resident forward for EVERY width the compiled programs cover: decode
        # (T <= bucket, the captured static twins), prefill / mixed chunked-prefill steps
        # (bucket < T <= prefill_capacity, the symbolic programs via ``run_device_sym``),
        # AND rider-carrying full-chunk steps (prefill_bucket < T <= prefill_bucket +
        # rider_width, the chunk+decode twin row split) — no per-layer host numpy
        # round-trip. The numpy path below survives only as the fallback for a runner built
        # without capacity (the standalone oracle) or an over-coverage T.
        decode_ok = self.runner.has_device_decode and 0 < t <= self.runner.decode_bucket
        prefill_ok = 0 < t <= max(self.runner.prefill_capacity, self.runner.prefill_bucket + self.runner.rider_width)
        if decode_ok or prefill_ok:
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
            attn_out = self._attn_aliased(layer, q, k, v) if self._alias_attn else None  # A4: any tier; the backing router decides
            if attn_out is None:
                attn_out = self.attn[layer](q, k, v)  # vLLM paged attention
            hidden = self.runner.forward_layer_post_device(layer, attn_out, residual)
        return self.runner.final_norm_device(hidden)

    def _attn_aliased(self, layer, q, k, v):
        """vLLM paged attention writing INTO the M=1 post twin's ``attn_out`` input backing —
        ``Attention.forward`` minus its own output allocation, so ``run_device``'s prefix upload
        self-copy-skips (the seam copy drops out of the captured decode graph). Returns ``None``
        to fall back to the ordinary module call (m1 twin absent for this layer, or an attention
        feature this tail doesn't replicate — kv scales / query quant)."""
        attn = self.attn[layer]
        if attn.calculate_kv_scales or getattr(attn, "query_quant", None) is not None:
            return None
        out = self.runner.post_attn_backing(layer, rows=q.shape[0])
        if out is None or out.dtype != q.dtype:
            return None
        query = q.view(-1, attn.num_heads, attn.head_size)
        key = k.view(-1, attn.num_kv_heads, attn.head_size)
        value = v.view(-1, attn.num_kv_heads, attn.head_size_v)
        output = out.view(-1, attn.num_heads, attn.head_size_v)
        kv_dep = None
        update_kv = not attn.attn_backend.forward_includes_kv_cache_update and attn.kv_sharing_target_layer_name is None
        if attn.use_direct_call:
            if update_kv:
                kv_dep = unified_kv_cache_update(key, value, attn.layer_name)
            unified_attention_with_output(query, key, value, output, attn.layer_name, kv_cache_dummy_dep=kv_dep)
        else:
            encoded = _encode_layer_name(attn.layer_name)
            if update_kv:
                kv_dep = torch.ops.vllm.unified_kv_cache_update(key, value, encoded)
            torch.ops.vllm.unified_attention_with_output(query, key, value, output, encoded, kv_cache_dummy_dep=kv_dep)
        return out

    def compute_logits(self, hidden_states, *args):
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None and self._suppress_token_ids:
            logits[:, self._suppress_token_ids] = -float("inf")
        return logits

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
            elif self.sinks is not None and name.endswith(".self_attn.sinks"):
                # gpt-oss attention sinks — the per-layer [num_heads] weight beside lm_head
                # (see __init__). Single GPU: the full copy (stock's tp narrow at tp=1).
                layer = int(name.split(".layers.")[1].split(".")[0])
                self.sinks[layer].data.copy_(w)
                loaded.add(name)
        # Tied checkpoints: the loaded head IS the raw embed table — hand it to the runner and drop
        # the runner's own ~2 GiB folded device copy (uploaded eagerly at construction for the
        # profiling-order contract). empty_cache + the cupy pool trim actually RELEASE the freed
        # blocks to the driver, so vLLM's KV-cache profiling (which runs after load) sees them —
        # this is where the reclaimed memory becomes KV blocks. The runner re-applies the gemma
        # embed_scale at gather; the shared table stays raw (the head must read it unscaled).
        if tied and "lm_head.weight" in loaded and param.data.is_cuda:
            self.runner.adopt_embed_table(param.data, scale=getattr(self.runner, "_embed_scale", 1.0))
            import cupy as cp

            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
        # The spec-decode shim shares this table with the drafter; verify the tie once, here,
        # where the adopt hand-off just happened (the compiled forward carries no guards).
        shim = getattr(self, "model", None)
        if shim is not None:
            shim.embed_tokens.validate_tied()
        return loaded
