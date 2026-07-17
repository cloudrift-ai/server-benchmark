"""``EmmyGenRunner`` — per-layer attention-split runner (Phase 2).

Sibling to ``EmmyForwardRunner`` (the embedding runner). Carves SDPA out of every
decoder layer (``build_attention_split_wrapper``), compiles **two programs per layer**
(``pre`` + ``post``) over the flattened ``[num_tokens, H]`` layout with ``num_tokens``
symbolic, and exposes the per-token, everything-but-attention compute:

- ``embed(input_ids) -> hidden[T, H]`` — token embedding lookup (the runner owns embedding).
- ``forward_layer_pre(L, hidden, positions) -> (q, k, v)`` — un-rotated 2-D seam q[T,Hq·D],
  k/v[T,Hkv·D] (RoPE is applied downstream by the caller / vLLM, A2 — ``positions`` unused here).
- ``forward_layer_post(L, attn_out, residual) -> hidden`` — o_proj + residual + post-norm + MLP.
- ``final_norm(hidden) -> hidden``.

The caller stitches attention between ``pre`` and ``post`` (a reference torch SDPA in the
Phase-2 host stitch; vLLM's paged ``Attention`` in Phase 3). I/O: the vLLM plugin runs
device-resident at EVERY width — decode (``T <= bucket``) through the captured static twins,
prefill / chunked-prefill (``bucket < T <= prefill_capacity``) through the symbolic programs'
``run_device_sym`` (grids sized per step, capacity buffers, no per-layer host hop); the host
numpy ``rebind`` path survives for the standalone oracle and as the over-capacity fallback.
NOTE: the per-layer ``CompiledProgram``s share BOTH their
weights (one device buffer per constant, ``_bind_device_constants``) and their activation
buffers + scratch slabs (one ``BufferArena`` per runner) — the footprint no longer scales
with ``num_layers`` (the memory budget the plan flagged, Phase 2 / Top risk #9).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class _Program:
    """One compiled dynamic-``num_tokens`` split subgraph, run via the host ``rebind`` path."""

    def __init__(self, program, input_names, output_names):
        self.program = program
        self.input_names = input_names
        self.output_names = output_names

    def run(self, arrays):
        """arrays: numpy arrays aligned to ``input_names`` (each ``[T, …]``). Returns the
        outputs in ``output_names`` order, sliced to the runtime ``T``."""
        from emmy.compiler.backend.gpu_lock import gpu_lock

        t = arrays[0].shape[0]
        feed = dict(zip(self.input_names, arrays, strict=True))
        with gpu_lock():
            self.program.rebind(feed)  # resolves num_tokens from the input shapes
            self.program.run_once()
            out = self.program.outputs({"num_tokens": t})
        return [out[n] for n in self.output_names]

    def run_device(self, arrays):
        """Device-resident captured-replay twin of :meth:`run` for the **static M=bucket** decode
        programs. ``arrays``: torch CUDA tensors aligned to ``input_names`` (each ``[T, …]``,
        ``T <= bucket``). Uploads the ``T`` real rows into the buffer prefix (device-to-device),
        captures-or-replays the whole-program graph, and returns the outputs as torch CUDA tensors
        sliced to ``T`` — no host round-trip. Stale prefix padding rows are safe (pre/post are
        per-token-independent; only ``[:T]`` is read out). All cupy work runs on torch's current
        stream so the upload, replay and output read stay ordered.

        Under an OUTER capture (vLLM's whole-step decode cudagraph — torch's current stream is
        capturing) the program's own graph machinery is illegal (nested stream capture aborts, and
        a graph launch cannot be recorded), so the raw launch sequence is issued instead
        (``run_once`` — the exact work ``capture_program_graph`` records: prebuilt buffers, no
        allocation, no sync). The outer graph absorbs the launches and the per-call Python
        overhead vanishes at replay."""
        import cupy as cp
        import torch

        from emmy.compiler.backend.gpu_lock import gpu_lock

        t = arrays[0].shape[0]
        with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
            feed = {n: cp.from_dlpack(a.detach().contiguous()) for n, a in zip(self.input_names, arrays, strict=True)}
            self.program.upload_prefix_device(feed)
            if torch.cuda.is_current_stream_capturing():
                self.program.run_once()
            else:
                self.program.capture_program_graph()  # static graph → one cached entry (empty sym_values)
                self.program.replay_program_graph()
            outs = self.program.output_prefix_device()
            return [torch.from_dlpack(outs[n])[:t].clone() for n in self.output_names]

    def run_device_sym(self, arrays):
        """Device-resident twin of :meth:`run` for the SYMBOLIC (prefill) programs at any width
        ``T`` up to the build capacity: size the launch grids to ``T`` (``set_sym_values`` — no
        re-allocation, the buffers stay at capacity), upload the ``T`` real rows into the buffer
        prefix device-to-device, issue the launch sequence on torch's current stream, and return
        the outputs as torch CUDA tensors at their resolved ``T`` shapes — no per-layer host numpy
        round-trip (the pre-device prefill path's ~2×48 ``.cpu()`` hops per step were the TTFT
        wall). No per-T graph capture: at prefill widths the dispatch hides behind the GPU work,
        and chunked-prefill ``T`` varies step to step — a per-T graph cache would re-capture
        constantly. Requires the program to have been built with a ``capacity`` feed
        (:func:`_compile_split`); ``set_sym_values`` raises past it."""
        import cupy as cp
        import torch

        from emmy.compiler.backend.gpu_lock import gpu_lock

        t = arrays[0].shape[0]
        with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
            feed = {n: cp.from_dlpack(a.detach().contiguous()) for n, a in zip(self.input_names, arrays, strict=True)}
            self.program.set_sym_values({"num_tokens": t})
            self.program.upload_prefix_device(feed)
            self.program.run_once()
            outs = self.program.output_prefix_device({"num_tokens": t})
            return [torch.from_dlpack(outs[n]).clone() for n in self.output_names]


def _pad_rows(arr, bucket):
    """Pad axis 0 from ``t`` up to ``bucket`` with zeros. The decode programs are static at
    M=bucket; padding rows are computed then sliced away — safe because pre/post are
    per-token-independent (pointwise + matmul over the hidden axis)."""
    import numpy as np

    t = arr.shape[0]
    if t == bucket:
        return arr
    out = np.zeros((bucket, *arr.shape[1:]), dtype=arr.dtype)
    out[:t] = arr
    return out


def _bind_device_constants(graph, sources, cache):
    """Upload each distinct ``(source_path, load_ops)`` constant ONCE and share the cupy
    array across program builds. The symbolic and decode-bucket twins bind the same
    weights; per-build numpy feeds would upload a second full on-GPU copy of the trunk
    (~2× the weight footprint). ``cache`` must be scoped to one wrapper — param paths
    are wrapper-relative, so a cross-wrapper cache would collide."""
    import cupy as cp

    from emmy.compiler.loader.binder import apply_load_ops

    out = {}
    for nid, op in graph.loadable_constants():
        if op.source_path not in sources:
            continue
        key = (op.source_path, repr(op.load_ops))
        arr = cache.get(key)
        if arr is None:
            arr = cp.asarray(apply_load_ops(sources[op.source_path], op.load_ops))
            cache[key] = arr
        out[nid] = arr
    return out


def _compile_split(wrapper, example_args, argnames, np_dtype, dev_consts=None, arena=None, capacity=None):
    """Trace ``wrapper`` and build a :class:`_Program`. ``argnames`` (a list) ties each named
    arg's axis-0 to a shared symbolic ``num_tokens`` Dim — the **prefill** program (one program,
    any width). ``argnames=None`` traces a **fully static** graph at the example shapes — the
    **decode-bucket** program (efficient at small M; the symbolic program's hint-sized M-tile is
    pathological at decode). ``dev_consts`` (a per-wrapper dict) shares each weight's device
    buffer across the builds that pass the same dict — see :func:`_bind_device_constants`.
    ``arena`` (one per runner) pools the activation buffers + scratch slab across every
    program built with it — layers run sequentially, so N layers hold ~one layer's worth.
    ``capacity`` (symbolic programs only) sizes the BUILD feed's token axis so the device
    buffers hold any step up to it — the :meth:`_Program.run_device_sym` prefill path needs
    fixed capacity buffers (``set_sym_values`` never re-allocates); without it the build feed
    is the example (the host ``rebind`` path re-sizes per call)."""
    import torch

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.trace.torch import trace_module

    dynamic_shapes = None
    if argnames:
        from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

        dynamic_shapes = build_torch_dynamic_shapes(parse_position_specs([f"num_tokens@{n}:0" for n in argnames]))
    graph = trace_module(wrapper, tuple(example_args), dynamic_shapes=dynamic_shapes)
    compiled = CudaBackend(tune_db="auto").compile(graph)

    sources = {}
    for path, t in wrapper.named_parameters(remove_duplicate=False):
        sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
    for path, t in wrapper.named_buffers(remove_duplicate=False):
        sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)

    build_args = example_args
    if capacity is not None and argnames:
        build_args = [torch.zeros((capacity, *a.shape[1:]), dtype=a.dtype) for a in example_args]
    feed = {n: a.detach().cpu().to(torch.float32).numpy().astype(np_dtype) for n, a in zip(compiled.inputs, build_args, strict=True)}
    with gpu_lock():
        if dev_consts is None:
            const_feed = bind_constants(compiled, sources)
        else:
            const_feed = _bind_device_constants(compiled, sources, dev_consts)
        program = CompiledProgram.build(compiled, {**const_feed, **feed}, arena=arena)
    return _Program(program, list(compiled.inputs), list(compiled.outputs))


class EmmyGenRunner:
    def __init__(
        self,
        *,
        embed_weight,
        norm,
        pre,
        post,
        attn_meta,
        np_dtype,
        pre_decode=None,
        post_decode=None,
        decode_bucket=16,
        prefill_capacity=None,
        pre_prefill=None,
        post_prefill=None,
        prefill_bucket=0,
    ):
        self._embed_weight = embed_weight  # numpy [vocab, H]
        self._norm = norm  # torch module
        self._pre = pre  # list[_Program] — symbolic (prefill / any width)
        self._post = post
        self._pre_decode = pre_decode  # list[_Program] — static M=decode_bucket (or None → no bucket)
        self._post_decode = post_decode
        self._decode_bucket = decode_bucket
        self._prefill_capacity = prefill_capacity  # symbolic programs' device-buffer token capacity (None -> host rebind only)
        self._pre_prefill = pre_prefill  # list[_Program] — static M=prefill_bucket chunk twins (or None → symbolic prefill)
        self._post_prefill = post_prefill
        self._prefill_bucket = prefill_bucket
        self._attn_meta = attn_meta  # per-layer list of (head_dim, num_heads, num_kv, scaling)
        # Layer-0 convenience scalars — correct for homogeneous models (Qwen3 / Llama). Gemma-4's
        # global layers differ, so the vLLM model reads per-layer dims via ``layer_meta``.
        self.head_dim, self.num_heads, self.num_kv_heads, self.scaling = attn_meta[0]
        self._np_dtype = np_dtype

    def layer_meta(self, layer: int) -> tuple[int, int, int, float]:
        """Per-layer ``(head_dim, num_heads, num_kv_heads, scaling)``. Not uniform for Gemma-4:
        its global (``full_attention``) layers use ``global_head_dim`` > the sliding layers' head_dim."""
        return self._attn_meta[layer]

    @property
    def num_layers(self) -> int:
        return len(self._pre)

    @property
    def decode_bucket(self) -> int:
        return self._decode_bucket

    @property
    def has_device_decode(self) -> bool:
        """True when the static decode-bucket programs exist → the device-resident decode path
        (``embed_device`` / ``forward_layer_*_device`` / ``final_norm_device``) is available."""
        return self._pre_decode is not None

    @property
    def prefill_capacity(self) -> int:
        """The symbolic programs' device-buffer token capacity — the widest step
        ``forward_layer_*_device`` serves without a host fallback (0 = built without
        capacity, host ``rebind`` only)."""
        return self._prefill_capacity or 0

    @property
    def prefill_bucket(self) -> int:
        """The static prefill-chunk twins' M (0 = twins not built — prefill rides the
        symbolic programs). Chunked prefill fills steps to ``max_num_batched_tokens``
        whenever the queue is deep, so a twin at that width runs exact static grids
        (and captured-replay) on the hot chunk shape."""
        return self._prefill_bucket if self._pre_prefill is not None else 0

    @classmethod
    def create(cls, model_id, *, dtype_str="float16", decode_bucket=16, max_tokens=None, prefill_bucket=0):
        import torch
        from transformers import AutoModelForCausalLM

        logger.info("[gen_runner] loading %s (%s, CPU trace)...", model_id, dtype_str)
        with torch.device("cpu"):
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype_str)).eval()
            return cls.from_model(
                model, dtype_str=dtype_str, decode_bucket=decode_bucket, max_tokens=max_tokens, prefill_bucket=prefill_bucket
            )

    @classmethod
    def from_model(cls, model, *, dtype_str="float16", decode_bucket=16, max_tokens=None, prefill_bucket=0):
        """Build from an already-loaded CausalLM module (the network-free path). ``model``
        must be on CPU for the trace."""
        import numpy as np
        import torch

        from emmy.compiler.trace.huggingface import build_attention_split_wrapper

        dtype = getattr(torch, dtype_str)
        np_dtype = np.dtype(dtype_str)
        trunk = getattr(model, "model", model)
        # Multimodal wrappers (gemma-4 "unified") nest the decoder stack + embed/norm under
        # ``language_model`` and carry the text dims on ``config.text_config``.
        trunk = getattr(trunk, "language_model", trunk)
        layers = trunk.layers
        text_config = getattr(model.config, "text_config", model.config)
        hidden = text_config.hidden_size

        def _meta(attn):
            # Per-layer attention dims. Gemma-4's global layers use a larger head_dim
            # (``global_head_dim``) than its sliding layers, so this is NOT uniform across layers.
            hd = attn.head_dim
            return hd, attn.q_proj.out_features // hd, attn.k_proj.out_features // hd, attn.scaling

        attn_meta = []  # per-layer (head_dim, num_heads, num_kv, scaling)
        from emmy.compiler.backend.cuda.program import BufferArena

        pre_programs, post_programs = [], []
        pre_decode, post_decode = [], []
        pre_prefill, post_prefill = [], []
        decode_ok = decode_bucket and decode_bucket > 0
        # The prefill-chunk twin only pays ABOVE the decode bucket (an equal-or-smaller
        # bucket is fully shadowed by the decode twins' routing).
        prefill_ok = prefill_bucket and prefill_bucket > max(decode_bucket or 0, 0)
        # One arena for every program this runner builds: layers run sequentially, so
        # all layers' activation buffers + scratch slabs share one layer's worth of
        # device memory instead of scaling with num_layers.
        arena = BufferArena()
        for i, block in enumerate(layers):
            meta = _meta(block.self_attn)
            attn_meta.append(meta)
            attn_width = meta[1] * meta[0]  # this layer's num_heads * head_dim (gemma-4: global ≠ sliding)
            logger.info("[gen_runner] compiling layer %d/%d (pre + post%s)...", i + 1, len(layers), " + decode" if decode_ok else "")
            pre_w, post_w = build_attention_split_wrapper(block)
            # Per-wrapper device-constant caches: the symbolic program and its static
            # decode-bucket twin bind the SAME weights — share one upload, not two.
            pre_consts: dict = {}
            post_consts: dict = {}
            with torch.device("cpu"):
                pre_programs.append(
                    _compile_split(
                        pre_w,
                        [torch.zeros(8, hidden, dtype=dtype)],
                        ["hidden"],
                        np_dtype,
                        dev_consts=pre_consts,
                        arena=arena,
                        capacity=max_tokens,
                    )
                )
                post_programs.append(
                    _compile_split(
                        post_w,
                        [torch.zeros(8, attn_width, dtype=dtype), torch.zeros(8, hidden, dtype=dtype)],
                        ["attn_out", "residual"],
                        np_dtype,
                        dev_consts=post_consts,
                        arena=arena,
                        capacity=max_tokens,
                    )
                )
                # Static M=decode_bucket twins — fast at decode (small M). If a layer's static
                # compile fails (e.g. a demoted-matmul lowering gap at this bucket), drop the
                # decode path entirely and fall back to the symbolic programs (slow but correct).
                if decode_ok:
                    try:
                        pre_decode.append(
                            _compile_split(
                                pre_w, [torch.zeros(decode_bucket, hidden, dtype=dtype)], None, np_dtype, dev_consts=pre_consts, arena=arena
                            )
                        )
                        post_decode.append(
                            _compile_split(
                                post_w,
                                [torch.zeros(decode_bucket, attn_width, dtype=dtype), torch.zeros(decode_bucket, hidden, dtype=dtype)],
                                None,
                                np_dtype,
                                dev_consts=post_consts,
                                arena=arena,
                            )
                        )
                    except Exception as ex:  # noqa: BLE001 — any lowering/compile failure → disable the bucket
                        logger.warning("[gen_runner] decode-bucket compile failed at layer %d (%s); decode falls back to symbolic", i, ex)
                        decode_ok = False
                # Static M=prefill_bucket chunk twins — exact grids on the hot chunked-prefill
                # width (the symbolic masked-tile programs at off-hint T are the residual
                # prefill cost). Same failure contract as the decode twins.
                if prefill_ok:
                    try:
                        pre_prefill.append(
                            _compile_split(
                                pre_w,
                                [torch.zeros(prefill_bucket, hidden, dtype=dtype)],
                                None,
                                np_dtype,
                                dev_consts=pre_consts,
                                arena=arena,
                            )
                        )
                        post_prefill.append(
                            _compile_split(
                                post_w,
                                [torch.zeros(prefill_bucket, attn_width, dtype=dtype), torch.zeros(prefill_bucket, hidden, dtype=dtype)],
                                None,
                                np_dtype,
                                dev_consts=post_consts,
                                arena=arena,
                            )
                        )
                    except Exception as ex:  # noqa: BLE001 — any lowering/compile failure → disable the twin
                        logger.warning("[gen_runner] prefill-bucket compile failed at layer %d (%s); prefill falls back to symbolic", i, ex)
                        prefill_ok = False

        embed_weight = trunk.embed_tokens.weight.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
        # Gemma scales embeddings by sqrt(hidden) (a ``Gemma3TextScaledWordEmbedding`` carries it as
        # an ``embed_scale`` buffer); a plain ``nn.Embedding`` has none (scale 1). Fold it into the
        # gather table so ``embed`` / ``embed_device`` both apply it with zero per-step cost.
        embed_scale = float(getattr(trunk.embed_tokens, "embed_scale", 1.0))
        if embed_scale != 1.0:
            embed_weight = embed_weight * np_dtype.type(embed_scale)
        use_decode = decode_ok and len(pre_decode) == len(layers)
        use_prefill = prefill_ok and len(pre_prefill) == len(layers)
        runner = cls(
            embed_weight=embed_weight,
            norm=trunk.norm,
            pre=pre_programs,
            post=post_programs,
            attn_meta=attn_meta,
            np_dtype=np_dtype,
            pre_decode=pre_decode if use_decode else None,
            post_decode=post_decode if use_decode else None,
            decode_bucket=decode_bucket,
            prefill_capacity=max_tokens,
            pre_prefill=pre_prefill if use_prefill else None,
            post_prefill=post_prefill if use_prefill else None,
            prefill_bucket=prefill_bucket,
        )
        if runner.has_device_decode:
            # EAGER, not lazy: vLLM sizes its KV cache from a profiling pass that runs
            # after model construction — anything allocated later (the embed table is
            # ~1.9 GiB for gemma-4-12B) lands on memory the KV cache already claimed
            # and OOMs at high --gpu-memory-utilization, only at the first
            # small-batch decode (T <= bucket). Allocating here puts the device
            # residents inside the profiled footprint.
            runner._ensure_device()
        return runner

    def embed(self, input_ids):
        """``input_ids``: list/1-D of ints → ``[T, H]`` numpy in the runner dtype."""
        import numpy as np

        return self._embed_weight[np.asarray(input_ids, dtype=np.int64)]

    def forward_layer_pre(self, layer, hidden, positions=None):
        """``hidden[T, H]`` numpy → un-rotated ``(q[T,Hq·D], k[T,Hkv·D], v[T,Hkv·D])``.
        ``positions`` is unused under A2 (RoPE applied downstream); kept for signature parity.
        Uses the static decode-bucket program when ``T <= decode_bucket`` (pad → run → slice)."""
        del positions
        h = hidden.astype(self._np_dtype, copy=False)
        t = h.shape[0]
        if self._pre_decode is not None and t <= self._decode_bucket:
            q, k, v = self._pre_decode[layer].run([_pad_rows(h, self._decode_bucket)])
            return q[:t], k[:t], v[:t]
        return tuple(self._pre[layer].run([h]))

    def forward_layer_post(self, layer, attn_out, residual):
        """``(attn_out[T,Hq·D], residual[T,H])`` numpy → ``layer_out[T, H]`` numpy. Decode-bucketed
        like ``forward_layer_pre``."""
        a = attn_out.astype(self._np_dtype, copy=False)
        r = residual.astype(self._np_dtype, copy=False)
        t = a.shape[0]
        if self._post_decode is not None and t <= self._decode_bucket:
            out = self._post_decode[layer].run([_pad_rows(a, self._decode_bucket), _pad_rows(r, self._decode_bucket)])[0]
            return out[:t]
        return self._post[layer].run([a, r])[0]

    def final_norm(self, hidden):
        """Apply the model's final norm (held as a torch module) to ``hidden[T, H]`` numpy."""
        import numpy as np
        import torch

        with torch.no_grad():
            out = self._norm(torch.from_numpy(np.ascontiguousarray(hidden)))
        return out.numpy()

    # --- Device-resident decode path (Phase A) ---
    # Used by the vLLM plugin for the decode hot path (T <= decode_bucket); the numpy methods
    # above stay for prefill / the standalone ``emmy generate`` oracle.

    def _ensure_device(self):
        """Build CUDA copies of the embed table + final norm (idempotent). Called EAGERLY by
        ``from_model`` when the device-decode path exists — vLLM's KV-cache sizing profiles the
        model right after construction, so these residents must be allocated before it, never
        at the first decode (see ``from_model``). A **deep copy** of the norm — `.to()` is
        in-place, and the host `final_norm` (oracle / prefill) still feeds it CPU tensors, so
        the shared module must stay on CPU."""
        if getattr(self, "_dev_ready", False):
            return
        import copy

        import torch

        self._embed_weight_dev = torch.from_numpy(self._embed_weight).cuda()
        self._norm_dev = copy.deepcopy(self._norm).to("cuda")
        self._dev_ready = True

    def embed_device(self, input_ids):
        """``input_ids``: 1-D int torch CUDA tensor → ``[T, H]`` CUDA tensor (on-device gather)."""
        self._ensure_device()
        return self._embed_weight_dev[input_ids.long()]

    def forward_layer_pre_device(self, layer, hidden):
        """Device twin of :meth:`forward_layer_pre`: ``hidden[T,H]`` CUDA → un-rotated
        ``(q, k, v)`` CUDA tensors. ``T <= decode_bucket`` rides the static decode twin
        (captured-replay); wider ``T`` (prefill / mixed chunked-prefill steps) rides the
        SYMBOLIC program device-resident (``run_device_sym``) when it was built with
        capacity — no per-layer host numpy hop either way."""
        t = hidden.shape[0]
        if self._pre_decode is not None and t <= self._decode_bucket:
            return tuple(self._pre_decode[layer].run_device([hidden]))
        if self._pre_prefill is not None and t <= self._prefill_bucket:
            # The static chunk twin: T real rows into the prefix, the bucket's exact grids
            # compute (stale padding rows sliced away by ``run_device``'s ``[:t]``).
            return tuple(self._pre_prefill[layer].run_device([hidden]))
        return tuple(self._pre[layer].run_device_sym([hidden]))

    def forward_layer_post_device(self, layer, attn_out, residual):
        """Device twin of :meth:`forward_layer_post`: ``(attn_out, residual)`` CUDA → ``[T,H]``
        CUDA. Decode-bucketed / symbolic-routed like :meth:`forward_layer_pre_device`."""
        t = attn_out.shape[0]
        if self._post_decode is not None and t <= self._decode_bucket:
            return self._post_decode[layer].run_device([attn_out, residual])[0]
        if self._post_prefill is not None and t <= self._prefill_bucket:
            return self._post_prefill[layer].run_device([attn_out, residual])[0]
        return self._post[layer].run_device_sym([attn_out, residual])[0]

    def final_norm_device(self, hidden):
        """Apply the final norm on CUDA to a ``hidden[T,H]`` CUDA tensor."""
        import torch

        self._ensure_device()
        with torch.no_grad():
            return self._norm_dev(hidden)
