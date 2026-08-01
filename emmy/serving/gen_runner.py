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
weights (one device buffer per constant, ``_bind_plan_constants``) and their activation
buffers + scratch slabs (one ``BufferArena`` per runner) — the footprint no longer scales
with ``num_layers`` (the memory budget the plan flagged, Phase 2 / Top risk #9).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class _Program:
    """One compiled dynamic-``num_tokens`` split subgraph, run via the host ``rebind`` path."""

    def __init__(self, program, input_names, output_names, const_bytes=0):
        self.program = program
        self.input_names = input_names
        self.output_names = output_names
        self.const_bytes = const_bytes  # deduped bound-constant footprint — the boot roofline audit's floor input

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
                # Under the outer whole-step capture the output CLONES are dead weight: the graph's
                # fixed kernel order guarantees every consumer (RoPE — in-place on the view is fine,
                # its target is rewritten fresh next replay — attention, the next twin's prefix
                # upload) reads before this program's next-replay overwrite, and each layer runs its
                # OWN program instance so no other layer touches these buffers. Dropping them removes
                # ~4 D2D copy nodes per layer per step from the captured graph (the emmy↔vLLM seam
                # traffic the decode-gap trace attributed). The uncaptured path keeps the clone —
                # there the program graph may replay again before the caller consumes the view.
                outs = self.program.output_prefix_device()
                return [torch.from_dlpack(outs[n])[:t] for n in self.output_names]
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
        (:func:`_compile_split`); ``set_sym_values`` raises past it.

        Under an OUTER capture (an over-bucket decode size in vLLM's capture ladder) the output
        clones drop, mirroring :meth:`run_device`'s captured branch: the outer graph's fixed
        kernel order guarantees every consumer (RoPE, attention, the next twin's prefix upload)
        reads before this program's next-replay overwrite, and each layer runs its own program
        instance. ``set_sym_values`` stays capture-safe because vLLM's uncaptured per-size warmup
        has already populated this sym key's TMA descriptor overlay — the descriptor H2D never
        lands inside the capture window. The uncaptured path keeps the clone: there the buffers
        may be rewritten before the caller consumes the view."""
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
            if torch.cuda.is_current_stream_capturing():
                return [torch.from_dlpack(outs[n]) for n in self.output_names]
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


def _bind_plan_constants(plan, sources, cache):
    """Build the constant feed from the plan's weight specs (works identically whether the
    plan came from a fresh compile or a pack). With a ``cache`` (per-wrapper), each distinct
    ``(source_path, load_ops)`` weight uploads ONCE and the cupy array is shared across
    program builds — the symbolic and decode/prefill-bucket twins bind the same weights;
    per-build numpy feeds would upload a second full on-GPU copy of the trunk (~2× the
    weight footprint). ``cache`` must be scoped to one wrapper — param paths are
    wrapper-relative, so a cross-wrapper cache would collide."""
    from emmy.compiler.backend.plan import apply_weight_loads
    from emmy.compiler.loader.binder import assemble_source

    out = {}
    for nid, w in plan.weights.items():
        src = assemble_source(w, sources)
        if src is None or w.load_ops is None:
            continue
        if cache is None:
            out[nid] = apply_weight_loads(src, w.load_ops)
            continue
        import cupy as cp

        key = (w.source_path, w.source_parts, w.load_ops)
        arr = cache.get(key)
        if arr is None:
            arr = cp.asarray(apply_weight_loads(src, w.load_ops))
            cache[key] = arr
        out[nid] = arr
    return out


def trace_split(wrapper, example_args, argnames):
    """Trace ``wrapper`` into the Graph :func:`_compile_split` compiles — the trace half,
    split out so a capture harness (``scripts/capture_gen_twins.py``) records the graphs
    serving actually runs rather than a hand-rolled approximation of them. ``argnames``
    ties each named arg's axis-0 to a shared symbolic ``num_tokens`` Dim; ``None`` traces
    fully static at the example shapes."""
    from emmy.compiler.trace.torch import trace_module

    dynamic_shapes = None
    if argnames:
        from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

        dynamic_shapes = build_torch_dynamic_shapes(parse_position_specs([f"num_tokens@{n}:0" for n in argnames]))
    return trace_module(wrapper, tuple(example_args), dynamic_shapes=dynamic_shapes)


def _compile_split(wrapper, example_args, argnames, np_dtype, dev_consts=None, arena=None, capacity=None, plan=None):
    """Trace ``wrapper`` and build a :class:`_Program`; returns ``(program, plan)``. ``argnames``
    (a list) ties each named arg's axis-0 to a shared symbolic ``num_tokens`` Dim — the
    **prefill** program (one program, any width). ``argnames=None`` traces a **fully static**
    graph at the example shapes — the **decode-bucket** program (efficient at small M; the
    symbolic program's hint-sized M-tile is pathological at decode). ``dev_consts`` (a
    per-wrapper dict) shares each weight's device buffer across the builds that pass the same
    dict — see :func:`_bind_plan_constants`. ``arena`` (one per runner) pools the activation
    buffers + scratch slab across every program built with it — layers run sequentially, so N
    layers hold ~one layer's worth. ``capacity`` (symbolic programs only) sizes the BUILD
    feed's token axis so the device buffers hold any step up to it — the
    :meth:`_Program.run_device_sym` prefill path needs fixed capacity buffers
    (``set_sym_values`` never re-allocates); without it the build feed is the example (the
    host ``rebind`` path re-sizes per call). ``plan`` (a pack-loaded ``ExecutionPlan``) skips
    the trace + compile entirely and builds from the stored plan — kernels load by cubin key,
    weights rebind from the live wrapper through the same shared-upload cache."""
    import torch

    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock

    if plan is None:
        from emmy.compiler.backend.cuda.backend import CudaBackend
        from emmy.compiler.backend.plan import plan_from_graph

        graph = trace_split(wrapper, example_args, argnames)
        plan = plan_from_graph(CudaBackend(tune_db="auto").compile(graph))

    sources = {}
    for path, t in wrapper.named_parameters(remove_duplicate=False):
        sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
    for path, t in wrapper.named_buffers(remove_duplicate=False):
        sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)

    build_args = example_args
    if capacity is not None and argnames:
        build_args = [torch.zeros((capacity, *a.shape[1:]), dtype=a.dtype) for a in example_args]
    feed = {n: a.detach().cpu().to(torch.float32).numpy().astype(np_dtype) for n, a in zip(plan.inputs, build_args, strict=True)}
    with gpu_lock():
        const_feed = _bind_plan_constants(plan, sources, dev_consts)
        program = CompiledProgram.build_from_plan(plan, {**const_feed, **feed}, arena=arena)
    # Deduped by storage: a weight bound under two nids (e.g. gemma's K=V projection reuse) is
    # one physical read stream, and the shared-upload cache aliases arrays across nids.
    const_bytes = sum(a.nbytes for a in {id(a): a for a in const_feed.values()}.values())
    return _Program(program, list(plan.inputs), list(plan.outputs), const_bytes=const_bytes), plan


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
        pre_m1=None,
        post_m1=None,
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
        self._pre_m1 = pre_m1  # list[_Program] — static M=1 gemv-class twins (or None → bucket twins take T=1)
        self._post_m1 = post_m1
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
        self._sym_decode_warned: set[int] = set()  # widths already reported by _warn_symbolic_decode

    def _warn_symbolic_decode(self, t: int) -> None:
        """Report ONCE per width when a decode-shaped step misses the decode twin and falls to
        the symbolic path. Silent here is a ~40% throughput loss that no audit sees: the twins
        only cover the widths in :mod:`emmy.serving.twins`, so a step landing just above the
        bucket looks like ordinary symbolic traffic. The live cause is the cudagraph ladder —
        vLLM pads a step UP to a captured size before the runner ever sees it, so a rung sitting
        one step above the bucket silently retires the twin (see ``_gen_graph_args``).

        The advice deliberately leads with the ladder rather than with the bucket: widening the
        bucket to an UNCOVERED width trades a throughput loss for a correctness risk, because such
        a width resolves its kernels cold at boot, which is neither reproducible nor accuracy-gated
        (bucket 256 on the gemma-4 image returns empty completions)."""
        if t in self._sym_decode_warned or t > 2 * self._decode_bucket:
            return
        self._sym_decode_warned.add(t)
        logger.warning(
            "[gen_runner] decode-shaped step of %d tokens missed the decode twin (bucket %d) and fell to the "
            "symbolic path — check that the cudagraph capture ladder has a rung <= the bucket (speculative "
            "decoding re-rounds it), or raise EMMY_GEN_DECODE_BUCKET to >= %d. Prefer a width this deployment "
            "has tuned kernels for: an uncovered width resolves its kernels cold, which is neither reproducible "
            "nor accuracy-gated",
            t,
            self._decode_bucket,
            t,
        )

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

    @property
    def rider_width(self) -> int:
        """Static-tier coverage ABOVE the chunk twin: a step of ``T`` in
        ``(prefill_bucket, prefill_bucket + rider_width]`` splits row-wise into the chunk
        twin (first ``prefill_bucket`` rows) + the decode twin (the rider rows) — correct
        because pre/post are per-token-independent, so the row split is request-agnostic.
        This is what lets ``--max-num-batched-tokens`` sit ABOVE the chunk width: a full
        chunk step keeps carrying its decode riders (and the previous prompt's tail
        tokens) instead of freezing every decoding request for the whole chunk. 0 = no
        split coverage (either twin family missing)."""
        if self._pre_prefill is None or self._pre_decode is None:
            return 0
        return self._decode_bucket

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
        pre_m1, post_m1 = [], []
        pre_prefill, post_prefill = [], []
        decode_ok = decode_bucket and decode_bucket > 0
        from emmy import config as emmy_config

        # The gemv-class T=1 tier is OFF by default (EMMY_GEN_M1_TIER=1 to enable): the M=1
        # matvec forms are gemv-fast, but the fused norm→merged edges lift with ZERO free axes
        # at M=1 and schedule as grid-1 kernels — recognition cannot bind the degenerate
        # composition yet, so an enabled tier would deploy ms-class kernels. Infra is complete
        # (build/routing/chaining/pack names); flip the gate once the recognizer gap closes.
        m1_ok = bool(decode_ok and decode_bucket > 1 and emmy_config.gen_m1_tier())
        # The prefill-chunk twin only pays ABOVE the decode bucket (an equal-or-smaller
        # bucket is fully shadowed by the decode twins' routing).
        prefill_ok = prefill_bucket and prefill_bucket > max(decode_bucket or 0, 0)

        # Pack lookup (EMMY_PACK_DIR): one pack for the whole per-layer program set. The
        # validity key is the model's config hash + the serving shape — deliberately NO model
        # id/path: the baked image resolves the model to a snapshot *path* offline while the
        # warm boot uses the hub id, and the config hash already pins identity.
        import hashlib

        # The config hash must strip the VOLATILE fields or the path sneaks back in:
        # ``_name_or_path`` records the load spelling (the hub id at warm time, the snapshot
        # path on a baked offline boot), so hashing to_json_string() keyed the warm pack and
        # the baked boot APART — the 2026-07-24 verify failure (66 runtime recompiles on an
        # image whose warm had converged). ``transformers_version`` churns the same way.
        import json as _json

        from emmy import config as emmy_config
        from emmy.compiler.backend.pack import load_pack, pack_path

        cfg_dict = model.config.to_dict()
        for volatile in ("_name_or_path", "transformers_version"):
            cfg_dict.pop(volatile, None)
            for sub in cfg_dict.values():
                if isinstance(sub, dict):
                    sub.pop(volatile, None)
        pack_key = {
            "kind": "gen-split",
            "model": str(getattr(text_config, "model_type", "gen")),  # label + key; config-derived, path-stable
            "config_sha": hashlib.sha1(_json.dumps(cfg_dict, sort_keys=True, default=str).encode()).hexdigest()[:16],
            "dtype": dtype_str,
            "decode_bucket": int(decode_bucket or 0),
            "max_tokens": int(max_tokens or 0),
            "prefill_bucket": int(prefill_bucket or 0),
        }
        pack_at = pack_path(emmy_config.pack_dir(), pack_key) if emmy_config.pack_dir() is not None else None
        loaded = load_pack(pack_at, key=pack_key) if pack_at is not None else None
        if loaded is not None:
            # "pack hit" is a contract: the serving image's verify.sh greps docker logs for it.
            logger.info("[gen_runner] pack hit at %s — skipping trace + compile for %d program(s)", pack_at, len(loaded))
            # A logger-independent hit marker: the container logging config swallows this
            # module's INFO lines, so the image verify (docker/vllm-emmy-serve/verify.sh)
            # asserts the pack hit on this file instead of a log grep.
            try:
                (pack_at / ".pack_hit").touch()
            except OSError:
                pass
            # The pack records which twin sets survived their compiles — honor that instead
            # of re-attempting a twin the save-time boot already saw fail.
            decode_ok = decode_ok and "L00.pre.decode" in loaded
            prefill_ok = prefill_ok and "L00.pre.prefill" in loaded

        def stored(name):
            return loaded.get(name) if loaded is not None else None

        plans: dict = {}

        def build(name, *args, **kw):
            prog, built_plan = _compile_split(*args, plan=stored(name), **kw)
            plans[name] = built_plan
            return prog

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
                    build(
                        f"L{i:02d}.pre.sym",
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
                    build(
                        f"L{i:02d}.post.sym",
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
                            build(
                                f"L{i:02d}.pre.decode",
                                pre_w,
                                [torch.zeros(decode_bucket, hidden, dtype=dtype)],
                                None,
                                np_dtype,
                                dev_consts=pre_consts,
                                arena=arena,
                            )
                        )
                        post_decode.append(
                            build(
                                f"L{i:02d}.post.decode",
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
                # Static M=1 twins — the gemv-class c=1 decode tier: at one row the contractions
                # demote to the PLANAR coop-reduce forms, which at b64/b128 stream the weights at
                # ~1.68 TB/s (>= cuBLAS gemv) where the bucket-32 twins' computed-A forms run
                # ~1.5. Routed at T == 1 only; T in [2, bucket] keeps the bucket twins. Same
                # failure contract: any layer's compile failure disables the tier.
                if m1_ok:
                    try:
                        pre_m1.append(
                            build(
                                f"L{i:02d}.pre.m1",
                                pre_w,
                                [torch.zeros(1, hidden, dtype=dtype)],
                                None,
                                np_dtype,
                                dev_consts=pre_consts,
                                arena=arena,
                            )
                        )
                        post_m1.append(
                            build(
                                f"L{i:02d}.post.m1",
                                post_w,
                                [torch.zeros(1, attn_width, dtype=dtype), torch.zeros(1, hidden, dtype=dtype)],
                                None,
                                np_dtype,
                                dev_consts=post_consts,
                                arena=arena,
                            )
                        )
                    except Exception as ex:  # noqa: BLE001 — any lowering/compile failure → disable the tier
                        logger.warning("[gen_runner] M=1 twin compile failed at layer %d (%s); T=1 decode rides the bucket twins", i, ex)
                        m1_ok = False
                # Static M=prefill_bucket chunk twins — exact grids on the hot chunked-prefill
                # width (the symbolic masked-tile programs at off-hint T are the residual
                # prefill cost). Same failure contract as the decode twins.
                if prefill_ok:
                    try:
                        pre_prefill.append(
                            build(
                                f"L{i:02d}.pre.prefill",
                                pre_w,
                                [torch.zeros(prefill_bucket, hidden, dtype=dtype)],
                                None,
                                np_dtype,
                                dev_consts=pre_consts,
                                arena=arena,
                            )
                        )
                        post_prefill.append(
                            build(
                                f"L{i:02d}.post.prefill",
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

        # post→pre buffer CHAINING (decode twins): rewire every post twin's OUTPUT array onto the
        # pre twins' shared hidden-INPUT backing (one arena backing per role:name across layers).
        # The next layer's pre "upload" then sees its own buffer as the source and self-copy-skips
        # (``upload_prefix_device``) — one D2D seam copy per layer per step drops out of the
        # captured decode graph. Safe because within a step post[l] writes the backing only AFTER
        # its residual upload copied the previous hidden out of it (the residual copy is the
        # protective copy and stays), and the decode/symbolic paths never interleave within one
        # step (the runner routes on T). Rewiring happens before any run or capture, so both the
        # per-program graphs and vLLM's outer whole-step capture bake the chained pointers.
        if decode_ok and pre_decode and post_decode:
            pre_in = pre_decode[0].input_names[0]
            shared_in = pre_decode[0].program.arrays[pre_in]
            for prog in post_decode:
                out_name = prog.output_names[0]
                if prog.program.arrays[out_name].nbytes == shared_in.nbytes:
                    prog.program.arrays[out_name] = shared_in
        if m1_ok and pre_m1 and post_m1:
            import cupy as cp

            # Same chaining for the M=1 twins. Their pre input is a [1, H] view into the SAME
            # role:name arena backing the bucket twins use, so the post outputs rewire onto a
            # [1, H] view over that backing's memory (nbytes differ from the bucket view; the
            # pointer is what the upload self-copy skip keys on).
            m1_in = pre_m1[0].input_names[0]
            m1_shared = pre_m1[0].program.arrays[m1_in]
            for prog in post_m1:
                out_name = prog.output_names[0]
                cur = prog.program.arrays[out_name]
                if cur.shape == m1_shared.shape and cur.dtype == m1_shared.dtype:
                    prog.program.arrays[out_name] = cp.ndarray(m1_shared.shape, dtype=m1_shared.dtype, memptr=m1_shared.data)
        # A2: the same chaining for the SYMBOLIC programs (capacity-view buffers — device path
        # only, `max_tokens` set; the oracle's host `rebind` re-takes arena views per call and
        # neither needs nor keeps the rewire) and for the static prefill-chunk twins. Every
        # family's post output now aliases its pre twins' shared hidden-INPUT backing, so the
        # between-layer seam copy self-skips on eager chunk steps (the measured ~0.9 ms/layer
        # host+copy chunk overhead's copy share) and drops out of captured over-bucket sym
        # steps. Rider steps stay safe under the cross-tier aliasing (all tiers' views share
        # one backing base): they are eager by construction (prefill is never captured), and
        # `run_device`'s uncaptured path CLONES the chunk-twin head before the decode-twin
        # tail overwrites the shared rows.
        if max_tokens is not None and pre_programs and post_programs:
            sym_in = pre_programs[0].input_names[0]
            sym_shared = pre_programs[0].program.arrays[sym_in]
            for prog in post_programs:
                out_name = prog.output_names[0]
                if prog.program.arrays[out_name].nbytes == sym_shared.nbytes:
                    prog.program.arrays[out_name] = sym_shared
        if prefill_ok and pre_prefill and post_prefill:
            pf_in = pre_prefill[0].input_names[0]
            pf_shared = pre_prefill[0].program.arrays[pf_in]
            for prog in post_prefill:
                out_name = prog.output_names[0]
                if prog.program.arrays[out_name].nbytes == pf_shared.nbytes:
                    prog.program.arrays[out_name] = pf_shared

        embed_weight = trunk.embed_tokens.weight.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
        # Gemma scales embeddings by sqrt(hidden) (a ``Gemma3TextScaledWordEmbedding`` carries it as
        # an ``embed_scale`` buffer); a plain ``nn.Embedding`` has none (scale 1). Fold it into the
        # gather table so ``embed`` / ``embed_device`` both apply it with zero per-step cost.
        embed_scale = float(getattr(trunk.embed_tokens, "embed_scale", 1.0))
        if embed_scale != 1.0:
            embed_weight = embed_weight * np_dtype.type(embed_scale)
        use_decode = decode_ok and len(pre_decode) == len(layers)
        use_m1 = m1_ok and len(pre_m1) == len(layers) and len(post_m1) == len(layers)
        use_prefill = prefill_ok and len(pre_prefill) == len(layers)
        if pack_at is not None and loaded is None:
            # Best-effort save after a full compile: only the program sets that survived
            # (a mid-run twin failure leaves partial lists — those must not be recorded).
            keep = {
                name: p
                for name, p in plans.items()
                if (".decode" not in name or use_decode) and (".prefill" not in name or use_prefill) and (".m1" not in name or use_m1)
            }
            if any(w.load_ops is None for p in keep.values() for w in p.weights.values()):
                logger.warning("[gen_runner] not writing pack: a weight load-op chain is outside the pack vocabulary")
            else:
                try:
                    from emmy.compiler.backend.pack import save_pack

                    save_pack(pack_at, keep, key=pack_key)
                except Exception:  # noqa: BLE001 — the pack is an optimization, never a boot blocker
                    logger.warning("[gen_runner] pack write failed at %s", pack_at, exc_info=True)
        runner = cls(
            embed_weight=embed_weight,
            norm=trunk.norm,
            pre=pre_programs,
            post=post_programs,
            attn_meta=attn_meta,
            np_dtype=np_dtype,
            pre_decode=pre_decode if use_decode else None,
            post_decode=post_decode if use_decode else None,
            pre_m1=pre_m1 if use_m1 else None,
            post_m1=post_m1 if use_m1 else None,
            decode_bucket=decode_bucket,
            prefill_capacity=max_tokens,
            pre_prefill=pre_prefill if use_prefill else None,
            post_prefill=post_prefill if use_prefill else None,
            prefill_bucket=prefill_bucket,
        )
        runner._embed_scale = embed_scale  # raw-table scale for adopt_embed_table (host table keeps it folded)
        if runner.has_device_decode:
            # EAGER, not lazy: vLLM sizes its KV cache from a profiling pass that runs
            # after model construction — anything allocated later (the embed table is
            # ~1.9 GiB for gemma-4-12B) lands on memory the KV cache already claimed
            # and OOMs at high --gpu-memory-utilization, only at the first
            # small-batch decode (T <= bucket). Allocating here puts the device
            # residents inside the profiled footprint.
            runner._ensure_device()
        # Boot roofline audit over the STATIC twins (symbolic programs sit at capacity shape at
        # boot — wrong width to time). One layer per attention class: homogeneous models audit
        # layer 0 only; gemma-4 adds its first global layer (different head_dim → different
        # programs). Advisory only — audit_boot_programs never raises.
        from emmy.serving.roofline import audit_boot_programs

        audit_layers = [0]
        hetero = next((i for i in range(1, len(attn_meta)) if attn_meta[i] != attn_meta[0]), None)
        if hetero is not None:
            audit_layers.append(hetero)
        named = [
            (f"L{li}.{role}.{tag}", plist[li])
            for li in audit_layers
            for tag, role, plist in (
                (f"decode.m{decode_bucket}", "pre", runner._pre_decode),
                (f"decode.m{decode_bucket}", "post", runner._post_decode),
                ("decode.m1", "pre", runner._pre_m1),
                ("decode.m1", "post", runner._post_m1),
                (f"chunk.m{prefill_bucket}", "pre", runner._pre_prefill),
                (f"chunk.m{prefill_bucket}", "post", runner._post_prefill),
            )
            if plist is not None and plist[li] is not None
        ]
        audit_boot_programs(named)
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

        if getattr(self, "_embed_weight_dev", None) is None:
            self._embed_weight_dev = torch.from_numpy(self._embed_weight).cuda()
        self._norm_dev = copy.deepcopy(self._norm).to("cuda")
        self._dev_ready = True

    def adopt_embed_table(self, weight, *, scale=1.0):
        """Share an already-resident device copy of the RAW (unscaled) embed table — gemma ties
        ``lm_head.weight`` to ``embed_tokens.weight``, so a server that holds the head can hand
        it here and the runner skips its own ~2 GiB upload (the KV-cache budget gets it back).
        ``weight``: torch CUDA ``[vocab, H]`` in the trunk dtype; ``scale``: the model's
        ``embed_scale`` (sqrt(H) for gemma), applied at gather time since the SHARED table must
        stay raw — the head reads it unscaled (a folded scale would retemper every logit)."""
        self._embed_weight_dev = weight
        self._embed_dev_scale = float(scale)

    def embed_device(self, input_ids):
        """``input_ids``: 1-D int torch CUDA tensor → ``[T, H]`` CUDA tensor (on-device gather).
        An adopted (raw, shared) table applies the embed scale here — in fp32, matching the
        host table's fold-at-fp32-then-cast numerics."""
        self._ensure_device()
        rows = self._embed_weight_dev[input_ids.long()]
        scale = getattr(self, "_embed_dev_scale", 1.0)
        if scale != 1.0:
            rows = (rows.float() * scale).to(rows.dtype)
        return rows

    def forward_layer_pre_device(self, layer, hidden):
        """Device twin of :meth:`forward_layer_pre`: ``hidden[T,H]`` CUDA → un-rotated
        ``(q, k, v)`` CUDA tensors. ``T <= decode_bucket`` rides the static decode twin
        (captured-replay); ``T == prefill_bucket`` — the FULL chunked-prefill step, the
        width the twin was built for — rides the static chunk twin's exact grids;
        ``prefill_bucket < T <= prefill_bucket + rider_width`` — a full chunk step carrying
        decode riders / a prompt tail — splits row-wise across the chunk twin + the decode
        twin (see :attr:`rider_width`); every other width (an over-bucket decode batch, a
        partial tail chunk) rides the SYMBOLIC program device-resident
        (``run_device_sym``) — no per-layer host numpy hop any way.
        The twin boundary is EXACT equality, not ``<=``: the twin always computes
        ``prefill_bucket`` rows (pad → run → slice), so routing a T≈32 over-bucket decode
        step or a T≈450 tail chunk through it pays the full-bucket grids for a sliver of
        real rows — up to ~bucket/T× the useful work per layer, in the default-config
        steady state (mnbt-default bucket 4096, ``--max-concurrency 32`` decode)."""
        t = hidden.shape[0]
        if t == 1 and self._pre_m1 is not None:
            return tuple(self._pre_m1[layer].run_device([hidden]))
        if self._pre_decode is not None and t <= self._decode_bucket:
            return tuple(self._pre_decode[layer].run_device([hidden]))
        if self._pre_prefill is not None and t == self._prefill_bucket:
            return tuple(self._pre_prefill[layer].run_device([hidden]))
        if 0 < t - self._prefill_bucket <= self.rider_width:
            import torch  # noqa: PLC0415

            pb = self._prefill_bucket
            head = self._pre_prefill[layer].run_device([hidden[:pb]])
            tail = self._pre_decode[layer].run_device([hidden[pb:]])
            return tuple(torch.cat(pair, dim=0) for pair in zip(head, tail, strict=True))
        self._warn_symbolic_decode(t)
        return tuple(self._pre[layer].run_device_sym([hidden]))

    def forward_layer_post_device(self, layer, attn_out, residual):
        """Device twin of :meth:`forward_layer_post`: ``(attn_out, residual)`` CUDA → ``[T,H]``
        CUDA. Decode-bucketed / exact-chunk / symbolic-routed like
        :meth:`forward_layer_pre_device`."""
        t = attn_out.shape[0]
        if t == 1 and self._post_m1 is not None:
            return self._post_m1[layer].run_device([attn_out, residual])[0]
        if self._post_decode is not None and t <= self._decode_bucket:
            return self._post_decode[layer].run_device([attn_out, residual])[0]
        if self._post_prefill is not None and t == self._prefill_bucket:
            return self._post_prefill[layer].run_device([attn_out, residual])[0]
        if 0 < t - self._prefill_bucket <= self.rider_width:
            import torch  # noqa: PLC0415

            pb = self._prefill_bucket
            head = self._post_prefill[layer].run_device([attn_out[:pb], residual[:pb]])[0]
            tail = self._post_decode[layer].run_device([attn_out[pb:], residual[pb:]])[0]
            return torch.cat((head, tail), dim=0)
        return self._post[layer].run_device_sym([attn_out, residual])[0]

    def post_attn_backing(self, layer: int, rows: int):
        """A torch CUDA view of the M=1 post twin's ``attn_out`` INPUT backing (first ``rows``
        rows), or ``None`` when the m1 tier is off / this layer fell back. vLLM's paged attention
        writes into this view under ``EMMY_GEN_ALIAS_ATTN`` so :meth:`_Program.run_device`'s
        prefix upload self-copy-skips — the seam copy drops out of the captured decode graph.
        The backing is allocated once per program, so the pointer is stable across steps.

        The wrap is CACHED per layer: ``torch.from_dlpack`` on a cupy array negotiates a stream
        sync, which INVALIDATES an in-flight CUDA-graph capture — the view must be minted on an
        uncaptured (warmup) step and only re-served under capture. An uncached layer asked for
        mid-capture returns ``None`` (the caller falls back to the ordinary copy path)."""
        if self._post_m1 is None:
            return None
        handle = self._post_m1[layer]
        if handle is None:
            return None
        views = getattr(self, "_post_m1_attn_views", None)
        if views is None:
            views = self._post_m1_attn_views = {}
        view = views.get(layer)
        if view is None:
            import torch  # noqa: PLC0415

            if torch.cuda.is_current_stream_capturing():
                return None
            arr = handle.program.arrays.get("attn_out")
            if arr is None:
                return None
            view = views[layer] = torch.from_dlpack(arr)
        return view[:rows]

    def final_norm_device(self, hidden):
        """Apply the final norm on CUDA to a ``hidden[T,H]`` CUDA tensor."""
        import torch

        self._ensure_device()
        with torch.no_grad():
            return self._norm_dev(hidden)
