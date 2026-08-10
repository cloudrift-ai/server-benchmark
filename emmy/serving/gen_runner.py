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
``run_device_sym`` (grids sized per step, capacity buffers, no per-layer host hop). When the
configured capacity fits entirely inside the static decode bucket, the runner omits the
redundant symbolic tier so its dynamic upper-bound arena does not consume the model's VRAM.
The host numpy ``rebind`` path survives for ordinary runners; a static-only runner rejects
over-capacity calls explicitly.
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

    def __init__(self, program, input_names, output_names, const_bytes=0, input_weight_bytes=0):
        self.program = program
        self.input_names = input_names
        self.output_names = output_names
        self.const_bytes = const_bytes  # deduped bound-constant footprint — the boot roofline audit's floor input
        # Weight bytes that arrive as INPUTS, not constants (an expert program's per-expert
        # slices). Same role in the floor: every one is read at least once per forward.
        self.input_weight_bytes = input_weight_bytes

    @property
    def weight_bytes(self) -> int:
        """Bytes this program must read per forward however the weights arrive — the roofline
        audit's floor input. An expert program holds ZERO constants, so counting only the
        constant side would give it no floor and audit nothing."""
        return self.const_bytes + self.input_weight_bytes

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

    def run_device(self, arrays, out=None):
        """Device-resident captured-replay twin of :meth:`run` for the **static M=bucket** decode
        programs. ``arrays``: torch CUDA tensors aligned to ``input_names`` (each ``[T, …]``,
        ``T <= bucket``). Uploads the ``T`` real rows into the buffer prefix (device-to-device),
        captures-or-replays the whole-program graph, and returns the outputs as torch CUDA tensors
        sliced to ``T`` — no host round-trip. Stale prefix padding rows are safe (pre/post are
        per-token-independent; only ``[:T]`` is read out). All cupy work runs on torch's current
        stream so the upload, replay and output read stay ordered.

        ``out`` (A3, eager only): a list of torch CUDA destination views aligned to
        ``output_names``, each ``[T, …]``. The single protective copy lands directly in the
        caller's destination (``dest.copy_(view)``) instead of a freshly allocated clone — the
        rider path passes slices of one shared joint tensor, so the halves need no
        ``torch.cat`` (which paid a second full copy plus the allocation). Illegal under an
        outer capture: the captured branch returns raw views by design (dropping copies is the
        point there), so ``out`` would silently change semantics — asserted against.

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
                assert out is None, "run_device(out=...) is an eager-path contract — captured steps return views"
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
            if out is not None:
                for o, n in zip(out, self.output_names, strict=True):
                    o.copy_(torch.from_dlpack(outs[n])[:t])
                return out
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


def combine_routed_experts(xn, gated, run_expert):
    """Route + combine for one MoE layer — shared by :meth:`EmmyGenRunner._moe_combine` and its
    parity tests so both always exercise the same math. ``gated`` is the HF router module's
    return, whose LAST two entries are ``scores[T, k]`` / ``indices[T, k]``; each HIT expert
    gets one ``run_expert(e, rows)`` call on its routed rows and the partials weighted-scatter
    into a fresh ``[T, H]``. Scores cast to the activation dtype — some HF routers (Mixtral)
    return fp32 scores, and ``index_add_`` requires matching dtypes."""
    import torch

    scores, indices = gated[-2], gated[-1]
    scores = scores.to(xn.dtype)
    out = torch.zeros_like(xn)
    for e in indices.unique().tolist():
        tok, pos = torch.where(indices == e)
        out.index_add_(0, tok, run_expert(e, xn[tok]) * scores[tok, pos, None])
    return out


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


# The static prefill expert twin's row count: the mean per-expert width of a full 2048-token
# chunk (T·k/E — OLMoE 2048·8/64 = 256, gpt-oss 2048·4/32 = 256). Routed rows in
# (decode_bucket, 256] ride this twin via the prefix upload (pad-up; per-token-independent),
# wider row sets fall to the symbolic program.
_EXPERT_PREFILL_M = 256


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

        key = (w.source_path, w.source_parts, w.generated, w.load_ops)
        arr = cache.get(key)
        if arr is None:
            arr = cp.asarray(apply_weight_loads(src, w.load_ops))
            cache[key] = arr
        out[nid] = arr
    return out


def _retarget_constants(graph, wrapper, id_to_key) -> None:
    """Re-address every traced constant of ``graph`` from its WRAPPER-relative parameter path to
    the checkpoint key that parameter came from.

    A split wrapper holds submodules of one decoder block, so ``named_parameters`` spells
    ``o_proj.weight`` where the checkpoint says ``model.layers.7.self_attn.o_proj.weight``. The
    trace stamps the wrapper spelling onto ``ConstantOp.source_path``, which is why the
    checkpoint-driven spellers (``spell_trellis_constants``) can never fire on a serving trunk
    and why the trunk had to bind off the live module. ``id_to_key`` — built once over the whole
    twin — maps a parameter TENSOR's identity to its full path, so the mapping needs no naming
    convention and survives any wrapper shape."""
    key_map = {}
    for path, t in list(wrapper.named_parameters(remove_duplicate=False)) + list(wrapper.named_buffers(remove_duplicate=False)):
        full = id_to_key.get(id(t))
        if full is not None:
            key_map[path] = full
    for _nid, op in graph.loadable_constants():
        if op.source_path in key_map:
            op.source_path = key_map[op.source_path]
        if op.source_parts:
            op.source_parts = tuple((key_map.get(p, p), s) for p, s in op.source_parts)


def _plan_sources(plan, wrapper, np_dtype, ckpt_dir, id_to_key):
    """The constant feed's raw sources for a plan whose weights are addressed by CHECKPOINT key.

    Reads exactly the paths the plan asks for — straight from the shards — and falls back to the
    live wrapper only for what the checkpoint has no key for (buffers a module computes at init,
    e.g. rotary tables). Deliberately NOT the whole ``named_parameters`` sweep the module lane
    does: on a compressed trunk those parameters are uninitialized placeholders, and
    materializing them as f32 numpy would cost more host memory than the decoded checkpoint.
    Raises when a weight cannot be sourced at all — a silently unbound trunk weight computes
    garbage, and the folded checkpoint-basis cone (which the plan cannot reproduce) reaches here
    exactly that way."""
    import torch

    from emmy.compiler.loader.safetensors import load_sources_by_path

    unbindable = [
        nid
        for nid, w in plan.weights.items()
        if w.load_ops is None or (w.source_path is None and not w.source_parts and w.generated is None)
    ]
    if unbindable:
        raise RuntimeError(
            f"checkpoint-sourced compile: {len(unbindable)} weight(s) carry no plan-reproducible source "
            f"(e.g. {unbindable[0]!r}) — a trellis linear that fell back to the folded checkpoint-basis cone "
            f"binds nothing here and the program would run weightless"
        )
    want = set()
    for w in plan.weights.values():
        want.update(p for p, _shape in w.source_parts)
        if w.source_path is not None:
            want.add(w.source_path)
    sources = load_sources_by_path(ckpt_dir, want) if want else {}
    missing = want - set(sources)
    if missing:
        for path, t in list(wrapper.named_parameters(remove_duplicate=False)) + list(wrapper.named_buffers(remove_duplicate=False)):
            key = id_to_key.get(id(t), path)
            if key in missing:
                sources[key] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
        still = want - set(sources)
        if still:
            raise RuntimeError(f"checkpoint-sourced compile: no source for {sorted(still)[:3]} (of {len(still)})")
    return sources


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


def _compile_split(
    wrapper,
    example_args,
    argnames,
    np_dtype,
    dev_consts=None,
    arena=None,
    capacity=None,
    plan=None,
    indirect_inputs=None,
    quant_specs=None,
    trellis_specs=None,
    aux_examples=None,
    weight_inputs=(),
    ckpt=None,
    plan_cache=None,
):
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
    weights rebind from the live wrapper through the same shared-upload cache.
    ``indirect_inputs`` names graph INPUT buffers to compile as indirect operands (the base
    pointer resolved in-kernel from a device table — the MoE fixed-slot dispatch): set as a
    graph-level hint before compile, so the schedule search is untouched and only the final
    kernel lowering changes the ABI. Ignored when ``plan`` is supplied (the stored plan already
    carries — or lacks — the indirect encoding; callers validate coverage). ``quant_specs``
    (the fp8 expert path) feeds ``spell_quantized_inputs`` post-trace: the named weight inputs
    become fp8 bits inputs with appended scale inputs; ``example_args`` then carries the scale
    EXAMPLES as its LAST ``len(quant_specs)`` entries (matching the speller's append order),
    while only the leading entries are the wrapper's forward args for the trace.
    ``trellis_specs`` (the EXL3 expert path) feeds ``spell_trellis_inputs`` the same way — the
    named weight inputs become int16 CODES inputs with appended channel-vector inputs — but its
    extra examples come by NAME in ``aux_examples`` rather than by position, so the build feed
    is independent of the speller's append order. ``weight_inputs`` names the inputs that carry
    WEIGHT bytes (an expert program's per-expert slices): they count toward the program's
    weight-streaming floor in the boot roofline audit, where a constant-free expert program
    would otherwise report no floor at all.
    ``ckpt`` — ``(checkpoint_dir, id_to_key)`` — switches the TRUNK to the checkpoint-sourced
    lane: every traced constant is re-addressed to its checkpoint key
    (:func:`_retarget_constants`), ``spell_trellis_constants`` then fires on a serving wrapper
    for the first time (the compressed lane forced, not env-gated), and the constant feed comes
    from the shards rather than the live module (:func:`_plan_sources`). This is what puts an
    EXL3 trunk on the card at its CODED size; without it a trunk linear binds decoded values.
    ``plan_cache`` is a session-scoped, binding-neutral plan-template cache. It runs only on the
    cold path (an explicit pack ``plan`` still wins), after every loader spelling and ABI hint has
    reached the graph; each hit returns a fresh plan carrying THIS wrapper's real source paths."""
    import torch

    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock

    if plan is None:
        from emmy.compiler.backend.cuda.backend import CudaBackend
        from emmy.compiler.backend.plan import plan_from_graph

        n_fwd = len(example_args) - (len(quant_specs) if quant_specs else 0)
        graph = trace_split(wrapper, example_args[:n_fwd], argnames)
        if ckpt is not None:
            from emmy.compiler.loader.quant import spell_trellis_constants

            _retarget_constants(graph, wrapper, ckpt[1])
            spell_trellis_constants(graph, ckpt[0])
        if quant_specs:
            from emmy.compiler.loader.quant import spell_quantized_inputs

            spell_quantized_inputs(graph, quant_specs)
        if trellis_specs:
            from emmy.compiler.loader.quant import spell_trellis_inputs

            spell_trellis_inputs(graph, trellis_specs)
        if indirect_inputs:
            graph.hints.set("cuda.indirect_inputs", tuple(indirect_inputs))

        def compile_plan(g):
            return plan_from_graph(CudaBackend(tune_db="auto").compile(g))

        plan = plan_cache.resolve(graph, compile_plan) if plan_cache is not None else compile_plan(graph)

    if aux_examples:
        # By NAME, over the plan's own input list: the trellis speller both RE-MINTS inputs at a
        # new shape (a dense weight arg becomes its packed codes) and APPENDS new ones, so the
        # traced positional args no longer describe the build feed.
        padded = list(example_args) + [None] * (len(plan.inputs) - len(example_args))
        example_args = [aux_examples.get(n, a) for n, a in zip(plan.inputs, padded, strict=True)]

    if ckpt is not None:
        sources = _plan_sources(plan, wrapper, np_dtype, ckpt[0], ckpt[1])
    else:
        sources = {}
        for path, t in wrapper.named_parameters(remove_duplicate=False):
            sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
        for path, t in wrapper.named_buffers(remove_duplicate=False):
            sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)

    build_args = example_args
    if capacity is not None and argnames:
        # Only the args whose axis 0 IS the symbolic token axis get resized to capacity — a
        # non-symbolic arg (an expert program's weight slice) keeps its true shape AND its
        # example dtype (an fp8 bits slice stays fp8; the per-input feed below binds it).
        build_args = [
            torch.zeros((capacity, *a.shape[1:]), dtype=a.dtype) if n in argnames else a
            for n, a in zip(plan.inputs, example_args, strict=True)
        ]

    # Each plan input binds at its own traced dtype, not one blanket model dtype: an
    # f8-dtype input (an input-sourced fp8 expert weight) binds the raw bit pattern on the
    # uint8 carrier — the same rule as the constant side in ``loader/safetensors.py`` — and
    # a scale input keeps its traced f32. A torch fp8 example tensor reinterprets via
    # ``.view``; a value cast would decode it.
    import numpy as np  # noqa: PLC0415

    input_dtypes = {b.name: b.dtype for b in plan.buffers}
    torch_f8 = (torch.float8_e4m3fn, torch.float8_e5m2)

    def _np_in(n, a):
        dt = input_dtypes.get(n)
        if dt is not None and dt.name in ("f8e4m3", "f8e5m2"):
            if a.dtype in torch_f8:
                a = a.view(torch.uint8)
            return a.detach().cpu().numpy().astype(np.uint8, copy=False)
        if dt is not None and not a.dtype.is_floating_point:
            # An integer example (an EXL3 expert's packed CODES) binds its stored word pattern;
            # the f32 round-trip below is a value cast and means nothing for codes. Keyed on the
            # EXAMPLE's dtype, not the buffer's — a bf16 buffer also carries an integer numpy
            # dtype (the uint16 bits carrier) but arrives as a float tensor.
            return a.detach().cpu().numpy().astype(dt.np, copy=False)
        return a.detach().cpu().to(torch.float32).numpy().astype(dt.np if dt is not None else np_dtype, copy=False)

    feed = {n: _np_in(n, a) for n, a in zip(plan.inputs, build_args, strict=True)}
    with gpu_lock():
        const_feed = _bind_plan_constants(plan, sources, dev_consts)
        program = CompiledProgram.build_from_plan(plan, {**const_feed, **feed}, arena=arena)
    # Deduped by storage: a weight bound under two nids (e.g. gemma's K=V projection reuse) is
    # one physical read stream, and the shared-upload cache aliases arrays across nids.
    const_bytes = sum(a.nbytes for a in {id(a): a for a in const_feed.values()}.values())
    weight_in = sum(feed[n].nbytes for n in weight_inputs if n in feed)
    return _Program(program, list(plan.inputs), list(plan.outputs), const_bytes=const_bytes, input_weight_bytes=weight_in), plan


def _indirect_covered(plan, names):
    """True iff every launch that takes one of ``names`` as an arg carries its indirect-operand
    encoding — the validity check for both a fresh compile (a kernel could consume the weight
    under a derived buffer name, which the hint would miss) and a stored pack plan (a stale
    pre-indirect pack must not half-hit the fixed-slot tier)."""
    for lc in plan.launches:
        marked = {a for a, _, _, _ in lc.indirect_args}
        if any(a in names and a not in marked for a in lc.arg_names):
            return False
    return True


def _plan_with_slot(plan, slot):
    """A shallow plan copy whose indirect operands read table entry ``sel[slot]`` — the per-slot
    stamp for the fixed-slot instances (slot ``i`` of k passes ``slot=i`` as a plain int arg;
    buffers/kernels/weights are shared read-only with the source plan)."""
    import dataclasses

    launches = [
        dataclasses.replace(lc, indirect_args=tuple((a, t, s, slot) for a, t, s, _ in lc.indirect_args)) if lc.indirect_args else lc
        for lc in plan.launches
    ]
    return dataclasses.replace(plan, launches=launches)


def _static_decode_covers_capacity(max_tokens, decode_bucket, prefill_bucket=0) -> bool:
    """Whether every scheduler-visible width is covered by the static decode twins.

    This is deliberately a strict capacity proof, not a performance heuristic.  In this lane
    symbolic programs are unnecessary and their dynamic-shape arena can be much larger than the
    quantized model's remaining VRAM.  A static compile failure must therefore fail the boot
    rather than silently reintroduce the symbolic tier after earlier layers omitted it.
    """
    return bool(
        max_tokens is not None
        and decode_bucket
        and 0 < max_tokens <= decode_bucket
        and not (prefill_bucket and prefill_bucket > decode_bucket)
    )


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
        moe=None,
        expert_tiers=None,
    ):
        self._embed_weight = embed_weight  # numpy [vocab, H]
        self._norm = norm  # torch module
        self._pre = pre  # list[_Program] — symbolic (prefill / any width), empty when static-covered
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
        # MoE third seam: per-layer dict (router module + the E-stacked expert INPUT tensors
        # under ``inputs`` — weights, and for gpt-oss biases + fp8 bits/scales — plus ``top_k``
        # and the expert SHAPE GROUP the layer belongs to) or None for dense layers.
        # ``expert_tiers`` is one program set per group: ``sym`` (any width), ``bucket`` (static
        # M=decode_bucket) and ``one`` (static M=1) for the decode hot path, ``m256`` for
        # prefill-width routed row sets (captured replay — see ``_launch_expert``), and
        # ``slots`` — the FIXED-SLOT decode tier, k instances of the INDIRECT M=1 twin (one
        # dedicated, lifetime-compatible arena shared across the ordered slot launches) whose
        # kernels resolve their weight base pointers from the
        # per-group device tables built in ``_ensure_device`` (the capture-legal T=1 dispatch,
        # ``_moe_combine_slots``). A homogeneous MoE has exactly one group; EXL3's mixed bit
        # allocation gives one per distinct codes shape. A tier whose schedule stages no operand
        # through a TMA descriptor (descriptors bake pointers at build) launches with
        # POINTER-SWAPPED per-expert weight slices — no D2D weight copy; descriptor-bearing
        # tiers fall back to the upload copy, still correct.
        self._moe = moe
        self._expert_tiers = expert_tiers
        # The fixed-slot tier is ALL-OR-NOTHING across groups: a whole-step CUDA graph capture
        # records one launch set for every layer, so a group left on the routed (eager) path
        # would make the captured step wrong for its layers.
        self._slots_ok = bool(expert_tiers) and all(t["slots"] is not None for t in expert_tiers)
        self._expert_swap_safe = {
            id(prog): not any("_desc" in a for launch in prog.program.compiled.launches for a in launch.arg_names)
            for tiers in expert_tiers or ()
            for prog in (tiers["sym"], tiers["bucket"], tiers["one"])
            if prog is not None
        }
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
        return len(self._attn_meta)

    @property
    def decode_bucket(self) -> int:
        return self._decode_bucket

    @property
    def has_device_decode(self) -> bool:
        """True when the static decode-bucket programs exist → the device-resident decode path
        (``embed_device`` / ``forward_layer_*_device`` / ``final_norm_device``) is available."""
        return self._pre_decode is not None

    @property
    def has_moe_fixed_slot(self) -> bool:
        """True when the fixed-slot MoE decode tier exists → a single-token decode step is
        capture-legal (``_moe_combine_slots``: no host sync, fixed launch set). The boot guard
        in ``EmmyGenModel`` keys the MoE capture decision on this. Every shape group must have
        its slots (see ``_slots_ok``)."""
        return self._moe is not None and self._slots_ok

    # Group-0 views of the expert tiers: the shapes an introspecting caller (tests, the boot
    # audit) means when a model has one group, which is every homogeneous MoE.
    @property
    def _expert_sym(self):
        return self._expert_tiers[0]["sym"] if self._expert_tiers else None

    @property
    def _expert_m256(self):
        return self._expert_tiers[0]["m256"] if self._expert_tiers else None

    @property
    def _expert_slots(self):
        return self._expert_tiers[0]["slots"] if self._slots_ok else None

    @property
    def prefill_capacity(self) -> int:
        """The widest configured device-resident step.  It may be served wholly by the
        static decode twins, in which case no symbolic program or symbolic arena exists."""
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
        """``model_id`` is a local checkpoint directory or an HF repo id, the latter optionally
        carrying its revision as ``<repo>@<revision>`` (the serving shim tags vLLM's
        ``--revision`` on). Everything the runner resolves from the checkpoint — detection,
        snapshot, the coded allocation, the pack key — goes through the tagged id, so a repo
        publishing one rung per branch serves the rung that was asked for."""
        import torch
        from transformers import AutoModelForCausalLM

        from emmy.compiler.loader.safetensors import split_revision, warn_if_unpinned
        from emmy.compiler.trace.huggingface import quantized_checkpoint_dir

        warn_if_unpinned(model_id)  # covers both lanes below, including the plain from_pretrained one
        # A quantized checkpoint cannot go through ``from_pretrained`` (transformers would
        # engage its own fp8 machinery); build the twin from config and stream the shards —
        # dense trunk loaded as real values, expert tensors kept fp8 (``load_quantized_split``).
        qdir = quantized_checkpoint_dir(model_id)
        if qdir is not None:
            # EXL3 keeps the TRUNK coded too — the whole point of a 2-bit checkpoint is that the
            # decoded trunk (GLM-4.5-Air: ~14 GiB) does not fit beside the experts. fp8 trunks
            # stay on the decoded lane, where the values are what the fp8 expert path expects.
            from emmy.compiler.loader.quant import checkpoint_quant_digest, checkpoint_quant_format, checkpoint_quant_summary
            from emmy.compiler.trace.huggingface import load_quantized_split

            # EXL3's generic reconstruction algebra is dissolved before lowering, so its
            # checkpoint sources can stay coded on the card. FP8 keeps the existing
            # value-trunk lane; only its routed experts are input-spelled today.
            coded_trunk = checkpoint_quant_format(qdir) == "exl3"
            # The RESOLVED directory and the scheme summary are logged, not just the requested id:
            # a repo that publishes one rung per branch resolves to a per-commit snapshot, and this
            # line is how a boot proves which rung it actually opened.
            logger.info(
                "[gen_runner] loading %s (%s, quantized checkpoint: dense trunk %s, experts compressed) — resolved %s, %s, key %s...",
                model_id,
                dtype_str,
                "coded" if coded_trunk else "as values",
                qdir,
                checkpoint_quant_summary(qdir),
                checkpoint_quant_digest(qdir),
            )
            model, expert_store = load_quantized_split(qdir, getattr(torch, dtype_str), compress_trunk=coded_trunk)
            with torch.device("cpu"):
                return cls.from_model(
                    model,
                    dtype_str=dtype_str,
                    decode_bucket=decode_bucket,
                    max_tokens=max_tokens,
                    prefill_bucket=prefill_bucket,
                    expert_store=expert_store,
                )
        logger.info("[gen_runner] loading %s (%s, CPU trace)...", model_id, dtype_str)
        repo, revision = split_revision(model_id)
        with torch.device("cpu"):
            model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, dtype=getattr(torch, dtype_str)).eval()
            return cls.from_model(
                model, dtype_str=dtype_str, decode_bucket=decode_bucket, max_tokens=max_tokens, prefill_bucket=prefill_bucket
            )

    @classmethod
    def from_model(cls, model, *, dtype_str="float16", decode_bucket=16, max_tokens=None, prefill_bucket=0, expert_store=None):
        """Build from an already-loaded CausalLM module (the network-free path). ``model``
        must be on CPU for the trace. ``expert_store`` (a quantized checkpoint's
        ``load_quantized_split`` result) supplies the per-layer expert input tensors —
        fp8 bits + scales + biases — in place of the twin's expert params (which stay meta
        on the quantized path); the expert programs are then compiled with the decode + scale
        algebra spelled in-graph (``spell_quantized_inputs`` via ``_compile_split``), one program
        set per distinct expert SHAPE. A store whose ``trunk`` is ``"codes"`` (EXL3) also puts the
        TRUNK on the checkpoint-sourced lane, so its coded linears stay compressed on the card."""
        import numpy as np
        import torch

        from emmy.compiler.trace.huggingface import (
            build_attention_split_wrapper,
            build_moe_split_wrapper,
            deinterleave_gate_up,
            moe_block_parts,
            moe_expert_layout,
        )

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
        from emmy.compiler.backend.plan_cache import PlanTemplateCache
        from emmy.compiler.loader.quant import checkpoint_quant_digest

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
        # The config hash alone does NOT identify a compressed checkpoint: the twin is built from a
        # config the loader band strips of its compression scheme, so two rungs of one repo — same
        # architecture, different per-tensor rates and therefore different coded extents and
        # programs — hash identically and would share this pack. Key on the loader's own digest of
        # the checkpoint too (``None`` for an uncompressed one, which keeps its existing key).
        if (ckpt_dir := (expert_store or {}).get("dir")) and (qd := checkpoint_quant_digest(ckpt_dir)) is not None:
            pack_key["quant_sha"] = qd
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

        static_only = _static_decode_covers_capacity(max_tokens, decode_bucket, prefill_bucket) and bool(decode_ok)
        if static_only:
            logger.info(
                "[gen_runner] static decode tier covers configured capacity %d (bucket %d) — omitting symbolic arenas",
                max_tokens,
                decode_bucket,
            )

        def stored(name):
            return loaded.get(name) if loaded is not None else None

        plans: dict = {}
        # One cold-boot compile session for every split. Plans remain per program (and retain
        # each layer's actual checkpoint provenance); only binding-neutral compiled structure
        # is shared. A pack hit supplies ``plan=`` and bypasses this cache entirely.
        plan_cache = PlanTemplateCache()

        def build(name, *args, **kw):
            prog, built_plan = _compile_split(*args, plan=stored(name), plan_cache=plan_cache, **kw)
            plans[name] = built_plan
            return prog

        # A CODED trunk (``load_quantized_split(..., compress_trunk=True)``) binds its constants
        # from the CHECKPOINT rather than from the twin's parameters, which are placeholders
        # there. ``id_to_key`` maps a parameter tensor's identity to its full checkpoint path, so
        # each wrapper's trace can be re-addressed and the trellis speller can fire (see
        # :func:`_retarget_constants`). Everything else keeps the module lane.
        ckpt = None
        if (expert_store or {}).get("trunk") == "codes":
            id_to_key = {
                id(t): path
                for path, t in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))
            }
            ckpt = (expert_store["dir"], id_to_key)

        # One arena for every program this runner builds: layers run sequentially, so
        # all layers' activation buffers + scratch slabs share one layer's worth of
        # device memory instead of scaling with num_layers.
        arena = BufferArena()
        # Fixed-slot programs also execute serially, but their scratch must not alias the trunk
        # arena because the post program's h/xn outputs stay live across every slot launch.
        # One separate arena can safely serve every slot and every expert shape group.
        slot_arena = BufferArena()
        moe_meta: list = []  # per-layer: None (dense) or the router + 3-D expert weight tensors
        # ONE expert program set per SHAPE GROUP, shared by every MoE layer in that group
        # (weights are inputs): the symbolic any-width program plus static decode-bucket / M=1
        # twins (the decode hot path — the symbolic grids are the wrong shape at decode widths)
        # and the fixed-slot instances. Homogeneous MoEs (gpt-oss, OLMoE) have exactly one
        # group, and group 0 keeps the original pack names so their packs stay valid; EXL3's
        # mixed bit allocation is what makes a second group appear — K varies per layer, so the
        # codes shape does, so the baked program does.
        expert_tiers: list[dict] = []
        expert_groups: dict[tuple, int] = {}  # shape key → group index
        moe_top_k = int(getattr(text_config, "num_experts_per_tok", 0) or 0)

        def _build_expert_group(g, experts, einputs, expert_w, expert_fmt, trellis, has_bias, layer):
            """Compile one expert SHAPE GROUP's whole program set: the symbolic any-width
            program, the static decode-bucket / M=1 / M=256 twins, and the ``top_k`` fixed-slot
            instances of the indirect M=1 twin. Returns the tier dict every layer of this group
            routes through (:meth:`_launch_expert`, :meth:`_moe_combine_slots`).

            Group 0 keeps the original pack names (``moe.expert.sym`` and friends), so a
            single-group model — every shipped MoE image today — packs exactly as before; later
            groups take a ``g<N>`` infix. Each twin keeps its own failure contract: a compile
            failure drops that tier FOR THIS GROUP and the symbolic program stays the fallback.
            The fixed-slot tier is the one all-or-nothing decision, resolved by the caller: a
            whole-step capture cannot mix a captured group with a routed one."""
            pre = "moe.expert." if g == 0 else f"moe.expert.g{g}."
            # Forward-arg examples in wrapper-signature order (weights, then biases) at
            # the VALUE dtype — the trace is quantization-blind; fp8 bits and trellis
            # codes alike trace as the dense logical weight. Quantized experts append
            # the scale examples (the speller appends the scale inputs after the forward
            # args, in spec order); EXL3 passes its channel vectors by NAME instead.
            if trellis:
                inter, hidden_e = experts.gate_up_proj.shape[1] // 2, experts.gate_up_proj.shape[2]
                logical = {"w_gate": (inter, hidden_e), "w_up": (inter, hidden_e), "w_down": (hidden_e, inter)}
                w_names = list(logical)
            else:
                w_names = ["w_gate_up", "w_down"] + (["b_gate_up", "b_down"] if has_bias else [])
                logical = {n: tuple(einputs[n].shape[1:]) for n in w_names}
            example_w = [torch.zeros(*logical[n], dtype=dtype) for n in w_names]
            quant_specs = trellis_specs = aux_examples = None
            if expert_fmt is not None and not trellis and einputs["w_gate_up"].dtype == torch.uint8:
                quant_specs = {
                    "w_gate_up": (expert_fmt, tuple(einputs["w_gate_up_scale"].shape[1:]), "f32"),
                    "w_down": (expert_fmt, tuple(einputs["w_down_scale"].shape[1:]), "f32"),
                }
                example_w += [torch.zeros(*einputs[f"{n}_scale"].shape[1:], dtype=torch.float32) for n in ("w_gate_up", "w_down")]
            if trellis:
                cbs = (expert_store or {}).get("codebooks", {}).get(layer) or {}
                trellis_specs = {n: (int(cbs.get(n, 0)), tuple(einputs[n].shape[1:])) for n in w_names}
                # The build feed by NAME: the codes replace their dense trace placeholder
                # (the speller re-mints the input at the packed shape) and the channel
                # vectors are new inputs. Each takes the store's own per-expert shape.
                aux_examples = {
                    f"{n}{leaf}": torch.zeros(*einputs[f"{n}{leaf}"].shape[1:], dtype=einputs[f"{n}{leaf}"].dtype)
                    for n in w_names
                    for leaf in ("", "_suh", "_svh")
                }
            # EVERY per-expert input, in a stable order: the fixed-slot tier
            # table-resolves exactly these, and they are the expert program's whole
            # weight-streaming floor (expert weights are graph inputs).
            ind_names = (
                tuple(w_names) + tuple(f"{n}_scale" for n in quant_specs or ()) + tuple(n for n in aux_examples or () if n not in w_names)
            )
            expert_kw = {
                "quant_specs": quant_specs,
                "trellis_specs": trellis_specs,
                "aux_examples": aux_examples,
                "weight_inputs": ind_names,
            }
            tiers = {"sym": None, "bucket": None, "one": None, "m256": None, "slots": None, "ind_names": ind_names}
            with torch.device("cpu"):
                if not static_only:
                    tiers["sym"] = build(
                        pre + "sym",
                        expert_w,
                        [torch.zeros(8, hidden, dtype=dtype), *example_w],
                        ["x"],
                        np_dtype,
                        arena=arena,
                        capacity=max_tokens,
                        **expert_kw,
                    )
                # Static expert twins — the decode hot path. Same failure contract as
                # the per-layer twins: a compile failure drops the tier, the symbolic
                # program stays the correct fallback. Names avoid the ".decode"/".m1"
                # pack keep-filter suffixes — these tiers are not gated by the per-layer
                # twin flags.
                for tier, m in (("bucket", decode_bucket if decode_ok else 0), ("one", 1), ("m256", _EXPERT_PREFILL_M)):
                    # The M=256 prefill twin is the mean per-expert chunk width (§4 of the
                    # dispatch design); it only pays above the decode bucket (an equal-or-larger
                    # bucket fully shadows it).
                    if not m or (tier == "m256" and (static_only or m <= max(decode_bucket or 0, 0))):
                        continue
                    try:
                        tiers[tier] = build(
                            pre + tier,
                            expert_w,
                            [torch.zeros(m, hidden, dtype=dtype), *example_w],
                            None,
                            np_dtype,
                            arena=arena,
                            **expert_kw,
                        )
                    except Exception as ex:  # noqa: BLE001 — any compile failure → the group's symbolic program
                        logger.warning(
                            "[gen_runner] expert %s twin compile failed for group %d (%s); that width rides a wider tier", tier, g, ex
                        )
                # FIXED-SLOT decode tier (MoE): k slot instances of the INDIRECT twin of the M=1
                # expert program (`moe.expert.one.ind`). The twin compiles the same graph with
                # every per-expert input marked as an indirect operand — each kernel fetches its
                # weight base pointer from a device table indexed by the router's indices tensor
                # (`table[sel[slot]]`), so the slots read the E-stacked expert tensors DIRECTLY
                # with no per-step weight staging. Slot instances share one DEDICATED arena with
                # each other, but not with routed tiers: all slot launches are ordered on one
                # stream, so their input/scratch lifetimes never overlap; each output is rebound
                # below to its own row in `_slot_partials`. This avoids multiplying a ~100-MiB
                # scratch set by top-k while preserving every captured pointer. Slot `j`'s plan
                # stamps `slot=j` (`_plan_with_slot`). A schedule that stages an indirect operand
                # through a TMA descriptor fails the compile LOUDLY (descriptors bake the base
                # address at encode) and single-token decode keeps the routed path (eager; the
                # boot capture guard sees the tier gone). A stale pre-indirect pack plan fails
                # `_indirect_covered` and recompiles — no half-hit.
                if tiers["one"] is not None and moe_top_k > 0:
                    stored_ind = stored(pre + "one.ind")
                    if stored_ind is not None and not _indirect_covered(stored_ind, ind_names):
                        logger.warning(
                            "[gen_runner] stored %sone.ind plan lacks the indirect-operand encoding (stale pack) — recompiling", pre
                        )
                        stored_ind = None
                    try:
                        slot_example = [torch.zeros(1, hidden, dtype=dtype), *example_w]
                        slot0, ind_plan = _compile_split(
                            expert_w,
                            slot_example,
                            None,
                            np_dtype,
                            plan=stored_ind,
                            plan_cache=plan_cache,
                            indirect_inputs=ind_names,
                            arena=slot_arena,
                            **expert_kw,
                        )
                        if not _indirect_covered(ind_plan, ind_names):
                            raise RuntimeError(
                                "indirect twin compile left a weight arg unmarked (the kernel consumes it under a derived buffer name)"
                            )
                        tiers["slots"] = [slot0] + [
                            _compile_split(
                                expert_w,
                                slot_example,
                                None,
                                np_dtype,
                                plan=_plan_with_slot(ind_plan, j),
                                aux_examples=aux_examples,
                                weight_inputs=ind_names,
                                arena=slot_arena,
                            )[0]
                            for j in range(1, moe_top_k)
                        ]
                        plans[pre + "one.ind"] = ind_plan
                    except Exception as ex:  # noqa: BLE001 — any compile failure → routed fallback, loudly
                        logger.warning(
                            "[gen_runner] indirect expert twin build failed for group %d (%s); single-token decode keeps the "
                            "routed expert path (eager, no whole-step capture)",
                            g,
                            ex,
                        )
                if static_only and tiers["bucket"] is None:
                    raise RuntimeError(
                        f"expert group {g} static decode compile failed while bucket {decode_bucket} "
                        f"covers the configured capacity {max_tokens}; no symbolic fallback was allocated"
                    )
            return tiers

        for i, block in enumerate(layers):
            meta = _meta(block.self_attn)
            attn_meta.append(meta)
            attn_width = meta[1] * meta[0]  # this layer's num_heads * head_dim (gemma-4: global ≠ sliding)
            logger.info("[gen_runner] compiling layer %d/%d (pre + post%s)...", i + 1, len(layers), " + decode" if decode_ok else "")
            moe_parts = moe_block_parts(block.mlp) if hasattr(block, "mlp") else None
            if moe_parts is not None:
                import copy

                gate, experts = moe_parts
                expert_fmt = (expert_store or {}).get("fmt")
                trellis = expert_fmt == "exl3"
                pre_w, post_w, expert_w = build_moe_split_wrapper(block, split_gate_up=trellis)
                transposed, interleaved, has_bias = moe_expert_layout(experts)
                # The per-layer expert INPUT tensors, keyed by the expert program's input
                # names. A quantized checkpoint supplies them from the shard stream (fp8 bits
                # + f32 scales + dtype biases — the twin's expert params stay meta); otherwise
                # they come off the twin, de-interleaved here when the layout needs it so the
                # wrapper's chunk-half gate/up split holds.
                einputs = (expert_store or {}).get("layers", {}).get(i)
                if einputs is None:
                    einputs = {"w_gate_up": experts.gate_up_proj.detach().to(dtype), "w_down": experts.down_proj.detach().to(dtype)}
                    if has_bias:
                        einputs["b_gate_up"] = experts.gate_up_proj_bias.detach().to(dtype)
                        einputs["b_down"] = experts.down_proj_bias.detach().to(dtype)
                    if interleaved:
                        for nm in ("w_gate_up", "b_gate_up"):
                            if nm in einputs:
                                einputs[nm] = deinterleave_gate_up(einputs[nm])
                moe_meta.append(
                    {
                        "gate": copy.deepcopy(gate).to(dtype),
                        "inputs": einputs,
                        # k routed experts per token — the fixed-slot tier's slot count.
                        "top_k": int(getattr(text_config, "num_experts_per_tok", 0) or 0),
                    }
                )
                # ONE expert program serves every MoE layer, so the expert shape and activation
                # must be uniform across layers — verify, don't assume.
                act_fn = getattr(experts, "act_fn", None)
                act_tag = (
                    type(act_fn).__name__
                    if act_fn is not None
                    else f"clamped_swiglu[{getattr(experts, 'alpha', None)},{getattr(experts, 'limit', None)}]"
                )
                # Every per-expert tensor's per-layer shape (plus the codebook ids on the EXL3
                # path) is in the key: the shared program bakes the CODES shape too, and EXL3's
                # mixed allocation lets K vary per layer.
                shape_key = (
                    tuple(sorted((n, tuple(t.shape[1:]), str(t.dtype)) for n, t in einputs.items())),
                    tuple(sorted(((expert_store or {}).get("codebooks", {}).get(i) or {}).items())),
                    act_tag,
                    transposed,
                    has_bias,
                )
                g = expert_groups.get(shape_key)
                if g is None:
                    g = expert_groups[shape_key] = len(expert_tiers)
                    expert_tiers.append(_build_expert_group(g, experts, einputs, expert_w, expert_fmt, trellis, has_bias, i))
                moe_meta[-1]["group"] = g
            else:
                pre_w, post_w = build_attention_split_wrapper(block)
                moe_meta.append(None)
            # Per-wrapper device-constant caches: the symbolic program and its static
            # decode-bucket twin bind the SAME weights — share one upload, not two.
            pre_consts: dict = {}
            post_consts: dict = {}
            with torch.device("cpu"):
                if not static_only:
                    pre_programs.append(
                        build(
                            f"L{i:02d}.pre.sym",
                            pre_w,
                            [torch.zeros(8, hidden, dtype=dtype)],
                            ["hidden"],
                            np_dtype,
                            dev_consts=pre_consts,
                            ckpt=ckpt,
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
                            ckpt=ckpt,
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
                                ckpt=ckpt,
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
                                ckpt=ckpt,
                                arena=arena,
                            )
                        )
                    except Exception as ex:  # noqa: BLE001 — any lowering/compile failure → disable the bucket
                        if static_only:
                            raise RuntimeError(
                                f"layer {i} static decode compile failed while bucket {decode_bucket} "
                                f"covers configured capacity {max_tokens}; no symbolic fallback was allocated"
                            ) from ex
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
                                ckpt=ckpt,
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
                                ckpt=ckpt,
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
                                ckpt=ckpt,
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
                                ckpt=ckpt,
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
        # MoE post programs are EXCLUDED from every chaining block below (the two-output guard):
        # their layer output is a fresh torch tensor (h + expert combine), so a rewire buys no
        # skip — and h must survive the torch interlude, not sit on the next pre's input backing.
        if decode_ok and pre_decode and post_decode:
            pre_in = pre_decode[0].input_names[0]
            shared_in = pre_decode[0].program.arrays[pre_in]
            for prog in post_decode:
                if len(prog.output_names) != 1:
                    continue
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
                if len(prog.output_names) != 1:
                    continue
                out_name = prog.output_names[0]
                cur = prog.program.arrays[out_name]
                if cur.shape == m1_shared.shape and cur.dtype == m1_shared.dtype:
                    prog.program.arrays[out_name] = cp.ndarray(m1_shared.shape, dtype=m1_shared.dtype, memptr=m1_shared.data)
        # A2: the same chaining for the SYMBOLIC programs (capacity-view buffers — device path
        # only, `max_tokens` set; the oracle's host `rebind` re-takes arena views per call and
        # neither needs nor keeps the rewire) and for the static prefill-chunk twins. The skip
        # fires on pointer equality, so it pays exactly where the caller receives a VIEW of the
        # post output — i.e. under an outer capture (the A1 no-clone branch): the per-layer seam
        # copy node drops from captured over-bucket sym decode graphs. On EAGER steps the
        # protective output clone breaks pointer equality and the upload still copies — the
        # chunk-twin chaining is groundwork that activates when chunk steps get whole-step
        # captured (the recorded future work), not an eager win today. Rider steps stay safe
        # under the cross-tier aliasing (all tiers' views share one backing base): they are
        # eager by construction (prefill is never captured), and `run_device`'s uncaptured path
        # CLONES the chunk-twin head before the decode-twin tail overwrites the shared rows.
        if max_tokens is not None and pre_programs and post_programs:
            sym_in = pre_programs[0].input_names[0]
            sym_shared = pre_programs[0].program.arrays[sym_in]
            for prog in post_programs:
                if len(prog.output_names) != 1:
                    continue
                out_name = prog.output_names[0]
                if prog.program.arrays[out_name].nbytes == sym_shared.nbytes:
                    prog.program.arrays[out_name] = sym_shared
        if prefill_ok and pre_prefill and post_prefill:
            pf_in = pre_prefill[0].input_names[0]
            pf_shared = pre_prefill[0].program.arrays[pf_in]
            for prog in post_prefill:
                if len(prog.output_names) != 1:
                    continue
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
        if plan_cache.hits or plan_cache.misses:
            logger.info(
                "[gen_runner] structural plan cache: %d hit(s), %d miss(es), %d template(s)",
                plan_cache.hits,
                plan_cache.misses,
                len(plan_cache),
            )
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
            moe=moe_meta if any(m is not None for m in moe_meta) else None,
            expert_tiers=expert_tiers or None,
        )
        runner._embed_scale = embed_scale  # raw-table scale for adopt_embed_table (host table keeps it folded)
        if runner.has_device_decode or (max_tokens is not None and runner._moe is not None):
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
        named += [
            (f"moe.expert.{'' if g == 0 else f'g{g}.'}{tag}", tiers[key])
            for g, tiers in enumerate(expert_tiers)
            for key, tag in (("bucket", f"bucket.m{decode_bucket}"), ("one", "one.m1"), ("m256", "m256"))
            if tiers[key] is not None
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
        if not self._pre:
            raise RuntimeError(f"token width {t} exceeds static-only capacity {self.prefill_capacity}")
        return tuple(self._pre[layer].run([h]))

    def forward_layer_post(self, layer, attn_out, residual):
        """``(attn_out[T,Hq·D], residual[T,H])`` numpy → ``layer_out[T, H]`` numpy. Decode-bucketed
        like ``forward_layer_pre``."""
        if self._moe is not None and self._moe[layer] is not None:
            raise NotImplementedError(
                "MoE layers run device-resident only — the routed expert dispatch has no host numpy path; "
                "serve within the compiled programs' token capacity"
            )
        a = attn_out.astype(self._np_dtype, copy=False)
        r = residual.astype(self._np_dtype, copy=False)
        t = a.shape[0]
        if self._post_decode is not None and t <= self._decode_bucket:
            out = self._post_decode[layer].run([_pad_rows(a, self._decode_bucket), _pad_rows(r, self._decode_bucket)])[0]
            return out[:t]
        if not self._post:
            raise RuntimeError(f"token width {t} exceeds static-only capacity {self.prefill_capacity}")
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

        from emmy import config as emmy_config

        if getattr(self, "_embed_weight_dev", None) is None:
            if emmy_config.gen_embed_host():
                self._embed_weight, self._embed_weight_dev = self._map_embed_table_to_host()
            else:
                self._embed_weight_dev = torch.from_numpy(self._embed_weight).cuda()
        self._norm_dev = copy.deepcopy(self._norm).to("cuda")
        if self._moe is not None:
            # The routers and the 3-D expert weight tensors move to CUDA HERE — eagerly, inside
            # vLLM's profiled footprint (same contract as the embed table above). The expert
            # tensors ARE the model's dominant weights. The per-expert cupy views are minted
            # ONCE here (``cp.from_dlpack`` negotiates a stream sync per call — 2·E·layers of
            # them don't belong on the per-step path) and reused by every expert launch.
            import cupy as cp

            from emmy.compiler.backend.gpu_lock import gpu_lock

            with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
                # Program construction churns both allocator pools. Release their FREE blocks
                # before the dominant expert-store transfer: vLLM cannot reclaim them until
                # ``load_weights`` returns, and on a 32-GiB card that ordering alone can make
                # the final layer's codes fail despite enough live-memory headroom.
                before = torch.cuda.mem_get_info()[0]
                torch.cuda.empty_cache()
                cp.get_default_memory_pool().free_all_blocks()
                free, total = torch.cuda.mem_get_info()
                logger.info(
                    "[gen_runner] pre-expert allocator reclaim: %.3f GiB released; %.3f of %.3f GiB free",
                    (free - before) / 2**30,
                    free / 2**30,
                    total / 2**30,
                )
                for m in self._moe:
                    if m is None:
                        continue
                    m["gate"] = m["gate"].to("cuda")
                    m["inputs"] = {n: t.cuda() for n, t in m["inputs"].items()}
                    m["inputs_cp"] = {n: [cp.from_dlpack(w) for w in t] for n, t in m["inputs"].items()}
                if self._slots_ok:
                    # Fixed-slot indirect tables: PER SHAPE GROUP, one flat [group layers * E]
                    # int64 pointer table per expert-input kind (weights, biases, and — on the
                    # fp8 path — the scale tensors; entry = the device address of one expert's
                    # slice of a layer's E-stacked tensor), a [k] int32 SELECTOR the router
                    # rewrites per layer per step (top-k indices + the layer's offset within its
                    # OWN group's table), and a [k, H] partials tensor the slot OUTPUTS land in.
                    # Selector and partials are SHARED across groups — one layer runs at a time,
                    # and the tables are what a group's slot kernels differ on. A table is
                    # layer-invariant within its group — the layer offset rides in the selector
                    # VALUES, which are device data — so per-slot cached graphs and vLLM's
                    # whole-step capture both bake one fixed table/selector pointer and the slot
                    # kernels read the E-tensors directly (no per-step weight staging; the
                    # ~100 MB stage-1 staging pair is gone, returned to vLLM's KV-cache budget).
                    # Slot i's table/selector/output arrays are bound here ONCE.
                    import numpy as np  # noqa: PLC0415

                    k = len(self._expert_tiers[0]["slots"])
                    h = self._embed_weight.shape[1]
                    self._slot_sel = torch.zeros(k, dtype=torch.int32, device="cuda")
                    act_dtype = torch.from_numpy(np.empty(0, dtype=self._np_dtype)).dtype
                    self._slot_partials = torch.empty(k, h, dtype=act_dtype, device="cuda")
                    sel_cp = cp.from_dlpack(self._slot_sel)
                    self._slot_tables = []
                    for g, tiers in enumerate(self._expert_tiers):
                        members = [m for m in self._moe if m is not None and m["group"] == g]
                        num_e = next(iter(members[0]["inputs"].values())).shape[0]  # every per-expert tensor is E-leading
                        tables: dict[str, list[int]] = {n: [] for n in members[0]["inputs"]}
                        for m in members:
                            m["sel_off"] = len(next(iter(tables.values())))
                            for name, ptrs in tables.items():
                                w = m["inputs"][name]
                                ptrs.extend(w.data_ptr() + e * w.stride(0) * w.element_size() for e in range(num_e))
                        dev_tables = {n: torch.tensor(p, dtype=torch.int64, device="cuda") for n, p in tables.items()}
                        self._slot_tables.append(dev_tables)
                        table_cps = {n: cp.from_dlpack(t) for n, t in dev_tables.items()}
                        for j, slot in enumerate(tiers["slots"]):
                            p = slot.program
                            for name, tcp in table_cps.items():
                                p.arrays[f"{name}__table"] = tcp
                                p.arrays[f"{name}__sel"] = sel_cp
                            p.arrays[slot.output_names[0]] = cp.from_dlpack(self._slot_partials[j : j + 1])
                            # The build allocated the (never-read) direct per-expert input
                            # buffers — drop them (~two experts' weights per slot) so vLLM's
                            # KV-cache profiling sees the memory.
                            for name in table_cps:
                                p.arrays[name] = cp.empty(0, dtype=p.arrays[name].dtype)
                    cp.get_default_memory_pool().free_all_blocks()
        self._dev_ready = True

    def _map_embed_table_to_host(self):
        """Move the token-embedding table into **mapped host memory** and return
        ``(numpy view, torch CUDA view)`` of that one allocation (``EMMY_GEN_EMBED_HOST``).

        The allocation is ``cudaHostAlloc``-mapped, so under unified virtual addressing its host
        address is also a valid device address: a gather kernel indexes it exactly as it indexes
        device memory, reading the rows over PCIe. That keeps ``embed_device`` a plain device-side
        op with no host round trip, so the whole-step decode capture still records it — the reason
        this is worth doing rather than gathering on the host.

        The table then costs no VRAM at all. On a card the weights already fill, that is straight
        KV-cache budget: at a 151552-row fp16 vocab it is 1.156 GiB, and an untied checkpoint
        shares it with nothing. The price is PCIe latency per embedded token (~2 bytes x hidden
        per row), paid on every step in proportion to its width.

        Host memory does not double: the numpy table the oracle / prefill fallback gathers from is
        rebound to a view of the SAME buffer, and the original host array is dropped."""
        import ctypes  # noqa: PLC0415

        import cupy as cp  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        table = np.ascontiguousarray(self._embed_weight)
        rt = cp.cuda.runtime
        pinned = cp.cuda.PinnedMemory(table.nbytes, flags=rt.hostAllocMapped | rt.hostAllocPortable)
        host = np.frombuffer((ctypes.c_char * table.nbytes).from_address(pinned.ptr), dtype=table.dtype).reshape(table.shape)
        host[...] = table
        # UVA makes the host address the device address; assert it rather than trust it — a
        # platform where it does not hold would hand the gather kernel an unmapped pointer.
        attrs = rt.pointerGetAttributes(pinned.ptr)
        if int(attrs.devicePointer) != int(pinned.ptr):
            raise RuntimeError("EMMY_GEN_EMBED_HOST: this platform does not map host allocations into the device address space")
        mem = cp.cuda.UnownedMemory(pinned.ptr, table.nbytes, pinned)
        arr = cp.ndarray(table.shape, dtype=table.dtype, memptr=cp.cuda.MemoryPointer(mem, 0))
        self._embed_pinned = (pinned, arr)  # keepalive: the views below do not own the allocation
        logger.info(
            "[gen_runner] embed table (%d x %d, %.3f GiB) mapped in host memory — 0 device bytes",
            table.shape[0],
            table.shape[1],
            table.nbytes / 2**30,
        )
        return host, torch.from_dlpack(arr)

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

    def _rider_dest(self, name, rows, cols, ref):
        """A3: the shared joint destination for a rider step's combined result — one persistent
        tensor per ``(name, cols)`` sized ``[prefill_bucket + rider_width, cols]``, sliced to the
        step's rows. Shared across layers: q/k/v are consumed within their layer (RoPE + paged
        attention + KV-cache write), and the hidden dest's residual read happens at the NEXT
        post's upload, before that post's rider halves overwrite the dest. Cached per column
        width because gemma-4's global layers are wider than its sliding ones."""
        import torch  # noqa: PLC0415

        dests = getattr(self, "_rider_dests", None)
        if dests is None:
            dests = self._rider_dests = {}
        cap = self._prefill_bucket + self.rider_width
        d = dests.get((name, cols))
        if d is None:
            d = dests[(name, cols)] = torch.empty(cap, cols, dtype=ref.dtype, device=ref.device)
        return d[:rows]

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
            # A3: both halves copy ONCE, straight into slices of one shared joint destination —
            # no torch.cat (which allocated 3 tensors and re-copied every row per layer per
            # rider step). Rider steps are eager by construction, run_device's out= contract.
            pb = self._prefill_bucket
            hd, nh, nkv, _ = self._attn_meta[layer]
            widths = (nh * hd, nkv * hd, nkv * hd)
            dests = [self._rider_dest(nm, t, w, hidden) for nm, w in zip(("q", "k", "v"), widths, strict=True)]
            self._pre_prefill[layer].run_device([hidden[:pb]], out=[d[:pb] for d in dests])
            self._pre_decode[layer].run_device([hidden[pb:]], out=[d[pb:] for d in dests])
            return tuple(dests)
        self._warn_symbolic_decode(t)
        if not self._pre:
            raise RuntimeError(f"token width {t} exceeds static-only capacity {self.prefill_capacity}")
        return tuple(self._pre[layer].run_device_sym([hidden]))

    def forward_layer_post_device(self, layer, attn_out, residual):
        """Device twin of :meth:`forward_layer_post`: ``(attn_out, residual)`` CUDA → ``[T,H]``
        CUDA. Decode-bucketed / exact-chunk / symbolic-routed like
        :meth:`forward_layer_pre_device`. A MoE layer's post program returns ``(h, xn)``; the
        routed expert dispatch + weighted combine run here in torch (the third seam) and the
        layer output is ``h + combine``."""
        outs = self._route_post_device(layer, attn_out, residual)
        moe = self._moe[layer] if self._moe is not None else None
        if moe is None:
            return outs[0]
        h, xn = outs
        # Single-token decode rides the FIXED-SLOT combine (capture-legal — no host sync, a
        # fixed launch set); every wider step keeps the routed dispatch, whose launch set
        # varies with the routing (eager only).
        if self._slots_ok and xn.shape[0] == 1:
            return h + self._moe_combine_slots(moe, xn)
        return h + self._moe_combine(moe, xn)

    def _route_post_device(self, layer, attn_out, residual):
        """Tier-route one post program launch; returns the full output list (1 output for a
        dense layer's post, 2 — ``h, xn`` — for a MoE layer's post_attn)."""
        t = attn_out.shape[0]
        if t == 1 and self._post_m1 is not None:
            return self._post_m1[layer].run_device([attn_out, residual])
        if self._post_decode is not None and t <= self._decode_bucket:
            return self._post_decode[layer].run_device([attn_out, residual])
        if self._post_prefill is not None and t == self._prefill_bucket:
            return self._post_prefill[layer].run_device([attn_out, residual])
        if 0 < t - self._prefill_bucket <= self.rider_width:
            # A3: same slice-bound joint destination as the pre path. The residual reads are
            # ordered before the overwrites: each half's upload copies its residual slice into
            # the program's own buffer before that half's kernels run, and the NEXT layer's
            # post uploads its residual (this dest) before its own halves overwrite it.
            pb = self._prefill_bucket
            names = ("hidden",) if len(self._post_prefill[layer].output_names) == 1 else ("hidden", "moe_xn")
            dests = [self._rider_dest(nm, t, residual.shape[1], residual) for nm in names]
            self._post_prefill[layer].run_device([attn_out[:pb], residual[:pb]], out=[d[:pb] for d in dests])
            self._post_decode[layer].run_device([attn_out[pb:], residual[pb:]], out=[d[pb:] for d in dests])
            return dests
        if not self._post:
            raise RuntimeError(f"token width {t} exceeds static-only capacity {self.prefill_capacity}")
        return self._post[layer].run_device_sym([attn_out, residual])

    def _moe_combine(self, moe, xn):
        """The torch half of the MoE third seam: route via the HF router module (linear +
        softmax + top-k — ops the tracer cannot map), launch the tier-routed shared expert
        program once per HIT expert on that expert's routed rows, and weighted-scatter the
        partials. ``xn[T, H]`` → combined expert output ``[T, H]``. The GPU lock and the cupy
        external-stream bind are hoisted around the WHOLE per-expert loop — per-launch framing,
        not the weight bytes, was the measured decode wall (~0.23 ms/launch through the
        symbolic per-call path)."""
        import cupy as cp
        import torch

        from emmy.compiler.backend.gpu_lock import gpu_lock

        self._ensure_device()
        gated = moe["gate"](xn)
        with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
            return combine_routed_experts(xn, gated, lambda e, rows: self._launch_expert(moe, e, rows))

    def _moe_combine_slots(self, moe, xn):
        """Fixed-slot combine for single-token decode (``T == 1``) — the capture-legal twin of
        :meth:`_moe_combine`. The routing stays data-dependent in VALUES only: one fixed-shape
        write puts the router's top-k indices (plus this layer's table offset, cast to the
        kernel's int32) into the persistent selector — no ``unique()``, no ``.tolist()``, no
        host sync — and the k slot instances of the indirect M=1 expert program resolve their
        weight base pointers in-kernel from the device tables (``table[sel[slot]]``), reading
        the 3-D expert tensors directly; the partials combine through one score matmul. Every
        op is fixed-shape with a fixed launch set, so a whole-step decode CUDA graph records
        it; under an outer capture each slot issues its raw launch sequence, mirroring
        :meth:`_Program.run_device` (nested graph machinery is illegal in a capturing stream).
        ``combine_routed_experts`` (the routed path) stays the parity oracle."""
        import cupy as cp
        import torch

        from emmy.compiler.backend.gpu_lock import gpu_lock

        self._ensure_device()
        gated = moe["gate"](xn)
        scores, indices = gated[-2], gated[-1]  # [1, k] each
        scores = scores.to(xn.dtype)
        self._slot_sel.copy_(indices.reshape(-1) + moe["sel_off"])
        capturing = torch.cuda.is_current_stream_capturing()
        with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
            x_cp = cp.from_dlpack(xn.detach().contiguous())
            for slot in self._expert_tiers[moe["group"]]["slots"]:
                p = slot.program
                p.upload_prefix_device({"x": x_cp})
                if capturing:
                    p.run_once()
                else:
                    p.capture_program_graph()  # static program → one cached graph per slot
                    p.replay_program_graph()
        return scores @ self._slot_partials  # [1, k] @ [k, H] — the fixed-shape weighted combine

    def _launch_expert(self, moe, e, rows):
        """One expert-program launch on ``rows`` routed rows (caller holds the GPU lock + the
        external-stream bind). Tier routing mirrors the main programs: M=1 twin, then the
        decode-bucket twin (pad → run → slice; stale prefix rows are per-token-independent),
        then the static M=256 prefill twin for row sets up to ``_EXPERT_PREFILL_M`` (same
        pad-up contract), then the symbolic program. A swap-safe tier (no TMA descriptors —
        descriptors bake pointers at build) takes the per-expert weight slices by POINTER SWAP
        into ``program.arrays`` — no D2D weight copy; otherwise the slices upload normally.
        The M=256 twin instead replays its captured whole-program graph
        (``capture_program_graph`` — one host call per expert instead of per-kernel Python
        framing; at ~64 expert launches per chunk the framing was ~3× the GPU work). The
        capture bakes the twin's own buffer pointers, so its weights always arrive by UPLOAD
        (a pointer swap would freeze the first expert's slices into every later replay).
        Eager only — on torch's current stream, never under an outer capture (MoE
        serves eager above T=1; see the boot guard)."""
        import cupy as cp
        import torch

        t = rows.shape[0]
        per_e = {name: views[e] for name, views in moe["inputs_cp"].items()}
        tiers = self._expert_tiers[moe["group"]]  # this layer's shape group
        if tiers["m256"] is not None and self._decode_bucket < t <= _EXPERT_PREFILL_M:
            prog = tiers["m256"]
            p = prog.program
            p.upload_prefix_device({"x": cp.from_dlpack(rows.detach().contiguous()), **per_e})
            p.capture_program_graph()  # static program → one cached graph, replayed per expert
            p.replay_program_graph()
            outs = p.output_prefix_device()
            return torch.from_dlpack(outs[prog.output_names[0]])[:t]
        if t == 1 and tiers["one"] is not None:
            prog, sym = tiers["one"], False
        elif tiers["bucket"] is not None and t <= self._decode_bucket:
            prog, sym = tiers["bucket"], False
        else:
            prog, sym = tiers["sym"], True
            if prog is None:
                raise RuntimeError(f"expert row width {t} exceeds the static-only decode bucket {self._decode_bucket}")
        p = prog.program
        feed = {"x": cp.from_dlpack(rows.detach().contiguous())}
        if self._expert_swap_safe[id(prog)]:
            for name, view in per_e.items():
                p.arrays[name] = view
        else:
            feed.update(per_e)
        if sym:
            p.set_sym_values({"num_tokens": t})
        p.upload_prefix_device(feed)
        p.run_once()
        outs = p.output_prefix_device({"num_tokens": t} if sym else None)
        # A VIEW of the shared output buffer: the caller's weighted index_add_ consumes it
        # before the next expert launch overwrites it (same stream, ordered).
        return torch.from_dlpack(outs[prog.output_names[0]])[:t]

    def post_attn_backing(self, layer: int, rows: int):
        """A torch CUDA view (first ``rows`` rows) of the ``attn_out`` INPUT backing of the post
        program that :meth:`forward_layer_post_device` will route ``rows`` to — EVERY tier since
        A4: M=1, the decode bucket, the exact prefill chunk, and the symbolic program. vLLM's
        paged attention writes into this view under ``EMMY_GEN_ALIAS_ATTN`` so the post upload
        self-copy-skips — the attention→post seam copy drops from captured decode graphs (static
        and over-bucket sym alike) and skips on eager chunk steps. ``None`` when no tier covers
        ``rows`` or the alias cannot be honored — notably RIDER widths: the step splits across
        two programs, and one contiguous attention output cannot alias two buffers.

        The tier routing MUST mirror :meth:`forward_layer_post_device` — a mismatch is still
        correct (the upload simply copies) but silently loses the skip. The wrap is CACHED per
        ``(layer, tier)``: ``torch.from_dlpack`` on a cupy array negotiates a stream sync, which
        INVALIDATES an in-flight CUDA-graph capture — the view must be minted on an uncaptured
        (warmup) step and only re-served under capture. An uncached entry asked for mid-capture
        returns ``None`` (the caller falls back to the ordinary copy path)."""
        if rows == 1 and self._post_m1 is not None:
            handle, tier = self._post_m1[layer], "m1"
        elif self._post_decode is not None and rows <= self._decode_bucket:
            handle, tier = self._post_decode[layer], "decode"
        elif self._post_prefill is not None and rows == self._prefill_bucket:
            handle, tier = self._post_prefill[layer], "chunk"
        elif 0 < rows - self._prefill_bucket <= self.rider_width:
            return None  # rider split: two programs, no single contiguous attn_out backing
        elif self._prefill_capacity and rows <= self._prefill_capacity:
            handle, tier = self._post[layer], "sym"
        else:
            return None
        if handle is None:
            return None
        views = getattr(self, "_post_attn_views", None)
        if views is None:
            views = self._post_attn_views = {}
        view = views.get((layer, tier))
        if view is None:
            import torch  # noqa: PLC0415

            if torch.cuda.is_current_stream_capturing():
                return None
            arr = handle.program.arrays.get("attn_out")
            if arr is None:
                return None
            view = views[(layer, tier)] = torch.from_dlpack(arr)
        return view[:rows]

    def final_norm_device(self, hidden):
        """Apply the final norm on CUDA to a ``hidden[T,H]`` CUDA tensor."""
        import torch

        self._ensure_device()
        with torch.no_grad():
            return self._norm_dev(hidden)
