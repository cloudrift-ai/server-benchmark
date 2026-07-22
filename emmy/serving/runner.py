"""Trace + compile + per-sequence execution of an embedding-model trunk.

``EmmyForwardRunner`` owns the emmy side of the vLLM plugin: at
server start it traces the HuggingFace ``AutoModel`` trunk (hidden states out,
no lm_head) with a dynamic seq_len, compiles it through the CUDA backend
(greedy fork picks from the global prior — no GPU tuning), and builds ONE
``CompiledProgram`` over a single ``max_seq_len``-sized buffer set — one program
serves every request at any seq_len ≤ ``max_seq_len``.

Per request it captures (once per distinct seq_len, then replays) a whole-program
CUDA graph over that buffer set — one host-side launch instead of ~hundreds. Each
graph is captured at its EXACT seq_len, so every kernel runs at its exact grid (no
oversized-grid masking); the buffers are allocated once at ``max_seq_len`` and each
request's inputs upload into their contiguous prefix.

No vllm imports here — the class is driven by ``vllm_model.EmmyEmbedModel``
but is independently testable with torch + cupy alone.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from emmy.compiler.backend.cuda.program import CompiledProgram

logger = logging.getLogger(__name__)

# Example-tensor size handed to torch.export (the CLI's --seq-len default).
# The traced dim is symbolic, so the value never reaches the kernels.
_TRACE_SEQ = 32
# Per-seq_len causal-mask host cache (FIFO). At S=4096 one fp16 mask is
# ~32 MB, so the cap keeps the cache bounded while serving mixed lengths.
_MASK_CACHE_MAX = 16


def _causal_mask_np(s: int, np_dtype) -> np.ndarray:
    """``(1, 1, s, s)`` additive causal mask (0 on/below diagonal, -inf above),
    the numpy twin of ``trace.huggingface.build_causal_mask``."""
    mask = np.triu(np.full((s, s), -np.inf, dtype=np.float32), k=1)
    return mask.astype(np_dtype)[None, None, :, :]


class EmmyForwardRunner:
    """One compiled dynamic-seq_len trunk + its per-sequence execution."""

    def __init__(
        self,
        program: CompiledProgram,
        input_names: tuple[str, str, str],
        output_name: str,
        np_dtype,
        max_seq_len: int,
        batch_cap: int = 1,
        static: bool = False,
    ):
        self._program = program
        self._ids_name, self._mask_name, self._pos_name = input_names
        self._output_name = output_name
        self._np_dtype = np_dtype
        # Shared buffer set is sized for max_seq_len; every accepted request
        # (S ≤ max_seq_len) uses the captured-graph path.
        self.max_seq_len = max_seq_len
        # batch_cap == 1: symbolic-seq, one sequence per forward (the default).
        # batch_cap > 1: each scheduler step runs as a padded batched forward
        # (`forward_hidden_states_batched`) — either the batched SYMBOLIC-seq
        # program (static=False: batch extent baked at batch_cap, seq symbolic,
        # steps pad to the step's longest sequence) or the fully-STATIC
        # (batch_cap, max_seq_len) program (static=True: steps pad to max_seq_len).
        self.batch_cap = batch_cap
        self.static = static
        # int seq_len -> device-built (1,1,S,S) cupy causal mask (built on first
        # sight of each S, reused across same-S requests via a D2D prefix copy).
        self._mask_cache: dict = {}

    @classmethod
    def create(cls, model_id: str, max_seq_len: int, dtype_str: str = "float16", batch: int = 1, static: bool = False) -> EmmyForwardRunner:
        import hashlib

        import torch
        from transformers import AutoModel

        from emmy import config
        from emmy.compiler.backend.cuda.program import CompiledProgram
        from emmy.compiler.backend.gpu_lock import gpu_lock
        from emmy.compiler.backend.pack import load_pack, pack_path
        from emmy.compiler.backend.plan import apply_weight_loads
        from emmy.compiler.trace.dynamic import DYNAMIC_DIM_MAX
        from emmy.compiler.trace.huggingface import build_full_model_wrapper

        if max_seq_len > DYNAMIC_DIM_MAX:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds emmy's dynamic-dim max ({DYNAMIC_DIM_MAX}); "
                f"serve with --max-model-len {DYNAMIC_DIM_MAX} or lower"
            )
        dtype = getattr(torch, dtype_str)
        np_dtype = np.dtype(dtype_str)

        logger.info("[serving] loading %s trunk (dtype=%s)...", model_id, dtype_str)
        # vLLM instantiates models inside a CUDA device context; the HF trunk
        # is only traced + read for constants here (then freed), so force CPU —
        # this also sidesteps transformers' accelerate requirement for
        # non-default device contexts. The wrapper builds buffers and the trace
        # runs the forward on example tensors — all of it must sit on one device.
        with torch.device("cpu"):
            model = AutoModel.from_pretrained(model_id, dtype=dtype)
            model.eval()
            wrapper = build_full_model_wrapper(model, max_seq_len if static else _TRACE_SEQ, dtype, dynamic=True)

        # Pack lookup: model identity (id + config hash, so a same-config fine-tune shares the
        # pack) × the serving shape. A hit skips trace + pipeline + fork resolution + codegen;
        # any mismatch / missing cubin falls back to the full compile below.
        pack_key = {
            "kind": "embed-trunk",
            "model": model_id,
            "config_sha": hashlib.sha1(model.config.to_json_string().encode()).hexdigest()[:16],
            "max_seq_len": max_seq_len,
            "dtype": dtype_str,
            "batch": batch,
        }
        if batch > 1 and not static:
            # The batched SYMBOLIC-seq program compiles a different graph than the
            # historical fully-static (batch, max_seq_len) one at the same key —
            # disambiguate so an old static pack can never load into symbolic mode.
            pack_key["sym"] = True
        pack_at = pack_path(config.pack_dir(), pack_key) if config.pack_dir() is not None else None
        plan = None
        if pack_at is not None:
            plans = load_pack(pack_at, key=pack_key)
            plan = plans.get("trunk") if plans is not None else None

        # Weight sources in the traced dtype: named_buffers (NOT state_dict)
        # so non-persistent buffers — the wrapper's precomputed rotary
        # cos/sin — bind too; remove_duplicate=False so tied weights surface
        # under every traced alias.
        sources: dict[str, np.ndarray] = {}
        for path, t in wrapper.named_parameters(remove_duplicate=False):
            sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
        for path, t in wrapper.named_buffers(remove_duplicate=False):
            sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)

        if plan is not None:
            logger.info("[serving] pack hit at %s — skipping trace + compile", pack_at)
            from emmy.compiler.loader.binder import assemble_source

            const_feed = {
                nid: apply_weight_loads(src, w.load_ops)
                for nid, w in plan.weights.items()
                if w.load_ops is not None and (src := assemble_source(w, sources)) is not None
            }
        else:
            plan, const_feed = cls._trace_and_compile(wrapper, sources, max_seq_len, dtype, batch, static)
            if pack_at is not None:
                cls._save_pack(pack_at, plan, pack_key, model_id)

        if len(plan.inputs) != 3:
            raise RuntimeError(f"expected 3 graph inputs (input_ids, attention_mask, position_ids), got {plan.inputs}")
        if len(plan.outputs) != 1:
            raise RuntimeError(f"expected 1 graph output (hidden states), got {plan.outputs}")

        ids_name, mask_name, pos_name = plan.inputs
        # Shared buffer set sized at (batch, max_seq_len) capacity so each captured
        # graph (one per seq_len; a single entry for the static program) replays over
        # the same prefix-occupied buffers; every accepted request (S ≤ max_seq_len)
        # fits, so all use the captured path. For the static program the mask +
        # position_ids are request-independent (full causal at max_seq_len, arange
        # positions) and never updated; the symbolic paths re-upload per-S prefixes.
        feed = {
            ids_name: np.zeros((batch, max_seq_len), dtype=np.int64),
            mask_name: _causal_mask_np(max_seq_len, np_dtype),
            pos_name: np.tile(np.arange(max_seq_len, dtype=np.int64), (batch, 1)),
        }
        with gpu_lock():
            program = CompiledProgram.build_from_plan(plan, {**const_feed, **feed})
        output_name = plan.outputs[0]
        del model, wrapper, sources, const_feed
        logger.info(
            "[serving] ready: %d launches, max_seq_len=%d, batch_cap=%d",
            len(program.compiled.launches),
            max_seq_len,
            batch,
        )
        return cls(
            program=program,
            input_names=(ids_name, mask_name, pos_name),
            output_name=output_name,
            np_dtype=np_dtype,
            max_seq_len=max_seq_len,
            batch_cap=batch,
            static=static,
        )

    @classmethod
    def _trace_and_compile(cls, wrapper, sources: dict[str, np.ndarray], max_seq_len: int, dtype, batch: int, static: bool):
        """The full frontend: torch.export trace + CUDA pass pipeline, projected to an
        execution plan, plus the weight feed bound from the live wrapper."""
        import torch

        from emmy.compiler.backend.cuda.backend import CudaBackend
        from emmy.compiler.backend.plan import plan_from_graph
        from emmy.compiler.loader.binder import bind_constants
        from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs
        from emmy.compiler.trace.huggingface import build_causal_mask
        from emmy.compiler.trace.torch import trace_module

        with torch.device("cpu"):
            if batch > 1 and static:
                # Fully-static batched path: ONE (batch, max_seq_len) program with
                # static extents on both axes. Each step pads to this shape.
                logger.info("[serving] tracing STATIC batched (batch=%d, S=%d)...", batch, max_seq_len)
                example = (
                    torch.zeros((batch, max_seq_len), dtype=torch.long),
                    build_causal_mask(max_seq_len, dtype),
                    torch.arange(max_seq_len).unsqueeze(0).expand(batch, max_seq_len).contiguous(),
                )
                graph = trace_module(wrapper, example)  # no dynamic_shapes → fully static
            else:
                # Symbolic-seq path (batch extent baked at `batch`, seq_len symbolic).
                logger.info("[serving] tracing (dynamic seq_len, batch=%d, example S=%d)...", batch, _TRACE_SEQ)
                specs = parse_position_specs(
                    ["seq_len@input_ids:1", "seq_len@attention_mask:2", "seq_len@attention_mask:3", "seq_len@position_ids:1"]
                )
                example = (
                    torch.zeros((batch, _TRACE_SEQ), dtype=torch.long),
                    build_causal_mask(_TRACE_SEQ, dtype),
                    torch.arange(_TRACE_SEQ).unsqueeze(0).expand(batch, _TRACE_SEQ).contiguous(),
                )
                graph = trace_module(wrapper, example, dynamic_shapes=build_torch_dynamic_shapes(specs))

        logger.info("[serving] compiling...")
        compiled = CudaBackend(tune_db="auto").compile(graph)
        return plan_from_graph(compiled), bind_constants(compiled, sources)

    @staticmethod
    def _save_pack(pack_at, plan, pack_key: dict, model_id: str) -> None:
        """Best-effort pack write after a full compile — a failure costs nothing but the log
        line (the next boot recompiles). Skipped when any weight can't be expressed in the
        pack vocabulary (a pack-hit boot could then not rebind it)."""
        from emmy.compiler.backend.pack import save_pack

        if any(w.load_ops is None for w in plan.weights.values()):
            logger.warning("[serving] not writing pack: a weight load-op chain is outside the pack vocabulary")
            return
        try:
            save_pack(pack_at, {"trunk": plan}, key=pack_key, provenance={"model": model_id})
        except Exception:  # noqa: BLE001 — the pack is an optimization, never a boot blocker
            logger.warning("[serving] pack write failed at %s", pack_at, exc_info=True)

    @property
    def hidden_size(self) -> int:
        out = self._program.compiled.buf_by_name[self._output_name]
        return int(out.shape[-1].as_static())

    def _mask(self, s: int):
        """``(1, 1, s, s)`` additive causal mask as a cached cupy device array —
        the device twin of :func:`_causal_mask_np`, built once per S on the GPU so
        the hot path never builds/uploads it from host."""
        import cupy as cp  # noqa: PLC0415

        cached = self._mask_cache.get(s)
        if cached is not None:
            return cached
        mask = cp.triu(cp.full((s, s), float("-inf"), dtype=cp.float32), k=1).astype(self._np_dtype)[None, None, :, :]
        if len(self._mask_cache) >= _MASK_CACHE_MAX:
            self._mask_cache.pop(next(iter(self._mask_cache)))
        self._mask_cache[s] = mask
        return mask

    def forward_hidden_states(self, token_ids):
        """Run one sequence: ``token_ids`` a 1-D int torch CUDA tensor of length
        ``S <= max_seq_len``. Returns an ``(S, hidden)`` torch CUDA tensor in the
        trunk dtype.

        Zero-copy device path: bridge the torch input to cupy (``cp.from_dlpack``,
        no host copy), size the launch grids to S, copy ids / device-built causal
        mask / position_ids into the shared buffers' prefix (device-to-device),
        capture-or-reuse the whole-program graph for this S, replay it, and wrap
        the output buffer's prefix back as a torch tensor (``torch.from_dlpack``)
        — no GPU↔host round-trip. All cupy work runs on torch's current stream so
        the prefix copy, the graph replay, and the output read stay ordered; the
        result is cloned because the shared buffer is reused by the next request."""
        import cupy as cp  # noqa: PLC0415
        import torch

        from emmy.compiler.backend.gpu_lock import gpu_lock

        s = int(token_ids.shape[0])
        if s > self.max_seq_len:
            raise ValueError(f"seq_len {s} exceeds max_seq_len {self.max_seq_len}")
        with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
            feed = {
                self._ids_name: cp.from_dlpack(token_ids.detach().reshape(1, s)),
                self._mask_name: self._mask(s),
                self._pos_name: cp.arange(s, dtype=cp.int64).reshape(1, s),
            }
            self._program.set_sym_values({"seq_len": s})
            self._program.upload_prefix_device(feed)
            self._program.capture_program_graph()
            self._program.replay_program_graph()
            out = self._program.output_prefix_device({"seq_len": s})[self._output_name][0]
            return torch.from_dlpack(out).clone()

    def forward_hidden_states_batched(self, token_ids_list):
        """Run up to ``batch_cap`` sequences (each a 1-D int torch CUDA tensor of
        length ``S <= max_seq_len``) in ONE batched forward per group. Returns a
        list of ``(S_i, hidden)`` torch CUDA tensors in the same order.

        Padding is safe either way: causal masking makes every row's real prefix
        independent of its right-padding (a token only attends to earlier
        positions), and dummy rows below the batch cap are simply not read out.
        The **static** program (``static=True``) pads every step to
        ``(batch_cap, max_seq_len)``; the batched **symbolic-seq** program pads
        only to the step's longest sequence and replays the captured graph for
        that seq_len (one capture per distinct S, LRU-bounded — same cache as the
        per-sequence path). Inputs longer than ``batch_cap`` are processed in
        successive batched groups."""
        import cupy as cp  # noqa: PLC0415
        import torch

        from emmy.compiler.backend.gpu_lock import gpu_lock

        B = self.batch_cap
        results: list = [None] * len(token_ids_list)
        for start in range(0, len(token_ids_list), B):
            group = token_ids_list[start : start + B]
            lens = [int(t.shape[0]) for t in group]
            if any(s > self.max_seq_len for s in lens):
                raise ValueError(f"seq_len {max(lens)} exceeds max_seq_len {self.max_seq_len}")
            # Static: every step runs at the full (B, max_seq_len) shape. Symbolic:
            # pad only to the step's longest sequence and size the grids to it.
            s_step = self.max_seq_len if self.static else max(lens)
            with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
                ids = torch.zeros((B, s_step), dtype=torch.int64, device=group[0].device)
                for i, t in enumerate(group):
                    ids[i, : lens[i]] = t.detach().to(torch.int64)
                feed = {self._ids_name: cp.from_dlpack(ids)}
                if not self.static:
                    # The mask + position_ids prefixes move with s_step; the static
                    # program's were fed once at build and never change.
                    feed[self._mask_name] = self._mask(s_step)
                    feed[self._pos_name] = cp.tile(cp.arange(s_step, dtype=cp.int64), (B, 1))
                    self._program.set_sym_values({"seq_len": s_step})
                self._program.upload_prefix_device(feed)
                self._program.capture_program_graph()  # cached per seq_len (one entry when static)
                self._program.replay_program_graph()
                sym = None if self.static else {"seq_len": s_step}
                out = torch.from_dlpack(self._program.output_prefix_device(sym)[self._output_name])
                for i, si in enumerate(lens):
                    results[start + i] = out[i, :si].clone()
        return results
