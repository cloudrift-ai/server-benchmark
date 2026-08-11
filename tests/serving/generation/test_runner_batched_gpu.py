"""Batched serving paths (``EMMY_SERVING_STATIC=1`` / ``EMMY_SERVING_BATCHED=1``).

Needs CUDA + cupy + the Qwen3-Embedding config (skips itself off-GPU; the config comes
from the HF cache like the compiler e2e tests'). Builds a 1-layer ``(batch, S)`` trunk
(fully static, and batched symbolic-seq), wraps a ``EmmyForwardRunner`` around it, and checks that
``forward_hidden_states_batched`` runs several different-length sequences in ONE
padded batched forward and matches eager per row — the causal-independence claim
(a row's real prefix is unaffected by right-padding, dummy rows below the batch cap
are ignored). The whole-model accuracy gate lives in ``test_vllm_plugin_gpu.py``.
"""

import numpy as np
import pytest

# NOT perf-marked (correctness pin, must run under ``make test``; see tests/ARCHITECTURE.md).
pytestmark = [pytest.mark.xdist_group("cuda")]

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def test_runner_batched_matches_eager():
    pytest.importorskip("cupy")
    import torch
    from transformers import AutoConfig, AutoModel

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.trace.huggingface import build_causal_mask, build_full_model_wrapper
    from emmy.compiler.trace.torch import trace_module
    from emmy.serving.runner import EmmyForwardRunner

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg.num_hidden_layers = 1
    model = AutoModel.from_config(cfg).float().eval()

    B, S, dtype = 4, 32, torch.float32
    wrapper = build_full_model_wrapper(model, S, dtype, dynamic=True)
    example = (
        torch.zeros((B, S), dtype=torch.long),
        build_causal_mask(S, dtype),
        torch.arange(S).unsqueeze(0).expand(B, S).contiguous(),
    )
    compiled = CudaBackend().compile(trace_module(wrapper, example))  # fully static

    sources: dict[str, np.ndarray] = {}
    for path, t in wrapper.named_parameters(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    for path, t in wrapper.named_buffers(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    const_feed = bind_constants(compiled, sources)
    ids_name, mask_name, pos_name = compiled.inputs
    feed = {
        ids_name: np.zeros((B, S), dtype=np.int64),
        mask_name: build_causal_mask(S, dtype).numpy(),
        pos_name: np.tile(np.arange(S, dtype=np.int64), (B, 1)),
    }
    with gpu_lock():
        program = CompiledProgram.build(compiled, {**const_feed, **feed})
        runner = EmmyForwardRunner(
            program=program,
            input_names=(ids_name, mask_name, pos_name),
            output_name=compiled.outputs[0],
            np_dtype=np.dtype("float32"),
            max_seq_len=S,
            batch_cap=B,
        )
        # Three different-length sequences (full, mid, short) + an implicit dummy
        # 4th row (only 3 supplied < batch_cap=4). Each must match eager on its own
        # length despite right-padding to S.
        lens = [S, 20, 7]
        seqs = [torch.from_numpy(((np.arange(n, dtype=np.int64) * 97) % 100) + 1).cuda() for n in lens]
        outs = runner.forward_hidden_states_batched(seqs)
        for n, out, ids_t in zip(lens, outs, seqs, strict=True):
            with torch.no_grad():
                ref = wrapper(ids_t.reshape(1, n).cpu(), build_causal_mask(n, dtype), torch.arange(n).reshape(1, n)).numpy()[0]
            got = out.cpu().numpy()
            assert got.shape == (n, 1024)
            np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


def test_runner_batched_symbolic_matches_eager():
    """The batched SYMBOLIC-seq path (``EMMY_SERVING_BATCHED=1``): batch extent baked
    at the cap, seq_len symbolic — each step pads only to the step's longest sequence
    and replays the captured graph for that seq_len. Mixed lengths per row + a second
    step at a different step-length (a fresh capture) must both match eager."""
    pytest.importorskip("cupy")
    import torch
    from transformers import AutoConfig, AutoModel

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs
    from emmy.compiler.trace.huggingface import build_causal_mask, build_full_model_wrapper
    from emmy.compiler.trace.torch import trace_module
    from emmy.serving.runner import EmmyForwardRunner

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg.num_hidden_layers = 1
    model = AutoModel.from_config(cfg).float().eval()

    B, hint, cap, dtype = 4, 32, 64, torch.float32
    wrapper = build_full_model_wrapper(model, hint, dtype, dynamic=True)
    specs = parse_position_specs(["seq_len@input_ids:1", "seq_len@attention_mask:2", "seq_len@attention_mask:3", "seq_len@position_ids:1"])
    example = (
        torch.zeros((B, hint), dtype=torch.long),
        build_causal_mask(hint, dtype),
        torch.arange(hint).unsqueeze(0).expand(B, hint).contiguous(),
    )
    compiled = CudaBackend().compile(trace_module(wrapper, example, dynamic_shapes=build_torch_dynamic_shapes(specs)))

    sources: dict[str, np.ndarray] = {}
    for path, t in wrapper.named_parameters(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    for path, t in wrapper.named_buffers(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    const_feed = bind_constants(compiled, sources)
    ids_name, mask_name, pos_name = compiled.inputs
    feed = {
        ids_name: np.zeros((B, cap), dtype=np.int64),
        mask_name: np.triu(np.full((cap, cap), -np.inf, dtype=np.float32), k=1)[None, None],
        pos_name: np.tile(np.arange(cap, dtype=np.int64), (B, 1)),
    }
    with gpu_lock():
        program = CompiledProgram.build(compiled, {**const_feed, **feed})
        runner = EmmyForwardRunner(
            program=program,
            input_names=(ids_name, mask_name, pos_name),
            output_name=compiled.outputs[0],
            np_dtype=np.dtype("float32"),
            max_seq_len=cap,
            batch_cap=B,
            static=False,
        )
        # Two steps at different step-lengths: mixed lengths padded to max(lens)=32
        # (the trace hint), then to 48 (off-hint — a fresh per-S capture).
        for lens in ([32, 20, 7], [48, 11]):
            seqs = [torch.from_numpy(((np.arange(n, dtype=np.int64) * 97) % 100) + 1 + 3 * i).cuda() for i, n in enumerate(lens)]
            outs = runner.forward_hidden_states_batched(seqs)
            for n, out, ids_t in zip(lens, outs, seqs, strict=True):
                with torch.no_grad():
                    ref = wrapper(ids_t.reshape(1, n).cpu(), build_causal_mask(n, dtype), torch.arange(n).reshape(1, n)).numpy()[0]
                got = out.cpu().numpy()
                assert got.shape == (n, 1024)
                np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)
