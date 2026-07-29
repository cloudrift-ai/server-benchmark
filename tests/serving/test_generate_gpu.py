"""GPU spike for the Phase-0 generation oracle.

Needs CUDA + cupy (skips itself otherwise). Builds a TINY random-weight
Llama CausalLM (no network), compiles the whole-model fp16 dynamic path through
``_CompiledLM`` (full logits, last row sliced on the host), and checks the compiled
next-token logits against an eager fp16 reference across a few growing prefixes — the
plan's "compile-and-run spike" that de-risks whole-model lowering (int64 embedding-gather,
lm_head matmul) before the generate loop is trusted. ``test_slice_last_logits_lowers_cold``
pins the cold build + numerics of the in-graph last-token slice ``_CompiledLM`` uses.
"""

import numpy as np
import pytest

# NOT perf-marked (correctness pin, must run under ``make test``; see tests/ARCHITECTURE.md).
pytestmark = [pytest.mark.xdist_group("cuda")]


def _tiny_llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval().to(torch.float16)
    return model


def test_generate_oracle_matches_eager_fp16():
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from emmy.commands.generate import _CompiledLM

    # IMPORTANT: build_full_model_wrapper mutates the model in place (replaces rotary_emb
    # with _SlicedRotary, patches the causal-mask builder). Compare against an INDEPENDENT,
    # untouched model (same seed → identical weights, but HF's real rotary) so a wrapper/RoPE
    # bug can't hide by corrupting both sides.
    lm = _CompiledLM.from_model(_tiny_llama(), seq_len=8)
    ref = _tiny_llama().to("cuda").eval()

    prefixes = [[1, 2, 3], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10]]
    for prefix in prefixes:
        dep = lm.logits(prefix)  # [vocab] fp32
        ids = torch.tensor([prefix], dtype=torch.long, device="cuda")
        with torch.no_grad():
            eager = ref(ids).logits[0, -1, :].float().cpu().numpy()
        # fp16 path vs fp16 eager: same dtype, so the only gap is kernel numerics.
        assert dep.shape == eager.shape
        np.testing.assert_allclose(dep, eager, rtol=2e-2, atol=2e-2)
        assert int(np.argmax(dep)) == int(np.argmax(eager))  # greedy token agrees


def test_slice_last_logits_lowers_cold():
    """The in-graph last-token slice must build AND compute correctly COLD (``tune_db=None``,
    empty prior) — it is ``_CompiledLM``'s production graph. Two historical failure modes are
    pinned here: the M=1 demoted lm_head once stayed an unlowered ``LoopOp`` (fixed by the
    tile-IR rebuild: the unbindable ContractionView demotes to PLANAR), and the
    ``hidden[:, -1:, :]`` slice's negative row index once reached codegen as a raw ``-1``
    (``mul[-H + a1]``, an OOB read → silent zeros; fixed in ``140_slice`` by normalizing to
    ``seq_len - 1``). Runs the sliced graph at two lengths and checks logits against eager."""
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs
    from emmy.compiler.trace.huggingface import build_causal_mask, build_full_model_wrapper
    from emmy.compiler.trace.torch import trace_module

    model = _tiny_llama()
    s = 8
    wrapper = build_full_model_wrapper(model, s, torch.float16, dynamic=True, slice_last_logits=True)
    specs = parse_position_specs(["seq_len@input_ids:1", "seq_len@attention_mask:2", "seq_len@attention_mask:3", "seq_len@position_ids:1"])
    example = (torch.zeros((1, s), dtype=torch.long), build_causal_mask(s, torch.float16), torch.arange(s).unsqueeze(0))
    graph = trace_module(wrapper, example, dynamic_shapes=build_torch_dynamic_shapes(specs))
    compiled = CudaBackend(tune_db=None).compile(graph)

    sources = {}
    for path, t in wrapper.named_parameters(remove_duplicate=False):
        sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np.float16, copy=False)
    for path, t in wrapper.named_buffers(remove_duplicate=False):
        sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np.float16, copy=False)
    const_feed = bind_constants(compiled, sources)
    ids_name, mask_name, pos_name = compiled.inputs

    def feed_for(prefix):
        t = len(prefix)
        return {
            ids_name: np.asarray(prefix, dtype=np.int64).reshape(1, t),
            mask_name: build_causal_mask(t, torch.float16).numpy(),
            pos_name: np.arange(t, dtype=np.int64).reshape(1, t),
        }

    with gpu_lock():
        program = CompiledProgram.build(compiled, {**const_feed, **feed_for([0] * s)})
    for prefix in ([1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8]):
        t = len(prefix)
        with gpu_lock():
            program.rebind(feed_for(prefix))
            program.run_once()
            got = program.outputs({"seq_len": t})[compiled.outputs[0]][0, -1, :]  # [vocab]
        example_t = (torch.tensor([prefix], dtype=torch.long), build_causal_mask(t, torch.float16), torch.arange(t).unsqueeze(0))
        with torch.no_grad():
            want = wrapper(*example_t)[0, -1, :].float().numpy()
        np.testing.assert_allclose(got.astype(np.float32), want, rtol=2e-2, atol=2e-2)  # zeros today (OOB read)


def test_generate_loop_runs_end_to_end():
    """The full host loop over the compiled program produces a fixed-length output."""
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from emmy.commands.generate import _CompiledLM, generate
    from emmy.serving.sampling import Sampler

    lm = _CompiledLM.from_model(_tiny_llama(), seq_len=8)
    out = generate(lm.logits, [1, 2, 3], max_new_tokens=5, eos_ids=None, sampler=Sampler())
    assert len(out) == 5
    assert all(0 <= t < 64 for t in out)
