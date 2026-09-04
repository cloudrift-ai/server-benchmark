"""GPU dynamic-compile test for the Phase-1 attention-split wrappers (the compiler enabler).

Needs CUDA + cupy (skips itself otherwise). Traces the ``pre`` and ``post`` wrappers over the
flattened ``[num_tokens, H]`` layout with ``num_tokens`` **symbolic**, compiles each, and
runs at two different token counts — matching the eager wrapper output. This proves the
carved subgraphs actually lower + run dynamically (the core of Phase 2's gen_runner), and
that the ``post`` wrapper's two inputs share one ``Dim`` (without the second spec the
``residual`` axis would stay trace-sized). fp32 (carve correctness is dtype-independent;
the fp16 path is covered by the Phase-0 oracle).
"""

import numpy as np
import pytest

from tests.serving import helpers

# NOT perf-marked (correctness pin, must run under ``make test``; see tests/ARCHITECTURE.md).
pytestmark = [pytest.mark.xdist_group("cuda")]


def _compile_wrapper(wrapper, example_args, argnames):
    """Trace ``wrapper`` with axis 0 of every arg bound to a shared ``num_tokens`` Dim,
    compile on the CUDA backend, bind fp32 constants. Returns (program, input_names, output_names)."""

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs
    from emmy.compiler.trace.torch import trace_module

    specs = [f"num_tokens@{name}:0" for name in argnames]  # shared NAME ties all axes
    graph = trace_module(wrapper, tuple(example_args), dynamic_shapes=build_torch_dynamic_shapes(parse_position_specs(specs)))
    with helpers.evidence_scope():
        compiled = CudaBackend(tune_db="auto").compile(graph)

    sources = {}
    for path, t in wrapper.named_parameters(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    for path, t in wrapper.named_buffers(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    const_feed = bind_constants(compiled, sources)

    feed = {n: a.detach().cpu().numpy().astype(np.float32) for n, a in zip(compiled.inputs, example_args, strict=True)}
    with gpu_lock():
        program = CompiledProgram.build(compiled, {**const_feed, **feed})
    return program, list(compiled.inputs), list(compiled.outputs)


def _run(program, input_names, output_names, arrays):
    from emmy.compiler.backend.gpu_lock import gpu_lock

    t = arrays[0].shape[0]
    feed = {n: a.detach().cpu().numpy().astype(np.float32) for n, a in zip(input_names, arrays, strict=True)}
    with gpu_lock():
        program.rebind(feed)  # resolves num_tokens from the input shapes
        program.run_once()
        out = program.outputs({"num_tokens": t})
    return [out[n] for n in output_names]


def test_pre_wrapper_compiles_and_runs_dynamic():
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    pre, args, argnames, config = helpers.wrapper_case("qwen3.pre")
    h = config.hidden_size

    program, in_names, out_names = _compile_wrapper(pre, args, argnames)
    for t in (4, 7):  # different token counts → replay at new num_tokens, not recapture
        hidden = torch.randn(t, h)
        got = _run(program, in_names, out_names, [hidden])
        with torch.no_grad():
            want = pre(hidden)  # (q, k, v)
        assert len(got) == 3
        for g, w in zip(got, want, strict=True):
            np.testing.assert_allclose(g, w.numpy(), rtol=1e-3, atol=1e-3)


def test_post_wrapper_compiles_with_shared_dim():
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    post, args, argnames, config = helpers.wrapper_case("qwen3.post")
    h = config.hidden_size
    attn_width = config.num_attention_heads * (config.head_dim or h // config.num_attention_heads)

    program, in_names, out_names = _compile_wrapper(post, args, argnames)
    for t in (4, 7):
        attn_out, residual = torch.randn(t, attn_width), torch.randn(t, h)
        (got,) = _run(program, in_names, out_names, [attn_out, residual])
        with torch.no_grad():
            want = post(attn_out, residual)
        np.testing.assert_allclose(got, want.numpy(), rtol=1e-3, atol=1e-3)


def test_gemma_post_wrapper_compiles_and_runs_dynamic():
    """Gemma-3/4's 4-norm ``post`` (extra pre/post-feedforward norms) lowers + runs through the
    CUDA backend — the compiler side of the Gemma serving carve. ``pre`` is the shared q/k-norm
    path already covered by ``test_pre_wrapper_compiles_and_runs_dynamic``."""
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    post, args, argnames, config = helpers.wrapper_case("gemma3.post")
    h = config.hidden_size
    attn_width = config.num_attention_heads * (config.head_dim or h // config.num_attention_heads)

    program, in_names, out_names = _compile_wrapper(post, args, argnames)
    for t in (4, 7):
        attn_out, residual = torch.randn(t, attn_width), torch.randn(t, h)
        (got,) = _run(program, in_names, out_names, [attn_out, residual])
        with torch.no_grad():
            want = post(attn_out, residual)
        np.testing.assert_allclose(got, want.numpy(), rtol=1e-3, atol=1e-3)
