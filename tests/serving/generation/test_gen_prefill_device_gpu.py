"""Device-resident SYMBOLIC prefill path (``_Program.run_device_sym``). Needs CUDA + cupy
(skips itself off-GPU).

Builds a tiny random-weight Llama layer through the gen runner with a prefill ``capacity``
(``max_tokens``), then checks the device path at a width ABOVE the decode bucket — the
prefill/chunked-prefill regime that used to take the per-layer host numpy hops — against the
host ``rebind`` path on the same programs: identical kernels, so the outputs must match
bit-for-bit (both fp16, same launch sequence, only the I/O transport differs).
"""

import numpy as np
import pytest

# NOT perf-marked (correctness pin, must run under ``make test``; see tests/ARCHITECTURE.md).
pytestmark = [pytest.mark.xdist_group("cuda")]


def test_run_device_sym_matches_host_path():
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from transformers import LlamaConfig, LlamaForCausalLM

    from emmy.serving.gen_runner import EmmyGenRunner

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval().to(torch.float16)
    runner = EmmyGenRunner.from_model(model, dtype_str="float16", decode_bucket=16, max_tokens=64, prefill_bucket=32)
    assert runner.prefill_capacity == 64
    assert runner.prefill_bucket == 32
    assert runner.rider_width == 16  # chunk twin + decode twin -> split coverage above the chunk

    # A2 chaining pins: every family's post OUTPUT array aliases its pre twins' hidden-INPUT
    # backing, so the next layer's pre upload self-copy-skips. Checked BEFORE any host-path
    # call — the host ``rebind`` re-takes arena views at the call's shape, which unwinds the
    # rewire for that program (correct, just copies again; serving never runs the host path).
    def _ptr(prog, name):
        return prog.program.arrays[name].data.ptr

    assert _ptr(runner._post[0], runner._post[0].output_names[0]) == _ptr(runner._pre[0], runner._pre[0].input_names[0])
    assert _ptr(runner._post_prefill[0], runner._post_prefill[0].output_names[0]) == _ptr(
        runner._pre_prefill[0], runner._pre_prefill[0].input_names[0]
    )
    assert _ptr(runner._post_decode[0], runner._post_decode[0].output_names[0]) == _ptr(
        runner._pre_decode[0], runner._pre_decode[0].input_names[0]
    )

    # A4 routing pins: ``post_attn_backing`` must return the ``attn_out`` input backing of the
    # SAME program ``forward_layer_post_device`` routes each width to; rider widths (no single
    # contiguous backing) and over-capacity widths return None.
    def _attn_ptr(plist):
        return plist[0].program.arrays["attn_out"].data.ptr

    m1_expect = runner._post_m1 if runner._post_m1 is not None and runner._post_m1[0] is not None else runner._post_decode
    assert runner.post_attn_backing(0, 1).data_ptr() == _attn_ptr(m1_expect)
    assert runner.post_attn_backing(0, 8).data_ptr() == _attn_ptr(runner._post_decode)
    assert runner.post_attn_backing(0, 32).data_ptr() == _attn_ptr(runner._post_prefill)
    assert runner.post_attn_backing(0, 40) is None  # rider split
    assert runner.post_attn_backing(0, 56).data_ptr() == _attn_ptr(runner._post)
    assert runner.post_attn_backing(0, 65) is None  # over capacity

    rng = np.random.default_rng(0)
    H = config.hidden_size
    close = lambda dev, host: np.testing.assert_allclose(dev, host, rtol=2e-2, atol=2e-3)  # noqa: E731
    # T=24: decode_bucket < T <= prefill_bucket — routed to the symbolic device path (the
    # chunk twin is exact-width only); same program as the host path but the host reference
    # tolerance is kept loose. T=40: prefill_bucket < T <= prefill_bucket + rider_width —
    # the chunk+decode twin row SPLIT (different kernels from the host's symbolic program,
    # so fp16 accumulation order may differ by rounding). T=56: past the rider window
    # (48 = pb + rider is still split) — the SYMBOLIC device regime, same program as the
    # host path (bit-exact). TWO PHASES, all device calls first: the host ``rebind`` both
    # unwinds the chaining rewire AND re-sizes the shared capacity buffers to each call's
    # shape (a later device call at a larger T would then refuse on capacity) — serving never
    # mixes the two paths on one runner, and neither may this test.
    # A4 aliased-post value check (decode tier, eager): attention "writes" into the backing
    # view, the post upload self-copy-skips, and the result must match the host path on the
    # same values (compared in phase 2, after the host calls are allowed to unchain).
    rng_a4 = np.random.default_rng(7)
    attn8 = (rng_a4.standard_normal((8, config.num_attention_heads * 16)) * 0.3).astype(np.float16)
    res8 = (rng_a4.standard_normal((8, H)) * 0.3).astype(np.float16)
    backing8 = runner.post_attn_backing(0, 8)
    backing8.copy_(torch.from_numpy(attn8).cuda())
    out8_dev = runner.forward_layer_post_device(0, backing8, torch.from_numpy(res8).cuda()).cpu().numpy()

    cases = []
    for T in (24, 40, 56):
        hidden = (rng.standard_normal((T, H)) * 0.3).astype(np.float16)
        attn = (rng.standard_normal((T, config.num_attention_heads * 16)) * 0.3).astype(np.float16)
        # Snapshot each stage to host IMMEDIATELY: rider outputs (T=40) are views of the shared
        # joint destinations (A3), which the NEXT rider call overwrites — the same consume-
        # before-next-call discipline the vLLM plugin follows (attention reads q/k/v within the
        # layer), made explicit here.
        q, k, v = runner.forward_layer_pre_device(0, torch.from_numpy(hidden).cuda())
        q_np_dev, k_np_dev, v_np_dev = q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy()
        q_ptr = q.data_ptr()
        out = runner.forward_layer_post_device(0, torch.from_numpy(attn).cuda(), torch.from_numpy(hidden).cuda())
        out_np_dev = out.cpu().numpy()
        # Chained layer-to-layer handoff: feed the post output straight back as the next pre
        # input (what the vLLM plugin does layer to layer).
        q2, _, _ = runner.forward_layer_pre_device(0, out)
        q2_np_dev = q2.cpu().numpy()
        if T == 40:
            # A3 pin: rider outputs are slices of persistent shared joint destinations — the
            # handoff call above reused the same storage (no per-step allocation, no torch.cat).
            assert q2.data_ptr() == q_ptr
        cases.append((T, hidden, attn, q_np_dev, k_np_dev, v_np_dev, out_np_dev, q2_np_dev))

    for T, hidden, attn, q, k, v, out, q2 in cases:
        check = np.testing.assert_array_equal if T == 56 else close
        q_np, k_np, v_np = runner.forward_layer_pre(0, hidden)  # host rebind path
        for host, dev in ((q_np, q), (k_np, k), (v_np, v)):
            assert dev.shape[0] == T
            check(dev.astype(np.float32), host.astype(np.float32))
        out_np = runner.forward_layer_post(0, attn, hidden)
        check(out.astype(np.float32), out_np.astype(np.float32))
        q2_np, _, _ = runner.forward_layer_pre(0, out)
        check(q2.astype(np.float32), q2_np.astype(np.float32))

    out8_np = runner.forward_layer_post(0, attn8, res8)
    close(out8_dev.astype(np.float32), out8_np.astype(np.float32))
