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
    runner = EmmyGenRunner.from_model(model, dtype_str="float16", decode_bucket=16, max_tokens=64)
    assert runner.prefill_capacity == 64

    rng = np.random.default_rng(0)
    T, H = 48, config.hidden_size  # bucket < T <= capacity — the symbolic device regime
    hidden = (rng.standard_normal((T, H)) * 0.3).astype(np.float16)

    q_np, k_np, v_np = runner.forward_layer_pre(0, hidden)  # host rebind path
    q, k, v = runner.forward_layer_pre_device(0, torch.from_numpy(hidden).cuda())
    for host, dev in ((q_np, q), (k_np, k), (v_np, v)):
        assert dev.shape[0] == T
        np.testing.assert_array_equal(dev.cpu().numpy(), host)

    attn = (rng.standard_normal((T, config.num_attention_heads * 16)) * 0.3).astype(np.float16)
    out_np = runner.forward_layer_post(0, attn, hidden)
    out = runner.forward_layer_post_device(0, torch.from_numpy(attn).cuda(), torch.from_numpy(hidden).cuda())
    np.testing.assert_array_equal(out.cpu().numpy(), out_np)
