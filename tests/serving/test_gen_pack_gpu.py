"""Gen-runner pack round-trip (``EMMY_PACK_DIR``): the first ``from_model`` boot writes the
pack, the second boots from it (no trace / compile) and produces identical layer outputs.
Needs CUDA + cupy (skips itself otherwise); tiny random Qwen3, same pattern as
``test_gen_runner_gpu``."""

import logging

import numpy as np
import pytest

pytestmark = [pytest.mark.xdist_group("cuda")]


def test_gen_pack_second_boot_hits_and_matches(tmp_path, monkeypatch, caplog):
    pytest.importorskip("cupy")
    import torch
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from emmy.serving.gen_runner import EmmyGenRunner

    monkeypatch.setenv("EMMY_PACK_DIR", str(tmp_path))
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        use_sliding_window=False,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config).eval()

    caplog.set_level(logging.INFO, logger="emmy.serving.gen_runner")
    first = EmmyGenRunner.from_model(model, dtype_str="float32", decode_bucket=16)
    manifests = list(tmp_path.glob("*/manifest.json"))
    assert len(manifests) == 1, "the full-compile boot must write exactly one pack"
    assert not any("pack hit" in r.message for r in caplog.records)

    caplog.clear()
    second = EmmyGenRunner.from_model(model, dtype_str="float32", decode_bucket=16)
    assert any("pack hit" in r.message for r in caplog.records), "second boot must load the pack"
    assert second.has_device_decode == first.has_device_decode

    t = 7
    hidden = first.embed(list(range(1, t + 1)))
    position_ids = torch.arange(t).unsqueeze(0)
    q1, k1, v1 = first.forward_layer_pre(0, hidden, position_ids)
    q2, k2, v2 = second.forward_layer_pre(0, hidden, position_ids)
    np.testing.assert_array_equal(q2, q1)
    np.testing.assert_array_equal(k2, k1)
    np.testing.assert_array_equal(v2, v1)
    attn = np.ascontiguousarray(np.random.default_rng(0).standard_normal((t, 4 * 16)).astype(np.float32) * 0.1)
    np.testing.assert_array_equal(second.forward_layer_post(0, attn, hidden), first.forward_layer_post(0, attn, hidden))
