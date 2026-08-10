"""CPU tests for the generative plugin's served-length RoPE cache construction."""

from types import SimpleNamespace

import pytest


def test_rope_cache_limit_prefers_vllm_served_length():
    pytest.importorskip("vllm")
    from emmy.serving.vllm_model_gen import _rope_cache_limit

    hf_config = SimpleNamespace(max_position_embeddings=1_048_576)
    assert _rope_cache_limit(SimpleNamespace(max_model_len=4096), hf_config) == 4096
    assert _rope_cache_limit(SimpleNamespace(max_model_len=None), hf_config) == 1_048_576


def test_laguna_multi_rope_is_served_length_bounded_and_dtype_correct():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope

    from emmy.serving.vllm_model_gen import _build_rotaries

    yarn = {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 16,
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 0.5,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    config = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 1_000_000.0},
            "full_attention": yarn,
        },
    )
    runner = SimpleNamespace(layer_meta=lambda _i: (8, 2, 1, 8**-0.5))

    with set_current_vllm_config(VllmConfig()):
        rotaries = _build_rotaries(config, runner, 3, max_position=7, dtype=torch.float16)
        full_yarn = get_rope(8, max_position=64, rope_parameters=yarn, dtype=torch.float16)

    assert rotaries[0] is rotaries[2]
    assert tuple(rotaries[0].cos_sin_cache.shape) == (7, 8)
    assert tuple(rotaries[1].cos_sin_cache.shape) == (7, 4)
    assert rotaries[0].cos_sin_cache.dtype == torch.float16
    assert rotaries[1].cos_sin_cache.dtype == torch.float16
    # Bounding only changes the row count: YaRN's correction ramp is still anchored at the
    # original training context, so every served row matches vLLM's full cache exactly.
    torch.testing.assert_close(rotaries[1].cos_sin_cache, full_yarn.cos_sin_cache[:7], rtol=0, atol=0)
