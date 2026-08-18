import sys
from types import ModuleType


def test_deepseek_opt_in_registers_every_broad_adapter(monkeypatch):
    import emmy.serving
    from emmy import config
    from emmy.serving import (
        onecat,
        onecat_deepseek,
        onecat_experts,
        onecat_fp8_linear,
        onecat_indexer,
        onecat_linear,
        onecat_mhc,
        onecat_output,
        onecat_vocab,
    )

    calls = []

    class ModelRegistry:
        @staticmethod
        def get_supported_archs():
            return {"EmmyEmbedModel", "EmmyGenModel"}

    vllm = ModuleType("vllm")
    vllm.ModelRegistry = ModelRegistry
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setattr(config, "onecat_deepseek_v4", lambda: True)
    monkeypatch.setattr(config, "onecat_rms_norm", lambda: False)
    monkeypatch.setattr("emmy.logging_setup.ensure_plugin_logging", lambda: calls.append("logging"))
    monkeypatch.setattr(onecat, "register_onecat_kernels", lambda: calls.append("rms_norm"))
    monkeypatch.setattr(onecat_deepseek, "register_onecat_deepseek_kernels", lambda: calls.append("deepseek"))
    monkeypatch.setattr(onecat_deepseek, "register_onecat_q_cache_kernel", lambda: calls.append("q_cache"))
    monkeypatch.setattr(onecat_experts, "register_onecat_expert_kernels", lambda: calls.append("experts"))
    monkeypatch.setattr(onecat_fp8_linear, "register_onecat_fp8_linear_kernels", lambda: calls.append("fp8_linear"))
    monkeypatch.setattr(onecat_linear, "register_onecat_linear_kernels", lambda: calls.append("linear"))
    monkeypatch.setattr(onecat_mhc, "register_onecat_mhc_kernels", lambda: calls.append("mhc"))
    monkeypatch.setattr(onecat_output, "register_onecat_output_kernels", lambda: calls.append("output"))
    monkeypatch.setattr(onecat_vocab, "register_onecat_vocab_kernels", lambda: calls.append("vocab"))
    monkeypatch.setattr(onecat_indexer, "register_onecat_indexer_kernels", lambda: calls.append("indexer"))

    emmy.serving.register()

    assert calls == [
        "logging",
        "rms_norm",
        "deepseek",
        "q_cache",
        "experts",
        "fp8_linear",
        "linear",
        "mhc",
        "output",
        "vocab",
        "indexer",
    ]
