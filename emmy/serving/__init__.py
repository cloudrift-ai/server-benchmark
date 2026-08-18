"""vLLM out-of-tree model plugin: emmy-compiled kernels behind vLLM's serving shell.

``register`` is the ``vllm.general_plugins`` entry point (see pyproject.toml);
vLLM calls it in every process at engine start. The model class itself lives in
``vllm_model.py`` and is registered by lazy string path so importing this
package never pulls in vllm (or CUDA) by itself.
"""


def register() -> None:
    from vllm import ModelRegistry

    from emmy.logging_setup import ensure_plugin_logging

    # Under the bare vLLM entrypoint nothing handles emmy's INFO records (the CLI's
    # root-logger setup never ran) — the serving runners' boot/pack lines would vanish
    # from docker logs, and the gemma4 verify gate greps them there.
    ensure_plugin_logging()

    from emmy import config

    deepseek_v4 = config.onecat_deepseek_v4()
    if config.onecat_rms_norm() or deepseek_v4:
        from emmy.serving.onecat import register_onecat_kernels

        register_onecat_kernels()
    if deepseek_v4:
        from emmy.serving.onecat_deepseek import register_onecat_deepseek_kernels, register_onecat_q_cache_kernel
        from emmy.serving.onecat_experts import register_onecat_expert_kernels
        from emmy.serving.onecat_fp8_linear import register_onecat_fp8_linear_kernels
        from emmy.serving.onecat_indexer import register_onecat_indexer_kernels
        from emmy.serving.onecat_linear import register_onecat_linear_kernels
        from emmy.serving.onecat_mhc import register_onecat_mhc_kernels
        from emmy.serving.onecat_native_warmup import install_onecat_native_prefill_warmup
        from emmy.serving.onecat_output import register_onecat_output_kernels
        from emmy.serving.onecat_vocab import register_onecat_vocab_kernels

        register_onecat_deepseek_kernels()
        register_onecat_q_cache_kernel()
        register_onecat_expert_kernels()
        register_onecat_fp8_linear_kernels()
        register_onecat_linear_kernels()
        register_onecat_mhc_kernels()
        register_onecat_output_kernels()
        register_onecat_vocab_kernels()
        register_onecat_indexer_kernels()
        install_onecat_native_prefill_warmup()

    if "EmmyEmbedModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("EmmyEmbedModel", "emmy.serving.vllm_model:EmmyEmbedModel")
    if "EmmyGenModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("EmmyGenModel", "emmy.serving.vllm_model_gen:EmmyGenModel")
