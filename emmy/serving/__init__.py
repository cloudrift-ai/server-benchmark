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

    # vLLM 0.23 dummy-run seq lens exceed max_model_len for capture sizes above it,
    # overrunning the block table (capture-warmup illegal access); see vllm_patches.
    from emmy.serving.vllm_patches import clamp_dummy_run_seq_lens

    clamp_dummy_run_seq_lens()

    if "EmmyEmbedModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("EmmyEmbedModel", "emmy.serving.vllm_model:EmmyEmbedModel")
    if "EmmyGenModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("EmmyGenModel", "emmy.serving.vllm_model_gen:EmmyGenModel")
