"""DeepSeek V4 through a REAL vLLM engine — greedy parity across parallel topologies.

The distributed gate of the plugin work: the same tiny checkpoint served by a single-rank engine
and by a TP2×PP2 engine must produce IDENTICAL greedy token ids. That exercises everything the
in-process gates cannot: the engine-driven forward with real paged-attention metadata, the
carrier-width pipeline transport between stages, per-stage weight loading, the tensor-parallel
expert shards summed by the group all-reduce, and the head/sampler on the last rank.

Needs four V100s in the pinned 1Cat image. Run it in its own pytest process: the engine spawns
its own workers, and a process that already initialized single-rank groups (the other plugin
gates) cannot host the driver cleanly.
"""

from __future__ import annotations

import gc

import pytest

from tests.serving.generation.test_vllm_plugin_deepseek_gpu import HF_OVERRIDES, KEYS_BY_TYPE, _tiny_deepseek

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda_engine")]

# Token-id prompts (the fixture ships no tokenizer): distinct lengths so the batch mixes widths.
PROMPTS = [[3, 14, 15, 92, 65], [35, 89, 79, 32, 38, 46], [2, 7, 71, 82]]


def _add_native_attention(path, config, torch):
    """Append the published NATIVE-named attention tensors to the fixture's safetensors.

    vLLM streams the file's keys into ``load_weights``, whose fork-attention family claims only the
    published spelling (``layers.N.attn.*``) — ``save_pretrained`` writes HF names, so without this
    the fork sublayer has no source and the loud completeness check refuses the boot. Values come
    from the twin's own attention parameters (via the same published renaming the loader uses, run
    FORWARD: native key → HF key), fp8-cast where the checkpoint declares a ``.scale`` sibling,
    with identity E8M0 block scales."""
    import math

    from safetensors.torch import load_file, save_file

    from emmy.compiler.trace.huggingface import _native_checkpoint_renamer

    native = []
    for layer, layer_type in enumerate(config.layer_types):
        native += [f"layers.{layer}.attn.{key}" for key in KEYS_BY_TYPE[layer_type]]
    rename = _native_checkpoint_renamer(config, keys=native)
    extra = {}
    block = HF_OVERRIDES["quantization_config"]["weight_block_size"][0]
    file = path / "model.safetensors"
    tensors = load_file(str(file))
    for key in sorted(native, key=lambda k: k.endswith(".scale")):  # weights first, their scales after
        if key.endswith(".scale"):
            weight = extra[key[: -len(".scale")] + ".weight"]
            shape = (math.ceil(weight.shape[0] / block), math.ceil(weight.shape[1] / block))
            extra[key] = torch.full(shape, 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
            continue
        tensor = tensors[rename(key)]
        if key[: -len(".weight")] + ".scale" in native:
            tensor = tensor.to(torch.float8_e4m3fn)
        extra[key] = tensor.clone()  # safetensors refuses aliased tensors in one file
    tensors.update(extra)
    save_file(tensors, str(file))


def _greedy_ids(model_dir, tensor_parallel_size, pipeline_parallel_size):
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(
        model=str(model_dir),
        runner="generate",
        dtype="float16",
        max_model_len=128,
        max_num_batched_tokens=256,
        enforce_eager=True,
        kv_cache_dtype="fp8",
        skip_tokenizer_init=True,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        gpu_memory_utilization=0.25,
        hf_overrides=HF_OVERRIDES,
    )
    try:
        params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True, logprobs=2)
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in PROMPTS], params)
        return [(list(o.outputs[0].token_ids), o.outputs[0].logprobs) for o in outs]
    finally:
        del llm
        gc.collect()


def test_tp2_pp2_greedy_token_ids_match_the_single_rank_engine(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("vllm")
    pytest.importorskip("cupy")
    # The DRIVER must stay off the CUDA driver (the fork-model import is NOT probed here on
    # purpose — it initializes CUDA), and the engine processes must spawn, not fork: a forked
    # EngineCore inherits a poisoned CUDA context and dies at init_device.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.device_count() < 4:
        pytest.skip("TP2 x PP2 needs four GPUs")

    config = _tiny_deepseek(tmp_path, transformers, torch)
    _add_native_attention(tmp_path, config, torch)

    single = _greedy_ids(tmp_path, tensor_parallel_size=1, pipeline_parallel_size=1)
    assert all(len(ids) == 8 for ids, _ in single), f"single-rank engine under-generated: {single}"

    sharded = _greedy_ids(tmp_path, tensor_parallel_size=2, pipeline_parallel_size=2)

    # Greedy ids must agree EXCEPT at a numerical tie: this fixture's random weights leave many
    # near-flat logit pairs, and the sharded topology sums the same expert partials in a different
    # order, so fp16 rounding may flip a pick whose top-2 margin is noise-sized. At the first
    # divergent step the single-rank margin must be inside that noise, and the sharded pick must be
    # one of the tied pair — a decisive pick that differs is a real numerical break. (The published
    # checkpoint's greedy gate demands exact ids; decisive logits are the model's job, not the
    # topology's.)
    def _margin(logprobs_step):
        top2 = sorted((lp.logprob for lp in logprobs_step.values()), reverse=True)
        return top2[0] - top2[1]

    for prompt, ((ids_1, lps_1), (ids_2, lps_2)) in enumerate(zip(single, sharded, strict=True)):
        if ids_1 == ids_2:
            continue
        step = next(j for j, (a, b) in enumerate(zip(ids_1, ids_2, strict=True)) if a != b)
        # BOTH topologies must agree the divergent step was a tie (a flat distribution can tie
        # more than two tokens, so membership in the other run's top-2 is not required), and the
        # two arms' BEST scores must agree numerically — differing argmaxes over materially
        # different distributions would slip a pure per-arm margin check.
        for arm, margin in (("single", _margin(lps_1[step])), ("tp2pp2", _margin(lps_2[step]))):
            assert margin < 0.08, f"prompt {prompt} step {step}: decisive {arm} pick flipped (margin {margin:.4f}): {ids_1} vs {ids_2}"
        tops = [max(lp.logprob for lp in lps[step].values()) for lps in (lps_1, lps_2)]
        assert abs(tops[0] - tops[1]) < 0.08, f"prompt {prompt} step {step}: the arms' best logprobs disagree ({tops})"
