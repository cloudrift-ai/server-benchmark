"""In-process vLLM engine test of the emmy **generative** plugin (Phase 3).

DELIBERATELY ``perf``-marked (deselected by default; the in-process engine demands 40%
of the card FREE at startup — incompatible with the parallel suite's live CUDA contexts,
so it stays out of ``make test`` unlike the serving correctness pins): needs CUDA + cupy
+ vllm. Saves a TINY random
Llama (vocab matches a cached Llama tokenizer, 2 layers — no network), serves it through
``EmmyGenModel`` in an in-process vLLM engine (real paged ``Attention`` + KV cache +
``lm_head`` + ``get_rope``), and greedily generates — checking it runs end-to-end and that
the generated tokens agree with HF eager greedy on the same weights. This is the Phase-3
integration proof: vLLM accepts the model, allocates the KV-cache spec from the per-layer
``Attention`` prefixes, and the emmy↔attention forward interleave produces correct logits.
"""

import pytest

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]

TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # cached; vocab 32000


def _save_tiny_llama(path):
    import torch
    import transformers

    config = transformers.LlamaConfig(
        vocab_size=32000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval().to(torch.float16)
    model.save_pretrained(path)
    return config


def _generate(vllm, model_dir, prompt_ids, max_new, **extra):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    llm = vllm.LLM(
        model=str(model_dir),
        tokenizer=TOKENIZER,
        runner="generate",
        hf_overrides={"architectures": ["EmmyGenModel"]},
        enforce_eager=True,
        dtype="float16",
        max_model_len=128,
        max_num_batched_tokens=512,  # <= DYNAMIC_DIM_MAX (the flattened-width bound)
        gpu_memory_utilization=0.4,
        **extra,
    )
    out = llm.generate(TokensPrompt(prompt_token_ids=prompt_ids), SamplingParams(temperature=0.0, max_tokens=max_new))
    return list(out[0].outputs[0].token_ids)


def test_vllm_gen_plugin_matches_hf_eager(tmp_path, monkeypatch):
    pytest.importorskip("cupy")
    vllm = pytest.importorskip("vllm")
    import torch
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import emmy.serving

    # The test process has CUDA initialized (conftest seeds it); vLLM's forked
    # EngineCore would die on re-init. Run the engine in-process instead.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    model_dir = tmp_path / "tiny_llama"
    _save_tiny_llama(str(model_dir))
    emmy.serving.register()  # ModelRegistry.register_model("EmmyGenModel", ...)

    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    prompt_ids = tok("The quick brown fox", add_special_tokens=True)["input_ids"]
    max_new = 8

    gen = _generate(vllm, model_dir, prompt_ids, max_new)

    # end-to-end gate: ran, produced max_new valid in-vocab tokens
    assert len(gen) == max_new
    assert all(0 <= t < 32000 for t in gen)

    # correctness gate: greedy agrees with HF eager on the same weights
    ref = transformers.LlamaForCausalLM.from_pretrained(str(model_dir), dtype=torch.float16).to("cuda").eval()
    with torch.no_grad():
        ref_out = ref.generate(torch.tensor([prompt_ids], device="cuda"), do_sample=False, max_new_tokens=max_new, use_cache=True)
    ref_gen = ref_out[0, len(prompt_ids) :].tolist()
    # Compare the WHOLE greedy sequence: token 0 comes from prefill, tokens 1.. are KV-cache
    # DECODE steps (num_tokens=1), so this validates the decode path, not just prefill.
    assert gen == ref_gen, f"greedy mismatch (emmy vs HF eager): {gen} vs {ref_gen}"


def test_vllm_gen_plugin_fp8_kv_cache(tmp_path, monkeypatch):
    """``--kv-cache-dtype fp8_e4m3`` (the quantized-KV deployment path — ~2x KV tokens per byte).

    Two things are under test that the fp16 sibling cannot reach: vLLM allocates the KV spec at
    the fp8 dtype and its backend reads it back, and — the emmy-side glue —
    ``EmmyGenModel._attn_aliased`` replicates ``Attention.forward``'s static query quantization
    instead of bailing, so the A4 alias fast path survives an fp8-KV boot. The backend is PINNED
    to ``TRITON_ATTN`` because only some backends ask for a quantized query: its
    ``supports_quant_query_input`` is true on any CUDA device, whereas FlashInfer's (what vLLM
    auto-selects for fp8 KV on sm_120) additionally requires TRTLLM attention, i.e. SM100 — so an
    auto-selected boot would leave ``query_quant`` None and cover nothing. The test asserts the
    quantizer was really built. The output gate is LOOSE on purpose: fp8 K/V at the default scale
    1.0 perturbs attention, and this random-init 2-layer model has no meaningful greedy trajectory
    to match — the pin is that the alias path is exercised (counted through a spy) and generation
    stays in-vocab."""
    pytest.importorskip("cupy")
    vllm = pytest.importorskip("vllm")
    import torch
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import emmy.serving
    from emmy.serving.vllm_model_gen import EmmyGenModel

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("EMMY_GEN_ALIAS_ATTN", "1")  # the path this test exists for (default on)
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")  # quantizes the query (see above)

    model_dir = tmp_path / "tiny_llama"
    _save_tiny_llama(str(model_dir))
    emmy.serving.register()

    # Spy on the alias tail: `None` means it bailed to the plain `Attention.forward` module
    # call. Before the query-quant replication every fp8-KV step bailed.
    counts = {"aliased": 0, "bailed": 0, "quantized_query": 0}
    original = EmmyGenModel._attn_aliased

    def _spy(self, layer, q, k, v):
        if self.attn[layer].query_quant is not None:
            counts["quantized_query"] += 1
        out = original(self, layer, q, k, v)
        counts["bailed" if out is None else "aliased"] += 1
        return out

    monkeypatch.setattr(EmmyGenModel, "_attn_aliased", _spy)

    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    prompt_ids = tok("The quick brown fox", add_special_tokens=True)["input_ids"]
    max_new = 8

    gen = _generate(vllm, model_dir, prompt_ids, max_new, kv_cache_dtype="fp8_e4m3")

    assert len(gen) == max_new
    assert all(0 <= t < 32000 for t in gen)
    assert counts["quantized_query"] > 0, f"backend asked for no query quantization — test covers nothing: {counts}"
    assert counts["aliased"] > 0, f"the A4 alias never fired under fp8 KV: {counts}"
