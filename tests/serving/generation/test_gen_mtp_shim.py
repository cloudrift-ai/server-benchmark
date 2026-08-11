"""EmmyGenModel's speculative-decoding shim — the ``.model.embed_tokens`` surface vLLM's MTP
drafter shares off the target model (no GPU, no vllm).

vLLM's Gemma-4 MTP drafter (vllm#41745) reaches into ``target.model.embed_tokens``, adopts it as the
draft head's own embedding, and then applies a ``* sqrt(hidden)`` normalizer itself — so the shared
module must return the RAW, unscaled embedding rows over the tied ``lm_head`` weight the emmy runner
gathers from. These tests pin that contract and the untied-target guard directly against the shim
classes, without constructing the full model (needs vllm/CUDA).
"""

import types

import pytest

torch = pytest.importorskip("torch")


def _shim(model):
    vllm_model_gen = pytest.importorskip("emmy.serving.vllm_model_gen")
    return vllm_model_gen._EmmyTargetInner(vllm_model_gen._SharedRawEmbedding(model))


def _target(embed_dev_is_tied):
    """Stand-in for a constructed EmmyGenModel exposing only what the shim reads: ``lm_head.weight``
    and ``runner._embed_weight_dev``. ``embed_dev_is_tied`` chooses whether the runner gathers from
    the SAME tensor as ``lm_head.weight`` (the tied gemma-4 path) or a distinct output head."""
    param = torch.nn.Parameter(torch.randn(7, 4))
    embed_dev = param.data if embed_dev_is_tied else torch.randn(7, 4)
    return types.SimpleNamespace(
        lm_head=types.SimpleNamespace(weight=param),
        runner=types.SimpleNamespace(_embed_weight_dev=embed_dev),
    )


def test_drafter_contract_shares_raw_tied_embedding():
    model = _target(embed_dev_is_tied=True)
    inner = _shim(model)

    # The exact surface vLLM's base proposer._maybe_share_embeddings reaches for.
    assert getattr(inner, "model", inner) is not None
    assert hasattr(inner, "embed_tokens")
    embed = inner.embed_tokens

    # Shared, not copied: same storage as the target's tied embed/lm_head weight.
    assert embed.weight.data_ptr() == model.lm_head.weight.data_ptr()

    # RAW (unscaled) gather — the drafter applies its own sqrt(hidden) normalizer.
    ids = torch.tensor([0, 3, 6])
    assert torch.equal(embed(ids), torch.nn.functional.embedding(ids, model.lm_head.weight))


def test_untied_target_validate_raises_but_forward_stays_pure():
    embed = _shim(_target(embed_dev_is_tied=False)).embed_tokens
    with pytest.raises(RuntimeError, match="tied-embedding"):
        embed.validate_tied()
    # forward itself carries no guard (torch.compile traces it): it gathers regardless.
    assert embed(torch.tensor([0, 1])).shape == (2, 4)


def test_tied_target_validates():
    _shim(_target(embed_dev_is_tied=True)).embed_tokens.validate_tied()
