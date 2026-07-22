"""``EmmyGenModel.compute_logits`` honors the generation config's ``suppress_tokens`` (no GPU).

gemma-4 checkpoints list the mm delimiter tokens ('<image|>'/'<audio|>') there; a degenerate text
prompt can genuinely rank one top-1 (verified vs HF eager fp16), and an emitted mm token decodes to
empty text. HF generate and stock vLLM -inf those ids; the plugin must match.
"""

import pytest

torch = pytest.importorskip("torch")


def _model_with(suppress_ids):
    vllm_model_gen = pytest.importorskip("emmy.serving.vllm_model_gen")
    m = vllm_model_gen.EmmyGenModel.__new__(vllm_model_gen.EmmyGenModel)
    # compute_logits touches only these three attributes; the identity processor makes the
    # hidden-states argument stand in for the logits tensor.
    m.logits_processor = lambda lm_head, hidden: hidden
    m.lm_head = None
    m._suppress_token_ids = suppress_ids
    return m


def test_suppressed_ids_are_neg_inf_others_untouched():
    out = _model_with([3, 5]).compute_logits(torch.zeros((2, 8)))
    assert torch.isneginf(out[:, [3, 5]]).all()
    keep = [i for i in range(8) if i not in (3, 5)]
    assert (out[:, keep] == 0).all()


def test_no_suppress_list_is_noop():
    out = _model_with(None).compute_logits(torch.zeros((2, 8)))
    assert (out == 0).all()
