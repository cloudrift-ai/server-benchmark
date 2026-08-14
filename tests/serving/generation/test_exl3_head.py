"""EXL3 coded output-head eligibility; GPU execution is covered by the serving lane."""

import numpy as np
import pytest

from emmy.serving.exl3_head import Exl3HeadSpec
from tests.compiler.helpers import requires_cuda
from tests.support.checkpoints import exl3_linear_tensors


def _source(*, bits=6, hidden=128, vocab=256, marker="mul1"):
    source = {
        "lm_head.trellis": np.zeros((hidden // 16, vocab // 16, 16 * bits), dtype=np.int16),
        "lm_head.suh": np.ones(hidden, dtype=np.float16),
        "lm_head.svh": np.ones(vocab, dtype=np.float16),
    }
    if marker:
        source[f"lm_head.{marker}"] = np.array(0, dtype=np.int32)
    return source


def test_coded_head_is_architecture_independent_and_keeps_logical_vocab():
    spec = Exl3HeadSpec.from_source(
        _source(vocab=512),
        hidden_size=128,
        vocab_size=500,
        tensor_parallel_size=1,
        tied=False,
    )
    assert spec == Exl3HeadSpec(bits=6, codebook=2, hidden_size=128, stored_vocab_size=512, vocab_size=500)


@pytest.mark.parametrize(
    ("kwargs", "source"),
    [
        ({"tensor_parallel_size": 2}, _source()),
        ({"tied": True}, _source()),
        ({}, _source(marker=None)),
    ],
)
def test_unsupported_coded_head_keeps_decoded_fallback(kwargs, source):
    base = dict(hidden_size=128, vocab_size=256, tensor_parallel_size=1, tied=False)
    assert Exl3HeadSpec.from_source(source, **(base | kwargs)) is None


def test_coded_head_rejects_ambiguous_marker_and_bad_storage():
    source = _source()
    source["lm_head.mcg"] = np.array(0, dtype=np.int32)
    with pytest.raises(ValueError, match="both mcg and mul1"):
        Exl3HeadSpec.from_source(
            source,
            hidden_size=128,
            vocab_size=256,
            tensor_parallel_size=1,
            tied=False,
        )


@requires_cuda
def test_coded_head_matches_decoded_weight_and_keeps_checkpoint_pointers():
    import torch

    from emmy.serving.exl3_head import Exl3CodedHead

    tensors, decoded = exl3_linear_tensors("lm_head", 256, 128, K=2, cb=2)
    source = {name: tensor.numpy() for name, tensor in tensors.items()}
    spec = Exl3HeadSpec.from_source(
        source,
        hidden_size=128,
        vocab_size=251,
        tensor_parallel_size=1,
        tied=False,
    )
    assert spec is not None
    head = Exl3CodedHead(spec, source)
    assert len(head.program.compiled.launches) > 1
    assert all("exl3" not in launch.kernel_name.lower() for launch in head.program.compiled.launches)
    pointers = {name: head.program.arrays[name].data.ptr for name in ("lm_head.trellis", "lm_head.suh", "lm_head.svh")}
    x = (torch.randn(2, 128, device="cuda") * 0.1).half()
    got = head(x)
    reference = x.float() @ torch.from_numpy(decoded[:, :251]).float().cuda()
    torch.testing.assert_close(got, reference, rtol=2e-2, atol=2e-2)
    assert pointers == {name: head.program.arrays[name].data.ptr for name in pointers}

    source = _source()
    source["lm_head.svh"] = source["lm_head.svh"].astype(np.float32)
    with pytest.raises(ValueError, match="invalid svh"):
        Exl3HeadSpec.from_source(
            source,
            hidden_size=128,
            vocab_size=256,
            tensor_parallel_size=1,
            tied=False,
        )
