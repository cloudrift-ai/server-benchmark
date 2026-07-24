"""Routing test for the ragged-chunk pad-up band (CPU-only, stub programs).

A chunked-prefill step whose width is a large fraction of the prefill bucket (a tail chunk,
or a chunk sharing its step with decode tokens) must ride the STATIC chunk twin padded up
(pad -> run -> slice) rather than the hint-512 symbolic path — the c=4 TTFT excess of the
2026-07-23 lane audit. Below half-bucket the padding waste wins and the symbolic path keeps
the step. The stubs record which program ran and assert the pad/slice arithmetic.
"""

from __future__ import annotations

import torch

from emmy.serving.gen_runner import EmmyGenRunner


class _StubStatic:
    """Stands in for a static twin _Program: returns bucket-width outputs, records calls."""

    def __init__(self, width, n_outs=1):
        self.width, self.n_outs, self.calls = width, n_outs, []

    def run_device(self, arrays):
        t = arrays[0].shape[0]
        self.calls.append(t)
        assert t == self.width, f"static twin fed T={t}, built for {self.width}"
        return [torch.zeros(t, 8) + i for i in range(self.n_outs)]


class _StubSym:
    def __init__(self, n_outs=1):
        self.n_outs, self.calls = n_outs, []

    def run_device_sym(self, arrays):
        self.calls.append(arrays[0].shape[0])
        return [torch.zeros(arrays[0].shape[0], 8) + i for i in range(self.n_outs)]


def _runner(prefill_bucket=64, decode_bucket=8):
    r = EmmyGenRunner.__new__(EmmyGenRunner)
    r._prefill_bucket = prefill_bucket
    r._decode_bucket = decode_bucket
    r._pre_m1 = r._post_m1 = None
    r._pre_decode = r._post_decode = None
    r._pre_prefill = [_StubStatic(prefill_bucket, n_outs=3)]
    r._post_prefill = [_StubStatic(prefill_bucket, n_outs=1)]
    r._pre = [_StubSym(n_outs=3)]
    r._post = [_StubSym(n_outs=1)]
    return r


def test_large_ragged_chunk_pads_to_static_twin():
    r = _runner(prefill_bucket=64)
    t = 40  # > bucket/2 -> pad-up band
    outs = r.forward_layer_pre_device(0, torch.zeros(t, 8))
    assert r._pre_prefill[0].calls == [64] and not r._pre[0].calls
    assert all(o.shape[0] == t for o in outs), "outputs must slice back to the real rows"

    out = r.forward_layer_post_device(0, torch.zeros(t, 8), torch.zeros(t, 8))
    assert r._post_prefill[0].calls == [64]
    assert out.shape[0] == t


def test_small_tail_chunk_stays_symbolic():
    r = _runner(prefill_bucket=64)
    t = 20  # <= bucket/2 -> symbolic keeps it
    r.forward_layer_pre_device(0, torch.zeros(t, 8))
    assert r._pre[0].calls == [t] and not r._pre_prefill[0].calls


def test_exact_chunk_runs_unpadded():
    r = _runner(prefill_bucket=64)
    r.forward_layer_pre_device(0, torch.zeros(64, 8))
    assert r._pre_prefill[0].calls == [64] and not r._pre[0].calls
