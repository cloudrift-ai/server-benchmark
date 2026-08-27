"""The dummy-run seq-lens clamp (``serving/vllm_patches.py``) against a stub runner.

The real defect needs a CUDA capture warmup to fire; here we verify the patch mechanics —
over-limit seq lens are clamped on both buffers before delegation, legal ones pass through
untouched, and re-applying the patch never double-wraps.
"""

import sys
import types

import torch

from emmy.serving.vllm_patches import clamp_dummy_run_seq_lens


class _RunnerBase:
    def __init__(self, lens: list[int], max_model_len: int):
        self.optimistic_seq_lens_cpu = torch.tensor(lens, dtype=torch.int32)
        self.seq_lens = torch.zeros(len(lens), dtype=torch.int32)
        self.max_model_len = max_model_len
        self.built = 0

    def _build_attention_metadata(self, *args, **kwargs):
        self.built += 1
        return "metadata"


def _install_stub(monkeypatch) -> type:
    class Runner(_RunnerBase):  # fresh class per test: the patch mutates it
        pass

    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    mod = types.ModuleType("vllm.v1.worker.gpu_model_runner")
    mod.GPUModelRunner = Runner
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", mod)
    return Runner


def test_over_model_len_dummy_seq_lens_are_clamped(monkeypatch):
    _install_stub(monkeypatch)
    clamp_dummy_run_seq_lens()
    runner = sys.modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner([4128, 4128, 0], max_model_len=4096)
    assert runner._build_attention_metadata(num_tokens=1) == "metadata"
    assert runner.built == 1
    assert runner.optimistic_seq_lens_cpu.tolist() == [4096, 4096, 0]
    assert runner.seq_lens.tolist() == [4096, 4096, 0]


def test_legal_seq_lens_pass_through_untouched(monkeypatch):
    _install_stub(monkeypatch)
    clamp_dummy_run_seq_lens()
    runner = sys.modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner([4096, 17, 0], max_model_len=4096)
    runner._build_attention_metadata()
    assert runner.optimistic_seq_lens_cpu.tolist() == [4096, 17, 0]
    assert runner.seq_lens.tolist() == [0, 0, 0]  # no refresh issued


def test_patch_is_idempotent(monkeypatch):
    runner_cls = _install_stub(monkeypatch)
    clamp_dummy_run_seq_lens()
    once = runner_cls._build_attention_metadata
    clamp_dummy_run_seq_lens()
    assert runner_cls._build_attention_metadata is once
