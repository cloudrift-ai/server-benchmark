"""Runtime workarounds for vLLM defects, applied from the plugin ``register`` hook.

Each patch documents the upstream defect and the pinned vLLM version it targets; drop the
patch when the pin moves to a release that ships the fix.
"""

import logging

logger = logging.getLogger(__name__)


def clamp_dummy_run_seq_lens() -> None:
    """vLLM 0.23: ``GPUModelRunner._dummy_run`` fills every dummy request's seq_len with the
    step's total TOKEN count. When a capture size exceeds ``--max-model-len`` (emmy's
    rider-top rung — chunk quantum + decode bucket — is the one rung that can), each dummy
    request claims a longer KV prefix than the block table can address (its rows hold
    ``cdiv(max_model_len, block_size)`` page ids). The unified-attention kernel loads block
    tables without a bounds mask, so the last request's programs read page ids from past the
    end of the tensor, and the garbage ids send the K/V loads far out of bounds — the
    capture-warmup illegal memory access. FLEX_ATTENTION trips over the same arithmetic
    loudly: its sliding-window block mask builds ``cdiv(seq_len, block_size)`` columns
    against a block table sliced to the row width. Head width is irrelevant — any model
    faults once a capture size passes max_model_len (verified by standalone kernel repro on
    RTX 4080: clean at 4096, faults at 4097, both head_dim 256 and 512).

    Clamping the seq-lens buffers to ``max_model_len`` before attention metadata is built
    restores the invariant every real batch already holds; legal batches are untouched.

    Only the DEFAULT model runner (``gpu_model_runner.py``) has this defect — and out-of-tree
    architectures like ``EmmyGenModel`` always get it (vLLM 0.23's newer runner is opt-in for
    stock Llama/Mistral/Qwen3 only, and its dummy batches use legal per-request seq lens).
    Verified end to end on the default runner: a 4128-token capture with max-model-len 4096
    warms up and captures cleanly with the clamp, greedy output correct.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_emmy_seq_lens_clamped", False):
        return
    orig = GPUModelRunner._build_attention_metadata

    def build_with_clamped_seq_lens(self, *args, **kwargs):
        lens = self.optimistic_seq_lens_cpu
        if int(lens.max()) > self.max_model_len:  # only dummy batches ever exceed
            lens.clamp_(max=self.max_model_len)
            # Same H2D refresh protocol as _dummy_run's own seq-lens fill.
            self.seq_lens.copy_(lens, non_blocking=True)
            if not getattr(self, "_emmy_seq_lens_clamp_logged", False):
                self._emmy_seq_lens_clamp_logged = True
                logger.info("emmy vllm patch: dummy-run seq lens clamped to max_model_len")
        return orig(self, *args, **kwargs)

    GPUModelRunner._build_attention_metadata = build_with_clamped_seq_lens
    GPUModelRunner._emmy_seq_lens_clamped = True
