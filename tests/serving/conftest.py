"""Shared pytest fixtures for the serving test modules."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    """vLLM 0.23.0's ``set_default_torch_dtype`` context manager has no
    try/finally, so when an in-process engine load raises (the ``_gpu`` plugin
    tests run with ``VLLM_ENABLE_V1_MULTIPROCESSING=0``), the worker's torch
    default dtype is left at bfloat16 — and every later CPU fp32 test on that
    worker fails with mixed-dtype errors. Snapshot and restore around each
    test so one test's engine failure can't poison the rest of the worker."""
    dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(dtype)
