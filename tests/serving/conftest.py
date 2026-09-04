"""Shared pytest fixtures for the serving test modules.

The lane compiles under its own golden rather than deploying cold — see :mod:`tests.serving.helpers`
for why and for the shape table these fixtures build from.
"""

from dataclasses import dataclass

import pytest
import torch

from tests.serving import helpers


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


@dataclass(frozen=True)
class Built:
    """One built runner beside the eager module it came from — the reference half of a parity
    assertion, and the source of the rotary tables and config the stitch needs."""

    runner: object
    model: object

    @property
    def config(self):
        return self.model.config


@pytest.fixture(scope="session")
def _gen_session():
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from emmy.compiler.backend.plan_cache import PlanTemplateCache

    return {"cache": PlanTemplateCache(), "built": {}}


@pytest.fixture
def build_runner(_gen_session):
    """Build a FRESH :class:`Built` of one :data:`helpers.RUNNERS` shape.

    For a test that mutates its runner, or that asks about the state a build leaves behind. It
    still rides the session's plan cache and the golden, so a fresh build costs a rebind of
    compiled structure, not a search.
    """

    def build(runner_id: str) -> Built:
        model = helpers.RUNNERS[runner_id][0]()
        return Built(helpers.build(runner_id, model=model, plan_cache=_gen_session["cache"]), model)

    return build


@pytest.fixture
def built(_gen_session, build_runner):
    """A :class:`Built` per shape, SHARED across the tests that only read it — built on first ask."""

    def get(runner_id: str) -> Built:
        if runner_id not in _gen_session["built"]:
            _gen_session["built"][runner_id] = build_runner(runner_id)
        return _gen_session["built"][runner_id]

    return get
