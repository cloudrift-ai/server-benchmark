"""Fixtures for the serving-generation lane.

One compile session spans the lane: the binding-neutral plan cache is shared, so two runners of
different shapes still compile a layer program they have in common exactly once, and every build
happens under the lane's golden (``helpers.evidence_scope`` — strict, so a fork the golden does not
decide names its kernel instead of falling back to a cold search).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tests.serving.generation import helpers


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
