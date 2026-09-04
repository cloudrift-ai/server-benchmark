"""One measurement regime: what the prior's dataset accepts, and what a tune measures in.

A measurement is only evidence about the conditions it was taken under. Emmy tunes in the regime
it deploys into, so these two facts have to hold together — the sweep's default flags must BE the
deploy's, and a row from any other regime must not enter the model's training set.
"""

from __future__ import annotations

import argparse

import pytest

from emmy.commands.compile import apply_nvcc_flags
from emmy.commands.tune import TUNE_NVCC_DEFAULT
from emmy.compiler.context import split_opt_level
from emmy.compiler.pipeline.search.prior.base import Prior


class _P(Prior):
    """The concrete surface ``add_rows`` needs; the model half is irrelevant here."""

    @property
    def fitted(self) -> bool:
        return False

    def fit(self) -> None:
        pass

    def mean_score(self, knobs: dict) -> float:
        return 0.0

    def score_rows(self, group):
        return None


def _prior() -> _P:
    return _P()


def test_tune_measures_in_the_regime_it_deploys_into(monkeypatch) -> None:
    """``tune``'s default nvcc flags are ``compile`` / ``run``'s, and they resolve to the
    deployable opt level. Ranking in a regime nothing deploys in makes the search's winner and
    production's winner two different questions."""
    monkeypatch.delenv("EMMY_NVCC_FLAGS", raising=False)
    tune_flags = apply_nvcc_flags(argparse.Namespace(nvcc_flags=None), default=TUNE_NVCC_DEFAULT)

    monkeypatch.delenv("EMMY_NVCC_FLAGS", raising=False)
    deploy_flags = apply_nvcc_flags(argparse.Namespace(nvcc_flags=None), default="")

    assert tune_flags == deploy_flags
    assert split_opt_level(tune_flags)[0] == 3


def test_only_deployable_rows_train_the_prior() -> None:
    """``add_rows`` admits ``H_opt=3`` and nothing else. A measurement from another opt level is a
    proposal distribution, not a label source: its error is biased along tile area — the axis being
    tuned — so a model fit on the mixture learns a systematic mis-ranking instead of averaging it
    out."""
    prior = _prior()
    prior.add_rows(
        [
            ({"H_opt": 3.0, "S_ext_free_prod": 1024.0, "BM": 64}, 10.0),
            ({"H_opt": 1.0, "S_ext_free_prod": 1024.0, "BM": 64}, 90.0),
            ({"H_opt": 2.0, "S_ext_free_prod": 1024.0, "BM": 32}, 50.0),
        ]
    )

    assert [us for _knobs, us in prior._dataset] == [10.0]
    # The reservoir's Algorithm-R counter must not count what it refused, or a long run of
    # rejected rows would skew every later replacement probability.
    assert prior._seen == 1


def test_an_unstamped_row_is_not_admitted() -> None:
    """A row with no ``H_opt`` reads as non-deployable, the same default the evidence tier
    applies (``Prior._o3_evidence``). The tune path always stamps it (``base_knobs`` comes from
    ``ctx.features()``), so this only governs hand-built rows — and the two must not disagree
    about what an unstamped row means."""
    prior = _prior()
    prior.add_rows([({"S_ext_free_prod": 1024.0, "BM": 64}, 7.0)])
    assert prior._dataset == []


@pytest.mark.parametrize("route", ({"PLACE": "cut"}, {"PLACE@inner.1/map": "cut"}, {"PLACE@inner.1/map": "cut", "WORK": "t32"}))
def test_placement_route_rows_train_but_are_not_measured_deploy_evidence(route) -> None:
    prior = _prior()
    prior.add_rows(
        [
            ({"H_opt": 3.0, "S_shape": 128.0, **route}, 1.0),
            ({"H_opt": 3.0, "S_shape": 128.0, "WORK": "t64"}, 7.0),
        ]
    )

    assert len(prior._dataset) == 2
    assert prior._o3_evidence() == {frozenset({("S_shape", 128.0)}): [({"WORK": "t64"}, 7.0)]}
    assert prior.evidence_pick([{"H_opt": 3.0, "S_shape": 128.0, "WORK": "t64"}]) == (0, 7.0)
