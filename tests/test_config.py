"""Tests for ``emmy.config`` — the prior-rename legacy env aliases.

The 2026-07 offline/online prior rename respelled ``EMMY_PRIOR_FILE`` /
``EMMY_ANALYTIC_FILE`` / ``EMMY_ANALYTIC_TILT``. The old spellings live in shell
profiles and remote-run scripts, so they must keep resolving (with a
DeprecationWarning) and the new spelling must win when both are set.
"""

from __future__ import annotations

import warnings

import pytest

from emmy import config


def test_online_path_legacy_env_alias(monkeypatch, tmp_path):
    monkeypatch.delenv(config.ONLINE_FILE, raising=False)
    monkeypatch.setenv("EMMY_PRIOR_FILE", str(tmp_path / "legacy.json"))
    with pytest.warns(DeprecationWarning, match="EMMY_PRIOR_FILE"):
        assert config.online_path() == tmp_path / "legacy.json"


def test_online_path_new_name_wins_over_legacy(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ONLINE_FILE, str(tmp_path / "new.json"))
    monkeypatch.setenv("EMMY_PRIOR_FILE", str(tmp_path / "legacy.json"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a warning here means the legacy var was consulted
        assert config.online_path() == tmp_path / "new.json"


def test_offline_path_legacy_env_alias(monkeypatch, tmp_path):
    monkeypatch.delenv(config.OFFLINE_FILE, raising=False)
    monkeypatch.setenv("EMMY_ANALYTIC_FILE", str(tmp_path / "weights.json"))
    with pytest.warns(DeprecationWarning, match="EMMY_ANALYTIC_FILE"):
        assert config.offline_path() == tmp_path / "weights.json"


def test_offline_tilt_legacy_env_alias(monkeypatch):
    monkeypatch.delenv(config.OFFLINE_TILT, raising=False)
    monkeypatch.setenv("EMMY_ANALYTIC_TILT", "0.7")
    with pytest.warns(DeprecationWarning, match="EMMY_ANALYTIC_TILT"):
        assert config.offline_tilt() == 0.7
