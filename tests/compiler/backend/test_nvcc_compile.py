"""nvcc compile path: env-driven flags + cache/context partitioning by opt level.

These are hermetic (no GPU, no actual nvcc invocation) — they exercise the flag
resolution, the cubin cache key, the Context perf-cache key, and the CLI
override precedence, all of which gate whether -O1-tuned and -O3-run results
stay separate in the DB.
"""

from __future__ import annotations

import argparse

from emmy.commands.compile import apply_nvcc_flags
from emmy.compiler.backend.cuda import nvcc
from emmy.compiler.context import Context, split_opt_level


def test_effective_flags_reads_env(monkeypatch) -> None:
    monkeypatch.delenv("EMMY_NVCC_FLAGS", raising=False)
    assert nvcc.effective_flags() == ["--use_fast_math"]
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "-Xcicc -O1")
    assert nvcc.effective_flags() == ["--use_fast_math", "-Xcicc", "-O1"]


def test_cubin_cache_key_partitions_by_flags(monkeypatch) -> None:
    """Same source compiled at different opt levels must key to different
    cubins — otherwise an -O1 sweep would serve its cubin to an -O3 run."""
    monkeypatch.setattr(nvcc, "_toolkit_tag", lambda: "tag")  # avoid the nvcc --version subprocess
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "")
    k_o3 = nvcc._cubin_key("src", "k", "sm_80")
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "-Xcicc -O1")
    k_o1 = nvcc._cubin_key("src", "k", "sm_80")
    assert k_o3 != k_o1


def test_context_key_reads_one_regime_however_it_is_spelled(monkeypatch) -> None:
    """``""`` (compile / run's default) and an explicit ``-Xcicc -O3`` are the SAME physical
    regime and must key the same. Keyed on the raw flag string they did not, so every row a tune
    wrote under an explicit -O3 pin was invisible to a default compile — declared deployable by
    ``H_opt`` and then unreadable by ``structural_key``, the two spellings disagreeing about one
    regime."""

    def key_for(flags: str) -> str:
        monkeypatch.setenv("EMMY_NVCC_FLAGS", flags)
        return Context.from_target((8, 0)).structural_key()

    assert key_for("") == key_for("-Xcicc -O3") == key_for("  -Xcicc   -O3  ")
    # A ranking-only regime still keys apart — it must never answer for a deployable measurement.
    assert key_for("-Xcicc -O1") != key_for("")
    # Any other flag genuinely changes codegen and keeps its own partition.
    assert key_for("--use_fast_math") != key_for("")
    assert key_for("--use_fast_math -Xcicc -O3") == key_for("--use_fast_math")
    assert key_for("--use_fast_math -Xcicc -O1") != key_for("--use_fast_math")


def test_h_opt_and_context_key_agree_on_the_regime(monkeypatch) -> None:
    """The deploy evidence tiers gate on ``H_opt`` and then look rows up by ``structural_key``,
    so the two must never disagree about what regime a compile is in."""
    for flags, opt in (("", 3.0), ("-Xcicc -O3", 3.0), ("-Xcicc -O1", 1.0), ("--use_fast_math", 3.0)):
        monkeypatch.setenv("EMMY_NVCC_FLAGS", flags)
        assert Context.from_target((8, 0)).features()["H_opt"] == opt
    # The residual keeps every other flag, so a same-opt pair still separates on codegen.
    assert split_opt_level("-Xcicc -O1") == (1, "")
    assert split_opt_level("--use_fast_math -Xcicc -O3") == (3, "--use_fast_math")
    assert split_opt_level("") == (3, "")


def test_apply_nvcc_flags_precedence(monkeypatch) -> None:
    # default applies when neither CLI flag nor env is set
    monkeypatch.delenv("EMMY_NVCC_FLAGS", raising=False)
    assert apply_nvcc_flags(argparse.Namespace(nvcc_flags=None), default="-Xcicc -O1") == "-Xcicc -O1"

    # CLI override wins over the default
    monkeypatch.delenv("EMMY_NVCC_FLAGS", raising=False)
    assert apply_nvcc_flags(argparse.Namespace(nvcc_flags="-Xcicc -O3"), default="-Xcicc -O1") == "-Xcicc -O3"

    # a pre-set env var is respected over the command default (but CLI still wins)
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "preset")
    assert apply_nvcc_flags(argparse.Namespace(nvcc_flags=None), default="-Xcicc -O1") == "preset"
    assert apply_nvcc_flags(argparse.Namespace(nvcc_flags="cliwins"), default="-Xcicc -O1") == "cliwins"
