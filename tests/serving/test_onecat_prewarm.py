from collections import Counter
from types import SimpleNamespace

import pytest

from emmy.serving.onecat_prewarm import (
    ExternalProgramProfile,
    deepseek_external_program_manifest,
    prewarm_deepseek_external_programs,
)


def _profile(name: str, *, inputs=("x",), outputs=1) -> ExternalProgramProfile:
    return ExternalProgramProfile(
        name=name,
        family="test",
        rows=1,
        symbolic=False,
        expected_inputs=inputs,
        output_count=outputs,
        graph_factory=lambda: name,
    )


def test_deepseek_external_manifest_is_complete_deterministic_and_names_the_failed_p1024_n64_profile():
    manifest = deepseek_external_program_manifest()
    names = tuple(profile.name for profile in manifest)

    assert len(manifest) == 114
    assert names == tuple(sorted(names))
    assert len(names) == len(set(names))
    assert Counter(profile.family for profile in manifest) == {
        "final_rms_norm": 1,
        "qkv_rms_norm": 1,
        "inverse_rope": 1,
        "linear": 45,
        "mhc": 45,
        "output": 3,
        "vocab_embedding": 8,
        "local_top1": 8,
        "rank_top1": 1,
        "indexer_q": 1,
    }
    assert "linear.n64.fp16.static.m1024" in names

    linear = [profile for profile in manifest if profile.family == "linear"]
    assert Counter((profile.rows, profile.symbolic) for profile in linear) == {
        (1, False): 5,
        (2, False): 5,
        (4, False): 5,
        (8, False): 5,
        (16, False): 5,
        (128, False): 5,
        (1024, False): 5,
        (4096, False): 5,
        (4096, True): 5,
    }

    for rank in range(8):
        assert f"vocab.embedding.rank{rank}.symbolic.m4096" in names
        assert f"vocab.local_top1.rank{rank}.symbolic.m4096" in names
    assert "vocab.rank_top1.symbolic.m4096" in names
    assert "indexer.q_rope_weights.symbolic.m4096" in names

    rms_norm = next(profile for profile in manifest if profile.family == "final_rms_norm")
    assert rms_norm.pins == (("REDUCE", "coop"), ("WORK", "t256"))
    assert rms_norm.symbolic_values == (("num_tokens", 4096),)


def test_prewarm_realizes_every_profile_in_order_and_checks_its_abi():
    profiles = (_profile("a", inputs=("x", "weight")), _profile("b"))
    calls = []

    def realize(graph, *, pins, symbolic_values):
        calls.append((graph, pins, symbolic_values))
        inputs = ("x", "weight") if graph == "a" else ("x",)
        return SimpleNamespace(inputs=inputs, outputs=("output",), launches=())

    assert prewarm_deepseek_external_programs(profiles, realize=realize) == ("a", "b")
    assert calls == [("a", {}, {}), ("b", {}, {})]


def test_prewarm_failure_identifies_the_exact_release_blocking_profile():
    profiles = (_profile("a"), _profile("linear.n64.fp16.static.m1024"))

    def realize(graph, **_kwargs):
        if graph != "a":
            raise RuntimeError("compiler failure")
        return SimpleNamespace(inputs=("x",), outputs=("output",), launches=())

    with pytest.raises(RuntimeError, match=r"linear\.n64\.fp16\.static\.m1024 \(2/2\)"):
        prewarm_deepseek_external_programs(profiles, realize=realize)


def test_prewarm_rejects_nondeterministic_or_duplicate_manifest_order():
    with pytest.raises(ValueError, match="deterministic order"):
        prewarm_deepseek_external_programs((_profile("b"), _profile("a")), realize=lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match="unique names"):
        prewarm_deepseek_external_programs((_profile("a"), _profile("a")), realize=lambda *_args, **_kwargs: None)
