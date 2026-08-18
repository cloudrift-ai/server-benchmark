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

    assert len(manifest) == 208
    assert names == tuple(sorted(names))
    assert len(names) == len(set(names))
    assert Counter(profile.family for profile in manifest) == {
        "final_rms_norm": 1,
        "qkv_rms_norm": 1,
        "inverse_rope": 1,
        "qnorm_rope": 1,
        "retained_fp8": 54,
        "route_learned": 9,
        "route_hash": 9,
        "routed_experts": 1,
        "routed_experts_wide": 20,
        "linear": 45,
        "mhc": 45,
        "output": 3,
        "vocab_embedding": 8,
        "local_top1": 8,
        "rank_top1": 1,
        "indexer_q": 1,
    }
    assert "linear.n64.fp16.static.m1024" in names
    for kind in ("learned", "hash"):
        routes = [profile for profile in manifest if profile.family == f"route_{kind}"]
        assert Counter((profile.rows, profile.symbolic) for profile in routes) == {
            (1, False): 1,
            (2, False): 1,
            (4, False): 1,
            (8, False): 1,
            (16, False): 1,
            (128, False): 1,
            (1024, False): 1,
            (4096, False): 1,
            (4096, True): 1,
        }
        route = next(profile for profile in routes if profile.symbolic)
        assert route.name == f"route.{kind}.symbolic.m4096"
        assert route.symbolic_values == (("num_tokens", 4096),)
    expert = next(profile for profile in manifest if profile.family == "routed_experts")
    assert expert.name == "experts.direct.symbolic.m4096"
    assert expert.symbolic_values == (("num_tokens", 4096),)
    assert expert.launch_count == 4
    wide = [profile for profile in manifest if profile.family == "routed_experts_wide"]
    assert Counter((profile.rows, profile.name.split(".")[2]) for profile in wide) == {
        (1024, "activation"): 1,
        (1024, "bucket"): 1,
        (1024, "combine"): 1,
        (1024, "pack"): 1,
        (1024, "unbucket0"): 1,
        (1024, "unbucket1"): 1,
        (1024, "w13"): 1,
        (1024, "w2"): 1,
        (4096, "activation"): 1,
        (4096, "bucket"): 1,
        (4096, "combine"): 1,
        (4096, "pack"): 1,
        (4096, "unbucket0"): 1,
        (4096, "unbucket1"): 1,
        (4096, "unbucket2"): 1,
        (4096, "unbucket3"): 1,
        (4096, "unbucket4"): 1,
        (4096, "unbucket5"): 1,
        (4096, "w13"): 1,
        (4096, "w2"): 1,
    }

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
    for profile in (
        "fused_wqa_wkv",
        "attention_wq_b_wo_b",
        "grouped_wo_a",
        "indexer_wq_b",
        "shared_gate_up",
        "shared_down",
    ):
        assert f"retained_fp8.{profile}.static.m1" in names
        assert f"retained_fp8.{profile}.static.m4096" in names
        assert f"retained_fp8.{profile}.symbolic.m4096" in names

    retained = [profile for profile in manifest if profile.family == "retained_fp8"]
    assert all(profile.expected_inputs == ("x", "weight", "weight_scale") for profile in retained)
    assert all(profile.launch_count == 1 for profile in retained)

    rms_norm = next(profile for profile in manifest if profile.family == "final_rms_norm")
    assert rms_norm.pins == (("REDUCE", "coop"), ("WORK", "t256"))
    assert rms_norm.symbolic_values == (("num_tokens", 4096),)
    qnorm_rope = next(profile for profile in manifest if profile.family == "qnorm_rope")
    assert qnorm_rope.pins == (("REDUCE", "coop"), ("WORK", "t128"))
    assert qnorm_rope.launch_count == 1


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
