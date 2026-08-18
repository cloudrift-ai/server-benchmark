from types import ModuleType, SimpleNamespace

import pytest

from emmy.compiler.loader.onecat_sm70 import PROJECTION_SPECS, ProjectionProfile
from emmy.serving.onecat_fp8_linear import _build_program, register_onecat_fp8_linear_kernels


def test_build_program_is_strict_and_validates_the_one_launch_abi(monkeypatch) -> None:
    profile = ProjectionProfile(PROJECTION_SPECS[0], 1)
    calls = []
    runtime = object()
    plan = SimpleNamespace(inputs=("x", "weight", "weight_scale"), outputs=("output",), launches=(object(),), buffers=())

    def load(graph, *, pins, symbolic_values):
        calls.append((graph, pins, symbolic_values))
        return runtime, plan

    monkeypatch.setattr("emmy.serving.external.load_external_program", load)
    entry = _build_program(profile)

    assert entry.runtime is runtime
    assert entry.profile == profile
    assert len(calls) == 1 and calls[0][2] is None


def test_build_program_rejects_any_materialized_projection_scratch(monkeypatch) -> None:
    profile = ProjectionProfile(PROJECTION_SPECS[0], 1)
    plan = SimpleNamespace(
        inputs=("x", "weight", "weight_scale"),
        outputs=("output",),
        launches=(object(),),
        buffers=(SimpleNamespace(name="logical_weight", role="scratch"),),
    )
    monkeypatch.setattr("emmy.serving.external.load_external_program", lambda *_args, **_kwargs: (object(), plan))

    with pytest.raises(RuntimeError, match="materialized scratch storage"):
        _build_program(profile)


def test_registration_requires_the_pinned_apply_signature_and_is_idempotent() -> None:
    module = ModuleType("fake_fp8")

    class Fp8LinearMethod:
        def apply(self, layer, x, bias=None):
            return layer, x, bias

    module.Fp8LinearMethod = Fp8LinearMethod
    original = Fp8LinearMethod.apply

    assert register_onecat_fp8_linear_kernels(module)
    replacement = Fp8LinearMethod.apply
    assert replacement is not original
    assert replacement._emmy_onecat_fp8_linear_original is original
    assert register_onecat_fp8_linear_kernels(module)
    assert Fp8LinearMethod.apply is replacement


def test_registration_rejects_signature_drift_without_mutation() -> None:
    module = ModuleType("fake_fp8")

    class Fp8LinearMethod:
        def apply(self, layer, x):
            return layer, x

    module.Fp8LinearMethod = Fp8LinearMethod
    original = Fp8LinearMethod.apply

    assert not register_onecat_fp8_linear_kernels(module)
    assert Fp8LinearMethod.apply is original
