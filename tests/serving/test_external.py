from emmy.compiler.backend.plan import ExecutionPlan
from emmy.serving.external import _override_symbolic_hints


def _plan():
    return ExecutionPlan(
        backend="cuda",
        inputs=[],
        outputs=[],
        buffers=[],
        constants={},
        runtime_constants={},
        launches=[],
        kernels={},
        symbolic_hints={"num_tokens": 512},
        symbolic_caps={"num_tokens": 4096},
    )


def test_external_program_symbolic_capacity_override_is_immutable():
    original = _plan()
    updated = _override_symbolic_hints(original, {"num_tokens": 4096})
    assert original.symbolic_hints == {"num_tokens": 512}
    assert updated.symbolic_hints == {"num_tokens": 4096}


def test_external_program_symbolic_capacity_override_rejects_unknown_or_out_of_range():
    import pytest

    plan = _plan()
    with pytest.raises(KeyError, match="unknown dimensions"):
        _override_symbolic_hints(plan, {"batch": 8})
    with pytest.raises(ValueError, match=r"outside \[1,4096\]"):
        _override_symbolic_hints(plan, {"num_tokens": 4097})
