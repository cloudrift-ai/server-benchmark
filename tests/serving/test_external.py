import json
from contextlib import nullcontext
from types import SimpleNamespace

from emmy.compiler.backend.plan import ExecutionPlan
from emmy.serving import external
from emmy.serving.external import _external_pack_key, _override_symbolic_hints, _resolve_external_plan
from emmy.serving.onecat_linear import _linear_graph, _LinearProfile


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


def test_external_pack_key_carries_the_exact_graph_and_build_contract():
    graph = SimpleNamespace(to_dict=lambda: {"nodes": {"x": {"shape": [1, 4096]}}})
    key = _external_pack_key(graph, {"WORK": "t256", "REDUCE": "coop"}, "auto", {"num_tokens": 4096})
    assert key == {
        "model": "external-program",
        "graph": {"nodes": {"x": {"shape": [1, 4096]}}},
        "pins": {"REDUCE": "coop", "WORK": "t256"},
        "tune_db": "auto",
        "symbolic_values": {"num_tokens": 4096},
    }


def test_external_pack_key_is_json_stable_for_a_real_symbolic_graph():
    graph = _linear_graph(_LinearProfile(64, False, 4096, symbolic=True))
    key = _external_pack_key(graph, None, "auto", {"num_tokens": 4096})

    encoded = json.dumps(key, sort_keys=True)
    assert json.loads(encoded) == key
    assert '"num_tokens"' in encoded


def test_external_plan_pack_hit_skips_compilation(monkeypatch, tmp_path):
    graph = SimpleNamespace(to_dict=lambda: {"nodes": {}})
    packed = object()
    compile_calls = []

    from emmy import config
    from emmy.compiler.backend import pack

    monkeypatch.setattr(config, "pack_dir", lambda: tmp_path)
    monkeypatch.setattr(pack, "pack_path", lambda _root, _key: tmp_path / "external")
    monkeypatch.setattr(pack, "load_pack", lambda _path, *, key: {"external": packed})
    monkeypatch.setattr(external, "_pack_lock", lambda _path: nullcontext())

    got = _resolve_external_plan(graph, None, "auto", None, lambda: compile_calls.append(True))
    assert got is packed
    assert compile_calls == []


def test_external_plan_pack_miss_rechecks_under_lock_then_saves_once(monkeypatch, tmp_path):
    graph = SimpleNamespace(to_dict=lambda: {"nodes": {"linear": {"shape": [1, 64]}}})
    compiled = object()
    loads = []
    saves = []

    from emmy import config
    from emmy.compiler.backend import pack

    monkeypatch.setattr(config, "pack_dir", lambda: tmp_path)
    monkeypatch.setattr(pack, "pack_path", lambda _root, _key: tmp_path / "external")
    monkeypatch.setattr(pack, "load_pack", lambda path, *, key: loads.append((path, key)) or None)
    monkeypatch.setattr(pack, "save_pack", lambda path, plans, *, key, provenance: saves.append((path, plans, key, provenance)))
    monkeypatch.setattr(external, "_pack_lock", lambda path: saves.append(("lock", path)) or nullcontext())

    compile_calls = []
    got = _resolve_external_plan(graph, None, "auto", None, lambda: compile_calls.append(True) or compiled)

    assert got is compiled
    assert len(loads) == 2
    assert compile_calls == [True]
    assert saves[0] == ("lock", tmp_path / "external.lock")
    assert saves[1][0] == tmp_path / "external"
    assert saves[1][1] == {"external": compiled}
    assert saves[1][3] == {"kind": "external-program"}
