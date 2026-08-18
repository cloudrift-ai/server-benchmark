"""Guard and lifecycle tests for the pinned 1Cat routed-expert adapter."""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

import emmy.serving.onecat_experts as onecat_experts
from emmy.serving.onecat_experts import _Adapter, _build_experts, _Program, _run, register_onecat_expert_kernels


def _router(*, hashed=False):
    return SimpleNamespace(
        top_k=6,
        global_num_experts=256,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        e_score_correction_bias=None if hashed else torch.zeros(256, dtype=torch.float32),
        _hash_indices_table=torch.zeros((129280, 6), dtype=torch.int32) if hashed else None,
    )


def _route_inputs(*, hashed=False, rows=1):
    hidden = torch.zeros((rows, 4096), dtype=torch.float16)
    logits = torch.zeros((rows, 256), dtype=torch.float32)
    input_ids = torch.zeros((rows,), dtype=torch.int32) if hashed else None
    return _router(hashed=hashed), hidden, logits, input_ids


def _route_adapter(*, capturing=lambda: False, builder=None, runner=None, rows=1, reference_weights=None):
    reference_weights = reference_weights if reference_weights is not None else torch.full((rows, 6), 0.25, dtype=torch.float32)
    reference_ids = torch.arange(6, dtype=torch.int32).reshape(1, 6).expand(rows, -1).clone()
    calls = []

    def original(router, hidden, logits, indices_type, *, input_ids=None):
        calls.append((router, hidden, logits, indices_type, input_ids))
        return reference_weights.clone(), reference_ids.clone()

    def default_builder(rows, kind):
        inputs = ("router_logits", "bias") if kind == "learned" else ("router_logits", "table", "input_ids")
        return _Program(object(), inputs, ("weights", "ids"), symbolic=rows not in onecat_experts.PROFILE_ROWS)

    def default_runner(program, tensors, device):
        del program, device
        tensors["weights"].copy_(reference_weights)
        tensors["ids"].copy_(reference_ids)

    return (
        _Adapter(
            original,
            lambda *_args: None,
            build_route=builder or default_builder,
            run=runner or default_runner,
            is_capturing=capturing,
            platform_supported=lambda *_tensors: True,
        ),
        calls,
    )


def test_route_first_use_verifies_then_uses_custom_boundary_with_exact_output_order():
    adapter, calls = _route_adapter()
    router, hidden, logits, input_ids = _route_inputs()
    hot = (torch.full((1, 6), 7.0), torch.full((1, 6), 8, dtype=torch.int32))
    op_calls = []

    first = adapter.dispatch_route(router, hidden, logits, None, input_ids, lambda *_args: op_calls.append(1) or hot)
    torch.testing.assert_close(first[0], torch.full((1, 6), 0.25), rtol=0, atol=0)
    torch.testing.assert_close(first[1], torch.arange(6, dtype=torch.int32).reshape(1, 6), rtol=0, atol=0)
    second = adapter.dispatch_route(router, hidden, logits, None, input_ids, lambda *_args: op_calls.append(1) or hot)

    assert second is hot
    assert len(calls) == 1
    assert op_calls == [1]


def test_route_cold_capture_and_build_or_parity_failure_fall_back_permanently():
    builds = 0

    def builder(rows, kind):
        nonlocal builds
        builds += 1
        return _Program(object(), ("router_logits", "bias"), ("weights", "ids"))

    capturing = True
    adapter, calls = _route_adapter(capturing=lambda: capturing, builder=builder)
    router, hidden, logits, input_ids = _route_inputs()
    adapter.dispatch_route(router, hidden, logits, None, input_ids, lambda *_args: None)
    assert builds == 0 and len(calls) == 1

    capturing = False

    def mismatch(program, tensors, device):
        del program, device
        tensors["weights"].zero_()
        tensors["ids"].zero_()

    adapter._run = mismatch
    adapter.dispatch_route(router, hidden, logits, None, input_ids, lambda *_args: None)
    adapter.dispatch_route(router, hidden, logits, None, input_ids, lambda *_args: None)
    assert builds == 1 and len(calls) == 3
    assert (1, "learned") in adapter.disabled_routes


def test_route_accepts_arbitrary_width_and_reuses_one_symbolic_program():
    adapter, calls = _route_adapter(rows=3)
    router, hidden, logits, input_ids = _route_inputs(rows=3)
    hot = (torch.ones((3, 6)), torch.zeros((3, 6), dtype=torch.int32))

    first = adapter.dispatch_route(router, hidden, logits, None, input_ids, lambda *_args: hot)
    second = adapter.dispatch_route(router, hidden, logits, None, input_ids, lambda *_args: hot)

    assert first[0].shape == first[1].shape == (3, 6)
    assert second is hot
    assert len(calls) == 1
    assert adapter.routes[(3, "learned")].symbolic


def test_route_accepts_fast_math_payload_delta_when_ids_are_exact():
    reference = torch.full((1, 6), 0.25, dtype=torch.float32)

    def runner(_program, tensors, _device):
        tensors["weights"].copy_(reference + 5e-7)
        tensors["ids"].copy_(torch.arange(6, dtype=torch.int32).reshape(1, 6))

    adapter, calls = _route_adapter(reference_weights=reference, runner=runner)
    router, hidden, logits, input_ids = _route_inputs()
    hot = (torch.ones((1, 6)), torch.zeros((1, 6), dtype=torch.int32))

    first = adapter.dispatch_route(router, hidden, logits, None, input_ids, lambda *_args: hot)
    second = adapter.dispatch_route(router, hidden, logits, None, input_ids, lambda *_args: hot)

    torch.testing.assert_close(first[0], reference, rtol=0, atol=1e-6)
    assert second is hot
    assert len(calls) == 1


class _Retained:
    def __init__(self, shape, dtype, device):
        self.shape = shape
        self.dtype = dtype
        self.device = device

    @staticmethod
    def is_contiguous():
        return True


def _expert_layer(device):
    return SimpleNamespace(
        w13_tm_weight=_Retained((256, 4096, 64), torch.int32, device),
        w13_tm_scales=_Retained((256, 128, 512), torch.uint8, device),
        w2_tm_weight=_Retained((256, 256, 512), torch.int32, device),
        w2_tm_scales=_Retained((256, 8, 4096), torch.uint8, device),
        sm70_mxfp4_num_experts=256,
        local_num_experts=256,
        global_num_experts=256,
        expert_map=None,
        apply_router_weight_on_input=False,
        swiglu_limit=10.0,
    )


def test_expert_first_use_checks_each_width_and_reuses_one_symbolic_program():
    calls = []

    def original(method, layer, x, weights, ids, shared, shared_input):
        calls.append((method, layer, shared, shared_input))
        return torch.full_like(x, 3.0)

    def build(rows):
        return _Program(object(), (), ("output",))

    def run(program, tensors, device):
        del program, device
        tensors["output"].fill_(3.03)

    adapter = _Adapter(
        lambda *_args: None,
        original,
        build_experts=build,
        run=run,
        is_capturing=lambda: False,
        platform_supported=lambda *_tensors: True,
    )
    x = torch.zeros((1, 4096), dtype=torch.float16)
    weights = torch.full((1, 6), 0.25, dtype=torch.float32)
    ids = torch.arange(6, dtype=torch.int32).reshape(1, 6)
    layer = _expert_layer(x.device)
    first = adapter.dispatch_experts(object(), layer, x, weights, ids, None, None, lambda *_args: None)
    hot = torch.full_like(x, 4.0)
    second = adapter.dispatch_experts(object(), layer, x, weights, ids, None, None, lambda *_args: hot)
    assert torch.equal(first, torch.full_like(x, 3.03))
    assert second is hot
    assert len(calls) == 1

    wide = torch.zeros((1024, 4096), dtype=torch.float16)
    wide_weights = torch.zeros((1024, 6), dtype=torch.float32)
    wide_ids = torch.zeros((1024, 6), dtype=torch.int32)
    wide_first = adapter.dispatch_experts(object(), layer, wide, wide_weights, wide_ids, None, None, lambda *_args: None)
    assert len(calls) == 2
    assert torch.equal(wide_first, torch.full_like(wide, 3.03))
    assert adapter.experts[1].runtime is adapter.experts[1024].runtime

    odd = torch.zeros((17, 4096), dtype=torch.float16)
    odd_weights = torch.zeros((17, 6), dtype=torch.float32)
    odd_ids = torch.zeros((17, 6), dtype=torch.int32)
    odd_first = adapter.dispatch_experts(object(), layer, odd, odd_weights, odd_ids, None, None, lambda *_args: None)
    assert len(calls) == 3
    assert torch.equal(odd_first, torch.full_like(odd, 3.03))
    assert adapter.experts[1].runtime is adapter.experts[17].runtime


def test_expert_contract_rejects_a_cross_device_carrier_before_launch():
    adapter = _Adapter(
        lambda *_args: None,
        lambda *_args: None,
        platform_supported=lambda *_tensors: True,
    )
    x = torch.zeros((1, 4096), dtype=torch.float16)
    weights = torch.zeros((1, 6), dtype=torch.float32)
    ids = torch.zeros((1, 6), dtype=torch.int32)
    layer = _expert_layer(x.device)
    layer.w2_tm_scales = _Retained((256, 8, 4096), torch.uint8, torch.device("meta"))

    assert adapter._expert_contract(layer, x, weights, ids) is None


def test_external_views_are_created_inside_the_current_torch_stream(monkeypatch):
    events = []
    active = False
    torch_stream = object()

    class ExternalStream:
        def __enter__(self):
            nonlocal active
            active = True
            events.append("enter")

        def __exit__(self, *_args):
            nonlocal active
            active = False
            events.append("exit")

    cupy = ModuleType("cupy")
    cupy.cuda = SimpleNamespace(Stream=SimpleNamespace(from_external=lambda stream: events.append(stream) or ExternalStream()))

    def from_dlpack(tensor):
        assert active
        events.append(tensor)
        return tensor

    cupy.from_dlpack = from_dlpack
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: torch_stream)
    from emmy.compiler.backend import gpu_lock as gpu_lock_module

    monkeypatch.setattr(gpu_lock_module, "gpu_lock", nullcontext)
    runtime = SimpleNamespace(run_once_external=lambda bindings: events.append(bindings))
    program = _Program(runtime, ("x",), ("output",))
    device = object()
    _run(program, {"x": "x", "output": "output"}, device)

    assert events == [torch_stream, "enter", "x", "output", {"x": "x", "output": "output"}, "exit"]


def test_expert_builder_rejects_a_full_logical_weight_scratch(monkeypatch):
    class Buffer:
        name = "temporary"
        role = "scratch"

        @staticmethod
        def resolve_shape(_values):
            return 256, 4096, 256

    plan = SimpleNamespace(
        inputs=("x", "route_weights", "route_ids", "w13", "w2", "w13_scale", "w2_scale"),
        outputs=("output",),
        buffers=(Buffer(),),
        launches=(object(),) * 4,
        symbolic_hints={"num_tokens": 4096},
    )
    monkeypatch.setattr("emmy.serving.deepseek_experts.trace_deepseek_experts", lambda **_kwargs: object())
    monkeypatch.setattr("emmy.serving.external.load_external_program", lambda *_args, **_kwargs: (object(), plan))

    with pytest.raises(RuntimeError, match="full_weight_scratch"):
        _build_experts(17)


def _modules(*, bad_expert=False):
    class Router:
        def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
            return hidden_states, router_logits, indices_type, input_ids

    class Expert:
        if bad_expert:

            def apply(self, layer, x):
                return layer, x

        else:

            def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
                return layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input

    return SimpleNamespace(FusedTopKBiasRouter=Router), SimpleNamespace(Mxfp4SM70MoEMethod=Expert)


def test_registration_is_all_or_none_and_idempotent(monkeypatch):
    monkeypatch.setattr(onecat_experts, "_ACTIVE", None)
    monkeypatch.setattr(onecat_experts, "_custom_ops", lambda: (lambda *_args: None, lambda *_args: None))
    router_module, expert_module = _modules()
    original_route = router_module.FusedTopKBiasRouter._compute_routing
    original_expert = expert_module.Mxfp4SM70MoEMethod.apply

    assert register_onecat_expert_kernels(router_module, expert_module)
    assert router_module.FusedTopKBiasRouter._compute_routing is not original_route
    assert expert_module.Mxfp4SM70MoEMethod.apply is not original_expert
    replacements = (router_module.FusedTopKBiasRouter._compute_routing, expert_module.Mxfp4SM70MoEMethod.apply)
    assert register_onecat_expert_kernels(router_module, expert_module)
    assert replacements == (router_module.FusedTopKBiasRouter._compute_routing, expert_module.Mxfp4SM70MoEMethod.apply)

    bad_router_module, bad_expert_module = _modules(bad_expert=True)
    bad_original = bad_router_module.FusedTopKBiasRouter._compute_routing
    assert not register_onecat_expert_kernels(bad_router_module, bad_expert_module)
    assert bad_router_module.FusedTopKBiasRouter._compute_routing is bad_original
