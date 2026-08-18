"""Exact compiler graph contracts for DeepSeek V4 routing and experts."""

import torch

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.backend.plan import plan_from_graph
from emmy.compiler.ir.cuda.ir import resolve_dim
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.serving.deepseek_experts import RoutedExpertsModule, trace_deepseek_experts, trace_deepseek_route


def _plan(graph):
    return plan_from_graph(CudaBackend(tune_db=None).compile(graph))


def test_learned_and_hash_route_graphs_preserve_pinned_source_semantics():
    learned_graph = trace_deepseek_route(rows=1, kind="learned")
    subtracts = [node for node in learned_graph.nodes.values() if isinstance(node.op, ElementwiseOp) and node.op.name == "subtract"]
    assert subtracts == []
    stable = learned_graph.nodes["weights"]
    assert stable.inputs[0] != stable.inputs[1]
    learned = _plan(learned_graph)
    assert learned.inputs == ["router_logits", "bias"]
    assert learned.outputs == ["weights", "ids"]
    learned_source = "\n".join(kernel.source for kernel in learned.kernels.values())
    assert "__expf" in learned_source and "__logf" in learned_source
    assert "value > best_value" in learned_source and "value >= best_value" not in learned_source

    hashed = _plan(trace_deepseek_route(rows=1, kind="hash"))
    assert hashed.inputs == ["router_logits", "table", "input_ids"]
    assert hashed.outputs == ["weights", "ids"]
    hash_source = "\n".join(kernel.source for kernel in hashed.kernels.values())
    assert "__shfl_xor_sync" in hash_source
    assert "candidate >= 0 && candidate < 256" in hash_source


def test_symbolic_route_outputs_and_launches_follow_every_runtime_width():
    for kind in ("learned", "hash"):
        plan = _plan(trace_deepseek_route(rows=128, kind=kind, symbolic=True))
        assert plan.symbolic_bindings == {"num_tokens": ("router_logits", 0)}
        buffers = {buffer.name: buffer for buffer in plan.buffers}
        for rows in (3, 4096):
            values = {"num_tokens": rows}
            assert buffers["weights"].resolve_shape(values) == (rows, 6)
            assert buffers["ids"].resolve_shape(values) == (rows, 6)
            assert all(launch.runtime_args == ("num_tokens",) for launch in plan.launches)
            assert all(resolve_dim(launch.grid[0], values) > 0 for launch in plan.launches)
        assert all(
            resolve_dim(launch.grid[0], {"num_tokens": 3}) < resolve_dim(launch.grid[0], {"num_tokens": 4096}) for launch in plan.launches
        )


def test_routed_expert_eager_contract_clamps_and_combines_in_fp32():
    module = RoutedExpertsModule(top_k=2, hidden=4, swiglu_limit=10.0)
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float16)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    ids = torch.tensor([[1, 0]], dtype=torch.int32)
    w13 = torch.arange(2 * 6 * 4, dtype=torch.float16).reshape(2, 6, 4) / 10
    w2 = torch.arange(2 * 4 * 3, dtype=torch.float16).reshape(2, 4, 3) / 20

    actual = module(x, weights, ids, w13, w2)
    routed = x[:, None, :].expand(-1, 2, -1).reshape(-1, 4)
    gate_up = torch.bmm(routed.unsqueeze(1), w13[ids.flatten()].transpose(1, 2)).squeeze(1)
    gate, up = gate_up.chunk(2, dim=-1)
    intermediate = torch.nn.functional.silu(torch.clamp(gate, max=10.0)) * torch.clamp(up, min=-10.0, max=10.0)
    routed_output = torch.bmm(intermediate.unsqueeze(1), w2[ids.flatten()].transpose(1, 2)).squeeze(1).view(1, 2, 4)
    expected = (routed_output * weights.unsqueeze(-1)).sum(dim=1).half()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_retained_expert_plan_has_one_physical_representation_and_four_launches():
    plan = _plan(trace_deepseek_experts(rows=1))
    assert plan.inputs == ["x", "route_weights", "route_ids", "w13", "w2", "w13_scale", "w2_scale"]
    assert len(plan.outputs) == 1
    assert len(plan.launches) == 4

    symbolic = _plan(trace_deepseek_experts(rows=128, symbolic=True))
    assert len(symbolic.launches) == 4
    assert not [buffer.name for buffer in symbolic.buffers if "decoded" in buffer.name]
    assert symbolic.symbolic_bindings == {"num_tokens": ("x", 0)}


def test_wide_expert_gather_keeps_its_data_dependent_index_producer():
    plan = _plan(trace_deepseek_experts(rows=4096))
    w2 = next(launch for launch in plan.launches if launch.node_id == "mul_1")
    assert "flatten" in w2.arg_names
    assert "const int* flatten" in plan.kernels[w2.kernel_name].source
