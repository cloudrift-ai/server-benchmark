"""Compiler graphs for exact DeepSeek V4 routing and retained compact experts."""

from __future__ import annotations

EXPERTS = 256
TOP_K = 6
HIDDEN = 4096
INTERMEDIATE = 256
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
VOCAB = 129280
ROWS_PER_GROUP = 16
WIDE_EXPERT_PROFILES = {1024: (2, 312), 4096: (6, 296)}


def _sqrt_softplus(values):
    import torch

    return torch.sqrt(torch.where(values > 20.0, values, torch.log(1.0 + torch.exp(values))))


class LearnedRouteModule:
    """Pinned learned score algebra before stable top-k selection."""

    def __new__(cls):
        import torch

        class Module(torch.nn.Module):
            def forward(self, router_logits, bias):
                score = _sqrt_softplus(router_logits.float())
                ranking = score + bias
                # The released image's loaded _moe_C binary returns unbiased
                # score payloads. Its checked-in d7612660 CUDA source contains
                # two bias subtractions, but direct runtime parity disproves
                # that dead source path; the executable binary is authoritative.
                return ranking, score

        return Module()


class HashRouteModule:
    """Pinned hash route score algebra before indexed candidate lookup."""

    def __new__(cls):
        import torch

        class Module(torch.nn.Module):
            def forward(self, router_logits):
                return _sqrt_softplus(router_logits.float())

        return Module()


class RoutedExpertsModule:
    """Route rows through retained W13, clamp-SwiGLU, W2, and combine."""

    def __new__(
        cls,
        *,
        top_k: int = TOP_K,
        hidden: int = HIDDEN,
        swiglu_limit: float = SWIGLU_LIMIT,
    ):
        import torch
        import torch.nn.functional as F

        class Module(torch.nn.Module):
            def forward(self, x, route_weights, route_ids, w13, w2):
                ids = route_ids.flatten()
                routed = x[:, None, :].expand(-1, top_k, -1).reshape(-1, hidden)
                gate_up = torch.bmm(routed.unsqueeze(1), w13[ids].transpose(1, 2)).squeeze(1)
                gate, up = gate_up.chunk(2, dim=-1)
                intermediate = F.silu(torch.clamp(gate, max=swiglu_limit))
                intermediate = intermediate * torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
                output = torch.bmm(intermediate.unsqueeze(1), w2[ids].transpose(1, 2)).squeeze(1)
                output = output.view(x.shape[0], top_k, hidden)
                return (output * route_weights.unsqueeze(-1)).sum(dim=1).to(torch.float16)

        return Module()


class GroupedInputModule:
    """Gather one grouped route tile from the token-major activation."""

    def __new__(cls):
        import torch

        class Module(torch.nn.Module):
            def forward(self, x, grouped_routes):
                valid = grouped_routes >= 0
                safe_routes = torch.where(valid, grouped_routes, 0)
                grouped_x = x[torch.div(safe_routes, TOP_K, rounding_mode="floor")]
                return torch.where(valid.unsqueeze(-1), grouped_x, 0.0)

        return Module()


class GroupedW13Module:
    """Apply one retained W13 expert to each stable route tile."""

    def __new__(cls):
        import torch

        class Module(torch.nn.Module):
            def forward(self, grouped_x, w13, group_experts):
                return torch.matmul(grouped_x, w13[group_experts].transpose(1, 2))

        return Module()


class GroupedActivationModule:
    """Apply exact clamp-SwiGLU to one grouped W13 result."""

    def __new__(cls):
        import torch
        import torch.nn.functional as F

        class Module(torch.nn.Module):
            def forward(self, gate_up):
                gate, up = gate_up.chunk(2, dim=-1)
                intermediate = F.silu(torch.clamp(gate, max=SWIGLU_LIMIT))
                return intermediate * torch.clamp(up, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)

        return Module()


class GroupedW2Module:
    """Apply one retained W2 expert to each activated route tile."""

    def __new__(cls):
        import torch

        class Module(torch.nn.Module):
            def forward(self, intermediate, w2, group_experts):
                return torch.matmul(intermediate, w2[group_experts].transpose(1, 2))

        return Module()


def trace_deepseek_route(*, rows: int, kind: str, symbolic: bool = False):
    """Trace one exact learned or hash route program."""
    import torch

    from emmy.compiler.dtype import I32
    from emmy.compiler.graph import Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.elementwise import ElementwiseImpl
    from emmy.compiler.ir.tensor.ir import ElementwiseOp, IndexedTopKOp, StableTopKOp
    from emmy.compiler.trace.torch import trace_module

    if rows < 1:
        raise ValueError(f"DeepSeek routing rows must be positive, got {rows}")
    dynamic_shapes = None
    if symbolic:
        from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

        dynamic_shapes = build_torch_dynamic_shapes(parse_position_specs(["num_tokens@router_logits:0"]))
    logits = torch.empty((rows, EXPERTS), dtype=torch.float32, device="meta")
    if kind == "learned":
        graph = trace_module(
            LearnedRouteModule(),
            (logits, torch.empty((EXPERTS,), dtype=torch.float32, device="meta")),
            dynamic_shapes=dynamic_shapes,
        )
    elif kind == "hash":
        graph = trace_module(HashRouteModule(), (logits,), dynamic_shapes=dynamic_shapes)
    else:
        raise ValueError(f"DeepSeek route kind must be 'learned' or 'hash', got {kind!r}")
    for node in graph.nodes.values():
        if isinstance(node.op, ElementwiseOp) and node.op.name in {"exp", "log"}:
            node.op = ElementwiseOp(ElementwiseImpl(f"{node.op.name}_fast"))

    if kind == "learned":
        ranking, payload = graph.outputs
        inputs = [ranking, payload]
        op = StableTopKOp(k=TOP_K, scale=ROUTE_SCALE)
    else:
        (payload,) = graph.outputs
        graph.add_node(InputOp(), [], Tensor("table", (VOCAB, TOP_K), I32), node_id="table")
        graph.add_node(InputOp(), [], Tensor("input_ids", (graph.nodes["router_logits"].output.shape[0],), I32), node_id="input_ids")
        graph.inputs.extend(("table", "input_ids"))
        inputs = [payload, "table", "input_ids"]
        op = IndexedTopKOp(k=TOP_K, scale=ROUTE_SCALE, reduction_lanes=32, lane_chunk=4)
    graph.add_node(
        op,
        inputs,
        outputs=[
            Tensor("weights", (graph.nodes["router_logits"].output.shape[0], TOP_K), "f32"),
            Tensor("ids", (graph.nodes["router_logits"].output.shape[0], TOP_K), I32),
        ],
        node_id="weights",
    )
    graph.outputs = ["weights", "ids"]
    return graph


def trace_deepseek_experts(*, rows: int, symbolic: bool = False):
    """Trace one complete retained-layout routed expert program."""
    import torch

    from emmy.compiler.loader.onecat_sm70_experts import spell_expert_inputs
    from emmy.compiler.trace.torch import trace_module

    if rows < 1:
        raise ValueError(f"DeepSeek expert rows must be positive, got {rows}")
    dynamic_shapes = None
    if symbolic:
        from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

        dynamic_shapes = build_torch_dynamic_shapes(
            parse_position_specs(
                [
                    "num_tokens@x:0",
                    "num_tokens@route_weights:0",
                    "num_tokens@route_ids:0",
                ]
            )
        )
    graph = trace_module(
        RoutedExpertsModule(),
        (
            torch.empty((rows, HIDDEN), dtype=torch.float16, device="meta"),
            torch.empty((rows, TOP_K), dtype=torch.float32, device="meta"),
            torch.empty((rows, TOP_K), dtype=torch.int32, device="meta"),
            torch.empty((EXPERTS, 2 * INTERMEDIATE, HIDDEN), dtype=torch.float16, device="meta"),
            torch.empty((EXPERTS, HIDDEN, INTERMEDIATE), dtype=torch.float16, device="meta"),
        ),
        dynamic_shapes=dynamic_shapes,
    )
    spell_expert_inputs(graph)
    return graph


def _wide_shape(rows: int) -> tuple[int, int, int]:
    try:
        shards, groups = WIDE_EXPERT_PROFILES[rows]
    except KeyError as exc:
        raise ValueError(f"no grouped expert profile for M={rows}") from exc
    return shards, groups, shards * groups


def trace_expert_bucket(*, rows: int):
    """Build the route-to-expert grouping program."""
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.tensor.ir import ExpertBucketOp

    _shards, _groups, total_groups = _wide_shape(rows)
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("route_ids", (rows, TOP_K), "i32"), node_id="route_ids")
    graph.add_node(
        ExpertBucketOp(experts=EXPERTS, routes=TOP_K, rows_per_group=ROWS_PER_GROUP),
        ["route_ids"],
        outputs=[
            Tensor("grouped_routes", (total_groups, ROWS_PER_GROUP), "i32"),
            Tensor("group_experts", (total_groups,), "i32"),
            Tensor("inverse", (rows, TOP_K), "i32"),
        ],
        node_id="grouped_routes",
    )
    graph.inputs = ["route_ids"]
    graph.outputs = ["grouped_routes", "group_experts", "inverse"]
    return graph


def trace_grouped_w13(*, rows: int):
    """Trace one reusable grouped W13 shard."""
    import torch

    from emmy.compiler.loader.onecat_sm70_experts import spell_grouped_w13_input
    from emmy.compiler.trace.torch import trace_module

    _shards, groups, _total_groups = _wide_shape(rows)
    graph = trace_module(
        GroupedW13Module(),
        (
            torch.empty((groups, ROWS_PER_GROUP, HIDDEN), dtype=torch.float16, device="meta"),
            torch.empty((EXPERTS, 2 * INTERMEDIATE, HIDDEN), dtype=torch.float16, device="meta"),
            torch.empty((groups,), dtype=torch.int32, device="meta"),
        ),
    )
    spell_grouped_w13_input(graph)
    return graph


def trace_grouped_input(*, rows: int):
    """Trace one reusable route-tile gather shard."""
    import torch

    from emmy.compiler.trace.torch import trace_module

    _shards, groups, _total_groups = _wide_shape(rows)
    return trace_module(
        GroupedInputModule(),
        (
            torch.empty((rows, HIDDEN), dtype=torch.float16, device="meta"),
            torch.empty((groups, ROWS_PER_GROUP), dtype=torch.int32, device="meta"),
        ),
    )


def trace_grouped_w2(*, rows: int):
    """Trace one reusable grouped activation-and-W2 shard."""
    import torch

    from emmy.compiler.loader.onecat_sm70_experts import spell_grouped_w2_input
    from emmy.compiler.trace.torch import trace_module

    _shards, groups, _total_groups = _wide_shape(rows)
    graph = trace_module(
        GroupedW2Module(),
        (
            torch.empty((groups, ROWS_PER_GROUP, INTERMEDIATE), dtype=torch.float16, device="meta"),
            torch.empty((EXPERTS, HIDDEN, INTERMEDIATE), dtype=torch.float16, device="meta"),
            torch.empty((groups,), dtype=torch.int32, device="meta"),
        ),
    )
    spell_grouped_w2_input(graph)
    return graph


def trace_grouped_activation(*, rows: int):
    """Trace one reusable grouped clamp-SwiGLU shard."""
    import torch

    from emmy.compiler.trace.torch import trace_module

    _shards, groups, _total_groups = _wide_shape(rows)
    return trace_module(
        GroupedActivationModule(),
        (torch.empty((groups, ROWS_PER_GROUP, 2 * INTERMEDIATE), dtype=torch.float16, device="meta"),),
    )


def trace_expert_unbucket(*, rows: int, shard_index: int):
    """Build one grouped-shard restore program."""
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.tensor.ir import RouteUnbucketOp

    shards, groups, _total_groups = _wide_shape(rows)
    if not 0 <= shard_index < shards:
        raise ValueError(f"grouped expert shard must be in [0,{shards}), got {shard_index}")
    graph = Graph()
    for name, shape, dtype in (
        ("base", (rows * TOP_K, HIDDEN), "f16"),
        ("grouped", (groups, ROWS_PER_GROUP, HIDDEN), "f16"),
        ("inverse", (rows, TOP_K), "i32"),
    ):
        graph.add_node(InputOp(), [], Tensor(name, shape, dtype), node_id=name)
    graph.add_node(
        RouteUnbucketOp(rows_per_group=ROWS_PER_GROUP, shard_index=shard_index),
        ["base", "grouped", "inverse"],
        Tensor("output", (rows * TOP_K, HIDDEN), "f16"),
        node_id="output",
    )
    graph.inputs = ["base", "grouped", "inverse"]
    graph.outputs = ["output"]
    return graph


def trace_weighted_route_sum(*, rows: int):
    """Build the deterministic route-order weighted combine program."""
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.tensor.ir import WeightedRouteSumOp

    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("partials", (rows, TOP_K, HIDDEN), "f16"), node_id="partials")
    graph.add_node(InputOp(), [], Tensor("weights", (rows, TOP_K), "f32"), node_id="weights")
    graph.add_node(
        WeightedRouteSumOp(routes=TOP_K),
        ["partials", "weights"],
        Tensor("output", (rows, HIDDEN), "f16"),
        node_id="output",
    )
    graph.inputs = ["partials", "weights"]
    graph.outputs = ["output"]
    return graph
