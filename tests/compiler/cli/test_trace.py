from __future__ import annotations

import argparse
import json

import pytest

from emmy.commands.trace import handle_trace, register_trace_command, trace_inline_code
from emmy.compiler.backend import torch_ref
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp, GatherOp
from emmy.compiler.pipeline.search.golden import load_golden_file, load_golden_records
from emmy.compiler.pipeline.search.working_golden import write_trace_inventory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    register_trace_command(parser.add_subparsers(dest="command"))
    return parser


def test_trace_parser_has_one_golden_yaml_output() -> None:
    args = _parser().parse_args(["trace", "some/model", "-o", "work.yaml"])
    assert args.output == "work.yaml"
    assert not hasattr(args, "golden_output")
    with pytest.raises(SystemExit):
        _parser().parse_args(["trace", "some/model", "--golden-output", "legacy.yaml"])


def test_trace_parser_shares_model_adapter_and_dynamic_inputs() -> None:
    args = _parser().parse_args(
        ["trace", "some/model", "--adapter", "causal-lm", "--layer", "2", "--dynamic", "seq_len@x:1", "--target", "sm_90"]
    )
    assert (args.input, args.layer, args.dynamic, args.target) == ("some/model", 2, ["seq_len@x:1"], "sm_90")


def test_trace_command_writes_only_golden_yaml(monkeypatch, tmp_path) -> None:
    import emmy.commands.compile as compile_command

    graph = trace_inline_code("torch.relu(torch.randn(8))")["graph"]
    monkeypatch.setattr(compile_command, "load_or_trace", lambda _args, **_kwargs: (graph, "auto-name", (None, (), {})))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "trace.yaml"
    handle_trace(_parser().parse_args(["trace", "some/model", "-o", str(output)]))
    records = load_golden_records(load_golden_file(output))
    assert records and all(record.program.nodes for record in records)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["trace.yaml"]


def test_trace_accepts_debug_graph_json_as_input_but_emits_yaml(monkeypatch, tmp_path) -> None:
    source_graph = trace_inline_code("torch.relu(torch.randn(8))")["graph"]
    source = tmp_path / "source.json"
    source.write_text(json.dumps(source_graph.to_dict()))
    output = tmp_path / "working.yaml"
    handle_trace(_parser().parse_args(["trace", str(source), "-o", str(output)]))
    assert load_golden_records(load_golden_file(output))
    assert json.loads(source.read_text()) == source_graph.to_dict()


def test_trace_writes_deterministic_self_contained_programs(tmp_path) -> None:
    graph = trace_inline_code("torch.relu(torch.randn(16,32))")["graph"]
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    write_trace_inventory(graph.copy(), first, model="org/model")
    write_trace_inventory(graph.copy(), second, model="org/model")
    first_doc, second_doc = load_golden_file(first), load_golden_file(second)
    assert first_doc == second_doc
    assert first_doc["programs"] and first_doc["configs"]
    assert all(set(entry) == {"name", "program", "target"} for entry in first_doc["configs"])
    assert all(set(entry["target"]) == {"origins"} for entry in first_doc["configs"])


def test_trace_target_resolves_in_original_multi_op_fusion_context(tmp_path) -> None:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (16, 64), "f16"), node_id="x")
    graph.add_node(InputOp(), [], Tensor("w1", (32, 64), "f16"), node_id="w1")
    graph.add_node(InputOp(), [], Tensor("w2", (32, 64), "f16"), node_id="w2")
    graph.add_node(LinearOp(), ["x", "w1"], Tensor("gate", (16, 32), "f16"), node_id="gate")
    graph.add_node(LinearOp(), ["x", "w2"], Tensor("up", (16, 32), "f16"), node_id="up")
    graph.add_node(ElementwiseOp("multiply"), ["gate", "up"], Tensor("out", (16, 32), "f16"), node_id="out")
    graph.inputs, graph.outputs = ["x", "w1", "w2"], ["out"]

    path = tmp_path / "working.yaml"
    write_trace_inventory(graph, path)
    document = load_golden_file(path)
    (record,) = load_golden_records(document)

    assert len(document["programs"]) == 1
    assert set(record.origins) == {"gate", "up", "out"}
    assert record.shape_key.reduce_max == 64


def test_trace_inventory_keeps_flash_attention_as_one_target(tmp_path) -> None:
    graph = trace_inline_code(
        "F.scaled_dot_product_attention(torch.randn(1,2,8,16), torch.randn(1,2,8,16), torch.randn(1,2,8,16), is_causal=True)"
    )["graph"]
    path = tmp_path / "working.yaml"
    write_trace_inventory(graph, path)
    records = load_golden_records(load_golden_file(path))
    assert len(records) == 1
    assert records[0].origin_ops == ("torch.sdpa",)


def test_trace_serializes_target_without_a_torch_reference_mapping(tmp_path) -> None:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    graph.add_node(InputOp(), [], Tensor("index", (4, 8), "i64"), node_id="index")
    graph.add_node(GatherOp(axis=1), ["x", "index"], Tensor("gather", (4, 8)), node_id="gather")
    graph.inputs, graph.outputs = ["x", "index"], ["gather"]
    assert torch_ref.is_runnable(graph) is False

    path = tmp_path / "working.yaml"
    write_trace_inventory(graph, path)
    (record,) = load_golden_records(load_golden_file(path))
    assert record.origin_ops == ("tensor.gather",)
    assert record.program.nodes["gather"].op == GatherOp(axis=1)


def test_trace_refuses_to_replace_existing_yaml(tmp_path) -> None:
    graph = trace_inline_code("torch.relu(torch.randn(8))")["graph"]
    path = tmp_path / "working.yaml"
    write_trace_inventory(graph, path)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_trace_inventory(graph, path)
