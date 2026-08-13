from __future__ import annotations

import argparse
import json

import pytest

from emmy.commands.trace import handle_trace, register_trace_command, trace_inline_code
from emmy.compiler import provenance
from emmy.compiler.backend import torch_ref
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp, GatherOp
from emmy.compiler.pipeline.search.golden import load_golden_file, load_golden_records
from emmy.compiler.pipeline.search.working_golden import load_working_targets, write_trace_inventories, write_trace_inventory


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


def test_trace_serving_twins_requires_a_model_input() -> None:
    args = _parser().parse_args(["trace", "--serving-twins", "-o", "work.yaml"])
    with pytest.raises(SystemExit) as exc:
        handle_trace(args)
    assert exc.value.code == 2


def test_trace_serving_twins_requires_serving_config() -> None:
    args = _parser().parse_args(["trace", "org/model", "--serving-twins"])
    with pytest.raises(SystemExit) as exc:
        handle_trace(args)
    assert exc.value.code == 2


def test_trace_serving_twins_writes_one_exact_inventory_with_explicit_provenance(monkeypatch, tmp_path) -> None:
    import emmy.serving.twins as twins

    graphs = {
        "pre1@b2": trace_inline_code("torch.relu(torch.randn(8))")["graph"],
        "expert512@b2": trace_inline_code("torch.neg(torch.randn(16))")["graph"],
    }
    captured = {}

    def fake_capture(model, **kwargs):
        captured.update(model=model, **kwargs)
        return graphs

    monkeypatch.setattr(twins, "capture_twin_graphs", fake_capture)
    output = tmp_path / "serving.yaml"
    config = tmp_path / "release.env"
    config.write_text(
        f"SERVE_MODEL=cloudriftai/model-exl3\nSERVE_REVISION=0123456789abcdef0123456789abcdef01234567\n"
        f"SERVE_GPU=NVIDIA-Test\nSERVE_GOLDEN_FILE={output}\nSERVE_DECODE_BUCKET=64\n"
        'SERVE_MAX_NUM_BATCHED_TOKENS=1088\nSERVE_PREFILL_BUCKET=512\nSERVE_WARM_SHAPES=":: :1024:1088:fm"\n'.replace(":: ", "::1088 ")
    )
    handle_trace(
        _parser().parse_args(
            [
                "trace",
                str(tmp_path / "local-checkpoint"),
                "--serving-twins",
                "--serving-config",
                str(config),
                "--model-provenance",
                "cloudriftai/model-exl3@0123456789abcdef0123456789abcdef01234567",
                "-o",
                str(output),
            ]
        )
    )

    document = load_golden_file(output)
    records = load_golden_records(document)
    assert captured == {
        "model": str(tmp_path / "local-checkpoint"),
        "decode_bucket": 0,
        "prefill_bucket": 0,
    }
    assert document["model"] == "cloudriftai/model-exl3@0123456789abcdef0123456789abcdef01234567"
    assert {record.name.split(".", 1)[0] for record in records} == {"pre1@b2", "expert512@b2"}
    assert all(record.loop_wire is not None and not record.origins for record in records)
    assert {(record.bindings, record.pins) for record in records} >= {
        ((("num_tokens", 64),), (("FAST_MATH", False),)),
        ((("num_tokens", 1024),), (("FAST_MATH", True),)),
        ((), (("FAST_MATH", False),)),
        ((), (("FAST_MATH", True),)),
    }


def test_trace_serving_twins_static_only_release_forwards_exact_scope(monkeypatch, tmp_path) -> None:
    import emmy.serving.twins as twins

    graphs = {"pre1@b2": trace_inline_code("torch.relu(torch.randn(8))")["graph"]}
    captured = {}

    def fake_capture(model, **kwargs):
        captured.update(model=model, **kwargs)
        return graphs

    monkeypatch.setattr(twins, "capture_twin_graphs", fake_capture)
    output = tmp_path / "static.yaml"
    config = tmp_path / "static.env"
    config.write_text(
        f"SERVE_MODEL=org/model\nSERVE_GPU=NVIDIA-Test\nSERVE_GOLDEN_FILE={output}\nSERVE_STATIC_ONLY=1\n"
        "SERVE_MAX_NUM_BATCHED_TOKENS=1\nSERVE_DECODE_BUCKET=1\nSERVE_PREFILL_CAPACITY=1\n"
        "SERVE_PREFILL_BUCKET=0\nSERVE_M1_TIER=1\nSERVE_CAPTURE_SIZES=[1]\n"
    )
    handle_trace(
        _parser().parse_args(
            [
                "trace",
                "org/model",
                "--serving-twins",
                "--serving-config",
                str(config),
                "-o",
                str(output),
            ]
        )
    )

    assert captured == {
        "model": "org/model",
        "decode_bucket": 0,
        "prefill_bucket": 0,
    }
    records = load_golden_records(load_golden_file(output))
    assert {(record.bindings, record.pins) for record in records} == {((("num_tokens", 1),), (("FAST_MATH", False),))}


def test_trace_static_only_release_rejects_unsafe_config(tmp_path) -> None:
    config = tmp_path / "bad.env"
    config.write_text(
        "SERVE_MODEL=org/model\nSERVE_GPU=NVIDIA-Test\nSERVE_GOLDEN_FILE=golden.yaml\nSERVE_STATIC_ONLY=1\n"
        "SERVE_MAX_NUM_BATCHED_TOKENS=8\nSERVE_DECODE_BUCKET=1\n"
    )
    args = _parser().parse_args(["trace", "org/model", "--serving-twins", "--serving-config", str(config)])
    with pytest.raises(SystemExit) as exc:
        handle_trace(args)
    assert exc.value.code == 2


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
    assert all(set(entry) == {"program", "target", "realizations"} for entry in first_doc["configs"])
    assert all(set(entry["realizations"][0]) == {"name", "bindings", "pins"} for entry in first_doc["configs"])
    assert all(set(entry["target"]) == {"origins"} for entry in first_doc["configs"])


def test_trace_keeps_materialized_storage_outputs_and_quant_digest(tmp_path) -> None:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (4, 32), "f16"), node_id="x")
    bits = graph.add_node(ElementwiseOp("to_f8e4m3"), ["x"], Tensor("x_bits", (4, 32), "f8e4m3"), node_id="x_bits")
    graph.nodes[bits].hints.set("trace.materialize", True)
    graph.add_node(ElementwiseOp("from_f8e4m3"), [bits], Tensor("out", (4, 32), "f16"), node_id="out")
    graph.inputs, graph.outputs = ["x"], ["out"]

    path = tmp_path / "working.yaml"
    result = write_trace_inventory(
        graph,
        path,
        model_quant_digest="0123456789abcdef",
    )
    document = load_golden_file(path)

    assert result.target_count == 2
    assert document["model_quant_digest"] == "0123456789abcdef"
    assert "x_bits" in document["programs"][0]["outputs"]


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


def test_trace_inventory_keeps_every_kernel_even_without_cache_keys(monkeypatch, tmp_path) -> None:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x0", (16,)), node_id="x0")
    graph.add_node(InputOp(), [], Tensor("x1", (16,)), node_id="x1")
    graph.add_node(ElementwiseOp("relu"), ["x0"], Tensor("y0", (16,)), node_id="y0")
    graph.add_node(ElementwiseOp("relu"), ["x1"], Tensor("y1", (16,)), node_id="y1")
    graph.inputs, graph.outputs = ["x0", "x1"], ["y0", "y1"]
    monkeypatch.setattr(LoopOp, "cache_key", lambda _self: None)

    path = tmp_path / "working.yaml"
    result = write_trace_inventory(graph, path)
    records = load_golden_records(load_golden_file(path))

    assert result.target_count == 2
    assert len(records) == 2
    assert {record.origins for record in records} == {("y0",), ("y1",)}


def test_trace_inventory_embeds_loop_ir_when_frontend_provenance_is_missing(monkeypatch, tmp_path) -> None:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (16,)), node_id="x")
    graph.add_node(ElementwiseOp("relu"), ["x"], Tensor("y", (16,)), node_id="y")
    graph.inputs, graph.outputs = ["x"], ["y"]
    monkeypatch.setattr(provenance, "seed", lambda _graph: None)

    path = tmp_path / "working.yaml"
    write_trace_inventory(graph, path)
    document = load_golden_file(path)
    (record,) = load_golden_records(document)

    assert set(document) - {"gpu_name"} == {"compute_cap", "programs", "loops", "configs"}
    assert document["configs"][0]["target"] == {"loop": 0}
    assert record.origins == ()
    assert isinstance(record.target_program.nodes["y"].op, LoopOp)
    assert record.structural_features["S_pw_relu"] == 1.0
    _document, targets = load_working_targets(path)
    assert len(targets) == 1
    assert isinstance(targets[0].program.nodes["y"].op, LoopOp)


def test_trace_inventory_can_force_exact_loop_targets(tmp_path) -> None:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (16,)), node_id="x")
    graph.add_node(ElementwiseOp("relu"), ["x"], Tensor("y", (16,)), node_id="y")
    graph.inputs, graph.outputs = ["x"], ["y"]

    path = tmp_path / "working.yaml"
    write_trace_inventory(graph, path, force_loop_targets=True)
    (record,) = load_golden_records(load_golden_file(path))

    assert record.origins == ()
    assert record.loop_wire is not None


def test_combined_trace_inventory_deduplicates_identical_loop_targets(tmp_path) -> None:
    graph = trace_inline_code("torch.relu(torch.randn(8))")["graph"]
    path = tmp_path / "combined.yaml"
    result = write_trace_inventories({"pre1": graph, "pre8": graph.copy()}, path, model="org/model@revision")
    document = load_golden_file(path)
    records = load_golden_records(document)

    assert result.target_count == 1
    assert len(document["loops"]) == len(records) == 1
    assert records[0].name.startswith("pre1.")
    assert document["model"] == "org/model@revision"


def test_trace_yaml_uses_compact_graph_rows_but_block_candidate_rows(tmp_path) -> None:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x0", (512, 512), "f16"), node_id="x0")
    graph.add_node(InputOp(), [], Tensor("x1", (512, 512), "f16"), node_id="x1")
    graph.add_node(LinearOp(), ["x0", "x1"], Tensor("linear", (512, 512), "f16"), node_id="linear")
    graph.inputs, graph.outputs = ["x0", "x1"], ["linear"]

    path = tmp_path / "working.yaml"
    write_trace_inventory(graph, path)
    text = path.read_text()

    assert "inputs: [x0, x1]" in text
    assert "outputs: [[x0, f16, [512, 512]]]" in text
    assert "attrs: {has_bias: false}" in text
    assert "target:\n    origins:\n" in text


def test_trace_refuses_to_replace_existing_yaml(tmp_path) -> None:
    graph = trace_inline_code("torch.relu(torch.randn(8))")["graph"]
    path = tmp_path / "working.yaml"
    write_trace_inventory(graph, path)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_trace_inventory(graph, path)
