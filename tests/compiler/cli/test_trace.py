import argparse
import json
from types import SimpleNamespace

import pytest

from emmy.commands.trace import handle_trace, register_trace_command, trace_inline_code
from emmy.compiler.graph import Graph
from emmy.compiler.pipeline.search.golden import load_golden_file
from emmy.compiler.pipeline.search.working_golden import golden_sidecar_dir, write_trace_inventory


def test_trace_parser_accepts_working_golden_output():
    parser = argparse.ArgumentParser()
    register_trace_command(parser.add_subparsers(dest="command"))

    args = parser.parse_args(["trace", "some/model", "--golden-output", "work.yaml"])

    assert args.golden_output == "work.yaml"


def test_trace_parser_shares_model_adapter_and_dynamic_inputs():
    parser = argparse.ArgumentParser()
    register_trace_command(parser.add_subparsers(dest="command"))

    args = parser.parse_args(
        [
            "trace",
            "some/model",
            "--adapter",
            "causal-lm",
            "--layer",
            "2",
            "--dynamic",
            "seq_len@x:1",
            "--target",
            "sm_90",
        ]
    )

    assert args.input == "some/model"
    assert args.adapter == "causal-lm"
    assert args.layer == 2
    assert args.dynamic == ["seq_len@x:1"]
    assert args.target == "sm_90"


def test_golden_only_trace_reuses_shared_loader_without_auto_graph(monkeypatch, tmp_path):
    import emmy.commands.compile as compile_command

    graph = trace_inline_code("torch.relu(torch.randn(8))")["graph"]
    seen = {}

    def fake_load_or_trace(args):
        seen["adapter"] = args.adapter
        seen["dynamic"] = args.dynamic
        return graph, "auto-name", (None, (), {})

    def fake_write_golden(traced, path, *, graph_output, model):
        seen["graph"] = traced
        seen["model"] = model
        return SimpleNamespace(path=path, target_count=1)

    monkeypatch.setattr(compile_command, "load_or_trace", fake_load_or_trace)
    monkeypatch.setattr("emmy.commands.trace.write_trace_inventory", fake_write_golden)
    monkeypatch.chdir(tmp_path)

    parser = argparse.ArgumentParser()
    register_trace_command(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["trace", "some/model", "--dynamic", "seq_len@x:1", "--golden-output", str(tmp_path / "working.yaml")])
    handle_trace(args)

    assert seen == {
        "adapter": "causal-lm",
        "dynamic": ["seq_len@x:1"],
        "graph": graph,
        "model": "some/model",
    }
    assert not (tmp_path / "auto-name.json").exists()


def test_explicit_graph_output_is_kept_with_golden_output(monkeypatch, tmp_path):
    import emmy.commands.compile as compile_command

    graph = trace_inline_code("torch.relu(torch.randn(8))")["graph"]
    monkeypatch.setattr(compile_command, "load_or_trace", lambda _args: (graph, "auto-name", (None, (), {})))
    monkeypatch.setattr(
        "emmy.commands.trace.write_trace_inventory",
        lambda _graph, path, **_kwargs: SimpleNamespace(path=path, target_count=1),
    )

    parser = argparse.ArgumentParser()
    register_trace_command(parser.add_subparsers(dest="command"))
    output = tmp_path / "graph.json"
    args = parser.parse_args(["trace", "some/model", "--output", str(output), "--golden-output", str(tmp_path / "working.yaml")])
    handle_trace(args)

    assert Graph.from_dict(json.loads(output.read_text())).nodes


def test_golden_trace_accepts_pretraced_ir(monkeypatch, tmp_path):
    source_graph = trace_inline_code("torch.relu(torch.randn(8))")["graph"]
    source = tmp_path / "source.json"
    source.write_text(json.dumps(source_graph.to_dict()))
    seen = {}

    def fake_write_golden(graph, path, *, graph_output, model):
        seen["graph"] = graph
        seen["model"] = model
        return SimpleNamespace(path=path, target_count=1)

    monkeypatch.setattr("emmy.commands.trace.write_trace_inventory", fake_write_golden)
    parser = argparse.ArgumentParser()
    register_trace_command(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["trace", str(source), "--golden-output", str(tmp_path / "working.yaml")])

    handle_trace(args)

    assert seen["graph"].to_dict() == source_graph.to_dict()
    assert seen["model"] is None
    assert json.loads(source.read_text()) == source_graph.to_dict()


def test_trace_writes_deterministic_unmeasured_kernel_reproducers(tmp_path):
    graph = trace_inline_code("torch.relu(torch.randn(16,32))")["graph"]
    first = tmp_path / "first" / "working.yaml"
    second = tmp_path / "second" / "working.yaml"

    write_trace_inventory(graph, first, model="org/model")
    write_trace_inventory(graph, second, model="org/model")

    first_doc = load_golden_file(first)
    second_doc = load_golden_file(second)
    assert first_doc == second_doc
    assert first_doc["model"] == "org/model"
    assert first_doc["configs"]
    assert all(set(entry) == {"kernel", "name", "reproducer"} for entry in first_doc["configs"])
    assert all(entry["kernel"] == "traced" for entry in first_doc["configs"])
    assert len({entry["name"] for entry in first_doc["configs"]}) == len(first_doc["configs"])

    for entry in first_doc["configs"]:
        repro = first.parent / entry["reproducer"]
        assert repro.suffix == ".json"
        assert repro.name.endswith(".torch.json")
        assert Graph.from_dict(json.loads(repro.read_text())).nodes


def test_trace_inventory_keeps_flash_attention_as_one_fold_aware_target(tmp_path):
    graph = trace_inline_code(
        "F.scaled_dot_product_attention(torch.randn(1,2,8,16), torch.randn(1,2,8,16), torch.randn(1,2,8,16), is_causal=True)"
    )["graph"]
    path = tmp_path / "working.yaml"

    write_trace_inventory(graph, path, model=None)

    document = load_golden_file(path)
    assert len(document["configs"]) == 1
    reproducer = path.parent / document["configs"][0]["reproducer"]
    repro_graph = Graph.from_dict(json.loads(reproducer.read_text()))
    assert [type(node.op).__name__ for node in repro_graph.nodes.values()].count("SdpaOp") == 1


def test_trace_refuses_to_mix_with_existing_working_golden(tmp_path):
    graph = trace_inline_code("torch.relu(torch.randn(8))")["graph"]
    path = tmp_path / "working.yaml"
    write_trace_inventory(graph, path, model=None)

    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_trace_inventory(graph, path, model=None)

    orphan = tmp_path / "orphan.yaml"
    golden_sidecar_dir(orphan).mkdir()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_trace_inventory(graph, orphan, model=None)
