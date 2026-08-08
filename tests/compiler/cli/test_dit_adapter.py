"""CLI dispatch and validation for the experimental DiT adapter."""

from __future__ import annotations

import argparse
import logging
from types import SimpleNamespace

import pytest


def _namespace(**overrides):
    values = {
        "adapter": "dit",
        "code": None,
        "input": "facebook/DiT-XL-2-256",
        "layer": 0,
        "seq_len": 512,
        "dynamic": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_compiler_commands_expose_adapter_flag():
    """Every model compiler command accepts the same adapter choices."""
    from emmy.commands.compile import register_compile_command
    from emmy.commands.run import register_run_command
    from emmy.commands.trace import register_trace_command
    from emmy.commands.tune import register_tune_command

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_compile_command(subparsers)
    register_run_command(subparsers)
    register_trace_command(subparsers)
    register_tune_command(subparsers)

    for command in ("trace", "compile", "run", "tune"):
        args = parser.parse_args([command, "facebook/DiT-XL-2-256", "--adapter", "dit", "--layer", "0"])
        assert args.adapter == "dit"
        assert args.layer == 0


def test_default_adapter_remains_causal_lm():
    """Existing commands keep the CausalLM path unless explicitly opted into DiT."""
    from emmy.commands.compile import register_compile_command

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_compile_command(subparsers)
    args = parser.parse_args(["compile", "--code", "torch.relu(torch.randn(4))"])
    assert args.adapter == "causal-lm"


@pytest.mark.parametrize("command", ["trace", "compile", "run", "tune"])
def test_dit_cli_requires_layer_before_loading(command, run_cli):
    """Every public command rejects the missing layer before touching CUDA or the Hub."""
    rc, stdout, stderr = run_cli(command, "facebook/DiT-XL-2-256", "--adapter", "dit")
    assert rc == 2
    assert "--layer is required" in stdout + stderr


def test_dit_cli_rejects_layer_bounds_and_dynamic_shapes(run_cli):
    """Bounds and v1's fixed-shape constraint are usage errors."""
    for extra, message in [
        (["--layer", "28"], "expected 0-27"),
        (["--layer", "0", "--dynamic", "seq_len@hidden_states:1"], "--dynamic is not supported"),
    ]:
        rc, stdout, stderr = run_cli("compile", "facebook/DiT-XL-2-256", "--adapter", "dit", *extra)
        assert rc == 2
        assert message in stdout + stderr


def test_load_or_trace_dispatches_to_dit(monkeypatch):
    """The selected adapter owns model reconstruction and returns the standard bundle."""
    import emmy.commands.compile as compile_command

    sentinel_graph = object()
    sentinel_bundle = (object(), (), {})
    seen = []
    monkeypatch.setattr(
        compile_command,
        "_trace_dit_model",
        lambda model_id, layer: (seen.append((model_id, layer)) or sentinel_graph, sentinel_bundle),
    )

    graph, base_name, bundle = compile_command.load_or_trace(_namespace())
    assert graph is sentinel_graph
    assert bundle is sentinel_bundle
    assert base_name == "facebook-dit-xl-2-256-dit-layer0"
    assert seen == [("facebook/DiT-XL-2-256", 0)]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"layer": None}, "--layer is required"),
        ({"layer": -1}, "expected 0-27"),
        ({"layer": 28}, "expected 0-27"),
        ({"dynamic": ["seq_len@hidden_states:1"]}, "--dynamic is not supported"),
    ],
)
def test_dit_validation_rejects_unsupported_cli_combinations(overrides, message, caplog):
    """Required layer, fixed bounds, and fixed shapes fail before model download."""
    from emmy.commands.compile import validate_trace_adapter_args

    caplog.set_level(logging.ERROR)
    with pytest.raises(SystemExit) as exc:
        validate_trace_adapter_args(_namespace(**overrides))
    assert exc.value.code == 2
    assert message in caplog.text


def test_dit_missing_dependency_has_install_hint(monkeypatch, caplog):
    """A missing image extra produces the actionable combined-extra install command."""
    import builtins

    from emmy.commands.compile import _trace_dit_model

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "diffusers":
            raise ModuleNotFoundError("No module named 'diffusers'", name="diffusers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    caplog.set_level(logging.ERROR)
    with pytest.raises(SystemExit) as exc:
        _trace_dit_model("facebook/DiT-XL-2-256", 0)
    assert exc.value.code == 1
    assert ".[compile,image]" in caplog.text
