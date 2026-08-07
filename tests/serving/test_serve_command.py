"""``emmy serve`` command construction + flag routing + dry-run (no vllm/GPU)."""

import argparse
import json

import pytest

from emmy.commands.serve import build_bench_cmd, build_serve_cmd, handle_serve, register_serve_command

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _parse(argv):
    parser = argparse.ArgumentParser()
    register_serve_command(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def test_serve_cmd_plugin_defaults():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=[])
    assert cmd[:3] == ["vllm", "serve", MODEL]
    assert "--runner" in cmd and "pooling" in cmd
    assert "--enforce-eager" in cmd
    assert '{"architectures": ["EmmyEmbedModel"]}' in cmd
    assert cmd[cmd.index("--max-model-len") + 1] == "4096"
    assert "--gpu-memory-utilization=0.9" in cmd


def test_serve_cmd_stock_drops_plugin_but_keeps_parity():
    cmd = build_serve_cmd(MODEL, stock=True, vllm_args=[])
    assert "--enforce-eager" not in cmd
    assert "--hf-overrides" not in cmd
    assert "pooling" in cmd
    assert cmd[cmd.index("--max-model-len") + 1] == "4096"  # same cap as the plugin → apples-to-apples
    assert "--gpu-memory-utilization=0.9" in cmd


def test_serve_cmd_passthrough_and_max_model_len_override():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=["--max-model-len", "2048", "--gpu-memory-utilization", "0.8"])
    assert cmd.count("--max-model-len") == 1
    assert cmd[cmd.index("--max-model-len") + 1] == "2048"
    assert "--gpu-memory-utilization=0.9" not in cmd
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.8"


def test_serve_cmd_gpu_memory_utilization_equals_style_override():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=["--gpu-memory-utilization=0.8"])
    assert "--gpu-memory-utilization=0.9" not in cmd
    assert "--gpu-memory-utilization=0.8" in cmd


def test_serve_cmd_max_model_len_equals_style():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=["--max-model-len=2048"])
    assert "--max-model-len" not in cmd  # no separate default token added
    assert "--max-model-len=2048" in cmd
    assert "4096" not in cmd


def test_bench_cmd_targets_embeddings():
    cmd = build_bench_cmd(MODEL, port="8123", max_concurrency=8, num_prompts=32, random_input_len=64, seed=7)
    assert cmd[:3] == ["vllm", "bench", "serve"]
    assert cmd[cmd.index("--backend") + 1] == "openai-embeddings"
    assert cmd[cmd.index("--endpoint") + 1] == "/v1/embeddings"
    assert cmd[cmd.index("--base-url") + 1] == "http://localhost:8123"
    assert "--random-output-len" not in cmd


def test_bench_cmd_generate_targets_completions():
    cmd = build_bench_cmd(
        MODEL, port="8123", max_concurrency=8, num_prompts=32, random_input_len=64, seed=7, generate=True, random_output_len=96
    )
    assert cmd[cmd.index("--backend") + 1] == "openai"
    assert cmd[cmd.index("--endpoint") + 1] == "/v1/completions"
    assert cmd[cmd.index("--random-output-len") + 1] == "96"


def test_own_flags_after_model_are_extracted(capsys):
    # The argparse-REMAINDER footgun: everything after MODEL lands in
    # vllm_args, INCLUDING emmy's own flags. They must still be honored
    # (this exact case once exec'd a real server out of a --dry-run test).
    args = _parse(["serve", MODEL, "--bench", "--dry-run", "--random-input-len", "32", "--gpu-memory-utilization", "0.8"])
    handle_serve(args)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2, out
    assert "--bench" not in out[0] and "--dry-run" not in out[0]
    assert "--gpu-memory-utilization 0.8" in out[0]
    assert "--random-input-len 32" in out[1]  # bench param, not forwarded to serve
    assert "--random-input-len" not in out[0]


def test_verbatim_passthrough_after_double_dash(capsys):
    args = _parse(["serve", MODEL, "--bench", "--dry-run", "--", "--port", "8222", "--seed", "3"])
    handle_serve(args)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2
    assert out[0].startswith(f"vllm serve {MODEL}")
    assert "--port 8222" in out[0] and "--seed 3" in out[0]
    assert " -- " not in out[0]  # the separator itself is not forwarded
    assert "http://localhost:8222" in out[1]  # bench targets the passthrough port


def test_dry_run_serve_only(capsys):
    args = _parse(["serve", MODEL, "--dry-run"])
    handle_serve(args)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert "bench" not in out[0]


def test_bench_seed_is_distinct_from_vllm_seed(capsys):
    args = _parse(["serve", MODEL, "--bench", "--dry-run", "--bench-seed", "9", "--seed", "1"])
    handle_serve(args)
    out = capsys.readouterr().out.strip().splitlines()
    assert "--seed 1" in out[0]  # vllm serve gets --seed
    assert "--seed 9" in out[1]  # the bench client gets --bench-seed


def test_child_env_prepends_interpreter_bin_to_path(monkeypatch):
    """The vLLM child gets this interpreter's bin dir at the FRONT of PATH — so the venv's
    ninja (needed by the generative server's inductor compile) resolves even when emmy was
    invoked by absolute path, which does not activate the venv."""
    import sys
    from pathlib import Path

    from emmy.commands.serve import _child_env

    bin_dir = str(Path(sys.executable).parent)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = _child_env()
    assert env["PATH"].split(":")[0] == bin_dir
    assert "/usr/bin" in env["PATH"].split(":")  # the original entries are preserved


def test_child_env_is_idempotent_when_bin_already_leads(monkeypatch):
    """A bin dir already at the front of PATH is not duplicated (re-exec / nested invocation)."""
    import sys
    from pathlib import Path

    from emmy.commands.serve import _child_env

    bin_dir = str(Path(sys.executable).parent)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin")
    assert _child_env()["PATH"] == f"{bin_dir}:/usr/bin"


def test_serve_cmd_generate_branch():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=[], generate=True)
    assert cmd[cmd.index("--runner") + 1] == "generate"
    assert '{"architectures": ["EmmyGenModel"]}' in cmd
    assert cmd[cmd.index("--dtype") + 1] == "float16"  # forced for seam coherence
    # Default = dynamic-dim cap + decode bucket (16): the rider headroom the chunk+decode
    # twin split covers, so full chunk steps keep carrying their decode riders.
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "4112"
    # Whole-step decode capture is the DEFAULT: FULL_DECODE_ONLY graphs, capture sizes
    # following --max-num-seqs (vLLM default 256 when unset); no --enforce-eager.
    assert "--enforce-eager" not in cmd
    cfg = cmd[cmd.index("--compilation-config") + 1]
    assert '"cudagraph_mode": "FULL_DECODE_ONLY"' in cfg
    assert '"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256]' in cfg
    # The emmy generative arm defaults util to 0.97 — its cupy residents are invisible to
    # vLLM's torch-only profiler, so the 0.90 line can fall below them and fail the min-KV
    # fit at long model lens. Stock (and the embedding plugin) keep 0.90.
    assert "--gpu-memory-utilization=0.97" in cmd
    stock_cmd = build_serve_cmd(MODEL, stock=True, vllm_args=[], generate=True)
    assert "--gpu-memory-utilization=0.9" in stock_cmd


def test_serve_cmd_generate_capture_sizes_follow_max_num_seqs():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=["--max-num-seqs", "48"], generate=True)
    cfg = cmd[cmd.index("--compilation-config") + 1]
    # The decode bucket (16) and the cap itself always ride the list; the ladder fills between.
    assert '"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 48]' in cfg


def _spec(depth: int) -> list[str]:
    return ["--speculative-config", json.dumps({"method": "mtp", "model": "m", "num_speculative_tokens": depth})]


def _sizes(depth: int, vllm_args: list[str] | None = None) -> list[int]:
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=_spec(depth) + (vllm_args or []), generate=True)
    return json.loads(cmd[cmd.index("--compilation-config") + 1])["cudagraph_capture_sizes"]


@pytest.mark.parametrize("depth", [1, 2, 3, 5])
def test_serve_cmd_generate_spec_ladder_never_overshoots_the_bucket(depth):
    """THE invariant. vLLM rounds every capture size UP to a multiple of query_len and dispatches a
    step to the first size at or above its width, so the runner sees a PADDED width. If the first
    rung at or above a step's width exceeds the decode bucket, that step misses the static decode
    twin and silently runs the symbolic program. Every reachable verify width must therefore land on
    a rung that is still <= the bucket."""
    query_len = depth + 1
    sizes = _sizes(depth)
    bucket = 16  # emmy_config default
    assert all(s % query_len == 0 for s in sizes), sizes
    for width in range(query_len, bucket + 1, query_len):
        rung = min((s for s in sizes if s >= width), default=None)
        assert rung is not None and rung <= bucket, f"width {width} -> rung {rung} > bucket {bucket}"


def test_serve_cmd_generate_spec_ladder_is_dense_enough_to_avoid_padding():
    """A ladder that merely stays under the bucket can still waste rows: floored powers of two give
    15 then 30, so a 24-token step pads to 30. Mirroring vLLM's own dense default (stride 8) lands
    the common verify widths exactly, which is also why the shipped image never hit the fault."""
    sizes = _sizes(2, ["--max-num-seqs", "256"])
    for width in (24, 192):
        assert min(s for s in sizes if s >= width) == width, (width, sizes)


def test_serve_cmd_generate_capture_sizes_ignore_absent_spec_config():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=[], generate=True)
    cfg = cmd[cmd.index("--compilation-config") + 1]
    assert '"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256]' in cfg


def test_serve_cmd_generate_enforce_eager_opts_out_of_capture():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=["--enforce-eager"], generate=True)
    assert "--compilation-config" not in cmd
    assert cmd.count("--enforce-eager") == 1  # the user's own flag forwards, nothing added


def test_serve_cmd_generate_user_compilation_config_wins():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=["--compilation-config", "{}"], generate=True)
    assert cmd.count("--compilation-config") == 1
    assert "--enforce-eager" not in cmd


def test_serve_cmd_generate_bucket_off_forces_eager(monkeypatch):
    monkeypatch.setenv("EMMY_GEN_DECODE_BUCKET", "0")
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=[], generate=True)
    assert "--enforce-eager" in cmd
    assert "--compilation-config" not in cmd


def _force_moe_probe(monkeypatch):
    monkeypatch.setattr("emmy.commands.serve._is_moe_model", lambda model, vllm_args: True)


def test_serve_cmd_generate_moe_captures_at_size_one(monkeypatch):
    """MoE decode capture is fixed-slot: FULL_DECODE_ONLY with the ladder capped at capture
    size 1 (single-token steps ride the fixed-slot expert path; wider decode steps stay eager
    routed dispatch) — not --enforce-eager. The boot guard in EmmyGenModel validates
    authoritatively against the runner."""
    _force_moe_probe(monkeypatch)
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=[], generate=True)
    assert "--enforce-eager" not in cmd
    cfg = cmd[cmd.index("--compilation-config") + 1]
    assert '"cudagraph_mode": "FULL_DECODE_ONLY"' in cfg
    assert '"cudagraph_capture_sizes": [1]' in cfg


def test_serve_cmd_generate_moe_enforce_eager_forwards(monkeypatch):
    _force_moe_probe(monkeypatch)
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=["--enforce-eager"], generate=True)
    assert cmd.count("--enforce-eager") == 1
    assert "--compilation-config" not in cmd


def test_serve_cmd_generate_moe_user_compilation_config_forwards(monkeypatch):
    # A caller-supplied config forwards untouched; the EmmyGenModel boot guard validates it
    # against the runner (fixed-slot tier present, sizes capped at 1).
    _force_moe_probe(monkeypatch)
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=["--compilation-config", "{}"], generate=True)
    assert cmd.count("--compilation-config") == 1
    assert "--enforce-eager" not in cmd


def test_serve_cmd_generate_moe_bucket_off_forces_eager(monkeypatch):
    _force_moe_probe(monkeypatch)
    monkeypatch.setenv("EMMY_GEN_DECODE_BUCKET", "0")
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=[], generate=True)
    assert "--enforce-eager" in cmd
    assert "--compilation-config" not in cmd


def test_serve_cmd_generate_rejects_incompatible_dtype():
    with pytest.raises(ValueError, match="fp16"):
        build_serve_cmd(MODEL, stock=False, vllm_args=["--dtype", "bfloat16"], generate=True)


def test_serve_cmd_generate_rejects_oversized_batched_tokens():
    with pytest.raises(ValueError, match="dynamic-dim cap"):
        build_serve_cmd(MODEL, stock=False, vllm_args=["--max-num-batched-tokens", "8192"], generate=True)


def test_serve_cmd_generate_batched_tokens_rider_headroom(monkeypatch):
    # An explicit mnbt inside the rider headroom (cap + bucket) is accepted; one past it is not.
    monkeypatch.setenv("EMMY_GEN_DECODE_BUCKET", "32")
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=["--max-num-batched-tokens", "4128"], generate=True)
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "4128"
    with pytest.raises(ValueError, match="dynamic-dim cap"):
        build_serve_cmd(MODEL, stock=False, vllm_args=["--max-num-batched-tokens", "4129"], generate=True)
    # Bucket off -> no split coverage: the default falls back to the bare cap.
    monkeypatch.setenv("EMMY_GEN_DECODE_BUCKET", "0")
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=[], generate=True)
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "4096"


def test_serve_cmd_generate_honors_explicit_batched_tokens():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=["--max-num-batched-tokens", "2048"], generate=True)
    assert cmd.count("--max-num-batched-tokens") == 1  # the user's, no added default
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "2048"


def test_serve_generate_bench_targets_completions(capsys):
    args = _parse(["serve", MODEL, "--generate", "--bench", "--dry-run"])
    handle_serve(args)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2  # serve + bench
    assert "--runner generate" in out[0]
    assert "--backend openai" in out[1] and "/v1/completions" in out[1]
    assert "--random-output-len" in out[1]  # generative bench needs a generation length


def test_health_timeout_default():
    args = _parse(["serve", MODEL, "--bench"])
    assert args.health_timeout == 1800


def test_health_timeout_after_model_is_extracted_not_forwarded(capsys):
    # A fresh-shape first boot can compile past the 1800 s default; --health-timeout
    # widens the --bench /health window. Like every emmy flag it must be extracted
    # from the REMAINDER, not forwarded to vllm serve.
    args = _parse(["serve", MODEL, "--bench", "--dry-run", "--health-timeout", "5400"])
    handle_serve(args)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2
    assert "--health-timeout" not in out[0] and "--health-timeout" not in out[1]
    assert args.health_timeout == 5400
