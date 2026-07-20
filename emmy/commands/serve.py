"""``emmy serve`` — vLLM embedding serving with emmy kernels, one command.

Wraps ``vllm serve`` with the ``emmy/serving`` plugin boilerplate
(``--runner pooling --enforce-eager --hf-overrides …EmmyEmbedModel…``,
plus ``--max-model-len 4096`` — the dynamic-dim cap — and
``--gpu-memory-utilization=0.9`` unless overridden), and passes any
unrecognized flags through to vLLM verbatim. ``--stock`` drops the
plugin flags for the raw-vLLM baseline, so an A/B is two invocations of the
same command. ``--bench`` turns it into a one-shot benchmark: start the
server, wait for ``/health``, run ``vllm bench serve --backend
openai-embeddings`` against it, print the result table, and shut the server
down.

    emmy serve Qwen/Qwen3-Embedding-0.6B --gpu-memory-utilization 0.8
    emmy serve Qwen/Qwen3-Embedding-0.6B --bench --random-input-len 32
    emmy serve Qwen/Qwen3-Embedding-0.6B --bench --stock     # baseline

Flag routing: emmy's own flags (``--bench``, ``--stock``, the bench
params, ``--dry-run``) are extracted wherever they appear; everything else is
forwarded to ``vllm serve``. Tokens after a literal ``--`` are forwarded
verbatim without extraction (the escape hatch for a hypothetical name clash).
"""

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# The dynamic-dim cap (compiler/trace/dynamic.py DYNAMIC_DIM_MAX) — the plugin
# runner rejects anything larger at startup. Applied to --stock too, so the
# A/B compares both engines at the same max-model-len.
_DEFAULT_MAX_MODEL_LEN = "4096"
_DEFAULT_GPU_MEMORY_UTILIZATION = "0.9"

_PLUGIN_ARGS = ["--enforce-eager", "--hf-overrides", '{"architectures": ["EmmyEmbedModel"]}']

_HEALTH_TIMEOUT_S = 1800  # first boot compiles the whole model; warm cubin cache is much faster


def _add_own_flags(parser, *, suppress_defaults: bool) -> None:
    """The emmy-side flags, declared twice: on the subparser (for --help
    and flags placed before MODEL) and on the post-MODEL re-parse (with
    SUPPRESS defaults, so only flags actually present override)."""

    def d(value):
        return argparse.SUPPRESS if suppress_defaults else value

    parser.add_argument(
        "--stock", action="store_true", default=d(False), help="Serve stock vLLM kernels instead of the emmy plugin (A/B baseline)."
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        default=d(False),
        help="Serve a generative (chat) model via EmmyGenModel (--runner generate, fp16) instead of embeddings.",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        default=d(False),
        help="Start the server, run `vllm bench serve` against it, print results, shut down.",
    )
    parser.add_argument("--max-concurrency", type=int, default=d(32), help="Bench client concurrency (with --bench).")
    parser.add_argument("--num-prompts", type=int, default=d(256), help="Bench request count (with --bench).")
    parser.add_argument("--random-input-len", type=int, default=d(512), help="Bench tokens per request (with --bench).")
    parser.add_argument(
        "--random-output-len", type=int, default=d(128), help="Bench tokens generated per request (with --bench --generate)."
    )
    parser.add_argument(
        "--bench-seed", type=int, default=d(0), help="Bench prompt-sampling seed (with --bench; `--seed` itself forwards to vllm serve)."
    )
    parser.add_argument("--dry-run", action="store_true", default=d(False), help="Print the vllm command(s) without running.")


def register_serve_command(subparsers):
    parser = subparsers.add_parser(
        "serve",
        help="Serve an embedding model via vLLM with emmy-compiled kernels (optionally benchmark it)",
        description="Wraps `vllm serve` with the emmy plugin flags; unrecognized flags pass through to vLLM "
        "(tokens after a literal `--` pass through verbatim).",
        allow_abbrev=False,
    )
    parser.add_argument("model", help="HuggingFace model ID (e.g., Qwen/Qwen3-Embedding-0.6B)")
    _add_own_flags(parser, suppress_defaults=False)
    parser.add_argument(
        "vllm_args",
        nargs=argparse.REMAINDER,
        metavar="VLLM_ARGS",
        help="Extra args forwarded to `vllm serve` (e.g. --gpu-memory-utilization 0.8 --port 8123).",
    )
    parser.set_defaults(func=handle_serve)


def _split_own_flags(args) -> list[str]:
    """argparse REMAINDER swallows *everything* after MODEL — including our own
    flags. Re-parse the remainder: extract emmy flags into ``args``
    (SUPPRESS defaults → only flags actually present override), forward the
    rest; anything after a literal ``--`` forwards without extraction."""
    rem = list(args.vllm_args)
    verbatim: list[str] = []
    if "--" in rem:
        i = rem.index("--")
        rem, verbatim = rem[:i], rem[i + 1 :]
    own = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    _add_own_flags(own, suppress_defaults=True)
    _, passthrough = own.parse_known_args(rem, namespace=args)
    return passthrough + verbatim


def _has_flag(vllm_args: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(f"{flag}=") for a in vllm_args)


def _flag_value(vllm_args: list[str], flag: str, default: str) -> str:
    for i, a in enumerate(vllm_args):
        if a == flag and i + 1 < len(vllm_args):
            return vllm_args[i + 1]
        if a.startswith(f"{flag}="):
            return a.split("=", 1)[1]
    return default


def _gen_graph_args(vllm_args: list[str]) -> list[str]:
    """The eager/capture flags for emmy generative serving. DEFAULT is whole-step decode
    capture: a compilation-config asking for FULL_DECODE_ONLY graphs (full cudagraphs need no
    torch.compile — vLLM wraps the model in its ``CUDAGraphWrapper``) with capture sizes
    clamped to the decode bucket — sizes above the bucket would capture the model's symbolic
    fallback path, whose per-layer host numpy hops abort a stream capture. Opt out with
    vLLM's own ``--enforce-eager`` (forwards as-is); a caller-supplied ``--compilation-config``
    also wins over ours. With the decode bucket off (``EMMY_GEN_DECODE_BUCKET=0``) nothing is
    capturable, so eager is forced."""
    from emmy import config as emmy_config  # noqa: PLC0415

    if _has_flag(vllm_args, "--enforce-eager") or _has_flag(vllm_args, "--compilation-config"):
        return []  # the caller decided; forward theirs untouched
    bucket = emmy_config.gen_decode_bucket()
    if bucket <= 0:
        logger.warning("decode bucket is off (EMMY_GEN_DECODE_BUCKET=0) — the symbolic decode path is not capturable; serving eager")
        return ["--enforce-eager"]
    sizes = [s for s in (1, 2, 4, 8, 16, 32, 64) if s < bucket] + [bucket]
    cfg = f'{{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": {sizes}}}'
    return ["--compilation-config", cfg]


def build_serve_cmd(model: str, *, stock: bool, vllm_args: list[str], generate: bool = False) -> list[str]:
    cmd = ["vllm", "serve", model, "--runner", "generate" if generate else "pooling"]
    if not stock and generate:
        cmd += _gen_graph_args(vllm_args)
        cmd += ["--hf-overrides", '{"architectures": ["EmmyGenModel"]}']
        # Force fp16 across the emmy↔vLLM seam: vLLM defaults --dtype auto → bf16 for a
        # bf16 checkpoint, but the emmy trunk emits fp16. Reject an incompatible override.
        if _has_flag(vllm_args, "--dtype"):
            dt = _flag_value(vllm_args, "--dtype", "")
            if dt not in ("float16", "half", "fp16"):
                raise ValueError(f"generative serving requires fp16; --dtype {dt!r} is incompatible (use --dtype float16)")
        else:
            cmd += ["--dtype", "float16"]
        # The flattened width (sum of newly-scheduled tokens per step) must stay within
        # DYNAMIC_DIM_MAX (= _DEFAULT_MAX_MODEL_LEN). vLLM's default max_num_batched_tokens
        # (e.g. 8192 on a big GPU) would trip the model's startup bound, so cap it here.
        if _has_flag(vllm_args, "--max-num-batched-tokens"):
            mnbt = _flag_value(vllm_args, "--max-num-batched-tokens", "")
            if mnbt.isdigit() and int(mnbt) > int(_DEFAULT_MAX_MODEL_LEN):
                raise ValueError(f"--max-num-batched-tokens {mnbt} exceeds the dynamic-dim cap ({_DEFAULT_MAX_MODEL_LEN}); use it or lower")
        else:
            cmd += ["--max-num-batched-tokens", _DEFAULT_MAX_MODEL_LEN]
    elif not stock:
        cmd += _PLUGIN_ARGS
    if not _has_flag(vllm_args, "--max-model-len"):
        cmd += ["--max-model-len", _DEFAULT_MAX_MODEL_LEN]
    if not _has_flag(vllm_args, "--gpu-memory-utilization"):
        cmd += [f"--gpu-memory-utilization={_DEFAULT_GPU_MEMORY_UTILIZATION}"]
    return cmd + vllm_args


def build_bench_cmd(
    model: str,
    *,
    port: str,
    max_concurrency: int,
    num_prompts: int,
    random_input_len: int,
    seed: int,
    generate: bool = False,
    random_output_len: int = 128,
) -> list[str]:
    # Generative bench drives /v1/completions (token throughput, needs an output length);
    # embedding bench drives /v1/embeddings (no generation).
    backend, endpoint = ("openai", "/v1/completions") if generate else ("openai-embeddings", "/v1/embeddings")
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        model,
        "--backend",
        backend,
        "--endpoint",
        endpoint,
        "--base-url",
        f"http://localhost:{port}",
        "--max-concurrency",
        str(max_concurrency),
        "--num-prompts",
        str(num_prompts),
        "--random-input-len",
        str(random_input_len),
        "--seed",
        str(seed),
    ]
    if generate:
        cmd += ["--random-output-len", str(random_output_len)]
    return cmd


def _vllm_bin() -> str:
    """The vllm console script from this interpreter's environment (so the
    plugin entry point installed alongside emmy is the one that loads),
    falling back to PATH."""
    candidate = Path(sys.executable).parent / "vllm"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("vllm")
    if found:
        return found
    logger.error("vllm not found — install the serving extra: pip install -e '.[compile,serving]'")
    sys.exit(1)


def _child_env() -> dict:
    """Environment for the vLLM child with this interpreter's bin dir prepended to ``PATH``.
    Invoking ``./venv/bin/emmy`` by absolute path does NOT put the venv's bin on ``PATH``, so
    the generative server's inductor-compile subprocess dies with ``FileNotFoundError: ninja``
    (ninja is pip-installed into that same bin). Prepending it — what venv activation would do —
    makes the venv's ``ninja`` / console scripts resolvable to the child. Idempotent: a bin dir
    already at the front of ``PATH`` is not duplicated."""
    env = dict(os.environ)
    bin_dir = str(Path(sys.executable).parent)
    parts = env.get("PATH", "").split(os.pathsep)
    if parts[:1] != [bin_dir]:
        env["PATH"] = os.pathsep.join([bin_dir, *parts]) if parts != [""] else bin_dir
    return env


def handle_serve(args):
    vllm_args = _split_own_flags(args)  # re-parses own flags placed after MODEL into args
    serve_cmd = build_serve_cmd(args.model, stock=args.stock, vllm_args=vllm_args, generate=args.generate)
    port = _flag_value(vllm_args, "--port", "8000")
    bench_cmd = build_bench_cmd(
        args.model,
        port=port,
        max_concurrency=args.max_concurrency,
        num_prompts=args.num_prompts,
        random_input_len=args.random_input_len,
        seed=args.bench_seed,
        generate=args.generate,
        random_output_len=args.random_output_len,
    )

    if args.dry_run:
        # shlex.join so the printed line is shell-pastable (the JSON in
        # --hf-overrides needs quoting; the real execution uses argv lists).
        print(shlex.join(serve_cmd))
        if args.bench:
            print(shlex.join(bench_cmd))
        return

    vllm = _vllm_bin()
    serve_cmd[0] = vllm
    env = _child_env()
    if not args.bench:
        # Plain serving: replace this process so signals/Ctrl-C reach vLLM directly.
        os.execve(vllm, serve_cmd, env)

    bench_cmd[0] = vllm
    _serve_and_bench(serve_cmd, bench_cmd, port, env=env)


def _serve_and_bench(serve_cmd: list[str], bench_cmd: list[str], port: str, *, env: dict | None = None) -> None:
    log_file = tempfile.NamedTemporaryFile(mode="wb", prefix="emmy-serve-", suffix=".log", delete=False)
    logger.info("Starting server (logs: %s)...", log_file.name)
    logger.info("  %s", shlex.join(serve_cmd))
    server = subprocess.Popen(serve_cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    try:
        _wait_for_health(server, port, log_file.name)
        logger.info("Server healthy — running benchmark...")
        logger.info("  %s", shlex.join(bench_cmd))
        rc = subprocess.run(bench_cmd, env=env).returncode
    finally:
        logger.info("Shutting down server...")
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
        log_file.close()
    if rc != 0:
        logger.error("Benchmark failed (rc=%d); server logs: %s", rc, log_file.name)
        sys.exit(rc)


def _wait_for_health(server: subprocess.Popen, port: str, log_path: str) -> None:
    """Poll /health until the server answers; fail fast if it exits first
    (first boot may compile the whole model — be patient)."""
    import httpx

    deadline = time.monotonic() + _HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            tail = "".join(open(log_path, errors="replace").readlines()[-15:])
            logger.error("Server exited during startup (rc=%d). Log tail (%s):\n%s", server.returncode, log_path, tail)
            sys.exit(1)
        try:
            if httpx.get(f"http://localhost:{port}/health", timeout=3).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    logger.error("Server did not become healthy within %ds; logs: %s", _HEALTH_TIMEOUT_S, log_path)
    server.terminate()
    sys.exit(1)
