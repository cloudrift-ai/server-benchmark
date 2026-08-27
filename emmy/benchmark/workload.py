"""Benchmark workload execution."""

import logging
from pathlib import Path

from emmy.deploy.compose import calculate_num_instances
from emmy.recipe.types import Recipe, VllmConfig
from emmy.redact import redact_secrets


def _bench_args(recipe: Recipe) -> list[str]:
    """The vllm bench serve argument list shared by the display string and the
    docker invocation. Embedding recipes target /v1/embeddings via the
    openai-embeddings backend and have no output length (nothing is generated),
    so the generation-only sampling knobs (temperature, ignore_eos) don't apply."""
    bench = recipe.benchmark
    num_instances = calculate_num_instances(recipe)
    port = 8080 if num_instances > 1 else 8000
    args = [
        f"--model {recipe.model_name}",
        "--trust-remote-code",
    ]
    if recipe.is_embedding:
        args += ["--backend openai-embeddings", "--endpoint /v1/embeddings"]
    args += [
        f"--max-concurrency {bench.max_concurrency}",
        f"--num-prompts {bench.num_prompts}",
        f"--random-input-len {bench.random_input_len}",
    ]
    if bench.seed is not None:
        args.append(f"--seed {bench.seed}")
    if bench.num_warmups:
        args.append(f"--num-warmups {bench.num_warmups}")
    if not recipe.is_embedding:
        args.append(f"--random-output-len {bench.random_output_len}")
        if bench.temperature is not None:
            args.append(f"--temperature {bench.temperature}")
        if bench.ignore_eos:
            args.append("--ignore-eos")
    args.append(f"--base-url http://localhost:{port}")
    return args


def build_bench_command(recipe: Recipe) -> str:
    """Build the vllm bench serve command string (without docker wrapper).

    Returns the bench command as a human-readable multi-line string.
    """
    return "vllm bench serve\n" + "\n".join(f"    {a}" for a in _bench_args(recipe))


async def run_benchmark_workload(run_cmd, recipe: Recipe, dry_run=False):
    """Run vllm bench serve on the remote server and return output.

    Returns:
        (success: bool, output: str, stderr: str, bench_command: str)
    """
    bench = recipe.benchmark

    # Warn if input + output lengths risk exceeding context_length (embedding
    # workloads generate nothing — only the input length counts there).
    context_length = recipe.engine.llm.context_length
    request_len = bench.random_input_len + (0 if recipe.is_embedding else bench.random_output_len)
    if context_length is not None and request_len >= context_length:
        logging.getLogger().warning(f"benchmark request length ({request_len}) >= context_length ({context_length})")

    # vllm bench serve is an HTTP client that works against any OpenAI-compatible
    # endpoint, so we always use the vLLM image for benchmarking.
    image = recipe.engine.llm.image
    if recipe.engine.llm.engine_name != "vllm":
        image = VllmConfig().image

    # The ROCm vLLM image crashes on import without GPU devices, even for
    # the pure-HTTP benchmark client.  Pass device flags so it can start.
    is_amd = recipe.deploy.gpu is not None and recipe.deploy.gpu.startswith("AMD")
    device_flags = " --device /dev/kfd:/dev/kfd --device /dev/dri:/dev/dri" if is_amd else ""

    bench_cmd = (
        f"docker run --rm --network host{device_flags} --entrypoint bash {image} -c 'vllm bench serve {' '.join(_bench_args(recipe))}'"
    )

    # benchmark.repeats reruns the identical client workload against one deployed
    # server. The raw output retains one stanza per repeat; a failed repeat fails the task.
    bench_command_str = build_bench_command(recipe)
    outputs = []
    for _ in range(max(1, bench.repeats)):
        rc, output, stderr = await run_cmd(bench_cmd, stream=False, timeout=10800)
        outputs.append(output)
        if rc != 0:
            return False, "\n\n".join(outputs), stderr, bench_command_str
    return True, "\n\n".join(outputs), stderr, bench_command_str


async def capture_server_log(run_cmd, path: Path, *, dry_run: bool = False) -> dict:
    """Preserve the complete serving log without interpreting its contents."""
    if dry_run:
        return {"path": path.name, "status": "dry-run", "exit_code": None}

    rc, stdout, stderr = await run_cmd("docker compose logs --no-color", stream=False, timeout=120)
    content = "\n".join(part for part in (stdout, stderr) if part)
    path.write_text(redact_secrets(content) + ("\n" if content and not content.endswith("\n") else ""), encoding="utf-8")
    return {
        "path": path.name,
        "status": "collected" if rc == 0 else "failed",
        "exit_code": rc,
    }
