"""Deploy orchestration: run_deploy, run_teardown, deploy, teardown."""

import asyncio
import json
import logging
import math

from emmy.deploy.compose import (
    calculate_num_instances,
    generate_compose,
    generate_nginx_conf,
)
from emmy.deploy.log_phases import decompose_model_load, parse_engine_load_phases
from emmy.deploy.params import DeployParams
from emmy.provisioning.ssh_transport import make_run_cmd, make_write_file
from emmy.recipe.types import Recipe
from emmy.timing import (
    PHASE_IMAGE_PULL,
    PHASE_MODEL_DOWNLOAD,
    PHASE_MODEL_LOAD_AND_WARMUP,
    PHASE_SMOKE_TEST,
    PhaseTimer,
)

logger = logging.getLogger(__name__)


async def _baked_hf_cache(run_cmd, image):
    """The image's own HF cache directory, when it ships one — else None.

    A prebuilt per-model serving image (``docker/vllm-emmy-serve/``) bakes the model
    snapshot under its own HF_HOME and sets HF_HUB_OFFLINE=1, so it never reaches the
    hub. That pair is the image declaring itself self-contained: pointing HF_HOME at a
    host cache instead hides the baked snapshot while offline mode stays on, which makes
    the download step fail outright — and would re-fetch the full weights if it didn't.
    """
    fmt = "{{range .Config.Env}}{{println .}}{{end}}"
    # stream=False is what makes run_cmd return stdout rather than pass it through.
    rc, out, _ = await run_cmd(f"docker image inspect {image} --format '{fmt}'", stream=False, timeout=60)
    if rc != 0 or not out:
        return None
    env = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return env.get("HF_HOME") if env.get("HF_HUB_OFFLINE") == "1" else None


async def run_deploy(
    run_cmd,
    write_file,
    recipe: Recipe,
    model_dir,
    hf_token,
    host,
    dry_run=False,
    gpu_device_ids=None,
    port_mappings=None,
    timer: PhaseTimer | None = None,
    check_smoke_output: bool = True,
):
    """Shared deploy orchestration.

    Args:
        run_cmd: async callable(command, stream=True, timeout=600) -> (returncode, stdout, stderr)
        write_file: async callable(path, content) -> None - writes file to target
        recipe: resolved Recipe dataclass
        model_dir: model cache directory path
        hf_token: HuggingFace token
        host: hostname/IP for endpoint display
        dry_run: if True, skip sleep in health polling
        gpu_device_ids: optional list of GPU device IDs to restrict visibility
        port_mappings: optional list of (internal, external) port tuples
        timer: optional PhaseTimer; deploy step durations are recorded into it. A
            throwaway timer is used when None, so the timing log lines still print.
        check_smoke_output: whether a valid smoke response must also satisfy the
            model-specific deployment check. Benchmark orchestration disables this
            semantic judgment and uses the probe only for API readiness.
    """
    timer = timer or PhaseTimer()
    num_instances = calculate_num_instances(recipe)

    model_name = recipe.model_name
    image = recipe.engine.llm.image

    # Generate and write compose file
    compose_content = generate_compose(recipe, model_dir, hf_token, num_instances=num_instances, gpu_device_ids=gpu_device_ids)
    await write_file("docker-compose.yaml", compose_content)

    # Generate and write nginx config if multi-instance
    if num_instances > 1:
        nginx_content = generate_nginx_conf(num_instances, engine=recipe.engine.llm.engine_name)
        await write_file("nginx.conf", nginx_content)

    internal_port = 8080 if num_instances > 1 else 8000

    # Resolve external port from port mappings (e.g. CloudRift NAT)
    port_map = dict(port_mappings or [])
    external_port = port_map.get(internal_port, internal_port)

    # Step 1: Pull images. --ignore-pull-failures keeps a locally-built image
    # usable before it's pushed to a registry (e.g. testing a fresh
    # vllm-emmy build with `bench --local`), but compose's exit code is
    # unreliable across versions (non-zero for a registry-lookup miss on
    # some, zero on others) — so ignore it and enforce the actual
    # invariant instead: after the pull, every image the compose file
    # references must exist locally. Missing images abort here with their
    # names (instead of a confusing failure later); a never-pulled image
    # that exists locally proceeds with a stale-copy warning.
    logger.info("Pulling images...")
    async with timer.ameasure(PHASE_IMAGE_PULL):
        rc, _, _ = await run_cmd("docker compose pull --ignore-pull-failures", timeout=1800, log_output=True)
    # stream=False, else run_cmd passes stdout through and returns "" — which left this
    # guard iterating an empty list, so a genuinely missing image fell through to the
    # confusing later failure the check exists to replace.
    _, images_out, _ = await run_cmd("docker compose config --images", stream=False, timeout=60)
    missing = []
    for image in images_out.split():
        rc_i, _, _ = await run_cmd(f"docker image inspect {image}", timeout=60)
        if rc_i != 0:
            missing.append(image)
    if missing:
        logger.error(f"Images neither pullable nor available locally: {', '.join(missing)}")
        return False
    if rc != 0:
        logger.warning("Image pull reported failures, but every image exists locally — proceeding (local copies may be stale)")

    # Step 2: Model weights. An image that ships its own snapshot needs none of this —
    # and cannot do it, since it runs offline by construction. Defer to its cache and
    # rewrite the compose file so the service keeps the baked HF_HOME too (the first
    # write had to happen before the pull, which is what `docker compose pull` reads).
    baked_hf_home = await _baked_hf_cache(run_cmd, image)
    if baked_hf_home:
        logger.info(f"Image ships its model cache at {baked_hf_home} (offline) — skipping download")
        compose_content = generate_compose(
            recipe, model_dir, hf_token, num_instances=num_instances, gpu_device_ids=gpu_device_ids, baked_hf_home=baked_hf_home
        )
        await write_file("docker-compose.yaml", compose_content)
    else:
        logger.info(f"Downloading model {model_name}...")
        revision_arg = f" --revision {recipe.model.revision}" if recipe.model.revision else ""
        dl_cmd = (
            f"docker run --rm"
            f" -e HUGGING_FACE_HUB_TOKEN={hf_token}"
            f" -e HF_HOME={model_dir}"
            f" -v {model_dir}:{model_dir}"
            f" --entrypoint bash"
            f" {image}"
            f" -c 'HF_HUB_ENABLE_HF_TRANSFER=1 hf download {model_name}{revision_arg}'"
        )
        async with timer.ameasure(PHASE_MODEL_DOWNLOAD):
            rc, _, _ = await run_cmd(dl_cmd, timeout=7200, log_output=True)
        if rc != 0:
            logger.error("Failed to download model")
            return False

    # Step 3: Clean up old containers
    logger.info("Cleaning up old containers...")
    await run_cmd("docker compose down", timeout=300, log_output=True)

    # Step 4: Start services (blocks until /health passes, so this window covers
    # container start + weight load into GPU + CUDA graph capture + warmup)
    logger.info("Starting services...")
    async with timer.ameasure(PHASE_MODEL_LOAD_AND_WARMUP):
        rc, _, _ = await run_cmd("docker compose up -d --wait --wait-timeout 3600", timeout=3600, log_output=True)
    if rc != 0:
        logger.error("Failed to start services")
        logger.error("Container logs:")
        await run_cmd("docker compose logs --tail=100", timeout=60, log_output=True)
        return False

    # Best-effort: break the warmup window into startup / weights_load / torch_compile /
    # engine_warmup / cuda_graph_capture by scraping the container logs. The leaves sum to
    # model_load_and_warmup; absent/degraded to a single `other` when the format differs.
    if not dry_run:
        rc_logs, logs, _ = await run_cmd("docker compose logs --no-color", stream=False, timeout=120)
        if rc_logs == 0 and logs:
            raw = parse_engine_load_phases(logs, recipe.engine.llm.engine_name)
            mlw = timer.phases.get(PHASE_MODEL_LOAD_AND_WARMUP, 0.0)
            for name, seconds in decompose_model_load(raw, mlw).items():
                timer.record(name, seconds)

    # Step 5: Poll health
    logger.info("Waiting for health check...")
    health_url = f"http://localhost:{internal_port}/health"
    timeout = 3600  # 60 minutes
    interval = 10
    elapsed = 0
    while elapsed < timeout:
        rc, _, _ = await run_cmd(f"curl -sf {health_url}", stream=False, timeout=30, log_output=True)
        if rc == 0:
            break
        if dry_run:
            break
        await asyncio.sleep(interval)
        elapsed += interval
    else:
        logger.error(f"Health check timed out after {timeout}s")
        return False

    # Step 6: Print endpoint info
    status = "dry-run (not deployed)" if dry_run else "deployed"
    logger.info(f"\nEndpoint: http://{host}:{external_port}/v1")
    logger.info(f"Model: {model_name}")
    logger.info(f"Instances: {num_instances}")
    logger.info(f"Status: {status}")

    # Step 7: Probe inference (retry — first request may be slow due to warmup).
    # Standalone deploy checks model-specific content; benchmark callers request
    # transport readiness only and retain the response for later review.
    if not dry_run:
        logger.info("\nRunning smoke test...")
        async with timer.ameasure(PHASE_SMOKE_TEST):
            if recipe.is_embedding:
                smoke_cmd = (
                    f"curl --fail-with-body -s http://localhost:{internal_port}/v1/embeddings"
                    f" -H 'Content-Type: application/json'"
                    f''' -d '{{"model":"{model_name}","input":"What is 2+2?"}}' '''
                )
            elif recipe.model.smoke_test == "completion":
                prompt = "2 + 2 ="
                smoke_cmd = (
                    f"curl --fail-with-body -s http://localhost:{internal_port}/v1/completions"
                    f" -H 'Content-Type: application/json'"
                    f''' -d '{{"model":"{model_name}","prompt":"{prompt}","max_tokens":16,"temperature":0}}' '''
                )
            else:
                prompt = "What is 2+2? Answer with just the number."
                smoke_cmd = (
                    f"curl --fail-with-body -s http://localhost:{internal_port}/v1/chat/completions"
                    f" -H 'Content-Type: application/json'"
                    f''' -d '{{"model":"{model_name}","messages":[{{"role":"user","content":"{prompt}"}}],"max_tokens":128}}' '''
                )
            check = _smoke_response_check(recipe, check_smoke_output=check_smoke_output)
            smoke_timeout = 600
            smoke_interval = 10
            deadline = asyncio.get_event_loop().time() + smoke_timeout
            while asyncio.get_event_loop().time() < deadline:
                rc, stdout, _ = await run_cmd(smoke_cmd, stream=False, timeout=180)
                if rc != 0 or not stdout.strip():
                    # Server not ready yet, keep retrying
                    await asyncio.sleep(smoke_interval)
                    continue
                if not check_smoke_output:
                    logger.info("Smoke response: %s", stdout)
                verdict, detail = check(stdout)
                if verdict == "retry":
                    # Malformed/empty response, server may still be starting
                    await asyncio.sleep(smoke_interval)
                    continue
                if verdict == "pass":
                    logger.info("Smoke test passed.")
                    break
                # Valid response, wrong content — broken model
                logger.error(f"Smoke test failed: {detail}")
                logger.error("Container logs:")
                await run_cmd("docker compose logs --tail=100", timeout=60, log_output=True)
                return False
            else:
                logger.error(f"Smoke test timed out after {smoke_timeout}s. The endpoint is not ready.")
                logger.error("Container logs:")
                await run_cmd("docker compose logs --tail=100", timeout=60, log_output=True)
                return False

    # Print curl example
    logger.info("\nExample curl:")
    if recipe.is_embedding:
        logger.info(
            f"  curl http://{host}:{external_port}/v1/embeddings \\\n"
            f"    -H 'Content-Type: application/json' \\\n"
            f"    -d '{{\n"
            f'      "model": "{model_name}",\n'
            f'      "input": "Hello"\n'
            f"    }}'"
        )
    elif recipe.model.smoke_test == "completion":
        logger.info(
            f"  curl http://{host}:{external_port}/v1/completions \\\n"
            f"    -H 'Content-Type: application/json' \\\n"
            f"    -d '{{\n"
            f'      "model": "{model_name}",\n'
            f'      "prompt": "Hello",\n'
            f'      "max_tokens": 64\n'
            f"    }}'"
        )
    else:
        logger.info(
            f"  curl http://{host}:{external_port}/v1/chat/completions \\\n"
            f"    -H 'Content-Type: application/json' \\\n"
            f"    -d '{{\n"
            f'      "model": "{model_name}",\n'
            f'      "messages": [{{"role": "user", "content": "Hello"}}],\n'
            f'      "max_tokens": 64\n'
            f"    }}'"
        )

    return True


def _check_chat_response(stdout: str) -> tuple[str, str]:
    """Validate a /v1/chat/completions smoke response.

    Returns ``("pass" | "fail" | "retry", detail)`` — ``retry`` means the
    response was malformed/empty (server may still be starting)."""
    try:
        body = json.loads(stdout)
        message = body["choices"][0]["message"]
        answer = message.get("content") or message.get("reasoning_content") or message.get("reasoning") or ""
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return "retry", ""
    if not answer:
        return "retry", ""
    if "4" in answer:
        return "pass", ""
    return "fail", f"model returned wrong answer: {answer!r}"


def _check_readiness_response(stdout: str) -> tuple[str, str]:
    """Accept any nonempty JSON API response without interpreting model content."""
    try:
        body = json.loads(stdout)
    except json.JSONDecodeError:
        return "retry", ""
    return ("pass", "") if body else ("retry", "")


def _smoke_response_check(recipe: Recipe, *, check_smoke_output: bool):
    """Select semantic deployment checking or protocol-only benchmark readiness."""
    if not check_smoke_output:
        return _check_readiness_response
    if recipe.is_embedding:
        return _check_embedding_response
    if recipe.model.smoke_test == "completion":
        return _check_completion_response
    return _check_chat_response


def _check_completion_response(stdout: str) -> tuple[str, str]:
    """Validate the base-model completion smoke response."""
    try:
        answer = json.loads(stdout)["choices"][0]["text"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return "retry", ""
    if not answer:
        return "retry", ""
    if "4" in answer:
        return "pass", ""
    return "fail", f"model returned wrong answer: {answer!r}"


def _check_embedding_response(stdout: str) -> tuple[str, str]:
    """Validate a /v1/embeddings smoke response: a non-empty vector of finite
    floats with L2 norm ≈ 1 (the pooler normalizes; garbage/NaN models fail)."""
    try:
        body = json.loads(stdout)
        vec = body["data"][0]["embedding"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return "retry", ""
    if not isinstance(vec, list) or not vec:
        return "retry", ""
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vec):
        return "fail", "embedding contains non-finite values"
    norm = math.sqrt(sum(v * v for v in vec))
    if not 0.9 <= norm <= 1.1:
        return "fail", f"embedding L2 norm {norm:.4f} outside [0.9, 1.1] (dim={len(vec)})"
    return "pass", ""


async def run_teardown(run_cmd):
    """Tear down: docker compose down."""
    logger.info("Tearing down...")
    rc, _, _ = await run_cmd("docker compose down", timeout=300, log_output=True)
    if rc == 0:
        logger.info("Teardown complete.")
    else:
        logger.error("Teardown failed.")
    return rc == 0


async def deploy(params: DeployParams, timer: PhaseTimer | None = None, *, check_smoke_output: bool = True) -> bool:
    """Deploy a recipe to a server via SSH. Single entry point."""
    run_cmd = make_run_cmd(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)
    write_file = make_write_file(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)
    host = params.server.split("@")[-1] if "@" in params.server else params.server
    return await run_deploy(
        run_cmd,
        write_file,
        params.recipe,
        params.model_dir,
        params.hf_token,
        host,
        params.dry_run,
        gpu_device_ids=params.gpu_device_ids,
        port_mappings=params.port_mappings,
        timer=timer,
        check_smoke_output=check_smoke_output,
    )


async def teardown(params: DeployParams) -> bool:
    """Teardown containers on a server."""
    run_cmd = make_run_cmd(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)
    return await run_teardown(run_cmd)
