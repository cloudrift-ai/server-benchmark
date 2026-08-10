#!/usr/bin/env python3
"""Direct ExLlamaV3 v1.4.1 latency/throughput runner for Laguna S 2.1 EXL3."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import inspect
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

MODEL_ID = "cloudriftai/Laguna-S-2.1-exl3"
EXLLAMA_COMMIT = "4f8ad0121f483ba66a5336244a4c3b6d7210385e"
EXLLAMA_VERSION = "1.4.1+cu128.torch2.10.0"
TORCH_VERSION = "2.10.0+cu128"
INPUT_TOKENS = 512
OUTPUT_TOKENS = 128
CONCURRENCY_GRID = (1, 4, 8)
PROMPTS_PER_RUN = {1: 8, 4: 24, 8: 48}
REPEATS = 3
PAGE_SIZE = 256


class BenchError(RuntimeError):
    pass


def command(*args: str) -> str:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise BenchError(f"command failed: {' '.join(args)}\n{exc.stderr}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise BenchError("cannot summarize an empty metric")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in (
        "exllamav3",
        "torch",
        "huggingface_hub",
        "safetensors",
        "tokenizers",
        "numpy",
        "flash-linear-attention",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_metrics_or_error(model) -> dict:
    """Keep diagnostic-only metrics from invalidating a successful native load."""
    try:
        value = model.get_load_metrics()
    except AttributeError as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "value": json.loads(json.dumps(value, default=str))}


def verify_pins(source: Path, revision: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise BenchError("checkpoint identity must be exactly 40 lowercase hexadecimal characters")
    source = source.resolve()
    if command("git", "-C", str(source), "rev-parse", "HEAD") != EXLLAMA_COMMIT:
        raise BenchError(f"ExLlamaV3 checkout is not pinned commit {EXLLAMA_COMMIT}")
    if command("git", "-C", str(source), "status", "--porcelain"):
        raise BenchError("ExLlamaV3 checkout is dirty")
    versions = package_versions()
    if versions["exllamav3"] != EXLLAMA_VERSION:
        raise BenchError(f"exllamav3={versions['exllamav3']!r}, expected {EXLLAMA_VERSION!r}")
    if versions["torch"] != TORCH_VERSION:
        raise BenchError(f"torch={versions['torch']!r}, expected {TORCH_VERSION!r}")
    return versions


def validate_native_concurrency(source: Path) -> dict:
    """Fail closed unless the installed native scheduler matches the pinned source semantics."""
    import exllamav3.generator.generator as generator_module
    import exllamav3.generator.job as job_module
    from exllamav3.generator import Generator, Job

    checks = {}
    for relative, installed in (
        (Path("exllamav3/generator/generator.py"), Path(generator_module.__file__)),
        (Path("exllamav3/generator/job.py"), Path(job_module.__file__)),
    ):
        pinned = source / relative
        if not pinned.is_file() or sha256(pinned) != sha256(installed):
            raise BenchError(f"installed runtime source does not match pinned checkout: {relative}")
        checks[relative.as_posix()] = sha256(installed)

    enqueue_source = inspect.getsource(Generator.enqueue)
    iterate_source = inspect.getsource(Generator.iterate)
    job_init_source = inspect.getsource(Job.__init__)
    job_source = inspect.getsource(Job.prepare_for_queue)
    required_fragments = {
        "enqueue_list": "if isinstance(job, list):",
        "pending_queue": "self.pending_jobs.append(job)",
        "active_job_batch": "for job in self.active_jobs:",
        "per_job_enqueue_time": "self.time_enqueue = time.time()",
        "exact_output_limit": "self.max_new_tokens = max_new_tokens - 1 or 1",
    }
    sources = {
        "enqueue_list": enqueue_source,
        "pending_queue": enqueue_source,
        "active_job_batch": iterate_source,
        "per_job_enqueue_time": job_source,
        "exact_output_limit": job_init_source,
    }
    missing = [name for name, fragment in required_fragments.items() if fragment not in sources[name]]
    if missing:
        raise BenchError("native concurrency semantics could not be validated; refusing c=4/c=8 labels: " + ", ".join(missing))
    example = source / "examples" / "dynamic_gen.py"
    if not example.is_file() or "generator.enqueue(jobs)" not in example.read_text(encoding="utf-8"):
        raise BenchError("pinned native multi-job example is absent; refusing concurrency benchmark")
    return {
        "validated": True,
        "method": "Generator.enqueue(list[Job]) before first Generator.iterate()",
        "source_sha256": checks,
        "max_batch_size": max(CONCURRENCY_GRID),
    }


def checkpoint_manifest(model_dir: Path, revision: str) -> dict:
    model_dir = model_dir.resolve()
    if model_dir.name != revision:
        raise BenchError(f"snapshot path {model_dir} does not match checkpoint identity {revision}")
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    sidecar_path = model_dir / "quantization_config.json"
    for path in (config_path, index_path, sidecar_path):
        if not path.is_file():
            raise BenchError(f"checkpoint file missing: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    config_quant = config.get("quantization_config", {})
    if config_quant.get("quant_method") != "exl3" or sidecar.get("quant_method") != "exl3":
        raise BenchError("checkpoint is not consistently EXL3")
    bits = float(config_quant.get("bits"))
    if not 0 < bits <= 2.00 or bits != float(sidecar.get("bits")):
        raise BenchError(f"invalid or mismatched EXL3 bitrate: {bits}, {sidecar.get('bits')}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    if not weight_map:
        raise BenchError("checkpoint index is empty")
    missing_biases = [
        f"model.layers.{layer}.mlp.experts.e_score_correction_bias"
        for layer in range(1, 48)
        if f"model.layers.{layer}.mlp.experts.e_score_correction_bias" not in weight_map
    ]
    if missing_biases:
        raise BenchError(f"checkpoint lacks Laguna correction biases: {missing_biases}")
    shards = sorted(set(weight_map.values()))
    missing_shards = [name for name in shards if not (model_dir / name).is_file()]
    if missing_shards:
        raise BenchError(f"checkpoint has missing shards: {missing_shards}")
    return {
        "model_id": MODEL_ID,
        "revision": revision,
        "snapshot": str(model_dir),
        "actual_bits": bits,
        "config_sha256": sha256(config_path),
        "index_sha256": sha256(index_path),
        "sidecar_sha256": sha256(sidecar_path),
        "indexed_tensors": len(weight_map),
        "shards": len(shards),
        "shard_bytes": sum((model_dir / name).stat().st_size for name in shards),
        "correction_biases": 47,
    }


def assert_gpu_idle() -> dict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    selectors = [item.strip() for item in visible.split(",") if item.strip()]
    if visible and len(selectors) != 1:
        raise BenchError(f"CUDA_VISIBLE_DEVICES must select exactly one RTX 5090, found {visible!r}")
    selector_args = ["-i", selectors[0]] if selectors else []
    row = command(
        "nvidia-smi",
        *selector_args,
        "--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu,compute_mode,driver_version",
        "--format=csv,noheader,nounits",
    ).splitlines()
    if not selectors:
        matches = [line for line in row if line.split(",", 1)[0].strip() == "NVIDIA GeForce RTX 5090"]
        if len(row) != 1 or len(matches) != 1:
            raise BenchError("multiple GPUs are installed; set CUDA_VISIBLE_DEVICES to exactly one RTX 5090")
        row = matches
    if len(row) != 1:
        raise BenchError(f"expected exactly one selected benchmark GPU, found {len(row)}")
    fields = [field.strip() for field in row[0].split(",")]
    if len(fields) != 7:
        raise BenchError(f"unexpected nvidia-smi row: {row[0]}")
    name, uuid, total, used, utilization, compute_mode, driver = fields
    if name != "NVIDIA GeForce RTX 5090" or int(total) < 32000:
        raise BenchError(f"benchmark requires one 32 GB RTX 5090, found {name}, {total} MiB")
    # The local workstation drives its display from the benchmark GPU, which leaves a
    # stable sub-GiB graphics allocation even when no CUDA compute process exists.
    if int(used) > 1024 or int(utilization) != 0 or compute_mode != "Default":
        raise BenchError(f"GPU is not idle/default: used={used} MiB, util={utilization}%, mode={compute_mode}")
    processes = command(
        "nvidia-smi",
        *selector_args,
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    if processes.strip():
        raise BenchError(f"GPU has active compute processes:\n{processes}")
    return {
        "name": name,
        "uuid": uuid,
        "memory_total_mib": int(total),
        "memory_used_preflight_mib": int(used),
        "driver": driver,
    }


def make_prompt(tokenizer, request_id: int):
    # The unique prefix prevents page-cache reuse between requests while the fixed body keeps
    # tokenization deterministic. Tokenization occurs outside measured intervals.
    text = (
        f"Native benchmark request {request_id:08d}. Analyze this deterministic sequence: "
        + "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda. " * 160
    )
    input_ids = tokenizer.encode(text, add_bos=False)
    if input_ids.shape != (1, input_ids.shape[-1]) or input_ids.shape[-1] < INPUT_TOKENS:
        raise BenchError(f"could not build {INPUT_TOKENS}-token deterministic prompt")
    return input_ids[:, :INPUT_TOKENS].contiguous()


def tensor_hash(tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_wave(generator, tokenizer, concurrency: int, request_base: int, torch) -> tuple[list[dict], dict]:
    from exllamav3.generator import Job
    from exllamav3.generator.sampler import GreedySampler

    prompts = [make_prompt(tokenizer, request_base + offset) for offset in range(concurrency)]
    jobs = [
        Job(
            input_ids=prompt,
            # v1.4.1 Job stores this limit as requested-1; +1 produces exactly OUTPUT_TOKENS.
            max_new_tokens=OUTPUT_TOKENS + 1,
            min_new_tokens=OUTPUT_TOKENS,
            stop_conditions=[],
            sampler=GreedySampler(),
            seed=42 + request_base + offset,
            identifier=request_base + offset,
        )
        for offset, prompt in enumerate(prompts)
    ]
    torch.cuda.synchronize()
    started = time.perf_counter()
    serials = generator.enqueue(jobs)
    if not isinstance(serials, list) or len(serials) != concurrency:
        raise BenchError("native list enqueue did not return one serial per concurrent request")
    if generator.num_pending_jobs() != concurrency:
        raise BenchError("not all concurrent jobs were pending before the first native iterate()")

    eos_by_serial = {}
    output_ids = {serial: [] for serial in serials}
    first_started_serials = set()
    max_active = 0
    peak_device_used = 0
    deadline = time.monotonic() + 1800
    first_iteration = True
    while generator.num_remaining_jobs():
        if time.monotonic() > deadline:
            raise BenchError("native generation wave exceeded 30 minutes")
        events = generator.iterate()
        max_active = max(max_active, generator.num_active_jobs())
        if first_iteration:
            first_started_serials = {event["serial"] for event in events if event["stage"] == "started"}
            first_iteration = False
        free, total = torch.cuda.mem_get_info()
        peak_device_used = max(peak_device_used, total - free)
        for event in events:
            if event["stage"] == "streaming" and "token_ids" in event:
                output_ids[event["serial"]].append(event["token_ids"].cpu())
            if event.get("eos"):
                eos_by_serial[event["serial"]] = event
    torch.cuda.synchronize()
    ended = time.perf_counter()
    if first_started_serials != set(serials) or max_active < concurrency:
        raise BenchError(
            f"native scheduler did not admit all c={concurrency} jobs together; "
            f"first_started={len(first_started_serials)}, max_active={max_active}"
        )
    if set(eos_by_serial) != set(serials):
        raise BenchError(f"lost native requests: expected {serials}, completed {sorted(eos_by_serial)}")

    requests = []
    for serial in serials:
        event = eos_by_serial[serial]
        if (
            event.get("eos_reason") != "max_new_tokens"
            or event.get("new_tokens") != OUTPUT_TOKENS
            or event.get("prompt_tokens") != INPUT_TOKENS
        ):
            raise BenchError(f"request {serial} did not complete exact 512/128 protocol: {event}")
        emitted_tokens = sum(tensor.numel() for tensor in output_ids[serial])
        if emitted_tokens != OUTPUT_TOKENS:
            raise BenchError(f"request {serial} emitted {emitted_tokens} token IDs, expected {OUTPUT_TOKENS}")
        ttft = event["time_enqueued"] + event["time_prefill"]
        generation = event["time_generate"]
        requests.append(
            {
                "serial": serial,
                "identifier": event.get("identifier"),
                "prompt_tokens": event["prompt_tokens"],
                "output_tokens": event["new_tokens"],
                "queue_ms": event["time_enqueued"] * 1000,
                "prefill_ms": event["time_prefill"] * 1000,
                "ttft_ms": ttft * 1000,
                "generation_ms": generation * 1000,
                "tpot_ms": generation * 1000 / (OUTPUT_TOKENS - 1),
                "e2e_ms": (ttft + generation) * 1000,
                "output_ids_sha256": tensor_hash(output_ids[serial]),
            }
        )
    return requests, {
        "wall_seconds": ended - started,
        "prompt_ids_sha256": tensor_hash(prompts),
        "max_active_jobs": max_active,
        "peak_device_used_bytes_sampled": peak_device_used,
    }


def run_recorded(generator, tokenizer, concurrency: int, repeat: int, torch) -> dict:
    num_prompts = PROMPTS_PER_RUN[concurrency]
    if num_prompts % concurrency:
        raise BenchError("num_prompts must be divisible by concurrency")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    run_started = time.perf_counter()
    requests, waves = [], []
    for wave in range(num_prompts // concurrency):
        wave_requests, wave_record = run_wave(
            generator,
            tokenizer,
            concurrency,
            request_base=1_000_000 * concurrency + 10_000 * repeat + concurrency * wave,
            torch=torch,
        )
        requests.extend(wave_requests)
        waves.append(wave_record)
    torch.cuda.synchronize()
    harness_wall = time.perf_counter() - run_started
    model_wall = sum(row["wall_seconds"] for row in waves)
    output_tokens = sum(row["output_tokens"] for row in requests)
    return {
        "repeat": repeat,
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "input_tokens_each": INPUT_TOKENS,
        "output_tokens_each": OUTPUT_TOKENS,
        "model_wall_seconds": model_wall,
        "harness_wall_seconds": harness_wall,
        "output_tokens_per_second": output_tokens / model_wall,
        "median_ttft_ms": statistics.median(row["ttft_ms"] for row in requests),
        "p90_ttft_ms": percentile([row["ttft_ms"] for row in requests], 0.90),
        "median_tpot_ms": statistics.median(row["tpot_ms"] for row in requests),
        "p90_tpot_ms": percentile([row["tpot_ms"] for row in requests], 0.90),
        "median_e2e_ms": statistics.median(row["e2e_ms"] for row in requests),
        "p90_e2e_ms": percentile([row["e2e_ms"] for row in requests], 0.90),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_device_used_bytes_sampled": max(row["peak_device_used_bytes_sampled"] for row in waves),
        "waves": waves,
        "requests": requests,
    }


def summarize(repeats: list[dict]) -> dict:
    return {
        "output_tokens_per_second_median": statistics.median(row["output_tokens_per_second"] for row in repeats),
        "output_tokens_per_second_min": min(row["output_tokens_per_second"] for row in repeats),
        "output_tokens_per_second_max": max(row["output_tokens_per_second"] for row in repeats),
        "median_ttft_ms": statistics.median(row["median_ttft_ms"] for row in repeats),
        "median_tpot_ms": statistics.median(row["median_tpot_ms"] for row in repeats),
        "median_e2e_ms": statistics.median(row["median_e2e_ms"] for row in repeats),
        "peak_torch_reserved_gib": max(row["peak_torch_reserved_bytes"] for row in repeats) / 1024**3,
        "peak_torch_allocated_gib": max(row["peak_torch_allocated_bytes"] for row in repeats) / 1024**3,
        "peak_device_used_gib_sampled": max(row["peak_device_used_bytes_sampled"] for row in repeats) / 1024**3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--revision",
        required=True,
        help="40-hex snapshot identity; this need not be a Hugging Face commit",
    )
    parser.add_argument("--exllama-source", type=Path, required=True)
    parser.add_argument("--cache-mode")
    parser.add_argument("--cache-tokens", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--versions-output", type=Path, required=True)
    parser.add_argument("--validate-api-only", action="store_true")
    args = parser.parse_args()

    versions = verify_pins(args.exllama_source, args.revision)
    concurrency = validate_native_concurrency(args.exllama_source)
    base_manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": versions,
        "exllamav3_commit": EXLLAMA_COMMIT,
        "checkpoint_revision": args.revision,
        "native_concurrency": concurrency,
    }
    if args.validate_api_only:
        base_manifest["validation_only"] = True
        atomic_json(args.versions_output, base_manifest)
        atomic_json(args.output, {"native_concurrency": concurrency, "status": "PASS"})
        print(json.dumps(base_manifest, indent=2, sort_keys=True))
        return

    if args.model_dir is None or args.cache_mode is None or args.cache_tokens is None:
        raise BenchError("full benchmark requires model-dir, cache-mode, and cache-tokens")
    if args.cache_mode not in {"fp16", "q8", "q6", "q4", "q3", "q2"}:
        raise BenchError(f"unsupported explicit cache mode: {args.cache_mode!r}")
    minimum_cache = max(CONCURRENCY_GRID) * math.ceil((INPUT_TOKENS + OUTPUT_TOKENS) / PAGE_SIZE) * PAGE_SIZE
    if args.cache_tokens < minimum_cache or args.cache_tokens % PAGE_SIZE:
        raise BenchError(f"cache-tokens must be a multiple of {PAGE_SIZE} and >= {minimum_cache} for c=8 exact 512/128")
    checkpoint = checkpoint_manifest(args.model_dir, args.revision)
    gpu = assert_gpu_idle()
    base_manifest.update(
        {
            "status": "GPU_PREFLIGHT_PASS",
            "checkpoint": checkpoint,
            "gpu": gpu,
            "cache_mode": args.cache_mode,
            "cache_tokens": args.cache_tokens,
        }
    )
    atomic_json(args.versions_output, base_manifest)

    import torch
    from exllamav3 import Cache, Config, Model, Tokenizer
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.generator import Generator

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
        raise BenchError("CUDA runtime must expose exactly one RTX 5090")
    if torch.version.cuda != "12.8":
        raise BenchError(f"torch CUDA runtime is {torch.version.cuda}, expected 12.8")
    torch.cuda.reset_peak_memory_stats()
    config = Config.from_directory(str(args.model_dir))
    config.override_dynamic_seq_len(args.cache_tokens)
    model = Model.from_config(config)
    cache_kwargs = {}
    if args.cache_mode != "fp16":
        bits = int(args.cache_mode.removeprefix("q"))
        cache_kwargs = {"layer_type": CacheLayer_quant, "k_bits": bits, "v_bits": bits}
    cache = Cache(
        model,
        max_num_tokens=args.cache_tokens,
        max_batch_size=max(CONCURRENCY_GRID),
        **cache_kwargs,
    )
    load_started = time.perf_counter()
    model.load(
        device="cuda:0",
        progressbar=True,
        max_chunk_size=2048,
        max_output_size=max(CONCURRENCY_GRID),
        max_batch_size=max(CONCURRENCY_GRID),
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    generator = Generator(
        model=model,
        cache=cache,
        tokenizer=Tokenizer.from_config(config),
        max_batch_size=max(CONCURRENCY_GRID),
        max_chunk_size=2048,
        enable_defrag=True,
    )
    tokenizer = generator.tokenizer
    base_manifest.update(
        {
            "status": "MODEL_LOADED",
            "torch_cuda": torch.version.cuda,
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "load_seconds": load_seconds,
            "load_peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
            "load_peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "load_metrics": load_metrics_or_error(model),
        }
    )
    atomic_json(args.versions_output, base_manifest)

    results = {
        "status": "RUNNING",
        "protocol": {
            "input_tokens": INPUT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "concurrency": list(CONCURRENCY_GRID),
            "prompts_per_recorded_run": PROMPTS_PER_RUN,
            "recorded_repeats": REPEATS,
            "warmup_waves_per_cell": 1,
            "sampling": "greedy",
            "stop_conditions": [],
        },
        "versions_manifest": str(args.versions_output),
        "cells": [],
    }
    atomic_json(args.output, results)
    for cell_index, cell_concurrency in enumerate(CONCURRENCY_GRID):
        # One exact-shape warmup wave is deliberately discarded.
        run_wave(
            generator,
            tokenizer,
            cell_concurrency,
            request_base=900_000_000 + cell_index * 100,
            torch=torch,
        )
        recorded = [run_recorded(generator, tokenizer, cell_concurrency, repeat, torch) for repeat in range(REPEATS)]
        cell = {
            "concurrency": cell_concurrency,
            "summary": summarize(recorded),
            "recorded_runs": recorded,
        }
        results["cells"].append(cell)
        atomic_json(args.output, results)
        print(json.dumps(cell, indent=2, sort_keys=True))
    results["status"] = "PASS"
    results["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(args.output, results)
    print(f"NATIVE_EXLLAMAV3_BENCHMARK_PASS output={args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
