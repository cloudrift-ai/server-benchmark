"""Recipe dataclass types."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VllmConfig:
    """vLLM engine-specific configuration."""

    image: str = "vllm/vllm-openai:v0.17.0"
    entrypoint: str | None = None
    extra_args: str = ""
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class SglangConfig:
    """SGLang engine-specific configuration."""

    image: str = "lmsysorg/sglang:v0.5.9"
    extra_args: str = ""
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Engine-agnostic LLM serving configuration."""

    context_length: int | None = None
    max_concurrent_requests: int | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    vllm: VllmConfig | None = None
    sglang: SglangConfig | None = None
    docker_options: dict[str, Any] = field(default_factory=dict)

    @property
    def gpus_per_instance(self) -> int:
        """Number of GPUs consumed by one model instance."""
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size

    @property
    def engine_name(self) -> str:
        """Active engine name: 'vllm' or 'sglang'."""
        if self.sglang is not None:
            return "sglang"
        return "vllm"

    @property
    def image(self) -> str:
        """Docker image for the active engine."""
        if self.sglang is not None:
            return self.sglang.image
        if self.vllm is not None:
            return self.vllm.image
        return VllmConfig().image

    @property
    def entrypoint(self) -> str | None:
        """Docker entrypoint override for the active engine.

        vLLM normally uses the image entrypoint but may override it for a same-image
        control. SGLang images do not provide one, so its launcher is explicit.
        """
        if self.sglang is not None:
            return "python3 -m sglang.launch_server"
        if self.vllm is not None:
            return self.vllm.entrypoint
        return None

    @property
    def extra_args(self) -> str:
        """Extra CLI flags for the active engine."""
        if self.sglang is not None:
            return self.sglang.extra_args
        if self.vllm is not None:
            return self.vllm.extra_args
        return ""

    @property
    def extra_env(self) -> dict[str, str]:
        """Extra environment variables for the active engine."""
        if self.sglang is not None:
            return self.sglang.extra_env
        if self.vllm is not None:
            return self.vllm.extra_env
        return {}


@dataclass
class EngineConfig:
    """Top-level engine configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class ModelConfig:
    """Model configuration."""

    huggingface: str = ""
    # Immutable Hugging Face revision used by both model prefetch and the serving engine.
    revision: str | None = None
    # What the model serves: "generate" (completion/chat, the default) or
    # "embed" (/v1/embeddings). Drives the smoke test and the bench workload.
    task: str = "generate"
    # Generative checkpoints default to the semantic chat smoke test. Base models that
    # are not instruction-tuned use the completion endpoint instead.
    smoke_test: str = "chat"


@dataclass
class BenchmarkConfig:
    """Benchmark workload configuration.

    ``seed`` / ``temperature`` / ``ignore_eos`` pin the workload for controlled cross-engine
    comparisons: a fixed seed reproduces the same random prompt set, ``temperature: 0`` forces
    greedy decoding, and ``ignore_eos: true`` makes every request generate exactly
    ``random_output_len`` tokens. Unset fields emit no flag, keeping the client's defaults
    (note: an unset temperature is the server's default sampling, not greedy).

    ``repeats`` reruns the identical bench-client workload N times against the one deployed
    server; the JSON result then reports per-field mean and stddev across the runs, so the
    spread is run-to-run noise, not workload variation. ``output_probe_file`` optionally
    names a repo-relative JSONL prompt set captured sequentially after the throughput
    workload and before teardown; the raw deterministic responses are stored with the task.
    ``comparison_arm`` and ``process_repeat`` identify paired fresh-process tasks. When
    ``require_output_equivalence`` is true, ``emmy bench`` requires byte-exact probe outputs
    across the two arms before reporting success. ``comparison_order`` records the planned
    interleaving order without affecting execution. ``require_complete_requests`` requires
    every client repeat to report exactly ``num_prompts`` successful requests and zero failed
    requests. ``required_server_log_patterns`` and
    ``forbidden_server_log_patterns`` are regular-expression gates over the complete server
    log captured after the workload; requesting either makes log retrieval and every verdict
    fail closed."""

    max_concurrency: int = 128
    num_prompts: int = 256
    random_input_len: int = 8000
    random_output_len: int = 8000
    seed: int | None = None
    temperature: float | None = None
    ignore_eos: bool = False
    repeats: int = 1
    output_probe_file: str | None = None
    comparison_arm: str | None = None
    process_repeat: int | None = None
    comparison_order: int | None = None
    require_output_equivalence: bool = False
    require_complete_requests: bool = False
    required_server_log_patterns: list[str] = field(default_factory=list)
    forbidden_server_log_patterns: list[str] = field(default_factory=list)


@dataclass
class CommandConfig:
    """Generic command workload configuration.

    Used by command-style recipes that run an arbitrary tool on the
    provisioned VM (instead of deploying an inference server).

    Fields:
        stage: Repo paths to stage to the remote VM (via git ls-files +
            tar). Empty list = no staging; $repo_dir is unavailable.
        run: Shell command template (string.Template $-syntax). Substituted
            with variant params (flattened to leaf names) plus injected
            $task_dir, $repo_dir, $gpu_device_ids.
        result_files: List of result file names or shell globs (e.g.
            "result.json", "*.log"). Globs expand on the remote; each
            matched file is pulled back as {variant}_{basename}.
        timeout: Per-task command timeout in seconds.
        env: Extra environment variables to set on the remote command.
        strict: Require a clean staged source tree, every declared result file,
            and complete source/GPU/CUDA provenance.
    """

    stage: list[str] = field(default_factory=list)
    run: str = ""
    result_files: list[str] = field(default_factory=list)
    timeout: int = 1800
    env: dict[str, str] = field(default_factory=dict)
    strict: bool = False


@dataclass
class AggregateConfig:
    """Small, self-contained post-processing step run after a recipe matrix."""

    run: str = ""
    timeout: int = 300


@dataclass
class DeployConfig:
    """Optional deploy section — GPU info for cloud provisioning."""

    gpu: str | None = None
    gpu_count: int = 1
    driver_version: str | None = None
    cuda_version: str | None = None


@dataclass
class Recipe:
    """Complete recipe configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    command: CommandConfig | None = None
    aggregate: AggregateConfig | None = None

    @property
    def kind(self) -> str:
        """Recipe kind: 'command' if a command block is set, else 'inference'."""
        return "command" if self.command is not None else "inference"

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        """Build a Recipe from a (post-merge, post-migrate) config dict."""
        model_dict = d.get("model", {})
        model = ModelConfig(
            huggingface=model_dict.get("huggingface", ""),
            revision=model_dict.get("revision"),
            task=model_dict.get("task", "generate"),
            smoke_test=model_dict.get("smoke_test", "chat"),
        )

        engine_dict = d.get("engine", {})
        llm_dict = engine_dict.get("llm", {})

        vllm_dict = llm_dict.get("vllm")
        vllm = VllmConfig(**vllm_dict) if vllm_dict is not None else None

        sglang_dict = llm_dict.get("sglang")
        sglang = SglangConfig(**sglang_dict) if sglang_dict is not None else None

        llm = LLMConfig(
            context_length=llm_dict.get("context_length"),
            max_concurrent_requests=llm_dict.get("max_concurrent_requests"),
            tensor_parallel_size=llm_dict.get("tensor_parallel_size", 1),
            pipeline_parallel_size=llm_dict.get("pipeline_parallel_size", 1),
            data_parallel_size=llm_dict.get("data_parallel_size", 1),
            gpu_memory_utilization=llm_dict.get("gpu_memory_utilization", 0.9),
            vllm=vllm,
            sglang=sglang,
            docker_options=llm_dict.get("docker_options", {}),
        )

        bench_dict = d.get("benchmark", {})
        workload_fields = {
            "max_concurrency",
            "num_prompts",
            "random_input_len",
            "random_output_len",
            "seed",
            "temperature",
            "ignore_eos",
            "repeats",
        }
        unsupported_benchmark_fields = set(bench_dict) - workload_fields
        if unsupported_benchmark_fields:
            names = ", ".join(sorted(unsupported_benchmark_fields))
            raise ValueError(f"unsupported benchmark fields: {names}; benchmark only describes workload generation")
        benchmark = BenchmarkConfig(
            max_concurrency=bench_dict.get("max_concurrency", 128),
            num_prompts=bench_dict.get("num_prompts", 256),
            random_input_len=bench_dict.get("random_input_len", 8000),
            random_output_len=bench_dict.get("random_output_len", 8000),
            seed=bench_dict.get("seed"),
            temperature=bench_dict.get("temperature"),
            ignore_eos=bench_dict.get("ignore_eos", False),
            repeats=bench_dict.get("repeats", 1),
            output_probe_file=bench_dict.get("output_probe_file"),
            comparison_arm=bench_dict.get("comparison_arm"),
            process_repeat=bench_dict.get("process_repeat"),
            comparison_order=bench_dict.get("comparison_order"),
            require_output_equivalence=bench_dict.get("require_output_equivalence", False),
            require_complete_requests=bench_dict.get("require_complete_requests", False),
            required_server_log_patterns=list(bench_dict.get("required_server_log_patterns", [])),
            forbidden_server_log_patterns=list(bench_dict.get("forbidden_server_log_patterns", [])),
        )

        deploy_dict = d.get("deploy", {})
        deploy = DeployConfig(
            gpu=deploy_dict.get("gpu"),
            gpu_count=deploy_dict.get("gpu_count", 1),
            driver_version=deploy_dict.get("driver_version"),
            cuda_version=deploy_dict.get("cuda_version"),
        )

        command = None
        cmd_dict = d.get("command")
        if cmd_dict is not None:
            removed_strict_fields = {"require_clean_stage", "require_result_files", "require_provenance"} & cmd_dict.keys()
            if removed_strict_fields:
                names = ", ".join(sorted(removed_strict_fields))
                raise ValueError(f"command fields {names} were replaced by the single 'strict' field")
            command = CommandConfig(
                stage=list(cmd_dict.get("stage", [])),
                run=cmd_dict.get("run", ""),
                result_files=list(cmd_dict.get("result_files", [])),
                timeout=cmd_dict.get("timeout", 1800),
                env=dict(cmd_dict.get("env", {})),
                strict=cmd_dict.get("strict", False),
            )

        aggregate = None
        agg_dict = d.get("aggregate")
        if agg_dict is not None:
            aggregate = AggregateConfig(
                run=agg_dict.get("run", ""),
                timeout=agg_dict.get("timeout", 300),
            )

        return cls(
            model=model,
            engine=EngineConfig(llm=llm),
            benchmark=benchmark,
            deploy=deploy,
            command=command,
            aggregate=aggregate,
        )

    @property
    def model_name(self) -> str:
        """Shortcut for model.huggingface."""
        return self.model.huggingface

    @property
    def is_embedding(self) -> bool:
        """True for embedding recipes (``model.task: embed``) — /v1/embeddings
        smoke test + ``vllm bench serve --backend openai-embeddings`` workload."""
        return self.model.task == "embed"
