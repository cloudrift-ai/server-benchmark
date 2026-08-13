# Recipe Architecture

## Overview

The `recipe` package owns all recipe-related logic: YAML loading, matrix expansion for benchmark parameter sweeps,
typed configuration dataclasses, engine flag mapping, catalog queries, onboarding shell creation, and `extra_args`
validation.

## Modules

- `types.py` — dataclasses: `Recipe`, `DeployConfig`, `ModelConfig`, `EngineConfig`, `LLMConfig`, `VllmConfig`,
  `SglangConfig`, `BenchmarkConfig`, `CommandConfig`
- `lifecycle.py` — lifecycle tag validation and the runnable/disabled predicate
- `catalog.py` — compact repository inventory, deployment extraction, and validated onboarding shell creation
- `recipe.py` — `deep_merge()`, `load_recipe()`, `resolve_for_hardware()`, `validate_extra_args()`, `_load_raw_config()`, `_validate_and_build()`
- `matrix.py` — `expand_matrix()`, `_expand_cross()`, `_expand_zip()`, `filter_combinations()`, `dot_to_nested()`, `build_override()`
- `engines.py` — `VLLM_FLAG_MAP`, `SGLANG_FLAG_MAP`, `banned_extra_arg_flags()`, `build_engine_args()`
- `bundled.py` — `bundled_names()`, `resolve_recipe_dir()` — the recipes shipped inside an installed wheel

## Key Design Decisions

### Bundled Recipes Are Copied Out Before Use

A wheel carries every runnable `recipes/<model>/recipe.yaml` under `emmy/recipes/`, staged at build time because
`recipes/` sits outside the package (see `scripts/prepare_dist.py`). Those copies are read-only — they live in
site-packages, whereas `deploy` writes its compose file into the recipe directory and `bench` creates run directories
there. So `resolve_recipe_dir()` treats a bare name as a request for a **working copy**: it copies the bundled recipe
into the current directory and returns that path. An existing directory always takes precedence, so a name that
matches both a local directory and a bundled recipe resolves to the local one and an edited copy is never clobbered.

### Recipe Lifecycle Tags

The optional top-level `tags` list records whether a recipe belongs to the actively supported set:

```yaml
tags:
  - maintained
```

The lifecycle tags are mutually exclusive. `maintained` is a tested recipe selected for periodic testing and
optimization. `best-effort` is a useful runnable recipe outside that periodic set. `obsolete` preserves a recipe in
git while disabling deploy, benchmark, publish, and wheel staging; it is used only when an all-around better model for
the same task is available at a comparable or lower practical VRAM footprint, or when a technical limitation means
the recipe should no longer be used. Low demand or age alone is not enough. Discovery stores every lifecycle reason in
`model.rationale`. A new discovery shell carries both `onboarding` and `untested`; it contains the model ID, task,
rationale, and one to three proposed `deploy.gpu`/`deploy.gpu_count` matrix entries, but is not runnable until
onboarding replaces it with a qualified `best-effort` recipe. Untagged recipes remain runnable for backward
compatibility and are classified by the next discovery lifecycle run.

Tag values are unique lowercase kebab-case strings. `onboarding` and `untested` must appear together. The runtime
rejects direct use of disabled recipes, while bulk benchmark enumeration and package staging skip them.

`model.rationale` is descriptive lifecycle metadata. It records why the model currently belongs in the inventory and
does not affect engine arguments, deployment, or benchmark behavior.

`recipe_catalog()` is the shared repository scan behind `emmy recipe list` and model-discovery validation. Its compact
records contain only the identity, tags, task, rationale, and expanded deployment setups needed for lifecycle work.
`create_recipe_stub()` is likewise shared by `emmy recipe create` and discovery: it validates one to three canonical
GPU/count setups and writes the minimal disabled shell without duplicating YAML rendering in workflow scripts.

### Matrix Expansion for Benchmark Sweeps

Recipes use `matrices` — a dict that defines benchmark configurations using `cross` and `zip` combinators:

```yaml
# Simple single-point entry (implicit zip, all scalars)
matrices:
  deploy.gpu: "NVIDIA GeForce RTX 5090"
  deploy.gpu_count: 1

# Cross-product: 3 GPUs × 2 configs = 6 variants
matrices:
  cross:
    deploy.gpu_count: 1
    deploy.gpu:
      - "NVIDIA GeForce RTX 5090"
      - "NVIDIA H100 80GB"
      - "NVIDIA H200 141GB"
    zip:
      engine.llm.max_concurrent_requests: [128, 512]
      benchmark.max_concurrency: [128, 512]

# Concurrency sweep (zip: 8 runs)
matrices:
  deploy.gpu: "NVIDIA GeForce RTX 5090"
  engine.llm.max_concurrent_requests: [1, 2, 4, 8, 16, 32, 64, 128]
  benchmark.max_concurrency: [1, 2, 4, 8, 16, 32, 64, 128]
```

Rules:
- **`cross` node**: scalars broadcast, lists are independent cross-product axes, nested `zip` dicts are one compound axis
- **`zip` node**: scalars broadcast, lists are zipped element-wise (must all be the same length), nested `cross` dicts are one zip axis
- A plain dict (no `cross`/`zip` key) is an implicit `zip`
- `cross`/`zip` keys are only combinators when their value is a dict
- **`deploy.gpu`** is required
- **Dot-notation** is used for all parameter paths (`deploy.gpu`, `engine.llm.max_concurrent_requests`, etc.)

The entry point is `expand_matrix(matrices)` which dispatches to `_expand_cross()` or `_expand_zip()` recursively. Each resulting combination is converted to a nested override dict via `build_override()` and deep-merged with the base config.

#### Variant Filtering

`filter_combinations(combinations, filters)` applies `--filter KEY=PATTERN` flags (fnmatch glob, AND logic) after expansion, before building Recipe objects. This allows running a subset of variants (e.g. `--filter "deploy.gpu=*5090*"`).

### Deep Merge

Override dicts are applied to base configs via recursive deep merge (`deep_merge()`). Nested dicts are merged key-by-key; scalars in the override replace the base. This allows matrix entries to selectively override any field at any depth:

```yaml
# base
engine:
  llm:
    tensor_parallel_size: 8
    context_length: 16384
    vllm:
      extra_args: "--kv-cache-dtype fp8"

# matrix entry overrides only what changes
matrices:
  deploy.gpu: "NVIDIA H100 80GB"
  engine.llm.max_concurrent_requests: 256
```

The merged result keeps `tensor_parallel_size: 8`, `context_length: 16384`, and the vllm block from the base, while adding `max_concurrent_requests: 256` from the matrix entry.

### Auto-Generated Run Identifiers (Variant)

Variant naming is handled by `Variant` in `emmy.planner.variant`. Each matrix combination produces a `Variant(params=combo)` that derives its string representation from the raw params dict:

- GPU part: `{gpu_short}x{gpu_count}` (e.g. `rtx5090x1`, `h100x8`)
- Non-deploy params: abbreviated via first-letter-of-each-word heuristic (`max_concurrency` → `mc`, `num_prompts` → `np`, `max_concurrent_requests` → `mcr`), sorted alphabetically, appended with `_`
- All params appear in the label, not just the variable ones

Examples: `rtx5090x1_mc8_mcr8_np80_vllm_benchmark.txt`, `rtx5090x1_vllm_benchmark.txt` (deploy-only params).

### Driver / CUDA Version Pinning

`deploy.driver_version` and `deploy.cuda_version` (both optional) request a specific NVIDIA driver / CUDA toolkit on the target host. If the installed version already matches (prefix-match — `"550"` matches `550.127.05`), provisioning is a no-op. On a mismatch, a remote (`ssh`/`cloud`) deploy installs the requested version, reboots the host, and waits for SSH to come back. Local deploys refuse to run privileged commands and error out instead — these fields are intended for remote machines only.

```yaml
matrices:
  deploy.gpu: "NVIDIA H200 141GB"
  deploy.gpu_count: 8
  deploy.driver_version: "550"
  deploy.cuda_version: "12.4"
```

### DeployConfig

GPU provisioning info is encapsulated in `DeployConfig` (nested under `Recipe.deploy`):

```python
@dataclass
class DeployConfig:
    gpu: str | None = None
    gpu_count: int = 1
```

Matrix entries use `deploy.gpu` and `deploy.gpu_count` via dot-notation override. The `deploy` section is optional in the base recipe — it's only needed when `deploy cloud` requires GPU info directly (without matrix expansion).

### First-Class Named Parameters

Engine-agnostic serving parameters are promoted to first-class named fields on `LLMConfig` rather than being buried in `extra_args` strings:

| Field | vLLM flag | SGLang flag |
|---|---|---|
| `tensor_parallel_size` | `--tensor-parallel-size` | `--tp` |
| `pipeline_parallel_size` | `--pipeline-parallel-size` | `--pp` |
| `data_parallel_size` | `--data-parallel-size` | `--dp` |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | `--mem-fraction-static` |
| `context_length` | `--max-model-len` | `--context-length` |
| `max_concurrent_requests` | `--max-num-seqs` | `--max-running-requests` |

This design provides:

1. **Type safety** — numeric values are validated at parse time, not when Docker fails.
2. **Engine portability** — the same recipe field maps to different CLI flags per engine via `VLLM_FLAG_MAP` / `SGLANG_FLAG_MAP` in `engines.py`.
3. **Computed properties** — `LLMConfig.gpus_per_instance` derives from `tensor_parallel_size * pipeline_parallel_size * data_parallel_size` without parsing strings.
4. **Deep merge support** — named fields participate in matrix merging naturally. An `extra_args` string cannot be partially overridden.

### Controlled Workload Flags

`BenchmarkConfig` carries three optional knobs that pin the bench-client workload for controlled cross-engine
comparisons: `seed` reproduces the same random prompt set across runs and engines, `temperature: 0` forces greedy
decoding, and `ignore_eos: true` makes every request generate exactly `random_output_len` tokens so all engines do
identical work. Each unset field emits no client flag (prior behavior); note that no `temperature` means the server's
default sampling, not greedy. `temperature` / `ignore_eos` are generation-only and are skipped for embedding recipes
(`benchmark/workload.py`).

`benchmark.num_warmups` (default 0) makes the bench client complete that many requests before it starts measuring.
Use it when the serving engine performs request-time initialization after the deployment health check. The warmup
requests use the same controlled workload configuration and run before every measured repeat.

`benchmark.repeats` (default 1) reruns the identical bench-client workload N times against the one deployed server —
the model is deployed once, only the client run repeats. The text result then holds one stanza per repeat, and the
JSON result's `metrics` becomes the per-field mean, with `metrics_stddev` (sample stddev) and `metrics_repeats` (the
raw per-repeat metrics) added alongside (`benchmark/results.py`). Because the seed and prompts are identical across
repeats, the spread measures run-to-run noise, not workload variation.

The `benchmark` block describes workload generation only. Unknown fields are rejected rather than becoming implicit
result validators. `emmy bench` preserves raw observations but does not interpret whether they support an experiment's
claim; that decision belongs to review of the completed run directory.

### Extra Args Ban Enforcement

Users must not duplicate named fields in `extra_args`. The `validate_extra_args()` function enforces this by:

1. Building a banned set from the active engine's flag map (`VLLM_FLAG_MAP` or `SGLANG_FLAG_MAP`) plus hardcoded flags
   (`--trust-remote-code`, `--host`, `--port`, `--model`, `--model-path`, `--served-model-name`, `--revision`).
2. Tokenizing the `extra_args` string and checking each token (handling both `--flag value` and `--flag=value` forms).
3. Raising `ValueError` listing all offending flags if any are found.

This validation runs inside `_validate_and_build()` before returning the `Recipe`, so invalid configs fail fast at load time rather than at Docker runtime.

`extra_args` is the escape hatch for engine-specific flags that don't have a named field (e.g. `--kv-cache-dtype fp8`, `--enable-expert-parallel`). It is passed through verbatim to `build_engine_args()`.

### Engine-Specific Model Flag

`build_engine_args()` emits the model path using the flag expected by each engine:
- vLLM: `--model {name}`
- SGLang: `--model-path {name}`

Both `--model` and `--model-path` are in the hardcoded banned set, so they cannot appear in `extra_args` regardless of
which engine is active. Immutable checkpoints use `model.revision`; `build_engine_args()` emits `--revision` for both
engines, and deployment passes the same value to `hf download`. `--revision` is therefore also banned from
`extra_args`, preventing the prefetch and server revisions from drifting apart.

### Engine-Specific Nesting

Engine-specific config (`image`, `extra_args`, `extra_env`) nests under `engine.llm.vllm` or `engine.llm.sglang`, while engine-agnostic config lives at `engine.llm`. `LLMConfig.engine_name` is determined by which sub-config is present (SGLang takes priority if both exist). The `image`, `extra_args`, and `extra_env` properties delegate to the active engine's config.

### Extra Environment Variables

`extra_env` is a `dict[str, str]` on `VllmConfig` / `SglangConfig` that injects arbitrary environment variables into the Docker Compose container. It defaults to an empty dict. `LLMConfig.extra_env` delegates to the active engine's config, mirroring the pattern used by `extra_args`.

```yaml
engine:
  llm:
    vllm:
      extra_env:
        VLLM_ATTENTION_BACKEND: FLASHINFER
        CUDA_LAUNCH_BLOCKING: "1"
```

Each key-value pair is rendered as a `- KEY=VALUE` line in the `environment` section of the generated Docker Compose file.

### Embedding Recipes (`model.task`)

`model.task` declares what the model serves: `"generate"` (the default — completion/chat) or `"embed"`
(`/v1/embeddings`). `_validate_and_build()` rejects anything else. `Recipe.is_embedding` drives two behavior switches
downstream: the post-deploy smoke test POSTs `/v1/embeddings` and checks for a finite unit-norm vector
(`deploy/orchestrate.py`), and the bench workload runs `vllm bench serve --backend openai-embeddings --endpoint
/v1/embeddings` without `--random-output-len` (`benchmark/workload.py`). Everything else — matrices, images,
`extra_args`, deploy — is unchanged; serving the emmy compiler plugin instead of stock vLLM is just a different
image + `extra_args` pair (see `emmy/serving/ARCHITECTURE.md` and `recipes/Qwen3-Embedding-*`).

```yaml
model:
  huggingface: "Qwen/Qwen3-Embedding-0.6B"
  task: embed
```

Generative recipes use the semantic chat smoke test by default. A base checkpoint that is not instruction-tuned sets
`model.smoke_test: completion`; deployment then sends `2 + 2 =` to `/v1/completions` and still requires the correct
answer. The choice changes only the post-health correctness gate, not the benchmark endpoint or serving task.

### Command Recipes (Generic Workload)

In addition to inference recipes (`engine.llm` block), a recipe may declare a `command` block to run an arbitrary tool on the provisioned VM. The two are mutually exclusive — `_validate_and_build()` raises if both are set. `Recipe.kind` is `"command"` when `command` is set, else `"inference"`.

```yaml
command:
  stage: ["scripts"]              # repo paths to ship to the VM (git ls-files); empty = no staging
  run: |
    nvidia-smi --query-gpu=name --format=csv > $task_dir/result.csv
    echo "marker,$marker" >> $task_dir/result.csv
  result_files:                    # filenames or shell globs; expanded on the remote
    - result.csv
    - "*.log"
  timeout: 60
  env: {FOO: bar}                  # optional, prepended as KEY=value to the command
  strict: true                     # clean source, required artifacts and provenance

matrices:
  deploy.gpu: "NVIDIA GeForce RTX 5090"
  deploy.gpu_count: 1
  marker: [a, b, c]
```

The `run` template uses `string.Template` `$var` syntax. Substitution variables come from variant params (flattened to leaf names: `deploy.gpu` → `gpu`, `marker` → `marker`) plus harness-injected `$task_dir`, `$gpu_device_ids`, and `$repo_dir` (only when `stage` is non-empty). Conflicting leaf names (e.g. two matrix keys flattening to `gpu`) raise at substitution time.

Command recipes skip `validate_extra_args()` since they don't go through engine flag mapping.

Without `strict`, artifact transfer is best effort and staged files may include local edits. With `strict`, the staged
paths must be clean before execution, their exact file digests and Git revision are recorded, every `result_files`
entry must be retrieved, and GPU/CUDA provenance must be available. A failed command still attempts to retrieve its
declared artifacts so partial evidence is not lost.

### Inline Post-Processing

A recipe may declare an `aggregate` block for a short local command after its variants complete:

```yaml
aggregate:
  run: |
    rows="$run_dir/small_m_results.tsv"
    printf 'gpu\tstrategy\tm\tn\tk\tbatch\tkernel_ms\tcublas_ms\n' > "$rows"
    find "$run_dir" -maxdepth 1 -type f -name '*.json' -print |
      sort |
      while IFS= read -r result; do
        jq -r '
          . as $run
          | ($run.results // [])[]
          | select(.dimensions.M <= 128)
          | [$run.system_info.gpu, $run.strategy, .dimensions.M, .dimensions.N,
             .dimensions.K, .dimensions.batch, .kernel_time_ms, .cublas_time_ms]
          | @tsv
        ' "$result"
      done >> "$rows"
  timeout: 60
```

The template receives `$run_dir`. The example performs transparent structural processing: it selects the small-M
rows from each SGEMM JSON and assembles one TSV table. Keep such commands self-contained and readable in the recipe;
do not invoke an external result-analysis script. This hook may select fields, reshape rows, sort, join, or tabulate
structured data, but it must not interpret the results or generate a human-readable report such as `RESULTS.md`.
Agents inspect the raw run and write model-specific reports when richer analysis is required.

### Docker Options

`docker_options` is a `dict[str, Any]` on `LLMConfig` that injects arbitrary docker-compose service-level keys into the generated container definition. It defaults to an empty dict. Unlike `extra_env` and `extra_args`, which are engine-specific and live on `VllmConfig`/`SglangConfig`, `docker_options` lives directly on `LLMConfig` because Docker container options (security, capabilities, ulimits) are tied to GPU hardware, not the inference engine.

```yaml
engine:
  llm:
    vllm:
      image: "rocm/vllm:latest"
    docker_options:
      security_opt:
        - seccomp=unconfined
      cap_add:
        - SYS_PTRACE
```

Each key-value pair is rendered as a top-level service key in the generated Docker Compose file, inserted after `ipc: host` and before `command:`. Values are serialized via `yaml.dump()` to handle nested structures (lists, dicts, scalars) correctly.

Keys managed by the compose template (`image`, `container_name`, `entrypoint`, `deploy`, `devices`, `group_add`, `volumes`, `environment`, `ports`, `shm_size`, `ipc`, `command`, `healthcheck`, `restart`) are rejected at validation time via `validate_docker_options()`, following the same pattern as `validate_extra_args()`.

The compose template hard-codes `restart: unless-stopped` on every engine service (and the nginx load balancer in multi-instance deployments). Containers therefore come back automatically after a host reboot or after a process crash, but a manual `docker stop` / `docker compose down` is still honored — which is what the bench teardown path relies on.

Matrix overrides work naturally via deep merge:
```yaml
matrices:
  deploy.gpu: "AMD Instinct MI350X"
  engine.llm.docker_options:
      security_opt:
        - seccomp=unconfined
```

### SGLang Matrix Entry Example

To benchmark with SGLang alongside vLLM, use a cross-product with the engine image. An empty string selects vLLM (no SGLang sub-config is created), while a non-empty string activates SGLang:

```yaml
matrices:
  cross:
    deploy.gpu: "NVIDIA GeForce RTX 5090"
    deploy.gpu_count: 1
    engine.llm.sglang.image: ["", "lmsysorg/sglang:v0.5.9"]
```

### SGLang Quantization for AWQ MoE Models

SGLang does not automatically detect AWQ quantization for MoE architectures. For AWQ-quantized MoE models, `--quantization moe_wna16` must be passed via `extra_args`.

## Data Flow

```
recipe.yaml
    |
    v
_load_raw_config(recipe_dir) -> raw dict
    |
    +-- load_recipe(): strips matrices, calls _validate_and_build()
    |       -> base Recipe (for bench/cloud commands that don't need matrix resolution)
    |
    +-- resolve_for_hardware(recipe_dir, gpu_name): expands full matrix,
    |       finds best combo matching gpu_name, deep_merges with base,
    |       calls _validate_and_build()
    |       -> hardware-resolved Recipe (for deploy local/ssh commands)
    |
    +-- enumerate_tasks(): reads matrices, expands via cross/zip:
            |-- expand_matrix() -> list of combinations
            |-- filter_combinations() -> filtered list (if --filter flags)
            |-- Variant(params=combo) -> typed variant
            |-- build_override() -> nested override dict
            |-- deep_merge(base, override) -> merged config
            |-- _validate_and_build() -> Recipe per combination
            v
        list[BenchmarkTask] (for bench command)

Recipe dataclass
    |
    v
build_engine_args(recipe.engine.llm, model_name, model_revision=recipe.model.revision) -> ["--flag value", ...]
    |
    v
generate_compose() -> docker-compose.yaml string
```
