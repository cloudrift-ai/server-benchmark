# Commands Architecture

## Layered Design

```
commands/bench ──► benchmark (config, tasks, execution)
commands/bench ──► deploy (DeployParams, deploy/teardown)
commands/bench ──► provisioning (cloud VM lifecycle)
commands/deploy ─► deploy (DeployParams, deploy/teardown)
commands/deploy ─► provisioning (remote setup, cloud VMs)
commands/vm ────► provisioning (create/delete instances)
commands/agent ─► agent (tracked skill runner and tool schemas)
commands/publish ─► publish (image naming, metadata, collision and digest gates)
```

**Dependency rule:** `commands/` is the CLI-only layer. All reusable business logic lives in top-level library packages:
- `emmy/recipe/` — recipe loading, dataclass types (`Recipe`, `LLMConfig`, etc.), engine flag mapping
- `emmy/deploy/` — compose generation, deploy orchestration
- `emmy/provisioning/` — VM types, SSH polling, shell helpers, cloud providers
- `emmy/agent/` — OpenAI-compatible tracked-skill runner, bounded tools, and tool schemas
- `emmy/publish.py` — the canonical serving-image name parser, model slug, Docker metadata gates, and publication
  runner
- `emmy/serving/release.py` — shell-free pinned serving-config parsing and the exact realization matrix shared by
  trace and eval
- `emmy/logging_setup.py` — CLI logging setup (`setup_cli_logging()`), plus `ensure_plugin_logging()` — makes emmy
  INFO logs visible when nothing configured logging (a bare vLLM entrypoint; called by `emmy.serving.register()`)
- `emmy/config.py` — the single owner of `os.environ` for all `EMMY_*` config vars. Typed getters
  (`tune_db_path`, `nvcc_flags`, `debug_enabled`, `dump_dir`, `tune_patience`, `bench_backends_raw`, `cubin_cache_dir`,
  …) read the env live; `set_nvcc_flags(cli_value, default)` holds the `--nvcc-flags` > env > command-default precedence
  that used to live in this CLI layer, so every callsite (CLI, programmatic, tests) shares it. The thin
  `compile.apply_nvcc_flags` / `compile.resolve_tune_db` wrappers just adapt argparse to it. (Provider/secret vars stay
  with `redact.py`; the dynamic `EMMY_<KNOB>` namespace stays with `compiler/pipeline/knob.py`, which borrows
  `config.knob_var` / `config.knob_raw`.)
- `emmy/redact.py` — `redact_secrets()`, `SecretRedactingFilter`, `install_redaction()` (attach the filter to a handler — must be a handler, not a logger, so child-logger records that propagate up are still redacted), `register_secret()` (call after resolving any secret from a CLI flag — `--hf-token`, `--api-key` — or env var so its value is added to the redaction set)
- `emmy/benchmark/` — config, logging, workload, tasks, execution, structured JSON results

## Layers

### `emmy/recipe/` — Recipe Library

Recipe loading, configuration dataclasses (`Recipe`, `ModelConfig`, `EngineConfig`, `LLMConfig`, `VllmConfig`,
`SglangConfig`, `BenchmarkConfig`), deep merge / hardware resolution / extra-arg validation, and engine flag mapping
(`VLLM_FLAG_MAP`, `SGLANG_FLAG_MAP`). `load_recipe()` returns a `Recipe` dataclass; all consumers use attribute access
(e.g. `recipe.engine.llm.tensor_parallel_size`).

### `emmy/deploy/` — Deploy Library

The central orchestration layer. Provides a single entry point (`run_deploy()` / `run_teardown()` and the lower-level
`deploy()` / `teardown()`) for deploying recipes to servers, plus compose / nginx generation, SSH and local-subprocess
transports, and the scale-out strategies (`DataParallelismScaleOutStrategy`, `ReplicaParallelismScaleOutStrategy`).
`DeployParams` carries the `recipe`, `gpu_device_ids`, etc. `run_deploy()` / `deploy()` accept an optional
`timer: PhaseTimer` that records per-step durations (see [Timing metrics](#timing-metrics)).

Standalone deploy commands use a post-health **smoke test** that branches on the recipe model config. Embedding
models (`model.task: embed`) POST
`/v1/embeddings` and require a non-empty finite vector with L2 norm in [0.9, 1.1]. Generative models use the chat
endpoint by default ("What is 2+2?" must contain "4"); base checkpoints set `model.smoke_test: completion` to test the
same arithmetic through `/v1/completions`. All paths share the retry, timeout, and log-dump loop. Benchmark
orchestration uses the same probe as a transport-readiness check but does not interpret the returned model content.
`parse_engine_load_phases()` extracts best-effort `weights_load` / `cuda_graph_capture` from container logs.

`model.revision` is the one immutable Hugging Face revision for a deployment. The model-download phase passes it to
`hf download`, and Compose passes the same revision to vLLM or SGLang; recipes must not duplicate it in `extra_args`.

**GPU visibility:** `generate_compose()` accepts a `gpu_device_ids` parameter to restrict GPU visibility via
`device_ids: [...]` instead of `count: all`. Used by bench when a task needs fewer GPUs than the VM has.

### `emmy/provisioning/` — Provisioning Library

VM lifecycle management and cloud provisioning. `VMConnectionInfo` is the connection dataclass; `wait_for_ssh()` does
provider-agnostic SSH polling. The `Host` / `LocalHost` / `RemoteHost` hierarchy is a sudo-gated command runner
(`LocalHost.run(sudo=True)` raises so local deploys can't modify the dev box). `provision_remote()` installs Docker, the
NVIDIA container toolkit, and optional NVIDIA driver/CUDA (rebooting and waiting for the host on driver/CUDA install).
`provision_cloud_vm()` / `delete_cloud_vm()` orchestrate cloud VMs over the CloudRift (REST API) and GCP (gcloud)
providers.

### `emmy/benchmark/` — Benchmark Library

Benchmark configuration (`load_config()` / `validate_config()`), per-run logging, task enumeration
(`enumerate_tasks()`), execution (`run_execution_group()` — times provisioning per group + deploy/bench/teardown per
task; task results are `(task, ok, timing)` triples), and structured results (`BenchmarkMetrics` / `SystemInfo`
dataclasses, `parse_benchmark_metrics()`, `compose_json_result()` / `compose_result()` — both take an optional `timing`
arg feeding the `"timing"` JSON key / `=== Timing ===` text section).

The benchmark library is an experiment-agnostic runner: it records complete client output, server logs, timing,
system information, and partial observations after failure. It treats execution and evidence-collection failures as
authoritative, but does not judge model output, metrics, backend selection, or scientific claims. A recipe may use a
small self-contained post-processing command, but cannot delegate result interpretation or report generation to a
script. Command recipes may opt into the single `command.strict` integrity contract for clean/content-addressed staged
inputs, required declared artifacts, and source/GPU/CUDA provenance. The complete boundary lives in
`emmy/benchmark/ARCHITECTURE.md`.

`run_benchmark_workload()` drives `vllm bench serve`. Embedding recipes (`model.task: embed`) bench with
`--backend openai-embeddings --endpoint /v1/embeddings` and drop `--random-output-len` (nothing is generated); the
output's labels (request/token throughput, E2EL percentiles — no TTFT/TPOT/ITL) already parse via
`parse_benchmark_metrics`' missing-field-tolerant label regexes.

### `planner/` — Planner Layer

Groups benchmark tasks into execution groups for VM allocation. The abstract interface defines `BenchmarkTask` (one
recipe+variant combination, with run-directory management helpers), `ExecutionGroup` (tasks sharing one VM; its `label`
property returns a unique label like `rtx5090_x_8` or `rtx5090_x_8_r01` when an index is set), and the
`BenchmarkPlanner` ABC (`plan(tasks) -> list[ExecutionGroup]`).

`GroupByModelAndGpuPlanner(gpu_concurrency=1)` groups by `(model_name, gpu_name)` so the same model on the same GPU
shares a VM and reuses cached weights; max gpu_count determines VM size, tasks sorted descending. With
`gpu_concurrency > 1`, each group is split into up to N sub-groups via round-robin, each provisioning its own VM (trades
weight-cache reuse for wall-clock time).

### `commands/` — CLI Layer (thin handlers only)

Each command module contains only argparse registration and `handle_*` functions that delegate to library packages. CLI handlers use `asyncio.run()` to bridge sync argparse entry points into async internals:

```python
def handle_foo(args):
    asyncio.run(_handle_foo(args))


async def _handle_foo(args):
    await ...
```

The compiler commands (`trace`, `compile`, `run`, and `tune`) share the same input loader and model-adapter selector.
They accept a Hugging Face model, debug Graph IR, or inline `--code`; `causal-lm` is the default and
keeps the existing Transformers path. `dit` delegates to the Diffusers block adapter in `compiler/trace/dit.py`; it
requires `--layer`, accepts the checkpoint's layers 0-27, and rejects dynamic shapes in v1. `run --bench` and
`tune --bench` include the adapter in the isolated worker's reconstruction payload, so eager PyTorch, `torch.compile`,
and Emmy always rebuild the same module and example inputs. Inductor compiles with
`fullgraph=True, mode="max-autotune"`; its output must match eager on the same inputs at `rtol=atol=1e-3` before its
latency is accepted. `run --strict` makes every requested backend, captured timing, exact pin, and direct
Emmy-vs-eager proof authoritative. It records max/mean/relative error in `--json` and exits
nonzero on any missing or failed evidence. Dynamic-shape parsing, quantized architecture twins and
their in-graph storage algebra, sliding-window stamps, and the guarded `trust_remote_code` fallback therefore behave
identically for all four commands (see `compiler/ARCHITECTURE.md`, "Quantized checkpoints").
For a single-layer trace, the loader derives a missing attention `layer_type` from
`config.layer_types[self_attn.layer_idx]`. Rotary modules keyed by that attention label supply one `(cos, sin)` tuple;
modules with independent rotary keys (for example DeepSeek V4's `main` / `compress`) supply the complete mapping.

The trace/tune handlers delegate working-golden inventory construction, target reconstruction, proposal measurement,
and atomic ranking persistence to `compiler/pipeline/search/working_golden.py`. Scoped exact knob pins shared by
`run` and working-golden tuning live beside that search lifecycle in `compiler/pipeline/search/pins.py`; command
handlers retain only the workflow's argument validation and user-facing error/reporting.

`emmy trace MODEL -o PATH` lowers through post-fusion Loop IR and writes one self-contained golden YAML inventory.
The YAML embeds stable frontend Torch IR programs and emits one target row for every post-fusion kernel occurrence;
structurally identical occurrences are not collapsed and a missing cache key never drops a target. A target uses
frontend provenance origins when that selector is non-empty and unique. Otherwise the document embeds its standalone
Loop IR slice in `loops` and selects that fallback by index. Flash score producers absorbed into their consumer are
stored as part of that one fused target rather than as a second kernel. Trace records neither knobs nor timings,
refuses replacement, and never writes a traced Graph JSON or provenance sidecar. Quantized traces store their
checkpoint-declaration digest in the same YAML.

`emmy trace LOCAL_CHECKPOINT --serving-twins --serving-config PATH -o PATH` is the release inventory variant. It
calls the config/allocation-metadata-only `serving.twins.capture_twin_graphs` path, combines every distinct
pre/post/expert and coded rate-profile kernel into one document, and stores each structural target once as symbolic
Loop IR. The pinned env supplies the model provenance and complete realization matrix: decode, prefill, M=1, extra
warm shapes, symbolic fallbacks, and standard/precision-trading input pin regimes. Each target receives a
`realizations` array with those named bindings and explicit registered input pins; trace no longer accepts an
independent serving-shape surface. A static-only release is accepted
only when the same env proves that no wider or symbolic path is reachable. The resulting working file is consumed
directly by `tune --golden-file` and verified by `run --golden PATH [--target NAME]`.

The in-model audit normally uses those serving twins. An architecture that cannot fit their external-attention ABI is
dispatched through a sound config-only provider instead: DeepSeek V4 traces one complete representative decoder layer
per attention/MLP pairing at sequence length 512, retaining its HCA/CSA compressor and hyper-connection operations.
This provider is audit-only; `emmy trace --serving-twins` does not claim a DeepSeek serving split it cannot execute,
and the file-scoped audit rejects a serving matrix the fixed full-layer provider cannot represent.

`emmy tune --golden-file PATH` consumes embedded programs directly. Realizations sharing one symbolic target are
specialized from their named bindings and grouped by target, bindings, and input pins. Knob-bearing
realizations are measured in file order before MCTS and written back as working-only `ranking` metadata.
`--max-candidates N`
is a per-tuned-kernel budget: every supplied proposal reserves one slot, while an MCTS DB cache hit does not spend a
remaining live-measurement slot. A traced target normally maps to one post-fusion kernel, but lowering may materialize
several CudaOps; conflicting multi-CudaOp knob rows are reported as ambiguous instead of being assigned an invented
winner. Proposal feedback is written immediately after measurement, before MCTS, so an interruption preserves it.
The final winner annotation is emitted only when one directly searched observation supplies both the knobs and cost;
the later greedy deploy replay cannot be paired with the search reward. The ranking pass stays at tune's fast compile
flags and never writes the trusted
`emmy_us` / `cublas_us` fields. With multiple homogeneous `--devices`, independent working-file targets share one
event loop, backend-slot queue, DB, and prior, so a file of one-kernel trace entries can use every selected GPU.
When the file has multiple targets, `--dump-dir` receives one stable indexed subdirectory per target; `--output` is
rejected because a single CUDA-IR path cannot represent several independent results. The command also resolves and
rejects any `--golden-file` inside the canonical repository `search/goldens/` tree, including symlink aliases.
With `--bench`, each target's `62_kernel_bench.json` records whether an eager reference was available and the
non-fatal accuracy verdict alongside the deployable O3 timings. A null verdict proves correctness only when the
reference-available field is true; reference-free Loop slices remain timing evidence rather than accuracy evidence.

`emmy compile --golden-file PATH --golden NAME` and `emmy run --golden PATH [--target NAME]` are the verification
counterparts. They resolve targets only in the explicit working YAML and compile its exact provenance or Loop IR,
without canonical-corpus or live-card filtering. `run` visits every distinct target sequentially in the current
process unless `--target` narrows the file to one exact or unambiguous substring match. With several targets,
`--json DIR` writes one readable JSON record per target; there is no repeat or child-process orchestration layer.
Invoke `emmy run` again when independent process observations are required. Inventory and proposal rows select the
graph but are not trusted as automatic A/B pins; only verified rows with paired measurements auto-pin, while a
proposal is tested explicitly with `run --bench --ab 'KNOBS…'`. Embedded Loop IR stores stable algebra rather than
derived structural stamps, so `run --golden` replays it through the full compiler pipeline. A direct `run --ir` input
remains a stage-complete artifact and runs only the later passes.

For a fair hybrid-vs-MCTS comparison, both working files start from the same inventory-only trace: do not copy verified
knob rows into either baseline as proposals. Canonical goldens remain the common implicit deploy context for both runs.

`emmy eval golden GOLDEN_YAML --serving-config PATH` is the release audit. The env must name that exact canonical
file. The command validates the nested schema and model provenance, requires the live GPU to match both the config
and YAML, proves that every structural target has every config-derived realization, reproduces the recorded rows,
and re-traces the exact static/symbolic precision matrix. Any missing realization, DRIFT, GAP, or compile failure is
a non-zero release failure. Model, revision, GPU, and serving widths therefore have no independent audit flags.

**Command modules:** `commands/bench/` (with `GitCommitter` for incremental result commits),
`commands/deploy/{ssh,local,cloud}.py` (`deploy ssh` auto-detects the remote GPU via SSH, `deploy local` the local GPU
via PCI sysfs, both resolve the matrix + apply a scale-out strategy; `deploy cloud` uses the recipe's `deploy.gpu` for
matrix resolution), `commands/teardown.py`, and `commands/vm/` (a CLI handler per provider). Each exposes a `handle_*`
and a `register_*` function.

## Data Flow

```
Recipe dirs (positional args)
    |
    v
enumerate_tasks() -> list[BenchmarkTask]
    |
    v
Create per-recipe run directories:
    +-- for each recipe_dir: create_run_dir(recipe_dir)
    +-- assign task.run_dir per task
    +-- write tasks.json per run_dir
    |
    v
GroupByModelAndGpuPlanner.plan() -> list[ExecutionGroup]
    |
    v
asyncio.gather(*groups)  -- each group runs as async task:
    |
    v
provision_cloud_vm() -> VMConnectionInfo
    |
    v
For each task in group:
    +-- set gpu_device_ids if task.gpu_count < group.gpu_count
    +-- deploy(DeployParams) -> compose up
    +-- run_benchmark_workload()
    +-- capture raw server log and save raw or partial results
    +-- on_task_done callback (--commit-results: git add + commit + push)
    +-- teardown() (skipped with --no-teardown)
    |
    v
delete_cloud_vm(conn.delete_info) (skipped with --no-teardown; writes instances.json)
```

## Timing metrics

`emmy/timing.py` provides `PhaseTimer` — an ordered collector of `phase -> seconds` durations, threaded by
mutation through the async deploy/bench chain (so `run_deploy()` keeps its `bool` return). Each phase is wrapped in
`async with timer.ameasure(name)` (sync `measure()` also exists); the elapsed is recorded even if the body raises,
and a `[timing] <name>: 12.3s` line is logged. Phase-name constants live in `timing.py`.

**Measured phases:** provisioning `vm_provision`, `remote_provision`; deploy `image_pull`, `model_download`,
`model_load_and_warmup` (the `compose up --wait` window — covers weight load into GPU + CUDA graph capture + warmup),
`smoke_test`; plus `benchmark`, `teardown`, and `command` (command recipes). After `model_load_and_warmup`,
`orchestrate.py` scrapes `docker compose logs` and runs `log_phases.parse_engine_load_phases()` +
`log_phases.decompose_model_load()` to break that window into a **non-overlapping** set of sub-phases that sums to the
parent: `startup` (container + CUDA init + imports) / `weights_load` / `torch_compile` (engine kernel / torch.compile
time) / `engine_warmup` (profile + KV cache + warmup, derived from vLLM's `init engine … took X s` line) /
`cuda_graph_capture`. When the engine-init line isn't present (older vLLM / SGLang) the unattributed time collapses into
a single `other` remainder. All of these are a breakdown of `model_load_and_warmup`, so they are **excluded from
`total`** (which would otherwise double-count). Near-zero phases
(`container_cleanup`, health poll, `system_info`) are intentionally not timed, so the phases don't fully sum to raw
wall-clock.

**Attribution:** provisioning runs once per `ExecutionGroup` (shared VM) but is seeded into each task's timer, so every
task's result reflects what it cost to stand up its host. `vm_provision` is omitted for fixed/local hosts (no VM
created). `timing["benchmark"]` is wall-clock (incl. the docker bench-client startup), distinct from
`metrics.benchmark_duration_s` (the server-measured window).

**Output:** `bench` persists timing into each task's `.json` (`"timing"` key) and `.txt` (`=== Timing ===` section) and
prints a per-task `TIMING` table in the end-of-run summary (`commands/bench/__init__.py::_format_timing_table`).
Standalone `deploy local/ssh/cloud` are display-only (no results dir) — they log the `PhaseTimer.format_table()`
breakdown at the end.

### Fixed-host mode (`--local` / `--ssh`)

When the user supplies pre-allocated hosts via `--local` and/or `--ssh user@host[:port]`,
`bench` skips cloud provisioning entirely. `benchmark/fixed_hosts.py` resolves each host
into an `AllocatedHost(conn, gpu_name, gpu_count)` (GPU detected via PCI sysfs through the
existing `detect_local_gpus()` / `detect_remote_gpus()` helpers), then validates that every
planned `ExecutionGroup` can run on at least one supplied host. The dispatcher
`_run_groups_on_hosts()` routes each group to a compatible idle host (locking per-host so
each runs at most one group at a time) and calls `run_execution_group(...,
preallocated_conn=host.conn)` — which skips both `provision_cloud_vm()` and
`delete_cloud_vm()`. `provision_remote()` (Docker, NVIDIA Container Toolkit, optional
driver/CUDA pinning) still runs and is idempotent, so already-provisioned hosts are a
fast no-op while bare VMs (e.g. straight from `vm create`) get set up on first use.

## CLI Command Tree

```
emmy
+-- deploy
|   +-- local    -- deploy locally via docker compose
|   +-- ssh      -- deploy to remote server via SSH
|   +-- cloud    -- provision cloud VM + deploy via SSH
+-- bench        -- deploy + benchmark + teardown on cloud VMs
+-- serve        -- vllm serve with the emmy embedding plugin (optional one-shot bench)
+-- teardown     -- clean up VMs left by bench --no-teardown
+-- publish      -- validate, tag, and push the canonical image named by one recipe
+-- vm
    +-- create
    |   +-- gpu        -- name a GPU from the hardware table (orchestrator: retries + fallback)
    |   +-- gcp
    |   +-- cloudrift
    +-- delete
        +-- gcp
        +-- cloudrift
```

## CLI Reference

`emmy --version` prints the installed distribution version (`unknown` when running straight from a source checkout
with nothing installed).

Everywhere a recipe directory is accepted — `deploy local` / `ssh` / `cloud` via `--recipe`, and `bench`'s
positional arguments — a bare name with no path component instead selects one of the recipes bundled in the
installed package, copying it into the current directory first. An existing path always wins over a bundled name.
See [`emmy/recipe/ARCHITECTURE.md`](../recipe/ARCHITECTURE.md) for why the copy is mandatory.

### `emmy deploy local`

Runs `docker compose` directly on the current machine. Auto-detects the local GPU via PCI sysfs and selects the matching `matrices` entry from the recipe.

```bash
emmy deploy local --recipe <path> [--dry-run] [--teardown]
emmy deploy local --recipe <path> --gpu "..." --gpu-count N    # override detection
```

### `emmy deploy ssh`

Deploys to a remote server via SSH + SCP. Auto-detects the remote GPU and resolves the matrix the same way as `deploy local`. The remote host is responsible for having Docker + NVIDIA toolkit installed (or supplying `deploy.driver_version` / `deploy.cuda_version` in the recipe — see Recipe ARCHITECTURE).

```bash
emmy deploy ssh --recipe <path> --ssh user@host[:port] [--ssh-key ~/.ssh/id_ed25519] [--dry-run] [--teardown]
```

### `emmy deploy cloud`

Provisions a cloud VM and deploys via SSH. Requires `--gpu` and `--gpu-count` to select the matching matrix entry from
the recipe (no auto-detection — there is no host yet). When several providers offer a GPU, their hardware-table order
sets fallback preference; pass `--provider {gcp,cloudrift}` to restrict the search to one provider.

```bash
emmy deploy cloud --recipe <path> --gpu "NVIDIA H200 141GB" --gpu-count 8 [--provider gcp] [--name prefix]
```

### Hardware-Aware Deploy (Local / SSH)

Both `deploy local` and `deploy ssh` auto-detect the target GPU by scanning PCI sysfs device IDs (locally or over SSH) and select the matching `matrices` entry. If more GPUs are available than the recipe's base configuration needs, a scale-out strategy is applied (`--scale-out-strategy {data-parallelism,replica-parallelism}`, default `data-parallelism`).

### `emmy serve`

Serves an embedding model (or a generative chat model via `EmmyGenModel` with `--generate` — `--runner generate` +
fp16) through vLLM with the emmy plugin flags baked in (`serving/` plugin; needs the `serving` extra). Unrecognized flags forward to `vllm serve`; tokens after a literal `--` forward verbatim (emmy's
own flags are otherwise extracted wherever they appear — argparse REMAINDER swallows everything after MODEL, so the
handler re-parses it; see `commands/serve.py::_split_own_flags`). `--max-model-len 4096` (the dynamic-dim cap) is
applied for both engines unless overridden, so `--stock` is an apples-to-apples baseline. **`--revision` forwards to
vLLM *and* reaches the emmy runner** — the plugin composes `<repo>@<revision>` and every checkpoint read inside emmy
resolves that commit (see `serving/ARCHITECTURE.md`); without it a repo publishing several branches warns loudly and
takes its default. `emmy pull` and `emmy compile` / `emmy run` accept the same `<repo>@<revision>` spelling directly,
so a served rung can be reproduced off the CLI. Generative serving
defaults to **whole-step decode CUDA graphs** (a `--compilation-config` with `FULL_DECODE_ONLY` + capture sizes
laddered up to `--max-num-seqs` — sizes above the decode bucket capture the device-resident symbolic programs; see
`serving/ARCHITECTURE.md`); pass vLLM's own `--enforce-eager` to opt out (forced automatically when
`EMMY_GEN_DECODE_BUCKET=0`, and for MoE models — the routed expert dispatch host-syncs, which a whole-step capture
cannot record; `_is_moe_model` probes the LOCAL config cache as UX, a caller-supplied `--compilation-config` on an
MoE model is rejected with the reason, and `EmmyGenModel.__init__` carries the authoritative boot guard for probe
misses). Under `--speculative-config` the ladder is derived from the resulting
`query_len = num_speculative_tokens + 1`: dense candidates, each floored to a multiple of `query_len`, so that vLLM's
round-up to that multiple cannot push a step's padded width past the decode bucket and off the static decode twin
(`serving/ARCHITECTURE.md` carries the rule and its invariant). The emmy generative arm also defaults
`--gpu-memory-utilization` to **0.97** (its
cupy residents are invisible to vLLM's torch-only profiler, so the 0.90 line can fail the min-KV fit at long
model lens; stock keeps 0.90) and `--max-num-batched-tokens` to **the runner's prefill capacity + the decode
bucket** — the bucket-sized rider headroom is covered by the chunk+decode twin row split
(`serving/ARCHITECTURE.md`), so full chunk steps keep carrying their decode riders; an explicit value past that cap
is rejected. Capacity is the dynamic-dim cap unless `EMMY_GEN_PREFILL_CAPACITY` pins it lower (the activation-arena
lever for a card the weights nearly fill), and the default follows it down. `EMMY_SERVING_BATCHED=1`
embedding serving defaults `--max-num-batched-tokens` to `max_num_seqs × max_model_len` so scheduler steps can fill
the batch. A checkpoint whose compressed weights emmy's loader owns end to end (today: **EXL3**, trellis-coded) is
additionally presented to vLLM as **unquantized** through the `--hf-overrides` — vLLM carries no method for the
scheme and refuses the boot outright, while nothing in the engine needs one, since the runner owns every coded
weight and the one vLLM-owned parameter (`lm_head`) decodes to fp16 at load. Which schemes those are is the loader
band's call (`compiler/loader/quant.py::engine_config_overrides`), not the command layer's.

```bash
emmy serve Qwen/Qwen3-Embedding-0.6B --gpu-memory-utilization 0.8   # plugin server (Ctrl-C to stop)
emmy serve Qwen/Qwen3-Embedding-0.6B --bench --random-input-len 32  # start → bench → results → shutdown
emmy serve Qwen/Qwen3-Embedding-0.6B --bench --stock                # raw-vLLM baseline of the same bench
```

Without `--bench` the process execs `vllm serve` (signals flow to vLLM directly). With `--bench` the server runs as a
subprocess (logs to a temp file), `/health` is polled (`--health-timeout` seconds, default 1800 — first boot compiles
the model; raise it when a fresh-serving-shape compile runs longer, e.g. a new prefill-bucket/max-num-batched-tokens
combination on a big model, or the kill lands mid-compile and no pack is saved), then
`vllm bench serve` runs against it (`--max-concurrency` / `--num-prompts` / `--random-input-len` / `--bench-seed`) and
the server is torn down. The bench backend follows the model: embeddings hit `--backend openai-embeddings --endpoint
/v1/embeddings`; **`--generate`** hits `--backend openai --endpoint /v1/completions` with `--random-output-len`.

The vLLM child inherits an environment with this interpreter's bin dir prepended to `PATH` (`serve.py::_child_env`):
invoking `./venv/bin/emmy` by absolute path does not activate the venv, so the generative server's inductor-compile
subprocess would otherwise die with `FileNotFoundError: ninja` (ninja is pip-installed into that same bin). A
**multimodal `--stock` baseline** (e.g. gemma-4-12B) still needs `--language-model-only` passed through — stock vLLM
loads the full multimodal checkpoint and the vision encoder's budget check rejects the small `--max-num-batched-tokens`
the emmy arm runs at; the flag is a plain vLLM passthrough, only meaningful for the raw-vLLM arm.

### `emmy bench`

Loads each recipe, expands its matrix, allocates GPU hosts, delegates execution to the recipe's workload adapter,
captures observations, and tears down. Compatible tasks are grouped onto the same VM by the planner.

The orchestration layer is experiment-agnostic. It expands matrices, provisions resources, invokes the declared
workload adapter, records raw observations, retrieves declared artifacts, reports execution failures, and tears down.
It never decides whether request counts, model outputs, backend choices, performance values, or comparisons are
scientifically acceptable. It provides no semantic gate and runs no aggregate or post-processing script. Those
judgments belong to intelligent review of the complete run directory against the frozen recipe and protocol.

For inference recipes, the deployment probe checks only that the API returns nonempty JSON; it does not judge the
model's answer. The raw probe response appears in the task log, and a complete redacted server log is saved beside
each result before teardown so later review has the evidence that orchestration deliberately does not interpret.

The only automatic acceptance boundary is generic execution integrity. A nonzero task, provisioning, transport, or
required artifact failure makes `emmy bench` exit nonzero. Command JSON records the rendered command, exit code,
timing, system information, and the content-addressed staged-source manifest. `command.strict` rejects dirty selected
paths before transfer, makes every declared artifact authoritative, and requires source, GPU, and CUDA-compiler
provenance. Result-file collection still runs after a nonzero command so partial evidence is retained, and later
matrix tasks continue.

```bash
emmy bench recipes/*                                    # All recipes
emmy bench experiments/.../optimal_mcr_rtx5090          # An experiment
emmy bench recipes/* --filter "deploy.gpu=*5090*"       # Subset (fnmatch glob, AND across multiple --filter)
emmy bench recipes/* --gpu-concurrency 4                # Split each (model, GPU) group across up to N VMs
emmy bench recipes/* --no-teardown                      # Skip teardown; writes instances.json for later cleanup
emmy bench recipes/* --local                            # Run on this machine via ssh to 127.0.0.1
emmy bench recipes/* --ssh user@host1 --ssh user@host2  # Pre-allocated host pool (no provisioning, no teardown)
```

Results are stored in `{recipe_dir}/{timestamp}_{hash}/` — each recipe directory holds its own run directories alongside `recipe.yaml`.

### `emmy publish`

Publishes the local serving image named by one concrete inference recipe. The recipe image is the destination and
must match `cloudriftai/(vllm-emmy|1cat-vllm)-<model-slug>:<runtime-version>-<source-sha>`, where the source SHA is
7–12 lowercase hexadecimal characters. The model slug comes from the same `emmy.publish.model_slug()` implementation
used by `docker/vllm-emmy-serve/model_slug.sh`; `latest`, hardware tags, and qualification suffixes are rejected.

The local source must carry `ai.emmy.publish.family`, `ai.emmy.model.id`, `org.opencontainers.image.version`, and
`org.opencontainers.image.revision` labels matching the recipe destination. `--source-image` retags a local build
whose temporary name differs from that destination. A matrix is accepted only when it expands to one concrete
variant.

Before any mutation, the command checks the registry destination. An existing destination is accepted only when its
digest is already among the local image's `RepoDigests`; a different or unprovable digest is never overwritten. After
a push, the registry digest must appear on the local destination image. `--dry-run` performs every read-only gate and
prints the pending Docker commands; an actual push requires the explicit noninteractive `--yes` confirmation.

```bash
emmy publish recipes/MyModel --source-image local/my-model:baked --dry-run
emmy publish recipes/MyModel --source-image local/my-model:baked --yes
```

**`--local` note:** runs the workload over SSH to `127.0.0.1` (same code path as remote hosts). Requires a running SSH server on localhost and that `--ssh-key` (default `~/.ssh/id_ed25519`) is in `~/.ssh/authorized_keys`. Quick check: `ssh -i ~/.ssh/id_ed25519 $USER@127.0.0.1 echo ok`.

**Fixed-host mode:** when `--local` and/or `--ssh` are supplied, `bench` detects each host's GPU and verifies that every planned execution group can run on at least one host (matching `deploy.gpu` and sufficient `deploy.gpu_count`). Unsatisfied groups abort the run before any work starts. Fixed hosts are never deleted at the end of the run.

### `emmy teardown`

Cleans up VMs left running by `bench --no-teardown`. Reads `instances.json` from the run directory.

```bash
emmy teardown <run_dir> [--ssh-key ~/.ssh/id_ed25519]
```

### `emmy vm create / delete / audit`

Manages cloud GPU VM lifecycles directly. Instances are ephemeral — `delete` removes them entirely. Run `emmy vm create {gpu,gcp,cloudrift} --help` for full flag lists.

There are two `vm create` modes:

* **`gpu`** (recommended) — name a GPU from the hardware table; the orchestrator picks the provider and instance type, retries transient failures, and falls back to alternative candidates on capacity errors. Same code path as `deploy cloud` and `bench`.
* **`gcp` / `cloudrift`** — single-shot manual create. You pass the exact `--machine-type` / `--instance-type`. No retries, no fallback. Useful for debugging an exact instance shape or for instance types not yet in the hardware table.

```bash
# GPU-based (uses orchestrator: retries, candidate fallback, orphan cleanup)
emmy vm create gpu --gpu "NVIDIA H200 141GB" --gpu-count 2 \
  --ssh-key ~/.ssh/id_ed25519 --provider cloudrift

# Manual single-shot
emmy vm create gcp --instance my-vm --zone us-central1-a --machine-type a2-highgpu-1g
emmy vm delete gcp --instance my-vm --zone us-central1-a

emmy vm create cloudrift --instance-type rtx4090.1 --ssh-key ~/.ssh/id_ed25519.pub
emmy vm delete cloudrift --instance-id <id>
```

Automated jobs can require an exact physical GPU count and persist an interrupt-safe ownership lease:

```bash
emmy vm create gpu --gpu "NVIDIA H200 141GB" --gpu-count 1 --exact-gpu-count \
  --lease /tmp/onboard-vm.json --owner cloudrift-ai/emmy/123-1 --json
emmy vm delete lease /tmp/onboard-vm.json --owner cloudrift-ai/emmy/123-1
emmy vm audit lease /tmp/onboard-vm.json --owner cloudrift-ai/emmy/123-1
```

The lease records the provider deletion handle before readiness polling, then adds SSH connection details. Delete and
audit accept only the exact recorded owner and never enumerate unrelated provider resources.

CloudRift attach to a specific network with `--network <name>` (on `vm create cloudrift`, `vm create gpu`, `deploy cloud`, and `bench`). The name must exist in the target datacenter; omit to let CloudRift pick a public network.

**Extra authorized keys.** `--ssh-key` is the *private* key emmy connects with; its `.pub` is always installed in
the VM's `authorized_keys`. To grant additional people access, pass `--authorized-key PATH` (repeatable, on `vm create
gpu` and `deploy cloud`) — each points to one public key file. The authorized set becomes `[ssh-key's .pub] + [every
--authorized-key]` (CloudRift via the rent payload's `PublicKeys` list; GCP via newline-joined `ssh-keys` metadata).
Missing or empty `--authorized-key` files fail fast before any VM is provisioned; a missing `--ssh-key` `.pub` warns
(no `ssh-keys` metadata is set) instead of aborting, since OS-Login-only projects can still SSH without it.

**GCP OS Login.** On GCP the `ssh-keys` metadata above is **instance-level** (temporary — it dies with the VM, no
project-wide key needed). But a project whose common metadata sets `enable-oslogin=TRUE` makes GCP **ignore** instance
`ssh-keys` entirely, so emmy also pins `enable-oslogin=FALSE` in the **same** `--metadata` flag (a second
`--metadata` would overwrite the first; `enable-oslogin` goes first so the multi-line `ssh-keys` value stays last). The
instance value overrides the project one — *unless* an org policy **enforces** `constraints/compute.requireOsLogin`, in
which case instance metadata can't turn OS Login off and keys must be registered through OS Login
(`gcloud compute os-login ssh-keys add`). Check with `gcloud resource-manager org-policies describe
compute.requireOsLogin --effective --project=PROJECT`: `booleanPolicy: {}` (or absent) = not enforced = the
`enable-oslogin=FALSE` path works.

```bash
emmy vm create gpu --gpu "NVIDIA H200 141GB" --gpu-count 2 --provider cloudrift \
  --authorized-key ~/.ssh/alice.pub --authorized-key ~/.ssh/bob.pub
```

**GCP provisioning model.** `vm create gpu` defaults to the per-GPU model from `hardware.GPU_GCP_PROVISIONING_MODEL`
(`DEFAULT_GCP_PROVISIONING_MODEL = FLEX_START`). Pass `--provisioning-model {FLEX_START,SPOT,STANDARD}` to override it —
`STANDARD` rents an **on-demand** VM. Because this routes through the orchestrator, on-demand still gets the full
`config.yaml` treatment (CUDA image, large boot disk, service account, SSH-key metadata) — the manual `vm create gcp`
path does not. GCP-only; ignored for CloudRift candidates.

```bash
emmy vm create gpu --gpu "NVIDIA B200" --gpu-count 8 --provider gcp --provisioning-model STANDARD
```

#### Allocation strategy (shared by `deploy cloud`, `bench`, `vm create gpu`)

All three commands go through `provision_cloud_vm()` in `emmy/provisioning/cloud.py`. It enumerates preference-ordered
candidates from `hardware.GPU_INSTANCE_TYPES` and fans GCP entries across `GPU_GCP_ZONES`. Each candidate gets up to
`SAME_CANDIDATE_RETRIES` transient attempts. `CapacityExhausted` advances; `TerminalProvisionError` aborts. Without a
filter, fallback can cross providers in hardware-table order; `--provider` restricts the complete search.

Capacity-class signals recognized today: CloudRift HTTP 503/429 on rent, CloudRift `Inactive` terminal status / readiness timeout, GCP `ZONE_RESOURCE_POOL_EXHAUSTED` / `QUOTA_EXCEEDED` / `STOCKOUT` in `gcloud` stderr, and GCP `RUNNING`-status timeout. Both providers terminate VMs they created but couldn't bring to readiness, so orchestrator fallback does not leak orphan instances.

GCP project is inferred from `gcloud` config. CloudRift reads `CLOUDRIFT_API_KEY` and `CLOUDRIFT_API_URL` from the environment by default. **H200 on CloudRift** is only available on on-prem clusters — set `CLOUDRIFT_API_URL` to the on-prem endpoint (the public `api.cloudrift.ai` does not offer H200).

### `emmy agent`

Runs a tracked repository skill non-interactively through an OpenAI-compatible Chat Completions endpoint. The API key
must arrive through a one-use mode-`0600` file or inherited file descriptor and is removed from every tool subprocess.

```bash
emmy agent run --skill .claude/skills/discover-models/SKILL.md --prompt /tmp/task.md \
  --model Qwen/Qwen3.6-35B-A3B-FP8 --api-key-file /tmp/agent-key --output /tmp/result.json
emmy agent tools --output /tmp/emmy-agent-tools.json
```

Repository writes are limited to the workspace plus explicit `--allow-write` paths. The generated tool JSON comes
from the same definitions the runner sends to the model. See `emmy/agent/ARCHITECTURE.md` for the security and
workflow-ownership boundary.

### `emmy fit`

Fit an offline-prior weights artifact and cross-validate it, GPU-free. Two orthogonal switches — `--trainer
{linear,catboost}` × `--data {golden,freeze:<path>}` — of which only `linear` × `golden` (the incumbent trainer on the
golden dataset) exists today; other combinations exit with "not yet supported". `--samples N` (default 0:
coordinate-descent-from-seed, the incumbent practice), `--l2 λ` (the raw-space L2 penalty strength in the fit loss —
default the declared tie-breaker strength `fit/linear.DEFAULT_L2`, `0` disables; keeps a rank-flat weight magnitude
identified, the D_pow2_threads 686 incident), `--seed`, `--folds {op_family,gpu,both,none}` (default `both`),
`--features SPEC` (the feature view — comma-separated names, trailing `*` = prefix glob; default
`D_*,MMA_tier,MMA_acc_bits`, recorded in the metrics header and artifact provenance so two fits are only compared
under matching views; `fit/group.MATMUL_FEATURES` is a ready spec holding just the 52 features that can move a
matmul ranking — the rest are either constant within every pool or affine copies of a kept feature, so excluding
them is expressiveness-neutral), `--out DIR`
(default `_tune/fits/<timestamp>-<trainer>-<data>/`). Writes `metrics.json` — the deterministic per-run record two fits
are diffed by: `full_train` (the shippable artifact's per-golden dual ranks + per-card aggregates) and one `cv.<axis>`
block per fold axis (pooled holdout / train tables, per-card gap, per-fold detail) — and `weights.json`, the full-train
artifact in the shipped format; `--artifact [PATH]` additionally writes the artifact to PATH (no value: the
repo-checked `offline_weights.json` — the regenerate-the-shipped-weights flow, formerly the retired
`scripts/golden_knob_heuristics.py`). `emmy/commands/fit.py` owns the snippet-tracing golden case builder
(`build_golden_groups` — `pipeline/` must not import the tracer) plus the trainer wiring and file writing; the run
harness and fold/metrics machinery are library code in `emmy/compiler/pipeline/search/prior/fit/` (`run.py` /
`cv.py`), documented there and in the pipeline ARCHITECTURE's prior sections.

```bash
emmy fit                                  # linear x golden, both fold axes, metrics under _tune/fits/
emmy fit --folds gpu --out _tune/fits/ab  # leave-one-card-out only, fixed run dir for an A/B
```

## Experiments

Experiments are self-contained parameter sweeps in `experiments/{model}/{name}/`. Each directory contains a
`recipe.yaml`; benchmark output is local and ignored by default:

```
experiments/Qwen3-Coder-30B-A3B-Instruct-AWQ/optimal_mcr_rtx5090/
  recipe.yaml
```

```bash
emmy bench experiments/Qwen3-Coder-30B-A3B-Instruct-AWQ/optimal_mcr_rtx5090
```

## CI Benchmark Workflow

External developers can submit experiment configurations via pull requests. A maintainer triggers benchmarks by
commenting `/run-experiment` on the PR. That explicit command authorizes the workflow to commit its selected results
back to the PR branch; ordinary local and onboarding runs do not commit experiment output.

```
/run-experiment                                                       # Auto-detect: all experiments changed in the PR
/run-experiment experiments/MyModel/my_experiment                      # Explicit
/run-experiment experiments/MyModel/my_experiment --gpu-concurrency 2  # Split groups across 2 VMs each
```

Only users with **write** or **admin** access can trigger benchmarks. For fork PRs, "Allow edits from maintainers" must be checked for results to be pushed back to the fork branch (otherwise results are downloadable as workflow artifacts).

## Adding a New VM Provider

1. Create `provisioning/<provider>.py` with `create_instance()` -> `VMConnectionInfo | None` and `delete_instance()`. The function must raise `CapacityExhausted` on no-capacity errors and `TerminalProvisionError` on auth/malformed-request errors, and terminate any VM it created but couldn't bring to readiness before re-raising (see `emmy/provisioning/errors.py`).
2. Add CLI handlers in `commands/vm/<provider>.py` (the single-shot manual subcommand).
3. Register CLI in `commands/vm/__init__.py`.
4. Add entries to `hardware.py` `GPU_INSTANCE_TYPES` table. If the provider has zone-affinity, add `GPU_<provider>_ZONES` and teach `iter_candidates` in `provisioning/candidates.py` to fan out across them.
5. Add provider dispatch in `provisioning/cloud.py` (`_provision_candidate`) and `delete_cloud_vm()`.
