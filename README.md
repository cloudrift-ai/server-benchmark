<p align="center">
  <a href="https://pypi.org/project/emmy-ml/"><img src="https://img.shields.io/pypi/v/emmy-ml" alt="PyPI"></a>
  <a href="https://github.com/cloudrift-ai/emmy/actions/workflows/tests.yml"><img src="https://github.com/cloudrift-ai/emmy/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://discord.gg/cloudrift"><img src="https://img.shields.io/discord/1150997934113030174?label=Discord" alt="Discord"></a>
</p>

**Compile → Benchmark → Deploy** any LLM on any GPU. Optimized compiler, LLM benchmarking, and deployment stack. Optimize inference via kernel fusion, autotuning, and advanced scheduling. See the blog post: [*Outperforming vLLM (cuBLAS and FlashAttention) on Gemma4-12B*](https://www.cloudrift.ai/blog/optimizing-gemma-4-12b-rtx).

## Install

```bash
pip install emmy-ml          # the CLI, with the recommended recipes bundled
emmy --version
```

The compiler needs its own extra (`pip install "emmy-ml[compile]"` — torch, transformers, cppyy). To hack on emmy
itself, clone instead:

```bash
git clone https://github.com/cloudrift-ai/emmy.git
cd emmy && make setup
```

## Compile

A hackable PyTorch → Graph IR → CUDA compiler. Trace any `nn.Module`, fuse it into one kernel, run it, and inspect the emitted CUDA. See the blog post: [*A Principled ML Compiler Stack in 5,000 Lines of Python*](https://www.cloudrift.ai/blog/building-gpu-compiler-from-scratch-1).

```bash
# Compile a single operation
emmy compile -c "nn.RMSNorm(2048)(torch.randn(1,32,2048))"
# Benchmark kernel on a local GPU
emmy run --bench --profile -c "torch.nn.Softmax(dim=-1)(torch.randn(1, 28, 2048, 2048))"
# Trace a dynamic model layer into an unmeasured working golden for remote tuning
emmy trace Qwen/Qwen3-0.6B --layer 0 --dynamic seq_len@x:1 -o _tune/qwen3/working.yaml
# Measure proposed rows, then spend the remaining per-kernel budget on MCTS
emmy tune --golden-file _tune/qwen3/working.yaml --devices 0,1 --max-candidates 64
```

Layer-norm-style reduction (two reductions, broadcast subtract, elementwise chain) fused into single kernel:

```bash
emmy compile -c "
class LN(torch.nn.Module):
    def forward(self, x):
        m = x.mean(-1, keepdim=True)
        v = ((x - m) ** 2).mean(-1, keepdim=True)
        return (x - m) * torch.rsqrt(v + 1e-6)
LN()(torch.randn(64, 2048))"
```

Principled compilation stack with six IR stages, each printable on demand via `--ir <stage>`:

1. **Torch IR** — captures the FX graph as a 1:1 mirror of PyTorch's op set (`rmsnorm`, `linear`, `softmax`, ...)
2. **Tensor IR** — decomposes Torch ops into generic elementwise, reduction, indexing, and value-conversion primitives
3. **Loop IR** — lifts each primitive to a `LoopOp` and fuses
4. **Tile IR** — schedules kernels onto GPU
5. **Kernel IR** — materializes the schedule into framework-agnostic hardware primitives
6. **CUDA** — optimized CUDA code ready for `nvcc`

**Readable Schedule**: `emmy compile -c "nn.RMSNorm(2048)(torch.randn(1,32,2048))" --ir tile`
```
kernel k_rms_norm_reduce  inputs: rms_norm_mean_count, rms_norm_eps, x, p_weight  outputs: rms_norm
    in0 = load rms_norm_mean_count[0]
    in1 = load rms_norm_eps[0]
    Tile(axes=(a0:256=THREAD, a1:32=BLOCK)):
        x_smem = Stage(x, origin=(0, a1, 0), slab=(a2:2048@2)) async
        p_weight_smem = Stage(p_weight, origin=(0), slab=(a3:2048@0)) async
        StridedLoop(a2 = a0; < 2048; += 256):  # reduce
            in2 = load x_smem[a2]
            v0 = multiply(in2, in2)
            acc0 <- add(acc0, v0)
        v1 = divide(acc0, in0)
        v2 = add(v1, in1)
        v3 = rsqrt(v2)
        StridedLoop(a3 = a0; < 2048; += 256):  # free
            in3 = load x_smem[a3]
            in4 = load p_weight_smem[a3]
            v4 = multiply(in3, v3)
            v5 = multiply(v4, in4)
            rms_norm[0, a1, a3] = v5
```

**Optimized CUDA kernel**: `emmy compile -c "nn.RMSNorm(2048)(torch.randn(1,32,2048))" --ir cuda`

```c
extern "C" __global__
__launch_bounds__(256) void k_rms_norm_reduce(const float* x, const float* p_weight, float* rms_norm) {
    float in0 = 2048.0f;
    float in1 = 1e-06f;
    {
        int a1 = blockIdx.x;
        int a0 = threadIdx.x;
        float acc0 = 0.0f;
        __syncthreads();
        __shared__ float x_smem[2048];
        for (int x_smem_flat = a0; x_smem_flat < 2048; x_smem_flat += 256) {
            {
                unsigned int _smem_addr = __cvta_generic_to_shared(&x_smem[x_smem_flat]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                             :: "r"(_smem_addr), "l"(&x[a1 * 2048 + x_smem_flat])
                             : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        __shared__ float p_weight_smem[2048];
        for (int p_weight_smem_flat = a0; p_weight_smem_flat < 2048; p_weight_smem_flat += 256) {
            {
                unsigned int _smem_addr = __cvta_generic_to_shared(&p_weight_smem[p_weight_smem_flat]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                             :: "r"(_smem_addr), "l"(&p_weight[p_weight_smem_flat])
                             : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        for (int a2 = a0; a2 < 2048; a2 += 256) {
            float in2 = x_smem[a2];
            float v0 = in2 * in2;
            acc0 += v0;
        }
        __shared__ float acc0_smem[256];
        acc0_smem[a0] = acc0;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (a0 < s) {
                acc0_smem[a0] = acc0_smem[a0] + acc0_smem[a0 + s];
            }
            __syncthreads();
        }
        __syncthreads();
        float acc0_b = acc0_smem[0];
        float v1 = acc0_b / in0;
        float v2 = v1 + in1;
        float v3 = rsqrtf(v2);
        for (int a3 = a0; a3 < 2048; a3 += 256) {
            float in3 = x_smem[a3];
            float in4 = p_weight_smem[a3];
            float v4 = in3 * v3;
            float v5 = v4 * in4;
            rms_norm[a1 * 2048 + a3] = v5;
        }
    }
}
```

## Benchmark

```bash
emmy bench experiments/gemma-4-12B/*                                    # All Gemma experiments
emmy bench experiments/gemma-4-12B/gsm8k_mtp_rtx5090                    # A single experiment
emmy bench experiments/gemma-4-12B/* --filter "deploy.gpu=*5090*"       # Subset
emmy bench experiments/gemma-4-12B/* --gpu-concurrency 4                # Parallel VMs per GPU
emmy bench experiments/gemma-4-12B/* --local                            # On this machine
emmy bench experiments/gemma-4-12B/* --ssh user@host1 --ssh user@host2  # Pre-allocated hosts
```

External contributors: open a PR with an experiment under `experiments/{model}/{name}/`, then a maintainer triggers a cloud run by commenting `/run-experiment` on the PR.

## Deploy

```bash
# Remote server via SSH
emmy deploy ssh --recipe recipes/gemma-4-12B-it --ssh user@host

# Local Docker Compose
emmy deploy local --recipe recipes/gemma-4-12B-it

# Cloud (auto-provisions a VM)
emmy deploy cloud --recipe recipes/gemma-4-12B-it --gpu "NVIDIA H200 141GB" --gpu-count 8
```

`--recipe` also takes the bare name of a recipe bundled with the installed package (`--recipe gemma-4-12B-it`),
which copies it into the current directory first — `deploy` writes its compose file next to the recipe, and `bench`
its run directories. A path that exists always wins, so an edited working copy is never overwritten.

## Publish a serving image

The serving recipe pins the canonical immutable image reference. Validate the local image, its provenance labels,
and the registry collision before requesting publication approval; only then log in and perform the push:

```bash
emmy publish recipes/DeepSeek-V4-Flash-0731 --dry-run
emmy publish recipes/DeepSeek-V4-Flash-0731 --source-image local-baked-image --yes
```

Published references use
`cloudriftai/<runtime-family>-<model-slug>:<runtime-version>-<source-sha>`; see the
[prebuilt-serving-image architecture](docker/vllm-emmy-serve/ARCHITECTURE.md) for the release gates and labels.

## Serve (compiled embeddings via vLLM)

```bash
# vLLM's OpenAI shell (/v1/embeddings, tokenizer, scheduler, pooler) over emmy-compiled kernels
emmy serve Qwen/Qwen3-Embedding-0.6B

curl localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-Embedding-0.6B","input":"Hello"}'

# One-shot benchmark (vllm bench serve against the started server), and the raw-vLLM baseline
emmy serve Qwen/Qwen3-Embedding-0.6B --bench --random-input-len 32
emmy serve Qwen/Qwen3-Embedding-0.6B --bench --random-input-len 32 --stock
```

## Recipe

```yaml
model:
  huggingface: "org/model-name"

engine:
  llm:
    tensor_parallel_size: 8
    gpu_memory_utilization: 0.9
    context_length: 16384
    max_concurrent_requests: 512
    vllm:
      image: "vllm/vllm-openai:v0.23.0"
      extra_args: "--kv-cache-dtype fp8"

benchmark:
  max_concurrency: 128
  num_prompts: 256
  random_input_len: 8000
  random_output_len: 8000

# Cross-product: 3 GPUs × 2 concurrency configs = 6 variants
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
```

Generic workload (run any tool on the VM, pull back result files):

```yaml
command:
  stage: ["scripts"]
  run: |
    nvidia-smi --query-gpu=name,memory.used --format=csv > $task_dir/result.csv
  result_files: ["result.csv"]
  timeout: 60

matrices:
  deploy.gpu: "NVIDIA GeForce RTX 5090"
  deploy.gpu_count: 1
```

## Virtual Machine Management

```bash
# GPU-based allocation with an interrupt-safe ownership lease
emmy vm create gpu --gpu "NVIDIA H200 141GB" --gpu-count 1 --exact-gpu-count \
  --lease /tmp/emmy-vm.json --owner local-run --json
emmy vm delete lease /tmp/emmy-vm.json --owner local-run

# GCP
emmy vm create gcp --instance my-vm --zone us-central1-a --machine-type a2-highgpu-1g
emmy vm delete gcp --instance my-vm --zone us-central1-a

# CloudRift
emmy vm create cloudrift --instance-type rtx4090.1 --ssh-key ~/.ssh/id_ed25519.pub
emmy vm delete cloudrift --instance-id <id>
```

## Agent Skills

```bash
emmy agent run --skill .claude/skills/discover-models/SKILL.md --prompt /tmp/task.md \
  --model Qwen/Qwen3.6-35B-A3B-FP8 --api-key-file /tmp/agent-key --output /tmp/result.json
emmy agent tools # the exact model tool definitions as JSON
```

## Development

```bash
make test      # run pytest
make lint      # ruff check + format check
make format    # auto-fix
make wheel     # build the wheel into dist/
```

### Release

Bump `version` in `pyproject.toml` on `main`, then run the **Publish to PyPI** workflow — it takes the version from
there, and refuses to run if that version is already tagged. It lints, tests, builds, uploads to PyPI via trusted
publishing, and only then creates the tag and GitHub release, so a failed upload leaves nothing behind. Publishing
a GitHub release by hand works too; the tag must agree with `pyproject.toml`.

`scripts/prepare_dist.py` stages the tree for a distribution build: `--recipes` copies `recipes/*/recipe.yaml` into
the package (`make wheel` runs this), and `--readme` rewrites this file's repo-relative links to absolute GitHub
URLs, which the workflow runs because PyPI renders the README detached from the repo.

## Project Structure

- [.github/](.github/) — Pull-request checks, releases, cloud experiments, and model discovery/onboarding workflows
  (see [ARCHITECTURE.md](.github/ARCHITECTURE.md))
- [emmy/](emmy/) — Python package
  - [emmy.py](emmy/emmy.py) — CLI entrypoint
  - [logging_setup.py](emmy/logging_setup.py) — CLI logging configuration
  - [hardware.py](emmy/hardware.py) — GPU specs and instance type mapping
  - [agent/](emmy/agent/) — tracked-skill runner and bounded model tools
    (see [ARCHITECTURE.md](emmy/agent/ARCHITECTURE.md))
  - [detect.py](emmy/detect.py) — GPU detection via PCI sysfs (local and remote)
  - [redact.py](emmy/redact.py) — Secret redaction for logs and dumps
  - [commands/](emmy/commands/) — CLI layer (thin argparse handlers, see [ARCHITECTURE.md](emmy/commands/ARCHITECTURE.md))
    - [deploy/](emmy/commands/deploy/) — `deploy local`, `deploy ssh`, `deploy cloud` commands
    - [bench/](emmy/commands/bench/) — `bench` command
    - [vm/](emmy/commands/vm/) — `vm create/delete/audit` commands (GCP, CloudRift, owned leases)
    - [teardown.py](emmy/commands/teardown.py) — `teardown` command
    - [pull.py](emmy/commands/pull.py) — `pull` command (download HF model)
    - [trace.py](emmy/commands/trace.py) — `trace` command (PyTorch → Graph IR)
    - [compile.py](emmy/commands/compile.py) — `compile` command (decomposition → optimization → fusion → kernel/CUDA lowering)
    - [run.py](emmy/commands/run.py) — `run` command (compile + execute on CUDA backend, optional benchmarks)
    - [inspect_graph.py](emmy/commands/inspect_graph.py) — `inspect` command (graph summary)
  - [compiler/](emmy/compiler/) — PyTorch → Graph IR → CUDA compiler (see [ARCHITECTURE.md](emmy/compiler/ARCHITECTURE.md))
    - [graph.py](emmy/compiler/graph.py) — `Graph`, `Node`, `Tensor`, `Hints` container
    - [ir/](emmy/compiler/ir/) — per-dialect op definitions (torch / tensor / loop / kernel / cuda) (see [ARCHITECTURE.md](emmy/compiler/ir/ARCHITECTURE.md))
    - [trace/](emmy/compiler/trace/) — PyTorch/HuggingFace → Graph IR capture (see [ARCHITECTURE.md](emmy/compiler/trace/ARCHITECTURE.md))
    - [pipeline/](emmy/compiler/pipeline/) — rewrite engine + passes + dump hooks (see [ARCHITECTURE.md](emmy/compiler/pipeline/ARCHITECTURE.md))
    - [backend/](emmy/compiler/backend/) — numpy / loop / CUDA execution (see [ARCHITECTURE.md](emmy/compiler/backend/ARCHITECTURE.md))
      - [cuda/](emmy/compiler/backend/cuda/) — CUDA backend internals (see [ARCHITECTURE.md](emmy/compiler/backend/cuda/ARCHITECTURE.md))
  - [recipe/](emmy/recipe/) — Recipe loading, dataclass types, engine flag mapping (see [ARCHITECTURE.md](emmy/recipe/ARCHITECTURE.md))
  - [serving/](emmy/serving/) — vLLM out-of-tree embedding plugin (see [ARCHITECTURE.md](emmy/serving/ARCHITECTURE.md))
  - [deploy/](emmy/deploy/) — Compose generation, deploy orchestration
  - [provisioning/](emmy/provisioning/) — Cloud provisioning, SSH transport, VM lifecycle
  - [benchmark/](emmy/benchmark/) — Benchmark tracking, config, task enumeration, execution
  - [planner/](emmy/planner/) — Groups benchmark tasks into execution groups for VM allocation
- [recipes/](recipes/) — The recommended serving configuration, one per model — what `emmy deploy` runs
  (see [ARCHITECTURE.md](recipes/ARCHITECTURE.md); benchmark grids belong in `experiments/`)
- [docker/](docker/) — Custom image builds ([vllm-emmy](docker/vllm-emmy/) — vLLM + the emmy plugin;
  [vllm-emmy-serve](docker/vllm-emmy-serve/) — prebuilt per-model images: warmed cubins + baked model snapshot;
  [1cat-vllm-sm70](docker/1cat-vllm-sm70/) — source-pinned 1Cat-vLLM runtimes and request-time GPU caches for Volta)
- [experiments/](experiments/) — Benchmark parameter sweeps, self-contained recipe + committed results —
  what `emmy bench` runs
- [kernels/](kernels/) — Standalone CUDA kernel sources
- [docs/](docs/) — Docusaurus user-docs site (getting started, benchmarking, custom configurations, deployment)
- [tests/](tests/) — pytest tests (see [ARCHITECTURE.md](tests/ARCHITECTURE.md))
  - [compiler/passes/](tests/compiler/passes/) — compiler pass tests (see [ARCHITECTURE.md](tests/compiler/passes/ARCHITECTURE.md))
- [scripts/](scripts/) — Analysis and visualization scripts
- [utils/](utils/) — Standalone utility scripts
- [config.yaml](config.yaml) — Benchmark configuration
- [Makefile](Makefile) — Build automation
- [pyproject.toml](pyproject.toml) — Package metadata and tool config

## Contributing

1. Fork and branch from `main` (e.g. `feature/my-change`)
2. Follow [STYLE.md](STYLE.md) and per-directory `ARCHITECTURE.md` files
3. Add tests in `tests/` (see [tests/ARCHITECTURE.md](tests/ARCHITECTURE.md))
4. `make test && make lint` (use `make format` to auto-fix)
5. Open a PR against trunk

## License

Licensed under the [Apache License 2.0](LICENSE).
