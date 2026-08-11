# TP8 runtime kernel audit protocol

Status: protocol-only. The repository validator is implemented; the rank-local vLLM operator/phase exporter and
manifest-to-reproducer integration are not. No runtime-derived TP8 claim is allowed until both exist and a measured
manifest from the exact deployment validates.

## Systems and fixed workloads

Run one tensor-parallel replica from the exact serving recipe. A100 EXL3 is a compatibility/refusal study until an
Emmy serving image can load the checkpoint, so it produces no kernel audit on refusal.

| System | Rank policy | Decode capture | Prefill capture |
| --- | --- | --- | --- |
| 8x A100 DeepSeek EXL3 | `uniform_tp` | input 256, output 256, concurrency 8, 40 requests | input 4096, output 1, concurrency 1, 20 requests |
| 8x H200 GLM-5.2 FP8 | `uniform_tp` | input 256, output 256, concurrency 32, 160 requests | input 8192, output 1, concurrency 1, 20 requests |
| 8x B200 GLM-5.2 NVFP4/EP | `rank_local_ep` | input 256, output 256, concurrency 32, 160 requests | input 8192, output 1, concurrency 1, 20 requests |

Generate requests with vLLM's random dataset, seed 0, temperature 0, ignored EOS, and the exact tokenizer revision.
Run one unprofiled warmup before each capture. Store the fully expanded command, generated request IDs, measured token
counts, and client output. Do not substitute natural-language prompts or hand-counted tokens.

```bash
vllm bench serve --backend vllm --host 127.0.0.1 --port 8000 --model MODEL \
  --dataset-name random --random-input-len INPUT --random-output-len OUTPUT \
  --num-prompts REQUESTS --max-concurrency CONCURRENCY --seed 0 --ignore-eos
```

Each decode window contains initial prefill work, and a prefill request with one output token contains one decode
step. The exporter must label every operator and CUDA launch `prefill`, `decode`, or excluded. Selection uses only the
named phase; the other phase and all remaining CUDA activity stay in whole-window accounting. A window boundary alone
is not phase isolation.

## Immutable provenance

Before capture, archive the following commands and outputs. Also download the pinned model's `config.json`, record its
SHA-256 digest, and record the expanded serving and client commands. A missing full revision or image digest closes
the gate.

```bash
git rev-parse HEAD
docker image inspect --format '{{index .RepoDigests 0}}' IMAGE
docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' IMAGE
nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.draw,power.limit --format=csv
nvcc --version
python -m pip freeze --all
```

The manifest records the model/config digest, vLLM and Emmy revisions, image digest, packages, CUDA/driver versions,
GPU UUIDs and operating state, exact rank list, parallel and rank policies, recipe, and expanded commands.

## Five required capture sources

The five sources are content-addressed artifacts. Shapes, layouts, ranks, phases, and quantization fields come from
instrumentation rather than manual entry. Torch and Nsight runs are separate fresh-server runs with distinct run IDs;
operator metadata is enabled in both so each profiler can be joined within its own run.

1. **Rank-local operator and phase metadata.** An instrumented vLLM image wraps the operator boundary before backend
   dispatch and emits JSONL on every rank. Each row contains profiler run ID, workload/request ID, phase, NVTX range,
   operator and kernel family, operand roles/shapes/element strides/layouts/dtypes, quantization method/backend, and
   reproducer source.

   ```bash
   EMMY_OPERATOR_METADATA_DIR=/artifacts/operator-metadata \
   EMMY_OPERATOR_METADATA_WORKLOAD=WORKLOAD EMMY_OPERATOR_METADATA_RUN_ID=RUN_ID \
     vllm serve MODEL SERVING_ARGS
   ```

   This exporter does not yet exist. Config arithmetic, Torch shapes without strides/layouts, or rank-0-only logs
   cannot replace it.

2. **Torch profiler.** Start a fresh instrumented server, warm it, bracket one fixed workload with the endpoints, and
   retain every worker trace and its matching operator JSONL.

   ```bash
   vllm serve MODEL SERVING_ARGS \
     --profiler-config '{"profiler":"torch","torch_profiler_dir":"/artifacts/torch",\
   "torch_profiler_record_shapes":true,"ignore_frontend":true}'
   curl -fsS -X POST http://127.0.0.1:8000/start_profile
   vllm bench serve FIXED_WORKLOAD_ARGS
   curl -fsS -X POST http://127.0.0.1:8000/stop_profile
   ```

3. **Nsight Systems.** Start another fresh instrumented server under CUDA-profiler-range capture. Preserve the report,
   SQLite export, per-launch trace, kernel summary, phase ranges, and matching operator JSONL from every rank.

   ```bash
   nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
     --capture-range=cudaProfilerApi --capture-range-end=stop \
     --output /artifacts/nsys/WORKLOAD vllm serve MODEL SERVING_ARGS \
     --profiler-config '{"profiler":"cuda"}'
   curl -fsS -X POST http://127.0.0.1:8000/start_profile
   vllm bench serve FIXED_WORKLOAD_ARGS
   curl -fsS -X POST http://127.0.0.1:8000/stop_profile
   nsys stats --report cuda_gpu_trace --report cuda_gpu_kern_sum --format csv \
     --output /artifacts/nsys/WORKLOAD /artifacts/nsys/WORKLOAD.nsys-rep
   ```

4. **Engine log.** Preserve complete logs from startup through shutdown for both profiler runs. Logs must identify the
   attention, GEMM/MoE, quantization, distributed, expert-parallel, and CUDA graph backends and all fallbacks.

5. **Workload client.** Preserve the expanded command, generated request identifiers, token counts, and full client
   output for warmup and measured runs. The manifest repeats all fixed workload fields.

## Reconciliation, selection, and reproducers

Within each profiler run, join by workload, rank, phase, NVTX/operator range, CUDA launch correlation, and operand
signature. Never join distinct profiler runs by request or correlation ID. Reconcile Torch and Nsight runs only by a
stable signature containing family, operator, operands, and quantization, then compare launch-count distributions.
The record stores the Torch and Nsight launch counts for that signature separately; both reconcile to the
selected-phase record count.

For every workload and rank, Nsight independently supplies whole-window CUDA time, selected model-forward phase time,
excluded CUDA time, and selected-phase launch count. Manifest records must exactly reconcile to those phase totals
and counts; therefore omitted kernels cannot shrink the denominator. Communication, copies, allocators, the other
model phase, and non-model activity remain visible in the whole-window/excluded accounting.

Under `uniform_tp`, a logical case has one record on every rank and identical operand/quantization signatures. Under
`rank_local_ep`, B200 expert routing may produce rank-local cases and data-dependent MoE shapes; cases are selected
and coverage is checked per rank, and the report shows their union plus per-rank incidence. Unsupported model-forward
kernels remain in the records and denominator with a failure reason.

For decode and prefill independently on every rank, sort cases by descending CUDA time and stable case ID, then select
supported cases until they cover at least 90% of the independent Nsight phase total. Ties use case ID. Failure to
reach 90% on any rank closes the gate. Every selected case has deterministic source and a SHA-256 digest.

The planned compiler interface is:

```bash
emmy trace --serving-manifest manifest.json --output working.yaml
```

It must consume validated measured operands and never infer TP8 sharding from model config. Until it exists, a valid
manifest is capture evidence but not a kernel benchmark. After implementation, tune selected programs with the common
bounded search and run five fresh exact `-O3` replays.

## Validation and reporting

Run the checked-in validator from the Emmy revision recorded by the manifest:

```bash
./venv/bin/python scripts/validate_serving_kernel_manifest.py manifest.json
```

The validator checks immutable provenance, distinct profiler runs, phase filtering, within-run/cross-run evidence,
independent Nsight time/count reconciliation, the declared rank policy, deterministic per-rank 90% selection, and
artifact digests. Protocol-only data, missing files, mismatches, unsupported hot coverage, or handwritten incomplete
metadata fail closed.

Report selected and unsupported cases, minimum and per-rank runtime coverage, whole-window CUDA share, per-rank
variation, and exact generated-program mapping. Do not call a result TP8-, FP8-, NVFP4-, EXL3-, or
serving-representative unless this gate passes for the exact named system.
