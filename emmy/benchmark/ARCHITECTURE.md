# Benchmark orchestration

The benchmark package expands recipe matrices, executes every experiment row, and preserves raw output without
judging or interpreting experiment data.

## Experiment record

An actual `emmy bench` invocation creates `<recipe_dir>/<YYYY-MM-DD_HH-MM-SS>/`. Dry runs never create, delete, or
modify a run directory. Every expanded row receives one collision-safe `*.experiment.yaml`; there is no JSON, TXT,
task manifest, instance manifest, or inline aggregate result path.

`ExperimentRecord` is a typed dataclass schema. It is initialized before provisioning, serialized directly to YAML,
and atomically rewritten at each lifecycle transition. A handled failure therefore leaves a terminal row rather than
only a log. An interrupted process may leave `running`, which is explicit incomplete evidence. The common fields are:

- record version, UTC timestamp, and terminal status;
- task ID, stable row ID, experiment directory, kind, variant, and raw matrix parameters;
- Git revision and dirty flag;
- execution timestamps, stage, phase timings, error, and infrastructure lifecycle;
- hostname, OS/kernel, CPU topology, memory, per-GPU identity/state and driver, NVCC and cuBLAS versions, Docker, root
  filesystem, and uptime from the experiment host.

The record contains no serving metrics, benchmark values, workload output, rendered command, compose document,
repeat aggregation, comparison, or conclusion. Those values remain only in raw artifacts.

NVIDIA GPUs use a structured `nvidia-smi` query. PCI device identity supplies a vendor-neutral fallback, including
AMD cards, while `amd-smi` or `rocm-smi` output is retained when available. Missing probes remain null; the runner does
not substitute requested hardware for an unavailable live observation.

## Boundary

The orchestrator may expand/filter matrices, allocate hosts, stage declared inputs, run workloads, capture raw
client/server logs, enforce generic execution integrity, and retain partial evidence. It must not parse experiment
measurements, interpret model responses, decide scientific thresholds, compare outputs, or generate `RESULTS.md`.

Command records preserve timing, status, errors, and system information. `command.strict` requires a clean Git
worktree, every declared result file, GPU identity, NVCC, and cuBLAS. A failed command still attempts to retrieve
declared results.

The repository `run-experiment` skill validates row coverage, copies the latest records beside the recipe, writes a
thoughtful interpretation grounded in the raw evidence, replaces the Git LFS-backed raw-results archive, and commits
the archive, records, and `RESULTS.md` as one durable last-run snapshot. The timestamped raw directory stays local and
ignored. Interpretation belongs to the skill's intelligent review, never the runner or a repository script.

`commands.bench` owns orchestration, `execution` runs execution groups, `experiment_record` owns the typed record
schema and YAML serialization, top-level `system_info` owns the typed host schema plus the one shared generic/PCI
probe used by records and GPU detection, `command_workload` executes command rows, and `workload` invokes the inference
client and captures raw server logs.
