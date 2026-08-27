# Benchmark orchestration

The benchmark package expands recipe matrices, executes every experiment row, and preserves raw output without
judging or interpreting experiment data.

## Experiment record

An actual `emmy bench` invocation creates `<recipe_dir>/<YYYY-MM-DD_HH-MM-SS>/`. Dry runs never create, delete, or
modify a run directory. Every expanded row receives one collision-safe `*.experiment.yaml`; each declared result file
uses the same readable variant plus stable row ID before its task-relative name, so abbreviated variants cannot
overwrite each other. There is no JSON, TXT, task manifest, instance manifest, or inline aggregate result path.

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

The orchestrator may:

- expand and filter recipe matrices;
- group experiment rows by requested hardware and provision or allocate hosts;
- stage declared inputs and invoke the selected workload adapter;
- capture stdout, stderr, raw server logs, timing, system information, recipe metadata, and declared result files;
- retain partial raw evidence after failure and continue independent experiment rows;
- return a nonzero status for generic execution-integrity failures; and
- tear down resources or record retained instances.

The orchestrator MUST NOT:

- parse experiment measurements or interpret model responses;
- decide whether request counts, latency, throughput, accuracy, or coverage are acceptable;
- aggregate repeat measurements or compare outputs;
- recognize compiler, quantization, serving-backend, or publication-specific evidence;
- match required or forbidden log text;
- reject a filtered matrix for scientific reasons; or
- generate `RESULTS.md` or invoke result-validation or report-generation scripts.

Command records preserve timing, status, errors, and system information. For a staged command, Git provenance names
the invoking worktree revision and the dirty state of the exact declared stage paths; neither the installed package
nor a reused remote source tree supplies that provenance. `command.strict` requires those staged paths to be clean,
every declared result file, GPU identity, NVCC, and cuBLAS. A failed command still attempts to retrieve declared
results.

The repository `run-experiment` skill validates row coverage, keeps the latest records inside the archived raw-run
tree, writes a thoughtful interpretation grounded in the raw evidence, replaces the exact GPU platform's named Git
LFS raw-results archive, and commits it with the shared `RESULTS.md`. Other platform snapshots remain unchanged. The
timestamped raw directory stays local and ignored. Interpretation belongs to the skill's intelligent review, never
the runner or a repository script.

`commands.bench` owns orchestration, `execution` runs execution groups, `experiment_record` owns the typed record
schema and YAML serialization, top-level `system_info` owns the typed host schema plus the one shared generic/PCI
probe used by records and GPU detection, `command_workload` executes command rows, and `workload` invokes the inference
client and captures raw server logs.
