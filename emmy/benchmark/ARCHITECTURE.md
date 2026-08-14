# Benchmark orchestration

The benchmark package expands recipe matrices, executes every experiment row, and preserves observations without
judging the experiment's scientific claim.

## Experiment record

An actual `emmy bench` invocation replaces `<recipe_dir>/results/`. Dry runs never create, delete, or modify that
directory. Every expanded row receives one collision-safe `*.experiment.yaml`; there is no JSON, TXT, task manifest,
instance manifest, or inline aggregate result path.

The versioned record is initialized before provisioning and atomically rewritten at each lifecycle transition. A
handled failure therefore leaves a terminal row rather than only a log. An interrupted process may leave `running`,
which is explicit incomplete evidence. The common fields are:

- record version, UTC timestamp, and terminal status;
- task ID, stable row ID, raw matrix parameters, and fully expanded recipe;
- Emmy code digest, Git state, and command-stage source manifest when applicable;
- execution timestamps, stage, phase timings, error, and infrastructure lifecycle;
- hostname, OS/kernel, CPU topology, memory, per-GPU identity/state, GPU compiler/runtime, Docker, root filesystem,
  and uptime from the measured host;
- inference metrics and repeats or rendered command/exit/transfer observations; and
- paths to raw logs and declared command artifacts.

NVIDIA GPUs use a structured `nvidia-smi` query. PCI device identity supplies a vendor-neutral fallback, including
AMD cards, while `amd-smi` or `rocm-smi` output is retained when available. Missing probes remain null; the runner does
not substitute requested hardware for an unavailable live observation.

## Boundary

The orchestrator may expand/filter matrices, allocate hosts, stage declared inputs, run workloads, capture raw
client/server logs, parse generic benchmark labels, enforce generic execution integrity, and retain partial evidence.
It must not interpret model responses, decide scientific thresholds, compare outputs, or generate `RESULTS.md`.

Command recipes preserve the rendered command, exit code, timing, source manifest, and artifact-transfer outcomes.
`command.strict` requires clean content-addressed staged inputs, every declared result file, GPU identity, and a CUDA
or HIP compiler. A failed command still attempts to retrieve declared artifacts.

Human interpretation belongs to the repository `run-experiment` skill. It validates row coverage, moves the latest
records beside the recipe, writes the free-form report, and commits the raw results, records, and `RESULTS.md` as one
durable last-run snapshot.

`commands.bench` owns orchestration, `execution` runs execution groups, `record` owns the schema/parsers/atomic writer,
`command_workload` executes command rows, and `workload` invokes the inference client and captures raw server logs.
