# Benchmark orchestration

The benchmark package executes recipe matrices and preserves their observations. It is deliberately ignorant of an
experiment's scientific question and permitted claims.

## Boundary

The orchestrator may:

- expand and filter recipe matrices;
- group tasks by requested hardware and provision or allocate hosts;
- stage declared inputs and invoke the selected workload adapter;
- capture stdout, stderr, complete raw server logs, timing, system information, recipe metadata, and declared result
  files;
- retain partial artifacts after failure and continue independent matrix tasks;
- return a nonzero status for generic execution-integrity failures; and
- tear down resources or record retained instances.

The orchestrator must not:

- interpret model responses or compare outputs;
- decide whether request counts, latency, throughput, accuracy, or coverage are acceptable;
- recognize compiler, quantization, serving-backend, or publication-specific evidence;
- match required or forbidden log text;
- reject a filtered matrix for scientific reasons; or
- run report-generation or result-validation scripts.

Generic execution integrity is the only automatic acceptance boundary. Process, provisioning, transport, and raw-log
collection failures are authoritative because they say whether the declared task executed and its evidence was
collected, not whether the evidence supports a claim.

Every command task records the rendered command, exit code, timing, system information, and declared artifact-transfer
outcomes in its JSON result. `command.strict` makes three generic integrity requirements fail closed: staged inputs
must be clean and content-addressed, every declared result file must be retrieved, and source, GPU, and CUDA compiler
provenance must be present. Dry runs do not require provenance from a host that was never contacted. These checks say
whether a command measurement is reproducible and complete; they do not interpret its output.

A recipe may declare a short post-processing command directly in its `aggregate.run` block. The command may arrange
or summarize files mechanically, but must stay readable in the recipe and cannot invoke an external result-analysis
script. A nonzero command or timeout is an execution failure. Complex interpretation and `RESULTS.md` writing belong
to an agent reviewing the complete run directory.

The inference adapter's pre-run probe is an API-readiness check only: a successful, nonempty JSON response permits
the benchmark regardless of its content. The response and complete server log are preserved for later review. The
standalone deploy commands may retain their model-specific smoke checks because they are outside benchmark result
acceptance.

`commands.bench` owns orchestration, `execution` runs execution groups, `command_workload` executes arbitrary command
tasks, and `workload` invokes the existing inference workload adapter. Result files remain raw evidence. Durable
recipe tests verify the intended configuration before measurement, and an agent evaluates every final run directory
against its protocol afterward.
