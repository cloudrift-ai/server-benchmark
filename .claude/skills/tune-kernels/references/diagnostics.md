# Tuning diagnostics and report contract

Use the existing CLI before writing ad-hoc scripts or SQL. Preserve the exact command, logs, JSON output, dump paths,
and target GPU for every finding.

## Triage a losing or failed kernel

Start with:

```bash
emmy eval variants --kernel <substring>
emmy eval failures
emmy eval online --dataset nodes --kernel <substring>
emmy eval online --dataset nodes --blame --ablate --kernel <substring>
```

For a specialized golden target, also run:

```bash
emmy eval golden --kernel <substring>
emmy eval offline --kernel <substring>
emmy eval online --dataset golden --kernel <substring>
```

Classify every meaningful loss:

1. **Search shortfall:** the best measured/replayed config exists, but the prior or patience does not reach it. Use
   variant rank, fork sibling regret, and per-feature blame. Keep the CLI's offline-prior and online-prior blocks
   separate: cold-start weight/feature errors and learned-model calibration errors require different fixes.
2. **Eligibility or optimization lockout:** the desired schedule family is never offered. Cite the responsible
   lowering/scheduler gate and explain which target property triggers it.
3. **Code generation quality:** the right execution tier is present but loses. Inspect emitted CUDA and profile the
   pinned target against the reference.
4. **Benchmark failure:** use the recorded error and shared failed knobs. Reproduce compile-only before spending a
   search budget on another run.

Confirm a suspected search shortfall by tuning only the reproducer with more patience without clearing useful state.
Diff before/after dumps with `emmy compare`; do not infer structural changes by comparing log text.

## Isolated O3 and NCU checks

Run the dumped target independently:

```bash
EMMY_NVCC_FLAGS= emmy run --ir <kernel>.torch.json --bench \
  --bench-backends eager,tcompile,emmy --ab "<fully realized knobs>" \
  --json _tune/<run>/o3-<kernel>.json
```

Append `--profile` for an NCU comparison when `ncu` is installed and performance counters are permitted. Use
occupancy, registers/thread, SM and DRAM throughput, LSU instructions, and shared-memory bank conflicts to test a
specific hypothesis. If counters are unavailable, record that limitation and continue with timing/source evidence.

Inspect source without a GPU when useful:

```bash
EMMY_KNOBS="<fully realized knobs>" emmy compile <kernel>.torch.json --ir cuda
```

Do not compare O1 tune-DB latency with O3 deployment latency. Re-run a surprising O3 result before reporting it.

## Whole-model validation

For a requested whole-model tune, produce a full eager / `torch.compile` / Emmy table after finalist selection. Label
an `emmy tune --bench` result as the deploy-evidence replay, not an arm's searched winner; use exact `emmy run --ab`
pins for arm conclusions. For a servable embedding model, additionally run matched `emmy serve <model> --bench` and
`emmy serve <model> --bench --stock` trials with identical request count, input length, concurrency, and seed. Skip
serving A/B for unsupported model types and state why.

## Canonical-golden decisions

Compare live O3 measurements within the same standard/fast-math lane and the same static/dynamic target. Copy the
winning `record_knobs` map verbatim, including every schedule family and explicit off spelling. Record the live
reference latency from the same measurement regime. Keep parity configurations as duplicate target entries; remove
an incumbent only after a repeated loss beyond the measured noise floor.

Generic traced inventory and working `ranking` metadata are analysis artifacts, not canonical deploy evidence.

## Findings report

Include:

- status, date, repository revision, exact hardware/device IDs, scope, dynamic hints, dtype/quantization, and commands;
- fairness controls for hybrid versus MCTS-only, including starting DB/prior hashes, golden source, measurement and
  wall-time budgets, run order, live measurement counts, and compilation lane;
- a candidate table with target, knobs, evidence/rationale, proposal status, measured knobs, O1 rank, and searched finalist;
- a per-target A/B table with both arms' O3 latency, reference latency, correctness, repeated-run range, and decision;
- whole-model and serving tables when applicable;
- one finding per root cause, ordered by deployable latency at stake, with symptom, evidence, root cause or
  distinguishing diagnostic, reproducer, and recommended fix;
- an offline-prior versus online-prior table for fork sibling regret, reachability, calibration, and labeled blame
  whenever search steering is implicated;
- promoted, tied, rejected, and unresolved candidates, plus the exact working/canonical artifact paths;
- workflow notes covering slow steps, retries/flakiness, multi-command detours, output friction, and a concrete CLI or
  skill improvement for each.

Use O1 values only as ranking evidence and label them. Use O3 values for performance conclusions. Never present a
single-layer table as a model result or aggregate across different target coverage.
