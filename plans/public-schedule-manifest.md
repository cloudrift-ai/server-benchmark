# Public schedule manifest capture and replay

Status: postponed. Do not implement until this work is explicitly resumed.

## Goal

Make a complete schedule chosen during an ordinary compile reproducible without consulting goldens, the tune DB, or
the prior again.

The public workflow should be:

```bash
emmy compile MODEL_OR_IR --schedule-out schedule.yaml
emmy run MODEL_OR_IR --schedule schedule.yaml
```

`emmy compile` must use its normal deploy evidence hierarchy and write the choices from the successful compilation.
`emmy run` must replay those choices authoritatively. Tests must use the same capture/replay implementation, either
through the public CLI or through its shared Python API. CUDA accuracy tests should not depend on whichever schedule a
cold prior happens to select.

## Success criteria

1. A normal `emmy compile --schedule-out` writes every structural and schedule fork chosen on the final successful
   greedy attempt, including placement cuts, cross-CTA reduction partitions, and the schedules of every resulting
   kernel.
2. `emmy run --schedule` lowers the same program without consulting deploy evidence and selects the exact recorded
   option at every fork.
3. Replay fails before launch when the program differs, a recorded option is no longer offered, a new unrecorded fork
   appears, a recorded fork disappears, or a target is ambiguous. It never falls back to the prior or option 0.
4. Capture records only the successful retry path. A rejected large tile or retired structural choice from an earlier
   greedy attempt must not appear in the file.
5. The public commands, `CudaBackend`, and tests share one manifest model, exact-option matcher, and replay policy.
   There is no CLI-only or test-only scheduler.
6. All CUDA accuracy tests use an exact manifest. Tests whose actual subject is greedy selection remain unpinned but
   are named and marked as policy tests rather than relying on an undocumented exception.
7. Manifest replay remains lazy: selecting an exact row descends the existing fork tree and does not enumerate the
   complete schedule space.

## Boundary: a schedule manifest is not a golden

A golden record is reviewed per-GPU measurement evidence. It embeds or references a reconstructable program target,
records timings, participates in the deploy evidence hierarchy, and trains the offline prior.

A schedule manifest is an unmeasured record of one complete compilation path. It does not embed the program, contain
measurements, enter a tuning dataset, or affect an unpinned compile. It references its input by structural identity
and exists only to reproduce explicit user intent.

Do not extend the golden schema or make manifests another evidence tier. Reuse the existing structural identities,
knob codecs, and exact golden-row descent where their contracts agree.

## Public interface

### Capture

Add `--schedule-out PATH` to `emmy compile`.

- Capture runs as part of the same lowering that produces the requested CUDA output. Do not compile once to display IR
  and a second time to discover the schedule.
- The manifest is written only after lowering succeeds, using an atomic replacement of a temporary file. Refuse to
  overwrite an existing path unless the command already has an established explicit overwrite convention to reuse.
- Initially require a complete CUDA pass sequence. Reject a truncated `--ir` / `--passes` selection because it cannot
  record later kernel-lowering forks and would look complete while leaving `run` choices unpinned.
- The compile keeps using the live verified goldens, measured evidence, prior, and final option-0 fallback exactly as
  it does without `--schedule-out`. The manifest records the result, not why it won.
- Record the final successful greedy attempt after validity fallback and structural retirement. A failed compile
  writes no manifest.

### Replay

Add `--schedule PATH` to `emmy run` and the corresponding optional argument to `CudaBackend.compile`.

- Replay selects the manifest entry matching the input program identity, then supplies an exact decide callback to
  the ordinary `Run.resolve` loop.
- Replay bypasses the deploy evidence hierarchy. It must work with no tuning DB and with a deliberately opposing
  prior loaded.
- `--schedule` is incompatible with `--ab`: the former asserts one complete compilation, while the latter asks to
  compile additional rows. A later design can accept a baseline manifest for an A/B, but that is not needed here.
- Registered input pins recorded by the manifest are installed for replay. Equal ambient pins are harmless; a
  conflicting `EMMY_<KNOB>` / `EMMY_KNOBS` setting is an error rather than an undocumented precedence rule.
- `run --golden PATH --target NAME --schedule MANIFEST` may replay one selected working-golden target. Reject an
  unqualified multi-target `run --golden` unless the manifest covers every selected program entry; do not silently
  reuse one target's decisions for another.
- The file records the GPU and compilation context that produced it. A different target emits a warning and attempts
  exact replay; ordinary legality and kernel validation remain authoritative. This permits intentionally portable
  scalar test manifests while making performance portability claims explicit.

### Diagnostics

Both commands should report:

- the manifest path;
- the matched program identity;
- the number of structural and schedule decisions captured or consumed;
- the first drift with its rule, recorded kernel label, structural identity, and requested knob row.

Kernel labels and graph node IDs are for diagnostics only. They must not decide whether a manifest entry matches.

## YAML format

Use a small, strict, versioned YAML format. One file may contain several program entries so a test module can share a
fixture without embedding each program or creating one file per parameter cell. `emmy compile` normally writes one
entry.

Illustrative shape:

```yaml
format: emmy-schedule
version: 1
programs:
- identity: 4f57a8700b55...
  created_for:
    gpu_name: NVIDIA GeForce RTX 5090
    compute_cap: [12, 0]
    nvcc_flags: ''
  passes: [frontend/decomposition, frontend/optimization, loop/lifting, loop/fusion,
           lowering/tile, lowering/kernel, lowering/cuda]
  input_pins:
    FAST_MATH: false
  decisions:
  - rule: 018_cut
    target:
      identity: 6d8e...
      occurrence: 0
      label: k_sdpa_linear_reduce
    knobs:
      PLACE@a3: cut
  - rule: 020_schedule
    target:
      identity: a41c...
      occurrence: 0
      label: k_sdpa_reduce
    knobs:
      WORK: w1x8
      TILE@a3: mma_m16n8k16_f16_f32/f2x2/k2
      REDUCE@a3: ''
      STAGE@a3: d2/smem-tma
      RASTER: ''
```

The final field names should follow the implementation's existing pass and knob vocabulary; do not serialize Python
class names.

### Identities

- Program identity should reuse `Graph.structural_key()` and fold in the symbolic hints/bindings that scheduling
  reads. It must ignore cosmetic buffer and node names, weight values, and runtime-only objects.
- A Tile scheduling fork should reuse `deploy_identity`, which already combines the Fold tree with dtypes and axis
  extents.
- Define one generic fork-identity function for non-Tile decision sites from the rule name, the root op's structural
  or cache identity before the choice, and the relevant output shape/dtype fingerprint. Do not key replay on
  `ForkPoint.node_id`.
- Identical fork identities are disambiguated by deterministic occurrence within the program. The label is retained
  only to make failures readable.

### Decisions

- Store the complete canonical knob row selected at a fork, not a partial prefix and not the option's list index.
- Structural choices such as `PLACE=cut` and `REDUCE=gNk` must be present even though the parent is replaced and the
  value cannot be recovered from a child kernel stamp.
- Record registered input pins separately because they control which rows are offered. Do not copy arbitrary
  environment variables into the file.
- Exclude structural `S_*` and hardware `H_*` features: they are facts used to rank rows, not decisions.
- Audit any multi-option fork whose selected option has no stable knob row before landing the format. Give it a stable
  structural choice key or make capture fail loudly; never serialize an ordinal as a compatibility shortcut.

The loader rejects unknown top-level fields, duplicate program/fork keys, malformed knob values, and unsupported
versions. Do not add legacy decoders or migration shims; regenerate stale manifests when compiler structure changes.

## Shared implementation

### 1. Exact option selection

Extract the verified golden tier's exact recorded-row descent into one generic helper. It should:

- compare knob values through the registered codecs;
- descend compatible fork branches lazily;
- select exactly one concrete leaf or structural option;
- report missing and ambiguous rows without ranking anything.

Use this helper from both verified golden replay and schedule-manifest replay. Do not create another implementation in
the command layer or tests.

### 2. Successful greedy result

Introduce a small result value containing the terminal graph and the successful `Decision` trace. Refactor
`GreedyStrategy` so:

- its current `run()` remains the graph-returning compatibility entry point;
- one shared resolution method returns the graph plus final trace;
- retries discard their provisional traces;
- the no-prior final fallback also returns its trace;
- normal compilation behavior and ranking stay unchanged.

Enrich each `Decision` at the point where `Run.resolve` still has the pre-choice root with the stable fork identity and
diagnostic label needed by serialization. Do not reconstruct this information from final CUDA nodes.

### 3. Capture and replay policy

Place the manifest value objects, strict YAML loader/dumper, capture conversion, and replay decide callback in one
pipeline/search module. The command modules should only validate arguments, load/save the file, and report errors.

Replay consumes entries as `Run.resolve` asks questions:

1. Verify the program identity before lowering.
2. Match the next fork by stable identity and occurrence.
3. Select the exact recorded option through the shared lazy matcher.
4. Mark the entry consumed.
5. At terminal, reject unconsumed entries and any live fork not represented in the manifest.
6. Run the existing kernel validation normally; a recorded but now invalid schedule is drift, not a reason to retry.

`Pipeline.run()` should keep its current return type. Add the smallest shared entry point needed by `emmy compile`,
`CudaBackend.compile`, and tests to request capture or replay without bypassing `GreedyStrategy`.

## Verification milestones

### Milestone 1: characterize the decision surface

- Capture decision traces for a scalar elementwise kernel, a regular MMA contraction, fused SDPA, a placement cut,
  split-K partial/finalize kernels, and a multi-output computed-operand graph.
- Prove every multi-option decision has a stable row. List and fix any unkeyed fork before defining schema version 1.
- Confirm that automatically replayed identical structural sites do not require duplicate manifest entries and that
  every resulting kernel still receives its own schedule entry.

Gate: a written inventory explains every decision that must be serialized; no ordinal choices remain.

### Milestone 2: manifest model and exact replay

- Unit-test strict YAML round trips, deterministic key ordering, empty/off values, axis-scoped keys, multiple program
  entries, duplicate rejection, unsupported versions, and atomic no-overwrite output.
- Unit-test exact lazy descent without materializing a large schedule space.
- Replay regular MMA, SDPA, a placement cut, split-K, and heterogeneous multi-kernel rows.
- Add drift tests for the wrong program, a missing/extra fork, an unavailable row, ambiguous identity, conflicting
  input pins, and an invalid target schedule.
- Load an opposing fake prior and DB row during replay and prove neither changes the selected graph.
- Compile twice and assert the replayed final knob rows and CUDA sources equal those captured from the original
  compile.

Gate: the shared Python API captures and replays complete graphs without command-layer help.

### Milestone 3: public commands

- Add parser and subprocess tests for `compile --schedule-out` and `run --schedule` across model/`--code` and frontend
  JSON inputs.
- Verify IR output and manifest output can be requested together without a second compile.
- Verify capture reflects whichever verified record, DB evidence, prior, or fallback the normal compile selected.
- Verify malformed, stale, conflicting, and incomplete manifests produce concise non-zero failures before execution.
- Verify existing compile/run behavior is byte-for-byte unchanged when neither argument is supplied.

Gate: a manifest produced by the public compile command reproduces the same kernel set and CUDA through public run
with the tune DB and online prior disabled.

### Milestone 4: migrate CUDA accuracy tests

Inventory every CUDA compile whose assertion is numerical correctness under `tests/compiler/e2e/`. Migrate by
contract rather than applying one global pin:

- Schedule-specific regression tests construct a small in-memory manifest with the exact row under test.
- Canonical and parameterized accuracy tests load module-level YAML fixtures keyed by program identity. Group several
  small programs in one file when that is clearer than one file per case.
- Full TinyLlama/Qwen blocks and full attention tests use checked-in whole-program manifests containing every
  placement, split, and kernel schedule decision. They must not consult a prior during the accuracy lane.
- Architecture-specific MMA/TMA tests keep their existing capability markers and use the corresponding recorded row.
- Portable correctness tests prefer conservative scalar rows; target mismatch is allowed only when exact replay and
  ordinary legality validation succeed on the live GPU.
- Replace `_chain_tile_pins`, broad `pinned_knobs` contexts, and other graph-wide accuracy pins. Retain env pinning in
  tests whose subject is env parsing, enumeration narrowing, or CLI A/B behavior.
- Move or mark tests whose subject is greedy policy selection so their intentional unpinned compile is explicit.
  Their numerical check may remain as a safety assertion, but they are not part of the deterministic accuracy lane.

Add one shared accuracy compile helper that accepts a manifest and calls the same backend replay entry point used by
`emmy run`. It must assert complete consumption. Do not add a helper that picks the first legal row or regenerates a
manifest during the test: that would merely hide greedy selection under another name.

Gate: searching the CUDA accuracy lane finds no unexplained direct greedy compile or broad schedule env pin, and the
suite remains deterministic with the tune DB and prior replaced by deliberately bad evidence.

### Milestone 5: documentation and cleanup

- Add “schedule manifest” to `GLOSSARY.md`.
- Add only a high-level policy to `AGENTS.md`: CUDA accuracy tests normally replay a complete schedule manifest;
  intentional greedy-policy tests are the explicit exception. Keep flag syntax, examples, manifest semantics, refresh
  instructions, and the distinction from golden evidence out of `AGENTS.md`; route those details through the README
  architecture index to the command, pipeline, and test documentation below.
- Add a concise `README.md` quickstart showing `compile --schedule-out` followed by `run --schedule`, and route readers
  to the detailed compiler/pipeline documentation. Explain that this captures the choices made by the active deploy
  evidence hierarchy; it does not tune or measure them.
- Document argument validation, supported input combinations, output ownership, and error reporting in
  `emmy/commands/ARCHITECTURE.md`.
- Document manifest identity, successful-attempt capture, authoritative exact replay, drift, and separation from
  golden evidence in `emmy/compiler/pipeline/ARCHITECTURE.md`. The deploy evidence hierarchy section must state
  clearly that manifest replay bypasses the hierarchy because it is explicit user input, rather than becoming a new
  tier.
- Update `emmy/compiler/ARCHITECTURE.md` with the persistence boundary: the input graph remains external, while the
  manifest stores program/fork structural identities and decisions only. Explain why node IDs and rendered kernel
  names are diagnostic rather than matching keys.
- Update `emmy/compiler/backend/ARCHITECTURE.md` and `emmy/compiler/backend/cuda/ARCHITECTURE.md` for the shared
  compile/replay entry point, the exact-manifest validation before launch, and `CudaBackend.compile`'s manifest
  argument. Do not imply that NumPy or Loop backends have GPU schedules.
- Update `tests/ARCHITECTURE.md` with the deterministic CUDA accuracy contract, fixture placement and refresh
  procedure, complete-consumption assertion, target-capability handling, and the explicit greedy-policy exception.
  State that tests must never regenerate a missing manifest automatically.
- Update the existing compiler tutorials rather than creating an isolated feature page:
  - `docs/docs/tutorials/03-forks-and-knobs.md`: contrast graph-wide ad hoc env pins with a complete per-kernel
    schedule manifest and show the public commands.
  - `docs/docs/tutorials/06-deploy-evidence-hierarchy.md`: show where normal capture gets its choices and why replay
    bypasses ranking.
  - `docs/docs/tutorials/07-golden-configurations.md`: contrast an unmeasured whole-compile manifest with measured,
    program-backed golden evidence.
- Remove capture/replay helpers made obsolete by the shared exact matcher. Do not remove golden measurement or A/B
  functionality.
- Read `STYLE.md` and every touched directory's `ARCHITECTURE.md` during implementation. Update `STYLE.md` only if the
  work introduces a genuinely new repository-wide coding convention; the manifest workflow itself belongs in
  `AGENTS.md`, the architecture docs, and the test contract above.
- Delete this plan once its conclusions are encoded in durable docs.

Gate: `make test`, `make lint`, a full diff audit, and the normal contribution checks pass before implementation is
committed.

## Non-goals

- Do not change schedule enumeration, greedy ranking, placement pricing, or the prior.
- Do not add kernel-targeted environment pin syntax in this work. The manifest is the scalable complete-program
  interface; `EMMY_KNOBS` remains the lightweight ad hoc override.
- Do not put manifests into the tune DB, prior reservoir, or golden corpus.
- Do not record measurements or claim that a manifest remains fast on hardware other than the target that produced
  it.
- Do not make tests regenerate manifests automatically. Intentional scheduler changes must produce visible fixture
  updates reviewed with the code change.
- Do not retain compatibility shims for stale manifest schemas or structural identities.
