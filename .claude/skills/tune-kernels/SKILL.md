---
name: tune-kernels
description: >-
  Tune Emmy kernels for a Hugging Face model, traced IR, or golden YAML. Use when asked to tune a model or golden
  set, seed MCTS with model-proposed knob configurations, compare hybrid proposals against MCTS-only search,
  diagnose slow or failing kernels, refresh per-GPU goldens, or produce a per-kernel tuning findings report.
---

# Tune Emmy kernels

Use one golden YAML format throughout the workflow. Keep its two trust levels separate:

- A **working golden** is an untracked experiment artifact. Each structural target contains a `realizations` array;
  each realization has named dimension `bindings`, explicit registered input `pins`, and optional knob proposals,
  measurements, or `ranking` feedback written by `emmy tune`.
- A **canonical golden** is reviewed deploy evidence under
  `emmy/compiler/pipeline/search/goldens/`. Every realization requires an explicit knobs mapping (empty for a
  forkless anchor) and paired positive deployable `emmy_us` / `reference_us` measurements with a named
  `reference_backend`. Never write search feedback into a canonical file.

`pins` and `knobs` are both registered knob mappings with different times of application: `pins` constrains candidate
enumeration, while `knobs` records the winner measured inside that regime. `FAST_MATH` has no special YAML field; write
it under `pins` exactly like any other input knob.

`emmy tune --golden-file` rejects canonical repository paths because it updates its input. Copy a canonical file to
a fresh `_tune/<run>/working.yaml` first. Do not commit a trace-created working golden automatically; leave that
decision to the author or agent after validation.

## 1. Prepare the supplied GPU host

When the caller supplies SSH, run tracing, tuning, O3 verification, and profiling on that host. The caller owns the
VM; never provision, stop, or delete it here. Record the SSH target and verify its GPU names/count before spending a
search budget. Fail on a mismatch rather than switching hardware.

Create a task-owned remote scratch directory and source tree:

1. capture the local repository URL and revision;
2. create an exact remote directory with `mktemp -d`, record it immediately, and clone the repository there;
3. check out the captured revision, then rsync the local tracked-file list so reviewed uncommitted source changes are
   represented; remove the exact tracked paths deleted locally, and add only explicitly enumerated task-owned
   untracked source files to that list;
4. rsync the selected self-contained working golden into `_tune/<run>/`;
5. run `make setup` remotely and verify `./venv/bin/emmy trace --help` and `./venv/bin/emmy tune --help` match the
   local workflow.

Build rsync input from `git ls-files`, filtering it to paths that still exist. Derive the deletion list from tracked
status, validate every deletion as repository-relative, and apply only those exact paths under the remote checkout.
Never send `.git`, `.env*`, virtual environments, caches, unrelated untracked files, or the whole checkout. Verify
the remote revision and a manifest hash after the overlay. Repository setup and task-artifact transfer are allowed
remote mutations. Do not patch installed packages, running containers, or deployed workloads by hand; use Emmy
commands for any workload lifecycle.

A typical setup is:

```bash
REPO_URL=$(git remote get-url origin)
REVISION=$(git rev-parse HEAD)
REMOTE_ROOT=$(ssh "$REMOTE" 'mktemp -d /tmp/emmy-tune.XXXXXX')
LOCAL_MANIFEST=$(mktemp /tmp/emmy-tune-files.XXXXXX)
ssh "$REMOTE" "git clone --no-checkout '$REPO_URL' '$REMOTE_ROOT/repo' && \
  git -C '$REMOTE_ROOT/repo' checkout --detach '$REVISION'"
git ls-files -z | while IFS= read -r -d '' tracked_path; do
  if [ -e "$tracked_path" ] || [ -L "$tracked_path" ]; then
    printf '%s\0' "$tracked_path"
  fi
done > "$LOCAL_MANIFEST"
rsync -a --from0 --files-from="$LOCAL_MANIFEST" ./ "$REMOTE:$REMOTE_ROOT/repo/"
rsync -a --relative _tune/<run>/working.yaml "$REMOTE:$REMOTE_ROOT/repo/"
ssh "$REMOTE" "cd '$REMOTE_ROOT/repo' && make setup"
```

Apply the validated NUL-delimited output of `git diff --name-only --diff-filter=D "$REVISION" --` as exact removals
under the remote checkout before hashing it. Use `mktemp` for local manifests and delete them locally. Quote
substituted values, and reject an empty or unexpected `REMOTE_ROOT` before cleanup. If the revision is not reachable
from the remote, rsync a locally created tracked-only source tree into `repo/` and record that the run used an overlay
rather than silently tuning different code.

Run long remote commands in a task-named `tmux` session or detached process, record the session/PID, enforce the
caller deadline, and poll logs. Scope Hugging Face or other required credentials to the command without printing
them; remove any task-owned temporary credential file during cleanup.

On success and failure, rsync back working YAML files, logs, DB/prior snapshots, O3 JSON, and
reports before cleanup. Stop only the recorded task-owned session/process, tear down any skill-created serving
workload through Emmy, then remove only the recorded scratch directory after validating its task-specific prefix.
Leave the VM running and return cleanup ownership to the caller.

## 2. Establish scope and artifacts

Record the repository revision, target GPU name and compute capability, selected device IDs, model or IR, trace
profile, dynamic dimensions, dtype/quantization, compilation flags, candidate budget, wall-time limit, and RNG seed.
Use a persistent untracked directory under `_tune/`; keep logs, DB/prior snapshots, dumps, and working goldens there.

Prefer the user-requested scope. Otherwise tune one representative layer with deployable dynamic dimensions before
expanding to a whole model. Never describe a single-layer result as whole-model performance.

Choose the input:

1. **Existing golden YAML:** copy it to the work directory. It is self-contained.
2. **Hugging Face model with no working golden:** create the inventory once:

   ```bash
   emmy trace <model> [--layer N] -o _tune/<run>/working.yaml
   ```

   The command refuses replacement and embeds every distinct post-fusion target, with one inventory realization and
   no knobs or measurements. On a supplied host, run it from the remote checkout, rsync the YAML back before
   proposing candidates, then rsync the completed arm files to the same remote paths.
   Preserve that skeleton even if tuning later fails.
3. **Existing Graph/Torch IR:** pass it through `emmy trace <ir.json> -o _tune/<run>/working.yaml`; do not hand-write
   a second persistence format.

For a serving release, derive the entire matrix from the pinned env and capture every structural target once:

```bash
emmy trace <checkpoint> --serving-twins --serving-config <models/slug.env> -o _tune/<run>/working.yaml
```

Do not add or remove serving widths or pin regimes by hand. The config is authoritative for decode, prefill, M=1,
warm-shape, symbolic, and precision-trading realizations.

Run `emmy tune --help` and `emmy trace --help` before a remote run so the commands and the checked-out revision agree.

## 3. Propose candidates from evidence

Inspect, in this order:

1. canonical goldens for the exact GPU;
2. entries with the same kernel kind, dtype, layout, dynamic/static status, input pins, and nearby shapes;
3. relevant goldens from GPUs with the same compute capability or closest architecture;
4. the embedded target program and emitted operation structure;
5. the scheduler/lowering code and nearby `ARCHITECTURE.md` files that define offered knobs and eligibility gates.

Use this context to add several distinct knob proposals inside the relevant realization. Duplicate that realization's
`name`, `bindings`, and `pins`, then add only a `knobs` map. Do not duplicate the structural config, invent
timings, copy latency from another GPU, or add a knob value that target/regime cannot offer. Prefer candidates that
test materially different schedule choices; omit low-confidence proposals so MCTS receives the remaining budget.

Keep every schedule family emitted in `record_knobs`, including explicit off values, when copying a known realized
configuration. Treat dynamic and static kernels as different targets. Keep different input pin regimes separate;
a precision-trading candidate never replaces the standard deploy path.

## 4. Run equal-budget hybrid and MCTS-only searches

Create an inventory-only base with one row per target and no knobs, timings, or `ranking`. Then create two working
files from that exact base:

- `mcts.yaml`: no agent-added proposals;
- `hybrid.yaml`: the same file plus agent-added proposals.

Do not copy knob-bearing canonical rows into either arm: canonical goldens remain the common implicit deploy context
consulted by Emmy, while copied rows would reserve candidate slots and corrupt the equal-budget comparison. Existing
goldens may guide proposal reasoning, but only the hybrid file contains agent proposals.

Give both arms identical starting tuning DB, online prior, canonical deploy context, compiler revision, GPU devices, RNG
seed, compilation flags, patience, wall-time limit, and `--max-candidates` value. Start both empty when no historical
state is required. Restore the same snapshots before each arm when measuring from a warm state. Clear compiled
caches equally and alternate run order when repeating the comparison.

Every supplied proposal reserves one slot in the target kernel's `--max-candidates` budget before MCTS, even when
its measurement is cached. MCTS cache hits do not spend the remaining live-measurement slots. Do not report a hybrid
win from a larger budget; compare reserved proposal slots and successful live measurements as well as the configured
budget.

Use every homogeneous target GPU. Pass exact device IDs; never mix GPU models or compute capabilities in one tune.
Multiple targets share the backend-slot queue and one prior, so keep them in one invocation:

```bash
EMMY_TUNE_DB=<arm.db> EMMY_ONLINE_FILE=<arm-online.json> \
  EMMY_CUBIN_CACHE=<arm-cubin-dir> \
  emmy tune --golden-file <arm.yaml> --devices 0,1 --max-candidates <B> \
  --patience <P> --seed <S> --dump-dir <arm-dump> 2>&1 | tee <arm.log>
```

Give cold arms separate empty `EMMY_CUBIN_CACHE` directories. For a warm comparison, restore the same DB/prior
snapshots into distinct arm paths and still start with empty arm-specific cubin directories. Enforce the same external
deadline around each command when wall time is part of the comparison. The tune ranking lane normally compiles at
`-Xcicc -O1`; its latencies rank search results but are not deployable performance. Read the per-entry `ranking`
blocks and search logs for proposal status, measured knobs, latency, and the exact searched finalists.

Do not use `emmy tune --bench` as an arm's winner measurement. Its assembled graph replays through the deploy
evidence hierarchy, where a canonical golden can override the searched DB result and make both arms benchmark the
same deployed configuration. Compare the primary arms with ranking/search feedback, then pin the exact actual
proposal and search finalists through `emmy run --ab` for fresh O3 measurements.

Compare arms per target: successful live measurements, best ranking latency, exact searched finalist, exact-pinned
deployable O3 result, correctness, wall time, and failures. Also report aggregate geometric mean or total latency only
when the target set and measurement coverage are identical.

## 5. Optionally refine once

Use failed pins, measured-knob mismatches, searched finalists, `emmy eval variants`, and same-family canonical winners
to form at most one refined proposal round. Keep it outside the primary A/B result unless both arms receive an equal
additional budget and deadline. Record which first-round evidence motivated each new proposal.

## 6. Verify deployable winners and promote cautiously

Shortlist the best hybrid proposal, MCTS-searched, and incumbent configurations from their exact measured knobs.
Re-run each with the same embedded target at deployable O3, selecting its working file and realization name and
pinning the exact fully realized knobs with `--ab`.
Use `EMMY_NVCC_FLAGS=` if necessary to prevent an inherited O1 override:

```bash
CUDA_VISIBLE_DEVICES=<selected-ordinal> EMMY_NVCC_FLAGS= \
  emmy run --golden-file _tune/<run>/working.yaml --golden <realization-name> \
  --bench --bench-backends eager,emmy \
  --ab "<fully realized knobs>" --json _tune/<run>/verification/<candidate>.json
```

Run `emmy run --help` from the checked-out revision before using this form. Under `CUDA_VISIBLE_DEVICES`, the selected
physical GPU becomes the command's ordinal 0; never pass an invented `--device` flag. Require:

- correct output on every run;
- no compile, pin-realization, watchdog, or benchmark failure;
- at least two fresh O3 measurements on the target GPU;
- a win that exceeds observed run-to-run noise.

Treat statistically indistinguishable configurations as ties and retain multiple candidates when their realized
knobs differ. Promote only specialized realizations with paired positive O3 `emmy_us` / `reference_us` measurements
and a live `reference_backend`. Never promote a `ranking` block, an O1 latency, an incomplete serving realization
matrix, or a timing copied from another run/GPU. Let the author or agent decide whether to update and commit the
canonical golden.

When the calling workflow has proved that its model inventory covers every required path, canonical promotion is a
required output, including a correct measured greedy fallback for every search miss. Withhold the repository golden
only when model tracing is incomplete; preserve that successful subset as a partial working golden instead.

Validate any proposed canonical edit with the golden schema tests before presenting it.

## 7. Diagnose and report

Use the existing CLI before writing ad-hoc scripts or SQL. Preserve the exact command, logs, JSON output, dump paths,
and target GPU for every finding.

### Triage a losing or failed kernel

Start with:

```bash
emmy eval variants --kernel <substring>
emmy eval failures
emmy eval online --dataset nodes --kernel <substring>
emmy eval online --dataset nodes --blame --ablate --kernel <substring>
```

For a serving golden, run the unified release audit on the pinned GPU; it validates that every structural target has
the exact config-derived realization matrix before reproducing and auditing it:

```bash
emmy eval golden <canonical-golden.yaml> --serving-config <models/slug.env>
emmy eval offline --kernel <substring>
emmy eval online --dataset golden --kernel <substring>
```

Classify every meaningful loss:

1. **Search shortfall:** the best measured or replayed configuration exists, but the prior or patience does not reach
   it. Use variant rank, fork sibling regret, and per-feature blame. Keep offline-prior and online-prior evidence
   separate because cold-start feature errors and learned-model calibration errors require different fixes.
2. **Eligibility or optimization lockout:** the desired schedule family is never offered. Cite the lowering or
   scheduler gate and the target property that triggers it.
3. **Code generation quality:** the correct execution tier is present but loses. Inspect emitted CUDA and profile the
   pinned target against the reference.
4. **Benchmark failure:** use the recorded error and shared failed knobs. Reproduce compile-only before spending
   another search budget.

Confirm a suspected search shortfall by tuning only the selected realization with more patience without clearing
useful state.
Diff before and after dumps with `emmy compare`; do not infer structural changes from log text.

For a deeper independent check, include `torch.compile` in the O3 comparison:

```bash
EMMY_NVCC_FLAGS= emmy run --ir <kernel>.torch.json --bench \
  --bench-backends eager,tcompile,emmy --ab "<fully realized knobs>" \
  --json _tune/<run>/o3-<kernel>.json
```

Append `--profile` for an NCU comparison when `ncu` is installed and performance counters are permitted. Use
occupancy, registers per thread, SM and DRAM throughput, LSU instructions, and shared-memory bank conflicts to test a
specific hypothesis. If counters are unavailable, record that limitation and continue with timing and source
evidence. Inspect source without a GPU when useful:

```bash
EMMY_KNOBS="<fully realized knobs>" emmy compile <kernel>.torch.json --ir cuda
```

Do not compare O1 tune-DB latency with O3 deployment latency. Re-run a surprising O3 result before reporting it.

### Validate whole-model impact

For a requested whole-model tune, produce a full eager / `torch.compile` / Emmy table after finalist selection. Label
an `emmy tune --bench` result as deploy-evidence replay, not an arm's searched winner; use exact `emmy run --ab` pins
for arm conclusions. For a servable embedding model, also run matched `emmy serve <model> --bench` and
`emmy serve <model> --bench --stock` trials with identical request count, input length, concurrency, and seed. Skip
serving A/B for unsupported model types and state why.

### Write the findings report

Save standalone findings under `_tune/<run>/findings.md` unless the calling workflow specifies a durable experiment
report. Include:

- status, date, repository revision, exact hardware/device IDs, scope, dynamic hints, dtype or quantization, and
  commands;
- fairness controls for hybrid versus MCTS-only, including starting DB/prior hashes, golden source, measurement and
  wall-time budgets, run order, live measurement counts, and compilation lane;
- a candidate table with target, knobs, rationale, proposal status, measured knobs, O1 rank, and searched finalist;
- a per-target A/B table with both arms' O3 latency, reference latency, correctness, repeated-run range, and decision;
- whole-model and serving tables when applicable;
- one finding per root cause, ordered by deployable latency at stake, with symptom, evidence, root cause or
  distinguishing diagnostic, embedded program target, and recommended fix;
- an offline-prior versus online-prior table for fork sibling regret, reachability, calibration, and labeled blame
  whenever search steering is implicated;
- promoted, tied, rejected, and unresolved candidates, plus exact working and canonical artifact paths;
- workflow notes covering slow steps, retries or flakiness, multi-command detours, output friction, and a concrete CLI
  or skill improvement for each.

Use O1 values only as labeled ranking evidence and O3 values for performance conclusions. Never present a
single-layer table as a model result or aggregate across different target coverage. End with workflow friction and
concrete improvements.
