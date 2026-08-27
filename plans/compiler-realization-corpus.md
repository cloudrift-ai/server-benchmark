# Compiler realization corpus

Give Emmy a durable, data-driven regression lane for one failure class: **a schedule that should be realizable is
not**, and the kernel loses to `torch.compile` as a result. Make `onboard-model` surface that class and record it,
instead of leaving it in an ephemeral `_tune/<run>/findings.md`.

Out of scope as *admission criteria*, deliberately: search shortfall (the schedule realizes, the prior does not pick
it) and code generation quality (the right tier is present and still loses). Neither is a realization gap, and
admitting cases on those grounds would make the lane's ratchet meaningless.

That is a rule about what earns a case, not about what a case then measures. Once a case exists it carries per-card
`emmy` and `torch.compile` timings (**Stage 5**), so the corpus does end up showing where the compiler is ahead of
and behind torch across the fleet — which is the point of **Part D**. Admission stays narrow so the ratchet means
something; measurement is broad because the cases are already the right minimized reproducers.

## 1. Why this lane does not exist yet

Every oracle it needs is already in the tree — `golden_eval.enumerate_graph`, `pins.pinned_knobs`,
`pins.unreproducible_pin_flag` — but they are only reachable from inside `emmy run` and the tuner. Nothing asserts
that a schedule known to be good is still realizable.

The consequences are visible in the repository today:

- `tests/compiler/e2e/test_attention_coverage.py` carries a nine-row measured table in its module docstring, annotated
  "they are not reproduced by the correctness lane ... they were measured once, on the named card, and they are
  recorded here because nothing else in the tree records them";
- `tune-kernels` already classifies losses into search shortfall / eligibility-or-optimization lockout / code
  generation quality / benchmark failure, and writes the result to a run directory nothing reads again;
- `onboard-model` never compares a target against `torch.compile`. Its golden gate accepts
  `reference_backend: emmy-greedy`, so a complete, verified golden can be an order of magnitude off torch and say
  nothing.

Measured feasibility, on this checkout:

| Probe | Result |
| --- | --- |
| minimal case file from `emmy trace -c "<snippet>" -o f.yaml` | 26 lines, ~500 B, self-contained stable Torch IR |
| pinned `enumerate_graph` at a forced compute capability, no GPU | 0.05 s, 8 rows |
| lower to CUDA source at a forced compute capability, no GPU | 0.04 s |
| pin that cannot be offered (`WORK=zzz9x9`, `WORK=w7x13`) | pinned enumeration returns 0 rows |
| the same pin through `unreproducible_pin_flag` | returns `None` — the family is unstamped, so the flag cannot see it |
| capability-illegal pin (`d1/smem-tma` at sm_89, an mma atom at sm_70) | raises `ValueError` with an exact message |
| `structural_features` / `kernel_identity` for one record | 0.08 s / 0.02 s |

Two conclusions drive the design. The whole "cannot be realized" class is provable without a GPU at roughly 0.1 s per
case, which is what lets it gate every commit. And `unreproducible_pin_flag` alone is not a sound oracle for a
hand-authored pin — pinned-enumeration membership has to be the primary check.

## 2. Design

### Obligation scales with the machine

The governing rule, from which the per-stage rules below are derived: **nothing in this lane blocks work the machine
in front of you cannot do.** A run carries exactly the obligation it is capable of discharging, and reports the rest.

- **Everywhere, GPU or not** — the derived half must be current. Checking it and fixing it are both GPU-free, so it
  is always an obligation and always blocking. On a machine with no GPU this is the *only* thing the corpus can ask
  of you, and it is always something you can finish.
- **On a card whose capability matches a case** — correctness is blocking. A kernel that does not build, or builds
  and computes the wrong answer, is a bug wherever it is found, and the strict-xfail ratchet on stages 1 to 4 is what
  keeps a closed gap from silently reopening.
- **In the perf lane, on a matching card** — recording a missing timing is an obligation, because that is the only
  way coverage grows. The lane is `perf`-marked and opt-in, so this never interrupts ordinary work; it applies to
  someone who has chosen to run it.
- **Performance regressions and improvements are always reported, never enforced.** A slower case does not fail a
  run. It produces a finding, and **Part D** turns that finding into a labeled pull request a human accepts or
  declines. Enforcement lives in that review, not in a red test that a legitimate correctness fix could pin red
  forever.

Repair what can be repaired here; report everything else.

### Layout

```
tests/compiler/realization/
  ARCHITECTURE.md
  helpers.py                 # load, regenerate, replay, the four oracles
  test_realization.py        # one parametrized walker over cases/
  cases/
    matmul/f16acc-splitk-sm89.yaml                 # closed — every stage must pass
    attention/hd512-flash-sm120_xfail_offered.yaml # open — expected to fail at `offered`
    fused/<name>.yaml
```

`tests/compiler/realization/` is a kind-organized directory in the sense `tests/ARCHITECTURE.md` already sanctions:
its cases span lowering, the CUDA backend, the pin machinery and the golden loader, and they share one workflow.

### Case file

A working golden document — byte-for-byte what the inventory writer already emits — plus the authored pin and two
optional keys. Nothing else is stored, because nothing else is necessary.

```yaml
compute_cap: [8, 9]
programs:
- inputs: [x0, x1]
  outputs: [matmul]
  nodes:
  - id: x0
    op: input
    outputs: [[x0, f16, [64, 4096]]]
  - id: x1
    op: input
    outputs: [[x1, f16, [4096, 512]]]
  - id: matmul
    op: torch.matmul
    attrs: {has_bias: false}
    inputs: [x0, x1]
    outputs: [[matmul, f16, [64, 512]]]
configs:
- program: 0
  target:
    origins: [matmul]
  realizations:
  - name: k_matmul_5b7645.167d5f47efce
    bindings: {}
    pins: {FAST_MATH: true}
    knobs: {WORK: w2x2, TILE: mma_m16n8k16_f16_f16/f4x8/k2, REDUCE: g2k, STAGE: ''}
    identity: 0302cbd2c129ae1851d5f529621a752756f6181d0d4cbaf57eb22f85028d11c2
    latency:                                    # per card; absent until a matching card records it
      NVIDIA GeForce RTX 4090: {emmy_us: 30.81, tcompile_us: 24.10}
model: synthetic/matmul-f16acc-splitk
```

Why each part is necessary, and why nothing else is:

- `programs` / `target` / `compute_cap` — the reproducer. Stable Torch IR rather than a code snippet, so a frontend
  change cannot silently alter what the corpus tests.
- `name` — already carries `Op.cache_key()[:12]` (`working_golden.py`), so it detects cache-key drift for free.
- `pins` / `knobs` — the authored schedule. Regeneration never produces these, which is what makes the mechanism
  safe.
- `identity` — the record's `deploy_identity`. `cache_key` folds the class name, the algebra key and the knobs;
  `deploy_identity` additionally folds the dtype, extent, shape and store fingerprints, so a new fingerprint fact —
  which `ir/tile/identity.py` documents as routine under "Adding a fact" — moves `identity` while leaving `name`
  untouched.
- `latency` — the per-card block described under **Stage 5**.

Those last two are the only additions to the format, and both are optional keys on the realization.

**Why `latency` rather than the existing `measurements`.** The schema already carries per-realization timings —
`{emmy_us, reference_us, reference_backend}` — and the reason not to reuse it is cardinality, not principle.
Repository goldens are one file per card (`recipes/<model>/golden/<gpu-slug>_<compute-cap>.yaml`, with `gpu_name` at
document level), so a flat block is exactly right: the card is implied by the file. Corpus cases are one file per
case across many cards, and a flat block cannot hold a 4090 entry beside a 5090 one.

The alternative is to adopt the golden layout — one case file per card — and reuse `measurements` unchanged with no
format extension at all. It is worse: the program wire would duplicate on every card, adding a card would mean
copying a case file rather than appending a line, and the GPU-free regeneration gate would re-check N identical
copies of one program on every pull request. Machine-independence is what lets a single case declare a capability and
any matching card run it; splitting by card discards it. Two further reasons stand regardless of shape: an open case
has no `emmy_us` because its schedule never ran, and filling `measurements` marks a case `VERIFIED`, which auto-pins
it in `emmy run --golden` and folds test data into the replay tooling's trusted evidence.

Deliberately not stored: the `S_*` feature dict and the featurizer version. The corpus asserts realization and
accuracy; the featurizer participates in neither, because every fork is pinned and the prior never decides. Also not
stored: a canonical copy of the knobs — regeneration rewrites `knobs` in place to its canonical spelling instead, so a
codec re-spelling shows up as a diff in the field that already exists.

### Expectations live in the filename

There is no manifest. A case's expectation is its filename suffix, so extending the corpus is writing one file:

```
<family>/<name>.yaml                  # closed — every applicable stage must pass
<family>/<name>_xfail_offered.yaml    # open — strict xfail at `offered`
<family>/<name>_xfail_realized.yaml
<family>/<name>_xfail_built.yaml
<family>/<name>_xfail_correct.yaml
```

That is enough because every other field a sidecar would have carried is either derivable or belongs in the file:

- **id / file** — the path is the identity, and the pytest parameter id.
- **realization** — a case file holds exactly one config with exactly one realization. The harness enforces that
  invariant, so there is nothing to select.
- **reference** — derived from the target. `target: {origins: […]}` is a frontend program and compares against the
  numpy backend; `target: {loop: N}` is an exact Loop target with no torch twin and compares against the same-input
  greedy execution, which is the rule `emmy run` already applies.
- **evidence** — a leading comment block in the case file, the way `recipes/*/golden/*.yaml` already opens with
  prose. An `_xfail_*` file must carry an `# evidence:` line; that is a grep, not a parser. Measured latency is not
  here: it is structured per-card data in `latency:`, described under **Stage 5**.

Three things this buys beyond ergonomics. The open-gap inventory becomes `ls cases/**/*_xfail_*.yaml`, and the
completion gate becomes "no file matches that glob" — sharper than a list shrinking to empty. Closing a gap is a
`git mv`, so the diff shows the closure as a rename. And two concurrent onboarding runs on different models can each
add a case without touching a shared file, which a single `index.yaml` would have turned into a merge conflict on
every parallel nightly.

The one cost is real but small: the filename is semantic, so an unrecognized `_xfail`-shaped token must be a hard
error rather than a silently-closed case — otherwise a typo quietly strengthens the assertion.

Prose still carries one thing the structured fields cannot. The **origin** of a gap — emmy's greedy pick against
`torch.compile` on the model kernel the case was minimized from — describes a different program on a different card
at a different revision. It is context for why the case exists, not a measurement of it, and it belongs beside the
`# evidence:` line rather than in `latency:`.

### Stages

| Stage | Assertion | GPU |
| --- | --- | --- |
| `offered` | under `pinned_knobs(pins + knobs)`, `enumerate_graph` at the declared capability returns at least one row carrying every pinned value | no |
| `realized` | the graph lowers through `TILE_PASSES + CUDA_PASSES` at that capability, `unreproducible_pin_flag` is `None`, and every pinned family is stamped | no |
| `built` | `CudaBackend().compile(...)` under the pin — nvcc accepts it | yes, exact capability |
| `correct` | run against the reference within `dtype_tol` | yes, exact capability |

An `_xfail_<stage>` suffix applies `pytest.mark.xfail(strict=True)` at that stage. When the compiler learns to
realize the case, the strict xfail fails and the fix is a `git mv` dropping the suffix — the ratchet. Stages 1 and 2
always run at the declared capability, so an sm_70 lockout is exercised on any box; stages 3 and 4 run only when the
live card's capability equals the declared one.

### Stage 5 — latency, as a separate lane

Stages 1 to 4 answer "can this schedule be realized and is it correct". A fifth answers "and is it still as fast as it
was", which catches the failure this corpus cares about most after a lockout: a schedule that quietly stops being
selected and falls back to a slower tier. It must not ride the same walker, for two reasons the tree already
establishes. `make test` compiles at `EMMY_NVCC_FLAGS="-Xcicc -O1"` — the correctness lane, which by the glossary's
own definition "is not a measurement lane, and nothing measured under one is read by a deploy" — so a latency
assertion there would measure the wrong regime entirely. And `tests/perf/` is deliberately non-asserting today: it
prints ratios and dumps JSON for a separate diffing script (itself retired in **Part C**, since a timing kept in the
case file is diffed by `git`). Turning that stance around belongs in its own lane, not as a side effect of a
correctness suite.

So: `perf`-marked, `-O3` forced, invoked explicitly (`make bench-kernels` or `pytest tests/compiler/realization
-m perf`), and skipped unless the live card's capability matches the case's. A matching card with no recorded timing
is the coverage gap below, not a skip.

Latency lives in an optional per-card block, keyed by `Context.hardware_id` — the identity that already separates
same-die SKUs like H100 from H200, which free-text card names do not. This keeps stages 1 to 4 machine-independent
while giving stage 5 the card a timing is only meaningful against:

```yaml
    latency:
      NVIDIA GeForce RTX 5090: {emmy_us: 12.28, tcompile_us: 15.90}
      NVIDIA GeForce RTX 4090: {emmy_us: 30.81, tcompile_us: 24.10}
```

Both numbers, because the block answers two questions and only one of them is a ratchet. `emmy_us` against its own
recorded value says *did we regress*. `tcompile_us` beside it says *are we ahead or behind*, per case, per card — and
that ratio, sorted, is the compiler's optimization worklist. Recording only the first would make the corpus a
regression net and nothing more. The torch side costs roughly 0.8 s of JIT per case, which is nothing on a rented
box.

The card key is `hardware_id`, not the compute capability, and that distinction is what makes the block a fleet view:
an RTX 4090, an RTX 4080 and an L40S are all sm_89 and all run the same cases, but they are three separate rows. The
capability decides which cases a card *can* run; the card decides what gets recorded.

**The two gates have deliberately different reach, and that asymmetry is the point.** The derived half — program wire,
name, `identity`, canonical knobs — is GPU-free, so its check fires everywhere and its fix works everywhere. Timings
are not: only a machine holding the card can produce one. If both gates fired everywhere, an agent on a CPU box would
face a failure it has no way to clear, so the timing gate is scoped to the machine that can actually answer it:

| Gate | Fires | Fixable |
| --- | --- | --- |
| derived half is stale | always, on any machine | anywhere — `make test-corpus-regen` |
| no timing recorded for the live card | only when the live card can run the case | only on that card, at `-O3` |

"Can run the case" is already defined by stages 3 and 4: the live compute capability equals the case's declared one.
Add two more bounds and the gate stays small. It applies only to **closed** cases, because an open case's schedule
never runs and demanding a latency for it would be the same false attribution the plan rejects elsewhere. And it
reports **once, at session end**, naming every case and the command that records them — the shape
`tests/conftest.py` already uses for the durations baseline, rather than N separate failures. Whether that report
fails the opt-in lane or merely ends it prominently is open decision 4.

The effect is that coverage accumulates on its own: an agent working on a 4090 is asked once for the sm_89 cases'
4090 timings, records them, and every later run on that card has a baseline to ratchet against. An agent on a CPU box
is never asked at all.

The comparison rule, and why each half is shaped this way:

- **Best of three.** Run the case three times and compare the fastest against the record. Interference is one-sided —
  a busy machine makes a kernel slower, never faster than the hardware can go — so the minimum of several runs is the
  honest estimator of capability, and requiring all three to be slow before reporting is what keeps the finding from
  crying wolf on a developer box that is also compiling something.
- **Within band: nothing.** Start at 10% and *measure* the band before fixing it — run the corpus repeatedly on an
  idle card and look at the spread. Ten percent is not obviously safe: `run --json` already documents a ~7% gap
  between two timing semantics for the same kernel, and this repository has found isolated microsecond benches to be
  launch-bound rather than measuring the kernel at all.
- **Faster: update, but never as a side effect of a test run.** A developer's suite must not rewrite checked-in data
  while they work: an automatic ratchet ends up pinned to the luckiest noise excursion ever observed, after which the
  band is effectively gone, and it dirties the working tree. The repository already made this choice once, in
  `--write-durations` behind `make test-durations`. Recording happens two ways instead, both deliberate: the explicit
  command that also restamps identity, and the nightly workflow of **Part D**, whose measurements come off an idle
  single-tenant rented card and land as a reviewable pull request rather than a silent local write.
- **Slower on all three: report.** Naming the case, the card, both numbers and the three samples — a finding, not a
  failure, per the principle above. It is `Part D` that escalates it into a reviewable pull request; a developer who
  ran the lane while investigating something else is told, not blocked.

Be honest about what this catches. Best-of-three with a 10% band will miss percent-level drift — a real 12%
regression can still show one lucky run inside the band. It catches cliffs, which is what this corpus's regressions
actually look like: a schedule that stops being realized does not get 12% slower, it gets several times slower.

There is a reuse opportunity here worth taking rather than adding a parallel mechanism. `tests/perf/cases.py` is a
hand-curated list of twelve Qwen3-Embedding kernels whose own `ARCHITECTURE.md` still claims "Emmy currently emits
FP32 only" — it has drifted. If the corpus becomes that lane's case source, the perf suite gets its cases from the
same minimized reproducers the realization stages use, and one stale list goes away instead of a second one appearing
beside it. And with `tcompile_us` recorded alongside `emmy_us`, stage 5 answers `tests/perf/`'s own question too —
"how do we compare to PyTorch" — so this is a replacement rather than a division of labour. What `tests/perf/`
uniquely contributes is its curated Qwen3-Embedding layer coverage; that survives as corpus cases, not as a second
list.

The PR test job is CPU-only in practice — 1322 s of the 1922 s in `tests/durations.json` is CUDA-marked, which alone
exceeds its 20-minute cap on the single serial CUDA chain — so stages 1 and 2 are the ones that gate a commit. Confirm
this before relying on it.

### Staleness: regeneration, not stamps

Kernel identity and schedule codec spellings change often, so a stored case rots. The failure mode that matters is
silent: a retired knob spelling canonicalizes to itself, matches no candidate, returns zero rows, and reports as a
lockout — a phantom compiler gap. For an open case the mirror failure applies: the xfail keeps passing and the
ratchet stops ratcheting.

The gate is regeneration equality, and **detection is a test, not a command**. `test_realization.py` recomputes each
case's derived half — decode `programs[0]`, re-run the inventory writer under `Context.from_target(compute_cap)`,
re-derive `identity`, re-canonicalize `knobs` — and asserts it equals what is stored, failing with the offending
cases named and the fix spelled out. `make test-corpus-regen` only *applies* the fix. That split is the shape the
repository already uses twice: `ruff format --check` detects while `make format` fixes, and the session-end durations
gate in `tests/conftest.py` names its offenders and asks for `make test-durations`.

This matters more than it looks. The check is GPU-free — decode, lower at a forced compute capability, derive the
identity, roughly 0.1 s per case — so it runs under `make test` on the CPU-only pull-request job. Kernel-identity and
codec drift is therefore caught on every pull request that causes it, by the commit that causes it, rather than
weeks later on a GPU box. Nothing in the tree does that for `recipes/*/golden/*.yaml` today.

Verified: the derived half round-trips byte-identically — same program wire, same target, same `name` including the
identity digest, no extra keys — and the only diff is the authored `pins` / `knobs`, which regeneration structurally
cannot produce. That property is what keeps a snapshot mechanism from burying a realization regression.

Five rules make it load-bearing:

1. **Regenerate from the stored program, through the library, not through a CLI.** `emmy compile` prints IR stages and
   `emmy trace` writes YAML, but `emmy trace --target sm_89 -c …` still stamps `gpu_name` from the live card, while
   the library path with an explicit context emits none. The library path is machine-independent and needs no torch.
2. **Canonicalize the authored knobs strictly.** `canon_family_value` swallows `ValueError` and returns the raw
   string, which is exactly how a retired spelling survives. Add a `strict=True` keyword to that one function; under
   it, regeneration fails loudly on `STAGE=d2/ring`, `WORK=zzz9x9`, `TILE=mma_m64n64k64_…` or `REDUCE=g2z` — all
   verified to raise — while a valid-but-unreachable pin (`WORK=w7x13`, `TILE=…/f99x99/k8`) parses cleanly and falls
   through to the `offered` stage, where a genuine lockout belongs.
3. **Refuse to write when a verdict changed.** If one commit moves identity and breaks realization, regeneration fixes
   the first and must not let the second ride along. It names the affected cases and exits non-zero; resolving them is
   a review conversation, not a mechanical step.
4. **Preserve what regeneration cannot produce.** `latency:` is measured on a card, not derived from the program, so
   the writer carries existing entries through untouched — a regeneration on a CPU box must not erase a 4090's
   recorded timings. Only `emmy run --golden-file … --bench --record` writes that block.
5. **Preserve the leading comment block.** `dump_golden_file` is a plain YAML dump and drops comments — verified — so
   a naive rewrite would eat the `# evidence:` line on every regeneration. The writer captures the leading `#` block
   before dumping and re-prepends it after. Roughly five lines, and without them the citation requirement is
   unenforceable.

### Boundary rule

- **Corpus, as data:** whether a pinned schedule is offered, realized, built and correct at a capability.
- **Python, as code:** how — emitted-source substrings, bit-identity between two configs, kernel counts, error text,
  compile-budget claims. None of these are expressible as a row.
- **Neither:** search shortfall; code generation quality; and a schedule the compiler *correctly* refuses. The corpus
  records schedules that are wanted and unavailable, so a refusal that is the right answer — and whose test asserts
  the refusal message — stays in Python.

The corpus is therefore overwhelmingly additive; Part A gives the exact retirement list.

## 3. Part A — the lane, and the tests it retires

Two commits in one change, not two changes. The lane is purely additive and lands first, so a bisect separates "the
lane itself broke something" from "a migrated row is wrong". The retirement follows in the same review, because a
mechanism that subsumes existing tests and leaves them in place is how a suite grows two ways to assert one thing.

### Commit 1 — additive

**A1. Build the lane.** `helpers.py`, `test_realization.py`, `ARCHITECTURE.md`, the regen target, and the
`canon_family_value(strict=)` keyword. Seed with two closed cases and one open case.
*Verify:* green off-GPU and on a 5090; each row under 0.3 s; `make test-corpus-regen` is a no-op on a clean tree.

### Commit 2 — retirement

Nine tests of roughly 1230 retire. That is the real number, and it is small on purpose: the existing suite asserts
*how* the compiler lowers — emitted source, kernel counts, bit-identity between two configs — which a row cannot
express. Anything not listed below stays.

**A2. Retire the accuracy-only half of `tests/compiler/e2e/test_knob_pinning.py`.** Seven of its fourteen tests assert
nothing but an accuracy comparison against a reference and are corpus cases exactly:
`test_norm_linear_fp16_scalar_reduce_tma_alignment`, `test_norm_linear_warp_fused_masked_m`,
`test_mma_matmul_k_split_staged`, `test_scalar_matmul_f16`, `test_unstaged_atom_mma_accuracy`,
`test_masked_tile_accuracy_configs`, `test_scalar_cpasync_mixed_dtype_slabs`. The other seven stay: five assert
emitted source (`test_sgemm_inner_reduce_is_unrolled`, `test_flat_output_sweep_lowers_with_its_axis_bound`,
`test_output_sweep_declines_the_warp_tier`, `test_unrealizable_warp_pin_falls_back_to_a_bound_scalar_grid`,
`test_unstaged_atom_lowers_gmem_direct`) and two assert a refusal message
(`test_warp_tma_pin_refuses_oversized_box`, `test_scalar_cpasync_pin_refuses_odd_stride`). The file roughly halves; it
does not go away.
*Verify:* the seven migrated configs are covered as cases; the remaining seven still pass; the suite is green.

**A3. Retire two tests from `test_matmul_coverage.py`.**
`test_warp_tier_is_offered_at_a_static_k_the_step_does_not_tile` is an `offered`-stage claim written in Python, and
`test_staged_scalar_matmul_matches_reference` is accuracy-only across three transports and three shapes. Of that
file's 53 asserting tests those are the only two: 41 assert structure alone and 11 mix accuracy with structure, and
splitting a mixed test across two mechanisms buys nothing. `test_tma_stage_pin_refuses_below_sm90` and
`test_scalar_masked_n_stage_pin_refuses` stay — they assert a refusal message, which the corpus does not express.
*Verify:* the two are gone; the bit-identity and refusal tests are untouched.

**A4. Convert the measured table in `test_attention_coverage.py`'s docstring** into `measured` blocks on rows, or drop
the entries that are pure code-generation-quality losses. Replace the prose table with one sentence pointing at the
corpus.

**A5. Seed from the two existing plans.** `plans/exl3-compiler-performance-gaps.md` and
`plans/v100-model-onboarding-compiler-gaps.md` are gap inventories already. Convert what qualifies into cases, then
delete both plan files.

Not in scope: splitting the 125 KB `test_matmul_coverage.py`. It is churn that does not serve this goal, and the
one-matrix-per-regime convention argues against a per-tier split.

## 4. Part B — `onboard-model` changes

One new section between "3. Fully qualify the compiler inventory" and "4. Decide Emmy eligibility", plus two
amendments. Everything else is untouched.

### New section 3b — Surface and record compiler realization gaps

Mandatory, and runs even when serving is blocked, matching the rule section 3 already carries.

1. After tuning, run one per-target torch comparison at deployable optimization:
   `emmy run --golden-file <working.yaml> --bench --bench-backends eager,tcompile,emmy --strict --json <out>`.
   Parse that record; it already carries `record_knobs`, `status`, `flags` and `lane` per row. Do not parse the
   terminal table.
2. Rank targets by `tcompile_us / emmy_us`, losers first. Selecting, sorting and tabulating structural fields is
   ordinary data handling; the classification below is agent reasoning and stays out of code.
3. Classify each material loss with the taxonomy `tune-kernels` already defines. Only an eligibility or optimization
   lockout, a pin that refuses or fails to lower, or a pin that runs wrong becomes a case. A search shortfall is fixed
   by measuring or reported as a prior finding. A code-generation-quality loss is reported, not recorded.
4. **Name the desired schedule with cited evidence.** Accept only a sibling card's golden carrying that family for the
   same structural identity, the same family already winning at a neighbouring binding, or an explicit roofline
   argument. It becomes the case file's `# evidence:` line. Without this the corpus fills with speculation.
5. Minimize to the smallest snippet that reproduces the refusal, then write the case with three shell steps and no
   file parsing:

   ```bash
   case=tests/compiler/realization/cases/<family>/<name>_xfail_<stage>.yaml
   emmy trace -c "<snippet>" --target sm_<cc> -o "$case"
   cat >> "$case" <<'EOF'   # the knobs block, copied from run --json record_knobs
   EOF
   make test-corpus-regen   # normalizes, stamps identity, re-prepends the comment block
   ```
6. Prove the case reproduces the gap: it must fail without the suffix and pass with it. Record the exact command.
7. Bounded: at most five new rows per run, and never at the expense of the artifact and cleanup reserve.

Restamping belongs to the pull request that changes a codec or an identity, not to an onboarding run. A run that hits
a stale row reports it and does not regenerate — it is not the author of the invalidation.

### Amendments

- Section 8 gains: every new case reproduces the stage its suffix names, `pytest tests/compiler/realization` is
  green, `make test-corpus-regen` is a no-op, and every `_xfail_*` file carries an `# evidence:` line.
- The summary JSON gains `compiler.realization_gaps` — a list of `{file, stage, emmy_us, tcompile_us}` — and lists the
  case files under `artifacts`.
- Section 3's golden gate should require a `tcompile` column for every retained target. Without it the skill has no
  signal that anything is losing, and gap surfacing has nothing to rank.

## 5. Part C — infrastructure

### One way to bench a golden

Nothing in this plan may introduce a new way to measure a kernel. Today there are three, which is the reason agents
keep writing scripts:

1. `emmy run --bench --json PATH` — the machine-readable record. Its own docstring says it "retires ad-hoc table
   parsing in the golden-sweep workflow".
2. `scripts/bench_golden_set.py` — shells out to `emmy run --bench --golden NAME` and **parses stdout with a regex**
   (`^(Eager PyTorch|torch\.compile|Emmy)\s+([\d.]+)`). It is precisely what `--json` was built to retire, and it
   was never migrated.
3. `tests/perf/conftest.py::_bench_via_subprocess` — shells out to `emmy run --code … --bench` and harvests
   `EMMY_DUMP_DIR/60_bench_compare.json`, a third harvest surface again.

**The capability.** `emmy run --golden-file FILE --bench --record` benches every target in the file at `-O3`, best of
three, and writes the results back into it as `latency:` entries keyed by `Context.hardware_id`. This is not a new
kind of operation: `emmy tune --golden-file` already persists ranking feedback atomically into a working golden, so
write-back to a golden is an established path with an owner (`working_golden.py`). `--record` refuses for the same
reasons the regeneration target does — a stale derived half, or a changed verdict — so a measurement run can never
launder a regeneration or bury a realization regression.

Everything downstream then uses it. Section 3b calls it instead of describing a bench-and-paste ritual, stage 5 calls
it, and Part D's workflow is a rental wrapped around it rather than a script that re-implements it. **An agent that
finds itself writing a benchmark script has found a missing flag; the fix is the flag.**

**What retires.** All three of these are referenced only by `AGENTS.md`'s helper list and by each other — no skill,
recipe, or experiment drives them, which is what makes the removal clean:

- `scripts/bench_golden_set.py` — fully subsumed. Delete it, and drop it from `AGENTS.md`.
- `scripts/diff_perf_results.py` — it diffs two `tests/perf/.results/*.json` snapshots. Once timings live in the case
  files, "diff two runs" is `git diff` on the corpus and "diff two cards" is reading one file. Delete it.
- `scripts/render_golden_bench_chart.py` — it renders the article figures from `bench_golden_set.py`'s JSON, so it
  cannot outlive its input unchanged. Re-point it at the corpus table emitter, or retire it with the other two if the
  figures are regenerated from the table instead.
- `tests/perf/conftest.py::_bench_via_subprocess` — keep the fixture, move it onto `--json` instead of the dump-dir
  artifact, so one record shape serves every consumer.

**Align `run` on the same flags.** `emmy run --golden` takes a *path*, while `emmy compile --golden` and
`emmy tune --golden` take a *name* with the path in `--golden-file`. One flag, two meanings, three commands — exactly
the friction that sends an agent to write a wrapper rather than use the CLI. `run` gains `--golden-file PATH` and
`--golden NAME`; `--golden-file` alone keeps today's behaviour of running every target in the file.

This is mostly *deleting* a translation layer rather than adding one. `_run_golden_targets` already converts the path
form into the aligned pair internally — it loads the file, enumerates names, and for each one sets
`target_args.golden_file = <path>` and `target_args.golden = <name>` before dispatching. And `run.py`'s own
re-invocation has to translate back the other way, emitting `["--golden", args.golden_file, "--target", args.golden]`.
The aligned pair is what the code already thinks in; only the argparse surface disagrees. `--target` becomes
redundant with `--golden NAME` and retires with it, as does the defensive `getattr(args, "golden_file", None)`.

Callers to migrate, all verified: `README.md`, `experiments/golden-bench-2026/kernels/recipe.yaml` in two places,
`tests/benchmark/models/test_golden_bench_2026.py` (which asserts the exact command substring),
`tests/compiler/cli/test_run_golden.py`, and the self-invocation above. The experiments recipe needs a note rather
than a silent rewrite: it is checked-in reproducibility input, and its archived results were produced with the old
spelling.

**This also settles `bench_golden_set.py`, which is already dead.** It passes a card-scoped golden *name* to
`emmy run --golden`, which has required a path for some time. Running it today gives
`cannot load --golden <name>: No such file or directory` for every case — verified. It is not merely superseded by
`--json`; it has not worked in a while, and nothing noticed, which is its own argument for the consolidation.

### The nightly workflow

`.github/workflows/onboard-model.yml` was audited against the section 3b changes. Four properties make the change
safe, and three places would break.

Safe as-is. The `.claude/skills/*` entries are symlinks to `.agents/skills/*`, so there is one copy of the skill to
edit, not two. Both the skill the agent reads and `onboarding_artifacts.py` are loaded from `$WORKFLOW_SOURCE`, a
`git archive` of `github.sha`, so the prompt and its validator are always the same commit and a coupled change lands
atomically — it simply takes effect on the first scheduled run after it reaches the default branch. The
modify-nothing guard covers `.github` and the skill directories, not `tests/`, so an agent may write case files. And
the deadline is 23h30m, which absorbs section 3b comfortably.

1. **`prompts/onboard-model/qualify.md`** — the most important one, and easy to miss because the skill is not the only
   contract. It is the canonical automation prompt, and its **Repository artifacts** section enumerates the allowed
   areas: `recipes/`, `experiments/`, `docker/vllm-emmy-serve/models/`, and a bounded fix under `emmy/`.
   `tests/compiler/realization/cases/` is not among them, so an automated run would be told by the skill to write case
   files and told by this prompt that it may not. Add the corpus to the allowed areas and to the **Output** section's
   summary contract, in the same change as the skill.
2. **`.github/workflows/onboard-model.yml`** — the `opencode run` invocation passes the agent an explicit `--file`
   list, which is its context: README, the onboard-model / tune-kernels / run-experiment skills, and the three
   `prompts/onboard-model/*.md` files. `tests/compiler/realization/ARCHITECTURE.md` is not there, so an automated
   agent would get the instruction to author a case without the conventions that make one valid. Add it to the list.
   This must ride the same pull request: the modify-nothing guard means the agent cannot add it at run time.
3. **`.github/scripts/onboarding_artifacts.py`** — `_relative_artifact` rejects any non-`.py` file under `tests/`, so
   a case file cannot be committed by the nightly job at all. Allow `.yaml` under
   `tests/compiler/realization/cases/`, and reject an `_xfail`-shaped suffix that is not one of the four stages.
   Two subtleties in `_validate_implementation_patch` decide whether the result is usable:

   - a corpus case **must satisfy** the "focused test change" requirement. That check raises when `emmy/*.py` changed
     and no `tests/**/*.py` did; a compiler fix landing with a corpus case and no Python test is exactly the workflow
     this plan wants, so the case has to count as the focused test rather than fail the rule;
   - a corpus case **must not consume** the eight-file / five-hundred-line budget, which exists to bound code churn.
     Evidence is not code.

   `stage_artifacts` separately rejects any file the agent created that the summary does not manifest, so listing
   case files in `artifacts` is mandatory, not a convention. The `compiler` block is unvalidated today, so
   `realization_gaps` is schema-safe to add, but validating it is new code rather than an extension of an existing
   check. `_invalid_result_artifacts` constrains only `experiments/` and `recipes/`, so it needs no change. Cover all
   of it in `tests/github/test_onboarding_artifacts.py`, which already pins these contracts.
2. **`tests/ARCHITECTURE.md`** — amend "Do not load checked-in golden YAML in the per-commit suite" to carve out the
   corpus, stating why it differs: hand-minimized reproducers with no measurement claim, targeting a declared
   capability rather than the live card. Add the directory to the exceptions table and describe the capability gating
   beside `requires_sm90`.
3. **`tests/compiler/realization/ARCHITECTURE.md`** — the four stages, the boundary rule, how to add a row, and why
   pinned-enumeration membership is the primary oracle rather than `unreproducible_pin_flag`.
4. **`emmy/compiler/pipeline/knob.py`** — the `strict=` keyword on `canon_family_value`. The same staleness class
   affects `recipes/*/golden/*.yaml`; wiring strictness into `validate_golden_file` at the promotion and repository
   levels would catch stale repository goldens too, but that is a follow-up, not this plan.
5. **`emmy/compiler/pipeline/search/golden.py`** — `identity` and `latency` as *optional* realization keys, and the
   only format change this plan makes. Optional on purpose:
   `recipes/gemma-4-12B-it/golden/rtx5090_sm120.yaml` is 17k lines with roughly 57 realizations, and the corpus should
   not force churn on files it does not own. The corpus requires the key for its own cases.
6. **`tests/architecture/test_layering.py`** — extend one of the identity invariants so a new fingerprint fact cannot
   be added without the corpus noticing.
7. **`make test-durations`** once after landing; rows land just above the 0.05 s recording threshold.
8. **`tests/perf/ARCHITECTURE.md`** — remove the stale "Emmy currently emits FP32 only" claim.
9. **`AGENTS.md`** (reached by agents through the one-line `CLAUDE.md` include) — the corpus changes what an agent
   does when a test fails locally, and the wrong reflex is cheap and silent, so it needs saying at the top level
   rather than only in the corpus's own `ARCHITECTURE.md`. Add `make test-corpus-regen` to **Key Make Targets**, and
   this to **Running Tests**:

   ```markdown
   ### The realization corpus

   `tests/compiler/realization/` replays pinned schedules from checked-in case files. A case's expectation is its
   filename: no suffix means every stage must pass, `_xfail_<stage>` means it is a known gap expected to fail at
   `offered`, `realized`, `built` or `correct`.

   - A case **without** a suffix that fails is a regression. Fix the compiler. **Never add an `_xfail_` suffix to
     make a red test green** — that converts a regression into a recorded gap and the ratchet stops meaning
     anything.
   - A case **with** a suffix that passes means the gap closed. `git mv` the file to drop the suffix; do not delete
     the case.
   - A **stale case** failure means a kernel identity or a schedule codec changed and the stored derived data no
     longer matches. `make test` detects this on its own, on any machine; `make test-corpus-regen` is the fix. It
     refuses to write when a case's verdict also changed; that refusal is the signal, not an obstacle to work
     around.
   - Latency is stage 5, is `perf`-marked, and needs `-O3`, so `make test` never measures it. It compares the best
     of three runs against the case's recorded latency for the live card, and only an explicit command records an
     improvement.
   - **Never write a benchmark script.** `emmy run --golden-file FILE --bench --record` benches a golden and writes
     its timings back. If it cannot express what you need, that is a missing flag to add, not a script to write.
   - **The corpus never asks for something this machine cannot do.** With no GPU, the only obligation is the stale
     case above, and it is always fixable where you are. In the perf lane on a matching card, a passing case with no
     recorded latency for that card is reported so you can record it — that is how coverage grows. A performance
     regression is reported, never a failure: the nightly turns it into a pull request someone reviews.
   ```

## 6. Part D — the timing-refresh workflow

Stage 5 only bites where a timing exists, and timings can only be produced on the card they describe. Section 3b
fills them opportunistically, on whatever card an onboarding run happened to rent. A scheduled workflow closes the
rest: `.github/workflows/corpus-timings.yml`, which rents one CloudRift GPU, records that card's timings for every
case it can run, and accumulates the result on a rolling pull request.

It reuses the two existing workflows rather than inventing a third shape. The rolling-branch machinery is
`discover-model.yml`'s, verbatim in structure: find the single open pull request carrying the label, refuse to run if
there is more than one, adopt an unpaired branch when the pull request was closed, rebase on the default branch, and
refuse to overwrite a branch whose head moved after selection. The rental, retry, and teardown machinery is
`onboard-model.yml`'s: `emmy vm create gpu` behind a three-attempt loop with `terminate_instances_by_tags` between
attempts, an ephemeral SSH key, and an `always()` teardown that deletes every VM tagged with the run.

### Choosing the card

Rent uniformly at random from what CloudRift currently offers — `provisioning.cloudrift.list_available_instance_types`
is the primitive, already used by `recipe/query.py` for the availability filter — seeded by the run id so the choice
is reproducible from the log, and constrained to a single GPU — this measures kernels, and there is no reason to
hold an eight-GPU node to do it.

Selection deliberately does **not** try to pick a card matching a case's declared capability. Nothing in the tree maps
a card name to a compute capability — `emmy/hardware.py` has `gpu_short_name` and `resolve_instance_type` and no
capability table — and adding one would be a second thing to maintain and get wrong. Instead, discover the live
capability and `Context.hardware_id` **on the host**, then select the cases whose declared `compute_cap` matches. A
card with no matching cases is a clean no-op: report it, tear down, exit zero. That also makes a new card graceful
rather than a failure.

### What it may change

A timing run may add or update `latency:` entries and nothing else. If the diff touches a program wire, a realization
name, an `identity` stamp, or a knob, the corpus it measured was stale and the run must fail rather than commit —
committing would fold a regeneration into a measurement run where nobody is reviewing it. Likewise the run fails, and
records nothing, if any case's verdict changed while it was measuring: a schedule that stopped realizing is a finding,
not a timing to refresh. Enforce both in `.github/scripts/corpus_timings.py` with tests in `tests/github/`, the same
way `onboarding_artifacts.py` bounds what an onboarding run may retain.

Bound the run by wall clock rather than by case count, record what fits, and **name what was skipped** in the pull
request body. A silent cap reads as "this card is fully covered" when it is not.

A regression is proposed, never absorbed. When a case is slower on all three runs the workflow writes the new number
into the pull request and labels it, rather than either accepting it silently or failing forever. A nightly that goes
permanently red because one legitimate correctness fix cost latency is a nightly nobody reads; a human merging the
pull request is what accepts a new baseline, and declining it is what keeps the finding open. The same applies to a
compiler change that moves timings everywhere — the nightly repairs the corpus and the diff shows the size of the
move.

### What the fleet buys

As CloudRift's fleet grows the corpus becomes a cross-GPU matrix: every case, on every card, emmy beside
`torch.compile`. Joining those blocks into a table — case, card, both latencies, the ratio, sorted — is mechanical
row-shaping and belongs in code, as a CSV or JSON emitter. Reading it is not: deciding which of those rows is worth a
compiler change is exactly the judgment `AGENTS.md` keeps out of scripts. Emit the table; let an agent write the
conclusion.

### Why the rolling pull request works here

Because each case is its own file and each card its own key, two runs on different cards touch disjoint lines, so the
branch accumulates coverage across cards without conflict — the same property the filename-per-case design bought for
concurrent onboarding runs. The single-open-pull-request guard plus the rebase handles the one genuine collision, two
runs on the same card.

One consequence worth taking: with this workflow filling timings on a schedule, the local missing-timing gate
(open decision 4) can settle on report-only. A developer is told their card has gaps; the nightly closes them.

## 7. Open decisions

1. Whether stages 3 and 4 run under plain `requires_cuda` (recommended, so a developer box exercises them in
   `make test`) or behind their own marker. They must not be `perf`-marked — that silently drops them everywhere.
2. Whether commit 2 rides this change or a later one. It no longer deletes a file, so it proves less about the
   mechanism than first assumed, and the case for keeping it here is now the ordinary one: a subsuming mechanism that
   leaves its predecessors in place does not subsume them. Split only if review load demands it.
3. The stage 5 band. Ten percent is a starting guess, not a measurement. Decide it after running the corpus
   repeatedly on an idle 5090 and an idle 4090 and looking at the actual spread, and record the number that produced
   it.
4. How loudly the perf lane asks for a missing timing. The principle settles that it is an obligation there and
   nowhere else; what is left is whether it fails the opt-in lane or ends it with a prominent summary. Failing is
   defensible because the lane is opt-in and coverage is its purpose.
5. Part D's cadence and cost ceiling. A random single-GPU rental per run is cheap, but the cadence sets the rate at
   which cards accumulate coverage against a recurring bill; pick it against how fast the corpus is expected to grow.
6. Whether stage 5 replaces `tests/perf/cases.py` as the perf lane's case source in this change or later. Replacing it
   removes a drifted list; deferring leaves two case inventories alive for a while, which is the outcome the boy-scout
   rule exists to prevent.
7. `plans/` currently holds ten files, its cap. A5 retires two of them during execution; until then this plan puts the
   directory one over.
