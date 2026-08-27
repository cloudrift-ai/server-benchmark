# Compiler realization corpus

Give Emmy a durable, data-driven regression lane for one failure class: **a schedule that should be realizable is
not**, and the kernel loses to `torch.compile` as a result. Make `onboard-model` surface that class and record it,
instead of leaving it in an ephemeral `_tune/<run>/findings.md`.

Out of scope, deliberately: search shortfall (the schedule realizes, the prior does not pick it) and code generation
quality (the right tier is present and still loses). Both are real findings; neither is a realization gap, and
recording them here would make the lane's ratchet meaningless.

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

### Layout

```
tests/compiler/realization/
  ARCHITECTURE.md
  helpers.py                 # load, regenerate, replay, the four oracles
  test_realization.py        # one parametrized walker over the manifest
  cases/
    index.yaml               # test expectations — the readable list of open gaps
    matmul/<name>.yaml
    attention/<name>.yaml
    fused/<name>.yaml
```

`tests/compiler/realization/` is a kind-organized directory in the sense `tests/ARCHITECTURE.md` already sanctions:
its cases span lowering, the CUDA backend, the pin machinery and the golden loader, and they share one workflow.

### Case file

A working golden document — byte-for-byte what the inventory writer already emits — plus the authored pin and exactly
one new field. Nothing else is stored, because nothing else is necessary.

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
model: synthetic/matmul-f16acc-splitk
```

Why each part is necessary, and why nothing else is:

- `programs` / `target` / `compute_cap` — the reproducer. Stable Torch IR rather than a code snippet, so a frontend
  change cannot silently alter what the corpus tests.
- `name` — already carries `Op.cache_key()[:12]` (`working_golden.py`), so it detects cache-key drift for free.
- `pins` / `knobs` — the authored schedule. Regeneration never produces these, which is what makes the mechanism
  safe.
- `identity` — the record's `deploy_identity`. This is the one genuinely new field. `cache_key` folds the class name,
  the algebra key and the knobs; `deploy_identity` additionally folds the dtype, extent, shape and store fingerprints,
  so a new fingerprint fact — which `ir/tile/identity.py` documents as routine under "Adding a fact" — moves
  `identity` while leaving `name` untouched.

Deliberately not stored: the `S_*` feature dict and the featurizer version. The corpus asserts realization and
accuracy; the featurizer participates in neither, because every fork is pinned and the prior never decides. Also not
stored: a canonical copy of the knobs — regeneration rewrites `knobs` in place to its canonical spelling instead, so a
codec re-spelling shows up as a diff in the field that already exists.

### Manifest

Test expectations stay out of the case file, because the config and realization key sets are shared with
`recipes/*/golden/*.yaml`, and a repository golden has no notion of a blocked stage.

```yaml
- id: matmul/f16acc-splitk-sm89
  file: matmul/f16acc-splitk-sm89.yaml
  realization: k_matmul_5b7645.167d5f47efce
  status: open                 # open | closed
  blocked_at: offered          # offered | realized | built | correct
  reference: numpy             # numpy | greedy   (greedy for exact Loop targets)
  evidence: "rtx5090_sm120.yaml records this structural identity with TILE=…; sm_89 offers no row"
  measured: {emmy_us: 412.0, tcompile_us: 96.0, card: "NVIDIA GeForce RTX 4090"}
```

`measured` is provenance only and is never asserted — the per-commit suite must not depend on a latency number.

### Stages

| Stage | Assertion | GPU |
| --- | --- | --- |
| `offered` | under `pinned_knobs(pins + knobs)`, `enumerate_graph` at the declared capability returns at least one row carrying every pinned value | no |
| `realized` | the graph lowers through `TILE_PASSES + CUDA_PASSES` at that capability, `unreproducible_pin_flag` is `None`, and every pinned family is stamped | no |
| `built` | `CudaBackend().compile(...)` under the pin — nvcc accepts it | yes, exact capability |
| `correct` | run against the reference within `dtype_tol` | yes, exact capability |

`status: open` applies `pytest.mark.xfail(strict=True)` at the declared `blocked_at` stage. When the compiler learns
to realize the row, the strict xfail fails and forces the row to `closed` — the ratchet. Stages 1 and 2 always run at
the declared capability, so an sm_70 lockout is exercised on any box; stages 3 and 4 run only when the live card's
capability equals the declared one.

The PR test job is CPU-only in practice — 1322 s of the 1922 s in `tests/durations.json` is CUDA-marked, which alone
exceeds its 20-minute cap on the single serial CUDA chain — so stages 1 and 2 are the ones that gate a commit. Confirm
this before relying on it.

### Staleness: regeneration, not stamps

Kernel identity and schedule codec spellings change often, so a stored case rots. The failure mode that matters is
silent: a retired knob spelling canonicalizes to itself, matches no candidate, returns zero rows, and reports as a
lockout — a phantom compiler gap. For an `open` row the mirror failure applies: the xfail keeps passing and the
ratchet stops ratcheting.

The gate is regeneration equality. `make test-corpus-regen` decodes `programs[0]`, re-runs the inventory writer under
`Context.from_target(compute_cap)`, re-derives `identity`, re-canonicalizes `knobs`, and compares. Verified: the
derived half round-trips byte-identically — same program wire, same target, same `name` including the identity
digest, no extra keys — and the only diff is the authored `pins` / `knobs`, which regeneration structurally cannot
produce. That property is what keeps a snapshot mechanism from burying a realization regression.

Three rules make it load-bearing:

1. **Regenerate from the stored program, through the library, not through a CLI.** `emmy compile` prints IR stages and
   `emmy trace` writes YAML, but `emmy trace --target sm_89 -c …` still stamps `gpu_name` from the live card, while
   the library path with an explicit context emits none. The library path is machine-independent and needs no torch.
2. **Canonicalize the authored knobs strictly.** `canon_family_value` swallows `ValueError` and returns the raw
   string, which is exactly how a retired spelling survives. Add a `strict=True` keyword to that one function; under
   it, regeneration fails loudly on `STAGE=d2/ring`, `WORK=zzz9x9`, `TILE=mma_m64n64k64_…` or `REDUCE=g2z` — all
   verified to raise — while a valid-but-unreachable pin (`WORK=w7x13`, `TILE=…/f99x99/k8`) parses cleanly and falls
   through to the `offered` stage, where a genuine lockout belongs.
3. **Refuse to write when a verdict changed.** If one commit moves identity and breaks realization, regeneration fixes
   the first and must not let the second ride along. It names the affected rows and exits non-zero; resolving them is
   a review conversation, not a mechanical step.

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
`canon_family_value(strict=)` keyword. Seed with two closed rows and one open row.
*Verify:* green off-GPU and on a 5090; each row under 0.3 s; `make test-corpus-regen` is a no-op on a clean tree.

### Commit 2 — retirement

Nine tests of roughly 1230 retire. That is the real number, and it is small on purpose: the existing suite asserts
*how* the compiler lowers — emitted source, kernel counts, bit-identity between two configs — which a row cannot
express. Anything not listed below stays.

**A2. Retire the accuracy-only half of `tests/compiler/e2e/test_knob_pinning.py`.** Seven of its fourteen tests assert
nothing but an accuracy comparison against a reference and are corpus rows exactly:
`test_norm_linear_fp16_scalar_reduce_tma_alignment`, `test_norm_linear_warp_fused_masked_m`,
`test_mma_matmul_k_split_staged`, `test_scalar_matmul_f16`, `test_unstaged_atom_mma_accuracy`,
`test_masked_tile_accuracy_configs`, `test_scalar_cpasync_mixed_dtype_slabs`. The other seven stay: five assert
emitted source (`test_sgemm_inner_reduce_is_unrolled`, `test_flat_output_sweep_lowers_with_its_axis_bound`,
`test_output_sweep_declines_the_warp_tier`, `test_unrealizable_warp_pin_falls_back_to_a_bound_scalar_grid`,
`test_unstaged_atom_lowers_gmem_direct`) and two assert a refusal message
(`test_warp_tma_pin_refuses_oversized_box`, `test_scalar_cpasync_pin_refuses_odd_stride`). The file roughly halves; it
does not go away.
*Verify:* the seven migrated configs are covered as rows; the remaining seven still pass; the suite is green.

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
`plans/v100-model-onboarding-compiler-gaps.md` are gap inventories already. Convert what qualifies into rows, then
delete both plan files.

Not in scope: splitting the 125 KB `test_matmul_coverage.py`. It is churn that does not serve this goal, and the
one-matrix-per-regime convention argues against a per-tier split.

## 4. Part B — `onboard-model` changes

One new section between "3. Fully qualify the compiler inventory" and "4. Decide Emmy eligibility", plus two
amendments. Everything else is untouched.

### New section 3b — Surface and record compiler realization gaps

Mandatory, and runs even when serving is blocked, matching the rule section 3 already carries.

1. After tuning, run one per-target torch comparison at deployable optimization:
   `emmy run --golden <working.yaml> --bench --bench-backends eager,tcompile,emmy --strict --json <out>`. Parse that
   record; it already carries `record_knobs`, `status`, `flags` and `lane` per row. Do not parse the terminal table.
2. Rank targets by `tcompile_us / emmy_us`, losers first. Selecting, sorting and tabulating structural fields is
   ordinary data handling; the classification below is agent reasoning and stays out of code.
3. Classify each material loss with the taxonomy `tune-kernels` already defines. Only an eligibility or optimization
   lockout, a pin that refuses or fails to lower, or a pin that runs wrong becomes a row. A search shortfall is fixed
   by measuring or reported as a prior finding. A code-generation-quality loss is reported, not recorded.
4. **Name the desired schedule with cited evidence.** Accept only a sibling card's golden carrying that family for the
   same structural identity, the same family already winning at a neighbouring binding, or an explicit roofline
   argument. The citation goes in the row's `evidence` field. Without this the corpus fills with speculation.
5. Minimize to the smallest snippet that reproduces the refusal, draft the case with
   `emmy trace -c "<snippet>" --target sm_<cc> -o tests/compiler/realization/cases/<family>/<name>.yaml`, paste
   `record_knobs` into `knobs`, then run the regen target to normalize the file and stamp `identity`.
6. Prove the row reproduces the gap: it must fail without the xfail and pass with it. Record the exact command.
7. Bounded: at most five new rows per run, and never at the expense of the artifact and cleanup reserve.

Restamping belongs to the pull request that changes a codec or an identity, not to an onboarding run. A run that hits
a stale row reports it and does not regenerate — it is not the author of the invalidation.

### Amendments

- Section 8 gains: every new row reproduces its declared `blocked_at` stage, `pytest tests/compiler/realization` is
  green, `make test-corpus-regen` is a no-op, and every row cites evidence.
- The summary JSON gains `compiler.realization_gaps` — a list of `{id, blocked_at, emmy_us, tcompile_us, file}` — and
  lists the case files under `artifacts`.
- Section 3's golden gate should require a `tcompile` column for every retained target. Without it the skill has no
  signal that anything is losing, and gap surfacing has nothing to rank.

## 5. Part C — infrastructure

1. **`.github/scripts/onboarding_artifacts.py`** — the only hard blocker. `_relative_artifact` rejects any non-`.py`
   file under `tests/`, so a case file cannot be committed by the nightly job. Allow `.yaml` under
   `tests/compiler/realization/cases/`; exclude those files from `_validate_implementation_patch`, since evidence must
   not consume the eight-file / five-hundred-line bounded-fix budget meant for code; require every new case file to
   appear in the summary's `compiler.realization_gaps`. Cover it in `tests/github/`.
2. **`tests/ARCHITECTURE.md`** — amend "Do not load checked-in golden YAML in the per-commit suite" to carve out the
   corpus, stating why it differs: hand-minimized reproducers with no measurement claim, targeting a declared
   capability rather than the live card. Add the directory to the exceptions table and describe the capability gating
   beside `requires_sm90`.
3. **`tests/compiler/realization/ARCHITECTURE.md`** — the four stages, the boundary rule, how to add a row, and why
   pinned-enumeration membership is the primary oracle rather than `unreproducible_pin_flag`.
4. **`emmy/compiler/pipeline/knob.py`** — the `strict=` keyword on `canon_family_value`. The same staleness class
   affects `recipes/*/golden/*.yaml`; wiring strictness into `validate_golden_file` at the promotion and repository
   levels would catch stale repository goldens too, but that is a follow-up, not this plan.
5. **`emmy/compiler/pipeline/search/golden.py`** — `identity` as an *optional* realization key. Optional on purpose:
   `recipes/gemma-4-12B-it/golden/rtx5090_sm120.yaml` is 17k lines with roughly 57 realizations, and the corpus should
   not force churn on files it does not own. The corpus requires the key for its own cases.
6. **`tests/architecture/test_layering.py`** — extend one of the identity invariants so a new fingerprint fact cannot
   be added without the corpus noticing.
7. **`make test-durations`** once after landing; rows land just above the 0.05 s recording threshold.
8. **`tests/perf/ARCHITECTURE.md`** — remove the stale "Emmy currently emits FP32 only" claim.

## 6. Open decisions

1. Whether stages 3 and 4 run under plain `requires_cuda` (recommended, so a developer box exercises them in
   `make test`) or behind their own marker. They must not be `perf`-marked — that silently drops them everywhere.
2. Whether commit 2 rides this change or a later one. It no longer deletes a file, so it proves less about the
   mechanism than first assumed, and the case for keeping it here is now the ordinary one: a subsuming mechanism that
   leaves its predecessors in place does not subsume them. Split only if review load demands it.
3. `plans/` currently holds ten files, its cap. A5 retires two of them during execution; until then this plan puts the
   directory one over.
