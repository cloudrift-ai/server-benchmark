---
name: tune-golden
description: Use this skill when the user asks to "tune the goldens", "update the golden configs", "re-tune the goldens", "run the golden sweep", "refresh golden matmul configs", "evaluate goldens and update", "tune/seed the dynamic goldens", or otherwise wants to re-tune the GOLDEN_CONFIGS matmul shapes (static and dynamic/.dynM symbolic-axis entries alike), A/B the greedy pick against the recorded golden, and record genuine improvements into the per-GPU golden YAML. Tunes the whole golden dataset, benches greedy-vs-golden per shape with `emmy tune` / `run --bench`, categorizes (better → replace, same → add, worse → leave), edits the goldens YAML by hand, and writes a findings report to plans/ — unlike tune-model, the target config is known here, so the report analyzes the offline/online prior's expectation against it (rank, per-knob misses, recommendations) plus workflow notes.
version: 0.4.0
---

# Evaluate and update the golden matmul configs

The golden set (`emmy/compiler/pipeline/search/goldens/<gpu>.yaml`, e.g. `rtx5090_sm120.yaml`) is the
deployable-latency ground truth: per shape, the best-known knobs + the emmy-vs-cuBLAS latencies. This skill
re-tunes every shape, checks whether the current deployable greedy pick beats the recorded golden, records the real
wins, and writes a findings report to `plans/` — which shapes the search/prior couldn't reach, why, what to do about
it, and how this workflow itself could improve.

What sets this apart from the `tune-model` skill: here the **target config is known** (the recorded golden), so the
analysis can evaluate the *expectation* directly — where the golden ranks under the offline and online priors, which
knobs the greedy pick misses, and whether the search ever measured it. Use the `emmy` CLI for all of it (`tune`,
`run --bench --golden/--ab`, the `eval` views) — no ad-hoc bench scripts or hand-written SQL.

Requires a CUDA GPU. The set is ~36 shapes on the RTX 5090 (six kinds × sizes × static/`.dynM`). Budget ~2.5 h for the
full single-invocation cold sweep (measured on a 4090, 2026-07-07); a per-shape loop of `tune -c <snippet> --clean`
costs the same wall time but forfeits within-sweep prior transfer — use it only when a report explicitly needs
unbiased per-shape cold-search behavior. Goldens are **hand-maintained YAML** — never dump them with PyYAML (it
destroys the flow-style `{BN: 16, ...}` knob dicts and key order); edit the YAML text directly.

Work dir: keep all temp data **inside the repo** under the untracked `_tune/` folder (e.g.
`_tune/golden-sweep-<gpu>/`) — `_tune/` is gitignored, so the tee'd logs and any `--dump-dir` dumps persist across
reboots (unlike `/tmp`) and the sweep stays reproducible later.

## Step 1 — Tune the whole golden dataset (one invocation, cold)

```
emmy tune --dataset golden --clean 2>&1 | tee _tune/golden-sweep-<gpu>/tune.log
```

This tunes every golden shape in one in-process loop (`handle_tune` over `_tune_targets` — the same codepath as a
single-shape tune, differing only in the dataset it builds), sharing one tune DB / bench worker / prior. This trains
the prior; it does not update goldens.

**Cold is the point — never reuse pre-sweep tuning data.** This skill runs when the existing tuning data is no
longer valid (the compiler / kernels changed), so `--clean` is mandatory and seeding `EMMY_ONLINE_FILE` from an
earlier checkpoint is forbidden: stale measurements don't just fail to help, they actively mislead — the 2026-07-07
4090 A/B (`local_data/golden-ab-4090/ab-findings.md` on the dev box) showed a stale-context seed steering the fp16
warp-`TILE` search into the wrong region (fp16 squares 1.6–2.6× slower than cold, even with more benches on some
shapes). Within-sweep accumulation is different: after the one `--clean`, the prior learns across the sweep's own
(fresh) shapes — which is why the sweep must be **one invocation, never a per-shape loop of `tune -c <snippet>
--clean`** (36 cold restarts forfeit that transfer; the h4096 GEMMs and static/`.dynM` twins are near-identical to
the search).

Time levers that do NOT reuse old data:

- **`--gpus N`** on a multi-GPU box: the shapes fan out near-linearly.
- **`EMMY_O3_TOL=0.10`** (default 0.15) trims -O3 recompiles of non-contenders on the fp16 warp kernels. Do not
  tighten below ~0.10: at 0.05 the 4090 A/B lost a real winner to the -O1/-O3 ranking inversion (a config 11% off
  the -O1 best was the -O3 best; the 5% band skipped its rebench).
- Narrow with `--kernel SUBSTR` (e.g. `--kernel square` or `--kernel q_proj`) to re-tune a subset. Routine sweeps
  may skip the `.dynM` twins of the memory-bound kinds (softmax / rms_norm / reduce / pointwise): they land within
  ~2% of their static twins (2026-07-06 5090 sweep — the masked-tile penalty is warp-tier-only), so the static
  twin's fresh result stands in for them. The matmul / attention `.dynM` twins ARE worth every sweep — the masked
  warp tile is where the static↔dynamic gap lives.
- Do NOT cut `--patience` below the default 50: with a cold start the wide warp-`TILE` fork is found late (the
  4090 A/B regressions were exactly early-stopped fp16 shapes).

## Step 2 — A/B the greedy pick vs the recorded golden, per shape

For each shape, run the deployable comparison:

```
emmy run --bench --golden NAME
```

The kernel table shows the **greedy pick** row (what `run`/`compile` deploy) and one `golden NAME` row per recorded
config, **all benched live this run** — a real A/B. Knob columns are in the header; rows carry positional values.
Compare the greedy `us` against the *best* (min) golden row's `us`.

Two failure semantics to know (both survive, nothing aborts the run — every row, greedy and pinned alike, benches
in its own SIGKILL-able worker job, so a hung kernel never wedges the run or poisons the parent process). One
measurement caveat: the greedy row benches interleaved with the live torch reference while pinned rows bench
emmy-only, so greedy-vs-pinned µs for the same config differ (~7% observed on split-K pairs) — compare pinned rows
against each other and record from `--ab`/golden rows only, never from the greedy row's number:

- **A greedy pick that hangs or blows the bench budget no longer aborts the A/B** — the greedy row reports
  `bench_fail` (with the reason as a `!` line / in `--json`) and the golden rows still bench. No
  `EMMY_KNOBS`-pinned-deploy workaround needed anymore; the run exits non-zero when any row failed.
- **A pin that matches no offered row FAILS its row without benching** (`pin_unmatched` in `--json`,
  `unreproducible pin … NOT benched` in the table) — fix the pin spelling; the number you would have gotten was the
  planner's own pick under the pin's name.

The same `run --bench` also prints a `Backend … vs Eager` latency table above the kernel table: the **`Eager
PyTorch`** row is the live cuBLAS reference (SGEMM for fp32 — `allow_tf32=False`, HGEMM for `*.fp16`), and `vs Eager`
is `eager_us / emmy_us` (>1 = emmy faster than PyTorch, <1 = slower). Note that eager number — it is the
emmy-vs-PyTorch/cuBLAS comparison the report surfaces (Step 7), and it is also the live cross-check on the
shape's recorded `cublas_us` (which is config-independent, so the two should agree within noise).

## Step 3 — Categorize each shape

Using the live A/B (greedy vs best golden row, same run):

- **better** — greedy >3% faster → **replace** the shape's golden(s) with the greedy config.
- **same** — within 3% AND different knobs → **add** the greedy config as an extra entry alongside the existing one(s).
- **same, same knobs** — greedy reproduces a golden → nothing to do.
- **worse** — greedy >3% slower → leave the golden untouched (the prior couldn't reach it).

## Optional fast-math lane (`EMMY_FAST_MATH=1`)

When the user asks for a fast-math sweep (or the card is a consumer die — sm_86/89/120 — where the f16-accumulate
atom pays), run the Step-1 sweep with the umbrella on:

```
EMMY_FAST_MATH=1 emmy tune --dataset golden --clean ...
```

The gate-on enumeration is a **superset** (every standard row stays a fork sibling), so one sweep measures both
regimes. Then categorize **within-regime** — the two lanes never mix:

- **Standard lane**: Step-2/3 exactly as written (`emmy run --bench --golden NAME`, no env). Only configs whose
  knobs carry NO precision-trading realization compete here; the Step-3 replace/add rules apply among them alone.
- **Fast-math lane**: A/B under the gate — `EMMY_FAST_MATH=1 emmy run --bench --golden NAME` — and compare the
  gate-on greedy against the shape's existing `[fm]` entries (those whose knobs carry an f16-accumulate `TILE`
  atom (`…_f16_f16`) or `FAST_EXP: true` — `GoldenConfig.fast_math` is the discriminator, derived from the knobs).
  A fast-math winner that beats the shape's standard golden is recorded as an **additional same-`name` entry
  beside it, never replacing it**: a default (gate-off) compile can't reach a fast-math config, so replacing
  would orphan the shape's deployable golden (and pins are authoritative — it would silently flip precision on
  for default users). Same noise gates as Step 4.

`emmy eval golden` prints the fast-math entries as separate `<name> [fm]` rows and evaluates each regime against
its own enumeration (the rank/reproduction views self-gate on `GoldenConfig.fast_math`), so no env is needed for
the diagnostics. Replay (`run --bench --golden`) needs no env either — the recorded `TILE` pin bypasses the gate.
In the findings report, keep the two lanes' tables separate and note the fast-math accuracy contract (torch-default
fp16 numerics; see the f16-accumulate evaluation in the PR #339 thread).

## Step 4 — Confirm wins above the noise floor (critical)

The `golden NAME` row is a *live re-bench*, and it swings ~10–13% run-to-run for the smaller shapes. So a marginal
"better" (<~13%) can be the golden benching slow this run, not a real greedy win. Before recording any win:

- Re-run `emmy run --bench --golden NAME` once or twice; only record wins that reproduce clearly above the ~10–13%
  noise band. Treat <~5% deltas as noise (no change).

## Step 5 — Edit the goldens YAML

Open the per-GPU file (`goldens/<gpu>.yaml`). Each entry is:

```yaml
  - kernel: matmul
    name: qwen3_06b.q_proj.s128
    M: 128
    N: 2048
    K: 1024
    knobs: {BN: 16, BM: 8, FM: 4, FN: 2, FK: 1, BK: 64, SPLITK: 1, BR: 1, STAGE: '11', RING: 3}
    emmy_us: 18.4
    cublas_us: 20.9
```

When recording a config:

- **`emmy_us`** — the greedy pick's `us` from the **-O3** A/B (`run --bench`), not the -O1 tune ranking pass.
- **`cublas_us`** — reuse the shape's existing value. cuBLAS is the config-independent torch reference (true-fp32 SGEMM
  for fp32 shapes, HGEMM for `*.fp16`), so it doesn't change when our knobs change. The derived `ratio` / `golden` flag
  recompute from the two latencies.
- **`knobs`** — copy the winning kernel's **`record_knobs`** map from the `--json` record verbatim: the realized
  tuning knobs with EVERY schedule family (TILE / REDUCE / STAGE / WSPEC / RASTER) explicitly stamped, OFF spelling
  (`''`) included. Do NOT record just the knobs you pinned or the non-empty table cells — an entry that omits a
  family leaves it to the planner's replay-time fill, which drifts as the planner evolves (the recurring
  unpinned-`REDUCE` phantom-regression class: a recorded un-split config replaying with a surprise `g2k` fill).
- A shape with the **same knobs** as an existing entry but a faster `us` is a codegen win — just lower that entry's
  `emmy_us`, don't add a duplicate.

### Pruning

Keep multiple entries per shape **only** when they're at parity (within ~3%). When a replace makes an old entry
strictly slower (>3%), **delete** the old entry — don't keep superseded-slower alternates.

## Dynamic (`.dynM`) goldens

A matmul golden may mark its M axis **symbolic** — the kernel is then a masked-tile artifact (ceil-div grid +
boundary guard, one cached kernel for any runtime seq_len), tuned and benched at the `Dim` hint. In the YAML this is
the `dynamic:` block; `M` doubles as the hint, and the name carries a `.dynM` suffix:

```yaml
  - kernel: matmul
    name: qwen3_06b.q_proj.s512.dynM
    M: 512            # the Dim hint — MUST equal DEFAULT_SEQ_HINT (512) today, see below
    N: 2048
    K: 1024
    dynamic: {seq_len: {input: x0, axis: 0}}
    knobs: {BN: 32, BM: 8, FM: 8, FN: 4, FK: 1, BK: 64, SPLITK: 1, BR: 1, STAGE: '11', RING: 2}
    emmy_us: 52.0
    cublas_us: 53.5
```

**`M` must equal `DEFAULT_SEQ_HINT` (512)** — the pipeline tiles/benches a symbolic axis at the *global* Dim hint,
not at the traced size, so an M=1024 "dynamic golden" would silently be measured at 512 and duplicate the
(N, K, hint-512) shape (seen live in the 2026-06-12 seeding: a 1024³ symbolic-M trace produced the exact
`kv_proj.s512.dynM` kernel). The schema rejects other values until per-Dim hints are plumbed.

Everything in steps 1–7 applies unchanged — the spec is **part of the config**, so `tune --dataset golden`,
`tune --golden NAME`, and `run --bench --golden NAME` all apply it to the (re-)trace automatically. Never pass a CLI
`--dynamic` next to `--golden` (it's rejected). Specifics:

- A dynamic golden is a **separate shape** from its static twin (different deployment artifact, own variant space,
  own `ShapeKey` — `is_dyn` keeps them apart in every eval/diagnostics join). Never merge or cross-compare their
  entries; the static `qwen3_06b.q_proj.s512` and `…s512.dynM` rows coexist.
- The A/B benches **at the hint** (the table prints a `benched at seq_len=… (symbolic hint)` note); `cublas_us` is
  the hint-shaped torch reference, so the ratio is apples-to-apples *at the hint*. Knob comparisons work exactly as
  for static shapes.
- **Seeding a new dynamic shape** (no recorded entry yet): tune + bench via the snippet + an explicit spec —
  `emmy tune -c "<matmul snippet>" --dynamic seq_len@x0:0` (accumulate into the existing DB/prior — no
  `--clean`), then `emmy run --bench -c "<snippet>" --dynamic seq_len@x0:0`; record the greedy kernel's -O3
  `us` as `emmy_us`, the same run's eager row as `cublas_us`, and the table's search knobs. `x0` is the
  snippet's lhs `(M,K)`; only the M axis may be symbolic (symbolic K is future work and the schema rejects it).
- `eval online --dataset golden` prints a per-shape `SKIPPED: no tuned rows` line for any golden (dynamic or static)
  whose shape has no tuned data — a `.dynM` entry skipping there means the symbolic shape was never tuned, not that
  the join failed.
- The cold `OfflinePrior` ranks `.dynM` shapes under a dedicated masked-tier weight set (`weights_dynamic`,
  selected on the stamped `S_ext_n_symbolic_axis`), fit by `scripts/golden_knob_heuristics.py` over the recorded
  dynamic goldens (2026-06-12 evening refit: five of eight dynM rows rank ≤1, median ~6). Re-run the script after
  recording new `.dynM` goldens — it writes both weight sets into the repo-checked
  `search/prior/offline_weights.json` artifact (use `--out` for a candidate file and A/B it via
  `emmy eval offline --offline-file <file>` before overwriting the default).

## Step 6 — Validate

```
emmy/../venv/bin/pytest tests/compiler/test_golden_configs.py -q
```

This loads every YAML and checks the schema + derived properties. Spot-check a few entries:

```
./venv/bin/python -c "from emmy.compiler.pipeline.search.golden import goldens_by_name; \
print(goldens_by_name('square.512'))"
```

## Step 7 — write the findings report

Save to `plans/golden-sweep-<gpu>-findings.md` (e.g. `golden-sweep-rtx5090-findings.md`). Mirror the tone and evidence
density of the `tune-model` reports (`plans/*-tune-findings.md`; executed ones are deleted — recover the latest via
`git log --diff-filter=D --name-only -- 'plans/*findings*'` + `git show <commit>^:<path>`). Note friction and findings
**as they happen** during steps 1–6 — don't reconstruct from memory at the end.

- **Header**: date, GPU, the exact sweep command, wall time, and the category tally (N replaced / N added / N
  unchanged / N worse).
- **A `## Fork sibling regret` section** (`emmy eval online --dataset nodes`, this card's `-O1` block). The command
  prints every metric TWICE, labeled `=== offline prior ===` / `=== online prior ===` — the two halves diagnose
  different failures, so the section must keep them apart. Structure it as: (1) a 2–3 sentence intro naming the
  columns (offline = the cold-start ranking that decides what a cold sweep measures at all; online = the CatBoost
  this sweep trained); (2) ONE comparison table, one metric per row, one column per half:

  ```
  | metric (-O1, N forks)                     | offline prior | online prior (CatBoost) |
  | --- | --- | --- |
  | TILE fork regret (median)                 | …× | …× |
  | <one row per other family: REDUCE / STAGE / WSPEC / structural medians> | | |
  | worst / 2nd / 3rd-worst TILE fork (one ROW each, shape-labeled)         | | |
  | leaf reachability (mean / median / worst) | | |
  | leaf calibration (median per-op Spearman) | | |
  ```

  Mark any entry whose "best" baseline is physically impossible (FLOP-roofline sanity check on the shape) with
  `(*)` and one footnote line under the table — worst-case entries sit on those baselines, medians are robust;
  (3) a closing diagnosis paragraph that names WHICH half misprices: offline regret ⇒ the cold-start
  weights/features are wrong (fix via `scripts/golden_knob_heuristics.py`); online regret with a cleaner offline
  ⇒ training-data / calibration / featurization problem (the online half also inherits the offline prior's censoring —
  it never sees regions the cold ranking steered away from). Never paste raw eval output as the section body and
  never quote an unlabeled "prior" number. A family ≫1.00x is a steering gap even if every golden A/B passed.
- **Per-shape outcome table**: shape name, greedy µs, best-golden µs, ratio (greedy/best-golden), category, **and a
  cuBLAS comparison when an estimate is available** — a `cuBLAS µs` column (the shape's recorded `cublas_us`, or the
  live `Eager PyTorch` row from the same `run --bench`) and a `vs cuBLAS` column (`greedy_us / cublas_us`, so >1.0 =
  emmy slower than PyTorch/cuBLAS — the "loser" view). Leave the two cuBLAS cells blank for any shape with no
  estimate (e.g. a kernel kind that has no cuBLAS analogue). This makes the absolute emmy-vs-PyTorch gap visible
  per shape, not just the relative greedy-vs-golden delta — a shape can win the golden A/B yet still trail cuBLAS
  (e.g. the fp16 squares: greedy beats a stale scalar golden 3× but is still ~0.6× cuBLAS HGEMM). All emmy /
  golden numbers from the -O3 `run --bench` A/B, never the -O1 tune DB.
- **One `## Finding N — <title>` per prior shortfall**, ordered by how far the pick lands from the golden. The
  "worse" shapes (greedy >3% slower) are the findings: the search/prior failed to reach a config it was literally
  shown. For each, gather the evidence:
  - `emmy eval golden --kernel <SUBSTR>` — the per-knob `found/golden` diff: *which* knobs the greedy pick got
    wrong.
  - `emmy eval offline --kernel <SUBSTR>` — the golden's rank under the cold `OfflinePrior` over the full
    enumeration. A deep rank here is the "poor offline heuristic on this config" signal: the hand-coded weights
    misprice this shape, and patience can't reach it cold.
  - `emmy eval online --dataset nodes --blame [--ablate] --kernel <SUBSTR>` — WHICH features caused the mispricing:
    the per-feature blame table (regret-weighted term diff between the prior's pick and the measured-best sibling,
    per fork family; a BLIND fork means no feature separates the siblings — a featurizer gap, not a weight problem)
    and the ablation Δ (median regret with one feature masked; `< 0` = actively misleading). The command prints one
    attribution per prior half — cite the `=== offline prior ===` blame when recommending a weight refit, the
    `=== online prior ===` blame when the trained model is the suspect; never quote a blame row without its label.
    Cite the blame row in the finding instead of hand-deriving per-knob misses from the weight table.
  - `emmy eval online --dataset golden --kernel <SUBSTR>` — the rank under the *online* prior + the `vs gold`
    -O3 perf column. Deep offline rank but shallow online rank → the heuristic is the problem, not the search.
  - `emmy eval variants --kernel <SUBSTR>` — was the golden config ever *measured* this sweep (reachability),
    and where the deployed pick ranks among measured variants.
  - `emmy eval online --dataset nodes --kernel <SUBSTR>` — the per-family **fork sibling regret** for this op:
    WHICH decision family (TILE / REDUCE / STAGE / …) the search was steered wrong at (regret ≫1.00x), and which
    families to exonerate (1.00x). Locates the miss at a fork, where `eval golden`'s per-knob diff only names the
    end-state knobs. Read the two labeled blocks separately: high ANALYTIC regret on a family ⇒ the cold-start
    ranking steered the sweep away (it also censors what the online model ever sees — fix weights/features);
    high ONLINE regret with low offline ⇒ the trained model unlearned a correct heuristic (training-data /
    calibration problem). The diagnosis differs, so a finding must name which half its regret came from.
  - **Before calling a knob mismatch a search shortfall**, check the `-O3 us` column in `eval variants` or run one
    `run --bench --ab "<golden knobs>"` A/B: -O1 ranking flags often invert at -O3, so an apparent pick miss can be
    the -O1/-O3 gap, not the prior.
- **Each finding ends with a recommendation**, with priority: refit the offline weights
  (`scripts/golden_knob_heuristics.py`) when a shape family is systematically mispriced, a missing `D_*` engineered
  feature when the prior can't see what distinguishes the golden, a patience bump when the golden ranks shallow but
  the search stops early, or an enumeration/eligibility gate (cite `file:line`) when the golden's config was never
  offered at all.
- **End with `## Workflow notes`** — a retrospective on this loop itself, for whoever maintains the CLI and this
  skill: slow steps (what dominated the wall time — tune loop vs per-shape `--bench` A/Bs — and whether a
  flag/cache/narrower `--kernel` could cut it; note the tune.log's `done: ... in N s` line and the sweep's total),
  many-step detours (answers assembled by hand across several commands — each is a missing `eval` view or column),
  flakiness (the noise-floor re-runs of step 4: which shapes, how many re-runs), and output friction (data only in
  logs, hand cross-referencing). One line of symptom + one line of proposed improvement each. If a previous sweep's
  report exists, say which of its workflow notes got fixed and whether the fix held.
- Wrap to ~120 chars (repo-wide markdown rule); tables may overflow.

## Notes

- The isolated `scripts/find_golden_configs.py` regen uses a cold per-shape search that lands well below the warm-prior
  greedy pick, so it can't reproduce these wins — that's why this workflow records from `run --bench` rather than that
  script.
- Commit the YAML edit with a message listing each shape's before→after and why (better / same-diff), per the repo's
  contribution rules (feature branch, `make lint`, `make test`).
