# Plan: extend the gemma-4 golden set, seed the deploy evidence, and make serving deploy the optimal kernels

## BLOCKER FOUND (2026-07-19, later session) — the tracer fix took mlp_down off the mma tier, 66×

The twin capture harness now exists (`scripts/capture_gen_twins.py` → `gen_runner.trace_split`, the same trace
`_compile_split` makes, so a captured twin IS the graph serving compiles) and the 12 gemma-4 twins were re-captured
post-tracer-fix. A golden-relevance audit (`scripts/diagnostics/audit_golden_match.py`, golden tier as the ONLY
evidence) over them says:

**35 MATCH, 10 DRIFT, 48 GAP** — against the pre-fix baseline's 21 / 0 / 60. Coverage nearly doubled (the `free_max`
aspect fix and the tracer fix's real cast nodes both pay off: the qk-norms now match `qknorm.k256`). But DRIFT went
0 → 10, and 8 of those are one regression:

| twin (M=32 post) | down-proj kernel | TOTAL |
| --- | --- | --: |
| pre-fix graph | `Contraction mul_4 @ linear_3_wt`, `mma_m16n8k16_f16_f32`, `d2/tma/ring`, on its golden | **274.0 µs** |
| post-fix graph | `Reduction[a2] planar` — no mma atom, `b128`, grid 122880 | **18,202.2 µs** |

The other 2 drifts (`mlp_geglu.m256.cut`) are correct behavior — the PLACE-golden refusal (8731f3ff) working.

**Mechanism, traced end to end.** The tracer fix makes gate/up read an f16 `type_as_1` instead of f32 `mul_3`. The
MLP edge consequently stops being ONE 2-channel fused node and becomes two separate contractions, which leaves the
GeGLU combine (`gelu·mul`) unmaterialized — so loop fusion inlines it into down_proj's K loop. `bind_contraction`
requires the A operand to be a plain `Load`; with the combine inlined there is no A tensor, so the cell demotes to
PLANAR → scalar. Ruled out: o_proj carries an identical f32 output cast and matches fine, so it is the computed-A
operand, not the cast. The old graph compiles fine under current code, so it is the graph, not a pass.

**`make test` (2535) does not catch this** — no test asserts the mlp edge stays on the mma tier.

### Two fixes attempted and REVERTED — record both, neither is the answer as written

1. **A fusion guard** (refuse to inline a stat-free compute producer into a contraction's A). Works — restores
   `linear_3` to its golden, DRIFT 0. Rejected on direction: the fix does not belong in fusion. Note for whoever
   picks this up: the pass's own docstring already knows this hazard (*"one harmful silu→down_proj fusion lands"*)
   and `_BLOWUP_FACTOR = 8` was tuned to exclude it — the ratio guards cannot see this cliff, because inlining a
   per-k pointwise chain into a big matmul grows work only ~2×.
2. **Binding the stat-free cone as computed-A** (relax `_is_clean_contraction` + let `bind_contraction` return the
   cone as a `Body` A operand). Structurally it works — the node binds as
   `Contraction linear_1_reduce @ linear_3_wt (mma_m16n8k16_f16_f32)` — but it does **not** compile:
   `ValueError: scratch buffer 'linear_2' has no consuming launch`. The emitted kernel signature is
   `(linear_1_reduce, linear_3_wt, linear_3)` — **the A compute-fill codegen plumbs exactly ONE operand buffer**,
   and the GeGLU cone reads two. It also broke 8 `test_fused_edge::test_fused_map_matmul` cases ("the fused edge
   must be ONE kernel, got 2" — the relaxation changes split-K for the existing single-buffer fused edges).

### (a) LANDED — bind the cone off the ⊗ lift; the cliff is gone, the gap is not

The second attempt's real defect was not "the codegen plumbs one buffer" — the sync compute-fill
(`_sync_operands.a_value`) already evaluates a whole cone inline, multi-buffer included. It was that
**`bind_contraction` bound A as "the first (m, k)-indexed load"**, and once fusion inlines a cone, a cone-INTERNAL
load is (m, k)-indexed too. It bound `linear_1_reduce` as A and emitted `linear_1_reduce @ W` — cp.async-ing the
gate projection into the A slab with the gelu and the up projection silently **gone**. A wrong kernel, not just a
missing parameter.

Fix: bind A off the ⊗ **lift** instead. The lift names the true operands — B is its (n, k)-indexed load, A is the
other argument, either a plain `Load` (clean gmem-direct, the old path, bit-identical) or a computed cone (which
rides the sync compute-fill exactly like the norm→linear cone, just with no statistic prologue). Plus
`_is_clean_contraction` now also classifies the computed-A cell as `CONTRACTION` so it reaches that binder.

Measured (5090, cold, goldens-only):

| twin | HEAD | with the fix | |
| --- | --: | --: | --: |
| `post32` | 18,202.2 µs | **1,137.6** | 16.0× |
| `post256` | 144,202.2 µs | **4,301.2** | 33.5× |
| `post-sym` | `bench_fail` | **2,290.3** | now runs |

`make test` 2535 passed. One test changed deliberately: `test_fused_map_matmul` asserted the fused edge is exactly
ONE kernel, but a stat-free cone now binds as a computed-A `Contraction` that legitimately offers split-K, and a
chosen split-K row emits a `__partial` + finalize pair which still carries the cone inline. The assertion was
stricter than the invariant it documents, so it now asserts the invariant itself — `xn` never round-trips through
gmem (no kernel takes or produces it) — and the warp cell still demands `emmy_mma`.

**What is NOT fixed.** `post32` is 1,137.6 µs against the pre-fix graph's 274.0 — still **4.2×** off. The fused
computed-A down-proj runs 892 µs at M=32 where the materialized `mlp_down.m32` golden records 74.1. So (a) removes
the scalar cliff but does not reach the materialized form, and the 8 `mlp_down` DRIFTs remain: those goldens
describe the CUT (materialized) form, and a computed-A node only offers `d1/sync`, so no offered candidate realizes
them. The audit is unchanged at **35 MATCH / 10 DRIFT / 48 GAP**.

### (b) LANDED — the stat-free cut, pinned-verified; selection is now a SEEDING task

Three changes: `020_cut_edge` accepts a bare computed-A `Contraction` (the stat-bearing MONOID entry is unchanged);
`010_recognize` offers a `PLACE@cone: cut` row beside the fused rows for a single-fold stat-free cone; and the cut's
free-axis guard admits `{m, n}` (the lift peels BOTH, where the MONOID composition keeps `n` inside the map body).

One non-obvious blocker, worth remembering: **`stat_prologue()` classified the GeGLU's scalar constants as
statistics.** The seam splits at the first K-indexed stmt, so a stat-free cone's `0.044715 / 0.797885 / 0.5` loads
land in the prologue and read as four "bridged statistics", tripping the one-statistic limit — the cut silently
skipped with `multi-statistic cone`. A loop-invariant scalar needs no bridging kernel and no smem row; only an
m-indexed prologue def is a real statistic. The re-homing is scoped to the stat-free branch (applying it to the
MONOID path breaks it: `Assign 'v0': arg 'in0' not defined`).

`post32` TOTAL, 5090 cold, goldens-only:

| form | µs |
| --- | --: |
| HEAD (PLANAR scalar demote) | 18,202.2 |
| (a) fused computed-A | 1,137.6 |
| **(a)+(b) cut, `EMMY_PLACE=cut`** | **317.0** |
| pre-fix graph, for reference | 274.0 |

The cut consumer lowers to `Contraction linear_3__cone @ linear_3_wt (mma_m16n8k16_f16_f32)` — exactly the plain
gmem-A matmul `mlp_down.m32` records. `make test` 2535 passed, lint clean.

**Unpinned it still deploys FUSED (1,171.6 µs), and that is the designed state, not a bug.** The cut row is offered;
nothing selects it, because the recognizer deliberately discards the cut row's own schedule (`020_cut_edge` re-enters
`010` per half) and the `mlp_down.m32` golden describes the CONSUMER half, not this fork. Selection needs a golden
recording **`PLACE@cone: cut` at this shape against the cut's measured TOTAL** — the same contract a split-K golden
uses. That is a measured YAML edit per the tune-golden skill, not a code change, and it is the remaining work:

1. Seed `PLACE@cone: cut` goldens for `mlp_down` at m32 / m256 / dynM (local + global) from the pinned A/B above.
2. Re-run the audit — the 8 `mlp_down` DRIFTs should become MATCHes and the consumer halves pick up the existing
   `mlp_down.*` plain-matmul goldens on their own forks.
3. Only then re-run the serving A/B; before that it measures the fused fallback.

### On seeding the DB from goldens — measured inert on this branch

- The 35 MATCHing forks already deploy from the golden tier, which sits ABOVE the DB tier in `greedy.decide`. A perf
  row changes nothing.
- The 8 drifting forks cannot be helped by evidence: the fork offers 12 knobless rows, so the golden's config is
  never enumerated. Seeding a config that is not offered is a no-op.
- The 48 GAPs have no golden to seed from.


## STATUS (2026-07-19) — Phases 1–4a landed, 5 blocked on a capture harness

11 commits on `feature/serve-boot-resolution-cache`, `make test` green (2535). Per-phase detail is inline below;
each phase carries an EXECUTED / ATTEMPTED-AND-REVERTED note with its measurements.

| phase | state | headline |
| --- | --- | --- |
| 1 boot pathologies | **done** | prior + DB-index memoized per process, ~37 s off a 96-program boot (the `sig_groups` memo was only ~2%, kept as drift-path insurance) |
| 2 fused-fork audit | **done** | 12 twins, 0 DRIFT; all three Phase 3b "NO class" holes need no new class; found + fixed a silent aspect-blind wrong deploy |
| 3a seeding | **done** | golden coverage **21 → 47 MATCH**, cold rescues up to **820×**; also fixed a computed-A misclassification |
| 3b new kinds | **dropped** | resolved by the Phase 2 audit — no new classes needed |
| 4a cut selection | **done** | `PLACE@cone` is a schedule knob; greedy chooses the cut unpinned at M=256, **921.8 → 365.0 µs (2.53×)**, M≤128 stays fused |
| 4b GeGLU epilogue | **closed, reverted** | 15× WORSE (drops the channel off the mma tier); prize was only 1.5% — see the section for the measurements |
| 5b cold-serve verify | **partial pass** | serves correctly from repo goldens alone (64/64, 0 failures); per-op deployed-kernel dump not done |
| 5c e2e A/B | **done, then invalidated** | +11.8% req/s / −13.8% TTFT / −14.5% TPOT vs the same-session baseline — but the later tracer fix invalidated the box's tuning state, so this must be re-run |
| 5a twin re-tune | **NOT started — now a prerequisite** | blocked: the twins have no capture mechanism |

**The one blocker.** The twin JSONs under `_tune/decode-twin-readiness/` were produced ad hoc in an earlier session;
there is no documented capture path. Every remaining measurement depends on regenerating them, so a capture harness
against `EmmyGenRunner` is the next piece of work.

### Remaining steps, in order

1. **Write the twin capture harness** (`EmmyGenRunner` → per-program graph JSONs). Blocker for everything below.
2. **Phase 5a — re-capture the twins and re-tune `twins.db`** (~70 min GPU). No longer optional: the tracer fix
   changed the graphs, invalidating both the cubin cache and `twins.db` (its evidence is keyed to the pre-fix
   structural signatures).
3. **Re-run 5c** on the re-tuned box. Only then is the tracer fix's serving impact measurable — it is currently
   **unmeasured**, and a run in the invalidated state produced a meaningless 0.04 req/s / 112 s TTFT.
4. **The prefill lever (~31% of prefill, precision-free)** — still closed. The norm→qkv cones cost 468 µs/layer,
   95% of the pre-attention half, and the cut is ~4×. Blocked on the cone being unbindable; the tracer fix did NOT
   unlock it (fusion re-inlines the f16 cone). Needs recognizer work — see the Phase 4 section.
5. **Optional / smaller**: the in-8 decode A/B arm; the fast-math lane (~31% of the mlp edge, but an f16-accumulate
   numerics trade needing an explicit decision); the dynM fused twins.

### Two rules this session paid for

- **Never compare emmy across sessions.** The earlier report's 7.11 req/s does not reproduce — but neither does it
  at the PRE-session commit (5.35), so the gap is the box, not a regression. Stock reproducing its old number
  exactly (14.27 vs 14.25) is misleading: it runs native kernels with no emmy compile and is far less sensitive.
  Every emmy-vs-emmy claim must be same-session.
- **Verify which graph form a claim is about.** Three proxies each disagreed with the serving graph and each cost a
  conclusion: the *snippet* fuses the cone where the model does not; the `--layer` trace keeps a leading batch dim
  `(1, 256, 3840)` where serving is flattened `(256, 3840)`, making seeded m256 goldens spuriously DRIFT; the
  captured *twins* carry the pre-fix f32 dtype.

## Goal (and the honest split)

Make `emmy serve` (gemma-4-12B) deploy the **verified-optimal kernel for every in-model op, cold, from the
repo** — dropping the box-local `twins.db` dependency and the 15–21 min serve boot. This is a **selection /
portability** goal, and it is where "extend goldens + seed the DB + optimal serving" actually lives.

It is NOT the same as "beat stock vLLM". Two independent findings bound that:

- The single fused computed-A decode kernel is at its memory-pipeline floor and **provably cannot beat cuBLAS**
  (`plans/computed-a-pipeline-and-sdpa-oproj.md`, VERDICT). emmy's decode TPOT floor is ~22.7 ms; stock is 18.1 ms
  (`plans/golden-sweep-gemma4-rtx5090-findings.md`, e2e A/B). Perfect selection reaches emmy's floor, not stock's.
- The way to WIN the fused edges is the **split** (`PLACE@cone=cut`, landed): N single-channel matmuls each beat
  cuBLAS ~1.1× (M=256: 270 µs = 1.17× the unfused eager pair). That is a *prefill* perf lever, folded in below as
  Phase 4, but the stock-beating decode kernel is a research-class rewrite explicitly out of scope here.

So: Phases 1–3 make serving deploy emmy's best-known kernels cold (the ask). Phase 4 adds the one perf lever that is
also a *selection* problem (cut-at-prefill). Phase 5 verifies and states the honest ceiling.

## Grounding — how serving picks kernels, and why goldens don't yet carry it (verified this session)

- **Serving compiles through the ordinary backend**: `serving/gen_runner.py::EmmyGenRunner` →
  `CudaBackend(tune_db="auto").compile` (`backend/cuda/backend.py:135`) → `Pipeline.run` →
  `greedy_decide` (`pipeline/search/policy/greedy.py`). The generative 12B runner compiles **~96 programs**
  (`pre`+`post` twins × 48 layers).
- **Deploy hierarchy** (`greedy_decide.decide`, greedy.py:598–619): (1) golden tier
  (`_golden_evidence_index`/`_golden_pick`), (2) `evidence_pick` (online reservoir, -O3), (3) `_db_measured_pick`
  (tune-DB `perf` rows), (4) model argmin. Goldens ship with the repo (`GOLDEN_CONFIGS`); `evidence`/`perf`/online
  are box-local caches (`EMMY_TUNE_DB` default `~/.cache/emmy/autotune.db`, `EMMY_ONLINE_FILE` default
  `~/.cache/emmy/online.json`; `config.py`). `_tune/` (twins.db, twins-online.json) is gitignored.
- **Why goldens don't carry serving yet**: the in-model fused (computed-A) forks stamp a pre-split geometry whose
  `ShapeKey` didn't match the isolated fused golden's key, so they fell through to model argmin. PR #398 fixed the
  **norm→qkv cones** (rebuild the pre-split fork key to `kind="fused"`); `mlp_geglu` (gate⊗up) already matched
  post-split. The residual: the fused prologue goldens exist **only at `.m32`**, so prefill (M=256) and symbolic
  (dynM) fused forks have no golden and misdeploy; and three fused forms have **no golden kind at all**.
- **Seeding fact** (verified, golden-sweep-5090 report): `run --bench --golden --record-nodes` writes the **`node`**
  table (offline-prior training feed) → **zero deploy effect**. Only `tune`-populated `perf`/reservoir evidence
  seeds the deploy pick. Isolated-golden replay does NOT carry serving; the in-model **twin** tune (`twins.db`) does.

## The exact golden gaps (5090 `rtx5090_sm120_gemma4.yaml`, verified inventory)

| in-model fused form | golden kind | decode `.m32` | prefill `.m256` | symbolic `.dynM` | schema models it? |
| --- | --- | :-: | :-: | :-: | --- |
| norm→q (`norm_q_proj`) | `NormLinearGoldenConfig` | ✓ (L951) | ✗ | ✗ | yes (prologue) |
| norm→k/v (`norm_kv_proj`) | `NormLinearGoldenConfig` | ✓ (L966) | ✗ | ✗ | yes (prologue) |
| gate⊗up (`mlp_geglu`) | `MlpGeGluGoldenConfig` | ✓ (L985) | — (declined, use cut) | ✗ | yes (multi-channel) |
| gate/up CUT consumer (`mlp_gate_up_split`) | `MatmulGoldenConfig` | — | ✓ std+fm (L1008/1017) | ✗ | yes (plain matmul) |
| per-head q_norm/k_norm **epilogue** on the projection output | — | ✗ | ✗ | ✗ | **NO class** |
| fused **sdpa→o_proj** (attention out as computed-A) | — | ✗ | ✗ | ✗ | **NO class** |
| **linear→norm epilogue** (o_proj+post_attn_norm, down_proj+post_ff_norm) | — | ✗ | ✗ | ✗ | **NO class** |

Plain matmul projections (q/kv/o/*_global, mlp_gate_up/down) and attention (hd256/hd512) are fully covered across
`.m16/.m32/.s2048/.dynM`. The gaps are exactly the **fused** forms at non-decode M, plus three schema holes.

---

## Phase 1 — fix the serve-boot pathologies (unblocks everything; bounded; ships in the repo)

Both are **per-program-compile** costs paid ~96× because the memo sentinels live in the `greedy_decide` closure, not
the process. This is the single biggest serving-usability lever and is independent of any golden work.

- **1a. Cache the parsed online prior per process.** `_load_prior_safe` (greedy.py:97) → `load_prior` →
  `OnlinePrior.load` (`prior/online.py:182`) base64-decodes + rehydrates the CatBoost blob on **every** program
  compile; there is no `@lru_cache` (contrast `_tile_pipeline`, greedy.py:49, which has one). Add a process-level
  memo keyed on `(online_path(), mtime)` so the 56 MB `online.json` parses **once**. → verify: py-spy a serve boot
  shows one `json.loads`/`from_json`, not ~96; boot drops from ~15 min-in-resolution to seconds.
- **1b. Cache the DB `perf` index per process.** `_db_measured_index` (greedy.py:247) is rebuilt once per
  `Pipeline.run` (greedy.py:551, `db_state=[None]` is per-closure). Add a process-level cache keyed on
  `(db path, mtime, frozenset of the three context keys)`; invalidate on mtime change. → verify: the ~96× rebuild
  over the 57-shape/twin DB collapses to 1×; the documented 21 min twins.db boot drops to ~11 (cubin) or less.
- **1c. (workflow, no code) `tune --clean` scope.** `--clean` wiped `~/.cache/emmy/cubin`, forcing a 13 min cold
  recompile on next serve. Out of scope to fix here; note it in the tune-golden skill.

Success: a serve boot with a populated `twins.db` + online file is compile-bound (cubin), not resolution-bound.
This is a prerequisite for any of the "does serving deploy the golden" verification below being fast enough to run.

## Phase 2 — make every in-model fused fork match its golden cold (extend PR #398)

PR #398 rebuilt the norm→qkv pre-split fork key. Confirm the rest of the fused-fork family also matches, using the
committed spy method (`scratchpad/golden_spy.py`: monkeypatch `greedy._golden_pick`, compile the twin JSONs under
`_tune/decode-twin-readiness/` with `CudaBackend(tune_db=None)`).

- **2a. Audit all twin programs** — `pre32/post32/pre256/post256/pre-sym/post-sym` and the `-global` variants — and
  record, per fork: rebuilt `ShapeKey`, matched golden, MATCH / DRIFT / no-golden-for-shape. → verify: table of
  every fused fork's verdict. (Already done for pre32/post32/pre-sym: norm→qkv MATCH, gate⊗up MATCH, o_proj/down
  MATCH; the qk-norm reduces are `no-golden-for-shape`.)

  **EXECUTED (2026-07-19, 5090, golden tier only: `CudaBackend(tune_db=None)` + empty online, deployable -O3).**
  All 12 twins compiled, 0 failures. **21 MATCH, 60 no-golden-for-shape, 0 DRIFT.** Zero drift is the headline:
  no golden is matched-but-unrealizable, so the enumeration still offers what every recorded golden pinned.

  | twin | MATCH | gap | note |
  | --- | :-: | :-: | --- |
  | `pre32` / `post32` | 3 / 3 | 7 / 2 | norm→q, norm→kv, o_proj, mlp_down, mlp_geglu all deploy from tier |
  | `pre32-global` | **0** | 7 | no global fused prologue golden at ANY shape |
  | `post32-global` | 3 | 2 | o_proj_global, mlp_down, mlp_geglu |
  | `pre256` | 1 | 9 | only `qknorm.k256`; every fused prologue misses |
  | `pre256-global` | 1 | 6 | the 1 "MATCH" is a **WRONG deploy** — see 2b |
  | `post256` / `post256-global` | **0 / 0** | 5 / 5 | prefill post-twin deploys NOTHING from the tier |
  | `pre-sym` / `post-sym` | 1 / 4 | 9 / 1 | symbolic o_proj/mlp_down/rms_norm dynM carry; fused prologues do not |
  | `pre-sym-global` / `post-sym-global` | 1 / 4 | 6 / 1 | same shape as the non-global sym twins |

  Two findings the plan did not anticipate:
  - **`post256` / `post256-global` deploy nothing at all** — the prefill post-twin (o_proj + mlp_down + gate⊗up at
    M=256) has zero golden coverage. This is exactly the edge behind the 2332 ms TTFT / 7.11 req/s prefill wall.
  - **`pre32-global` deploys nothing** while `pre32` gets 3 — the global (hd512) fused prologues have no golden at
    any M, so the 8 global layers misdeploy their projections cold even at decode.
- **2b. For each `no-golden-for-shape` fused fork**, decide: is it a MISSING GOLDEN (Phase 3) or a MISSING SHAPEKEY
  REBUILD (a #398-style pre-split classifier miss)? The dividing test: does `_fork_shape_key` produce the key the
  golden *would* carry? If the key is wrong → extend `_fork_shape_key`; if the key is right but no golden exists →
  Phase 3.

  **EXECUTED. Verdict: every fused fork is correctly kinded `"fused"` — PR #398's rebuild fires on all of them,
  including the global and dynM variants. So NO `_fork_shape_key` extension is needed; the 60 gaps are genuine
  missing goldens (Phase 3a).** With one exception, which is a real defect:

  **DEFECT — `ShapeKey` is aspect-blind on every kinded shape, and it causes a confirmed WRONG deploy.**
  `pre256-global`'s norm→kv fork (M=256 × N=512 global) produces
  `ShapeKey(free_prod=131072, reduce_max=3840, is_warp=True, is_dyn=False, kind='fused', free_max=0)` — byte-identical
  to `gemma4_12b.norm_q_proj.m32`'s key (M=32 × N=4096 local), because `32*4096 == 256*512 == 131072`. The audit shows
  it as MATCH deploying at a fabricated 24.1 µs. This is worse than a gap: it silently ships a config tuned for a
  different shape, reports someone else's latency, and emits no DRIFT warning.

  Root cause is by design, not a rebuild miss — `data/shape.py` `__post_init__`:
  ```python
  if (self.kind or self.is_dyn) and self.free_max:
      object.__setattr__(self, "free_max", 0)
  ```
  `free_max` (`max(M, N)`, the aspect disambiguator added by #386) is force-zeroed for ANY non-empty `kind` or
  dynamic shape, so `fused` / `rms_norm` / `flash` keys carry only the free PRODUCT. Plain matmul keys (`kind=""`,
  static) keep `free_max` and are therefore NOT aspect-blind — which is why o_proj/mlp_down never collide.

  Fixing it means letting `free_max` survive on kinded static shapes, which re-keys every recorded fused/rms_norm
  golden and can shadow or unshadow matches across the tier — the exact hazard 2c guards. **Gated on an explicit
  decision; do NOT fold it into a seeding change.**

  **Resolved for free: all three Phase 3b "NO class" schema holes need no new class.**
  - *Per-head qk-norm* — deploys as a STANDALONE `kind='rms_norm'` fork (reduce 256 local / 512 global), and
    `gemma4_12b.qknorm.k256` already matches it at `pre256`. It is a missing rms_norm SHAPE, not a missing class.
  - *Linear→norm epilogue* — o_proj/mlp_down appear as plain `kind=''` matmul forks with the norms as separate
    `rms_norm` forks at BOTH m32 and m256, i.e. they SPLIT. The plan's "add a kind only if they fuse-and-
    underperform" test resolves to: they never fuse. No class needed.
  - *Fused sdpa→o_proj* — o_proj is a plain matmul (`reduce_max` 4096 local / 8192 global) in every twin; the fused
    attention→projection form is never produced, so there is nothing to record a golden for.

  Incidental: the fused prologue forks offer **12k–14k candidates each** (one reached 41,568), which is the source
  of the `_db_measured_pick` cost measured in Phase 1d.
- **2c. Guard against over-firing / cross-kind shadows.** Re-run the `test_golden_evidence_deploy` /
  `test_shape_key_kinds` suites plus a full twin compile after every `_fork_shape_key` change (no new DRIFT
  warnings, no regression on the matmul/attention forks).

  **SATISFIED trivially — 2b concluded no `_fork_shape_key` change is needed, so there is nothing to over-fire.**
  Baseline recorded for the future `free_max` change: full 12-twin compile = 0 DRIFT, 21 MATCH, 60 gaps;
  `make test` green (2530 passed / 35 skipped) after the Phase 1 commits. Any `free_max` re-keying must reproduce
  the 21 MATCHes minus the one wrong `pre256-global` deploy, and must not introduce DRIFT.

## Phase 3 — extend the golden set with the necessary kernels

Two sub-tracks. **3a is bounded and unblocks prefill/dynamic serving. 3b is schema work gated on the perf pipeline
(do NOT seed a losing golden).**

**RE-TRIAGED against the Phase 2 audit + the YAML's recorded intent (2026-07-19).** Not every audit gap is a
seeding target — one class is seeded-by-design-NOT-to-be:

- **`mlp_geglu.m256` (post256 / post256-global `mul_4`) is NOT a Phase 3a gap — it is the Phase 4 problem.** The
  5090 YAML (the `mlp_geglu` block) already records the decision: the fused form is memory-pipeline-stall bound at
  M=256 and provably cannot beat cuBLAS, so *"a fused `mlp_geglu.m256` golden would only ever record a losing config
  and muddy the seeding"*. The perf form is the CUT, and its consumers **are already seeded**
  (`mlp_gate_up_split.m256`, std 187.3 / fm 125.6 = 1.20× cuBLAS). The audit shows `mul_4` still forking FUSED with
  no golden because *"the cut fires only under the `PLACE@cone=cut` pin today"* — exactly Phase 4. **Seeding a fused
  m256 golden here would be actively harmful. Do not.**
- **The same question is OPEN for `norm_*_proj.m256`** and must be measured, not assumed. The mlp verdict rested on
  "a 2-fold node can't ride `d2/tma/ring`", which does not transfer directly — norm→q/kv are SINGLE-channel. But the
  blocker the YAML names for the cut consumer is that *"a single-fold matmul reads a clean gmem A, so it rides
  d2/tma/ring"*, and a fused prologue's A is COMPUTED, not clean gmem. So fused-at-M=256 plausibly loses here too.
  → measure fuse-vs-cut before seeding either.

Resulting Phase 3a priority (highest confidence first):
1. **`norm_q_proj_global.m32` + `norm_kv_proj_global.m32`** — `pre32-global` has ZERO golden coverage, and this is
   DECODE M=32 where fusion is the known winner (the `mlp_geglu.m32` precedent, 0.92× eager). 8 global layers
   currently misdeploy cold. Highest value, least ambiguity.
2. **Plain matmul at M=256** — `o_proj.m256`, `o_proj_global.m256`, `mlp_down.m256` (`kind=""`, the same kind as the
   existing `.m32`/`.dynM` entries). Closes most of `post256`'s zero coverage with a well-understood form.
3. **Measure fuse-vs-cut for `norm_*_proj.m256`**, then seed whichever wins (fused golden, or cut-consumer matmuls).
4. **dynM fused twins** + the `rms_norm`/`qknorm` shapes at the uncovered `free_prod`s.

- **3a. Seed the missing fused-prologue + cut-consumer twins** (existing kinds, just missing shapes). Use the twin
  tune, NOT isolated `--golden --record-nodes` (which is deploy-inert):
  - `norm_q_proj.dynM`, `norm_kv_proj.dynM` — symbolic-M twins (`NormLinearGoldenConfig(dynamic=True)`); PR #398
    already routes the dynM cone key, so seeding these makes symbolic-prefill norm→qkv deploy from the tier.
  - `mlp_gate_up_split.dynM` — the CUT consumer at symbolic M (prefill is symbolic; today the split only has
    `.m256`). Required for Phase 4 to deploy cold at real prefill.
  - Decide per shape whether the `.m256` fused prologue (`norm_*_proj.m256`) is worth a golden or should also CUT —
    the cut is the perf answer for multi-fold; norm→q/kv are SINGLE-channel so they can fuse OR split. Measure
    fuse-vs-cut at M=256 before seeding (mirror the mlp finding).
  - Method: extend `sweep.sh` / the `emmy tune -c "<snippet>"` seeding flow (see
    `plans/gemma4-decode-goldens-seeding-findings.md`), one cold invocation with within-sweep transfer, then a -O3
    A/B (`run --bench --golden --json`, 3× reproduced), then hand-edit the YAML per the tune-golden skill
    (better→replace, same→add). fp16, torch-eager reference. → verify: `run --bench --golden <name>` deploys each
    new shape from an empty DB at the recorded µs.
- **3b. New golden KINDS for the three schema holes** (each gated on its perf work in
  `plans/computed-a-pipeline-and-sdpa-oproj.md`):
  - **Per-head qk-norm epilogue** (WS3a): the deployed `k_mean_linear_reduce` applies a per-head RMSNorm to the
    projection output that `NormLinearGoldenConfig` (prologue-only) doesn't model. Either extend the norm_linear
    snippet to include the per-head norm, or add a dedicated kind; verify its `shape_key()` matches the deployed op.
  - **Fused sdpa→o_proj** (WS2b): add an attention→projection computed-A kind ONLY after the fusion is reachable +
    well-scheduled. Until then the split o_proj consumer (WS2a) is the target — fix its staging (it drifts to a
    289 µs scalar today) and seed an accurate staged `o_proj` golden, NOT a gmem-direct one (a matched-but-slow
    golden previously regressed it 129→189).
  - **Linear→norm epilogue** (WS3b): confirm o_proj/down_proj SPLIT into covered matmul+rms_norm at decode; add a
    kind only if they fuse-and-underperform at prefill.

## Phase 4 — make the optimal FORM (cut) deploy at prefill, not just the optimal config

Even with a golden, cold serving takes the LOSING fused form at prefill: `PLACE@cone` defaults to `fuse`
(`search/space.py:216`, `_PLACE_DEFAULTS`), and the cut (`020_cut_edge`) is a **structural** option filtered cold
(`greedy_decide`: structural leaves need a trained prior to be priced, else dropped — greedy.py:576–582). The cut is
the cuBLAS-beating form for the multi-fold gate/up edge at M≥256.

- **4a. Choose cut by default for a multi-fold cone at prefill M.** Change `_PLACE_DEFAULTS["cone"]` resolution (or
  the structural pricing) so a multi-fold (gate⊗up) cone at large M resolves to `cut` without a pin, while decode
  M=32 stays fused (the `mlp_geglu.m32` golden). Cleanest: make the cut a *cost-priced* option the golden/evidence
  tier can select — i.e. seed the `mlp_gate_up_split` cut-consumer goldens (3a) so `_pick_structural` prices the
  split from measured evidence. → verify: cold compile of `pre256`/`post256` deploys the cut (N single-channel
  matmuls + combine), not the fused megakernel; `mlp` edge ≈ 270 µs (1.17× eager) not 435 µs.

  **ATTEMPTED 2026-07-19 AND REVERTED. The premise above is wrong on two counts; do not retry it as written.**

  1. **The cut is not "a structural option filtered cold".** `PLACE` is declared with no `hints=` —
     *"Pin-only (never enumerated)"* (`search/space.py`). `_pick_structural` / `prior.trustworthy` governs a
     DIFFERENT class (the demoted-matmul split). Verified by probe: no structural option is ever offered at these
     forks, so `_pick_structural` is never called. `020_cut_edge` fires only under an explicit pin.
  2. **The cut cannot be offered as a fork from `020_cut_edge`.** The two passes are coupled through the IR
     spelling: `bind_prologue_contraction` requires `Map(source=Reduction)` (the UNFUSED spelling), but when
     `010_recognize` fuses (the default) it produces `Map(source=Contraction)`. So by the time `020_cut_edge`
     runs on a fused cone, its bind returns `None` and the rule skips — it can only ever cut what `010` declined
     to fuse. Offering `[root.op, cut_graph]` there is a no-op in the default path.
  3. Worse, that offer is **non-terminating** where it does fire: choosing the keep-fused `root.op` leaves the
     graph unchanged, the rule re-matches, and the pipeline loops. Observed as a **hang** — `make test` stalled
     28 min at 86% with no output. (The rule's docstring says termination is structural because neither *split*
     half re-matches; a keep-fused option breaks that invariant.)
  4. Moving the `trustworthy` check to after the pricing (so provenance could be computed) also made structural
     pricing run cold on forks it never ran on before, and `_price_op_leaf`'s `single_node_graph` raised on
     several — 13 failures across e2e/serving. A price probe must be exception-safe end to end; only
     `_price_kernel`'s inner resolve is today.

  **What is true and worth keeping** (all measured on the 5090, cold, golden tier only):
  - With 3a's consumers seeded the cut now DEPLOYS correctly instead of a scalar tile:
    norm→q M=256 **58.0 µs vs fused 119.4 (2.06×)**; gate⊗up M=256 **369.2 vs fused 898.9 (2.43×)**.
  - **The cut is only viable where its consumers carry goldens.** Unpinned-M crossover for the gate⊗up edge
    (`EMMY_PLACE` pinned, cold): M=32 fuse 172.9 / cut **35,558**; M=64 281.2 / **71,160**; M=128 652.7 /
    **142,702**; M=256 905.1 / **370.5**. Only m256 has a `mlp_gate_up_split` golden. So a shape-threshold
    default (`cut` above some M) is NOT safe — it would put M=64/128 on a 71–143 ms path. The choice must be
    evidence-driven, never geometric.
  - Prize if solved: ~620 µs/layer × 48 layers ≈ **~30 ms per 256-token prefill chunk**.

  **What the real fix requires** (larger than this plan budgeted): the fuse/cut choice has to be offered where it
  is actually made — at `010_recognize`, as a genuine two-option fork carrying the fused schedule beside the cut
  `Graph`. That means extracting `020_cut_edge`'s graph construction into a helper `010` can call, so the
  structural option exists at the point the prologue would be bound. Then, and only then, the evidence-gated
  pricing below is what selects it. Budget this as its own change with its own validation pass.
- **4b. Fuse the GeGLU `gelu·mul` into the up-matmul epilogue** to drop the standalone combine kernel (perf tail
  from `020_cut_edge`'s combine). Bounded; measure the combine-kernel bandwidth saved.

  **ATTEMPTED 2026-07-19 AND REVERTED — the epilogue costs the mma tier.** Implemented as: channels `0..N-2` keep
  their F32 workspaces, the LAST channel's projection reloads them and runs `pro_map.body` straight into `out`
  (no combine node). Structurally correct — the combine kernel disappears and the last channel writes the real
  output — but the edge went **365 → 5714 µs (15× WORSE)**. The fused-epilogue channel drops off the warp tier
  onto a scalar loop (`k_mul` 5531.9 µs, no `TILE`, regs=40, grid=15360; only ONE `__partial` survives, ch0's).
  Adding gmem `Load`s of the sibling workspaces to the projection body defeats the mma tiering, so the channel
  can no longer ride `d2/tma/ring` — the exact property the cut exists to buy.

  **The prize did not justify it anyway, and this is the useful measurement.** Per-kernel breakdown of the
  deployed 365 µs cut at M=256 (5090, cold):

  | kernel | µs | share |
  | --- | ---: | ---: |
  | the two channel `__partial`s (the matmuls) | 168.4 + 170.1 | **93.1%** |
  | channel split-K finalizes | 8.0 + 8.1 | 4.4% |
  | `k_mul` — the GeGLU combine (4b's whole target) | 5.6 | **1.5%** |
  | `__stat` + `__cone` producers | 1.6 + 1.5 | 0.8% |

  The combine is cheap because the F32 channel workspaces (15.7 MB each) sit inside the 5090's ~96 MB L2, so the
  round-trip 4b would delete is already cache-resident rather than DRAM traffic. Making it work would need the mma
  epilogue to support gmem loads at the output stage — real codegen work for ≤1.5%. **Not worth it; closed.**

  **The real lever is in the 93%.** Those channel partials deploy `mlp_gate_up_split.m256` **std** at ~169 µs each;
  the **fm** golden for the identical shape records **125.6 vs 187.3 (1.49×)**. Routing this edge through the
  fast-math lane would take the two channels ~338 → ~226 µs — **~112 µs, ~31% of the edge**, versus 1.5% for 4b.
  It is gated behind `EMMY_FAST_MATH` because it is an f16-accumulate precision trade, so it needs an explicit
  decision on the serving numerics contract, not a compiler change.

## Phase 5 — seed the box + verify serving deploys optimal, end to end

**EXECUTED 2026-07-19 (5b partial + 5c). Headline: the session's compiler/golden work is worth ~12–15% on served
gemma-4, and emmy still trails stock ~2.4× on the mixed workload.**

Environment note that cost a detour: the serving venv was missing `ninja` (vLLM's C++ extension builder) and it is
not on the child PATH — `pip install ninja` into `venv-serving` AND `PATH="$PWD/venv-serving/bin:$PATH"`. Stock also
needs `--language-model-only` (gemma-4 is an MM checkpoint; the encoder budget check rejects `mnbt=256` otherwise).
Cold emmy boots exceed vLLM's default 600 s engine-ready timeout — set `VLLM_ENGINE_READY_TIMEOUT_S=2400`.

**5c — e2e A/B, all four arms measured in ONE session** (in-256 / out-64, concurrency 32, 64 prompts, fp16,
`--max-model-len 512 --max-num-batched-tokens 256 --gpu-memory-utilization 0.90`, empty online prior):

| arm | req/s | out tok/s | TTFT mean / median (ms) | TPOT mean / median (ms) |
| --- | --: | --: | --: | --: |
| stock vLLM (`--language-model-only`) | **14.27** | **913** | **464 / 218** | **25.9 / 28.0** |
| emmy @ `7d34e841` (pre-session baseline), twins.db | 5.35 | 343 | 2388 / 3292 | 45.6 / 47.5 |
| emmy @ `c42fffa5` (this session), twins.db | **5.98** | **383** | **2203 / 2838** | **39.2 / 40.6** |
| emmy @ `c42fffa5`, goldens ONLY (no tune DB) | 4.96 | 318 | 2768 / 3562 | 47.5 / 49.4 |

- **The session's work is a real gain: +11.8% req/s, −13.8% median TTFT, −14.5% median TPOT** against the SAME-session
  pre-session baseline at identical config. That is the only valid attribution.
- **Do not compare across sessions.** The earlier report's emmy row (7.11 req/s / 2332 ms / 32.0 ms) does not
  reproduce — but neither does it reproduce at the PRE-session commit (5.35), so the gap is the box, not a
  regression. Stock reproducing its old numbers exactly (14.27 vs 14.25) is misleading here: stock runs native
  kernels with no emmy compile, so it is far less sensitive to whatever differs. Every emmy-vs-emmy claim must be
  same-session.
- **The twin DB is still worth ~20%**: goldens-only 4.96 vs twins.db 5.98 req/s. So Phase 5b's portability goal is
  *functional* but not yet *free* — a fresh clone serves correctly and 17% slower than a locally-tuned box.
- **stock still leads 2.4× on req/s and 13× on median TTFT.** The prefill wall is narrowed, not broken.

**5b — cold-serve verification: PASSES functionally.** With `EMMY_TUNE_DB` at a nonexistent path and an empty
online prior — repo goldens as the ONLY evidence — the 12B serves correctly (64/64 requests, 0 failures). Boot is
~18 min and is entirely nvcc codegen (GPU idle at 0%, `cicc`/`nvcc` resident throughout), i.e. **compile-bound, not
resolution-bound — exactly Phase 1's success criterion**. Not done: the per-op deployed-kernel dump against the
golden tier (the twin audit covers this statically at 47 MATCH / 0 DRIFT).

**Not executed: 5a** (twin re-tune — the box's `twins.db` predates this session's forms, so the 5.98 row understates
what a re-tuned box would give) **and the in-8 decode arm.**

### The prefill profile — and the one lever that is NOT reachable yet

Per-kernel, cold, goldens-only, per layer (`pre256` 491 µs + `post256` 620 µs = **1111 µs → ~53 ms per 256-token
chunk over 48 layers**):

| half | kernel | µs | % of half |
| --- | --- | --: | --: |
| `pre256` | fused norm→q (`w1x1/f4x8`, **255 regs, 17% occ** — spilling) | 174.2 | 35.5% |
| `pre256` | fused norm→k, norm→v (`w2x2/f2x4 d1/sync`) | 147.2 + 147.0 | 59.9% |
| `post256` | geglu cut's two channel matmuls | 184.9 + 184.8 | 59.6% |
| `post256` | `mlp_down.m256`, `o_proj.m256` (both on their goldens) | 161.2 + 46.4 | 33.5% |

`post256` is healthy — every big kernel is on a seeded golden. **`pre256` is not: 95% of it is the three fused
norm→qkv cones**, and the q cone cold-picks a register-spilling `w1x1` tile. Standalone the CUT beats fused
**2.19×** (54.4 vs 119.4) on norm→q and **1.77×** (36.6 vs 64.7) on norm→kv — worth ~350 µs/layer, ~17 ms/chunk,
**~31% of prefill**.

**It is not reachable by seeding, and attempting it regressed the half 5.3× (491 → 2595 µs).** These cones fork
**PRE-split**: the cone is not recognized yet, so `PLACE@cone` appears on NONE of their ~13k candidate rows, while
`_fork_shape_key` rebuilds their key to `kind="fused"` anyway (the #398 classifier). A `PLACE@cone: cut` golden
matched them as "free" and deployed a bare map-form row on the scalar tier at 1244 µs while reporting 54.4.
Fixed defensively — an absent PLACE family is now a refusal, not a free match, so this is a loud drift warning
instead of a silent misdeploy — but the lever itself stays closed.

**To open it — and the first diagnosis of this was WRONG, so record the corrected one.** The blocker is NOT the
enumeration. Decisive experiment (run it FIRST next time, before touching any knob): pin `EMMY_PLACE=cut` on the
in-model `pre256` twin. Result — **490.6 µs, byte-identical kernels to unpinned** (`d1/sync`, `w1x1/f4x8`, no
`__cone`/`__stat`). The pin cannot help because **the cut is not CONSTRUCTIBLE for these ops**:
`bind_prologue_contraction` rejects them, so `020_cut_edge` has nothing to realize. Offering `PLACE@cone` at that
fork would therefore just re-create the 5.3× misdeploy the refusal now guards against.

Localized by instrumenting every `return None` in the binder — the in-model cones are rejected at **two** guards:

| guard (offset within `bind_prologue_contraction`) | source | hits |
| --- | --- | --: |
| `not isinstance(op, Map) or not isinstance(op.source, Reduction)` | header | 6 |
| `len(inner) < 2 or not inner[0].is_reduce or not isinstance(inner[-1], Write)` | column-loop body shape | 4 |

The second is the interesting one: those ops ARE `Map(source=Reduction)` with a valid statistic header, but their
column loop's body is not the expected `[reduce Loop, pointwise…, Write]`. The standalone snippet
(`F.rms_norm(x,(3840,),nw) @ w`) binds fine and cuts to 54.4 µs; the in-model op differs because `mul_1` feeds
THREE linears whose outputs then reshape into the per-head q/k-norms, so fusion leaves extra structure in the body.

### The dtype root cause — FIXED in the tracer, and what it invalidates

The f32 A operand was not a recognizer or enumeration problem at all. `trace/torch.py` aliased `to` / `type_as`
unconditionally, so Gemma's closing `.type_as(x)` was dropped and the norm OUTPUT stayed f32 (the f32 itself arrives
via the traced scalar constants — `f16 ** f32 -> f32` — not via `.float()`). Fixed: a dtype-CHANGING cast now emits a
real identity `IndexMapOp` carrying the target dtype; a same-dtype `to` stays an alias. Verified on a fresh
gemma-4-12B trace (`type_as_cast = copy(mul_1) -> f16`), `make test` 2535.

The statistic deliberately stays f32. Demoting the constants so the chain never widens — the tempting
"simplification" — is a CORRECTNESS bug: squaring gemma activations in f16 overflows above |x|=256 (measured
max|err| vs HF **60.7 at peak 300**, 61.8 at peak 1000; a corrupted mean destroys the row's normalization). HF
upcasts for exactly this reason.

**The fix does NOT by itself move serving, and it invalidates the box's tuning state.** Two measured consequences:
- Loop fusion RE-INLINES the now-f16 cone into the projections, so they stay computed-A. The cone is still not
  bindable (`PLACE@cone` absent on all ~13k rows of a fresh trace), so the cut remains unreachable.
- Changing the graphs invalidates BOTH the cubin cache and `twins.db` (its evidence is keyed to the pre-fix
  structural signatures). A serving A/B run immediately after the fix measured **0.04 req/s, median TTFT 112 s** —
  entirely compile-on-demand, and a second run still had not converged. **Benchmarking in this state measures "stale
  evidence applied to new graphs", which is strictly worse than either coherent state; the number is meaningless.**

**Correct sequence, and the reason Phase 5a is now a prerequisite rather than an option:** tracer fix → RE-CAPTURE the
twins and RE-TUNE `twins.db` against the new graphs (~70 min per the earlier report) → only then re-run the A/B. Note
the twins have no documented capture mechanism — they were produced ad hoc in an earlier session — so re-capture needs
a harness against `EmmyGenRunner` written first.

**Proxy warning, learned the hard way this session.** Three different proxies for "the serving graph" each disagreed
with it and each invalidated a conclusion: the *snippet* (`F.rms_norm(x) @ w`) fuses the cone where the model does
not; the `--layer` trace keeps a leading batch dim `(1, 256, 3840)` where serving is flattened `(256, 3840)`, which
makes the seeded m256 goldens spuriously DRIFT; and the captured *twins* carry the pre-fix f32 dtype. Verify which
form a claim is about before acting on it.

**So the real prerequisite is a RECOGNIZER change** — extend `bind_prologue_contraction`'s structural matcher in
`_atomize.py` to accept the in-model cone shape. Only once the binder accepts it does the Phase 4a enumeration +
a `PLACE@cone: cut` golden become meaningful. Budget it as matcher work with its own accuracy validation, not as
seeding or enumeration. It remains the largest measured, precision-free win left (~350 µs/layer, ~31% of prefill),
ahead of the fast-math lane (~31% of the mlp edge alone, but a numerics trade).

- **5a. Twin tune (box-local seed, near-term).** Run `_tune/decode-twin-readiness/tune-sym.sh` (and the static
  pre32/post32) to rebuild `twins.db` + `twins-online.json` **after** Phases 1–4 land, so the box serves the
  improved forms. This is the evidence path that actually carries serving today.
- **5b. Cold-serve verification (the portability goal).** With `EMMY_TUNE_DB`/`EMMY_ONLINE_FILE` pointed at EMPTY
  paths (golden tier only), serve the 12B and py-spy / dump the deployed kernel per in-model op. → verify: every
  fused fork deploys its golden/cut config cold (no `mean_scores` fallthrough on the decode-critical forms); the
  cold-serve decode TPOT matches the twin-tuned TPOT within noise. This is the proof that goldens now carry serving.
- **5c. E2e A/B vs stock** (`plans/golden-sweep-gemma4-rtx5090-findings.md` harness): decode (in-8) and mixed
  (in-256), emmy (goldens + boot-fixed, twins optional) vs stock vLLM `--language-model-only`. Report TPOT / TTFT /
  req/s honestly. **Expected**: decode reaches emmy's ~22.7 ms floor (still behind stock's 18.1 — structural);
  mixed improves materially from the cut-at-prefill (4a) closing the 2332 ms TTFT / 7.11 req/s prefill wall.
- **5d. Regression gates**: `make test` (esp. `test_golden_evidence_deploy`, `test_shape_key_kinds`,
  `test_golden_configs`); `make lint`; a full twin compile with zero new DRIFT warnings.

## Order of work — ORIGINAL plan, superseded by the STATUS block at the top

Kept for the record of what was intended and where it was wrong. The live ordering is
**Remaining steps** in the STATUS block; read that, not this list.

1. **Phase 1** (boot pathologies) — bounded, repo-shipped, unblocks fast verification. Do first.
2. **Phase 2** (audit + extend `_fork_shape_key`) — cheap, decides which gaps are Phase 3 vs a rebuild.
3. **Phase 3a** (seed dynM prologue + cut-consumer twins) — unblocks prefill/dynamic cold serving.
4. **Phase 4** (cut-by-default at prefill) — the one selection-side perf lever; gated on 3a's cut-consumer goldens.
5. **Phase 3b** (new golden kinds) — gated on the WS2/WS3 perf work; do not seed losing goldens.
6. **Phase 5** (twin re-seed + cold-serve verify + e2e A/B) — throughout, and as the final acceptance.

Where it was wrong, and worth remembering when writing the next plan of this shape:

- **Step 2 assumed `_fork_shape_key` would need extending.** It did not — every fused fork was already correctly
  kinded. The audit's real value was the two things it was not looking for: a silent wrong deploy, and the
  discovery that Phase 3b was unnecessary.
- **Step 4 mis-stated the mechanism** ("the cut is a structural option filtered cold"). `PLACE` was pin-only and
  never enumerated; `_pick_structural` governs a different class entirely. Two designs were built on the wrong
  premise before the right one (decide-in-the-schedule, realize-in-the-graph — split-K's existing pattern).
- **Step 5 treated the twin re-tune as a closing formality.** It is a prerequisite, and it is blocked on a capture
  harness that does not exist.
- **The plan never questioned the traced dtypes.** The single largest measured item in the prefill profile — 468 µs
  a layer — traced back to a dropped `type_as` in the tracer, four layers below where the plan was looking.

## Explicitly out of scope

- **Beating stock on decode.** The fused computed-A decode kernel is at its floor; a stock-beating decode kernel is
  a CUTLASS-class warp-specialized rewrite (`plans/computed-a-pipeline-and-sdpa-oproj.md`), not a selection or
  golden problem. This plan reaches emmy's floor, not below it.
- **hd512 flash cold-reachability.** Serving routes the 8 global (hd512) layers through vLLM paged-attention; emmy
  owns only the projections/MLP/norms there. hd512 is a `compile`/bench gap, not a serving lever.
- **Non-gemma models and the 4090.** Scope is gemma-4-12B on the 5090; the 4090 (which lacks the whole fused/decode
  golden tier) follows once the 5090 forms land.
- **The fast-math (f16-accumulate) precision lever** on the computed-A forms — orthogonal to staging/selection.
