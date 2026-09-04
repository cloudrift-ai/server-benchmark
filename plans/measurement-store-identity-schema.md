# Measurement store on unpacked identities: remove the in-memory evidence mirror

Working note, 2026-09-03. Follows PR #704 (golden replay as the one measured-evidence pick). Not tracked.

## Goal

The tune DB becomes the only measured store. A compile decides each fork by one query on the identity of the
kernel it is offered on. A golden reaches a deploy by being written into that DB as ordinary measurement rows
before the compile starts. Nothing measured lives in memory beside the DB: no per-compile evidence index, no
golden row import into that index, no reservoir tier.

Every identity the store keys on is a value object in ONE module, its fields are the table's columns, and its
digest is derived from those fields. Adding an element to an identity is adding a field: the column follows and a
migration recomputes the digest from the columns beside it.

## Why

- The pick matches on a proxy. Rows are found by their `S_*` feature set, with a subset rule that exists only
  because features drift. Different kernels never share a signature, so the signature is a stand-in for identity
  and the identity is what should be stored and queried.
- The DB holds three notions of kernel identity beside the one the deploy matches on: `perf.op_key` (the
  identity lattice with io and knobs, and for a rendered kernel the CUDA-source digest), `node.op_sig` (the
  identity pass's feature-signature digest) and the Loop-IR body identity goldens carry. Keys are assembled by
  hand in about ten places (`Context.structural_key`, `Op.identity_key`, `identity.op_sig`, `fork_signature`,
  `canonical_row_key`, `golden.kernel_identity`, `golden._set_key`, `scope_token`, two `node_key`s).
- The evidence index is a full scan of the perf table per process, memoized on mtime and golden scope, because the
  lookup key sits inside a JSON blob. With an indexed identity column the per-fork query is a point lookup.
- `perf` has no card column: a measurement on one sm_89 card answers a deploy on another.
- Goldens are imported into the live regime through `regime_live` and scoped through `records_override` /
  `EMMY_GOLDEN_FILE` / `sole_evidence`, a second isolation mechanism beside "which DB file".

## The identity module

One module (working name `emmy/compiler/pipeline/search/identity.py`; the DB module imports it, nothing in it
imports the DB) defines frozen dataclasses. `Structural.structural_key()` already digests a frozen dataclass's
fields, so the identity, the row and the digest are one object; no hand-assembled key survives.

```python
@dataclass(frozen=True)
class Regime(Structural):  # where a measurement holds
    gpu: str  # the card; µs is per card, capability alone is not enough
    sm_major: int
    sm_minor: int
    opt_level: int  # split nvcc opt level, never the raw flag string
    pins: Pins  # the input-pin regime (FAST_MATH and the precision gates); absent = OFF


@dataclass(frozen=True)
class OpIdentity(Structural):  # what a kernel IS, knob-free
    dialect: str  # the stage the identity is taken at (loop | tile)
    body: str  # canonical Loop-IR program, wire form
    io: str  # operand dtypes / shapes / strides, canonical JSON


@dataclass(frozen=True)
class Decision:  # the arm: canonical row of tunable knobs, open key set
    knobs: Mapping[str, str]  # canonical JSON in the table; '{}' for a forkless kernel
```

Every producer and consumer of a key imports these: the perf writer, the pick, the golden importer, the identity
pass, the corpus helpers, `eval`. `Op.identity_key` returns `OpIdentity(...).structural_key()` and its CUDA-source
override goes: a measurement keys on the Loop-IR body identity or the golden's identity and the DB's never meet.

## Schema

```sql
CREATE TABLE context (            -- Regime, unpacked; digest derived from the columns
    digest     TEXT PRIMARY KEY,
    gpu        TEXT NOT NULL,
    sm_major   INTEGER NOT NULL,
    sm_minor   INTEGER NOT NULL,
    opt_level  INTEGER NOT NULL,
    pins       TEXT NOT NULL
);

CREATE TABLE op (                 -- OpIdentity, unpacked; digest derived from the columns
    digest     TEXT PRIMARY KEY,
    dialect    TEXT NOT NULL,
    body       TEXT NOT NULL,
    io         TEXT NOT NULL
);

CREATE TABLE measurement (        -- one row per (regime, kernel, decision); the PK is the keep-best key
    context     TEXT NOT NULL REFERENCES context (digest),
    op          TEXT NOT NULL REFERENCES op (digest),
    decision    TEXT NOT NULL,    -- canonical JSON
    status      TEXT NOT NULL,    -- ok | failed | timeout
    us_median   REAL NOT NULL,
    us_min      REAL,
    us_max      REAL,
    n_samples   INTEGER NOT NULL,
    source      TEXT NOT NULL,    -- tune | bench | golden:<file digest>
    measured_at TEXT NOT NULL,
    error       TEXT,             -- a failure's message is the one thing not reconstructible
    PRIMARY KEY (context, op, decision)
);
CREATE INDEX measurement_fork ON measurement (context, op);
```

Not stored, reconstructed on demand: `S_*` features (featurize `op.body`), `H_*` values (from the regime), kernel
source / grid / block / smem / pretty text (re-lower the body under the decision in the regime; the cubin cache
already keys on that), parent-to-piece relations (replay the parent's op under its route decision; pieces are
identities of their own). Dropped tables: `loop_op`, `tile_op`, `kernel_op`, `cuda_op` (collapse into `op`),
`lowering`, `perf` (becomes `measurement`).

Open question, decide before step 5: the `node` table (search-tree samples with a features blob; the offline
fitter, `eval prior --dataset nodes` and the freeze read it). A leaf sample is a measurement row; a partial node's
value is an aggregate over the leaves beneath it. Dropping `node` from the durable store means the fitter
featurizes on load and a freeze is a snapshot of `measurement` joined with `op`.

## Steps

Each step lands green on its own, keeps the realization corpus under strict evidence as its acceptance test, and
should not grow the line count under `emmy/`.

1. **Identity module.** Add `Regime`, `OpIdentity`, `Decision`. Route `Context.structural_key`, `Op.identity_key`
   (drop the CUDA-source override), `canonical_row_key` and `golden.kernel_identity` through them. Fold the
   input pins into `Regime`. Verify: kernel-source digest battery (`scripts/digest_kernels.py`) unchanged; the
   corpus's `offered` / `realized` unchanged; the fork-time identity of a kernel equals the persisted identity
   the golden replay keys its rows by (`_replay.signatures` today) — this is the join the design rests on.
2. **Schema.** Create `context`, `op`, `measurement`; the perf writer (`persist_kernel_perf`, the one writer)
   writes them. One-time migration of an existing DB: `perf` rows whose `op_key` can be re-derived from the
   `cuda_op` inventory migrate, the rest are dropped with a count in the log. Verify: `emmy eval variants` and
   `eval knobs` read the new tables and agree with the old on a migrated DB.
3. **Per-fork query.** `greedy_decide` asks `db.measurements(regime, identity)` at each fork and runs the
   existing arm logic over that short list: schedule rows through `Fork.admits` / `leaf_for`, route rows through
   `pins.spelled_arm`, failed rows as disqualification. Delete `_Measured`, `_db_measured_index*`,
   `_DB_INDEX_CACHE`, `_sig_groups`, the subset branch of `Prior.sig_groups`, `fork_signature` as a lookup key.
   Verify: the cold-pick tests, the two serving stitch tests, a serve boot's compile count and wall time.
4. **Golden import is a DB write.** `golden.import_file(db, path)`: each entry with an identity is one
   `measurement` row (its knobs are the decision, its µs the measurement, `source = golden:<digest>`); an
   identity-less single-kernel entry derives its identity from its program once; re-import deletes by source
   first, and an `imports (source, compiler_fingerprint)` row makes it idempotent. `compile`, `run`, `tune` and
   the `serve` parent import the repository's per-card files, or the one `--golden PATH` names, before
   compiling. Decision for the user: import the YAML µs as-is (keeps today's whole-target pricing against
   per-kernel rows) or bench on import (`run --golden --bench` IS the import; a cold machine benches once before
   it serves). Recommendation: bench on import. Delete `evidence_rows`, `regime_live`, the replay's persisted
   "replays" section, `records_override`, `RECORDS_OVERRIDE`, `scope_token`, `scope_explicit`, `sole_evidence`,
   `EMMY_GOLDEN_FILE` / `config.golden_scope` / `golden_file_override`. The replay (`golden._replay`) stays only
   where identities are minted at record time (`helpers.complete`, the strict decode).
5. **Isolation is a DB file.** The realization corpus, the release gate (`eval golden --serving-config`) and
   `--golden PATH` compile against a scratch DB named through `EMMY_TUNE_DB`, import into it, and turn strict
   evidence on. `make test` sets `EMMY_TUNE_DB` to an empty scratch path instead of `EMMY_GOLDEN_FILE=`.
6. **Retire the reservoir as an evidence tier.** The tuner already writes every measurement as rows, so the
   reservoir is a subset of the DB. Keep it as the online prior's training sample only; delete `evidence_pick`
   and the reservoir-first branch of the hierarchy. Optional for the first five steps; it is the last in-memory
   mirror.

## What stays

`pins.spelled_arm`, `Fork.admits` / `leaf_for`, strict evidence and `EvidenceError`, the splice events carrying arm
knobs, the corpus's one-entry-per-kernel format, `run --bench` writing rows through the one writer.

## Risks

- **Identity at fork time vs persisted identity.** Step 1's verification. If they differ for cut pieces, the
  importer needs the replay after all and the design degrades to today's, keyed by identity instead of features.
- **Whole-target µs.** A golden's µs covers every kernel of its target. Written per kernel it over-prices the
  golden's arm against a per-kernel DB row of the same identity. Bench-on-import removes this; as-is import keeps
  it, exactly as today.
- **Concurrent writers.** Imports run in the command process before the compile; xdist workers and the corpus
  use their own scratch DB. SQLite WAL covers the occasional overlap on one machine.
- **Freeze and fitter.** Dropping `node` changes what a reported prior number is evaluated against. Decide before
  step 5; the first four steps do not touch `node`.
