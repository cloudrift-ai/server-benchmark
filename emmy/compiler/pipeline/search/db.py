"""SQLite-backed measurement store for the search package.

Pure persistence layer — no MCTS state, no propagation walks. Tables:

- ``context`` / ``op`` — the identities a measurement is keyed on, UNPACKED: one column per
  field of :class:`~emmy.compiler.identity.Regime` and :class:`~emmy.compiler.identity.OpIdentity`,
  with the row's ``digest`` derived from those columns. Adding an element to an identity is
  adding a field to that module: the column follows and no key is assembled by hand anywhere.
- ``measurement`` — one row per ``(regime, kernel, decision)``, which is also the keep-best key.
  The kernel is its knob-free identity and the decision is the canonical row of tunable knobs, so
  a compile decides a fork with ONE indexed query on the identity it is offered on
  (:meth:`SearchDB.measurements`) instead of scanning the table and matching on a feature proxy.
  What used to sit beside the measurement is reconstructed on demand: the ``S_*`` features
  (featurize ``op.body``), the ``H_*`` values (from the regime), the kernel source and launch
  geometry (re-lower the body under the decision — what the cubin cache already keys on).
- ``node`` — one row per search-tree node (every partial branch + leaf of a
  per-kernel autotune search), keyed by ``digest(context_key, op_sig,
  tunable-knob set)``. Each row carries the full feature dict passed to the
  prior (``H_*`` + ``S_*`` + knobs), a value-of-position latency (``1/best_reward``
  — the best latency reachable below the node; keep-min across sessions for
  branches, newest-measurement for leaves), and
  a ``parent_key`` pointer so ancestry between rows is recoverable. Content-keyed
  (parent-tree-independent), so it survives schema-version drops.
  Written once per finished search by :meth:`SearchDB.record_nodes`, fed by the
  post-order tree walk ``TuningSearch._collect_node_records`` — alongside (not
  replacing) the online prior's reservoir feed. Label-quality columns (additive
  migration; old rows degrade to unknowns): ``visits`` (benched-descendant count,
  SUM-accumulated — the label's confidence weight), ``is_leaf`` (directly-benched
  terminal vs branch), ``variance`` / ``n_samples`` (the leaf's own bench stats),
  ``status`` (``ok`` / ``bench_fail`` — fail leaves ARE recorded, with the bench
  watchdog's sentinel latency as ``value_us``; an ``ok`` row is never downgraded
  by a later fail), and ``run_id`` / ``measured_at`` (the tune session + time
  that produced the current ``value_us`` — replaced only on improvement). The
  Every write passes the physical-plausibility gate (:func:`implausible_value_reason`):
  an ``ok`` row whose latency implies throughput above its card's recorded peak is a
  mismeasurement, dropped with a warning — never stored (``purge_implausible`` is the
  one-time repair for stores written before the gate existed).

Concurrency: opened in WAL mode so parallel benches can read while one
writes. The connection is kept open for the DB's lifetime; callers can
share one ``SearchDB`` instance across threads (sqlite3 handles
locking).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from emmy.compiler.identity import OpIdentity, Regime
from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerfStats:
    """Summary statistics over per-iter kernel latencies (microseconds)."""

    median: float
    min: float
    max: float
    mean: float
    variance: float
    n_samples: int


@dataclass(frozen=True)
class Measurement:
    """One ``measurement`` row: what a kernel's arm cost, where it was measured.

    ``op`` is the knob-free kernel identity (:attr:`OpIdentity.digest`) and ``knobs`` the arm it
    decided — the canonical tunable row, which together with the regime is this row's key.
    ``captured``: the measurement ran under CUDA graph capture (pure GPU time); False = wall
    semantics including per-launch dispatch. Both kinds stay usable (replay, prior training); on
    write, a captured measurement supersedes an uncaptured one for the same key — see
    :meth:`SearchDB.record_measurement`."""

    op: str
    knobs: dict
    status: str
    us_median: float
    us_min: float
    us_max: float
    n_samples: int
    measured_at: str
    captured: bool = False
    error: str | None = None  # a failure's message — the one thing beside the µs that is not reconstructible


@dataclass(frozen=True)
class NodeRow:
    """One ``node`` row — a single node of a per-kernel autotune search tree.

    ``node_key`` is ``digest(context_key, gpu, op_sig, tunable-knob set)`` — the
    node's identity within its operation *on its hardware*; ``parent_key`` is the
    parent node's ``node_key`` (``None`` at the operation's top forks). ``gpu`` is the
    card's identity (``Context.hardware_id`` — the PCIe product name, or a device
    digest when unknown): folded into the key so same-die SKUs (H100 vs H200) never
    collide, and kept as a column so the dataset groups/filters by hardware.
    ``features`` is the full feature dict the prior sees (``H_*`` regime + ``S_*``
    structure + tunable knobs). ``value_us`` is the value-of-position latency (best
    reachable below the node); on re-encounter :meth:`SearchDB.record_nodes` keeps
    the minimum for branches (a coverage bound) but takes the NEWEST measurement for
    leaves (a re-measurement — keep-min would drift to the noise floor). ``depth``
    is the node's distance from the sentinel root (top forks = 1).

    Label-quality columns (all default to the pre-enrichment unknowns so old rows
    and positional constructions keep working): ``visits`` is the benched-descendant
    count — the confidence weight for the value-of-position label (distinct from the
    table's ``n_updates``, which counts write batches, and SUM-accumulated across
    writes/merges); ``is_leaf`` marks a directly-benched terminal (its ``value_us``
    is a real measurement) vs a branch (a min over explored descendants) — ``None``
    on pre-enrichment rows; ``variance`` / ``n_samples`` are the leaf's own direct
    bench stats (``None`` on branches; a leaf whose subtree found a faster descendant
    keeps ``value_us`` = min-over-subtree while these describe its own bench);
    ``status`` is ``'ok'`` or ``'bench_fail'`` — a fail row's ``value_us`` is the
    bench watchdog's sentinel latency, NOT a measurement; ``run_id`` /
    ``measured_at`` identify the tune session + time that produced the CURRENT
    ``value_us`` (they replace only when the value improves — ``updated_at`` is the
    one that refreshes on every write)."""

    node_key: str
    parent_key: str | None
    context_key: str
    op_sig: str
    features: dict
    value_us: float
    depth: int
    gpu: str = ""
    visits: int = 0  # benched-descendant count (0 = unknown / pre-enrichment)
    is_leaf: bool | None = None  # True = directly-benched terminal; None = unknown (old rows)
    variance: float | None = None  # leaf measurement stats (None on branches / old rows)
    n_samples: int | None = None
    status: str = "ok"  # 'ok' | 'bench_fail'
    run_id: str = ""  # tune-session id ('' = unknown / old rows)
    measured_at: str | None = None  # when the CURRENT value_us was measured (None -> record_nodes stamps now)
    # Featurizer-vocabulary version the ``features`` dict is spelled in. Defaults to the CURRENT
    # version (writers construct rows with live code); rows read back from a pre-stamp DB carry 1
    # and are excluded from prior evaluation (cross-vocabulary features score as garbage).
    feat_ver: int = FEATURIZER_VERSION


def implausible_value_reason(row: NodeRow) -> str | None:
    """THE physical-plausibility predicate for a node row's ``value_us`` — the reason it
    cannot be a real measurement, or ``None`` when it's plausible/ungateable. Shared by
    :meth:`SearchDB.record_nodes`'s write-time gate and the one-time
    :meth:`SearchDB.purge_implausible` repair (``scripts/purge_node_store.py``).

    The bound is the arithmetic-intensity floor the golden A/B integrity gate uses: the
    throughput a latency implies from the row's stamped shape must stay below the card's
    recorded peak. ``2·free·reduce_max`` FLOPs is the true work ONLY when the reduce axes
    are **disjoint** from the output — the iteration space is then exactly
    ``free_prod × reduce`` (a contraction, or a pure output-shrinking reduce) — which the
    stamps certify as ``S_loop_depth == n_free + n_reduce + n_symbolic`` (every loop of
    the nest is either a counted free/symbolic output axis or a counted reduce axis). A
    norm/softmax kernel fails that equality — its reduced axis is part of the full-size
    output, so ``free_prod`` already contains it and the product overcounts by the reduce
    extent (a cooperative norm legitimately runs ~100x its serial sibling, so a latency
    floor there would flag honest rows); a fused multi-node kernel (attention) fails it
    too. Both stay ungated rather than falsely flagged — the identity was verified
    against every stamp combination in the 2026-07 sweep stores. ``reduce_max``, not
    ``reduce_prod``, keeps the bound a lower estimate of work even off the exact case. A
    symbolic axis is excluded from the stamped products and benched at the dynamic hint,
    so it re-enters as one hint factor. Ungateable rows also pass on: non-``ok`` status
    (a fail sentinel is not a measurement), no stamped shape, unknown card or unrecorded
    peak, and rows outside the current featurizer vocabulary (their stamps aren't trusted
    enough to judge)."""
    if row.status != "ok" or row.value_us <= 0 or row.feat_ver != FEATURIZER_VERSION:
        return None
    f = row.features
    free = float(f.get("S_ext_free_prod") or 0.0)
    red = float(f.get("S_ext_reduce_max") or 0.0)
    if free <= 0 or red <= 0:
        return None
    # Work = free x red only when every loop multiplies the iteration space (disjoint axes).
    depth = float(f.get("S_loop_depth") or 0.0)
    n_sym = float(f.get("S_ext_n_symbolic_axis") or 0.0)
    n_axes = float(f.get("S_ext_n_free_axis") or 0.0) + float(f.get("S_ext_n_reduce_axis") or 0.0) + n_sym
    if depth <= 0 or depth != n_axes:
        return None
    from emmy import gpu  # noqa: PLC0415

    spec = gpu.by_name(row.gpu) if row.gpu else None
    if spec is None:
        return None
    half = any(k.startswith("S_dtype_") and "f16" in k and v for k, v in f.items())  # f16 / bf16
    peak = spec.peak_tflops("fp16" if half else "fp32")
    if not peak:
        return None
    from emmy.compiler.dim import DEFAULT_SEQ_HINT  # noqa: PLC0415

    hint = DEFAULT_SEQ_HINT if n_sym > 0 else 1
    implied = 2.0 * free * red * hint / row.value_us / 1e6  # FLOP / µs -> TFLOP/s
    if implied > peak:
        return f"implies {implied:.0f} TFLOP/s > {peak:.0f} device peak"
    return None


def impossible_kernel_reason(row: NodeRow) -> str | None:
    """The *validity* companion to :func:`implausible_value_reason` — the reason the row's
    stamped kernel could never have launched, or ``None``. A ``cp.async``-staged warp tile
    whose slab (``depth · (tile_m + tile_n) · bk_elems · elem_bytes``, the
    warp stage sizing) exceeds the card's dynamic-smem opt-in cap cannot
    materialize — pre-#330 code stamped such stages anyway, the materializer rejected the
    main kernel, and the bench recorded the surviving combine kernel's cached µs as an
    ``ok`` measurement of the whole op. On shapes too small for the latency floor to
    notice (square.512's combine implies a legal 133 TFLOP/s), THIS check is the only one
    that catches the class: the measurement is of a kernel set that provably didn't
    include the stamped kernel."""
    if row.status != "ok" or row.feat_ver != FEATURIZER_VERSION:
        return None
    f = row.features
    tile_spec = next((str(v) for k, v in f.items() if k.startswith("TILE") and v), "")
    stage_spec = next((str(v) for k, v in f.items() if k.startswith("STAGE") and v), "")
    if not tile_spec or not stage_spec.startswith("d"):
        return None
    from emmy.compiler.ir.schedule import Stage, Tile, Work  # noqa: PLC0415

    try:
        work = Work.parse(str(f.get("WORK") or ""))  # the row's unit widths live here, not in TILE
        tp, st = Tile.parse(tile_spec, work), Stage.parse(stage_spec)
    except ValueError:
        return None
    if not tp.is_warp:
        return None
    if st.transport != "smem-async":
        return None
    from emmy import gpu  # noqa: PLC0415

    spec = gpu.by_name(row.gpu) if row.gpu else None
    if spec is None:
        return None
    atom = tp.atom
    tile_m = tp.units_m * tp.reg_m * atom.atom_m
    tile_n = tp.units_n * tp.reg_n * atom.atom_n
    slab = st.depth * (tile_m + tile_n) * tp.bk * atom.atom_k * atom.operand_dtype("a").nbytes
    if slab > spec.smem_optin:
        return f"staged slab {slab} B > {spec.smem_optin} B dynamic-smem cap (kernel cannot launch)"
    return None


# The ``measurement`` SELECT column list — order must match :func:`_row_to_measurement`.
_MEASUREMENT_COLS = "op, decision, status, us_median, us_min, us_max, n_samples, measured_at, captured, error"


class SearchDB:
    """Persistent inventory of compiled ops + their measured perf.

    Pass ``path=None`` for an in-memory database (default — keeps tests
    hermetic; tuning runs pass an explicit path like
    ``~/.cache/emmy/autotune.db``).
    """

    # The store keys on identities, not on tree topology: a row survives any change that does not
    # change what a kernel IS. A bump means the identities themselves are spelled differently, and
    # the rows written under the old spelling are unreadable — not stale, unreadable — so they are
    # dropped on the next open.
    #
    # Version log:
    #   1-4: the fork-tree generations of the ``lowering``/``perf`` schema (planner-hoisted forks,
    #       explicit knob OFF sentinels, the RASTER codec, the ``S_ext_serial_cell_work`` stamp).
    #   5: the unpacked measurement store. ``perf`` (one composite ``op_key`` folding body, io AND
    #       knobs, inside a row found by its ``S_*`` feature set) becomes ``measurement`` keyed on
    #       ``(context, op, decision)`` — the identity a fork is offered on, beside the arm it
    #       decides. The op inventory (``loop_op`` / ``tile_op`` / ``kernel_op`` / ``cuda_op``) and
    #       ``lowering`` collapse into ``op``: a chain's every stage keys off the same Loop-IR
    #       content, so the parent-to-child links they existed to replay are the same row. No
    #       migration is possible in either direction — a ``perf`` row's ``op_key`` is a digest of
    #       a rendered CUDA source, and no amount of re-derivation recovers the Loop-IR body it
    #       was rendered from — so pre-5 stores are dropped whole, with a count in the log.
    _SCHEMA_VERSION = 5

    _LEGACY_TABLES = ("perf", "lowering", "loop_op", "tile_op", "kernel_op", "cuda_op")

    _SCHEMA = [
        # ``Regime``, unpacked. The digest is derived from the columns beside it, never stored
        # independently of them: a reader that wants to know what a regime WAS reads the row.
        """
        CREATE TABLE IF NOT EXISTS context (
            digest     TEXT PRIMARY KEY,
            gpu        TEXT NOT NULL,
            sm_major   INTEGER NOT NULL,
            sm_minor   INTEGER NOT NULL,
            opt_level  INTEGER NOT NULL,
            nvcc_flags TEXT NOT NULL,
            pins       TEXT NOT NULL
        )
        """,
        # ``OpIdentity``, unpacked. ``dialect`` is the stage the identity was taken at — a
        # descriptive column, deliberately NOT in the digest: every stage of one rewrite chain
        # keys off the same Loop-IR content, which is what lets a golden minted by lifting a
        # recorded target to Loop IR join the live Tile fork that offers it.
        """
        CREATE TABLE IF NOT EXISTS op (
            digest     TEXT PRIMARY KEY,
            dialect    TEXT NOT NULL,
            body       TEXT NOT NULL,
            io         TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS measurement (
            context     TEXT NOT NULL REFERENCES context (digest),
            op          TEXT NOT NULL REFERENCES op (digest),
            decision    TEXT NOT NULL,
            status      TEXT NOT NULL,
            us_median   REAL NOT NULL,
            us_min      REAL NOT NULL,
            us_max      REAL NOT NULL,
            n_samples   INTEGER NOT NULL,
            measured_at TEXT NOT NULL,
            captured    INTEGER NOT NULL DEFAULT 0,
            error       TEXT,
            PRIMARY KEY (context, op, decision)
        )
        """,
        # The per-fork query's index: one point lookup per fork, where the deploy used to scan
        # the whole table once per process and memoize the result on the file's mtime.
        "CREATE INDEX IF NOT EXISTS measurement_fork ON measurement (context, op)",
        # Content-keyed (``node_key`` folds context + op_sig + knob set), so —
        # unlike the retired ``lowering`` — it is parent-tree-independent and survives a
        # ``_SCHEMA_VERSION`` bump. ``CREATE … IF NOT EXISTS`` auto-creates it on
        # the next open of a pre-``node`` DB; no version bump / ALTER needed.
        """
        CREATE TABLE IF NOT EXISTS node (
            node_key     TEXT PRIMARY KEY,
            parent_key   TEXT,
            context_key  TEXT NOT NULL,
            op_sig       TEXT NOT NULL,
            gpu          TEXT NOT NULL DEFAULT '',
            features     TEXT NOT NULL DEFAULT '{}',
            value_us     REAL NOT NULL,
            depth        INTEGER NOT NULL,
            n_updates    INTEGER NOT NULL DEFAULT 1,
            updated_at   TEXT NOT NULL,
            visits       INTEGER NOT NULL DEFAULT 0,
            is_leaf      INTEGER,
            variance     REAL,
            n_samples    INTEGER,
            status       TEXT NOT NULL DEFAULT 'ok',
            run_id       TEXT NOT NULL DEFAULT '',
            measured_at  TEXT,
            feat_ver     INTEGER NOT NULL DEFAULT 1
        )
        """,
        "CREATE INDEX IF NOT EXISTS node_parent ON node (parent_key)",
        "CREATE INDEX IF NOT EXISTS node_op ON node (context_key, op_sig)",
        "CREATE INDEX IF NOT EXISTS node_gpu ON node (gpu)",
    ]

    def __init__(self, path: Path | str | None = None) -> None:
        # The backing file (``None`` for an in-memory DB) — read by the deploy-side
        # ``_db_measured_index`` cache to key its process-wide memo on (path, mtime).
        self._path = Path(path) if path is not None else None
        if path is None:
            self._conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        # Drop the ``lowering`` table when an older schema is detected;
        # everything else (op inventory, perf rows) is keyed off content
        # hashes and remains valid across fork-tree changes.
        cur_version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if cur_version != self._SCHEMA_VERSION:
            self._drop_legacy_tables(cur_version)
            self._conn.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
        # Additive ``node.gpu`` migration must run BEFORE the schema loop — the
        # ``node_gpu`` index in ``_SCHEMA`` references the column, so it has to exist
        # first. A brand-new DB has no ``node`` table yet (the loop's CREATE includes
        # ``gpu``), so this only fires for a pre-``gpu``-column table; old rows default
        # to '' (unknown card).
        if self._has_node_table() and not self._has_node_gpu_column():
            self._conn.execute("ALTER TABLE node ADD COLUMN gpu TEXT NOT NULL DEFAULT ''")
        for stmt in self._SCHEMA:
            self._conn.execute(stmt)
        # Additive ``node`` label-quality columns (visits / is_leaf / variance /
        # n_samples / status / run_id / measured_at). Per-column ALTERs so a
        # crash-interrupted migration self-heals on the next open; no index
        # references them, so unlike ``node.gpu`` this runs after the schema loop.
        have = {r[1] for r in self._conn.execute("PRAGMA table_info(node)")}
        for col, ddl in self._NODE_ENRICH_COLUMNS:
            if col not in have:
                self._conn.execute(f"ALTER TABLE node ADD COLUMN {col} {ddl}")  # noqa: S608 — fixed literal pairs

    # The label-quality columns added to ``node`` after the ``gpu`` generation —
    # kept as (name, DDL) pairs shared by the CREATE literal above and the additive
    # migration loop. Old rows degrade to the defaults (unknowns).
    _NODE_ENRICH_COLUMNS = (
        ("visits", "INTEGER NOT NULL DEFAULT 0"),
        ("is_leaf", "INTEGER"),
        ("variance", "REAL"),
        ("n_samples", "INTEGER"),
        ("status", "TEXT NOT NULL DEFAULT 'ok'"),
        ("run_id", "TEXT NOT NULL DEFAULT ''"),
        ("measured_at", "TEXT"),
        # Featurizer-vocabulary stamp (``features.FEATURIZER_VERSION``). Default 1 =
        # unknown/pre-stamp vocabulary: such rows are excluded from prior evaluation
        # (``data/group.group_measured``) — a cross-vocabulary row featurizes to garbage —
        # but kept in the DB and carried by ``merge_nodes`` (data, not judgement).
        # NOTE: rows written after the 2026-07 tile-IR rebuild but before this column
        # shipped are spelled in the v2 vocabulary yet default to 1 — they quarantine
        # conservatively; re-collect with the ``collect-node-data`` flow.
        ("feat_ver", "INTEGER NOT NULL DEFAULT 1"),
    )

    def _drop_legacy_tables(self, cur_version: int) -> None:
        """Drop the pre-5 measurement store. Its rows key on a digest of a rendered CUDA source
        folded with a knob dict — an alphabet the identity module cannot re-derive from anything
        the store kept — so they are unreadable rather than stale, and there is no migration to
        write. The count goes in the log so the loss is visible where it happens; a machine that
        wants its numbers back re-benches (``emmy run --golden-file FILE --bench --record``)."""
        for table in self._LEGACY_TABLES:
            if self._conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone() is None:
                continue
            rows = self._conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608 — fixed literal names
            if rows:
                logger.warning("[db] dropping %d %s row(s) from schema v%d — pre-identity keys, unreadable here", rows, table, cur_version)
            self._conn.execute(f"DROP TABLE {table}")  # noqa: S608 — fixed literal names

    def _has_node_gpu_column(self) -> bool:
        return any(r[1] == "gpu" for r in self._conn.execute("PRAGMA table_info(node)"))

    def _has_node_enrich_columns(self) -> bool:
        # ``status`` proxies the whole enrichment generation (the seven columns ship
        # as one migration; the writer-side loop self-heals partial applications).
        return any(r[1] == "status" for r in self._conn.execute("PRAGMA table_info(node)"))

    def _has_node_table(self) -> bool:
        # A read-only open of a pre-``node`` DB never ran ``CREATE TABLE IF NOT
        # EXISTS``, so ``SELECT … FROM node`` would raise; readers gate on this.
        return self._conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'node'").fetchone() is not None

    @classmethod
    def open_readonly(cls, path: Path | str) -> SearchDB:
        """Open an existing DB **read-only** — no schema creation, no version
        check, no legacy-table drop, no WAL pragma — so a read-side consumer
        (``eval``, the dataset layer) never contends with a concurrent ``tune``
        writer or mutates the file. The read methods (``measurement`` /
        ``measurements`` / ``iter_measurements``) work; any write raises (the
        connection is ``?mode=ro``). Raises ``sqlite3.OperationalError`` if the
        file is absent. A non-sqlite file fails HERE with a named reason (sqlite
        itself defers header validation to the first query, which would surface
        as a bare ``DatabaseError`` deep inside a PRAGMA) — the foreseeable case
        being a measurement freeze handed to a perf-table consumer: freezes are
        accepted only by the nodes-dataset readers (``data/freeze.py``)."""
        p = Path(path)
        if p.is_file():
            with p.open("rb") as fh:
                magic = fh.read(16)
            if magic and magic != b"SQLite format 3\x00":  # empty file = valid empty DB
                hint = (
                    " — this looks like a v1 JSONL measurement freeze; freezes are now per-GPU YAML directories, "
                    "accepted only by the nodes-dataset consumers, e.g. `eval prior --dataset nodes --db`"
                    if magic.startswith(b"{")
                    else ""
                )
                raise RuntimeError(f"{p} is not a sqlite database{hint}")
        self = cls.__new__(cls)
        self._conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
        return self

    # ------------------------------------------------------------------
    # Measurements — write
    # ------------------------------------------------------------------

    def record_measurement(
        self,
        regime: Regime,
        identity: OpIdentity,
        knobs: dict | None,
        *,
        status: str,
        stats: PerfStats,
        captured: bool = False,
        error: str | None = None,
    ) -> None:
        """Upsert one ``measurement`` row, minting its ``context`` and ``op`` rows.

        THE writer: everything measured enters the store here, so the identities are unpacked in
        one place and a reader can always answer "what regime / what kernel is this" from the
        columns. Keep-best-``ok`` policy on the ``(context, op, decision)`` key: a failure never
        overwrites a prior ``ok`` row, and among same-semantics ``ok`` rows the lowest median
        wins. ``captured`` (CUDA-graph-captured, pure GPU time) adds a precedence axis: a captured
        measurement supersedes an uncaptured (wall-semantics) one regardless of median — the
        numbers aren't comparable, and captured is the better truth — while an uncaptured
        measurement never overwrites a captured one. ``error`` is the failure text for a failed
        row (whitespace-collapsed, truncated) so failure forensics (``eval failures``) need no
        tune-log grepping."""
        context, op, decision = regime.digest, identity.digest, _decision(knobs)
        existing = self._row(context, op, decision)
        if existing is not None and existing.status == "ok":
            if status != "ok":
                return  # a failure never replaces a good measurement
            if existing.captured and not captured:
                return  # wall semantics never overwrites a captured row
            if not (captured and not existing.captured) and stats.median >= existing.us_median:
                return  # same semantics: keep the best median
        self._conn.execute(
            "INSERT OR IGNORE INTO context (digest, gpu, sm_major, sm_minor, opt_level, nvcc_flags, pins) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (context, regime.gpu, regime.sm_major, regime.sm_minor, regime.opt_level, regime.nvcc_flags, regime.pins),
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO op (digest, dialect, body, io) VALUES (?, ?, ?, ?)",
            (op, identity.dialect, identity.body, identity.io_json),
        )
        if error is not None:
            error = " ".join(str(error).split())[:300] or None
        self._conn.execute(
            "INSERT OR REPLACE INTO measurement "
            "(context, op, decision, status, us_median, us_min, us_max, n_samples, measured_at, captured, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                context,
                op,
                decision,
                status,
                stats.median,
                stats.min,
                stats.max,
                stats.n_samples,
                datetime.now(UTC).isoformat(),
                int(captured),
                error,
            ),
        )

    # ------------------------------------------------------------------
    # Search-tree nodes — write
    # ------------------------------------------------------------------

    def _drop_implausible(self, row: NodeRow) -> bool:
        reason = implausible_value_reason(row) or impossible_kernel_reason(row)
        if reason is None:
            return False
        kind = "leaf" if row.is_leaf else "branch"
        logger.warning("[node-store] dropping implausible %s row for %s on %s: %s", kind, row.op_sig[:12], row.gpu or "?", reason)
        return True

    def purge_implausible(self, *, dry_run: bool = False) -> dict[str, int]:
        """One-time repair of a store that predates the :meth:`record_nodes` gate:
        delete every row :func:`implausible_value_reason` flags, and REPAIR (not
        delete) a flagged **branch** that still has surviving ``ok`` leaf descendants
        — its value-of-position bound is recomputed as the min over them, so the
        fork-tree structure the diagnostics group on survives the purge. A flagged
        branch with no surviving ``ok`` leaf below it is deleted (matching the
        "a branch whose descendants all failed stays unrecorded" convention).
        Returns a receipt dict; ``dry_run`` computes it without writing.

        Motivation: the 2026-07-08/09 golden sweeps ran before the #330 fix, so
        split-K variants whose over-budget staged main kernel was rejected at
        materialize benched combine-kernel-only and landed as impossibly-fast ``ok``
        leaves, min-propagated up their ancestries."""
        rows = list(self.iter_nodes())
        flagged = [r for r in rows if implausible_value_reason(r) is not None or impossible_kernel_reason(r) is not None]
        flagged_keys = {r.node_key for r in flagged}
        children: dict[str | None, list[NodeRow]] = {}
        for r in rows:
            children.setdefault(r.parent_key, []).append(r)

        def surviving_leaf_min(key: str) -> float | None:
            # Flagged LEAVES are excluded from the min (they're being deleted); flagged
            # branches are still traversed — the honest leaves below them belong to
            # every ancestor's subtree regardless of the intermediate's own fate.
            best: float | None = None
            stack = [key]
            while stack:
                for child in children.get(stack.pop(), ()):
                    if child.is_leaf:
                        if child.status == "ok" and child.node_key not in flagged_keys:
                            best = child.value_us if best is None else min(best, child.value_us)
                    else:
                        stack.append(child.node_key)
            return best

        receipt = {"deleted_leaves": 0, "deleted_branches": 0, "repaired_branches": 0}
        for r in flagged:
            if r.is_leaf:
                receipt["deleted_leaves"] += 1
                if not dry_run:
                    self._conn.execute("DELETE FROM node WHERE node_key = ?", (r.node_key,))
                continue
            repaired = surviving_leaf_min(r.node_key)
            if repaired is None:
                receipt["deleted_branches"] += 1
                if not dry_run:
                    self._conn.execute("DELETE FROM node WHERE node_key = ?", (r.node_key,))
            else:
                receipt["repaired_branches"] += 1
                if not dry_run:
                    self._conn.execute("UPDATE node SET value_us = ? WHERE node_key = ?", (repaired, r.node_key))
        return receipt

    def record_nodes(self, rows: list[NodeRow]) -> None:
        """Upsert a batch of search-tree node rows (one finished per-kernel search's
        worth), with **per-kind value semantics**: a *branch*'s ``value_us`` is a
        value-of-position coverage bound (min over whatever its sessions explored
        below it), so keep-the-minimum is right — a session that found a better
        descendant genuinely tightened the bound. A *leaf*'s ``value_us`` is a
        re-measurement of ONE config, where min-of-K noisy medians drifts to the
        noise floor (selectively, on the configs revisited most) — so a leaf row is
        **newest-measurement-wins**, ordered by ``measured_at`` (ISO-8601 UTC compares
        lexicographically; a stale measurement never resurrects, which also makes
        ``merge_nodes`` direction-independent) — EXCEPT that a newer measurement of
        unambiguously lower quality (fewer ``n_samples`` AND higher ``variance``, both
        sides known) never replaces: with ``run --bench`` recording into the store by
        default (``bench_record``), a drive-by bench must not displace a tune-grade
        leaf, while comparable/unknown quality keeps plain newest-wins so honest
        re-measurement still heals stale or fake rows. Rows with unknown leaf-ness
        (pre-enrichment, ``is_leaf`` NULL) keep the conservative branch behavior.
        Consequence: a cross-session tree is no longer value-monotone — an ancestor
        branch may hold a historical bound below its leaf's current measurement.
        ``context_key`` / ``op_sig`` / ``parent_key`` are functions of ``node_key``
        and re-stamp identically. ``n_updates`` counts writes (incl. non-replacing
        re-encounters) and ``updated_at`` refreshes each time.

        Label-quality columns: ``visits`` SUM-accumulates on every write (the total
        benched descendants ever informing this node's label — a confidence weight,
        not an exact ledger); the value-paired columns (``is_leaf`` / ``variance`` /
        ``n_samples`` / ``status`` / ``run_id`` / ``measured_at``, stamped ``now``
        when the row carries none) travel WITH ``value_us`` — replaced together,
        untouched otherwise. Status follows ``record_measurement``'s keep-best-``ok``
        policy: an ``ok`` row is never downgraded by a later ``bench_fail``
        (whatever the fail's sentinel ``value_us``), while a fail row upgrades to
        ``ok`` unconditionally; fail-vs-fail follows the leaf rule (newest sentinel).

        Manual lookup-guard + INSERT/UPDATE (the ``record_measurement``
        idiom) rather than ``INSERT OR REPLACE`` — the latter would reset
        ``n_updates`` and drop the old value. Row-at-a-time autocommit like the rest
        of the file; a finished search is a few-hundred-row batch at most.

        Every incoming row first passes the physical-plausibility gate
        (:func:`implausible_value_reason`): an ``ok`` row whose ``value_us`` implies
        throughput above its card's recorded peak is a mismeasurement (a fragment's
        cost standing in for the whole op, a silently-failed launch, …), never a fast
        kernel — it is dropped with a warning instead of stored. Applying the gate
        HERE covers both writers (``_collect_node_records`` batches and
        ``merge_nodes``' cross-card imports, whose source rows may predate the gate),
        and dropping poisoned *branch* rows too (their value-of-position min trips
        the same physics) keeps a poisoned in-batch chain from landing at all."""
        rows = [r for r in rows if not self._drop_implausible(r)]
        now = datetime.now(UTC).isoformat()
        for r in rows:
            feats_json = json.dumps(r.features, sort_keys=True, default=str)
            is_leaf = None if r.is_leaf is None else int(r.is_leaf)
            measured = r.measured_at or now
            existing = self._conn.execute(
                "SELECT value_us, n_updates, visits, status, measured_at, variance, n_samples FROM node WHERE node_key = ?",
                (r.node_key,),
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO node "
                    "(node_key, parent_key, context_key, op_sig, gpu, features, value_us, depth, n_updates, updated_at, "
                    " visits, is_leaf, variance, n_samples, status, run_id, measured_at, feat_ver) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        r.node_key,
                        r.parent_key,
                        r.context_key,
                        r.op_sig,
                        r.gpu,
                        feats_json,
                        r.value_us,
                        r.depth,
                        1,
                        now,
                        r.visits,
                        is_leaf,
                        r.variance,
                        r.n_samples,
                        r.status,
                        r.run_id,
                        measured,
                        r.feat_ver,
                    ),
                )
                continue
            cur_val, n_upd, cur_visits, cur_status, cur_measured, cur_var, cur_n = existing
            visits = (cur_visits or 0) + r.visits
            if cur_status == "ok" and r.status != "ok":
                replace = False  # a failure never downgrades a good row
            elif cur_status != "ok" and r.status == "ok":
                replace = True  # a clean measurement always supersedes a fail sentinel
            elif r.is_leaf:
                # Re-measurement of one config: strictly-newer wins (NULL = pre-
                # timestamp row, always superseded); equal timestamps (same batch /
                # re-merged snapshot) don't churn the row. Quality guard: a newer
                # measurement of UNAMBIGUOUSLY lower quality — fewer samples AND
                # higher variance, both sides known — never replaces (a drive-by
                # ``run --bench`` must not displace a tune-grade leaf; comparable or
                # incomparable quality keeps plain newest-wins, so re-measurements
                # still heal stale/fake rows).
                newer = cur_measured is None or measured > cur_measured
                worse = (
                    r.n_samples is not None
                    and cur_n is not None
                    and r.n_samples < cur_n
                    and r.variance is not None
                    and cur_var is not None
                    and r.variance > cur_var
                )
                if newer and worse:
                    logger.debug("[node-store] keeping higher-quality leaf %s (incoming n=%s)", r.node_key[:12], r.n_samples)
                replace = newer and not worse
            else:
                replace = r.value_us < cur_val  # branch: keep the coverage bound
            if replace:
                self._conn.execute(
                    "UPDATE node SET value_us = ?, features = ?, parent_key = ?, n_updates = ?, updated_at = ?, "
                    "visits = ?, is_leaf = ?, variance = ?, n_samples = ?, status = ?, run_id = ?, measured_at = ?, "
                    "feat_ver = ? WHERE node_key = ?",
                    (r.value_us, feats_json, r.parent_key, n_upd + 1, now, visits, is_leaf, r.variance, r.n_samples)
                    + (r.status, r.run_id, measured, r.feat_ver, r.node_key),
                )
            else:
                self._conn.execute(
                    "UPDATE node SET n_updates = ?, updated_at = ?, visits = ? WHERE node_key = ?",
                    (n_upd + 1, now, visits, r.node_key),
                )

    # ------------------------------------------------------------------
    # Search-tree nodes — read
    # ------------------------------------------------------------------

    def iter_nodes(self, *, context_key: str | None = None, op_sig: str | None = None) -> Iterator[NodeRow]:
        """Yield one :class:`NodeRow` per stored search-tree node (the value-of-position
        dataset backing ``eval prior --dataset nodes``). Self-contained — no join.
        A read-only open of a pre-``node`` DB has no such table, so this degrades to
        yielding nothing instead of raising (mirrors ``iter_measurements``'
        missing-column degrade). Optional ``context_key`` / ``op_sig`` scope to one
        regime / operation."""
        if not self._has_node_table():
            return
        # ``gpu`` degrades to '' on a pre-``gpu``-column DB opened read-only (the
        # additive migration runs writer-side only) — same pattern as perf.error.
        # The enrichment columns degrade as one generation to their unknowns.
        gpu_col = "gpu" if self._has_node_gpu_column() else "''"
        enrich_cols = (
            "visits, is_leaf, variance, n_samples, status, run_id, measured_at"
            if self._has_node_enrich_columns()
            else "0, NULL, NULL, NULL, 'ok', '', NULL"
        )
        # ``feat_ver`` ships after the enrichment generation, so it needs its own
        # presence check (a read-only open never migrates); absent → 1 (unknown /
        # pre-stamp vocabulary).
        node_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(node)")}
        feat_ver_col = "feat_ver" if "feat_ver" in node_cols else "1"
        sql = (
            f"SELECT node_key, parent_key, context_key, op_sig, {gpu_col}, features, value_us, depth, "  # noqa: S608
            f"{enrich_cols}, {feat_ver_col} FROM node"
        )
        clauses: list[str] = []
        params: list = []
        if context_key is not None:
            clauses.append("context_key = ?")
            params.append(context_key)
        if op_sig is not None:
            clauses.append("op_sig = ?")
            params.append(op_sig)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        for row in self._conn.execute(sql, params):
            node_key, parent_key, ck, sig, gpu, feats_json, value_us, depth = row[:8]
            visits, is_leaf, var, n_samp, status, run_id, measured, feat_ver = row[8:]
            try:
                features = json.loads(feats_json) if feats_json else {}
            except (TypeError, json.JSONDecodeError):
                continue
            yield NodeRow(
                node_key=node_key,
                parent_key=parent_key,
                context_key=ck,
                op_sig=sig,
                features=features,
                value_us=value_us,
                depth=depth,
                gpu=gpu,
                visits=visits or 0,
                is_leaf=None if is_leaf is None else bool(is_leaf),
                variance=var,
                n_samples=n_samp,
                status=status,
                run_id=run_id,
                measured_at=measured,
                feat_ver=int(feat_ver) if feat_ver is not None else 1,
            )

    def merge_nodes(self, src_path: Path | str) -> int:
        """Merge the ``node`` table from the autotune DB at ``src_path`` into this
        DB, and return the number of source node rows processed.

        Source rows are read read-only (:meth:`open_readonly`) and re-upserted through
        :meth:`record_nodes`, so they inherit its exact per-kind semantics: on a shared
        ``node_key`` a source *branch* wins only when strictly faster (keep-min) and a
        source *leaf* only when its measurement is strictly newer — so the merge is
        direction-independent. Because
        ``node_key`` folds the ``gpu`` identity, that collision never happens *across
        cards* — so merging another GPU's node store into this one accumulates a
        cross-hardware dataset and leaves every other card's rows untouched. The use
        case is bringing per-card node data measured on a rented GPU back to a single
        canonical DB (``scripts/merge_node_db.py``).

        Caveats: :meth:`iter_nodes` doesn't carry ``n_updates``, so each merged row
        re-enters as one ``record_nodes`` bump rather than carrying the source's write
        count — acceptable, since ``n_updates`` is bookkeeping only. ``visits`` DOES
        carry and SUM-accumulates on a key collision — right for the one-shot
        rent→tune→merge flow; re-merging the same snapshot double-counts (accepted:
        ``visits`` is a confidence weight, not an exact ledger)."""
        src = SearchDB.open_readonly(src_path)
        try:
            rows = list(src.iter_nodes())
        finally:
            src.close()
        # One transaction around the whole batch. ``record_nodes`` runs row-at-a-time on
        # this autocommit connection (``isolation_level=None``), which for a cross-card
        # merge of 10k+ rows would mean one fsync per row; an explicit BEGIN/COMMIT
        # collapses it to a single commit (the per-kind upsert within the batch still
        # sees prior inserts on the same connection).
        self._conn.execute("BEGIN")
        try:
            self.record_nodes(rows)
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")
        return len(rows)

    # ------------------------------------------------------------------
    # Measurements — read
    # ------------------------------------------------------------------

    def measurement(self, regime: Regime, identity: OpIdentity, knobs: dict | None) -> Measurement | None:
        """The one row for an exact arm, or ``None``."""
        return self._row(regime.digest, identity.digest, _decision(knobs))

    def _row(self, context: str, op: str, decision: str) -> Measurement | None:
        row = self._conn.execute(
            f"SELECT {_MEASUREMENT_COLS} FROM measurement WHERE context = ? AND op = ? AND decision = ?",  # noqa: S608
            (context, op, decision),
        ).fetchone()
        return _row_to_measurement(row) if row else None

    def measurements(self, regime: Regime, identity: OpIdentity | None = None) -> list[Measurement]:
        """Every measured arm of ``identity`` in ``regime`` — THE per-fork query.

        One indexed point lookup on the identity the fork is offered on, which is what replaces
        the whole-table scan the deploy used to build once per process and match by feature
        subset. ``identity=None`` spans the regime (the dataset / eval readers)."""
        context = regime.digest
        if identity is None:
            cur = self._conn.execute(f"SELECT {_MEASUREMENT_COLS} FROM measurement WHERE context = ?", (context,))  # noqa: S608
        else:
            cur = self._conn.execute(
                f"SELECT {_MEASUREMENT_COLS} FROM measurement WHERE context = ? AND op = ?",  # noqa: S608
                (context, identity.digest),
            )
        return [_row_to_measurement(row) for row in cur]

    def min_latency_for_context(self, context_key: str) -> float | None:
        """Fastest ``ok`` median measured in a regime — the pricing floor probes read it."""
        row = self._conn.execute(
            "SELECT min(us_median) FROM measurement WHERE context = ? AND status = 'ok' AND us_median > 0",
            (context_key,),
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def iter_measurements(self, *, status: str = "ok", min_latency_us: float = 0.0) -> Iterator[Measurement]:
        """Every measured row across every regime — the dataset layer's read (``Dataset.from_db``).
        Filters to ``status`` (default ``ok``) and ``us_median > min_latency_us`` so callers don't
        re-filter stale / failed rows."""
        cur = self._conn.execute(
            f"SELECT {_MEASUREMENT_COLS} FROM measurement WHERE status = ? AND us_median > ?",  # noqa: S608
            (status, min_latency_us),
        )
        for row in cur:
            yield _row_to_measurement(row)

    def best_per_op_time(self, regime: Regime, identity: OpIdentity, knobs: dict | None) -> float | None:
        """Best measured median (µs) for a kernel, or ``None`` when it has no clean ``ok`` row.

        The exact arm answers when it was measured — the two-level inner search records the best
        *whole-slice* total (``Σ`` over the slice's kernels, so a split-K main + combine both
        count) under the slice's own arm, and that Σ is the honest price of the slice. Otherwise
        the identity's best ``ok`` arm answers: every stage of a rewrite chain shares one
        identity, so the terminal's own measurement is found here directly — the chain of
        parent-to-child links this walk used to follow was a way of spelling that join when a
        measurement keyed on a rendered kernel instead of on what the kernel is."""
        exact = self.measurement(regime, identity, knobs)
        if exact is not None and exact.status == "ok":
            return exact.us_median
        rows = [m.us_median for m in self.measurements(regime, identity) if m.status == "ok" and m.us_median > 0]
        return min(rows) if rows else None

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()


def _decision(knobs: dict | None) -> str:
    """The ``measurement.decision`` column — the arm a knob dict decides, as canonical JSON.
    ONE projection, shared by the writer and every lookup: the tunable row (``tuning_knob_items``
    drops the ``S_*`` / ``CTX_*`` stamps and the marker BOOLs, which ride ``Regime.pins``), so a
    caller hands over the op's raw knobs and cannot spell the key two ways."""
    from emmy.compiler.pipeline.knob import tuning_knob_items  # noqa: PLC0415

    return json.dumps(dict(tuning_knob_items(dict(knobs or {}))), sort_keys=True)


def _row_to_measurement(row: tuple) -> Measurement:
    """One ``measurement`` SELECT (:data:`_MEASUREMENT_COLS` order) as a :class:`Measurement`."""
    return Measurement(
        op=row[0],
        knobs=json.loads(row[1]) if row[1] else {},
        status=row[2],
        us_median=row[3],
        us_min=row[4],
        us_max=row[5],
        n_samples=row[6],
        measured_at=row[7],
        captured=bool(row[8]),
        error=row[9],
    )
