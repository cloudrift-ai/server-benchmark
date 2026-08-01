"""SQLite-backed inventory + measurement store for the search package.

Pure persistence layer — no MCTS state, no propagation walks. Tables:

- ``loop_op`` / ``tile_op`` / ``kernel_op`` / ``cuda_op`` — one row per
  op encountered along a lowering chain. Keyed by ``op_cache_key``.
  Each row stores the JSON form (for programmatic inspection) and the
  pretty-printed form (for human inspection).
- ``lowering`` — best-known child for each parent op, one row per
  rewrite hop along the lowering chain (Loop→Tile, every intra-Tile
  autotune step, Tile→Kernel, Kernel→Cuda). Each row carries the knob
  delta the rule stamped at that hop plus a best-median upsert — the
  chain :meth:`SearchDB.best_per_op_time` walks to resolve a pre-final
  op's measured cost (greedy fork picks come from the ``Prior``, never
  DB replay). ``record_lowering`` upserts uniformly across
  dialects: a strictly better measured median replaces the row; a
  None measurement (bench_fail terminal) never overwrites a
  known-good row. Deterministic rewrites (single option) trivially
  win their own slot via the same path.
- ``perf`` — backend-agnostic measurement store. ``op_key`` is whichever
  terminal op the backend measured (today: a CudaOp; tomorrow whatever
  other backends lower to). ``backend`` partitions the table so the
  loop interpreter and the CUDA backend can coexist in the same DB.
- ``node`` — one row per search-tree node (every partial branch + leaf of a
  per-kernel autotune search), keyed by ``digest(context_key, op_sig,
  tunable-knob set)``. Each row carries the full feature dict passed to the
  prior (``H_*`` + ``S_*`` + knobs), a value-of-position latency (``1/best_reward``
  — the best latency reachable below the node; keep-min across sessions for
  branches, newest-measurement for leaves), and
  a ``parent_key`` pointer so ancestry between rows is recoverable. Content-keyed
  (parent-tree-independent), so it survives schema-version drops like ``perf``.
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
  tune's deployable -O3 re-benches land as extra parentless ``H_opt=3`` leaf
  rows under their own -O3 ``context_key`` (never colliding with the -O1 twin).
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

from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION

logger = logging.getLogger(__name__)


def _jsonable_geometry(geometry) -> list:
    """Render a launch grid/block to a JSON-safe nested list for the inventory
    tables. Int / str pass through; nested specs recurse; composite ``Expr``
    factors (ceil-div block extents for hint-driven masked tiles) render to
    their pretty string — inventory rows are for human inspection, not
    re-execution."""

    def conv(x):
        if isinstance(x, (int, str)):
            return x
        if isinstance(x, (tuple, list)):
            return [conv(e) for e in x]
        return x.pretty()  # Expr

    return [conv(spec) for spec in geometry]


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
class PerfRow:
    """One ``perf`` row.

    ``captured``: the measurement ran under CUDA graph capture (pure GPU time);
    False = wall semantics including per-launch dispatch (all pre-capture rows).
    Both kinds stay usable (replay, prior training); on write, a captured
    measurement supersedes an uncaptured one for the same key — see
    :meth:`SearchDB.record_perf`."""

    context_key: str
    op_key: str
    backend: str
    status: str
    stats: PerfStats
    measured_at: str
    knobs: dict
    captured: bool = False


@dataclass(frozen=True)
class PerfSample:
    """One measured terminal kernel — a ``perf`` row joined to its ``cuda_op``.

    The minimal row the dataset layer reads: the kernel's pretty source (for the C
    identifier), the recorded knobs (which already carry the ``S_*`` / ``H_*``
    features the variant was stamped with), and the median latency. Backs
    :meth:`SearchDB.iter_perf_samples`."""

    pretty: str
    knobs: dict
    latency_us: float
    error: str | None = None  # bench_fail failure text (None on ok rows / pre-error-column DBs)


@dataclass(frozen=True)
class LoweringRow:
    """One ``lowering`` row — best-known child for a parent op.

    ``knobs`` is the delta added at this rewrite step (e.g.
    ``005_blockify_launch`` adds ``{"BN": 64, "BM": 64}``). Greedy
    replay picks the fork whose newly-stamped knobs agree with this
    delta — no need to compare structural keys per fork."""

    parent_key: str
    parent_dialect: str
    child_key: str
    child_dialect: str
    knobs: dict
    best_median_us: float | None


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
    # Declarative golden-spelled identity of the benched shape (the golden YAML entry minus
    # knobs/latencies: ``{"kernel": ..., <shape fields>, "dynamic": ...}``), captured at
    # collection time by ``run --bench --record-shape``. ``None`` = legacy row (tune-written or
    # pre-identity bench) — kept in the DB but excluded from goldens-format measurement freezes.
    shape_spec: dict | None = None


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
    from emmy.compiler.ir.schedule import Stage, TilePlan, Workers  # noqa: PLC0415

    try:
        work = Workers.parse(str(f.get("WORK") or ""))  # the row's unit widths live here, not in TILE
        tp, st = TilePlan.parse(tile_spec, work), Stage.parse(stage_spec)
    except ValueError:
        return None
    if not tp.is_warp:
        return None
    if st.transport != "cp.async":
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


# The ``perf`` SELECT column list — order must match ``_row_to_perf``.
_PERF_COLS = (
    "context_key, op_key, backend, status, latency_us_median, latency_us_min, latency_us_max, "
    "latency_us_mean, latency_us_variance, n_samples, measured_at, knobs, captured"
)


class SearchDB:
    """Persistent inventory of compiled ops + their measured perf.

    Pass ``path=None`` for an in-memory database (default — keeps tests
    hermetic; tuning runs pass an explicit path like
    ``~/.cache/emmy/autotune.db``).
    """

    # Bumped whenever the fork-tree topology shifts in ways that change
    # ``parent_key`` / ``child_key`` for the same physical decision —
    # stale ``lowering`` rows from older versions won't match the new
    # keys and would silently slow the next tune sweep. On version
    # mismatch we drop the ``lowering`` table only; ``perf`` /
    # ``loop_op`` / ``tile_op`` etc. survive (source-hash keyed,
    # parent-tree-independent).
    #
    # Version log:
    #   1: M9.4 — planner-hoisted FM / FN / BN / BM forks. Parent-tree
    #       topology shifted vs. the legacy downstream forks.
    #   2: explicit-knob OFF sentinels — every variant now stamps every planner
    #       knob (tier-foreign ones get an OFF value: WM/WN/MMA on scalar,
    #       BM/BN/BR/FK on warp), so ``op_cache_key`` (which folds the knob dict)
    #       shifts for every TileOp/KernelOp. Stale ``lowering`` rows won't match.
    #   3: the RASTER launch-order codec — every contraction row now spells a fifth
    #       schedule family (``RASTER: ''``/``gm8``), so ``op_cache_key`` shifts for every
    #       matmul TileOp/KernelOp; cached pre-RASTER chains would silently replay
    #       old-key kernels and starve the new rows of evidence.
    _SCHEMA_VERSION = 3

    _SCHEMA = [
        """
        CREATE TABLE IF NOT EXISTS loop_op (
            key       TEXT PRIMARY KEY,
            body_json TEXT NOT NULL,
            pretty    TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tile_op (
            key       TEXT PRIMARY KEY,
            body_json TEXT NOT NULL,
            pretty    TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS kernel_op (
            key       TEXT PRIMARY KEY,
            body_json TEXT NOT NULL,
            pretty    TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cuda_op (
            key           TEXT PRIMARY KEY,
            kernel_source TEXT NOT NULL,
            arg_order     TEXT NOT NULL,
            grid          TEXT NOT NULL,
            block         TEXT NOT NULL,
            smem_bytes    INTEGER NOT NULL,
            pretty        TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS lowering (
            parent_key      TEXT PRIMARY KEY,
            parent_dialect  TEXT NOT NULL,
            child_key       TEXT NOT NULL,
            child_dialect   TEXT NOT NULL,
            knobs           TEXT NOT NULL DEFAULT '{}',
            best_median_us  REAL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS perf (
            context_key          TEXT NOT NULL,
            op_key               TEXT NOT NULL,
            backend              TEXT NOT NULL,
            status               TEXT NOT NULL,
            latency_us_median    REAL NOT NULL,
            latency_us_min       REAL NOT NULL,
            latency_us_max       REAL NOT NULL,
            latency_us_mean      REAL NOT NULL,
            latency_us_variance  REAL NOT NULL,
            n_samples            INTEGER NOT NULL,
            measured_at          TEXT NOT NULL,
            knobs                TEXT NOT NULL DEFAULT '{}',
            captured             INTEGER NOT NULL DEFAULT 0,
            error                TEXT,
            PRIMARY KEY (context_key, op_key, backend)
        )
        """,
        # Content-keyed (``node_key`` folds context + op_sig + knob set), so —
        # unlike ``lowering`` — it is parent-tree-independent and survives a
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
            feat_ver     INTEGER NOT NULL DEFAULT 1,
            shape_spec   TEXT
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
            self._conn.execute("DROP TABLE IF EXISTS lowering")
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
        # Additive migration: pre-existing DBs lack the ``error`` column
        # (``CREATE TABLE IF NOT EXISTS`` never alters). Writer-side only —
        # read-only consumers tolerate its absence instead. (``perf`` has no index on
        # ``error``, so unlike ``node.gpu`` this can run after the schema loop.)
        if not self._has_perf_error_column():
            self._conn.execute("ALTER TABLE perf ADD COLUMN error TEXT")
        # Additive ``node`` label-quality columns (visits / is_leaf / variance /
        # n_samples / status / run_id / measured_at). Per-column ALTERs so a
        # crash-interrupted migration self-heals on the next open; no index
        # references them, so like ``perf.error`` this runs after the schema loop.
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
        # (``diagnostics.node_report``) — a cross-vocabulary row featurizes to garbage —
        # but kept in the DB and carried by ``merge_nodes`` (data, not judgement).
        # NOTE: rows written after the 2026-07 tile-IR rebuild but before this column
        # shipped are spelled in the v2 vocabulary yet default to 1 — they quarantine
        # conservatively; re-collect with the ``collect-node-data`` flow.
        ("feat_ver", "INTEGER NOT NULL DEFAULT 1"),
        # Declarative golden-spelled shape identity (canonical JSON; NULL = legacy row).
        # See :attr:`NodeRow.shape_spec`; only identity-carrying rows enter goldens-format
        # measurement freezes (``data/freeze.py``).
        ("shape_spec", "TEXT"),
    )

    def _has_perf_error_column(self) -> bool:
        return any(r[1] == "error" for r in self._conn.execute("PRAGMA table_info(perf)"))

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
        check, no ``DROP TABLE lowering``, no WAL pragma — so a read-side consumer
        (``eval``, the dataset layer) never contends with a concurrent ``tune``
        writer or mutates the file. The read methods (``iter_perf`` /
        ``iter_perf_samples`` / ``lookup_*``) work; any write raises (the
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
                    "accepted only by the nodes-dataset consumers, e.g. `eval online --dataset nodes --db`"
                    if magic.startswith(b"{")
                    else ""
                )
                raise RuntimeError(f"{p} is not a sqlite database{hint}")
        self = cls.__new__(cls)
        self._conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
        return self

    # ------------------------------------------------------------------
    # Op-inventory writes (idempotent INSERT OR IGNORE)
    # ------------------------------------------------------------------

    def record_loop_op(self, key: str, body_json: str, pretty: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO loop_op (key, body_json, pretty) VALUES (?, ?, ?)",
            (key, body_json, pretty),
        )

    def record_tile_op(self, key: str, body_json: str, pretty: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO tile_op (key, body_json, pretty) VALUES (?, ?, ?)",
            (key, body_json, pretty),
        )

    def record_kernel_op(self, key: str, body_json: str, pretty: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO kernel_op (key, body_json, pretty) VALUES (?, ?, ?)",
            (key, body_json, pretty),
        )

    def record_cuda_op(
        self,
        key: str,
        *,
        kernel_source: str,
        arg_order: list[str],
        grid: list[int],
        block: list[int],
        smem_bytes: int,
        pretty: str,
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO cuda_op (key, kernel_source, arg_order, grid, block, smem_bytes, pretty) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                kernel_source,
                json.dumps(list(arg_order)),
                json.dumps(_jsonable_geometry(grid)),
                json.dumps(_jsonable_geometry(block)),
                int(smem_bytes),
                pretty,
            ),
        )

    # ------------------------------------------------------------------
    # Lowering edges
    # ------------------------------------------------------------------

    def record_lowering(
        self,
        parent_key: str,
        parent_dialect: str,
        child_key: str,
        child_dialect: str,
        *,
        knobs: dict | None = None,
        measured_median_us: float | None,
    ) -> None:
        """Upsert one ``parent_key`` → ``child_key`` lowering edge.

        ``knobs`` is the delta this rewrite step stamps onto the child
        (e.g. partition_loops adds ``{"BN": 64, "BM": 64, ...}``;
        launch_geometry adds nothing). Greedy replay picks forks by
        knob-subset match against this delta, so the row is enough to
        reconstruct the chain without re-querying ``perf``.

        Best-of upsert across every dialect — autotune fork rules live
        at Tile→Tile (blockify, split_register_axes) and used to be excluded
        here; recording every hop is how the chain stays replayable.
        Rows where the rewrite is genuinely deterministic (a single
        option) still trivially win their own slot, just via the same
        upsert path.
        """
        knobs_json = json.dumps(knobs or {}, sort_keys=True, default=str)
        existing = self._conn.execute(
            "SELECT child_key, best_median_us FROM lowering WHERE parent_key = ?",
            (parent_key,),
        ).fetchone()
        if existing is None:
            self._conn.execute(
                "INSERT INTO lowering (parent_key, parent_dialect, child_key, child_dialect, knobs, best_median_us) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (parent_key, parent_dialect, child_key, child_dialect, knobs_json, measured_median_us),
            )
            return
        # Replace iff the new measurement is strictly better than the
        # stored best (or the stored best is NULL). A None measurement
        # never overwrites a known-good row.
        cur_best = existing[1]
        if measured_median_us is None:
            return
        if cur_best is None or measured_median_us < cur_best:
            self._conn.execute(
                "UPDATE lowering SET child_key = ?, child_dialect = ?, knobs = ?, best_median_us = ? WHERE parent_key = ?",
                (child_key, child_dialect, knobs_json, measured_median_us, parent_key),
            )

    # ------------------------------------------------------------------
    # Perf — write
    # ------------------------------------------------------------------

    def record_perf(
        self,
        context_key: str,
        op_key: str,
        *,
        backend: str,
        status: str,
        stats: PerfStats,
        knobs: dict | None = None,
        captured: bool = False,
        error: str | None = None,
    ) -> None:
        """Upsert one ``perf`` row. Keep-best-``ok`` policy: a ``bench_fail``
        never overwrites a prior ``ok`` row, and among same-semantics ``ok``
        rows the lowest median wins. ``captured`` (CUDA-graph-captured, pure GPU
        time) adds a precedence axis: a captured measurement supersedes an
        uncaptured (wall-semantics) one regardless of median — the numbers
        aren't comparable, and captured is the better truth — while an
        uncaptured measurement never overwrites a captured one. ``error`` is the
        failure text for a ``bench_fail`` row (whitespace-collapsed, truncated)
        so failure forensics (``eval failures``) need no tune-log grepping."""
        existing = self.lookup_perf(context_key, op_key, backend=backend)
        if existing is not None and existing.status == "ok":
            if status != "ok":
                return  # a failure never replaces a good measurement
            if existing.captured and not captured:
                return  # wall semantics never overwrites a captured row
            if not (captured and not existing.captured) and stats.median >= existing.stats.median:
                return  # same semantics: keep the best median
        knobs_json = json.dumps(knobs or {}, sort_keys=True, default=str)
        if error is not None:
            error = " ".join(str(error).split())[:300] or None
        self._conn.execute(
            "INSERT OR REPLACE INTO perf "
            "(context_key, op_key, backend, status, latency_us_median, latency_us_min, latency_us_max, "
            " latency_us_mean, latency_us_variance, n_samples, measured_at, knobs, captured, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                context_key,
                op_key,
                backend,
                status,
                stats.median,
                stats.min,
                stats.max,
                stats.mean,
                stats.variance,
                stats.n_samples,
                datetime.now(UTC).isoformat(),
                knobs_json,
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
        untouched otherwise. Status follows ``record_perf``'s keep-best-``ok``
        policy: an ``ok`` row is never downgraded by a later ``bench_fail``
        (whatever the fail's sentinel ``value_us``), while a fail row upgrades to
        ``ok`` unconditionally; fail-vs-fail follows the leaf rule (newest sentinel).

        Manual lookup-guard + INSERT/UPDATE (the ``record_perf`` / ``record_lowering``
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
            spec_json = json.dumps(r.shape_spec, sort_keys=True) if r.shape_spec is not None else None
            existing = self._conn.execute(
                "SELECT value_us, n_updates, visits, status, measured_at, variance, n_samples FROM node WHERE node_key = ?",
                (r.node_key,),
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO node "
                    "(node_key, parent_key, context_key, op_sig, gpu, features, value_us, depth, n_updates, updated_at, "
                    " visits, is_leaf, variance, n_samples, status, run_id, measured_at, feat_ver, shape_spec) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                        spec_json,
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
            # ``shape_spec`` deliberately deviates from the value-paired-column rule: identity is
            # config-INTRINSIC (a ``node_key`` has exactly one shape), so it is COALESCE-kept in
            # both directions — a later identity-less tune re-measurement must not erase
            # freezability, and a non-replacing identity-carrying bench still proves the shape.
            if replace:
                self._conn.execute(
                    "UPDATE node SET value_us = ?, features = ?, parent_key = ?, n_updates = ?, updated_at = ?, "
                    "visits = ?, is_leaf = ?, variance = ?, n_samples = ?, status = ?, run_id = ?, measured_at = ?, "
                    "feat_ver = ?, shape_spec = COALESCE(?, shape_spec) WHERE node_key = ?",
                    (r.value_us, feats_json, r.parent_key, n_upd + 1, now, visits, is_leaf, r.variance, r.n_samples)
                    + (r.status, r.run_id, measured, r.feat_ver, spec_json, r.node_key),
                )
            else:
                self._conn.execute(
                    "UPDATE node SET n_updates = ?, updated_at = ?, visits = ?, shape_spec = COALESCE(shape_spec, ?) WHERE node_key = ?",
                    (n_upd + 1, now, visits, spec_json, r.node_key),
                )

    # ------------------------------------------------------------------
    # Search-tree nodes — read
    # ------------------------------------------------------------------

    def iter_nodes(self, *, context_key: str | None = None, op_sig: str | None = None) -> Iterator[NodeRow]:
        """Yield one :class:`NodeRow` per stored search-tree node (the value-of-position
        dataset backing ``eval online --dataset nodes``). Self-contained — no join.
        A read-only open of a pre-``node`` DB has no such table, so this degrades to
        yielding nothing instead of raising (mirrors ``iter_perf_samples``'
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
        # ``shape_spec`` ships after ``feat_ver`` — own presence check, absent → NULL (legacy).
        spec_col = "shape_spec" if "shape_spec" in node_cols else "NULL"
        sql = (
            f"SELECT node_key, parent_key, context_key, op_sig, {gpu_col}, features, value_us, depth, "  # noqa: S608
            f"{enrich_cols}, {feat_ver_col}, {spec_col} FROM node"
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
            visits, is_leaf, var, n_samp, status, run_id, measured, feat_ver, spec_json = row[8:]
            try:
                features = json.loads(feats_json) if feats_json else {}
            except (TypeError, json.JSONDecodeError):
                continue
            try:
                shape_spec = json.loads(spec_json) if spec_json else None
                if not isinstance(shape_spec, dict):
                    shape_spec = None
            except (TypeError, json.JSONDecodeError):
                shape_spec = None  # unparseable identity degrades to legacy, never drops the row
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
                shape_spec=shape_spec,
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
    # Perf — read
    # ------------------------------------------------------------------

    def lookup_lowering(self, parent_key: str) -> LoweringRow | None:
        """Return the best-known child for ``parent_key``, or ``None``
        when no row exists. Used by :meth:`best_per_op_time`'s chain
        walk to resolve a pre-final op's measured cost."""
        row = self._conn.execute(
            "SELECT parent_key, parent_dialect, child_key, child_dialect, knobs, best_median_us FROM lowering WHERE parent_key = ?",
            (parent_key,),
        ).fetchone()
        if row is None:
            return None
        return LoweringRow(
            parent_key=row[0],
            parent_dialect=row[1],
            child_key=row[2],
            child_dialect=row[3],
            knobs=json.loads(row[4]) if row[4] else {},
            best_median_us=row[5],
        )

    def lookup_perf(self, context_key: str, op_key: str, *, backend: str) -> PerfRow | None:
        row = self._conn.execute(
            f"SELECT {_PERF_COLS} FROM perf WHERE context_key = ? AND op_key = ? AND backend = ?",  # noqa: S608
            (context_key, op_key, backend),
        ).fetchone()
        return _row_to_perf(row) if row else None

    def min_latency_for_context(self, context_key: str, *, backend: str | None = None) -> float | None:
        if backend is None:
            row = self._conn.execute(
                "SELECT MIN(latency_us_median) FROM perf WHERE context_key = ? AND status = 'ok'",
                (context_key,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT MIN(latency_us_median) FROM perf WHERE context_key = ? AND backend = ? AND status = 'ok'",
                (context_key, backend),
            ).fetchone()
        return row[0] if row and row[0] is not None else None

    def iter_perf(self, context_key: str, *, backend: str | None = None) -> Iterator[PerfRow]:
        if backend is None:
            cur = self._conn.execute(
                f"SELECT {_PERF_COLS} FROM perf WHERE context_key = ?",  # noqa: S608
                (context_key,),
            )
        else:
            cur = self._conn.execute(
                f"SELECT {_PERF_COLS} FROM perf WHERE context_key = ? AND backend = ?",  # noqa: S608
                (context_key, backend),
            )
        for row in cur:
            yield _row_to_perf(row)

    def iter_perf_samples(self, *, backend: str | None = "cuda", status: str = "ok", min_latency_us: float = 0.0) -> Iterator[PerfSample]:
        """Yield one :class:`PerfSample` per measured terminal kernel — ``perf``
        joined to ``cuda_op`` on ``op_key = key``. The single place the two tables
        are joined; backs ``Dataset.from_db``. ``backend=None`` spans every backend.
        Filters to ``status`` (default ``ok``) and ``latency_us_median >
        min_latency_us`` so callers don't re-filter stale / failed rows. The
        ``error`` select degrades to ``NULL`` on a pre-error-column DB opened
        read-only (the additive migration runs writer-side only)."""
        error_col = "perf.error" if self._has_perf_error_column() else "NULL"
        sql = (
            f"SELECT cuda_op.pretty, perf.knobs, perf.latency_us_median, {error_col} "  # noqa: S608 — column name from a fixed two-way choice
            "FROM perf JOIN cuda_op ON perf.op_key = cuda_op.key "
            "WHERE perf.status = ? AND perf.latency_us_median > ?"
        )
        params: list = [status, min_latency_us]
        if backend is not None:
            sql += " AND perf.backend = ?"
            params.append(backend)
        for pretty, knobs_json, us, error in self._conn.execute(sql, params):
            try:
                knobs = json.loads(knobs_json) if knobs_json else {}
            except (TypeError, json.JSONDecodeError):
                continue
            yield PerfSample(pretty=pretty, knobs=knobs, latency_us=us, error=error)

    # ------------------------------------------------------------------
    # Per-op best time (summed into the outer terminal reward)
    # ------------------------------------------------------------------

    def best_per_op_time(self, context_key: str, op_key: str, *, backend: str = "cuda") -> float | None:
        """Best measured median (us) for the kernel that ``op_key`` lowers
        to in ``context_key``, or ``None`` when it has no clean ``ok``
        measurement.

        ``op_key`` is typically a finalized ``LoopOp`` key (the unit the
        outer search hands to the inner per-op tuner). Two ways it carries a
        time:

        1. **Direct row** — the two-level inner search records the best
           *whole-slice* total (``Σ`` over the slice's CudaOps, so split-K
           main + combine are both counted) under the LoopOp key itself.
           Preferred when present.
        2. **Chain walk** — otherwise follow the ``lowering`` best-known child
           links down to the ``cuda`` dialect and read that terminal's
           context-keyed median. A ``CudaOp`` key resolves here directly (no
           lowering row as parent).
        """
        direct = self.lookup_perf(context_key, op_key, backend=backend)
        if direct is not None and direct.status == "ok":
            return direct.stats.median
        cur: str | None = op_key
        seen: set[str] = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            row = self.lookup_lowering(cur)
            if row is None:
                break
            cur = row.child_key
            if row.child_dialect == "cuda":
                break
        if cur is None or cur == op_key:
            return None
        perf = self.lookup_perf(context_key, cur, backend=backend)
        return perf.stats.median if perf is not None and perf.status == "ok" else None

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()


def _row_to_perf(row: tuple) -> PerfRow:
    stats = PerfStats(
        median=row[4],
        min=row[5],
        max=row[6],
        mean=row[7],
        variance=row[8],
        n_samples=row[9],
    )
    knobs = json.loads(row[11]) if row[11] else {}
    return PerfRow(
        context_key=row[0],
        op_key=row[1],
        backend=row[2],
        status=row[3],
        stats=stats,
        measured_at=row[10],
        knobs=knobs,
        captured=bool(row[12]),
    )
