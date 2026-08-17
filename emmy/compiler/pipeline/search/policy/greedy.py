"""The greedy compile pick — :func:`greedy_decide`, a ``Run.resolve`` decide
factory picking each fork point's globally-best **complete** leaf via the
global online prior when one is trained, else option-0.

This is the deterministic pick for ``compile`` / ``run``, the structural
pricing probes, and the assembled-graph lowering. It is NOT a search and not
a ``Search`` policy: there is no frontier to rank, no tree, no benching — a
deterministic resolution is a fold over the pipeline (at each fork, a pure
function of ``(options, op, prior)``, argmin, continue), so its process state
is :meth:`Run.resolve`'s returned trace, never accumulated policy attributes.
It can only *use* a prior trained earlier by ``tune``, never train one.
Exploration stays in :class:`~.mcts.TuningSearch` (``Pipeline.tune``).

**Flatten, don't descend.** The lazy fork tree (``lowering/tile`` planner) is an
MCTS data structure — it stages knob choices across levels (``BR`` → ``BM/BN`` →
``FM/FN``) so MCTS pays one node per pop. Greedy must NOT walk it level-by-level:
a branch carries only a *partial* tile, and ``features.knob_features`` can't compute
the tile's area / occupancy until ``FM/FN`` are pinned — so the prior is blind at
the ``BM/BN`` choice and defaults to ``BN=16`` for every shape. Instead greedy
**flattens** each fork point to its complete leaves
(:func:`~emmy.compiler.pipeline.fork.flatten_leaves` — cheap, ``expand``
builds only knob dicts; materialization stays deferred to the one chosen leaf)
and picks the one with the lowest :meth:`Prior.mean_scores` over the
full feature vector the prior trained on: the ``H_*`` host/hardware regime + the
op's ``S_*`` structural knobs (read off the offer op) + the leaf's complete knob
row. The pick equals scoring the flat candidate set, invariant to the tree's
level order. With no trained prior the model is unfit → it falls back to the first
emitted sibling.

**Greedy is ranked by evidence and by nothing else.** Its tiers are recorded
goldens, then measurements, then the fitted prior — every one of them a
recording of something that ran. There is no hand-written tier: no leaf is
promoted, demoted, withheld or given a head start here, and no fallback
default is chosen for being safe. Where all three tiers are silent the pick
degenerates to the enumeration's first leaf, which carries no meaning and can
be arbitrarily slow. That is the accepted cost of the rule, not a defect to
patch: a bad unmeasured pick is fixed by measuring (a tune, a recorded golden)
or by fitting the prior better, never by teaching this module a preference.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING

from emmy.compiler.graph import Graph
from emmy.compiler.pipeline.fork import Fork, flatten_leaves

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.pipeline import ForkPoint


@lru_cache(maxsize=1)
def _tile_pipeline():
    """The ``lowering/tile``-only pipeline the structural price probes drive —
    frozen and shareable, so one load serves every nested descent."""
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415

    return Pipeline.build(["lowering/tile"])


# The rule whose fork prices a kernel: the prior's predicted µs for the chosen
# complete schedule row at the contraction fork (the one hierarchical tile → stage → reduce
# fork the tile schedule offers) is the per-kernel cost the
# structural pricing sums (defined here, not in ``two_level``, because that module imports
# this package at module scope — the reverse would cycle). The fork moved out of recognition
# when the two halves split: ``010_recognize`` emits the unmapped ``TileOp`` and
# ``020_schedule`` offers the row fork, so the scored trace decision records under the latter.
PARTITION_RULE = "020_schedule"


def tile_identity(knobs: dict) -> frozenset:
    """The blocklist key for a tile — its canonical tuning-knob view
    (:func:`~emmy.compiler.pipeline.knob.tuning_knob_items`: the ``S_*`` / ``H_*``
    features and marker booleans dropped, values stringified) as a hashable set.
    Computed identically for a greedy leaf's fork knobs and for a rejected node's
    realized knobs — the honest-stamping rule makes the two agree — so
    :func:`greedy_decide` can skip a leaf that already failed ``validate(ctx)``
    downstream (the smem / thread-budget gate)."""
    from emmy.compiler.pipeline.knob import tuning_knob_items  # noqa: PLC0415

    return frozenset(tuning_knob_items(knobs))


def _tile_blocked(fork_knobs: dict, blocked: set[frozenset]) -> bool:
    """True if a leaf's complete knob row matches a blocklisted tile. Only a leaf
    fork carries every identity knob, so a partial (branch) fork — whose identity
    is a strict subset — never equals a full-row entry and is never skipped."""
    return tile_identity(fork_knobs) in blocked


# ---------------------------------------------------------------------------
# ``greedy_decide`` — the greedy pick as a ``Run.resolve`` decide callback.
# ``Pipeline.run`` and the structural pricing probes route through this.
# ---------------------------------------------------------------------------

# Sentinel distinguishing "load the global prior lazily on the first fork"
# (the ``Pipeline.run`` default) from an explicitly injected prior — which may
# legitimately be ``None`` (= no prior, option-0 emission order).
_LOAD_PRIOR = object()


@lru_cache(maxsize=1)
def _load_prior_cached(path_str: str, mtime: int):  # noqa: ARG001 — args are the cache key
    """The rehydrated global prior for one ``(online-file path, mtime)`` — the
    process-wide memo behind :func:`_load_prior_safe`. ``maxsize=1`` evicts on any
    key change, so a rewritten checkpoint (new mtime) reloads and a stale one is
    dropped. The deploy path only *reads* this prior (``mean_scores`` / ``pick`` /
    ``evidence_pick``), never trains it, so one shared instance is safe across the
    ~96 program compiles of a serve boot."""
    from emmy.compiler.pipeline.search.prior import load_prior  # noqa: PLC0415

    return load_prior()


def _load_prior_safe():
    """Load the one global prior (``OnlinePrior`` behind the
    ``OfflinePrior`` cold-start fallback), memoized per process on the online
    file's ``(path, mtime)`` — a serve boot compiles ~96 programs and each would
    otherwise ``json.loads`` the 56 MB checkpoint again (the dominant boot-time
    resolution cost). Best-effort: any load failure → ``None`` → emission order
    (option-0) — a bad/missing prior must never break compile."""
    try:
        from emmy import config  # noqa: PLC0415

        path = config.online_path()
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            mtime = -1  # missing file → stable key; a fresh prior is loaded once
        return _load_prior_cached(str(path), mtime)
    except Exception:  # noqa: BLE001
        return None


def _first_leaf(option: object) -> object:
    """Descend an option to its first leaf (branch Forks take child 0) — the
    no-information emission-order pick the no-prior fallback keeps."""
    while isinstance(option, Fork) and not option.is_leaf:
        option = option.expand()[0]
    return option


def _leaf_knobs(leaf: object) -> dict:
    """A flattened leaf's complete knob row: a leaf ``Fork`` carries it as
    ``knobs``; a concrete ``Op`` carries its own; a ``Graph`` splice has no
    single row (scored structurally, never by knobs) — empty, matching how
    ``LazyCandidate.from_option`` treats it during the tuning search."""
    if isinstance(leaf, Fork):
        return dict(leaf.knobs)
    return dict(getattr(leaf, "knobs", None) or {}) if not isinstance(leaf, Graph) else {}


def _decision_key(fp: ForkPoint, blocked: dict | None) -> tuple | None:
    """The decision memo's key for one schedule fork, or ``None`` where the memo does not apply.

    GREEDY-ONLY and scoped to one factory call (one compile attempt), because a decision is a
    CONCLUSION over evidence — MCTS must explore, and evidence may move between attempts. Within
    one attempt the pick is deterministic, so N same-shape kernels — 28 identical per-layer
    matmuls — decide once and the rest replay by tree descent instead of a flatten-and-score.

    ``TileOp``-rooted forks only, keyed on the scheduler's own ``pool_key``: it carries the
    dtype / hint / pin discriminators op identity deliberately excludes, so the memo can never
    serve a twin with different atom eligibility (the pool cache's f16-serves-f32 hazard, same
    class). The rule identity separates two forks offered on one op, and the node's blocklist
    CONTENT keys the validate-retry path — a retry with a blocked tile is a different decision."""
    from emmy.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

    if not isinstance(fp.root_op, TileOp):
        return None
    from emmy.compiler.pipeline.passes.lowering.tile._schedule import pool_key  # noqa: PLC0415

    rule = fp.match.rule
    node_blocked = blocked.get(fp.node_id) if blocked else None
    return (
        getattr(getattr(rule, "pass_", None), "name", None),
        getattr(rule, "name", None),
        pool_key(fp.root_op),
        frozenset(node_blocked) if node_blocked else frozenset(),
    )


def _find_decided_leaf(options: list, want: dict) -> object | None:
    """The leaf carrying exactly the memoized row ``want`` — replayed by DESCENDING the lazy fork
    tree: a branch expands only when every knob it pins matches the row, so the walk instantiates
    O(path × siblings) Forks, never the flat leaf set (the descent ``build_fork_tree`` was built
    for, which the scoring pass cannot use because it must rank complete rows). ``None`` when no
    leaf matches — emission drift between two offers of one key — and the caller re-decides."""
    for o in options:
        if isinstance(o, Fork) and not o.is_leaf:
            if all(want.get(name) == value for name, value in o.knobs.items()):
                found = _find_decided_leaf(o.expand(), want)
                if found is not None:
                    return found
        elif _leaf_knobs(o) == want:
            return o
    return None


def _leaf_op(leaf: object):
    """The concrete ``Op`` behind a flattened leaf, or ``None``. Reads
    ``OptionFork.option`` rather than firing ``expand()`` — a planner tree
    ``_Leaf``'s thunk would materialize a TileOp just to inspect it."""
    from emmy.compiler.ir.base import Op  # noqa: PLC0415

    if isinstance(leaf, Op):
        return leaf
    option = getattr(leaf, "option", None)
    return option if isinstance(option, Op) else None


def _leaf_graph(leaf: object) -> Graph:
    """The ``Graph`` behind a structural leaf (raw or ``OptionFork``-wrapped)."""
    return leaf if isinstance(leaf, Graph) else leaf.option


def _price_kernel(graph: Graph, nid: str, ctx: Context, prior, memo: dict[str, float | None], db: object | None = None) -> float | None:
    """One kernel's price: a nested deterministic resolution of its
    single-node slice through ``lowering/tile`` only (the partition fork is
    where the prior prices a complete tile row; the kernel/cuda passes add
    nothing and cost real CPU), reading the chosen leaf's µs off the
    slice-resolve's trace entry at the partition fork. ``db`` rides into the
    nested decide, so the partition-fork pick follows the same deploy evidence
    hierarchy as a top-level knob pick (reservoir -O3 rows, then the tune DB's
    -O1 ranking rows, model prediction only where nothing was measured) — the
    priced µs is a measurement wherever the tune benched this kernel. Memoized
    per ``Op.cache_key`` so 28 identical per-layer kernels price once.
    Best-effort: any resolve failure prices as ``None`` (→ the caller keeps
    the op-variant path)."""
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415
    from emmy.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415

    key = graph.nodes[nid].op.cache_key()
    if key in memo:
        return memo[key]
    us: float | None = None
    try:
        nested = greedy_decide(prior=prior, price_structural=False, db=db)
        _, trace = Run(pipeline=_tile_pipeline(), ctx=ctx).resolve(single_node_graph(graph, nid), nested)
        us = next((d.score for d in trace if d.rule_name == PARTITION_RULE and d.node_id == nid), None)
    except Exception:  # noqa: BLE001 — a price-probe failure must never break compile
        us = None
    memo[key] = us
    return us


def _price_graph(graph: Graph, ctx: Context, prior, memo: dict[str, float | None], db: object | None = None) -> float | None:
    """Σ of per-kernel best-µs prices over ``graph``'s kernel-bearing
    nodes, or ``None`` when any kernel is unpriceable (no partition fork —
    e.g. a pre-tiled combine ``TileOp`` — or a failed nested resolve)."""
    prices = [_price_kernel(graph, nid, ctx, prior, memo, db) for nid, n in graph.nodes.items() if n.op.cache_key() is not None]
    if not prices or any(p is None for p in prices):
        return None
    return sum(prices)


def _price_op_leaf(fp: ForkPoint, leaf: object, prior, memo: dict[str, float | None], db: object | None = None) -> float | None:
    """The keep-fused side's price: the leaf's ``Op`` rebound into a
    single-node slice of the current graph, priced like any kernel."""
    from emmy.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415

    option = _leaf_op(leaf)
    if option is None:
        return None
    sub = single_node_graph(fp.match.graph, fp.node_id)
    sub.nodes[fp.node_id].op = option
    return _price_graph(sub, fp.ctx, prior, memo, db)


def _priced_pick(fp: ForkPoint, leaves: list, prior, memo: dict[str, float | None], db: object | None = None) -> object | None:
    """The priced argmin over a kernel-set fork's leaves — the structural
    (``Graph``-splicing) options and the keep-fused ``Op`` side alike — or
    ``None`` when some leaf cannot be priced.

    This exists because the per-op prior scores ONE kernel's knob row, so its
    score for a multi-kernel ``Graph`` option is meaningless: the leaf carries
    no row of its own. It is a way of ASKING the evidence about a leaf the
    ordinary ranking cannot featurize, not a rule about which leaf should win.
    Every leaf is priced the same way: the best µs at each kernel's partition
    fork, obtained by a nested deterministic resolution of the kernel's
    single-node slice (``lowering/tile`` only, no backend, CPU-only —
    :func:`_price_kernel`); a structural option's price is the Σ over its
    fragment's kernels. The nested pick follows the deploy evidence hierarchy
    (``db`` threads the tune DB down), so each side's price is a *measurement*
    wherever the tune benched that kernel, and the loaded prior prices the
    unmeasured remainder. A Σ-of-predictions comparison across two different
    kernel families is exposed to the model's absolute-µs error, which does not
    cancel across sides the way it does among siblings of one fork — that is a
    fitting requirement on the prior, and it is the prior's problem to fix.

    ``None`` (an unpriceable leaf) hands the fork back to the ordinary leaf
    ranking with EVERY leaf still in it, structural ones included: an option
    nothing can price is just an option, and greedy is not shielded from
    picking it."""
    from emmy.compiler.pipeline.pipeline import _is_structural_option  # noqa: PLC0415

    priced = [
        (o, _price_graph(_leaf_graph(o), fp.ctx, prior, memo, db) if _is_structural_option(o) else _price_op_leaf(fp, o, prior, memo, db))
        for o in leaves
    ]
    if any(us is None for _, us in priced):
        return None
    return min(priced, key=lambda op_us: op_us[1])[0]


# The default nvcc flags of the tune ranking pass (``emmy/commands/tune.py``'s
# ``apply_nvcc_flags(default=...)``) — the deploy-side DB consult also queries the
# perf rows recorded under a ``context_key`` with these flags, beside the deploy's own.
_TUNE_RANKING_FLAGS = "-Xcicc -O1"


# Process-wide memo for the built DB index, keyed on (db path, mtime, the three
# context keys). The index depends only on the DB file and cc+nvcc-flags (NOT the
# op shape — ``structural_key`` folds neither), so for a serve boot it is identical
# across all ~96 program compiles; without this the 527 MB perf scan reran each time.
# Bounded to the current key (cleared on miss), like ``_load_prior_cached``.
_DB_INDEX_CACHE: dict = {}


def _db_measured_index(db, ctx) -> dict[frozenset, list[tuple[dict, float, bool]]]:
    """Caching wrapper over :func:`_db_measured_index_build` — memoizes the built
    index per process on ``(db path, mtime, context keys)``, invalidated when the
    DB file's mtime changes. An in-memory DB (no ``_path``) or an unstatable file
    bypasses the cache and rebuilds. Best-effort throughout: a failed key
    computation just rebuilds."""
    path = getattr(db, "_path", None)
    if path is None:
        return _db_measured_index_build(db, ctx)
    try:
        from emmy.compiler.pipeline.search.policy.mcts import O3_NVCC_FLAGS  # noqa: PLC0415

        # Stat the main file AND its ``-wal`` sidecar: in WAL mode a ``record_perf``
        # commit can land in the WAL without bumping the main file's mtime, so a
        # main-mtime-only key could serve a stale index to a same-process
        # write-then-read (the tune lane). ``os.stat`` on a missing WAL → skip it.
        wal = path.with_name(path.name + "-wal")
        mtime = (path.stat().st_mtime_ns, wal.stat().st_mtime_ns if wal.exists() else 0)
        ctx_keys = frozenset(
            {
                ctx.structural_key(),
                replace(ctx, compile_flags=_TUNE_RANKING_FLAGS).structural_key(),
                replace(ctx, compile_flags=O3_NVCC_FLAGS).structural_key(),
            }
        )
        key = (str(path), mtime, ctx_keys)
    except Exception:  # noqa: BLE001 — any key-build failure → just rebuild uncached
        return _db_measured_index_build(db, ctx)
    hit = _DB_INDEX_CACHE.get(key)
    if hit is not None:
        return hit
    index = _db_measured_index_build(db, ctx)
    _DB_INDEX_CACHE.clear()  # keep only the current (path, mtime, keys)
    _DB_INDEX_CACHE[key] = index
    return index


def _db_measured_index_build(db, ctx) -> dict[frozenset, list[tuple[dict, float, bool]]]:
    """The tune DB's measured ``ok`` cuda perf rows, indexed by their ``S_*``
    structural signature (stringified values — perf knobs round-trip JSON) —
    the deploy-side analogue of ``Prior._o3_evidence``. Queries three context
    keys (``context_key`` folds the nvcc flags): the deploy's own, and the same
    key with the ``-Xcicc -O1`` tune ranking flags and with ``-Xcicc -O3``,
    where the tune's deployable re-benches land. Each entry carries a
    ``deployable`` flag: an -O1 median is a *ranking* signal with known -O3
    inversions, so the pick must prefer rows measured at deployable flags
    wherever they exist (letting -O1 rows override a well-trained model
    regressed qkv/mlp_down ~15% in the ninth-4090-sweep verification).
    Best-effort: any failure returns an empty index (deploys fall back to the
    prior)."""
    from emmy.compiler.pipeline.search.policy.mcts import O3_NVCC_FLAGS  # noqa: PLC0415

    index: dict[frozenset, list[tuple[dict, float, bool]]] = {}
    try:
        o1_key = replace(ctx, compile_flags=_TUNE_RANKING_FLAGS).structural_key()
        keys = {ctx.structural_key(), o1_key, replace(ctx, compile_flags=O3_NVCC_FLAGS).structural_key()}
        # Sorted — set iteration order is per-process (hash-seeded), and the index's
        # per-signature row order must not vary across boots (ties resolve through it).
        for ck in sorted(keys):
            for row in db.iter_perf(ck, backend="cuda"):
                if row.status != "ok" or row.stats.median <= 0:
                    continue
                sig = frozenset((k, str(v)) for k, v in row.knobs.items() if k.startswith("S_"))
                tun = {k: str(v) for k, v in row.knobs.items() if not k.startswith(("S_", "H_"))}
                index.setdefault(sig, []).append((tun, float(row.stats.median), ck != o1_key))
    except Exception:  # noqa: BLE001 — a DB consult failure must never break compile
        return {}
    return index


def _sig_groups(index: dict[frozenset, list[tuple[dict, float, bool]]], sig: frozenset) -> list[list[tuple[dict, float, bool]]]:
    """Drift-tolerant signature match — see :meth:`Prior.sig_groups` (one
    contract for the reservoir tier and this DB tier)."""
    from emmy.compiler.pipeline.search.prior.base import Prior  # noqa: PLC0415

    return Prior.sig_groups(index, sig)


def _db_measured_pick(index: dict[frozenset, list[tuple[dict, float, bool]]], rows: list[dict]) -> tuple[int, float] | None:
    """Measured-evidence argmin over candidate knob rows against the DB index —
    the same prefix-consistency contract as ``Prior.evidence_pick`` (every
    tunable knob the candidate specifies must match the measured row; undecided
    knobs are free). Signature matching is drift-tolerant (:func:`_sig_groups`).
    Two-tier: rows measured at deployable flags (the deploy's own + ``-O3``
    context keys) decide outright; ``-O1`` ranking rows decide only when no
    candidate has deployable evidence — an -O1 median is a ranking signal with
    known -O3 inversions, and letting it override a well-trained model regressed
    qkv/mlp_down ~15% in the ninth-4090-sweep verification. Keeps a config the
    tune *measured* fastest from losing the deploy to an unmeasured model
    extrapolation (eighth golden sweep, finding 2). -O3 reservoir evidence,
    where present, still takes precedence at the call site."""
    from emmy.compiler.pipeline.knob import canonical_row_key, evidence_row_vouches  # noqa: PLC0415

    row_key: dict[int, tuple] = {}  # i → canonical_row_key(rows[i]), computed at most once

    def key_of(i: int) -> tuple:
        if i not in row_key:
            row_key[i] = canonical_row_key(rows[i])
        return row_key[i]

    def better(us: float, i: int, cur: tuple[int, float] | None) -> bool:
        # Tie on µs (one measured row matching several candidates) breaks by the
        # candidates' canonical content, never their enumeration order.
        return cur is None or us < cur[1] or (us == cur[1] and key_of(i) < key_of(cur[0]))

    # Every candidate at one fork shares the offer op's ``S_*`` base (``rows`` is
    # ``{**base, **leaf_knobs}``), so one signature covers the whole candidate set —
    # measured: exactly one distinct sig per call, over sets up to ~41.5k rows.
    # On an exact index hit ``_sig_groups`` is already O(1), so this memo buys
    # little (~2%) in the common case. It matters on the DRIFT path: when the
    # candidate's sig is NOT a key (the #311 ``S_warp_eligible`` vocabulary drift
    # this tier's shared-key matching exists to absorb), every call rescans EVERY
    # index signature building a dict per entry — 41.5k candidates x 61 signatures
    # for a single fork. The memo bounds that at one scan per distinct sig.
    # Per-call scope, so a rebuilt index is never served stale.
    groups_memo: dict[frozenset, list[list[tuple[dict, float, bool]]]] = {}

    best: tuple[int, float] | None = None  # deployable lane
    best_rank: tuple[int, float] | None = None  # fallback: rows from the -O1 ranking pass
    for i, cand in enumerate(rows):
        sig = frozenset((k, str(v)) for k, v in cand.items() if k.startswith("S_"))
        cand_tun = {k: str(v) for k, v in cand.items() if not k.startswith(("S_", "H_"))}
        if sig not in groups_memo:  # not ``.get`` — an empty group list is a valid, falsy hit
            groups_memo[sig] = _sig_groups(index, sig)
        for measured in groups_memo[sig]:
            for row_tun, us, deployable in measured:
                # A row counts as evidence when it matches every knob the candidate
                # has decided; undecided knobs are free (``evidence_row_vouches``).
                if not evidence_row_vouches(cand_tun, row_tun):
                    continue
                if deployable:
                    if better(us, i, best):
                        best = (i, us)
                elif better(us, i, best_rank):
                    best_rank = (i, us)
    return best if best is not None else best_rank


def _warn_disjoint_evidence(index: dict[frozenset, list[tuple[dict, float, bool]]], rows: list[dict], node_id: str) -> None:
    """Warn when a fork's candidate set is DISJOINT from its measured evidence:
    the DB holds rows for this kernel's structural signature, yet
    :func:`_db_measured_pick` matched none of them against any offered
    candidate. That condition is exactly "the tune measured a schedule tier
    the deploy did not offer" — the model then extrapolates over an
    evidence-free candidate set, which shipped gemma o_proj on a scalar tile
    16x its own measured mma rows (the stale-placeholder offer gap). A cold
    compile (no rows for the signature at all) stays silent — extrapolation
    is expected there."""
    sigs = {frozenset((k, str(v)) for k, v in r.items() if k.startswith("S_")) for r in rows}
    n_measured = sum(len(g) for sig in sigs for g in _sig_groups(index, sig))
    if n_measured:
        logger.warning(
            "deploy: node %r has %d measured DB row(s) for its structural signature, but none matches any of the "
            "%d offered candidates — the tune measured a schedule tier this compile did not offer; falling back to "
            "the model prediction. Investigate the enumeration (offer gates) for this kernel.",
            node_id,
            n_measured,
            len(rows),
        )


#: The precision-trading pin universe the regime check must cover in BOTH directions — a record
#: that omits one of these was measured with it OFF, and must not decide when it is live-ON.
_PRECISION_PINS = ("FAST_MATH", "FAST_EXP", "F16_MMA_F32_ACC", "FP8_MMA")


def _pins_live(pins: dict) -> bool:
    """Whether the record's input-pin regime IS the live one — exact per pin: a BOOL pin compares
    against the live env pin (unset = the knob's off state), anything else against the raw env
    string. Strict BOTH ways: a record measured under FAST_MATH decides nothing on a standard
    deploy, and a standard record decides nothing under a live precision-trading pin — the
    precision universe (:data:`_PRECISION_PINS`, umbrella semantics per ``space.precision_pin``)
    is compared even for pins the record omits (omitted = measured OFF)."""
    from emmy import config  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import KnobType, registry  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import precision_pin  # noqa: PLC0415

    knobs = registry()
    for name, value in pins.items():
        kn = knobs.get(str(name))
        raw = kn.raw() if kn is not None else config.knob_raw(str(name))
        if kn is not None and kn.type is KnobType.BOOL:
            live = kn.parse(raw) if raw is not None else False
            if bool(value) != live:
                return False
        elif (raw or "") != str(value):
            return False
    umbrella = bool(pins.get("FAST_MATH", False))
    for name in _PRECISION_PINS:
        recorded = bool(pins.get(name, umbrella))
        kn = knobs.get(name)
        live = bool(precision_pin(kn)) if kn is not None else False
        if recorded != live:
            return False
    return True


# Optional per-consultation verdict sink for the drift audit (``search/audit.py``). ``None``
# (the default, and what every real compile runs with) is zero-cost: one identity test per
# consulted fork. ``golden_audit`` installs a list that :func:`_verified_pick` appends one record
# per consulted SCHEDULE fork to — the supported hook the audit reads.
_AUDIT_SINK: list[dict] | None = None


@contextmanager
def golden_audit(records: list[dict]):
    """Collect one ``{node, key, verdict, golden, us, n_rows, unrealized}`` record per
    verified-tier consultation into ``records`` for the duration of the block. ``key`` is the
    fork's ``deploy_identity`` — the strict structural identity the tier joins on, not a
    classified shape. Verdicts:

      MATCH  a record carrying the fork's identity decided it (its spelled row equalled exactly
             one enumerated leaf)
      DRIFT  records carry the identity but NO offered leaf equals any of their rows — the
             recording no longer realizes under the current enumeration, so the tier decides
             nothing and falls through (:func:`_verified_pick` already warns)
      GAP    no record carries the fork's identity (coverage information, not a defect)

    ``unrealized`` (MATCH/DRIFT only; ``None`` on GAP) lists the identity's records that no
    offered leaf realizes — the per-entry signal the ``eval golden`` offer audit reads (one entry
    can be individually unrealizable while a sibling still MATCHes and floors the deploy)."""
    global _AUDIT_SINK
    prev = _AUDIT_SINK
    _AUDIT_SINK = records
    try:
        yield records
    finally:
        _AUDIT_SINK = prev


def _audit_record(
    node_id: str, key, verdict: str, golden: str | None, us: float | None, n_rows: int, unrealized: list | None = None
) -> None:
    if _AUDIT_SINK is not None:
        _AUDIT_SINK.append(
            {"node": node_id, "key": key, "verdict": verdict, "golden": golden, "us": us, "n_rows": n_rows, "unrealized": unrealized}
        )


def _live_leaf_rows(leaves: list, node_blocked) -> list[tuple[object, dict]]:
    """The fork's enumerated schedule rows — every non-structural leaf paired with its complete
    knob row, minus the tiles this node already failed to lower."""
    from emmy.compiler.pipeline.pipeline import _is_structural_option  # noqa: PLC0415

    rows = [(o, _leaf_knobs(o)) for o in leaves if not _is_structural_option(o)]
    return rows if node_blocked is None else [(o, k) for o, k in rows if not _tile_blocked(k, node_blocked)]


def _verified_index(ctx: Context) -> tuple[dict, dict]:
    """The card's recorded goldens keyed by STRICT structural identity — the recognized term's
    algebra digest + dtype fingerprint (``_schedule.deploy_identity``), derived record-side from
    each record's own persisted program through the shared recognition core. Returns
    ``(schedule rows, routing rows)`` as ``{identity: [records fastest-first]}``, scoped to the
    live ``(gpu_name, compute_cap)`` and the live pin regime. Best-effort per record (an
    underivable row is skipped — the decode tripwire is where that is loud); classification-free:
    no shape key, no matching heuristic, identity or nothing."""
    from emmy.compiler.pipeline.search.golden import flush_identity_store, kernel_identity, records_for_card  # noqa: PLC0415

    gpu_name = getattr(ctx, "gpu_name", None)
    if not gpu_name:
        return {}, {}
    sched: dict = {}
    routing: dict = {}
    try:
        cap = tuple(ctx.compute_capability)
        for g in records_for_card(gpu_name, cap):
            if not g.knobs or not _pins_live(g.pin_map):
                continue
            identity = kernel_identity(g)
            if identity is None:
                continue
            (routing if g.is_routing else sched).setdefault(identity, []).append(g)
        for entries in (*sched.values(), *routing.values()):
            entries.sort(key=lambda g: g.emmy_us or float("inf"))
        flush_identity_store()
    except Exception:  # noqa: BLE001 — a golden consult failure must never break compile
        return {}, {}
    return sched, routing


def _verified_pick(fp: ForkPoint, sched_idx: dict, routing_idx: dict, blocked) -> tuple[object, float, dict | None] | None:
    """The strict verified-tier decision for one fork, or ``None``.

    A SCHEDULE fork (the recognized ``TileOp`` root): the fork's ``deploy_identity`` selects the
    records; the fastest record whose spelled row is EXACTLY one enumerated leaf
    (``canonical_row_key`` equality — no prefix, no any-of) decides. A record that matches the
    identity but equals no leaf is DRIFT: warn loudly and decide nothing (fail-closed — the fuzzy
    acceptance this tier replaced is what deployed wrong kernels).

    A PLACEMENT fork (``Graph`` cut options beside the fused ``TileOp``): a ROUTING record picks
    the cut fragment whose parent piece stamps the record's ``PLACE`` keys; otherwise a schedule
    record for the fused identity keeps the fused side (the verified µs is fused truth, so the
    prior must not cut against it).

    Under an active :func:`golden_audit` sink every SCHEDULE consultation also appends its verdict
    (MATCH / DRIFT / GAP) — the drift audit's only reading of this tier."""
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415
    from emmy.compiler.ir.tile.path import ROUTING_FAMILIES  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import family_of, schedule_row_key  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.tile._schedule import deploy_identity  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import _is_structural_option  # noqa: PLC0415

    leaves = flatten_leaves(fp.options)
    structural = [o for o in leaves if _is_structural_option(o)]
    if structural:
        tile = next((o for o in leaves if isinstance(o, TileOp)), None)
        if tile is None:
            return None
        identity = deploy_identity(tile)
        for rec in routing_idx.get(identity, ()):
            route = {k: str(v) for k, v in rec.knobs.items()}
            families = {family_of(k) for k in route}

            def _stamps(opt) -> dict:
                ops = [n.op for n in _leaf_graph(opt).nodes.values()]
                return {k: str(v) for op in ops for k, v in (getattr(op, "knobs", {}) or {}).items() if family_of(k) in ROUTING_FAMILIES}

            hit = next((o for o in structural if _stamps(o) == route), None)
            if hit is None and all("@" not in k for k in route):
                # A BARE spelling names the primary site, which the realizer stamps canonically
                # (``PLACE@<seam>`` / ``SPLIT@<axis>``), so it cannot equal the stamp. Accept the
                # first option routing the same families to the same values — recognition offers
                # them shallowest-first, which is what the bare key means.
                hit = next(
                    (
                        o
                        for o in structural
                        if {family_of(k) for k in _stamps(o)} == families and set(_stamps(o).values()) == set(route.values())
                    ),
                    None,
                )
            if hit is not None:
                return hit, float(rec.emmy_us or 0.0), None
            logger.warning(
                "deploy: node %r matches routing golden %s by identity, but no structural option routes %s — "
                "routing drift; falling through.",
                fp.node_id,
                rec.name,
                route,
            )
        if identity in sched_idx:
            return tile, float(sched_idx[identity][0].emmy_us or 0.0), None
        return None
    root = fp.root_op
    if not isinstance(root, TileOp) or root.op is None:
        return None
    identity = deploy_identity(root)
    recs = sched_idx.get(identity)
    node_blocked = blocked.get(fp.node_id) if blocked else None
    if not recs:
        if _AUDIT_SINK is not None:  # the row count is the audit's alone — never paid on a deploy
            _audit_record(fp.node_id, identity, "GAP", None, None, len(_live_leaf_rows(leaves, node_blocked)))
        return None
    live = _live_leaf_rows(leaves, node_blocked)
    # Both sides normalize through the ONE schedule-row identity (``schedule_row_key``: the
    # recording canonicalizer restricted to what THIS fork decides) — equality after it is exact
    # realized identity, never a prefix or any-of acceptance.
    by_key = {schedule_row_key(k): (o, k) for o, k in live}
    # Per-entry realizability, computed only under an active audit sink: the ``eval golden`` offer
    # audit reads which records the enumeration no longer offers. The deploy below still stops at
    # the first record whose row is offered.
    unrealized = None if _AUDIT_SINK is None else [g for g in recs if schedule_row_key(g.knobs) not in by_key]
    for rec in recs:
        hit = by_key.get(schedule_row_key(rec.knobs))
        if hit is not None:
            _audit_record(fp.node_id, identity, "MATCH", rec.name, float(rec.emmy_us or 0.0), len(live), unrealized=unrealized)
            return hit[0], float(rec.emmy_us or 0.0), dict(hit[1])
    _audit_record(fp.node_id, identity, "DRIFT", ", ".join(g.name for g in recs), None, len(live), unrealized=unrealized)
    logger.warning(
        "deploy: node %r matches %d recorded golden(s) by structural identity, but none equals an enumerated row — "
        "the recording no longer realizes under the current enumeration (drift); falling through to measured "
        "evidence / the prior. Records: %s",
        fp.node_id,
        len(recs),
        ", ".join(g.name for g in recs),
    )
    return None


def greedy_decide(
    blocked: dict[str, set[frozenset]] | None = None,
    *,
    prior: object = _LOAD_PRIOR,
    price_structural: bool = True,
    db: object | None = None,
) -> Callable[[ForkPoint], object]:
    """The greedy compile pick as a :meth:`Run.resolve` ``decide`` callback:
    flatten the fork point to its complete leaves (:func:`flatten_leaves`),
    skip ``blocked`` tile identities, and take the prior's ``mean_scores``
    argmin — the ``OnlinePrior`` once trained, the ``OfflinePrior``
    cold-start heuristic otherwise (both behind ``load_prior``'s
    ``FallbackPrior``). With no prior at all (a failed load, or the explicit
    ``prior=None`` emission-order resolve) every fork falls to emission order
    (option-0, first leaf). Stamps the pick's predicted µs on
    ``fp.score``, so the resolve trace carries the per-fork price (the
    structural pricing probe reads a kernel's cost off the partition fork's
    trace entry).

    ``blocked`` (``{node_id: {tile_identity, ...}}``) lists tiles that failed
    ``validate(ctx)`` on a previous compile attempt — ``Pipeline.run`` retries
    the deterministic resolution with the failed leaf blocklisted so the next
    best non-blocked leaf is picked (the analogue of how ``tune``
    benches-and-skips an unviable tile; greedy benches nothing, so the
    validity signal must come from the retry).

    Structural (``Graph``-splicing) options are priced against the fused side
    with the same evidence — :func:`_priced_pick` — because a ``Graph`` leaf
    carries no knob row the ordinary ranking could score; when a leaf cannot
    be priced, all of them go on to that ranking anyway. Nothing withholds a
    structural leaf to keep a kernel set unchanged.
    ``price_structural=False`` withdraws the splices for reasons that are not
    about speed — ``Pipeline.run``'s retry after a structural pick failed to
    LOWER, and the nested pricing probes, which must not re-split the slice
    they are pricing. The price memo is per-factory-call (one compile
    attempt), keyed by ``Op.cache_key``."""
    from emmy.compiler.pipeline.pipeline import _is_structural_option  # noqa: PLC0415

    memo: dict[str, float | None] = {}  # Op.cache_key → predicted µs (None = unpriceable)
    #: The DECISION memo (:func:`_decision_key` → the winning row + its price) — same lifetime as
    #: the price memo, one compile attempt. A repeat offer replays by :func:`_find_decided_leaf`
    #: descent; only genuinely new (key, blocklist) states pay the flatten-and-score.
    decisions: dict[tuple, tuple[dict, float | None]] = {}
    loaded = prior is not _LOAD_PRIOR
    the_prior = prior if loaded else None
    # Lazily-built per-compile DB evidence index (needs a fork point's ctx for the
    # context keys); ``None`` sentinel = not built yet, ``{}`` = built and empty.
    db_state: list = [None]
    # Lazily-built per-compile verified-golden identity index — same sentinel convention.
    verified_state: list = [None]

    def db_index() -> dict:
        return db_state[0] or {}

    def decide(fp: ForkPoint) -> object:
        nonlocal loaded, the_prior
        if db is not None and db_state[0] is None:
            db_state[0] = _db_measured_index(db, fp.ctx)
        if not loaded:
            loaded = True
            the_prior = _load_prior_safe()
        dkey = _decision_key(fp, blocked)
        if dkey is not None and dkey in decisions:
            want, price = decisions[dkey]
            found = _find_decided_leaf(fp.options, want)
            if found is not None:
                fp.score = price
                return found
        # The VERIFIED tier: the card's recorded goldens, joined by strict structural identity
        # and decoded by exact spelled-row equality. Needs no prior, and applies only in the
        # deployable regime — a recorded µs is -O3 truth and must never arbitrate an -O1 compile.
        from emmy.compiler.pipeline.search.prior.base import _O3_OPT  # noqa: PLC0415

        if float(fp.ctx.features().get("H_opt", _O3_OPT)) == _O3_OPT:
            if verified_state[0] is None:
                verified_state[0] = _verified_index(fp.ctx)
            sched_idx, routing_idx = verified_state[0]
            # An empty index still consults under an audit sink: "this card records nothing for
            # any fork" is the audit's all-GAP coverage answer, not silence.
            if sched_idx or routing_idx or _AUDIT_SINK is not None:
                got_verified = _verified_pick(fp, sched_idx, routing_idx, blocked)
                if got_verified is not None:
                    leaf, price, row = got_verified
                    fp.score = price
                    if dkey is not None and row is not None:
                        decisions[dkey] = (row, price)
                    return leaf
        if the_prior is None:
            # No prior on this resolve — a failed ``load_prior`` (corrupt/unreadable
            # checkpoint) or ``Pipeline.run``'s explicit emission-order fallback
            # (``prior=None``): emission order (option-0, first leaf).
            return _first_leaf(fp.options[0])
        # Flatten: greedy benches nothing, so it must pick the globally best
        # COMPLETE tile, not a partial branch — see ``flatten_leaves`` (the
        # prior is blind at a partial ``BM/BN`` branch: ``knob_features``
        # can't compute the tile's area / occupancy until ``FM/FN`` exist).
        # The pick equals scoring the flat candidate set, invariant to how
        # the lazy tree's levels are arranged.
        leaves = flatten_leaves(fp.options)
        # Structural options (Graph splices that change the kernel set): the
        # per-op prior prices ONE kernel's knob row, so its score for a
        # multi-kernel Graph option is meaningless. :func:`_priced_pick` asks
        # the same evidence about them properly — Σ of nested per-kernel
        # bests, every leaf priced the same way — and returns the argmin. When
        # it cannot price some leaf it decides nothing and every leaf, the
        # structural ones included, goes on to the ordinary ranking below.
        # ``tune`` explores them regardless (MCTS walks every sibling); an env
        # pin makes the Graph the rule's only option, which applies inline and
        # never reaches a decide.
        if any(_is_structural_option(o) for o in leaves):
            if not price_structural:
                # Structural RETIREMENT, not a ranking rule: a fragment kernel
                # that failed to lower cannot be blocklisted at the fork site
                # (the splice minted fresh node ids), so ``Pipeline.run``
                # re-resolves with the splices withdrawn — the same role
                # ``blocked`` plays for a tile. It is also what stops a nested
                # price probe from re-splitting the slice it is pricing.
                leaves = [o for o in leaves if not _is_structural_option(o)] or leaves
            else:
                pick = _priced_pick(fp, leaves, the_prior, memo, db)
                if pick is not None:
                    return pick
        if len(leaves) <= 1:
            return leaves[0] if leaves else _first_leaf(fp.options[0])
        # The constant base under this fork's deltas: the offer op's knobs
        # (its ``S_*`` structural identity) plus the ``H_*`` host/hardware
        # regime — the feature base tune trained on (``two_level.inner_reward``).
        base = {**fp.ctx.features(), **dict(fp.root_op.knobs)}
        # Tiles this node already failed to lower on an earlier attempt — skip
        # the matching leaf so greedy falls back to the next prior-ranked one.
        node_blocked = blocked.get(fp.node_id) if blocked else None
        live = [(o, _leaf_knobs(o)) for o in leaves]
        if node_blocked is not None:
            live = [(o, k) for o, k in live if not _tile_blocked(k, node_blocked)]
        if not live:  # every leaf blocklisted → no valid alternative left
            return leaves[0]
        rows = [{**base, **k} for _, k in live]
        # The deploy evidence hierarchy, top first: (1) measured -O3 reservoir
        # evidence (``Prior.evidence_pick`` — deployable-regime truth); (2) the
        # tune DB's measured best on an exact ``S_*`` match (a config the tune
        # measured must not lose the deploy to an unmeasured extrapolation —
        # eighth-sweep finding 2); (3) the model argmin only when no candidate
        # has evidence at all. An env pin overrides everything upstream of the
        # fork (a pinned family never reaches a decide).
        picker = getattr(the_prior, "pick", None)
        if picker is not None:
            ev = getattr(the_prior, "evidence_pick", None)
            got = ev(rows) if ev is not None else None
            if got is None and db_index():
                got = _db_measured_pick(db_index(), rows)
                if got is None:
                    _warn_disjoint_evidence(db_index(), rows, fp.node_id)
            best_i, price = got if got is not None else picker(rows)
        else:  # bare-mean_scores prior object (tests / custom callers)
            from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

            s = the_prior.mean_scores(rows)
            best_i = min(range(len(rows)), key=lambda i: (s[i], canonical_row_key(rows[i])))
            price = s[best_i]
        fp.score = price  # measured µs when evidence decided, predicted µs otherwise
        if dkey is not None:
            decisions[dkey] = (dict(live[best_i][1]), price)
        return live[best_i][0]

    return decide
