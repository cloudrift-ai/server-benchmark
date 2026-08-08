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
emitted sibling (option-0).
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
    resolution cost). Best-effort: any load failure → ``None`` → recorded goldens
    still decide the forks they match, the rest falls to emission order — a
    bad/missing prior must never break compile."""
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


def _pool_group(fp: ForkPoint) -> list | None:
    """The schedule fork's enumerated rows, read off the lazy tree's ROOT branch — the pool
    already materialized them, so the golden tier can consult candidates without building one
    leaf. ``None`` for any other option shape (several top options, a leaf fork, a concrete op —
    the flash arm's eagerly-built ``TileOp`` options among them): the caller keeps the flatten
    path."""
    if len(fp.options) != 1:
        return None
    root = fp.options[0]
    if not isinstance(root, Fork) or root.is_leaf:
        return None
    group = getattr(root, "group", None)
    return list(group) if group else None


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


def _pick_structural(
    fp: ForkPoint, leaves: list, prior, memo: dict[str, float | None], price_structural: bool, db: object | None = None
) -> object | None:
    """Price the structural (``Graph``-splicing) leaves of one fork against
    the keep-fused ``Op`` side and return the winning structural leaf, or
    ``None`` to keep the op-variant path (cold prior, unpriceable option, or
    fused priced faster).

    Both sides are priced the same way: the best µs at each kernel's partition
    fork, obtained by a nested deterministic resolution of the kernel's
    single-node slice (``lowering/tile`` only, no backend, CPU-only —
    :func:`_price_kernel`); a structural option's price is the Σ over its
    fragment's kernels. The nested pick follows the deploy evidence hierarchy
    (``db`` threads the tune DB down), so each side's price is a *measurement*
    wherever the tune benched that kernel, and a structure the tune measured
    slower cannot displace a measured-faster fused config on a model
    extrapolation alone — a Σ-of-predictions comparison across two different
    kernel families is exposed to the model's absolute-µs error, which doesn't
    cancel across sides the way it does among siblings of one fork. Gated on
    the *trusted* ``OnlinePrior`` (``prior.trustworthy``
    — trained AND passing the reservoir calibration gate): Σ-comparisons
    through the offline cold-start model are unvalidated, and neither a cold
    compile nor a mis-calibrated model may change kernel sets."""
    from emmy.compiler.pipeline.pipeline import _is_structural_option  # noqa: PLC0415

    if not price_structural or prior is None or not getattr(prior, "trustworthy", False):
        return None
    op_leaves = [o for o in leaves if not _is_structural_option(o)]
    if not op_leaves:
        return None  # nothing to compare against — the no-op-variant edge keeps today's scoring path
    fused_prices = [_price_op_leaf(fp, o, prior, memo, db) for o in op_leaves]
    if any(p is None for p in fused_prices):
        return None
    split_prices = [(o, _price_graph(_leaf_graph(o), fp.ctx, prior, memo, db)) for o in leaves if _is_structural_option(o)]
    split_prices = [(o, p) for o, p in split_prices if p is not None]
    if not split_prices:
        return None
    best_split, best_split_us = min(split_prices, key=lambda op_us: op_us[1])
    return best_split if best_split_us < min(fused_prices) else None


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


def _golden_evidence_index(ctx: Context) -> dict:
    """The deploy card's recorded goldens — every kind: matmul, attention (flash), rms_norm, softmax,
    reduce, pointwise, norm_linear / mlp_geglu (the fused RMSNorm→linear / gate⊗up computed-A), and the
    rope / embedding entries kept only as regression references (they fork nothing, so they never match here) — grouped by
    :class:`~emmy.compiler.pipeline.search.data.shape.ShapeKey` (whose ``kind``
    discriminator keeps the sweep kinds apart from extent-coincident contractions)
    and sorted fastest-first — the verified-evidence tier a greedy compile consults
    before the reservoir / DB tiers. Scoped to the ctx's ``(gpu_name, compute_cap)``
    exactly like the live-GPU golden scoping: no card identity (off-GPU
    pure-logic runs) or an unseeded card ⇒ empty index ⇒ no consultation.
    Golden files ship with the repo, so this is the only evidence tier that
    exists on a fresh machine (the reservoir and tune DB are machine-local
    caches written by local tunes). Goldens are consulted, never inserted into
    the reservoir or the online prior's training data. Best-effort: any load
    failure returns an empty index (deploys fall back to the normal hierarchy)."""
    from emmy.compiler.pipeline.search.golden import GOLDEN_CONFIGS  # noqa: PLC0415

    gpu_name = getattr(ctx, "gpu_name", None)
    if not gpu_name:
        return {}
    index: dict = {}
    try:
        cap = tuple(ctx.compute_capability)
        for g in GOLDEN_CONFIGS:
            if g.gpu_name != gpu_name or tuple(g.compute_cap) != cap:
                continue
            if g.is_routing:
                continue  # a ROUTING entry (PLACE-only) decides cuts pre-schedule — never a fork row
            index.setdefault(g.shape_key(), []).append(g)
        for entries in index.values():
            entries.sort(key=lambda g: g.emmy_us or float("inf"))  # unmeasured entries rank last
    except Exception:  # noqa: BLE001 — a golden consult failure must never break compile
        return {}
    return index


def _golden_matches_row(golden_knobs: dict, row: dict) -> bool:
    """Prefix-consistency of a golden's recorded tuning knobs against one offered
    candidate row. Keys compare through
    :func:`~emmy.compiler.pipeline.knob.pin_key_matches` (a golden key recorded
    without an axis suffix matches the axis-stamped realization) and values through
    :func:`~emmy.compiler.pipeline.knob.values_equal` (registry-canonical, so a
    TILE value written with an alias atom name matches the canonically-stamped
    row). A family the candidate hasn't decided at this fork is free — a later
    pass decides it (the ``evidence_pick`` convention: undecided knobs are free).

    An AXIS-KEYED golden key must be satisfied by every candidate key it names — the
    flash all-or-nothing pin contract: a static attention golden records ``TILE@dd``
    AND ``TILE@pj``, and a row matching one but not the other is a different form. A
    BARE golden key on a multi-axis family mirrors the PIN-RESOLUTION semantics
    instead: it names ONE plan the kernel realizes across its axes (a dynamic
    attention golden is schema-required to record a single bare ``TILE`` — the
    dd-plan or, fast-math, the sibling PV plan), so it is satisfied when ANY
    same-family realization equals it. For single-axis families (every matmul row)
    any-of and all-of coincide, so matmul matching is unchanged."""
    from emmy.compiler.pipeline.knob import family_of, pin_key_matches, values_equal  # noqa: PLC0415

    for gk, gv in golden_knobs.items():
        fam = family_of(gk)
        hits = [(rk, rv) for rk, rv in row.items() if not rk.startswith(("S_", "H_")) and family_of(rk) == fam and pin_key_matches(gk, rk)]
        if not hits:
            continue  # family not decided at this fork — free
        matched = [values_equal(rk, gv, rv) for rk, rv in hits]
        if "@" in gk:  # axis-keyed: names exactly one realization — all-or-nothing
            if not all(matched):
                return False
        elif not any(matched):  # bare: one plan, satisfied by any same-family realization
            return False
    return True


def _fork_shape_key(rows: list[dict], base: dict | None = None):
    """The deploy-time :class:`ShapeKey` of a fork's candidate rows. ``base`` carries the shared
    ``S_*`` stamps when ``rows`` are the pool's RAW rows (the golden probe — the stamps live on
    the offer op, not in a pool row); merged rows carry them on ``rows[0]``. The base case is
    ``from_s_features`` over the shared ``S_*`` stamps — but two restructured-op forks stamp a
    histogram the classifier can't kind, so each is rebuilt from an OFFER / dtype signature the
    stamped final op would carry.

    FLASH: the tile pass's RESTRUCTURED twisted op carries re-derived extents ONLY (no
    ``S_loop_depth``, no op histogram — measured off a live greedy resolve), so the classifier can
    never mark it ``kind="flash"`` there. It is instead unmistakable from its OFFER: only the twisted
    lowering forks the two contractions as ``TILE@dd`` + ``TILE@pj``. When the rows carry that pair,
    the key is rebuilt flash-kinded, with the masked (dynamic) form's reduce extent normalized to the stamped
    final-op convention (the fork-time masked op still shows head_dim as a visible reduce; the golden
    keys — and the diagnostics/A-B joins over final stamped ops — have every reduce axis
    symbolic-excluded, ``reduce_max=0``).

    COMPUTED-A CONE (norm→linear / gate⊗up): the fork op is the fused megakernel evaluated on its
    PRE-SPLIT geometry — the RMSNorm statistic reduce has not yet lifted to a second axis
    (``S_ext_n_reduce_axis == 1``) and the rsqrt lives in the nested A-cone sub-body, so the
    histogram can't fire ``kind="fused"`` (which needs ``>= 2`` and a top-level ``S_pw_rsqrt``).
    Like flash, it is unmistakable from its OFFER: only a computed-A contraction enumerates the
    mandatory ``sync`` compute-fill (``d*/sync`` STAGE rows — ``space.stage_moves`` offers only
    gmem-direct / ``cp`` / ``tma``, and the ``sync`` transport is born in the computed-A resolvers
    alone), so a fork whose rows carry a sync-STAGE candidate IS the cone form. (The earlier
    mixed-dtype sniff — f16/bf16 operands + an f32 statistic constant over an add-reduce — fired on
    ANY plain-nest kernel with that stamp combination, e.g. a genuine mixed-dtype plain matmul,
    silently re-keying it out of its ``kind=""`` goldens; the offer signal cannot, and it also
    covers the all-f16 stat-free cone the dtype sniff missed.) Rebuild to the fused convention:
    ``is_warp=True`` (a computed-A contraction is a warp mma) and ``kind="fused"``; ``free_max``
    carries the stamped aspect (both key builders keep it for the fused kind — see the aspect note
    below); ``reduce_max`` stays the contraction extent (the fused goldens key on it even when
    dynamic — unlike the flash reduce).

    A DECODED B (``dtype_class == "trellis"``) is excluded from the cone rebuild: the sync-STAGE
    offer signal is no longer unique to a computed-A cone now that the trellis compute fill spells
    the same transport, and a trellis contraction is a plain two-free-axis nest with a computed B,
    not a fused megakernel. Left un-excluded, every prefill-M coded matmul re-keyed to
    ``kind="fused"`` AND lost its storage class — a key that collides with a real RMSNorm→linear
    cone of equal extents and can never join a coded shape's own golden. Both rebuilds carry
    ``dtype_class`` and the coded ``k_bits`` rate through for the same reason — and carry them
    STRUCTURALLY, by stating only what changes (``dataclasses.replace``) rather than re-listing
    the fields, so the next key dimension cannot be dropped here by omission."""
    from emmy.compiler.pipeline.search.data.shape import ShapeKey  # noqa: PLC0415

    stamps = base if base is not None else rows[0]
    key = ShapeKey.from_s_features(stamps)
    # ANY row carrying the pair marks the fork (like the cone's sync-STAGE scan below):
    # the pair may appear only on the warp leaves, and row 0 — whichever leaf the
    # planner emitted first — need not be one of them.
    if any({"dd", "pj"} <= {k.split("@", 1)[-1] for k in row if k.startswith("TILE@")} for row in rows):
        # ``replace``, not a fresh ``ShapeKey(...)``: a rebuild states only what it CHANGES, so a
        # dimension added to the key rides through instead of silently resetting to its default —
        # which would merge this fork with every other key that differs only there. (``free_max``
        # needs no mention: ``__post_init__`` normalizes it to 0 off the ``flash`` kind.)
        key = replace(key, reduce_max=0 if key.is_dyn else key.reduce_max, kind="flash")
    elif (
        key.kind == ""
        # A computed-B decode fill also spells ``sync``; see the docstring.
        and key.dtype_class != "trellis"
        # A computed-A cone CONTRACTS: its output is 2-D ``(M, N)``. A standalone RMSNorm
        # STATISTIC kernel produces ONE value per row — without this the cut's own ``__stat``
        # producer was rebuilt to ``kind="fused"``, which both locked it out of the plain
        # reduce goldens and left it able to shadow a real cone of equal extents.
        and stamps.get("S_ext_n_free_axis", 0) >= 2
        # The OFFER signal (exact, like flash's TILE@dd+TILE@pj): a ``d*/sync`` STAGE row
        # exists only on a computed-A contraction fork. Segment-match so ``cp.async``'s
        # substring can never false-positive.
        and any("sync" in str(v).split("/") for row in rows for k, v in row.items() if k.startswith("STAGE"))
    ):
        # ``free_max`` carries through: the pre-split key was built ``kind=""``, which
        # preserves the stamped aspect, and the fused kind keeps it (a computed-A cone is a
        # plain two-free-axis ``(M, H) @ (H, N)``). Dropping it here collapsed the M=256
        # global norm→kv cone onto the M=32 local norm→q golden — equal free_prod (131072)
        # and reduce (3840) — deploying the wrong config at a fabricated µs. Carrying it (and
        # every other dimension) is what ``replace`` guarantees structurally.
        key = replace(key, is_warp=True, kind="fused")
    return key


# Optional per-consultation verdict sink for the golden-tier audit (``search/audit.py``).
# ``None`` (the default) is zero-cost; ``golden_audit`` installs a list that
# ``_golden_pick`` appends one record per consulted fork to — the supported hook the
# drift audit reads, replacing the old monkey-patch interception in ``scripts/diagnostics``.
_AUDIT_SINK: list[dict] | None = None


@contextmanager
def golden_audit(records: list[dict]):
    """Collect one ``{node, key, verdict, golden, us, n_rows, unrealized}`` record per
    golden-tier consultation into ``records`` for the duration of the block. Verdicts:
    ``MATCH`` (a recorded golden realized on an offered candidate), ``DRIFT`` (goldens
    keyed to the fork's shape but none realizes — a graph/enumeration change invalidated
    them), ``GAP`` (no golden recorded for the shape). ``unrealized`` (MATCH/DRIFT only;
    ``None`` on GAP) lists the shape's recorded :class:`GoldenConfig` entries that NO
    offered candidate realizes — the per-entry "pin-only" signal the ``eval golden``
    offer audit reads (an entry can be individually unrealizable while a sibling still
    MATCHes and floors the deploy)."""
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
        # ``key`` stays the ShapeKey object — coverage policy (major-gap classification)
        # reads its fields; presentation layers stringify at print/JSON time.
        _AUDIT_SINK.append(
            {"node": node_id, "key": key, "verdict": verdict, "golden": golden, "us": us, "n_rows": n_rows, "unrealized": unrealized}
        )


def _golden_pick(index: dict, rows: list[dict], node_id: str, base: dict | None = None) -> tuple[int, float] | None:
    """Verified-golden pick over candidate knob rows: the first candidate
    prefix-consistent with the fastest recorded golden of the op's shape
    (:class:`ShapeKey` off the shared ``S_*`` base). Sits ABOVE the reservoir /
    DB evidence tiers — a golden is an A/B-verified, integrity-gated, reproduced
    deployable measurement; a reservoir row is a single tune sample. Applies only
    in the deployable regime (mirroring ``Prior.evidence_pick``'s guard): the
    recorded µs is -O3 truth and must never arbitrate an -O1 compile. Among a
    shape's entries (std + fm + parity alternates) the fastest one whose config is
    actually offered decides — a fast-math golden self-excludes on a default
    deploy because its atom isn't in the offer when the fm gate is off. A shape
    match with NO realizable golden logs a loud drift warning (the enumeration no
    longer offers what the golden recorded) and falls through to the normal
    hierarchy. Returns ``(candidate_index, recorded_µs)`` or ``None``."""
    from emmy.compiler.pipeline.knob import canonical_row_key, tuning_knob_items  # noqa: PLC0415
    from emmy.compiler.pipeline.search.prior.base import _O3_OPT  # noqa: PLC0415

    if not rows or float((base if base is not None else rows[0]).get("H_opt", _O3_OPT)) != _O3_OPT:
        return None  # deploying a non--O3 regime — golden µs is deployable-regime truth
    key = _fork_shape_key(rows, base=base)
    goldens = index.get(key)
    if not goldens:
        _audit_record(node_id, key, "GAP", None, None, len(rows))
        return None
    # Per-entry realizability, computed only under an active audit sink: the ``eval golden``
    # offer audit reads which recorded entries the un-pinned enumeration never offers
    # (pin-only rows); the deploy hot path below still stops at the first realizing golden.
    unrealized = None
    if _AUDIT_SINK is not None:
        unrealized = [g for g in goldens if not any(_golden_matches_row(dict(tuning_knob_items(g.knobs)), row) for row in rows)]
    for g in goldens:  # fastest recorded entry first
        gold = dict(tuning_knob_items(g.knobs))
        # Several rows can realize one golden (a golden pins a knob PREFIX); the
        # canonically-smallest realization wins, never the first-enumerated.
        matches = [i for i, row in enumerate(rows) if _golden_matches_row(gold, row)]
        if matches:
            best = min(matches, key=lambda i: canonical_row_key(rows[i]))
            _audit_record(node_id, key, "MATCH", g.name, float(g.emmy_us or 0.0), len(rows), unrealized=unrealized)
            return best, float(g.emmy_us or 0.0)
    _audit_record(node_id, key, "DRIFT", ", ".join(g.name for g in goldens), None, len(rows), unrealized=unrealized)
    logger.warning(
        "deploy: node %r matches golden shape %s (%d recorded entr%s), but no offered candidate realizes any of "
        "them — the golden(s) no longer realize under the current enumeration; falling through to the normal "
        "evidence hierarchy. Investigate enumeration drift for: %s",
        node_id,
        goldens[0].shape_key(),
        len(goldens),
        "y" if len(goldens) == 1 else "ies",
        ", ".join(g.name for g in goldens),
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
    ``prior=None`` emission-order resolve) the card's recorded goldens still
    decide the forks they match — golden evidence needs no prior — and only
    a fork without a realizable golden falls to emission order (option-0,
    first leaf). Stamps the pick's predicted µs on
    ``fp.score``, so the resolve trace carries the per-fork price (the
    structural pricing probe reads a kernel's cost off the partition fork's
    trace entry).

    ``blocked`` (``{node_id: {tile_identity, ...}}``) lists tiles that failed
    ``validate(ctx)`` on a previous compile attempt — ``Pipeline.run`` retries
    the deterministic resolution with the failed leaf blocklisted so the next
    best non-blocked leaf is picked (the analogue of how ``tune``
    benches-and-skips an unviable tile; greedy benches nothing, so the
    validity signal must come from the retry).

    Structural (``Graph``-splicing) options are priced with the trained prior
    grounded in measured DB evidence — :func:`_pick_structural` — so an
    unpinned ``compile`` / ``run`` can deploy the kernel sets ``tune`` measured
    best (the demoted-matmul split); cold, the structural leaf is filtered and
    kernel sets stay unchanged.
    ``price_structural=False`` keeps the filter behavior — used by
    ``Pipeline.run``'s retry after a structural pick failed to lower, and by
    the nested pricing probes themselves (no recursive splitting inside a
    price probe). The price memo is per-factory-call (one compile attempt),
    keyed by ``Op.cache_key``."""
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
    # Lazily-built per-compile golden evidence index (needs a fork point's ctx
    # for the card scoping) — same sentinel convention.
    golden_state: list = [None]

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
        # The golden PROBE — the covered path never flattens: a schedule fork's enumerated rows
        # already sit on the lazy tree's root branch (the pool), so the golden tier consults them
        # directly and a MATCH descends to its ONE leaf. The verdict (MATCH / DRIFT / GAP) is
        # recorded here exactly once; the fall-through paths below skip their own golden consult.
        golden_consulted = False
        if dkey is not None:
            if golden_state[0] is None:
                golden_state[0] = _golden_evidence_index(fp.ctx)
            group = _pool_group(fp) if golden_state[0] else None
            if group is not None:
                node_blocked = blocked.get(fp.node_id) if blocked else None
                if node_blocked is not None:
                    group = [r for r in group if not _tile_blocked(r, node_blocked)]
                base = {**fp.ctx.features(), **dict(fp.root_op.knobs)}
                got = _golden_pick(golden_state[0], group, fp.node_id, base=base) if group else None
                golden_consulted = bool(group)
                if got is not None:
                    best_i, price = got
                    found = _find_decided_leaf(fp.options, dict(group[best_i]))
                    if found is not None:
                        if the_prior is None:
                            logger.warning(
                                "deploy: node %r resolved WITHOUT a prior (load failure or emission-order fallback), "
                                "but the golden floor holds — deploying the recorded golden realization (%.1f us) "
                                "over option-0.",
                                fp.node_id,
                                price,
                            )
                        fp.score = price
                        decisions[dkey] = (dict(group[best_i]), price)
                        return found
        if the_prior is None:
            # No prior on this resolve — a failed ``load_prior`` (corrupt/unreadable
            # checkpoint) or ``Pipeline.run``'s explicit emission-order fallback
            # (``prior=None``). The card's recorded goldens are tier (1) of the
            # evidence hierarchy and depend on no prior, so they still apply:
            # a fork whose golden realizes on an offered candidate deploys the
            # golden, and only the rest falls to emission order. (A blocklisted
            # tile — one that failed ``validate(ctx)`` earlier — stays excluded, so
            # this path can never re-pick a tile the retry loop already rejected.)
            if golden_state[0] is None:
                golden_state[0] = _golden_evidence_index(fp.ctx)
            if golden_state[0] and not golden_consulted:
                leaves = [o for o in flatten_leaves(fp.options) if not _is_structural_option(o)]
                live = [(o, _leaf_knobs(o)) for o in leaves]
                node_blocked = blocked.get(fp.node_id) if blocked else None
                if node_blocked is not None:
                    live = [(o, k) for o, k in live if not _tile_blocked(k, node_blocked)]
                base = {**fp.ctx.features(), **dict(fp.root_op.knobs)}
                rows = [{**base, **k} for _, k in live]
                got = _golden_pick(golden_state[0], rows, fp.node_id) if rows else None
                if got is not None:
                    best_i, price = got
                    logger.warning(
                        "deploy: node %r resolved WITHOUT a prior (load failure or emission-order fallback), but the "
                        "golden floor holds — deploying the recorded golden realization (%.1f us) over option-0.",
                        fp.node_id,
                        price,
                    )
                    fp.score = price
                    if dkey is not None:
                        decisions[dkey] = (dict(live[best_i][1]), price)
                    return live[best_i][0]
            return _first_leaf(fp.options[0])  # no realizable golden → emission order
        # Flatten: greedy benches nothing, so it must pick the globally best
        # COMPLETE tile, not a partial branch — see ``flatten_leaves`` (the
        # prior is blind at a partial ``BM/BN`` branch: ``knob_features``
        # can't compute the tile's area / occupancy until ``FM/FN`` exist).
        # The pick equals scoring the flat candidate set, invariant to how
        # the lazy tree's levels are arranged.
        leaves = flatten_leaves(fp.options)
        # Structural options (Graph splices that change the kernel set): the
        # per-op prior prices ONE kernel's knob row, so its score for a
        # multi-kernel Graph option is meaningless. With the *trained* prior
        # loaded, :func:`_pick_structural` prices the option properly — Σ of
        # nested per-kernel predicted-bests vs the keep-fused side — and
        # returns the split when it predicts faster. Cold (offline / no
        # prior), or when an option can't be priced, the structural leaf is
        # filtered so a cold compile never changes kernel sets. ``tune``
        # explores them regardless (MCTS walks every sibling); an env pin
        # makes the Graph the rule's only option, which applies inline and
        # never reaches a decide.
        if any(_is_structural_option(o) for o in leaves):
            pick = _pick_structural(fp, leaves, the_prior, memo, price_structural, db)
            if pick is not None:
                return pick
            op_leaves = [o for o in leaves if not _is_structural_option(o)]
            if op_leaves:
                leaves = op_leaves
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
        # The deploy evidence hierarchy, top first: (1) the card's recorded
        # GOLDENS — A/B-verified deployable measurements that ship with the
        # repo, the only evidence a fresh machine has (consulted, never
        # trained on); (2) measured -O3 reservoir evidence
        # (``Prior.evidence_pick`` — deployable-regime truth); (3) the tune
        # DB's measured best on an exact ``S_*`` match (a config the tune
        # measured must not lose the deploy to an unmeasured extrapolation —
        # eighth-sweep finding 2); (4) the model argmin only when no
        # candidate has evidence at all.
        if golden_state[0] is None:
            golden_state[0] = _golden_evidence_index(fp.ctx)
        got = _golden_pick(golden_state[0], rows, fp.node_id) if golden_state[0] and not golden_consulted else None

        picker = getattr(the_prior, "pick", None)
        if picker is not None:
            ev = getattr(the_prior, "evidence_pick", None)
            if got is None and ev is not None:
                got = ev(rows)
            if got is None and db_index():
                got = _db_measured_pick(db_index(), rows)
                if got is None:
                    _warn_disjoint_evidence(db_index(), rows, fp.node_id)
            best_i, price = got if got is not None else picker(rows)
        elif got is not None:  # golden decides even for bare-mean_scores priors
            best_i, price = got
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
