"""The greedy compile pick — :func:`greedy_decide`, a ``Run.resolve`` decide
factory choosing one **complete** leaf via direct evidence or the global online
prior, else option-0.

This is the deterministic pick for ``compile`` / ``run``, the structural
pricing probes, and the assembled-graph lowering. It is NOT a search and not
a ``Search`` policy: there is no frontier to rank, no tree, no benching — a
deterministic resolution is a fold over the pipeline (at each fork, a pure
function of ``(options, op, prior)``, argmin, continue), so its process state
is :meth:`Run.resolve`'s returned trace, never accumulated policy attributes.
It can only *use* a prior trained earlier by ``tune``, never train one.
Exploration stays in :class:`~.mcts.TuningSearch` (``Pipeline.tune``).

**Evaluate complete rows.** A branch carries only a partial schedule, so a prior cannot score it as
though it were a complete row. Measured rows descend to their exact spelling;
otherwise greedy scores the complete offered rows and chooses the argmin — streamed off the
lazy walk in bounded chunks (:func:`_stream_tiers`), so the scan is O(chunk) memory however large
the pool, and bounded in descent work by the cold-pool budget except when one complete path itself
has a larger declared bound: that pool gets exactly one descent attempt. A pool whose minted size
bound exceeds :data:`_POOL_BUDGET` is ranked over a deterministic drawn subset of its complete rows
instead of walked at full length. Sampling complete rows is not branch substitution — no branch is
ever scored as a stand-in for the schedules it contains — and the argmin is global again the moment
evidence exists, because measured rows descend directly whatever the pool size.

**Greedy is ranked by evidence and by nothing else.** Measured rows first —
the reservoir, then ONE index holding the tune DB's rows and the golden rows in
scope, compared on µs alone — then the fitted prior; every measured row is a
recording of something that ran. There is no hand-written step: no leaf is
promoted, demoted, withheld or given a head start here, and no fallback
default is chosen for being safe. Where nothing measured and no prior speaks
the pick degenerates to the enumeration's first leaf, which carries no meaning
and can be arbitrarily slow. That is the accepted cost of the rule, not a
defect to patch: a bad unmeasured pick is fixed by measuring (a tune, a benched
golden row) or by fitting the prior better, never by teaching this module a
preference — or refused outright under strict evidence
(:func:`_require_evidence`), which raises :class:`EvidenceError` for a fork
no measurement decides.

**Kernel-set forks follow the same rule.** Every measured row of the kernel's
signature spells one offered arm (:func:`~emmy.compiler.pipeline.search.pins.spelled_arm`): a
route row (:func:`_is_route_row`) the cut or split it records, a schedule row the fused /
unsplit arm — the kernel it decorates ran that way. :func:`_route_candidates` turns them into
candidates priced at the row's µs, nothing installed on the kernel, and a measured candidate
outranks every arm priced by nested resolution. Only with no measured arm are the arms priced
against each other. The pieces an arm mints are brand-new kernels, decided at their own forks
from rows of their own signatures.

**A measurement can also DISQUALIFY.** The measured sources above all RANK, and a
ranking needs a latency — which a ``bench_fail`` row does not have, only the
watchdog's timeout sentinel. Those rows are still a recording of something that
ran (or failed to), so they are read, but as an elimination rather than a score:
where every measured variant of one structural signature failed, a slice
containing that kernel prices ``inf`` (:func:`_resolved_price`) and any
structural arm holding it loses the kernel-set argmin. Still evidence, still no
preference — the alternative is that an all-failed kernel has no ``ok`` row,
therefore no evidence at all, and falls through to the prior as though nothing
were known about it. That is how DeepSeek-V4's post block kept a fused arm whose
every benched variant hung.

**One physical bound, and it is not a preference.** The kernel-set Σ
(:func:`_resolved_price`) clamps a summand to its serial-work lower bound where
that bound is decisively large (:data:`_SERIAL_FLOOR_ENFORCE_US`). This is the
one exception to "no hand-written tier", and it is an exception in the same
sense the disqualification is: not a ranking opinion but a fact no measurement
can overrule — no thread retires 2^30 dependent-nest trips in microseconds, and
no measurement can even EXIST at such magnitudes (the bench watchdog fires
first), so "fix it by measuring" is unavailable exactly where the bound binds.
A measured µs is never below the bound, so evidence always wins where evidence
can exist.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

from emmy.compiler.graph import Graph
from emmy.compiler.pipeline.fork import Fork, flatten_leaves, fork_signature, iter_leaves, leaf_for, leaf_knobs
from emmy.compiler.pipeline.knob import schedule_pin_fingerprint
from emmy.compiler.pipeline.search.features import serial_floor_us

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


def _find_decided_leaf(options: list, want: dict) -> object | None:
    """The leaf carrying exactly the memoized row ``want`` — replayed by DESCENDING the lazy fork
    tree: a branch expands only when every knob it pins matches the row, so the walk instantiates
    O(path × siblings) Forks, never the flat leaf set. ``None`` when no leaf matches — emission
    drift between two offers of one key — and the caller re-decides."""
    for o in options:
        if isinstance(o, Fork) and not o.is_leaf:
            if o.admits(want):
                found = _find_decided_leaf(o.expand(), want)
                if found is not None:
                    return found
        elif leaf_knobs(o) == want:
            return o
    return None


def _leaf_op(leaf: object):
    """The concrete ``Op`` behind a leaf, or ``None``. A schedule-tree leaf exposes its concrete
    option directly; a deferred non-structural leaf is materialized only when asked for."""
    from emmy.compiler.ir.base import Op  # noqa: PLC0415

    if isinstance(leaf, Op):
        return leaf
    option = getattr(leaf, "option", None)
    if option is None and isinstance(leaf, Fork) and leaf.is_leaf and not leaf.structural:
        option = leaf.expand()[0]
    return option if isinstance(option, Op) else None


def _leaf_graph(leaf: object) -> Graph:
    """The ``Graph`` behind a raw, concrete, or deferred structural leaf."""
    if isinstance(leaf, Graph):
        return leaf
    option = getattr(leaf, "option", None)
    return option if option is not None else leaf.expand()[0]


def _decision_key(fp: ForkPoint, blocked: dict | None) -> tuple | None:
    """The decision memo's key for one schedule fork, or ``None`` where the memo does not apply.

    GREEDY-ONLY and scoped to one factory call (one compile attempt), because a decision is a
    CONCLUSION over evidence — MCTS must explore, and evidence may move between attempts. Within
    one attempt the pick is deterministic, so N same-shape kernels — 28 identical per-layer
    matmuls — decide once and the rest replay by tree descent instead of a flatten-and-score.

    ``TileOp``-rooted forks only, keyed on the enumeration's MINTED pool identity
    (:attr:`~emmy.compiler.pipeline.fork.Fork.pool_id` — the enumeration's minted stamp: the
    variant key + hints + pins + the sample identity). One minting site, one spelling; the memo
    fails safe on anything the stamp cannot see, because a replayed row that no longer decodes
    (``_find_decided_leaf`` → ``None``) simply re-decides. A fork carrying no stamp
    (offered outside the schedule enumeration) falls back to the kernel's variant key
    (``identity_key`` with io + knobs) + pins. The rule identity separates two
    forks offered on one op, and the node's blocklist CONTENT keys the validate-retry path — a
    retry with a blocked tile is a different decision."""
    from emmy.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

    if not isinstance(fp.root_op, TileOp):
        return None
    pid = next((p for o in fp.variants if (p := getattr(o, "pool_id", None)) is not None), None)
    rule = fp.match.rule
    node_blocked = blocked.get(fp.node_id) if blocked else None
    return (
        getattr(getattr(rule, "pass_", None), "name", None),
        getattr(rule, "name", None),
        pid if pid is not None else (fp.root_op.identity_key(with_io=True, with_knobs=True), schedule_pin_fingerprint()),
        frozenset(node_blocked) if node_blocked else frozenset(),
    )


#: Enforcement guard for the serial-work bound (µs): the clamp in :func:`_resolved_price` applies
#: only to a kernel whose bound exceeds this — 1 ms, three orders above launch overhead and three
#: below the bench watchdog. Below it the bound sits inside the range launch overhead and memory
#: traffic legitimately dominate, and the model's ranking (however uncalibrated) must stand;
#: above it per-thread serial work alone makes the kernel un-servable and nothing may price it
#: lower. The lower edge is pinned by measurement: the largest legitimate ``serial_floor_us``
#: across the qwen3emb realization corpus family is **6.55 µs** (``sdpa-s512``'s fused kernel,
#: 2^16 serial trips — a shape whose fused election is correct and pinned by its ``realized``
#: replay), so 1 ms stands ~150x above the biggest bound the guard must ignore.
_SERIAL_FLOOR_ENFORCE_US = 1e3


def _resolved_price(terminal: Graph, trace: list, ctx: Context, prior, failed: dict | None = None) -> float | None:
    """Σ over a resolved slice's kernels of each one's estimated µs — the ONE cost rule.

    ``failed`` is the measured DISQUALIFICATION (:class:`_Measured`): the structural signatures
    whose every benched variant failed. A slice containing one prices ``inf``, so a structural arm
    holding a kernel the tune watched hang loses the argmin to any arm that does not. That is not
    a preference, it is the measurement — and without it those rows are invisible at deploy,
    because the ranking index carries ``ok`` rows only, so an all-failed kernel has NO evidence
    and falls through to the prior.

    Per kernel: the price the resolution's own fork stamped (the winning leaf's µs, which the
    deploy evidence hierarchy chose), or — where the trace carries no score for it: a decide that
    stamped none (no prior at that fork), or a kernel resolved without a traced fork at all (an
    inline structural replay of an already-decided offer site) — the prior's estimate for the row
    it realized. ``None`` when any surviving kernel can be priced by neither, which hands the
    caller back to the ordinary leaf ranking. (A one-option fork is still traced — every pass
    returns even a forced decision as a fork — so "never forked" is no longer a case here.)

    A slice that a structural fork changed the kernel SET of (a ``PLACE`` cut, a cross-CTA split)
    ends with several kernels, and this Σ is exactly why that needs no special case: the kernel
    those replaced does not run and has no latency of its own, so its estimate IS the sum over the
    kernels it produced.

    The summands are not all the same quantity: a fork a measured row decided contributes a
    measured µs, one the model decided contributes the model's ranking score. Mixing them in a Σ is
    the known cost of comparing kernel SETS with a per-kernel ranker — the exposure the module
    docstring names, and the prior's to fix by being calibrated, not this function's to paper
    over. What this Σ DOES enforce is the physical bound no calibration could relax: a summand
    whose serial-work lower bound (:func:`~..features.serial_floor_us` — the kernel's per-thread
    serial trips at a per-trip time conservative for any GPU clock) exceeds
    :data:`_SERIAL_FLOOR_ENFORCE_US` is clamped to that bound. A measured µs is never below the
    bound, so the clamp only ever lifts model garbage: the cold proxy priced DeepSeek-V4
    ``post4096``'s fused 2^30-trip recomputation nest at 4.29e-37 µs, under every one of its
    recomputation-free composed-cut arms, and no fitted weight can guarantee the bound at
    magnitudes no measurement can reach. The guard is jurisdiction, not tuning: the bound ignores
    launch overhead and memory traffic, so at ordinary magnitudes it must not adjudicate a
    fused-vs-cut µs delta (an ungated draft flipped three qwen3emb sdpa corpus replays to a cut
    election by comparing trip counts alone) — while a bound past the guard is un-servable
    whatever those effects are. The guard's honest escape: a nest under ~2^23 per-thread trips
    (bound < 1 ms; several ms real) is still adjudicated by the uncalibrated proxy, so a
    2^23-class recomputation defect stays electable — smaller than the 2^30 class this bound
    exists for, but not free. Sibling ranking within one kernel is decided upstream and never
    reads this Σ, which is what makes the clamp safe here and NOT on the prior's scoring
    surfaces (there any µs bound collapses live-range deltas — the plateau failure
    ``latency_proxy``'s history warns about)."""
    scored: dict[str, float | None] = {d.node_id: d.score for d in trace}
    total = 0.0
    for nid, node in terminal.nodes.items():
        if node.op.identity_key(with_io=True, with_knobs=True) is None:
            continue
        knobs = getattr(node.op, "knobs", None) or {}
        if failed:
            sig = frozenset((k, str(v)) for k, v in knobs.items() if k.startswith("S_"))
            # The ONE signature rule (:func:`_sig_groups`): a stored signature describes this
            # kernel when the kernel carries every recorded fact — a candidate that only ADDS stamps
            # the featurizer has since gained is the same measured shape (the stamp derives from
            # the same body), or one added ``S_*`` feature would silently disable every
            # disqualification (measured live when ``S_ext_serial_cell_work`` landed). A recorded
            # key the kernel lacks is a different shape: agreeing on merely the SHARED keys
            # condemned all 17 leaves of a fork on DeepSeek-V4's post block and decided nothing.
            # An empty stored signature (an op that stamped nothing) describes nothing.
            if _sig_groups(failed, sig):
                return math.inf
        us = scored.get(nid)
        if us is None:
            rows = [{**ctx.features(), **knobs}]
            us = prior.mean_scores(rows)[0] if prior is not None else None
        if us is None:
            return None
        floor = serial_floor_us(knobs)
        total += max(us, floor) if floor > _SERIAL_FLOOR_ENFORCE_US else us
    return total


def _price_kernel(
    graph: Graph, nid: str, ctx: Context, prior, memo: dict[object, float | None], db: object | None = None, decisions: dict | None = None
) -> float | None:
    """One kernel's price: a nested deterministic resolution of its
    single-node slice through ``lowering/tile`` only (the schedule fork is
    where the prior prices a complete tile row; the kernel/cuda passes add
    nothing and cost real CPU), summed over the kernels that resolution ends
    with (:func:`_resolved_price`). ``db`` rides into the
    nested decide, so each fork's pick follows the same deploy evidence
    hierarchy as a top-level knob pick (reservoir rows, then the tune DB's
    measured rows, model prediction only where nothing was measured) — the
    priced µs is a measurement wherever the tune benched this kernel. Memoized
    per exact variant key (``Op.identity_key(structural=False, with_io=True,
    with_knobs=True)``) so identically computing kernels price once — the key is
    α-invariant, so mirror cut pieces and re-spelled siblings share the memo entry
    (``identity_key(with_io=True, with_knobs=True)`` is the fallback for ops with no body-derived identity).
    Best-effort: any resolve failure prices as ``None`` (→ the caller keeps
    the op-variant path)."""
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415
    from emmy.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415

    op = graph.nodes[nid].op
    key = op.identity_key(structural=False, with_io=True, with_knobs=True) or op.identity_key(with_io=True, with_knobs=True)
    if key in memo:
        return memo[key]
    us: float | None = None
    try:
        nested = greedy_decide(prior=prior, price_structural=False, db=db, decisions=decisions)
        if getattr(ctx, "kernel_cache", None) is not None:
            from dataclasses import replace as _replace  # noqa: PLC0415

            ctx = _replace(ctx, kernel_cache=None)  # a replayed kernel offers no fork to price
        terminal, trace = Run(pipeline=_tile_pipeline(), ctx=ctx).resolve(single_node_graph(graph, nid), nested)
        failed = _db_measured_index(db, ctx).failed if db is not None else None
        us = _resolved_price(terminal, trace, ctx, prior, failed=failed)
    except Exception:  # noqa: BLE001 — a price-probe failure must never break compile
        us = None
    memo[key] = us
    return us


def _price_graph(
    graph: Graph, ctx: Context, prior, memo: dict[object, float | None], db: object | None = None, decisions: dict | None = None
) -> float | None:
    """Σ of per-kernel best-µs prices over ``graph``'s kernel-bearing
    nodes, or ``None`` when any kernel is unpriceable (no partition fork —
    e.g. a pre-tiled combine ``TileOp`` — or a failed nested resolve)."""
    prices = [
        _price_kernel(graph, nid, ctx, prior, memo, db, decisions)
        for nid, n in graph.nodes.items()
        if n.op.identity_key(with_io=True, with_knobs=True) is not None
    ]
    if not prices or any(p is None for p in prices):
        return None
    return sum(prices)


def _price_op_leaf(
    fp: ForkPoint, leaf: object, prior, memo: dict[object, float | None], db: object | None = None, decisions: dict | None = None
) -> float | None:
    """The keep-fused side's price: the leaf's ``Op`` rebound into a
    single-node slice of the current graph, priced like any kernel."""
    from emmy.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415

    option = _leaf_op(leaf)
    if option is None:
        return None
    sub = single_node_graph(fp.match.graph, fp.node_id)
    sub.nodes[fp.node_id].op = option
    return _price_graph(sub, fp.ctx, prior, memo, db, decisions)


def _priced_pick(
    fp: ForkPoint, leaves: list, prior, memo: dict[str, float | None], db: object | None = None, decisions: dict | None = None
) -> object | None:
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
        (
            o,
            _price_graph(_leaf_graph(o), fp.ctx, prior, memo, db, decisions)
            if _is_structural_option(o)
            else _price_op_leaf(fp, o, prior, memo, db, decisions),
        )
        for o in leaves
    ]
    if any(us is None for _, us in priced):
        return None
    return min(priced, key=lambda op_us: op_us[1])[0]


# Process-wide memo for the built DB index, keyed on (db path, mtime, context key).
# The index depends only on the DB file and cc+nvcc-flags (NOT the
# op shape — ``structural_key`` folds neither), so for a serve boot it is identical
# across all ~96 program compiles; without this the 527 MB perf scan reran each time.
# Bounded to the current key (cleared on miss), like ``_load_prior_cached``.
_DB_INDEX_CACHE: dict = {}


def _db_measured_index(db, ctx) -> _Measured:
    """Caching wrapper over :func:`_db_measured_index_build` — memoizes the built index per
    process on ``(db path, mtime, context keys, golden scope)``, invalidated when the DB file's
    mtime or the golden scope changes. An in-memory DB (no ``_path``) or an unstatable file
    bypasses the cache and rebuilds. Best-effort throughout: a failed key computation just
    rebuilds."""
    from emmy.compiler.pipeline.search.golden import scope_token  # noqa: PLC0415

    path = getattr(db, "_path", None)
    if db is not None and path is None:
        return _db_measured_index_build(db, ctx)
    try:
        # Stat the main file AND its ``-wal`` sidecar: in WAL mode a ``record_perf``
        # commit can land in the WAL without bumping the main file's mtime, so a
        # main-mtime-only key could serve a stale index to a same-process
        # write-then-read (the tune lane). ``os.stat`` on a missing WAL → skip it.
        if path is not None:
            wal = path.with_name(path.name + "-wal")
            mtime = (path.stat().st_mtime_ns, wal.stat().st_mtime_ns if wal.exists() else 0)
        else:
            mtime = None
        key = (str(path), mtime, ctx.structural_key(), scope_token())
    except Exception:  # noqa: BLE001 — any key-build failure → just rebuild uncached
        return _db_measured_index_build(db, ctx)
    hit = _DB_INDEX_CACHE.get(key)
    if hit is not None:
        return hit
    index = _db_measured_index_build(db, ctx)
    _DB_INDEX_CACHE.clear()  # keep only the current (path, mtime, keys)
    _DB_INDEX_CACHE[key] = index
    return index


class _Measured(NamedTuple):
    """One evidence scan's answers, because the scan is expensive and all come from the same rows.

    ``ok`` RANKS — the measured schedule rows a pick argmins over, by ``S_*`` signature. ``failed``
    DISQUALIFIES — the signatures whose every measured variant failed, a different kind of answer
    that cannot be expressed as a latency. ``routes`` PRICES A KERNEL SET — rows whose keys spell a
    placement (``PLACE@…``) or a cross-CTA split, each a measured µs for applying that decision to
    the kernel of its signature."""

    ok: dict[frozenset, list[tuple[dict, float]]]
    failed: dict[frozenset, list[float]]
    routes: dict[frozenset, list[tuple[dict, float]]]


_EMPTY_MEASURED = _Measured({}, {}, {})


def _is_route_row(tun: dict) -> bool:
    """Whether a measured row spells a kernel-SET decision beside (or instead of) a schedule: a
    ``PLACE`` key, or a ``REDUCE`` value carrying a cross-CTA ``g<n>`` half."""
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import parse_reduce  # noqa: PLC0415

    for key, value in tun.items():
        family = family_of(key)
        if family == "PLACE":
            return True
        if family == "REDUCE" and (plan := parse_reduce(value)) is not None and plan.needs_split:
            return True
    return False


def _db_measured_index_build(db, ctx) -> _Measured:
    """Every measured row this compile may deploy, split into what ranks, what disqualifies and
    what prices a kernel set: the tune DB's CUDA ``perf`` rows for this compile's regime (``db``
    may be ``None`` — a machine that never tuned) and the golden rows in scope
    (:func:`~emmy.compiler.pipeline.search.golden.evidence_rows` — the repository's per-card
    files, or the file ``--golden PATH`` named), all in one index.

    Rows are indexed by their ``S_*`` structural signature (stringified values because perf knobs
    round-trip JSON). One context key is sufficient: tune measures in the deployable regime, and
    ``Context.structural_key`` gives that regime one key however its flags are spelled. Rows from a
    deliberately non-deployable compile key elsewhere and are not consulted.

    A non-``ok`` row is evidence too — the bench watchdog measured that variant not finishing — but
    it is evidence a ranker cannot use, since its sentinel latency is a timeout constant rather
    than a speed. It lands in ``failed`` instead, and only where NO variant of that signature was
    measured ``ok``: one surviving row means the shape is realizable and merely has bad rows.

    A row spelling a placement or a cross-CTA split (:func:`_is_route_row`) is the measured price
    of applying that decision to the kernel it was recorded on, and lands in ``routes``: at that
    kernel's fork it names one offered arm (:func:`_route_candidates`); the pieces the arm mints
    are brand-new kernels, decided by rows of their own signatures.

    Best-effort: any failure returns an empty index so deploy falls back to the prior.
    """
    from emmy.compiler.pipeline.search.golden import evidence_rows, scope_explicit  # noqa: PLC0415

    index: dict[frozenset, list[tuple[dict, float]]] = {}
    routes: dict[frozenset, list[tuple[dict, float]]] = {}
    survived: set[frozenset] = set()
    failures: dict[frozenset, list[float]] = {}
    try:
        for row in db.iter_perf(ctx.structural_key(), backend="cuda") if db is not None else ():
            sig = frozenset((k, str(v)) for k, v in row.knobs.items() if k.startswith("S_"))
            if row.status != "ok":
                failures.setdefault(sig, []).append(float(getattr(row.stats, "median", 0.0) or 0.0))
                continue
            survived.add(sig)
            if row.stats.median <= 0:
                continue
            tun = {k: str(v) for k, v in row.knobs.items() if not k.startswith(("S_", "H_"))}
            (routes if _is_route_row(tun) else index).setdefault(sig, []).append((tun, float(row.stats.median)))
        gpu_name = getattr(ctx, "gpu_name", None) or ""
        if gpu_name or scope_explicit():
            for sig, tun, us, _name in evidence_rows(gpu_name, tuple(ctx.compute_capability)):
                (routes if _is_route_row(tun) else index).setdefault(sig, []).append((tun, us))
    except Exception:  # noqa: BLE001 — an evidence consult failure must never break compile
        logger.debug("measured-evidence index build failed", exc_info=True)
        return _EMPTY_MEASURED
    return _Measured(index, {sig: us for sig, us in failures.items() if sig not in survived}, routes)


def _sig_groups(index: dict[frozenset, list[tuple[dict, float]]], sig: frozenset) -> list[list[tuple[dict, float]]]:
    """Drift-tolerant signature match — see :meth:`Prior.sig_groups` (one
    contract for the reservoir and the evidence index)."""
    from emmy.compiler.pipeline.search.prior.base import Prior  # noqa: PLC0415

    return Prior.sig_groups(index, sig)


def _db_measured_pick(
    index: dict[frozenset, list[tuple[dict, float]]],
    rows: list[dict],
    *,
    exact_families: frozenset[str] = frozenset(),
) -> tuple[int, float] | None:
    """Measured-evidence argmin over candidate knob rows against the DB index —
    the same prefix-consistency contract as ``Prior.evidence_pick`` (every
    tunable knob the candidate specifies must match the measured row; undecided
    knobs are free). Signature matching tolerates stamps a row predates (:func:`_sig_groups`).
    Every indexed row was measured in this compile's regime, so the argmin over matching rows is
    the answer. This keeps a config tune measured fastest from losing deploy to an unmeasured
    model extrapolation. Reservoir evidence, where present, still takes precedence at the call
    site.
    """
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
    # this tier's subset matching exists to absorb), every call rescans EVERY
    # index signature building a dict per entry — 41.5k candidates x 61 signatures
    # for a single fork. The memo bounds that at one scan per distinct sig.
    # Per-call scope, so a rebuilt index is never served stale.
    groups_memo: dict[frozenset, list[list[tuple[dict, float]]]] = {}

    best: tuple[int, float] | None = None
    for i, cand in enumerate(rows):
        sig = frozenset((k, str(v)) for k, v in cand.items() if k.startswith("S_"))
        cand_tun = {k: str(v) for k, v in cand.items() if not k.startswith(("S_", "H_"))}
        if sig not in groups_memo:  # not ``.get`` — an empty group list is a valid, falsy hit
            groups_memo[sig] = _sig_groups(index, sig)
        for measured in groups_memo[sig]:
            for row_tun, us in measured:
                # A row counts as evidence when it matches every knob the candidate
                # has decided; undecided knobs are free (``evidence_row_vouches``).
                if not evidence_row_vouches(cand_tun, row_tun, exact_families=exact_families):
                    continue
                if better(us, i, best):
                    best = (i, us)
    return best


def _warn_disjoint_evidence(
    index: dict[frozenset, list[tuple[dict, float]]], rows: list[dict], node_id: str, *, n_rows: int | None = None
) -> None:
    """Warn when a fork's candidate set is DISJOINT from its measured evidence:
    the DB holds rows for this kernel's structural signature, yet
    :func:`_db_measured_pick` matched none of them against any offered
    candidate. That condition is exactly "the tune measured a schedule tier
    the deploy did not offer" — the model then extrapolates over an
    evidence-free candidate set, which shipped gemma o_proj on a scalar tile
    16x its own measured mma rows (the stale-placeholder offer gap). A cold
    compile (no rows for the signature at all) stays silent — extrapolation
    is expected there. ``n_rows`` reports the full candidate count when ``rows`` is a
    representative sample (the streamed scan passes one row — every candidate at one fork shares
    the offer op's ``S_*`` signature, so one row carries the whole set's signature)."""
    sigs = {frozenset((k, str(v)) for k, v in r.items() if k.startswith("S_")) for r in rows}
    n_measured = sum(len(g) for sig in sigs for g in _sig_groups(index, sig))
    if n_measured:
        logger.warning(
            "deploy: node %r has %d measured DB row(s) for its structural signature, but none matches any of the "
            "%d offered candidates — the tune measured a schedule tier this compile did not offer; falling back to "
            "the model prediction. Investigate the enumeration (offer gates) for this kernel.",
            node_id,
            n_measured,
            n_rows if n_rows is not None else len(rows),
        )


def _schedule_fork(fp: ForkPoint) -> bool:
    """Whether ``fp`` decides one kernel's schedule (its options carry the enumeration's ``pool_id``
    stamp) rather than which kernels exist (the cut pass's placement / split fork)."""
    return any(getattr(o, "pool_id", None) is not None for o in fp.options)


def _fork_signature(fp: ForkPoint) -> frozenset:
    """The ``S_*`` signature every candidate at this fork shares (:func:`~emmy.compiler.pipeline.fork.fork_signature`)."""
    return fork_signature(fp.root_op, fp.options, fp.ctx)


class EvidenceError(RuntimeError):
    """Raised under strict evidence (``config.strict_evidence``) when a fork must be decided and
    no measured row — reservoir, tune DB or golden — vouches for any of its candidates."""


def _require_evidence(fp: ForkPoint, why: str) -> None:
    """Under strict evidence, refuse to decide ``fp`` by anything but a measurement."""
    from emmy import config  # noqa: PLC0415

    if not config.strict_evidence():
        return
    name = getattr(fp.root_op, "name", None) or fp.node_id
    raise EvidenceError(
        f"strict evidence: kernel {name!r} (node {fp.node_id!r}) has no measured evidence for its {fp.match.rule.name} fork "
        f"({why}) — record it with `emmy run --golden PATH --bench`, tune it, or load a golden that covers it"
    )


def _route_candidates(fp: ForkPoint, index: _Measured) -> list[tuple[object, float]]:
    """The measured arms at this kernel-set fork: one ``(option, µs)`` per measured row of
    the kernel's signature that spells an arm on the ballot
    (:func:`~emmy.compiler.pipeline.search.pins.spelled_arm`) — a schedule row the fused /
    unsplit arm (the kernel it decorates ran that way), a ``PLACE`` row its cut, a split-carrying
    ``REDUCE`` row its split. The option is the cut pass's own offer; the pieces it mints are
    brand-new kernels whose own forks consult their own rows. A schedule fork has none."""
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import _structural_domain  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import spelled_arm  # noqa: PLC0415

    root = fp.root_op
    if not isinstance(root, TileOp) or root.op is None or _schedule_fork(fp):
        return []
    if _structural_domain(fp.options) not in (("PLACE",), ("REDUCE",)):
        return []
    sig = _fork_signature(fp)
    measured = [entry for source in (index.ok, index.routes) for group in _sig_groups(source, sig) for entry in group]
    out: list[tuple[object, float]] = []
    for row, us in measured:
        arm = spelled_arm(fp.options, row)
        if arm is not None:
            out.append((arm[0], us))
    return out


def _direct_measured_pick(fp: ForkPoint, blocked, db_index: dict) -> tuple[object, dict, float] | None:
    """Descend directly to the fastest offered tune-DB row.

    Evidence rows already spell complete schedules, so scoring branch representatives would be
    both slower and less exact. Expansions are memoized across records; each tree branch is opened
    at most once during the lookup.
    """
    from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

    db_signature = _fork_signature(fp)
    node_blocked = blocked.get(fp.node_id) if blocked else None

    def offered(records):
        # ``leaf_for`` descends only the branches that admit the row (``Fork.admits``): a family
        # the row leaves undecided is free here as everywhere, so a partial row still reaches its
        # leaf whatever the pool size — the one path that does not depend on the cold-pool budget.
        ordered = sorted(records, key=lambda item: (item[1], canonical_row_key(item[0])))
        skip = (lambda knobs: _tile_blocked(knobs, node_blocked)) if node_blocked is not None else None
        for record, price in ordered:
            if (hit := leaf_for(fp.options, record, skip=skip)) is not None:
                return hit[0], hit[1], float(price)
        return None

    if db_index:
        groups = _sig_groups(db_index, db_signature)
        records = [(row, price) for group in groups for row, price in group]
        if records and (picked := offered(records)) is not None:
            return picked
    return None


#: Leaves scored per batch in the streamed scan: large enough to amortize CatBoost's per-``predict``
#: overhead (its batched surface exists because per-row calls pay it N times), small enough that the
#: transient row dicts stay bounded — the flat 486k-row pools this replaces held ~GBs of them at once.
_CHUNK = 4096

#: The cold-pool budget: a pool whose minted size bound (``Fork.pool_bound`` — Π of the per-node
#: option tuples, legality only shrinks it) exceeds this is not walked at full length on a cold
#: deploy. The research-class fused terms enumerate millions of legal schedules, and a model
#: argmin over all of them buys nothing a bounded sample doesn't: the cold pick only needs a
#: REASONABLE kernel — the optimal one comes from evidence (a tune, a benched golden row), which
#: the measured descent deploys directly regardless of pool size.
_POOL_BUDGET = 65_536

#: Maximum complete rows drawn for a budgeted pool: seeded uniform descents through the lazy tree
#: cover every level's values, unlike an emission-order prefix. The option-check budget below may
#: reduce this count for a wide, deep tree; each emitted row remains a legal complete schedule.
_POOL_DRAW = 2_048

#: Maximum option checks spent drawing one cold pool, including the four-attempt allowance for a
#: dead or blocklisted descent. A fixed row count is not a work bound for wide, deep terms. When
#: one descent's declared bound already exceeds this value, exactly one complete-row attempt is
#: the deliberate soft-cap exception; the Fork interface cannot pause one sibling expansion.
_POOL_DESCENT_WORK = 262_144


def _descent_sample(options, pool_id: str, node_blocked) -> list:
    """Up to :data:`_POOL_DRAW` complete leaves drawn by seeded uniform descents. Dead ends (a
    branch whose expansion is empty — legality killed the subtree) and blocklisted rows retry, up
    to a bounded attempt count. When one descent is wider than the work budget, make exactly one
    attempt: completing a legal row is indivisible through the Fork interface. Duplicates are kept
    (a repeat costs a scoring slot, never a wrong pick). Structural options never appear here —
    the caller samples only the variant side."""
    import random  # noqa: PLC0415

    rng = random.Random(pool_id)
    sample: list = []
    descent_bound = max((getattr(option, "pool_descent_bound", None) or 1 for option in options), default=1)
    one_descent_exceeds_budget = descent_bound > _POOL_DESCENT_WORK
    attempt_budget = max(1, _POOL_DESCENT_WORK // descent_bound)
    draw = 1 if one_descent_exceeds_budget else min(_POOL_DRAW, max(1, attempt_budget // 4))
    attempts = 1 if one_descent_exceeds_budget else min(4 * draw, attempt_budget)
    while len(sample) < draw and attempts > 0:
        attempts -= 1
        option = options[rng.randrange(len(options))]
        dead = False
        while isinstance(option, Fork) and not option.is_leaf:
            kids = option.expand()
            if not kids:
                dead = True
                break
            option = kids[rng.randrange(len(kids))]
        if dead:
            continue
        if node_blocked is not None and _tile_blocked(leaf_knobs(option), node_blocked):
            continue
        sample.append(option)
    return sample


def _stream_tiers(
    fp: ForkPoint, the_prior, node_blocked, db_idx: dict, options: list | None = None
) -> tuple[object, dict | None, float | None, str | None] | None:
    """The deploy evidence hierarchy over a non-structural pool, in ONE streamed walk.

    The lazy walk is not free — each branch expansion re-spells its schedule step, and on the
    research-class pools (a 486k-row explicit-mask softmax term) the walk itself costs minutes —
    so this scan walks exactly once, like the flatten it replaces, and evaluates every source
    chunk-wise as the leaves go by: measured reservoir evidence, the evidence index's measured
    best (tune DB rows and golden rows), and the model score, each folded into its own running
    best. The PRIORITY is applied after the stream ends (reservoir > index > model, the same
    hierarchy as before); the one behavioral trade is that the model's ``mean_scores`` runs even
    when a later chunk turns up evidence — acceptable because measured forks are normally decided
    upstream by the direct measured descent, never here. The pick is EXACTLY the flattened argmin: every source
    breaks ties by candidate content (``canonical_row_key``), never enumeration order, so
    per-chunk winners folded through a running ``(price, key)`` min are chunk-invariant.

    A pool whose minted size bound exceeds :data:`_POOL_BUDGET` is not walked: the scan ranks a
    deterministic drawn subset instead (:func:`_descent_sample` — seeded uniform descents, legal
    complete rows only). Above the budget the pick is the argmin over the draw, not the pool —
    the accepted cold-deploy trade: a reasonable kernel now, the optimal one from evidence (the
    measured descent reaches it directly whatever the pool size, and a bad cold pick is fixed by
    measuring, as ever).

    Returns ``None`` when a structural (``Graph``-splicing) option is present — those forks carry
    a handful of options and keep the flatten path, where :func:`_priced_pick` needs the whole
    leaf set. ``(leaf, None, None, None)`` is the degenerate plain return (≤1 leaf, or every leaf
    blocklisted — no score, no decision memo); ``(leaf, knobs, price, tier)`` is the ranked pick,
    ``tier`` being ``"evidence"`` (a measured row decided) or ``"model"``."""
    from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import NO_OPTION, _is_structural_option  # noqa: PLC0415

    base = {**fp.ctx.features(), **dict(fp.root_op.knobs)}
    picker = getattr(the_prior, "pick", None)
    ev = getattr(the_prior, "evidence_pick", None) if picker is not None else None
    use_db = picker is not None and bool(db_idx)
    # Per-source running bests: (price, canonical_row_key, leaf, knobs).
    best_ev: tuple | None = None
    best_db: tuple | None = None
    best_model: tuple | None = None

    def fold(best: tuple | None, chunk: list, got: tuple[int, float] | None) -> tuple | None:
        if got is None:
            return best
        i, price = got
        key = (price, canonical_row_key(chunk[i][2]))
        if best is None or key < (best[0], best[1]):
            return (price, key[1], chunk[i][0], chunk[i][1])
        return best

    def scan(chunk: list) -> None:
        nonlocal best_ev, best_db, best_model
        rows = [row for _, _, row in chunk]
        if ev is not None:
            best_ev = fold(best_ev, chunk, ev(rows))
        if use_db:
            best_db = fold(best_db, chunk, _db_measured_pick(db_idx, rows))
        scorer = getattr(the_prior, "mean_scores", None)
        if scorer is not None:
            scores = scorer(rows)
            # Two-stage argmin: find the min score first, spell ``canonical_row_key`` only for
            # the tied rows — the key is a canonicalizing sort over the whole row, and computing
            # it for every candidate (as the flattened ``min`` key did) dominated large-pool
            # scans.
            lo = min(scores)
            ties = [j for j, score in enumerate(scores) if score == lo]
            i = ties[0] if len(ties) == 1 else min(ties, key=lambda j: canonical_row_key(rows[j]))
            best_model = fold(best_model, chunk, (i, lo))
        else:
            # A ``pick``-only prior (no per-row scoring surface): ask it per chunk and fold on the
            # ``(index, score)`` it returns — exact for any single-chunk pool, and for the real
            # ``Prior`` classes generally (their pick IS the mean_scores argmin).
            best_model = fold(best_model, chunk, picker(rows))

    opts = fp.options if options is None else options
    # The cold-pool budget: a pool whose minted bound exceeds _POOL_BUDGET is sampled by seeded
    # descents instead of walked — the sources below then rank the drawn complete rows exactly as
    # they would the full pool. An empty draw fails explicitly: walking the full oversized pool
    # would silently discard the bound. Report the empty subtree to the resolver so it can keep
    # walking the current rule batch; returning a partial branch would violate the prior's
    # complete-row contract.
    bound = next((b for o in opts if (b := getattr(o, "pool_bound", None)) is not None), None)
    drawn = None
    if bound is not None and bound > _POOL_BUDGET:
        pid = next((p for o in opts if (p := getattr(o, "pool_id", None)) is not None), "")
        drawn = _descent_sample(opts, pid, node_blocked)
        if not drawn:
            return NO_OPTION, None, None, None
    n_leaves = n_live = 0
    first: object = None
    sample_row: dict | None = None  # one live row — carries the fork's shared ``S_*`` signature
    chunk: list = []
    for leaf in drawn if drawn is not None else iter_leaves(opts):
        if _is_structural_option(leaf):
            return None
        n_leaves += 1
        if first is None:
            first = leaf
        knobs = leaf_knobs(leaf)
        if node_blocked is not None and _tile_blocked(knobs, node_blocked):
            continue
        n_live += 1
        row = {**base, **knobs}
        if sample_row is None:
            sample_row = row
        chunk.append((leaf, knobs, row))
        if len(chunk) >= _CHUNK:
            scan(chunk)
            chunk = []
    if n_leaves == 0:
        return NO_OPTION, None, None, None
    if n_leaves == 1 or n_live == 0:
        return first, None, None, None
    if chunk:
        scan(chunk)
    if best_ev is not None:
        return best_ev[2], best_ev[3], best_ev[0], "evidence"
    if best_db is not None:
        return best_db[2], best_db[3], best_db[0], "evidence"
    if use_db:
        _warn_disjoint_evidence(db_idx, [sample_row], fp.node_id, n_rows=n_live)
    return best_model[2], best_model[3], best_model[0], "model"


def greedy_decide(
    blocked: dict[str, set[frozenset]] | None = None,
    *,
    prior: object = _LOAD_PRIOR,
    price_structural: bool = True,
    db: object | None = None,
    decisions: dict | None = None,
) -> Callable[[ForkPoint], object]:
    """The greedy compile pick as a :meth:`Run.resolve` ``decide`` callback:
    descend directly to exact evidence when available, otherwise stream the complete rows in
    bounded chunks (:func:`_stream_tiers`), skip ``blocked`` tile identities, and take the
    prior's global argmin. The prior is the
    ``OnlinePrior`` once trained and the ``OfflinePrior``
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
    attempt), keyed by ``identity_key(with_io=True, with_knobs=True)``."""
    from emmy.compiler.pipeline.pipeline import _is_structural_option  # noqa: PLC0415

    memo: dict[str, float | None] = {}  # exact variant key → predicted µs (None = unpriceable)
    #: The DECISION memo (:func:`_decision_key` → the winning row + its price) — same lifetime as
    #: the price memo, one compile attempt. A repeat offer replays by :func:`~emmy.compiler.pipeline.fork.find_leaf`
    #: descent; only genuinely new (key, blocklist) states pay the stream-and-score. The outer
    #: compile's memo is SHARED into every nested pricing resolve (``decisions=`` — each
    #: ``_price_kernel`` used to build a fresh factory, so N placement variants re-walked and
    #: re-scored one identical schedule pool N times; the key already carries the pool identity,
    #: the rule, and the blocklist, so sharing is exactly the replay the memo was built for).
    decisions = {} if decisions is None else decisions
    loaded = prior is not _LOAD_PRIOR
    the_prior = prior if loaded else None
    # Lazily-built per-compile measured-evidence index (needs a fork point's ctx for the
    # context keys): the tune DB's rows and the golden rows in scope, one index.
    # ``None`` sentinel = not built yet.
    db_state: list = [None]

    def db_index() -> dict:
        return (db_state[0].ok if db_state[0] is not None else None) or {}

    def decide(fp: ForkPoint) -> object:
        nonlocal loaded, the_prior
        from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

        if db_state[0] is None:
            db_state[0] = _db_measured_index(db, fp.ctx)
        index: _Measured = db_state[0]
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
        # A kernel-set fork: every measured row of this kernel spells one offered arm, and a
        # measured arm outranks anything priced by nested resolution (a Σ that may hold
        # predictions). Among measured arms the fastest wins.
        arms = _route_candidates(fp, index) if price_structural else []
        if arms:
            # Fastest first; a tie breaks by the arm's content, never by emission order.
            best_o, best_us = min(arms, key=lambda c: (c[1], canonical_row_key(leaf_knobs(c[0]))))
            fp.score = best_us
            return best_o
        if the_prior is None:
            # No prior on this resolve — a failed ``load_prior`` (corrupt/unreadable
            # checkpoint) or ``Pipeline.run``'s explicit emission-order fallback
            # (``prior=None``): emission order (option-0, first leaf).
            if len(fp.options) > 1:
                _require_evidence(fp, "no prior loaded; emission order would decide")
            return next(iter_leaves(fp.options))
        if dkey is not None and _schedule_fork(fp):
            picked = _direct_measured_pick(fp, blocked, db_index())
            if picked is not None:
                leaf, row, price = picked
                fp.score = price
                decisions[dkey] = (dict(row), price)
                return leaf
        # Greedy benches nothing, so it must pick the globally best COMPLETE
        # tile, not a partial branch (the prior is blind at a partial ``BM/BN``
        # branch: ``knob_features`` can't compute the tile's area / occupancy
        # until ``FM/FN`` exist). ``_stream_tiers`` scores those complete rows
        # off the lazy walk in bounded chunks — the pick equals the flattened
        # scoring's argmin exactly (content-keyed tie rules make the running
        # min chunk-invariant), without ever retaining the O(pool) leaf and
        # row lists that made a 486k-row cold pool an OOM.
        #
        # Structural (``Graph``-splicing) options are TOP-LEVEL siblings by construction
        # (``_is_structural_option``: schedule-product branches contain only ``TileOp`` leaves),
        # so they are split off here without a walk. The keep-fused side is then ONE streamed
        # scan — its best price is the same quantity ``_price_op_leaf``'s nested resolve computes
        # (the deploy evidence hierarchy's best at this kernel's own fork), which the old
        # per-leaf pricing re-derived through one nested compile per flattened leaf: a
        # 9k-leaf placement fork paid 9k nested resolves for 9k identical answers. A splice's
        # price stays the nested Σ over its fragment kernels (:func:`_price_graph`); an op-vs-op
        # score tie keeps the content tie rule, an op-vs-splice tie keeps the fused side.
        node_blocked = blocked.get(fp.node_id) if blocked else None
        splices, plain = fp.splices, fp.variants
        if splices and not price_structural:
            # Structural RETIREMENT, not a ranking rule: a fragment kernel that failed to lower
            # cannot be blocklisted at the fork site (the splice minted fresh node ids), so
            # ``Pipeline.run`` re-resolves with the splices withdrawn — the same role ``blocked``
            # plays for a tile. It also stops a nested price probe from re-splitting the slice it
            # is pricing. All-splice forks keep their options (the old ``or leaves`` fallback).
            splices, plain = (), plain or fp.options
        streamed = _stream_tiers(fp, the_prior, node_blocked, db_index(), options=plain) if plain else None
        if streamed is not None:
            leaf, row, price, tier = streamed
            if row is None and not splices:
                return leaf  # degenerate pool (≤1 leaf / all blocklisted): plain, unscored return
            if row is not None:
                if splices:
                    # No measured row spelled an arm here (those return above), so this is a
                    # comparison of prices, which strict evidence refuses.
                    _require_evidence(fp, "no measured row spells a kernel-set arm")
                    # ONE price definition: a price is the Σ of a resolution's trace. The splices
                    # price by nested resolution of their fragments; the fused side prices by one
                    # nested resolution of the STREAMED winner — the scan already found the best
                    # row, so this is a single resolve, not one per enumerated leaf — keeping the
                    # two sides of the kernel-set comparison the same quantity (a fork-local row
                    # score would omit any further scored forks the fused resolution hits).
                    priced = [(o, _price_graph(_leaf_graph(o), fp.ctx, the_prior, memo, db, decisions)) for o in splices]
                    fused_us = _price_op_leaf(fp, leaf, the_prior, memo, db, decisions)
                    if fused_us is not None and all(us is not None for _, us in priced):
                        best_o, best_us = min(priced, key=lambda o_us: o_us[1])
                        if best_us < fused_us:
                            fp.score = best_us
                            return best_o
                    else:
                        # An unpriceable side: the old contract sends EVERY leaf to the ordinary
                        # ranking, structural ones included — the flatten path below keeps that.
                        streamed = None
                if streamed is not None and row is not None:
                    if tier == "model":
                        _require_evidence(fp, "no measured row vouches for any offered candidate")
                        fp.score = price
                    if dkey is not None:
                        decisions[dkey] = (dict(row), price)
                    return leaf
                if streamed is not None:
                    return leaf
        # Reached on: an unpriceable splice, an all-splice fork, a degenerate op side beside
        # splices, or a structural leaf that surfaced mid-stream (outside the top-level
        # construction) — all small-pool corners; the flatten path handles them as before.
        leaves = flatten_leaves(fp.options if price_structural else (plain or fp.options))
        base = {**fp.ctx.features(), **dict(fp.root_op.knobs)}
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
                _require_evidence(fp, "no measured row spells a kernel-set arm")
                pick = _priced_pick(fp, leaves, the_prior, memo, db, decisions)
                if pick is not None:
                    return pick
        if len(leaves) <= 1:
            if leaves:
                return leaves[0]
            from emmy.compiler.pipeline.pipeline import NO_OPTION  # noqa: PLC0415

            return NO_OPTION
        # The constant base under this fork's deltas: the offer op's knobs
        # (its ``S_*`` structural identity) plus the ``H_*`` host/hardware
        # regime — the feature base tune trained on (``two_level.inner_reward``).
        # Tiles this node already failed to lower on an earlier attempt — skip
        # the matching leaf so greedy falls back to the next prior-ranked one.
        live = [(o, leaf_knobs(o)) for o in leaves]
        if node_blocked is not None:
            live = [(o, k) for o, k in live if not _tile_blocked(k, node_blocked)]
        if not live:  # every leaf blocklisted → no valid alternative left
            return leaves[0]
        rows = [{**base, **k} for _, k in live]
        # The deploy evidence hierarchy, top first: (1) measured reservoir
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
            if got is None:
                _require_evidence(fp, "no measured row vouches for any offered candidate")
            best_i, price = got if got is not None else picker(rows)
        else:  # bare-mean_scores prior
            from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

            _require_evidence(fp, "a bare scoring prior decides")
            s = the_prior.mean_scores(rows)
            best_i = min(range(len(rows)), key=lambda i: (s[i], canonical_row_key(rows[i])))
            price = s[best_i]
        fp.score = price  # measured µs when evidence decided, predicted µs otherwise
        if dkey is not None:
            decisions[dkey] = (dict(live[best_i][1]), price)
        return live[best_i][0]

    return decide
