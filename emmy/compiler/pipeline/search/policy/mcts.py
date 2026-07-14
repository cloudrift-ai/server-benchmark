"""Single-player MCTS for ``emmy tune`` with max-reward propagation and
**PUCT** selection — the online
:class:`~emmy.compiler.pipeline.search.prior.Prior` is the *only* selection
signal (greedy and the ``+∞``-unvisited UCB rule are gone).

    select   — descend from root, picking at each level
               ``argmax_c [ Q(c) + c · P(c) · √(N_parent+1) / (1+N_c) ]``
               where ``Q = best_reward / global_best`` and ``P`` is the prior's
               *predicted* reward on the same scale: the prior predicts latency,
               which this loop converts to reward (``1/û``) and normalizes by the
               same ``global_best`` as ``Q``. No softmax, no ``+∞``-unvisited rule
               → no forced breadth: a confidently-bad sibling gets a small ``P``
               and is skipped. A cold or absent prior gives a uniform ``P = 1``
               (PUCT still explores via the exploration term; a single-shot
               compile with no prior descends emission-order). Live-count
               filtering skips drained subtrees.
    expand   — :meth:`TuningSearch.push` adds the engine's spawned
               candidates as children of the ``parent`` token (the
               ``SearchNode`` their spawning candidate was popped with);
    simulate — the engine runs the popped candidate and benches it;
    backprop — :meth:`SearchTree.record_terminal` walks ``parent``
               links from the observed terminal's token, bumping
               ``visits`` and updating
               ``best_reward = max(best_reward, leaf_reward)``.

Reward — both measured and prior-predicted — is normalized against the global
best so the exploration constant ``c`` is unit-free.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from emmy import config
from emmy.compiler.pipeline.search.candidate import LazyCandidate
from emmy.compiler.pipeline.search.db import NodeRow, PerfStats
from emmy.compiler.pipeline.search.policy.base import Search
from emmy.compiler.structural import digest

if TYPE_CHECKING:
    from emmy.compiler.pipeline.search.prior import Prior

# Re-bench at -O3 any config whose -O1 latency is within this fraction of the best
# -O1 so far (not just a strict new best). -O1 is the fast ranking compile but its
# ranking diverges at -O3 (deployable): register-tile mma kernels run 1.5–3× slower at
# -O1 (the cicc unroll blowup the ranking compile dodges), so an mma config outside a
# narrow -O1 band can still be the -O3 deploy winner — and without a deployable sample
# the evidence-first deploy hierarchy only ever sees scalar -O3 rows and keeps deploying
# them (the qwen3-emb layer-0 projections: scalar g2a evidence at 394 µs deployed over
# the 70 µs-class mma picks). The band covers that skew (within 3× of best -O1); cost is
# bounded by ``_o3_done``'s per-config dedup — one -O3 compile + short bench each.
# Env-overridable via ``EMMY_O3_TOL`` (a fraction, e.g. ``0.15`` for 15%).
O3_REBENCH_TOL = 2.0

# The nvcc flags of that deployable re-bench (``pipeline._rebench_o3_async``) — also
# the regime the re-bench's node rows are keyed under (``two_level`` derives their
# ``context_key`` from the tune context with these flags substituted).
O3_NVCC_FLAGS = "-Xcicc -O3"


@dataclass
class SearchNode:
    candidate: LazyCandidate | None  # None for the root sentinel
    parent: SearchNode | None = field(default=None, repr=False)
    children: list[SearchNode] = field(default_factory=list, repr=False)
    visits: int = 0
    best_reward: float = 0.0  # max reward over this subtree's measured leaves
    live: int = 0  # count of un-popped frontier leaves in this subtree
    # The benched terminal's FULL realized knob set (S_/H_ base + every tunable
    # knob, incl. those stamped at deterministic lowering steps that never fork).
    # Set on directly-benched leaves in ``observe``; ``None`` on branches, which
    # keep their partial fork-prefix for value-of-position.
    realized_knobs: dict | None = field(default=None, repr=False)
    # The direct bench's measurement stats + status (``'ok'`` / ``'bench_fail'``),
    # set alongside ``realized_knobs`` in ``observe``; ``None`` on branches (never
    # directly benched). Read back by ``_collect_node_records`` so the node store
    # keeps the leaf's variance / n_samples and its failure outcome.
    bench_stats: PerfStats | None = field(default=None, repr=False)
    bench_status: str | None = field(default=None, repr=False)
    # The leaf's deployable -O3 re-bench median (``observe_o3``), ``None`` when the
    # config wasn't in the re-bench tolerance band (or the sweep already ran at -O3).
    # Read back by ``_collect_node_records`` to emit the leaf's -O3-regime node row.
    o3_us: float | None = field(default=None, repr=False)


class SearchTree:
    def __init__(self) -> None:
        self.root = SearchNode(candidate=None)

    @property
    def best_reward(self) -> float:
        return self.root.best_reward

    def attach(self, candidates: list[LazyCandidate], parent: SearchNode) -> None:
        nodes = [SearchNode(candidate=c, parent=parent, live=1) for c in candidates]
        parent.children.extend(nodes)
        # Each new child is a fresh frontier — bump live count on every ancestor.
        cur: SearchNode | None = parent
        while cur is not None:
            cur.live += len(nodes)
            cur = cur.parent

    def record_terminal(self, node: SearchNode, reward: float) -> None:
        """Max-propagate ``reward`` from ``node`` (the terminal's own
        SearchNode — the token it was popped with) up to the root,
        bumping ``visits`` along the way."""
        cur: SearchNode | None = node
        while cur is not None:
            cur.visits += 1
            if reward > cur.best_reward:
                cur.best_reward = reward
            cur = cur.parent


class TuningSearch(Search):
    """SP-MCTS with PUCT selection — the online prior is the sole signal."""

    DEFAULT_UCB_C = math.sqrt(2)

    def __init__(
        self,
        tree: SearchTree | None = None,
        *,
        patience: int = 50,
        ucb_c: float = DEFAULT_UCB_C,
        explore_eps: float = 0.0,
        seed: int = 0,
        max_visits: int | None = None,
        prior_model: Prior | None = None,
        base_knobs: dict | None = None,
    ) -> None:
        super().__init__()
        self.tree = tree if tree is not None else SearchTree()
        self._ucb_c = ucb_c
        # ε-greedy exploration: with probability ``explore_eps`` a selection step
        # descends a UNIFORMLY RANDOM live child instead of the PUCT argmax. PUCT
        # alone is deterministic — a tie (cold prior → uniform ``P``) always goes
        # to the first-in-list (= heuristic enumeration order), so each fork is
        # visited once and takes its heuristic-preferred child; a binary fork like
        # ``WARPSPEC`` then never benches its option-1 branch even when that's the
        # real win. ε-randomness makes ~half the visits to such a fork take the
        # other branch, so tuning finds good configs WITHOUT relying on the
        # heuristic/prior ordering. ``0.0`` (the default) restores deterministic
        # PUCT — kept for the unit tests and single-shot compile. Seeded for
        # reproducibility (vary ``seed`` per op/run upstream, not via wall clock).
        self._explore_eps = explore_eps
        self._rng = random.Random(seed)
        self._patience = patience
        self._max_visits = max_visits
        # Online prior driving PUCT selection — a fixed global model for the run
        # (it refits in batches between ops, not within one). ``None`` for a
        # single-shot compile (no benching) → uniform PUCT → emission-order pick.
        self.prior_model = prior_model
        # The kernel's identity knobs (the ``S_*`` structural features stamped on
        # the LoopOp) — merged under every node's accumulated fork deltas so the
        # GLOBAL prior sees op-structure and can tell kernels apart.
        self._base_knobs = dict(base_knobs) if base_knobs else {}
        self._best_reward = 0.0
        self._visits_at_best = 0
        # Why the search stopped (set by ``_should_stop``): a patience /
        # max_visits message, or ``None`` when the queue drained — the
        # exhaustion signal ``two_level.inner_reward`` records as ``inf``
        # effort.
        self.stop_reason: str | None = None
        # Last benched variant's measurement — read by the tune progress bar
        # after each yielded terminal (the engine calls ``observe`` right before
        # yielding). Carries no role in the search itself.
        self.last_stats: PerfStats | None = None
        self.last_status: str | None = None
        # Set in ``observe`` when a bench sets a new global best.
        self.last_improved_best = False
        # Set in ``observe`` when the just-benched config is a *deployable-bench
        # candidate* — within ``O3_REBENCH_TOL`` of the best -O1 so far and not
        # already re-benched. The engine reads it to trigger an -O3 re-bench
        # (``observe_o3``). Widening from "strict new best" to a tolerance band is
        # what lets configs that TIE at -O1 but differ at -O3 (the warp-tier
        # WARPSPEC / occupancy split is the motivating case) get a deployable
        # sample, so the prior can rank them by -O3 cost. ``_o3_done`` dedups so
        # each config is -O3'd at most once; ``o3_rows`` holds the samples
        # (knobs tagged ``H_opt=3``) for the prior.
        self.last_o3_worthy = False
        self._o3_done: set[tuple] = set()
        self.o3_rows: list[tuple[dict, float]] = []
        # Best -O1 latency among STANDARD-regime rows (no precision-trading realization). Under a
        # precision gate (FAST_MATH / F16_MMA_F32_ACC) the global best is usually a fast-math row,
        # and a regime-blind tolerance band would then starve the standard lane of -O3 samples —
        # the gate-off deploy hierarchy is evidence-first, so a shape whose only -O3 rows are
        # fast-math (absent from the gate-off enumeration) falls to the model argmin (the
        # qkv.h4096 ~1000x scalar deploy, 2026-07-09 fm sweep). A standard row therefore competes
        # in its OWN band; with no gate every row is standard and this equals the global best.
        self._best_std_lat: float | None = None

    def observe(self, token: object | None, stats: PerfStats, status: str, candidate: object | None = None) -> None:
        self.last_stats = stats
        self.last_status = status
        assert isinstance(token, SearchNode), f"TuningSearch.observe needs the terminal's pop token, got {type(token).__name__}"
        # Record the benched leaf's FULL realized knobs (from the resolved graph),
        # not just its fork-prefix — so knobs stamped at deterministic lowering
        # steps (FK / BK / STAGE / …) reach the prior. Falls back to the
        # fork-prefix when no candidate is supplied.
        token.realized_knobs = self._realized_knobs(candidate) if candidate is not None else self._node_knobs(token)
        token.bench_stats = stats
        token.bench_status = status
        reward = (1.0 / stats.median) if status == "ok" and stats.median > 0 else 0.0
        prev_best = self.tree.best_reward
        self.tree.record_terminal(token, reward)
        self.last_improved_best = status == "ok" and self.tree.best_reward > prev_best
        # Re-bench at -O3 not only a strict new best but any config within the
        # tolerance band of the best -O1 so far — configs that tie at -O1 can
        # differ sharply at -O3 (the warp WARPSPEC / occupancy split), so the
        # prior needs an -O3 sample for every near-best contender, not just the
        # winner. Dedup via ``_o3_done`` so each config is -O3'd at most once.
        self.last_o3_worthy = False
        if status == "ok" and stats.median > 0 and self.tree.best_reward > 0:
            from emmy.compiler.pipeline.search.golden import fast_math_knobs  # noqa: PLC0415 — layering: policy → search core

            # A standard-regime row competes in its own band (see ``_best_std_lat``): the best
            # standard row must reach -O3 even when a fast-math row owns the global best, or the
            # gate-off deploy lane is left with no deployable evidence for this shape.
            is_fm = fast_math_knobs(token.realized_knobs or {})
            if not is_fm and (self._best_std_lat is None or stats.median < self._best_std_lat):
                self._best_std_lat = stats.median
            ref_lat = 1.0 / self.tree.best_reward
            if not is_fm and self._best_std_lat is not None:
                ref_lat = self._best_std_lat
            tol = config.o3_tol(O3_REBENCH_TOL)
            sig = self._o3_sig(token.realized_knobs)
            if stats.median <= ref_lat * (1.0 + tol) and sig not in self._o3_done:
                self._o3_done.add(sig)
                self.last_o3_worthy = True
        if self.prior_model is not None:
            # Record the leaf for the end-of-run stats. The model itself is fixed
            # during a run — it refits in batches between ops (see ``Prior``), not
            # per bench — so there is nothing to refit here.
            self.prior_model.record_bench(token.realized_knobs, stats.median, status)

    def observe_o3(self, token: object | None, o3_us: float) -> None:
        """Record an extra training row for an -O1 winner re-benched at -O3: the
        same realized knobs but tagged ``H_opt=3`` (the deployable regime) and
        labeled with the -O3 median latency (µs) — the prior's regression target
        is latency, converted to reward only in the MCTS selection loop. The prior
        can then rank winners by -O3 cost where -O1 ties them; ``H_opt`` lets the
        -O1 and -O3 rows coexist. Also stashed on the token so
        :meth:`_collect_node_records` emits the leaf's -O3-regime node row."""
        if o3_us <= 0 or not isinstance(token, SearchNode) or token.realized_knobs is None:
            return
        token.o3_us = o3_us
        knobs = dict(token.realized_knobs)
        knobs["H_opt"] = 3.0
        self.o3_rows.append((knobs, o3_us))
        if self.prior_model is not None:
            self.prior_model.record_bench(knobs, o3_us, "ok")

    @staticmethod
    def _o3_sig(knobs: dict | None) -> tuple:
        """A hashable signature of a realized knob set for -O3 dedup. Values are
        ``str()``-ified for a uniform hashable key, and the ``H_opt`` regime tag is
        excluded so the -O1 row and its -O3 re-bench share one signature."""
        if not knobs:
            return ()
        return tuple(sorted((k, str(v)) for k, v in knobs.items() if k != "H_opt"))

    def _realized_knobs(self, candidate: object) -> dict:
        """The terminal's complete knob set: the kernel's ``base_knobs`` (``S_*``
        identity + ``H_*`` regime) merged with the realized op ``knobs`` off the
        resolved graph (every tunable knob, including deterministically-stamped
        ones that ``_node_knobs`` can't see). Unions all op knob dicts — a
        single-kernel slice has one kernel-bearing op, constants carry none."""
        merged: dict = dict(self._base_knobs)
        graph = getattr(candidate, "graph", None)
        if graph is not None:
            for node in graph.nodes.values():
                knobs = getattr(node.op, "knobs", None)
                if knobs:
                    merged.update(knobs)
        return merged

    def push(self, *cands: LazyCandidate, parent: object | None = None, structural: bool = False) -> None:
        # ``parent`` is the token the spawning candidate was popped with;
        # ``None`` seeds the run under the root sentinel. ``structural``
        # (kernel-set-changing fork) is accepted for protocol uniformity;
        # MCTS explores structural siblings like any other fork.
        del structural
        assert parent is None or isinstance(parent, SearchNode), f"TuningSearch.push needs a SearchNode token, got {type(parent).__name__}"
        self.tree.attach(list(cands), parent=parent if parent is not None else self.tree.root)

    def pop(self) -> tuple[object | None, LazyCandidate] | None:
        if self._should_stop():
            return None
        # The prior is a fixed global model during a run (it refits in batches
        # between ops, not per descent), so selection just reads its scores.
        node = self.tree.root
        if node.live == 0:
            return None
        while node.children:
            descendable = [c for c in node.children if c.live > 0]
            if not descendable:
                return None
            node = self._select(descendable, node)
        # Frontier just got handed off — drop it from the live count
        # on every ancestor. The engine may push children of this node
        # before the next pop; those pushes re-grow the count.
        cur: SearchNode | None = node
        while cur is not None:
            cur.live -= 1
            cur = cur.parent
        return node, node.candidate

    def _prior_score(self, child: SearchNode) -> float:
        """The prior's predicted *latency* (µs) for a child (``0`` when no model
        is attached, the model is unfit, or the child is the root sentinel — the
        ``_select`` loop treats a non-positive prediction as a uniform ``P``)."""
        if self.prior_model is None or child.candidate is None:
            return 0.0
        return self.prior_model.score(self._node_knobs(child))

    def _select(self, children: list[SearchNode], parent: SearchNode) -> SearchNode:
        """PUCT is the *only* selection rule — the prior is the sole signal.

            score(c) = Q(c) + c_ucb · P(c) · √(N_parent + 1) / (1 + N_c)

        where ``Q = best_reward / global_best`` (``0`` for an unvisited child) and
        ``P`` is the prior's *predicted reward* on the same scale: the prior
        predicts latency ``û(c)``, which this loop converts to reward (``1/û``)
        and normalizes by the same ``global_best`` as ``Q`` — no softmax. A
        confidently-bad sibling gets a small ``P`` → tiny exploration term → it is
        deprioritized rather than force-visited. The prior is always consulted —
        the ``FallbackPrior`` returns the online model's prediction once trained
        and the ``OfflinePrior`` heuristic cold, so even a fresh ``tune`` is
        prior-guided, not uniform. Only when there is NO usable prediction (no
        prior attached, or a non-positive score) does ``P`` fall to a uniform
        ``1`` so the exploration term still drives breadth. ``c_ucb`` is
        ``--ucb-c``. With ``explore_eps > 0`` (tune, opt-in) a fraction of steps
        instead descend a uniformly random live child (ε-greedy); off by default
        so a single-shot compile / the unit tests stay deterministic. NOTE: a
        *random tie-break* under a cold prior was tried and reverted — it discarded
        the heuristic ordering and regressed fp16 tuning ~2×; exploration must
        perturb the prior order, not replace it."""
        if self._explore_eps and self._rng.random() < self._explore_eps:
            return self._rng.choice(children)
        global_best = self.tree.best_reward or 1.0
        sqrt_parent = math.sqrt(parent.visits + 1)
        best, best_v = children[0], float("-inf")
        for c in children:
            q = (c.best_reward / global_best) if c.visits > 0 else 0.0
            pred_us = self._prior_score(c)
            if pred_us > 0:
                p = (1.0 / pred_us) / global_best
            else:
                p = 1.0  # cold / absent prior → uniform exploration
            v = q + self._ucb_c * p * sqrt_parent / (1 + c.visits)
            if v > best_v:
                best_v, best = v, c
        return best

    def _node_knobs(self, node: SearchNode) -> dict:
        """Accumulated knob dict for a node — the kernel's ``base_knobs`` (its
        ``S_*`` structural identity) merged with every ``fork.knobs`` delta from
        the root down to ``node``. A branch pins its level slice; a leaf carries
        the complete row, so deeper nodes hold a superset. This is the
        (partial-or-full) feature input the prior featurizes. A RESOLVED
        ancestor's pending fork is gone (``resolve`` drops it) — its delta is
        read from ``LazyCandidate.resolved_knobs`` instead, so descendants of
        a resolved branch keep the full feature prefix (else a structural
        branch's continuation would score as a knob-less generic row against
        its fully-knobed unresolved sibling)."""
        chain: list[dict] = []
        cur: SearchNode | None = node
        while cur is not None and cur.candidate is not None:
            fork = cur.candidate.fork
            knobs = fork.knobs if fork is not None else cur.candidate.resolved_knobs
            if knobs:
                chain.append(knobs)
            cur = cur.parent
        merged: dict = dict(self._base_knobs)
        for knobs in reversed(chain):
            merged.update(knobs)
        return merged

    def _collect_rows(self) -> list[tuple[dict, float]]:
        """Value-of-position training rows from the live tree: every node with
        a benched descendant (``visits > 0`` and ``best_reward > 0``) — leaves
        *and* branches — labeled with the best (min) median latency µs over its
        subtree (``1/best_reward``; the prior regresses on latency, and the
        reward conversion lives in the MCTS selection loop). Re-read each refit
        since labels only fall.

        A directly-benched leaf uses its ``realized_knobs`` (the FULL config);
        a branch (no realized knobs of its own) uses its partial fork-prefix
        (``_node_knobs``) — the value-of-position label still rides on it."""
        rows: list[tuple[dict, float]] = []
        stack = list(self.tree.root.children)
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            if node.candidate is None or node.visits == 0 or node.best_reward <= 0:
                continue
            knobs = node.realized_knobs if node.realized_knobs is not None else self._node_knobs(node)
            rows.append((knobs, 1.0 / node.best_reward))
        return rows

    def _node_key(self, feats: dict, op_sig: str, context_key: str, gpu: str) -> str:
        """Identity of a node within its operation *on its hardware*: a digest over the
        deploy regime (``context_key``), the card identity (``gpu`` —
        ``Context.hardware_id``), the op's ``S_*`` signature (``op_sig``), and the
        canonical tunable-knob set. ``gpu`` is folded in because ``context_key`` (cc +
        opt only) can't tell same-die SKUs apart (H100 vs H200), so without it their
        rows would collide and keep-min would silently drop one card's data. ``S_*`` /
        ``H_*`` features are excluded from the set — already folded via ``op_sig`` /
        ``context_key`` (and ``gpu``) — so the key is the within-op node identity.
        ``str()``-ified values mirror :meth:`_o3_sig` so non-string knob values
        stay stable, and the sorted tuple keeps :func:`digest`
        (order-sensitive) deterministic."""
        tun = tuple(sorted((k, str(v)) for k, v in feats.items() if not k.startswith(("S_", "H_"))))
        return digest(context_key, gpu, op_sig, tun)

    def _collect_node_records(
        self, *, context_key: str, op_sig: str, gpu: str = "", run_id: str = "", o3_context_key: str | None = None
    ) -> list[NodeRow]:
        """Post-search tree walk producing keyed, parent-linked :class:`NodeRow`
        records for :meth:`SearchDB.record_nodes` — the persistent/keyed/deduped
        sibling of :meth:`_collect_rows` (which feeds the prior's in-memory
        reservoir).

        Pre-order descent from the top forks (the sentinel root is skipped); each
        node passing the same ``visits > 0 and best_reward > 0`` guard as
        ``_collect_rows`` emits an ``ok`` row: ``features`` is the full dict the
        prior sees (``realized_knobs`` on a benched leaf — incl. deterministically-
        stamped knobs — else the partial fork-prefix from ``_node_knobs``),
        ``value_us`` the value-of-position ``1/best_reward``, ``visits`` the benched-
        descendant count (the label's confidence weight), ``is_leaf`` whether the
        node was directly benched, and ``variance`` / ``n_samples`` the leaf's OWN
        bench stats — a leaf whose subtree found a faster descendant keeps
        ``value_us`` = min-over-subtree while the stats describe its direct bench.

        A directly-benched leaf whose bench FAILED (``bench_status == 'bench_fail'``,
        reward 0 — previously invisible) is now emitted too, as a ``bench_fail`` row
        with the watchdog's sentinel latency as ``value_us``: the negative examples a
        search prior needs. Fail rows are leaf-only and never value anchors — they
        skip the monotone assert, don't update ``parent_value``, and children keep
        the inherited ``parent_key``; a branch whose descendants ALL failed stays
        unrecorded (``best_reward == 0``).

        A leaf carrying an ``o3_us`` (the deployable :data:`O3_NVCC_FLAGS` re-bench
        of a near-best config — ``observe_o3``) additionally emits an **-O3-regime
        row** when ``o3_context_key`` is given: keyed under that context (so it never
        collides with its -O1 twin), features stamped ``H_opt=3.0`` (the reservoir
        convention), ``parent_key=None`` (a regime re-measurement, not part of this
        tree — it never enters a fork sibling group), ``visits=1`` per re-bench, and
        no variance/n_samples (the re-bench returns a bare median).

        ``parent_key`` is the *nearest emitted ok ancestor*'s ``node_key`` (a skipped
        intermediate node passes its own inherited parent down), so it always
        references a recorded row — true ancestry from the live ``parent`` edge,
        not knob-subset inference (which a leaf's extra stamped knobs would break).
        Asserts the monotone ``parent.value_us <= child.value_us`` invariant — it
        holds because ``record_terminal`` max-propagates ``best_reward`` up the
        chain, transitively across skipped nodes.

        Deterministic single-option steps can stamp no knob delta, so a child's
        accumulated knob set (and thus ``node_key``) can equal its parent's;
        duplicates within the batch are collapsed to one row per key — preferring
        the directly-benched (``is_leaf``) row, which carries the bench stats, and
        taking the max (not sum) of the duplicates' ``visits`` so
        ``record_nodes``'s SUM accumulation never double-counts within one run."""
        out: dict[str, NodeRow] = {}

        def emit(row: NodeRow) -> None:
            prev = out.get(row.node_key)
            if prev is None:
                out[row.node_key] = row
                return
            # Within-batch duplicate (empty knob-delta chain): keep one row per key.
            # An ``ok`` row always beats a ``bench_fail`` (record_perf's policy);
            # among same-status rows prefer the directly-benched one (it carries the
            # bench stats) with the min value. Either way the survivor takes the
            # first-seen depth/parent and the max (not sum) of the visits, so
            # ``record_nodes``'s SUM accumulation never double-counts one run.
            keep = prev
            if (row.status, prev.status) == ("ok", "bench_fail") or (row.status == prev.status and row.is_leaf and not prev.is_leaf):
                keep = row
            value_us = min(prev.value_us, row.value_us) if prev.status == row.status else keep.value_us
            out[row.node_key] = replace(
                keep, visits=max(prev.visits, row.visits), depth=prev.depth, parent_key=prev.parent_key, value_us=value_us
            )

        def visit(node: SearchNode, parent_key: str | None, parent_value: float | None, depth: int) -> None:
            nk = parent_key
            if node.candidate is not None:
                is_leaf = node.realized_knobs is not None
                stats = node.bench_stats if is_leaf else None
                if node.visits > 0 and node.best_reward > 0:
                    feats = node.realized_knobs if is_leaf else self._node_knobs(node)
                    value_us = 1.0 / node.best_reward
                    assert parent_value is None or value_us >= parent_value - 1e-9, "value-of-position not monotone up the tree"
                    nk = self._node_key(feats, op_sig, context_key, gpu)
                    emit(
                        NodeRow(
                            node_key=nk,
                            parent_key=parent_key,
                            context_key=context_key,
                            op_sig=op_sig,
                            features=feats,
                            value_us=value_us,
                            depth=depth,
                            gpu=gpu,
                            visits=node.visits,
                            is_leaf=is_leaf,
                            variance=stats.variance if stats is not None else None,
                            n_samples=stats.n_samples if stats is not None else None,
                            status="ok",
                            run_id=run_id,
                        )
                    )
                    parent_value = value_us
                    if is_leaf and node.o3_us is not None and o3_context_key is not None:
                        feats_o3 = {**node.realized_knobs, "H_opt": 3.0}
                        emit(
                            NodeRow(
                                node_key=self._node_key(feats_o3, op_sig, o3_context_key, gpu),
                                parent_key=None,
                                context_key=o3_context_key,
                                op_sig=op_sig,
                                features=feats_o3,
                                value_us=node.o3_us,
                                depth=depth,
                                gpu=gpu,
                                visits=1,
                                is_leaf=True,
                                status="ok",
                                run_id=run_id,
                            )
                        )
                elif is_leaf and node.bench_status == "bench_fail" and stats is not None:
                    # Sentinel latency from the failed bench; NOT a value anchor — no
                    # assert, no parent_value update, children keep the inherited nk.
                    emit(
                        NodeRow(
                            node_key=self._node_key(node.realized_knobs, op_sig, context_key, gpu),
                            parent_key=parent_key,
                            context_key=context_key,
                            op_sig=op_sig,
                            features=node.realized_knobs,
                            value_us=stats.median,
                            depth=depth,
                            gpu=gpu,
                            visits=node.visits,
                            is_leaf=True,
                            variance=stats.variance,
                            n_samples=stats.n_samples,
                            status="bench_fail",
                            run_id=run_id,
                        )
                    )
            for child in node.children:
                visit(child, nk, parent_value, depth + 1)

        for child in self.tree.root.children:
            visit(child, None, None, 1)
        return list(out.values())

    def _should_stop(self) -> bool:
        if self.stop_reason is not None:
            return True
        visits = self.tree.root.visits
        if visits == 0:
            return False
        if self._max_visits is not None and visits >= self._max_visits:
            best_us = 1.0 / self._best_reward if self._best_reward > 0 else float("inf")
            self.stop_reason = f"max_visits ({visits} reached, best {best_us:.2f} us)"
            return True
        if self.tree.best_reward > self._best_reward:
            self._best_reward = self.tree.best_reward
            self._visits_at_best = visits
        stagnant = visits - self._visits_at_best
        if stagnant >= self._patience:
            best_us = 1.0 / self._best_reward if self._best_reward > 0 else float("inf")
            self.stop_reason = f"patience ({stagnant} stagnant, best {best_us:.2f} us)"
            return True
        return False
