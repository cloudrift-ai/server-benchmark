"""Single-player MCTS for ``emmy tune`` with max-reward propagation and
**PUCT** selection — the online
:class:`~emmy.compiler.pipeline.search.prior.Prior` is the *only* selection
signal (greedy and the ``+∞``-unvisited UCB rule are gone).

    select   — descend from root, picking at each level
               ``argmax_c [ Q(c) + c · P(c) · √(N_parent+1) / (1+N_c) ]``
               where ``Q = best_reward / global_best`` and ``P`` is
               ``Prior.policy`` over the sibling set — each sibling's predicted
               preference relative to the best of them, one batched call per
               fork. No ``+∞``-unvisited rule → no forced breadth: a
               confidently-bad sibling gets a small ``P`` and is skipped. A cold
               or absent prior gives a uniform ``P = 1`` (PUCT still explores via
               the exploration term; a single-shot compile with no prior descends
               emission-order). Live-count filtering skips drained subtrees.
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

from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.schedule import ReducePlan, Workers
from emmy.compiler.pipeline.knob import (
    canonical_row_key,
    context_view,
    decision_view,
    family_of,
    stamp_schedule_families,
    tuning_knob_items,
)
from emmy.compiler.pipeline.search.candidate import LazyCandidate
from emmy.compiler.pipeline.search.db import NodeRow, PerfStats, node_key
from emmy.compiler.pipeline.search.pins import unreproducible_pin_flag
from emmy.compiler.pipeline.search.policy.base import Search
from emmy.compiler.pipeline.search.policy.terminal_bench import bench_terminal_async

if TYPE_CHECKING:
    from emmy.compiler.pipeline.search.prior import Prior


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
    # Number of CudaOps in the directly-benched terminal. Usually one for an
    # inner kernel search, but lowering can legitimately materialize several
    # CUDA kernels (for example a split reduction + combine). A candidate-file
    # annotation is only unambiguous in the one-CudaOp case.
    realized_cuda_ops: int | None = field(default=None, repr=False)
    # Decision rows from the directly measured per-kernel receipts. Structural winners use them to
    # reject a parent row whose ordinary schedule pins describe a different
    # independently tuned child than the terminal actually measured.
    realized_cuda_knobs: list[dict] | None = field(default=None, repr=False)


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
        max_measurements: int | None = None,
        prior_model: Prior | None = None,
        base_knobs: dict | None = None,
    ) -> None:
        super().__init__()
        self.tree = tree if tree is not None else SearchTree()
        self._ucb_c = ucb_c
        # ε-greedy exploration: with probability ``explore_eps`` a selection step
        # descends a UNIFORMLY RANDOM live child instead of the PUCT argmax. PUCT
        # alone is deterministic — a tie (cold prior → uniform ``P``) always goes
        # to the first-in-list (= the enumeration's emission order, which means
        # nothing), so each fork is visited once and takes whichever child the walk
        # emitted first; a binary fork like ``WARPSPEC`` then never benches its
        # second branch even when that's the real win. ε-randomness makes ~half the
        # visits to such a fork take the other branch, so tuning finds good configs
        # WITHOUT relying on emission order at all. ``0.0`` (the default) restores deterministic
        # PUCT — kept for the unit tests and single-shot compile. Seeded for
        # reproducibility (vary ``seed`` per op/run upstream, not via wall clock).
        self._explore_eps = explore_eps
        self._rng = random.Random(seed)
        self._patience = patience
        self._max_visits = max_visits
        self._max_measurements = max_measurements
        self.measurements = 0
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

    def note_bench(self, *, measured: bool) -> None:
        """Count only terminals that reached the live backend.

        A DB cache hit remains useful evidence and is observed by the tree, but
        does not spend ``max_measurements``. This keeps a resumed tune's candidate
        budget about new measurements rather than replay work.
        """
        if measured:
            self.measurements += 1

    def prepare_ctx(self, ctx):
        """Policy-owned ctx setup, applied by the engine's run construction: the tune search is
        exempt from the strict knob-pin validator — it explores tier-foreign forks and steers
        heterogeneous multi-op graphs with a union pin vector (each op takes its tier's subset).
        A per-op contradiction is a pruned branch here, not the loud user error the greedy
        compile wants."""
        return replace(ctx, validate_pins=False) if ctx.validate_pins else ctx

    async def evaluate(self, token: object | None, cand, *, backend, db) -> None:
        """Value one terminal the engine's loop yielded — the whole of what a terminal is worth
        is policy: bench every CudaOp (or serve the cache / stub), persist the per-kernel
        ``perf`` / inventory / lowering rows, and feed the tree and the prior (:meth:`observe`).
        Every bench is taken in the deployable regime, so a terminal earns exactly one
        measurement. The engine awaits this and nothing else."""
        stats, status, measured, per_kernel = await bench_terminal_async(cand, backend=backend, db=db)
        self.note_bench(measured=measured)
        self.observe(token, stats, status, candidate=cand, kernels=per_kernel)

    def observe(
        self, token: object | None, stats: PerfStats, status: str, candidate: object | None = None, kernels: list | None = None
    ) -> None:
        self.last_stats = stats
        self.last_status = status
        assert isinstance(token, SearchNode), f"TuningSearch.observe needs the terminal's pop token, got {type(token).__name__}"
        # Record the benched leaf's FULL realized knobs (from the resolved graph),
        # not just its fork-prefix — so knobs stamped at deterministic lowering
        # steps (FK / BK / STAGE / …) reach the prior. Falls back to the
        # fork-prefix when no candidate is supplied.
        token.realized_knobs = self._realized_knobs(candidate) if candidate is not None else self._node_knobs(token)
        token.realized_cuda_ops = self._realized_cuda_op_count(candidate)
        token.realized_cuda_knobs = [dict(decision_view(knobs)) for knobs, _stats, _status in kernels] if kernels is not None else None
        token.bench_stats = stats
        token.bench_status = status
        reward = (1.0 / stats.median) if status == "ok" and stats.median > 0 else 0.0
        prev_best = self.tree.best_reward
        self.tree.record_terminal(token, reward)
        self.last_improved_best = status == "ok" and self.tree.best_reward > prev_best
        if self.prior_model is not None:
            # Train on the rows that actually earned a latency: the terminal's own when it lowered
            # to one kernel, else one per KERNEL (its own decisions, its own measured µs, under
            # this run's host regime). The Σ is the tree's reward; it is not any single row's
            # label, and a terminal a structural fork made several kernels of has no row of its
            # own to give the prior.
            # The model itself is fixed during a run — it refits in batches between ops (see
            # ``Prior``), not per bench — so there is nothing to refit here.
            if token.realized_knobs is not None:
                self.prior_model.record_bench(token.realized_knobs, stats.median, status)
            else:
                regime = context_view(self._base_knobs)
                for knobs, kstats, st in kernels or ():
                    self.prior_model.record_bench({**regime, **knobs}, kstats.median, st)

    def _realized_knobs(self, candidate: object) -> dict | None:
        """The terminal's ONE knob row — the kernel's ``base_knobs`` (``S_*`` identity + ``H_*``
        regime) merged with the realized op ``knobs`` off the resolved graph (every tunable knob,
        including deterministically-stamped ones that ``_node_knobs`` can't see), or ``None`` when
        the terminal has no single row.

        A terminal is a Σ over the kernels it lowered to. When it lowered to ONE, that kernel's row
        earned the whole measurement and the merge is exact. When a structural fork made it
        several — a cut, a cross-CTA split — the kernels carry DIFFERENT decisions for the same
        families, and merging them fabricates a row no kernel realized (last write wins: the
        finalize's OFF ``WORK`` used to overwrite the partial's real one in the row that fed the
        online prior, the node table and the tune winner). There is no such row, so this answers
        ``None`` and the per-KERNEL rows — which the bench hands over intact — carry the training
        signal instead."""
        graph = getattr(candidate, "graph", None)
        if graph is None:
            return dict(self._base_knobs)
        merged: dict = dict(self._base_knobs)
        decided: dict = {}
        for node in graph.nodes.values():
            knobs = getattr(node.op, "knobs", None)
            if not knobs:
                continue
            for k, v in decision_view(knobs).items():
                if k in decided and decided[k] != v:
                    return None  # two kernels, two decisions — no single row to attribute the Σ to
                decided[k] = v
            merged.update(knobs)
        return merged

    @staticmethod
    def _realized_cuda_op_count(candidate: object | None) -> int | None:
        """Number of CUDA kernels in a directly observed terminal."""
        graph = getattr(candidate, "graph", None)
        if graph is None:
            return None
        return sum(isinstance(node.op, CudaOp) for node in graph.nodes.values())

    @staticmethod
    def _structural_row(knobs: dict | None) -> dict[str, str] | None:
        """The exact kernel-set-changing replay row in ``knobs``, if any."""
        if not knobs:
            return None
        row = dict(tuning_knob_items(knobs))
        cuts = {key: value for key, value in row.items() if family_of(key) == "PLACE" and value == "cut"}
        if cuts:
            return cuts
        work = Workers.parse(row.get("WORK"))
        if any(ReducePlan.parse(value, work).needs_split for key, value in row.items() if family_of(key) == "REDUCE"):
            return stamp_schedule_families(row)
        return None

    def _structural_replay_row(self, node: SearchNode) -> dict[str, str] | None:
        """The first exact kernel-set-changing row on ``node``'s path."""
        path: list[SearchNode] = []
        cur: SearchNode | None = node
        while cur is not None:
            path.append(cur)
            cur = cur.parent
        for cur in reversed(path):
            knobs = getattr(cur.candidate, "resolved_knobs", None)
            structural = self._structural_row(knobs)
            if structural is not None:
                return structural
        return None

    def _best_terminal_node(self) -> SearchNode | None:
        """The fastest directly observed successful terminal node."""
        best: tuple[float, bool, tuple, SearchNode] | None = None
        stack = list(self.tree.root.children)
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            stats = node.bench_stats
            if node.bench_status != "ok" or stats is None or stats.median <= 0:
                continue
            key = canonical_row_key(node.realized_knobs) if node.realized_knobs is not None else ()
            candidate = (float(stats.median), node.realized_knobs is None, key, node)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        return best[3] if best is not None else None

    def best_realized(self, *, validated_input_route: dict | None = None) -> tuple[dict, float, int | None, bool] | None:
        """Return an exact replay row for the fastest directly observed successful terminal.

        Unlike ``tree.best_reward``, this preserves the knobs and direct cost as
        one indivisible observation. Callers must not reconstruct the knobs from
        a later greedy/deploy replay, whose evidence hierarchy can select a
        different configuration. When the winner changes the kernel set, its
        first structural row is the replay contract and the independently
        scheduled pieces remain separate targets. If neither that row nor one
        terminal knob row exists, never fall back to a slower representable
        sibling. An authoritative ``validated_input_route`` may supply the row
        when pinning made the structural choice deterministic rather than a tree
        node, but only for a conflicting multi-CUDA terminal. Equal medians
        prefer a representable row and then break deterministically by its
        canonical key.
        """
        node = self._best_terminal_node()
        if node is None:
            return None
        structural = self._structural_replay_row(node)
        # A compatible multi-CUDA terminal has one exact merged row even though
        # its kernel-set-changing choice never appeared as a search-tree fork.
        if structural is None and (node.realized_cuda_ops or 0) > 1:
            structural = self._structural_row(node.realized_knobs)
        if structural is None and node.realized_knobs is None and (node.realized_cuda_ops or 0) > 1:
            structural = self._structural_row(validated_input_route)
        if structural is not None:
            place_only = all(family_of(key) == "PLACE" for key in structural)
            if not place_only and (
                not node.realized_cuda_knobs
                or unreproducible_pin_flag(structural, node.realized_cuda_knobs, reject_conflicts=True) is not None
            ):
                return None
            return structural, float(node.bench_stats.median), node.realized_cuda_ops, True
        if node.realized_knobs is None:
            return None
        return dict(node.realized_knobs), float(node.bench_stats.median), node.realized_cuda_ops, False

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

    def _prior_policy(self, children: list[SearchNode]) -> list[float]:
        """The prior's ``P`` for each child — ONE batched call over the whole sibling
        set, so a vectorized model pays its per-call overhead once per fork rather
        than once per candidate.

        A child the prior cannot speak about takes the uniform ``1.0`` that keeps the
        exploration term driving breadth: no model attached, or a node with no candidate
        (the root sentinel). Such a node is EXCLUDED from the call rather than passed an
        empty knob dict — an empty row is a row the model has an opinion about (the
        linear model scores it its neutral value), and normalizing it against real
        siblings would rank a sentinel among them. A cold model needs no special case
        here: its all-zero predictions come back uniform from ``Prior.policy`` itself."""
        if self.prior_model is None:
            return [1.0] * len(children)
        live = [i for i, c in enumerate(children) if c.candidate is not None]
        if not live:
            return [1.0] * len(children)
        out = [1.0] * len(children)
        for i, p in zip(live, self.prior_model.policy([self._node_knobs(children[i]) for i in live]), strict=True):
            out[i] = p
        return out

    def _select(self, children: list[SearchNode], parent: SearchNode) -> SearchNode:
        """PUCT is the *only* selection rule — the prior is the sole signal.

            score(c) = Q(c) + c_ucb · P(c) · √(N_parent + 1) / (1 + N_c)

        where ``Q = best_reward / global_best`` (``0`` for an unvisited child) and
        ``P`` is ``Prior.policy`` over this sibling set: each sibling's preference
        relative to the best of them, so the best scores ``1.0`` and one the model
        prices 10× slower scores ``0.1``. A confidently-bad sibling therefore gets a
        small ``P`` → tiny exploration term → it is deprioritized rather than
        force-visited. The prior is always consulted — the composite prior answers
        with the online model once trained and the offline one cold, so even a fresh
        ``tune`` is prior-guided, not uniform. Only where there is NO usable
        prediction does ``P`` fall back to a uniform ``1`` so the exploration term
        still drives breadth.

        ``P`` is normalized within the SIBLING SET, not against the tree's
        ``global_best``: a fork is a choice among its own children, ``global_best``
        is a moving target set elsewhere in the tree, and the offline prior's scores
        are not µs at all — pushing their raw magnitude through ``1/û`` is what left
        the cold policy meaningless.

        ``c_ucb`` is ``--ucb-c``. With ``explore_eps > 0`` (tune, opt-in) a fraction
        of steps instead descend a uniformly random live child (ε-greedy); off by
        default so a single-shot compile / the unit tests stay deterministic. NOTE: a
        *random tie-break* under a cold prior was tried and reverted — it discarded
        the heuristic ordering and regressed fp16 tuning ~2×; exploration must
        perturb the prior order, not replace it."""
        if self._explore_eps and self._rng.random() < self._explore_eps:
            return self._rng.choice(children)
        global_best = self.tree.best_reward or 1.0
        sqrt_parent = math.sqrt(parent.visits + 1)
        policy = self._prior_policy(children)
        best, best_v = children[0], float("-inf")
        for c, p in zip(children, policy, strict=True):
            q = (c.best_reward / global_best) if c.visits > 0 else 0.0
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
        (``_node_knobs``) — the value-of-position label still rides on it. A leaf that was benched
        but has NO single row (a structural fork made it several kernels with different decisions)
        contributes nothing here: its measurement was already attributed per kernel at
        :meth:`observe`, and its fork-prefix would merge the pieces' rows into one that no kernel
        realized — the fabrication this whole path exists to avoid."""
        rows: list[tuple[dict, float]] = []
        stack = list(self.tree.root.children)
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            if node.candidate is None or node.visits == 0 or node.best_reward <= 0:
                continue
            if node.bench_stats is not None and node.realized_knobs is None:
                continue
            knobs = node.realized_knobs if node.realized_knobs is not None else self._node_knobs(node)
            rows.append((knobs, 1.0 / node.best_reward))
        return rows

    def _collect_node_records(
        self,
        *,
        context_key: str,
        op_sig: str,
        gpu: str = "",
        run_id: str = "",
        validated_input_route: dict | None = None,
    ) -> list[NodeRow]:
        """Post-search tree walk producing keyed, parent-linked :class:`NodeRow`
        records for :meth:`SearchDB.record_nodes` — the persistent/keyed/deduped
        sibling of :meth:`_collect_rows` (which feeds the prior's in-memory
        reservoir).

        ``validated_input_route`` is an authoritative proposal row whose
        realized-pin check already passed. When that row is structural and the
        directly measured winner is a conflicting multi-CUDA terminal with no
        structural node on its path, the walk records the original Loop parent
        and its measured structural child. This is the pinned full-fork case:
        the input route was applied deterministically instead of becoming an
        ordinary search-tree node.

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
        structural_leaf = self._best_terminal_node()
        structural_input = self._structural_row(validated_input_route)
        if (
            structural_leaf is None
            or structural_leaf.realized_knobs is not None
            or (structural_leaf.realized_cuda_ops or 0) <= 1
            or self._structural_replay_row(structural_leaf) is not None
        ):
            structural_input = None

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
                is_leaf = node.bench_stats is not None
                stats = node.bench_stats if is_leaf else None
                # A benched leaf with no single row (several kernels with different
                # decisions) has nothing to key a node record on — see :meth:`_collect_rows`.
                skip = is_leaf and node.realized_knobs is None
                if node is structural_leaf and structural_input is not None and node.visits > 0 and node.best_reward > 0:
                    value_us = 1.0 / node.best_reward
                    route_parent = parent_key
                    route_depth = depth
                    if route_parent is None:
                        base_features = dict(self._base_knobs)
                        route_parent = node_key(context_key, gpu, op_sig, base_features)
                        emit(
                            NodeRow(
                                node_key=route_parent,
                                parent_key=None,
                                context_key=context_key,
                                op_sig=op_sig,
                                features=base_features,
                                value_us=value_us,
                                depth=depth,
                                gpu=gpu,
                                visits=node.visits,
                                is_leaf=False,
                                status="ok",
                                run_id=run_id,
                            )
                        )
                        route_depth += 1
                    route_features = {**self._base_knobs, **structural_input}
                    nk = node_key(context_key, gpu, op_sig, route_features)
                    emit(
                        NodeRow(
                            node_key=nk,
                            parent_key=route_parent,
                            context_key=context_key,
                            op_sig=op_sig,
                            features=route_features,
                            value_us=value_us,
                            depth=route_depth,
                            gpu=gpu,
                            visits=node.visits,
                            is_leaf=True,
                            variance=stats.variance if stats is not None else None,
                            n_samples=stats.n_samples if stats is not None else None,
                            status="ok",
                            run_id=run_id,
                        )
                    )
                elif node.visits > 0 and node.best_reward > 0 and not skip:
                    feats = node.realized_knobs if is_leaf else self._node_knobs(node)
                    value_us = 1.0 / node.best_reward
                    assert parent_value is None or value_us >= parent_value - 1e-9, "value-of-position not monotone up the tree"
                    nk = node_key(context_key, gpu, op_sig, feats)
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
                elif is_leaf and node.bench_status == "bench_fail" and stats is not None and node.realized_knobs is not None:
                    # ``skip`` above sends an unrealized leaf here, so the same
                    # nothing-to-key-on rule has to hold: a variant that bench-failed before
                    # its knobs were realized (a run-stage timeout, a missing bench input)
                    # carries ``realized_knobs is None`` and cannot be keyed at all.
                    # Sentinel latency from the failed bench; NOT a value anchor — no
                    # assert, no parent_value update, children keep the inherited nk.
                    emit(
                        NodeRow(
                            node_key=node_key(context_key, gpu, op_sig, node.realized_knobs),
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
        if self._max_measurements is not None and self.measurements >= self._max_measurements:
            best_us = 1.0 / self._best_reward if self._best_reward > 0 else float("inf")
            self.stop_reason = f"max_measurements ({self.measurements} reached, best {best_us:.2f} us)"
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
