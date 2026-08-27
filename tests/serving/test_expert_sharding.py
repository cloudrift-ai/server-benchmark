"""Tensor-parallel routed-expert sharding (no GPU, no vLLM).

A DeepSeek V4 rank cannot hold all 256 routed experts: one pipeline stage's experts are ~9.4 GB at
MXFP4 and ~17.7 GB expanded, against a 32 GB card that also carries attention, arenas and the KV
cache. So each rank loads one shard (``expert_range``) and the routed combine must produce the SAME
sum as the unsharded oracle once the ranks' partials are added — which is what the all-reduce does.

These tests pin that equality directly against ``combine_routed_experts``, the routing math serving
actually runs, with the shards' partials summed the way a tensor-parallel group sums them.
"""

from __future__ import annotations

import pytest


def _router_return(torch, tokens: int, experts: int, top_k: int, seed: int = 0):
    """One HF-router-shaped ``(scores, indices)`` pair over the GLOBAL expert space."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(tokens, experts, generator=generator)
    scores, indices = torch.topk(logits.softmax(dim=-1), top_k, dim=-1)
    return scores, indices


def test_sharded_combine_sums_to_the_unsharded_oracle():
    """Splitting the experts across ranks and adding the partials reproduces the single-rank result."""
    torch = pytest.importorskip("torch")

    from emmy.serving.gen_runner import combine_routed_experts, local_expert_slice

    tokens, hidden, experts, top_k, ranks = 6, 8, 8, 3, 4
    generator = torch.Generator().manual_seed(1)
    xn = torch.randn(tokens, hidden, generator=generator)
    weights = [torch.randn(hidden, hidden, generator=generator) for _ in range(experts)]
    gated = _router_return(torch, tokens, experts, top_k)

    oracle = combine_routed_experts(xn, gated, lambda e, rows: rows @ weights[e])

    per_rank = experts // ranks
    total = torch.zeros_like(oracle)
    for rank in range(ranks):
        lo = rank * per_rank
        expert_range = (lo, lo + per_rank)
        # Every rank runs the SAME router over the global expert space and keeps only its own hits;
        # its expert weights are indexed rank-locally, exactly as the loader stacks the shard.
        shard = weights[lo : lo + per_rank]
        run_shard = lambda e, rows, shard=shard: rows @ shard[e]  # noqa: E731 — bound per rank on purpose
        total += combine_routed_experts(xn, gated, run_shard, expert_range=expert_range)

    torch.testing.assert_close(total, oracle)
    assert local_expert_slice(0, (2, 5)) is None and local_expert_slice(3, (2, 5)) == 1


def test_a_rank_runs_only_its_own_experts():
    """A shard must never launch an expert it does not hold — that would index past its own table."""
    torch = pytest.importorskip("torch")

    from emmy.serving.gen_runner import combine_routed_experts

    tokens, hidden, experts, top_k = 4, 8, 8, 4
    xn = torch.randn(tokens, hidden)
    gated = _router_return(torch, tokens, experts, top_k, seed=2)
    launched: list[int] = []

    def run_expert(e, rows):
        launched.append(e)
        return rows

    combine_routed_experts(xn, gated, run_expert, expert_range=(2, 4))
    assert launched, "the shard ran no experts at all — the routing never reached it"
    assert all(0 <= e < 2 for e in launched), f"a rank launched non-local expert indices {sorted(set(launched))}"


def test_an_unrouted_shard_contributes_exactly_zero():
    """A rank whose experts win no token still returns a zero partial, so the all-reduce is safe."""
    torch = pytest.importorskip("torch")

    from emmy.serving.gen_runner import combine_routed_experts

    xn = torch.randn(3, 8)
    # Every token routes to experts 0 and 1, so the rank holding experts 6-7 has nothing to do.
    gated = (torch.full((3, 2), 0.5), torch.zeros(3, 2, dtype=torch.long) + torch.tensor([0, 1]))
    partial = combine_routed_experts(xn, gated, lambda e, rows: rows + 1.0, expert_range=(6, 8))

    assert partial.shape == xn.shape
    torch.testing.assert_close(partial, torch.zeros_like(partial))


def test_hash_routing_needs_the_steps_token_ids():
    """A hash router selects experts by token id; the router call must pass the ids through and
    refuse to route without them (silently routing on garbage would serve noise)."""
    pytest.importorskip("torch")

    from emmy.serving.gen_runner import EmmyGenRunner

    seen = []
    moe = {"hash": True, "layer": 7, "gate": lambda xn, ids: seen.append((xn, ids)) or ("logits", "scores", "indices")}
    xn, ids = object(), object()
    assert EmmyGenRunner._route(None, moe, xn, ids) == ("logits", "scores", "indices")
    assert seen == [(xn, ids)]
    with pytest.raises(RuntimeError, match="hash-routed"):
        EmmyGenRunner._route(None, moe, xn, None)

    plain = {"hash": False, "gate": lambda xn: ("l", "s", "i")}
    assert EmmyGenRunner._route(None, plain, xn, None) == ("l", "s", "i")
