"""The candidate-pool draw — determinism, exact membership, and the exact count beside the rows.

Three properties, and each is load-bearing somewhere else:

- the draw is a pure function of the stream and ``(sample size, seed)`` — the reservoir never reads
  a row, and the schedule walk's leaf order is deterministic — so two byte-identical pools draw
  byte-identical samples. That is what keeps ``emmy fit`` reproducible and what keeps two goldens
  over one pool merging into one training case rather than two;
- a signature in ``keep`` survives the draw wherever it sits in the stream. The fit locates a golden
  by scanning its pool for that signature and DROPS the golden on a miss, and ``eval golden`` reads
  the same miss as a pin or dtype mismatch — a draw that could lose the row would turn a real defect
  signal into noise;
- a pool no larger than the draw is taken whole, so the sampled and unsampled paths agree wherever
  sampling has nothing to do — and the exact count comes back beside the rows either way.
"""

from __future__ import annotations

import pytest

from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.pool import Candidates, PoolSample


@pytest.fixture
def by_id(monkeypatch):
    """Row identity as the row's own index, so a test can name the row it means. Patched on the
    ``features`` module, which is how :meth:`PoolSample.take` reaches it."""
    monkeypatch.setattr(features, "tile_signature", lambda row: row["id"])


def _space(n: int) -> list[dict]:
    return [{"id": i} for i in range(n)]


def test_the_draw_reads_only_the_stream_its_size_and_its_seed() -> None:
    drawn = PoolSample(rows=10, seed=0).take(_space(500))
    assert drawn == PoolSample(rows=10, seed=0).take(iter(_space(500))), "a list and its stream are one pool"
    assert drawn != PoolSample(rows=10, seed=1).take(_space(500)), "the seed must move the draw"
    ids = [row["id"] for row in drawn.rows]
    assert len(ids) == len(set(ids)) == 10, "a draw is without replacement"
    assert sorted(ids) == ids, "retained in stream order"
    assert all(0 <= i < 500 for i in ids)
    assert drawn.total == 500, "the exact count rides beside the rows — the reservoir knows it when the stream ends"


def test_the_same_stream_twice_draws_the_same_rows() -> None:
    sample = PoolSample(rows=10, seed=7)
    assert sample.take(_space(500)) == sample.take(_space(500))
    assert PoolSample(rows=10, seed=7).take(_space(500)) == sample.take(_space(500)), "and so does an equal sample"


def test_a_kept_signature_survives_wherever_it_sits(by_id) -> None:
    """Row 0 and row N-1 — the two positions a draw is least likely to reach by accident."""
    space = _space(500)
    sample = PoolSample(rows=10, seed=0, keep=frozenset({0, 499}))
    ids = [row["id"] for row in sample.take(space).rows]
    assert 0 in ids and 499 in ids
    assert sorted(ids) == ids, "retained in stream order, drawn and kept alike"
    unkept = [row["id"] for row in PoolSample(rows=10, seed=0).take(space).rows]
    assert set(unkept) <= set(ids), "the keep-set ADDS to the draw; it never replaces it"


def test_every_candidate_is_visited_even_though_most_are_dropped(by_id) -> None:
    """The keep scan is what makes membership exact: an absent signature must read as absent
    because the pool really does not contain it, not because the draw missed it."""
    space = _space(500)
    assert PoolSample(rows=10, keep=frozenset({12345})).take(space) == PoolSample(rows=10).take(space)


def test_a_pool_no_larger_than_the_draw_is_taken_whole() -> None:
    space = _space(10)
    assert PoolSample(rows=10).take(space) == Candidates(space, 10)
    assert PoolSample(rows=99).take(space) == Candidates(space, 10)
    assert PoolSample(rows=0).take(space) == Candidates(space, 10), "0 is 'enumerate everything', the live and unsampled default"


def test_the_cache_identity_ignores_the_size_sink_and_the_keep_set_order() -> None:
    """The pool memo keys on this, so it must be stable across processes: a ``frozenset``'s
    iteration order follows randomized string hashing, and its ``repr`` would key one sample two
    ways on two runs."""
    a = PoolSample(rows=10, seed=1, keep=frozenset({("x",), ("y",)}))
    b = PoolSample(rows=10, seed=1, keep=frozenset({("y",), ("x",)}))
    b.totals["some-pool"] = 999
    assert a.key == b.key and a == b, "a size sink is not part of a sample's identity"
    assert a.key != PoolSample(rows=10, seed=2, keep=a.keep).key
    assert a.key != PoolSample(rows=11, seed=1, keep=a.keep).key
    assert a.key != PoolSample(rows=10, seed=1, keep=frozenset({("x",)})).key
