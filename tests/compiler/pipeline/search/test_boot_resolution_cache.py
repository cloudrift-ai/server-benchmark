"""Process-wide memoization of the two per-program deploy-resolution inputs — the
parsed online prior (``_load_prior_cached``) and the DB perf index
(``_db_measured_index``). A generative serve boot compiles ~96 programs and, before
these caches, each re-``json.loads``'d the 56 MB online checkpoint and re-scanned the
whole perf table — even though both are identical across programs (``structural_key``
folds only cc + nvcc flags, never the op shape). The memo must reuse within a process
yet invalidate when the backing file changes on disk."""

from __future__ import annotations

import os

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.db import PerfStats, SearchDB
from emmy.compiler.pipeline.search.policy import greedy

_SIG = {"S_ext_free_prod": 2048.0, "S_dtype_f32": 1.0}
_OK = PerfStats(median=47.1, min=46.0, max=48.0, mean=47.2, variance=0.1, n_samples=5)


def _record(db: SearchDB, ctx: Context, node: str, tile: str) -> None:
    db.record_perf(ctx.structural_key(), node, backend="cuda", status="ok", stats=_OK, knobs={**_SIG, "TILE": tile})


def _bump_mtime(path) -> None:
    """Force the file's mtime forward a full second so the (path, mtime) key differs
    regardless of the filesystem's mtime granularity — deterministic, not timing-based."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


def test_db_index_built_once_per_process(tmp_path, monkeypatch):
    greedy._DB_INDEX_CACHE.clear()
    ctx = Context.from_target((12, 0))
    db = SearchDB(path=tmp_path / "tune.db")
    _record(db, ctx, "op-a", "a")

    builds = []
    orig = greedy._db_measured_index_build
    monkeypatch.setattr(greedy, "_db_measured_index_build", lambda d, c: (builds.append(1), orig(d, c))[1])

    i1 = greedy._db_measured_index(db, ctx)
    i2 = greedy._db_measured_index(db, ctx)
    assert i1 is i2  # same object handed back — no rebuild
    assert len(builds) == 1  # the 527 MB scan ran once, not once per program


def test_db_index_cache_invalidates_when_db_changes(tmp_path):
    greedy._DB_INDEX_CACHE.clear()
    ctx = Context.from_target((12, 0))
    p = tmp_path / "tune.db"
    db = SearchDB(path=p)
    _record(db, ctx, "op-a", "a")
    sig = frozenset((k, str(v)) for k, v in _SIG.items())

    i1 = greedy._db_measured_index(db, ctx)
    assert len(i1[sig]) == 1

    _record(db, ctx, "op-b", "b")  # a second measured row lands
    _bump_mtime(p)  # its commit changes the file → key must differ
    i2 = greedy._db_measured_index(db, ctx)
    assert len(i2[sig]) == 2  # rebuilt and reflects the new row, not the stale index


def test_in_memory_db_bypasses_index_cache(tmp_path, monkeypatch):
    """An in-memory DB has no ``_path`` (its rows are process-local, not a shared
    file) — it must never be cached under a bogus key nor collide with a file DB."""
    greedy._DB_INDEX_CACHE.clear()
    ctx = Context.from_target((12, 0))
    db = SearchDB()  # :memory:
    _record(db, ctx, "op-a", "a")

    builds = []
    orig = greedy._db_measured_index_build
    monkeypatch.setattr(greedy, "_db_measured_index_build", lambda d, c: (builds.append(1), orig(d, c))[1])

    greedy._db_measured_index(db, ctx)
    greedy._db_measured_index(db, ctx)
    assert len(builds) == 2  # rebuilt each call — never cached
    assert not greedy._DB_INDEX_CACHE


def test_db_pick_scans_signatures_once_per_distinct_sig(monkeypatch):
    """Every candidate at a fork shares the offer op's ``S_*`` base, so one signature
    covers the whole set — the drift-path rescan of every index signature must happen
    once, not once per candidate (a real fork offers up to ~41.5k). The candidate sig
    here carries a key no index row has (#311-style vocabulary drift), so it misses the
    exact-hit fast path and takes the scanning branch."""
    index = {frozenset({("S_ext_free_prod", str(2048 + s)), ("S_dtype_f32", "1.0")}): [({"TILE": "t"}, 40.0 + s, True)] for s in range(8)}
    base = {"S_ext_free_prod": "2048", "S_dtype_f32": "1.0", "S_warp_eligible": "1.0"}
    rows = [{**base, "TILE": "t", "REDUCE": f"r{i}"} for i in range(50)]
    assert frozenset((k, str(v)) for k, v in rows[0].items() if k.startswith("S_")) not in index

    scans = []
    orig = greedy._sig_groups
    monkeypatch.setattr(greedy, "_sig_groups", lambda ix, sg: (scans.append(1), orig(ix, sg))[1])

    got = greedy._db_measured_pick(index, rows)
    assert len(scans) == 1  # one scan for the one distinct sig, not 50
    assert got is not None and got[1] == 40.0  # drift match still resolves to the cheapest row


def test_online_prior_parsed_once_per_mtime(tmp_path, monkeypatch):
    greedy._load_prior_cached.cache_clear()
    p = tmp_path / "online.json"
    p.write_text("{}")
    monkeypatch.setenv("EMMY_ONLINE_FILE", str(p))

    import emmy.compiler.pipeline.search.prior as prior_pkg

    parses = []
    orig = prior_pkg.load_prior
    monkeypatch.setattr(prior_pkg, "load_prior", lambda **kw: (parses.append(1), orig(**kw))[1])

    a = greedy._load_prior_safe()
    b = greedy._load_prior_safe()
    assert a is b  # same rehydrated prior reused across program compiles
    assert len(parses) == 1  # the 56 MB checkpoint parsed once

    _bump_mtime(p)  # tune rewrote the checkpoint → reparse
    greedy._load_prior_safe()
    assert len(parses) == 2
    greedy._load_prior_cached.cache_clear()  # don't leak the tmp-path prior to other tests
