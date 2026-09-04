# Perf suite architecture

`tests/perf/` measures Emmy against PyTorch on the GPU. It is gated by the `perf` marker, which the
root `tests/conftest.py` deselects for any `tests/` collection unless `-m perf` is passed — so
`make test` never runs it. Invoke it explicitly:

```bash
make bench-kernels                 # the stock lane (no tune DB)
make bench-kernels-tuned           # the same cases with EMMY_TUNE_DB pointed at a tuned DB
pytest tests/perf/ -m perf -v      # directly
```

## The case list is the realization corpus

There is no case table here. `test_corpus.py` parametrizes over every **closed**
`tests/compiler/realization/cases/**.yaml` whose declared capability matches the live card, and
benches each one pinned to the schedule that case authors.

That is deliberate, and it replaced a hand-curated list of twelve Qwen3-Embedding-0.6B layer-0
kernels. A curated list describes its programs in prose and shapes, so it drifts from the compiler
without anything noticing: this one still claimed "Emmy currently emits FP32 only" long after that
stopped being true, and its `EMMY_BK` parametrization pinned a knob that no longer existed, so both
of its cells ran the same compile. A corpus case stores the program itself, is replayed by the
correctness lane on every commit, and carries its own realization contract.

Those twelve kernels survive as `cases/qwen3emb/` — RMSNorm at the layer and per-head widths, the
Q and K/V projections, causal GQA attention, `o_proj` and `down_proj` with the residual fused into
the epilogue, and the gated MLP, each at seq 32 / 128 / 512. Read
`tests/compiler/realization/ARCHITECTURE.md` before adding a case.

## One bench, two answers

Each case is measured once and the result answers both questions the lane exists for:

- **How do we compare to PyTorch** — the row joins the session-end table, emmy beside eager and
  `torch.compile`, sorted worst-first.
- **Did we regress** — the same measurement is compared against the case's stored `latency:` entry
  for this card. Samples are taken lazily: a run inside the band settles the case, so only a case
  that looks slow pays for the extra runs.

**A regression is reported, never enforced.** A slower case prints a finding; the timing-refresh
workflow turns it into a labelled pull request a human accepts or declines. A lane that goes red
because one legitimate correctness fix cost latency is a lane nobody reads. What *does* fail is a
case that cannot be benched at all — that is a broken measurement, not a slow kernel.

A closed case with no stored latency for the live card is named at session end, with the command
that records it. Only on a card that can answer: an agent on a machine without it is never asked.

## How it runs

- `conftest.py` — the `bench_pair` fixture, the session-end table, the JSON dump to `.results/`,
  and the ECharts plot. Benching goes through `emmy run --golden <case> --realization <name> --bench
  --json` at `-O3`, one fresh process per case: reusing the CLI keeps the
  eager / `torch.compile` / emmy comparison and the ncu metrics on the code path users invoke
  directly, and reusing the corpus keeps one case inventory in the tree instead of two.
- `test_corpus.py` — the parametrized walker.
- `test_dit_comparison.py` — a separate subject: one fixed-shape Diffusers DiT block. It needs the
  `image` extra and `EMMY_RUN_DIT_PRETRAINED=1` for the real checkpoint, so normal CI never
  downloads it.

`EMMY_BENCH_NCU=1` adds an ncu pass with a curated metric set (occupancy, bank conflicts,
SM/DRAM/FMA throughput, registers), appended as extra table columns and nested per row in the JSON.
`EMMY_TUNE=1` switches the fixture to a tune-only path that populates the autotune DB and measures
nothing.

Bench subprocesses coordinate on a per-uid advisory GPU lock (`EMMY_GPU_LOCK`), so only the
kernel-launch phase serializes across xdist workers; trace, compile and dump-writing run unlocked.

**Never write a benchmark script.** `emmy run --bench --json` is the record every consumer reads,
and `--record` writes a measured latency back into a case. A missing capability there is a flag to
add, not a script to write.
