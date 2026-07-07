# Golden seeding findings — RTX 4090 (sm_89): first rms_norm entries

- **Date / GPU / box:** 2026-07-07, NVIDIA GeForce RTX 4090 24 GB (CloudRift rental, driver 580.65.06, nvcc 12.9 via
  `CUDA_HOME=/usr/local/cuda`), emmy at `main` (71bd34f3) + this branch.
- **Scope:** NOT a full-dataset sweep — a seeding session adding the first `rms_norm` entries to `rtx4090_sm89.yaml`
  (the card had none; only the 5090 was seeded in #315). Shapes chosen from the gemma-4-12B RMSNorm tune session
  (`gemma-4-12b-rmsnorm-rtx4090-tune-findings.md`): hidden-size norm K=3840 (static + `.dynM`) and the per-head
  QK-norm geometry 4096×256.
- **Commands** (per shape, accumulating DB/prior — no `--clean`): `emmy tune -c "torch.nn.RMSNorm(K)(torch.randn(M,K))"
  [--dynamic seq_len@x:0]` then `emmy run --bench -c …` for the -O3 deployable numbers; validation via
  `emmy run --bench --golden NAME`. Logs: `_tune/golden-seed-rtx4090/` on the box. ~1 min/shape wall.
- **Tally:** 3 added (fresh seeds), 0 replaced, 0 worse.

## Per-shape outcome

| Shape | greedy µs | golden µs (A/B) | eager µs | vs eager | knobs | category |
|---|--:|--:|--:|--:|---|---|
| `rms_norm.k3840` (512×3840) | 17.7–18.4 | 17.7 | 7 | 2.5× slower | `REDUCE: b32` | added |
| `rms_norm.k3840.dynM` (hint 512) | 17.7 | 17.7 | 7 | 2.5× slower | `REDUCE: b32` | added |
| `gemma4_12b.qknorm.k256` (4096×256) | 4.5 | 4.4 | 5 | **1.1× faster** | `REDUCE: b32` | added |

`cublas_us` here is torch eager `nn.RMSNorm` (fp32), per the rms_norm golden convention. The k3840 ratio matches the
5090's recorded profile (its seeds also trail eager 2.7–4×), so the 0.40× is the current reduce-tier ceiling, not a
4090 regression. Greedy reproduces every golden with identical knobs (rank 1/8 in `eval variants`; the thread-serial
lane sits at rank 8/8, ~1800 µs, correctly rejected).

## Finding 1 — seeding cannot fix the fused-norm kernels: the coop fork is never offered (recognition gap)

The motivating problem — gemma-4-12B's fused per-head norm kernels at 626–1540 µs — did **not** move after seeding:
the greedy pick for `k_mean_linear_reduce_323fd5` is byte-identical pre/post (625 µs). Root cause (from the generated
CUDA): the fused kernels' mean-count and epsilon arrive as materialized 1-element buffers (`mean_2_count[0]`,
`add_3_c1[0]`), the RMSNorm `Reduction` lift never fires, and lowering emits **one thread per output row** (`_gid <
4096` guards off 127/128 threads). The standalone golden `gemma4_12b.qknorm.k256` proves the same geometry runs 4.5 µs
when recognized. Priors rank options; they cannot rank an option that doesn't exist. **Recommendation:** fix in the
loop-dialect lift (recognize buffer-fed count/eps as the RMSNorm statistic), then re-bench the three fused kernels —
expected ~100× on the norm epilogues. This is the highest-leverage item of the session.

## Finding 2 — the AI-floor integrity gate false-fires on dynamic rms_norm shapes

`run --bench --golden rms_norm.k3840.dynM` flags `impossible: implies 114 TFLOP/s > 83 device peak` for a 17.7 µs
measurement — while the static twin with the **identical latency and knobs** passes. The dyn-shape FLOP model is
overcounting (~2 GFLOP implied for a 512×3840 norm, ~250× the real work). Harmless here (the row still prints), but a
wrong-bench gate that cries wolf on valid rows will get ignored. **Recommendation:** fix the dynamic FLOP estimate in
the A/B gate before porting the gates into tune (Finding 3 depends on trusting them).

## Finding 3 — the e4bb2c broken-winner class reproduces cold: tune needs the integrity gates

An isolated `--clean --explore-eps 0.25` retune of `k_mean_linear_reduce_e4bb2c` (post-FFN norm + residual, 512×3840,
eager 635 µs / tcompile 385 µs) reproduced the pathology exactly: search best **18.15 µs at -O1 ranking**, -O3 re-bench
of that winner **~98 ms**, greedy `run --bench` aborts at the 10 s GPU budget (100 iters × 98 ms). So it is not prior
poisoning — the winner class is deterministically mis-measured at -O1 (18 µs is impossible for this memory-bound op:
wrong-bench class) and pathological at -O3. `eval failures` is blind to it: the bogus rows are "ok" rows. This kernel
is also in the Finding-1 recognition-gap family (buffer-fed `mean_6_count[0]` / `add_9_c1[0]`). **Recommendations:**
(a) port the golden A/B's wrong-answer + AI-floor gates into tune's winner selection (an 18 µs winner for a 5.7 MB
memory-bound op should be re-verified, not crowned); (b) the Finding-1 lift fix likely dissolves this kernel's bad
lane entirely; (c) until then, gemma-4-12B deploys on sm_89 carry a ~98 ms kernel per layer — do not ship.

## Finding 4 — rms_norm goldens don't feed the analytic-weight refit

`scripts/golden_knob_heuristics.py` handles Reduce / Pointwise / Matmul golden kinds and silently `continue`s past
`RmsNormGoldenConfig` (and Softmax), so the skill's "refit after recording new `.dynM` goldens" step is a no-op for
this session. Not urgent — the reduce-tier analytic pricing already ranks `b32` first on all three shapes (rank-1
picks) — but the refit script and the golden schema have drifted. **Recommendation:** either teach the script the
rms_norm/softmax kinds (they share the reduce regime) or document the skip in the script docstring.

## Workflow notes

- **Remote goldens flow works but is manual:** tune/bench on the card, hand-copy numbers, edit local YAML, rsync the
  YAML back, `run --bench --golden` to validate. A `--record NAME` flag on `run --bench` that prints the ready-to-paste
  YAML entry would remove the transcription step (and the integer-rounded `Eager` row: `cublas_us` had to be recorded
  at whole-µs precision, coarser than the 5090 seeds').
- **`goldens_by_name` is matmul-only** (`isinstance(g, MatmulGoldenConfig)`), so the documented spot-check returns `[]`
  for freshly added rms_norm entries — misleading during seeding; filter `GOLDEN_CONFIGS` by name instead. Worth
  widening to all kinds.
- **Winner knobs are hard to harvest:** neither the tune summary nor the bench table prints the policy knobs
  (`VECTORIZE_LOADS` / `INTERLEAVE_LOADS` / `FAST_EXP`); values were mirrored from the 5090 entries and validated via
  the A/B knob-diff (no red cells). `format_tuning_knobs` output in the tune log would fix this.
- **CloudRift box prep** (apt update + `python3.12-venv`/`-dev`, `CUDA_HOME=/usr/local/cuda`) cost ~4 failed rounds
  before the first kernel compiled; see the memory note / consider a `scripts/remote_setup.sh`.
- vs the previous sweep reports (rtx5090/rtx4080): their headline friction (noise-floor re-runs, step 4) didn't bite
  here — all three seeds reproduced within 0.1 µs across runs; small reduce-tier shapes are far more stable than the
  matmul shapes those sweeps fought with.
