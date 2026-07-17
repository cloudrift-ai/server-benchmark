# Gemma-4-12B decode-bucket golden seeding (M=16/32) + whole-step decode capture (RTX 5090)

- **Status**: complete. Both serving levers from the post-fix report's roadmap landed on
  `feature/gemma4-decode-goldens-graph-capture`: the decode-shape golden family is seeded (16 shapes, all beating
  cuBLAS) and `emmy serve --generate` defaults to whole-step decode CUDA graphs (vLLM `FULL_DECODE_ONLY`, no
  `--enforce-eager`).
- **Context**: the 12B serving A/B (`plans/gemma4-12b-postfix-golden-serving-findings.md`) measured emmy ~100×
  behind stock vLLM, structural: decode-bucket twins deployed COLD (TPOT 3 639 ms at bucket 32 — slower than the
  symbolic fallback) and ~96 per-token host launches. These are the two named levers.
- **Run commands** (2026-07-16, local RTX 5090, per-run `EMMY_TUNE_DB`/`EMMY_ONLINE_FILE` under
  `_tune/decode-goldens-5090/`): `sweep.sh` — per-shape `EMMY_FAST_MATH=1 emmy tune -c "<matmul snippet>"` (one
  shared DB/prior, within-sweep transfer, fm-superset enumeration), then `emmy run --bench -c … --json` per shape
  per regime; manual `--ab` family pins afterwards. Tune phase ≈ 3.2 h (16 shapes), A/B ≈ 25 min, 0 bench_fails.

## Outcome — 16 new goldens, emmy beats cuBLAS on every decode shape

| shape (all fp16) | N | K | emmy µs | cuBLAS µs | vs cuBLAS |
| --- | --: | --: | --: | --: | --: |
| q_proj.m16 | 4096 | 3840 | 9.4 | 53.2 | **5.7×** |
| kv_proj.m16 | 2048 | 3840 | 5.7 | 12.3 | 2.2× |
| o_proj.m16 | 3840 | 4096 | 8.8 | 22.5 | 2.6× |
| q_proj_global.m16 | 8192 | 3840 | 16.7 | 47.1 | 2.8× |
| k_proj_global.m16 | 512 | 3840 | 5.3 | 6.1 | 1.2× |
| o_proj_global.m16 | 3840 | 8192 | 15.4 | 38.6 | 2.5× |
| mlp_gate_up.m16 | 30720 | 3840 | 145.2 | 147.1 | 1.01× |
| mlp_down.m16 | 3840 | 15360 | 72.9 | 79.7 | 1.09× |
| q_proj.m32 | 4096 | 3840 | 10.7 | 16.4 | 1.5× |
| kv_proj.m32 | 2048 | 3840 | 6.6 | 10.2 | 1.5× |
| o_proj.m32 | 3840 | 4096 | 11.1 | 16.4 | 1.5× |
| q_proj_global.m32 | 8192 | 3840 | 17.6 | 26.6 | 1.5× |
| k_proj_global.m32 | 512 | 3840 | 4.3 | 6.1 | 1.4× |
| o_proj_global.m32 | 3840 | 8192 | 18.7 | 26.6 | 1.4× |
| mlp_gate_up.m32 | 30720 | 3840 | 144.8 | 151.6 | 1.05× |
| mlp_down.m32 | 3840 | 15360 | 74.1 | 77.1 | 1.04× |

- The MLP giants are at the weight-streaming roofline (gate_up moves ~236 MB/pass ⇒ ~130 µs floor at ~1.8 TB/s),
  so ~1.0× vs cuBLAS is the correct ceiling there — the wins live on the projection shapes.
- std ≡ fm pick on every shape (plain `f16_f32` atoms; the big-tile f16acc family doesn't pay at small M) — no
  `[fm]` siblings recorded.
- Serving arithmetic: a decode step's per-layer matmul work at M=32 now sums to ~230 µs/layer sliding /
  ~250 global ⇒ ~11 ms/token across 48 layers — the right order against stock vLLM's 24.9 ms TPOT once the
  launches collapse (the capture lever below).

## Finding 1 — within-sweep prior transfer is a shape-ORDER effect: the first shapes' picks lost 20–45%

- **Evidence**: the sweep tuned m16 shapes first. Its early greedy picks (q_proj.m16 13.3 µs, kv_proj.m16 10.3,
  o_proj.m16 11.0, q_proj_global.m16 23.0) were all beaten by the `w1x8/f2x2/k[24] + g{4,8}k + d2/tma/ring`
  family config that every LATER shape's search converged on (9.4 / 5.7 / 8.8 / 16.7 — pinned via `--ab`, then
  re-recorded). The last-tuned shapes' picks were already optimal.
- **Class**: search shortfall at cold start, healed by transfer — the tune-golden skill's "one invocation, never
  a per-shape loop" rationale, observed as a first-shapes tax instead. `eval variants` on the early kernels shows
  the family configs measured but ranked below the deployed pick in the -O1 lane.
- **Recommendation**: for future SEEDING sweeps (new shape families), run two passes — tune all shapes once, then
  re-tune (no `--clean`) the first ~quarter of the list; or simply A/B the converged family config across all
  shapes at the end (what this session did by hand, ~15 min).

## Finding 2 — ShapeKey `free_prod` is aspect-blind: a 512×512 golden shadows the 32×8192 twin

- **Symptom**: cold greedy on `q_proj_global.m32` (32×8192, K=3840) deploys `w2x2/f2x4/k2` at 26.5 µs — NOT its
  recorded golden (16.3 µs pinned) and with NO drift warning.
- **Root cause** (spy on `_golden_pick` during a cold compile): the golden index buckets by
  `ShapeKey(free_prod=M·N, reduce_max=K, …)`, and 32×8192 = 512×512 = 262144 — the bucket also holds
  `k_proj_global.s512` (K=3840 too). Entries sort fastest-first, the s512 entry (15.3 µs) outranks the m32 twin
  (17.6), its config prefix-matches an offered row, and the pick "succeeds" — deploying a square-shape config on
  a thin shape at 1.6× its cost. Silent by design: the tier believes same key ⇒ same shape.
- **Two more collisions are BENIGN after the finding-1 re-pins**: 16×4096 == 32×2048 and 16×8192 == 32×4096 —
  the shadowing twin records the SAME family config the shadowed shape wants (the q_proj.m16 cold deploy actually
  IMPROVED through the cross-match before the re-pin).
- **Fix (compiler follow-up, ~small)**: add a `free_max` discriminator to `ShapeKey` — `S_ext_free_max` is
  already stamped on every row (`diagnostics.py` reads it), the golden side knows `max(M, N)`, and ShapeKeys are
  recomputed per process (never persisted), so the change is a constructor-pair edit
  (`from_matmul`/`from_s_features`) plus the per-kind `shape_key()` builders and the flash rebuild in
  `_fork_shape_key`. Until then the YAML section comment documents the shadow; the affected shape cold-deploys at
  eager parity (26.5 vs eager 26.6), not a regression vs stock.

## Whole-step decode CUDA graphs (the second lever) — DEFAULT for `emmy serve --generate`

- vLLM 0.23's `CUDAGraphMode.FULL_DECODE_ONLY` captures full decode graphs **without torch.compile** (the
  `CUDAGraphWrapper` wraps an undecorated OOT model), gated only on the attention backend's cudagraph support.
  `emmy serve --generate` now passes `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY",
  "cudagraph_capture_sizes": [1, …, bucket]}'` instead of `--enforce-eager`; sizes are clamped to the decode
  bucket because an over-bucket batch takes the model's symbolic fallback path whose per-layer host numpy hops
  abort a stream capture. Opt out with vLLM's own `--enforce-eager`; `EMMY_GEN_DECODE_BUCKET=0` forces eager.
- The enabling change is in `_Program.run_device`: under `torch.cuda.is_current_stream_capturing()` it issues the
  raw launch sequence (`run_once` — prebuilt buffers, no alloc, no sync) instead of its own
  `capture_program_graph`/`replay_program_graph` (nested stream capture and graph launch are both illegal in a
  capturing stream). The whole decode step (embed + 48× pre/RoPE/paged-attention/post + final norm) then records
  into ONE vLLM graph; the ~96 per-token host launches and per-call Python/dlpack overhead vanish at replay.
- **Verified**: `tests/serving/test_gen_capture_gpu.py` — a `run_device` call captured inside an outer
  `torch.cuda.graph` replays LIVE (new input values flow through on replay). A full serving A/B (12B or
  TinyLlama, capture vs eager) is the remaining validation — needs a serving-capable box/session.

## Repro / artifacts

- Work dir: `_tune/decode-goldens-5090/` — `sweep.sh` + `sweep.log`, `tune.db` / `online.json`,
  `ab-<shape>-{std,fm}.json` (32 records), `ab-*-family.json` (the manual family pins),
  `ab-q_proj_global.m32-std-r2.json` (the pinned win).
- Cold-deploy verification: `emmy run --bench --golden gemma4_12b.<name>.m{16,32}` with an empty
  `EMMY_TUNE_DB`/`EMMY_ONLINE_FILE` — greedy row ≡ golden row on every shape except the finding-2 shadow.

## Workflow notes

- **The seeding loop is ~90% waiting on per-variant nvcc** (~11 min/shape × 16). The shapes share one weight
  matrix family; a `tune --dataset` -style single invocation over a NEW shape list (not just the recorded
  goldens) would let the DB/prior transfer do its thing without a hand-rolled sweep script — third session to
  hand-roll one (`sweep.sh` here, `sweep.sh` in the 12B repro, the s2048 seeding).
- **`tail -3` in the sweep script destroyed the per-variant evidence** — the enumeration questions in finding 1/2
  had to be answered from the DB (`eval variants`) instead of the log. Tee the full tune output; disk is free.
- **The `_golden_pick` spy** (monkeypatch + `CudaBackend.compile` on the snippet) answered the finding-2 "why no
  warning" question in one run — worth folding into `eval golden` as a `--explain-deploy` view: shape key, bucket
  contents, which entry matched which row.
- The fm-superset sweep (`EMMY_FAST_MATH=1` tune + per-regime A/B) cost nothing extra and proved std ≡ fm on this
  family — keep it as the default seeding recipe on consumer dies.
