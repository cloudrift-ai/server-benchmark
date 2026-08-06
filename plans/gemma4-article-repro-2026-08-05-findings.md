# Gemma-4-12B article reproduction — RTX 5090 (local) + RTX 4090 (rented), 2026-08-05/06

Brief: re-run every experiment behind "Beating vLLM and Llama.cpp on Gemma4-12B" at current `main`
(`703a0948`, the #465 tile-scheduler rebuild), fix problems as they surface. Article read live from the
landing checkout (`optimizing-gemma-4-12b-rtx/index.md`, the published 2026-08-01 16K-protocol tables).

**Result: the article reproduces at HEAD — after this session fixed a #465 regression that had silently
pushed every routed-cut kernel to its fused floor.** Serving cells within 0.7% (many exact), GSM8K all
four lanes inside one standard error, the numerics sweep digit-identical, per-kernel geomeans at or above
published on both cards. Branch `repro/gemma4-article-2026-08-05` (`870e698f`, `6c9082ac`).

## Setup

| | |
|---|---|
| 5090 host | local box, driver 580.173.02 (article: 580.159.03), CUDA 13.0 toolkit |
| 4090 host | riftuser@211.21.50.85:57002, driver 580.159.03 (matches article), CUDA 13.3 toolkit (article: 13.0) |
| Emmy code | working tree at `703a0948` + this branch's fixes (command recipes: kernels / accum / gsm8k / llama.cpp) |
| Emmy image | `cloudriftai/vllm-emmy-gemma-4-12b-it:latest` = `0.23.0-78f5364f`, digest 5add12d3 (serving + MTP; predates #465, so the deploy suites never saw the regression) |
| Drift | `origin/main` is 10 commits past the image (#465, #461, serving Milestone A tasks 2–3) — serving numbers are the released artifact's, not HEAD's |

## The regression found and fixed (the session's main result)

`_cut._ws_dtype` under the one-kind Fold IR: with `Map`/`Contraction` dissolved into `Fold` (#465),
`isinstance(child, Fold)` matched every node, so every routed-cut seam workspace materialized **f32**. An
f32 A cannot ride the warp mma atoms bare, so each cut consumer re-wrapped into a demoting cone, keyed
`fused` off the sync-STAGE offer signal, joined its **parent's** fused golden, and deployed the fused floor
— both precision lanes, both cards (5090 fm `norm_qkv.m256.lin` 1.31x → 0.62x, `norm_gate_up.dynM.lin`
1.38x → 0.44x, std hit identically). #465 had recorded the coverage loss in `EXPECTED_MAJOR_GAPS` as an
accepted re-keying ("reproduced identically at the pre-rebuild commit") — it had not; the pre-#465 A/B in
this session measured the published ratios. Fix: carrier state ⇔ the child folds an axis; a zero-axis
projection is the value seam and keeps its leaf dtype. All eleven 5090 major-gap keys (and six m1
generic-gap siblings) are covered again with **no re-seeding**; both ratchets tightened to empty.

Also fixed: the command-recipe result pull flattened `std/*` and `fm/*` to one basename (fm silently
overwrote std locally — recovered for all runs of this session), and the stale "no longer reproduces the
published table" recipe headers (the published article IS the 16384 protocol).

## Measured vs published

### Serving (RTX 5090, image `78f5364f`; tok/s | median TTFT ms | median TPOT ms)

| point | lane | tok/s | pub | TTFT | pub | TPOT | pub |
|---|---|--:|--:|--:|--:|--:|--:|
| 4k/4k c=1 | stock | 57.2 | 57.2 | 565 | 565 | 17.4 | 17.3 |
| | emmy | 54.4 | 54.4 | 622 | 625 | 18.2 | 18.2 |
| | fm | 54.5 | 54.5 | **469** | 471 | 18.2 | 18.2 |
| 4k/4k c=4 | stock | 216.5 | 216.6 | 1086 | 1086 | 18.2 | 18.2 |
| | emmy | 203.9 | 203.9 | 1264 | 1271 | 19.3 | 19.3 |
| | fm | 205.4 | 205.2 | **1061** | 1070 | 19.2 | 19.2 |
| 4k/4k c=8 | stock | 383.4 | 383.8 | 1100 | 1099 | 20.6 | 20.6 |
| | emmy | 372.0 | 371.9 | 1219 | 1224 | 21.2 | 21.2 |
| | fm | 376.4 | 375.9 | **997** | 1007 | 21.0 | 21.0 |
| 8k/256 c=4 | stock | 112.7 | 112.7 | 2029 | 2027 | 27.3 | 27.3 |
| | emmy | 100.8 | 99.9 | 2636 | 2666 | 28.9 | 29.4 |
| | fm | 111.9 | 112.5 | **2157** | 2176 | 27.4 | 26.6 |
| 256/256 c=64 | stock | 1434.5 | 1435.9 | — ¹ | | 27.7 | 27.7 |
| | emmy | 1135.2 | 1139.0 | — ¹ | | 30.9 | 29.3 |
| | fm | 1210.2 | 1218.6 | — ¹ | | 29.2 | 28.1 |

¹ the published c=64 TTFT is the single-wave (`--num-prompts 64`) protocol; the recipe's c=64 row is the
np=256 queue-drain and is not comparable (measured queue-drain TTFTs: 1692 / 3277 / 2458). The only cells
past 1%: the c=64 emmy-lane TPOTs (+4–5%).

### GSM8K (strict exact-match, 200 questions, ±0.033)

stock **0.690** (pub 0.685), emmy **0.680** (0.670), fm **0.700** (0.695), llama.cpp **0.685** (0.665) —
all four inside one standard error; hybrid accumulation still costs no task quality.

### Numerics (accum_error): digit-identical, all 15 cells.

### Per-kernel catalogs (post-fix; geomean of eager/emmy per shape)

| card | lane | full catalog | shared-shape: pub → measured |
|---|---|--:|--:|
| 5090 (309 cases; article had 277) | std | 1.14x | 1.177x → **1.199x** |
| | fm | **1.30x** (article headline 1.30x) | 1.341x → **1.348x** |
| 4090 (150 cases; article had 139) | std | 0.94x | 0.894x → **0.954x** |
| | fm | 1.10x | 1.051x → **1.118x** |

The catalogs grew (post-article m512/m1024/m192 golden tiers), so the full-catalog numbers are not
like-for-like against the article's; the shared-shape columns are. `attention.hd512` on the 4090 moved
0.25x → 0.86x. 5090 lane counts: std 169/309 ≥ eager, fm 256/309.

### llama.cpp serving lane (fresh master build, article pinned `0a50d99` — version drift applies)

4k/4k c=1 58.1 (pub 56.4) | c=4 182.3 (153.1) | c=8 270.9 (294.0) | 8k/256 86.3 (80.2) | c=64 OOM as
published. Emmy-independent control; deltas track llama.cpp's own movement. NOTE: the recipe's point
list has no 8192/256 cell although the article's table does — measured here manually with the recipe's
serve/bench spelling; worth adding to the recipe.

### MTP smoke-test (tok/s; the article's own disclaimer: compare within the table only)

The first full-grid run failed every emmy+MTP cell at boot: `daaec3e5` keeps a self-contained image's baked
`HF_HOME` (+ its `HF_HUB_OFFLINE=1`), so the drafter (`google/gemma-4-12B-it-assistant`, host-cache only)
could not resolve — "Invalid repository ID". Fixed in `compose.py`: a `--speculative-config` lane needs a
model beyond the baked one, so the host-cache `HF_HOME` override returns for exactly that case. Re-run
(measured vs published): stock d2/d3/d5 at 4k/4k c=1 = 111.9/144.6/197.8 (pub 109.3/145.9/197.2); emmy
d2/d3 there = **107.1/135.7 (pub 107.2/135.8, exact)**; 4k/4k c=4 stock d2/d3 349.9/443.6 (355.8/443.7),
emmy d2 341.3 (352.1); c=8 emmy d2 450.9 (446.2); 256/256 c=64 stock d2 1426.7 (1423.5), emmy d2 **904.2
(pub 882.8)** — the 2.6x-recovered cell holds. Drifted: stock d2/d3 at c=8 measured 534/605 vs pub 610/717
(emmy-independent lane; within the disclaimer's scope). Not measured (run terminated by request): the
8192/256 row, emmy d3 at c=8, and both emmy d5 cells.

## Not reproducing, and why (no code change owed)

`mlp_geglu.m4096` fm (pub 1.39x, measures 0.55x) and the last stretch of `mlp_down_fused.m256.lin`
(1.29x vs 1.06x) do not reproduce **even at the article-measurement revision on the same box**: the
published picks rode tune-DB / online-prior evidence never recorded into the goldens — open lead 1 of
`plans/gemma4-article-repro-16k-findings.md`, now confirmed by bisection. Remedy: seed fm golden rows for
those shapes (manual `--ab`, the golden-sweep method). Also still below published: `mlp_down_fused.m8.lin`
and two 1-µs-resolution micro-shapes (bench quantization, not signal).

## Harness findings

- **emmy's ssh transport dropped twice** (rc=255, sshd logs "disconnected by user"), both times exactly
  when `llama-server` began its 23.8 GB model load on the loopback host (gsm8k 01:57, llamacpp 05:33);
  `rift-service` logged worker errors in the same minute. Keepalives are generous (30s x 20), so it is not
  a plain silence timeout; not root-caused here. Both lanes were completed manually from the surviving
  task-dir artifacts.
- A background `( … ) &` subshell launched from a wrapper dies with it, and a `pkill -f` whose pattern
  appears in the wrapper's own command line kills the session — run multi-step lanes from a script file.
- The leftover `vllm_0` container between suites (known gotcha) was torn down before gsm8k.
