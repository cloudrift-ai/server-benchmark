# Gemma-4 global-layer (full_attention) golden seeding + hd512 flash analysis — RTX 5090, 2026-07-14

- **Trigger**: a kernel-inventory diff of real layer compiles (`emmy compile google/gemma-4-12B --layer {0,5}`)
  against every golden file showed the 8 `full_attention` layers (of 48, every 6th) compile to shapes NO golden
  covers: `global_head_dim=512`, **one KV head** (k_proj weight `(512, 3840)`), and `attention_k_eq_v` (no
  v_proj — V is a V-norm of the k_proj output). The 40 sliding layers' kernels all map onto existing entries.
- **Method**: the manual pinned `--ab` golden-sweep workflow (no tuner) — per shape one broad candidate round
  from the card's known winner families, 1–2 refinement rounds, every winner 3× reproduced (spreads ≤0.7%),
  matmul runs deploy-pinned via `EMMY_KNOBS` (dodges the greedy hazards), attention runs with NO env tile pin.
  `cublas_us` = median live `Eager PyTorch` row (cuBLAS HGEMM / torch SDPA / torch RMSNorm).
- **Deliverable**: new `global`-suffixed entries in `goldens/rtx5090_sm120_gemma4.yaml` (static + dynM + s2048
  tiers), a re-sweep of the file's slower-than-ref s2048 std rows, and the hd512 flash codegen root-cause below.

## Headline: 1 in 6 gemma-4 layers ran on fully unseeded shapes; q_proj_global cold greedy is ~800× off

`run --bench --golden gemma4_12b.q_proj_global` (post-seeding validation) shows raw cold greedy deploying a
scalar `b256` tile at **143 868 µs vs cuBLAS 180** on this unseeded shape — the same misdeploy class the
original gemma seeding found on kv_proj (~770×), reproduced on a new shape. The attention.hd512 greedy does
NOT hang (unlike unseeded hd256) but deploys the `w2x1 d3/cp/ring` form at 210 µs = 0.47× — a 2× rank miss.

## Static tier (M=512) — recorded winners, 3× reproduced

| shape | lane | config | emmy µs | ref µs | vs ref |
| --- | --- | --- | --: | --: | --: |
| q_proj_global 512×8192 K3840 | std | `w4x2/f2x4/k2 g2k d2/tma/ring` | 169.0 | 180.3 | 1.07× |
| | fm | `w4x2/f4x8/k4 SERIAL d2/tma/ring` | 108.6 | 180.3 | **1.66×** |
| k_proj_global 512×512 K3840 | std | `w2x2/f2x4/k2 g8k d2/tma/ring` | 15.3 | 14.3 | 0.94× |
| | fm | `w2x4/f2x4/k2 g8k d2/tma/ring` | 11.8 | 14.3 | **1.21×** |
| o_proj_global 512×3840 K8192 | std | `w4x2/f2x4/k2 g2k d2/tma/ring` | 154.3 | 158.1 | 1.02× |
| | fm | `w4x2/f4x8/k4 g2k d2/tma/ring` | 118.2 | 158.1 | **1.34×** |
| attention.hd512 16h | std | `dd w4x1/f1x2/k32, pj w4x1/f1x64, d1/cp/alt, g2k` | 113.0 | 98.2 | 0.87× |
| qknorm.k512 8192×512 | — | `b128` | 8.7 | 8.2 | 0.95× |

Split-K is shape-specific across the three projections — exactly the kind of structure a cold prior misses:

- **q_proj_global fm wants SERIAL** (N=8192 fills the 170-SM grid alone; g2k costs 13% in finalize).
- **o_proj_global REQUIRES the split** (fm serial collapses 220 vs 118.2 — the K=8192 grid starves at N=3840).
- **k_proj_global (N=512, grid 8 unsplit) needs the deep `g8k` on the k2 atom**: the fm k4-atom `g8k` is
  REFUSED (slice 480 % 64 ≠ 0) and the big `w4x2/f4x8` tile collapses to 53.5 µs at grid 16. The fm win
  needed the narrow `w2x4/f2x4/k2` spelling — no other seeded shape uses it.

## attention.hd512 — the frontier is codegen-bound at 0.87×; full gate/limit root-cause

The knob space is exhausted (every direction 3× reproduced):

| form | µs | vs SDPA 98.2 |
| --- | --: | --: |
| **nt2 `d1/cp/alt` + split-KV `g2k`** (recorded) | **113.0** | **0.87×** |
| nt2 `d1/cp/alt` | 118.8 | 0.83× |
| nt2 `d1/cp` + g2k (no alt) | 127.5 | 0.77× |
| nt2 `d1/cp/alt` + g4k | 130.7 | 0.75× |
| nt4 `d1/cp` + g2k | 143.9 | 0.68× |
| nt2 `d3/cp/ring` + g2k | 144.0 | 0.68× |
| nt2 `d2/cp/ring` | 157.7 | 0.62× |
| nt4 `d1/cp` | 165.3 | 0.59× |
| greedy (`w2x1 d3/cp/ring`) | 210.1 | 0.47× |
| fm P·V sibling (`d2/cp/ring`) | 263.8 | 0.37× |

Root-caused limits (all verified against `_schedule.py::_resolve_twisted_stage` + the live ~100 KB budget):

1. **The 255-register ceiling is intrinsic at hd512**: the `w4x1` form's 64×512 f32 O-accumulator alone is
   256 regs/thread — every realized form sits at 255 regs / 4–8% occupancy. The f2 ILP chains are
   register-impossible (2×256 O regs), and the fm P·V demote — the lever that pays at hd64–256 — LOSES
   (263.8): the f16 promote traffic at d_v=512 outweighs the accumulator savings. Fix direction: a **d_v
   fold** (split the 512-wide O accumulation into per-pass halves — the P·V analogue of split-KV, new
   codegen), or cross-step software pipelining to hide the spill latency.
2. **Only the nt=2 geometry stages**: nt4's K/V slot is 32·(512+512+16)·2 = 66 560 B, so the ring depth
   clamps to 1 (`d2/cp/ring` resolves as `d1/cp` — benched, loses) and the alt pipeline needs staged-Q
   (66 560 B) + slabs > budget. Not a divisibility issue.
3. **TMA declines wholesale**: head_dim 512 > the 256 TMA box-dim hardware cap (`bn/head_dim/d_v ≤ 256`
   gate). Fix direction: split the K/V slab encode into two 256-wide boxes per step.
4. Diagnostic fix landed on this branch: the STAGE-pin decline warning claimed "static, N-key-block-divisible
   kv" for EVERY decline reason; it now names the geometry and the full gate list (transport / divisibility /
   TMA caps / smem budget).

## dynM + s2048 tiers

dynM (symbolic M @ hint 512, 3× reproduced ≤0.5% spread): **dynM ≈ static parity holds on all three global
matmuls** (fm: q 109.6 / k 11.9 / o 117.1; std: 168.7 / 15.4 / 154.4 — every value within 1% of its static
twin), same knob spellings recorded. Two attention.hd512.dynM findings:

- **The static winner has no dynM twin**: split-KV `g2k` fails on the symbolic form — "cross-CTA split of a
  symbolic reduce axis is not built yet" — so the dynM primary is the plain `d1/cp/alt` at 135.2 = 0.77×
  (ring d2/cp 146.7).
- **Cold greedy on the unseeded symbolic hd512 shape deploys a scalar kernel at ~289 ms (~2800× off)** — it
  does not hang, it is just enormous; the recorded row pins it.

s2048 (static M=2048, 3× reproduced; refs live same-session, attention ref measured directly — see below):

| shape | lane | config | emmy µs | ref µs | vs ref |
| --- | --- | --- | --: | --: | --: |
| q_proj_global.s2048 | std | `w4x2/f2x4/k2 SERIAL` | 601.4 | 600.9 | 1.00× |
| | fm | `w4x2/f4x8/k4 g2k` | 439.8 | 600.9 | **1.37×** |
| k_proj_global.s2048 | std | `w2x2/f2x4/k2 g8k` | 47.5 | 47.1 | 1.00× |
| | fm | `w2x4/f2x4/k2 g2k` | 37.8 | 47.1 | **1.24×** |
| o_proj_global.s2048 | std | `w4x2/f2x4/k2 SERIAL` | 595.1 | 570.2 | 0.96× |
| | fm | `w4x2/f4x8/k4 g2k` | 389.5 | 570.2 | **1.46×** |
| attention.hd512.s2048 | std | `dd w4x1/f1x2/k32, pj w4x1/f1x64, d1/cp/alt` | 1034.6 | 1040.4 | **1.00×** |

- **The split-K story differs from the hd256 s2048 tier**: the fm lane KEEPS a shallow `g2k` on all three
  global projections at M=2048 (hd256's q_proj dropped it), while the std lane drops the split on q/o.
- **attention.hd512 reaches PARITY at seq 2048** — the 255-reg/8%-occ penalty that costs 13% at seq 512
  washes out once the 512-CTA grid fills the card. The split-KV advantage INVERTS (plain alt 1034.6 beats
  g2k 1150.4): the finalize is pure cost at a saturated grid.
- **Cold greedy on unseeded hd512+s2048 deploys a scalar b256 over a 16.7M-block grid that HANGS the 1 s
  watchdog** (the hd256-s2048 hazard reproduced at hd512) — and the hang aborts the e2e backends bench, so
  the recorded `cublas_us` is torch SDPA measured directly (1039.2/1040.0/1043.5, 3×).

## Re-sweep of the recorded slower-than-ref s2048 std rows — one replacement, two confirmations

- **mlp_gate_up.s2048 std REPLACED**: the wide-fragment `w4x2/f4x8/k4` std geometry (the gate_up.h4096 std
  pick) beats the recorded `w4x2/f2x4/k2` serial by **9.2%** (2191.0 vs 2413, 3× at 0.6% spread) —
  0.87× → 0.96× vs cuBLAS. The square-2048 `w2x4/f2x2/k4` winner collapses to 4554 here; `d4` depth is
  neutral-to-worse. The residual 4% is the std-lane wall, not search.
- **o_proj.s2048 std: no change** — the recorded config re-measures at 291.4 vs ref 290.8 (1.00×) this
  session and every alternative loses (303–393; the f4x8 geometry that won gate_up loses 26% here). The
  recorded 0.97× was session noise.
- **attention.hd256.s2048 std: no change** — nt4 cp-alt (266.6), nt8 tma-alt (266.9), alt+g2k (270.9) and
  ring nt4 (280.5) all lose to the recorded nt8 cp-alt 262.7. The 0.94× residual is the per-step pipeline
  depth item, not search.

## Slower-than-reference audit of the recorded gemma4 goldens (both cards, from recorded values)

Best row per (name, lane): **every sub-reference entry is the std (f32-acc) lane** — the fm lane beats its
reference everywhere except one row.

- **4090**: mlp_gate_up.s2048 std **0.68×** (worst in the set), kv_proj std 0.84–0.86×, mlp_gate_up std
  0.91×, attention.hd256.s2048 std 0.91×, q_proj/kv_proj/mlp_down .s2048 std 0.93–0.95×, o_proj.s2048 0.98×.
- **5090**: attention.hd256 std 0.94–0.97× (static/dynM/s2048), mlp_gate_up.s2048 std 0.92×, o_proj.s2048
  std 0.97×; rms_norm.k3840 0.97× is noise-level.
- The one fm loss: 5090 attention.hd256.dynM fm 0.86× — the static fm-alt win does not transfer to the
  symbolic form (std stays the dynM primary, as the yaml records).

## Part 2 — RTX 4090 twin (same day, riftuser@176.124.69.202, the box from the original Part-2 sweep)

Same workflow (branch rsynced, sweeps under nohup, JSONs pulled back, replays validated on-box within
noise). **16 entries added to `rtx4090_sm89_gemma4.yaml`** — the same name set as the 5090. Winners
(3× reproduced; static/dynM/s2048; ref = live eager):

| shape | std (f32-acc) | fm (f16-acc) |
| --- | --- | --- |
| q_proj_global | `w2x2/f4x4/k2 g2k` 220.8 (0.96×) — s2048 drops the split (serial, bimodal 8%) | `w2x2/f4x8/k2 SERIAL` 144.2 (**1.47×**); s2048 583.2 (1.29×) |
| k_proj_global | `w2x2/f2x4/k2 g8k` 19.6 (0.85×) | `w2x4/f2x4/k2 g8k` 17.9 (0.93×) — the 5090 winner spelling transfers; s2048 shallows to g2k |
| o_proj_global | `w2x2/f4x4/k2 g2k` 223.5 (1.02×) | `w2x2/f4x8/k2 g2k` 162.1 (**1.40×**); s2048 goes SERIAL 589.3 (1.34×) |
| attention.hd512 | alt+g2k 116.1 (0.82×); dynM alt 139.6 (0.79×, 9% jitter); **s2048 alt+g2k 1069.0 (0.87×)** | PV lose (not recorded) |
| qknorm.k512 | `b128` 9.9 (1.05×) — greedy already picks it | — |

4090-specific findings:

- **k_proj_global is below parity on BOTH lanes** (fm 0.93×, std 0.85×; at s2048 fm g2k 0.95×, std 0.74×)
  — the sm_89 kv_proj latency residual, worse at N=512. Nothing in the knob space closes it.
- **Split-KV does NOT invert at s2048 on this card** (unlike the 5090): alt+g2k 1069.0 (0.1% stable)
  beats plain alt 1391.6 (8% jitter) and ring 1541.1. And split-KV PAYS at seq 512 on the alt pipeline
  where ring+g2k loses (172.6) — the transport/split interaction is lane- AND card-specific.
- **Greedy on the unseeded hd512 shapes misdeploys but never hangs here**: static 163.3 (0.59×), dynM
  803.8 (0.14×), s2048 5 419 µs (0.17× — under the watchdog that the 5090's scalar pick blows).
- The 4090 alt jitter (8–9% run-to-run on plain-alt rows) recurs from the hd256 sweep; the g2k rows are
  0.1%-stable throughout — worth preferring stable rows when ratios tie.

## Follow-ups

1. **hd512 d_v fold / TMA box split** — the only routes past 0.87× at short seq (5090; the 4090 residual
   is the same ceiling).
2. **Symbolic split-KV** ("cross-CTA split of a symbolic reduce axis is not built yet") — it is the static
   winner on BOTH cards' hd512 (and the 4090's s2048), so the dynM lane leaves 8–16% on the table.
3. The whole-model seq>1024 explicit-mask attention form (PR #365) has no golden coverage for the hd512
   global layers either — the mask+hd512 combination is untested end to end.

## Repro / artifacts

Work dir `/tmp/…/scratchpad/g4/` (session-scoped): per-shape 3-rep `--json` records (`seed/<shape>.r{1..3}.json`),
layer dumps (`l0/`, `l5/`), sweep scripts. Every entry's repro command is recoverable from the yaml knobs, e.g.:

```bash
# any global matmul entry (pinned replay):
EMMY_KNOBS="TILE=a:mma_m16n8k16_f16_f16/w4x2/f4x8/k4,REDUCE=,RASTER=,STAGE=d2/tma/ring,WSPEC=" \
  venv/bin/emmy run --bench -c "torch.matmul(torch.randn(512,3840,dtype=torch.float16), torch.randn(3840,8192,dtype=torch.float16))"
# golden replay:
venv/bin/emmy run --bench --golden gemma4_12b.attention.hd512
```
