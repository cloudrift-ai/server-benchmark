# gemma-4-12B serving optimization on the RTX 5090 (post gap-closure roadmap, 2026-07-21)

Successor to the executed gap-closure plan, scoped to the gemma-4 generative path on the local 5090. Baselines
(this branch, #407 protocol + extensions — full tables in `plans/serving-gap-closure-findings.md`):

| workload | stock | emmy | gap |
| --- | --- | --- | --- |
| decode c=32 TPOT med | 18.2 ms | 20.7 ms | 1.14× |
| decode c=64 TPOT med (over-bucket, captured) | — | 35.4 ms | soft vs c=32 |
| mixed in-256 req/s | 13.90 | 10.04 | 0.72× |
| mixed in-256 TTFT med | 231 ms | 400 ms | 1.73× |
| 4K-in/4K-out out tok/s | 204.5 | **235.5** | emmy leads |

Protocol invariants for every A/B: `EMMY_GEN_DECODE_BUCKET` matches the seeded golden set, empty online prior
(`{}`), `EMMY_TUNE_DB=_tune/decode-twin-readiness/twins.db`, seed 0, and compare TPOT for decode claims (the
2026-07-21 stock decode arm's TTFT was anomalous — 473 ms median vs its own 95 ms baseline — while its TPOT
reproduced exactly; re-baseline stock once before quoting req/s leads).

## 1. Over-bucket decode: extend the m32 golden playbook to M=64 (cheap, proven method)

Concurrency > 32 decode now rides CAPTURED symbolic programs (+10.6% req/s at c=64 from the capture-ladder
change), but those widths run cold-tuned shapes — TPOT 35.4 ms at c=64. The m32 flow (static decode twins +
manually-seeded goldens) is what got 20.7 ms at c=32; mirror it at M=64:

1. Capture static M=64 twins: `scripts/capture_gen_twins.py --model google/gemma-4-12B --out _tune/decode-m64
   --decode-bucket 64 --prefill-bucket 0 --no-symbolic` (**done — 4 twins captured**, sliding + global).
2. Bench cold picks per twin (`emmy run <twin>.json --bench` under the serving evidence env) and identify
   laggards vs the m32 configs scaled 2× in M.
3. Manual pinned `--ab` sweep on laggards (NOT the tuner — the established method), starting from each shape's
   m32 golden config as the anchor; seed `rtx5090_sm120_gemma4.yaml` m64 entries from same-regime medians.
4. Verify deploy (`emmy eval golden --in-model` MATCH, no shadowing of the m32 set — the ShapeKey aspect-blind
   shadow class), then A/B `EMMY_GEN_DECODE_BUCKET=64` + capture sizes to 64 vs today's bucket-32+symbolic at
   c=64. Exit: c=64 TPOT clearly below 35.4 ms with c=32 protocol numbers unregressed (bucket 64 pads c=32
   decode steps to 64 rows — if that costs c=32 TPOT, keep bucket 32 and instead seed the SYMBOLIC `.dynM`
   evidence at the M≈64 regime; decide by measurement, not preference).

M=128 only if the c=64 exit lands and a real workload needs c>64 — same recipe.

## 2. Decode TPOT residual (1.14× stock) — the fused-form pipeline, research-class

The whole remaining decode gap is two per-layer kernels in the post twin at M=32: the fused gate⊗up computed-A
edge (170 µs, 64% of the step) and down_proj (74 µs). Knob optimum is within 4% of the memory floor
(`plans/computed-a-pipeline-and-sdpa-oproj.md`) — the lever is real pipeline work (async multi-stage weight
prefetch on the sync compute-fill), not tuning. Schedule as its own session; do not burn A/B cycles re-tuning
these shapes.

## 3. Mixed/prefill gap (0.72× stock req/s, TTFT 1.73×)

Two known residuals, both research-class, both documented: the large-M computed-A prefill pipeline (same lever
as #2 — one pipeline design serves both) and hd512 flash cold-unreachability (needs the hd256 lever port +
symbolic split-KV, its own session per the round-2 findings). The 4K/4K result shows the prefill side is
already competitive when amortized over long decodes — prioritize #2's pipeline work first since it feeds both.

## 4. Workflow: cut the A/B iteration cost

- Wire `EMMY_PACK_DIR` into the serving A/B protocol — gen-runner pack support exists (`test_gen_pack_gpu`);
  boots are ~11 min of evidence resolution per arm, the dominant cost of every optimization loop on this list.
- Re-baseline the stock decode arm (TTFT anomaly above).

## Non-goals here

- Embedding-path work (Qwen3-Embedding flash/varlen) — separate track, out of scope for this plan.
- WS2.1 (RoPE into the pre-trace) — premise stale (findings); revisit only with a flash B-track consumer.
- Re-tuning the m32 decode set or the gate⊗up/down_proj shapes — measured at/near their floors.
