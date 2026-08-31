# NVFP4 serving follow-up: native, tuned serving for Qwen 3.6 / 3.8

Branch-lifetime working note for the follow-up to PR #499. May be discarded before the PR is finalized.

**Goal.** `emmy serve Inferact/Qwen3.8-27B-NVFP4 --generate` on an RTX 5090 doing what the vLLM recipe command
(`--tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice`) does, plus
`nvidia/Qwen3.6-27B-NVFP4` as the architecture-identical official stand-in — emmy kernels, native W4A4, tuned.
The vLLM parser flags need no work: `emmy serve` forwards unknown flags verbatim (`_split_own_flags`).

Evidence discipline: every number or state inherited from PR #499's description or notes is a HYPOTHESIS
(marked *claimed*) until re-run on our heads. The docs there were written across many heads by many agents.

## The gaps

1. **Serving never declares 4-bit activations.** `gen_runner._compile_split` stamps quantized weights
   (`spell_quantized_constants`) but never calls `spell_static_fp4_activations` — the one call the compile
   lane makes and serving doesn't. Serving therefore runs the W4A16 scaffolding. Verified by reading.
2. **Nothing is tuned.** `serving/twins.py`'s format dispatch has EXL3/fp8/mxfp4 arms, no NVFP4 arm; no NVFP4
   model golden exists in `recipes/`. *Claimed:* untuned coded-trunk decode sits 58431x over its roofline
   floor and profiling never completes.
3. **No hybrid dispatch.** Qwen3.6/3.8-27B run gated-DeltaNet linear attention on 3 of 4 layers. The trace-side
   carve was removed in `b3056eea0` (re-applies cleanly — tested 2026-08-31). Known consumer bug:
   `vllm_model_gen.py` unpacks 3 tensors where a gated layer's `pre` returns 4. Per-layer-kind routing and
   DeltaNet cache wiring in serving were never written.
4. **No recipe, no qualification** for any NVFP4 Qwen; Inferact repack calibration unvouched.
5. **Compile-lane efficiency gaps** (backlog): re-measured 2026-08-31 on this head (workstation, layer 0,
   seq-len 512, tile IR): the block-scaled cell lands on **1 of 6** marked linears (`down_proj` only) — the
   readiness table's 3-of-6 (v/o/down, 2026-08-26) did not reproduce; the PR description's final "lands on one
   kernel" matches. `q_proj`/`k_proj` ride as planar decodes fused into mean-reduce kernels. The activation
   encode itself is present (8 `to_f4e2m1` sites), so the declared W4A4 program compiles. *Claimed, un-rerun:*
   the cell is 2.0x slower than cuBLAS's same instruction; `PLACE`/`WORK` pins are whole-program.

## Environment

Workstation: RTX 5080 Laptop, 16 GB VRAM, cc 12.0 (sm_120 — the block-scaled instruction runs), 30 GB host RAM.
Good for: 8B serving, sm_120 correctness, kernel dumps. Not for: recorded perf numbers, goldens, 27B. Those go
to a rented 5090. Setup on the workstation: `nix develop` + venv in the worktree. NixOS gotcha: CPU LoopOp
execution (cppyy/Cling) segfaults locally — keep local tests on graph-structure checks and numpy forwards;
LoopOp-executing suites run on CI/GPU boxes.

PR stacking: step 1's PR is based on `feature/nvfp4`; each later step stacks on the previous. GitHub retargets
to `main` when #499 merges; then one `git rebase --onto origin/main <499-head>` per open branch.

## Step 1 — serving declares W4A4 (dense 8B) — PR based on `feature/nvfp4`

Scope: `emmy serve nvidia/Qwen3-8B-NVFP4 --generate` runs the declared W4A4 program instead of the W4A16
scaffolding. Touches `_compile_split` (activation speller joins its stamp), `engine_config_overrides` (null the
modelopt quant config toward vLLM as already done for exl3/awq/mxfp4), plus whatever plan plumbing spelled
activations need. No compiler-pass changes. Also: re-verify the 499 claims this step stands on (compile-lane
cell coverage, serving's current W4A16 state) on the workstation.

| # | deliverable | verified by |
| --- | --- | --- |
| 1.1 | `_compile_split` on a synthetic NVFP4 checkpoint produces pre/post graphs carrying the activation quantize algebra: `to_f4e2m1` encode ahead of each marked linear, packed `f4e2m1x2` activation buffer | new CPU test, written first, failing now — `tests/serving/generation/test_gen_runner.py`, fixture style of `test_create_keeps_storage_coded_trunks_packed` |
| 1.2 | vLLM no longer receives the modelopt quant config when emmy owns the weights | extend `engine_config_overrides` coverage with a modelopt case, written first |
| 1.3 | one 8B layer through the runner matches eager torch within the declared program's envelope (#499 recorded median 1.5e-3 / max 1.9e-2; bar: max <= 5e-2) | parity run on the workstation — pre-commit |
| 1.4a | pulled forward from 2.1/2.2 after the first boot attempts (2026-08-31): untuned W4A4 makes even booting impractical (vLLM's startup forwards run 36 layers of untuned programs; #499 measured one such window at 69+ s). So: the NVFP4 twins arm in `scripts/capture_gen_twins.py`'s lane, then a budgeted local tune of the decode-width twins on the workstation. The local tune DB is machine-local evidence, not step 2's recorded 5090 goldens | twins-arm test (written first) + a completed local tune run |
| 1.4 | `emmy serve nvidia/Qwen3-8B-NVFP4 --generate` boots, picks the coded trunk, chat probe returns coherent text | manual run, workstation, after 1.4a; `--enforce-eager` + `--num-gpu-blocks-override` is the accepted graphless smoke fallback |
| 1.5 | prefill-tier layer-0 program carries the block-scaled cell on >= 1 marked linear — parity with the compile lane's re-measured coverage (1 of 6, `down_proj`, this head); widening coverage is backlog, not step 1 | kernel dump inspection via `EMMY_DUMP_DIR`, workstation |
| 1.6 | full `tests/serving` green | pytest (GPU box or CI where local Cling breakage applies) |

Deliberately excluded: any speed claim. Parity is this step's only number.

## Step 2 — tuned and measured (dense 8B) — stacked on step 1

Scope: NVFP4 twins arm (`serving/twins.py` + `scripts/capture_gen_twins.py`), tune, record goldens. Rented
5090 for every recorded number.

| # | deliverable | verified by |
| --- | --- | --- |
| 2.1 | twin capture on a synthetic NVFP4 checkpoint yields coded W4A4 twins | new CPU test, written first, beside `test_twins_coded.py`'s exl3/fp8/mxfp4 patterns |
| 2.2 | `emmy tune` over captured 8B twins completes; `emmy eval knobs` shows the packed rows | tune run on 5090, output in PR |
| 2.3 | `recipes/Qwen3-8B-NVFP4/golden/rtx5090_sm120.yaml` exists and passes golden validation | file in PR |
| 2.4 | the coded-trunk decode step completes and has a number (today: two timed-out profile runs, none) | `scripts/profile_gen_decode.py --bucket 16` on 5090 |
| 2.5 | coded W4A4 TPOT <= the decoded-trunk baseline (160.03 ms/step per #499) and no kernel > 10x over its roofline floor | same profile + runner roofline report |
| 2.6 | `--generate --bench` vs stock vLLM, same 5090: comparison recorded (target, not promise: within 1.5x of stock TPOT) | bench table in PR |

| 2.7 | regression tracking for the marked-linear kernels: realization corpus cases per projection shape at prefill width, pinned to the block-scaled cell — `q_proj`/`k_proj` as `_xfail_offered` (the ratchet records the binding gap and forces acknowledging its closure); `down_proj` green | new cases under `tests/compiler/realization/cases/`, per its ARCHITECTURE; `offered`/`realized` run on any machine |

Search-selection drift (the `v_proj` class — schedule realizes, search stops picking it) is deliberately NOT a
corpus case (its ARCHITECTURE excludes search shortfalls); the recorded model golden (2.3) is the tracker for
that, plus `make bench-kernels` drift findings.

**Step 2b is no longer conditional — the trigger fired structurally on 2026-08-31**, from the workstation
manual sweep (`_tune/run1/`, 112 offerability-verified proposals over all 32 serving twin kernels): the
block-scaled cell is offered on only 2 of 8 serving matmul shapes (unfused `v_proj`/`down_proj`, static widths
only), and every FUSED linear (q+norm, k+norm, o+residual+norm+requant, gate+up+SiLU+requant — the shapes
serving compiles) refuses the warp tier outright, at every width, as does every symbolic twin. Tuned bests
show the consequence: pre graphs land at 11-35 us across tiers, post graphs at 48-330 ms — a scalar-tier
floor no schedule search can cross. Giving the fused shapes a tensor-core tier is the critical path to any
honest speed number; q/k binding is a sub-case of it. Two team invariants bound HOW (Ivan, 2026-08-31): no
fusion stop-gaps — fuse everything, and where a boundary helps performance, CUT the graph (the `PLACE`
lane), never refuse the fusion up front; and the resulting coverage is pinned as
`tests/compiler/realization` cases (the 2.7 deliverable), not new custom Python tests.

Findings logged along the way, each needing its own fix: the coded-trunk weight load takes 23 minutes for the
6 GB 8B (host-side, independent of kernel picks); `--dump-dir`'s frontend-reproducer capture crashes on a
spelled W4A4 graph (`Input buffer 'attn_out_static_fp4_bits' does not exist` — slice taken over the pre-spell
input graph); an expected `bench_fail` watchdog verdict prints a full child traceback and reads like a crash.

## Step 3 — hybrid serving (Qwen3.6-27B-NVFP4) — stacked on step 2

Scope: the serving-contract work. Revert `b3056eea0`, fix the 3-vs-4 unpack, give the runner per-layer kinds,
route full-attention layers through emmy programs and linear-attention layers through vLLM's DeltaNet module
with its cache, raise the twin-memory ceiling for the 27B.

| # | deliverable | verified by |
| --- | --- | --- |
| 3.1 | stock vLLM 0.23 serves `nvidia/Qwen3.6-27B-NVFP4` — or the recorded fact it doesn't, which re-plans the step | first task, before any code — run on rented 5090 |
| 3.2 | carve restored: `tests/serving/test_linear_attention_split.py` green again | pytest |
| 3.3 | a gated layer's 4-tensor `pre` is consumed correctly | new CPU test, written first, failing on the unpack bug |
| 3.4 | runner exposes per-layer kind; `EmmyGenModel.forward` routes by it | new CPU tests, hybrid config, mock weights |
| 3.5 | both layer kinds hold parity on real 27B weights (#499 claims carve max_abs 0.0 — re-verify) | `emmy run --layer <N>` for one layer of each kind, 5090 |
| 3.6 | 27B twin capture fits in <= 30 GB host RSS | measured capture run |
| 3.7 | `emmy serve nvidia/Qwen3.6-27B-NVFP4 --generate` generates coherent text with full-attention layers on emmy kernels | manual run + kernel dump inspection, 5090 |

## Step 4 — Qwen3.8, qualification, recipes — stacked on step 3

| # | deliverable | verified by |
| --- | --- | --- |
| 4.1 | `emmy serve Inferact/Qwen3.8-27B-NVFP4 --generate` + the three parser flags: a chat completion with a parsed tool call round-trips | probe on 5090 |
| 4.2 | 27B twins tuned; `recipes/Qwen3.8-27B-NVFP4/golden/rtx5090_sm120.yaml` recorded | files + tune runs |
| 4.3 | calibration vouched: lm-eval score within an agreed delta of the fp16/AWQ sibling recipe's score | `scripts/run_lmeval_gate.py` on 5090 |
| 4.4 | TPOT/throughput vs stock vLLM recorded in the recipe's RESULTS | bench table |

## Backlog (named, not scheduled)

q/k contraction binding; two-channel `gate`+`up` pair reading; per-kernel `PLACE`/`WORK` pins + the
evidence-path direct test; the 2.0x cell gap to cuBLAS; `graph.to_dict()` round-trip on packed constants;
#499's parity-gated consolidations. Pulled in only when a step's numbers demand it.

## Risks

- vLLM 0.23 may not serve `qwen3_5` — checked at 3.1; fallbacks: newer vLLM pin in the recipe, or another
  DeltaNet route. Re-planning trigger, not a silent assumption.
- #689 (kernel identity off the canonical lowered body) landed after #499's measurements — recorded goldens
  may already be orphaned; step 2 re-records rather than trusts.
- Inferact calibration unvouched until 4.3; alternates (`RadixArk/…`, `gittensor-model-hub/…`) queue behind it.
