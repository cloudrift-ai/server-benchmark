# Next steps from the qwen3-emb / gemma-4-e2b layer-0 tune findings

## Context

The two findings reports asked "what next, and which kernel can be made competitive with cuBLAS?" Code exploration
**corrected both reports' hypotheses**, and the corrections change the work:

1. **Qwen3 attention (9.4 ms/layer, the dominant gap).** The tensor-core flash tier the qwen report calls a
   "follow-up" **already exists and ships** (`_schedule._twisted_warp_options` + `kernel/_twist.py::realize_warp_twist`
   — it's what Gemma deploys at 81 µs). And RoPE'd Q/K is **not** the blocker: the fusion boundary already keeps RoPE
   out of the score producer, so `_extract_qk` succeeds. Recognition actually dies on the **V operand**: v_proj output
   is `(1, S, 1, 1024)` and GQA folds the kv-head into V's trailing index (`(head/2)*128 + d`); `_extract_v_layout` →
   `_slot_of` matches only plain per-slot vars → `_fuse_degraded("P@V's V load is not plainly indexed")`
   (`_flash.py:735-737`) → the un-fused 3-kernel scalar SDPA. Once V is slot-plain, qwen (causal, f16, d=128, GQA g2)
   passes every warp-tier gate and deploys mma flash. **Fix A is a small fusion-boundary change, not a new tier.**
2. **Projection linears — the "competitive with cuBLAS" answer (~1.5 ms qwen + ~2.4 ms gemma).** The MMA tile **is
   enumerated** for the split reduce-epilogue linears (q/k/o_proj mains, PLE) — it's a **pick miss**, not a missing
   tier: atomic split-K (`g?a`) exists only on the scalar tier by design (an mma C-fragment can't atomicAdd), so a
   prior wanting split-K gets nudged off tensor cores; the `__partial` reproducers pick `a:mma_m16n8k16_f16` rank-1
   (70 µs vs 1463 µs deployed). **Fix B lives in search/prior ranking** — and MMA routing also dodges the gemma
   k_proj >1 s hang (a masked-M scalar split-K boundary bug, a different class from the masked-N fix in cc21dd14).
3. The uncommitted gemma PLE trace fix (`emmy/compiler/trace/huggingface.py`) must land with the full checklist.

Scope confirmed: **A + B + land the PLE fix.** Three PRs, in order below. `plans/` holds 12 files (cap 10) — prune
along the way.

---

## PR-0 — land the PLE trace fix (branch `feature/ple-layer-trace`)

The working-tree diff in `emmy/compiler/trace/huggingface.py` (synthetic seeded `per_layer_input` buffer in
`build_layer_wrapper`, sliced in-graph like cos/sin; non-PLE archs byte-identical). To land:

- **Test** (new `tests/compiler/trace/` test, no HF download): fake `nn.Module` block with
  `hidden_size_per_layer_input` whose forward asserts `per_layer_input is not None` and multiplies; plus a non-PLE
  fake block asserting it's called *without* the kwarg. Assert buffer registration only for PLE blocks, wrapper runs
  at two seq lens (slicing contract).
- **Docs**: one sentence in `emmy/compiler/trace/ARCHITECTURE.md` under `build_layer_wrapper` (shapes/latencies real,
  numerics synthetic; non-PLE unchanged).
- **Prune `plans/` to ≤ 10**: check `git log` for executed reports (candidates: `golden-sweep-rtx4090-findings.md`,
  `golden-sweep-rtxpro6000-findings.md`, `splitk-structural-fork.md`) — verify each landed before deleting.
- `make test`, `make lint`, PR.

## PR-A — flash recognition over a GQA-folded V (branch off PR-0; the 9.4 ms win)

**Fix shape: keep the axis-folding V indexmap materialized at the fusion boundary** (mirrors what the boundary
already does for RoPE cones on Q/K and compute-bearing V producers), rather than teaching every flash surface
(recognition, fragment builder, staging/TMA box encoding) a "folded slot" vocabulary. Cost: one V layout-copy kernel
(~3–20 µs, same class as the rotary layout kernels) vs 9.4 ms recovered. The rule is algebraic, not shape-keyed: *an
indexmap whose load index combines ≥2 axis vars in one slot leaves the operand no longer plainly indexed; at a flash
offer site it stays materialized.* Permutation/broadcast/slice indexmaps (≤1 var per slot, incl. GQA `head//g` in its
own slot) keep fusing.

### Edits

1. **`emmy/compiler/pipeline/passes/loop/fusion/_helpers.py`** — new predicate `folds_axes(loop_op)`: any Load index
   slot with ≥2 free axis vars.
2. **`emmy/compiler/pipeline/passes/loop/fusion/010_merge_loop_ops.py`** — narrow the `_is_pure_indexmap` exemption
   at the three guard sites: the flash-consumer protection (line ~286), the score-producer protection (line ~298),
   and the pending-contraction-half timing hole (line ~272 — a folding indexmap may fuse into the bare P@V product
   half before the offer site forms; extend `_sum_contracts_exp_producer` or a sibling helper to fire on an
   accum-free product half whose other multiplicand chain is exp-bearing). Module docstring gets the
   narrowed-exemption sentence.
3. **`emmy/compiler/pipeline/passes/loop/fusion/005_split_shared_indexmap.py`** — same gate for the multi-consumer
   indexmap rule (`RuleSkipped` when `folds_axes(producer)` and any consumer is flash-offer-shaped).
4. **`emmy/compiler/pipeline/passes/lowering/tile/_flash.py` — per-operand GQA groups.** With the fold blocked, V
   materializes post-repeat (16 heads) while K stays 8-head, so the single `gqa_group(q_shape, k_shape)` +
   `k_batch[-1] != v_batch[-1]` eligibility check would degrade the fuse. Generalize: compute `gk`/`gv` per operand,
   `flash_shape_eligible(..., group_k, group_v)`, `build_flash_frag`/`_flash_op` divide K's head var by `gk`, V's by
   `gv` (`group == 1` degenerates to today's path). No external callers of these helpers exist.
5. **Stale docstrings (same PR):** `_flash.py:26-29` and `_atomize.py:21-28` still say "flash lowers only on the
   scalar tier" (pre-#300 — the warp tier exists); `tests/compiler/e2e/test_attention_coverage.py:8-10` has the same
   stale bullet plus a reference to a removed `FLASH` knob / `025_recognize_flash` pass.

### Tests (in `tests/compiler/e2e/test_attention_coverage.py`)

- `test_tensorcore_flash_folded_v_projection_fuses` — model-style V: `(1,S,Hkv*D)` f16 → view/transpose → SDPA with
  `enable_gqa=True` (e.g. Hq=4, Hkv=2, S=128, D=32). Assert **fusion** (one SDPA kernel containing
  `emmy_mma_m16n8k16_f16`, kernel count = flash + V copy), plus accuracy — pattern of
  `test_generated_tensorcore_flash_matches_torch`.
- `test_flash_folded_v_with_repeated_kv_fuses` — HF idiom (explicit KV repeat) exercising per-operand groups
  (`gk=2, gv=1`).
- A no-GPU structural test (pattern of `test_flash_form_fork_offers_geometry_grid`): folded-V module →
  `enumerate_graph` → flash fork rows present.
- TinyLlama full-attn tests: add a fusion assert if the scalar tier certifies post-fix (fp32 + additive mask keeps it
  off the warp tier); otherwise keep accuracy-only with a docstring note. Watch seq512 accuracy thresholds (flash
  reorders FMAs).

### Verification (A)

1. Compile-only, no GPU: `source _tune/tune-model-qwen3-emb-l0-4080/env.sh && EMMY_KNOBS="" emmy compile
   _tune/…/08_lowering_cuda.kernels/k_sdpa_linear_reduce_deb3d9.torch.json --ir cuda | grep -c mma.sync` → expect ≥3
   (was 0), one fused SDPA kernel + V copy instead of the 3-kernel cut.
2. **Gate before claiming the win**: confirm the recognized `mask_kind` is causal (an additive mask buffer would fuse
   but stay scalar — `_twisted_warp_options` refuses additive, and d_v=128 > `_CHAIN_MAX_D=64` kills the chain). If
   additive fires, warp-tier additive-mask support becomes a scoped follow-up decision.
3. Layer re-tune + bench (fresh isolated `EMMY_TUNE_DB`/`EMMY_PRIOR_FILE`/`EMMY_CUBIN_CACHE` work dir): expect
   attention → one mma-flash kernel in the ~0.1–0.3 ms class, layer e2e **8880 µs → ~2.5 ms** (residual = the PR-B
   projections).
4. `make test` (`-n auto --dist=loadgroup`) + `make lint`. Delete `plans/qwen3-embedding-06b-layer0-tune-findings.md`.

## PR-B — projection linears onto MMA (the cuBLAS-competitiveness fix)

### B1. Diagnostic first — pin the MMA fork on the full layer (no code; can run while PR-A is in review)

Pin via **`EMMY_KNOBS`** (`knob.py::apply_knobs_env`; a pin narrows authoritatively at every fork). `run --ab` is not
usable (needs `--code/--golden/--ir`), so A/B = two `run --bench` invocations per model:

```
source _tune/tune-model-<x>/env.sh
emmy run <model> --layer 0 --dynamic seq_len@x:1 --bench                     # baseline
EMMY_KNOBS="TILE=a:mma_m16n8k16_f16/w2x1/f4x4/k8,REDUCE=" \
  emmy run <model> --layer 0 --dynamic seq_len@x:1 --bench                   # MMA pinned, split-K off
```

(`k8` divides all projection Ks; `w2x1` keeps the flash pin-contract satisfiable on qwen post-PR-A. Try 2–3 tile
geometries.) Record per-kernel -O3 deltas. Then attribute the miss with `emmy eval analytic` vs `emmy eval prior`:
does the **analytic** dyn weight set rank scalar first (plausible — `_W_A_DYN` has `MMA_tier=+1.76` vs
`D_bn_ge_bm=+29.1`/`D_pow2_threads=+23.7`, features a warp row structurally can't earn), or does the **learned**
-O1-trained prior own the pick? **Gate:** proceed to B2 only if pinned-MMA beats baseline materially; if the learned
prior is the culprit, pull the "-O3 ranking column" tune-infra item into scope (otherwise out of scope).

### B2. The ranking lever (primary hypothesis: analytic featurization gap + unpriced split-K asymmetry)

General rules only — no kernel names, no shapes (moveset invariant; `test_golden_configs.py` permanence):

1. **`_schedule.py::_tile_rows`** (~line 442): when `_warp_atoms` is non-empty, stamp `S_warp_eligible=1.0` into the
   knob base every row inherits (same mechanism as `S_masked_*` riding `STRUCT_PREFIX` through `knob_features`).
2. **`search/features.py`**: `D_scalar_on_warp_eligible` (= 1.0 when `S_warp_eligible` and not warp-tier) and
   `D_splitk_roundtrip` (= log2(free_prod) when the `g?k` finalize-kernel fires — the analogue of the existing
   `D_cut_roundtrip` that prices the demoted-cone cut but not the split-K workspace round-trip).
3. **`search/prior/analytic.py::AnalyticPrior.score`**: hard-coded interaction penalties for both (the
   `atomic_free_weight` precedent — no training rows carry the stamps yet). Optional inside the PR: refit
   `_W_A`/`_W_A_DYN` via `scripts/golden_knob_heuristics.py` and validate golden median rank; else note the refit.
4. **No enumeration edits** — scalar split-K rows stay offered (goldens permanence). The mma-fragment atomic finalize
   (closing the `g?a` asymmetry in codegen, `030_split.py:164-170`) is the long-term close, noted in the PR, not built.

Tests: new `tests/compiler/pipeline/search/test_analytic_tier_preference.py` (no GPU; pattern of
`test_structural_push.py`) — enumerate a warp-eligible f16 [512,1024]×[1024,·] contraction (static + symbolic-M),
assert `AnalyticPrior` ranks some `a:mma` row above every scalar `g?a`/`g?k` row; `S_warp_eligible` stamped on all
rows, absent on fp32. Extend `tests/compiler/passes/test_move_catalog.py` to confirm row key-sets unchanged.

### B3. The gemma k_proj hang (masked-M scalar split-K — correctness-adjacent, MMA routing only *dodges* it)

1. Repro-confirm (`source _tune/tune-model-gemma-4-e2b-l0-4080/env.sh && emmy run google/gemma-4-E2B --layer 0
   --dynamic seq_len@x:1 --bench` → `HungKernelError` on `k_linear_reduce_8f622a`?).
2. Bisect: `emmy eval variants --kernel k_linear_reduce_8f622a --top 0`; pin the suspect masked-M `g4a` config via
   `EMMY_KNOBS`; inspect the emitted boundary guard.
3. If it's the decline-gate class (the cc21dd14 precedent): structural *refusal* of the unboundable masked-M scalar
   split-K combination in `_schedule.py` + sibling test to
   `tests/compiler/e2e/test_matmul_coverage.py::test_scalar_masked_n_stage_declines`.
4. If deeper: timebox ~a day; land the decline as mitigation + repro test + follow-up note in the PR description.

### Verification (B)

1. B1's measured pinned-vs-baseline delta (go/no-go).
2. Post-B2 cold-prior compile (`EMMY_PRIOR_FILE` → empty path): qwen q/v_proj mains deploy `a:mma` in `eval variants`.
3. Full re-tune of both layers (fresh work dirs): qwen e2e ~2.5 ms → **≤ ~1.0 ms**; gemma **full-layer e2e obtainable**
   (hang gone), PLE/q/k_proj rows `a:mma`.
4. `emmy eval analytic` on the golden dataset — median golden rank not regressed by the new weights.
5. `make test` + `make lint`. Delete `plans/gemma-4-e2b-layer0-tune-findings.md`.

## Key risks → catch

| Risk | Caught by |
|---|---|
| Qwen mask is additive, not causal → fuses but stays scalar | A-verify step 2 gate (+ step 1 grep = 0) |
| V fold lands in bare P@V half before guards fire | A-edit 2's pending-half extension + the e2e fusion tests |
| 16-head V vs 8-head K breaks eligibility | A-edit 4 per-operand groups + the `gk=2,gv=1` test |
| Folding-indexmap refusal false-positives elsewhere → extra copies | Always correctness-safe; re-tune kernel tables show no unexpected copy kernels |
| Learned prior (not analytic) owns the scalar pick | B1 attribution; explicit gate pulls -O3-ranking into scope |
| B2 weights over-rotate on goldens | B-verify step 4 + goldens permanence test |
| Gemma hang not decline-class | B3 timebox → mitigation + follow-up |

## Out of scope (noted in PRs, not built)

Tune-infra items unless B1's gate fires (-O3 ranking): 4 s compile budget (gemma f3), reproducer coverage flags
(both reports), full-layer bench skip-and-continue, `tune --work-dir`. MMA-fragment atomic finalize. Warp-tier
additive-mask + causal tile-skip. Folded-slot support inside flash (fix shape (i), rejected). NCU enablement,
serving A/Bs, whole-model tunes.
