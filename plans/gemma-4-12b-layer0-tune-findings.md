# Gemma 4 12B layer-0 tune findings — RTX 5090 (sm_120), 2026-07-02

- **Status: emmy is 69× behind eager end-to-end on this layer (96.5 ms vs 1.39 ms)** — dominated entirely by the
  sdpa/attention kernels falling to the serial reduce schedule (the same dead reduce fork as
  `plans/golden-sweep-rtx5090-findings-8.md` finding 3, here costing milliseconds instead of microseconds).
  Correctness is fine: `emmy run google/gemma-4-12B --layer 0` and `--layer 5` (sliding + global attention) both pass
  the fatal accuracy check vs eager.
- **Command:** `emmy tune google/gemma-4-12B --layer 0 --dynamic seq_len@x:1 --clean --bench --dump-dir
  _tune/tune-model-gemma4-12b-l0/dump` — layer 0 is a `sliding_attention` layer (window 1024; the first
  `full_attention` layer is 5). Dynamic scope: symbolic `seq_len`, **benched at the 512 hint** (masked-tile kernels,
  the deployable artifact).
- **Run stats:** ~23 min wall (21:28–21:51). 538 benches (61 warmup / 477 post); DB: **421 ok / 37 `bench_fail`**
  (36 of the 37 are one misaligned-address cluster — Finding 4). Branch `tuning/gemma-4-5090` (includes the golden
  sweep's warp-TMA box-gate fix, which is why zero TMA box-extent failures appear here).
- Numbers below are the `--bench` -O3 re-bench (deployable, CUDA-graph captured); tune-DB latencies quoted for
  ranking context are -O1 and say so.

## Bench results

Full layer (torch inputs tiled to the seq_len=512 hint; single layer, NOT the model e2e):

| Backend | Latency (µs) | vs Eager |
|---|---|---|
| Eager PyTorch | 1394 | 1.00× |
| torch.compile | 1211 | 1.15× |
| **Emmy** | **96459** | **0.01×** |

Per-kernel -O3 re-bench (sorted by emmy µs; Layer-op labels from the dump's `.torch.txt` coverage headers; `-` =
the slicer wired no torch reference for that fused chain):

| Kernel | Layer op | eager | tcompile | emmy | vs eager |
|---|---|---|---|---|---|
| k_sdpa_reduce_60081d | sdpa (1/28 slice — attention·V reduce) | 35 | 31 | 73118 | 0.00× |
| k_linear_sdpa_reduce_fb39d3 | linear_3 (o_proj, 6/7) + sdpa drain | 129 | 127 | 72998 | 0.00× |
| k_sdpa_mean_linear_reduce_f98a03 | sdpa mask/softmax + norm + linear chain | - | - | 50821 | - |
| k_sdpa_reduce_75af34 | sdpa (softmax/score reduce slice) | 254 | 36 | 33718 | 0.01× |
| k_mean_linear_reduce_cd923e | post-FFN layernorm chain (add_9/10 + norm) | 480 | 305 | 2586 | 0.19× |
| k_mean_linear_reduce_f2c987 | post-attn layernorm chain (add_6 + norm) | 270 | 107 | 2332 | 0.12× |
| k_mean_5708a9 | input layernorm (add + mean + rsqrt·w) | 164 | 13 | 1521 | 0.11× |
| k_linear_mean_reduce_00fd4a | linear + norm fused edge | - | - | 842 | - |
| k_mean_linear_reduce_46ea14 | norm→linear fused edge | - | - | 679 | - |
| k_linear_reduce_5dd170 | linear split combine (fp32) | - | - | 597 | - |
| k_mean_linear_reduce_9aa814 | norm→linear fused edge | - | - | 377 | - |
| k_linear_reduce_dbf34f | linear (matmul) | 292 | 292 | 367 | 0.80× |
| k_linear_reduce_a17f46 | linear split combine | - | - | 297 | - |

The four sdpa-family kernels (73.1 + 73.0 + 50.8 + 33.7 ms as isolated reproducers) account for essentially the
whole 96.5 ms e2e total; every other kernel combined is < 10 ms. The only healthy matmul comparison is
`k_linear_reduce_dbf34f` at 0.80× eager.

## Finding 1 — sdpa lowers to the serial reduce schedule: ~230 ms of attention vs eager's ~0.4 ms

**Symptom**: `k_sdpa_reduce_60081d` 73.1 ms vs eager 35 µs (~2000×); `k_sdpa_reduce_75af34` 33.7 ms vs
tcompile 36 µs; reproducer re-bench confirms (72.6 ms second run — stable, not a bench artifact).

**Evidence**: `eval variants --kernel k_sdpa_reduce` shows `_75af34` has **exactly one measured config** — `REDUCE@a3`
*empty* (the serial option-0 schedule), 41.5 ms at -O1 / 31.9 ms at -O3. `_60081d` has zero ok rows (its single
variant SIGKILL'd the 8 s bench wall — the one non-cluster `bench_fail`). The emitted CUDA is the serial form
(one output element per thread, the key-axis reduce a bare loop). No flash/streaming (TWISTED), chain, coop, or
block sibling was ever offered.

**Root cause**: the reduce schedule fork emits zero rows through the model lowering — the same dead fork the golden
sweep quantified on bare reduces (`reduce.2048x2048` greedy 53× behind its pinned golden). PR #300's
chain/coop/serial prior-ranked siblings demonstrably work on the isolated flash-article kernel (the flash campaign
hit FA-2 parity, 206 µs, on this GPU), but nothing reaches `Run.resolve` for a `Reduction`/TWISTED node traced out
of a model graph. Class 2 (schedule lockout), not codegen: the flash kernels prove the fast form exists.

**Repro**: `emmy run --ir _tune/tune-model-gemma4-12b-l0/dump/08_lowering_cuda.kernels/k_sdpa_reduce_60081d.torch.json
--bench --bench-backends eager,emmy`.

**Fix (priority 1, shared with golden finding 3)**: restore the bare/TWISTED reduce schedule fork through the model
path, then re-tune. Until then every attention-bearing model deploys unusable sdpa kernels and single-layer e2e
numbers are meaningless.

## Finding 2 — the fused linear→sdpa / sdpa→norm→linear mega-chains: 124 ms in two kernels, scalar-tier locked

**Symptom**: `k_linear_sdpa_reduce_fb39d3` (o_proj matmul fused into the sdpa drain) 73.0 ms vs eager 129 µs;
`k_sdpa_mean_linear_reduce_f98a03` (sdpa mask/softmax + norm + linear) 50.8 ms, no torch reference wired.

**Evidence**: `eval variants --kernel sdpa`: all 50 measured configs of `fb39d3` are **scalar-tier** (no `a:` mma
atom anywhere in the leaderboard; best -O1 row 1.35 ms, pick rank 14/50 at 2.12× of best — a search shortfall
stacked on top of the lockout). Its `__partial` split, by contrast, **did** get the warp tier
(`a:mma_m16n8k16_f16/w1x2/f4x4 g2k d2/cp/ring`, 121 µs -O1 / 107 µs -O3) — the mma tier works on Gemma's 3840-wide
fp16 shapes when it is offered; the fused chain form is what locks it out.

**Root cause**: the mega-fusion pathology carried from the Qwen whole-model tune
(`plans/qwen3-embedding-06b-tune-findings.md` finding 1): fused chains (norm→linear at large N, linear→sdpa) have
no structural un-fuse escape, and the fused computed-A cone keeps the contraction off the warp tier. Known
limitation, recorded here because it is 124 ms of this layer.

**Fix (priority 1, two complementary tracks — CUT is not the only lever)**: verified with a `PLACE@fold=fuse`
pin: `try_flash`'s `_recognize` returns `None` on Gemma's graph BEFORE any eligibility guard (no degrade warning
fires; the kernel set is unchanged) — the sliding-window mask lands as a separate op between score and softmax,
and Q/K are q/k-RMSNorm + in-graph-RoPE cones, not the plain Loads `_extract_qk` demands. So the fused
schedule-friendly kernel already exists (the flash TWISTED form, FA-2 parity in isolation) and simply never
certifies on a model trace. (a) **Widen flash recognition** — mask-tolerant matching plus computed Q/K cones,
the same computed-operand precedent `bind_prologue_contraction` set for norm→linear — keeping QK+softmax+PV as
ONE warp-tier kernel; this is why the flash campaign's isolated win never appears in any model tune (every
per-layer trace has in-graph RoPE). (b) **The bandwidth-gated structural CUT for the chain edges** (qkv-proj /
o_proj / norm drains — the boundaries every production attention stack cuts at; the `__partial` mma evidence
shows the pieces are healthy alone). Scheduling the whole sdpa→norm→linear composite on the warp tier as-is (a
chained second contraction inside the streaming loop) is ruled out — highest effort, no production precedent,
and (a)+(b) deliver the same µs.

## Finding 3 — every norm/mean chain lowers one-thread-per-row: 6.4 ms across five kernels

**Symptom**: `k_mean_5708a9` (input layernorm) 1521 µs vs torch.compile **13 µs** (117×); the two
`k_mean_linear_reduce` layernorm chains 2586 / 2332 µs vs eager 480 / 270 µs.

**Evidence**: the emitted CUDA guards `if (_gid < seq_len)` — **512 threads total** on a 21k-core GPU, each thread
serially reducing H = 3840 (`k_mean_5708a9`) or walking the whole per-row chain (`cd923e`, `f2c987`). No coop/block
reduce sibling was measured for any of them (same `REDUCE@axis` emptiness as Finding 1 — the fork never offers the
cooperative forms the golden `g2a`/`b16` knobs prove exist).

**Root cause**: same dead reduce fork, per-row norm flavor. Class 2. Fixing Finding 1 should re-open these; re-check
this table afterwards.

## Finding 4 — misaligned-address / hang cluster on the fp32 linear split-combine: 36 wasted bench slots

**Symptom**: 36 `bench_fail` rows (of 477 post-warmup benches — ~8%) across `k_linear_reduce_5dd170` /
`k_linear_reduce_a17f46(__partial)`: half `HungKernelError` (1 s watchdog), half hard
`CUDA_ERROR_MISALIGNED_ADDRESS` that kills the bench worker (EOF).

**Evidence**: `eval failures` clusters all of them with shared knobs `STAGE@a2=d1/cp` (or `REDUCE@a2=g2k` on the
partials) + `VECTORIZE_LOADS=True` + `INTERLEAVE_LOADS=True`. These are fp32 kernels (`linear_reduce__partial`
buffers are `float*`) with a 4096-wide N — the depth-1 cp.async stage with vectorized+interleaved fill misaligns on
this fused split-combine shape. The final picks for both kernels are healthy (297–597 µs), so the cost is search
slots + ~40 s of watchdog stalls, not a wrong deploy — but the same class wedged the #244 dynamic tune, and a hard
fault in a bench worker is one gate away from a deploy fault.

**Repro (compile-only, validated)**: `EMMY_KNOBS="STAGE@a2=d1/cp,VECTORIZE_LOADS=1,INTERLEAVE_LOADS=1" emmy compile
_tune/tune-model-gemma4-12b-l0/dump/08_lowering_cuda.kernels/k_linear_reduce_5dd170.torch.json --ir cuda` (then run
under compute-sanitizer for the faulting address). Class 4; priority 2 — fix the alignment gate (or decline
vectorized fill on this stage form) in the scalar cp.async path.

## Finding 5 — no torch reference for 6 of 13 kernels

The op-slicing bench wired eager/tcompile references only where the slice maps to a clean torch closure — exactly
the fused-chain kernels (Findings 2–3's subjects) bench emmy-only, so their vs-eager column is blank and layer-level
attribution leans on the four referenced rows. `plans/bench-attribution-by-slicing.md` is the standing plan for
this; Gemma is a good test case (its every kernel is a fusion chain).

## Repro / artifacts

- Tune log: `_tune/tune-model-gemma4-12b-l0/tune.log`; dump: `_tune/tune-model-gemma4-12b-l0/dump/` (kernels +
  `.torch.json` reproducers under `08_lowering_cuda.kernels/`, bench JSON `62_kernel_bench.json`, `kernels.html`).
- Isolated worst kernel: `emmy run --ir …/k_sdpa_reduce_60081d.torch.json --bench --bench-backends eager,emmy`.
- Accuracy (both attention flavors): `emmy run google/gemma-4-12B --layer 0` / `--layer 5` — exit 0.
- NCU was available (`ncu 2025.3.1`) but skipped: the 2000× gaps are schedule-class, not counter-class; profile
  after the reduce fork returns.

## Workflow notes

- **Log spam**: the misaligned cluster printed ~36 near-identical multi-line tracebacks into the tune log (and the
  live monitor). `eval failures` dedups perfectly after the fact; the tune loop itself should collapse repeats of
  an already-seen (kernel, error) cluster to one line + a counter.
- **Silent -O3 compile bursts**: 8–12 min stretches with zero log output at ~2000% CPU (N=15360 register tiles at
  -O3) — liveness needed `ps`/`nvidia-smi` checks. A `[bench] compiling K variants (n/N)…` heartbeat line would
  make hang-vs-compile obvious.
- **Hash cross-referencing**: the printed kernel table drops hash suffixes while `eval variants`/`eval failures`
  key on them — labeling the report table meant joining through `62_kernel_bench.json` by hand. Print the suffixed
  name (or add `--full-names`).
- **Zero-ok kernels invisible in `eval variants`**: `k_sdpa_reduce_60081d` (0 ok / 1 fail) simply doesn't appear,
  which reads as "never enumerated" — a `0 ok rows (1 bench_fail)` stub row would distinguish lockout from wipeout.
- **From the Qwen report's notes**: the `.torch.json` reproducer + `--bench-backends` flow worked unchanged; the
  missing-reference gap (Finding 5) was already on file and remains the biggest attribution hole.

## Post-fix retune (2026-07-03, same branch — findings 1/3/4 addressed, no CUTs)

Fixes applied scheduler-first (per the "cuts hide weak spots" rule — no structural CUT was added), then the same
clean tune + -O3 bench re-run (`_tune/tune-model-gemma4-12b-l0-v2/`, 918 benches vs the baseline's 538 — the
restored forks nearly doubled the explored space):

- **Findings 1+3 (serial reduce schedules)** were largely closed by the ninth-golden-sweep scheduler fixes already
  on the branch: the restored `_reduce_specs` catalog fork reaches every norm/mean chain (verified: `k_mean` and
  both `k_mean_linear_reduce` chains now fork the full `b4..b32/r2/r4` catalog), and the sdpa fragments enumerate
  the TWISTED coop rows plus the flash warp move grid (`k_sdpa_reduce_60081d`: 32 leaves incl.
  `a:mma…/w1x1/f1x32`-class tiles; was ONE serial variant).
- **Finding 4 (misaligned cluster) — root-caused and fixed for real**: the staged fill/slab/descriptor/budget all
  sized both operands with A's element width; a mixed fp32-A × fp16-B contraction issued 16 B ``cp.async`` chunks
  at fp32 spacing over fp16 memory. Per-operand `Operand.dtype`/`elem_bytes` now flow through both transports
  (`test_scalar_cpasync_mixed_dtype_slabs`, red pre-fix / green post-fix; the pinned reproducer runs 421 µs
  clean). **Zero misaligned-address rows in the retune** (was 36); the 18 remaining `bench_fail`s are slow serial
  sdpa variants exceeding wall budgets — search cost, not faults.

| Backend | baseline | post-fix | |
|---|---|---|---|
| Eager PyTorch | 1394 | 1394 | |
| torch.compile | 1211 | 1220 | |
| **Emmy** | **96459** | **27695** | **3.5× better; 69× → 20× behind eager** |

Per-kernel -O3 (isolated reproducers, µs):

| Kernel | baseline | post-fix | vs eager now |
|---|---|---|---|
| k_sdpa_mean_linear_reduce | 50821 | 50885 | — (unchanged — see below) |
| k_sdpa_reduce (score slice) | 33718 | 3076 | 0.08× (11× better, still 12× behind) |
| k_sdpa_reduce (drain slice) | 73118 | 115 | 0.27× (635× better) |
| k_mean (input layernorm) | 1521 | 21 | **7.8× FASTER than eager** |
| k_mean_linear_reduce (post-attn) | 2332 | 117 | **2.15× faster** |
| k_mean_linear_reduce (post-FFN) | 2586 | 412 | **1.16× faster** |
| k_linear_reduce (matmul) | 367 | 367 | 0.79× |

Accuracy: `emmy run google/gemma-4-12B --layer 0` passes post-fixes.

**What remains is exactly the finding-2 flash-certification case**: the residual gap is concentrated in
`k_sdpa_mean_linear_reduce` (50.9 ms, byte-for-byte the baseline number — the fused sdpa+norm+linear chain whose
TWISTED reduce sits inside a `Map` composite no reduce-partition tier can rescue) and the score-slice
`k_sdpa_reduce` (3.1 ms — coop-partitioned now, but the streaming flash form is what eager's 251 µs corresponds
to). Both are the finding-2 fix track (a): widen `try_flash`'s `_recognize`/`_extract_qk` to Gemma's
mask-as-separate-op + q/k-norm'd/RoPE'd Q/K cones so QK+softmax+PV certifies as ONE flash kernel in model context
— the fused schedule-friendly form, not a CUT.

## v3 flash-certification run (2026-07-03 04:10, commit cb1a5ea6) — CERTIFIES, but HUNG on broken flash codegen

Track (a) landed as commit cb1a5ea6 ("Flash certifies on model graphs: fusion boundaries + layout-agnostic frag") and
this run (`_tune/tune-model-gemma4-12b-l0-v3/`) is the first model tune against it. **The certification worked; the
run did not.** Status as of 08:48: the `emmy tune` process (pid 3826767) is **deadlocked** — sleeping at 0 % CPU, no
worker children, no nvcc/cicc running, GPU idle, log frozen since **04:27** (~4 h20 m with zero progress). No exit
line, no `62_kernel_bench.json`, no `08_lowering_cuda.kernels/` — only `00_input.*` dumped. **There are no v3 bench
numbers to compare against the v2 table**; the tune wedged inside the very first kernel's bench sweep.

- **The certified flash kernel was numerically WRONG on Gemma (NaN) — FOUND + FIXED this session.** Greedy
  `emmy run google/gemma-4-12B --layer 0` (and `--layer 5`) failed `CORRECTNESS FAIL: output mul_ contains NaN`;
  `EMMY_KNOBS=PLACE@fold=cut` (flash OFF) passed → pinned to the flash kernel. **Root cause — a codegen bug, NOT the
  activations** (the isolated `.torch.json` reproducer passed only because the slicer wired it a clean 4-D
  grid-matched output; the bug needs the real model buffer). cb1a5ea6 made the input LOADS layout-agnostic
  (`_permute_idx`) but left the output STORE writing the bare 4-var grid tuple `(b0, b1, m, d)`. Gemma's sdpa output
  is 5-D `[1, 16, 32, 1, 256]` (a size-1 broadcast dim between seq and head_dim, plus the absorbed `transpose_2`), so
  the 4-component index mis-strided against the 5-D buffer → `[b0 + b1 + m + d]` (all outputs alias addresses 0–301,
  the rest uninitialized) → NaN in the downstream `mul_` (o_proj). **Fix:** `_out_store_index` (the output counterpart
  to `_permute_idx`) reproduces the root buffer's real rank + layout — mapping grid axes onto the output slots by dim
  extent, keeping the size-1 `Literal` slots — and every lowering tier's store consumes it (scalar `with_store` via a
  Map-body `Write`; the chain `_realize_chain` and the warp `_twist` `RegStore` substitute the row/col axis motion).
  An un-reproducible layout declines to cut. Files: `tile/_flash.py`, `kernel/_factor.py`, `kernel/_twist.py`.
  Verified: both Gemma layers pass (exit 0), the store is now `[b1*8192 + m*256 + d]` (correct strides); regression
  tests `test_flash_transposed_output_matches_torch` (e2e, absorbed transpose) +
  `test_out_store_index_reproduces_output_layout` (unit — unit-dim / transpose / decline), red pre-fix / green
  post-fix; full attention suite 70/70, full compiler suite 1589 passed / 2 skipped.

What the 23-line log *does* establish, all on one kernel now named **`k_scaled_dot_product_attention_reduce`**:

- **Flash certifies on the model graph (the intended win).** The attention op is no longer split into v2's
  `k_sdpa_reduce` / `k_sdpa_mean_linear_reduce` / `k_linear_sdpa_reduce` fragments — it forms a single
  `scaled_dot_product_attention` kernel (the flash TWISTED form). Finding 2 track (a) — mask-tolerant `_recognize`
  plus computed-Q/K `_extract_qk` — is real on a Gemma trace, exactly as the commit claims (182 µs vs eager 86 µs on
  the *pinned* warp variant in isolation).
- **…but the flash reduce's ILP register fold did not compile.** Variants nvcc-fail with `identifier "dd__r1"
  undefined` / `"dd__r3" undefined`. Root cause (found + fixed): `_coop_carrier` accepts the flash TWISTED reduce, so
  the full `coop_reduce_moves()` catalog — **including the `r2`/`r4` ILP register folds** — reaches the `kv` streaming
  reduce. But that reduce body holds the NESTED `dd` (Q@K) / `j` (P@V) contraction loops (flash is the first
  register-tiled reduce with a nested loop in its body — a plain matmul's reduce body is flat). `_factor._replicate`'s
  `copy_cell` renames a var's USES but not a nested `Loop`'s own axis DECLARATION, and the nested reduce axis was
  absent from `protected` — so copy r1 emitted `for (int dd …)` (unrenamed) with loads reading `dd__r1` (renamed):
  undefined. **Not** a frag-permute-layout bug (my first read) — it reproduces with plain non-permuted loads. Fix:
  add every nested loop-axis name in the reduce body to `protected` so `dd`/`j` stay shared (each copy re-declares its
  own loop under the one name). `emmy/compiler/pipeline/passes/lowering/kernel/_factor.py`; regression test
  `test_ilp_reg_flash_matches_torch` (red pre-fix / green post-fix, plain+causal × reg 2/4, nvcc + accuracy vs torch).
- **The misaligned-address fault-class is back (separate, still open).** A bench worker died on
  `CUDA_ERROR_MISALIGNED_ADDRESS` (worker EOF) — the same hard fault-class v2 eliminated for the fp32 split-combine
  (Finding 4), now resurfacing on a flash sdpa variant (a *different* shape, so v2's per-operand-dtype fix doesn't
  cover it). Root-cause needs the specific variant reproduced under compute-sanitizer.
- **The misaligned fault then wedged the whole tune (deadlock — root-caused, not yet fixed).** The EOF path *does*
  catch + pin `bench_fail @ 2e6` (`pipeline.finalize_exc`, and that line IS in the log) — so the parent did not fail
  to pin. The 4-hour hang is *downstream*: post-mortem shows the parent **sleeping (state `S`) at 0 % CPU, no worker
  child/zombie, no nvcc, 113 threads, ~7 GB GPU left allocated with zero compute-apps**. That signature — a futex
  sleep, not a `D`-state driver ioctl, with a leaked context — points to a hard GPU/driver wedge from the illegal
  memory access hanging a *parent-side* CUDA call on a helper thread, which the main thread then awaits forever. This
  is a GPU-wedge-induced deadlock; the tractable, verifiable fix is to **prevent the misaligned fault** (bullet
  above), not to make the parent survive a wedged device — so it is not patched speculatively here. Follow-up: capture
  a `py-spy dump` on the next occurrence to name the wedged thread, and consider a hard tune-level watchdog as
  defense-in-depth.

**Net vs the v2 checkpoint:** v2 is still the best *measured* state (27.7 ms, 20× behind eager, healthy reduce/norm
kernels). v3 proved the flash-certification structural goal is met, and this session made flash on Gemma **correct**:
greedy `emmy run --layer 0`/`--layer 5` now pass (were NaN). Status after this session (branch
`fix/gemma-flash-frag-codegen`): the ILP-fold codegen bug AND the output-layout NaN are **fixed + tested**; the
misaligned fault and its induced deadlock remain open (root-caused). Next steps, in priority order:
1. ✅ Kill the hung pid; ✅ fix the flash ILP-fold codegen (`_factor.py` `protected` set +
   `test_ilp_reg_flash_matches_torch`); ✅ fix the flash output-layout NaN (`_out_store_index` across all tiers +
   `test_flash_transposed_output_matches_torch` / `test_out_store_index_reproduces_output_layout`). Full attention
   70/70, full compiler 1589 passed / 2 skipped. Both Gemma layers pass greedy.
2. Re-run the clean layer-0 tune on this branch. Flash now compiles AND is numerically correct; watch whether the
   misaligned fault still fires and whether the tune completes.
3. If the misaligned fault recurs: reproduce that single variant, run under compute-sanitizer, fix the alignment gate
   (Finding-4 class) — and grab a `py-spy dump` if the parent hangs again to confirm the wedged-thread hypothesis.
4. Only after a completed tune does a v3 bench table become meaningful. Expect the certified single-kernel flash to
   collapse v2's residual 50.9 ms `k_sdpa_mean_linear_reduce` + 3.1 ms score-slice into one warp-tier kernel near the
   isolated 182 µs.
