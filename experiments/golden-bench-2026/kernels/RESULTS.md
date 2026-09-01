# Golden-bench kernel corpus

## Manual child-schedule qualification after #689 (2026-08-29)

Manual route selection confirmed that child scheduling works once placement exposes ordinary kernels. The exact source
was `226619ca`; the retained A100, V100, and RTX 4090 lanes used deployable O3, bounded searches, strict eager
correctness for admissible rows, and fresh `torch.compile` measurements where the current CLI could produce them.

The A100 s512 attention all-four cut produced five kernels, but initially failed before launch with
`KeyError: linear_1_wt`. Cut-piece graph inputs came from lowered root statements, which omit contraction operands
retained structurally beneath a projection region even though kernel materialization later expands and reads them.
The historical cut-local fix was commit `a9672422`. Main's later generic schedule reconstruction superseded it:
`loaded_buffers` inventories the complete stored Fold tree, and both piece inputs and producer ordering consume that
shared reading. At the historical source, the route carried all 23 required inputs and exercised A100 tuning. Its gate
passed 3,993 tests with 1,012 skips and five expected failures; focused replay passed 22 tests with one skip.

A second bounded reduction found that legal placement sites beneath a projection region were absent from provider
closure: the lexical-environment walk only visited direct Fold members while placement used the complete stored-child
walk. Historical commit `8c2bb5cb` fixed that locally; main now supersedes it with the generic `_stored_folds` traversal
and its regression test. At that source, the normalized-K projection plus rotary seam lowered as two
kernels, but a complete correct s512 assembly still had another child that exceeded the 15-second bound.

Manually selected A100 child rows demonstrate that schedule quality is not the immediate blocker:

| child role | best correct O3 latency | schedule highlight |
| --- | ---: | --- |
| s512 attention statistics | 2.322 µs | `WORK=t128`, cooperative reduction |
| s512 elementwise split | 5.3 µs | `WORK=t128`, cooperative reduction |
| s512 MLP product | 86.229 µs | `WORK=w2x2`, f16 MMA `f4x8/k8` |
| s1 final consumer | 60.536 µs | `WORK=w1x1`, f16 MMA `f1x4/k8`, synchronous shared-memory stages |

No complete A100 route is promotable yet. Every legal composed cut retains at least one large child that recomputes a
Q/K/V projection, rotary transform, SDPA, or output projection inside another projection region. Those children
either expose no applicable MMA schedule or take 10-15 seconds per launch. The next compiler boundary is therefore
value materialization across independent projection regions, not a larger schedule search.

The V100 manual screen found one correct and repeatable schedule improvement. The standalone model-sized Volta
projection fell from 3,571.7 µs to 1,868.8 µs with `WORK=w4x2`, f16 MMA `f2x4/k4`, and `STAGE=d1/smem`, while
`torch.compile` measured 756.736 µs. A fresh repeat measured 1,881.088 versus 757.760 µs with strict zero error. The
exact-card CLI recorded the first pair in the realization case. These measurements predate main's #691 generic
schedule reconstruction and are historical rather than current-head latency evidence; the authored row still passes
the current GPU-free realization checks but needs an exact V100 timing refresh. At the measured source it was a 1.9x
Emmy improvement but remained about 2.5x behind; its CUDA used two CTA barriers in a 192-iteration K loop, 123
registers per thread, and 25% occupancy.

The V100 linear-cut diagnostic fell from 1,160.2 µs to 247.0-247.8 µs by selecting the unsplit child, versus
65.7-66.0 µs for `torch.compile`, but it is not admissible: both strict repeats fail with 26,441 mismatches and maximum
absolute error 0.25. Eight tensor-core schedules failed identically, and a plain large FP16 `F.linear` reproducer
without placement or residual addition has the same numerical character. That row is retained only as diagnostic
evidence and must not be promoted under the strict contract.

The numerical discrepancy reduces to FP16 `[1,17,1536] × [17,1536]`. Emmy emits FP32-accumulating Volta MMA across
the complete K dimension and performs one final FP32-to-FP16 store. Disabling PyTorch's reduced-precision FP16 GEMM
reduction makes the unchanged Emmy row pass the strict comparison, identifying reference reduction policy—not an
intermediate Emmy FP16 carrier, split-K, or store—as the distinguishing semantic. A proposed `_xfail_correct` corpus
case correctly XPASSed because corpus correctness uses the NumPy backend and its documented narrow-operand tolerance;
it cannot encode a PyTorch-eager policy mismatch. Reproducing cuBLAS's reduced-precision chunking would require an
explicit compiler numerical-policy decision, so no tolerance or speculative code change was committed.

The RTX 4090 selector again found zero exact sm89 corpus cases. A fresh Qwen3-0.6B s512 trace produced six targets;
the 29-origin target exceeded a six-second watchdog, while manual cuts produced best measured child totals of about
152,231 + 64 µs and 153,764 + 56 µs. The dominant child preserves the same repeated attention computation as the
A100 route, so further knob search is not a plausible route to parity. No sm89 realization or canonical golden was
promoted without repeated correct reference measurements.

Manual scheduling therefore improved the measured ceilings and exposed one fixed compiler bug, but did not establish
parity. Across Ampere and Ada the next shared problem is forming reusable value boundaries before child scheduling;
on Volta the independent remaining problems are synchronous staging efficiency and the strict large-linear numerical
contract. At the historical source, the combined local gate passed 3,994 tests with 1,012 skips and five expected
failures; lint remained clean. The A100 VM remained running.

## Post-#689 structural-identity qualification (2026-08-29)

The draft was rebased cleanly onto main `e782e991`; exact qualification source was `79f14e3b`. Main's canonical
Loop-IR identity migration deliberately orphans earlier tune-DB rows. `make test-corpus-regen` reported the realization
corpus current, and its GPU-free replay passed 623 cases with 424 skips and five expected failures, so its 210 identity
stamps need no branch-local regeneration.

The two staged A100 experiment goldens do need more than a key update. Reconstructing their embedded stable Torch IR
on current main yields four s1 targets and six s512 targets, instead of the nine provenance targets in each checked
file. The s1 layer now has three small targets plus one 73-origin fused target. The s512 layer has four small targets
plus 53-origin and 29-origin fused targets. The old provenance selections therefore cannot be renamed onto the new
identities.

A fresh, isolated O3 A100 tune used a four-candidate cap and patience two. The three s1 small targets completed in
3.5-29.2 seconds with 1.1-2.0 µs winners. Four s512 small targets completed in 4.5-5.8 seconds with 1.684-3.081 µs
winners. The first candidate for every fused target exceeded the 60-second kernel watchdog: the s1 73-origin target,
the s512 53-origin target, and the s512 29-origin target. None produced a measured cut assembly or child schedule
receipts. The regenerated working files are retained as host-local evidence, but were not promoted because an
incomplete file would replace replayable goldens with unqualified fused rows. Current regeneration therefore needs
the tuner to select a cut before benchmarking the fused terminal, then tune and persist the resulting child identities
as one replayable assembly.

The exact post-rebase A100 realization corpus retained two wins, three losses, and the existing stat-fill watchdog:

| A100 case | Emmy | `torch.compile` | result |
| --- | ---: | ---: | --- |
| GQA B cut | 7.026 µs | 10.030 µs | 1.43x faster |
| computed-value attention cut | 6.072 µs | 8.018 µs | 1.32x faster |
| Q/K workspace chain | 84.224 µs | 11.947 µs | 7.05x slower |
| unit-row split-K | 9.113 µs | 2.856 µs | 3.19x slower |
| batched PV transpose | 17.682 µs | 11.658 µs | 1.52x slower |
| stat-fill | watchdog | — | unchanged failure |

Both current SM70 cases passed correctness on V100. Linear-cut measured 1,160.192 µs versus 66.048 µs for
`torch.compile`; Volta MMA measured 3,129.344 µs versus 754.688 µs. The command's nonzero status was only the
missing-duration guard for these newly measured rows. The regenerated corpus still contains no exact SM89 closed case.

The failed fused candidates exposed one independent bounded-tuning bug: after reporting `HungKernelError`, the child
returned through normal Python teardown, where CUDA waited forever for the still-running kernel. The worker now
hard-exits after writing the failure response, and a CPU-only regression test proves that Python exit handlers cannot
wedge the next candidate. This restores the existing clean-worker continuation contract; it does not improve a kernel.
No new golden or large serving result was promoted, and the retained A100 remains running.

## Post-merge qualification and measured-row replay fix (2026-08-29)

Qualification resumed from merged main `5f0a2076` on branch `codex/post-679-corpus-qualification`. The exact-main
corpus run reproduced the prior platform verdicts; the merge's review follow-up did not regress a pinned kernel.

| platform | exact coverage | result |
| --- | --- | --- |
| A100 sm80 | 6 cases | 2 wins, 3 measurable losses, and the same stat-fill watchdog |
| V100 sm70 | 2 cases | Volta MMA remains a correct 4.13x loss; linear-cut remains incorrect and inadmissible |
| RTX 4090 sm89 | 0 cases | 201 collected and skipped; no exact sm89 closed case |

The exact-main A100 rows were 7.041 µs versus 10.016 µs for GQA and 6.076 µs versus 8.009 µs for computed-value
attention. The remaining pinned rows were 84.224 µs versus 12.902 µs for the workspace chain, 9.060 µs versus
4.175 µs for split-K, and 17.664 µs versus 11.106 µs for PV transpose. Stat-fill again exceeded its watchdog. V100
reproduced 3,130.368 µs versus 756.736 µs for the promoted Volta row and 1,160.192 µs versus 66.048 µs for
linear-cut; strict linear-cut correctness still found 26,418 mismatches with maximum absolute error 0.25.

A fresh 10-candidate workspace tune exposed one deploy replay gap. Its fastest DB row was 76.012 µs, but direct
descent compared the schedule tree's intermediate `S_warp_eligible` feature with a feature-free measured row after
the `S_*` signature had already joined them. Every measured row therefore appeared disjoint and deploy fell back to
the prior at about 85 µs. Historical commit `d2505362` ignored `S_*` / `H_*` fork keys during tuning-knob descent.
That implementation is now in main; this branch retains its focused measured-row regression coverage. At the measured
source, exact A100 strict replay selected `WORK=t8` at 76.0 µs with direct eager correctness rather than the prior's
`WORK=t4` fallback. This closed evidence replay but not the roughly 6.9x workspace performance gap.

Stat-fill's remaining persistence gap is larger. A four-candidate fused probe stayed at 277,229 µs and produced no
children. Replaying the previously correct six-cut route did enroll its resulting kernels: a four-candidate cap per
kernel completed 53 measurements and found a 2.093 µs child. The written working golden nevertheless contained only
the route seed and a 9,950.886 µs placement total—no child identities and no child schedule receipts. That total is
not deploy evidence because a reload cannot reconstruct the measured ordered assembly. The required change is
split-first persistence: record the route decision, realize the resulting kernel identities, then join each identity
to its measured schedule before writing a replayable winner. This is not a safe small follow-up to the DB descent
fix, and no unqualified stat-fill row was promoted.

Draft PR #687 carried the original replay fix. Its local verification was 3,982 passed, 1,012 skipped, five expected
failures, and clean lint; its GitHub test and lint jobs were green. The A100 VM remained running.

## Current-head corpus requalification (2026-08-29)

The draft is based on current main `b88763fa`; the exact combined source for this pass is `857ba7e9`. Every hardware
run used deployable O3, the exact GPU capability, task-owned tuning and cubin state, the repository CLI, and strict
direct correctness where a timing was admissible. The retained A100 VM stayed running.

### A100 corpus

The exact sm80 lane collected 201 tests and found six applicable cases. It completed in 189 seconds with five passes,
one independent stat-fill watchdog failure, and 195 skips. The result JSON and full log are retained host-local on the retained A100 VM (untracked
`_tune/a100-corpus-857ba7e9/evidence/full-corpus/`; `_tune/` is not in the repository).

| A100 case | Emmy | `torch.compile` | launches | result |
| --- | ---: | ---: | ---: | --- |
| `attention/rmsnorm-gqa-b-cut.yaml` | 6.907 µs | 10.026 µs | 2 | Correct; 1.45x faster |
| `attention/sdpa-computed-value-cut-mma.yaml` | 6.193 µs | 8.039 µs | 2 | Correct; 1.30x faster |
| `attention/rmsnorm-qk-sdpa-workspace-chain.yaml` | 75.855 µs | 11.469 µs | 1 | Correct; 6.61x slower |
| `matmul/f16-cut-splitk-unit-row.yaml` | 9.183 µs | 2.839 µs | 3 | Correct; 3.23x slower |
| `matmul/f16-mma-broadcast-batched-pv-transpose.yaml` | 17.664 µs | 11.894 µs | 1 | Correct; 1.49x slower |
| `attention/rmsnorm-gqa-sdpa-stat-fill.yaml` | about 280 ms/iteration | 19.456 µs | — | Pinned row exceeds the aggregate 10-second watchdog |

The GQA win required both a better route and one replay fix. Materializing the clustered normalized-K value produces
one cooperative producer and one consumer. The A/B path then incorrectly dropped scoped OFF exceptions such as
`REDUCE@a7=''`, allowing bare `REDUCE=coop` to fan out and change the consumer source. Replay now preserves a scoped
OFF when it overrides a non-OFF bare family. The unchanged perf command fell from 10.193 to 6.857-7.100 µs and uses
the same fast source as direct full-row replay.

The PV golden now carries the strict `w8x1`, `f1x8/k8`, `d1/smem-async` schedule. This is about four times faster than
the prior checked row, but an eight-row neighbour search found no further schedule gain. The kernel uses 48 KiB shared
memory, 126 registers per thread, and 25% occupancy. Its transpose epilogue emits 32 stride-512 scalar stores per
thread; toggling vector stores produces byte-identical CUDA. Parity therefore needs a transpose-aware store path or a
different contraction orientation.

The workspace-chain golden improved 13.5% by using `WORK=t8` and cooperative reductions, with the combined lane
reaching 75.855 µs. Its CUDA still recomputes Q RMSNorm within output-key work, K RMSNorm within each dot product, and
the softmax score scan across value-output lanes. No offered schedule changes that structure; reuse or materialization
must move those cones outside the repeated loops.

The split-K case remains a launch-structure gap. Its fastest strict forced row with a genuine split used `g2k+w1x1`
at 4.681 µs versus 2.761 µs for `torch.compile`, but the corpus row remains the repeatable authored 9.183 µs result.
Deferred split-K requires partial, finalize, and cut-consumer kernels. Atomic split-K removes the finalize kernel but
needs a runtime accumulator reset because its first kernel has no predecessor to own zero initialization. A safe
improvement needs a cross-CTA last-arriver primitive or a proved consumer-side reset protocol.

The stat-fill case now builds and passes strict correctness after preserving provider evaluation domains, ordering
sibling operands by dependency, closing tiled provider cones, and retaining computed-B projection providers. Its
authored fused schedule is nevertheless about 0.28 seconds per iteration and trips the benchmark watchdog. A correct
six-seam route reached 3,342 µs versus 17.238 µs for `torch.compile`; its best measured factor-16 child was 184.525 µs.
The parent instead selected an unmeasured factor-32 split, so a 206.592 µs parent canary is not qualified evidence.
The remaining storage gap is ordering in the child-identity schedule receipts: record the split choice first, then join the identities
and measured schedules of the children that choice creates.

### Other exact platforms

| platform | exact corpus coverage | current result |
| --- | --- | --- |
| V100 sm70 | 2 cases | Volta MMA is correct at 3,130.368 µs versus 757.760 µs; linear-cut latency is inadmissible because strict correctness still has 26,418/524,288 mismatches |
| RTX 4090 sm89 | 0 cases | 201 collected, 201 skipped; no exact sm89 closed case and therefore no parity claim |

The promoted V100 row uses `WORK=w4x2` and a single `d1/smem` stage. It shares A tiles across two N warps and B tiles
across four M warps, reducing shared-memory requests, but remains 4.13x behind `torch.compile`. Volta lacks
`ldmatrix` and `cp.async`; the next useful schedule primitive is a producer/consumer warp band with named barriers,
not another depth or geometry row. The linear-cut target still differs in contraction accumulation order after the
public f16 boundary is restored, so its perf-harness number is reported only as diagnostic output.

### Retained fixes and conclusion

This pass retains small, separately tested fixes for provider evaluation domains, dependency-ordered operand splicing,
scalar-atom dump replay, direct tuning of persisted unscheduled Tile children, provider-cone closure, and scoped-OFF
A/B replay. It promotes the GQA, PV, workspace-chain, and V100 schedules. The combined local gate is 3,981 passed,
1,012 skipped, and five expected failures; lint is clean.

Parity is not achieved: A100 has two wins, three measurable losses, and one watchdog; V100 has one correct loss and
one correctness failure; RTX 4090 has no exact corpus coverage. No large serving experiment was started while those
kernel-level gaps remain. The next compiler work is structural reuse/materialization for attention, transpose-aware
PV stores, a cross-CTA completion/reset primitive for split-K, split-first ordering for stat-fill's child-identity
schedule receipts, and a Volta
producer/consumer staging primitive.

## Cut-pinned attention qualification after the computed-B changes (main through `d2950079`, 2026-08-28)

### Corrected protocol

The first screen put `PLACE` choices in proposal `knobs`. That measured one structural candidate whose new kernels
kept greedy schedules; it did not test the route the compiler is designed to tune. This follow-up instead froze the
kernel set in realization `pins`, ran the normal two-level tuner on every minted kernel identity, and replayed the
assembled route from the same isolated evidence state. A CPU regression test now protects that exact contract: a
pinned placement cut enrolls both children and the assembly replays different `WORK` and `STAGE` rows.

The full route has four distinct value seams: shared statistics (`PLACE@map.fold.a21`), Q
(`PLACE@a.map.a`), K (`PLACE@a.map.b`), and softmax weight (`PLACE@map.fold.a1`). The three previously tested K
spellings resolve to three different Fold occurrences, but value clustering groups them into one K-value `CutSite`;
pinning any occurrence replaces all three. They are one cut, not three composed cuts.

The first measurement lanes used exact `6e6181d5` source, deployable `-O3`, isolated tuning state, seed 0, at most 12
candidates per independent kernel, patience 4, and an outer wall bound. The receipt-aware follow-up rebased the draft
onto exact `a597f15d` and regenerated the working files before inspecting or measuring any schedule. A fresh trace
still produces one maximal whole-layer target, so these diagnostics use untrusted copies of the checked self-contained
score/statistics slice. They are host-local compiler qualification, not replacement publication evidence; no results
archive was changed.

### Hardware result

| platform | frozen route | result |
| --- | --- | --- |
| V100 | K-value cut; two children | Correct replay: 595,101 µs versus 1,920 µs eager. No child candidate finished inside the bounded search, so replay correctly used the offline fallback. |
| A100 | statistics + Q + K + softmax weight; five primary children plus one recursive statistics split | 67 clean benches in 219 s. Best children were 7-37 µs except the softmax-weight producer at 110,744 µs; tuned route 110,882 µs. Fresh replay was 110,723 µs versus 758 µs eager and passed direct correctness. |
| RTX 4090 | statistics + Q + K; four launches | Per-identity DB replay used different child schedules: 4,561 µs versus 494 µs eager, direct correctness passed. The remaining consumer was 4,396 µs. |
| RTX 4090 | add softmax weight; five launches | The consumer fell to 19-31 µs, but the new producer's best row was 100,416 µs; replay was 101,233 µs versus 504 µs eager and passed direct correctness. |
| RTX 5090 | statistics + Q + K + softmax weight; structural replay only | Exact lowering produced the expected five children. Timing was deferred because an unrelated task owned the host's only compatible GPU; it was not interrupted. |

### Receipt-aware current-main retune

The follow-up regenerated the working targets after rebasing. V100, A100, and RTX 4090 measurements used exact
`043f1f25`, which adds only the route-contract test to `a597f15d`; RTX 5090 used exact `5ddf7816`, whose receipt
decoder change does not alter kernel source. All rows used deployable O3 and bounded candidate or explicit-row
budgets.

| platform | child | best bounded row | result |
| --- | --- | --- | --- |
| V100 | K-cut cast producer | `TILE=f2`; other schedule families off | 2.924 µs |
| V100 | K-cut pointwise producer | `TILE=f4`; other schedule families off | 2.686 µs |
| V100 | K-cut attention consumer | only offered row: all schedule families off | exceeded the 15 s watchdog; no accepted latency |
| A100 | softmax materialization (`c3d`) | `WORK=t128, REDUCE@a3=coop, REDUCE@a4=coop` | qualifying repeats 110,768 and 110,786 µs |
| RTX 4090 | softmax materialization (`c3d`) | `WORK=t128, REDUCE@a3=coop, REDUCE@a4=coop` | search observations 100,351 and 100,335 µs; a noise-scale tie with the prior 100,416 µs row |
| RTX 4090 | other four-cut children | per-child rows: statistic `t32/coop`, Q all-off, K `t128/coop`, consumer f2x8 MMA with async stage | 4.15-27.89 µs |
| RTX 5090 | softmax materialization (`c3d`) | `WORK=t128, REDUCE@a3=coop, REDUCE@a4=coop` | after the compiler fix, qualifying repeats 72,380 and 71,966 µs |

The A100 c3d candidate-pool bound is 1,094,745,632 rows and the consumer bound is 1,066,670,432. Whole-target MCTS
spent 8m30s in first-candidate CPU descent without measuring a row. On RTX 4090, the 24-live-candidate MCTS-only arm
took 407.5 s and the equally bounded evidence-seeded refinement reached its 600 s wall; neither found a different c3d
schedule. Exhaustive child-row listing and strict receipt decoding were each stopped at 60 s. The useful schedules
are visible by deploy identity, but flattening these pools is not a usable listing or validation algorithm.

Current main's child-identity schedule receipts close the representation gap, and `5ddf7816` fixes strict decoding
when a regenerated target lowers to several kernels. Exact deployment is not closed yet. On RTX 4090 the canonical
t128 receipt joined the correct c3d identity but reported row DRIFT and fell back to t8: 397,303 µs for c3d and
397,477 µs for the four-cut route versus 480 µs eager, with direct correctness passing. The explicit working-file
path also treats receipt siblings as independent flat A/B rows rather than installing them together for base
lowering. No receipt was promoted; the remaining work is a child-directed exact-row descent shared by strict decode
and the verified tier, plus grouped working-file replay.

RTX 5090 exposed one independent built-stage gap: every screened row initially emitted ambiguous `float * __half`
expressions under readable CUDA rendering. The readability fold had inlined a mixed-dtype single-use `Assign` before
the target-aware renderer could insert `__half2float`. The compiler now keeps such assignments named; the new closed
sm120 realization case proves offered, realized, built, and correct. The repaired explicit rows measured 72,488 µs
at t32, 72,405 µs at t64, 71,969 µs at t128, 73,936 µs at t256, and 73,362 µs at t512. This closes compilation but
does not change the repeated 512×128 work.

### Statistics-sharing replay after #682

PR #682 (`d2950079`) directly closes the repeated-statistics gap identified above: Tile normalization restores object
sharing between structurally equal cones, and two provider-closed statistics seams make the shared row state
materializable. On the exact Qwen s512 target, adding those two cuts to the prior four-cut route produces six launches.
The statistics producer writes max and normalization state once per `(head, query)` row; the softmax-weight child
loads that workspace and no longer contains the 512-key scan. The consumer also loads the shared state rather than
recomputing it.

The retained A100 was replayed at exact branch source `31e7e629` (PR #682 plus this draft's receipt and readable-CUDA
fixes), deployable O3, isolated DB/prior/cubin state, five warmups, and 20 iterations. Both standard repeats passed the
strict direct eager check.

| lane | eager (µs) | Emmy route (µs) | dominant statistics child (µs) | result |
| --- | ---: | ---: | ---: | --- |
| standard repeat 1 | 744.653 | 11,717.632 | 11,501.568 | correct; 6 launches |
| standard repeat 2 | 744.795 | 11,720.704 | 11,505.664 | correct; 6 launches |
| `FAST_MATH` | 744.590 | 11,704.320 | 11,501.568 | correct; noise-scale 0.1% change |

The prior four-cut full replay was 110,723 µs, so the shared-statistics route is 9.45x faster. The old c3d child falls
from about 110,780 µs to 65-66 µs; the consumer is 81 µs and the other three producers are 7-37 µs. This is a real
algorithmic improvement, but the route remains 15.7x slower than eager. The matched `torch.compile` request again
produced no positive timing for this embedded target, so it still cannot supply a parity ratio.

The new bottleneck is the one correctly shared statistics producer, not duplicated work. Its greedy row is
`TILE@a4=f1x2, WORK=t32x8` and takes 11.50 ms. A bounded MCTS-only follow-up measured
`TILE@a4=f4x6, WORK=t32x16` at 11.535 ms and found no improvement; after 2m55s the next candidate remained in CPU
descent with the GPU idle, so the arm was stopped. The remaining performance work is to make the nested 128-channel
score contraction inside the online 512-key statistics reduction eligible for an efficient tensor-core schedule,
then reduce the still-large candidate descent. No receipt was promoted. Raw host-local evidence is retained under
`_tune/pr682-a100/remote/`; the task-owned remote scratch was removed while the A100 VM stayed running.

`torch.compile` produced no positive latency for these score/statistics strict replays, including the post-#682
attempt, so no parity ratio is claimed.
The earlier output-projection strict result remains a valid separate finding: 1,594,544 µs for Emmy versus 52.6 µs
for `torch.compile` on RTX 4090. The V100 down-projection also remains a direct correctness failure and was not
admitted as a performance result.

### Bottleneck and receipt-aware replay

Before #682, the correction changed the diagnosis. Placement worked, and resulting kernels were independently
schedulable. On A100 and RTX 4090, four children tuned into the tens-of-microseconds range; materializing the softmax
weight isolated one producer that remained about 100-111 ms. Its lowered loop had free query and output-key axes and,
for every output weight, recomputed the complete 512-key reduction whose body performs a 128-channel score
contraction. Ordinary `WORK` and `REDUCE` choices changed the constant factor but preserved that repeated scan. PR #682
closes that reuse gap; the statistics-sharing replay above supersedes this performance state. On V100, even the K-cut
consumer did not complete a candidate inside the original search wall.

The earlier cold-replay drift was a separate persistence gap: the DB keyed different rows by child structural
identity, but the old flat realization could not serialize conflicting child-global `WORK`, `TILE`, `REDUCE`, `STAGE`,
or `RASTER` values. Main `a597f15d` resolves that representation gap with child-identity schedule receipts. Each
sibling realization carries the route cuts in `pins`, one child's row in `knobs`, and that child's `deploy_identity`
in `identity`; strict decoding checks the row only against that child's candidate pool. Copying child rows into the
parent flat map remains invalid, but the sibling receipts make exact per-child replay representable in the schema.
The current-main retune above shows that strict enumeration and deploy equality still need a child-directed descent
before those receipts are promotion-ready for this large route.

No realization was promoted. Offered, realized, built, and correctness stages are closed for the composed route, so
there was no small compiler failure or new realization-corpus gap to patch. `FAST_MATH` was not promoted because the
standard route remained far behind eager and changing contraction math does not remove the isolated producer cost.

## Host-local exact-card qualification checkpoint (2026-08-28)

### Question and scope

Can the current Qwen3-0.6B FP16 layer-0 kernel inventory match or beat `torch.compile` on the exact V100, A100,
RTX 4090, and RTX 5090 cards after bounded retuning? This pass covered the same nine targets at sequence lengths 1
and 512 on each card: 18 targets per platform. It did not retune the historical FP8, 32B, H200, B200, or serving
lanes, so it supports no claim about those workloads.

This was bounded tuning and exact `emmy run --golden-file ... --bench` qualification, not a new `emmy bench` recipe
snapshot. The checked-in `results_*.tar.gz` files and the later platform sections therefore remain the earlier
archived runs. Current raw evidence is retained in the ignored `_tune` directories listed below; it is not presented
as a replacement experiment record or archive.

Consequently, this section is a host-local working checkpoint for compiler and golden review, not durable publication
evidence. The paper must not cite its exact counts or timing ranges until the recipe is rerun and the per-platform
raw-results archives are replaced through the normal experiment workflow.

### Protocol and acceptance rule

- Model: `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, layer 0, FP16, static sequence lengths
  1 and 512.
- All measurements used deployable `-O3`. Ordinary screens used five warmups and 20 iterations; promoted finalists
  used 10 warmups and 100 iterations in at least two fresh processes. Hard targets used one warmup and one iteration
  under a bounded watchdog rather than extending the run indefinitely.
- Direct Emmy-versus-eager correctness was the admission gate. `torch.compile` was compared only when it compiled
  the whole target and passed its own eager check; an unavailable baseline did not turn a correct Emmy target into a
  failure. A hung or incorrect Emmy target remained unresolved.
- Win/tie/loss follows the preregistered two-percent rule in the experiment README. Counts use only targets with a
  valid `torch.compile` result, and the denominator is reported explicitly.
- `FAST_MATH` was considered only when the standard row had empty compile flags and the fast-math row independently
  passed direct eager correctness. It was promoted only when it changed the performance conclusion.

### Platform summary

| platform | direct eager correct | comparable with `torch.compile` | win / tie / loss | baseline unavailable | unresolved | current golden status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| V100 | 16/18 | 14 | 1 / 0 / 13 | 2 | 2 | REPOSITORY validation passes 18/18; one row fails runtime correctness |
| A100 | 16/18 | 15 | 5 / 1 / 9 | 1 | 2 | sequence 1: REPOSITORY 9/9; sequence 512: WORKING 7/9 |
| RTX 4090 | 17/18 | 15 | 3 / 1 / 11 | 2 | 1 | REPOSITORY validation passes 18/18 |
| RTX 5090 | 18/18 | 16 | 3 / 1 / 12 | 2 | 0 | REPOSITORY validation passes 18/18 |

The result does not establish cross-platform parity with `torch.compile`. Emmy matches or beats it on 1/14
comparable V100 targets, 6/15 A100 targets, 4/15 RTX 4090 targets, and 4/16 RTX 5090 targets. Several targets,
especially attention targets, are orders of magnitude slower or hang; the V100 MLP down-projection remains
numerically incorrect.

The strongest new positive promotion is the RTX 4090 sequence-512 input RMSNorm: `WORK=t256, REDUCE=coop` measured
2.3641-2.3663 µs in the qualifying repeats, against 4.6793-4.6999 µs for `torch.compile` and
126.61-126.68 µs for eager. The repaired A100 sequence-1 score/statistics schedule repeated at 34.99 µs and passed
direct correctness, but it still loses to the 11.26-12.16 µs `torch.compile` result.

### Main unresolved targets and performance losses

| platform | target role | result |
| --- | --- | --- |
| V100 | sequence-512 down-projection + residual | all four bounded schedules are incorrect; the stored row has 40/524,288 mismatches |
| V100 | sequence-512 o-projection + residual | the selected kernel exceeds the internal watchdog |
| A100 | sequence-512 softmax times V | correct greedy execution is about 327 ms versus about 46 µs for `torch.compile`; no safe recorded route |
| A100 | sequence-512 o-projection + residual | bounded execution hangs, so no timing or correctness result is admitted |
| A100 | sequence-512 score/statistics | the corrected lowering path still hangs under the bounded greedy replay |
| RTX 4090 | sequence-512 score/statistics | correct, but 164-372 ms; `torch.compile` is unavailable |
| RTX 4090 | sequence-512 softmax times V | correct, but 202-235 ms versus 36.1 µs for `torch.compile` |
| RTX 4090 | sequence-512 o-projection + residual | warmups take about 3.69 s before the internal watchdog aborts |
| RTX 5090 | sequence-512 score/statistics | correct at about 397 ms; `torch.compile` is unavailable |
| RTX 5090 | sequence-512 softmax times V | correct at about 152 ms versus 30.8 µs for `torch.compile` |
| RTX 5090 | sequence-512 o-projection + residual | correct at about 3.35 s versus 40.1 µs for `torch.compile` |

The V100 correctness diagnosis found that fusion removes a public f16 rounding boundary before the residual add.
Restoring that boundary reduced mean error but did not reproduce eager matmul accumulation and rounding, and it
changed the kernel identity. The partial change and its provisional realization case were therefore reverted; the
target remains an explicit compiler correctness gap rather than a promoted schedule.

### `FAST_MATH`

Only the sequence-512 q-projection changed category: RTX 4090 improved from about 20.7 µs to 16.4 µs against a
20.4 µs `torch.compile` result, and RTX 5090 improved from about 19.1 µs to 16.6 µs against roughly 19-20 µs.
Other direct-correct fast-math rows on A100, V100, RTX 4090, and RTX 5090 were ties or losses and were not promoted.
Rows that changed outputs were rejected regardless of speed.

### Compiler changes established by this pass

- Singleton reduction collapse, computed-A statistic-fold selection, and guarded unit-row recovery close reduced
  correctness or realization failures seen in the Qwen targets.
- Golden feature derivation, exact path retry, measured-row latency recording, and bounded cold-pool descent repair
  replay and search without adding a separate benchmark harness.
- Placement-cut capture preservation, unit-axis preservation, scoped-cut consumption, and output-sweep promotion
  make the affected schedules realizable. The affected closed realization cases reached `built` with nvcc on their
  exact-capability cards.
- A proposed structural-route receipt was audited and reverted. The final design fails closed: whole-slice
  `PLACE` measurements are not written as deploy evidence, and both tuning-database and online-reservoir measured
  tiers reject legacy rows containing `PLACE` or `PLACE@...`. Search may retain those rows for ranking and training,
  but automatic deployment cannot use them without an exact child-schedule receipt.

The final compiler suite reports 3,911 passed, 990 skipped, and 5 xfailed; `make lint` passes. The route contract
is now a positive fail-closed test replacing one prior xfail.

### Systems and evidence

| platform | exact GPU and software | qualification source | ignored local evidence |
| --- | --- | --- | --- |
| V100 | Tesla V100-SXM3-32GB, `GPU-b415579d-cdad-42bb-23d1-32c20cdb729d`; driver 580.159.03, nvcc 12.9, PyTorch 2.13.0+cu126 | `0e4729d5` | `_tune/v100-current/current-head/` |
| A100 | A100-SXM4-80GB, `GPU-80df657e-2e14-421c-32a5-cb2429dc93e6`; driver 580.65.06, nvcc 12.9, PyTorch 2.13.0+cu130 | `15c27422` | `_tune/a100-current/` |
| RTX 4090 | GeForce RTX 4090, `GPU-81d79c00-868e-3ec5-2948-745283b756f6`; driver 580.159.03, nvcc 13.3, PyTorch 2.13.0+cu130 | `7b5161e8` | `_tune/rtx4090-current/safe-head/` |
| RTX 5090 | GeForce RTX 5090, `GPU-bb78f2c5-11d6-02d6-f124-08b719623110`; driver 580.173.02, nvcc 13.0, PyTorch 2.13.0+cu130 | `5e95d2de` | `_tune/rtx5090-current-source-revalidate/` |

The compiler tree after the final safety patch is `c9d8a19e`. Later changes relative to a card's qualification source
are search or safety changes, except for measurement-only golden promotions already qualified at the listed source.
They do not supply unmeasured performance claims for that card.

### End-to-end decision

Large serving experiments were intentionally skipped. The kernel gate is not healthy: the common corpus is
incomplete on V100 and A100, and every card has major `torch.compile` losses in attention. Running a long serving
matrix now would consume hardware without supporting the requested across-platform compiler claim.

## Earlier archived refresh: first fix wave (main @ 001d4f44)

## What this refresh is

Second measurement pass over the corpus, after the maintainer's fixes landed (#561 split-axis re-fusion, #556
conv1d/einsum lowering, #549/#547 FA restoration, #513-era search changes). Same three GPUs, fresh hosts,
recipe budgets, three `-O3` repeats per target. Each `goldens/<model>_<gpu>.yaml` was rebuilt from a fresh
trace + tune + 3-repeat verification on `001d4f44`; the previous values remain in git history for comparison.

Runs were driven by an updated benchmark flow — one row per committed golden on its exact GPU, three `-O3`
repeats of `emmy run --golden <file> --bench --bench-backends eager,tcompile,emmy`, no tracing or tuning at
measurement time — kept UNCOMMITTED in this PR per review direction; `recipe.yaml` in-tree is unchanged.

## Before/after (sum of measured kernel targets, median of 3 repeats, µs; old = previous committed goldens)

| platform | model/seq | old emmy | new emmy | gain | new vs eager |
| --- | --- | ---: | ---: | ---: | ---: |
| v100x1 | 0.6B s512 | 71714 | 58981 | 1.2x | 0.07x |
| v100x1 | 0.6B-FP8 s512 | 147160 | 71084 | 2.1x | 0.10x |
| v100x1 | 32B-FP8 s512 | 408241 | 395773 | 1.0x | 0.06x |
| rtx4090x1 | 0.6B s512 | 35305 | 34466 | 1.0x | 0.03x |
| rtx4090x1 | 0.6B-FP8 s512 | 34496 | 24525 | 1.4x | 0.04x |
| rtx4090x1 | 32B-FP8 s512 | 650248 | 117839 | **5.5x** | 0.08x |
| rtx5090x1 | 0.6B s512 | 32870 | 27620 | 1.2x | 0.03x |
| rtx5090x1 | 0.6B-FP8 s512 | 28830 | 17843 | 1.6x | 0.10x |

Decode (s1) rows improved 4-13x on the FP8 corpora but changed measurement coverage (19 -> 8-11 targets, from
new fusion identity plus bench failures), so their sums are not clean ratios; per-target values are in the
archives. Matched-kernel gains on the V100 0.6B corpus: geomean 2.6x, led by q_proj 4.5x (992 -> 219 µs, 0.88x
eager on Volta — #561's tensor-core unlock confirmed in silicon), k/v_proj 3.4x, and the SDPA matmul fusions
21.5x / 16.8x (FA restoration).

## Why layer totals still trail torch.compile: the remaining defects, diagnosed

1. **RoPE-fusion statistic replay (dominates every s512 total; unchanged).** `k_sdpa_mean_reduce`'s loop nest
   recomputes the k-norm statistic (a full 512x128 reduce) inside every q-row iteration — a 512x replay
   (~23000-29300 µs of each card's s512 total; the s1 variant is fine, replay factor 1). Consistent with the
   #513 guard removal enumerating this fused form and cold greedy deploying it. Fix directions: hoist
   loop-invariant statistics in loop/canonicalize, or make the placement-cut alternative evidence-reachable
   cold. Diagnosis-only here per review direction.
2. **Computed-A (fused norm+gate/up) misdeploys, worst at decode.** New extreme case: on the rtx4090,
   `k_linear_mean_reduce_549927.s1` deploys KNOBLESS at **116445 µs vs eager 108** (~1000x); the V100 s512
   sibling regressed 679 -> 1035 µs. The search reaches no schedule for this form and the fallback is
   catastrophic.
3. **Qwen3.6-27B capture advanced one op and is blocked again**: conv1d now lowers (#556), the trace now stops
   at `aten.masked_fill requires resolved self, mask, and fill inputs`. Still no 27B golden.
4. **Hung kernel under the 32B corpus on V100**: `k_mul_12__partial` exceeds the 2 s bench watchdog in the
   tuned deploy (16/19 targets measured around it).
5. **Unmeasured golden rows are real bench failures, kept as inventory**: fp8 files carry 11-22 unmeasured
   realizations each (hangs, compile failures, or the coverage change); only the 0.6B BF16 files validate at
   REPOSITORY level on all three cards, the rest at WORKING level.

## Environment caveats (hosts are rented and heterogeneous)

- The rtx4090 host ran an old driver (CUDA 12.2-era) and nvcc 12.1: the default cu130 torch cannot initialize
  (fixed with the cu126 wheel + matching cu12 libraries), and a subset of fp8 kernels fail to COMPILE under
  nvcc 12.1 that compiled under CUDA 13.3 on the first pass — its fp8 numbers carry that asterisk.
- V100 requires torch cu126 and `cupy-cuda12x==13.6.0` + `fastrlock` (nvrtc 13 dropped Volta), as before.
- Pre-run canaries must check BOTH `cupy.full` AND `torch.cuda.is_available()`; a cupy-only canary passed on
  the old-driver 4090 while every torch-side measurement failed.

## Platform a100x1 — earlier routing measurements (2026-08-23; not current deploy evidence)

An earlier pass measured two positive routing realizations at deployable `-O3` on the exact
NVIDIA A100-SXM4-80GB (`GPU-b0354a1a-37c2-086d-f6fe-953b6fac5c3e`):

| seq | target | routing realization | Emmy | fused/cold Emmy | eager | result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 512 | score + softmax statistics | `PLACE@b=cut` | 299.69 µs | 21232 µs | 745.47 µs | 2.49x eager |
| 1 | post-attention norm + gate/up + SiLU | `PLACE@map=cut` | 66.97 µs | 4612 µs | 154.62 µs | 2.31x eager |

Both routing realizations passed strict Emmy-versus-eager correctness in that pass. They are not current automatic
deploy evidence: the 2026-08-28 audit showed that a `PLACE` total does not identify the exact child schedules it
measured, and the final compiler fails closed on such rows. The current qualification above supersedes this section.

At the 2026-08-23 boundary this was direct tuning evidence; no matching experiment snapshot was produced.
`results_a100x1.tar.gz` below remains the 2026-08-21 run and does not measure the two routing realizations.

## Platform a100x1 — historical chain-form replay (2026-08-21)

### Question

`main` now carries the chain root formation: a fold closes over the values its projection body defines, so the RoPE
gathers and the k-norm no longer survive as their own kernels — they become part of the score kernel's tree. That
re-keys every computed-A attention target. Which committed realizations survive the re-key, what do the re-keyed
targets cost once they are tuned again on this card, and — now that every fold is a node with a `PLACE` seam — does
any cut beat the fused form on the targets that lose?

### Identity diff

Both inventories were re-traced from `Qwen/Qwen3-0.6B@c1899de2` layer 0 and every target name was compared with the
committed golden. A surviving identity kept its committed knobs and measurements verbatim.

| seq | committed | re-traced | carried verbatim | re-keyed or unpinned | absorbed by the score kernel |
| --- | ---: | ---: | ---: | ---: | --- |
| 512 | 12 | 9 | 6 | 3 | RoPE cos gather, RoPE sin gather, k-norm + RoPE |
| 1 | 10 | 9 | 6 | 3 | q/k norm + RoPE |

The three re-keyed targets per sequence are the score/statistics kernel, softmax·V, and o_proj + residual. The last
two keep their names but had no committed schedule (their knob maps have never been recordable), so they were tuned
from scratch as well.

### Protocol

Each re-keyed target was hybrid-tuned on this card: agent proposals drawn from the card's own measured schedules plus
every `PLACE` seam the recognize rule enumerates, then `emmy tune --max-candidates 48 --patience 12 --seed 0` under a
per-target wall budget, with an isolated tuning DB, online checkpoint, and cubin cache. Every finalist was re-measured
at deployable `-O3` against the cold greedy pick, then verified in a fresh `emmy run --golden … --target … --bench
--strict` process. The recipe then replayed both committed goldens in five fresh
`emmy run --golden … --bench --strict --bench-backends eager,tcompile,emmy --warmup 10 --iters 100` processes with an
empty per-task tuning DB, online checkpoint, and cubin cache.

### Per-kernel result (median of five processes, µs)

`greedy` is the cold pick re-benched in the same process; `deployed` is the committed realization where one exists and
the greedy pick otherwise.

| seq | target | role | eager | torch.compile | greedy | deployed | vs eager | vs tcompile |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | k_mean_20f978 | input RMSNorm | 191.54 | 8.69 | 3.82 | 3.17 | 60.4x | 2.74x |
| 512 | k_linear_reduce_06a42b | v_proj | 12.97 | 13.70 | 83.03 | 17.50 | 0.74x | 0.78x |
| 512 | k_linear_1fd3d5 | q_proj + reshape to heads | 100.64 | 21.65 | 132.97 | 22.35 | 4.50x | 0.97x |
| 512 | k_linear_a09c5a | k_proj + reshape to heads | 59.86 | 14.73 | 71.21 | 17.28 | 3.46x | 0.85x |
| 512 | k_sdpa_linear_reduce_c0a378 | softmax·V, computed V | 45.78 | 46.04 | 159.57 | 159.57 | 0.29x | 0.29x |
| 512 | k_linear_sdpa_reduce_e24efe | o_proj + residual | 60.16 | 61.27 | 315.39 | 190.46 | 0.32x | 0.32x |
| 512 | k_linear_mean_reduce_dc067d | post-attn norm + gate/up + SiLU | 246.39 | 49.62 | 67.27 | 67.27 | 3.66x | 0.74x |
| 512 | k_linear_6b4b5f | down_proj + residual | 32.09 | 37.21 | 303.10 | 46.57 | 0.69x | 0.80x |
| 512 | k_sdpa_mean_reduce_29d3df | q/k norm + RoPE + scores + softmax stats | 745.29 | failed | 21286.91 | 19609.60 | 0.04x | — |
| 1 | k_mean_b8e46d | input RMSNorm | 121.85 | 2.89 | 3.05 | 2.38 | 51.2x | 1.21x |
| 1 | k_linear_reduce_7ef15d | v_proj | 7.42 | 7.21 | 12.39 | 4.74 | 1.57x | 1.52x |
| 1 | k_linear_49a16b | q_proj + reshape to heads | 35.09 | 6.79 | 21.45 | 4.84 | 7.25x | 1.40x |
| 1 | k_linear_dfb21f | k_proj + reshape to heads | 33.67 | 5.25 | 11.06 | 4.74 | 7.10x | 1.11x |
| 1 | k_sdpa_linear_reduce_d0f5c0 | softmax·V, computed V | 11.91 | 10.62 | 25.22 | 20.72 | 0.57x | 0.51x |
| 1 | k_linear_sdpa_reduce_14c8c7 | o_proj + residual | 14.06 | 12.16 | 36.86 | 24.44 | 0.58x | 0.50x |
| 1 | k_linear_mean_reduce_549927 | post-attn norm + gate/up + SiLU | 154.62 | 16.38 | 4605.95 | 393.22 | 0.39x | 0.04x |
| 1 | k_linear_2dcd0c | down_proj + residual | 9.57 | 7.64 | 35.13 | 9.46 | 1.01x | 0.81x |
| 1 | k_sdpa_mean_reduce_0a2624 | q/k norm + RoPE + scores + softmax stats | 334.55 | failed | 39.82 | 36.53 | 9.16x | — |

Every target measured in all five repeats of both rows: the two `torch.compile`-less targets are ordered last in their
golden, so their strict failure no longer costs the later targets their measurement.

Layer totals as the sum of those medians: sequence 512 is 1494.7 eager, 252.9 Inductor (eight of nine targets), 22423
untuned Emmy and 20134 deployed Emmy; sequence 1 is 722.7 eager, 68.9 Inductor (eight of nine), 4791 untuned and 501
deployed. Sequence 1 improves on the previous revision (522 → 501 µs, 1.44x eager). Sequence 512 does not: one target,
the fused score/statistics kernel, is 19610 µs of the 20134 µs total. Over the other eight targets the sequence-512
layer is 524 µs against 749 µs eager, or 1.43x.

### Why the fused score kernel costs 19.6 ms

Its tile IR is unambiguous. The kernel places `free=(head, query)` and sweeps the key axis on the store; inside that
sweep it runs the whole k cone per cell — a 128-element k-norm fold over `to_4`, then the k RoPE — so every k vector is
recomputed once per query row rather than once per key. At sequence 512 that is 512x redundant arithmetic, and it is
the whole cost: 21243.9 µs of the 21286.9 µs untuned total is that single kernel, at 94% occupancy and 34 registers.
No thread tier changes it (`WORK=t256/t512/t1024` measure 21558 / 21835 / 22126 µs; `coop-t` measures 37873 µs).

### Cut options

The recognize rule enumerates three seams on the score root — bare `PLACE` (the score dot `acc2`), `PLACE@a2` (the
q-norm fold), and `PLACE@fold.fold.a4` (the k-norm fold) — plus one on the softmax·V root. Every one was measured at
deployable `-O3` against the fused form in the same process.

| seq | target | fused | cut option | cut | verdict |
| --- | --- | ---: | --- | ---: | --- |
| 512 | k_sdpa_mean_reduce_29d3df | 21289.98 | `PLACE@fold.fold.a4=cut` (k-norm fold) | 19625.98 | 1.08x — committed |
| 512 | k_sdpa_mean_reduce_29d3df | 21289.98 | `PLACE@a2=cut` (q-norm fold) | 335166.47 | 0.06x |
| 512 | k_sdpa_mean_reduce_29d3df | 21289.98 | both seams | 335165.44 | 0.06x |
| 512 | k_sdpa_linear_reduce_c0a378 | 159.57 | `PLACE=cut` | 4466.69 | 0.04x |
| 512 | k_linear_sdpa_reduce_e24efe | 315.39 | `PLACE=cut` | 366.59 | 0.86x |
| 1 | k_sdpa_linear_reduce_d0f5c0 | 25.17 | `PLACE=cut` | 26.22 | 0.96x |
| 1 | k_linear_sdpa_reduce_14c8c7 | 36.82 | `PLACE=cut` | 37.56 | 0.98x |
| 1 | k_sdpa_mean_reduce_0a2624 | 39.58 | no legal seam on this tree | — | — |

One cut won in this historical run, and it was the k-norm fold: materializing that reduction once removed 8% of the
redundant work. It is not current automatic deploy evidence for the reason stated above. The seam that would matter
is not in the set. Cutting `a2`
promotes the key axis to a free axis of the residue, whose grid becomes 4.2M blocks and whose cost rises 16x, because
the k cone is then recomputed with no reuse at all. What the target needs is the RoPE'd k vector materialized once as
the dot's B operand; that is a binding the contraction binder still declines, not a seam the placement fork can spell.
For the softmax·V and o_proj forms the cut is a straight loss: it splits a working mma contraction into a scalar
producer plus a workspace zero-fill (`__zp524288`, 48.5 µs on its own in the o_proj cut).

### Schedules that measure but cannot be recorded

Four targets measured a deployable win that the golden's one-knob-map-per-realization format rejects. A multi-kernel
target whose kernels include a knob-free one (an elementwise epilogue such as `k_add_5`, or `k_sdpa_reduce_fe4eb9` at
sequence 1) always fails the merge: that kernel records the empty value for every schedule family while the others
record the pinned value, so `realized_tuning_knobs` sees `WORK: '' != 't512'` and returns nothing. The pin itself is
uniform and replays, so these four realizations record the exact `--ab` pin instead of the merged realized map, and
each was re-verified by replaying the committed golden in a fresh strict process.

| seq | target | greedy | recorded pin | deployed |
| --- | --- | ---: | --- | ---: |
| 512 | k_linear_sdpa_reduce_e24efe | 315.39 | `WORK=w8x2,TILE=mma_m16n8k16_f16_f32/f1x8/k8,STAGE=d2/smem` | 190.46 |
| 1 | k_linear_sdpa_reduce_14c8c7 | 36.86 | `WORK=t512,REDUCE=coop-t` | 24.44 |
| 1 | k_sdpa_linear_reduce_d0f5c0 | 25.22 | `WORK=t512,REDUCE=coop-t` | 20.72 |
| 1 | k_sdpa_mean_reduce_0a2624 | 39.82 | `WORK=t128,REDUCE=coop/r2` | 36.53 |

### Repeat variation

Every target's five paired latencies agree to within 0.6% of their median (0.57% at sequence 1, 0.52% at 512), and
every committed realization reproduces its tuning-time `-O3` measurement to within 0.5%.

### Defects this round surfaced

1. **The online-prior refit aborts the tune.** `emmy tune` raises `_catboost.CatBoostError: All features are either
   constant or ignored` from `OnlinePrior.fit`, reached through `measure_proposals`' `prior.maybe_refit()`, when the
   first measured proposal contributes a run of rows whose feature vectors are identical. It killed the whole
   invocation for four of the six re-keyed targets on a cold online checkpoint. Re-running the same command against a
   checkpoint that already carries a varied dataset succeeds, which is the workaround used here.
2. **The tune-lane bench watchdog censors an expensive target completely.** Every candidate of the 21 ms fused score
   kernel exceeds the 2 s accumulated-GPU-time budget and is marked `bench_fail`, so a full 1800 s search ranked
   nothing at all and wrote no `ranking` block. `EMMY_BENCH_RUN_TIMEOUT_S` raises the budget; at 60 s the search
   instead spent its whole 2400 s in re-lowering without completing a candidate, so this target's schedules were
   priced by direct `emmy run --ab` instead.
3. **`torch.compile` still cannot compile the RoPE-bearing attention reference** on PyTorch 2.13.0, so `--strict`
   rejects those two targets in every repeat and both rows report `failed`. This is the same Inductor limitation the
   previous run recorded.
4. **A bare `PLACE=cut` pin re-cuts every piece it produces.** Because the pin is authoritative on each freshly
   recognized tree, the resolution recurses through the fragments; on the sequence-512 score tree a single
   `emmy compile --ir tile` under that pin had not terminated after ten minutes. Named seams resolve promptly.

### Conclusion

The re-key is mostly benign: six of nine realizations per sequence carried over verbatim and reproduce their committed
numbers, and sequence 1 is faster than the previous revision (1.44x eager). Sequence 512 regressed by construction —
folding the k cone into the score kernel made it recompute that cone once per (query, key) pair, which costs 19.6 ms
against 745 µs eager and turns a 1.56x layer win into a 0.07x loss. The placement fork is real and usable: its seams
enumerate, resolve by name, and one of them is now committed evidence, but the seam that would undo this particular
regression is a contraction binding rather than a placement.

### Limitations

- Layer-0 evidence only, one model, one card; never a whole-model claim.
- Both rows report `failed` because `--strict` requires a `torch.compile` latency the two RoPE-bearing targets cannot
  produce. Every other target passed strict Emmy-vs-eager correctness in all five repeats.
- The Inductor column is missing for those two targets, so no geometric mean over the full corpus is available; the
  measured denominators are stated above.
- A target's program includes the producers its output needs, so the attention targets overlap and the layer total is
  a sum of overlapping sub-programs on both the Emmy and the eager side, not a disjoint decomposition.
- The five repeats share one deployed host and run back to back, so they capture process-level, not day-level,
  variation.

### Run and system

- Status: failed (2/2 rows, `torch.compile` reference unavailable on the two RoPE-bearing targets; every target
  measured in every repeat)
- Result timestamp: 2026-08-21T23:03:34Z; run ID: `20260821T230334Z`
- Rows: `…sl1_scommon` (row ID `551082cef77b`, 440.74 s) and `…sl512_scommon` (row ID `3a4d139974b8`, 2131.30 s)
- Git revision: `213c443a`; dirty: false
- Host: `riftvm`; Ubuntu 24.04.1 LTS; AMD EPYC 7742 64-Core Processor
- GPU: NVIDIA A100-SXM4-80GB, UUID `GPU-b0354a1a-37c2-086d-f6fe-953b6fac5c3e`; PyTorch 2.13.0+cu130

### Durable files

- Raw-results archive: `results_a100x1.tar.gz`; archived root `2026-08-21_23-03-34/`
- Members: both `*.experiment.yaml` records, both `*_artifacts.tar.gz` task archives (per-repeat verification JSON per
  target, per-repeat logs and exit statuses, package freeze, replayed working golden), and the two runner logs
- Committed goldens: `golden/qwen3-06b-s1_a100.golden.yaml`, `golden/qwen3-06b-s512_a100.golden.yaml`

## Platform sections

### v100x1 — full pipeline (rebench + retune, 4 models attempted, 27B blocked at trace)
Goldens: `qwen3-06b_v100.yaml` (18/18, REPOSITORY), `qwen3-06b-fp8_v100.yaml` (27/38), `qwen3-32b-fp8_v100.yaml`
(23/38). Archive: `results_v100x1.tar.gz`.

### rtx4090x1 — full pipeline; measurements re-run after the driver fix
Goldens: `qwen3-06b_rtx4090.yaml` (18/18, REPOSITORY), `qwen3-06b-fp8_rtx4090.yaml` (16/38),
`qwen3-32b-fp8_rtx4090.yaml` (19/38). Archive: `results_rtx4090x1.tar.gz`.

### rtx5090x1 — full pipeline on a replacement host (first instance had unstable SSH and a failing toolchain)
Goldens: `qwen3-06b_rtx5090.yaml` (18/18, REPOSITORY), `qwen3-06b-fp8_rtx5090.yaml` (27/38).
Archive: `results_rtx5090x1.tar.gz`. Large models excluded by RAM fit (30 GB host), as before.

## Limitations

Layer-0 evidence only; `-O3` numbers throughout; tcompile per-target values live in the archives (its lane
fails on some SDPA targets); s1 sums are not comparable across passes due to coverage changes; multi-kernel
targets record `knobs: {}` with per-kernel `record_knobs` in the archives.

## Platform h200x8 — earlier failed attempt (retained)

### Conclusion

The latest non-dry invocation failed before tracing, tuning, or kernel benchmarking began. It produced no latency,
correctness, or coverage measurements and supports no kernel-performance claim. The failure is retained because
dry-run validation is not a result.

### Protocol and failure

The invocation selected one `common` row on a pre-allocated host detected as eight NVIDIA H200 141GB GPUs. The task
used one GPU and targeted `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, layer 0, sequence length 512,
with search budget 12, patience 4, and seed 0.

Remote setup reached the command after staging a clean source tree. The command then failed because `make` was not
installed. Its exit trap encountered a second error because `task_dir` was unset, so the intended `artifacts.tar.gz`
was never created or retrieved. The command result records exit code 1 and the missing-result transfer error. The
runner summary is 0/1 successful tasks.

### System and provenance

- Hardware detected: NVIDIA H200 141GB x8; the task requested one GPU.
- Timestamp: `2026-08-13_16-08-20_e1c8d16a`.
- Git revision: `030b6d58182bb3da1748c4954d7d2fd0211e8d3b`; staged source was clean.
- Workload status: failed before measurement.
- The legacy command result has no assembled experiment record and no complete typed system-information section.

### Durable files

- Raw-results archive: `results.tar.gz`.
- Archived root: `2026-08-13_16-08-20_e1c8d16a/`.
- Supporting evidence: runner logs, the executed recipe snapshot, the task manifest, and the command result JSON are
  in the archive.
