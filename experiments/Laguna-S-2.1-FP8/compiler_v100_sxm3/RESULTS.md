# Laguna-S-2.1-FP8 compiler qualification on 8× V100 SXM3 32 GB

Status: complete model-level compiler inventory and canonical V100 SM70 golden. The compiler result is
architecture-derived rather than checkpoint-accuracy or serving evidence. The independent 1Cat serving qualification
is documented in
[`recipes/Laguna-S-2.1-FP8/RESULTS.md`](../../../recipes/Laguna-S-2.1-FP8/RESULTS.md).

## Exact scope

- Date: 2026-08-09, with placement-routing continuation on 2026-08-11
- Model: `poolside/Laguna-S-2.1-FP8`
- Immutable model revision: `9e0b8ba630080b0e6f20a7b43294a9f2232fd247`
- Emmy base revision: `4438c84a2027b87091fefd43f5cbbd5ea2bb4a5f`, plus this PR's trace and lowering fixes
- Hardware: 8× `Tesla V100-SXM3-32GB`, compute capability 7.0, driver 580.159.03
- Trace profile: decoder, embedding, and final-normalization sequence length 512; single-token output head; FP16
  architecture twins; target `sm_70`

Architecture-only tracing replaces each sparse MoE block with representative expert compute. Router sorting,
dispatch, combination, and the serving engine's FP8 storage decoder remain host/runtime orchestration. The inventory
therefore covers every compiler-visible model path and layout, but does not claim to reproduce the checkpoint's
256-expert routing or preserve its on-disk FP8 representation in Emmy serving.

## Complete model coverage

The exact checkpoint has five layer/checkpoint-layout classes and three distinct stable Torch programs:

| Layers | Attention | MLP | Routed expert storage | Representative |
| --- | --- | --- | --- | ---: |
| 0 | full | dense | BF16 dense MLP | 0 |
| 1-43 except multiples of four | sliding | sparse | E4M3 FP8 plus FP32 scales | 1 |
| 4, 8, ..., 40 | full | sparse | E4M3 FP8 plus FP32 scales | 4 |
| 44 | full | sparse | BF16 routed experts | 44 |
| 45-47 | sliding | sparse | BF16 routed experts | 45 |

The final four layers are sparse, not dense. Separate layer-44 and layer-45 traces have exactly the same target sets
as layers 4 and 1, respectively. Across the five decoder representatives, 91 emitted targets deduplicate by stable
target identity to 33 unique targets. Independent token-embedding, final-normalization, and output-head traces add
three targets, making the complete model inventory 36.

[coverage.json](coverage.json) maps every layer 0-47 and all three non-layer seams to its representative and target
list.

## Canonical golden and validation

- [v100_sm70_laguna_s_2_1_fp8.yaml](../../../emmy/compiler/pipeline/search/goldens/v100_sm70_laguna_s_2_1_fp8.yaml)
  is the canonical repository golden. It contains 36 self-contained single-target Loop IR records and 41 measured
  realizations: one schedule realization for every target plus five placement-routing realizations. Every target has
  paired positive deployable O3/reference measurements; the file has no O1 `ranking` metadata.
- Every target reconstructed and lowered on the live SM70 compiler. The repeated O3 promotion run compiled and
  executed all 36 candidates on the requested V100 system and checked their output against their measured reference.
- Two targets used Torch eager as the independent accuracy and timing reference. The 34 targets whose self-contained
  Loop IR has no independently callable Torch geometry used a separately compiled Emmy greedy configuration on
  identical deterministic inputs; these rows prove compiler-configuration parity, not independent framework
  correctness.

The ordinary origin-based trace format exposed a durability defect: unique origin sets can reconstruct a larger
multi-target cone, and three fused rotary/attention targets did not resolve at all after reload. The canonical file
therefore uses the exact single-target Loop IR for all 36 rows. Each row lowered to exactly one CUDA kernel under its
stored schedule knobs. The five placement-routing realizations intentionally split a computed operand into a producer
and a residue kernel; the full current trace program remains embedded for provenance.

## Bounded continuation tuning

The original layer-1 inventory had 18 unique kernel names; complete coverage added 15. Only those 15 new targets
were tuned, with seed 0, six candidates per target, patience 4, all eight GPUs, and `-Xcicc -O1` ranking compiles.

The sweep completed 15/15 targets and recorded 111 prior benches, 58 successful and 11 `bench_fail` distinct perf
rows, and post-warmup Spearman +0.77. Thirteen targets received a winner. The two large reductions without a winner
were `k_linear_mean_reduce_4425bf.0b4325f6d5d4` and `k_linear_reduce_093b9a.731834dc09a4`; their candidates crossed
the two-second GPU-run watchdog.

The final exact-target promotion lane submitted one explicit configuration for every row and compared it with a fresh
greedy compile. Twenty-three submitted configurations were retained; thirteen regressed at O3 and fell back to the
measured greedy configuration. Thus every canonical row is deployable O3 evidence; the O1 values remain search
rankings only.

### Placement-routing continuation

The five largest computed-operand linear reductions were still using one fused scalar kernel. SM70 tensor-core
lowering requires materialized matrix operands, so the existing placement-routing move `PLACE@a=cut` was measured as
an alternative: it materializes the computed operand and lets the residue use the Volta `mma_m8n8k4` atom. Each
candidate was compiled and measured twice at deployable O3. Every routed measurement comprises two positive kernel
latencies and has no correctness or benchmark flag.

| Target | Original O3 (µs), repeat 1 / 2 | Routed O3 (µs), repeat 1 / 2 | Speedup, repeat 1 / 2 |
| --- | ---: | ---: | ---: |
| `k_linear_mean_reduce_01ee55` | 59,516.930 / 59,510.784 | 13,846.994 / 13,840.384 | 4.298× / 4.300× |
| `k_linear_mean_reduce_4425bf` | 358,027.519 / 357,429.260 | 82,245.123 / 82,162.233 | 4.353× / 4.350× |
| `k_linear_mean_reduce_6f21a4` | 164,318.207 / 164,395.004 | 15,296.073 / 15,320.064 | 10.743× / 10.731× |
| `k_linear_mean_reduce_a2b51c` | 119,848.961 / 119,735.298 | 27,659.584 / 27,665.664 | 4.333× / 4.328× |
| `k_linear_reduce_093b9a` | 55,886.848 / 55,863.297 | 18,286.406 / 18,325.784 | 3.056× / 3.048× |

The first `k_linear_mean_reduce_4425bf` original value is the existing canonical O3 measurement. Its fresh 50-iteration
repeat exceeded the aggregate ten-second benchmark-stage limit, while the fresh 20-iteration repeat and both routed
measurements completed. The canonical file retains both the original schedule rows and the faster routing rows so the
live-card deploy policy can choose from measured evidence.

The replay path was corrected in two places before the final deploy check: scoped programmatic pins now reach the
`EMMY_KNOBS` aggregate consumed by placement routing, and an embedded golden Loop target reruns the structural stamp
instead of being treated as a stage-complete direct IR file. With no explicit placement pin, all five exact golden
targets selected the recorded two-kernel route and completed O3 benchmarking at 13,855.665, 82,242.389, 16,417.327,
27,667.798, and 18,311.819 µs in the table's order. The stat-free activation target uses the same fused shape-key
convention on the persisted and live sides; this avoids a plain-key match against its own materialized producer.

## Accuracy limit

Exact full-checkpoint layer parity did not reach GPU execution. Host materialization reached 653,208,220 KiB RSS
(93.1% of host RAM), leaving about 39 GiB and no swap, so it was stopped before host OOM. The canonical golden is
valid architecture, lowering, per-target correctness, and tuning evidence; it does not imply checkpoint or
whole-model Emmy accuracy. The successful 1Cat serving checks provide the separate end-to-end model accuracy result.

## Artifact hashes

- `v100_sm70_laguna_s_2_1_fp8.yaml`:
  `37373fe93f9fe10a284095682034080e653996b6c11c7245d07ea6640066c719`
- `coverage.json`: `9a373a1a39536164555a65ab69d1a6c92623b90324a34e4d5b299bedd91518e3`

## Sources

- [Exact model card](https://huggingface.co/poolside/Laguna-S-2.1-FP8)
- [Exact configuration](https://huggingface.co/poolside/Laguna-S-2.1-FP8/blob/9e0b8ba630080b0e6f20a7b43294a9f2232fd247/config.json)
- [OpenMDW-1.1 license](https://huggingface.co/poolside/Laguna-S-2.1-FP8/blob/9e0b8ba630080b0e6f20a7b43294a9f2232fd247/LICENSE.md)
