# V100 Qwen3.5-122B-A10B golden-sweep findings

## Outcome

Phase 1 produced 44 provisional FP16 projection configs for the V100 SM70 target. They cover 11 logical shape
families at static `M={32,128,512}` plus a `dynM` entry with hint 512. Forty candidate schedules came from live
single-shape tuning on the 16×V100 host. The four vocabulary-projection schedules used explicit manual transfer
because every tuner candidate exceeded the default worker budget.

These are the best reviewed V100 configs available now, not a claim of cuBLAS parity. The current Volta warp-MMA
implementation is global-memory direct and several shapes remain much slower than eager PyTorch. The YAML therefore
acts as provisional deploy evidence for a newly enabled target. The model-serving engine uses its own SM70 kernels;
this sweep evaluates Emmy's isolated projection programs, not end-to-end Qwen serving performance.

The clean O3 review selected 38 candidate schedules and retained the conservative bootstrap for six shapes. No row
reached the ordinary 0.95 measured-golden threshold: eager/Emmy ranged from 0.025× to 0.587× with a 0.122× median.
The below-threshold timing pairs are retained deliberately because the user requested the best available configs for
this newly supported card, and the file labels them as provisional V100 entries.

## Scope and provenance

- Host: `riftuser@185.165.50.65`, 16× Tesla V100-SXM3-32GB, driver 580.159.03.
- Topology: every GPU pair reports `PHB`; CUDA peer read/write and NVLink are unavailable. This does not affect the
  single-GPU schedule search, but it disqualifies the host from Phase 2 serving numbers.
- Compiler target: `sm_70`, standard FP16 inputs with FP32 accumulation.
- Qualified image ID: `sha256:7a62dcfeb1622665035eab501ffe0aea159662cb63872730c0c1ea0d61f43b2e`.
- Candidate snapshot checksum: `d6b60cd148b0411bb13e6d2f7f8ebe0fadbbcf36a76e27dace51fd6764b0579b`.
- Reviewed selection checksum: `ebfe22dcbbbab178fe2e0d0bb4a9728e2d7252655d4a73a17420de20554dfed2`.
- Inventory status: architecture-derived, not an observed Emmy model trace. The current model runner cannot emit
  Qwen3.5's linear-attention and gated shared-expert graph. The YAML intentionally has no `model:` audit tag.

The set excludes `M=1`, MTP, and vision. Static expert rows are representative prefill policies rather than observed
per-expert routing counts; exact routed `M_e` values still require a serving trace on a later integration.

## Search workflow

The normal whole-dataset `emmy tune --dataset golden --gpus 16` path was attempted first. Its shape loop was serial,
so it completed only two shapes before the bounded attempt was stopped. The documented fallback split the 44 exact
golden names over 16 isolated V100 workers, each with its own tuning DB, online checkpoint, cubin cache, and GPU lock.
That run completed in 2,219.8 seconds.

The sharded O1 search yielded usable best candidates for 40 shapes. All four `N=248320, K=3072` vocabulary rows hit
the tuner worker budget. Their schedules were selected manually from legal, measured same-geometry donors. For
`dynM`, an initial split-reduction donor measured about 105.6 ms; the unsplit static-`M=512` donor measured about
100.3 ms and was retained for the final review.

The final O3 review used one immutable image ID, all 16 GPUs, explicit warmup/iteration counts, and three independent
repetitions for both the conservative bootstrap and candidate lane. Every repetition wrote the machine-readable
`emmy run --json` record. The harness rejects a row unless both lanes have three clean records with:

- command return code zero;
- canonical V100 identity and the exact benchmark protocol;
- standard precision lane, `status == ok`, and no integrity flags;
- one realized kernel whose complete schedule families match the requested config; and
- positive isolated Emmy and eager timings.

The selector keeps the lower three-repeat isolated median for each shape. Requested knobs are never copied directly;
the YAML is written from the verified `record_knobs` in the winning lane.

## Prior behavior

The tune logs expose the best candidate's measurement position, but they do not persist the prior's first complete
schedule. For the first shape on each shard, the offline cold-start prior found the best O1 row at median measurement
position 56.5, or the 58.6th percentile of the measured pool. Only 3/16 winners landed in the top quartile. For the
24 later shapes searched after an online update, the median best position was 54, or the 49.2nd percentile; 4/24
landed in the top quartile. Across all 40 successful searches, none found its best row in the first 10 measurements
and only six did so in the first 25.

The result is consistent with a cold target whose legal schedule vocabulary differs substantially from the trained
SM80+ corpus. Exact per-knob misses of the first prior choice cannot be reconstructed from the current logs. Comparing
the winning configs to the conservative seed still shows where additional prior features are most valuable: register
tile depth and extent first, work decomposition second, and reduction splitting for selected large contractions.
Among the 38 candidate winners, all changed `TILE`, 18 changed `WORK`, 18 changed `REDUCE`, two changed `RASTER`, and
none used `STAGE`; that last result reflects the current global-memory-direct Volta implementation rather than proof
that operand staging would not help.

## Main findings

1. Volta MMA works and produces deployable schedules, but global-memory-direct operand movement is the dominant
   performance limitation. The final results support enablement, not an optimization-parity claim.
2. The tuner can search SM70 successfully for ordinary Qwen projection sizes. The vocabulary projection needs a
   configurable accumulated GPU-time budget or a dedicated large-shape lane; treating every timeout as an invalid
   schedule hides useful candidates.
3. Fixed-host command workloads share one staged checkout. Concurrent Emmy benchmark invocations can replace that
   tree while an older container still imports modules. Run the sweep recipes serially or stage immutable per-run
   source directories.
4. The current whole-dataset tuner does not distribute exact golden names across `--gpus`; the explicit sharding
   fallback is materially faster and should become the native dataset behavior.
5. O1 is useful for candidate discovery but is not a recording lane. Several O1 winners regressed under O3, so the
   three-repeat isolated O3 comparison remains necessary before updating YAML.

## Recommendations

1. Add a legal synchronous shared-memory Volta transport and tune its swizzle/load layout before expanding the
   schedule space further. This is the highest-leverage path toward cuBLAS parity.
2. Make golden-dataset tuning shard exact names over GPUs, and expose compile/run wall budgets through the supported
   tuning interface rather than a harness-local backend override.
3. Have tuning emit the same integrity JSON used by `emmy run --json`, including realized schedule families, prior
   position, failure classification, and immutable GPU/image provenance.
4. Collect runtime expert-token histograms and an observed projection trace once Emmy can emit the Qwen3.5 graph.
   Replace representative expert `M` policies only with measured serving shapes.
5. On the Phase 2 NVLink host, recheck a representative subset of single-GPU schedules for GPU identity and drift,
   but do not redo the full sweep unless the V100 SKU or compiler revision changes.

## Per-shape O3 decisions

The table below is generated from the final clean A/B records. `eager/Emmy` below 1.0 means Emmy is slower; these
values are diagnostic kernel evidence and are not Phase 2 serving numbers.

| golden | O1 µs | bootstrap O3 µs | candidate O3 µs | decision | eager/Emmy |
| --- | ---: | ---: | ---: | --- | ---: |
| `qwen35_122b.attn_gdn_out.lin.dynM` | 4901.888 | 13954.048 | 5706.752 | candidate | 0.06× |
| `qwen35_122b.attn_gdn_out.m128.lin` | 1727.488 | 3484.672 | 1110.016 | candidate | 0.13× |
| `qwen35_122b.attn_gdn_out.m32.lin` | 1128.448 | 984.064 | 1162.240 | bootstrap | 0.09× |
| `qwen35_122b.attn_gdn_out.m512.lin` | 4886.528 | 13930.496 | 4080.640 | candidate | 0.08× |
| `qwen35_122b.attn_kv.lin.dynM` | 668.672 | 990.208 | 720.896 | candidate | 0.07× |
| `qwen35_122b.attn_kv.m128.lin` | 213.811 | 306.176 | 214.835 | candidate | 0.10× |
| `qwen35_122b.attn_kv.m32.lin` | 81.323 | 96.870 | 78.297 | candidate | 0.16× |
| `qwen35_122b.attn_kv.m512.lin` | 737.280 | 995.328 | 604.672 | candidate | 0.08× |
| `qwen35_122b.attn_q.lin.dynM` | 9070.592 | 27436.031 | 7615.488 | candidate | 0.09× |
| `qwen35_122b.attn_q.m128.lin` | 2526.208 | 7411.712 | 1938.432 | candidate | 0.20× |
| `qwen35_122b.attn_q.m32.lin` | 1575.936 | 1993.728 | 950.272 | candidate | 0.21× |
| `qwen35_122b.attn_q.m512.lin` | 9850.880 | 27508.736 | 8181.760 | candidate | 0.08× |
| `qwen35_122b.down.lin.dynM` | 660.224 | 2112.512 | 503.296 | candidate | 0.15× |
| `qwen35_122b.down.m128.lin` | 313.003 | 483.328 | 234.752 | candidate | 0.14× |
| `qwen35_122b.down.m32.lin` | 192.102 | 125.952 | 157.867 | bootstrap | 0.15× |
| `qwen35_122b.down.m512.lin` | 647.680 | 2086.912 | 500.736 | candidate | 0.15× |
| `qwen35_122b.gate_up.lin.dynM` | 1231.211 | 3536.896 | 1141.760 | candidate | 0.12× |
| `qwen35_122b.gate_up.m128.lin` | 460.440 | 1009.664 | 365.909 | candidate | 0.17× |
| `qwen35_122b.gate_up.m32.lin` | 191.470 | 312.320 | 180.053 | candidate | 0.23× |
| `qwen35_122b.gate_up.m512.lin` | 1165.609 | 3816.448 | 999.424 | candidate | 0.14× |
| `qwen35_122b.gdn_ba.lin.dynM` | 339.374 | 159.232 | 294.571 | bootstrap | 0.09× |
| `qwen35_122b.gdn_ba.m128.lin` | 67.038 | 96.768 | 38.321 | candidate | 0.24× |
| `qwen35_122b.gdn_ba.m32.lin` | 21.243 | 97.075 | 12.480 | candidate | 0.59× |
| `qwen35_122b.gdn_ba.m512.lin` | 98.054 | 160.427 | 96.461 | candidate | 0.14× |
| `qwen35_122b.gdn_qkv.lin.dynM` | 6349.440 | 20667.393 | 6154.240 | candidate | 0.08× |
| `qwen35_122b.gdn_qkv.m128.lin` | 1773.475 | 5672.960 | 2586.624 | candidate | 0.12× |
| `qwen35_122b.gdn_qkv.m32.lin` | 863.232 | 1528.832 | 777.216 | candidate | 0.16× |
| `qwen35_122b.gdn_qkv.m512.lin` | 6366.891 | 20695.040 | 6203.392 | candidate | 0.08× |
| `qwen35_122b.gdn_z.lin.dynM` | 5768.989 | 13857.792 | 4360.192 | candidate | 0.08× |
| `qwen35_122b.gdn_z.m128.lin` | 1387.520 | 3614.720 | 2098.176 | candidate | 0.11× |
| `qwen35_122b.gdn_z.m32.lin` | 863.232 | 1006.592 | 813.056 | candidate | 0.15× |
| `qwen35_122b.gdn_z.m512.lin` | 4803.145 | 13952.000 | 47396.866 | bootstrap | 0.03× |
| `qwen35_122b.lm_head.lin.dynM` | fallback | 416461.823 | 100302.849 | candidate | 0.08× |
| `qwen35_122b.lm_head.m128.lin` | fallback | 103092.224 | 26258.432 | candidate | 0.19× |
| `qwen35_122b.lm_head.m32.lin` | fallback | 26003.456 | 10541.056 | candidate | 0.22× |
| `qwen35_122b.lm_head.m512.lin` | fallback | 411325.439 | 100307.968 | candidate | 0.08× |
| `qwen35_122b.router.lin.dynM` | 305.897 | 529.920 | 669.696 | bootstrap | 0.08× |
| `qwen35_122b.router.m128.lin` | 108.362 | 160.256 | 144.677 | candidate | 0.12× |
| `qwen35_122b.router.m32.lin` | 54.784 | 96.870 | 46.127 | candidate | 0.20× |
| `qwen35_122b.router.m512.lin` | 307.541 | 531.968 | 531.968 | candidate | 0.07× |
| `qwen35_122b.shared_gate_or_up.lin.dynM` | 618.066 | 1982.464 | 624.128 | candidate | 0.12× |
| `qwen35_122b.shared_gate_or_up.m128.lin` | 906.493 | 539.136 | 385.707 | candidate | 0.08× |
| `qwen35_122b.shared_gate_or_up.m32.lin` | 297.984 | 164.352 | 202.547 | bootstrap | 0.12× |
| `qwen35_122b.shared_gate_or_up.m512.lin` | 606.336 | 1990.656 | 746.496 | candidate | 0.11× |

## Evidence

- O1 shard run: `experiments/volta-sm70-golden-shards/2026-08-07_21-37-11_4487904d`.
- Candidate snapshot: `experiments/volta-sm70-golden-o3/candidates.json`.
- Reviewed selection snapshot: `experiments/volta-sm70-golden-o3/selections.json`.
- Final O3 run: `experiments/volta-sm70-golden-o3/2026-08-07_23-41-20_4487904d`.
- Full environment, topology, model-stack, and serving evidence: `_tune/volta-qwen35/`.

No Phase 1 serving latency belongs in the article or final benchmark manifest. Phase 2 remains blocked only on the
replacement 16×V100 host with working NVLink.
