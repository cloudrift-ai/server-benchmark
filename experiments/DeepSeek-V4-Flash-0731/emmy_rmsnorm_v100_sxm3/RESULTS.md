# DeepSeek V4 Flash 0731 bounded Emmy RMSNorm on 16× V100

Status: qualified for the exact bounded leaf described here. The full model remains a 1Cat/vLLM deployment; Emmy
compiles and runs the final decode RMSNorm only. This result does not claim `EmmyGenModel`, attention, KV-cache, MoE,
checkpoint-loading, tensor-parallel, or pipeline-parallel coverage.

## Question and result

Can a real Emmy-generated SM70 kernel run inside the qualified DeepSeek V4 serving path on 16 V100s without changing
model output or degrading endpoint performance? Yes. The guarded adapter activated on all eight TP ranks of the last
PP stage, survived CUDA graph capture, passed the live first-use exact comparison, and produced the same deterministic
chat response and token counts as stock. Warm endpoint throughput and TPOT were statistically tied with stock.

The retained one-launch `WORK=t256, REDUCE=coop` kernel measured 4.5692 ± 0.0021 µs at deployable O3, 1.578× faster
than Emmy's 7.2084 µs greedy schedule for this leaf. The faster two-launch split was rejected because it expands the
serving ABI and showed slightly more one-ULP drift in randomized checks for only 11% additional leaf-level speed.

## Exact scope

| Item | Value |
| --- | --- |
| Date / run ID | 2026-08-17 / `20260817T094438Z` |
| Repository base | `776b234e82ac2e248b1540a3ec27c35cb4d01d5d` plus this branch's reviewed overlay |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Model revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Hardware | 16× Tesla V100-SXM3-32GB, device IDs 0–15, compute capability 7.0 |
| Host software | Ubuntu 24.04.1, driver 580.173.02 |
| Serving image | local image ID `sha256:8e81b26d6c42427826e71ac27079349a5d79c051a0e2a30dfe707c9995b94582` |
| Runtime base | 1Cat/vLLM `d76126608155c334df7c2fb9b75096f879624859`, CUDA 12.9 |
| Serving layout | TP8×PP2, context 1,048,576, concurrency 8, FP8 KV cache |
| Emmy boundary | final PP-stage RMSNorm, decode shape `[1,4096]`, FP16 input/weight/output, epsilon `1e-6` |
| Tuning environment | PyTorch 2.13.0+cu126, CuPy 14.1.1, CUDA 12.9 nvcc, target `sm_70` |

The model continues to use packed FP4 experts, block-FP8 attention weights, FP8 KV cache, and 1Cat's SM70 kernels.
Quantized formats never enter the Emmy leaf. Unsupported shapes, dtypes, residual fusion, cold CUDA-graph capture,
build failures, and a failed first-use comparison route to the original runtime implementation.

## Kernel search and decision

The self-contained working golden contains one static target. Both primary arms started from SHA-256
`b9af896875e1f4de7c8e34173f4504558a40d8a86cd9240d2a6f7c0bea05021d`; the hybrid file added four proposals and
started as `67a1d9ef8da6356d8701b2b8c9bbbf1733825d06927f86051e5cb5667dacad07`. Each arm used separate empty DB, online
prior, and cubin directories, all 16 homogeneous devices, seed 731, patience 6, and a six-candidate target budget in
the O1 ranking lane. MCTS-only ran first in 20.6 s; hybrid followed in 28.9 s. Hybrid reserved four of its six slots
for proposals. The logs report 10 and 13 prior trajectory observations respectively; these include nested kernel and
warmup observations rather than extra target-candidate budget.

| Candidate | Source | O1 ranking evidence (µs) | Deployable O3 evidence | Decision |
| --- | --- | ---: | ---: | --- |
| `WORK=t32, REDUCE=coop` | Hybrid proposal | 73.964 | Not shortlisted | Reject |
| `WORK=t64, REDUCE=coop` | Hybrid proposal | 35.257 | Not shortlisted | Reject |
| `WORK=t128, REDUCE=coop` | Hybrid proposal / greedy | 19.436 | 7.2084 µs mean | Control |
| `WORK=t256, REDUCE=coop` | Hybrid proposal | 9.943 | 4.5692 ± 0.0021 µs | Deploy |
| `PLACE=cut, TILE=f4` | MCTS structural candidate | Best trajectory reached 4.508 | 4.0498 µs, two launches | Reject |

The primary-arm logs do not persist one complete searched-finalist row when the search ends after its single full
terminal; the per-kernel DB does preserve the structural split. Replaying the exact `PLACE=cut,TILE=f4` pins made the
candidate explicit and passed strict O3 integrity. This CLI feedback gap is a workflow limitation, not evidence that
the split should be deployed.

Three fresh O3 runs used 50 warmups and 1,000 timed iterations. The retained schedule measured 4.5664, 4.5701, and
4.5713 µs; the pin realized exactly, CUDA timing was captured, and the direct eager comparison passed every time. The
corresponding isolated greedy measurements were 7.2139, 7.1986, and 7.2128 µs. The working inventory remains an
experiment artifact because it covers one bounded serving leaf, not the checkpoint's required serving-twin matrix;
no canonical full-model golden was promoted.

Randomized numerical checks used 100 seeds and vLLM's declared FP16 RMSNorm tolerance (`atol=1e-2`, `rtol=2e-3`).
Both candidates passed all 100 cases. Relative to eager, the one-launch winner differed in 3 cases and 10 elements;
the split differed in 5 cases and 18 elements. Both maximum absolute differences were `0.0009765625`. The deployed
adapter additionally requires the first real request to compare exactly with the original 1Cat result before it
latches active.

## Serving A/B

The direct comparison used separate fresh stock and Emmy deployments on the same host. Each lane ran one cold repeat
followed by three reported steady repeats, each with eight unique 1,024-token prompts, concurrency 8, 64 requested
output tokens, greedy decoding, ignored EOS, and seed 731. Every reported request returned exactly 64 output tokens.
Spread is population standard deviation across the three steady repeats.

| Metric | Stock 1Cat | Bounded Emmy | Difference |
| --- | ---: | ---: | ---: |
| Successful / failed requests | 24 / 0 | 24 / 0 | tie |
| Request throughput (requests/s) | 0.534127 ± 0.003913 | 0.534351 ± 0.002167 | +0.04% |
| Output throughput (tokens/s) | 34.1841 ± 0.2505 | 34.1984 ± 0.1387 | +0.04% |
| Mean TTFT (ms) | 2,488.00 ± 49.48 | 2,483.12 ± 43.43 | -4.89 ms |
| Mean TPOT / ITL (ms) | 197.414 ± 1.514 | 197.376 ± 0.666 | -0.04 ms |

The differences are much smaller than repeat variation, so the endpoint result is a tie rather than a speedup claim.
The deterministic chat probe returned identical content and usage. Logs show the Emmy activation message on each of
`Worker_PP1_TP0` through `Worker_PP1_TP7`; no attention, cache, router, expert, TP, or PP operation was intercepted.

The clean recipe-driven reproduction succeeded with 32/32 requests across four client repeats. Its first repeat was
cold (25.94 output tokens/s); repeats two through four averaged 34.013 ± 0.589 tokens/s, 2,816.73 ± 21.19 ms TTFT,
and 193.743 ± 3.137 ms TPOT. The final two repeats measured 34.42 and 34.44 output tokens/s. This run independently
verified model download, load, smoke response, eight-rank Emmy activation, CUDA graph capture, and client completion.

## Compiler issue fixed during onboarding

The DeepSeek layer-0 Loop graph exposed unbounded dependency growth before the existing normalized-work guard could
run. `LoopBuilder` now uses a construction-only mutable tree plus monotone fresh-name cursors, and fusion applies the
existing eight-times structural growth factor while dependencies are emitted. The exact graph that remained at 99%
CPU after more than 22 minutes 59 seconds now completes in 3.825650 seconds (3.848140-second repeat), a censored
improvement greater than 360×. The emitted artifact is stable across repeats. This is a model-agnostic boundedness fix;
it does not add a DeepSeek- or V100-specific compiler branch.

## Conclusion and limitations

This experiment proves real Emmy compiler execution inside the production-shaped DeepSeek V4 decode path on
16 V100s, with a tuned leaf and no measurable endpoint regression. It deliberately retains 1Cat for the hard parts:
checkpoint conversion, TP8×PP2, HCA/CSA state, sparse attention, FP8 cache traffic, FP4 experts, routing, and mHC.
Calling this full Emmy model support would violate the serving-twin and coverage invariants.

The final RMSNorm runs once per decode row only on the last PP stage, so even a 1.578× leaf improvement is too small
to move whole-model throughput visibly. The standard run also observed first-shape 1Cat Triton JIT warnings during
the cold repeat; steady repeats were retained separately. The local derived image was not pushed: registry
publication requires a separate human approval and a stable target repository.

Workflow friction: `make setup` initially selected CUDA 13 PyTorch, which no longer contains Volta kernels; the run
needed a CUDA 12.6 PyTorch reinstall and removal of stale CUDA 13 shared libraries. The tuning CLI should reject a
live GPU absent from `torch.cuda.get_arch_list()` before starting, and working-golden tuning should always persist the
complete directly searched finalist even when only one terminal is reached.

## Reproduce and evidence

Build the local image from the repository root, then run the experiment:

```bash
make wheel
docker build -f docker/1cat-vllm-sm70/Dockerfile.emmy \
  --build-arg BASE_IMAGE=cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608 \
  -t emmy/1cat-vllm-deepseek-v4-flash-0731:local .
emmy bench experiments/DeepSeek-V4-Flash-0731/emmy_rmsnorm_v100_sxm3 \
  --ssh riftuser@185.165.50.61
```

The durable archive is `results_v100x16.tar.gz`, rooted at `2026-08-17_09-44-38/`. Its terminal system record is
`v100x16_661253606d45.experiment.yaml`; serving logs and client output are beside it. `manual_ab/{stock,emmy}/`
contains the matched direct comparison, while `kernel_tuning/` contains both primary-arm logs and DB/prior snapshots,
the working goldens, strict O3 JSON, and randomized numerical output. The record status is `succeeded`, with no failed
row, execution error, or cleanup error. The supplied host remains running; all task serving containers were stopped.
