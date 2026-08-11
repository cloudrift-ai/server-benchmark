# Ling 3.0 Flash FP8 compiler qualification on 8× V100 SXM2 16 GB

Status: **exact trace and SM70 CUDA lowering complete; deployable O3 qualification partial; no repository
golden**. The exact model inventory is preserved as
[`exact_traced_working.yaml`](exact_traced_working.yaml). It is a non-canonical continuation artifact, not deploy
evidence.

## Scope

- Date: 2026-08-11
- Repository revision: `09a26def732dc19df8be525e3d0a0876b5847a9e`, plus the compiler changes under review
- Model: `inclusionAI/Ling-3.0-flash-fp8`
- Immutable model revision: `a5d248fcca98b9d9a0c225cc22372f2fd1b3540b`
- Hardware: 8× `Tesla V100-SXM2-16GB`, compute capability 7.0, driver 580.159.03
- Trace shape: batch 1, sequence length 8, hidden size 2560
- Environment: Python 3.12, PyTorch `2.13.0+cu126`, Transformers `4.56.2`, FLA `0.5.2`, CUDA 12.9, and
  CuPy `13.6.0` with CUDA 12.9 NVRTC

The checkpoint contains 128,443,021,752 tensor bytes (119.62 GiB). Perfectly balanced over eight 16 GiB cards,
weights alone consume about 14.95 GiB per GPU and leave about 1.05 GiB before CUDA contexts, collectives,
workspaces, attention state, and KV cache. This fails the serving fit gate. Independently, the official Ling vLLM
fork's FP8 path requires compute capability 7.5 or newer, while this host is 7.0. No serving deployment, recipe,
image build, or publication was attempted.

## Architecture coverage

The immutable configuration defines 42 decoder layers and one MTP layer. The machine-readable mapping is in
[`coverage.json`](coverage.json).

| Architecture path | Layer indices | Traced representative |
| --- | --- | --- |
| Dense SwiGLU + KDA | 0–1 | Layer 0 |
| Sparse MoE + KDA | 2–4, 6–10, 12–16, 18–22, 24–28, 30–34, 36–40 | Layer 2 |
| Sparse MoE + MLA | 5, 11, 17, 23, 29, 35, 41 | Layer 41 |
| MTP + MLA + sparse MoE | 42 | Layer 42 |

The model inventory also covers token embedding, rotary generation, final RMSNorm, the main output head,
shifted-token embedding, the MTP transition, and the MTP output head. The representative sparse block contains one
exact routed expert and the always-on shared expert. Routing, top-k, sort, and weighted combination remain host
orchestration under Emmy's representative-MoE trace contract.

The artifact records checkpoint-compatible source paths for `model.word_embeddings.weight`,
`model.rotary_emb.inv_freq`, `model.norm.weight`, `lm_head.weight`, and layers 0, 2, 41, and 42. It spells 38
checkpoint weights as FP8 E4M3 constants from immutable safetensors metadata while preserving the checkpoint's
dynamic-activation, 128×128 weight-block layout.

## Exact frontend support and safety bounds

The exact MTP token shift exports `aten.roll -> aten.select -> aten.fill_`. The frontend now represents this path
without a model-specific replacement:

- Static one-dimension `roll` is two bounded affine `IndexMapOp` regions.
- Static rank-reducing `select` fixes one source coordinate in an `IndexMapOp`.
- `fill_` returns a functional filled value. When a later live read observes the base, a static unit-step
  slice/select view rooted at locally produced storage reassembles the base with a two-source `IndexMapOp`.
- Static integer `arange` uses the zero-input `RangeOp`; bind-time constant replay no longer calls NumPy `arange`
  elementwise over a broadcast stop tensor.

The alias gate requires the base to be a locally produced call-function value with its own storage, the mutation
return to be unused when the base must be rebound, and every observable pre-write alias to be absent. Later aliases
created from the rebound root see the new version. Multidimensional roll, dynamic shifts/dimensions/ranges,
non-integer ranges, dynamic or non-unit slices, input/parameter roots, used mutation returns, and stale aliases fail
closed. `copy_` retains its stricter pre-existing constructor-root and slice-only restrictions.

Focused tests cover positive and negative roll shifts, shift normalization, select parity, exact Ling MTP parity
through the NumPy and Loop pipelines, input/stale-alias rejection, static range replay, and non-integer range
rejection. The complete trace suite reports 98 passed and one existing XPASS.

## Exact current-head inventory and lowering

The exact whole-architecture trace completed in 8 minutes 2 seconds on the supplied host. The inventory SHA256 is
`9c1d8a53bf657b7747ef7c284c554af6acfb42b1978366b99a571061c00953de`.

| Artifact | Frontend nodes | FP8 constants | Targets | SM70 CUDA lowering |
| --- | ---: | ---: | ---: | ---: |
| Exact whole architecture on current HEAD | 1,968 | 38 | 48 | 48 / 48 |

Every exact target reconstructs and lowers on compute capability 7.0. The lowering replay took 48.75 seconds and
recorded no failures; see [`exact_lowering_summary.json`](exact_lowering_summary.json).

## Deployable O3 qualification

`emmy tune --golden-file` used all eight GPUs, seed 0, one candidate per target, the O1 ranking lane, and a fresh DB
and online prior. `--bench` then rebuilt each winning or fallback target at nvcc default O3 with five warmups and 20
iterations. Full-model comparison is unavailable because the embedded inventory has no runnable eager module;
correctness is therefore recorded per provenance reproducer when its frontend slice can build an eager reference.

| O3 result | Count |
| --- | ---: |
| Exact targets attempted | 48 |
| Targets with at least one positive timing | 47 |
| Targets without a positive timing | 1 |
| Provenance reproducers attempted | 69 |
| Reproducers with positive Emmy timing | 68 |
| Eager-reference correctness passes | 30 |
| Eager-reference correctness failures | 0 |
| Timing-only reproducers without an eager reference | 38 |

The exact timing distribution and failure records are in [`exact_o3_summary.json`](exact_o3_summary.json). A null
accuracy error is considered a pass only when `reference_available=true`; reference-free Loop slices are timing
evidence, not correctness evidence.

All 68 successful rows used CUDA-graph-captured timing. Emmy latencies range from 1.456 µs to 1.305 seconds, with a
498.18 µs median. Of the 30 eager-backed rows, Emmy is faster on 10 and slower on 20; every row passed its random-input
reference comparison. The five prior bind-time truth-value failures are gone after static `arange` became `RangeOp`.

The sole timing gap is target 36, `k_linear_mean_reduce_e0ee1d.a08b04cc6fc6`. It fuses the exact MTP
embedding/roll/fill path, two norms, the FP8 `eh_proj`, and two means. O3 first iterations took 2.58 and 2.28 seconds;
iteration 1 then exceeded the 2-second watchdog. Bounded scoped `PLACE@a=cut` and bare `PLACE=cut` evaluations both
retained the original single launch, so the exact multi-use cone has no legal cut under the captured-value safety
gate. Their O1 first iterations took 4.47/4.16 and 4.48/4.21 seconds respectively; there is no cut O3 timing.

The one-candidate ranking pass persisted 23 explicit O1 winners. The remaining 25 targets still reached deployable
O3 through their assembled fallback, but have no explicit searched winner. This is a second reason the continuation
artifact cannot become a repository golden.

## Historical diagnostic evidence

The retained [`partial_traced_working.yaml`](partial_traced_working.yaml), [`lowering_summary.json`](lowering_summary.json),
[`tuning_summary.json`](tuning_summary.json), and [`o3_summary.json`](o3_summary.json) come from the earlier
`9daede61bab735aca99b1f4afc4b0f4af905fa74` compiler revision. That diagnostic replacement inventory had 1,978
frontend nodes and 341 post-fusion targets, all of which lowered on SM70; 300 had positive O1 rankings and three
representatives were measured at O3. Compiler fusion and inventory semantics subsequently changed: a same-revision
control before the final static-range fix produced 48 targets, and the final exact current-head inventory is the
48-target artifact above. The 341-target files remain historical diagnostics and are not current coverage counts.

The library-only FLA and rotary replacements used by both traces retain their prior numerical checks. Original FLA
KDA and its replacement agreed at maximum absolute error 0.00390625 and passed FP16 `rtol=5e-3, atol=5e-3`;
original rotary and its replacement agreed bit-for-bit. The exact token shift now uses frontend support rather than
diagnostic slice/concatenate substitution.

## Promotion decision

No file was added under `emmy/compiler/pipeline/search/goldens/`. Exact trace and lowering are complete, but a
canonical golden requires every target to have a valid realization, positive O3 timing, and explicit correctness
evidence where an eager reconstruction is available. The remaining O3 and serving gaps fail that gate. The working
YAML must not be presented as complete deploy evidence.
