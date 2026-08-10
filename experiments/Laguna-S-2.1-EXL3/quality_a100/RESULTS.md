# Laguna S 2.1 EXL3 1.98 bpw accuracy on A100

Status: complete. The checkpoint is structurally valid and all evaluated values are finite, but
the 1.98-bpw allocation has a material quality loss relative to BF16.

## Exact scope

- Date: 2026-08-10
- Source: `poolside/Laguna-S-2.1@00af5a51782109b587a3b3bbf11875e566036fa7`
- Checkpoint: EXL3 v1.4.1, configured `bits=1.98`, head bits 6
- Quantizer: ExLlamaV3 v1.4.1 at `4f8ad0121f483ba66a5336244a4c3b6d7210385e`
- Hardware: NVIDIA A100-SXM4-80GB, driver 580.159.03, CUDA 13.0
- Software: Torch 2.10.0+cu128, ExLlamaV3 1.4.1+cu128.torch2.10.0,
  datasets 5.0.1
- Data: saved WikiText-2 sample, 10 rows x 2,048 tokens, stride 2,048
- Scored positions: 20,470 for perplexity; 20,480 for KL and exact top-1 agreement
- Runtime: 524 seconds end to end; the persistent reference cache was reused

The saved input IDs have SHA-256
`c67479a6778a04b69ed1aa066a168ba9fd3ef6d2f2164c1984c17a8d7a4e2551`.

## Results

| Metric | BF16 | BF16 self-noise floor | EXL3 1.98 bpw |
| --- | ---: | ---: | ---: |
| Perplexity | 48.1599458559 | 48.5762423100 | 67.0894867506 |
| PPL change from BF16 | - | +0.8644% | +39.3056% |
| Mean `KL(BF16 || candidate)` | - | 0.1822370738 | 0.5950700045 |
| Median `KL(BF16 || candidate)` | - | 0.0625737682 | 0.3187871575 |
| p90 `KL(BF16 || candidate)` | - | 0.4261093438 | 1.4466252327 |
| Mean reverse KL | - | 0.1839387417 | 0.5823022127 |
| Exact top-1 agreement | - | 16,817 / 20,480 (82.1143%) | 14,087 / 20,480 (68.7842%) |
| Non-finite positions | 0 | 0 | 0 |

qbench reports an aggregate layer-storage rate of 2.0061055771 bpw, a head-storage rate of
6.0053678759 bpw, and a model-memory estimate of 27.527048 GiB. The memory estimate is not an
RTX 5090 fit measurement. The checkpoint's explicit EXL3 allocation field is `bits=1.98`.

## Interpretation

The output is numerically stable, but this quantization does not preserve BF16 quality closely.
The +39.31% perplexity increase, KL well above the BF16 noise floor, and 68.78% exact top-1
agreement must remain prominent in any model card. No BF16 downstream coding score is claimed
for this quantization.

## Evidence

- Original A100 `qbench-results.json` SHA-256:
  `f47a041a204d1fb22de8188e586c29c35f69a4b9a9e3e7e44c0990dbb6f911fc`
- Repository [qbench-results.json](qbench-results.json), normalized with a final newline, SHA-256:
  `0428e491c79f50d912e22302cd11063472d3072ac0146c1257c6a5f301c4fc38`
- Input/cache manifest SHA-256:
  `1b99b79e37ee204889e2fcbea70711dad72d850607a99d8da51aef4df67f5947`
- Version manifest SHA-256:
  `8e2893a4eb1ca98d22f972ba6913eaafff06da0dd30477768445b9369687c4ce`
- Output verification SHA-256:
  `aa59bb7ee66d68da120df40090be634b983c21d63a75e168801dd7acb47b7555`
- Checkpoint index SHA-256:
  `dfc19be2e337430fc0e0c012e71a2b178d281e5d0c5d54e5ec11a235e69eef5c`
- Complete isolated run:
  `/home/riftuser/laguna/accuracy-runs/20260810T043432Z-7412`

The run exited zero, all four phases passed, stderr was empty, all 47 Laguna router correction
biases were present, and the pinned ExLlamaV3 checkout remained clean.
