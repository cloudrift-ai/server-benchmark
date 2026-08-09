# Ling Flash FP8 qualification on 8× V100 SXM2 16 GB

Status: blocked before a representative compiler inventory or correct SM70 serving path. Ling 3.0 fails the runtime
memory-fit gate; the requested Ling 2.6 fallback fails both its exact serving and trace gates. No recipe or golden is
published.

## Scope

- Date: 2026-08-09
- Hardware: 8× `Tesla V100-SXM2-16GB`, compute capability 7.0, driver 580.159.03
- Primary: `inclusionAI/Ling-3.0-flash-fp8` at `a5d248fcca98b9d9a0c225cc22372f2fd1b3540b`
- Fallback: `inclusionAI/Ling-2.6-flash-fp8` at `8bc416b60fe28be33303d57bb77dd826445a1eb1`
- Emmy base: `4438c84a2027b87091fefd43f5cbbd5ea2bb4a5f`, plus this PR's guarded custom-code loader retry

## Ling 3.0 fit gate

The exact Ling 3.0 checkpoint contains 128,443,021,752 tensor bytes (119.62 GiB). Perfectly balanced over eight 16
GiB cards, weights alone use 14.95 GiB per GPU and leave about 1.05 GiB for CUDA contexts, collectives, attention
state, workspaces, and KV cache. That is not a viable serving allocation. The pinned 1Cat build also has no
`BailingMoeV3ForCausalLM` registry entry, so the primary model was rejected without a wasteful full download.

## Ling 2.6 topology and serving gates

The public fallback's 26 serving shards contain 105,627,730,504 bytes (98.37 GiB). Its separate 3.10 GiB MTP file
was excluded from the no-speculation qualification. The 32-layer checkpoint is divisible as TP4×PP2; TP8 is invalid
in the pinned 1Cat implementation because tensor parallelism may not exceed `group_norm_size=4`.

The live stock qualification used the official vLLM v0.26.0 image at digest
`sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8` on all eight target GPUs. It resolved
`BailingMoeV2_5ForCausalLM`, accepted TP4×PP2 and a 256-token context, then stopped before weight loading:

```text
The quantization method fp8 is not supported for the current GPU.
Minimum capability: 75. Current capability: 70.
```

The pinned 1Cat source at commit `91aca502d2bb1f05d9208ab2edec9fae53ff0d0b` registers Bailing V2.5 and has an
SM70 block-FP8 path. Its Bailing prefill nevertheless calls `lightning_attention`, whose implementation
unconditionally rejects compute capability below 8.0. Therefore the service cannot return its first real prompt even
if TP4×PP2 weight loading fits. A redundant cross-host image transfer was stopped after the exact stock live gate and
the pinned-source prefill gate established the outcome; no 1Cat workload was left running.

## Compiler trace gates

The V100 environment first required the CUDA 12.6 PyTorch wheel: the default CUDA 13.0 PyTorch 2.13 wheel excludes
SM70, while `torch==2.13.0+cu126` includes `sm_70` and completed a real FP16 V100 matrix multiplication. This PR's
guarded quantized custom-code loader tests passed 19/19 there, and the exact trace exercised that retry.

The checkpoint's remote code cannot import under Transformers 5.14 because it imports
`is_torch_fx_available`, which was removed after the checkpoint's declared Transformers 4.56.2. A bounded retry with
4.56.2 reached `torch.export` on layer 0, then FLA's `FusedRecurrentFunction` entered its Triton autotuner under
FakeTensor mode and failed on a tensor data-pointer access. Lightning attention needs an opaque custom operation or
frontend mapping before this layer is traceable.

The MLA layers do not offer a representative escape hatch: their remote-code ModuleList MoE uses `topk`, `argsort`,
and scatter rather than Emmy's supported packed expert interface. Tracing a dense-only substitute would omit the
checkpoint's hybrid attention and expert routing, so it was not labeled as model evidence.

No non-empty post-fusion inventory was emitted. Equal-budget tuning, O3 verification, a working golden, and accuracy
testing were therefore not started.

## Decision

Do not create or publish a V100 recipe or serving image for either checkpoint. A future qualification can resume when
Ling 3.0 has more device memory, a Lightning attention backend supports SM70, and Emmy gains a traceable Bailing/FLA
frontend contract.

## Sources

- [Ling 3.0 Flash FP8 model card](https://huggingface.co/inclusionAI/Ling-3.0-flash-fp8)
- [Ling 2.6 Flash FP8 model card](https://huggingface.co/inclusionAI/Ling-2.6-flash-fp8)
