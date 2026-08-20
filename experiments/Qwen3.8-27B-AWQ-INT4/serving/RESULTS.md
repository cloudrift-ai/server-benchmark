# Qwen3.8-27B W4A16 serving on RTX 4090

Factual artifact index for the shared serving protocol. Each platform section records what was executed and where the
raw evidence lives; it does not interpret the measurements. The recommended-configuration report lives beside the
recipe in `recipes/Qwen3.8-27B-AWQ-INT4/RESULTS.md`.

## NVIDIA GeForce RTX 4090 x1

- Archive: `results_rtx4090x1.tar.gz`, archived root `2026-08-20_01-47-44/`.
- Run timestamp: `2026-08-20T01:47:44Z`; repository revision `a20b10790824a04d195c707d0dda3d8fa5e1cf68`, staged
  source clean.
- Host: `riftvm`, Ubuntu 24.04.1, kernel 6.8.0-134-generic, Intel Xeon Platinum 8352V, 7 logical CPUs, 46 GiB RAM.
- GPU: one NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9, driver 580.159.03,
  UUID `GPU-81d79c00-868e-3ec5-2948-745283b756f6`.
- Model: `philbert440/Qwen3.8-27B-W4A16-AWQ@7908d42a71077a5e4dc458f273682b12dfe384a0`.
- Engine image: `vllm/vllm-openai@sha256:dae7af23ea9b66b4f15de3d5e4ddebfdafa7be636be91d400184c1666f1b1462`.
- Controls: seed 0, temperature 0, ignored EOS, prefix caching disabled, text-only path
  (`--language-model-only`), context 8,192, TP1.
- Rows executed: 3 of 3 succeeded, 0 failed requests across all rows.

| Row | Input / output | Concurrency | Prompts |
| --- | ---: | ---: | ---: |
| `rtx4090x1_mc4_np32_ril128_rol128` | 128 / 128 | 4 | 32 |
| `rtx4090x1_mc4_np16_ril1024_rol256` | 1,024 / 256 | 4 | 16 |
| `rtx4090x1_mc2_np8_ril4096_rol512` | 4,096 / 512 | 2 | 8 |

The archive contains the per-row system-only experiment records, the client benchmark logs, the engine server logs,
and the executed recipe snapshot.

## Reproduce

```bash
emmy bench experiments/Qwen3.8-27B-AWQ-INT4/serving --ssh USER@HOST
```
