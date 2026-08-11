# Golden-set bench — filter 'gemma4_12b.mlp_geglu'

| kernel | eager µs | torch.compile µs | emmy µs | vs eager | vs torch.compile |
|---|--:|--:|--:|--:|--:|
| gemma4_12b.mlp_geglu.m4096.lin | 4702 | 4515 | 10400.0 | 0.45x | 0.43x |
| gemma4_12b.mlp_geglu.m4096 | 4746 | 4567 | 8910.0 | 0.53x | 0.51x |
| gemma4_12b.mlp_geglu.m32 | 156 | 153 | 169.0 | 0.92x | 0.91x |
| gemma4_12b.mlp_geglu.m32.lin | 163 | 160 | 171.0 | 0.95x | 0.94x |
