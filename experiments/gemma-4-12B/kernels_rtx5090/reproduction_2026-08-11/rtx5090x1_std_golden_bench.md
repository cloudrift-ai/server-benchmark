# Golden-set bench — filter 'gemma4_12b.mlp_geglu'

| kernel | eager µs | torch.compile µs | emmy µs | vs eager | vs torch.compile |
|---|--:|--:|--:|--:|--:|
| gemma4_12b.mlp_geglu.m4096.lin | 4699 | 4531 | 10056.0 | 0.47x | 0.45x |
| gemma4_12b.mlp_geglu.m4096 | 4757 | 4584 | 8863.0 | 0.54x | 0.52x |
| gemma4_12b.mlp_geglu.m32 | 156 | 153 | 174.0 | 0.90x | 0.88x |
| gemma4_12b.mlp_geglu.m32.lin | 162 | 160 | 173.0 | 0.94x | 0.92x |
