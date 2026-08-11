# Gemma 4 12B targeted kernel replay on RTX 5090

This bounded replay uses the existing `bench_golden_set.py` command path and the current RTX 5090 golden. It selects
the four `gemma4_12b.mlp_geglu` realizations most relevant to the long-prompt article lane instead of starting the
complete 309-realization, two-lane cold-cache inventory.

The standard and fast-math structured results are preserved alongside this report. Both lanes keep the small M=32
realizations within 6--10% of eager PyTorch. The M=4096 realizations are 1.9--2.2x slower than eager and
`torch.compile`; they remain an explicit performance gap. Scoped and unscoped placement probes did not yield a legal,
fully lowered split for this two-branch MLP graph, so the existing golden was not changed without executable evidence.
