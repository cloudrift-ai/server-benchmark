# Committed agent kernel sets

One directory per generator run, `kernels/<agent>/`, holding `MANIFEST.yaml` (generator name, revision, model,
budget, date, host GPU) and one `<target>/` directory per common-corpus golden target with `reference.py` and
`kernel.py` in the KernelBench module convention. The recipe measures what is committed here and generates
nothing; see `../recipe.yaml`.
