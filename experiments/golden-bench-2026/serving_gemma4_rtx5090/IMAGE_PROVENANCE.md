# Gemma same-image control provenance

The matched end-to-end A/B uses the exact same immutable derivative image for both arms:
`cloudriftai/vllm-emmy-gemma-4-12b-it@sha256:5add12d3b7f4673790b435b76635082433538e3615fbc40227fa1c0db64c9ff3`.
The stock control overrides that image's `/opt/emmy/serve.sh` entrypoint with
`python3 -m vllm.entrypoints.openai.api_server` and does not set the `EmmyGenModel` architecture override. Therefore
the container filesystem, CUDA runtime, Python environment, vLLM installation, model snapshot, and weights are
identical; the model route and arm-specific compiler settings differ. This controls the runtime image but does not
isolate compiler kernels from `EmmyGenModel` integration behavior.

The image records vLLM source revision `91df0fad4dc98a67c7659d9dbd915245d5c43d96`. For reference, the upstream stock
multi-platform image is
`vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`, whose
Linux/amd64 manifest is `sha256:3a1e7f5904e1a1192a02aa0086ceaffc33985d7044c7bb25b3a43d61bdbe3ac0`; it is not
an arm in the matched-system table.

The machine-readable contract is [IMAGE_PROVENANCE.json](IMAGE_PROVENANCE.json). Its relationship to the expanded
recipe is a checked repository invariant:

```bash
./venv/bin/pytest tests/benchmark/models/test_golden_bench_2026.py
```

Also archive `docker image inspect` output and `python -m pip freeze --all` from the one image. A different digest,
stock entrypoint, package inventory, model snapshot, or scheduler setting invalidates the matched-system claim. A
compiler-caused end-to-end claim additionally requires a within-`EmmyGenModel` reference-kernel arm.
