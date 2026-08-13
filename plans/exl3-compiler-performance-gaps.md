# EXL3 compiler integration: remaining performance work

## Selected boundary

Every EXL3 coded linear enters the compiler through `spell_factored_linear` at graph birth. Trunk linears, sparse
experts, and the coded output head therefore expose the same factorized contractions and pointwise algebra to frontend
decomposition, Loop IR, scheduling, and CUDA lowering. No EXL3 operation survives the spelling band, and the runtime
contains no vendored CUDA source or native EXL3 helper.

Sparse single-token decode keeps the existing fixed-slot program selection. Each selected expert replays an ordinary
compiler program over E-leading coded tensors, and the generic combine accumulates routed outputs in fp32. The coded
head remains compressed and builds its row program from the same generic algebra.

## Known performance gaps

- Top-k sparse decode replays one generic expert program per selected slot, then runs a separate combine. This repeats
  codebook decode and Hadamard work and creates more launches than a persistent grouped implementation.
- A coded linear currently exposes three factorized contractions plus layout and pointwise work. Single-row scheduling
  is correct but requires exact-device tuning; no specialized M=1 path is selected by this change.
- The coded head evaluates sampled rows through a multi-launch generic program. It retains the decoded-head memory
  saving, but its latency, capture behavior, and multi-row profile path require fresh qualification.
- Historical exact-checkpoint diagnostics measured a removed fused route-and-expert path near 86 microseconds per
  sparse layer, a grouped generic experiment near 1.0 milliseconds, and ten direct M=1 programs near 1.3 milliseconds.
  These figures are diagnostic context only: they were not measured on this source and must not be used as results.
- Existing model reports pin older images. They do not qualify throughput, context capacity, or request-time behavior
  for the compiler-only implementation.

## Planned compiler work

1. Tune the exact generic M=1 contractions at deploy optimization and verify every promoted row twice against the
   same-input reference before adding it to a golden.
2. Make row-indexed expert selection visible to scheduling so a grouped realization cannot reuse one expert's coded
   operand across rows assigned to another expert.
3. Express any grouped or persistent optimization as structural generic IR fusion. Do not add a format-specific
   operation, native helper, source-backed launch, or backend branch.
4. Before selecting an optimization, measure exact layer outputs, workspace, launch count, captured replay, context
   capacity, and end-to-end serving. Correctness remains the acceptance gate for the current compiler-only boundary.
