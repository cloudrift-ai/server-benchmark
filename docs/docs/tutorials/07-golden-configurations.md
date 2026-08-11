---
sidebar_position: 7
title: "7. Golden Configurations"
description: The reviewed, per-GPU measurements that ship with the repository — how they are recorded, what makes one unusable, and the audits that catch it.
keywords: [Emmy, golden configuration, evidence, audit, benchmark, pin, realize]
---

# 7. Golden Configurations

A **golden configuration** is one persisted symbolic program target. Its **realizations** are the dimension bindings
and precision regimes measured for that target on a specific GPU. They matter more than their modest description
suggests, because as [the stores page](./04-measuring-and-recalling.md) noted, **they are the only measured data that
travels with the repository**. On a freshly rented machine they are the difference between deploying on evidence and
deploying on a guess.

They do three jobs at once:

1. **The first tier** of the deploy evidence hierarchy.
2. **The training data** for the offline prior, which is fitted on them.
3. **A regression reference** — if today's compiler produces something slower than the recording, that is a defect
   with a number attached.

## What one looks like

Golden configurations live in one file per GPU. A file embeds its program pool, then lists structural targets whose
realization arrays hold bindings, regimes, schedules, and paired measurements:

```yaml
gpu_name: NVIDIA GeForce RTX 5090
compute_cap: [12, 0]
model: google/gemma-4-12B-it
programs:
  - {inputs: [...], outputs: [...], nodes: [...]}
configs:
  - program: 0
    target: {origins: [linear_7]}
    realizations:
      - name: gemma4_12b.norm_q_proj.m32
        bindings: {num_tokens: 32}
        pins: {FAST_MATH: false}
        knobs: {WORK: w1x16, TILE: mma_m16n8k16_f16_f32/f2x2/k2, REDUCE: g8k, RASTER: '', STAGE: d2/sync}
        measurements: {emmy_us: 26.7, reference_us: 19.8, reference_backend: cublas}
```

The program and target identify the structural kernel. Empty `bindings` keep the program symbolic; a mapping such as
`num_tokens: 32` specializes that symbolic dimension before lowering. `pins` applies registered knob values before
enumeration; `knobs` records the configuration selected and measured inside that regime. The knobs use the exact
spelling from [the forks page](./03-forks-and-knobs.md), and `measurements` records the candidate beside a named
reference. `FAST_MATH` follows this same rule and appears under `pins`; it has no dedicated realization field. Keeping
all realizations below one target makes it explicit which input dimension changes and prevents static copies of the
same program from drifting apart.

Names repeat across files — every GPU has its own `matmul.square.512` — with different shapes, different data types
and different measured times. So a compile only ever consults the file for the GPU it is targeting. Pooling them would
mean deploying one card's configuration on another.

## Recording one

A golden is recorded from a side-by-side comparison run:

```bash
emmy run --golden matmul.square.512 --bench --ab "WORK=w2x2,TILE=f2x8,STAGE=d2/cp"
```

To verify a target still living in a working YAML—including an exact Loop IR fallback—select both the file and row:

```bash
emmy compile --golden-file _tune/model/working.yaml --golden target.name --ir cuda
emmy run --golden-file _tune/model/working.yaml --golden target.name --bench --ab "WORK=w2x2,TILE=f2x8"
```

The working row supplies the graph regardless of state. Only rows with verified paired timings are auto-pinned;
inventory and proposal knobs run only when supplied explicitly through `--ab`.

That compiles the shape the way the compiler would on its own, then compiles it again with the given knob values
pinned, and prints both. Two rules about which number to copy:

- **Record from a pinned row, never from the ordinary comparison row.** The ordinary row is measured interleaved with
  the PyTorch baselines, so the allocator state and cache contents of another framework are resident while it runs. It
  is the right number to compare against PyTorch and the wrong number to compare against a pinned row — the gap
  observed in practice is around 7 percent. Whenever pinned rows are measured, the ordinary configuration is
  additionally re-measured on its own, and *that* is the baseline the pinned rows are compared against.
- **Copy the whole knob map, including the families that are off.** Each row prints the knob values the compile
  actually produced, with every schedule family written out explicitly. An entry that omits a family leaves that
  family to whatever the compiler fills in when the entry is replayed, which shifts as the compiler evolves — a
  recurring source of regressions that look real and are not.

Three checks guard the measurement before it is believed:

1. **The pin must have been honored.** The knob values the compile produced are compared against the pinned ones, and
   a mismatched row fails without being measured at all. A structurally invalid pin silently falls back to the
   compiler's own choice, so measuring it would compare the compiler against itself and report a flattering result
   under the pin's name.
2. **The arithmetic must be plausible.** A row implying more arithmetic per second than the card can physically
   perform is flagged as a bad measurement rather than celebrated as a fast kernel.
3. **The answer must be right.** Each pinned configuration is executed once on the same inputs as the ordinary run and
   its outputs compared. A kernel that skips a step it should not have produces plausible-looking garbage very
   quickly.

Every row is measured in a separate worker process that can be killed, so one configuration that hangs takes down its
own process, is reported as a failure, and the remaining rows continue.

## The failure that matters most

A recorded configuration is only useful if the compiler still offers it. The word for that is **realize**: a recording
realizes at a fork when the options the compiler offers there include one that matches it.

**A golden that realizes nowhere is worse than no golden at all.** When a shape matches but none of its entries
matches anything on offer, the compile warns loudly and falls through to the tiers below — and those tiers, for a
shape somebody bothered to record, can be hundreds of times slower than the entry claims. The recording promises a
number the deployment cannot produce.

Two ways to record one by accident, both worth knowing:

- **A recorded cut and a tile choice cannot go in the same entry.** A cut is decided before schedules are chosen, so
  no single offered option carries both, and an entry containing both matches nothing. Record one or the other. Here
  is a real cut entry, recorded for the same shape as the fused one above:

  ```yaml
  - name: gemma4_12b.norm_q_proj.m32.cut
    bindings: {num_tokens: 32}
    pins: {FAST_MATH: false}
    knobs: {PLACE@cone: cut}
    measurements: {emmy_us: 16.0, reference_us: 19.0, reference_backend: cublas}
  ```

  It stores the split and nothing else. Each resulting piece is recognized on its own afterwards and finds its own
  schedule through the hierarchy. In this case the split is 1.8 times faster than keeping the work fused, which is
  exactly the kind of finding that has to be recorded to be usable — a cold compile keeps work fused unless told
  otherwise.

- **A row must be verified to deploy, not merely to reproduce.** Pinning reproduces configurations the compiler would
  never offer on its own, so a recording that only works when pinned still looks healthy in an isolated check. Only
  the in-model audit catches it.

## The release audit

One command checks the own-program and served-model views together:

```bash
emmy eval golden <canonical-golden.yaml> --serving-config <models/slug.env>
```

The serving config names that exact file and supplies the model, revision, GPU, and reachable realization matrix.
The command must run on that GPU. It validates the schema and provenance, proves every structural target contains
every expected static/symbolic precision realization, reproduces the recorded rows, runs the pin-only offer check in
each input pin regime, and then rebuilds the model as a weight-free stand-in to audit the exact serving widths. Every
consultation yields:

| Verdict | Meaning |
| --- | --- |
| `MATCH` | a recorded configuration matched an option the compiler offered |
| `DRIFT` | the shape matched but none of its entries did — always a defect, since the recording claims a time the deployment can no longer produce |
| `GAP` | there is no recording for this shape |

The two views genuinely differ, which is why the command runs both. Isolated checks have passed 68 out of 68 while
the same configurations drifted inside the model, where fusion changed the offered set. For release, any missing
realization, all-pin-only target, DRIFT, GAP, or compile failure is fatal.

Coverage is gated in continuous integration so that it can only improve: each GPU's set of gaps is pinned exactly, a
new gap fails until a configuration is recorded, and a gap that has been closed fails until its line is removed from
the baseline. The stand-in models deliberately follow the installed modeling library, so an upgrade that changes the
forward pass changes the audit exactly as it changes serving.

## Two smaller rules

**Fast math never loses.** Entries recorded with faster, less precise arithmetic are only kept when they are faster
than the best ordinary sibling. A slower one could never be used anyway — the ordinary entry is picked first, whether
fast math is enabled or not — so such rows are dropped, and a missing one simply means a fast-math deployment uses the
ordinary configuration there.

**The two halves of the prior treat goldens differently.** The online prior never trains on them: a recorded
configuration enters no training data anywhere, which leaves the goldens as a clean acceptance set — data the model is
judged against but never learns from. The offline prior *is* fitted on them. That distinction is the subject of the
next page.

## See it yourself

Read a real file — they are plain YAML, and the comments in them are the record of why each entry is what it is:

```bash
ls emmy/compiler/pipeline/search/goldens/
grep -n -A9 "kernel: norm_linear" emmy/compiler/pipeline/search/goldens/rtx5090_sm120_gemma4.yaml | head -30
```

Then run the file-scoped audit on the GPU named by the serving config. It needs model configuration and allocation
metadata, but no weight payload:

```bash
emmy eval golden emmy/compiler/pipeline/search/goldens/rtx5090_sm120_gemma4.yaml \
  --serving-config docker/vllm-emmy-serve/models/gemma-4-12b-it.env
```

Next: [8. Inside the prior](./08-inside-the-prior.md).
