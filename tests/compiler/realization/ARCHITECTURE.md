# The realization corpus

A data-driven regression lane for one failure class: **a schedule that should be realizable is not**. Each case is a
checked-in minimized reproducer — one program, one authored schedule — and the lane replays it against the compiler in
front of you.

This directory is kind-organized in the sense `tests/ARCHITECTURE.md` sanctions: its cases span lowering, the CUDA
backend, the pin machinery and the golden loader, and they share one workflow.

## Layout

```
helpers.py            # load, regenerate, and the four oracles
regen.py              # `make test-corpus-regen` — applies the fix the staleness test detects
test_realization.py   # one parametrized walker over cases/
cases/<family>/<name>.yaml
```

## What earns a case

Only a realization gap: a schedule family that is never offered, a pin that refuses or fails to lower, or a pin that
runs wrong. Two neighbouring failure classes deliberately do **not** earn one, because admitting them would make the
ratchet meaningless:

- **search shortfall** — the schedule realizes and the prior simply does not pick it. Fix it by measuring, or report it
  as a prior finding.
- **code generation quality** — the right tier is present and still loses. Report it; do not record it.

A schedule the compiler *correctly refuses* is not a gap either. A slab that does not fit, a byte transport with no
sibling on a computed operand, a masked axis with no register block to stage — these are right answers, and their
tests assert the refusal message, which a row cannot express. Keep them in Python.

That rule is about what earns a case, not about what a case then measures. Once a case exists it is an ordinary
reproducer and carries whatever the lane can learn from it.

## The case file

A working golden document — byte-for-byte what the inventory writer emits — plus the authored pin, and nothing else:

```yaml
compute_cap: [12, 0]
programs:
- inputs: [a, b]
  outputs: [c]
  nodes: …
configs:
- program: 0
  target: {origins: [c]}
  realizations:
  - name: k_matmul_5b7645.167d5f47efce
    bindings: {}
    pins: {FAST_MATH: true}
    knobs: {WORK: w2x2, TILE: mma_m16n8k16_f16_f16/f4x8/k2, REDUCE: g2k, STAGE: ''}
    identity: 0302cbd2c129ae1851d5f529621a752756f6181d0d4cbaf57eb22f85028d11c2
```

Why each part, and why nothing else:

- `programs` / `target` / `compute_cap` — the reproducer. Stable Torch IR rather than a code snippet, so a frontend
  change cannot silently alter what the corpus tests.
- `name` — already carries `Op.cache_key()[:12]`, so it detects cache-key drift for free.
- `pins` / `knobs` — the authored schedule. Regeneration structurally cannot produce these, which is what makes the
  staleness mechanism safe.
- `identity` — the record's `deploy_identity`. `cache_key` folds the class name, the algebra key and the knobs;
  `deploy_identity` additionally folds the dtype, extent, shape and store fingerprints, so a new fingerprint fact moves
  `identity` while leaving `name` untouched.
- `identity` and the optional per-card `latency` block are the only additions the corpus makes to the golden schema,
  and both are optional keys the model goldens do not carry.

Three spelling rules decide what a case actually asserts:

- **A knob present with `''` is pinned OFF; a knob absent is free.** `''` is a decided value — the schedule declined
  that family — while an absent key lets the fork choose. Several of the tests this corpus replaces `delenv` a family
  rather than setting it empty, and the two are different pins.
- **`PLACE` goes in `pins`, not `knobs`.** Graph placement is consumed by a splice, so it is not a schedule row and
  the golden validator refuses to see it beside one.
- **Binding a symbolic dimension specializes the program.** A case with `bindings: {}` keeps its symbolic axis and runs
  at the dimension's own `Dim` hint — the size `emmy run` already resolves a symbolic reproducer to. The corpus has no
  spelling for "compile at the hint, run at some other size", so a sweep of one symbolic kernel across many runtime
  sizes stays in Python.

## Expectations live in the filename

There is no manifest. Extending the corpus is writing one file:

```
<family>/<name>.yaml                  # closed — every applicable stage must pass
<family>/<name>_xfail_offered.yaml    # open — strict xfail at that stage
<family>/<name>_xfail_realized.yaml
<family>/<name>_xfail_built.yaml
<family>/<name>_xfail_correct.yaml
```

The open-gap inventory is `ls cases/**/*_xfail_*.yaml`, and the completion gate is "no file matches that glob". Closing
a gap is a `git mv`, so the diff shows the closure as a rename. And two concurrent runs on different models can each
add a case without touching a shared file.

The cost is that the filename is semantic, so an `_xfail`-shaped token naming something other than the four stages is a
hard error rather than a silently-closed case.

An `_xfail_*` file must carry a leading `# evidence:` comment naming why the schedule *should* be realizable — a
sibling card's golden carrying that family for the same structural identity, the same family already winning at a
neighbouring binding, or an explicit roofline argument. Without it the corpus fills with speculation. The rest of the
leading comment block is prose about where the gap came from; regeneration preserves it.

## The four stages

| Stage | Assertion | GPU |
| --- | --- | --- |
| `offered` | under `pinned_knobs(pins + knobs)`, `enumerate_graph` at the declared capability returns at least one row satisfying the pin | no |
| `realized` | the graph lowers through `CUDA_PASSES` at that capability, `unreproducible_pin_flag` is `None`, and every authored family is stamped | no |
| `built` | `CudaBackend().compile(...)` under the pin — nvcc accepts it | yes, exact capability |
| `correct` | run against the reference within tolerance | yes, exact capability |

Each is its own test node, so an `_xfail_<stage>` suffix lands on exactly the stage it names; the stages past a
declared gap are skipped, because a schedule that never realizes has nothing to run.

`offered` and `realized` always run at the **declared** capability, so an sm_70 lockout is exercised on any box,
GPU or not. `built` and `correct` run only when the live capability **equals** the declared one — a pinned schedule is
a claim about one capability, never about a merely newer card.

The reference for `correct` is derived from the target, the way `emmy run` derives it: a frontend program
(`target: {origins: …}`) compares against the numpy backend; an exact Loop target has no torch twin and compares
against the same-input greedy execution of the same program.

`offered` asks whether the pin *can be honoured*, not whether the tier would be offered to an **unpinned** search.
Those differ, and the difference is load-bearing: a pin narrows the candidate grid authoritatively, so a schedule the
cold search never enumerates can still be offered here. A tier the search will not reach on its own is a search
shortfall, and the corpus does not express it.

**Pinned-enumeration membership is the primary oracle, not `unreproducible_pin_flag` alone.** The flag answers `None`
for a registered family that nothing stamped — serialized IR can omit knob stamps — so a pin that cannot be offered at
all would read as satisfied. Membership is asked per row *through* the flag, so the families it already reads correctly
(a `PLACE` consumed by a splice, the structural `g<n>` half of a cross-CTA `REDUCE` split) stay correctly read; and
`realized` closes the flag's hole with an explicit stamping check over the authored knobs.

## Stage 5 — latency

`perf`-marked, `-O3`, and skipped unless the live card's capability matches the case's. It compares
the best of three runs against the case's stored latency for this card, and a slower case
**reports** rather than fails: enforcement belongs in a human reviewing the timing-refresh pull
request, not in a red test a legitimate correctness fix could pin red forever. Nothing auto-updates
a stored number — an automatic ratchet ends up pinned to the luckiest noise excursion ever
observed. `emmy run --golden-file FILE --bench --record` is the only writer.

The band is 5%, measured rather than guessed: ten cases spanning 1.5 us to 579 us, four estimates
each on an idle RTX 5090, put the best-of-three estimator's own spread at a median of 0.17% and a
maximum of 0.74%.

Latency lives in an optional per-card block keyed by `Context.hardware_id` — the identity that
already separates same-die SKUs like H100 from H200, which a free-text card name does not. Both
numbers are stored, because the block answers two questions and only one of them is a ratchet:
`emmy_us` against its own stored value says *did we regress*, and `tcompile_us` beside it says *are
we ahead of or behind torch*, per case, per card. That ratio, sorted, is the optimization worklist.

A closed case with no timing for the live card is reported once at session end, on a card that can
answer it and nowhere else. That asymmetry is deliberate: the derived-half check is GPU-free so it
fires everywhere and its fix works everywhere, while a timing can only be produced on the machine
holding the card.

## Staleness: regeneration, not stamps

Kernel identity and schedule codec spellings change often, so a stored case rots. The failure mode that matters is
silent: a retired knob spelling canonicalizes to itself, matches no candidate, returns zero rows, and reports as a
lockout — a phantom compiler gap. For an open case the mirror applies: the xfail keeps passing and the ratchet stops
ratcheting.

**Detection is a test, not a command.** `test_case_derived_half_is_current` recomputes each case's derived half —
decode the program, re-run the inventory writer under `Context.from_target(compute_cap)`, re-derive `identity`,
re-canonicalize `knobs` — and asserts it equals what is stored. The check is GPU-free at roughly 0.02 s per case, so
codec and kernel-identity drift is caught on the pull request that causes it, by the commit that causes it. Nothing in
the tree does that for `recipes/*/golden/*.yaml` today.

`make test-corpus-regen` only *applies* the fix. That split is the shape the repository already uses twice:
`ruff format --check` detects while `make format` fixes, and the session-end durations gate names its offenders and
asks for `make test-durations`.

Five rules make it load-bearing:

1. **Regenerate through the library, not a CLI.** `emmy trace --target sm_89` still stamps `gpu_name` from the live
   card; the library path with an explicit context emits none. That is what makes the check machine-independent, so it
   fires and its fix works on any box.
2. **Canonicalize the authored knobs strictly.** `canon_family_value` normally swallows a `ValueError` and returns the
   raw string, which is exactly how a retired spelling survives. Under `strict=True` regeneration fails loudly on
   `STAGE=d2/ring`, `WORK=zzz9x9`, `TILE=mma_m64n64k64_…` or `REDUCE=g2z`, while a valid-but-unreachable pin
   (`WORK=w7x13`, `TILE=…/f99x99/k8`) parses cleanly and falls through to `offered`, where a genuine lockout belongs.
3. **Refuse to write when a verdict changed.** If one commit moves an identity and breaks realization, regeneration
   fixes the first and must not let the second ride along. It names the affected cases and exits non-zero; resolving
   them is a review conversation, not a mechanical step.
4. **Preserve what regeneration cannot produce.** A `latency` block is measured on a card, not derived from the
   program, so a regeneration on a machine without that card carries existing entries through untouched.
5. **Preserve the leading comment block.** `dump_golden_file` is a plain YAML dump and drops comments, so a naive
   rewrite would eat the `# evidence:` line on every regeneration.

## Adding a case

```bash
case=tests/compiler/realization/cases/<family>/<name>_xfail_<stage>.yaml
emmy trace -c "<snippet>" --target sm_<cc> -o "$case"
cat >> "$case" <<'EOF'   # the knobs block, copied from `run --json`'s record_knobs
EOF
make test-corpus-regen   # normalizes, stamps identity, re-prepends the comment block
```

Then prove the case reproduces the gap: it must fail without the suffix and pass with it.

## What stays in Python

The boundary is what a row can express.

- **Corpus, as data:** whether a pinned schedule is offered, realized, built and correct at a capability.
- **Python, as code:** *how* — emitted-source substrings, bit-identity between two configs, kernel counts, refusal
  messages, compile-budget claims, and a sweep of one symbolic kernel across several runtime sizes.

The corpus is therefore overwhelmingly additive. It subsumes the accuracy-only tests whose whole assertion was "this
program under this pinned schedule computes the right answer"; a test that also asserts structure keeps its structural
half.
