# The realization corpus

A data-driven regression lane for one failure class: **a schedule that should be realizable is not**. Each case is a
checked-in minimized reproducer — one program, one authored kernel set — and the lane replays it against the compiler
in front of you: once as a hand pin, to ask whether each schedule can be offered at all, and then as the compile's only
evidence, strict, to ask whether the compiler realizes, builds and runs the set the way a deploy would.

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

- **search shortfall** — the schedule realizes and the prior simply does not pick it when nothing measured is in
  scope. Fix it by measuring, or report it as a prior finding. (A row that *is* in scope and still is not picked is
  not a shortfall: it is the replay contract failing, and `realized` reports it.)
- **code generation quality** — the right tier is present and still loses. Report it; do not record it.

A schedule the compiler *correctly refuses* is not a gap either. A slab that does not fit, a byte transport with no
sibling on a computed operand, a masked axis with no register block to stage — these are right answers, and their
tests assert the refusal message, which a row cannot express. Keep them in Python.

That rule is about what earns a case, not about what a case then measures. Once a case exists it is an ordinary
reproducer and carries whatever the lane can learn from it.

## The case file

A working golden document — byte-for-byte what the inventory writer emits — plus the authored entries, and nothing
else. The first entry is the target's own; every further entry decides one more kernel of the set the target compiles
to and names it by `identity`:

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
- `name` — already carries the variant key (`identity_key(with_io=True, with_knobs=True)`)`[:12]`, so it detects
  cache-key drift for free.
- `pins` / `knobs` — the authored schedule, one entry per kernel of the set. `pins` are the input regime; `knobs` are
  the row the entry's kernel realizes, spelled on that kernel's own tree — a kernel-set decision (`PLACE@seam: cut`,
  `REDUCE@k: g2k`) is an entry whose `identity` is the kernel the fork was offered on. Regeneration structurally cannot
  produce these, which is what makes the staleness mechanism safe.
- `identity` — the record's deploy identity — `identity_key(with_io=True)`, structural flavor — is the digest of the
  complete schedule-free Loop-IR body the term lowers to, folded with the io dtype/shape fingerprint. The variant key
  (in `name`) is the
  variant key — the same body + io folded with the knob row — so a knob-only change moves `name` while leaving
  `identity` untouched.
- `identity` and the optional per-card `latency` block are the only additions the corpus makes to the golden schema,
  and both are optional keys the model goldens do not carry. On a further entry `identity` is authored: it is the
  selector that lets the replay apply that entry at its own kernel's forks (`golden._replay` walks a target's entries
  as one set, deciding each fork by the entry whose identity is the kernel being offered).

Three spelling rules decide what a case actually asserts:

- **A knob present with `''` is pinned OFF; a knob absent is free.** `''` is a decided value — the schedule declined
  that family — while an absent key lets the fork choose. Several of the tests this corpus replaces `delenv` a family
  rather than setting it empty, and the two are different pins.
- **A placement is an entry of its own.** `PLACE@seam: cut` in the `knobs` of an entry whose identity is the kernel
  the cut is offered on; the golden validator refuses a placement key beside a schedule row. Older cases carry the
  route in the first entry's `pins`, which the replay reads the same way.
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
| `realized` | with the case as the compile's only evidence, the graph lowers through `CUDA_PASSES` at that capability, `unreproducible_pin_flag` is `None`, every authored family is stamped, and every kernel-set decision the case spells was taken | no |
| `built` | lower the same way on the live card, then build a `CompiledProgram` — nvcc accepts it | yes, exact capability |
| `correct` | run against the reference within tolerance | yes, exact capability |

**Only `offered` is a hand pin, asked of each entry.** The other three run under `helpers.evidence_scope`: the case's
entries are the whole golden scope, strictly (`golden.sole_evidence`, the scope the release gate compiles under too;
each entry standing in as a measured row — a case authors schedules rather than measuring them, and a proposal is no
evidence), so a fork no entry decides is an `EvidenceError` naming the kernel, never a prior's guess; the machine-local
online prior is out of the way, the tune DB is not consulted, and the environment carries the case's input pins alone — the regime
it was measured under (`FAST_MATH` and the precision gates), never its route or its schedule row. The route and the
row reach the compile as measured rows of the kernels they decide, through the same evidence pick every `compile` /
`run` / `serve` uses (`golden.evidence_rows`, `greedy._route_candidates`), or they do not reach it at all. That is the
deploy contract, asked of every case on every commit: a row the compiler can honour under a pin but does not select
when it is the evidence — a stale spelling, a route key no offered seam carries, a schedule that equals no leaf of the
kernel that deploys — fails `realized`, and the failure names what was lost. A kernel-set decision is checked through
the engine's own splice events (`PipelineStrategy.on_splice`), because no stamp on the resulting kernels can show a
placement cut or a cross-CTA split that was not taken.

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
shortfall, and the corpus does not express it. `realized` then asks the complementary question of the same schedule:
given as evidence rather than as a pin, is it what the compiler picks.

**Pinned-enumeration membership is the primary oracle, not `unreproducible_pin_flag` alone.** The flag answers `None`
for a registered family that nothing stamped — serialized IR can omit knob stamps — so a pin that cannot be offered at
all would read as satisfied. Membership is asked per row *through* the flag, so the families it already reads correctly
(a `PLACE` consumed by a splice, the structural `g<n>` half of a cross-CTA `REDUCE` split) stay correctly read;
`realized` closes the flag's hole with an explicit stamping check over the authored knobs, and asks the splice events
whether those two structural decisions were taken.

## Latency

A case may also carry measured microseconds. **The measuring lives in `tests/perf/`**, not here:
`make test` compiles at `-Xcicc -O1`, which is not a measurement lane, so a latency assertion in
this directory would measure the wrong regime entirely. That lane benches every closed case its
card can run, joins the result to its comparison table, and compares the same measurement against
the stored number — one bench, both answers. See `tests/perf/ARCHITECTURE.md`.

A slower case **reports** rather than fails: enforcement belongs in a human reviewing the
timing-refresh pull request, not in a red test a legitimate correctness fix could pin red forever.
Nothing auto-updates a stored number — an automatic ratchet ends up pinned to the luckiest noise
excursion ever observed. `emmy run --golden FILE --realization NAME --bench --record` is the only writer.

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

The perf command names the case's target entry (`run --golden <case> --realization <name>`), which
benches it as a pinned row whatever its measurement state: that entry's input `pins` and schedule
`knobs` — a placement cut included — are published as a hand pin for that one compile, and the
case's further entries are the compile's golden evidence, so the set the case authors is the one
measured, never the planner's own pick under its name.

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
2. **Validate authored knobs strictly.** `validate_family_value` requires every classic value to use its sole wire
   spelling. Regeneration fails loudly on `STAGE=d2/ring`, `WORK=zzz9x9`, `TILE=mma_m64n64k64_…` or `REDUCE=g2z`,
   while a canonical but unreachable pin (`WORK=w7x13`, `TILE=…/f99x99/k8`) parses cleanly and falls through to
   `offered`, where a genuine lockout belongs.
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
make test-corpus-regen COMPLETE=1   # normalizes, stamps identity, adds an entry per undescribed kernel
```

`COMPLETE=1` (`regen.py --complete`, `helpers.complete`) replays the set the way the deploy reads it and appends an
entry for every scheduled kernel no entry names and no entry's row vouches for — that kernel's identity, the input
regime and the row the replay realized on it — so strict evidence has a row at every fork. That is authoring: the
added rows are enumerable schedules the case pins from then on, and a kernel the author cares about should get its
row by hand before the completion fills in the rest.

Then prove the case reproduces the gap: it must fail without the suffix and pass with it.

## What stays in Python

The boundary is what a row can express.

- **Corpus, as data:** whether a pinned schedule is offered, realized, built and correct at a capability.
- **Python, as code:** *how* — emitted-source substrings, bit-identity between two configs, kernel counts, refusal
  messages, compile-budget claims, and a sweep of one symbolic kernel across several runtime sizes.

The corpus is therefore overwhelmingly additive. It subsumes the accuracy-only tests whose whole assertion was "this
program under this pinned schedule computes the right answer"; a test that also asserts structure keeps its structural
half.
