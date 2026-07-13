# cp.async slab XOR-swizzle — prototype win on the sm_89 gap (−12–17%, past cuBLAS on gate_up)

> Prototype measured 2026-07-12 (rented 4090, 176.124.69.204), same session as the WSPEC refutation.
> **Gate passed decisively — this is the implementation candidate for the sm_89 residual.** Harness:
> `_tune/codegen/swizzle-proto.py` on the box (pattern-rewrites the dumped fm gate_up kernel; A-only /
> B-only / both variants for attribution).

## The finding

The cp.async transport stages its slabs **plain row-major with no swizzle** — `_atom.py` says so explicitly
("a cp.async slab is plain row-major (NONE-swizzle), a TMA slab is hardware-swizzled in-copy with the drain
XOR undoing it"), and `slab_swizzles`' own docstring records what that costs on the TMA path ("the rebuilt
NONE-swizzle transport left the ldmatrix drain bank-conflict-bound"). On sm_89 there is no TMA, so every
warp-tier matmul eats it: in the gate_up fm winner the A slab's 64B row stride makes each ldmatrix phase
4-way bank-conflicted and the B slab's 128B stride (a full 32-bank period) makes it **8-way** — the B
pair-loads are the bulk of the drain. No sweep could ever find this: staging layout is not a search
dimension on the cp transport.

Fix prototyped: XOR the 16B-unit column with row bits — A: `c ^= (row/2) & 3`, B: `c ^= row & 7` — applied
identically to the cp.async fill destinations and the ldmatrix drain addresses (contents per logical element
unchanged, so outputs are **bit-identical**; fills stay conflict-free since the XOR permutes units within a
row). cp.async 16B alignment holds; zero smem growth; +0–4 registers, no spills.

## Measurements (gate_up fm winner, 512×4096 @ 4096×28672, 6 interleaved rounds)

| variant | regs | µs (rounds) | vs baseline |
| --- | --- | --- | --- |
| fm baseline (NONE-swizzle) | 250 | 796–844 | — |
| A swizzled only | 251 | 758–762 | −5–10% |
| B swizzled only | 254 | 715–752 | −9–15% |
| **both** | 254 | **693–700** | **−12–17%** |

NCU (same kernel, launch-matched): shared-mem bank conflicts **146.8M → 14K** (~zero), shared wavefronts
180.6M → 33.8M (**5.3× fewer** — conflicts were 81% of all shared traffic), issue-active 23.6% → 32.1%,
eligible warps 0.26 → 0.38. This also re-reads the earlier NCU record correctly: the 72.7% "Memory
throughput" that motivated RASTER was the **shared-memory pipe**, not DRAM — which is why halving DRAM
traffic moved nothing. At ~695 µs emmy passes cuBLAS on gate_up (~728 µs from the recorded 1.08× ratio):
the "structural" sm_89 residual was bank conflicts.

## Implementation — LANDED (same branch, same day)

Exactly the sketch below: `_MmaOps.slab_swizzles` now feeds BOTH transports (the `transport == "tma"` gate at
`_atom._staged` is gone), `cp_async_fill` threads the mode onto `CpAsyncCopy.swizzle`, and the render XORs the
fill's flattened destination index through the same `emmy_swizzle_<mode>` helper the ldmatrix drain uses — fill
and drain agree by construction, purely address-based, zero smem growth. The scalar tier (plain-`Load` drains)
and the sync compute-fill stay NONE; the flash K/V slabs keep their `pad_cols` fix (pad and swizzle are asserted
mutually exclusive). `096_pair_ldmatrix_loads` pairs swizzled loads as before (equal-mode check; the XOR commutes
with the paired lane map — verified in the emitted sm_89 kernel: paired `x4.trans` drains read through the XOR).
Covered by `test_cp_staged_slab_is_swizzled` (mode pinned ON, fill+drain XOR presence, CPU render at sm_89) and
the pre-existing staged-vs-gmem bit-identity suite, which now exercises swizzled cp kernels on GPU.

**End-to-end verification (golden replays through the real pipeline, same 4090, `emmy run --bench --golden`):**

| golden (rtx4090) | recorded µs | live w/ swizzle | delta | vs cuBLAS |
| --- | --- | --- | --- | --- |
| mlp_gate_up.h4096 [fm] | 786.9 | **662.0–664.6** (3×) | **−16%** | **0.92** (721.0) — was 1.08 |
| mlp_gate_up.h4096 std | 964.6 (live ~940) | 918.4–919.5 | −2–5% | — |
| mlp_down.h4096 [fm] | 348.5 | **273.6** (268.8 + 4.8 finalize) | **−21%** | **0.92** (297) — was 0.90 vs live-940-era std |
| mlp_down.h4096 std | 371 | 372.9 | parity | — |
| square.4096.fp16 [fm] | 750.6 | **631.3** | **−16%** | **0.78** (eager ~808) — was 0.93 |

Accuracy checks pass (silent-success `emmy run`). **Every cp-transport matmul golden's recorded `emmy_us` is now
stale-high across the sm_89 cards.** The 4090 file was refreshed the same day by a manual sweep — 15 value
refreshes, 2 fm replaces, 4 fm adds, fm lane past cuBLAS on 9 of 11 fp16 shapes; see
`golden-postswizzle-refresh-rtx4090-findings.md`. The 4080 file (and any other cp-transport cards) still needs the
same refresh, and the tune DB / learned prior have no post-swizzle measurements at all.

## Implementation sketch (as scoped before landing)

- The drain side already has the machinery: `LdmatrixLoad` carries a `swizzle` field and the pair-fusion
  pass (`096_pair_ldmatrix_loads`) is swizzle-aware ("the swizzle XOR is per-lane address-based, so it
  commutes with the paired lane map"). Today `_atom.py` only sets non-NONE modes when
  `stage.transport == "tma"` (hardware in-copy swizzle); the change is to pick software swizzle modes for
  cp transports (`pick_swizzle_atom` on the slab row stride, as the TMA path does) and emit the **matching
  XOR in the cp.async fill destination index** — the one genuinely new piece, since TMA swizzles in-copy
  and cp.async must swizzle in the store address.
- Not a schedule knob: it is a pure staging-layout property, unconditionally better (conflicts only go
  down; the XOR costs ~1 LOP per fill address and is thread-constant in the drains). No search-space
  change, no golden-schema change.
- Applies to every cp.async-staged kernel: both lanes (std gate_up should gain too — same layout), all
  sm_89 cards (4090/4080), and any cp-staged shapes on sm_120. Expect golden re-benches to move; the
  recorded `emmy_us` values for cp-transport matmul entries go stale wholesale once this lands.
- Validation: bit-identical outputs per shape (same accumulation order), the usual coverage tests plus a
  bank-conflict NCU spot-check; A/B per golden shape as with RASTER.

## Prototype-gated status of the sm_89 lever list (for the record)

RASTER (neutral), address-CSE (refuted), WSPEC-over-cp.async (refuted, SMSP banking), forced occupancy
(refuted) — and now slab swizzle: **the first measured win of the series**, found by reading the emitted
addressing against the bank layout rather than permuting schedule knobs.
