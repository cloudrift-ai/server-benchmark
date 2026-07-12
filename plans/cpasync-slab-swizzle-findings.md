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

## Implementation sketch (not started)

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
