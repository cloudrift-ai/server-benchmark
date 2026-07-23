# Golden seeding — merged sibling-linear edges, RTX 5090 (2026-07-22)

**GPU**: RTX 5090 (sm_120, local). **Trigger**: WS1 of the decode kernel-count plan
(`decomposition/035_merge_sibling_linears`) re-keyed every merged edge in the gemma-4 twins — the drift gate went
red with MATCH 52 / GAP 18 / DRIFT 2 per card (pre-merge baseline was 101 MATCH clean). **Method**: manual pinned
`--ab` exploration per shape (`emmy run --bench -c "<snippet>" --ab ...`), 3–4 pins per round seeded from the
family-analogous recorded rows, 1–2 rounds per shape — no tuner sweep (the cold greedy misdeploys every one of
these shapes; see Finding 1). **Wall time**: ~50 min of benching for 15 recorded entries. **Tally**: 13 new
entries seeded + 2 dynM entries re-recorded (replacing 4 stale ones) + 4 stale 4090 dynM entries deleted; 5090
audit healed to **MATCH 74 / GAP 0 / DRIFT 0**; 4090 keys baselined in the gate pending a 4090 session.

Logs and `--json` records: `_tune/merged-seed-5090/`.

## Per-shape outcomes

All `.lin` (serving `F.linear`) layout, fp16, `d1/sync` computed-A rows unless noted. "eager" is the same-run
torch reference (unfused norm + matmul chain for the fused kinds). Split totals include the finalize kernel.

| shape (new key) | winning knobs | emmy µs | eager µs | vs eager | cold-greedy µs |
| --- | --- | --- | --- | --- | --- |
| norm_gate_up.m32.lin (N=30720) | w1x16/f2x2/k4, unsplit | 161.6 | 154.1 | 0.95x | 636.8 (g8a w2x2/f4x4) |
| norm_gate_up.m256.lin | w1x16/f2x2/k4, unsplit | 559.4 | 362.0 | 0.65x | 853.0 |
| norm_gate_up.dynM.lin | w1x16/f2x2/k4, unsplit | 1079.2 | 664.0 | 0.62x | 1720.3 |
| norm_qkv.m256.lin (N=8192) | w1x16/f2x2/k4, unsplit | 174.5 | 114.0 | 0.65x | 376.0 |
| norm_qkv.dynM.lin | w2x8/f4x4/k4, unsplit | 299.8 | 224.0 | 0.75x | 407.4 |
| norm_qk_global.m32.lin (N=8704) | w1x16/f2x2/k2 g8k | 40.1 | 37.0 | 0.92x | 209.6 |
| norm_qk_global.m256.lin | w1x16/f2x2/k4, unsplit | 176.3 | 117.0 | 0.66x | 405.3 |
| norm_qk_global.dynM.lin | w2x8/f4x4/k4, unsplit | 297.9 | 222.0 | 0.75x | 423.8 |
| mlp_down_fused.m32.lin | w1x8/f2x2/k4 g4k | 95.9 | 90.0 | 0.94x | 1518.4 |
| mlp_down_fused.m256.lin | w1x8/f2x2/k4 g4k | 293.0 | 231.0 | 0.79x | 434.0 |
| mlp_down.dynM (+.lin) re-record | w1x8/f2x2/k4 g4k | 530.5 | 397.0 (cone ref) | 0.75x | 1361.5 |
| pw.n8704.m32 | f2 | 0.8 | 2.0 | 2.5x | 0.8 (greedy fine) |
| pw.n8704.m256 | f4 | 2.2 | 4.0 | 1.8x | 2.6 |
| pw.n8704.dynM | f4 | 3.5 | 4.0 | 1.1x | 4.6 |

## Finding 1 — cold greedy misdeploys every merged computed-A shape (2.2–16x)

With an empty online prior (the audit / fresh-boot regime), the greedy pick lands on `w2x2/f4x*` `g*a` rows at
2.2–4x the pinned best on every fused shape, and 16x on the down cone m32 (1518 vs 96 µs). This is the known
cold-deploy hazard class the goldens exist to pin — now covered for these keys on the 5090. The offline prior's
computed-A weights systematically prefer narrow-un tiles; a refit over the new rows
(`scripts/golden_knob_heuristics.py`) is worthwhile once the 4090 twin data exists too.

## Finding 2 — the fused forms trail eager at prefill M; the cut sibling is unreachable

At M=32 the fused megakernels sit at 0.92–0.95x eager (fine — they replace 2–4 launches). At M=256/dynM they
trail eager 0.62–0.79x: the computed-A form still cannot ride the `d2/tma/ring` transports (the WS1-era known
hazard). The `PLACE@cone=cut` escape exists on these forks but its consumer half (the plain N=30720/8192/8704
matmul at prefill M) has no goldens — an `--ab PLACE@cone=cut` probe deployed the consumer cold at 43.8 ms.
Seeding the cut requires first seeding the merged-shape *plain* matmul keys (they never appear in-model today, so
the audit doesn't force them). That plus a twin-e2e A/B of cut-vs-fused at m256/dynM is the natural next lever if
the serving A/B shows prefill regression.

## Finding 3 — the sym down fork keys `kind=""` while the static twins key `"fused"`

`_fork_shape_key`'s computed-A rebuild requires `S_ext_n_free_axis >= 2`; at symbolic M the fork has one static
free axis, so the same cone-bearing kernel keys `kind=""` on the sym twins and `kind="fused"` on the static ones.
Consequence: the dynM re-record stays on the existing `kernel: matmul` entries (key-compatible), but their STAGE
had to be left UNRECORDED — the fork only offers `d*/sync` fills, which the plain-matmul stage catalog cannot
spell (`test_golden_knobs_are_members_of_the_move_catalog`). The TILE+REDUCE prefix is unambiguous there, so
matching is safe; a `sync` spelling in the plain-matmul stage catalog (or a fused-kind dynamic key convention)
would remove both warts.

## Finding 4 — down-cone entries recorded under ShapeKey-equivalent `norm_linear` dims

The static down fork's form is `(gelu(ws[:, :15360]) * ws[:, 15360:]) @ W` — a stat-free GeGLU-combine cone. No
golden kind traces that form, but `NormLinearGoldenConfig(H=15360, N=3840)` produces the identical ShapeKey and
its fork enumerates the same computed-A rows, so the entries are recorded under those dims with µs measured on
the TRUE cone snippet (documented in the YAML comment). A dedicated combine-cone golden kind would make
`--golden` replays exact; follow-up if these entries ever need re-tuning.

## Workflow notes

- The single most time-consuming part was not benching but **key forensics**: confirming which snippet form
  reproduces each in-model fork's ShapeKey and row spelling took a monkeypatched `_golden_pick` spy over the twin
  compiles. An `emmy eval golden --in-model --rows <node>` view that dumps a fork's offered rows (knob families +
  a few sample rows) would have cut ~40 min.
- `run --bench --ab` with 3–4 pins per invocation was smooth; the split-total arithmetic (partial + finalize) is
  manual — a `total_us` per pin in the table (it IS in the `--json`) would remove a footgun.
- The audit probe (`audit_card` on the twins) takes ~2 min per card per iteration; fine for the 3 iterations this
  needed.
- 4090 follow-up: rent a 4090 (`start-remote-server` / tune-golden flow), re-run the same 14-key seeding there,
  then empty `_MERGED_EDGE_KEYS_4090` and restore its `MIN_MATCH` toward the healed count. The fm-lane twins for
  the merged edges (serving boots fm) are also unseeded on BOTH cards — fold them into the serving A/B session.
