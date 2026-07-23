"""Accumulation-error sweep: FP32 vs pure-FP16 vs hybrid (chunk-64 promote) matmul accumulation.

Measures the relative L2 error of ``C = A @ B`` (FP16 inputs drawn from N(0,1)) against an FP64
reference computed over the *identical* FP16-rounded operands — so the sweep isolates
*accumulation* error, not input rounding. Three strategies, each modeled as an element-sequential
K-chain (the tensor-core accumulate order):

- ``fp32``   — every partial-product add lands in an FP32 accumulator (the accurate default).
- ``fp16``   — the naive fast path: the running sum itself is rounded to FP16 on every one of the
  K adds, so the error grows with K.
- ``hybrid`` — emmy's FAST_MATH scheme (``emmy_mma_promote_f16acc``): FP16 accumulation within a
  64-element chunk, each chunk folded into an FP32 shadow and rezeroed, so FP16 error can never
  accumulate past one chunk.

This is the generator behind the accumulation-error table in the gemma-4-12B blog article.

    python scripts/bench_accum_error.py --out-dir _tune/accum-error
    python scripts/bench_accum_error.py --ks 256,3840,4096,15360,32768 --mn 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ATOM_K = 16  # the m16n8k16 HMMA's K — one internal full-precision dot, one rounding
PROMOTE_CADENCE = 64  # the K-slab / promote stride the generated kernels use


def _accum_error(k: int, mn: int, device: str, seed: int) -> dict[str, float]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn((mn, k), generator=g).half().to(device)  # FP16-rounded operands are the
    b = torch.randn((k, mn), generator=g).half().to(device)  # ground truth's inputs too
    ref = a.double() @ b.double()

    # One m16n8k16 HMMA computes a 16-element dot product internally in full precision and
    # rounds ONCE into the accumulator — so the rounding cadence is per-atom (ATOM_K), not
    # per element, and each strategy differs only in its running-sum dtype: fp32 keeps the
    # sum in fp32 (the accurate default), fp16 rounds the fused add to fp16 every atom, and
    # hybrid additionally folds the fp16 partial into an fp32 shadow every PROMOTE_CADENCE
    # elements (= 4 atoms), rezeroing it.
    acc32 = torch.zeros((mn, mn), dtype=torch.float32, device=device)
    acc16 = torch.zeros((mn, mn), dtype=torch.float16, device=device)
    hyb32 = torch.zeros((mn, mn), dtype=torch.float32, device=device)
    hyb16 = torch.zeros((mn, mn), dtype=torch.float16, device=device)
    for kk in range(0, k, ATOM_K):
        p32 = a[:, kk : kk + ATOM_K].float() @ b[kk : kk + ATOM_K, :].float()  # one atom's dot
        acc32 += p32
        acc16 = (acc16.float() + p32).half()  # one fp16 rounding per atom
        hyb16 = (hyb16.float() + p32).half()
        if (kk + ATOM_K) % PROMOTE_CADENCE == 0 or kk + ATOM_K >= k:
            hyb32 += hyb16.float()
            hyb16 = torch.zeros_like(hyb16)

    def rel_l2(c: torch.Tensor) -> float:
        return ((c.double() - ref).norm() / ref.norm()).item()

    return {"fp32": rel_l2(acc32), "fp16": rel_l2(acc16), "hybrid": rel_l2(hyb32)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ks", default="256,3840,4096,15360,32768", help="comma-separated accumulation depths")
    ap.add_argument("--mn", type=int, default=256, help="square M=N of the test matmul")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="_tune/accum-error", help="where the .md/.json land")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ks = [int(s) for s in args.ks.split(",")]
    rows = {k: _accum_error(k, args.mn, device, args.seed) for k in ks}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Accumulation error sweep — M=N={args.mn}, promote cadence {PROMOTE_CADENCE}, seed {args.seed}",
        "",
        "| K (accumulation depth) | FP32-accum | FP16-accum | Hybrid |",
        "|--:|--:|--:|--:|",
    ]
    for k, r in rows.items():
        lines.append(f"| {k} | {r['fp32']:.1e} | {r['fp16']:.1e} | {r['hybrid']:.1e} |")
    (out_dir / "accum_error.md").write_text("\n".join(lines) + "\n")
    (out_dir / "accum_error.json").write_text(json.dumps({"mn": args.mn, "seed": args.seed, "rows": rows}, indent=2))
    print("\n".join(lines))
    print(f"\ntable → {out_dir / 'accum_error.md'}\n json → {out_dir / 'accum_error.json'}")


if __name__ == "__main__":
    main()
