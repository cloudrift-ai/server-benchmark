"""Kernel-source digest harness — the byte-identity gate for the tile-IR migration steps.

Compiles a battery of representative kernels OFF-GPU (sources render without a device, target
sm_120) at a pinned knob config each, and prints ``case/kernel_name sha1`` per rendered kernel
source. Run at the baseline commit and after each migration commit; diff the outputs — a
byte-neutral storage migration must leave every digest unchanged. Covers the scalar/warp/f16acc/
split-K/raster matmuls and the producer band, the coop-t matvec, dynM forms, the reduce tiers
(rms/softmax/reduce/pointwise), the computed-A fused kinds (norm_linear canonical + .lin +
split-K + coop, mlp_geglu) and flash (hd128 tma/cp, hd256 split, fm sibling, chain, scalar).

Every case additionally asserts LIVENESS — that its pins reached a kernel (see :func:`_liveness`).
A digest alone cannot tell a covered path from an unscheduled one: both render, both hash stably.

Usage: venv/bin/python scripts/digest_kernels.py [case ...] > digests.txt
"""

from __future__ import annotations

import hashlib
import os
import sys
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from emmy.compiler.context import Context  # noqa: E402
from emmy.compiler.dim import Dim  # noqa: E402
from emmy.compiler.dtype import F16  # noqa: E402
from emmy.compiler.graph import Graph, Tensor  # noqa: E402
from emmy.compiler.ir.base import InputOp  # noqa: E402
from emmy.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp, SoftmaxOp  # noqa: E402
from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp  # noqa: E402
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline  # noqa: E402
from emmy.compiler.pipeline.fork import flatten_leaves  # noqa: E402
from emmy.compiler.pipeline.pipeline import Run  # noqa: E402

WARP = "mma_m16n8k16_f16_f32"  # the site TILE value's leading atom token; the warps live in WORK


def _inp(g: Graph, name: str, shape: tuple, dt=F16) -> str:
    return g.add_node(
        op=InputOp(), inputs=[], output=Tensor(name, tuple(Dim(s) if not isinstance(s, Dim) else s for s in shape), dtype=dt), node_id=name
    )


def matmul(M=512, N=512, K=512, lin=False):
    g = Graph()
    _inp(g, "x", (M, K))
    _inp(g, "w", (N, K) if lin else (K, N))
    op = LinearOp() if lin else MatmulOp()
    g.add_node(op=op, inputs=["x", "w"], output=Tensor("y", (Dim(M) if not isinstance(M, Dim) else M, Dim(N)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "w"], ["y"]
    return g


def norm_linear(S=32, H=1024, inter=3072, lin=False):
    g = Graph()
    Sd = S if isinstance(S, Dim) else Dim(S)
    _inp(g, "x", (1, Sd, H))
    _inp(g, "wn", (H,))
    _inp(g, "w", (inter, H) if lin else (H, inter))
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (1, Sd, Dim(H)), dtype=F16), node_id="xn")
    g.add_node(LinearOp() if lin else MatmulOp(), ["xn", "w"], Tensor("y", (1, Sd, Dim(inter)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "wn", "w"], ["y"]
    return g


def mlp_geglu(S=32, H=1024, inter=3072):
    g = Graph()
    _inp(g, "x", (1, S, H))
    _inp(g, "wn", (H,))
    _inp(g, "wg", (inter, H))
    _inp(g, "wu", (inter, H))
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (1, Dim(S), Dim(H)), dtype=F16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("gate", (1, Dim(S), Dim(inter)), dtype=F16), node_id="gate")
    g.add_node(LinearOp(), ["xn", "wu"], Tensor("up", (1, Dim(S), Dim(inter)), dtype=F16), node_id="up")
    g.add_node(ElementwiseOp("silu"), ["gate"], Tensor("sg", (1, Dim(S), Dim(inter)), dtype=F16), node_id="sg")
    g.add_node(ElementwiseOp("multiply"), ["sg", "up"], Tensor("o", (1, Dim(S), Dim(inter)), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["x", "wn", "wg", "wu"], ["o"]
    return g


def rms_norm(S=64, H=4096):
    g = Graph()
    _inp(g, "x", (S, H))
    _inp(g, "wn", (H,))
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("y", (Dim(S), Dim(H)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "wn"], ["y"]
    return g


def softmax(S=64, H=4096):
    g = Graph()
    _inp(g, "x", (S, H))
    g.add_node(SoftmaxOp(axis=-1), ["x"], Tensor("y", (Dim(S), Dim(H)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    return g


def reduce_sum(S=64, H=4096):
    g = Graph()
    _inp(g, "x", (S, H))
    g.add_node(ReduceOp(op="sum", axis=-1), ["x"], Tensor("y", (Dim(S), Dim(1)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    return g


def pointwise(S=64, H=4096):
    g = Graph()
    _inp(g, "x", (S, H))
    g.add_node(ElementwiseOp("relu"), ["x"], Tensor("y", (Dim(S), Dim(H)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    return g


def sdpa(head_dim=128, seq=128, heads=4):
    import torch  # noqa: F401

    from emmy.commands.trace import graph_from_code

    code = (
        f"torch.nn.functional.scaled_dot_product_attention(torch.randn(1,{heads},{seq},{head_dim},dtype=torch.float16), "
        f"torch.randn(1,{heads},{seq},{head_dim},dtype=torch.float16), torch.randn(1,{heads},{seq},{head_dim},dtype=torch.float16), "
        "is_causal=False)"
    )
    graph, _, _ = graph_from_code(code)
    return graph


CASES = [
    ("matmul_scalar", lambda: matmul(), {"TILE": "f4x8", "WORK": "t16x8"}),
    ("matmul_warp_tma", lambda: matmul(), {"TILE": f"{WARP}/f4x1/k4", "WORK": "w1x8", "STAGE": "d3/tma"}),
    (
        "matmul_warp_f16acc",
        lambda: matmul(2048, 2048, 2048),
        {"TILE": "mma_m16n8k16_f16_f16/f4x8/k4", "WORK": "w4x2", "STAGE": "d2/tma"},
    ),
    ("matmul_splitk", lambda: matmul(), {"TILE": f"{WARP}/f2x4/k2", "WORK": "w2x2", "REDUCE": "g2k", "STAGE": "d2/tma"}),
    (
        "matmul_raster",
        lambda: matmul(2048, 2048, 2048),
        {"TILE": f"{WARP}/f2x2/k4", "WORK": "w2x4", "STAGE": "d2/tma", "RASTER": "gm8"},
    ),
    # The producer band is inventory: it rides WORK's ``+p`` suffix, never the retired WSPEC key.
    (
        "matmul_pband",
        lambda: matmul(2048, 2048, 2048),
        {"TILE": f"{WARP}/f2x2/k4", "WORK": "w2x4+p1", "STAGE": "d2/tma", "REDUCE": ""},
    ),
    ("matvec_coopt", lambda: matmul(1, 4096, 4096, lin=True), {"REDUCE": "g16k/coop-t", "WORK": "t256"}),
    ("matmul_dynm", lambda: matmul(Dim("seq_len"), 512, 512), {"TILE": f"{WARP}/f4x1/k4", "WORK": "w1x8"}),
    ("rms_norm", lambda: rms_norm(), {"REDUCE": "coop", "WORK": "t256"}),
    ("softmax", lambda: softmax(), {"REDUCE": "coop", "WORK": "t128"}),
    ("reduce", lambda: reduce_sum(), {"REDUCE": "coop", "WORK": "t128"}),
    ("pointwise", lambda: pointwise(), {"TILE": "f2"}),
    ("norm_linear", lambda: norm_linear(), {"TILE": f"{WARP}/f2x2/k2", "WORK": "w1x16", "REDUCE": ""}),
    ("norm_linear_lin", lambda: norm_linear(lin=True), {"TILE": f"{WARP}/f2x2/k2", "WORK": "w1x16", "REDUCE": ""}),
    ("norm_linear_splitk", lambda: norm_linear(), {"TILE": f"{WARP}/f2x2/k2", "WORK": "w1x16", "REDUCE": "g8k"}),
    ("norm_linear_coop", lambda: norm_linear(), {"REDUCE": "coop", "WORK": "t128"}),
    ("norm_linear_dynm", lambda: norm_linear(S=Dim("seq_len")), {"TILE": f"{WARP}/f2x2/k2", "WORK": "w1x16", "REDUCE": ""}),
    ("mlp_geglu", lambda: mlp_geglu(), {"TILE": f"{WARP}/f4x8", "WORK": "w2x2", "REDUCE": ""}),
    ("flash_hd128", lambda: sdpa(128), {"TILE@dd": f"{WARP}/f1x2/k8", "WORK": "w4x1", "TILE@pj": f"{WARP}/f1x16", "STAGE": "d2/tma"}),
    (
        "flash_hd128_cp",
        lambda: sdpa(128),
        {"TILE@dd": f"{WARP}/f1x2/k8", "WORK": "w4x1", "TILE@pj": f"{WARP}/f1x16", "STAGE": "d2/cp"},
    ),
    (
        "flash_hd256_alt",
        lambda: sdpa(256),
        {"TILE@dd": f"{WARP}/f1x8/k16", "WORK": "w4x1", "TILE@pj": f"{WARP}/f1x32/k4", "STAGE": "d1/cp/split"},
    ),
    (
        "flash_hd256_fm",
        lambda: sdpa(256),
        {"TILE@dd": f"{WARP}/f1x8/k16", "TILE@pj": "mma_m16n8k16_f16_f16/f1x32/k4", "WORK": "w4x1", "STAGE": "d1/cp/split"},
    ),
    ("flash_chain", lambda: sdpa(64), {"TILE": "a:scalar", "TILE@pj": "f64"}),
    ("flash_scalar", lambda: sdpa(64), {"REDUCE": "coop", "WORK": "t128"}),
]


# The cases whose pins reach NO kernel today, each a consequence of the ONE enumerator gap left: the
# flash streaming pair emits no schedule rows, so ``020_schedule`` leaves those terms unmapped and the
# case digests the un-scheduled lowering path instead. Recorded, not tolerated — like the xfail
# registry this set is STRICT: a case that starts landing its pins FAILS here until it is deleted, so
# the remaining phase reports itself.
UNSCHEDULED = frozenset(
    {
        "flash_hd128",
        "flash_hd128_cp",
        "flash_hd256_alt",
        "flash_hd256_fm",
        "flash_chain",
    }
)


def _liveness(name, pins, realized):
    """Did this case's pins reach a kernel? Returns a message when the answer is not the expected one.

    A digest is blind to the difference: an unmapped term still renders and still hashes stably, so
    without this the baseline would pin recognition, term storage and the un-scheduled lowering path
    while leaving the tiered/placed contraction path — the one the pins name — entirely uncovered.
    The pins must land TOGETHER on one kernel; under split-K that kernel is the ``__partial``.

    A pin here is spelled BARE, and a bare pin fans out to every eligible site — so it lands when
    SOME same-family realized key carries its value (``knob.pin_key_matches`` / ``values_equal``,
    the same reading the golden matcher and the replay gate use). A fused kernel spells its
    contraction's K fold bare and the cone's statistic at ``REDUCE@<axis>``; requiring the bare key
    to carry the value would call that miss a drop.
    """
    from emmy.compiler.pipeline.knob import family_of, pin_key_matches, values_equal

    def lands(knobs, pin, want):
        hits = [(k, v) for k, v in knobs.items() if family_of(k) == family_of(pin) and pin_key_matches(pin, k)]
        return any(values_equal(k, want, v) for k, v in hits)

    landed = [kname for kname, knobs in realized if all(lands(knobs, f, v) for f, v in pins.items())]
    if name in UNSCHEDULED:
        return None if not landed else f"{name}: pins now land on {landed[0]} — delete it from UNSCHEDULED"
    if landed:
        return None
    got = {kname: {f: knobs.get(f) for f in pins} for kname, knobs in realized}
    return f"{name}: no kernel stamped {pins}; realized {got}"


def run_case(name, build, pins):
    from emmy.commands.run import _pinned_knobs

    g = build()

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        return leaves[0]

    with _pinned_knobs(pins):
        out, _ = Run(pipeline=Pipeline.build(CUDA_PASSES), ctx=Context.from_target((12, 0))).resolve(g, decide)
    lines, realized = [], []
    for nid, node in sorted(out.nodes.items()):
        src = getattr(node.op, "kernel_source", None)
        if src:
            kname = getattr(node.op, "kernel_name", nid)
            lines.append(f"{name}/{kname} {hashlib.sha1(src.encode()).hexdigest()}")
            realized.append((kname, getattr(node.op, "knobs", None) or {}))
    if not lines:
        lines.append(f"{name}/<no-kernel-source> -")
    return lines, _liveness(name, pins, realized)


BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_digests.txt")


def _check(lines: list[str]) -> int:
    """Diff the rendered digests against the committed baseline. Reports every drifted, missing and
    unexpected kernel, and returns a nonzero exit when any differ."""
    with open(BASELINE) as f:
        want = dict(ln.split() for ln in f.read().splitlines() if ln.strip())
    got = dict(ln.split() for ln in lines)
    drift = [(k, want[k], got[k]) for k in sorted(want.keys() & got.keys()) if want[k] != got[k]]
    for k, w, g in drift:
        print(f"DRIFT {k}\n  baseline {w}\n  rendered {g}")
    for k in sorted(want.keys() - got.keys()):
        print(f"MISSING {k} (baseline {want[k]})")
    for k in sorted(got.keys() - want.keys()):
        print(f"UNEXPECTED {k} ({got[k]})")
    ok = not drift and want.keys() == got.keys()
    print("digests match the baseline" if ok else f"{len(drift)} drifted, {len(want.keys() ^ got.keys())} added/removed")
    return 0 if ok else 1


def main():
    argv = sys.argv[1:]
    check = "--check" in argv
    only = [a for a in argv if not a.startswith("-")] or None
    failures, lines, dead = 0, [], []
    for name, build, pins in CASES:
        if only and name not in only:
            continue
        try:
            case_lines, verdict = run_case(name, build, pins)
            lines.extend(case_lines)
            if verdict:
                dead.append(verdict)
        except Exception as e:  # noqa: BLE001
            failures += 1
            lines.append(f"{name}/<ERROR> {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stderr)
    if not check:
        for line in lines:
            print(line)
    sys.stdout.flush()
    for verdict in dead:
        print(f"LIVENESS {verdict}", file=sys.stderr)
    if failures or dead:
        return 1
    return _check(lines) if check else 0


if __name__ == "__main__":
    sys.exit(main())
