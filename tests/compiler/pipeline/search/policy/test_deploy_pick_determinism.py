"""Deploy-pick determinism — the fork RESOLUTION twin of
``tests/compiler/backend/test_source_determinism.py`` (which pins the rendered bytes).

A deploy pick must be a function of the candidates' CONTENT, never of their
enumeration order: the offline prior can score many same-featurized siblings
identically (8 exact ties at the gemma-4 m16 mlp_down / o_proj forks, where the
recorded goldens don't realize against the serving trace), and an order-broken tie
flips the deployed kernel whenever leaf order shifts across processes — the 2026-07
RTX 5090 gemma-4 image's bimodal boot-time cubin set. Two layers of pinning:

1. Unit: every pick tier (model argmin, measured-evidence argmin, golden
   realization) selects the same CONTENT under any permutation of the candidate rows.
2. Subprocess: two fresh-interpreter greedy compiles of a serving-shape graph (fresh
   hash seed + address space, CUDA hidden, deployable -O3 regime, 5090 golden card)
   produce the identical SELECTED kernel set — knobs and source bytes.
"""

from __future__ import annotations

import subprocess
import sys

from emmy.compiler.pipeline.search.policy import greedy as greedy_mod
from emmy.compiler.pipeline.search.prior.base import Prior


class _ConstPrior(Prior):
    """Every candidate scores identically — the pick must fall to the canonical
    content tiebreak, so any order dependence shows immediately."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def fitted(self) -> bool:
        return True

    def fit(self) -> None:  # pragma: no cover — never trained
        pass

    def score(self, knobs: dict) -> float:
        return 1.0

    def mean_score(self, knobs: dict) -> float:
        return 1.0


def _rows() -> list[dict]:
    base = {"H_opt": 3.0, "S_ext_k": "512"}
    # SITE-LOCAL ``TILE`` values (the worker widths ride the row's one ``WORK`` entry) — the same
    # spelling a stamped row carries, so the tiebreak is exercised over live content.
    tiles = [
        "mma_m16n8k16_f16_f32/f2x4",
        "mma_m16n8k16_f16_f32/f4x2",
        "mma_m16n8k16_f16_f32/f2x4/k2",
        "mma_m16n8k16_f16_f32/f4x2/k2",
    ]
    return [{**base, "TILE@k": t, "REDUCE@k": r, "STAGE@k": "", "RASTER": "", "WORK": "w1x1"} for t in tiles for r in ("", "g2k")]


def _selected(pick: tuple[int, float] | None, rows: list[dict]):
    assert pick is not None
    return {k: v for k, v in rows[pick[0]].items() if not k.startswith(("S_", "H_"))}


def test_model_argmin_is_order_invariant():
    prior = _ConstPrior()
    rows = _rows()
    want = _selected(prior.pick(rows), rows)
    for perm in (rows[::-1], rows[3:] + rows[:3]):
        assert _selected(prior.pick(perm), perm) == want


def test_evidence_pick_tie_is_order_invariant():
    prior = _ConstPrior()
    rows = _rows()
    # One measured row that matches EVERY candidate (records no tunable knob) — all
    # candidates tie at its µs, so only the canonical tiebreak can decide.
    prior.add_rows([({"H_opt": 3.0, "S_ext_k": "512"}, 12.5)])
    want = _selected(prior.evidence_pick(rows), rows)
    assert _selected(prior.evidence_pick(rows[::-1]), rows[::-1]) == want


def test_db_measured_pick_tie_is_order_invariant():
    rows = _rows()
    sig = frozenset({("S_ext_k", "512")})
    index = {sig: [({}, 33.0, True)]}  # matches every candidate at one µs — a full tie
    rows_fwd, rows_rev = rows, rows[::-1]
    a = _selected(greedy_mod._db_measured_pick(index, rows_fwd), rows_fwd)
    b = _selected(greedy_mod._db_measured_pick(index, rows_rev), rows_rev)
    assert a == b


# --- subprocess pin: the SELECTED kernel set across fresh interpreters -------------

_SNIPPET = """
import hashlib
import os

os.environ["EMMY_ONLINE_FILE"] = "/nonexistent/online.json"
os.environ["EMMY_TUNE_DB"] = "/nonexistent/autotune.db"
os.environ["EMMY_NVCC_FLAGS"] = ""  # deployable -O3 regime — the golden tier's guard

from emmy.compiler import dtype as _dt
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp, RmsNormOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.knob import tuning_knob_items

F16 = _dt.get("f16")

# The gemma-4-12B decode-twin shapes whose deploy flapped on the 5090 image: the m16
# mlp_down matmul (8-way offline-prior tie behind a drifted golden) and the fused
# norm->gate/up->GeGLU cone (the multi-channel computed-A fork from #389).
def mlp_down(m=16, n=3840, k=15360):
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (1, m, k), F16), node_id="a")
    g.add_node(InputOp(), [], Tensor("w", (n, k), F16), node_id="w")
    g.add_node(LinearOp(), ["a", "w"], Tensor("o", (1, m, n), F16), node_id="o")
    g.inputs, g.outputs = ["a", "w"], ["o"]
    return g

def geglu(m=16, h=3840, inter=15360):
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, m, h), F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (h,), F16), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (inter, h), F16), node_id="wg")
    g.add_node(InputOp(), [], Tensor("wu", (inter, h), F16), node_id="wu")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, m, h), F16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("gate", (1, m, inter), F16), node_id="gate")
    g.add_node(LinearOp(), ["xn", "wu"], Tensor("up", (1, m, inter), F16), node_id="up")
    g.add_node(ElementwiseOp("gelu"), ["gate"], Tensor("sg", (1, m, inter), F16), node_id="sg")
    g.add_node(ElementwiseOp("multiply"), ["sg", "up"], Tensor("o", (1, m, inter), F16), node_id="o")
    g.inputs, g.outputs = ["x", "nw", "wg", "wu"], ["o"]
    return g

ctx = Context.from_target((12, 0), gpu_name="NVIDIA GeForce RTX 5090")
pipe = Pipeline.build(CUDA_PASSES)
for name, g in (("mlp_down.m16", mlp_down()), ("mlp_geglu.m16", geglu())):
    terminal = pipe.run(g, ctx=ctx)
    for nid, node in sorted(terminal.nodes.items()):
        src = getattr(node.op, "kernel_source", None)
        if not src:
            continue
        knobs = ",".join(f"{k}={v}" for k, v in tuning_knob_items(getattr(node.op, "knobs", {}) or {}))
        print(name, nid, hashlib.sha1(src.encode()).hexdigest(), f"[{knobs}]")
"""


def _resolve_once(tag: str) -> str:
    import os

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("PYTHONHASHSEED", None)  # each subprocess gets its own hash seed — the point
    out = subprocess.run([sys.executable, "-c", _SNIPPET], capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, f"resolve {tag} failed: {out.stderr[-800:]}"
    return out.stdout


def test_selected_kernel_set_identical_across_processes():
    a = _resolve_once("a")
    b = _resolve_once("b")
    assert a.strip(), "no kernels selected — the serving-shape resolve is broken, repick the graphs"
    assert a == b, f"selected kernel set differs across processes (boot-to-boot deploy flap):\n--- a\n{a}\n--- b\n{b}"
