"""The golden-spelling tripwire: every stored golden knob spelling is canonical.

Resolves every knob dict in ALL golden YAML files against its kernel kind's recognized tree
through the tree-path codec (:mod:`emmy.compiler.ir.tile.path`) and asserts the stored spelling
is the canonical (shortest-unique) one. The golden loader does not validate spellings, so this is
the only guard on the corpus: a structural change to a recognized tree that re-keys a canonical
site, or a hand-seeded entry with a stale/wrong-site spelling, would otherwise just stop matching
the stamped rows and the golden would silently never deploy. Documented exceptions, asserted
exactly:

- a DYNAMIC attention golden records ONE bare ``TILE`` naming its PV plan (the masked-flash
  golden form — schema-enforced in ``search/golden.py``); on the flash tree bare ``TILE`` is
  ambiguous by design (two contraction sites), and the golden layer matches it any-of.
- a bare key whose family has NO site on the kind's tree is the decided-empty stamp — its value
  must be the empty spelling.

A future structural change that breaks a stored short key fails here LOUDLY — the entry is then
re-spelled by hand, never silently re-keyed."""

from __future__ import annotations

import glob
import os

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp, SoftmaxOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.path import PATH_FAMILIES, canonical, resolve, sites
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import flatten_leaves
from emmy.compiler.pipeline.knob import family_of
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.golden_v2 import load_golden_file, load_golden_records

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "../../../../emmy/compiler/pipeline/search/goldens")


def _inp(g: Graph, name: str, shape: tuple, dt=F16) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, tuple(Dim(s) for s in shape), dtype=dt), node_id=name)


def _matmul() -> Graph:
    g = Graph()
    _inp(g, "x", (64, 128))
    _inp(g, "w", (128, 64))
    g.add_node(MatmulOp(), ["x", "w"], Tensor("y", (Dim(64), Dim(64)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "w"], ["y"]
    return g


def _rms_norm() -> Graph:
    g = Graph()
    _inp(g, "x", (64, 512))
    _inp(g, "wn", (512,))
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("y", (Dim(64), Dim(512)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "wn"], ["y"]
    return g


def _softmax() -> Graph:
    g = Graph()
    _inp(g, "x", (64, 512))
    g.add_node(SoftmaxOp(axis=-1), ["x"], Tensor("y", (Dim(64), Dim(512)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    return g


def _reduce() -> Graph:
    g = Graph()
    _inp(g, "x", (64, 512))
    g.add_node(ReduceOp(op="sum", axis=-1), ["x"], Tensor("y", (Dim(64), Dim(1)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    return g


def _pointwise() -> Graph:
    g = Graph()
    _inp(g, "x", (64, 512))
    g.add_node(ElementwiseOp("relu"), ["x"], Tensor("y", (Dim(64), Dim(512)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    return g


def _norm_linear() -> Graph:
    g = Graph()
    _inp(g, "x", (1, 32, 1024))
    _inp(g, "wn", (1024,))
    _inp(g, "w", (3072, 1024))
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (1, Dim(32), Dim(1024)), dtype=F16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "w"], Tensor("y", (1, Dim(32), Dim(3072)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "wn", "w"], ["y"]
    return g


def _mlp_geglu() -> Graph:
    g = Graph()
    _inp(g, "x", (1, 32, 1024))
    _inp(g, "wn", (1024,))
    _inp(g, "wg", (3072, 1024))
    _inp(g, "wu", (3072, 1024))
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (1, Dim(32), Dim(1024)), dtype=F16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("gate", (1, Dim(32), Dim(3072)), dtype=F16), node_id="gate")
    g.add_node(LinearOp(), ["xn", "wu"], Tensor("up", (1, Dim(32), Dim(3072)), dtype=F16), node_id="up")
    g.add_node(ElementwiseOp("silu"), ["gate"], Tensor("sg", (1, Dim(32), Dim(3072)), dtype=F16), node_id="sg")
    g.add_node(ElementwiseOp("multiply"), ["sg", "up"], Tensor("o", (1, Dim(32), Dim(3072)), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["x", "wn", "wg", "wu"], ["o"]
    return g


def _attention() -> Graph:
    pytest.importorskip("torch")
    from emmy.commands.trace import graph_from_code

    code = (
        "torch.nn.functional.scaled_dot_product_attention(torch.randn(1,4,128,128,dtype=torch.float16), "
        "torch.randn(1,4,128,128,dtype=torch.float16), torch.randn(1,4,128,128,dtype=torch.float16), is_causal=False)"
    )
    graph, _, _ = graph_from_code(code)
    return graph


def _tree(build, pick=None):
    """The recognized tile tree for one kernel kind — resolve the tile passes, take the (single)
    ``TileOp``'s stored op. ``pick`` selects a fork leaf (default option-0; the schedule slices a
    row stamps never change the walker's structure)."""

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        if pick is not None:
            for leaf in leaves:
                if pick(dict(getattr(leaf, "knobs", {}) or {})):
                    return leaf
        return leaves[0]

    rg, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(build(), decide)
    tiles = [n.op for n in rg.nodes.values() if isinstance(n.op, TileOp)]
    assert len(tiles) == 1, f"expected one kernel, got {len(tiles)}"
    return tiles[0].op


def _is_warp_row(row: dict) -> bool:
    # F1 site grammar: the tier discriminator is the row's ONE WORK entry, not an ``a:`` token.
    return str(row.get("WORK", "")).startswith("w")


@pytest.fixture(scope="module")
def trees():
    out = {
        "matmul": _tree(_matmul),
        "rms_norm": _tree(_rms_norm),
        "softmax": _tree(_softmax),
        "reduce": _tree(_reduce),
        "pointwise": _tree(_pointwise),
        # The fused kinds resolve against the CONTRACTION form — the tree the stored warp
        # spellings decorate (the map-form sibling is a different tree; its keys spell against
        # its own, where the stat fold is itself the primary).
        "norm_linear": _tree(_norm_linear, pick=_is_warp_row),
        "mlp_geglu": _tree(_mlp_geglu, pick=_is_warp_row),
        "attention": _tree(_attention),
    }
    return out


def _entries():
    files = sorted(glob.glob(os.path.join(_GOLDEN_DIR, "*.yaml")))
    assert files, f"no golden YAMLs under {_GOLDEN_DIR}"
    for f in files:
        doc = load_golden_file(f)
        for record in load_golden_records(doc):
            yield os.path.basename(f), record


def _tree_kind(record) -> str | None:
    """Derive the reference tree from the persisted frontend target.

    Golden v2 deliberately stores no kernel-class discriminator.  The current
    compiler category is an index over the stable origin operations, just like
    ShapeKey and the structural features.
    """
    origins = record.origin_ops
    if "torch.sdpa" in origins:
        return "attention"
    if record.shape_key.kind == "fused":
        contractions = sum(op in {"torch.linear", "torch.matmul"} for op in origins)
        return "mlp_geglu" if contractions > 1 else "norm_linear"
    if record.shape_key.kind in {"rms_norm", "softmax"}:
        return record.shape_key.kind
    if "tensor.reduce" in origins:
        return "reduce"
    if any(op in {"torch.linear", "torch.matmul"} for op in origins):
        return "matmul"
    if origins == ("tensor.elementwise",):
        return "pointwise"
    return None


def test_every_stored_golden_spelling_is_canonical(trees):
    checked = 0
    for fname, cfg in _entries():
        kind = _tree_kind(cfg)
        if kind not in trees:
            # rope / embedding entries carry no tree-path families — nothing to resolve.
            path_keys = [k for k in cfg.knobs if family_of(k) in PATH_FAMILIES]
            assert not path_keys, f"{fname}:{cfg.name}: derived kind {kind!r} has path-family keys but no reference tree"
            continue
        root = trees[kind]
        all_sites = sites(root)
        for key, value in cfg.knobs.items():
            if family_of(key) not in PATH_FAMILIES:
                continue
            ctx = f"{fname}:{cfg.name}: {key}={value!r}"
            if key == family_of(key):  # bare — the canonical spelling of the primary site
                if kind == "attention" and key == "TILE":
                    # The documented exception: a dynamic attention golden records ONE bare TILE
                    # (its PV plan) — ambiguous on the tree by design, matched any-of by the
                    # golden layer. Assert the exception holds exactly (dynamic entries only).
                    assert cfg.dynamic, f"{ctx}: a STATIC attention golden must spell TILE@dd/TILE@pj"
                    with pytest.raises(ValueError, match="ambiguous"):
                        resolve(root, key, all_sites=all_sites)
                    continue
                site = resolve(root, key, all_sites=all_sites)  # ambiguity would raise — the tripwire
                if site is None:
                    assert value in ("", None), f"{ctx}: no {key} site on the {kind} tree, yet a non-empty value"
                else:
                    assert canonical(root, key, all_sites=all_sites) == key, f"{ctx}: bare is not this site's canonical spelling"
            else:
                assert canonical(root, key, all_sites=all_sites) == key, f"{ctx}: stored spelling is not canonical"
            checked += 1
    assert checked > 1000, f"suspiciously few golden keys checked ({checked}) — did the loader change?"
