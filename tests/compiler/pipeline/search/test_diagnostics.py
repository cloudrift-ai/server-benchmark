"""Diagnostics join keys — every golden ↔ measured-data join goes through
``ShapeKey`` (``from_matmul`` on the golden side, ``from_s_features`` on the op
side) with the ``_matmul_sig`` histogram gate.

The load-bearing regression here is the dead ``S_n_mma`` marker: the stamp pass
runs at fusion end, before the tile tier emits ``Mma`` stmts, so ``S_n_mma`` is
0.0 on every stamped row. Gating on it made ``_golden_coverage`` permanently
empty, mislabeled every matmul in ``reachability``, and silently dropped every
fp16 golden from ``golden_prior_eval``'s rank join (the hidden fp16 lockout).
"""

from __future__ import annotations

from emmy.compiler.pipeline.search.golden import goldens_by_name
from emmy.compiler.pipeline.search.prior import diagnostics


def _sig(free_prod, reduce_max, *, fp32=True, matmul=True):
    """An op-group signature (sorted ``S_*`` items) as ``Dataset.group_by_op``
    keys it — a matmul histogram (product → reduce-add, 2 distinct inputs) or a
    reduce-shaped one. ``S_n_mma`` is stamped 0.0, as on every real row."""
    d = {
        "S_ext_free_prod": float(free_prod),
        # matmul fixtures here are SQUARE (free_max = sqrt); a reduce has one free dim (= free_prod)
        "S_ext_free_max": float(round(free_prod**0.5)) if matmul else float(free_prod),
        "S_ext_reduce_max": float(reduce_max),
        "S_n_mma": 0.0,
        "S_reduce_add": 1.0,
        "S_n_distinct_input": 2.0 if matmul else 1.0,
        ("S_dtype_f32" if fp32 else "S_dtype_f16"): 2.0,
    }
    if matmul:
        d["S_pw_multiply"] = 1.0
    return tuple(sorted(d.items()))


def test_op_label_recognizes_stamped_matmuls():
    """Matmuls label as ``matmul`` from the histogram (S_n_mma is always 0.0);
    a plain reduce keeps its label."""
    assert diagnostics._op_label(_sig(512 * 512, 512)).startswith("matmul")
    assert diagnostics._op_label(_sig(512, 4096, matmul=False)).startswith("reduce")


def test_golden_coverage_counts_dtype_twins_separately():
    """Coverage joins on ShapeKey: an fp32 group at square.512's extents covers the
    fp32 golden only — its ``.fp16`` twin needs its own group; a reduce-shaped group
    at the same extents counts nothing. (The old ``S_n_mma > 0`` gate made coverage
    permanently 0/N.)"""
    fp = 512 * 512
    base = diagnostics._golden_coverage({})[0]
    assert base == 0
    only_fp32 = diagnostics._golden_coverage({_sig(fp, 512): []})[0]
    assert only_fp32 == 1  # square.512, not square.512.fp16
    both = diagnostics._golden_coverage({_sig(fp, 512): [], _sig(fp, 512, fp32=False): []})[0]
    assert both == 2
    collider = diagnostics._golden_coverage({_sig(fp, 512, matmul=False): []})[0]
    assert collider == 0  # non-matmul group at matmul extents doesn't count


def test_golden_coverage_splits_dynamic_twin(monkeypatch):
    """A ``.dynM`` golden is covered only by a symbolic-marked op group
    (``S_ext_n_symbolic_axis > 0``, ``free_prod`` excluding the symbolic axis —
    what the 992 stamp emits for a masked-tile kernel); its static twin's group
    does not satisfy it, and vice versa."""
    from emmy.compiler.pipeline.search import golden as gmod

    dyn_g = gmod.MatmulGoldenConfig(
        name="matmul.square.512.dynM",
        M=512,
        N=512,
        K=512,
        knobs={"BM": 8},
        emmy_us=10.0,
        cublas_us=11.0,
        dynamic=True,
    )
    monkeypatch.setattr(gmod, "GOLDEN_CONFIGS", [dyn_g])

    static_group = _sig(512 * 512, 512)  # the static twin's stamped extents
    assert diagnostics._golden_coverage({static_group: []}) == (0, 1)

    sym = dict(_sig(512, 512))  # free_prod = N only; symbolic axis flagged
    sym["S_ext_n_symbolic_axis"] = 1.0
    assert diagnostics._golden_coverage({tuple(sorted(sym.items())): []}) == (1, 1)


class _FakePrior:
    """Constant-score prior with a hand-built reservoir (``golden_prior_eval``
    only needs ``_dataset`` for the op-group index and ``mean_score``)."""

    def __init__(self, rows):
        self._dataset = rows

    def mean_score(self, _feats):
        return 0.0


def test_golden_prior_eval_joins_fp16_goldens():
    """An fp16-tuned op group must rank its ``.fp16`` golden (under the old
    ``S_n_mma``-keyed join the op side was always ``False``, so fp16 goldens
    silently dropped while fp32 goldens could steal the fp16 group's histogram)."""
    g16 = goldens_by_name("matmul.square.512.fp16")[0]
    rows = [({**dict(_sig(g16.M * g16.N, g16.K, fp32=False)), "WM": 4}, 5.0)]
    out = diagnostics.golden_prior_eval(_FakePrior(rows), kernel_filter="matmul.square.512")
    assert "matmul.square.512.fp16" in out and "rank" in out
    # The fp32 square.512 has no tuned group here → a per-shape SKIPPED line, not silence.
    skip_lines = [line for line in out.splitlines() if "SKIPPED" in line]
    assert any(line.strip().startswith("matmul.square.512 ") and "no tuned rows" in line for line in skip_lines)


def test_golden_prior_eval_warns_per_unjoinable_shape():
    """Shapes with no tuned rows must each print a SKIPPED line — the silent drop
    hid the fp16 lockout in the 2026-06-12 sweep."""
    out = diagnostics.golden_prior_eval(_FakePrior([]), kernel_filter="matmul.square.512")
    skip_lines = [line for line in out.splitlines() if "SKIPPED" in line and "no tuned rows" in line]
    # square.512 and its .fp16 twin (one line per recorded config name, deduped by name set)
    assert {line.split()[0] for line in skip_lines} >= {"matmul.square.512", "matmul.square.512.fp16"}


# ---------------------------------------------------------------------------
# node-store fork sibling value regret
# ---------------------------------------------------------------------------


class _BMPrior:
    """A prior whose predicted latency is ``sign * features['BM']`` — so ``sign=+1``
    predicts smaller BM as faster, ``sign=-1`` reverses the order. ``BM`` is an
    unregistered numeric knob, so ``knob_features`` passes it through by name and
    the same read works on raw knob dicts and featurized rows alike."""

    fitted = True

    def __init__(self, sign: float = 1.0):
        self._sign = sign

    def mean_score(self, feats) -> float:
        return self._sign * float(feats.get("BM", 0.0))

    def mean_scores_features(self, feats_list) -> list[float]:
        return [self.mean_score(f) for f in feats_list]


class _FlatPrior:
    """A constant-score prior — every child ties, exercising the pessimistic tie rule."""

    fitted = True

    def mean_score(self, _feats) -> float:
        return 1.0

    def mean_scores_features(self, feats_list) -> list[float]:
        return [1.0] * len(feats_list)


def _child(node_key, bm, value_us, *, parent="P"):
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    return NodeRow(node_key, parent, "ctx", "mm", {"BM": bm}, value_us, 2)


def _fork_nodes():
    """A root P with three children whose value-of-position falls as BM falls."""
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    return [NodeRow("P", None, "ctx", "mm", {}, 1.0, 1), _child("c8", 8, 1.0), _child("c32", 32, 2.0), _child("c64", 64, 3.0)]


def test_sibling_regret_perfect_prior_scores_one():
    """A prior that orders the fork's children by value-of-position picks the true-best
    subtree: regret 1.00; the fork buckets under the family its children decide (BM)."""
    recs = diagnostics.sibling_regret(_BMPrior(), _fork_nodes())
    assert len(recs) == 1
    _op, family, regret, n_children = recs[0]
    assert (family, regret, n_children) == ("BM", 1.0, 3)


def test_sibling_regret_prices_a_reversed_prior():
    """A prior that ranks the children backwards pays the value gap, not a rank: its
    pick is the slowest subtree (3.0 µs, fork best 1.0), so regret = 3.00x."""
    recs = diagnostics.sibling_regret(_BMPrior(sign=-1.0), _fork_nodes())
    assert recs[0][2] == 3.0


def test_sibling_regret_ties_resolve_pessimistically():
    """A constant-score prior ties every child; a tie is a miss (greedy breaks score
    ties by emission order — the ``rank_of_golden`` ``>=`` rule), so the pick prices
    as the WORST tied value: 3.00x, not 1.00x."""
    recs = diagnostics.sibling_regret(_FlatPrior(), _fork_nodes())
    assert recs[0][2] == 3.0


def test_sibling_regret_ignores_singletons_and_top_forks():
    """A top fork (parent None) and a single-child parent are not multi-child forks,
    so there is nothing to price."""
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    nodes = [NodeRow("P", None, "ctx", "mm", {}, 1.0, 1), _child("only", 8, 1.0)]
    assert diagnostics.sibling_regret(_BMPrior(), nodes) == []


def test_sibling_regret_buckets_by_family_from_parent_delta():
    """Fork family = the knob keys the children pin beyond their parent row, mapped
    through ``family_of`` (``TILE@d`` → ``TILE``) — a TILE fork and a REDUCE fork under
    one op land in their own buckets, keyed in the current axis-named vocabulary."""
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    nodes = [
        NodeRow("Pt", None, "ctx", "mm", {"REDUCE@d": "b64"}, 1.0, 1),
        NodeRow("t1", "Pt", "ctx", "mm", {"REDUCE@d": "b64", "TILE@d": "n8x8/f2x2"}, 1.0, 2),
        NodeRow("t2", "Pt", "ctx", "mm", {"REDUCE@d": "b64", "TILE@d": "n16x16/f4x4"}, 2.0, 2),
        NodeRow("Pr", None, "ctx", "mm", {}, 1.0, 1),
        NodeRow("r1", "Pr", "ctx", "mm", {"REDUCE@d": "b64"}, 1.0, 2),
        NodeRow("r2", "Pr", "ctx", "mm", {"REDUCE@d": "g2k/b64"}, 2.0, 2),
    ]
    fams = sorted(fam for _, fam, _, _ in diagnostics.sibling_regret(_FlatPrior(), nodes))
    assert fams == ["REDUCE", "TILE"]


def test_sibling_regret_family_falls_back_without_parent_row():
    """When the parent row isn't in the store (cross-session gap), the family comes from
    the keys whose values differ ACROSS the siblings — a shared identical key (the
    inherited ``REDUCE@d``) doesn't count."""
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    nodes = [
        NodeRow("s1", "GONE", "ctx", "mm", {"REDUCE@d": "b64", "STAGE@d": "d2/cp"}, 1.0, 5),
        NodeRow("s2", "GONE", "ctx", "mm", {"REDUCE@d": "b64", "STAGE@d": "d3/tma"}, 2.0, 5),
    ]
    recs = diagnostics.sibling_regret(_BMPrior(), nodes)
    assert len(recs) == 1
    assert recs[0][1] == "STAGE"


def test_node_report_combines_fork_and_leaf_sections():
    """``node_report`` renders both the fork sibling regret and the leaf reachability
    over the node store."""
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    s = {"S_reduce_add": 1.0, "S_pw_multiply": 1.0, "S_n_distinct_input": 2.0, "S_ext_free_max": 512.0}
    nodes = [
        NodeRow("P", None, "ctx", "mm", s, 1.0, 1),
        NodeRow("c8", "P", "ctx", "mm", {**s, "BM": 8}, 1.0, 2),
        NodeRow("c64", "P", "ctx", "mm", {**s, "BM": 64}, 3.0, 2),
    ]
    text = diagnostics.node_report(_BMPrior(), nodes)
    assert "node store: 3 nodes" in text
    assert "fork sibling regret" in text
    assert "leaf reachability" in text


def test_node_report_separates_hardware():
    """The report groups by card, so an H100 op and a same-die H200 op (identical S_*
    signature, different latencies) are evaluated as separate blocks — never mixed."""
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    mm = {"S_reduce_add": 1.0, "S_pw_multiply": 1.0, "S_n_distinct_input": 2.0, "S_ext_free_max": 512.0}
    nodes = [
        NodeRow("Ph", None, "ctx", "mm", mm, 5.0, 1, gpu="NVIDIA H100 80GB"),
        NodeRow("h8", "Ph", "ctx", "mm", {**mm, "BM": 8}, 5.0, 2, gpu="NVIDIA H100 80GB"),
        NodeRow("h64", "Ph", "ctx", "mm", {**mm, "BM": 64}, 7.0, 2, gpu="NVIDIA H100 80GB"),
        NodeRow("Pn", None, "ctx", "mm", mm, 3.0, 1, gpu="NVIDIA H200 141GB"),
        NodeRow("n8", "Pn", "ctx", "mm", {**mm, "BM": 8}, 3.0, 2, gpu="NVIDIA H200 141GB"),
        NodeRow("n64", "Pn", "ctx", "mm", {**mm, "BM": 64}, 4.0, 2, gpu="NVIDIA H200 141GB"),
    ]
    text = diagnostics.node_report(_BMPrior(), nodes)
    assert "across 2 card(s)" in text
    assert "[NVIDIA H100 80GB]" in text and "[NVIDIA H200 141GB]" in text


def test_node_report_excludes_bench_fail_rows():
    """``bench_fail`` rows (sentinel latency, not a measurement) are dropped before
    any metric: injecting a huge-value failed sibling leaves the fork ranking and
    the node count identical to the fail-free store."""
    from dataclasses import replace  # noqa: PLC0415

    clean = _fork_nodes()
    with_fail = [*clean, replace(_child("cfail", 16, 3e7), status="bench_fail")]
    text_clean = diagnostics.node_report(_BMPrior(), clean)
    text_fail = diagnostics.node_report(_BMPrior(), with_fail)
    assert text_fail == text_clean  # metrics AND the "N nodes" count unaffected
    assert "node store: 4 nodes" in text_clean


def test_node_report_skips_cross_vocabulary_rows():
    """Rows recorded under another ``FEATURIZER_VERSION`` are excluded from every
    metric — their features are spelled in a vocabulary the prior can't read, so
    scoring them reports worse-than-random about a healthy model — and the skip is
    printed so a mostly-stale store is visible rather than silently empty."""
    from dataclasses import replace  # noqa: PLC0415

    clean = _fork_nodes()
    stale = [replace(n, node_key=n.node_key + "_v1", feat_ver=1) for n in clean]
    text = diagnostics.node_report(_BMPrior(), clean + stale)
    assert "node store: 4 nodes" in text  # the 4 stale rows don't count
    assert "skipped 4 row(s) from another featurizer vocabulary" in text


def test_node_report_splits_compile_regimes_per_card():
    """One card whose rows span two ``H_opt`` regimes (the -O1 tune tree + the
    deployable -O3 re-bench rows) renders one sub-block per regime — their
    latencies aren't comparable, so pooling them would corrupt leaf reachability."""
    from dataclasses import replace  # noqa: PLC0415

    mm = {"S_reduce_add": 1.0, "S_pw_multiply": 1.0, "S_n_distinct_input": 2.0, "S_ext_free_max": 512.0, "H_opt": 1.0}
    o1 = [
        _child("P", 0, 1.0, parent=None),
        _child("c8", 8, 1.0),
        _child("c64", 64, 3.0),
    ]
    o1 = [replace(n, features={**mm, **n.features}) for n in o1]
    o3 = [replace(_child("d8", 8, 0.5, parent=None), features={**mm, "H_opt": 3.0, "BM": 8})]
    text = diagnostics.node_report(_BMPrior(), o1 + o3)
    assert "@ -O1" in text and "@ -O3" in text  # one sub-block per regime
    assert "3 nodes" in text and "1 nodes" in text


# ---------------------------------------------------------------------------
# per-feature regret attribution: blame decomposition + ablation Δ
# ---------------------------------------------------------------------------


def _planted_offline(weights: dict[str, float]):
    """An OfflinePrior with ONLY the planted linear weights (both regimes), no
    hand-fit table — attribution over it must recover exactly the planted terms."""
    from emmy.compiler.pipeline.search.prior.offline import OfflinePrior  # noqa: PLC0415

    return OfflinePrior(weights=weights, weights_dynamic=weights)


def test_offline_explain_sums_to_the_scored_quality():
    """The exact-sum invariant: ``explain_features`` terms (linear + the three
    ``gate:*`` pseudo-terms) sum to the same quality ``mean_score_features``
    exponentiates — so a two-row term diff IS the model's preference gap. The
    atomic-free gate is enabled explicitly (it defaults OFF since the 2026-07-07
    golden-gate check) so the interaction's sign flip stays covered."""
    import math  # noqa: PLC0415

    from emmy.compiler.pipeline.search.prior.offline import OfflinePrior  # noqa: PLC0415

    p = OfflinePrior(atomic_free_weight=5.0)
    feats = {
        "D_pow2_threads": 1.0,
        "D_near_waves": 0.7,
        "D_splitk": 8.0,  # ≥ threshold → the atomic-free gate REWARDS the deferred finalize
        "D_finalize_kernel": 1.0,
        "D_scalar_on_warp_eligible": 1.0,
        "D_splitk_roundtrip": 19.0,
    }
    terms = p.explain_features(feats)
    assert {"gate:atomic_free", "gate:scalar_on_warp", "gate:splitk_roundtrip"} <= set(terms)
    assert terms["gate:atomic_free"] > 0 and terms["gate:scalar_on_warp"] < 0
    assert math.isclose(p.mean_score_features(feats), math.exp(-p._scale * sum(terms.values())))
    # Below the split threshold the gate flips to a penalty — and the sum still matches.
    narrow = {**feats, "D_splitk": 2.0}
    terms_n = p.explain_features(narrow)
    assert terms_n["gate:atomic_free"] < 0
    assert math.isclose(p.mean_score_features(narrow), math.exp(-p._scale * sum(terms_n.values())))


def test_offline_explain_selects_the_dynamic_weight_set():
    """A symbolic-axis row decomposes under the dynamic weight set — the same
    selection ``mean_score_features`` makes."""
    from emmy.compiler.pipeline.search.prior.offline import OfflinePrior  # noqa: PLC0415

    p = OfflinePrior()
    # A sentinel key the two shipped sets genuinely differ on — found, not hardcoded: the
    # dynamic fit chains from the static one, so any coordinate it never moves carries the
    # static raw weight verbatim and would make a fixed key's selection assert vacuous.
    key = next(k for k in sorted(p._w) if k in p._w_dyn and p._w[k] != p._w_dyn[k])
    static = p.explain_features({key: 1.0})
    dyn = p.explain_features({key: 1.0, "S_ext_n_symbolic_axis": 1.0})
    assert static[key] == p._w[key]
    assert dyn[key] == p._w_dyn[key]


def _planted_fork():
    """One fork the planted prior mispicks: D_x argues (wrongly) for the slow child,
    D_y argues (rightly, but too weakly) for the fast one."""
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    return [
        NodeRow("P", None, "ctx", "mm", {}, 1.0, 1),
        NodeRow("slow", "P", "ctx", "mm", {"D_x": 0.0, "D_y": 0.0}, 2.0, 2),
        NodeRow("fast", "P", "ctx", "mm", {"D_x": 1.0, "D_y": 1.0}, 1.0, 2),
    ]


def test_blame_recovers_the_planted_weight():
    """Blame on the planted misranking names D_x with exactly the regret-weighted
    planted gap: term diff = 0 − (−10) = +10, weight = regret − 1 = 1.0."""
    prior = _planted_offline({"D_x": -10.0, "D_y": 1.0})
    out = diagnostics.attribution_report(prior, _planted_fork())
    line = next(ln for ln in out.splitlines() if ln.split() and ln.split()[0] == "D_x")
    assert "+10.00" in line
    # D_y argued for the right child and was overruled → negative blame.
    line_y = next(ln for ln in out.splitlines() if ln.split() and ln.split()[0] == "D_y")
    assert "-1.00" in line_y
    assert "blind forks overall: 0" in out


def test_ablation_flags_the_misleading_feature():
    """Masking D_x lets D_y pick the fast child: Δ = 1.0 − 2.0 = −1.00x (actively
    misleading); masking D_y leaves the D_x mispick → Δ = 0 (collapsed into the
    zero line); single-fork support is flagged as noise."""
    prior = _planted_offline({"D_x": -10.0, "D_y": 1.0})
    out = diagnostics.attribution_report(prior, _planted_fork(), blame=False, ablate=True)
    line = next(ln for ln in out.splitlines() if ln.split() and ln.split()[0] == "D_x")
    assert "-1.00x" in line and "1~" in line  # misleading, on one-fork support
    assert "Δ = 0.00x everywhere" in out and "D_y" in out.split("everywhere")[1]


def test_blame_counts_blind_forks():
    """A missed fork whose siblings the prior's terms cannot separate (no weighted
    feature varies) is BLIND — the identical-vector failure class — and surfaces as
    a counter, not a silent empty row."""
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    nodes = [
        NodeRow("P", None, "ctx", "mm", {}, 1.0, 1),
        NodeRow("a", "P", "ctx", "mm", {"KNOB": 1.0}, 2.0, 2),
        NodeRow("b", "P", "ctx", "mm", {"KNOB": 2.0}, 1.0, 2),
    ]
    out = diagnostics.attribution_report(_planted_offline({"D_x": -10.0}), nodes)
    assert "BLIND" in out and "blind forks overall: 1" in out


def test_blame_unavailable_without_a_decomposition():
    """A prior with no ``explain_features`` decomposition (the base default) reports
    blame as unavailable instead of crashing — ablation still works (it only
    re-scores)."""

    class _NoExplain(_BMPrior):
        def explain_features(self, _feats):
            return None

    out = diagnostics.attribution_report(_NoExplain(sign=-1.0), _fork_nodes())
    assert "blame unavailable" in out


def test_attribution_pools_cards_and_respects_kernel_filter():
    """Fork records build per card (identity never crosses), the tables pool; the
    ``--kernel`` filter drops whole ops before attribution."""
    from dataclasses import replace  # noqa: PLC0415

    nodes = [replace(n, gpu="A") for n in _planted_fork()] + [replace(n, node_key=n.node_key + "b", gpu="B") for n in _planted_fork()]
    prior = _planted_offline({"D_x": -10.0, "D_y": 1.0})
    out = diagnostics.attribution_report(prior, nodes)
    assert "2 multi-child forks" in out and "pooled across 2 card(s)" in out
    miss = diagnostics.attribution_report(prior, nodes, kernel_filter="zzz")
    assert "no usable nodes" in miss


def test_node_report_kernel_filter_selects_ops_by_label():
    """``--kernel`` filters the node store by op label (derived from each node's
    ``S_*`` features) — whole ops drop atomically, and a non-matching filter prints
    an explanatory message, not the empty-store one."""
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    mm = {"S_reduce_add": 1.0, "S_pw_multiply": 1.0, "S_n_distinct_input": 2.0, "S_ext_free_max": 512.0}
    rd = {"S_reduce_add": 1.0, "S_ext_reduce_max": 64.0}  # a reduce op: no pw-multiply / 2nd input
    nodes = [
        NodeRow("Pm", None, "ctx", "mm", mm, 1.0, 1),
        NodeRow("m8", "Pm", "ctx", "mm", {**mm, "BM": 8}, 1.0, 2),
        NodeRow("m64", "Pm", "ctx", "mm", {**mm, "BM": 64}, 3.0, 2),
        NodeRow("Pr", None, "ctx", "rd", rd, 2.0, 1),
        NodeRow("r1", "Pr", "ctx", "rd", {**rd, "FK": 1}, 2.0, 2),
    ]
    out = diagnostics.node_report(_BMPrior(), nodes, kernel_filter="matmul")
    assert "3 nodes matching --kernel 'matmul'" in out  # only the matmul op's nodes survive
    assert "fork sibling regret" in out
    miss = diagnostics.node_report(_BMPrior(), nodes, kernel_filter="zzz")
    assert "no nodes match --kernel 'zzz'" in miss


# ---------------------------------------------------------------------------
# golden-anchored descent
# ---------------------------------------------------------------------------


def _anchor_golden(**over):
    """A synthetic fp16 512^3 matmul golden on TESTGPU. Knob families (BM/BK) are
    unregistered, so ``values_equal`` reduces to string equality — the canonical-parse
    path is exercised by the real-knob integration the pin gate already tests."""
    from emmy.compiler.pipeline.search.golden import MatmulGoldenConfig  # noqa: PLC0415

    kw = dict(
        name="test.anchor",
        M=512,
        N=512,
        K=512,
        dtype="fp16",
        knobs={"BM": "8", "BK": "2"},
        gpu_name="TESTGPU",
        compute_cap=(8, 9),
        emmy_us=5.0,
    )
    kw.update(over)
    return MatmulGoldenConfig(**kw)


def _anchor_feats(**knobs):
    """A stamped fp16 512^3 matmul histogram (ShapeKey (262144, 512, warp)) + knobs."""
    return {
        "S_ext_free_prod": 262144.0,
        "S_ext_free_max": 512.0,
        "S_ext_reduce_max": 512.0,
        "S_reduce_add": 1.0,
        "S_pw_multiply": 1.0,
        "S_n_distinct_input": 2.0,
        "S_dtype_f16": 2.0,
        **knobs,
    }


def _anchor_tree():
    """Root fork {BM=8, BM=32}, then under BM=8 a BK fork {2, 4} of measured leaves.
    The golden's path is BM=8 -> BK=2; BK does not vary the _BMPrior's score, so the
    BK fork is a predicted TIE resolved pessimistically to the worse-valued BK=4."""
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    return [
        NodeRow("A", None, "ctx", "mm", _anchor_feats(BM=8), 1.0, 1, gpu="TESTGPU"),
        NodeRow("B", None, "ctx", "mm", _anchor_feats(BM=32), 3.0, 1, gpu="TESTGPU"),
        NodeRow("A1", "A", "ctx", "mm", _anchor_feats(BM=8, BK=2), 1.0, 2, gpu="TESTGPU", is_leaf=True),
        NodeRow("A2", "A", "ctx", "mm", _anchor_feats(BM=8, BK=4), 2.0, 2, gpu="TESTGPU", is_leaf=True),
    ]


def test_anchor_full_match_with_tie_pessimistic_descent():
    """The walk follows BM=8 -> BK=2 to a measured leaf (matched 2); the BM fork is
    kept (the prior genuinely prefers BM=8), the BK fork is a tie -> pessimistically
    lost, with the measured gap 2.00x vs the golden's branch."""
    lines = diagnostics._golden_anchor_block(_BMPrior(), "TESTGPU", _anchor_tree(), None, goldens=[_anchor_golden()])
    row = next(line for line in lines if "test.anchor" in line)
    assert "followed 2/2 fork levels to a measured leaf" in row
    assert "descent kept 1/2" in row and "lost @BK 2.00x vs golden branch" in row
    assert "summary: 0/1" in lines[-1]


def test_anchor_partial_match_reports_unbuilt_branch():
    """A BK fork exploring only {4, 8} never built the golden's BK=2 branch: matched
    stops at 1 with the loud 'never built' reason (vs silence before)."""
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    nodes = [
        NodeRow("A", None, "ctx", "mm", _anchor_feats(BM=8), 1.0, 1, gpu="TESTGPU"),
        NodeRow("B", None, "ctx", "mm", _anchor_feats(BM=32), 3.0, 1, gpu="TESTGPU"),
        NodeRow("A4", "A", "ctx", "mm", _anchor_feats(BM=8, BK=4), 1.0, 2, gpu="TESTGPU", is_leaf=True),
        NodeRow("A8", "A", "ctx", "mm", _anchor_feats(BM=8, BK=8), 2.0, 2, gpu="TESTGPU", is_leaf=True),
    ]
    lines = diagnostics._golden_anchor_block(_BMPrior(), "TESTGPU", nodes, None, goldens=[_anchor_golden()])
    row = next(line for line in lines if "test.anchor" in line)
    assert "followed 1 of ~2 fork levels" in row and "never built below @BK (2 sibling(s) explored)" in row


def test_anchor_absence_is_loud():
    """A golden whose shape has zero node rows gets its own NO TREE DATA row and the
    summary headline counts it — absence must never render as health."""
    lines = diagnostics._golden_anchor_block(_BMPrior(), "TESTGPU", [], None, goldens=[_anchor_golden()])
    row = next(line for line in lines if "test.anchor" in line)
    assert "NO TREE DATA" in row
    assert "summary: 1/1" in lines[-1]


def test_anchor_o3_endpoint_and_regime_discipline():
    """The golden's recorded us enters ONLY the -O3 endpoint (pick 5.4 / golden 5.0 =
    1.08x over the H_opt=3 regime rows); the -O1 walk's descent gap stays the measured
    sibling ratio (2.00x) and never involves the golden's us."""
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    nodes = _anchor_tree() + [
        NodeRow("o3a", None, "ctx-o3", "mm", _anchor_feats(BM=8, H_opt=3.0), 5.4, 1, gpu="TESTGPU", is_leaf=True),
        NodeRow("o3b", None, "ctx-o3", "mm", _anchor_feats(BM=32, H_opt=3.0), 9.0, 1, gpu="TESTGPU", is_leaf=True),
    ]
    lines = diagnostics._golden_anchor_block(_BMPrior(), "TESTGPU", nodes, None, goldens=[_anchor_golden()])
    row = next(line for line in lines if "test.anchor" in line)
    assert "-O3 pick/golden 1.08x" in row  # 5.4 us pick over the 5.0 us golden, same regime
    assert "lost @BK 2.00x" in row  # the -O1 gap is 2.0/1.0 from sibling values, NOT 5.0-derived
    assert "followed 2/2 fork levels" in row  # the parentless -O3 rows never join the tree walk


def test_anchor_renders_inside_node_report(monkeypatch):
    """node_report appends the anchored block per card and the goldens-without-rows
    summary line for cards absent from the store."""
    from emmy.compiler.pipeline.search import golden as golden_mod  # noqa: PLC0415

    other = _anchor_golden(name="test.other", gpu_name="OTHERGPU")
    monkeypatch.setattr(golden_mod, "GOLDEN_CONFIGS", [_anchor_golden(), other])
    out = diagnostics.node_report(_BMPrior(), _anchor_tree())
    assert "golden-anchored descent" in out and "test.anchor" in out
    assert "NO node rows" in out and "OTHERGPU (1 golden(s))" in out


def test_anchor_prefix_matching_is_registry_canonical():
    """Real-knob matching goes through the pin gate's helpers: a bare atom-ALIAS
    golden TILE matches its canonically-stamped axis-keyed realization; a different
    geometry does not; a family the golden doesn't record is no constraint."""
    gold = {"TILE": "a:mma_m16n8k16_f16/w2x2/f4x4/k2"}
    stamped = {"S_ext_free_prod": 1.0, "TILE@a2": "a:mma_m16n8k16_f16_f32/w2x2/f4x4/k2", "STAGE@a2": "d2/cp/ring"}
    assert diagnostics._golden_prefix_consistent(stamped, gold)
    assert not diagnostics._golden_prefix_consistent({**stamped, "TILE@a2": "a:mma_m16n8k16_f16_f32/w1x1/f1x1"}, gold)


def test_anchor_level_zero_unbuilt_branch_and_o3_only_ops():
    """Two sentinel edges: a root group with no golden-consistent child reports
    'never built' at matched 0 (not the empty-tree text), and an op with ONLY -O3
    regime rows reports 'no explored forks' while the -O3 endpoint still renders."""
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    roots_only = [
        NodeRow("X", None, "ctx", "mm", _anchor_feats(BM=16), 1.0, 1, gpu="TESTGPU"),
        NodeRow("Y", None, "ctx", "mm", _anchor_feats(BM=32), 2.0, 1, gpu="TESTGPU"),
    ]
    lines = diagnostics._golden_anchor_block(_BMPrior(), "TESTGPU", roots_only, None, goldens=[_anchor_golden()])
    row = next(line for line in lines if "test.anchor" in line)
    assert "followed 0 of ~1 fork levels" in row and "never built below @BM (2 sibling(s) explored)" in row

    o3_only = [NodeRow("o3", None, "ctx-o3", "mm", _anchor_feats(BM=8, H_opt=3.0), 5.4, 1, gpu="TESTGPU", is_leaf=True)]
    lines = diagnostics._golden_anchor_block(_BMPrior(), "TESTGPU", o3_only, None, goldens=[_anchor_golden()])
    row = next(line for line in lines if "test.anchor" in line)
    assert "no explored forks" in row and "-O3 pick/golden 1.08x" in row
