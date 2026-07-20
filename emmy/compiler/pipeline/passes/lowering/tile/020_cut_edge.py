"""Realize ``PLACE@cone=cut`` — split the fused producer-cone kernel at the A seam.

``030_split_reduce``'s sibling for the PLACE codec's cone element: loop fusion fuses greedily and the
recognizer nodifies the MONOID-producer composition (the fused norm→linear edge) — this rule
realizes the ``cut`` placement as a **graph rewrite** on the recognized kernel, never a fusion
veto. Under the pin the recognizer's own gate already withholds the fused-form warp rows (the
kernel arrives here in its coop ``Map(source=Reduction)`` spelling); this rule re-binds the cone
structure (:func:`bind_prologue_contraction` — structure-only, so the same probe both halves
use) and splits:

- **producer kernel** — a plain ``LoopOp`` materializing the cone value ``x̂[m, k]`` into a
  ``<out>__cone`` workspace: the per-row statistic prologue hoisted over the k sweep
  (``for m: [stat…, for k: [cell…, Write]]``), so the statistic is computed ONCE per row —
  never per N-stripe (the fused form's recompute tax) and never per cell.
- **consumer kernel** — the same projection ``Map`` re-lowered (``ops.lower``) with
  ``a_operand`` swapped for a plain gmem ``Load`` of the workspace, wrapped back in its free
  loops.

Both halves are UN-mapped ``LoopOp``\\ s: the pass scan restarts on any applied rewrite, so
``010_recognize`` re-enters and each half gets its own recognition, scheduling and fork — the
producer lands on the reduce tier (coop statistic), the consumer on the plain-matmul tiers
(gmem-direct / cp.async / TMA staging, the schedules a computed A can never ride). Measured on
the gemma gate_up fused-edge shape (5090): cut + the golden-family config = 496–503 µs e2e vs
the exhaustively-optimized fused kernel's ~660 and the unfused eager pair's ~570.

**Multi-fold (gate/up) split.** A multi-fold cone (``swiglu(x̂@Wg, x̂@Wu)``) cannot cut into ONE
consumer — a multi-fold *plain* matmul has no mma tier (the gmem-direct / cp / TMA transports are
single-fold; the sync compute-fill is keyed on a *computed* A) — so it splits into **N
single-fold channel matmuls**, each reading the shared workspace A and writing an ``out__ch<i>``
workspace, plus a downstream pointwise **combine** kernel carrying the ⊗-combine tail
(``pro_map.body`` — the GeGLU), which reloads the channel accumulators and writes ``out``. Each
channel matmul is single-fold, so it rides ``d2/tma/ring`` and BEATS cuBLAS (5090, M=256:
~113 µs/channel f16acc); the whole cut is ~270 µs e2e = 1.17× the unfused eager pair (~317), the
form that the fused computed-A megakernel provably cannot reach (its compute floor ≈ eager). This
is why the cut is the perf answer for the mlp edge at prefill M — the fused ``mlp_geglu.m256``
golden anchors only the DEFAULT (fused) deploy.

Termination is structural: the consumer's A is a ``Load`` (``bind_prologue_contraction`` finds
no cone), and the producer's column sweep carries no ⊗-fold ``Accum`` (not the composition) —
neither half re-matches. Restrictions (each a ``RuleSkipped``, the kernel stays fused): no lead
(batch) axes / exactly one free axis (the ``m`` rows) this cut, and at most one bridged
statistic."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import Loop, LoopOp
from emmy.compiler.ir.stmt import Body, Load, Write
from emmy.compiler.ir.tile import Contraction, Map, TileOp
from emmy.compiler.ir.tile.ops import lower as tile_lower
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.stamp._stamp import restamp_structural_features
from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_prologue_contraction
from emmy.compiler.pipeline.search.space import place_decision

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> Graph | None:
    # A pure REALIZER, like ``030_split_reduce``: the decision lives in the SCHEDULE (the
    # ``PLACE@cone`` knob the recognizer stamps on the chosen row), the graph carries the kernel
    # count. Reading the stamp — not the global pin — is what lets the cone cut be *selected* per
    # op by the ordinary evidence pick (a golden recording ``PLACE@cone: cut`` with the cut's
    # measured total cost), exactly as ``REDUCE: g8k`` selects the split-K cut. Falls back to the
    # global pin for rows that carry no stamp (the pinned path, where the recognizer emits the
    # unfused map form outright).
    tile: TileOp = root.op
    stamped = tile.knobs.get("PLACE@cone")
    decision = stamped if stamped in ("fuse", "cut") else place_decision("cone")
    if decision != "cut":
        raise RuleSkipped("PLACE@cone is fuse — the cone stays register-resident")
    if tile.op is None:
        raise RuleSkipped("kernel-less TileOp — nothing to cut")
    bound = bind_prologue_contraction(tile.op, tile.place.free)
    if bound is not None:
        # The MONOID composition: the column loop lives INSIDE the map body, so the only free
        # (grid) axis here is ``m``, and the cone carries a per-row statistic prologue.
        pro_map, _n_ax = bound
        c: Contraction = pro_map.source
        free_ok, stat_free = len(tile.place.free) == 1, False
    else:
        # A STAT-FREE cone — the node is ALREADY the computed-A ``Contraction`` (loop fusion
        # inlined a pointwise operand cone: gemma's GeGLU combine ahead of down_proj). Same cut,
        # minus the statistic. Here BOTH ``m`` and ``n`` are peeled free axes, not just ``m``.
        node = tile.op.source if isinstance(tile.op, Map) and isinstance(tile.op.source, Contraction) else tile.op
        if not isinstance(node, Contraction) or not node.a_computed or len(node.folds) != 1:
            raise RuleSkipped("not a producer-cone composition — nothing to cut")
        pro_map, c = (tile.op if isinstance(tile.op, Map) else None), node
        free_ok, stat_free = {a.name for a in tile.place.free} <= {c.m_axis.name, c.n_axis.name}, True
    if c.lead_axes or not free_ok:
        raise RuleSkipped("lead/batch axes on the cone — the 2-D workspace cut doesn't cover them yet")
    pro, cell, stats = c.stat_prologue()
    # A loop-invariant SCALAR load is not a per-row statistic — the seam splits at the first
    # K-indexed stmt, so a stat-free cone's scalar constants (the GeGLU's 0.044715 / 0.797885 /
    # 0.5) land in the prologue and read as four "statistics". They need no bridging kernel and no
    # smem row: the cell just re-loads them. Only an m-indexed prologue def is a real statistic.
    scalars = [s for s in pro if isinstance(s, Load) and not any(e.free_vars() for e in s.index)] if stat_free else []
    if scalars:
        scalar_ids = {id(s) for s in scalars}
        scalar_names = {nm for s in scalars for nm in s.defines()}
        pro = tuple(s for s in pro if id(s) not in scalar_ids)
        cell = (*scalars, *cell)
        stats = tuple(nm for nm in stats if nm not in scalar_names)
    if len(stats) > 1:
        raise RuleSkipped("multi-statistic cone — the per-statistic workspace cut covers one bridged stat this cut")
    m_ax, k_ax = c.m_axis, c.k_axis
    out = root.output

    # The workspace element dtype: the cone's k-indexed gmem read's tensor (the norm's ``x``) —
    # the same dtype the fused form's A slab stored, so numerics match the fused kernel.
    ws_dtype = out.dtype
    for s in cell:
        if isinstance(s, Load) and k_ax.name in {v for e in s.index for v in e.free_vars()}:
            t = tile.inputs.get(s.input) if tile.inputs else None
            if t is not None:
                ws_dtype = t.dtype
            break
    ws = f"{out.name}__cone"
    # Idempotence. The stat-BEARING cut terminates structurally: its producer carries the statistic
    # reduce, so ``010_merge_loop_ops``'s reduce-heavy guard will not fuse it back into the
    # consumer. A STAT-FREE producer is pure pointwise and carries no such brake — fusion re-inlines
    # it into the consumer's K loop on the next sweep, the consumer is computed-A again, and the
    # rule re-fires on a node it has already cut (observed on the gemma-4 pre256 norm→kv cone:
    # ``Node id 'linear_2_reduce__cone' already exists``). Refuse when this node's workspace is
    # already in the graph: the cut has been applied, and applying it twice is never the intent.
    if ws in match.graph.nodes:
        raise RuleSkipped(f"cone already cut — {ws} exists")

    # Producer(s): the cone value materializes through the same two kernels the UNFUSED graph
    # had — a per-row STATISTIC kernel (``for m: [stat reduce…, Write(stat[m])]``, m on the
    # grid, the reduce tier's coop fold) and a POINTWISE scale kernel (``for m: for k:
    # [Load(stat[m]), cell…, Write(ws[m,k])]``, both axes on the grid). A single nested
    # ``for m: [stat…, for k: …]`` kernel measured 1.8 ms here: only m lifts to the grid, so
    # every thread sweeps K serially — the statistic must live in its own kernel to give the
    # scale sweep its 2-D grid. No-statistic cones (a plain MAP producer) skip the stat kernel.
    stat_nodes: list[tuple[LoopOp, str]] = []
    cell_body: tuple = tuple(cell)
    if stats:
        (stat_name,) = stats
        stat_ws = f"{out.name}__stat"
        stat_op = LoopOp(body=Body((Loop(axis=m_ax, body=Body((*pro, Write(output=stat_ws, index=(Var(m_ax.name),), value=stat_name)))),)))
        stat_nodes.append((stat_op, stat_ws))
        cell_body = (Load(name=stat_name, input=stat_ws, index=(Var(m_ax.name),)), *cell)
    k_loop = Loop(axis=k_ax, body=Body((*cell_body, Write(output=ws, index=(Var(m_ax.name), Var(k_ax.name)), value=c.a_name))))
    producer = LoopOp(body=Body((Loop(axis=m_ax, body=Body((k_loop,))),)))

    # Consumer(s): each ⊗-fold channel becomes its OWN plain single-fold matmul reading the shared
    # workspace A (``a_load``) — a single-fold node rides the plain-matmul mma tiers (gmem-direct /
    # cp.async / TMA / d2/tma/ring, the schedules a computed A can never ride), which is the whole
    # point of the cut. A single-fold cone (norm→linear) keeps the original projection ``Map`` (its
    # combine, if any, IS the projection and writes ``out`` directly). A multi-fold cone (gate/up)
    # can't keep one node — a multi-fold plain matmul has no mma tier (the transports are single-fold,
    # the sync fill needs a computed A) — so it splits into N channel matmuls, each writing a
    # per-channel ``out__ch<i>`` workspace, plus a downstream pointwise COMBINE kernel carrying the
    # ⊗-combine tail (``pro_map.body`` — the geglu), which loads the channel accumulators back and
    # writes ``out``. Each channel matmul beats cuBLAS on ``d2/tma/ring``; the combine is a cheap
    # elementwise pass.
    a_load = Load(name=c.a_name, input=ws, index=(Var(m_ax.name), Var(k_ax.name)))
    n_axis = c.n_axis

    def _consumer(proj_map) -> LoopOp:
        return LoopOp(body=Body((Loop(axis=m_ax, body=Body((Loop(axis=n_axis, body=Body(tuple(tile_lower(proj_map)))),))),)))

    consumer_nodes: list[tuple[LoopOp, str]] = []  # (op, workspace/out name)
    if len(c.folds) == 1:
        # A stat-free cone has no ``Map`` wrapper — its projection rides the node's own epilogue,
        # so the re-bound Contraction IS the consumer.
        cut_c = replace(c, a_operand=a_load)
        consumer_nodes.append((_consumer(cut_c if pro_map is None else replace(pro_map, source=cut_c)), out.name))
        combine = None
    else:
        chan_accs: list[tuple[str, str]] = []  # (workspace name, accumulator SSA name)
        for i, fold in enumerate(c.folds):
            ch_ws = f"{out.name}__ch{i}"
            proj = Map(
                body=Body((Write(output=ch_ws, index=(Var(m_ax.name), Var(n_axis.name)), value=fold[1]),)),
                source=replace(c, folds=(fold,), a_operand=a_load),
            )
            consumer_nodes.append((_consumer(proj), ch_ws))
            chan_accs.append((ch_ws, fold[1]))
        # The combine kernel: reload each channel accumulator from its workspace, then run the
        # original ⊗-combine tail (``pro_map.body`` writes ``out``). Pure pointwise over (m, n).
        acc_loads = tuple(Load(name=acc, input=cws, index=(Var(m_ax.name), Var(n_axis.name))) for cws, acc in chan_accs)
        combine = LoopOp(body=Body((Loop(axis=m_ax, body=Body((Loop(axis=n_axis, body=Body((*acc_loads, *pro_map.body))),))),)))

    frag = Graph()
    for inp in root.inputs:
        n = match.graph.nodes[inp]
        frag.add_node(op=InputOp(), inputs=[], output=n.output, node_id=inp)
    from emmy.compiler.dtype import F32  # noqa: PLC0415 — the bridged statistic rows are fp32, as in the fused form

    for stat_op, stat_ws in stat_nodes:
        frag.add_node(
            op=stat_op, inputs=[i for i in root.inputs if i in stat_op.inputs], output=Tensor(stat_ws, (m_ax.extent,), F32), node_id=stat_ws
        )
    stat_ids = [nm for _, nm in stat_nodes]
    frag.add_node(
        op=producer,
        inputs=[*(i for i in root.inputs if i in producer.inputs), *(nm for nm in stat_ids if nm in producer.inputs)],
        output=Tensor(ws, (m_ax.extent, k_ax.extent), ws_dtype),
        node_id=ws,
    )
    for cons_op, cons_out in consumer_nodes:
        # A split channel writes its accumulator to an F32 workspace (NOT out.dtype): the downstream
        # combine reads it back and runs the ⊗-combine tail (e.g. GeGLU's ``gelu(gate)`` cubes the
        # value — ``gate³`` overflows f16 for |gate| ≳ 40), so the channel value must keep the f32
        # accumulator precision the fused form's register epilogue had. Single-fold (no combine)
        # writes the real output directly at its own dtype.
        node_out = out if combine is None else Tensor(cons_out, (m_ax.extent, n_axis.extent), F32)
        frag.add_node(op=cons_op, inputs=[*(i for i in root.inputs if i in cons_op.inputs), ws], node_id=cons_out, output=node_out)
    if combine is not None:
        frag.add_node(
            op=combine,
            inputs=[*(i for i in root.inputs if i in combine.inputs), *(nm for _, nm in consumer_nodes)],
            output=out,
            node_id=out.name,
        )
    # Re-stamp structural features: ``020_stamp_structural_features`` ran at fusion end and never
    # revisits these spliced fragments, so without this the split kernels carry the FUSED op's
    # features (or none) — they'd never match their own golden / prior rows (the cut consumer is a
    # plain matmul that should join the ``mlp_gate_up`` matmul evidence, not the ``fused`` kind).
    for node in frag.nodes.values():
        if isinstance(node.op, LoopOp):
            restamp_structural_features(node.op, frag)
    frag.outputs = [out.name]
    return frag
