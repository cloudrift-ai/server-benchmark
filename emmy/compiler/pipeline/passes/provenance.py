"""ProvenanceStrategy — op provenance, end to end, as pipeline strategy.

Provenance (which original model ops a kernel realizes — the ``k_…`` naming and coverage data)
is a concern of THIS pipeline, not of the engine or the graph: ``Graph.splice`` is pure surgery
and the engine emits events. A pipeline built without this strategy simply has no provenance
anywhere; the consumers (kernel naming's ``name_for``, golden origin selection) fall back to
node-id names or are specific to pipelines that carry it. A splice performed outside a run
(e.g. the checkpoint loader's spelling splices) threads no provenance either — the next run's
seed gives every unseeded node a fresh self-origin.
"""

from __future__ import annotations

from emmy.compiler import provenance
from emmy.compiler.pipeline.strategy import PipelineStrategy, RunStartEvent, SplicedEvent


class ProvenanceStrategy(PipelineStrategy):
    """Threads op provenance through every rewrite of a run. A decomposition's fragments MINT —
    each new compute node becomes a fresh piece of the consumed origins (one op expanding into
    many distinct primitives); every other splice AGGREGATES — fragment outputs union the pieces
    their consumed nodes carried. Rebinds are the same kernel decided further: provenance rides
    the node untouched. Stateless: everything it needs arrives on the event."""

    MINTING_PASSES = ("frontend/decomposition",)

    def on_run_start(self, e: RunStartEvent) -> None:
        provenance.seed(e.graph)

    def on_spliced(self, e: SplicedEvent) -> None:
        r = e.receipt
        consumed_prov = {nid: hints.get(provenance.PROV) or {} for nid, hints in r.consumed_hints.items()}
        # Resolve each redirected buffer to its producing node id (identity for primary
        # outputs) — prov hints live on nodes.
        new_by_old = {}
        for old, new_buf in r.redirected.items():
            producer = e.graph.producer(new_buf)
            if producer is not None:
                new_by_old[old] = producer.id
        _propagate(
            e.graph,
            consumed_prov=consumed_prov,
            new_compute_ids=[nid for nid in r.new_compute_ids if nid in e.graph.nodes],
            new_by_old=new_by_old,
            output_map=r.output_map,
            mint_pieces=e.pass_name in self.MINTING_PASSES,
        )


def _propagate(
    graph,
    *,
    consumed_prov: dict[str, dict],
    new_compute_ids: list[str],
    new_by_old: dict[str, str],
    output_map: dict[str, str],
    mint_pieces: bool,
) -> None:
    """Thread provenance across one ``Graph.splice``.

    Called by the provenance STRATEGY from the post-splice event (``on_spliced``), off the
    splice's receipt — the graph itself carries no provenance knowledge. Node ids are looked up
    defensively: orphan removal runs inside the splice, so a receipt id may no longer exist.

    ``consumed_prov`` is ``{consumed_node_id: prov}`` snapshotted before the
    consumed nodes were removed; ``new_compute_ids`` are the graph ids of the
    freshly-added non-boundary fragment nodes; ``new_by_old`` maps each
    redirected consumed node to its fragment output's producing NODE id.

    - **mint** (``mint_pieces=True``, decomposition): each new fragment node
      becomes a fresh piece of the consumed origins — one op expanding into
      many distinct primitives.
    - **aggregate** (otherwise — fusion / lifting / optimization folds): each
      fragment output inherits its own consumed node's pieces unioned with the
      ``shared`` pieces of every *dissolved* consumed node (those not in
      ``output_map`` — e.g. a producer inlined into all its consumers), so no
      origin is dropped at a multi-output splice.

    A fragment output that is a boundary sentinel (e.g. a fold collapsing a
    transpose into its parameter ``ConstantOp``) gets its prov scrubbed instead:
    the splice's generic hint merge copied the consumed node's hints — prov
    included — onto it, and a boundary must never carry provenance (its pieces
    would inflate :func:`totals` and make every other kernel of the origin read
    as partial coverage)."""
    for new_out in new_by_old.values():
        node = graph.nodes.get(new_out)
        if node is not None and provenance.is_boundary(node.op):
            node.hints.remove(provenance.PROV)
    if mint_pieces:
        origins_kind = provenance.origins(provenance.union(*consumed_prov.values())) if consumed_prov else {}
        for nid in new_compute_ids:
            node = graph.nodes.get(nid)
            if node is not None:
                provenance.put(node, provenance.mint(origins_kind, nid))
        return

    shared = provenance.union(*[p for cid, p in consumed_prov.items() if cid not in output_map])
    for old_id, new_out in new_by_old.items():
        node = graph.nodes.get(new_out)
        if node is not None:
            provenance.put(node, provenance.union(consumed_prov.get(old_id, {}), shared))
    outputs = set(new_by_old.values())
    for nid in new_compute_ids:
        node = graph.nodes.get(nid)
        if node is not None and nid not in outputs:
            provenance.put(node, provenance.union(shared))
