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
from emmy.compiler.pipeline.strategy import RunStartEvent, SplicedEvent, Strategy


class ProvenanceStrategy(Strategy):
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
        provenance.propagate(
            e.graph,
            consumed_prov=consumed_prov,
            new_compute_ids=[nid for nid in r.new_compute_ids if nid in e.graph.nodes],
            new_by_old=new_by_old,
            output_map=r.output_map,
            mint_pieces=e.pass_name in self.MINTING_PASSES,
        )
