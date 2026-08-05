"""Registry of tests expected to fail while the tile schedule enumeration is INCOMPLETE — **EMPTY**.

The row enumerator was deleted and rebuilt RECURSIVELY over the site tree. Its first rebuild had
been single-site — ``_site_keys`` returned three keys for ``ops.head`` alone and ``_assemble`` took
one ``(TilePlan, stage, ReducePlan)`` triple, so every emitter hand-wrote a flat product over ONE
node's families. The two families that could not land in that shape — a COMPUTED operand edge (the
fused norm→linear / gate⊗up cone) and the flash streaming pair — are both cases where two SITES
must agree, which is exactly what the recursion is for. Both are carried now, so the list below is
empty and the rebuild's acceptance gate is MET.

The file is KEPT, not deleted, because it is the shape a scheduling coverage gap is recorded in and
the contract around it is what made the rebuild reviewable:

- a LIST OF EXACT NODE IDS, never a module or path glob — each id is a concrete acceptance
  obligation, not a suppressed area;
- the marker is **strict**, so an id that starts passing FAILS the run until it is deleted. That is
  what makes "the file shrinking to empty IS the completion gate" an enforced claim rather than an
  aspiration;
- genuinely flaky or card-conditional expectations do not belong here — they stay inline
  ``@pytest.mark.xfail(strict=False)`` at their own test, where the reason can name the condition.

Two limits to know about, should the list ever be repopulated:

- **Collection errors cannot be xfailed.** A demolition that takes a module's imports with it needs
  the probing tests deleted alongside the helpers they probed, not a mark.
- **GPU-gated ids must be added from a full GPU run**, and the registry matches the RAW
  ``item.nodeid``, so the ``@cuda`` / ``@cuda-cli`` suffix xdist's loadgroup scheduler bakes into
  the REPORTED id must be stripped before an id is added.

Applied by the root ``conftest.py``'s ``pytest_collection_modifyitems``.
"""

from __future__ import annotations

REASON = "tile schedule incomplete - awaiting the flash streaming pair's rows"

#: Node ids that fail because scheduling is missing. EMPTY — the enumeration is complete.
NODEIDS: frozenset[str] = frozenset()


#: Modules whose registered ids fail only as CONSEQUENCES of an enumerator gap — whole-model and
#: serving end-to-end graphs, and the golden drift gate. None is an independent obligation, so their
#: ids are marked ``run=False``: the expectation is recorded, the minutes are not spent. Kept beside
#: the empty list as the shape a future gap is recorded in.
CONSEQUENCE_MODULES: frozenset[str] = frozenset(
    {
        "tests/compiler/e2e/test_block.py",
        "tests/compiler/ir/test_dynamic_shapes.py",
        "tests/compiler/test_golden_drift_gate.py",
        "tests/serving/test_attention_split_gpu.py",
        "tests/serving/test_gen_prefill_device_gpu.py",
        "tests/serving/test_gen_runner_gpu.py",
        "tests/serving/test_generate_gpu.py",
        "tests/serving/test_runner_batched_gpu.py",
    }
)


def scheduling_xfail(nodeid: str) -> tuple[str, bool] | None:
    """``(reason, run)`` for ``nodeid``, or ``None`` if it is not a scheduler casualty. ``run`` is
    ``False`` for the consequence modules — record the expectation, skip the execution."""
    if nodeid not in NODEIDS:
        return None
    return REASON, nodeid.split("::")[0] not in CONSEQUENCE_MODULES
