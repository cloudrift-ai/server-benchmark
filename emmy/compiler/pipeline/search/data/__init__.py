"""Harmonized read-view over the measurement-data sources (golden configs, the tune
DB, the online-prior reservoir, and the digest-pinned measurement freeze): one
:class:`Sample` row type, one :class:`Dataset` query surface, the cheap
:class:`ShapeKey` structural identity, the freeze writer/loader (``freeze.py``), and
``group.py``'s :class:`Group` — one candidate pool packed as a matrix plus one label per
row, the candidate pools a ranking question is asked over, with :class:`GoldenGroup` the
kind whose labels MARK the rows goldens verified and :class:`MeasuredGroup` the kind whose
labels ARE the measured microseconds. Pools are built by plain functions there
(``group_measured`` for benched rows, ``pack_features`` for the packing both kinds share).
See ``sample.py`` for the featurization-fidelity contract.

Nothing here may import :mod:`~..prior` — the data layer describes candidates and
their labels; deciding what a score MEANS is the layer above (guarded by
``tests/architecture/test_layering.py``)."""

from __future__ import annotations

from emmy.compiler.pipeline.search.data.dataset import Dataset
from emmy.compiler.pipeline.search.data.freeze import FREEZE_KIND, FREEZE_VER, freeze_reason, load_freeze, load_node_rows, write_freeze
from emmy.compiler.pipeline.search.data.sample import KERNEL_NAME_RE, Sample
from emmy.compiler.pipeline.search.data.shape import ShapeKey, is_matmul, op_label

__all__ = [
    "FREEZE_KIND",
    "FREEZE_VER",
    "KERNEL_NAME_RE",
    "Dataset",
    "Sample",
    "ShapeKey",
    "freeze_reason",
    "is_matmul",
    "load_freeze",
    "load_node_rows",
    "op_label",
    "write_freeze",
]
