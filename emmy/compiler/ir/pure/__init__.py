"""Pure terms — the value vocabulary every IR level shares.

A pure term denotes a value. It binds names, it carries an algebra, it can be substituted and
compared up to α-renaming; it does not occupy a position in an instruction stream and it has no
scope of its own to seed or tear down. The statements live in :mod:`emmy.compiler.ir.stmt`; the
two vocabularies meet in exactly one direction — a term is RENDERED into statements at the point
of use (``Fold.merge`` / ``Fold.lower``), never spliced in as one.

- :mod:`.lam` — :class:`~emmy.compiler.ir.pure.lam.Lambda`, the ONE binder kind; a fold's ``base``
  is its ``componentwise`` program, and ``components`` reads that shape back.
- :mod:`.fold` — :class:`~emmy.compiler.ir.pure.fold.Fold`, the ONE reduce term (what Tile IR
  stores) plus its derived readings, its render-to-statements (``merge`` / ``step`` / ``lower``)
  and ``fuse``.
- :mod:`.twist` — the twist RECIPES (a twisted monoid as data: the base ⊕, its per-element lift,
  ``psi`` / ``psi_inv``, the channel patterns and the stable ⊕ program) that ``Fold.fuse`` fuses a
  two-pass reduce pair into, and that a fused ``Fold`` then NAMES in its ``twist`` field.

The invariant these modules exist to state lives in ``ir/ARCHITECTURE.md`` ("Pure terms vs statements").
"""

from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.pure.lam import Lambda

__all__ = ["Fold", "Lambda"]
