"""The closed Layer-1 dialect unions hold their module contracts.

``FrontendOp`` / ``TensorOp`` / ``BoundaryOp`` are closed union aliases: the checker
relies on them for exhaustive ``match``, and this file enforces the runtime side —
membership stays in sync with the classes actually defined in each module, and
every frontend variant has the decomposition rule its module doc promises.
"""

from typing import final, get_args

from emmy.compiler.ir import base as base_ir
from emmy.compiler.ir.base import Op
from emmy.compiler.ir.frontend import ir as frontend_ir
from emmy.compiler.ir.tensor import ir as tensor_ir
from emmy.compiler.pipeline.pipeline import Pass


def _variants(alias) -> set[type]:
    """The member classes of a PEP 695 union alias (lazy — unwrap before get_args)."""
    return set(get_args(alias.__value__))


def _final_ops_defined_in(module) -> set[type]:
    """All ``@final`` Op subclasses defined (not just imported) in *module*."""
    return {
        v
        for v in vars(module).values()
        if isinstance(v, type) and issubclass(v, Op) and v.__module__ == module.__name__ and getattr(v, "__final__", False)
    }


# --- union membership ↔ module contents ---


def test_union_variants_are_exactly_the_final_op_classes():
    """Each union lists exactly the ``@final`` op classes of its module — a class
    added to a module without joining its union (or vice versa) fails here."""
    for module, alias in (
        (frontend_ir, frontend_ir.FrontendOp),
        (tensor_ir, tensor_ir.TensorOp),
        (base_ir, base_ir.BoundaryOp),
    ):
        assert _variants(alias) == _final_ops_defined_in(module), module.__name__


def test_unions_are_disjoint():
    frontend, tensor, boundary = (
        _variants(frontend_ir.FrontendOp),
        _variants(tensor_ir.TensorOp),
        _variants(base_ir.BoundaryOp),
    )
    assert not (frontend & tensor or frontend & boundary or tensor & boundary)


def test_final_marker_is_importable():
    # ``@final`` classes carry ``__final__`` per the typing spec — the membership
    # test above silently weakens if that attribute convention ever changes.
    @final
    class Probe: ...

    assert getattr(Probe, "__final__", False)


# --- the FrontendOp module contract: everything decomposes ---


def test_every_frontend_op_has_a_decomposition_rule():
    """``ir/frontend/ir.py`` promises every frontend op is rewritten to tensor
    primitives by a rule under ``passes/frontend/decomposition/``. Enforced
    exhaustively over the union: a new FrontendOp variant without a rule whose
    pattern roots on it fails here."""
    rules = Pass.load("frontend/decomposition", index=0).rules
    rooted = {p.op_type for rule in rules for p in rule.pattern}
    missing = _variants(frontend_ir.FrontendOp) - rooted
    assert not missing, f"frontend ops without a decomposition rule: {sorted(c.__name__ for c in missing)}"
