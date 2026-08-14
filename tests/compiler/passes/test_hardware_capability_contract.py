"""Guard the compiler-pass boundary between target facts and transformations."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PASS_ROOT = Path(__file__).parents[3] / "emmy" / "compiler" / "pipeline" / "passes"
PASS_FILES = tuple(sorted(PASS_ROOT.rglob("*.py")))
MEASURED_ROUTING = PASS_ROOT / "lowering" / "tile" / "_cut.py"
ALGEBRA_RECOGNITION = {PASS_ROOT / "lowering" / "tile" / name for name in ("010_recognize.py", "_atomize.py", "_flash.py", "_softmax.py")}
FUSION_FILES = tuple(sorted((PASS_ROOT / "loop" / "fusion").glob("*.py")))


def test_structural_placement_reaches_a_pass_fixpoint_before_scheduling() -> None:
    """The canonical pipeline cannot enumerate a schedule while a new cut piece is still placeable."""
    from emmy.compiler.pipeline import TILE_PASSES

    assert TILE_PASSES.index("lowering/tile") < TILE_PASSES.index("lowering/schedule")
    structural_rules = {path.name for path in (PASS_ROOT / "lowering" / "tile").glob("[0-9]*.py")}
    schedule_rules = {path.name for path in (PASS_ROOT / "lowering" / "schedule").glob("[0-9]*.py")}
    assert structural_rules == {"010_recognize.py", "015_place.py"}
    assert schedule_rules == {"020_schedule.py", "030_split_reduce.py"}


@pytest.mark.parametrize("path", PASS_FILES, ids=lambda path: str(path.relative_to(PASS_ROOT)))
def test_passes_do_not_probe_process_global_target(path: Path):
    """A pass receives target facts through ``Context``; it never probes global device state."""
    tree = ast.parse(path.read_text(), filename=str(path))
    forbidden: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "emmy.compiler.target":
            if any(alias.name in {"compute_capability", "live_compute_capability"} for alias in node.names):
                forbidden.append(node.lineno)
        elif isinstance(node, ast.Import):
            if any(alias.name == "emmy.compiler.target" for alias in node.names):
                forbidden.append(node.lineno)
    assert not forbidden, f"target probes at lines {forbidden}; add a Context capability predicate"


def _reads_compute_capability(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Attribute) and child.attr == "compute_capability" for child in ast.walk(node))


def _getattr_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "getattr" or len(node.args) < 2:
        return None
    name = node.args[1]
    return name.value if isinstance(name, ast.Constant) and isinstance(name.value, str) else None


@pytest.mark.parametrize("path", PASS_FILES, ids=lambda path: str(path.relative_to(PASS_ROOT)))
def test_passes_do_not_compare_compute_capability_thresholds(path: Path):
    """Primitive legality is named once on ``Context``, not restated as SM-number branches."""
    tree = ast.parse(path.read_text(), filename=str(path))
    forbidden = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Compare) and _reads_compute_capability(node)]
    assert not forbidden, f"direct capability comparisons at lines {forbidden}; use a Context capability predicate"


@pytest.mark.parametrize("path", PASS_FILES, ids=lambda path: str(path.relative_to(PASS_ROOT)))
def test_product_identity_is_only_used_for_measured_routing(path: Path):
    """A product name may select measured placement evidence, never an implementation path."""
    tree = ast.parse(path.read_text(), filename=str(path))
    allowed_lines: set[int] = set()
    if path == MEASURED_ROUTING:
        routing = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_routing_entry")
        allowed_lines.update(range(routing.lineno, routing.end_lineno + 1))
    forbidden = []
    for node in ast.walk(tree):
        reads_identity = (isinstance(node, ast.Attribute) and node.attr in {"gpu_name", "hardware_id"}) or _getattr_name(node) in {
            "gpu_name",
            "hardware_id",
        }
        if reads_identity and node.lineno not in allowed_lines:
            forbidden.append(node.lineno)
    assert not forbidden, f"product-identity reads at lines {forbidden}; only measured routing may use them"


@pytest.mark.parametrize("path", sorted(ALGEBRA_RECOGNITION), ids=lambda path: path.name)
def test_algebra_recognition_does_not_read_hardware_capabilities(path: Path):
    """Hardware filters the atom/schedule domain after recognition; it cannot change the algebra."""
    tree = ast.parse(path.read_text(), filename=str(path))
    forbidden = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr.startswith("has_")]
    assert not forbidden, f"hardware-capability reads at lines {forbidden}; recognition must stay algebra-only"


@pytest.mark.parametrize("path", FUSION_FILES, ids=lambda path: path.name)
def test_fusion_does_not_consult_placement_schedule_or_hardware(path: Path):
    """Fusion forms the maximal reasonable algebraic region; later layers decide kernel cuts."""
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
    forbidden_imports = [
        (line, module)
        for line, module in imports
        if module.startswith(
            (
                "emmy.compiler.context",
                "emmy.compiler.target",
                "emmy.config",
                "emmy.compiler.pipeline.knob",
                "emmy.compiler.pipeline.search",
                "emmy.compiler.pipeline.passes.lowering",
            )
        )
    ]
    forbidden_names = [
        node.lineno
        for node in ast.walk(tree)
        if (isinstance(node, ast.Constant) and node.value in {"PLACE", "TILE", "REDUCE", "STAGE", "WORK", "RASTER"})
        or (
            isinstance(node, ast.Attribute)
            and (node.attr.startswith("has_") or node.attr in {"knobs", "gpu_name", "hardware_id", "compute_capability"})
        )
        or _getattr_name(node) in {"knobs", "gpu_name", "hardware_id", "compute_capability"}
        or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(arg.arg == "ctx" for arg in node.args.args))
    ]
    assert not forbidden_imports, f"fusion imports later-layer policy: {forbidden_imports}"
    assert not forbidden_names, f"fusion reads placement/schedule/hardware facts at lines {forbidden_names}"
