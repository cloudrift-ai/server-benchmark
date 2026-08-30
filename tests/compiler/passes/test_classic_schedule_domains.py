"""Production classic scheduling obeys the independent-domain contract."""

from emmy.compiler.context import Context
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.classic_schedule import ClassicProblem, ClassicScheduleCodec, enumerate_reference
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.stmt import Assign, Body
from emmy.compiler.ir.tile import Placement, TileOp
from emmy.compiler.pipeline.passes.lowering.tile._classic import project_domains, schedule


def _signature(codec, assignment) -> tuple[tuple[str, str], ...]:
    return tuple(codec.encode(assignment).items())


def test_production_enumeration_is_the_compatible_independent_product() -> None:
    root = Fold.projection(body=Body((Assign("y", "add", ("x", "x")),)), results=("y",))
    tile = TileOp(op=root, place=Placement(free=(Axis("n", 8),)))
    target = Context.from_target((12, 0))
    problem = ClassicProblem(tile.op, target)
    domains = project_domains(problem)
    codec = ClassicScheduleCodec(problem, domains)

    reference = {_signature(codec, assignment) for assignment in enumerate_reference(problem, domains)}
    leaves = schedule(tile, "pointwise", {}, target)

    assert {_signature(codec, leaf.schedule) for leaf in leaves} == reference
    assert len(reference) == domains.product_size == 1
    (materialized,) = leaves[0].expand()
    assert materialized.classic == leaves[0].schedule
    assert materialized.place == tile.place.on_grid()
