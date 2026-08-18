"""Birth-time spelling for caller-owned physical input storage."""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Expr


@dataclass(frozen=True)
class PhysicalCarrier:
    """One caller-owned tensor addressed over a logical input's coordinates."""

    name: str
    input_suffix: str
    shape: tuple[int, ...]
    dtype: str
    coord_map: tuple[Expr, ...]


@dataclass(frozen=True)
class TypedAlgebra:
    """One shape-preserving typed elementwise operation over carrier values."""

    name: str
    op: str
    inputs: tuple[str, ...]
    dtype: str


@dataclass(frozen=True)
class PhysicalInputStorage:
    """Physical carriers and typed algebra that reproduce one logical input."""

    logical_shape: tuple[int, ...]
    carriers: tuple[PhysicalCarrier, ...]
    algebra: tuple[TypedAlgebra, ...]
    output: str


def _validate_storage(name: str, logical_dtype: str, storage: PhysicalInputStorage) -> None:
    from emmy.compiler.dtype import get as get_dtype  # noqa: PLC0415
    from emmy.compiler.ir.expr import PLACEHOLDER_PREFIX  # noqa: PLC0415

    logical_shape = tuple(int(dim) for dim in storage.logical_shape)
    if not logical_shape or any(dim <= 0 for dim in logical_shape):
        raise ValueError(f"physical input {name!r}: logical shape must contain only positive dimensions")
    if not storage.carriers:
        raise ValueError(f"physical input {name!r}: at least one carrier is required")

    allowed_vars = {f"{PLACEHOLDER_PREFIX}{axis}" for axis in range(len(logical_shape))}
    values: dict[str, str] = {}
    suffixes: set[str] = set()
    for index, carrier in enumerate(storage.carriers):
        if not carrier.name or carrier.name in values:
            raise ValueError(f"physical input {name!r}: carrier names must be non-empty and unique")
        if index == 0 and carrier.input_suffix:
            raise ValueError(f"physical input {name!r}: the first carrier must retain the logical input id")
        if index and not carrier.input_suffix:
            raise ValueError(f"physical input {name!r}: only the first carrier may use an empty input suffix")
        if carrier.input_suffix in suffixes:
            raise ValueError(f"physical input {name!r}: carrier input suffixes must be unique")
        shape = tuple(int(dim) for dim in carrier.shape)
        if any(dim <= 0 for dim in shape) or len(carrier.coord_map) != len(shape):
            raise ValueError(f"physical input {name!r}: carrier {carrier.name!r} has invalid shape or coordinate rank")
        if any(not coordinate.free_vars() <= allowed_vars for coordinate in carrier.coord_map):
            raise ValueError(f"physical input {name!r}: carrier coordinates may only use logical output placeholders")
        get_dtype(carrier.dtype)
        values[carrier.name] = carrier.dtype
        suffixes.add(carrier.input_suffix)

    for operation in storage.algebra:
        if not operation.name or operation.name in values:
            raise ValueError(f"physical input {name!r}: algebra names must be non-empty and unique")
        if not operation.inputs or any(input_name not in values for input_name in operation.inputs):
            raise ValueError(f"physical input {name!r}: algebra inputs must name earlier carrier or algebra values")
        get_dtype(operation.dtype)
        values[operation.name] = operation.dtype
    if storage.output not in values:
        raise ValueError(f"physical input {name!r}: output must name a carrier or algebra value")
    if values[storage.output] != logical_dtype:
        raise ValueError(
            f"physical input {name!r}: output dtype {values[storage.output]!r} does not reproduce logical dtype {logical_dtype!r}"
        )


def spell_physical_inputs(graph: Graph, specs: dict[str, PhysicalInputStorage]) -> dict[str, tuple[str, ...]]:
    """Replace logical graph inputs with physical carriers and typed algebra.

    A storage descriptor exists only for this birth-time rewrite. The resulting
    graph contains ordinary input tensors, coordinate maps, and elementwise
    algebra; later compiler stages receive no storage-format object.
    """
    from emmy.compiler.ir.base import InputOp  # noqa: PLC0415
    from emmy.compiler.ir.tensor.ir import ElementwiseOp, IndexMapOp, IndexSource  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    prepared: list[tuple[str, PhysicalInputStorage]] = []
    new_input_ids: set[str] = set()
    for name, storage in specs.items():
        node = graph.nodes.get(name)
        if node is None or not isinstance(node.op, InputOp) or name not in graph.inputs:
            raise ValueError(f"physical input {name!r}: node is not a graph input")
        logical = node.output
        if any(not dim.is_static for dim in logical.shape):
            raise ValueError(f"physical input {name!r}: logical shape must be static")
        logical_shape = tuple(dim.as_static() for dim in logical.shape)
        if logical_shape != tuple(int(dim) for dim in storage.logical_shape):
            raise ValueError(f"physical input {name!r}: descriptor shape {storage.logical_shape} does not reproduce {logical_shape}")
        _validate_storage(name, logical.dtype.name, storage)
        for carrier in storage.carriers:
            input_id = f"{name}{carrier.input_suffix}"
            if input_id in new_input_ids or (input_id != name and input_id in graph.nodes):
                raise ValueError(f"physical input {name!r}: carrier input id {input_id!r} already exists")
            new_input_ids.add(input_id)
        prepared.append((name, storage))

    result: dict[str, tuple[str, ...]] = {}
    appended_inputs: list[str] = []
    for name, storage in prepared:
        parked = f"{name}__physical_src"
        graph.rename_node(name, parked)
        values: dict[str, str] = {}
        carrier_inputs: list[str] = []
        for carrier in storage.carriers:
            input_id = f"{name}{carrier.input_suffix}"
            source = graph.add_node(
                op=InputOp(),
                inputs=[],
                output=Tensor(input_id, carrier.shape, carrier.dtype),
                node_id=input_id,
            )
            carrier_inputs.append(source)
            indexed = graph.add_node(
                op=IndexMapOp(
                    out_shape=storage.logical_shape,
                    sources=(IndexSource(input_idx=0, coord_map=carrier.coord_map),),
                ),
                inputs=[source],
                output=Tensor(f"{name}_{carrier.name}_logical", storage.logical_shape, carrier.dtype),
            )
            values[carrier.name] = indexed
        graph.inputs = [carrier_inputs[0] if item == parked else item for item in graph.inputs]
        appended_inputs.extend(carrier_inputs[1:])

        for operation in storage.algebra:
            values[operation.name] = graph.add_node(
                op=ElementwiseOp(op=operation.op),
                inputs=[values[input_name] for input_name in operation.inputs],
                output=Tensor(f"{name}_{operation.name}", storage.logical_shape, operation.dtype),
            )
        graph.replace_node(parked, values[storage.output])
        graph.remove_node(parked)
        result[name] = tuple(carrier_inputs)
    graph.inputs.extend(appended_inputs)
    return result
