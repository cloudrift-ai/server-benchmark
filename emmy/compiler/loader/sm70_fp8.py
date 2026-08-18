"""Birth-time description of TurboMind's retained SM70 HMMA884 FP8 operands."""

from __future__ import annotations

from emmy.compiler.loader.physical import PhysicalCarrier, PhysicalInputStorage, TypedAlgebra


def retained_sm70_fp8_storage(
    logical_shape: tuple[int, int],
    *,
    weight_shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
    metadata: tuple[int, int],
    interleave_halves: bool = False,
    group_index: int | None = None,
) -> PhysicalInputStorage:
    """Validate an exported carrier triple and describe its logical values.

    The physical shapes and metadata come from the runtime that prepared the
    caller-owned tensors. A mismatch fails before graph construction. The
    returned descriptor contains only carrier geometry, coordinate maps, and
    typed algebra, and is dissolved by :func:`spell_physical_inputs`.
    """
    from emmy.compiler.ir.expr import Literal, placeholder  # noqa: PLC0415

    n, k = (int(dim) for dim in logical_shape)
    if n <= 0 or k <= 0 or n % 128 or k % 128:
        raise ValueError("retained SM70 FP8 logical N and K must be positive multiples of 128")
    if interleave_halves and n % 2:
        raise ValueError("retained SM70 FP8 interleaving requires an even output width")
    if group_index is not None and group_index < 0:
        raise ValueError("retained SM70 FP8 group index must be non-negative")

    expected_weight = (k, n) if group_index is None else (group_index + 1, k, n)
    expected_scale = (k // 128, n) if group_index is None else (group_index + 1, k // 128, n)
    if tuple(int(dim) for dim in weight_shape) != expected_weight:
        raise ValueError(f"retained SM70 FP8 weight carrier {weight_shape} does not match {expected_weight}")
    if tuple(int(dim) for dim in scale_shape) != expected_scale:
        raise ValueError(f"retained SM70 FP8 scale carrier {scale_shape} does not match {expected_scale}")
    expected_metadata = (32 * k, n)
    if tuple(int(value) for value in metadata) != expected_metadata:
        raise ValueError(f"retained SM70 FP8 metadata {metadata} does not match {expected_metadata}")

    row = placeholder(0)
    col = placeholder(1)
    physical_row = row
    if interleave_halves:
        half = Literal(n // 2, "int")
        physical_row = (row % half) * Literal(2, "int") + row / half

    eight = Literal(8, "int")
    thirty_two = Literal(32, "int")
    # The converter stores one 32x8 logical tile in 256 bytes. Within each
    # eight-value K fragment, Volta's HMMA884 operand order swaps bits 0 and 1.
    lane = col % eight
    permuted_lane = (
        (lane / Literal(4, "int")) * Literal(4, "int")
        + (lane % Literal(2, "int")) * Literal(2, "int")
        + (lane / Literal(2, "int")) % Literal(2, "int")
    )
    flat = (
        ((physical_row / thirty_two) * Literal(k // 8, "int") + col / eight) * Literal(256, "int")
        + (physical_row % thirty_two) * eight
        + permuted_lane
    )
    weight_coords = (flat / Literal(n, "int"), flat % Literal(n, "int"))
    scale_coords = (col / Literal(128, "int"), physical_row)
    if group_index is not None:
        group = Literal(group_index, "int")
        weight_coords = (group, *weight_coords)
        scale_coords = (group, *scale_coords)

    return PhysicalInputStorage(
        logical_shape=(n, k),
        carriers=(
            PhysicalCarrier("values", "", expected_weight, "f8e4m3", weight_coords),
            PhysicalCarrier("multiplier", "_scale", expected_scale, "f16", scale_coords),
        ),
        algebra=(
            TypedAlgebra("decoded", "from_f8e4m3", ("values",), "f16"),
            TypedAlgebra("scaled", "multiply", ("decoded", "multiplier"), "f16"),
        ),
        output="scaled",
    )


def expected_sm70_fp8_metadata(logical_shape: tuple[int, int]) -> tuple[int, int]:
    """Return the exact metadata values exported for a supported logical shape."""
    n, k = (int(dim) for dim in logical_shape)
    if n <= 0 or k <= 0 or n % 128 or k % 128:
        raise ValueError("retained SM70 FP8 logical N and K must be positive multiples of 128")
    return 32 * k, n
