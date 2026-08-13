"""Host C++ spellings for the float32-reference Loop backend."""

from __future__ import annotations

_FLOAT_DTYPES = frozenset({"f32", "f16", "f16x2", "f8e4m3", "f8e5m2"})
_TYPE_NAME = {
    **{dtype: "float" for dtype in _FLOAT_DTYPES},
    "i16": "short",
    "i32": "int",
    "i64": "long long",
    "u16": "unsigned short",
    "u32": "unsigned int",
    "u64": "unsigned long long",
    "bool": "bool",
}
_HOST_NBYTES = {
    "float": 4,
    "short": 2,
    "int": 4,
    "long long": 8,
    "unsigned short": 2,
    "unsigned int": 4,
    "unsigned long long": 8,
    "bool": 1,
}
_INTEGER_DTYPES = frozenset({"i16", "i32", "i64", "u16", "u32", "u64"})
_INTEGER_NATIVE_OPS = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "floor_divide",
        "remainder",
        "mod",
        "left_shift",
        "right_shift",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_count",
    }
)


class LoopRenderTarget:
    """Render every floating dtype through the Loop runner's float ABI.

    ``execute_loop_op_cpp`` deliberately coerces its numpy buffers to
    float32. Narrow floating assignments therefore remain numeric identity
    operations in this correctness-reference backend; CUDA retains the real
    storage types and conversion intrinsics through ``CudaRenderTarget``.
    """

    def type_name(self, dtype: str) -> str:
        return _TYPE_NAME.get(dtype, "float")

    def literal(self, text: str, dtype: str) -> str:  # noqa: ARG002
        return text

    def convert(self, value: str, src_dt: str, dst_dt: str) -> str:
        src_type = self.type_name(src_dt)
        dst_type = self.type_name(dst_dt)
        if src_type == dst_type:
            return value
        return f"static_cast<{dst_type}>({value})"

    def bitcast(self, value: str, src_dt: str, dst_dt: str) -> str:
        if src_dt == dst_dt:
            return value
        src_type = self.type_name(src_dt)
        dst_type = self.type_name(dst_dt)
        if _HOST_NBYTES[src_type] != _HOST_NBYTES[dst_type]:
            raise ValueError(f"Loop float32 reference cannot bitcast {src_dt} to {dst_dt}")
        return f"emmy_bitcast<{dst_type}>({value})"

    def intrinsic(self, op_name: str, result_dt: str) -> str:
        if op_name == "bitwise_count":
            return "__builtin_popcountll" if result_dt in ("i64", "u64") else "__builtin_popcount"
        return op_name

    def has_native_op(self, op_name: str, dtype: str) -> bool:
        if dtype == "f32":
            return True
        if dtype in _INTEGER_DTYPES:
            return op_name in _INTEGER_NATIVE_OPS
        return False

    def vector_type(self, dtype: str, n: int) -> tuple[str, str] | None:  # noqa: ARG002
        return None
