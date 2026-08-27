"""Register epilogues retain the Loop tail's per-Assign dtype semantics."""

from emmy.compiler.dtype import F16
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import RegStore
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, RenderCtx, Write
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _warp_epilogue


def _render(*assigns: Assign) -> tuple[str, object]:
    tail = [*assigns, Write(output="out", index=(Var("m"), Var("n")), value=assigns[-1].name)]
    epilogue = _warp_epilogue(tail, "acc", "m", "n", Sigma.IDENTITY)
    assert epilogue is not None
    store = RegStore(
        dst_buffer="out",
        dst_index=(Var("m"), Var("n")),
        frag="_c",
        shape=(16, 8, 16),
        ldm=16,
        epilogue=epilogue,
    )
    ctx = RenderCtx(shapes={"out": (16, 16)}, buffer_dtypes={"out": "f16"})
    return "\n".join(store.render(ctx)), epilogue


def test_typed_copy_narrows_the_fragment_value_before_its_consumer() -> None:
    source, epilogue = _render(
        Assign(name="narrow", op=ElementwiseImpl("copy"), args=("acc",), dtype=F16),
        Assign(name="result", op=ElementwiseImpl("copy"), args=("narrow",)),
    )

    assert epilogue.ops[0][3] is F16
    assert "const __half narrow_e0 = __float2half(_c[0]);" in source
    assert "const __half result_e0 = narrow_e0;" in source


def test_untyped_copy_keeps_the_existing_f32_epilogue() -> None:
    source, epilogue = _render(Assign(name="result", op=ElementwiseImpl("copy"), args=("acc",)))

    assert epilogue.ops[0][3] is None
    assert "const float result_e0 = _c[0];" in source
    assert "__float2half(_c[0])" not in source


def test_transposed_fragment_store_uses_both_output_strides() -> None:
    store = RegStore(
        dst_buffer="out",
        dst_index=(Var("n"), Var("m")),
        frag="_c",
        shape=(16, 8, 16),
        row_dim=1,
        col_dim=0,
    )
    source = "\n".join(store.render(RenderCtx(shapes={"out": (16, 16)}, buffer_dtypes={"out": "f16"})))

    assert "reinterpret_cast<__half2*>" not in source
    assert "(_t * 2 + 1) * 16" in source
    assert "out[n * 16 + m + _g + (_t * 2 + 0) * 16]" in source
