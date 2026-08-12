"""CUDA launch-ABI tests that do not require a device."""

from types import SimpleNamespace

from emmy.compiler.backend.cuda.program import _launch
from emmy.compiler.backend.plan import LaunchSpec


class _Kernel:
    def __init__(self):
        self.calls = []
        self.max_dynamic_shared_size_bytes = 0

    def __call__(self, grid, block, args, *, shared_mem):
        self.calls.append((grid, block, args, shared_mem))


def _spec(smem_bytes: int, *, dynamic_smem: bool) -> LaunchSpec:
    return LaunchSpec(
        node_id="node",
        kernel_name="kernel",
        arg_names=(),
        grid=((1,), (1,), (1,)),
        block=((32,), (1,), (1,)),
        smem_bytes=smem_bytes,
        dynamic_smem=dynamic_smem,
        zero_outputs=(),
    )


def test_native_dynamic_smem_is_supplied_below_opt_in_threshold():
    """An extern-shared native kernel still receives its sub-48 KiB slab."""
    kernel = _Kernel()
    _launch(_spec(32 * 1024, dynamic_smem=True), SimpleNamespace(kernels={"kernel": kernel}), {})
    assert kernel.calls == [((1, 1, 1), (32, 1, 1), (), 32 * 1024)]
    assert kernel.max_dynamic_shared_size_bytes == 0


def test_dynamic_smem_above_threshold_sets_opt_in_attribute():
    """Large dynamic slabs opt the kernel in before launch, regardless of source family."""
    kernel = _Kernel()
    _launch(_spec(64 * 1024, dynamic_smem=False), SimpleNamespace(kernels={"kernel": kernel}), {})
    assert kernel.calls == [((1, 1, 1), (32, 1, 1), (), 64 * 1024)]
    assert kernel.max_dynamic_shared_size_bytes == 64 * 1024
